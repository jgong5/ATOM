# SPDX-License-Identifier: MIT
"""The ``tcp`` serve side under ``select_serve`` (`compass run --select-serve`):
one thread reads every connection through a selector and serves each frame
where it is whole, in place of a reader thread a connection.

What each group defends:

* **The same replies.** Scripted runs give each LP the same reply bytes
  through both serve sides; a bound connection ending inside a frame, an abort,
  and the refusals a connection is answered and served on after give every LP
  the same refusal or reply through both.
* **One thread.** Every connection is served on the selector's thread, a frame
  once whole however the reads cut it, and a fault no reader expects ends its
  own connection alone.
* **The order.** The selector serves two LPs' frames in the order it reads
  them, one their readers could also take the serve side's lock in. On
  generated runs under wide ceilings, as `compass run --wide-local-grants`
  runs, no submit order changes a grant any LP sees or the timeline, though
  the orders change how many calls the LPs make.
* **The key.** `compass run --select-serve` writes ``select_serve``, off by
  default and outside the configuration's name, and the run's authority
  refuses it as anything but a bool.
"""

import dataclasses
import functools
import json
import socket
import threading
import time

import pytest
from test_drive_runfile import build
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import (
    NER,
    TAR,
    ClockAuthority,
    LpId,
    prefill_decode_table,
)
from atom.compass.clock_transport import MalformedMessage, connect, serve, service
from atom.compass.clock_transport.stream import _framed, _receive, _SelectServer
from atom.compass.clock_transport.wire import (
    BIND,
    GRANT,
    JOINED,
    REFUSED,
    decode,
    encode,
)

from . import test_grant_rule as generated
from .test_clock_transport import AB, A, B, _ask, _later, _result, _table
from .test_clock_transport_socket import (
    FIVE_LP,
    PREFILL_DECODE_SCRIPTS,
    RUNS,
    _connect,
    _replies,
)
from .test_clock_transport_threads import INF, LOCAL, _Threads

SIDES = {"readers": False, "selector": True}


@pytest.fixture
def served():
    servers = []

    def start(authority, select):
        servers.append(serve(authority, LOCAL, select))
        return servers[-1].endpoint

    yield start
    for server in servers:
        server.close()


def _seen(box):
    """A reply, or a refusal as its type, reason and LP table."""
    try:
        return _result(box)
    except Exception as refused:  # noqa: BLE001 - compared across the sides
        return (
            type(refused).__name__,
            getattr(refused, "reason", str(refused)),
            getattr(refused, "table", None),
        )


def _until_served(ca, count):
    deadline = time.monotonic() + 10.0
    while sum(ca.requests_served.values()) < count:
        assert time.monotonic() < deadline, ca.requests_served
        time.sleep(0.005)


# --- the same replies -----------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
def test_a_scripted_run_gives_each_lp_the_same_reply_bytes_through_both_sides(
    served, run
):
    make, kwargs, scripts = RUNS[run]
    table = make(**kwargs)
    sides = [
        _replies(served(ClockAuthority(table), select), table, scripts)
        for select in SIDES.values()
    ]
    assert sides[0] == sides[1]
    assert all(f[-1].startswith(b'{"G":"+inf"') for f in sides[1][1].values())


def _a_connection_ends_inside_a_frame(served, select):
    """engine-P's connection ends inside a frame while every LP but traffic
    waits in NER; then traffic asks."""
    ca = ClockAuthority(prefill_decode_table(**FIVE_LP))
    endpoint = served(ca, select)
    conns = {name: _connect(LpId(name), endpoint) for name in PREFILL_DECODE_SCRIPTS}
    waiting = [name for name in conns if name != "traffic"]
    for name in waiting:
        _result(_later(conns[name].send, (NER, INF, [], INF)))
    _until_served(ca, len(waiting))
    pending = [_later(conns[name].recv) for name in waiting if name != "engine-P"]
    conns["engine-P"]._server._sock.sendall(b"\x00\x00\x00\x40{")
    conns["engine-P"].close()
    seen = [_seen(box) for box in pending]
    return seen + [_seen(_later(_ask, conns["traffic"], (NER, INF, [], INF)))]


def _an_abort_and_refusals(served, select):
    """A reply kind sent as a request and a name the clock does not hold are
    refused, and the connection serves on; then a backdated message ends the
    run for its sender and for the LP parked on its channel."""
    ca = ClockAuthority(_table())
    endpoint = served(ca, select)
    a, b = _connect(A, endpoint), _connect(B, endpoint)
    a.send((GRANT, 1.0, {}))
    seen = [_seen(_later(a.recv)), _seen(_later(_connect, LpId("x"), endpoint))]
    for k, (target, arrival) in enumerate([(1.0, 0.7), (2.0, 1.2)]):
        _result(_later(b.send, (NER, INF, [], INF)))
        _until_served(ca, 1 + 2 * k)
        parked = _later(b.recv)
        seen.append(_seen(_later(_ask, a, (TAR, target, [(AB, k, arrival)], INF))))
        seen.append(_seen(parked))
    return seen


@pytest.mark.parametrize(
    "case", [_a_connection_ends_inside_a_frame, _an_abort_and_refusals]
)
def test_an_end_an_abort_and_a_refusal_reach_every_lp_alike_through_both_sides(
    served, case
):
    readers, selector = (case(served, select) for select in SIDES.values())
    assert readers == selector
    refused = [kind for kind, *_ in selector if isinstance(kind, str)]
    assert {"ClockAbort", "BackdatedEvent", "ValueError", "KeyError"} & set(refused)


# --- one thread -----------------------------------------------------------------


def test_every_connection_is_served_on_the_one_selector_thread(served):
    ca = _Threads(_table())
    before = set(threading.enumerate())
    endpoint = served(ca, True)
    started = set(threading.enumerate()) - before
    a, b = _connect(A, endpoint), _connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], INF)))
    held = _later(b.recv)
    assert _ask(a, (TAR, 1.0, [(AB, 0, 0.7)], INF)) == (1.0, {})
    # b's held reply, released by a's request, is written on that thread.
    assert _result(held) == (0.7, {AB: [(0, 0.7)]})
    # One thread, of those the server started, served both connections.
    assert ca.served_on["a"] == ca.served_on["b"] and len(ca.served_on["a"]) == 1
    assert ca.served_on["a"] <= {t.ident for t in started}


def test_a_frame_split_across_reads_and_two_frames_in_one_are_each_served(served):
    endpoint = served(ClockAuthority(_table()), True)
    host, port = endpoint.removeprefix("tcp://").rsplit(":", 1)
    raw = socket.create_connection((host, int(port)))
    bind = _framed(encode((BIND, A, None)))
    raw.sendall(bind[:3])  # the length itself split
    time.sleep(0.05)
    raw.sendall(bind[3:])
    reader = raw.makefile("rb")
    assert decode(_receive(reader))[0] == JOINED
    b = _connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], INF)))
    # A request and a stray bind in one write: the request is served before
    # the bind, which a bound connection refuses, is read.
    raw.sendall(
        _framed(encode((TAR, 1.0, [(AB, 0, 0.7)], INF)))
        + _framed(encode((BIND, A, None)))
    )
    assert decode(_receive(reader))[:3] == (GRANT, 1.0, {})
    assert _result(_later(b.recv)) == (0.7, {AB: [(0, 0.7)]})
    assert decode(_receive(reader))[:2] == (REFUSED, "ValueError")
    raw.close()


def test_a_fault_no_reader_expects_ends_its_own_connection_alone(served, monkeypatch):
    ca = ClockAuthority(_table())
    endpoint = served(ca, True)
    b = _connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], INF)))
    _until_served(ca, 1)
    held = _later(b.recv)
    decoding = service.decode
    with monkeypatch.context() as m:
        m.setattr(service, "decode", lambda frame: 1 / 0)
        with pytest.raises(MalformedMessage, match="with no reply"):
            _connect(A, endpoint)
    assert service.decode is decoding
    # The selector served on: a binds, and its request releases b's reply.
    a = _connect(A, endpoint)
    assert _ask(a, (TAR, 1.0, [(AB, 0, 0.7)], INF)) == (1.0, {})
    assert _result(held) == (0.7, {AB: [(0, 0.7)]})


# --- the order ------------------------------------------------------------------


def test_under_wide_ceilings_no_submit_order_changes_a_grant_or_the_timeline(
    monkeypatch,
):
    widened = functools.partial(ClockAuthority, wide_ceilings=True)
    monkeypatch.setattr(generated, "ClockAuthority", widened)
    shuffled = generated.PICKS["shuffled"]
    calls = set()
    for seed in generated.SEEDS:
        runs = [
            generated._drive_below_ceilings(seed, p, True)
            for p in generated.PICKS.values()
        ]
        runs += [
            generated._drive_below_ceilings(seed, shuffled, True, s)
            for s in range(1, 5)
        ]
        (seen, ca, _), *others = runs
        for other, other_ca, _ in others:
            assert other == seen, seed
            assert generated._by_lp(other_ca) == generated._by_lp(ca), seed
        calls.add(len({tuple(c.requests_served.values()) for _, c, _ in runs}))
    # Not vacuous: on some seed the orders made the LPs ask a different number
    # of times, so the replies' ceilings differed.
    assert max(calls) > 1


# --- the key ----------------------------------------------------------------------


@pytest.mark.parametrize("flags, on", [((), False), (("--select-serve",), True)])
def test_the_selector_is_off_unless_the_command_turns_it_on(tmp_path, flags, on):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    run = build(tmp_path, "--step-stub", str(stub), *flags)
    assert run[compass_run.SELECT_SERVE] is on
    off = dict(run, **{compass_run.SELECT_SERVE: not on})
    assert compass_run.configuration(run) == compass_run.configuration(off)


def test_the_run_refuses_a_select_serve_that_is_not_a_bool(tmp_path):
    run = json.loads(_run_file(tmp_path, select_serve="yes").read_text())
    with pytest.raises(TypeError, match="select_serve is 'yes'"):
        compass_run._RecordingAuthority(run)


def test_an_in_process_endpoint_has_no_reader_for_a_selector_to_replace():
    server = serve(ClockAuthority(_table()), "inproc:select", True)
    try:
        assert not isinstance(server, _SelectServer)
        a, b = connect(A, server.endpoint), connect(B, server.endpoint)
        b.send((NER, INF, [], INF))
        a.send((TAR, 1.0, [(AB, 0, 0.7)], INF))
        assert (a.recv(), b.recv()) == ((1.0, {}), (0.7, {AB: [(0, 0.7)]}))
    finally:
        server.close()
