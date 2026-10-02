# SPDX-License-Identifier: MIT
"""`atom.compass.clock_transport`: the serve loop, held replies, refusals and the wire.

Every LP here talks to a real `ClockAuthority` through `connect`, the way
`LPRuntime` does: ``send((kind, t, log, t_daemon))`` with ``(channel, seq,
arrival)`` log entries, then ``recv() -> (G, {channel: [(seq, arrival)]})``. Each
call runs on a daemon thread with a bounded wait, so a reply that never comes
fails the test instead of hanging it.
"""

import json
import math
import queue
import threading

import pytest

from atom.compass.clock import (
    NER,
    TAR,
    BackdatedEvent,
    ChannelTable,
    ClockAuthority,
    LpId,
    LpRegistry,
    single_engine_table,
)
from atom.compass.clock_transport import (
    MalformedMessage,
    connect,
    decode,
    encode,
    serve,
    service,
)

A, B = LpId("a"), LpId("b")
AB = "a->b:m"
INF = math.inf
WAIT_S = 10.0


def _table():
    """`a` and `b`, one channel from `a` to `b` with 0.5 s of lookahead."""
    registry = LpRegistry()
    registry.register(A)
    registry.register(B)
    table = ChannelTable(registry)
    table.declare(AB, A, B, 0.5, "inline")
    return table


class _Recorded(ClockAuthority):
    """Records every request the serve loop hands the rule."""

    def __init__(self, channels):
        super().__init__(channels)
        self.requests = []

    def on_request(self, lp, kind, t, log, t_daemon=INF):
        self.requests.append((lp, (kind, t, log, t_daemon)))
        return super().on_request(lp, kind, t, log, t_daemon)


@pytest.fixture
def served():
    servers = []

    def start(authority, endpoint="inproc:test"):
        servers.append(serve(authority, endpoint))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def _later(fn, *args):
    box = queue.Queue()

    def run():
        try:
            box.put((True, fn(*args)))
        except Exception as e:  # noqa: BLE001 - re-raised by _result
            box.put((False, e))

    threading.Thread(target=run, daemon=True).start()
    return box


def _result(box):
    ok, value = box.get(timeout=WAIT_S)
    if not ok:
        raise value
    return value


def _ask(conn, request):
    _result(_later(conn.send, request))
    return _result(_later(conn.recv))


# --- held replies and the finish ----------------------------------------------


def test_a_held_reply_reaches_its_lp_with_no_second_request(served):
    ca = _Recorded(_table())
    endpoint = served(ca).endpoint
    a, b = connect(A, endpoint), connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], INF)))
    held = _later(b.recv)
    # Served while b is parked in recv, after b's request.
    assert _ask(a, (TAR, 0.2, [], INF)) == (0.2, {})
    assert held.empty()
    assert _ask(a, (TAR, 1.0, [(AB, 0, 0.7)], INF)) == (1.0, {})
    assert _result(held) == (0.7, {AB: [(0, 0.7)]})
    assert [kind for lp, (kind, *_) in ca.requests if lp == B] == [NER]


def test_a_daemon_deadline_crosses_the_wire_and_the_finish_answers_every_lp(served):
    endpoint = served(ClockAuthority(_table())).endpoint
    a, b = connect(A, endpoint), connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], 0.5)))
    held = _later(b.recv)
    assert _ask(a, (TAR, 1.0, [], INF)) == (1.0, {})
    # a's target reaches b's deadline, so it fires.
    assert _result(held) == (0.5, {AB: []})
    _result(_later(b.send, (NER, INF, [], 2.0)))
    held = _later(b.recv)
    # Nothing essential reaches 2.0: every LP waits, and the run finishes.
    assert _ask(a, (NER, INF, [], INF)) == (INF, {})
    assert _result(held) == (INF, {AB: []})


IPC_S, STREAM_S = 0.001, 0.002
HTTP = "traffic->frontend:http"
STREAM = "frontend->traffic:stream"
REQUEST = "frontend->engine:request#dp0"
OUTPUT = "engine->frontend:output#dp0"

#: One request through the single-engine table. ``("send", channel)`` registers
#: a message at the LP's clock plus the channel's lookahead; a TAR's time is
#: relative to the LP's clock, and a third element is an NER's daemon deadline.
SCRIPTS = {
    "traffic": [("send", HTTP), (NER, INF), (NER, INF, 16.0)],
    "frontend": [
        (NER, INF),
        ("send", REQUEST),
        (NER, INF),
        ("send", STREAM),
        (NER, INF),
    ],
    "engine": [
        (NER, INF, 0.005),
        (NER, INF),
        (TAR, 0.05),
        ("send", OUTPUT),
        (NER, INF),
    ],
}


def _drive(conn, table, script):
    """Run `script` as one LP: its request count and its grants, with released seqs."""
    now, next_seq, log, sent, grants = 0.0, {}, [], 0, []
    for kind, x, *daemon in script:
        if kind == "send":
            next_seq[x] = next_seq.get(x, -1) + 1
            log.append((x, next_seq[x], now + table.lookahead(x)))
            continue
        conn.send((kind, now + x if kind == TAR else x, log, *(daemon or [INF])))
        log, sent = [], sent + 1
        now, released = conn.recv()
        seqs = {
            ch: [seq for seq, _ in pairs] for ch, pairs in released.items() if pairs
        }
        grants.append((round(now, 9), seqs))
    return sent, grants


def _three_lp_run(cls, served):
    """Serve a `cls` over the single-engine table and run `SCRIPTS` on it."""
    table = single_engine_table(
        admission_path="serving", ipc_s=IPC_S, stream_s=STREAM_S
    )
    ca = cls(table)
    endpoint = served(ca).endpoint
    conns = {name: connect(LpId(name), endpoint) for name in SCRIPTS}
    runs = {name: _later(_drive, conns[name], table, SCRIPTS[name]) for name in SCRIPTS}
    return ca, {name: _result(run) for name, run in runs.items()}


def test_every_request_gets_exactly_one_reply_through_the_finish(served):
    _, runs = _three_lp_run(ClockAuthority, served)
    sent = {name: sent for name, (sent, _) in runs.items()}
    received = {name: len(grants) for name, (_, grants) in runs.items()}
    assert sent == received == {"traffic": 2, "frontend": 3, "engine": 4}
    assert {name: grants for name, (_, grants) in runs.items()} == {
        "traffic": [(0.063, {STREAM: [0]}), (INF, {})],
        "frontend": [(0.009, {HTTP: [0]}), (0.061, {OUTPUT: [0]}), (INF, {})],
        "engine": [(0.005, {}), (0.01, {REQUEST: [0]}), (0.06, {}), (INF, {})],
    }


def test_every_request_reaches_the_rule_as_the_frame_it_was_sent_as(
    served, monkeypatch
):
    frames = []

    def recording(frame):
        message = decode(frame)
        frames.append(message)
        return message

    monkeypatch.setattr(service, "decode", recording)
    ca, _ = _three_lp_run(_Recorded, served)
    requests = [m for m in frames if m[0] in (TAR, NER)]
    assert len(requests) == len(ca.requests) == 9
    assert sorted(map(repr, requests)) == sorted(repr(r) for _, r in ca.requests)
    assert [m[0] for m in frames].count("GRANT") == 9
    assert [m[0] for m in frames].count("BIND") == 3


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    "request_, error, match",
    [
        ((TAR, 1.0, [(AB, 0, 0.1)], INF), BackdatedEvent, "before a's clock 0.0"),
        ((TAR, INF, [], INF), ValueError, "must be a finite number"),
        ((TAR, 1.0, [("a->x:m", 0, 1.0)], INF), KeyError, "not a declared channel"),
    ],
    ids=["backdated", "value", "key"],
)
def test_a_refusal_raises_at_its_requester_and_the_loop_serves_on(
    served, request_, error, match
):
    a = connect(A, served(ClockAuthority(_table())).endpoint)
    with pytest.raises(error, match=match) as refused:
        _ask(a, request_)
    assert type(refused.value) is error
    assert _ask(a, (TAR, 1.0, [], INF)) == (1.0, {})


def test_a_backdated_event_arrives_with_the_lp_table(served):
    ca = ClockAuthority(_table())
    endpoint = served(ca).endpoint
    a, b = connect(A, endpoint), connect(B, endpoint)
    _result(_later(b.send, (NER, INF, [], INF)))
    assert _ask(a, (TAR, 2.0, [(AB, 0, 3.0)], INF)) == (2.0, {})
    with pytest.raises(BackdatedEvent) as refused:
        _ask(a, (TAR, 3.0, [(AB, 1, 2.1)], INF))
    assert refused.value.reason.startswith(f"{AB} seq 1 arrives at 2.1")
    table = refused.value.table
    assert table == ca.lp_table()
    # b is waiting, so its state, target and undelivered all cross the wire.
    assert [(r.state, r.target, r.undelivered) for r in table if r.lp == B] == [
        (NER, INF, ((AB, 0, 3.0),))
    ]


def test_a_log_behind_its_receiver_is_refused(served):
    endpoint = served(ClockAuthority(_table())).endpoint
    a, b = connect(A, endpoint), connect(B, endpoint)
    assert _ask(b, (TAR, 0.3, [], INF)) == (0.3, {AB: []})
    with pytest.raises(BackdatedEvent, match="behind b's clock at 0.3"):
        _ask(a, (NER, INF, [(AB, 0, 0.2)], INF))


def test_a_reply_is_not_something_a_participant_may_send(served):
    a = connect(A, served(ClockAuthority(_table())).endpoint)
    for message in [("GRANT", 1.0, {}), ("BIND", A)]:
        with pytest.raises(MalformedMessage, match="sends only TAR, NER$"):
            a.send(message)
    assert _ask(a, (TAR, 1.0, [], INF)) == (1.0, {})


# --- one entry into the rule --------------------------------------------------


def test_a_carriage_can_reach_the_rule_only_through_a_frame(served):
    public = sorted(name for name in vars(service._Server) if not name.startswith("_"))
    assert public == ["bind", "close", "submit"]
    server = served(ClockAuthority(_table()))
    with pytest.raises(MalformedMessage, match="never message objects"):
        server.bind(("BIND", A))
    with pytest.raises(MalformedMessage, match="never message objects"):
        server.submit(A, (TAR, 1.0, [], INF))


def test_no_public_attribute_of_the_transport_hands_back_the_clock(served):
    server = served(ClockAuthority(_table()))
    conn = connect(A, server.endpoint)
    leaks = [
        f"{type(holder).__name__}.{name}"
        for holder in (server, conn)
        for name in dir(holder)
        if not name.startswith("_")
        and isinstance(getattr(holder, name), ClockAuthority)
    ]
    assert not leaks


# --- endpoints ----------------------------------------------------------------


def test_a_name_the_clock_does_not_hold_is_refused_at_the_first_message(served):
    endpoint = served(ClockAuthority(_table())).endpoint
    with pytest.raises(KeyError, match="not a participant"):
        connect(LpId("pp-stage-7"), endpoint)
    connect(A, endpoint)
    with pytest.raises(KeyError, match="already bound"):
        connect(A, endpoint)


def test_nothing_is_reachable_in_this_process_before_it_is_served():
    with pytest.raises(KeyError, match="nothing is served at inproc:not-started"):
        connect(A, "inproc:not-started")


def test_two_clocks_at_one_endpoint_are_refused():
    first = serve(ClockAuthority(_table()), "inproc:twice")
    with pytest.raises(ValueError, match="already served"):
        serve(ClockAuthority(_table()), "inproc:twice")
    first.close()
    serve(ClockAuthority(_table()), "inproc:twice").close()


@pytest.mark.parametrize("endpoint", ["clock", "tcp://127.0.0.1:9"])
def test_an_endpoint_that_reaches_nothing_is_refused_rather_than_guessed(endpoint):
    with pytest.raises(ValueError, match="is not inproc:<name>"):
        connect(A, endpoint)
    with pytest.raises(ValueError, match="is not inproc:<name>"):
        serve(ClockAuthority(_table()), endpoint)


# --- the wire -----------------------------------------------------------------


def _messages():
    ca = ClockAuthority(_table())
    ca.on_request(B, NER, INF, [])
    ca.on_request(A, TAR, 2.0, [(AB, 0, 3.0)])
    with pytest.raises(BackdatedEvent) as refused:
        ca.on_request(A, TAR, 3.0, [(AB, 1, 2.1)])
    return [
        ("BIND", A),
        (TAR, 1.0, [(AB, 0, 1.5)], INF),
        (NER, INF, [], 2.0),
        ("GRANT", INF, {AB: [(0, 0.5), (1, 0.75)]}),
        ("REFUSED", "BackdatedEvent", refused.value.reason, refused.value.table),
        ("REFUSED", "KeyError", "a reason", None),
    ]


def test_a_message_survives_the_round_trip_it_is_built_for():
    for message in _messages():
        assert decode(encode(message)) == message


def test_a_message_encodes_the_same_bytes_every_time():
    assert encode(("GRANT", 1.0, {"x": [], "y": []})) == encode(
        ("GRANT", 1.0, {"y": [], "x": []})
    )
    assert encode((NER, INF, [], 2.0)) == (
        b'{"kind":"NER","log":[],"t":"+inf","t_daemon":2.0}'
    )


def test_a_frame_is_json_a_stranger_can_read():
    for message in _messages():
        json.loads(
            encode(message),
            parse_constant=lambda name: pytest.fail(f"bare {name} on the wire"),
        )


def test_a_duration_that_is_not_one_is_refused_where_it_would_be_written():
    with pytest.raises(MalformedMessage, match="not JSON compliant: nan"):
        encode((TAR, math.nan, [], INF))


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"{}",
        b'{"kind":"gossip"}',
        b"\xff\xfe",
        "a string",
        b'{"kind":"NER","log":[],"t":Infinity,"t_daemon":"+inf"}',
        b'{"kind":"TAR","log":[[["x"],0,1.0]],"t":1.0,"t_daemon":"+inf"}',
        b'{"kind":"TAR","log":[["x",0.0,1.0]],"t":1.0,"t_daemon":"+inf"}',
        b'{"kind":"NER","log":[],"t":"+inf"}',
        b'{"kind":"REFUSED","error":"SystemExit","reason":"","table":null}',
    ],
)
def test_a_frame_that_is_not_a_message_is_refused(frame):
    with pytest.raises(MalformedMessage):
        decode(frame)
