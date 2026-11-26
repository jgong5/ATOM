# SPDX-License-Identifier: MIT
"""Wide local grants, `compass run --wide-local-grants` (#3968).

What each group defends:

* **The cycle term.** A ceiling passes the static cycle only when the cycle's
  first hop waits in TAR on a channel that is not offering; a receiver that
  waits in NER, or one reached on an offering channel, caps it as before, and
  the least such return is the term.
* **A TAR's grant sends messages ahead.** What lies below its ceiling goes
  with the grant, where without the flag the next arrival caps the ceiling.
  Nothing on an offering channel goes ahead: a TAR past such an arrival is
  granted it first, and the lookup its handler offers there is registered
  before any LP moves past it.
* **Nothing else changes.** On generated runs, LPs answering their own calls
  below wide ceilings see every grant a run without local grants gives them,
  and the authority keeps the same timeline, in fewer calls than under the
  ceilings without the flag.
* **The frontend's loop.** A frontend polling its clock connection walks a
  message sent ahead by its own step, not a request, and its timeline is that
  of a run without local grants.
* **The flag.** `compass run --wide-local-grants` writes the run-file key, off
  by default and per run; the run's authority reads it, and refuses it
  without local grants or as anything but a bool.
"""

import dataclasses
import functools
import json
import math
import threading
import uuid

import pytest
from test_drive_runfile import build
from test_vertical_slice import _run_file

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import (
    NER,
    TAR,
    ClockAuthority,
    LpId,
    TimelineLog,
    single_engine_table,
)
from atom.compass.clock.channels import add_lmcache
from atom.utils import clock, compass_loop
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import CompassEventLoop
from tests.compass.clock import test_grant_rule as generated

INF = math.inf
TRAFFIC, FRONTEND, ENGINE = LpId("traffic"), LpId("frontend"), LpId("engine")
LMCACHE = LpId("lmcache")
OUT, REQ = "engine->frontend:output#dp0", "frontend->engine:request#dp0"
CTL, LOOK = "frontend->engine:control#dp0", "engine->lmcache:lookup#dp0"
TABLE = single_engine_table(admission_s=0.54e-3, ipc_s=0.001, stream_s=0.002)
# The frontend's returns by first hop: through the engine, request then
# output; through the traffic LP, stream then http.
VIA_ENGINE, VIA_TRAFFIC = 0.001 + 0.001, 0.002 + 0.54e-3
LBTS = 1.0 + 0.001  # the frontend's, with the engine at 1.0 and traffic at 2.0


def _table(offering=False):
    table = single_engine_table(admission_s=0.54e-3, ipc_s=0.001, stream_s=0.002)
    if offering:
        table.mark_offering(REQ)
    return table


# --- the cycle term -----------------------------------------------------------


@pytest.mark.parametrize(
    "engine, traffic, offering, wide, ceiling",
    [
        (TAR, TAR, False, True, LBTS),
        (NER, TAR, False, True, 0.1 + VIA_ENGINE),
        (TAR, NER, False, True, 0.1 + VIA_TRAFFIC),
        (TAR, TAR, True, True, 0.1 + VIA_ENGINE),
        (TAR, TAR, False, False, 0.1 + VIA_ENGINE),
    ],
    ids=["both-in-TAR", "engine-in-NER", "traffic-in-NER", "offering", "no-flag"],
)
def test_the_cycle_passes_an_lp_only_while_it_waits_in_tar(
    engine, traffic, offering, wide, ceiling
):
    ca = ClockAuthority(_table(offering), ceilings=True, wide_ceilings=wide)
    assert ca.on_request(TRAFFIC, traffic, 2.0, []) == []
    assert ca.on_request(ENGINE, engine, 1.0, []) == []
    [(lp, g, released)] = ca.on_request(FRONTEND, TAR, 0.1, [])
    assert (lp, g, released.ceiling) == (FRONTEND, 0.1, ceiling)


def test_wide_ceilings_need_local_grants():
    with pytest.raises(ValueError, match="wide ceilings widen the ceilings"):
        ClockAuthority(TABLE, wide_ceilings=True)


# --- a TAR's grant sends messages ahead ---------------------------------------


def _frontend_through_two_outputs(**kw):
    """The engine has sent the frontend outputs at 0.3 and 0.6 and waits in
    TAR(1.0), traffic in TAR(2.0); the frontend charges 0.1, then 0.4, then
    idles, each below its ceiling on its own where it has one. Returns the
    authority and the frontend's first grant."""
    ca = ClockAuthority(TABLE, timeline=TimelineLog(), **kw)
    ca.on_request(TRAFFIC, TAR, 2.0, [])
    ca.on_request(ENGINE, TAR, 1.0, [(OUT, 0, 0.3), (OUT, 1, 0.6)])
    [(_, _, first)] = ca.on_request(FRONTEND, TAR, 0.1, [])
    if kw.get("wide_ceilings"):
        steps = [(TAR, 0.4, 0.4, 0), (NER, INF, 0.6, 0)]
        ca.on_request(FRONTEND, NER, INF, [], steps=steps)
    else:
        ca.on_request(FRONTEND, TAR, 0.4, [])
        ca.on_request(FRONTEND, NER, INF, [])  # granted 0.6
        ca.on_request(FRONTEND, NER, INF, [])
    return ca, first


def test_a_tar_grant_sends_what_lies_below_its_ceiling_ahead():
    _, first = _frontend_through_two_outputs(ceilings=True, wide_ceilings=True)
    assert first[OUT] == [(0, 0.3), (1, 0.6)] and first.ceiling == LBTS
    # Without the flag, the first output caps the ceiling, and none goes ahead.
    _, first = _frontend_through_two_outputs(ceilings=True)
    assert first[OUT] == [] and first.ceiling == 0.1 + VIA_ENGINE


def test_the_steps_through_what_was_sent_ahead_are_the_grants_of_a_run_without():
    off, _ = _frontend_through_two_outputs()
    on, _ = _frontend_through_two_outputs(ceilings=True, wide_ceilings=True)
    assert on.timeline.records() == off.timeline.records()
    assert on.grants == off.grants
    served = (off.requests_served["frontend"], on.requests_served["frontend"])
    assert served == (4, 2)


def _driven_lmcache(**kw):
    """An engine at DP 1 offloading to a driven lmcache server, so its request
    channel is offering; lookaheads of 1.0. lmcache waits in NER(4.0)."""
    table = single_engine_table(admission_s=1.0, ipc_s=1.0, stream_s=1.0, dp=1)
    add_lmcache(table, "engine", 1.0, 1, True)
    ca = ClockAuthority(table, ceilings=True, wide_ceilings=True, **kw)
    assert ca.on_request(TRAFFIC, NER, INF, []) == []
    assert ca.on_request(LMCACHE, NER, 4.0, []) == []
    return ca


@pytest.mark.parametrize("kind", [NER, TAR])
def test_no_message_on_an_offering_channel_goes_ahead(kind):
    """The engine's grant, an NER's at the control message or a TAR's short of
    it, has its ceiling at the request's arrival on the offering channel, 1.5,
    and sends nothing on that channel ahead. The engine's next TAR, past 1.5,
    is granted 1.5 first, and the lookup it sends there registers while
    lmcache is still at 0."""
    ca = _driven_lmcache()
    if kind == NER:
        assert ca.on_request(ENGINE, NER, INF, [], ahead=True) == []
        [(_, g, released)] = ca.on_request(
            FRONTEND, NER, INF, [(CTL, 0, 1.4), (REQ, 0, 1.5)]
        )
        assert (g, released[CTL], released[REQ]) == (1.4, [(0, 1.4)], [])
        steps = []
    else:
        assert ca.on_request(FRONTEND, NER, INF, [(CTL, 0, 1.4), (REQ, 0, 1.5)]) == []
        [(_, g, released)] = ca.on_request(ENGINE, TAR, 1.2, [])
        assert (g, released[CTL], released[REQ]) == (1.2, [(0, 1.4)], [])
        steps = [(NER, INF, 1.4, 0)]
    assert released.ceiling == 1.5
    [(lp, g, released)] = ca.on_request(ENGINE, TAR, 4.0, [], steps=steps)
    assert (lp, g, released[REQ]) == (ENGINE, 1.5, [(0, 1.5)])
    replies = ca.on_request(ENGINE, TAR, 4.0, [(LOOK, 0, 2.5)])
    assert [(lp.name, g) for lp, g, _ in replies] == [("lmcache", 2.5)]


# --- nothing else changes, on generated runs ----------------------------------


def test_wide_ceilings_leave_every_lps_grants_and_the_timeline_as_they_are(
    monkeypatch,
):
    narrow, wide, ahead = 0, 0, {}
    for seed in generated.SEEDS:
        for pick in generated.PICKS.values():
            off, off_ca, _ = generated._drive_below_ceilings(seed, pick, False)
            _, narrow_ca, ahead_narrow = generated._drive_below_ceilings(
                seed, pick, True
            )
            with monkeypatch.context() as m:
                widened = functools.partial(ClockAuthority, wide_ceilings=True)
                m.setattr(generated, "ClockAuthority", widened)
                on, on_ca, ahead_wide = generated._drive_below_ceilings(
                    seed, pick, True
                )
            assert on == off, seed
            assert on_ca.grants == off_ca.grants, seed
            assert generated._by_lp(on_ca) == generated._by_lp(off_ca), seed
            narrow += sum(narrow_ca.requests_served.values())
            wide += sum(on_ca.requests_served.values())
            ahead["narrow"] = ahead.get("narrow", 0) + ahead_narrow
            ahead["wide"] = ahead.get("wide", 0) + ahead_wide
    # Wider ceilings: the LPs answered more of their own calls, and more
    # messages went ahead.
    assert wide < narrow and ahead["wide"] > ahead["narrow"]


# --- the frontend's loop ------------------------------------------------------


def _idle_to_the_finish(conn) -> None:
    g = conn.recv()[0]
    while g != INF:
        conn.send((NER, INF, [], INF))
        g = conn.recv()[0]


def _frontend_loop(monkeypatch, wide):
    """A frontend's loop on its in-process authority, polling the connection
    as a co-hosted frontend does. The engine has sent the frontend an output
    at 0.3 and waits in TAR(1.0), the traffic LP in TAR(2.0); the frontend
    charges 0.1, then idles to its timer at 0.5. A thread takes the output, as
    the frontend's output thread does. Returns the authority, the frontend's
    runtime and the clock the output was taken at."""
    endpoint = f"inproc:test-wide-grants-{uuid.uuid4().hex}"
    ca = ClockAuthority(
        TABLE, timeline=TimelineLog(), ceilings=wide, wide_ceilings=wide
    )
    server = clock_transport.serve(ca, endpoint)
    others = []
    for lp, t, log in ((ENGINE, 1.0, [(OUT, 0, 0.3)]), (TRAFFIC, 2.0, [])):
        conn = clock_transport.connect(lp, endpoint)
        conn.send((TAR, t, log, INF))
        others.append(threading.Thread(target=_idle_to_the_finish, args=(conn,)))
    rt = LPRuntime(FRONTEND, TABLE, clock_transport.connect_here(FRONTEND, endpoint))
    rt.start_run()
    monkeypatch.setattr(clock, "_installed", rt)
    loop = CompassEventLoop()
    taken = []

    def output_thread():
        wake = rt.my_wakeup([OUT])
        rt.back_at_wait_point()
        while True:
            wake.clear()
            with rt.lock:
                pending = rt.released[OUT] - rt.handled[OUT]
            if pending:
                break
            wake.wait()
        rt.handed_over(OUT, min(pending))
        taken.append(rt.now)
        rt.back_at_wait_point()

    others.append(threading.Thread(target=output_thread))
    for thread in others:
        thread.daemon = True
        thread.start()
    loop.call_at(0.5, lambda: None)

    async def charge_then_idle():
        compass_loop.charge(0.1)

    guard = threading.Timer(20, lambda: loop.call_soon_threadsafe(loop.stop))
    guard.start()
    try:
        loop.run_until_complete(charge_then_idle())
        loop.run_forever()
        for thread in others:
            thread.join(10)
    finally:
        guard.cancel()
        server.close()
        loop.close()
    return ca, rt, taken


def test_a_frontend_walks_a_message_sent_ahead_by_its_own_step(monkeypatch):
    off, _, taken_off = _frontend_loop(monkeypatch, False)
    on, rt, taken_on = _frontend_loop(monkeypatch, True)
    assert rt.refusals == [] and rt.now == INF
    assert taken_on == taken_off == [0.3]
    mine = [[r for r in ca.timeline.records() if r.lp == FRONTEND] for ca in (off, on)]
    assert mine[1] == mine[0]
    assert [r.time_to for r in mine[1]] == [0.1, 0.3, 0.5, INF]
    # The step to 0.3 went with the request for 0.5.
    served = (off.requests_served["frontend"], on.requests_served["frontend"])
    assert served == (4, 3)


# --- the flag -----------------------------------------------------------------


@pytest.mark.parametrize("flags, on", [((), False), (("--wide-local-grants",), True)])
def test_wide_local_grants_are_off_unless_the_command_turns_them_on(
    tmp_path, flags, on
):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))

    run = build(tmp_path, "--step-stub", str(stub), *flags)

    assert run[compass_run.WIDE_LOCAL_GRANTS] is on
    off = dict(run, **{compass_run.WIDE_LOCAL_GRANTS: not on})
    assert compass_run.configuration(run) == compass_run.configuration(off)


@pytest.mark.parametrize("wide", [False, True])
def test_the_run_s_authority_reads_the_key(tmp_path, wide):
    run = json.loads(_run_file(tmp_path, wide_local_grants=wide).read_text())
    authority = compass_run._RecordingAuthority(run)
    authority.on_request(TRAFFIC, TAR, 2.0, [])
    authority.on_request(ENGINE, TAR, 1.0, [(OUT, 0, 0.3)])

    [(_, g, released)] = authority.on_request(FRONTEND, TAR, 0.1, [])

    assert (g, released[OUT]) == (0.1, [(0, 0.3)] if wide else [])


@pytest.mark.parametrize(
    "keys, refusal",
    [
        ({"wide_local_grants": "yes"}, TypeError),
        ({"local_grants": False, "wide_local_grants": True}, ValueError),
    ],
)
def test_the_run_refuses_wide_ceilings_it_cannot_give(tmp_path, keys, refusal):
    run = json.loads(_run_file(tmp_path, **keys).read_text())
    with pytest.raises(refusal):
        compass_run._RecordingAuthority(run)
