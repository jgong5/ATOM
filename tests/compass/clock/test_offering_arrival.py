# SPDX-License-Identifier: MIT
"""#3767's variant (b): a TAR holding an arrival on an offering channel is
granted that arrival first, then asks again.

An engine's input thread offers a tier lookup for each request it is handed
(`EngineCore._offer_tier_lookups`), and the clock owner makes the send at the
request's arrival (`LPRuntime.offer`). With a lookup driver, `add_lmcache`
marks each rank's request channel and lookup replies offering. The groups
below defend:

* **The backdated probe.** An engine in ``TAR(1.0)``, a request at 0.5 on an
  offering channel: the authority grants 0.5 first, and the lookup the engine
  sends there registers at 0.5 plus the lookahead; on a channel not marked
  offering, the base's case, the same lookup is refused as backdated.
* **What stays.** An arrival on a channel not offering is released at the
  TAR's own grant; the step table's grant at the arrival is ``ARRIVAL``, and
  the scenario check reads the TAR it splits as one row.
* **DP members.** Only the member the message is for is granted it first and
  asks again; a member whose call named later rounds gets its reply there; a
  parked member leaves its park for it and parks again; no round is granted
  ahead past it.
* **The LP's side.** The owner asks the rounds left again after a grant short
  of them, a parked member parks again, and an own step counts the sends its
  walk offered before it.
"""

import json
import math
import queue
import threading

import pytest

from atom.compass.clock import (
    ARRIVAL,
    NER,
    PARK,
    TAR,
    BackdatedEvent,
    ClockAuthority,
    LpId,
    Released,
    single_engine_table,
)
from atom.compass.clock.channels import add_lmcache
from atom.compass.detect.determinism import StepTable
from atom.compass.detect.scenario import _rows
from atom.utils.clock import LPRuntime

INF = math.inf
IPC = 2.0**-10
ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
TRAFFIC, LMCACHE = LpId("traffic"), LpId("lmcache")
RANKS = ("dp0", "dp1")


def req(r=0):
    return f"frontend->engine:request#dp{r}"


def look(r=0):
    return f"engine->lmcache:lookup#dp{r}"


def _table(dp=1, driven=True, ipc=IPC):
    table = single_engine_table(admission_s=ipc, ipc_s=ipc, stream_s=ipc, dp=dp)
    add_lmcache(table, "engine", ipc, dp, driven)
    return table


class _Rows:
    """The authority's timeline, as ``(lp, from, to, kind)``."""

    def __init__(self):
        self.rows = []

    def record(self, lp, time_from, time_to, kind, recovered, released, sent):
        self.rows.append((lp.name, time_from, time_to, kind))


def _ca(dp=1, driven=True, ipc=IPC, **kw):
    table = _table(dp, driven, ipc)
    if dp > 1:
        names = [c.name for c in table.channels_into(ENGINE)]
        names += [c.name for c in table.channels_from(ENGINE)]
        kw["members"] = {
            ENGINE: {r: [n for n in names if n.endswith(f"#{r}")] for r in RANKS}
        }
    ca = ClockAuthority(table, **kw)
    for lp in (TRAFFIC, LMCACHE):
        assert ca.on_request(lp, NER, INF, []) == []
    return ca


def _grants(replies):
    return [(to[1] if isinstance(to, tuple) else to.name, g) for to, g, _ in replies]


def test_add_lmcache_marks_the_request_and_lookup_reply_channels_offering_when_driven():
    table = _table(2)
    offering = {c.name for c in table.channels_into(ENGINE) if table.offering(c.name)}
    assert offering == {
        req(0),
        req(1),
        "lmcache->engine:lookup#dp0",
        "lmcache->engine:lookup#dp1",
    }
    assert not any(_table(2, driven=False).offering(n) for n in offering)


# --- the backdated probe ------------------------------------------------------


@pytest.mark.parametrize("driven", [True, False], ids=["offering", "base"])
def test_the_backdated_probe(driven):
    """Engine in TAR(1.0), an ADD at 0.5 released by the grant, its lookup sent
    at 0.5: registered at 0.5 plus the lookahead, or, at the base, refused."""
    rows = _Rows()
    ca = _ca(driven=driven, timeline=rows)
    assert ca.on_request(ENGINE, TAR, 1.0, []) == []  # the frontend still runs
    [(lp, g, released)] = ca.on_request(FRONTEND, NER, INF, [(req(), 0, 0.5)])
    assert lp == ENGINE and released[req()] == [(0, 0.5)]
    lookup = [(look(), 0, 0.5 + IPC)]
    if not driven:
        assert g == 1.0
        with pytest.raises(BackdatedEvent, match="before engine's clock 1.0 plus"):
            ca.on_request(ENGINE, TAR, 2.0, lookup)
        return
    assert g == 0.5 and ca.lp_table()[0].now == 0.5  # engine
    # It asks again, with the lookup it sent at the arrival.
    replies = ca.on_request(ENGINE, TAR, 1.0, lookup)
    assert _grants(replies) == [("lmcache", 0.5 + IPC)]
    assert replies[0][2][look()] == [(0, 0.5 + IPC)]
    assert _grants(ca.on_request(LMCACHE, NER, INF, [])) == [("engine", 1.0)]
    engine = [r for r in rows.rows if r[0] == "engine"]
    assert engine == [("engine", 0.0, 0.5, ARRIVAL), ("engine", 0.5, 1.0, TAR)]
    assert ca.grants["engine"] == 2


@pytest.mark.parametrize("driven", [True, False], ids=["offering", "base"])
def test_a_tars_n_is_its_earliest_offering_arrival(driven):
    """No LP is granted past what the engine's handler may send at 0.5."""
    ca = _ca(driven=driven)
    ca.on_request(FRONTEND, TAR, 0.25, [(req(), 0, 0.5)])
    replies = ca.on_request(ENGINE, TAR, 1.0, [])
    assert _grants(replies) == [("frontend", 0.25)]
    [row] = [row for row in ca.lp_table() if row.lp == ENGINE]
    assert (row.state, row.n) == (TAR, 0.5 if driven else 1.0)


def test_an_arrival_on_a_channel_not_offering_waits_for_the_tars_grant():
    ca = _ca()
    ctl = "frontend->engine:control#dp0"
    ca.on_request(ENGINE, TAR, 1.0, [])
    [(lp, g, released)] = ca.on_request(FRONTEND, NER, INF, [(ctl, 0, 0.5)])
    assert (lp, g, released[ctl]) == (ENGINE, 1.0, [(0, 0.5)])


def test_the_scenario_check_reads_a_tar_granted_an_arrival_first_as_one_row():
    table = StepTable(json.dumps({}))
    table.record("engine", 0.0, 0.25, ARRIVAL)
    table.record("engine", 0.25, 0.5, ARRIVAL)
    table.record("engine", 0.5, 1.0, TAR)
    table.record("engine", 1.0, 2.0, TAR)
    _, tars, _ = _rows(table.text())
    assert tars["engine"] == [(0.0, 1.0), (1.0, 2.0)]


# --- DP members ---------------------------------------------------------------


def test_only_the_member_the_message_is_for_is_granted_it_first():
    ca = _ca(dp=2)
    assert ca.on_request(ENGINE, TAR, 1.0, [], member="dp0") == []
    assert ca.on_request(ENGINE, TAR, 1.0, [], member="dp1") == []
    replies = ca.on_request(FRONTEND, NER, INF, [(req(1), 0, 0.5)])
    assert _grants(replies) == [("dp1", 0.5)]  # dp0 stays in the round
    assert replies[0][2][req(1)] == [(0, 0.5)]
    lookup = [(look(1), 0, 0.5 + IPC)]
    assert _grants(ca.on_request(ENGINE, TAR, 1.0, lookup, member="dp1")) == [
        ("lmcache", 0.5 + IPC)
    ]
    assert _grants(ca.on_request(LMCACHE, NER, INF, [])) == [("dp0", 1.0), ("dp1", 1.0)]


def test_a_member_whose_call_named_later_rounds_gets_its_reply_at_the_message():
    ca = _ca(dp=2)
    ca.on_request(FRONTEND, NER, INF, [(req(1), 0, 1.0)])
    assert ca.on_request(ENGINE, TAR, 1.0, [], member="dp0") == []
    replies = ca.on_request(ENGINE, TAR, 1.0, [], member="dp1", rounds=[1.5])
    # Round 1.0 releases dp1's request: dp1 is answered there, not at 1.5.
    assert _grants(replies) == [("dp0", 1.0), ("dp1", 1.0)]
    assert replies[1][2][req(1)] == [(0, 1.0)]
    lookup = [(look(1), 0, 1.0 + IPC)]
    assert ca.on_request(ENGINE, TAR, 1.5, lookup, member="dp1") == []
    assert _grants(ca.on_request(ENGINE, TAR, 1.5, [], member="dp0")) == [
        ("lmcache", 1.0 + IPC)
    ]
    assert _grants(ca.on_request(LMCACHE, NER, INF, [])) == [("dp0", 1.5), ("dp1", 1.5)]


def test_a_parked_member_leaves_its_park_for_the_message_and_parks_again():
    ca = _ca(dp=2)
    ca.on_request(FRONTEND, NER, INF, [(req(1), 0, 0.25)])
    assert ca.on_request(ENGINE, PARK, INF, [], member="dp1") == []
    replies = ca.on_request(ENGINE, NER, 0.3, [], member="dp0")
    assert _grants(replies) == [("dp0", 0.25), ("dp1", 0.25)]
    assert replies[1][2][req(1)] == [(0, 0.25)]
    lookup = [(look(1), 0, 0.25 + IPC)]
    assert ca.on_request(ENGINE, PARK, INF, lookup, member="dp1") == []
    assert _grants(ca.on_request(ENGINE, NER, 0.3, [], member="dp0")) == [
        ("lmcache", 0.25 + IPC)
    ]
    assert _grants(ca.on_request(LMCACHE, NER, INF, [])) == [("dp0", 0.3)]
    assert _grants(ca.on_request(ENGINE, PARK, INF, [], member="dp0")) == [
        ("dp0", 0.3),
        ("dp1", 0.3),
    ]


def test_no_round_is_granted_ahead_past_an_offering_arrival():
    ca = _ca(dp=2, ipc=1.0, ceilings=True)
    ca.on_request(FRONTEND, NER, INF, [(req(1), 0, 1.25)])
    ca.on_request(ENGINE, TAR, 1.0, [], member="dp0")
    replies = ca.on_request(ENGINE, TAR, 1.0, [], member="dp1", rounds=[1.5])
    # dp0 returns at 1.0 owing round 1.5, which is not granted ahead of it.
    assert _grants(replies) == [("dp0", 1.0)]
    assert getattr(replies[0][2], "ceiling", None) is None
    assert not ca._owing[ENGINE]
    replies = ca.on_request(ENGINE, TAR, 1.5, [], member="dp0")
    assert _grants(replies) == [("dp1", 1.25)]
    assert replies[0][2][req(1)] == [(0, 1.25)]
    replies = ca.on_request(ENGINE, TAR, 1.5, [(look(1), 0, 2.25)], member="dp1")
    assert _grants(replies)[:2] == [("dp0", 1.5), ("dp1", 1.5)]


# --- the LP's side ------------------------------------------------------------


class FakeConn:
    def __init__(self, *replies, member=None):
        self.replies, self.sent, self.member = list(replies), [], member

    def send(self, msg):
        self.sent.append(msg)

    def recv(self):
        return self.replies.pop(0)


def _runtime(*replies, member=None):
    """An engine runtime on a scripted authority, its owner a thread, and
    `call(fn, *args)`, which runs fn there."""
    jobs = queue.Queue()

    def loop():
        while True:
            fn, box = jobs.get()
            try:
                box.put((True, fn()))
            except Exception as e:  # noqa: BLE001 - re-raised on the caller
                box.put((False, e))

    owner = threading.Thread(target=loop, daemon=True)
    owner.start()

    def call(fn, *args):
        box = queue.Queue()
        jobs.put((lambda: fn(*args), box))
        ok, value = box.get(timeout=10)
        if not ok:
            raise value
        return value

    conn = FakeConn(*replies, member=member)
    return LPRuntime(ENGINE, _table(2), conn, owner=owner), call


def _offering_handler(rt, ch, n, offer_at=()):
    """A thread taking `n` released messages off `ch`; the k-th for k in
    `offer_at` offers a lookup on that rank's channel."""
    lookup = look(int(ch[-1]))

    def run():
        wake = rt.my_wakeup()
        for k in range(n):
            rt.back_at_wait_point()
            while True:
                wake.clear()
                with rt.lock:
                    pending = rt.released[ch] - rt.handled[ch]
                if pending:
                    break
                wake.wait()
            rt.handed_over(ch, min(pending))
            if k in offer_at:
                rt.offer(lambda: rt.stamp_send(lookup))
        rt.back_at_wait_point()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_the_owner_asks_again_after_a_grant_short_of_its_tar():
    rt, call = _runtime((0.5, {req(): [(0, 0.5)]}), (1.0, {}))
    rt.start_run()
    handler = _offering_handler(rt, req(), 1, offer_at=(0,))
    call(rt.advance_to, 1.0)
    assert rt.conn.sent == [
        ("TAR", 1.0, [], INF),
        ("TAR", 1.0, [(look(), 0, 0.5 + IPC)], INF),
    ]
    assert rt.now == 1.0 and rt.calls == 2 and not rt.refusals
    handler.join(5)


def test_a_member_asks_its_rounds_left_again():
    replies = (0.5, {req(1): [(0, 0.5)]}), (1.0, {req(1): [(1, 1.0)]}), (1.5, {})
    rt, call = _runtime(*replies, member="dp1")
    rt.start_run()
    handler = _offering_handler(rt, req(1), 2, offer_at=(0, 1))
    call(rt.advance_to, 1.0, 1.5)
    sent = [(m[0], m[1], m[2], m[-1]) for m in rt.conn.sent]
    assert sent == [
        ("TAR", 1.0, [], [1.5]),
        ("TAR", 1.0, [(look(1), 0, 0.5 + IPC)], [1.5]),
        ("TAR", 1.5, [(look(1), 1, 1.0 + IPC)], INF),
    ]
    assert rt.now == 1.5
    handler.join(5)


def test_a_parked_member_given_an_offering_message_parks_again():
    replies = (0.25, {req(1): [(0, 0.25)]}), (0.3, {})
    rt, call = _runtime(*replies, member="dp1")
    rt.start_run()
    handler = _offering_handler(rt, req(1), 1, offer_at=(0,))
    call(rt.park)
    assert rt.conn.sent == [
        ("PARK", INF, [], INF),
        ("PARK", INF, [(look(1), 0, 0.25 + IPC)], INF),
    ]
    assert rt.now == 0.3
    handler.join(5)


def _under(ceiling, released=None):
    out = Released(released or {})
    out.ceiling = ceiling
    return out


def test_an_own_step_counts_the_sends_its_walk_offered_before_it():
    ahead = _under(1.0, {req(): [(0, 0.3), (1, 0.6)]})
    rt, call = _runtime((0.1, _under(0.2)), (0.3, ahead), (1.5, {}))
    rt.start_run()
    handler = _offering_handler(rt, req(), 2, offer_at=(1,))
    call(rt.advance_to, 0.1)
    call(rt.next_event, INF)  # asks its messages ahead: 0.6 is sent ahead
    call(rt.advance_to, 0.8)  # below the ceiling: walks 0.6, whose handler offers
    call(rt.advance_to, 1.5)
    assert rt.conn.sent[-1] == (
        "TAR",
        1.5,
        [(look(), 0, 0.6 + IPC)],
        INF,
        [("TAR", 0.8, 0.8, 1)],
        False,
    )
    handler.join(5)
