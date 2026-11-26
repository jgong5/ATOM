# SPDX-License-Identifier: MIT
"""Fast mode's pads of the router's paths, and a frontend's send floor.

* **Pads.** `prefill_decode_table`'s `pad_http_s` and `pad_relay_s` lengthen
  the request's and the relay's paths by their hop out of the router, which is
  what the router stamps with; the stream's path keeps its lookahead.
* **Send floor.** A frontend's floor bounds its term into an engine by
  min(max(N, floor), its earliest unreleased trigger) over a covered first
  hop; a trigger released to it voids the floor; a covered send below it
  ends the run; it rides the wire as ``floor``.
"""

import math

import pytest

from atom.compass.clock import (
    NER,
    TAR,
    ChannelTable,
    ClockAbort,
    ClockAuthority,
    LpId,
    LpRegistry,
    prefill_decode_table,
)
from atom.compass.clock_transport import wire

IPC, STREAM, ROUTER, KV, ADMIT, R_IN = 1.0e-4, 2.0e-3, 5.0e-4, 3.0e-4, 5.4e-4, 1.0e-5


def _pd(**pads):
    return prefill_decode_table(
        admission_s=ADMIT,
        ipc_s=IPC,
        stream_s=STREAM,
        router_s=ROUTER,
        kv_write_req_s=KV,
        router_in_s=R_IN,
        **pads,
    )


def test_the_pads_lengthen_the_request_and_relay_hops_out_of_the_router_only():
    plain, padded = _pd(), _pd(pad_http_s=3e-3, pad_relay_s=2e-3)
    grew = {
        name: padded.lookahead(name) - plain.lookahead(name)
        for name in (c.name for lp in plain.registry.ids() for c in plain.channels_from(lp))
    }
    assert grew.pop("router->frontend-P:http") == pytest.approx(3e-3)
    assert grew.pop("router->frontend-D:relay") == pytest.approx(2e-3)
    assert all(d == 0.0 for d in grew.values()), grew


def test_no_pad_is_the_exact_table():
    plain, zero = _pd(), _pd(pad_http_s=0.0, pad_relay_s=0.0)
    for lp in plain.registry.ids():
        for c in plain.channels_from(lp):
            assert zero.lookahead(c.name) == plain.lookahead(c.name)


R, F, E = LpId("router"), LpId("frontend"), LpId("engine")
TRIGGER, COVERED, OUT, STREAM_CH = (
    "router->frontend:m",
    "frontend->engine:m",
    "engine->frontend:m",
    "frontend->router:m",
)


def _loop_table():
    """A router whose messages trigger a frontend, which feeds its engine."""
    registry = LpRegistry()
    for lp in (E, F, R):
        registry.register(lp)
    table = ChannelTable(registry)
    for name, src, dst, la in (
        (TRIGGER, R, F, 3.0),
        (COVERED, F, E, 0.1),
        (OUT, E, F, 0.1),
        (STREAM_CH, F, R, 0.01),
    ):
        table.declare(name, src, dst, la, "inline")
    return table


def _engine_lbts(ca):
    (row,) = [row for row in ca.lp_table() if row.lp == E]
    return min(term for _, term in row.row)


@pytest.fixture(autouse=True)
def _distance_rule_unless_asked(monkeypatch, request):
    """The floor tests read the distance rule's terms; the TAR-cut tests set their own."""
    if "cut" not in request.node.name and "refined" not in request.node.name and "offering" not in request.node.name:
        monkeypatch.setenv("COMPASS_CA_TAR_CUT", "0")


def _started(floor, engine_at=7.0):
    ca = ClockAuthority(_loop_table())
    assert ca.on_request(R, TAR, 100.0, []) == []
    assert ca.on_request(E, TAR, engine_at, []) == []
    return ca, ca.on_request(F, TAR, 5.0, [], floor=floor)


def test_without_a_floor_the_engine_waits_on_its_frontends_clock():
    ca, replies = _started(None)
    assert [(str(i), g) for i, g, _ in replies] == [("frontend", 5.0)]
    assert _engine_lbts(ca) == pytest.approx(5.1)


def test_a_floor_lets_the_engine_past_its_frontends_clock_up_to_its_other_paths():
    ca, replies = _started(8.0)
    assert [(str(i), g) for i, g, _ in replies] == [("frontend", 5.0), ("engine", 7.0)]
    # frontend at 5.0: its floor 8.0 plus the request hop, below its path through
    # the router (5.0 + 0.01 + 3.0 + 0.1).
    assert _engine_lbts(ca) == pytest.approx(min(8.1, 103.1))


def test_the_path_through_the_router_still_bounds_a_floored_frontend():
    ca, _ = _started(50.0)
    assert _engine_lbts(ca) == pytest.approx(5.0 + 0.01 + 3.0 + 0.1)


def test_a_pending_trigger_bounds_the_floor_and_its_release_voids_it():
    ca = ClockAuthority(_loop_table())
    assert ca.on_request(R, TAR, 4.0, []) == []
    assert ca.on_request(E, TAR, 7.5, []) == []
    ca.on_request(F, TAR, 5.0, [], floor=20.0)
    # router at 4.0 sends a trigger arriving at 7.0 and waits far ahead.
    ca.on_request(R, TAR, 100.0, [(TRIGGER, 0, 7.0)])
    assert _engine_lbts(ca) == pytest.approx(7.0 + 0.1)
    # The frontend is released the trigger: its floor is void.
    replies = ca.on_request(F, NER, 9.0, [], floor=20.0)
    assert any(i == F and TRIGGER in released for i, _, released in replies)
    assert _engine_lbts(ca) == pytest.approx(7.0 + 0.1)


def test_a_covered_send_below_the_floor_ends_the_run():
    ca, _ = _started(8.0, engine_at=5.05)
    with pytest.raises(ClockAbort, match="below frontend's send floor 8.0"):
        ca.on_request(F, TAR, 6.0, [(COVERED, 0, 5.5 + 0.1)])


def test_a_covered_send_after_a_released_trigger_is_no_breach():
    ca = ClockAuthority(_loop_table())
    ca.on_request(R, TAR, 4.0, [])
    ca.on_request(E, TAR, 7.5, [])
    ca.on_request(F, TAR, 5.0, [], floor=20.0)
    ca.on_request(R, TAR, 100.0, [(TRIGGER, 0, 7.0)])
    ca.on_request(F, NER, 9.0, [], floor=20.0)  # released the trigger at 7.0
    ca.on_request(F, TAR, 9.5, [(COVERED, 0, 7.0 + 0.1)])


def test_the_floor_rides_the_wire_and_its_absence_changes_nothing():
    plain = wire._request_out(TAR, 1.0, [], math.inf)
    assert "floor" not in plain
    assert wire._request_in(plain) == wire._request_in(wire._request_out(TAR, 1.0, [], math.inf))
    body = wire._request_out(TAR, 1.0, [], math.inf, floor=2.5)
    assert wire._request_in(body)[-1] == 2.5
    assert wire._request_in(body)[:4] == wire._request_in(plain)[:4]


# --- the refined LBTS: an LP waiting in TAR passes nothing on before its target


def _chain_abc():
    registry = LpRegistry()
    for lp in ("a", "b", "c"):
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    table.declare("a->b:m", LpId("a"), LpId("b"), 0.1, "inline")
    table.declare("b->c:m", LpId("b"), LpId("c"), 0.1, "inline")
    return table


@pytest.mark.parametrize("middle, cut, granted", [
    (TAR, "1", True),   # b sends at 10 at the earliest, whatever reaches it
    (NER, "1", False),  # b may send the instant a's message lands, at 1.1
    (TAR, "0", False),  # the distance rule alone
])
def test_a_tar_lp_in_the_middle_cuts_the_path_behind_it(monkeypatch, middle, cut, granted):
    monkeypatch.setenv("COMPASS_CA_TAR_CUT", cut)
    a, b, c = LpId("a"), LpId("b"), LpId("c")
    ca = ClockAuthority(_chain_abc())
    assert ca.on_request(c, TAR, 5.0, []) == []
    # a runs at 0: its message reaches b at 0.1 at the earliest.
    got = [(str(i), g) for i, g, _ in ca.on_request(b, middle, 10.0, [])]
    assert (("c", 5.0) in got) is granted


def test_an_offering_channel_lets_a_tar_lp_pass_on_at_its_arrival(monkeypatch):
    monkeypatch.setenv("COMPASS_CA_TAR_CUT", "1")
    table = _chain_abc()
    table.mark_offering("a->b:m")
    a, b, c = LpId("a"), LpId("b"), LpId("c")
    ca = ClockAuthority(table)
    ca.on_request(c, TAR, 5.0, [])
    got = [(str(i), g) for i, g, _ in ca.on_request(b, TAR, 10.0, [])]
    assert ("c", 5.0) not in got


def test_the_refined_lbts_is_never_below_the_distance_rules(monkeypatch):
    monkeypatch.setenv("COMPASS_CA_TAR_CUT", "1")
    ca = ClockAuthority(_loop_table())
    ca.on_request(R, TAR, 4.0, [])
    ca.on_request(E, TAR, 7.5, [])
    ca.on_request(F, TAR, 5.0, [], floor=20.0)
    eit = ca._refined_eit()
    for i in (R, F, E):
        assert eit[ca._index[i]] >= ca._lbts(i) - 1e-12


def test_a_trigger_reaching_an_idle_floored_lp_before_its_n_bounds_its_engine(monkeypatch):
    """F waits in NER at N 5.0 (an engine output due then) under a floor of 20; P runs at 0 and can send a trigger
    through the router that reaches F at 1.01, so F may send into its engine at 1.01 and E's EIT is 1.11, not 5.1."""
    monkeypatch.setenv("COMPASS_CA_TAR_CUT", "1")
    registry = LpRegistry()
    names = ("engine", "frontend", "prefill", "router")
    for lp in names:
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    for name, src, dst, la in (
        ("prefill->router:m", "prefill", "router", 0.01),
        ("router->frontend:m", "router", "frontend", 1.0),
        ("frontend->engine:m", "frontend", "engine", 0.1),
        ("engine->frontend:m", "engine", "frontend", 0.1),
    ):
        table.declare(name, LpId(src), LpId(dst), la, "inline")
    ca = ClockAuthority(table)
    e, f, p, r = (LpId(n) for n in names)
    ca._state.update({e: TAR, f: NER, p: "running", r: NER})
    ca._nv.update({e: 3.0, f: 5.0, p: 0.0, r: math.inf})
    ca._now.update({e: 0.5, f: 0.5, p: 0.0, r: 0.0})
    ca._floor[f] = 20.0
    eit = ca._refined_eit()
    assert eit[ca._index[e]] == pytest.approx(0.01 + 1.0 + 0.1)
    assert eit[ca._index[e]] >= ca._lbts(e) - 1e-12
