"""Channel reactions (D3.6): an LP's declared reactions set the distances LBTS
and the local-grant cycle take, so a stream event pending at the router does not
hold a decode frontend at the router's 1.5 ms, while a relay or a cancel does."""

import math

import pytest

from atom.compass.clock import NER, TAR, ClockAuthority, LpId
from atom.compass.clock.channels import prefill_decode_table

FLOOR = 2**-14
ROUTER_S, ROUTER_IN = 1.5e-3, 1e-5
RELAY = ROUTER_S - ROUTER_IN  # the router's relay hop out
FD, FP, ROUTER, TRAFFIC = (
    LpId("frontend-D0"),
    LpId("frontend-P"),
    LpId("router"),
    LpId("traffic"),
)
# a stream event on to traffic, traffic's next request, the router's cancel as a relay
STREAM_BACK = (FLOOR + ROUTER_S - ROUTER_IN) + ROUTER_IN + RELAY


def _table(reactions=None):
    """One prefill and two decode deployments behind LP router; `reactions`
    replaces the router's declaration when given."""
    table = prefill_decode_table(
        admission_s=FLOOR,
        ipc_s=1e-5,
        stream_s=FLOOR,
        router_s=ROUTER_S,
        kv_write_req_s=1e-4,
        router_in_s=ROUTER_IN,
        prefill=1,
        decode=2,
    )
    if reactions is not None:
        table.mark_reactions(ROUTER, reactions)
    return table


def test_the_router_declares_its_own_kind_and_a_cancels_relay():
    table = _table()
    assert table.declares(ROUTER) and not table.declares(FD)
    assert table.reach("frontend-D0->router:stream", FD) == pytest.approx(STREAM_BACK)
    assert table.reach("frontend-P->router:relay", FD) == pytest.approx(RELAY)
    assert table.reach("traffic->router:http", FD) == pytest.approx(RELAY)
    assert table.distance(ROUTER, FD) == pytest.approx(RELAY)
    assert table.reaction_distance(ROUTER, FD) == pytest.approx(RELAY)


def test_an_undeclared_kind_counts_as_able_to_react_on_every_channel():
    table = _table({"http": ("http", "relay"), "relay": ("relay",)})
    assert table.reach("frontend-D0->router:stream", FD) == pytest.approx(RELAY)


def _router_term(wide, pending):
    """The router's term in frontend-D0's LBTS, the router waiting in NER with
    `pending` registered for it and held there by traffic, which runs at 0."""
    ca = ClockAuthority(_table(), ceilings=True, wide=wide)
    for lp in ca._ids:
        if lp not in (TRAFFIC, FD, FP):
            assert ca.on_request(lp, NER, math.inf, []) == []
    assert ca.on_request(FP, NER, math.inf, [p for p in pending if "frontend-P" in p[0]]) == []
    ca.on_request(FD, TAR, 0.002, [p for p in pending if "frontend-D0" in p[0]])
    row = next(r for r in ca.lp_table() if r.lp == FD)
    assert next(r for r in ca.lp_table() if r.lp == ROUTER).state == NER
    return dict(row.row)[ROUTER]


@pytest.mark.parametrize("wide", [1, 2])
def test_a_stream_event_pending_at_the_router_holds_a_frontend_only_without_reactions(
    wide,
):
    term = _router_term(wide, [("frontend-D0->router:stream", 0, ROUTER_IN)])
    back = STREAM_BACK if wide == 2 else RELAY
    assert term == pytest.approx(ROUTER_IN + back)


@pytest.mark.parametrize("wide", [1, 2])
def test_a_relay_pending_at_the_router_holds_a_frontend_at_its_relay_hop(wide):
    term = _router_term(wide, [("frontend-P->router:relay", 0, ROUTER_IN)])
    assert term == pytest.approx(ROUTER_IN + RELAY)


def test_a_cancel_pending_at_the_router_holds_a_frontend_at_its_relay_hop():
    ca = ClockAuthority(_table(), ceilings=True, wide=2)
    for lp in ca._ids:
        if lp not in (TRAFFIC, FD, FP):
            assert ca.on_request(lp, NER, math.inf, []) == []
    ca.on_request(FD, TAR, 0.002, [])
    # traffic sends a request or a cancel, then idles; the router is granted
    # nothing while frontend-P still runs at 0
    ca.on_request(TRAFFIC, NER, math.inf, [("traffic->router:http", 0, ROUTER_IN)])
    rows = {r.lp: r for r in ca.lp_table()}
    assert rows[ROUTER].state == NER
    assert dict(rows[FD].row)[ROUTER] == pytest.approx(ROUTER_IN + RELAY)


def test_a_running_router_holds_a_frontend_at_its_clock_plus_its_relay_hop():
    ca = ClockAuthority(_table(), ceilings=True, wide=2)
    for lp in ca._ids:
        if lp not in (ROUTER, FD):
            ca.on_request(lp, NER, math.inf, [])
    ca.on_request(FD, TAR, 0.002, [])
    rows = {r.lp: r for r in ca.lp_table()}
    assert rows[ROUTER].state == "running"
    assert dict(rows[FD].row)[ROUTER] == pytest.approx(0.0 + RELAY)
