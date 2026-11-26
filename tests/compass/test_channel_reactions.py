# SPDX-License-Identifier: MIT
"""Channel reactions under wide local grants (#4144).

What each group defends:

* **The router's declaration.** On the 8P16D C1228 table, as
  `prefill_decode_table` builds it, LP router declares that each message goes
  on as one of its own kind and a cancel on ``http`` as ``http`` or
  ``relay``; the distances follow that, and a kind left undeclared reacts on
  every channel.
* **The router's term in a decode frontend's LBTS.** While the router waits
  in NER, a stream event pending there holds the frontend at the event's way
  back through traffic, about 3.06 ms, where #3968 alone holds it at the
  relay hop, 1.49 ms; a relay or a cancel pending there holds it at the relay
  hop. While the router runs or waits in TAR, and for its own events in NER,
  its term is its ``N`` plus the relay hop. Without wide ceilings the
  declaration changes nothing.
* **The cycle.** A decode frontend whose engine waits in TAR has its ceiling
  at its grant plus the stream's way back, 3.06 ms past it, not 1.5 ms.
* **Nothing else changes, on generated runs.** LPs whose sends follow a
  declaration get every grant, and the authority keeps the timeline, of a run
  without local grants, in fewer calls than without the declaration.
"""

import math

import pytest

from atom.compass.clock import (
    NER,
    TAR,
    ChannelTable,
    ClockAuthority,
    LpId,
    LpRegistry,
    TimelineLog,
)
from atom.compass.clock.channels import kind_of, prefill_decode_table
from tests.compass.clock import test_grant_rule as generated

INF = math.inf
FLOOR = 2**-14  # decode output and stream, and admission
ROUTER_S, ROUTER_IN, DECODE_IPC = 1.5e-3, 10.3e-6, 9.57e-6
RELAY = ROUTER_S - ROUTER_IN  # the router's relay hop out
STREAM_OUT = FLOOR + ROUTER_S - ROUTER_IN  # its stream hop out, to traffic
# A stream event on to traffic, traffic's next request, which the router
# relays to the frontend as a cancel.
STREAM_BACK = STREAM_OUT + ROUTER_IN + RELAY
FD, FD1, FP = LpId("frontend-D0"), LpId("frontend-D1"), LpId("frontend-P0")
ED, ROUTER, TRAFFIC = LpId("engine-D0"), LpId("router"), LpId("traffic")
STREAM, RELAY_IN = "frontend-D0->router:stream", "frontend-P0->router:relay"
HTTP_IN = "traffic->router:http"


def _table(reactions=None):
    """The 8P16D C1228 table: 4 prefill deployments at DP 2 and 4 decode ones
    at DP 4 behind LP router. `reactions` replaces the router's declaration
    when given; ``{}`` declares that every kind reacts on every channel."""
    table = prefill_decode_table(
        admission_s=FLOOR,
        ipc_s=1e-5,
        stream_s=FLOOR,
        router_s=ROUTER_S,
        kv_write_req_s=1e-4,
        decode_ipc=(DECODE_IPC, FLOOR),
        dp=(2, 4),
        router_in_s=ROUTER_IN,
        prefill=4,
        decode=4,
    )
    if reactions is not None:
        table.mark_reactions(ROUTER, reactions)
    return table


# --- the router's declaration -------------------------------------------------


def test_the_router_declares_its_own_kind_and_a_cancels_relay():
    table = _table()
    assert table.declares(ROUTER) and not table.declares(FD)
    assert table.reach(STREAM, FD) == pytest.approx(STREAM_BACK)
    assert table.reach(RELAY_IN, FD) == pytest.approx(RELAY)
    assert table.reach(HTTP_IN, FD) == pytest.approx(RELAY)
    assert table.reach("router->frontend-D0:relay", FD) == 0.0
    assert table.reaction_distance(ROUTER, FD) == pytest.approx(RELAY)
    assert table.reaction_distance(TRAFFIC, FD) == pytest.approx(ROUTER_IN + RELAY)
    # Another decode frontend's stream no longer reaches this one through the
    # router's relay hop; its engine's KV write request to a prefill engine,
    # whose output frontend-P relays, does.
    assert table.distance(FD1, FD) == pytest.approx(ROUTER_IN + RELAY)
    via_prefill = DECODE_IPC + 1e-4 + 1e-5 + ROUTER_S
    assert table.reaction_distance(FD1, FD) == pytest.approx(via_prefill)


def test_an_undeclared_kind_counts_as_able_to_react_on_every_channel():
    table = _table({"http": ("http", "relay"), "relay": ("relay",)})
    assert table.reach(STREAM, FD) == pytest.approx(RELAY)
    assert table.reach(RELAY_IN, FD) == pytest.approx(RELAY)


def test_a_message_provokes_what_its_receiver_declares():
    table = _table()
    assert table.reacts(STREAM, "router->traffic:stream")
    assert not table.reacts(STREAM, "router->frontend-D0:relay")
    assert table.reacts("router->traffic:stream", "traffic->router:http")
    assert kind_of("engine-D0->engine-P0:kv_write_req#dp0.dp1") == "kv_write_req"


# --- the router's term in a decode frontend's LBTS ----------------------------


def _router_term(pending, reactions=None, wide=True):
    """The router's term in frontend-D0's LBTS, the router waiting in NER with
    `pending` registered for it, held there by traffic running at 0, and every
    other LP idle; frontend-D0 waits in TAR."""
    ca = ClockAuthority(_table(reactions), ceilings=True, wide_ceilings=wide)
    mine = lambda lp: [p for p in pending if p[0].startswith(f"{lp.name}->")]
    for lp in ca._ids:
        if lp not in (TRAFFIC, FD):
            assert ca.on_request(lp, NER, INF, mine(lp)) == []
    assert ca.on_request(FD, TAR, 0.004, mine(FD)) == []
    rows = {r.lp: r for r in ca.lp_table()}
    assert rows[ROUTER].state == NER and rows[TRAFFIC].state == "running"
    return dict(rows[FD].row)[ROUTER]


@pytest.mark.parametrize(
    "reactions, back", [(None, STREAM_BACK), ({}, RELAY)], ids=["declared", "#3968"]
)
def test_a_stream_event_pending_at_the_router_holds_a_frontend_at_its_way_back(
    reactions, back
):
    term = _router_term([(STREAM, 0, ROUTER_IN)], reactions)
    assert term == pytest.approx(ROUTER_IN + back)


@pytest.mark.parametrize("kind", [RELAY_IN, HTTP_IN], ids=["relay", "cancel"])
def test_a_relay_or_a_cancel_pending_at_the_router_holds_a_frontend_at_the_relay_hop(
    kind,
):
    """A cancel comes from traffic, which then runs no longer: frontend-P0,
    running at 0, holds the router instead."""
    if kind == RELAY_IN:
        assert _router_term([(kind, 0, ROUTER_IN)]) == pytest.approx(ROUTER_IN + RELAY)
        return
    ca = ClockAuthority(_table(), ceilings=True, wide_ceilings=True)
    for lp in ca._ids:
        if lp not in (TRAFFIC, FD, FP):
            assert ca.on_request(lp, NER, INF, []) == []
    assert ca.on_request(FD, TAR, 0.004, []) == []
    assert ca.on_request(TRAFFIC, NER, INF, [(kind, 0, ROUTER_IN)]) == []
    rows = {r.lp: r for r in ca.lp_table()}
    assert rows[ROUTER].state == NER
    assert dict(rows[FD].row)[ROUTER] == pytest.approx(ROUTER_IN + RELAY)


def test_the_earliest_of_each_kind_counts_until_it_is_released():
    """A stream event at 2 ms and a relay at 3 ms pending at the router: the
    relay's term, 3 ms plus the relay hop, is the lesser. With every other LP
    idle, the router is granted the stream event, and only the relay is left."""
    pending = [(STREAM, 0, 0.002), (RELAY_IN, 0, 0.003)]
    assert _router_term(pending) == pytest.approx(0.003 + RELAY)
    ca = ClockAuthority(_table(), ceilings=True, wide_ceilings=True)
    for lp in ca._ids:
        if lp not in (ROUTER, FD):
            log = [p for p in pending if p[0].startswith(f"{lp.name}->")]
            assert ca.on_request(lp, NER, INF, log) == []
    assert ca.on_request(FD, NER, INF, pending[:1]) == []
    [(lp, g, released)] = ca.on_request(ROUTER, NER, INF, [])
    assert (lp, g, released[STREAM]) == (ROUTER, 0.002, [(0, 0.002)])
    assert ca._cfirst[ROUTER] == [INF, 0.003, INF]  # http, relay, stream


@pytest.mark.parametrize("state", ["running", TAR, NER])
def test_the_router_s_own_events_hold_a_frontend_at_the_relay_hop(state):
    """Running at 0, in TAR(0.5), or in NER(0.5) with nothing pending."""
    ca = ClockAuthority(_table(), ceilings=True, wide_ceilings=True)
    for lp in ca._ids:
        if lp not in (ROUTER, TRAFFIC, FD):
            ca.on_request(lp, NER, INF, [])
    if state != "running":
        assert ca.on_request(ROUTER, state, 0.5, []) == []
    ca.on_request(FD, TAR, 0.004, [])
    rows = {r.lp: r for r in ca.lp_table()}
    assert rows[ROUTER].state == state
    now = 0.0 if state == "running" else 0.5
    assert dict(rows[FD].row)[ROUTER] == pytest.approx(now + RELAY)


def test_without_wide_ceilings_the_declaration_changes_nothing():
    term = _router_term([(STREAM, 0, ROUTER_IN)], wide=False)
    assert term == pytest.approx(ROUTER_IN + RELAY)
    plain = ClockAuthority(_table({}), ceilings=True)
    declared = ClockAuthority(_table(), ceilings=True)
    assert plain._from == declared._from and plain._cycle == declared._cycle


# --- the cycle ----------------------------------------------------------------


@pytest.mark.parametrize(
    "reactions, back",
    [(None, ROUTER_IN + STREAM_BACK), ({}, ROUTER_S)],
    ids=["declared", "#3968"],
)
def test_a_decode_frontend_s_ceiling_passes_the_router_s_relay_hop(reactions, back):
    """engine-D0 waits in TAR(1.0) and every other LP idles: frontend-D0's
    grant at 0.002 has its ceiling at its cycle through the router."""
    ca = ClockAuthority(_table(reactions), ceilings=True, wide_ceilings=True)
    for lp in ca._ids:
        if lp not in (ED, FD):
            assert ca.on_request(lp, NER, INF, []) == []
    assert ca.on_request(ED, TAR, 1.0, []) == []
    [(lp, g, released)] = ca.on_request(FD, TAR, 0.002, [])
    assert (lp, g) == (FD, 0.002)
    assert released.ceiling == pytest.approx(0.002 + back)


# --- nothing else changes, on generated runs ----------------------------------

DECLARING = LpId("a")
KINDS = ("m", "n")


def _generated_table(seed):
    """`seed`'s LPs, two to four of a, b, c and d, joined by channels of
    kinds m and n with positive lookaheads, and a's reactions: per inbound
    kind, the outbound kinds it provokes, a kind left out provoking every one."""
    names = generated.NAMES[: 2 + seed % 3]
    registry = LpRegistry()
    for name in names:
        registry.register(LpId(name))
    table, x = ChannelTable(registry), seed + 1
    for src in names:
        for dst in names:
            for kind in KINDS:
                x = generated._lcg(x)
                if src != dst and x % 3:
                    lookahead = (0.25, 0.5, 1.0)[(x >> 8) % 3]
                    table.declare(
                        f"{src}->{dst}:{kind}",
                        LpId(src),
                        LpId(dst),
                        lookahead,
                        "inline",
                    )
    reactions = {}
    for kind in KINDS:
        x = generated._lcg(x)
        provoked = ((), ("m",), ("n",), None)[(x >> 8) % 4]
        if provoked is not None:
            reactions[kind] = provoked
    return table, reactions


def _request(table, lp, own, seed, provokes):
    """`generated._request`, each send but on a channel `provokes` allows."""
    if own["k"] >= generated.STEPS:
        period = (INF, 0.75, 1.5)[(seed + ord(lp.name[0])) % 3]
        return NER, INF, [], own["g"] + period
    x = generated._lcg(
        generated._lcg(
            seed * 7919 + own["k"] * 104729 + own["received"] * 31 + ord(lp.name[0])
        )
    )
    log = []
    for channel in table.channels_from(lp):
        x = generated._lcg(x)
        if x % 2 and provokes(lp, own["cause"], channel.name):
            seq = own["seq"][channel.name]
            log.append((channel.name, seq, own["g"] + channel.lookahead_s))
            own["seq"][channel.name] += 1
    d = (0.0, 0.5, 1.0)[(x >> 8) % 3]
    return ((TAR, own["g"] + d), (NER, own["g"] + d), (NER, INF), (TAR, own["g"] + d))[
        (x >> 4) % 4
    ] + (log, INF)


def _drive(seed, pick, ceilings, declared):
    """`generated._drive_below_ceilings` on `_generated_table(seed)`, with
    ceilings wide when `ceilings`: LP a sends, after a grant at an arrival,
    only on the kinds its reactions let the messages released there provoke.
    The table declares `declared` for it when given. Returns each LP's grants
    as it sees them, and the authority."""
    table, reactions = _generated_table(seed)
    if declared is not None:
        table.mark_reactions(DECLARING, declared)

    def provokes(lp, cause, out):
        if lp != DECLARING or cause is None:  # an own event provokes any send
            return True
        return any(
            reactions.get(kind) is None or kind_of(out) in reactions[kind]
            for kind in cause
        )

    ca = ClockAuthority(
        table, timeline=TimelineLog(), ceilings=ceilings, wide_ceilings=ceilings
    )
    ids = table.registry.ids()
    own = {
        i: {"k": 0, "g": 0.0, "received": 0, "cause": None}
        | {"seq": {c.name: 0 for c in table.channels_from(i)}}
        for i in ids
    }
    lp = {
        i: {"ceiling": -INF, "ahead": [], "log": [], "steps": [], "asked": None}
        for i in ids
    }
    seen = {i: [] for i in ids}

    def took(i, g, got, asked):
        kind, t, t_daemon = asked
        own_event = kind == TAR or g in (t, t_daemon)
        cause = None if own_event else {kind_of(c) for c, _, _ in got}
        seen[i].append((g, sorted(got)))
        own[i].update(
            k=own[i]["k"] + 1, g=g, received=own[i]["received"] + len(got), cause=cause
        )

    running, x = list(ids), 0
    for _ in range(100000):
        x = generated._lcg(x)
        i = pick(sorted(running), x)
        running.remove(i)
        mine = lp[i]
        while True:
            kind, t, log, t_daemon = _request(table, i, own[i], seed, provokes)
            mine["log"] += log
            first = mine["ahead"][0][0] if mine["ahead"] else INF
            to = t if kind == TAR else min(t, t_daemon, first)
            essential = kind == TAR or t < INF or mine["ahead"]
            if not (own[i]["g"] <= to < mine["ceiling"] and essential):
                break
            reached = [m for m in mine["ahead"] if m[0] <= to]
            mine["ahead"] = mine["ahead"][len(reached) :]
            mine["steps"].append((kind, t, to, len(mine["log"])))
            took(i, to, [(c, s, a) for a, c, s in reached], (kind, t, t_daemon))
        steps, log = mine["steps"], mine["log"]
        mine["steps"], mine["log"] = [], []
        mine["asked"] = (kind, t, t_daemon)
        ahead = ceilings and kind == NER
        replies = ca.on_request(i, kind, t, log, t_daemon, None, steps, ahead)
        for j, g, released in replies:
            theirs = lp[j]
            pairs = [(c, s, a) for c, r in released.items() for s, a in r]
            reached = [(c, s, a) for a, c, s in theirs["ahead"]]
            got = [m for m in pairs if m[2] <= g] + reached
            theirs["ahead"] = sorted((a, c, s) for c, s, a in pairs if a > g)
            ceiling = getattr(released, "ceiling", None)
            theirs["ceiling"] = -INF if ceiling is None else ceiling
            if g == INF:
                seen[j].append((g, sorted(got)))
                continue
            took(j, g, got, theirs["asked"])
            running.append(j)
        if any(r and r[-1][0] == INF for r in seen.values()):
            return seen, ca
    raise AssertionError(f"seed {seed} did not finish")


def test_declared_reactions_leave_every_lps_grants_and_the_timeline_as_they_are():
    alone = declared = 0
    for seed in generated.SEEDS:
        reactions = _generated_table(seed)[1]
        for pick in generated.PICKS.values():
            off, off_ca = _drive(seed, pick, False, None)
            plain, plain_ca = _drive(seed, pick, True, {})
            on, on_ca = _drive(seed, pick, True, reactions)
            for run, ca in ((plain, plain_ca), (on, on_ca)):
                assert run == off, seed
                assert ca.grants == off_ca.grants, seed
                assert generated._by_lp(ca) == generated._by_lp(off_ca), seed
            alone += sum(plain_ca.requests_served.values())
            declared += sum(on_ca.requests_served.values())
    # The declarations widened some ceilings: the LPs answered more of their
    # own calls.
    assert declared < alone
