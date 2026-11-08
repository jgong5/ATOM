# SPDX-License-Identifier: MIT
"""`atom.compass.clock.authority`: the grant rule and the Clock Authority's states.

What the groups below defend:

* **N by state.** A peer waiting in TAR bounds the others by its target, not by
  its clock. `TestNByState` runs the rejected reading against the real state
  machine, so the only difference between the two measurements is that term.
* **Distance, not neighbours.** An idle LP in the middle of a chain does not
  hide the LP behind it. A rule that reads only direct channels grants too far
  and a legal message then lands in the receiver's past.
* **Every registration check is a raise carrying the LP table**, never an
  `assert`, which `python -O` deletes.
* **Recovery and the end of a run are grants, not aborts.**
* **Daemon deadlines** are held until essential time reaches them, delay no
  essential grant, and never fire once the run has finished.
* **Generated runs**, with daemon deadlines: no reply grants past an unreleased
  message, and each LP's replies do not depend on the order in which running
  LPs submit. With a DP group beside them, a member asking each run of its
  rounds in one call leaves every LP's grants and the timeline as one call a
  round does.
"""

import ast
import math
from pathlib import Path

import pytest

from atom.compass.clock import (
    NER,
    TAR,
    BackdatedEvent,
    ChannelTable,
    ClockAbort,
    ClockAuthority,
    LpId,
    LpRegistry,
    TimelineLog,
    single_engine_table,
)
from atom.compass.clock import authority as authority_module

A, B = LpId("a"), LpId("b")
INF = math.inf


def _table(lps, rows):
    registry = LpRegistry()
    for lp in lps:
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    for src, dst, lookahead_s in rows:
        table.declare(f"{src}->{dst}:m", LpId(src), LpId(dst), lookahead_s, "inline")
    return table


def _two_way(lookahead_s):
    return _table(("a", "b"), [("a", "b", lookahead_s), ("b", "a", lookahead_s)])


class _Recorded(ClockAuthority):
    """Records each grant the recovery branch issues."""

    def __init__(self, channels):
        super().__init__(channels)
        self.recovered = []
        self.daemon_grants = 0

    def _recover(self, i, g):
        self.recovered.append((str(i), g))
        return super()._recover(i, g)

    def _grant(self, i, g, recovered=False):
        self.daemon_grants += self._state[i] == NER and g == self._daemon[i]
        return super()._grant(i, g, recovered)


class _NowForWaitingTar(_Recorded):
    """The rejected rule: a peer waiting in TAR bounds the others by its clock."""

    def _row(self, i):
        return tuple(
            (
                j,
                (self._now[j] if self._state[j] == TAR else self._n(j))
                + self._channels.distance(j, i),
            )
            for j in self._ids
            if j != i
        )


class _NeighbourRow(_Recorded):
    """The rejected rule: only direct channels bound a peer, not paths."""

    def _row(self, i):
        return tuple(
            (
                j,
                self._n(j)
                + min(
                    (
                        c.lookahead_s
                        for c in self._channels.channels_from(j)
                        if c.target == i
                    ),
                    default=INF,
                ),
            )
            for j in self._ids
            if j != i
        )


def _grants(replies):
    return [(str(i), g) for i, g, _ in replies]


# --- the grant rule -----------------------------------------------------------


def test_a_receiver_is_held_until_its_sender_reports_and_gets_exactly_its_messages():
    table = _table(("decode", "prefill"), [("prefill", "decode", 0.5)])
    ca = ClockAuthority(table)
    prefill, decode = LpId("prefill"), LpId("decode")
    assert _grants(ca.on_request(prefill, TAR, 9.8, [])) == [("prefill", 9.8)]
    # prefill runs at 9.8, so a message into decode can still arrive at 10.3.
    assert ca.on_request(decode, TAR, 10.5, []) == []
    log = [("prefill->decode:m", 0, 9.8 + 0.5), ("prefill->decode:m", 1, 10.8)]
    replies = ca.on_request(prefill, TAR, 10.2, log)
    assert replies == [
        (prefill, 10.2, {}),
        (decode, 10.5, {"prefill->decode:m": [(0, 9.8 + 0.5)]}),
    ]
    (decode_row,) = [row for row in ca.lp_table() if row.lp == decode]
    assert decode_row.undelivered == (("prefill->decode:m", 1, 10.8),)


def _chain(cls):
    table = _table(
        ("decode", "frontend", "prefill"),
        [("frontend", "prefill", 0.1), ("prefill", "decode", 0.5)],
    )
    ca = cls(table)
    frontend, prefill = LpId("frontend"), LpId("prefill")
    assert _grants(ca.on_request(frontend, TAR, 9.0, [])) == [("frontend", 9.0)]
    assert ca.on_request(prefill, NER, INF, []) == []
    return ca


def test_an_idle_lp_in_the_middle_does_not_hide_the_one_behind_it():
    ca = _chain(ClockAuthority)
    decode = LpId("decode")
    (row,) = [row for row in ca.lp_table() if row.lp == decode]
    assert str(row.binding) == "frontend"
    assert dict(row.row)[LpId("frontend")] == pytest.approx(9.6)
    assert dict(row.row)[LpId("prefill")] == INF
    assert ca.on_request(decode, TAR, 9.7, []) == []
    assert _grants(_chain(ClockAuthority).on_request(decode, TAR, 9.5, [])) == [
        ("decode", 9.5)
    ]


@pytest.mark.parametrize("cls", [ClockAuthority, _NeighbourRow])
def test_a_rule_that_reads_only_neighbours_puts_a_legal_message_in_the_past(cls):
    ca = _chain(cls)
    frontend, prefill, decode = LpId("frontend"), LpId("prefill"), LpId("decode")
    held = ca.on_request(decode, TAR, 9.8, []) == []
    replies = ca.on_request(
        frontend, TAR, 10.0, [("frontend->prefill:m", 0, 9.0 + 0.1)]
    )
    assert (prefill, 9.0 + 0.1, {"frontend->prefill:m": [(0, 9.0 + 0.1)]}) in replies
    arrival = 9.0 + 0.1 + 0.5
    if cls is ClockAuthority:
        assert held
        ca.on_request(prefill, NER, INF, [("prefill->decode:m", 0, arrival)])
    else:
        assert not held
        with pytest.raises(BackdatedEvent, match="behind decode's clock at 9.8"):
            ca.on_request(prefill, NER, INF, [("prefill->decode:m", 0, arrival)])


def test_an_idle_lp_gets_no_reply_until_a_message_is_registered_for_it():
    ca = ClockAuthority(_table(("a", "b"), [("b", "a", 0.5)]))
    assert ca.on_request(A, NER, INF, []) == []
    assert _grants(ca.on_request(B, TAR, 1.0, [])) == [("b", 1.0)]
    assert ca.on_request(B, TAR, 2.0, []) == [(B, 2.0, {})]
    row, _ = ca.lp_table()
    assert (row.state, row.n) == (NER, INF)
    replies = ca.on_request(B, TAR, 5.0, [("b->a:m", 0, 2.5)])
    assert replies == [(A, 2.5, {"b->a:m": [(0, 2.5)]}), (B, 5.0, {})]


def test_an_ner_is_granted_its_own_target_when_that_comes_first():
    ca = ClockAuthority(_table(("a", "b"), [("b", "a", 0.5)]))
    ca.on_request(B, TAR, 3.0, [("b->a:m", 0, 0.5)])
    replies = ca.on_request(A, NER, 0.25, [])
    assert replies == [(A, 0.25, {"b->a:m": []})]
    assert ca.on_request(A, NER, INF, []) == [(A, 0.5, {"b->a:m": [(0, 0.5)]})]


# --- N by state: the named result ---------------------------------------------


def _named_result(cls):
    ca = cls(_two_way(0.5))
    first = ca.on_request(A, TAR, 0.8, [])
    second = ca.on_request(B, TAR, 0.9, [])
    grants = _grants(first + second)
    return (len(grants) - len(ca.recovered), len(ca.recovered)), grants, first


class TestNByState:
    def test_a_peer_waiting_in_tar_is_bounded_by_its_target(self):
        branches, grants, first = _named_result(_Recorded)
        assert first == []
        assert branches == (2, 0)
        assert grants == [("a", 0.8), ("b", 0.9)]

    def test_bounding_it_by_its_clock_needs_the_recovery_branch(self):
        branches, grants, first = _named_result(_NowForWaitingTar)
        assert first == []
        assert branches == (1, 1)
        assert grants == [("a", 0.8), ("b", 0.9)]


# --- recovery and serialization -----------------------------------------------


def test_zero_lookahead_is_granted_one_lp_at_a_time_in_name_order():
    ca = _Recorded(_two_way(0.0))
    assert ca.on_request(A, TAR, 10.3, []) == []
    assert ca.on_request(B, TAR, 10.3, []) == [(A, 10.3, {"b->a:m": []})]
    assert ca.recovered == [("a", 10.3)]
    replies = ca.on_request(A, NER, INF, [("a->b:m", 0, 10.3)])
    assert replies == [(B, 10.3, {"a->b:m": [(0, 10.3)]})]
    # b answers at the same instant: a's next round, not a backdated message.
    replies = ca.on_request(B, NER, INF, [("b->a:m", 0, 10.3)])
    assert replies == [(A, 10.3, {"b->a:m": [(0, 10.3)]})]


def test_the_bound_stops_a_recovery_grant():
    ca = ClockAuthority(_two_way(0.0), bound_s=10.0)
    assert ca.on_request(A, TAR, 10.3, []) == []
    with pytest.raises(ClockAbort, match="a grant to 10.3 for a passes"):
        ca.on_request(B, TAR, 10.3, [])


def test_every_lookahead_at_zero_is_serialized_and_not_an_error():
    names = ("alpha", "beta", "gamma")
    rows = [(s, d, 0.0) for s in names for d in names if s != d]
    ca = _Recorded(_table(names, rows))
    alpha, beta, gamma = (LpId(n) for n in names)
    assert ca.on_request(alpha, TAR, 1.0, []) == []
    assert ca.on_request(beta, TAR, 1.0, []) == []
    assert _grants(ca.on_request(gamma, TAR, 1.0, [])) == [("alpha", 1.0)]
    assert _grants(ca.on_request(alpha, TAR, 2.0, [])) == [("beta", 1.0)]
    assert _grants(ca.on_request(beta, TAR, 2.0, [])) == [("gamma", 1.0)]
    assert ca.recovered == [("alpha", 1.0), ("beta", 1.0)]


def test_one_participant_advances_straight_to_its_own_next_event():
    ca = _Recorded(_table(("only",), []))
    only = LpId("only")
    for kind, t in ((TAR, 4.0), (TAR, 9.0), (NER, 9.5)):
        assert ca.on_request(only, kind, t, []) == [(only, t, {})]
    assert ca.recovered == []
    assert ca.on_request(only, NER, INF, []) == [(only, INF, {})]


# --- daemon deadlines and the finish ------------------------------------------


def test_a_daemon_deadline_is_held_until_essential_time_reaches_it():
    ca = ClockAuthority(_two_way(0.5))
    assert ca.on_request(A, NER, INF, [], t_daemon=1.0) == []
    # Held: nothing essential has reached 1.0, and b's grant is not delayed.
    assert _grants(ca.on_request(B, TAR, 0.75, [])) == [("b", 0.75)]
    assert _grants(ca.on_request(B, TAR, 2.0, [])) == [("a", 1.0)]
    assert _grants(ca.on_request(A, NER, INF, [], t_daemon=3.0)) == [("b", 2.0)]
    # Every LP waits and a's next deadline is past every essential time.
    assert _grants(ca.on_request(B, NER, INF, [])) == [("a", INF), ("b", INF)]


def test_a_daemon_deadline_equal_to_the_horizon_fires():
    ca = ClockAuthority(_two_way(0.5))
    assert ca.on_request(A, NER, INF, [], t_daemon=1.0) == []
    assert _grants(ca.on_request(B, TAR, 1.0, [])) == [("a", 1.0), ("b", 1.0)]


def test_a_daemon_deadline_is_only_an_ner_and_never_behind_the_clock():
    ca = _one_way_at(0.5)
    with pytest.raises(ValueError, match="only NER has one"):
        ca.on_request(A, TAR, 1.0, [], t_daemon=2.0)
    with pytest.raises(BackdatedEvent, match="daemon deadline 0.25 behind its own"):
        ca.on_request(B, NER, INF, [], t_daemon=0.25)


def test_the_finish_grants_every_lp_infinity_once_and_then_changes_nothing():
    table = single_engine_table(admission_path="serving", ipc_s=1.0e-4, stream_s=2.0e-3)
    ca = ClockAuthority(table)
    engine, frontend, traffic = table.registry.ids()
    assert ca.on_request(engine, NER, INF, [], t_daemon=5.0) == []
    assert ca.on_request(frontend, NER, INF, []) == []
    replies = ca.on_request(traffic, NER, INF, [], t_daemon=16.0)
    assert replies == [
        (
            engine,
            INF,
            {"frontend->engine:control#dp0": [], "frontend->engine:request#dp0": []},
        ),
        (
            frontend,
            INF,
            {"engine->frontend:output#dp0": [], "traffic->frontend:http": []},
        ),
        (traffic, INF, {"frontend->traffic:stream": []}),
    ]
    table_at_end = ca.lp_table()
    assert (
        ca.on_request(engine, TAR, 1.0, [("frontend->engine:request#dp0", 7, 0.0)])
        == []
    )
    assert ca.lp_table() == table_at_end


def test_every_n_infinite_finishes_the_run():
    ca = _Recorded(_two_way(0.5))
    assert ca.on_request(A, NER, INF, []) == []
    assert ca.on_request(B, NER, INF, []) == [
        (A, INF, {"b->a:m": []}),
        (B, INF, {"a->b:m": []}),
    ]
    assert ca.recovered == []


# --- registration checks ------------------------------------------------------


def _one_way_at(b_now):
    """a -> b with lookahead 1; b has been granted `b_now`, a runs at 0."""
    ca = ClockAuthority(_table(("a", "b"), [("a", "b", 1.0)]))
    if b_now:
        assert _grants(ca.on_request(B, TAR, b_now, [])) == [("b", b_now)]
    return ca


REFUSALS = {
    "wrong sender": (0.0, B, TAR, 1.0, [("a->b:m", 0, 1.0)], "whose sender is a"),
    "repeated seq": (
        0.0,
        A,
        TAR,
        0.5,
        [("a->b:m", 0, 1.0), ("a->b:m", 0, 1.0)],
        "logged seq 0 on a->b:m, which expects seq 1",
    ),
    "seq gap": (
        0.0,
        A,
        TAR,
        0.5,
        [("a->b:m", 1, 1.0)],
        "logged seq 1 on a->b:m, which expects seq 0",
    ),
    "behind the receiver": (
        0.5,
        A,
        TAR,
        0.5,
        [("a->b:m", 0, 0.25)],
        "behind b's clock at 0.5",
    ),
    "inside the lookahead": (
        0.0,
        A,
        TAR,
        0.5,
        [("a->b:m", 0, 0.75)],
        "plus the channel's lookahead 1.0",
    ),
    "behind its own clock": (
        0.5,
        B,
        NER,
        0.25,
        [],
        "asked for NER(0.25) behind its own clock",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_each_registration_check_raises_with_the_lp_table(case):
    b_now, lp, kind, t, log, needle = REFUSALS[case]
    ca = _one_way_at(b_now)
    with pytest.raises(BackdatedEvent) as abort:
        ca.on_request(lp, kind, t, log)
    assert needle in abort.value.reason
    assert [row.lp for row in abort.value.table] == [A, B]
    assert abort.value.table[1].now == b_now
    assert repr(abort.value.table[1]) in str(abort.value)


def test_a_waiting_participant_cannot_produce_a_request():
    ca = _one_way_at(0.0)
    assert ca.on_request(B, TAR, 5.0, []) == []
    with pytest.raises(BackdatedEvent, match="sent NER while waiting in TAR") as abort:
        ca.on_request(B, NER, 6.0, [])
    assert abort.value.table[1].state == TAR


@pytest.mark.parametrize(
    "kind, t",
    [(TAR, INF), (TAR, math.nan), (NER, math.nan), (NER, -INF), ("ADVANCE", 1.0)],
)
def test_a_request_that_is_not_a_time_is_refused(kind, t):
    with pytest.raises(ValueError):
        ClockAuthority(_two_way(0.5)).on_request(A, kind, t, [])


def test_the_safety_check_is_not_an_assert_statement():
    tree = ast.parse(Path(authority_module.__file__).read_text())
    lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert not lines, f"authority.py asserts at {lines}; raise instead"


# --- generated runs -----------------------------------------------------------

NAMES = ("a", "b", "c", "d")
STEPS = 6


def _lcg(x):
    return (x * 1103515245 + 12345) % (1 << 31)


def _generated_table(seed):
    lps = NAMES[: 2 + seed % 3]
    rows = []
    x = seed + 1
    for src in lps:
        for dst in lps:
            x = _lcg(x)
            if src != dst and x % 3:
                rows.append((src, dst, (0.0, 0.5, 1.0)[(x >> 8) % 3]))
    return _table(lps, rows)


def _request(table, lp, own, seed):
    """A legal LP's next request, a function of its own history only."""
    if own["k"] >= STEPS:
        # Some LPs keep a periodic housekeeping timer that sends nothing.
        period = (INF, 0.75, 1.5)[(seed + ord(lp.name[0])) % 3]
        return NER, INF, [], own["g"] + period
    x = _lcg(
        _lcg(seed * 7919 + own["k"] * 104729 + own["received"] * 31 + ord(lp.name[0]))
    )
    log = []
    for channel in table.channels_from(lp):
        x = _lcg(x)
        if x % 2:
            log.append(
                (channel.name, own["seq"][channel.name], own["g"] + channel.lookahead_s)
            )
            own["seq"][channel.name] += 1
    d = (0.0, 0.5, 1.0)[(x >> 8) % 3]
    return ((TAR, own["g"] + d), (NER, own["g"] + d), (NER, INF), (TAR, own["g"] + d))[
        (x >> 4) % 4
    ] + (log, INF)


PICKS = {
    "name order": lambda running, x: running[0],
    "reverse": lambda running, x: running[-1],
    "shuffled": lambda running, x: running[x % len(running)],
}


def _drive(seed, pick, cls=_Recorded, shuffle=0):
    """Run legal LPs to the end; return each LP's replies and every violation seen."""
    table = _generated_table(seed)
    ca = cls(table)
    ids = table.registry.ids()
    own = {
        i: {
            "k": 0,
            "g": 0.0,
            "received": 0,
            "seq": {c.name: 0 for c in table.channels_from(i)},
        }
        for i in ids
    }
    sent = {
        i: [] for i in ids
    }  # registered into i and not yet released, as the LPs see it
    replies = {i: [] for i in ids}
    violations = []
    running = list(ids)
    x = shuffle
    for _ in range(10000):
        x = _lcg(x)
        lp = pick(sorted(running), x)
        running.remove(lp)
        kind, t, log, t_daemon = _request(table, lp, own[lp], seed)
        for name, seq, arrival in log:
            dst = table.channel(name).target
            if arrival < own[dst]["g"]:
                violations.append(("into the past", name, seq, arrival, own[dst]["g"]))
            sent[dst].append((name, seq, arrival))
        for i, g, released in ca.on_request(lp, kind, t, log, t_daemon):
            got = [
                (name, seq, a) for name, pairs in released.items() for seq, a in pairs
            ]
            replies[i].append((g, got))
            if g == INF:
                continue
            for message in got:
                sent[i].remove(message)
            violations += [("stepped over", str(i), g, m) for m in sent[i] if m[2] <= g]
            violations += [("released early", str(i), g, m) for m in got if m[2] > g]
            own[i].update(
                k=own[i]["k"] + 1, g=g, received=own[i]["received"] + len(got)
            )
            running.append(i)
        if any(r and r[-1][0] == INF for r in replies.values()):
            return replies, violations, ca
    raise AssertionError(f"seed {seed} did not finish")


SEEDS = range(60)


def test_no_reply_grants_past_an_unreleased_arrival():
    recovered = strict = fired = held = 0
    for seed in SEEDS:
        for name, pick in PICKS.items():
            replies, violations, ca = _drive(seed, pick)
            assert violations == [], (seed, name, violations)
            assert all(r[-1][0] == INF for r in replies.values())
            recovered += len(ca.recovered)
            strict += sum(len(r) - 1 for r in replies.values()) - len(ca.recovered)
            fired += ca.daemon_grants
            held += sum(d < INF for d in ca._daemon.values())
    # Both branches are exercised, so neither half of the property is vacuous;
    # so are daemon deadlines that fire and ones still held at the finish.
    assert recovered > 0 and strict > 0
    assert fired > 0 and held > 0


def test_the_generated_runs_catch_a_rule_that_reads_only_neighbours():
    caught = 0
    for seed in SEEDS:
        try:
            _, violations, _ = _drive(seed, PICKS["name order"], cls=_NeighbourRow)
        except BackdatedEvent:
            caught += 1
            continue
        caught += bool(violations)
    assert caught > 0


def test_each_lps_replies_do_not_depend_on_the_order_running_lps_submit():
    for seed in SEEDS:
        runs = [_drive(seed, pick)[0] for pick in PICKS.values()]
        runs += [_drive(seed, PICKS["shuffled"], shuffle=s)[0] for s in range(1, 9)]
        assert all(run == runs[0] for run in runs), seed


def _rule_n(ca, row):
    """N of `row`'s LP from its state alone, as the rule defines it."""
    if row.state == authority_module.RUNNING:
        return row.now
    if row.state == TAR:
        return row.target
    return min([row.target, ca._daemon[row.lp]] + [a for _, _, a in row.undelivered])


class _FromScratch(_Recorded):
    """Recomputes every LP's N from the LP table before each grant scan, and
    counts the scans at which the authority's own N had drifted from it."""

    drifted = 0

    def _grant_due(self):
        rule = {row.lp: _rule_n(self, row) for row in self.lp_table()}
        self.drifted += self._nv != rule
        self._nv.update(rule)
        return super()._grant_due()


def test_the_kept_n_grants_what_n_recomputed_at_every_request_grants():
    for seed in SEEDS:
        for pick in PICKS.values():
            kept, _, finished = _drive(seed, pick)
            recomputed, _, ca = _drive(seed, pick, cls=_FromScratch)
            assert ca.drifted == 0, seed
            assert kept == recomputed, seed
            # The table a run ends with reads every N from the state too.
            table = finished.lp_table()
            n = {row.lp: _rule_n(finished, row) for row in table}
            for row in table:
                d = finished._channels.distance
                assert row.row == tuple(
                    (j, n[j] + d(j, row.lp)) for j in n if j != row.lp
                ), seed


# --- local grants -------------------------------------------------------------

AB, BA = "a->b:m", "b->a:m"


def _ceilinged(**kw):
    """a <-> b, a sending at 0.5 and b back at 0.25, with a timeline."""
    table = _table(("a", "b"), [("a", "b", 0.5), ("b", "a", 0.25)])
    return ClockAuthority(table, timeline=TimelineLog(), **kw)


def _by_lp(ca):
    """The authority's timeline, each LP's records in its own order."""
    return sorted(ca.timeline.records(), key=lambda r: (str(r.lp), r.time_to))


def _first_grant(ca, a_target, log, ahead=False):
    """a waits in TAR(`a_target`) having sent `log`; b's NER is granted."""
    assert ca.on_request(A, TAR, a_target, log) == []
    return ca.on_request(B, NER, INF, [], ahead=ahead)


@pytest.mark.parametrize(
    "a_target, log, ceiling",
    [
        (1.0, [(AB, 0, 0.5)], 0.5 + 0.75),  # b's cycle, 0.25 + 0.5
        (0.6, [(AB, 0, 0.5)], 0.6 + 0.5),  # LBTS(b), a's target plus 0.5
        (1.0, [(AB, 0, 0.5), (AB, 1, 0.9)], 0.9),  # the arrival still unreleased
    ],
)
def test_a_ceiling_is_the_least_of_lbts_the_cycle_and_the_earliest_arrival(
    a_target, log, ceiling
):
    lp, g, released = _first_grant(_ceilinged(ceilings=True), a_target, log)[0]
    assert (lp, g, released) == (B, 0.5, {AB: [(0, 0.5)]})
    assert released.ceiling == pytest.approx(ceiling)
    assert _first_grant(_ceilinged(), a_target, log)[0][2].ceiling is None


def test_steps_below_the_ceiling_replay_as_the_grants_their_calls_would_have_had():
    """b takes 0.9 and its message ahead, then a TAR to 1.0, on its own; the
    authority records and counts them as if b had asked each time."""
    log = [(AB, 0, 0.5), (AB, 1, 0.9)]
    on = _ceilinged(ceilings=True)
    ((_, _, released),) = _first_grant(on, 1.0, log, ahead=True)
    assert released == {AB: [(0, 0.5), (1, 0.9)]} and released.ceiling == 1.25
    steps = [(NER, INF, 0.9, 0), (TAR, 1.0, 1.0, 0)]
    replies = on.on_request(B, TAR, 2.0, [(BA, 0, 1.25)], steps=steps)
    assert _grants(replies) == [("a", 1.0)]
    off = _ceilinged()
    _first_grant(off, 1.0, log)
    assert _grants(off.on_request(B, NER, INF, [])) == [("b", 0.9), ("a", 1.0)]
    assert _grants(off.on_request(B, TAR, 1.0, [])) == [("b", 1.0)]
    assert off.on_request(B, TAR, 2.0, [(BA, 0, 1.25)]) == []
    assert _by_lp(on) == _by_lp(off)
    assert on.grants == off.grants == {"a": 1, "b": 3}
    assert on.requests_served == {"a": 1, "b": 2}
    assert off.requests_served == {"a": 1, "b": 4}


def test_a_step_out_of_bounds_or_past_a_registered_message_aborts():
    def ceilinged():
        ca = _ceilinged(ceilings=True)
        _first_grant(ca, 1.0, [(AB, 0, 0.5)])  # b's ceiling: 1.25
        return ca

    for steps, needle in [
        ([(TAR, 1.25, 1.25, 0)], "below its ceiling 1.25"),
        ([(NER, 0.75, 0.8, 0)], "answered 0.8"),
        ([(TAR, 0.75, 0.75, 1)], "over a log of 0"),
    ]:
        with pytest.raises(ClockAbort, match=needle):
            ceilinged().on_request(B, NER, INF, [], steps=steps)
    ca = ceilinged()
    ca._undelivered[AB][1] = 0.95  # a message the ceiling should have kept out
    with pytest.raises(BackdatedEvent, match=r"past \(0.95"):
        ca.on_request(B, NER, INF, [], steps=[(TAR, 1.0, 1.0, 0)])
    ca = _ceilinged(ceilings=True)
    _first_grant(ca, 1.0, [(AB, 0, 0.5), (AB, 1, 0.9)], ahead=True)
    with pytest.raises(ClockAbort, match="sent ahead of it unreached"):
        ca.on_request(B, NER, INF, [])


def test_a_ceiling_stops_at_the_bound_so_a_tar_past_it_aborts_as_without_one():
    """b's ceiling, 1.25 with no bound, stops at the bound 1.0: b asks for
    1.1, which aborts as it does without ceilings, and a step to 1.1 that b
    reports having taken on its own aborts too."""
    for ceilings in (False, True):
        ca = _ceilinged(ceilings=ceilings, bound_s=1.0)
        ((_, _, released),) = _first_grant(ca, 1.0, [(AB, 0, 0.5)])
        assert released.ceiling == (1.0 if ceilings else None)
        with pytest.raises(ClockAbort, match="a grant to 1.1 for b passes"):
            ca.on_request(B, TAR, 1.1, [])
    ca = _ceilinged(ceilings=True, bound_s=1.0)
    _first_grant(ca, 1.0, [(AB, 0, 0.5)])
    with pytest.raises(ClockAbort, match=r"TAR\(1.1\) answered 1.1, .* ceiling 1.0"):
        ca.on_request(B, NER, INF, [], steps=[(TAR, 1.1, 1.1, 0)])


def _drive_below_ceilings(seed, pick, ceilings, shuffle=0):
    """`_drive`'s LPs, each answering its own requests below its last grant's
    ceiling as `atom.utils.clock.LPRuntime` does, and asking for messages
    ahead; returns each LP's grants as it sees them, its own included, the
    authority and how many messages were sent ahead."""
    table = _generated_table(seed)
    ca = ClockAuthority(table, timeline=TimelineLog(), ceilings=ceilings)
    ids = table.registry.ids()
    own = {
        i: {"k": 0, "g": 0.0, "received": 0}
        | {"seq": {c.name: 0 for c in table.channels_from(i)}}
        for i in ids
    }
    lp = {i: {"ceiling": -INF, "ahead": [], "log": [], "steps": []} for i in ids}
    seen = {i: [] for i in ids}

    def took(i, g, got):
        seen[i].append((g, sorted(got)))
        own[i].update(k=own[i]["k"] + 1, g=g, received=own[i]["received"] + len(got))

    running, x, sent_ahead = list(ids), shuffle, 0
    for _ in range(100000):
        x = _lcg(x)
        i = pick(sorted(running), x)
        running.remove(i)
        mine = lp[i]
        while True:
            kind, t, log, t_daemon = _request(table, i, own[i], seed)
            mine["log"] += log
            first = mine["ahead"][0][0] if mine["ahead"] else INF
            to = t if kind == TAR else min(t, t_daemon, first)
            essential = kind == TAR or t < INF or mine["ahead"]
            if not (own[i]["g"] <= to < mine["ceiling"] and essential):
                break
            reached = [m for m in mine["ahead"] if m[0] <= to]
            mine["ahead"] = mine["ahead"][len(reached) :]
            mine["steps"].append((kind, t, to, len(mine["log"])))
            took(i, to, [(c, s, a) for a, c, s in reached])
        steps, log = mine["steps"], mine["log"]
        mine["steps"], mine["log"] = [], []
        ahead = ceilings and kind == NER
        replies = ca.on_request(i, kind, t, log, t_daemon, None, steps, ahead)
        for j, g, released in replies:
            theirs = lp[j]
            pairs = [(c, s, a) for c, r in released.items() for s, a in r]
            reached = [(c, s, a) for a, c, s in theirs["ahead"]]
            got = [m for m in pairs if m[2] <= g] + reached
            theirs["ahead"] = sorted((a, c, s) for c, s, a in pairs if a > g)
            sent_ahead += len(theirs["ahead"])
            ceiling = getattr(released, "ceiling", None)
            theirs["ceiling"] = -INF if ceiling is None else ceiling
            if g == INF:
                seen[j].append((g, sorted(got)))
                continue
            took(j, g, got)
            running.append(j)
        if any(r and r[-1][0] == INF for r in seen.values()):
            return seen, ca, sent_ahead
    raise AssertionError(f"seed {seed} did not finish")


def test_local_grants_leave_every_lps_grants_and_the_timeline_as_they_are():
    asked = answered = ahead = 0
    for seed in SEEDS:
        for pick in PICKS.values():
            off, off_ca, _ = _drive_below_ceilings(seed, pick, False)
            on, on_ca, sent = _drive_below_ceilings(seed, pick, True)
            assert on == off, seed
            assert on_ca.grants == off_ca.grants, seed
            assert _by_lp(on_ca) == _by_lp(off_ca), seed
            asked += sum(off_ca.requests_served.values())
            answered += sum(on_ca.requests_served.values())
            ahead += sent
    # The LPs answered some of their own requests, some below a message ahead.
    assert answered < asked and ahead > 0


# --- a DP group's rounds, one call a round or each run in one call ------------

GROUP = LpId("e")
GROUP_STEPS = 4


def _group_table(seed):
    """`_generated_table`'s LPs, two or three, beside `e`, a DP group of two or
    three members, each owning a channel each way to every other LP."""
    lps = NAMES[: 2 + seed % 2]
    members = ("m0", "m1", "m2")[: 2 + seed // 2 % 2]
    rows, x = [], seed + 1
    for src in lps:
        for dst in lps:
            x = _lcg(x)
            if src != dst and x % 3:
                rows.append((src, dst, "m", (0.0, 0.5, 1.0)[(x >> 8) % 3]))
    for m in members:
        for other in lps:
            for src, dst in ((other, "e"), ("e", other)):
                x = _lcg(x)
                rows.append((src, dst, m, (0.0, 0.5, 1.0)[(x >> 8) % 3]))
    registry = LpRegistry()
    for lp in (*lps, "e"):
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    for src, dst, tag, lookahead_s in rows:
        table.declare(
            f"{src}->{dst}:{tag}", LpId(src), LpId(dst), lookahead_s, "inline"
        )
    owned = {m: [f"{s}->{d}:{m}" for s, d, tag, _ in rows if tag == m] for m in members}
    return table, owned


def _group_step(seed, k, base, members):
    """Step `k` of the group from its clock `base`: each member's return and
    pull, which every member knows, as from the group's gather."""
    x = _lcg(seed * 7919 + k * 104729 + 17)
    ends = {}
    for m in members:
        x = _lcg(x)
        end = base + (0.0, 0.5, 1.0)[(x >> 8) % 3]
        ends[m] = (end, end + (0.0, 0.25, 0.5)[(x >> 12) % 3])
    return ends


def _drive_group(seed, pick, batched, shuffle=0):
    """`_drive`'s LPs beside `e`, whose members take `GROUP_STEPS` steps, then
    idle. A member asks a step's rounds up to its own pull, sends, and asks the
    rest: each run of rounds in one call when `batched`, else one round a call.
    Returns what each LP and member saw, a member's run of rounds as its last
    grant with every message the run released, and the authority."""
    table, owned = _group_table(seed)
    ca = ClockAuthority(table, timeline=TimelineLog(), members={GROUP: owned})
    own = {
        i.name: {"k": 0, "g": 0.0, "received": 0}
        | {"seq": {c.name: 0 for c in table.channels_from(i)}}
        for i in table.registry.ids()
        if i != GROUP
    }
    group = {m: {"k": 0, "g": 0.0, "runs": [], "asked": [], "got": []} for m in owned}
    logs = {m: [] for m in owned}
    seq = {c: 0 for names in owned.values() for c in names}
    seen = {name: [] for name in [*own, *owned]}
    running, x = sorted(seen), shuffle
    for _ in range(100000):
        x = _lcg(x)
        name = pick(sorted(running), x)
        running.remove(name)
        if name in own:
            lp, member, rounds = LpId(name), None, []
            kind, t, log, t_daemon = _request(table, lp, own[name], seed)
        else:
            lp, member, me = GROUP, name, group[name]
            if not me["asked"] and not me["runs"] and me["k"] < GROUP_STEPS:
                ends = _group_step(seed, me["k"], me["g"], owned)
                every = sorted({t for end in ends.values() for t in end})
                pull = ends[name][1]
                mine = ([t for t in every if t <= pull], [t for t in every if t > pull])
                me["runs"], me["k"] = [run for run in mine if run], me["k"] + 1
            if not me["asked"] and me["runs"]:
                me["asked"] = me["runs"].pop(0)
            asked = me["asked"] if batched else me["asked"][:1]
            kind, t, t_daemon = (TAR, asked[0], INF) if asked else (NER, INF, INF)
            rounds, log, logs[name] = asked[1:], logs[name], []
        for to, g, released in ca.on_request(
            lp, kind, t, log, t_daemon, member, (), False, rounds
        ):
            got = [(c, s, a) for c, r in released.items() for s, a in r]
            key = to[1] if isinstance(to, tuple) else to.name
            if g == INF:
                seen[key].append((g, sorted(got)))
                continue
            running.append(key)
            if key in own:
                seen[key].append((g, sorted(got)))
                mine = own[key]
                mine.update(k=mine["k"] + 1, g=g, received=mine["received"] + len(got))
                continue
            me, was = group[key], group[key]["asked"]
            me["g"], me["got"] = g, me["got"] + got
            me["asked"] = [r for r in was if r > g]
            if me["asked"]:
                continue  # one round a call: the run's next round
            seen[key].append((g, sorted(me["got"])))
            me["got"] = []
            y = _lcg(seed * 31 + len(seen[key]) * 7 + ord(key[-1]))
            for c in owned[key] if was else ():  # a run ended: send from its clock
                y = _lcg(y)
                if table.channel(c).source == GROUP and y % 2:
                    logs[key].append((c, seq[c], g + table.lookahead(c)))
                    seq[c] += 1
        if any(r and r[-1][0] == INF for r in seen.values()):
            return seen, ca
    raise AssertionError(f"seed {seed} did not finish")


def _timeline(ca):
    """The authority's timeline, each LP's records in its own order, with each
    record's sends sorted: the members' calls carry them in submit order."""
    return [
        (r.lp, r.time_from, r.time_to, r.kind, r.recovered, r.released, sorted(r.sent))
        for r in _by_lp(ca)
    ]


def test_a_members_rounds_in_one_call_leave_every_grant_as_one_call_a_round():
    calls = {False: 0, True: 0}
    for seed in SEEDS:
        base, base_ca = _drive_group(seed, PICKS["name order"], False)
        for batched in (False, True):
            runs = [_drive_group(seed, pick, batched) for pick in PICKS.values()]
            runs += [
                _drive_group(seed, PICKS["shuffled"], batched, shuffle=s)
                for s in range(1, 5)
            ]
            for seen, ca in runs:
                assert seen == base, (seed, batched)
                assert ca.grants == base_ca.grants, (seed, batched)
                assert _timeline(ca) == _timeline(base_ca), (seed, batched)
            calls[batched] += runs[0][1].requests_served["e"]
    # Fewer calls for the same grants.
    assert calls[True] < calls[False]
