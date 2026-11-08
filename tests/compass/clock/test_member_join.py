# SPDX-License-Identifier: MIT
"""`ClockAuthority` with members: one LP whose time is asked for by several processes.

A data-parallel group is one engine LP, and each rank holds its own runtime for
it, so the authority joins the ranks' calls into one request per round. The
groups below defend:

* **The join.** No grant before every member has called, and a member's send
  log is registered on its own call, so no LP is granted past it meanwhile.
* **The combination.** NER takes the least target and the least daemon
  deadline over the members, and each member is released only its own
  channels.
* **The refusals**, each naming the LP and both members, and the ownership the
  caller declares: every channel into or out of the LP has exactly one owning
  member, and a member sends only on its own.
* **One member is no member.** It is granted exactly as an LP with none.
* **The deadlock the join exists for.** With one LP per rank, a rank woken by a
  message waits in the lockstep all-reduce for a rank that waits in NER with no
  message; with the join, both ranks are woken together.
* **Later rounds in one call.** A member's TAR naming later rounds leaves every
  grant and the timeline as one call a round does, and is answered once, at
  its last round, with what each round released to it; a queued round that
  cannot join is refused as a call would be.
"""

import math
import re

import pytest

from atom.compass.clock import (
    NER,
    TAR,
    ChannelTable,
    ClockAbort,
    ClockAuthority,
    LpId,
    LpRegistry,
    TimelineLog,
    single_engine_table,
)

INF = math.inf
ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
RANKS = ("dp0", "dp1")
STEP_S = 2.0


def _dp_table(engines):
    """`frontend` and one engine per rank, or one engine for every rank, lookahead 1."""
    registry = LpRegistry()
    for lp in ("frontend", *dict.fromkeys(engines)):
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    for engine, rank in zip(engines, RANKS):
        e = LpId(engine)
        table.declare(f"frontend->{engine}:request#{rank}", FRONTEND, e, 1.0, "thread")
        table.declare(f"{engine}->frontend:output#{rank}", e, FRONTEND, 1.0, "thread")
    return table


def _own(rank):
    """The channels rank `rank` owns on the joined engine."""
    return (f"frontend->engine:request#{rank}", f"engine->frontend:output#{rank}")


def _joined(members=None):
    members = members or {rank: _own(rank) for rank in RANKS}
    return ClockAuthority(_dp_table(("engine", "engine")), members={ENGINE: members})


def _grants(replies):
    return [(i[1], g) for i, g, _ in replies]


def test_no_grant_until_every_member_calls_and_none_past_a_logged_arrival():
    ca = _joined()
    assert ca.on_request(FRONTEND, NER, 10.0, []) == []
    # dp0 alone: the engine keeps running at 0, so the frontend may not take 10.
    assert ca.on_request(ENGINE, TAR, 9.5, [], member="dp0") == []
    out = "engine->frontend:output#dp1"
    replies = ca.on_request(ENGINE, TAR, 9.5, [(out, 0, 1.0)], member="dp1")
    assert replies == [
        (FRONTEND, 1.0, {out: [(0, 1.0)], "engine->frontend:output#dp0": []})
    ]


def test_ner_joins_to_the_least_target_and_deadline_and_releases_each_its_own():
    ca = _joined()
    req0, req1 = "frontend->engine:request#dp0", "frontend->engine:request#dp1"
    assert ca.on_request(FRONTEND, NER, INF, [(req1, 0, 1.0)]) == []
    assert ca.on_request(ENGINE, NER, 3.0, [], member="dp0") == []
    assert ca.on_request(ENGINE, NER, 2.0, [], member="dp1") == [
        ((ENGINE, "dp0"), 1.0, {req0: []}),
        ((ENGINE, "dp1"), 1.0, {req1: [(0, 1.0)]}),
    ]
    ca.on_request(ENGINE, NER, 3.0, [], member="dp0")
    assert _grants(ca.on_request(ENGINE, NER, 2.0, [], member="dp1")) == [
        ("dp0", 2.0),
        ("dp1", 2.0),
    ]
    ca.on_request(ENGINE, NER, INF, [], t_daemon=4.0, member="dp0")
    assert _grants(ca.on_request(ENGINE, NER, 5.0, [], t_daemon=6.0, member="dp1")) == [
        ("dp0", 4.0),
        ("dp1", 4.0),
    ]


@pytest.mark.parametrize(
    "first, second, refusal",
    [
        (
            ("dp0", TAR, 1.0),
            ("dp1", TAR, 2.0),
            "dp1 asked TAR(2.0) while dp0 asked TAR(1.0)",
        ),
        (("dp0", TAR, 1.0), ("dp1", NER, 1.0), "dp1 asked NER while dp0 asked TAR"),
        (("dp1", TAR, 1.0), ("dp1", TAR, 1.0), "dp1 called again before dp0 called"),
    ],
    ids=["unequal-target", "mixed-kinds", "call-count"],
)
def test_a_round_that_cannot_join_is_refused_naming_the_lp_and_both_members(
    first, second, refusal
):
    ca = _joined()
    ca.on_request(FRONTEND, NER, INF, [])
    member, kind, t = first
    assert ca.on_request(ENGINE, kind, t, [], member=member) == []
    member, kind, t = second
    with pytest.raises(ClockAbort, match="^engine: " + re.escape(refusal)):
        ca.on_request(ENGINE, kind, t, [], member=member)


@pytest.mark.parametrize(
    "lp, member", [(ENGINE, None), (ENGINE, "dp2"), (FRONTEND, "dp0")]
)
def test_a_call_from_no_member_of_the_lp_is_refused(lp, member):
    with pytest.raises(KeyError, match="was called by member"):
        _joined().on_request(lp, NER, INF, [], member=member)


def test_members_are_named_by_the_caller_not_by_their_channels():
    ca = _joined({"rank0": _own("dp0"), "rank1": _own("dp1")})
    req0, req1 = "frontend->engine:request#dp0", "frontend->engine:request#dp1"
    ca.on_request(FRONTEND, NER, INF, [(req1, 0, 1.0)])
    assert ca.on_request(ENGINE, NER, INF, [], member="rank0") == []
    assert ca.on_request(ENGINE, NER, INF, [], member="rank1") == [
        ((ENGINE, "rank0"), 1.0, {req0: []}),
        ((ENGINE, "rank1"), 1.0, {req1: [(0, 1.0)]}),
    ]


def test_a_send_on_a_channel_another_member_owns_is_refused_naming_its_owner():
    out0 = "engine->frontend:output#dp0"
    refusal = f"engine: dp1 logged a send on {out0}, which dp0 owns"
    with pytest.raises(ClockAbort, match="^" + re.escape(refusal)):
        _joined().on_request(ENGINE, NER, INF, [(out0, 0, 1.0)], member="dp1")


@pytest.mark.parametrize(
    "members, refusal",
    [
        (
            {"dp0": _own("dp0"), "dp1": _own("dp1")[:1]},
            (
                "engine->frontend:output#dp1 goes into or out of engine, and none of "
                "its members dp0, dp1 owns it"
            ),
        ),
        (
            {"dp0": _own("dp0") + _own("dp1")[:1], "dp1": _own("dp1")},
            (
                "frontend->engine:request#dp1 goes into or out of engine, and its "
                "members dp0 and dp1 each own it"
            ),
        ),
        (
            {"dp0": _own("dp0") + ("engine->frontend:output#dp2",), "dp1": _own("dp1")},
            (
                "dp0 of engine owns engine->frontend:output#dp2, which neither goes "
                "into nor comes out of engine"
            ),
        ),
    ],
    ids=["unowned", "doubly-owned", "not-the-lps"],
)
def test_a_channel_not_owned_by_exactly_one_member_is_refused(members, refusal):
    with pytest.raises(ValueError, match="^" + re.escape(refusal) + "$"):
        _joined(members)


def test_an_empty_member_list_is_refused():
    with pytest.raises(ValueError, match="^engine is declared with an empty member"):
        ClockAuthority(_dp_table(("engine", "engine")), members={ENGINE: {}})


def test_a_one_member_lp_is_granted_exactly_as_an_lp_with_none():
    req, out = "frontend->engine:request#dp0", "engine->frontend:output#dp0"
    script = [
        (LpId("traffic"), NER, INF, []),
        (FRONTEND, NER, INF, [(req, 0, 1.0)]),
        (ENGINE, NER, INF, []),
        (ENGINE, TAR, 3.0, []),
        (ENGINE, NER, INF, [(out, 0, 4.0)]),
        (FRONTEND, NER, INF, []),
    ]

    def run(members, member):
        ca = ClockAuthority(
            single_engine_table(admission_path="serving", ipc_s=1.0, stream_s=1.0),
            members=members,
        )
        return [
            (i if isinstance(i, LpId) else i[0], g, released)
            for lp, kind, t, log in script
            for i, g, released in ca.on_request(
                lp, kind, t, log, member=member if lp == ENGINE else None
            )
        ]

    plain = run(None, None)
    assert [g for _, g, _ in plain] == [1.0, 3.0, 4.0, INF, INF, INF]
    every = (req, "frontend->engine:control#dp0", out)
    assert run({ENGINE: {"dp0": every}}, "dp0") == plain


# --- the deadlock the join exists for ----------------------------------------


def _lockstep(ca, ranks):
    """Drive a frontend that sends one request to rank 0, and two DP ranks in lockstep.

    `ranks` maps each rank to ``(lp, member)``. A rank granted time goes to the
    lockstep all-reduce, and leaves it only once every rank is there: then each
    steps (TAR) if any rank has work, else idles (NER). The frontend's request
    and both ranks' first idle calls are the requests at time 0. Returns what
    is left when no request remains: the ranks inside the all-reduce, the
    responses the frontend got, and whether the run finished.
    """
    req = f"frontend->{ranks['dp0'][0]}:request#dp0"
    ready = [(FRONTEND, None, NER, INF, [(req, 0, 1.0)])]
    ready += [(lp, m, NER, INF, []) for lp, m in ranks.values()]
    who = {(lp if m is None else (lp, m)): rank for rank, (lp, m) in ranks.items()}
    work, stepping, outbox, at_barrier = {}, {}, {}, {}
    responses = 0
    while ready:
        lp, member, kind, t, log = ready.pop(0)
        for i, g, released in ca.on_request(lp, kind, t, log, member=member):
            if g == INF:
                continue
            if i == FRONTEND:
                responses += sum(map(len, released.values()))
                ready.append((FRONTEND, None, NER, INF, []))
                continue
            rank = who[i]
            work[rank] = work.get(rank, 0) + sum(map(len, released.values()))
            if stepping.pop(rank, False) and work[rank]:
                work[rank] -= 1
                out = f"{ranks[rank][0]}->frontend:output#{rank}"
                outbox[rank] = [(out, 0, g + 1.0)]
            at_barrier[rank] = g
            if len(at_barrier) == len(ranks):
                step = any(work.values())
                for r, now in at_barrier.items():
                    lp_r, m_r = ranks[r]
                    stepping[r] = step
                    request = (TAR, now + STEP_S) if step else (NER, INF)
                    ready.append((lp_r, m_r, *request, outbox.pop(r, [])))
                at_barrier.clear()
    return tuple(sorted(at_barrier)), responses, ca.final_clocks is not None


def test_one_lp_per_rank_deadlocks_and_the_member_join_does_not():
    per_rank = ClockAuthority(_dp_table(("engine-dp0", "engine-dp1")))
    ranks = {r: (LpId(f"engine-{r}"), None) for r in RANKS}
    # Rank 0 is woken by the request and waits in the all-reduce; rank 1 waits
    # in NER with no message, so neither ever calls again.
    assert _lockstep(per_rank, ranks) == (("dp0",), 0, False)
    joined = {r: (ENGINE, r) for r in RANKS}
    assert _lockstep(_joined(), joined) == ((), 1, True)


# --- later rounds in one call -------------------------------------------------

REQ0, REQ1 = "frontend->engine:request#dp0", "frontend->engine:request#dp1"


def _two_rounds(batched):
    """Requests to both ranks, then a step of rounds 2.0 and 3.0 in which dp0
    returns at 3.0 and dp1 at 2.0, owing 3.0: dp0 asks both rounds in one call
    when `batched`, else one a call. Returns each member's replies and the
    authority."""
    ca = ClockAuthority(
        _dp_table(("engine", "engine")),
        timeline=TimelineLog(),
        members={ENGINE: {rank: _own(rank) for rank in RANKS}},
    )
    log = [(REQ0, 0, 1.5), (REQ0, 1, 2.5), (REQ1, 0, 1.5)]
    assert ca.on_request(FRONTEND, NER, INF, log) == []
    calls = [("dp0", 2.0, [3.0] if batched else []), ("dp1", 2.0, [])]
    calls += [("dp1", 3.0, [])] if batched else [("dp0", 3.0, []), ("dp1", 3.0, [])]
    replies = []
    for member, t, rounds in calls:
        got = ca.on_request(ENGINE, TAR, t, [], member=member, rounds=rounds)
        replies += [(i[1], g, released) for i, g, released in got]
    return replies, ca


def test_a_call_naming_later_rounds_is_answered_once_with_what_each_released():
    replies, ca = _two_rounds(batched=True)
    assert replies == [
        ("dp1", 2.0, {REQ1: [(0, 1.5)]}),
        ("dp0", 3.0, {REQ0: [(0, 1.5), (1, 2.5)]}),
        ("dp1", 3.0, {REQ1: []}),
    ]
    one_a_call, each = _two_rounds(batched=False)
    assert [(m, g) for m, g, _ in one_a_call] == [
        ("dp0", 2.0),
        ("dp1", 2.0),
        ("dp0", 3.0),
        ("dp1", 3.0),
    ]
    # Each round is still its own grant, releasing what it released.
    assert ca.timeline.records() == each.timeline.records()
    assert ca.grants == each.grants and ca.grants["engine"] == 2
    assert (ca.requests_served["engine"], each.requests_served["engine"]) == (3, 4)


@pytest.mark.parametrize(
    "lp, member, kind, t, rounds, needle",
    [
        (FRONTEND, None, TAR, 1.0, [2.0], "only a member's TAR asks later rounds"),
        (ENGINE, "dp0", NER, 1.0, [2.0], "only a member's TAR asks later rounds"),
        (ENGINE, "dp0", TAR, 2.0, [2.0], "each past the one before"),
        (ENGINE, "dp0", TAR, 2.0, [3.0, 2.5], "each past the one before"),
        (ENGINE, "dp0", TAR, 2.0, [INF], "must be a finite number"),
    ],
    ids=["no-member", "ner", "repeated", "descending", "infinite"],
)
def test_later_rounds_are_a_members_tar_each_past_the_one_before(
    lp, member, kind, t, rounds, needle
):
    ca = _joined()
    with pytest.raises(ValueError, match=re.escape(needle)):
        ca.on_request(lp, kind, t, [], member=member, rounds=rounds)
    assert ca.lp_table() == _joined().lp_table()


def test_a_queued_round_that_cannot_join_is_refused_as_a_call_would_be():
    def asked(*calls):
        ca = _joined()
        ca.on_request(FRONTEND, NER, INF, [])
        for member, t, rounds in calls:
            ca.on_request(ENGINE, TAR, t, [], member=member, rounds=rounds)
        return ca

    unequal = "engine: dp1 asked TAR(3.0) while dp0 asked TAR(2.0)"
    with pytest.raises(ClockAbort, match="^" + re.escape(unequal)):
        asked(("dp0", 1.0, [2.0]), ("dp1", 1.0, [3.0]))
    ca = asked(("dp0", 1.0, [2.0]), ("dp1", 1.0, []))
    again = "engine: dp0 called again before dp1 called"
    with pytest.raises(ClockAbort, match="^" + re.escape(again)):
        ca.on_request(ENGINE, NER, INF, [], member="dp0")
