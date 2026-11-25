"""Wide local grants (#3968, #4135): a TAR's messages below its ceiling go
ahead, the cycle counts only first hops that can react, and with reactions LBTS
and the cycle follow what each LP declares it can provoke. Every LP must see
the grants it sees with plain local grants."""

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
from atom.compass.clock.channels import (
    ROUTER_REACTIONS,
    kind_of,
    prefill_decode_table,
)

INF = math.inf
R = LpId("r")
FLOOR = 2**-14


def _lcg(x):
    return (x * 1103515245 + 12345) % (1 << 31)


def test_a_stream_event_at_the_router_comes_back_to_a_frontend_only_through_traffic():
    table = prefill_decode_table(
        admission_s=FLOOR,
        ipc_s=1e-5,
        stream_s=FLOOR,
        router_s=1.5e-3,
        kv_write_req_s=1e-4,
        dp=(2, 4),
        router_in_s=1e-5,
        prefill=2,
        decode=2,
    )
    fd = LpId("frontend-D0")
    router = LpId("router")
    assert table.declares(router) and not table.declares(fd)
    relay = 1.5e-3 - 1e-5
    # stream on to traffic, traffic's next request, the router's cancel as a relay
    back = (FLOOR + 1.5e-3 - 1e-5) + 1e-5 + relay
    assert table.reach("frontend-D0->router:stream", fd) == pytest.approx(back)
    assert table.reach("frontend-P0->router:relay", fd) == pytest.approx(relay)
    assert table.reach("traffic->router:http", fd) == pytest.approx(relay)
    # LP-level distance takes the stream in and the relay out
    assert table.distance(router, fd) == pytest.approx(relay)
    assert table.reaction_distance(router, fd) == pytest.approx(relay)
    # the cycle through the router: today 1.5 ms, over the reactions twice that
    assert 1e-5 + table.distance(router, fd) == pytest.approx(1.5e-3)
    assert 1e-5 + table.reach("frontend-D0->router:stream", fd) > 3e-3


def _forwarding_table(seed):
    """`a`, `b`, `c` with channels among them, and `r`, which forwards each
    message it takes on one of its own kind, `p` or `q`."""
    lps = ("a", "b", "c")
    registry = LpRegistry()
    for lp in (*lps, "r"):
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    x = seed + 1
    for src in lps:
        for dst in lps:
            x = _lcg(x)
            if src != dst and x % 3:
                lookahead = (0.25, 0.5, 1.0)[(x >> 8) % 3]
                name = f"{src}->{dst}:m"
                table.declare(name, LpId(src), LpId(dst), lookahead, "inline")
    for lp in lps:
        for kind in ("p", "q"):
            x = _lcg(x)
            if x % 4:
                table.declare(f"{lp}->r:{kind}", LpId(lp), R, 0.25, "inline")
            x = _lcg(x)
            if x % 4:
                lookahead = (0.5, 1.0, 1.5)[(x >> 8) % 3]
                table.declare(f"r->{lp}:{kind}", R, LpId(lp), lookahead, "inline")
    table.mark_reactions(R, {"p": ("p",), "q": ("q",)})
    return table


def _request(table, lp, own, seed, steps=6):
    """A legal LP's next request, a function of its own history only."""
    if own["k"] >= steps:
        return NER, INF, [], INF
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


def _drive(seed, wide):
    """The LPs to the end, each answering its own requests below its last
    grant's ceiling as `atom.utils.clock.LPRuntime` does; `r` forwards each
    message at its arrival. Returns each LP's grants as it sees them and the
    authority."""
    table = _forwarding_table(seed)
    ca = ClockAuthority(table, timeline=TimelineLog(), ceilings=True, wide=wide)
    ids = table.registry.ids()
    own = {
        i: {"k": 0, "g": 0.0, "received": 0, "forward": []}
        | {"seq": {c.name: 0 for c in table.channels_from(i)}}
        for i in ids
    }
    lp = {i: {"ceiling": -INF, "ahead": [], "log": [], "steps": []} for i in ids}
    seen = {i: [] for i in ids}

    def took(i, g, got):
        seen[i].append((g, sorted(got)))
        own[i].update(k=own[i]["k"] + 1, g=g, received=own[i]["received"] + len(got))
        if i != R:
            return
        for name, seq, a in sorted(got, key=lambda m: (m[2], m[0], m[1])):
            kind = kind_of(name)
            outs = [c for c in table.channels_from(R) if kind_of(c.name) == kind]
            if outs:
                out = outs[seq % len(outs)]
                sent = (out.name, own[R]["seq"][out.name], a + out.lookahead_s)
                own[R]["forward"].append(sent)
                own[R]["seq"][out.name] += 1

    def request(i):
        if i == R:
            log, own[R]["forward"] = own[R]["forward"], []
            return NER, INF, log, INF
        return _request(table, i, own[i], seed)

    running = list(ids)
    for _ in range(100000):
        i = sorted(running)[0]
        running.remove(i)
        mine = lp[i]
        while True:
            kind, t, log, t_daemon = request(i)
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
        replies = ca.on_request(i, kind, t, log, t_daemon, None, steps, kind == NER)
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
            took(j, g, got)
            running.append(j)
        if any(r and r[-1][0] == INF for r in seen.values()):
            return seen, ca
    raise AssertionError(f"seed {seed} did not finish")


def test_wide_local_grants_leave_every_lps_grants_and_the_timeline_as_they_are():
    served = {0: 0, 1: 0, 2: 0}
    for seed in range(40):
        plain, plain_ca = _drive(seed, 0)
        for wide in (1, 2):
            seen, ca = _drive(seed, wide)
            assert seen == plain, (seed, wide)
            assert ca.grants == plain_ca.grants, (seed, wide)
            served[wide] += sum(ca.requests_served.values())
        served[0] += sum(plain_ca.requests_served.values())
    # Each widening answers more of the LPs' own requests.
    assert served[2] < served[1] < served[0]


def test_the_router_declares_its_reactions_and_wide_needs_ceilings():
    assert ROUTER_REACTIONS["stream"] == ("stream",)
    with pytest.raises(ValueError, match="wide grants"):
        ClockAuthority(_forwarding_table(0), wide=1)
