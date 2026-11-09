# SPDX-License-Identifier: MIT
"""The Clock Authority: which waiting process may move its clock, to when, and
with which messages.

A logical process (LP) asks for time with one of two requests:

* ``TAR`` -- advance to ``t``: its owner has priced an event, and the grant is
  exactly ``t``;
* ``NER`` -- idle until ``t``, which may be ``+inf``: the grant is ``t``, its
  daemon deadline, or the earliest message registered for it, whichever comes
  first. A daemon deadline is a housekeeping timer (a metrics push, a scrape)
  that fires as usual but does not keep the run alive.

Each request carries the requester's send log since its previous request, as
``(channel, seq, arrival)``. The log is registered before the requester's state
changes, so every message an LP produced is known before its clock moves.

``N[j]`` is the earliest time LP *j* could still produce a message: its clock
while it runs, its target while it waits in TAR (its owner is inside this call
and sends nothing), and the least of its target, its daemon deadline and its
earliest unreleased message while it waits in NER. A waiting LP *i* is granted
exactly ``N[i]``, and only when ``N[i] < min over j != i of (N[j] + D(j->i))``.
``D`` is the channel table's least summed lookahead over every channel path, so
an idle LP between two others does not hide the first from the third. The comparison is strict: a
message not yet reported arrives no earlier than its sender's ``N`` plus that
distance, so a strict grant never reaches it. Waiting LPs are tried in
``(N, name)`` order and the scan restarts after every grant.

The essential horizon ``H`` is the largest TAR target, finite NER target or
registered arrival seen so far; it only grows. A waiting LP is grantable only
while ``N <= H``, so one whose ``N`` is its daemon deadline is held until some
essential time or arrival reaches that deadline. A held deadline delays no
essential grant: its ``N`` exceeds ``H``, and every essential ``N`` is at most
``H``. Because ``H`` is a function of the schedule, not of which running LP
reports first, a daemon deadline fires under every request order or none.

When every LP waits and the strict rule grants none, which only a
zero-lookahead cycle allows, the grantable LP with the least ``(N, name)`` is
granted: nothing is earlier, so nothing earlier can reach it. A message that
then lands at its receiver's current instant belongs to that instant's next
round. When every LP waits and none is grantable, nothing is undelivered and
no essential target is pending: the run is finished, every LP is granted
``+inf``, and the daemon deadlines still held never fire.

An authority built naming a client LP serves `client_runs` client runs, the
count ``+inf`` for no limit, and that test ends one; only the last one's end is
the finish. The end of an earlier one grants the client ``+inf``, which resigns
it, and every other LP its own clock with nothing released, each such reply
marked with the client run's index. The client's row then runs at the latest
clock, holding every other LP until the next client binds and asks.
`on_bind` gives a binding LP its row's clock and the next sequence number of
each channel out of it. `ended` holds every ended client run.

An LP may be declared with members: processes that each hold a runtime for it,
as the ranks of a data-parallel group do. A member names itself in each call,
and each channel into or out of the LP is owned by exactly one member, as the
caller declares; a member may log sends only on channels it owns. The members'
requests are joined into one per round: each member asks every round, all ask
one kind, every TAR asks the same target, and the joined NER's target and
daemon deadline are each the least over the members. A member's TAR may name
later rounds, ascending, which queue behind it: once a round is granted, each
member's next queued round joins the next one, so a round joins once it heads
every member's queue, and each is still its own grant. A call before the
round is complete registers its send log and nothing else: the LP stays
running, so its ``N`` stays at its clock. Every member gets the common grant,
carrying the releases on its own channels only; a member whose call named
later rounds gets one reply, at its last round's grant, with what each of its
rounds released to it.

A member may instead *park*: one call, before it dispatches a forward, that
stands in every later round as NER with no target and no daemon deadline, so
the members still running are joined without it. A TAR meeting a park is
refused as a TAR meeting an NER is. A parked member gets no reply while it is
parked; what a round releases on its channels is held for it. The call that
parks the last member is answered at once: every member leaves at the LP's
clock, the latest grant any of them was given, with the messages held for it.

A finite grant past the simulated-time bound aborts the run with the LP table.
A housekeeping timer nobody declared daemon keeps raising ``H`` and would
otherwise keep the run alive forever.

An authority built with `ceilings` gives each grant to an LP without members,
but a recovery's, a ceiling: the least of ``LBTS(i)``, the grant plus the LP's
shortest cycle ``min over j of D(i->j) + D(j->i)``, its earliest message still
unreleased, and the simulated-time bound. Nothing reaches the LP below it: a
message not yet sent arrives at or past ``LBTS(i)``, and one its own sends
provoke at or past the cycle; a time past the bound is asked for, and aborts
as above. Below it the LP's owner answers its own TARs and NERs, as this
authority would answer them, and reports each such step with its next request,
with how many of the request's logged sends it made before that step. The
authority replays each step as the grant its own request would have had:
registering the sends made before it, recording the reply with what it
releases and counting the grant, so the timeline is that of an LP that asked
every time. An NER asking for messages ahead has every message registered for
it below the ceiling sent with its grant, and the LP releases each in the step
that reaches it. A step outside the LP's clock and ceiling, or past a message
registered for it since, aborts the run.

With `ceilings`, an LP with members gets one at a grant at which some of its
members return while others still ask later rounds: the least of ``LBTS(i)``,
the grant plus the LP's shortest cycle, and the bound. Each later round below
it is granted at once, each its own grant, the members that returned standing
in it, since they owe it; what a round releases to such a member goes ahead in
that member's reply, which carries the ceiling. The member steps its own clock
through those rounds and reports its steps with its send log: in a ``REPORT``,
which gets no reply, or in its call for the rounds left. Until it has, the LP's
``N`` is no later than that member's clock, so no LP is granted past a send it
has yet to log. A request from such a member that does not report exactly
those rounds, each with its own target, aborts the run.

Nothing here reads a clock, opens a socket or starts a thread. The caller
carries requests in and replies out.
"""

import math
from dataclasses import dataclass, field

from .channels import ChannelTable
from .identity import LpId

TAR, NER, PARK, REPORT = "TAR", "NER", "PARK", "REPORT"
RUNNING = "running"
_STANDING = (PARK, math.inf, math.inf)  # a parked member's place in each round


@dataclass(frozen=True, slots=True)
class LpRow:
    """One LP as the authority sees it."""

    lp: LpId
    state: str  # running, TAR or NER
    now: float
    target: float | None
    n: float  # the earliest time it could still produce a message
    undelivered: tuple[tuple[str, int, float], ...]  # (channel, seq, arrival)
    row: tuple[tuple[LpId, float], ...]  # (j, N[j] + D(j->lp)), every other LP
    binding: LpId | None  # the first least term of `row`, None if none is finite


@dataclass(frozen=True, slots=True)
class ClientRun:
    """One ended client run: its client's clock at binding, the latest LP clock
    at its end, each LP's clock, finite grants and calls served by name at its
    end, and the stall diagnostics printed during it."""

    index: int
    start: float
    end: float
    final_clocks: tuple[tuple[LpId, float], ...]
    grants: dict = field(repr=False)
    diagnostics: int = 0
    requests_served: dict = field(default_factory=dict, repr=False)


class ClockAbort(Exception):
    """The run cannot be trusted any further. Carries every LP's row."""

    def __init__(self, reason: str, table: tuple[LpRow, ...]) -> None:
        super().__init__("\n".join([reason, *map(repr, table)]))
        self.reason = reason
        self.table = table


class BackdatedEvent(ClockAbort):
    """A request that would put a message or a clock into some LP's past."""


class Released(dict):
    """What a grant releases, ``{channel: [(seq, arrival)]}``, and as `ceiling`
    the time below which its LP may advance on its own, or None."""

    def __init__(self, released=(), /) -> None:
        super().__init__(released)
        self.ceiling: float | None = None


def _seconds(value, finite: bool, what: str, *args) -> float:
    """`value` as seconds; `what.format(*args)` names it in the refusal, and is
    only formatted for one."""
    seconds = float(value)
    if math.isnan(seconds) or seconds == -math.inf or (finite and seconds == math.inf):
        kind = "a finite number" if finite else "a number"
        raise ValueError(
            f"{what.format(*args)} must be {kind} of seconds, got {value!r}"
        )
    return seconds


class ClockAuthority:
    """Every LP's clock and state, and the rule that grants them time.

    `timeline`, when given, receives one ``record(lp, from, to, kind,
    recovered, released, sent)`` call per reply, in issue order: ``released``
    the ``(channel, seq, arrival)`` messages the reply released, ``sent`` those
    logged by the requests it answers. `grants` counts each LP's finite grants
    by name, its replayed steps included, and `requests_served` its calls;
    `diagnostics`, the stall diagnostics its serve side printed. `final_clocks`
    is every LP's clock just before the run finished, and ``None`` until it
    has. `members` maps an LP to
    ``{member: names of the channels into or out of it that the member owns}``;
    an LP not in it is its own single caller. `bound_s` is the simulated-time
    bound, ``+inf`` for none. `client` names the LP that joins and resigns for
    each of `client_runs` client runs. `ceilings` gives grants their ceilings.
    """

    def __init__(
        self,
        channels: ChannelTable,
        timeline=None,
        members=None,
        bound_s: float = math.inf,
        client: LpId | None = None,
        client_runs: float = 1,
        ceilings: bool = False,
    ) -> None:
        if not client_runs >= 1 or client is None and client_runs != 1:
            raise ValueError(
                f"{client_runs} client runs over client {client}: a client LP "
                "serves one or more"
            )
        self._channels = channels
        self._bound = _seconds(bound_s, False, "the simulated-time bound")
        self._ids = channels.registry.ids()
        self._now = dict.fromkeys(self._ids, 0.0)
        self._state = dict.fromkeys(self._ids, RUNNING)
        self._target = dict.fromkeys(self._ids)
        self._daemon = dict.fromkeys(self._ids, math.inf)
        self._horizon = -math.inf  # H: the latest essential time seen
        self._into = {
            i: tuple(c.name for c in channels.channels_into(i)) for i in self._ids
        }
        # Per channel, seq -> arrival of every registered message not yet
        # released, in registration order.
        self._undelivered = {
            name: {} for names in self._into.values() for name in names
        }
        self._next_seq = dict.fromkeys(self._undelivered, 0)
        # N[j] of every LP, kept current by each call that changes one of its
        # inputs, and D(j->i) of every pair: a grant scan reads both, never
        # recomputing them.
        self._nv = dict.fromkeys(self._ids, 0.0)
        self._lp = {i: i for i in self._ids}
        self._from = {
            i: tuple((j, channels.distance(j, i)) for j in self._ids if j != i)
            for i in self._ids
        }
        self.timeline = timeline
        self._sent = {i: [] for i in self._ids}  # logged since its last reply
        self.grants = {i.name: 0 for i in self._ids}
        self.requests_served = {i.name: 0 for i in self._ids}  # by LP name
        self.diagnostics = 0
        self._diagnosed = 0  # `diagnostics` at the open client run's start
        self.final_clocks = None
        self._client = client and self._lp[channels.registry.require(client)]
        self._client_runs = client_runs
        self.ended: list[ClientRun] = []
        self._start = 0.0  # the open client run's start
        self._members = {}  # LP -> its member names, sorted
        self._owner = {}  # LP -> channel into or out of it -> the member owning it
        self._round = {}  # LP -> member -> (kind, t, daemon) of the round being joined
        self._parked = {}  # LP -> parked member -> {channel: releases held for it}
        self._queued = {}  # LP -> member -> later TAR rounds asked, not yet joined
        # LP -> member whose reply is held -> (the round that sends it,
        # {channel: releases held for it until then}, the grant it carries)
        self._held = {}
        # LP -> member -> [its clock, the rounds granted ahead of it that it
        # has not yet reported]
        self._owing = {}
        self._ceilings = ceilings
        self._ceiling = {}  # LP -> the ceiling of its latest grant
        self._ahead = {}  # LP -> channel -> {seq: arrival} sent ahead, not yet reached
        self._asks = {}  # LP -> whether its pending NER asked for messages ahead
        self._cycle = {}  # LP -> its shortest cycle, min over j of D(i->j) + D(j->i)
        for i in self._ids if ceilings else ():
            back = [channels.distance(i, j) + d for j, d in self._from[i]]
            self._cycle[i] = min(back, default=math.inf)
        for lp, owned in (members or {}).items():
            channels.registry.require(lp)
            if not owned:
                raise ValueError(f"{lp} is declared with an empty member list")
            touching = dict.fromkeys(
                [*self._into[lp], *(c.name for c in channels.channels_from(lp))]
            )
            listed = dict.fromkeys(name for names in owned.values() for name in names)
            self._owner[lp] = {}
            for name in sorted(touching | listed):
                owners = sorted(m for m, names in owned.items() if name in names)
                if name not in touching:
                    raise ValueError(
                        f"{' and '.join(owners)} of {lp} owns {name}, which "
                        f"neither goes into nor comes out of {lp}"
                    )
                if len(owners) != 1:
                    raise ValueError(
                        f"{name} goes into or out of {lp}, and "
                        + (
                            f"its members {' and '.join(owners)} each own it"
                            if owners
                            else f"none of its members {', '.join(sorted(owned))} owns it"
                        )
                    )
                self._owner[lp][name] = owners[0]
            self._members[lp] = tuple(sorted(owned))
            self._round[lp], self._parked[lp] = {}, {}
            self._queued[lp], self._held[lp], self._owing[lp] = {}, {}, {}

    def on_request(
        self,
        lp: LpId,
        kind: str,
        t: float,
        log,
        t_daemon: float = math.inf,
        member: str | None = None,
        steps=(),
        ahead: bool = False,
        rounds=(),
    ) -> list:
        """Register `lp`'s send log, record its request, and grant what is due.

        `t_daemon` is an NER's daemon deadline, ``+inf`` for none. `member`
        names the caller when `lp` has members, and is ``None`` otherwise.
        `steps` are the LP's own steps below its ceiling since its last grant,
        as ``(kind, t, G, n)``: asked `t`, answered `G`, after the first `n`
        entries of `log`. `ahead` asks an NER's grant to send every message
        below its ceiling. `rounds` are a member's later TAR rounds, each past
        the one before, asked in the same call. A ``REPORT`` is a member's
        `steps` through the rounds granted ahead of it, `t` the clock they
        reach, with its send log; it gets no reply.
        Returns the replies made due, as ``(lp, G, released)`` in the order
        issued, where ``released`` maps each channel into that LP to its newly
        released ``(seq, arrival)`` pairs; a member's reply is addressed
        ``(lp, member)`` and carries its own channels only. The requester is
        absent when its reply is held. Once the run is finished, every LP has
        had its ``+inf`` reply, and a request changes nothing and returns none.
        """
        if self.final_clocks is not None:
            return []
        # The registered object itself: every lookup below then matches by
        # identity rather than by calling LpId.__eq__.
        lp = self._lp.get(lp) or self._channels.registry.require(lp)
        self.requests_served[lp.name] += 1
        members = self._members.get(lp, ())
        if member not in (members or (None,)):
            raise KeyError(
                f"{lp} was called by member {member!r}; its members: "
                + (", ".join(members) or "none")
            )
        if self._state[lp] != RUNNING:
            raise BackdatedEvent(
                f"{lp} sent {kind} while waiting in {self._state[lp]}; only a "
                "running LP can produce a request",
                self.lp_table(),
            )
        if kind not in (TAR, NER, PARK, REPORT):
            raise ValueError(f"{kind!r} is not one of {TAR}, {NER}, {PARK}, {REPORT}")
        if kind in (PARK, REPORT) and not members:
            raise ValueError(
                f"{lp} sent {kind}; only a member of an LP with members does"
            )
        t = _seconds(t, kind in (TAR, REPORT), "the {} target of {}", kind, lp)
        daemon = _seconds(t_daemon, False, "the daemon deadline of {}", lp)
        if kind == TAR and daemon != math.inf:
            raise ValueError(
                f"{lp} sent TAR with daemon deadline {daemon}; only NER has one"
            )
        rounds = [_seconds(r, True, "a round {} asked", lp) for r in rounds]
        ascending = all(a < b for a, b in zip([t, *rounds], rounds))
        if rounds and (kind != TAR or not members or not ascending):
            raise ValueError(
                f"{lp} asked {kind} rounds {[t, *rounds]}; only a member's TAR "
                "asks later rounds, each past the one before"
            )
        log = self._catch_up(lp, steps, log, member)
        owing = self._owing.get(lp, {}).get(member)
        if owing is not None and owing[1]:
            raise ClockAbort(
                f"{lp}: {member} sent {kind} before reporting the rounds "
                f"{owing[1]} granted ahead of it",
                self.lp_table(),
            )
        if kind == REPORT and (owing is None or t != owing[0]):
            raise ClockAbort(
                f"{lp}: {member} reported its clock at {t}, but "
                + (
                    "no round was granted ahead of it"
                    if owing is None
                    else f"its last round granted ahead is {owing[0]}"
                ),
                self.lp_table(),
            )
        now = self._now[lp]
        if t < now or daemon < now:
            asked = f"{kind}({t})" if t < now else f"daemon deadline {daemon}"
            raise BackdatedEvent(
                f"{lp} asked for {asked} behind its own clock at {now}",
                self.lp_table(),
            )
        if kind != TAR and any(self._ahead.get(lp, {}).values()):
            raise ClockAbort(
                f"{lp} asked {kind} with messages sent ahead of it unreached",
                self.lp_table(),
            )
        if members and kind != REPORT:
            self._refuse_unjoinable(lp, member, kind, t)
        for name, seq, arrival in log:
            self._register(lp, name, seq, arrival, member)
        if self.timeline is not None:
            self._sent[lp] += map(tuple, log)
        if owing is not None:  # its sends are logged: N may rise
            del self._owing[lp][member]
            self._nv[lp] = self._n(lp)
            if kind == REPORT:
                return self._address(self._grant_due())
        self._asks[lp] = ahead and kind == NER
        if members:
            joined, parked = self._round[lp], self._parked[lp]
            joined[member] = _STANDING if kind == PARK else (kind, t, daemon)
            if rounds:
                self._queued[lp][member] = rounds
                own = self._own_into(lp, member)
                self._held[lp][member] = (rounds[-1], own, rounds[-1])
            if kind == PARK:
                parked[member] = self._own_into(lp, member)
            if len(joined) < len(members):
                return [] if owing is None else self._address(self._grant_due())
            if len(parked) == len(members):
                return self._release(lp)
            kind = NER if kind == PARK else kind
            t = min(asked for _, asked, _ in joined.values())
            daemon = min(deadline for _, _, deadline in joined.values())
            joined.clear()
            joined.update(dict.fromkeys(parked, _STANDING))
        self._wait(lp, kind, t, daemon)
        return self._address(self._grant_due())

    def on_bind(self, lp: LpId) -> tuple[float, dict[str, int]]:
        """What `lp` starts from when it binds: its row's clock, and the next
        sequence number of each channel out of it."""
        out = self._channels.channels_from(lp)
        return self._now[lp], {c.name: self._next_seq[c.name] for c in out}

    def _wait(self, lp: LpId, kind: str, t: float, daemon: float) -> None:
        """`lp` waits in `kind`, asking `t` with daemon deadline `daemon`."""
        if t < math.inf:
            self._horizon = max(self._horizon, t)
        self._state[lp], self._target[lp], self._daemon[lp] = kind, t, daemon
        self._nv[lp] = t if kind == TAR else self._n(lp)

    def _join_queued(self, lp: LpId) -> None:
        """After a round of `lp` is granted, each member's next queued round
        joins the next one, which waits in TAR once every member has asked it."""
        joined, queued = self._round[lp], self._queued[lp]
        for member in list(queued):
            t = queued[member].pop(0)
            if not queued[member]:
                del queued[member]
            self._refuse_unjoinable(lp, member, TAR, t)
            joined[member] = (TAR, t, math.inf)
        if len(joined) == len(self._members[lp]):
            (_, t, _), *_ = joined.values()
            joined.clear()
            self._wait(lp, TAR, t, math.inf)

    def _own_into(self, lp: LpId, member: str) -> Released:
        """An empty list for each channel into `lp` that `member` owns."""
        return Released({c: [] for c in self._into[lp] if self._owner[lp][c] == member})

    def _refuse_unjoinable(self, lp: LpId, member: str, kind: str, t: float) -> None:
        joined = self._round[lp]
        reason = None
        if member in joined:
            behind = ", ".join(m for m in self._members[lp] if m not in joined)
            reason = (
                f"{lp}: {member} called again before {behind} called; every "
                "member asks each round once"
            )
        elif joined:
            other, (other_kind, other_t, _) = next(iter(joined.items()))
            if (other_kind == TAR) != (kind == TAR):  # a park joins as an NER
                reason = f"{lp}: {member} asked {kind} while {other} asked {other_kind}"
            elif kind == TAR and other_t != t:
                reason = (
                    f"{lp}: {member} asked TAR({t}) while {other} asked TAR({other_t})"
                )
        if reason is not None:
            raise ClockAbort(reason, self.lp_table())

    def _address(self, replies: list) -> list:
        """One reply per member for an LP with members, each with its own
        channels; a parked member's are held for it, unless the reply finishes
        the run or ends a client run, and so are those of a member whose call
        named later rounds, until the last one's grant, and of a member that
        returned, until the last round granted ahead of it: that reply carries
        the grant it returned at."""
        out = []
        for i, g, released, *ended in replies:
            for m in self._members.get(i, (None,)):
                if m is None:
                    out.append((i, g, released, *ended))
                    continue
                mine = {c: r for c, r in released.items() if self._owner[i][c] == m}
                held, at = self._parked[i].get(m), g
                if held is not None:
                    self._hold(held, mine)
                    if g < math.inf and not ended:
                        continue
                    del self._parked[i][m]
                    self._round[i].pop(m, None)
                    mine = held
                elif m in self._held[i]:
                    last, held, back = self._held[i][m]
                    self._hold(held, mine)
                    if g < last and not ended:
                        continue
                    del self._held[i][m]
                    mine, at = held, back if g == last and not ended else g
                out.append(((i, m), at, mine, *ended))
        return out

    @staticmethod
    def _hold(held: dict, released: dict) -> None:
        """Keep what a grant released to a member until its reply goes."""
        for c, r in released.items():
            held[c] += r

    def _release(self, lp: LpId) -> list:
        """The last member parked: each leaves at the LP's clock with the
        messages held for it."""
        self._round[lp].clear()
        held, self._parked[lp] = self._parked[lp], {}
        g = self._now[lp]
        return [((lp, m), g, held[m]) for m in self._members[lp]] + self._address(
            self._grant_due()
        )

    def lp_table(self) -> tuple[LpRow, ...]:
        """Every LP's row, in name order."""
        rows = []
        for i in self._ids:
            row = self._row(i)
            least = min(row, key=lambda term: term[1], default=(None, math.inf))
            rows.append(
                LpRow(
                    i,
                    self._state[i],
                    self._now[i],
                    self._target[i],
                    self._n(i),
                    self._pending(i),
                    row,
                    least[0] if least[1] < math.inf else None,
                )
            )
        return tuple(rows)

    def _register(
        self, lp: LpId, name: str, seq: int, arrival: float, member: str | None
    ) -> None:
        channel = self._channels.channel(name)
        a = _seconds(arrival, True, "the arrival of {} seq {}", name, seq)
        expected = self._next_seq[name]
        clock = self._clock_of(lp, member)
        floor = clock + channel.lookahead_s
        if channel.source != lp:
            refusal = f"{lp} logged a send on {name}, whose sender is {channel.source}"
        elif member is not None and self._owner[lp][name] != member:
            refusal = (
                f"{lp}: {member} logged a send on {name}, which "
                f"{self._owner[lp][name]} owns"
            )
        elif seq != expected:
            refusal = (
                f"{lp} logged seq {seq} on {name}, which expects seq {expected} "
                "next; a sender numbers each channel consecutively"
            )
        elif a < self._now[channel.target]:
            refusal = (
                f"{name} seq {seq} arrives at {a}, behind {channel.target}'s "
                f"clock at {self._now[channel.target]}"
            )
        elif a < floor:
            refusal = (
                f"{name} seq {seq} arrives at {a}, before {lp}'s clock "
                f"{clock} plus the channel's lookahead {channel.lookahead_s}"
            )
        else:
            self._undelivered[name][seq] = a
            self._horizon = max(self._horizon, a)
            self._next_seq[name] = seq + 1
            if self._state[channel.target] == NER:
                self._nv[channel.target] = self._n(channel.target)
            return
        raise BackdatedEvent(refusal, self.lp_table())

    def _pending(self, i: LpId) -> tuple[tuple[str, int, float], ...]:
        return tuple(
            (name, seq, a)
            for name in self._into[i]
            for seq, a in self._undelivered[name].items()
        )

    def _clock_of(self, lp: LpId, member: str | None) -> float:
        """The clock `member` of `lp` sends at: its own while rounds granted
        ahead of it are unreported, else the LP's."""
        owing = self._owing.get(lp, {}).get(member)
        return self._now[lp] if owing is None else owing[0]

    def _n(self, j: LpId) -> float:
        if self._state[j] == RUNNING:  # no later than a member yet to report
            return min([self._now[j], *(c for c, _ in self._owing.get(j, {}).values())])
        if self._state[j] == TAR:
            return self._target[j]
        n = min(self._target[j], self._daemon[j])
        for name in self._into[j]:
            for a in self._undelivered[name].values():
                n = min(n, a)
        return n

    def _row(self, i: LpId) -> tuple[tuple[LpId, float], ...]:
        nv = self._nv
        return tuple((j, nv[j] + d) for j, d in self._from[i])

    def _lbts(self, i: LpId) -> float:
        return min((term for _, term in self._row(i)), default=math.inf)

    def _grant_due(self) -> list:
        replies = []
        state, nv = self._state, self._nv
        while True:
            # `state` iterates in name order and the sort is stable, so equal
            # N stay in name order without comparing two LpIds.
            waiting = sorted(
                ((nv[i], i) for i, s in state.items() if s != RUNNING),
                key=lambda waiter: waiter[0],
            )
            grantable = [(n, i) for n, i in waiting if n <= self._horizon]
            due = next(((i, n) for n, i in grantable if n < self._lbts(i)), None)
            if due is not None:
                replies.append(self._grant(*due))
                replies += self._grant_ahead(due[0])
            elif len(waiting) < len(self._ids):
                return replies
            elif not grantable:
                return replies + self._end_client_run()
            else:
                replies.append(self._recover(grantable[0][1], grantable[0][0]))

    def _grant_ahead(self, i: LpId) -> list:
        """With ceilings, after a grant at which members of `i` returned while
        others ask later rounds: each later round below that grant's ceiling is
        granted now, the members that returned standing in it, as they owe it.
        Each such member's reply is held until the last of them, to carry
        what they release to it ahead, beside the ceiling."""
        joined = self._round.get(i)
        if not (self._ceilings and joined) or len(joined) == len(self._members[i]):
            return []
        ceiling, replies = self._ceiling_of(i, self._now[i]), []
        while joined and (t := next(iter(joined.values()))[1]) < ceiling:
            for m in self._members[i]:
                if m not in joined:
                    self._owing[i].setdefault(m, [self._now[i], []])[1].append(t)
                    own = (t, self._own_into(i, m), self._now[i])
                    _, held, back = self._held[i].get(m, own)
                    held.ceiling = ceiling
                    self._held[i][m] = (t, held, back)
            joined.clear()
            self._wait(i, TAR, t, math.inf)
            replies.append(self._grant(i, t))
        self._nv[i] = self._n(i)
        return replies

    def _recover(self, i: LpId, g: float) -> tuple:
        # The one grant whose guarantee is weaker: a message may still land at g.
        return self._grant(i, g, recovered=True)

    def _grant(self, i: LpId, g: float, recovered: bool = False) -> tuple:
        if g > self._bound:
            raise ClockAbort(
                f"a grant to {g} for {i} passes the simulated-time bound "
                f"{self._bound}; either the bound is shorter than the workload, "
                "or something essential, such as a housekeeping timer not "
                "declared daemon, keeps the run alive",
                self.lp_table(),
            )
        released = Released()
        for name in self._into[i]:
            pending = self._undelivered[name]
            released[name] = [(seq, a) for seq, a in pending.items() if a <= g]
            for seq, _ in released[name]:
                del pending[seq]
        got = [(c, seq, a) for c, r in released.items() for seq, a in r]
        if self._ahead.get(i):  # a TAR past the messages sent ahead of it
            reached = self._ahead.pop(i)
            if any(a > g for msgs in reached.values() for a in msgs.values()):
                raise ClockAbort(
                    f"{i} was granted {g}, short of messages sent ahead of it",
                    self.lp_table(),
                )
            got = [
                (c, seq, a)
                for c in self._into[i]
                for seq, a in sorted([*reached.get(c, {}).items(), *released[c]])
            ]
        self._ceiling.pop(i, None)
        if self._ceilings and not recovered and g < math.inf and i not in self._members:
            released.ceiling = self._ceiling[i] = self._ceiling_of(i, g, released)
        if self.timeline is not None:
            self._record(i, g, self._state[i], recovered, tuple(got))
        self.grants[i.name] += 1
        # g is N[i], so N[i] is unchanged: a running LP's N is its clock.
        self._now[i], self._state[i], self._target[i] = g, RUNNING, None
        if self._queued.get(i):
            self._join_queued(i)
        return (i, g, released)

    def _ceiling_of(self, i: LpId, g: float, released: Released | None = None) -> float:
        """The ceiling of `i`'s grant to `g`: `released` takes every message
        below it when `i` asked for them ahead. An LP with members takes none:
        the rounds granted below it release them."""
        ceiling = min(self._lbts(i), g + self._cycle[i], self._bound)
        if i in self._members:
            return ceiling
        for name in self._into[i] if self._asks.get(i) else ():
            pending = self._undelivered[name]
            early = [(seq, a) for seq, a in pending.items() if a < ceiling]
            for seq, _ in early:
                del pending[seq]
            if early:
                self._ahead.setdefault(i, {})[name] = dict(early)
                released[name] += early
        rest = [a for n in self._into[i] for a in self._undelivered[n].values()]
        return min([ceiling, *rest])

    def _catch_up(self, lp: LpId, steps, log, member) -> list:
        """Register the sends `lp` logged before each of its own `steps`, and
        replay each step after them; returns the rest of `log`."""
        if not steps:
            return log
        try:
            steps = [
                (kind, _seconds(t, False, "a target"), _seconds(g, True, "a time"), n)
                for kind, t, g, n in steps
            ]
        except (TypeError, ValueError) as malformed:
            raise ClockAbort(
                f"{lp} reported malformed steps: {malformed}", self.lp_table()
            ) from malformed
        ns = [n for *_, n in steps]
        if any(type(n) is not int for n in ns) or not (
            ns == sorted(ns) and 0 <= ns[0] and ns[-1] <= len(log)
        ):
            raise ClockAbort(
                f"{lp} reported steps {steps} over a log of {len(log)}",
                self.lp_table(),
            )
        k = 0
        for kind, t, to, n in steps:
            for name, seq, arrival in log[k:n]:
                self._register(lp, name, seq, arrival, member)
            if self.timeline is not None:
                self._sent[lp] += map(tuple, log[k:n])
            if member is None:
                self._replay(lp, kind, t, to)
            else:
                self._step_member(lp, member, kind, t, to)
            k = n
        return log[k:]

    def _step_member(
        self, i: LpId, member: str, kind: str, t: float, to: float
    ) -> None:
        """`member` of `i` stepped on its own to `kind`(`t`), answered `to`:
        the next round granted ahead of it, which it now sends from."""
        owing = self._owing[i].get(member)
        if not (owing and owing[1] and kind == TAR and t == to == owing[1][0]):
            raise ClockAbort(
                f"{i}: {member} stepped on its own to {kind}({t}) answered {to}; "
                f"the rounds granted ahead of it: {owing[1] if owing else []}",
                self.lp_table(),
            )
        owing[0] = owing[1].pop(0)

    def _replay(self, i: LpId, kind: str, t: float, to: float) -> None:
        """One step `i` took on its own, asking `t` and answered `to`, as the
        grant its own request would have had."""
        ceiling = self._ceiling.get(i, -math.inf)
        if t < math.inf:
            self._horizon = max(self._horizon, t)
        if (
            kind not in (TAR, NER)
            or not self._now[i] <= to < ceiling
            or to > min(t, self._horizon)
            or (kind == TAR and to != t)
        ):
            raise ClockAbort(
                f"{i} stepped on its own to {kind}({t}) answered {to}, from its "
                f"clock {self._now[i]}, below its ceiling {ceiling}, by the "
                f"horizon {self._horizon}",
                self.lp_table(),
            )
        late = [
            (a, name, seq)
            for name in self._into[i]
            for seq, a in self._undelivered[name].items()
            if a <= to
        ]
        if late:
            raise BackdatedEvent(
                f"{i} stepped on its own to {to}, past {min(late)} registered for it",
                self.lp_table(),
            )
        ahead, got = self._ahead.get(i, {}), []
        for name in self._into[i]:
            reached = [(seq, a) for seq, a in ahead.get(name, {}).items() if a <= to]
            for seq, a in reached:
                del ahead[name][seq]
                got.append((name, seq, a))
        if self.timeline is not None:
            self._record(i, to, kind, False, tuple(got))
        self.grants[i.name] += 1
        self._now[i] = self._nv[i] = to

    def _end_client_run(self) -> list:
        """Every LP waits and none is grantable: the open client run ends, and
        unless it is the last, the client resigns and every other LP is told."""
        end, k = max(self._now.values()), len(self.ended)
        self._ceiling.clear()  # no LP runs below one now
        clocks, grants = tuple(self._now.items()), dict(self.grants)
        diagnostics = self.diagnostics - self._diagnosed
        served = dict(self.requests_served)
        self.ended.append(
            ClientRun(k, self._start, end, clocks, grants, diagnostics, served)
        )
        if self._client is None or k + 1 == self._client_runs:
            return self._finish()
        self._start, self._diagnosed = end, self.diagnostics
        replies = []
        for i in self._ids:
            none = {name: [] for name in self._into[i]}
            self._state[i], self._target[i] = RUNNING, None
            if i is not self._client:
                self._nv[i] = self._now[i]
                replies.append((i, self._now[i], none, k))
                continue
            if self.timeline is not None:
                self._record(i, math.inf, NER)
            self._now[i] = self._nv[i] = end
            replies.append((i, math.inf, none))
        return replies

    def _finish(self) -> list:
        self.final_clocks = tuple(self._now.items())
        for owing in self._owing.values():
            owing.clear()
        for i in self._ids:
            if self.timeline is not None:
                self._record(i, math.inf, self._state[i])
            self._now[i], self._state[i], self._target[i] = math.inf, RUNNING, None
            self._nv[i] = math.inf
        return [(i, math.inf, {name: [] for name in self._into[i]}) for i in self._ids]

    def _record(self, i: LpId, g: float, kind: str, recovered=False, released=()):
        """Hand the timeline `i`'s reply, with what it sent since its last one."""
        sent, self._sent[i] = tuple(self._sent[i]), []
        self.timeline.record(i, self._now[i], g, kind, recovered, released, sent)
