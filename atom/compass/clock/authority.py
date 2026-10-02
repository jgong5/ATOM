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

An LP may be declared with members: processes that each hold a runtime for it,
as the ranks of a data-parallel group do. A member names itself in each call
and owns the channels into the LP whose names end ``#<member>``. The members'
calls are joined into one request per round: each member makes the same number
of calls, all ask one kind, every TAR asks the same target, and the joined
NER's target and daemon deadline are each the least over the members. A call
before the round is complete registers its send log and nothing else: the LP
stays running, so its ``N`` stays at its clock. Every member gets the common
grant, carrying the releases on its own channels only.

Nothing here reads a clock, opens a socket or starts a thread. The caller
carries requests in and replies out.
"""

import math
from dataclasses import dataclass

from .channels import ChannelTable
from .identity import LpId

TAR, NER = "TAR", "NER"
RUNNING = "running"


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


class ClockAbort(Exception):
    """The run cannot be trusted any further. Carries every LP's row."""

    def __init__(self, reason: str, table: tuple[LpRow, ...]) -> None:
        super().__init__("\n".join([reason, *map(repr, table)]))
        self.reason = reason
        self.table = table


class BackdatedEvent(ClockAbort):
    """A request that would put a message or a clock into some LP's past."""


def _seconds(value, what: str, finite: bool) -> float:
    seconds = float(value)
    if math.isnan(seconds) or seconds == -math.inf or (finite and seconds == math.inf):
        kind = "a finite number" if finite else "a number"
        raise ValueError(f"{what} must be {kind} of seconds, got {value!r}")
    return seconds


class ClockAuthority:
    """Every LP's clock and state, and the rule that grants them time.

    `timeline`, when given, receives one ``record(lp, from, to, kind,
    recovered)`` call per reply, in issue order. `grants` counts each LP's
    finite grants by name. `final_clocks` is every LP's clock just before the run
    finished, and ``None`` until it has. `members` maps an LP to the names of
    its members; an LP not in it is its own single caller.
    """

    def __init__(self, channels: ChannelTable, timeline=None, members=None) -> None:
        self._channels = channels
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
        self.timeline = timeline
        self.grants = {i.name: 0 for i in self._ids}
        self.final_clocks = None
        self._members = {}  # LP -> its member names, sorted
        self._calls = {}  # LP -> member -> calls made
        self._round = {}  # LP -> member -> (kind, t, daemon) of the round being joined
        for lp, names in (members or {}).items():
            channels.registry.require(lp)
            names = tuple(sorted(names))
            for name in self._into[lp]:
                if name.rpartition("#")[2] not in names:
                    raise ValueError(
                        f"{name} goes into {lp}, and none of its members "
                        f"{', '.join(names)} owns it"
                    )
            self._members[lp] = names
            self._calls[lp] = dict.fromkeys(names, 0)
            self._round[lp] = {}

    def on_request(
        self,
        lp: LpId,
        kind: str,
        t: float,
        log,
        t_daemon: float = math.inf,
        member: str | None = None,
    ) -> list:
        """Register `lp`'s send log, record its request, and grant what is due.

        `t_daemon` is an NER's daemon deadline, ``+inf`` for none. `member`
        names the caller when `lp` has members, and is ``None`` otherwise.
        Returns the replies made due, as ``(lp, G, released)`` in the order
        issued, where ``released`` maps each channel into that LP to its newly
        released ``(seq, arrival)`` pairs; a member's reply is addressed
        ``(lp, member)`` and carries its own channels only. The requester is
        absent when its reply is held. Once the run is finished, every LP has
        had its ``+inf`` reply, and a request changes nothing and returns none.
        """
        if self.final_clocks is not None:
            return []
        self._channels.registry.require(lp)
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
        if kind not in (TAR, NER):
            raise ValueError(f"{kind!r} is not one of {TAR}, {NER}")
        t = _seconds(t, f"the {kind} target of {lp}", finite=kind == TAR)
        daemon = _seconds(t_daemon, f"the daemon deadline of {lp}", finite=False)
        if kind == TAR and daemon != math.inf:
            raise ValueError(
                f"{lp} sent TAR with daemon deadline {daemon}; only NER has one"
            )
        for asked, value in (
            (f"{kind}({t})", t),
            (f"daemon deadline {daemon}", daemon),
        ):
            if value < self._now[lp]:
                raise BackdatedEvent(
                    f"{lp} asked for {asked} behind its own clock at {self._now[lp]}",
                    self.lp_table(),
                )
        if members:
            self._refuse_unjoinable(lp, member, kind, t)
        for name, seq, arrival in log:
            self._register(lp, name, seq, arrival)
        if members:
            joined = self._round[lp]
            joined[member] = (kind, t, daemon)
            self._calls[lp][member] += 1
            if len(joined) < len(members):
                return self._address(self._grant_due())
            t = min(asked for _, asked, _ in joined.values())
            daemon = min(deadline for _, _, deadline in joined.values())
            joined.clear()
        if t < math.inf:
            self._horizon = max(self._horizon, t)
        self._state[lp], self._target[lp], self._daemon[lp] = kind, t, daemon
        return self._address(self._grant_due())

    def _refuse_unjoinable(self, lp: LpId, member: str, kind: str, t: float) -> None:
        calls, joined = self._calls[lp], self._round[lp]
        behind = min(calls, key=calls.__getitem__)
        reason = None
        if calls[member] > calls[behind]:
            reason = (
                f"{lp}: {member} made call {calls[member] + 1} while {behind} has "
                f"made {calls[behind]}; every member calls once per round"
            )
        elif joined:
            other, (other_kind, other_t, _) = next(iter(joined.items()))
            if other_kind != kind:
                reason = f"{lp}: {member} asked {kind} while {other} asked {other_kind}"
            elif kind == TAR and other_t != t:
                reason = (
                    f"{lp}: {member} asked TAR({t}) while {other} asked TAR({other_t})"
                )
        if reason is not None:
            raise ClockAbort(reason, self.lp_table())

    def _address(self, replies: list) -> list:
        """One reply per member for an LP with members, each with its own channels."""
        return [
            (
                (i, g, released)
                if m is None
                else (
                    (i, m),
                    g,
                    {c: r for c, r in released.items() if c.rpartition("#")[2] == m},
                )
            )
            for i, g, released in replies
            for m in self._members.get(i, (None,))
        ]

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

    def _register(self, lp: LpId, name: str, seq: int, arrival: float) -> None:
        channel = self._channels.channel(name)
        a = _seconds(arrival, f"the arrival of {name} seq {seq}", finite=True)
        expected = self._next_seq[name]
        floor = self._now[lp] + channel.lookahead_s
        if channel.source != lp:
            refusal = f"{lp} logged a send on {name}, whose sender is {channel.source}"
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
                f"{self._now[lp]} plus the channel's lookahead {channel.lookahead_s}"
            )
        else:
            self._undelivered[name][seq] = a
            self._horizon = max(self._horizon, a)
            self._next_seq[name] = seq + 1
            return
        raise BackdatedEvent(refusal, self.lp_table())

    def _pending(self, i: LpId) -> tuple[tuple[str, int, float], ...]:
        return tuple(
            (name, seq, a)
            for name in self._into[i]
            for seq, a in self._undelivered[name].items()
        )

    def _n(self, j: LpId) -> float:
        if self._state[j] == RUNNING:
            return self._now[j]
        if self._state[j] == TAR:
            return self._target[j]
        return min(
            [self._target[j], self._daemon[j]] + [a for _, _, a in self._pending(j)]
        )

    def _row(self, i: LpId) -> tuple[tuple[LpId, float], ...]:
        return tuple(
            (j, self._n(j) + self._channels.distance(j, i)) for j in self._ids if j != i
        )

    def _lbts(self, i: LpId) -> float:
        return min((term for _, term in self._row(i)), default=math.inf)

    def _grant_due(self) -> list:
        replies = []
        while True:
            waiting = sorted(
                (self._n(i), i) for i in self._ids if self._state[i] != RUNNING
            )
            grantable = [(n, i) for n, i in waiting if n <= self._horizon]
            due = next(((i, n) for n, i in grantable if n < self._lbts(i)), None)
            if due is not None:
                replies.append(self._grant(*due))
            elif len(waiting) < len(self._ids):
                return replies
            elif not grantable:
                return replies + self._finish()
            else:
                replies.append(self._recover(grantable[0][1], grantable[0][0]))

    def _recover(self, i: LpId, g: float) -> tuple:
        # The one grant whose guarantee is weaker: a message may still land at g.
        return self._grant(i, g, recovered=True)

    def _grant(self, i: LpId, g: float, recovered: bool = False) -> tuple:
        released = {}
        for name in self._into[i]:
            pending = self._undelivered[name]
            released[name] = [(seq, a) for seq, a in pending.items() if a <= g]
            for seq, _ in released[name]:
                del pending[seq]
        if self.timeline is not None:
            self.timeline.record(i, self._now[i], g, self._state[i], recovered)
        self.grants[i.name] += 1
        self._now[i], self._state[i], self._target[i] = g, RUNNING, None
        return (i, g, released)

    def _finish(self) -> list:
        self.final_clocks = tuple(self._now.items())
        for i in self._ids:
            if self.timeline is not None:
                self.timeline.record(i, self._now[i], math.inf, self._state[i], False)
            self._now[i], self._state[i], self._target[i] = math.inf, RUNNING, None
        return [(i, math.inf, {name: [] for name in self._into[i]}) for i in self._ids]
