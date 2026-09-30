# SPDX-License-Identifier: MIT
"""The Clock Authority: which waiting process may move its clock, to when, and
with which messages.

A logical process (LP) asks for time with one of three requests:

* ``TAR`` -- advance to ``t``: its owner has priced an event, and the grant is
  exactly ``t``;
* ``NER`` -- idle until ``t``, which may be ``+inf``: the grant is ``t`` or the
  earliest message registered for it, whichever comes first;
* ``END`` -- the workload is over: every LP is granted ``+inf`` and the run
  finishes.

Each request carries the requester's send log since its previous request, as
``(channel, seq, arrival)``. The log is registered before the requester's state
changes, so every message an LP produced is known before its clock moves.

``N[j]`` is the earliest time LP *j* could still produce a message: its clock
while it runs, its target while it waits in TAR (its owner is inside this call
and sends nothing), and the lesser of its target and its earliest unreleased
message while it waits in NER. A waiting LP *i* is granted exactly ``N[i]``, and
only when ``N[i] < min over j != i of (N[j] + D(j->i))``. ``D`` is the channel
table's least summed lookahead over every channel path, so an idle LP between
two others does not hide the first from the third. The comparison is strict: a
message not yet reported arrives no earlier than its sender's ``N`` plus that
distance, so a strict grant never reaches it. Waiting LPs are tried in
``(N, name)`` order and the scan restarts after every grant.

When every LP waits and the strict rule grants none, which only a
zero-lookahead cycle allows, the LP with the least ``(N, name)`` is granted:
nothing is earlier, so nothing earlier can reach it. A message that then lands
at its receiver's current instant belongs to that instant's next round. When
every ``N`` is infinite the run is finished, as with ``END``.

Nothing here reads a clock, opens a socket or starts a thread. The caller
carries requests in and replies out.
"""

import math
from dataclasses import dataclass

from .channels import ChannelTable
from .identity import LpId

TAR, NER, END = "TAR", "NER", "END"
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
    """Every LP's clock and state, and the rule that grants them time."""

    def __init__(self, channels: ChannelTable) -> None:
        self._channels = channels
        self._ids = channels.registry.ids()
        self._now = dict.fromkeys(self._ids, 0.0)
        self._state = dict.fromkeys(self._ids, RUNNING)
        self._target = dict.fromkeys(self._ids)
        self._into = {
            i: tuple(c.name for c in channels.channels_into(i)) for i in self._ids
        }
        # Per channel, seq -> arrival of every registered message not yet
        # released, in registration order.
        self._undelivered = {
            name: {} for names in self._into.values() for name in names
        }
        self._next_seq = dict.fromkeys(self._undelivered, 0)
        self._done = False

    def on_request(self, lp: LpId, kind: str, t: float, log) -> list:
        """Register `lp`'s send log, record its request, and grant what is due.

        Returns the replies made due, as ``(lp, G, released)`` in the order
        issued, where ``released`` maps each channel into that LP to its newly
        released ``(seq, arrival)`` pairs. The requester is absent when its
        reply is held. Once the run is finished, every LP has had its ``+inf``
        reply, and a request changes nothing and returns none.
        """
        if self._done:
            return []
        self._channels.registry.require(lp)
        if self._state[lp] != RUNNING:
            raise BackdatedEvent(
                f"{lp} sent {kind} while waiting in {self._state[lp]}; only a "
                "running LP can produce a request",
                self.lp_table(),
            )
        if kind not in (TAR, NER, END):
            raise ValueError(f"{kind!r} is not one of {TAR}, {NER}, {END}")
        if kind != END:
            t = _seconds(t, f"the {kind} target of {lp}", finite=kind == TAR)
            if t < self._now[lp]:
                raise BackdatedEvent(
                    f"{lp} asked for {kind}({t}) behind its own clock at "
                    f"{self._now[lp]}",
                    self.lp_table(),
                )
        for name, seq, arrival in log:
            self._register(lp, name, seq, arrival)
        if kind == END:
            return self._finish()
        self._state[lp], self._target[lp] = kind, t
        return self._grant_due()

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
        return min([self._target[j]] + [a for _, _, a in self._pending(j)])

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
            due = next(((i, n) for n, i in waiting if n < self._lbts(i)), None)
            if due is not None:
                replies.append(self._grant(*due))
            elif len(waiting) < len(self._ids):
                return replies
            elif waiting[0][0] == math.inf:
                return replies + self._finish()
            else:
                replies.append(self._recover(waiting[0][1], waiting[0][0]))

    def _recover(self, i: LpId, g: float) -> tuple:
        # The one grant whose guarantee is weaker: a message may still land at g.
        return self._grant(i, g)

    def _grant(self, i: LpId, g: float) -> tuple:
        released = {}
        for name in self._into[i]:
            pending = self._undelivered[name]
            released[name] = [(seq, a) for seq, a in pending.items() if a <= g]
            for seq, _ in released[name]:
                del pending[seq]
        self._now[i], self._state[i], self._target[i] = g, RUNNING, None
        return (i, g, released)

    def _finish(self) -> list:
        self._done = True
        for i in self._ids:
            self._now[i], self._state[i], self._target[i] = math.inf, RUNNING, None
        return [(i, math.inf, {name: [] for name in self._into[i]}) for i in self._ids]
