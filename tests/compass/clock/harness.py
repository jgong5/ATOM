# SPDX-License-Identifier: MIT
"""Drives synthetic LPs against the real Clock Authority, in process.

No transport and no threads: the driver calls `ClockAuthority.on_request`
directly. Every LP starts running at 0. While several LPs are running, which
one's request reaches the authority next is the driver's `order`, a priority
over LP names. A reply goes to its LP, which returns its next request; a
running LP that receives the `+inf` reply at the end of the run still submits
the request it was holding, as it would on a wire, so each LP issues as many
requests as it receives replies.

The driver holds the check the authority cannot make about itself: no grant
may move an LP past a message registered for it that it was not handed, and no
LP is handed a message that arrives after its grant.
"""

import collections
import math
import time
from dataclasses import dataclass

from atom.compass.clock import END, ClockAuthority

from .deployments import DEPLOYMENTS, build
from .participants import Engine


class SteppedOverEvent(AssertionError):
    """An LP was moved past a message it had not been handed."""


class GrantsExhausted(Exception):
    """A run hit its grant cap."""


class _Counted(ClockAuthority):
    """Counts the grants the recovery branch issues."""

    recovered = 0

    def _recover(self, i, g):
        self.recovered += 1
        return super()._recover(i, g)


@dataclass(frozen=True)
class RunReport:
    """What one run did."""

    deployment: str
    submitted: tuple[str, ...]  # the LP of each request, in the order submitted
    requests: dict  # LP name -> requests issued, by kind
    replies: dict  # LP name -> replies received
    reply_log: dict  # LP name -> ((G, ((channel, seq, arrival), ...)), ...)
    recovered: int
    steps: int
    messages: int
    handled: int
    handled_before_grant: int  # released messages handled below their grant
    channels: dict  # channel -> messages sent on it
    timers: int
    modelled_seconds: float
    wall_seconds: float
    unfinished: tuple[str, ...]
    stopped_by: str

    @property
    def grants(self) -> int:
        return sum(self.replies.values())


class SyntheticRun:
    """One deployment, one workload, one request order, driven to the end."""

    def __init__(self, deployment, workload, order=None, grant_cap=None):
        self.deployment = deployment
        self.table = DEPLOYMENTS[deployment]()
        self.clock = _Counted(self.table)
        self.lps = build(self.table, workload)
        order = tuple(order or map(str, self.table.registry.ids()))
        self.rank = {lp: order.index(str(lp)) for lp in self.lps}
        self.grant_cap = grant_cap
        self.in_flight = {lp: {} for lp in self.lps}  # (channel, seq) -> (a, payload)
        self.requests = {lp: collections.Counter() for lp in self.lps}
        self.reply_log = {lp: [] for lp in self.lps}
        self.channels = collections.Counter()
        self.submitted = []
        self.messages = self.handled = self.handled_before_grant = 0

    def run(self):
        started = time.perf_counter()
        ready = {lp: person.run(0.0) for lp, person in self.lps.items()}
        stopped_by = None
        try:
            while ready:
                lp = min(ready, key=self.rank.__getitem__)
                kind, t = ready.pop(lp)
                self.requests[lp][kind] += 1
                self.submitted.append(str(lp))
                if kind == END and stopped_by is None:
                    stopped_by = f"END from {lp}"
                for i, g, released in self.clock.on_request(
                    lp, kind, t, self._post(lp)
                ):
                    self._check_cap()
                    messages = self._hand_over(i, g, released)
                    self.reply_log[i].append(
                        (g, tuple((c, s, a) for a, c, s, _ in messages))
                    )
                    if g == math.inf:
                        stopped_by = stopped_by or "every N infinite"
                        continue
                    self._refuse_step_over(i, g)
                    ready[i] = self.lps[i].on_grant(g, messages)
        except GrantsExhausted:
            stopped_by = f"the grant cap of {self.grant_cap} was reached"
        return self._report(time.perf_counter() - started, stopped_by)

    def _post(self, lp):
        """Take the LP's sends since its last request as its send log."""
        person = self.lps[lp]
        log = []
        for channel, seq, arrival, payload in person.outbox:
            target = self.table.channel(channel).target
            self.in_flight[target][channel, seq] = (arrival, payload)
            self.channels[channel] += 1
            log.append((channel, seq, arrival))
        self.messages += len(log)
        person.outbox = []
        return log

    def _check_cap(self):
        grants = sum(map(len, self.reply_log.values()))
        if self.grant_cap is not None and grants >= self.grant_cap:
            raise GrantsExhausted

    def _hand_over(self, lp, g, released):
        """The released messages, in `(arrival, channel, seq)` order."""
        messages = []
        for channel, pairs in released.items():
            for seq, a in pairs:
                arrival, payload = self.in_flight[lp].pop((channel, seq))
                if a != arrival or a > g:
                    raise SteppedOverEvent(
                        f"{lp} was handed {channel} seq {seq} arriving at {a} "
                        f"(sent for {arrival}) on a grant to {g}"
                    )
                messages.append((a, channel, seq, payload))
        messages.sort(key=lambda m: m[:3])
        self.handled += len(messages)
        self.handled_before_grant += sum(a < g for a, *_ in messages)
        return messages

    def _refuse_step_over(self, lp, g):
        """The one thing the authority cannot check about itself."""
        missed = sorted(
            (a, channel, seq)
            for (channel, seq), (a, _) in self.in_flight[lp].items()
            if a <= g
        )
        if missed:
            raise SteppedOverEvent(
                f"a grant to {g} moved {lp} past {len(missed)} message(s) it was "
                f"never handed, the earliest {missed[0]}\n"
                + "\n".join(map(repr, self.clock.lp_table()))
            )

    def _report(self, wall_seconds, stopped_by):
        finite = [g for log in self.reply_log.values() for g, _ in log if g < math.inf]
        return RunReport(
            deployment=self.deployment,
            submitted=tuple(self.submitted),
            requests={str(lp): dict(c) for lp, c in self.requests.items()},
            replies={str(lp): len(log) for lp, log in self.reply_log.items()},
            reply_log={str(lp): tuple(log) for lp, log in self.reply_log.items()},
            recovered=self.clock.recovered,
            steps=sum(p.steps for p in self.lps.values() if isinstance(p, Engine)),
            messages=self.messages,
            handled=self.handled,
            handled_before_grant=self.handled_before_grant,
            channels=dict(self.channels),
            timers=sum(p.timers for p in self.lps.values()),
            modelled_seconds=max(finite, default=0.0),
            wall_seconds=wall_seconds,
            unfinished=tuple(str(lp) for lp, p in self.lps.items() if not p.finished),
            stopped_by=stopped_by or "the driver ran out of requests",
        )
