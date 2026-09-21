# SPDX-License-Identifier: MIT
"""The receive-side check: a message may not take effect in the past.

Every message that crosses between participants already passes through a send
and a receive wrapper, and the wrapper stamps two numbers on it: when the sender
sent it, and when it takes effect on the receiver. The second is the one that is
checked, because a message legally takes effect after the sender sent it -- one
declared floor later at the earliest -- so the send time on its own says nothing
about whether the receiver is late.

The check is `takes_effect_at >= now` on the receiver, one float comparison per
message, and it is the direct test of the property the whole coordination
protocol exists to provide. A failure means the receiver has already decided
what it does after that moment, so the schedule it produced is not the schedule
the modelled system would have produced. Nothing downstream of it is salvage:
the check raises and the run stops.

**Why the sender cannot make this check instead.** The sender knows the delay it
declared; it does not know where the receiver's clock stands. The coordinator
knows both, but only for the messages it was told about, and the messages most
likely to be wrong are exactly the ones nobody told it about -- a socket hop
that was annotated rather than intercepted carries its timestamps on the wire
and is invisible to the coordinator by design. The receiver is the only place
where the stamp and the clock are both in hand.

**What the report has to be good for.** Two different defects land here and they
have opposite fixes, so the report works out which it is. If the declared floor
is larger than the delay the message actually suffered, the declaration is the
defect and lowering it to the observed delay is the whole repair. If the floor
already matches the path, the floor is innocent: the receiver was moved past an
arrival that nothing had told the coordinator about, and the repair is on the
send path, not in the constant.
"""

import math
from dataclasses import dataclass


class CausalityViolation(Exception):
    """A message took effect behind the receiver's clock. The run is over.

    Carries the formatted report as `report`, and the numbers that produced it
    as fields, so a caller can act on either without parsing text.
    """

    def __init__(self, arrival: "Arrival", now: float, report: str) -> None:
        super().__init__(report)
        self.arrival = arrival
        self.now = now
        self.report = report


@dataclass(frozen=True)
class Arrival:
    """One message as the receiver sees it, with the two stamps it carries.

    `declared_floor_seconds` is the delay the sender promised on this ordered
    pair. It is carried on the message rather than looked up, because the
    receiver is checking the promise and has to see the promise that was made.
    """

    sender: str
    receiver: str
    sent_at: float
    takes_effect_at: float
    declared_floor_seconds: float

    @property
    def observed_delay_seconds(self) -> float:
        """What the path actually cost, against what was declared for it."""
        return self.takes_effect_at - self.sent_at


class StragglerCheck:
    """One comparison per inbound message, and a count of how many it made.

    `enabled=False` exists so a run can be made to demonstrate what it looks
    like without the check -- which is a run that finishes and reports numbers.
    It is not a production setting and nothing turns it off by configuration.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.checked = 0
        self.violations = 0

    def arriving(self, arrival: Arrival, now: float, context: str = "") -> Arrival:
        """Check one message against the receiver's clock and return it.

        `context` is whatever the caller can say about the state of every
        participant at this moment. It is appended verbatim, because the value
        of one of these is in seeing which clock was where when it fired.
        """
        if not self.enabled:
            return arrival
        self.checked += 1
        if arrival.takes_effect_at >= now:
            return arrival
        self.violations += 1
        report = self.report(arrival, now)
        if context:
            report = f"{report}\n\n{context}"
        raise CausalityViolation(arrival, now, report)

    @staticmethod
    def report(arrival: Arrival, now: float) -> str:
        """What a violation prints. The pair, both clocks, and which fix applies.

        The floor and the delay reach here by different arithmetic and a link
        sized exactly at its path agrees only to within rounding, so "larger
        than the path" is asked for as a real difference rather than as `>`. A
        floor out by one part in a billion is not what moved anybody.
        """
        behind = now - arrival.takes_effect_at
        declared = arrival.declared_floor_seconds
        observed = arrival.observed_delay_seconds
        if declared > observed and not math.isclose(declared, observed, rel_tol=1e-9):
            cause = (
                f"the declared floor on {arrival.sender} -> {arrival.receiver} is "
                f"{declared:.9g}s but the message took {observed:.9g}s, so the "
                f"declaration is {declared - observed:.9g}s longer than the path it "
                f"stands for. Lower it to {observed:.9g}s or less"
            )
        else:
            cause = (
                f"the declared floor on {arrival.sender} -> {arrival.receiver} is "
                f"{declared:.9g}s and the message took {observed:.9g}s, so the "
                f"declaration is not what let this happen. {arrival.receiver} was "
                f"granted past an arrival nothing had declared -- look at the send "
                f"path, not at the constant"
            )
        return (
            f"causality violation: {arrival.sender} -> {arrival.receiver} took "
            f"effect {behind:.9g}s into the receiver's past\n"
            f"  sent at              {arrival.sent_at:.9g}s\n"
            f"  takes effect at      {arrival.takes_effect_at:.9g}s\n"
            f"  receiver's clock at  {now:.9g}s\n"
            f"  declared floor       {declared:.9g}s\n"
            f"{cause}.\n"
            f"Everything {arrival.receiver} decided after {arrival.takes_effect_at:.9g}s "
            f"was decided without this message, so the run stops here rather than "
            f"reporting numbers taken off it."
        )

    def summary(self) -> str:
        """One line for a run's record. The violation count has to read zero."""
        if not self.enabled:
            return "straggler check: disabled, so this run makes no claim about its time order"
        return (
            f"straggler check: {self.checked} message(s) checked, "
            f"{self.violations} violation(s)"
        )
