# SPDX-License-Identifier: MIT
"""Three checks that a simulated run did not quietly break its own time order.

Breaking it does not crash. A participant that is moved past a message it had
not received still runs, still finishes, and still prints a latency table; the
table is simply wrong, by an amount too small to argue with. So each of these is
always on, and each fails in a different way because each catches a different
kind of mistake:

* `StragglerCheck` -- the receive side, one comparison per message that crosses
  between participants. It **fails the run**, because a message that landed in
  the past invalidates every number the receiver produced after that moment.
* `AnnotationWatchdog` -- a sampler that notices a participant sitting in real
  time while its simulated clock stands still. It **warns**, because the harm it
  predicts is a straggler, and the straggler check is what proves the harm
  happened.
* `ClockSourceLint` -- a static pass over the simulated path for reads of a real
  clock. It **fails CI**, on the day the read is added rather than on the day a
  number taken off it is disputed.

None of the three questions whether the rule that hands out time is sound. They
check the numbers that rule is fed: a declared delay larger than the path it
stands for, a wait nobody declared, a clock read nobody substituted. If one of
them fires and none of those three explains it, the problem is bigger than the
detector found.
"""

from .clock_source import (
    CLOCK_READS,
    DEFAULT_ALLOW_LIST,
    ClockRead,
    ClockSourceLint,
)
from .straggler import Arrival, CausalityViolation, StragglerCheck
from .watchdog import AnnotationWatchdog, WatchdogWarning

__all__ = [
    "CLOCK_READS",
    "DEFAULT_ALLOW_LIST",
    "AnnotationWatchdog",
    "Arrival",
    "CausalityViolation",
    "ClockRead",
    "ClockSourceLint",
    "StragglerCheck",
    "WatchdogWarning",
]
