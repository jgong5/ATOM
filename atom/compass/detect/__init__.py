# SPDX-License-Identifier: MIT
"""Five checks that a simulated run did not quietly break its own time order.

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
* `SetIterationLint` -- a static pass for members of a set read in the order the
  set happens to hold them. It **fails CI** for the same reason: the order is
  decided by a seed the interpreter picks per process, so the run it breaks is
  one nobody has run yet.
* `StepTable` and `compare` -- the record of what moved and when, and the diff
  of two runs of one configuration. It **fails CI** on the first row the two
  disagree on, and it is the only one of the five that needs two runs to say
  anything at all.

None of the five questions whether the rule that hands out time is sound. They
check the numbers that rule is fed and the record it produced: a declared delay
larger than the path it stands for, a wait nobody declared, a clock read nobody
substituted, an order nobody chose, and two runs that were meant to be one. If
one of them fires and none of those explains it, the problem is bigger than the
detector found.
"""

from .clock_source import (
    CLOCK_READS,
    DEFAULT_ALLOW_LIST,
    ClockRead,
    ClockSourceLint,
)
from .determinism import StepRow, StepTable, compare_step_tables
from .set_iteration import SetIteration, SetIterationLint
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
    "SetIteration",
    "SetIterationLint",
    "StepRow",
    "StepTable",
    "StragglerCheck",
    "WatchdogWarning",
    "compare_step_tables",
]
