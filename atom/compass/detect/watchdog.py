# SPDX-License-Identifier: MIT
"""A sampler that notices real time passing inside something nobody declared.

A participant declares itself idle before it waits for a peer and declares
itself running again when it is given time. The declaration is what lets the
coordinator look past it; without it the participant keeps its peers pinned at
its own clock plus one floor, and they crawl or stop.

The failure this watches for is a wait that nobody wrapped. It shows up as a
participant that has been executing for longer than any modelled step could take
while its simulated clock has not moved at all -- real seconds going by with no
simulated seconds charged for them. Two hundred milliseconds is far above any
step the cost model produces and far below any wait worth blocking on, so it
separates the two cleanly.

**This one warns; it does not stop the run.** An undeclared wait costs
correctness only when it also lets a message land in somebody's past, and the
receive-side check is what proves that happened. What this buys is the name of
the missing declaration while the inventory of waits is still being worked
through, rather than a wrong number six months later. It reports the innermost
frame the participant is standing in, because the annotation goes around that
call and nowhere else.

**Real time is the point here, so this module reads a real clock deliberately.**
It is on the clock-source allow-list for that reason: a watchdog measured in
simulated seconds would stop whenever the thing it is watching stopped, which is
exactly when it is needed.

Sampling is one thread for every participant rather than one per participant.
The thing being measured is a participant that is not running any code of its
own, so nothing is gained by asking it about itself, and a single sampler is one
thread to shut down instead of many.
"""

import contextlib
import logging
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)

#: Where the standard library lives, so a report can step out of it.
_STDLIB = sysconfig.get_paths()["stdlib"]

#: How long a participant may execute with its simulated clock standing still
#: before that is a missing declaration rather than a slow step.
DEFAULT_STALL_SECONDS = 0.2

#: How often the sampling thread looks, when one is used.
DEFAULT_SAMPLE_SECONDS = 0.02


@dataclass(frozen=True)
class WatchdogWarning:
    """One participant, stalled: how long, at what clock, and standing where."""

    lp_id: str
    running_for_seconds: float
    clock_at: float
    frame: str

    def __str__(self) -> str:
        return (
            f"annotation watchdog: {self.lp_id} has been executing for "
            f"{self.running_for_seconds:.3g}s of real time with its simulated clock "
            f"still at {self.clock_at:.9g}s. Real seconds are passing inside "
            f"something nothing declared, and until it is declared every peer is "
            f"bounded at {self.clock_at:.9g}s plus one floor. Innermost frame: "
            f"{self.frame}. The run continues."
        )


class AnnotationWatchdog:
    """Watches which participants are executing, and for how long in real time.

    `wall_clock` is injectable so a test can drive the sampler at exact
    durations instead of sleeping for them; the default is the machine's.
    """

    def __init__(
        self,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        wall_clock=time.monotonic,
        enabled: bool = True,
    ) -> None:
        self.stall_seconds = stall_seconds
        self.enabled = enabled
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._executing: dict[str, tuple[float, float, int]] = {}
        self._warned: dict[str, None] = {}
        self._warnings: list[WatchdogWarning] = []

    def running(self, lp_id, now: float) -> None:
        """This participant has been given time and is executing at `now`.

        Called again on every grant, which is what makes a moving clock look
        different from a stalled one: the recorded simulated time changes and
        the real-time count starts over.
        """
        with self._lock:
            self._executing[str(lp_id)] = (
                float(now),
                self._wall_clock(),
                threading.get_ident(),
            )
            self._warned.pop(str(lp_id), None)

    def parked(self, lp_id) -> None:
        """This participant has declared itself idle. Nothing to watch."""
        with self._lock:
            self._executing.pop(str(lp_id), None)
            self._warned.pop(str(lp_id), None)

    def sample(self) -> tuple[WatchdogWarning, ...]:
        """Look once at every executing participant; warn about the stalled ones.

        At most one warning per stall, because a sampler that fires every tick
        buries the name it exists to report.
        """
        if not self.enabled:
            return ()
        wall_now = self._wall_clock()
        fresh = []
        with self._lock:
            for lp_id in sorted(self._executing):
                clock_at, since, ident = self._executing[lp_id]
                running_for = wall_now - since
                if running_for < self.stall_seconds or lp_id in self._warned:
                    continue
                self._warned[lp_id] = None
                fresh.append(
                    WatchdogWarning(lp_id, running_for, clock_at, _frame_of(ident))
                )
            self._warnings.extend(fresh)
        for warning in fresh:
            _LOG.warning("%s", warning)
        return tuple(fresh)

    @property
    def warnings(self) -> tuple[WatchdogWarning, ...]:
        """Every warning raised so far, oldest first."""
        with self._lock:
            return tuple(self._warnings)

    @contextlib.contextmanager
    def sampling(self, interval: float = DEFAULT_SAMPLE_SECONDS):
        """Run the sampler on a background thread for the length of a block.

        The thread is a daemon and is joined on the way out, so a run that ends
        badly does not leave one behind.
        """
        stop = threading.Event()

        def loop():
            while not stop.wait(interval):
                self.sample()

        thread = threading.Thread(target=loop, name="compass-annotation-watchdog")
        thread.daemon = True
        thread.start()
        try:
            yield self
        finally:
            stop.set()
            thread.join(timeout=interval * 50)

    def summary(self) -> str:
        """One line for a run's record."""
        if not self.enabled:
            return "annotation watchdog: disabled"
        names = ", ".join(sorted({warning.lp_id: None for warning in self.warnings}))
        return f"annotation watchdog: {len(self.warnings)} warning(s)" + (
            f" on {names}" if names else ""
        )


def _frame_of(ident: int) -> str:
    """Where the stalled participant is standing: the innermost frame, and ours.

    The innermost frame is almost always inside the standard library, because
    that is where a blocking call ends up, and on its own it names the same
    three files whatever stalled. The first frame outside it is the call the
    missing declaration belongs around, so both are reported.

    This module's own frames are skipped first. A sampler on its own thread
    never sees them, but one called straight from the thread it is reporting on
    sees nothing else: the top of that stack is this function. Naming it would
    put the detector where the answer goes.

    The thread read here is the one that declared itself running, which is not
    always the one that is blocked -- a participant whose blocking read happens
    on a background thread it started is reported at the frame the declaration
    was made from.
    """
    frame = sys._current_frames().get(ident)
    if frame is None:
        return "thread gone"
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    if frame is None:
        return "nothing outside the watchdog"
    innermost = _where(frame)
    while frame is not None and frame.f_code.co_filename.startswith(_STDLIB):
        frame = frame.f_back
    if frame is None or _where(frame) == innermost:
        return innermost
    return f"{innermost} <- {_where(frame)}"


def _where(frame) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
