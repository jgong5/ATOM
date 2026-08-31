"""Injectable time source for the request path.

ATOM reads wall-clock time in a handful of places that decide observable
behaviour: queue-delay accounting, the scheduler's delay gate, and the
first-token timestamps behind TTFT. Those call sites go through this module so
a caller can substitute a different notion of "now".

The default :class:`WallClock` delegates straight to :mod:`time`, so ATOM
behaves exactly as it did before this module existed. :class:`VirtualClock`
advances only when told to, which lets a simulated run report timings for work
it never actually performed, and lets tests pin time to make scheduling
deterministic.

The clock is process-local. In a multi-process deployment the process that owns
scheduling owns the clock; workers report durations back to it rather than
holding a clock of their own.
"""

from __future__ import annotations

import time as _time
from typing import Optional, Protocol, runtime_checkable

__all__ = [
    "Clock",
    "WallClock",
    "VirtualClock",
    "get_clock",
    "set_clock",
    "reset_clock",
    "now",
    "perf_counter",
]


@runtime_checkable
class Clock(Protocol):
    """A source of the two time readings ATOM depends on."""

    def time(self) -> float:
        """Seconds since the epoch, comparable to :func:`time.time`."""
        ...

    def perf_counter(self) -> float:
        """Monotonic seconds, comparable to :func:`time.perf_counter`."""
        ...


class WallClock:
    """Real time. The default, and behaviourally identical to bare `time` calls."""

    __slots__ = ()

    def time(self) -> float:
        return _time.time()

    def perf_counter(self) -> float:
        return _time.perf_counter()

    @property
    def epoch(self) -> Optional[float]:
        """Real time has no start-of-run, so a declared offset is meaningless.

        None rather than 0.0: a caller offsetting from the Unix epoch would get
        a timestamp from 1970 and a duration in the billions, which is exactly
        the failure this property exists to prevent.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "WallClock()"


class VirtualClock:
    """Time that only moves when :meth:`advance` is called.

    ``time()`` is offset from a real epoch so timestamps remain plausible to
    anything that formats or logs them; ``perf_counter()`` starts at zero.
    Advancing is monotonic — a negative step is a bug, not a rewind.
    """

    __slots__ = ("_epoch", "_elapsed")

    def __init__(self, epoch: Optional[float] = None) -> None:
        self._epoch = _time.time() if epoch is None else float(epoch)
        self._elapsed = 0.0

    def time(self) -> float:
        return self._epoch + self._elapsed

    def perf_counter(self) -> float:
        return self._elapsed

    def advance(self, seconds: float) -> None:
        """Move time forward by ``seconds``."""
        if seconds < 0.0:
            raise ValueError(f"cannot advance a clock backwards: {seconds}")
        self._elapsed += float(seconds)

    @property
    def elapsed(self) -> float:
        """Virtual seconds since construction."""
        return self._elapsed

    @property
    def epoch(self) -> float:
        """Where this clock started, so an offset into the run can be placed.

        ``time()`` is ``epoch + elapsed`` and the epoch is a real timestamp, so
        a caller declaring "half a second into the run" has to add it. Passing
        0.5 straight through instead yields a first-token time in the billions
        minus an arrival of 0.5, which is not a duration.
        """
        return self._epoch

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"VirtualClock(elapsed={self._elapsed:.6f})"


_clock: Clock = WallClock()


def get_clock() -> Clock:
    """Return the process-wide clock."""
    return _clock


def set_clock(clock: Clock) -> Clock:
    """Install ``clock`` process-wide and return the one it replaced."""
    global _clock
    previous, _clock = _clock, clock
    return previous


def reset_clock() -> None:
    """Restore the default wall clock."""
    global _clock
    _clock = WallClock()


def now() -> float:
    """Current time in seconds since the epoch, per the installed clock."""
    return _clock.time()


def perf_counter() -> float:
    """Current monotonic reading, per the installed clock."""
    return _clock.perf_counter()
