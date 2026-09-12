"""When each on-demand derivation happened, on the wall clock.

`SourceComposition.build_seconds` already says how long building the oracle
took, but that is only the derivation a run does while it is coming up. The
source oracle also derives a structure the first time the schedule shows it --
`TemplateGraphs.graph_for` calls its deriver on a miss -- and a miss has no
phase. A shape first seen mid-schedule is derived *inside* the served window,
so those seconds are already in the execution the gate divides by; a shape
derived during startup is not. Charging the whole term beside the execution
window counts the mid-schedule part twice, and charging none of it beside the
window drops the startup part.

Nothing in the process can tell the two apart, because a runner does not know
when the harness decided startup ended. So this records the interval rather
than the phase: one row per derivation, ``t0``/``t1`` on the wall clock, which
is the clock `cc_traces_run.py` stamps its own windows on. The merge
intersects the two and the containment is measured on both sides rather than
asserted on either.

Off unless ``ATOM_COMPASS_DERIVATION_LOG`` names a file, and a no-op when off.
The path is suffixed with the pid, because the engine core and each runner are
different processes and would otherwise interleave into one file -- the same
rule `lifecycle.py` follows, for the same reason.

`time.time()` and not the engine clock: this is a duration of machine time
spent producing a prediction, and a predicting server's own clock is virtual.
Recording it there would put the cost in the same time domain as the thing it
is a cost of, which is exactly the confusion the cost record exists to avoid.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

__all__ = ["record", "log", "DerivationLog"]


class DerivationLog:
    """One JSON object per derivation, flushed as it happens."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = None
        self._failed = False

    @property
    def enabled(self) -> bool:
        return bool(self.path) and not self._failed

    def record(self, t0: float, t1: float, **fields: Any) -> None:
        if not self.enabled:
            return
        if self._fh is None:
            base, ext = os.path.splitext(self.path)
            try:
                self._fh = open(
                    f"{base}.{os.getpid()}{ext or '.jsonl'}", "a", encoding="utf-8"
                )
            except OSError:
                self._failed = True
                return
        row = {"t0": float(t0), "t1": float(t1), "seconds": float(t1) - float(t0)}
        row.update(fields)
        try:
            self._fh.write(json.dumps(row, default=str) + "\n")
            self._fh.flush()
        except (OSError, TypeError, ValueError):
            self._failed = True


_LOG: Optional[DerivationLog] = None


def log() -> DerivationLog:
    global _LOG
    if _LOG is None:
        _LOG = DerivationLog(os.environ.get("ATOM_COMPASS_DERIVATION_LOG", ""))
    return _LOG


def record(t0: float, t1: float, **fields: Any) -> None:
    """Write one derivation's interval. Never raises: a witness to a cost
    must not be able to fail the run whose cost it is witnessing."""
    try:
        log().record(t0, t1, **fields)
    except Exception:  # noqa: BLE001, S110 - deliberately swallowed, see above
        pass


def now() -> float:
    """The clock the windows are stamped on, named once so it cannot drift."""
    return time.time()
