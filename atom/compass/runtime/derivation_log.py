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

import atexit
import json
import os
import time
import weakref
from typing import Any, Optional

__all__ = ["record", "log", "watch", "DerivationLog"]


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


#: Caches whose final counters are wanted, held weakly so watching one cannot
#: keep a runner's graphs alive past the run.
_WATCHED: list = []


def watch(cache: Any, role: str) -> None:
    """Snapshot `cache`'s counters at exit, under `role`. No-op when off."""
    if not log().enabled:
        return
    if not _WATCHED:
        atexit.register(_write_counters)
    _WATCHED.append((role, weakref.ref(cache)))


def _counters_path() -> str:
    base, _ = os.path.splitext(log().path)
    return f"{base}.counters.{os.getpid()}.json"


def _write_counters() -> None:
    """One object per watched cache. Never raises."""
    try:
        rows = []
        for role, ref in _WATCHED:
            cache = ref()
            if cache is None:
                continue
            rows.append({
                "role": role,
                "at": now(),
                # The cache's own accessors, not a reconstruction: `hits` and
                # `derivations` are incremented on the two branches of one
                # `if`, so they partition the answered lookups and a refusal
                # is in neither.
                "templates": len(getattr(cache, "_templates", ()) or ()),
                "hits": getattr(cache, "hits", None),
                "representative_hits": getattr(cache, "representative_hits",
                                               None),
                "derivations": getattr(cache, "derivations", None),
                "derivation_seconds": getattr(cache, "derivation_seconds",
                                              None),
                "binds": getattr(cache, "binds", None),
                "refusals": len(getattr(cache, "refusals", ()) or ()),
                "describe": cache.describe(),
            })
        if not rows:
            return
        with open(_counters_path(), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
    except Exception:  # noqa: BLE001, S110 - see `record`
        pass
