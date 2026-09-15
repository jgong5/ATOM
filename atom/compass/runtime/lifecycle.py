"""A witness for the token lifecycle, on whichever side is running.

The short-run comparison in POC_STATUS E2a inferred a timestamp discrepancy
from a step table: the walk credited each request a token at the completion of
the step it appeared in, and the reported TTFT did not agree. That inference
cannot stand on its own. ATOM defers output by one *meaningful* step, a pure
middle chunk takes no turn in the buffer, and `postprocess` skips any sequence
absent from `fwd_output` -- so "which step produced this request's first token"
is a question about three separate mechanisms:

* **buffering** -- which step's tokens a given `fwd_output` carries;
* **batch membership** -- which sequences that output names, and which of them
  `postprocess` actually walks;
* **publication** -- when `first_token_time` is stamped, on which clock.

A step table records none of the three. This records all three, on both a real
run and a simulated one, so the two contracts can be compared at the events
themselves rather than reconstructed from durations.

Off unless ``ATOM_COMPASS_LIFECYCLE`` names a file, and a no-op when off: this
sits inside `Scheduler.postprocess`, which runs every step of every deployment.
The path is suffixed with the pid, because the engine core and each runner are
different processes and would otherwise interleave into one file.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

__all__ = ["trace", "LifecycleTrace"]


class LifecycleTrace:
    """One JSON object per lifecycle event, flushed as it happens.

    Flushed per event rather than buffered: the runs worth examining are the
    ones that end badly, and a buffer is lost exactly then.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = None
        self._failed = False

    @property
    def enabled(self) -> bool:
        return bool(self.path) and not self._failed

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        if self._fh is None:
            base, ext = os.path.splitext(self.path)
            try:
                self._fh = open(f"{base}.{os.getpid()}{ext or '.jsonl'}", "w",
                                encoding="utf-8")
            except OSError:
                self._failed = True
                return
        row = {"event": event}
        row.update(fields)
        # The engine's own clock -- virtual on a simulated run, wall on a real
        # one. Recording `time.time()` here instead would put the witness in a
        # different time domain from the thing it is witnessing, which is the
        # very confusion it exists to resolve.
        try:
            from atom.utils.clock import get_clock

            row["t"] = get_clock().time()
        except Exception:  # noqa: BLE001 - a witness must never break a run
            row["t"] = None
        try:
            self._fh.write(json.dumps(row, default=str) + "\n")
            self._fh.flush()
        except (OSError, TypeError, ValueError):
            self._failed = True


_TRACE: Optional[LifecycleTrace] = None


def trace() -> LifecycleTrace:
    global _TRACE
    if _TRACE is None:
        _TRACE = LifecycleTrace(os.environ.get("ATOM_COMPASS_LIFECYCLE", ""))
    return _TRACE
