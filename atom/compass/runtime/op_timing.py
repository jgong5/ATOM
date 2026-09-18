"""How long each operator took, and whether they add up to the step.

The cost model predicts from a step's *shape*. Everything the graph work
produces — the operators, their shapes, the derivation that reproduces a rank
without the hardware — feeds none of it. Attributing cost per operator is what
would change that, and it needs a per-operator number that does not exist yet.

The obstacle is CUDA graphs. Per-operator timing requires dispatch, and dispatch
only happens eagerly; production replays a captured graph, which is one opaque
submission with nothing to observe inside it. On this model a replayed decode
step runs 8.9x faster than the same step eagerly. So these numbers are eager
numbers and are **not** production costs: what they can support is a summary of
the work a step performs, which a small calibrated function then maps onto the
replayed cost. That mapping is the next phase and is deliberately not assumed
here.

Before any of that is worth building, one thing has to hold: the operators must
account for the step that contains them. This module measures both in the same
forward — each operator between its own pair of CUDA events, and the whole
region between one more pair — so the comparison is free of the run-to-run
variance that has repeatedly misled this project. Two numbers, one step, no
cross-run inference.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)

__all__ = ["OpTimingTracer", "OpTiming"]


class OpTiming:
    """One operator's measured device time, in dispatch order."""

    __slots__ = ("index", "name", "seconds")

    def __init__(self, index: int, name: str, seconds: float) -> None:
        self.index = index
        self.name = name
        self.seconds = seconds

    def as_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "seconds": self.seconds}


class OpTimingTracer(TorchDispatchMode):
    """Times every dispatched operator, and the region containing them.

    Events are recorded during dispatch and read back only at ``__exit__``,
    after one synchronise. Reading each pair as it is produced would serialise
    the very pipeline being measured — the mistake that made an earlier version
    of the step timer report a machine 33% slower than the real one.

    Nothing here interprets an operator. A name, a position and a duration is
    all that is kept; what the operator *means* is the graph's business, and the
    two artifacts are joined by index.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pairs: list[tuple[int, str, torch.cuda.Event, torch.cuda.Event]] = []
        self._region: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._index = 0
        #: Filled at exit: per-operator timings, in dispatch order.
        self.timings: list[OpTiming] = []
        #: Filled at exit: the whole region, measured the same way.
        self.region_seconds = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._region = (start, end)
        return super().__enter__()

    def __exit__(self, *exc):
        result = super().__exit__(*exc)
        if self._region is not None:
            start, end = self._region
            end.record()
            # One synchronise, here, rather than one per operator.
            torch.cuda.synchronize()
            self.region_seconds = start.elapsed_time(end) / 1000.0
            self.timings = [
                OpTiming(i, name, a.elapsed_time(b) / 1000.0)
                for i, name, a, b in self._pairs
            ]
        return result

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(getattr(func, "name", lambda: func)() if callable(
            getattr(func, "name", None)) else func)

        if self._region is None:  # no device: nothing to time against
            return func(*args, **kwargs)

        began = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        began.record()
        out = func(*args, **kwargs)
        ended.record()
        self._pairs.append((self._index, name, began, ended))
        self._index += 1
        return out

    def summary(self) -> dict:
        """What was measured, and whether the parts account for the whole.

        ``covered`` is the fraction of the region the operators account for.
        Above 1 means they overlap — several operators in flight at once, each
        credited with wall time the others were also using — and below 1 means
        time passed inside the region that no operator claimed. Either way the
        gap is the thing to explain before per-operator costs can be summed into
        a step cost.
        """
        total = sum(t.seconds for t in self.timings)
        return {
            "operators": len(self.timings),
            "sum_of_operators": total,
            "region": self.region_seconds,
            "covered": (total / self.region_seconds
                        if self.region_seconds else float("nan")),
        }
