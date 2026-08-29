"""A fixed-cost oracle.

Every step costs the same regardless of what it contains, which is wrong in
every way that matters for prediction. It exists to prove the plumbing: if a
request completes and its timings come out of a virtual clock, the seams work,
and a real oracle is then a change of one class.
"""

from __future__ import annotations

from atom.compass.core.cost.base import CostOracle, StepCost, StepShape

__all__ = ["ConstantCostOracle"]


class ConstantCostOracle(CostOracle):
    """Costs prefill and decode steps at two fixed rates."""

    def __init__(
        self,
        prefill_seconds: float = 0.010,
        decode_seconds: float = 0.001,
    ) -> None:
        if prefill_seconds < 0.0 or decode_seconds < 0.0:
            raise ValueError("step costs must be non-negative")
        self._prefill_seconds = float(prefill_seconds)
        self._decode_seconds = float(decode_seconds)

    def estimate(self, shape: StepShape) -> StepCost:
        seconds = self._prefill_seconds if shape.is_prefill else self._decode_seconds
        phase = "prefill" if shape.is_prefill else "decode"
        return StepCost(seconds=seconds, breakdown={phase: seconds})

    def describe(self) -> str:
        return (
            f"ConstantCostOracle(prefill={self._prefill_seconds:.6f}s, "
            f"decode={self._decode_seconds:.6f}s)"
        )
