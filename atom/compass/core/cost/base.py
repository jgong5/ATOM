"""The oracle interface and the shapes it is asked about."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

__all__ = ["StepShape", "StepCost", "CostOracle"]


@dataclass(frozen=True)
class StepShape:
    """What a single engine step was asked to compute.

    Sequence lengths are kept per request rather than reduced to a scalar.
    Collapsing them is the standard shortcut in this field and it is the
    standard source of error: a decode batch mixing short and long histories
    does not cost what its mean history would suggest.

    Attributes:
        num_scheduled_tokens: New tokens per request, in batch order.
        context_lens: KV history length per request, in batch order.
        num_prefill_tokens: Of the scheduled tokens, how many are prefill.
        topology: Communication group sizes, e.g. ``{"tp": 2, "dp": 4}``.
            Compass does not interpret these names; a group is a size and a
            membership, and the operators recorded against it carry the meaning.
        rank_coords: This rank's index within each group.
        capture_bucket: The batch size this step's CUDA graph replay was padded
            up to, or None when nothing was replayed (eager, or a build that
            never captured). A replay runs the padded bucket, not the batch, so
            cost steps at the ladder rather than rising with batch size --
            twelve sequences cost more than eight because twelve pads to
            sixteen. An engine-agnostic number: the runner resolves the ladder,
            since the padding rule is the engine's, and the oracle only sees
            which rung was used.
        compiled: Whether the step ran through a compiled graph without being
            replayed. ``None`` means the runner did not say, and is treated as
            eager. Between "replayed" and "eager" there is a third way a step
            runs, and it pays neither cost: a compiled step submits its kernels
            from generated code rather than through the dispatcher, so it does
            not pay eager dispatch, and it is not one submission, so it does not
            pay a replay's per-launch boundary. Measured, its idle is 5.7ms
            against the 21.5ms the eager term charges -- see the step-accounting
            section of DESIGN_NOTES.
    """

    num_scheduled_tokens: tuple[int, ...]
    context_lens: tuple[int, ...]
    num_prefill_tokens: int = 0
    topology: Mapping[str, int] = field(default_factory=dict)
    rank_coords: Mapping[str, int] = field(default_factory=dict)
    capture_bucket: int | None = None
    compiled: bool | None = None

    @property
    def batch_size(self) -> int:
        return len(self.num_scheduled_tokens)

    @property
    def total_tokens(self) -> int:
        return sum(self.num_scheduled_tokens)

    @property
    def num_decode_tokens(self) -> int:
        return self.total_tokens - self.num_prefill_tokens

    @property
    def is_prefill(self) -> bool:
        return self.num_prefill_tokens > 0


@dataclass(frozen=True)
class StepCost:
    """A predicted duration, and optionally where it went.

    Attributes:
        seconds: Total predicted wall time for the step.
        breakdown: Optional per-operator or per-category attribution. Present
            when the oracle can attribute; empty when it cannot. A gap analysis
            is the difference between two oracles' breakdowns.
    """

    seconds: float
    breakdown: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seconds < 0.0:
            raise ValueError(f"step cost must be non-negative, got {self.seconds}")


@runtime_checkable
class CostOracle(Protocol):
    """Predicts how long a step takes."""

    def estimate(self, shape: StepShape) -> StepCost:
        """Return the predicted cost of executing ``shape``."""
        ...

    def describe(self) -> str:
        """Short human-readable identity, for logs and artifact provenance."""
        ...
