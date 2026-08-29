"""What a rank did, recorded as operators and shapes.

An op graph is the sequence of operations one rank executed for one batch,
with concrete shapes and dtypes. It carries no notion of tensor, expert or data
parallelism: a collective simply names the communication group it ran on, and
the shapes around it already reflect whatever sharding produced them. That is
what lets one representation serve every parallel strategy, including
combinations of them, without teaching Compass what any of them mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

__all__ = ["OpSpec", "OpGraph", "GraphKey"]


@dataclass(frozen=True)
class GraphKey:
    """Identifies the graph a rank produces for a batch.

    Attributes:
        model_id: Model identity, including anything that changes its structure
            (quantisation, for instance).
        topology: Communication group sizes, e.g. ``{"tp": 2, "dp": 4}``. A rank
            usually belongs to several groups at once.
        rank_coords: This rank's index within each group it belongs to.
        batch_signature: The batch's shape, kept exact.
    """

    model_id: str
    topology: tuple[tuple[str, int], ...]
    rank_coords: tuple[tuple[str, int], ...]
    batch_signature: tuple[int, ...]

    @staticmethod
    def of(
        model_id: str,
        topology: Mapping[str, int],
        rank_coords: Mapping[str, int],
        batch_signature,
    ) -> "GraphKey":
        return GraphKey(
            model_id=model_id,
            topology=tuple(sorted(topology.items())),
            rank_coords=tuple(sorted(rank_coords.items())),
            batch_signature=tuple(int(x) for x in batch_signature),
        )


@dataclass(frozen=True)
class OpSpec:
    """One operation, as executed.

    Attributes:
        name: Operator identity, e.g. ``aten::mm`` or ``aiter::fused_moe``.
        input_shapes: Shape of each tensor argument, in order.
        output_shapes: Shape of each tensor result, in order.
        dtypes: Dtype of each tensor argument, in order.
        group: For a collective, the communication group it ran on. ``None``
            for local computation.
    """

    name: str
    input_shapes: tuple[tuple[int, ...], ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[str, ...] = ()
    group: Optional[str] = None

    @property
    def is_collective(self) -> bool:
        return self.group is not None


@dataclass
class OpGraph:
    """The ordered operations a rank executed for one batch."""

    key: Optional[GraphKey] = None
    ops: list[OpSpec] = field(default_factory=list)

    def add(self, op: OpSpec) -> None:
        self.ops.append(op)

    def __len__(self) -> int:
        return len(self.ops)

    def op_names(self) -> list[str]:
        """Distinct operator names, in first-seen order."""
        seen: dict[str, None] = {}
        for op in self.ops:
            seen.setdefault(op.name, None)
        return list(seen)

    def counts(self) -> dict[str, int]:
        """How many times each operator ran."""
        out: dict[str, int] = {}
        for op in self.ops:
            out[op.name] = out.get(op.name, 0) + 1
        return out
