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
from typing import Any, Mapping, Optional

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
        int_values: Contents of the small integer tensor arguments, as
            ``(position, values)``. Shapes describe how much memory an operator
            touches; for a data-dependent kernel they do not describe how much
            work it does. Attention walks as much KV cache as ``context_lens``
            says, so a benchmark handed a zero-filled tensor of the right shape
            measures the wrong thing -- it priced one decode step's attention at
            more than the whole step cost. Only integer tensors, and only small
            ones: metadata is a handful of numbers per sequence, while the data
            an operator computes over is large and float, and its values do not
            decide the cost.
        context: Ambient state the operator reads that its arguments do not
            describe, as ``(name, value)`` pairs. Empty for all but a handful of
            operators. Attention takes its metadata from a module-global forward
            context, so a recorded call cannot be replayed without it -- and
            giving the operator those arguments instead does not work, because
            ``torch.compile`` constant-folds every one that is not a tensor. See
            ``atom.compass.runtime.forward_ctx``.
        scalars: The operator's non-tensor arguments, positional then keyword,
            as ``(name, value)`` pairs. Shapes alone do not describe a call:
            ``aiter::rmsnorm2d_fwd_`` takes an ``eps`` and refuses without one,
            so a graph that records only tensors cannot be replayed to find out
            what its operators cost. Kept only for values a JSON artifact can
            hold; anything else is dropped rather than guessed at.
        launch: How to launch a Triton kernel that is not a torch operator, as
            ``(name, value)`` pairs: ``grid`` and ``origin``. A torch operator
            can be found again from its name alone, through ``torch.ops``; a
            raw ``@triton.jit`` kernel cannot, and a grid is not an argument but
            decides how much work runs. Without both, a recorded Triton launch
            describes a kernel nobody can call back -- which left the KV gather
            of every chunked prefill unpriced. Empty for everything else.
    """

    name: str
    input_shapes: tuple[tuple[int, ...], ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[str, ...] = ()
    group: Optional[str] = None
    scalars: tuple[tuple[str, Any], ...] = ()
    int_values: tuple[tuple[int, tuple[int, ...]], ...] = ()
    context: tuple[tuple[str, Any], ...] = ()
    launch: tuple[tuple[str, Any], ...] = ()

    @property
    def is_collective(self) -> bool:
        return self.group is not None


@dataclass
class OpGraph:
    """The ordered operations a rank executed for one batch."""

    key: Optional[GraphKey] = None
    ops: list[OpSpec] = field(default_factory=list)
    #: How this graph came to exist — device, compilation level, tracer mode.
    #: A graph is compared long after it is written, often against one produced
    #: another way, and the conditions of its recording decide whether that
    #: comparison means anything. Carrying them in the artifact is the only way
    #: they survive the trip.
    provenance: dict = field(default_factory=dict)

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

    # -- persistence ---------------------------------------------------------
    #
    # A graph outlives the process that produced it: derivation and capture run
    # separately (ATOM registers attention layers globally, so one process can
    # only build a model once), and a derived graph is reused across a sweep
    # rather than recomputed.

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "key": None if self.key is None else {
                "model_id": self.key.model_id,
                "topology": [list(t) for t in self.key.topology],
                "rank_coords": [list(t) for t in self.key.rank_coords],
                "batch_signature": list(self.key.batch_signature),
            },
            "provenance": dict(self.provenance),
            "ops": [
                {
                    "name": op.name,
                    "input_shapes": [list(s) for s in op.input_shapes],
                    "output_shapes": [list(s) for s in op.output_shapes],
                    "dtypes": list(op.dtypes),
                    "group": op.group,
                    "scalars": [list(kv) for kv in op.scalars],
                    "int_values": [[i, list(v)] for i, v in op.int_values],
                    "context": [list(kv) for kv in op.context],
                    "launch": [list(kv) for kv in op.launch],
                }
                for op in self.ops
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpGraph":
        version = data.get("version")
        if version not in (1, 2):
            raise ValueError(f"unsupported op-graph version: {version!r}")
        key = None
        raw_key = data.get("key")
        if raw_key:
            key = GraphKey(
                model_id=raw_key["model_id"],
                topology=tuple(tuple(t) for t in raw_key["topology"]),
                rank_coords=tuple(tuple(t) for t in raw_key["rank_coords"]),
                batch_signature=tuple(raw_key["batch_signature"]),
            )
        graph = cls(key=key, provenance=dict(data.get("provenance") or {}))
        for op in data["ops"]:
            graph.add(
                OpSpec(
                    name=op["name"],
                    input_shapes=tuple(tuple(s) for s in op["input_shapes"]),
                    output_shapes=tuple(tuple(s) for s in op["output_shapes"]),
                    dtypes=tuple(op["dtypes"]),
                    group=op["group"],
                    # Absent from graphs written before scalars were recorded;
                    # those simply cannot be replayed to price their operators.
                    scalars=tuple(tuple(kv) for kv in op.get("scalars") or ()),
                    int_values=tuple(
                        (int(i), tuple(v)) for i, v in op.get("int_values") or ()
                    ),
                    context=tuple(tuple(kv) for kv in op.get("context") or ()),
                    launch=tuple(tuple(kv) for kv in op.get("launch") or ()),
                )
            )
        return graph

    def save(self, path) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path) -> "OpGraph":
        import json

        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
