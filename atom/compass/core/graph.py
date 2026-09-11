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


def _deaths_of(recorded) -> tuple:
    """A per-output death tuple, from either shape a record may carry."""
    if recorded is None:
        return ()
    if isinstance(recorded, int):  # a record written per operator, not output
        return (int(recorded),)
    return tuple(int(d) for d in recorded)


@dataclass(frozen=True)
class OpSpec:
    """One operation, as executed.

    Attributes:
        name: Operator identity, e.g. ``aten::mm`` or ``aiter::fused_moe``.
        input_shapes: Shape of each tensor argument, in order.
        output_shapes: Shape of each tensor result, in order. An operator that
            returns no tensor is an out-variant, and what it produces is the
            destination it was handed -- recorded here when no live tensor
            already owns that storage, because then the buffer was allocated
            somewhere a dispatch tracer cannot see. `torch.empty` inside a
            custom operator is such a place: re-entering the operator from
            `__torch_dispatch__` runs below the mode. At TP=1 the MLP's silu
            destination is 13.6 MB a layer and is exactly where the allocator's
            high-water mark sits.
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
        int_ranges: The span each integer tensor argument covered, as
            ``(position, (low, high, monotone))``. Where ``int_values`` records
            what an index tensor held, this records only its shape in the other
            sense -- how far it reached and whether it climbed -- which is three
            numbers however large the tensor, so it covers the block tables and
            per-token maps that ``int_values`` is too small to hold. A rebuilt
            index of zeros points every access at one block and prices a walk
            over the whole cache as a walk over one resident page; a rebuilt
            index spread across the recorded span does not. Safe where replaying
            the values themselves is not, because every tensor argument is
            rebuilt at its recorded shape: a value that indexed the real tensor
            in range indexes the rebuilt one in range too.
        inputs_from: For each tensor input, the index of the operator that
            produced it, or -1 for one this step did not produce -- a weight, an
            embedding table, a buffer allocated before the forward. Shapes say
            how much memory an operator *touches*; only knowing which tensor is
            which says how much is live at once, and that is the activation
            term of a memory budget. Recorded by matching storage addresses as
            the trace runs, because a graph of shapes alone cannot be walked for
            liveness -- two tensors of the same shape are indistinguishable in
            it. The address map forgets an address when the tensor holding it
            dies, since the allocator hands it straight back and the next
            tensor there is a different tensor.

        output_aliases: Per output, whether the operator allocated it. ``None``
            means it did; an index ``k >= 0`` means it wrote into the tensor
            operator ``k`` produced, and ``-1`` that it wrote into one from
            before the step. An in-place operator allocates nothing, and
            counting its output as a fresh tensor inflates the activation term
            by one tensor per in-place call -- at TP=2 the 57 in-place
            all-reduces made that 12.6%. Decided by whether the output's
            storage is one of the operator's own inputs, which cannot be
            confused with the allocator handing back a freed address.
        dies_at: Per output, the operator after which it was released, or -1 if
            it outlived the step. Per output rather than per operator because a
            fused add-and-norm's two outputs do not have the same life: the
            normed activation dies into the next gemm and the new residual
            carries to the end of the block. Recorded, not inferred: a tensor is
            freed
            when its last *Python reference* goes, which is not the same as its
            last read -- a local held across a block keeps a tensor alive long
            after the operator that last looked at it. Inferring from last-read
            also resurrects the dead, since the allocator hands a freed address
            straight back and a producer map that never forgets then credits
            the new tensor to whoever held that address before.
        layouts: Where a tensor argument sat inside its allocation, as
            ``(position, (stride, storage_offset, storage_elements,
            storage_key))``. A shape says how big a tensor is; it cannot say
            that the tensor is a *view* into something larger. A Triton kernel
            takes pointers and strides as separate plain ints, so when the
            argument was a view the stride it was handed belongs to the base
            allocation -- and a dense rebuild of the recorded shape, launched
            with that stride, walks off the end and faults the device (§8 of
            G4_TRANSFER). Recorded only where the shape cannot already say it:
            a contiguous tensor owning its whole storage alone gets no entry.
            ``storage_key`` is the position of the first argument sharing that
            storage, so two views of one buffer rebuild as two views of one
            buffer rather than as two unrelated tensors -- a different amount
            of traffic and a different price.
        param_names: What the kernel calls each positional argument, as
            ``(position, name)``. The tracer records a non-tensor positional
            argument as ``#7 = 16384`` -- a number with no meaning attached --
            and the only way to tell a stride from a token count then is to
            guess from its size. That guess is wrong in both directions: at 16k
            prefill ``num_tokens`` is 16384 and looks like a stride, while at
            batch 4 a genuine ``q_in_stride0`` of 14336 sits beside tensors
            whose rows are 6144. A ``@triton.jit`` kernel already declares the
            names, so they are recorded rather than reconstructed. Empty for a
            torch operator, whose arguments are not raw pointers and strides.
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
    int_ranges: tuple[tuple[int, tuple[int, int, bool]], ...] = ()
    layouts: tuple[tuple[int, tuple], ...] = ()
    param_names: tuple[tuple[int, str], ...] = ()
    inputs_from: tuple[int, ...] = ()
    output_aliases: tuple = ()
    dies_at: tuple = ()

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
                    "int_ranges": [[i, list(v)] for i, v in op.int_ranges],
                    "layouts": [[i, [list(v[0]), v[1], v[2], v[3]]]
                                for i, v in op.layouts],
                    "param_names": [[i, n] for i, n in op.param_names],
                    "inputs_from": list(op.inputs_from),
                    "output_aliases": list(op.output_aliases),
                    "dies_at": list(op.dies_at),
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
                    int_ranges=tuple(
                        (int(i), (int(v[0]), int(v[1]), bool(v[2])))
                        for i, v in op.get("int_ranges") or ()
                    ),
                    layouts=tuple(
                        (int(i), (tuple(int(s) for s in v[0]), int(v[1]),
                                  int(v[2]), int(v[3])))
                        for i, v in op.get("layouts") or ()
                    ),
                    param_names=tuple(
                        (int(i), str(n))
                        for i, n in op.get("param_names") or ()
                    ),
                    inputs_from=tuple(
                        int(i) for i in op.get("inputs_from") or ()),
                    output_aliases=tuple(
                        None if a is None else int(a)
                        for a in op.get("output_aliases") or ()),
                    dies_at=_deaths_of(op.get("dies_at")),
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
