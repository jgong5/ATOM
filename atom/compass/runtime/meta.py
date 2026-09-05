"""Derive a rank's op graph by running the model on meta tensors.

Meta tensors carry shape and dtype but no storage, so a forward pass propagates
shapes without computing anything or touching a GPU. That is how Compass obtains
a graph for a configuration nobody has run: ATOM's own model code decides what
operations to emit and how they are sharded, so no parallelism rules have to be
written down or kept in step.

The catch is custom operators. An ``aten`` op almost always has a meta kernel;
AITER's do not necessarily, and AITER registers lazily through JIT, so the set
that matters cannot be listed by reading the source — it has to be discovered by
running. :class:`MetaOpTracer` therefore records what a forward *did* execute and
what it *could not*, and the second list is the work to be done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from atom.compass.runtime import forward_ctx

from atom.compass.core.graph import OpGraph, OpSpec

__all__ = [
    "MissingMetaKernel", "MetaTrace", "MetaOpTracer",
    "derived_inputs", "AMBIGUOUS_GROUP",
]

# Collectives name the group they ran on; everything else is local compute.
_COLLECTIVE_HINTS = (
    "all_reduce", "allreduce", "all_gather", "allgather",
    "reduce_scatter", "broadcast", "all_to_all", "alltoall",
)


#: Recorded when an operator is a collective but the group it ran on cannot be
#: determined. Distinct from ``None``, which asserts local computation.
AMBIGUOUS_GROUP = "?"


def _is_collective(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in _COLLECTIVE_HINTS)


def _resolve_group(name: str, topology: Optional[dict]) -> Optional[str]:
    """Name the communication group a collective ran on.

    The op graph's one concession to parallelism is that a collective names its
    group; the shapes around it carry everything else. So a collective recorded
    without a name is a graph that cannot distinguish an all-reduce over tensor
    ranks from one over expert ranks — which is the whole distinction the
    representation exists to preserve.

    The dispatcher does not hand us the group. What it does hand us is enough
    when the rank belongs to only one group of size greater than one: there is
    nothing else the collective could have run on. With several such groups the
    ambiguity is real, and is recorded as such rather than guessed at.

    Resolving the remaining case means intercepting at the group object rather
    than the dispatcher — ATOM routes collectives through ``get_tp_group()`` and
    friends, which know their own identity. That is the replacement for this
    function, not an addition to it.
    """
    if not _is_collective(name):
        return None
    candidates = [g for g, size in (topology or {}).items() if size > 1]
    return candidates[0] if len(candidates) == 1 else AMBIGUOUS_GROUP


#: Collectives whose output has the same shape as their input, and which ATOM
#: performs in place. These can be stood in for on meta by handing back the
#: tensor that went in — the graph still records that the collective happened,
#: on which group, over how many bytes, which is all a cost model needs.
#:
#: Deliberately not a catch-all. ``all_gather`` grows its output and
#: ``reduce_scatter`` shrinks it, so guessing "same shape" for those would
#: corrupt every downstream shape while looking like it worked. They are
#: reported as missing until each is given its own rule.
_SHAPE_PRESERVING = ("all_reduce", "allreduce", "broadcast")


def _collective_stand_in(name: str, tensors: list) -> Optional[Any]:
    """Result of a collective that cannot run, or None if we must not guess."""
    lowered = name.lower()
    if any(h in lowered for h in _SHAPE_PRESERVING) and tensors:
        return tensors[0]
    return None


def _shape_of(x: Any):
    return tuple(int(d) for d in x.shape) if isinstance(x, torch.Tensor) else None


def _flat_tensors(args, kwargs):
    out = []
    for value in list(args) + list((kwargs or {}).values()):
        if isinstance(value, torch.Tensor):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(v for v in value if isinstance(v, torch.Tensor))
    return out


@dataclass(frozen=True)
class MissingMetaKernel:
    """An operator that could not run on meta, and why."""

    name: str
    reason: str
    input_shapes: tuple

    def __str__(self) -> str:
        return f"{self.name}  ({self.reason})"


@dataclass
class MetaTrace:
    """Everything one meta forward revealed."""

    graph: OpGraph = field(default_factory=OpGraph)
    missing: list[MissingMetaKernel] = field(default_factory=list)
    seconds: float = 0.0
    completed: bool = False
    failure: Optional[str] = None

    def missing_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for m in self.missing:
            seen.setdefault(m.name, None)
        return list(seen)

    def report(self) -> str:
        lines = [
            "ATOMCompass meta probe",
            "=" * 66,
            f"  operators executed   : {len(self.graph)} "
            f"({len(self.graph.op_names())} distinct)",
            f"  derivation time      : {self.seconds:.4f}s",
            f"  forward completed    : {'yes' if self.completed else 'no'}",
        ]
        if self.failure:
            lines.append(f"  stopped by           : {self.failure}")
            tail = [op.name for op in self.graph.ops[-5:]]
            if tail:
                lines.append(f"  last operators ran   : {' -> '.join(tail)}")
                lines.append(
                    "  the blocker is whatever this model reaches next; if it is a"
                )
                lines.append(
                    "  Triton kernel it bypasses the dispatcher, so a meta kernel"
                )
                lines.append(
                    "  cannot help - it needs wrapping in a custom op that has one."
                )
        names = self.missing_names()
        lines += ["", f"  operators without a meta kernel: {len(names)}"]
        if names:
            lines += [f"    - {n}" for n in names]
            lines += [
                "",
                "  Each needs a shape/dtype propagation rule only — no math.",
                "  Register with torch.library.register_fake, then re-run: the",
                "  probe advances to the next one it cannot execute.",
            ]
        else:
            lines.append("    none — every operator ran on meta")
        lines.append("=" * 66)
        return "\n".join(lines)


#: Largest integer tensor whose contents are worth keeping. Metadata is a few
#: numbers per sequence; anything much bigger is data, and an artifact should not
#: grow with the size of a batch.
MAX_RECORDED_INTS = 4096


def _int_ranges_of(tensors) -> tuple:
    """How far each integer tensor argument reached, and whether it climbed.

    ``_int_values_of`` keeps contents, and cannot keep large ones -- an artifact
    must not grow with a batch -- so the block tables and per-token maps, which
    are exactly the tensors that decide how much memory a kernel walks, are the
    ones it drops. Three numbers describe them well enough: the span they
    covered and whether they were sorted. Rebuilt from that, an index tensor
    walks as many distinct blocks as the real one did instead of re-reading
    block zero.

    A device-to-host copy per tensor, so trace mode only, like its neighbour.
    """
    import torch

    out = []
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        if t.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint8,
                           torch.int8):
            continue
        flat = t.reshape(-1)
        low = int(flat.min())
        high = int(flat.max())
        climbing = (flat.numel() < 2
                    or bool(torch.all(flat[1:] >= flat[:-1])))
        out.append((i, (low, high, climbing)))
    return tuple(out)


def _int_values_of(tensors) -> tuple:
    """Contents of the small integer tensor arguments, by position.

    Shapes say how much memory an operator touches, not how much work it does.
    Attention reads as much KV cache as ``context_lens`` says it should, so a
    benchmark given a zero-filled tensor of the right shape measures something
    else entirely -- it priced one step's attention above the cost of the whole
    step. Floating-point arguments are skipped: they are the data, they are
    large, and their values do not decide what a kernel costs.

    Each read is a device-to-host copy, so this is confined to trace mode, which
    already runs eagerly and exists to produce an artifact rather than to serve.
    """
    import torch

    out = []
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint8,
                           torch.int8, torch.bool):
            continue
        if t.numel() == 0 or t.numel() > MAX_RECORDED_INTS:
            continue
        try:
            out.append((i, tuple(int(x) for x in t.flatten().tolist())))
        except Exception:  # noqa: BLE001 - a value that will not move is skipped
            continue
    return tuple(out)


def _scalars_of(args, kwargs) -> tuple:
    """An operator's non-tensor arguments, in a form an artifact can hold.

    Shapes do not describe a call. `aiter::rmsnorm2d_fwd_` takes an `eps` and
    raises without one, so a graph recording only tensors cannot be replayed to
    price it -- which is how 113 of 330 operators went unpriced.

    Positional arguments are named by position, since the dispatcher does not
    hand over the schema's parameter names. Only values a JSON artifact can hold
    are kept: a tensor is already recorded as a shape, and anything else is
    dropped rather than turned into a string that cannot be passed back.
    """
    keep = (bool, int, float, str, type(None))
    out = []
    for i, a in enumerate(args):
        if isinstance(a, keep):
            out.append((f"#{i}", a))
        elif isinstance(a, (list, tuple)) and a and all(
                isinstance(x, (bool, int, float)) for x in a):
            out.append((f"#{i}", list(a)))
    for k, v in (kwargs or {}).items():
        if isinstance(v, keep):
            out.append((k, v))
        elif isinstance(v, (list, tuple)) and v and all(
                isinstance(x, (bool, int, float)) for x in v):
            out.append((k, list(v)))
    return tuple(out)


class MetaOpTracer(TorchDispatchMode):
    """Records every dispatched operator, and any that meta cannot execute.

    A missing meta kernel raises rather than returning, so the trace stops at the
    first one on any given run. That is why discovery is iterative: stub the
    reported operator, run again, learn the next.
    """

    def __init__(self, graph: Optional[OpGraph] = None,
                 topology: Optional[dict] = None) -> None:
        super().__init__()
        self.graph = graph if graph is not None else OpGraph()
        #: Group sizes this rank participates in, e.g. ``{"tp": 2}``. Used only
        #: to name the group a collective ran on; see :func:`_resolve_group`.
        self.topology = dict(topology or {})
        self.missing: list[MissingMetaKernel] = []
        self._t0 = 0.0
        self.seconds = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return super().__enter__()

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return super().__exit__(*exc)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(getattr(func, "name", lambda: func)() if callable(
            getattr(func, "name", None)) else func)
        tensors = _flat_tensors(args, kwargs)
        in_shapes = tuple(s for s in (_shape_of(t) for t in tensors) if s is not None)
        dtypes = tuple(str(t.dtype).replace("torch.", "") for t in tensors)

        # A collective on meta has no group to talk to and no storage to send.
        # Standing in for the shape-preserving ones is what lets a single
        # process derive a sharded rank's graph: the collective is recorded --
        # its group, its bytes -- without a peer having to exist.
        stand_in = None
        if _is_collective(name) and any(
            isinstance(t, torch.Tensor) and t.device.type == "meta" for t in tensors
        ):
            stand_in = _collective_stand_in(name, tensors)
            if stand_in is None:
                self.missing.append(
                    MissingMetaKernel(
                        name=name,
                        reason="collective changes shape; no stand-in rule yet",
                        input_shapes=in_shapes,
                    )
                )
                raise NotImplementedError(
                    f"{name}: collective is not shape-preserving, so meta "
                    "derivation cannot synthesise its result"
                )
            out = stand_in
        try:
            out = out if stand_in is not None else func(*args, **kwargs)
        except NotImplementedError as exc:
            self.missing.append(
                MissingMetaKernel(name=name, reason=_short(exc), input_shapes=in_shapes)
            )
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the report
            self.missing.append(
                MissingMetaKernel(
                    name=name, reason=f"{type(exc).__name__}: {_short(exc)}",
                    input_shapes=in_shapes,
                )
            )
            raise

        outs = out if isinstance(out, (list, tuple)) else (out,)
        out_shapes = tuple(
            s for s in (_shape_of(o) for o in outs) if s is not None
        )
        self.graph.add(
            OpSpec(
                name=name,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
                dtypes=dtypes,
                group=_resolve_group(name, self.topology),
                scalars=_scalars_of(args, kwargs),
                int_values=_int_values_of(tensors),
                int_ranges=_int_ranges_of(tensors),
                # An operator that reads ambient state needs that state recorded
                # with it; its arguments do not describe it, and cannot be made
                # to. Empty for everything but attention.
                context=forward_ctx.capture(name),
            )
        )
        return out


def _short(exc: BaseException, limit: int = 120) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("…" if len(text) > limit else "")


def derived_inputs(tokens: int, device="meta"):
    """The token and position tensors ATOM's runner would hand the model.

    The dtypes are part of the contract, not a detail, and the two differ:
    ATOM stages ``input_ids`` as ``int32`` and ``positions`` as ``int64``
    (``model_runner.py`` lines 189 and 1277). They are easy to get wrong in the
    same way and the consequence is disproportionate — a derivation using
    PyTorch's ``int64`` default produces a graph whose embedding differs from
    the captured one in dtype alone, and one using ``int32`` for both diverges
    at the first attention operator instead. Either way the comparison rejects
    every operator from that point on, and the rejection reads as a real
    structural disagreement rather than a wrong probe.
    """
    import torch

    return (
        torch.zeros(tokens, dtype=torch.int32, device=device),
        torch.arange(tokens, dtype=torch.int64, device=device),
    )
