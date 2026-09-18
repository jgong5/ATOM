"""Record Triton kernel launches instead of performing them.

Triton kernels are launched directly rather than dispatched, so they never reach
a ``TorchDispatchMode`` and a meta kernel cannot stand in for them. On meta
tensors the launch fails outright: there is no storage behind the pointers.

They do not need shape inference, though. AITER's Triton kernels take their
destination as an argument — ``run_pa_decode_gluon(output, q, k_cache, ...)`` —
so the caller has already allocated every output with the right shape by the
time the launch happens. Nothing about the result is decided by the kernel.

So tracing one is: record what it was asked to do, and skip it. The recorded
descriptor — kernel identity, launch grid, argument shapes and dtypes, and the
compile-time tile constants — is also exactly the key a cost model needs, which
is why this records the constexprs rather than only the tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from atom.compass.core.graph import OpGraph, OpSpec
from atom.compass.runtime.meta import _int_ranges_of, inside_an_operator

logger = logging.getLogger(__name__)

__all__ = ["TritonLaunch", "TritonLaunchTracer"]

#: Prefix for kernels torch.compile generated, as opposed to hand-written ones.
#: Worth keeping separate: a hand-written kernel is a stable identity a cost
#: model can be fitted to, while an inductor kernel is generated per compilation
#: and named by hash, so its cost has to come from its shapes and its fused body
#: rather than from recognising it again.
INDUCTOR_PREFIX = "inductor"


@dataclass(frozen=True)
class TritonLaunch:
    """One kernel launch, as requested.

    Attributes:
        kernel: Kernel function name.
        grid: Launch grid, resolved if it was given as a callable.
        arg_shapes: Shape of each tensor argument, in order.
        arg_dtypes: Dtype of each tensor argument, in order.
        constexprs: Compile-time constants — tile sizes, warp and stage counts.
            These change the kernel that runs, so a cost model needs them.
    """

    kernel: str
    grid: tuple = ()
    arg_shapes: tuple = ()
    arg_dtypes: tuple = ()
    constexprs: tuple = ()


def _jsonable(v: Any) -> bool:
    """Whether an argument is a value a JSON artifact can hold and hand back.

    Anything else -- a dtype, a device, a callable -- is not recorded, and a
    call carrying one cannot be rebuilt at all, so it is marked rather than
    rebuilt wrongly.
    """
    return v is None or isinstance(v, (int, float, bool, str))


#: Marks an origin that is a file to load rather than a module to import.
GENERATED = "path:"


def _origin(fn: Any) -> str:
    """Where the kernel can be got back from, or empty.

    A hand-written kernel is an attribute of the module that defined it, and
    that module attribute is the decorated `JITFunction` -- what a launch
    subscripts with a grid -- so `module:name` is enough to import it back.

    Inductor's kernels have no importable name: they are generated into a file
    under the codecache and loaded by inductor's own machinery. But the
    autotuner keeps the path, the file is named by a hash of the code it holds,
    and it defines the kernel at module level under the same name inductor calls
    it. So the file is the origin, and it is stable for as long as the codecache
    is -- which is the assumption this rests on, and the reason it is recorded
    as a path rather than pretended to be an import.
    """
    filename = getattr(fn, "filename", None)
    if filename:
        return f"{GENERATED}{filename}::{_kernel_name(fn)}"
    inner = getattr(fn, "fn", fn)
    module = getattr(inner, "__module__", "") or ""
    name = getattr(inner, "__name__", "") or ""
    if not module or not name or module.startswith("torch._inductor"):
        return ""
    return f"{module}:{name}"


def _kernel_name(fn: Any) -> str:
    """Best available identity for a kernel.

    Hand-written kernels carry their function name. Inductor's autotuner wraps a
    generated function and records the name it chose in ``inductor_meta``, which
    is more informative than the wrapper's own ``__name__``.
    """
    meta = getattr(fn, "inductor_meta", None)
    if isinstance(meta, dict) and meta.get("kernel_name"):
        return str(meta["kernel_name"])
    inner = getattr(fn, "fn", fn)
    return getattr(inner, "__name__", None) or str(inner)


def _is_meta(x: Any) -> bool:
    return isinstance(x, torch.Tensor) and x.device.type == "meta"


def _tensors(args, kwargs):
    out = []
    for v in list(args) + list((kwargs or {}).values()):
        if isinstance(v, torch.Tensor):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(t for t in v if isinstance(t, torch.Tensor))
    return out


def _resolve_grid(grid, meta) -> tuple:
    if grid is None:
        return ()
    try:
        resolved = grid(meta) if callable(grid) else grid
    except Exception:  # noqa: BLE001 - a grid we cannot resolve is still worth recording
        return ("<unresolved>",)
    if isinstance(resolved, (list, tuple)):
        return tuple(_as_dim(x) for x in resolved)
    return (_as_dim(resolved),)


def _as_dim(x: Any):
    """A grid dimension as an int where it is one, and text where it is not.

    A dimension arrives as whatever computed it -- a numpy integer, a tensor
    scalar, a plain int -- and testing `isinstance(x, int)` recorded the first
    two as their own repr, which reads back as an unresolved grid and leaves a
    perfectly resolvable kernel unpriced. Anything that converts cleanly to an
    integer is one; anything that does not stays text, which is what the
    unresolved marker is.
    """
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)


class TritonLaunchTracer:
    """Context manager that records Triton launches, skipping them on meta.

    Every launch is recorded, whether or not it runs, so that a graph captured
    on real hardware can be compared against one derived on meta. Only the
    launch itself is conditional, which makes this safe to leave enabled outside
    a meta run.
    """

    def __init__(self, graph: Optional[OpGraph] = None) -> None:
        self.graph = graph if graph is not None else OpGraph()
        self.launches: list[TritonLaunch] = []
        self._original = None
        self._patched_cls = None
        self._inductor_original = None
        self._inductor_cls = None
        # An inductor launch may reach JITFunction.run underneath. Recording it
        # at both levels would double-count the same kernel.
        self._in_inductor = False

    def __enter__(self) -> "TritonLaunchTracer":
        try:
            from triton.runtime.jit import JITFunction
        except ImportError:  # pragma: no cover - triton absent
            logger.debug("triton not importable; launches will not be traced")
            return self

        tracer = self
        original = JITFunction.run

        # The signature mirrors triton's exactly: `grid` and `warmup` are
        # keyword-only after *args, and triton calls this as
        # `self.run(grid=..., warmup=False, *args, **kwargs)`. Forwarding through
        # a generic (*args, **kwargs) re-binds them into the kernel's own
        # parameters and the launch fails with a duplicate-argument error.
        def run(self, *args, grid=None, warmup=False, **kwargs):  # noqa: ANN001
            tensors = _tensors(args, kwargs)
            # Record either way, so a graph captured on real hardware is
            # comparable to one derived on meta. Only the launch is conditional.
            tracer._record(self, args, dict(kwargs, grid=grid), tensors)
            if any(_is_meta(t) for t in tensors):
                return None
            return original(self, *args, grid=grid, warmup=warmup, **kwargs)

        JITFunction.run = run
        self._original, self._patched_cls = original, JITFunction
        self._patch_inductor()
        return self

    def _patch_inductor(self) -> None:
        """Also intercept the kernels ``torch.compile`` generates.

        Inductor does not launch through ``JITFunction.run``. It compiles each
        fused kernel and calls it through ``CachingAutotuner``, so a graph built
        only from the dispatcher and ``JITFunction`` silently omits everything
        inductor fused — and omits it without leaving a mark, which is worse
        than failing.

        This matters because compilation is on by default. A capture taken with
        it off describes a configuration nobody deploys.
        """
        try:
            from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        except ImportError:  # pragma: no cover - inductor absent or moved
            logger.debug("inductor autotuner not importable; fused kernels "
                         "will not be traced")
            return

        tracer = self
        original = CachingAutotuner.run

        def run(self, *args, stream=None, **kwargs):  # noqa: ANN001
            tensors = _tensors(args, kwargs)
            tracer._record(self, args, kwargs, tensors, prefix=INDUCTOR_PREFIX)
            if any(_is_meta(t) for t in tensors):
                return None
            previously = tracer._in_inductor
            tracer._in_inductor = True
            try:
                return original(self, *args, stream=stream, **kwargs)
            finally:
                tracer._in_inductor = previously

        CachingAutotuner.run = run
        self._inductor_original, self._inductor_cls = original, CachingAutotuner

    def __exit__(self, *exc) -> None:
        if self._original is not None and self._patched_cls is not None:
            self._patched_cls.run = self._original
            self._original = self._patched_cls = None
        if self._inductor_original is not None and self._inductor_cls is not None:
            self._inductor_cls.run = self._inductor_original
            self._inductor_original = self._inductor_cls = None

    def _record(self, fn, args, kwargs, tensors, prefix: str = "triton") -> None:
        if self._in_inductor and prefix == "triton":
            return  # already recorded one level up, as the fused kernel
        # A kernel launched inside a dispatched operator is already part of that
        # operator's price. Recording it separately adds it to the step twice:
        # the KV gather runs inside chunked-prefill attention, and pricing both
        # charged 4.4ms of gather on top of an attention price that already
        # contained 5.9ms of it.
        if inside_an_operator():
            return
        kwargs = kwargs or {}
        name = _kernel_name(fn)
        constexprs = tuple(
            sorted(
                (k, v) for k, v in kwargs.items()
                if isinstance(v, (int, float, bool, str)) and k != "grid"
            )
        )
        # Positional arguments that are not tensors, by index. A Triton kernel
        # takes its strides and extents as plain ints alongside its pointers --
        # the KV gather takes six -- and recording only the tensors loses both
        # the values and the positions of everything else, so the call cannot be
        # rebuilt. `None` is recorded too: it is an argument that holds a place.
        scalars = tuple(
            (f"#{i}", v) for i, v in enumerate(args)
            if not isinstance(v, torch.Tensor) and _jsonable(v)
        )
        unrebuildable = any(
            not isinstance(v, torch.Tensor) and not _jsonable(v) for v in args)
        launch = TritonLaunch(
            kernel=name,
            grid=_resolve_grid(kwargs.get("grid"), kwargs),
            arg_shapes=tuple(tuple(int(d) for d in t.shape) for t in tensors),
            arg_dtypes=tuple(str(t.dtype).replace("torch.", "") for t in tensors),
            constexprs=constexprs,
        )
        self.launches.append(launch)
        # `origin` is where the kernel can be imported from again. A torch
        # operator is found by name through `torch.ops`; a raw @triton.jit
        # kernel is only reachable as an attribute of the module that defined
        # it, and an inductor-generated one is not reachable at all.
        origin = "" if unrebuildable else _origin(fn)
        self.graph.add(
            OpSpec(
                name=f"{prefix}::{name}",
                input_shapes=launch.arg_shapes,
                output_shapes=(),
                dtypes=launch.arg_dtypes,
                scalars=scalars + constexprs,
                launch=(("grid", list(launch.grid)), ("origin", origin)),
                int_ranges=_int_ranges_of(tensors),
            )
        )

    def summary(self) -> str:
        if not self.launches:
            return "  triton launches      : none"
        counts: dict[str, int] = {}
        for launch in self.launches:
            counts[launch.kernel] = counts.get(launch.kernel, 0) + 1
        lines = [f"  triton launches      : {len(self.launches)} "
                 f"({len(counts)} distinct kernels)"]
        for kernel, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {n:5d}  {kernel}")
        return "\n".join(lines)
