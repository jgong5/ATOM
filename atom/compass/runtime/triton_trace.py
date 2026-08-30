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

logger = logging.getLogger(__name__)

__all__ = ["TritonLaunch", "TritonLaunchTracer"]


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
        return tuple(int(x) if isinstance(x, int) else str(x) for x in resolved)
    return (resolved,)


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
        return self

    def __exit__(self, *exc) -> None:
        if self._original is not None and self._patched_cls is not None:
            self._patched_cls.run = self._original
            self._original = self._patched_cls = None

    def _record(self, fn, args, kwargs, tensors) -> None:
        kwargs = kwargs or {}
        name = getattr(getattr(fn, "fn", fn), "__name__", str(fn))
        constexprs = tuple(
            sorted(
                (k, v) for k, v in kwargs.items()
                if isinstance(v, (int, float, bool, str)) and k != "grid"
            )
        )
        launch = TritonLaunch(
            kernel=name,
            grid=_resolve_grid(kwargs.get("grid"), kwargs),
            arg_shapes=tuple(tuple(int(d) for d in t.shape) for t in tensors),
            arg_dtypes=tuple(str(t.dtype).replace("torch.", "") for t in tensors),
            constexprs=constexprs,
        )
        self.launches.append(launch)
        self.graph.add(
            OpSpec(
                name=f"triton::{name}",
                input_shapes=launch.arg_shapes,
                output_shapes=(),
                dtypes=launch.arg_dtypes,
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
