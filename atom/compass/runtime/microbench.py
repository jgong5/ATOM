"""What each kernel costs, measured without instrumenting the run.

Timing operators individually inside a real forward does not work: on a decode
step at batch 4, wrapping each dispatch in its own pair of CUDA events reported
45.664 ms for a step that replays in 3.946 ms, and the 113 gemms alone read
15.368 ms. The instrumentation costs several times the kernel it measures,
because these kernels run for tens of microseconds.

So the kernels are priced away from the run instead: each distinct
``(name, input shapes, dtypes)`` is called a few thousand times inside **one**
pair of events and the total divided. Nothing is measured per call, so there is
nothing per call to pay for, and the result is a price list rather than a
recording of one step — reusable across every configuration that runs the same
kernel at the same shape, instead of being remeasured per deployment.

This has to run **inside the model-runner process, after warmup**. ``aiter``
registers its operators lazily through a JIT that fires on first call, so
``torch.ops.aiter.gemm_a16w16`` does not exist until something has called it —
and a parent process that created an engine and generated tokens still does not
have it, because the model runs in a worker. Warmup in the right process is what
makes the lookup succeed, and it also means the kernels priced are the
deployment's own, already autotuned for the shapes it uses.

A graph records every tensor argument in dispatch order, which is what makes a
call reconstructible: ``aiter::silu_and_mul`` is
``(Tensor(a0!) out, Tensor(a1!) input, float limit=0.)`` and both tensors are in
the record, so allocating them in order and letting the scalar default reproduces
it. Operators that need a non-default scalar cannot be rebuilt this way and are
reported unpriced rather than guessed at.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["price_graph", "signature_of"]


def signature_of(op: dict) -> str:
    """A kernel plus the shapes it ran on — what a price is a price *of*."""
    shapes = ";".join(",".join(str(d) for d in s) for s in op["input_shapes"])
    return f"{op['name']}|{shapes}|{','.join(op['dtypes'])}"


def _resolve(name: str):
    """Find the callable behind a recorded operator name.

    ``aten::slice.Tensor`` is namespace ``aten``, operator ``slice``, overload
    ``Tensor``. A bare name means the default overload.
    """
    import torch

    namespace, _, rest = name.partition("::")
    opname, _, overload = rest.partition(".")
    ns = getattr(torch.ops, namespace, None)
    if ns is None:
        return None
    op = getattr(ns, opname, None)
    if op is None or not overload:
        return op
    return getattr(op, overload, op)


def _make_tensor(shape, dtype_name: str):
    import torch

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        return None
    size = tuple(int(d) for d in shape)
    if dtype.is_floating_point:
        return torch.randn(size, dtype=dtype, device="cuda")
    if dtype is torch.bool:
        return torch.zeros(size, dtype=dtype, device="cuda")
    # Integer arguments are usually indices or lengths. Zero is in range for
    # anything that indexes, which a random value would not be.
    return torch.zeros(size, dtype=dtype, device="cuda")


def _time(callable_, iters: int, warmup: int) -> float:
    """Seconds per call, measured over ``iters`` calls and one pair of events."""
    import torch

    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    began = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    began.record()
    for _ in range(iters):
        callable_()
    ended.record()
    torch.cuda.synchronize()
    return began.elapsed_time(ended) / 1000.0 / iters


def price_graph(graph_path: str, iters: int = 2000,
                warmup: int = 20) -> dict[str, Any]:
    """Price every distinct operator signature in a captured graph.

    Returns the price list and what it could not reach. Coverage is reported by
    operator count *and* by how many of the graph's operators a priced signature
    accounts for, because the two differ enormously: a handful of signatures
    cover most of a step.
    """
    with open(graph_path, encoding="utf-8") as fh:
        graph = json.load(fh)
    ops = graph["ops"]

    counts: dict[str, int] = {}
    example: dict[str, dict] = {}
    for op in ops:
        sig = signature_of(op)
        counts[sig] = counts.get(sig, 0) + 1
        example.setdefault(sig, op)

    priced: dict[str, dict] = {}
    unpriced: dict[str, str] = {}
    for sig, op in example.items():
        fn = _resolve(op["name"])
        if fn is None:
            unpriced[sig] = "operator not registered in this process"
            continue
        try:
            args = [_make_tensor(s, d)
                    for s, d in zip(op["input_shapes"], op["dtypes"])]
        except Exception as exc:  # noqa: BLE001 - allocation can fail many ways
            unpriced[sig] = f"could not build inputs: {type(exc).__name__}"
            continue
        if any(a is None for a in args):
            unpriced[sig] = "unknown dtype"
            continue
        try:
            seconds = _time(lambda: fn(*args), iters, warmup)
        except Exception as exc:  # noqa: BLE001 - a call can fail many ways
            # Almost always a non-default scalar the record does not carry.
            unpriced[sig] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        priced[sig] = {
            "name": op["name"],
            "seconds": seconds,
            "occurrences": counts[sig],
        }

    ops_priced = sum(counts[s] for s in priced)
    return {
        "version": 1,
        "provenance": {
            "graph": graph_path,
            "iters": iters,
            "note": "steady state, one event pair per signature",
        },
        "coverage": {
            "signatures": len(counts),
            "signatures_priced": len(priced),
            "operators": len(ops),
            "operators_priced": ops_priced,
            "fraction_of_operators": ops_priced / len(ops) if ops else 0.0,
        },
        "prices": priced,
        "unpriced": unpriced,
    }
