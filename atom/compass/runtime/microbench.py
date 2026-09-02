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
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["price_graph", "signature_of"]


def signature_of(op: dict) -> str:
    """A kernel plus the shapes it ran on — what a price is a price *of*."""
    shapes = ";".join(",".join(str(d) for d in s) for s in op["input_shapes"])
    sig = f"{op['name']}|{shapes}|{','.join(op['dtypes'])}"
    # Two calls with the same shapes but different metadata are different
    # amounts of work -- one decode step reading 40 tokens of history and
    # another reading 4000 have identical signatures until the contents are part
    # of the key.
    values = op.get("int_values") or ()
    if values:
        sig += "|" + ";".join(
            f"{i}:" + ",".join(str(x) for x in v) for i, v in values)
    # Scalars belong in the key for the same reason. A decode attention passing
    # max_qlen=1 and one passing 16384 are the same shapes and the same tensors
    # and are not the same amount of work; without this they collapse to one
    # entry and whichever was seen first prices both.
    scalars = op.get("scalars") or ()
    if scalars:
        sig += "|" + ";".join(f"{k}={v}" for k, v in scalars)
    return sig


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


def _make_tensor(shape, dtype_name: str, values=None):
    """A stand-in for one tensor argument, from its shape and its contents.

    ``values`` are the recorded contents of a small integer tensor, and where
    they exist they are used, because for a data-dependent kernel the numbers
    decide the work. Attention reads as much KV cache as ``context_lens`` says;
    handed zeros it measured something that is not attention, and priced one
    step of it above the cost of the whole step.

    Without recorded values an integer tensor falls back to zeros -- in range
    for anything that indexes, where a random value would not be -- and the
    price that results should be read as describing the shape only.
    """
    import torch

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        return None
    size = tuple(int(d) for d in shape)
    if values is not None and not dtype.is_floating_point:
        flat = torch.tensor(list(values), dtype=dtype, device="cuda")
        if flat.numel() == int(torch.Size(size).numel()):
            return flat.reshape(size)
    if dtype.is_floating_point:
        return torch.randn(size, dtype=dtype, device="cuda")
    return torch.zeros(size, dtype=dtype, device="cuda")


def _rebuild_args(op: dict, tensors: list) -> tuple[list, dict]:
    """Put the tensors and scalars back in the order the operator wants them.

    The tracer records tensor arguments as an ordered list of shapes and
    non-tensor ones as ``(name, value)``, where a positional argument is named
    by its index. Interleaving them again recovers the call: for
    ``rmsnorm2d_fwd_(out, input, weight, eps)`` the first three positions come
    from the tensor list and the fourth from the scalars.
    """
    scalars = {k: v for k, v in (tuple(x) for x in op.get("scalars") or ())}
    positional = {int(k[1:]): v for k, v in scalars.items() if k.startswith("#")}
    keywords = {k: v for k, v in scalars.items() if not k.startswith("#")}

    args: list = []
    remaining = list(tensors)
    width = (max(positional) + 1) if positional else 0
    for i in range(max(width, len(tensors) + len(positional))):
        if i in positional:
            args.append(positional[i])
        elif remaining:
            args.append(remaining.pop(0))
        else:
            break
    args.extend(remaining)
    return args, keywords


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


#: Bytes of distinct input to cycle through in ``cold`` mode. Has to exceed the
#: last-level cache comfortably or the rotation is pointless; capped so a large
#: weight does not exhaust the device building copies of itself.
COLD_WORKING_SET_BYTES = 1 << 30


def _build_arg_sets(op: dict, cache: str, fn) -> Optional[list]:
    """One or many argument sets, depending on what cache state is wanted.

    Cache state is really a property of each *argument*, not of the kernel. In a
    real decode step a gemm's activation input is hot -- the previous operator
    just wrote it -- while its weight is cold, streamed from memory, and every
    one of the 113 gemms in a step uses a different weight. Calling one kernel
    repeatedly on one buffer measures the hot case for everything, which
    flatters any kernel that moves a lot of memory.

    ``hot`` reuses a single set: an upper bound on speed, and the right answer
    for a value the previous operator just produced. ``cold`` rotates over
    enough distinct sets to overflow the cache, which is what reading a weight
    or a KV block actually costs. ``graph`` is hot inputs with the launch
    amortised into a CUDA graph, which is the only one of the three that can
    price a kernel smaller than the harness's own call overhead.

    Doing this properly would price each argument separately, which needs to
    know which inputs a previous operator produced. The graph records shapes,
    not identity, so it cannot distinguish them today -- these two modes bracket
    the answer rather than giving it.
    """
    import torch

    recorded = {int(i): v for i, v in (op.get("int_values") or ())}

    def one():
        tensors = [_make_tensor(s, d, recorded.get(i))
                   for i, (s, d) in enumerate(
                       zip(op["input_shapes"], op["dtypes"]))]
        if any(t is None for t in tensors):
            return None
        return _rebuild_args(op, tensors)

    first = one()
    if first is None:
        return None
    if cache == "hot":
        return [first]

    per_set = sum(
        int(torch.empty(0, dtype=getattr(torch, d)).element_size())
        * max(1, int(torch.Size(tuple(sh)).numel()))
        for sh, d in zip(op["input_shapes"], op["dtypes"])
        if getattr(torch, d, None) is not None
    ) or 1
    n = max(2, min(64, COLD_WORKING_SET_BYTES // per_set))
    sets = [first]
    for _ in range(n - 1):
        nxt = one()
        if nxt is None:
            break
        sets.append(nxt)
    return sets


#: Calls captured into one graph. Large enough that replaying the graph costs
#: far more than submitting it, small enough to capture quickly.
#: Calls captured into one graph. A replay costs about 5.5 microseconds
#: whatever it contains, so a capture of one call charges that to one kernel and
#: overstates it; measured per-call cost falls from B=1 to B=8 and is flat
#: thereafter, at every shape from M=1 to M=2048. Flat is the important part --
#: it means the captured calls run one after another, as stream-ordered capture
#: implies, so what is measured is a latency and not a throughput. 64 is
#: comfortably past the knee.
GRAPH_BATCH = int(os.environ.get("COMPASS_GRAPH_BATCH", "64"))


#: Signatures whose key contains this substring get taken apart rather than just
#: priced. A price is a duration and says nothing about where the duration went;
#: this says whether it went into kernels or into the gaps between them, which is
#: the difference between "this kernel is slow" and "this measurement is wrong".
PROFILE_MATCH = os.environ.get("COMPASS_BENCH_PROFILE", "")


def _time_in_graph(fn, sets: list, iters: int, warmup: int) -> float:
    """Seconds per call, with the launch amortised the way production does.

    A per-call loop cannot price a kernel smaller than its own call overhead. On
    this hardware that overhead is about 30 microseconds, and a gemm of
    [M,1024]x[4096,1024] costs the same 30 microseconds for every M from 1 to
    256 -- 256 times the work for the same price. Decode-shape kernels are far
    below the floor, so a loop measures dispatch and reports it as kernel time.
    That is why summed prices came to 2.2x a step: 298 operators times a 30
    microsecond floor is 8.9ms, against a priced total of 9.2ms.

    Capturing the calls into a CUDA graph removes exactly what production
    removes -- the graph is submitted once and the kernels run back to back with
    no host in the loop. What is left is the work.
    """
    import torch

    n = len(sets)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(max(warmup, 3)):
            a, k = sets[i % n]
            fn(*a, **k)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(GRAPH_BATCH):
            a, k = sets[i % n]
            fn(*a, **k)

    replays = max(1, iters // GRAPH_BATCH)
    graph.replay()
    torch.cuda.synchronize()
    began = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    began.record()
    for _ in range(replays):
        graph.replay()
    ended.record()
    torch.cuda.synchronize()
    total = began.elapsed_time(ended) / 1000.0
    return total / (replays * GRAPH_BATCH), 0.0


def _time_isolated(fn, sets: list, iters: int, warmup: int) -> tuple[float, float]:
    """Host dispatch and kernel, in series -- **not** the kernel alone.

    Named for what it was meant to measure and kept for what it does measure.
    The intent was one kernel with nothing else in flight. But ``began`` is
    recorded before the call, the stream is empty so the device timestamps it at
    once, and the device then sits idle through the whole host dispatch before
    the kernel even arrives. What comes back is dispatch **plus** kernel, with
    no overlap between them, which is why it is the largest of the three numbers
    rather than the smallest:

    | M | loop | host | this | in graph |
    | --- | --- | --- | --- | --- |
    | 4 | 33.9 µs | 33.9 µs | 52.2 µs | 9.2 µs |
    | 1024 | 136.0 µs | 30.7 µs | 174.5 µs | 137.6 µs |

    At M=1024 that is 31 + 138 almost exactly. There is no way with this API to
    start the clock after dispatch and before execution, so a single call cannot
    be timed in isolation at all -- the alternatives are to pipeline (the loop,
    which measures ``max(host, kernel)``) or to capture (the graph, which
    measures throughput with overlap allowed).

    Its use is as the third leg of a triangle: three numbers that only fit
    together if host dispatch is around 31 µs and the kernel runs from 9 µs at
    M=4 to 138 µs at M=1024. Median rather than mean, so one scheduling hiccup
    does not decide a price.
    """
    import statistics

    import torch

    n = len(sets)
    for i in range(warmup):
        a, k = sets[i % n]
        fn(*a, **k)
    torch.cuda.synchronize()

    # Far fewer iterations than the loop modes: each one pays a synchronise.
    rounds = max(20, min(200, iters // 20))

    # Created once, outside the window. A torch.cuda.Event builds its underlying
    # CUDA event lazily on first record(), so constructing `ended` inside the
    # loop puts that construction *between* the two timestamps -- after fn() has
    # been dispatched and before the closing event is enqueued. Recording an
    # event again simply overwrites its timestamp, so two suffice for the run.
    began = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    began.record()
    ended.record()
    torch.cuda.synchronize()

    samples = []
    for i in range(rounds):
        a, k = sets[i % n]
        began.record()
        fn(*a, **k)
        ended.record()
        torch.cuda.synchronize()
        samples.append(began.elapsed_time(ended) / 1000.0)
    return statistics.median(samples), 0.0


def _time_over(fn, sets: list, iters: int, warmup: int) -> tuple[float, float]:
    """Seconds per call on the device, and seconds per call on the host.

    Both, because one without the other cannot say what was measured. CUDA
    events are timestamped when the *device* reaches them, so the elapsed time
    between them is device-side wall clock across the whole loop -- not a sum of
    kernel durations. If the host enqueues faster than the device drains, the
    queue stays full and that elapsed time is real kernel work. If the host is
    slower, the device runs dry and waits, and the waiting is inside the window.

    The host figure is the wall time of the enqueue loop alone, taken before any
    synchronise, so it measures Python plus the dispatcher plus the operator
    wrapper plus the driver call, and nothing of the kernel. When the two agree,
    the device was idle waiting for the host and the "price" is the host's.
    """
    import time as _time

    import torch

    n = len(sets)
    for i in range(warmup):
        a, k = sets[i % n]
        fn(*a, **k)
    torch.cuda.synchronize()
    began = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    began.record()
    host0 = _time.perf_counter()
    for i in range(iters):
        a, k = sets[i % n]
        fn(*a, **k)
    host = _time.perf_counter() - host0
    ended.record()
    torch.cuda.synchronize()
    return began.elapsed_time(ended) / 1000.0 / iters, host / iters


def price_graph(graph_path: str, iters: int = 2000, warmup: int = 20,
                cache: str = "hot") -> dict[str, Any]:
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

    # Nothing priced here is inside a live forward, and an operator that reads
    # ambient state must not silently inherit the last one. Capture leaves a
    # forward context installed -- the final rung of the ladder, one sequence at
    # the model's maximum context -- so attention, which reads its metadata from
    # the context rather than its arguments, walked 16384 tokens of KV whatever
    # it was handed. It priced at 163.7us against a true 23.0us, and was
    # invariant to every argument, because the arguments were never read.
    from atom.utils.forward_context import reset_forward_context

    priced: dict[str, dict] = {}
    unpriced: dict[str, str] = {}
    for sig, op in example.items():
        # Per signature, not once: an operator that rebuilds the context from
        # its arguments leaves that context behind for whatever is priced next.
        reset_forward_context()
        fn = _resolve(op["name"])
        if fn is None:
            unpriced[sig] = "operator not registered in this process"
            continue
        try:
            sets = _build_arg_sets(op, cache, fn)
        except Exception as exc:  # noqa: BLE001 - allocation can fail many ways
            unpriced[sig] = f"could not build inputs: {type(exc).__name__}"
            continue
        if sets is None:
            unpriced[sig] = "unknown dtype"
            continue
        try:
            timer = {"graph": _time_in_graph,
                     "isolated": _time_isolated}.get(cache, _time_over)
            seconds, host_seconds = timer(fn, sets, iters, warmup)
        except Exception as exc:  # noqa: BLE001 - a call can fail many ways
            unpriced[sig] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        priced[sig] = {
            "name": op["name"],
            "seconds": seconds,
            "occurrences": counts[sig],
            "cache": cache,
            "arg_sets": len(sets),
            # Host enqueue cost per call. Where this matches `seconds`, the
            # device was idle waiting and the price is the host's, not the
            # kernel's.
            "host_seconds": host_seconds,
        }
        if PROFILE_MATCH and PROFILE_MATCH in sig:
            try:
                _profile_signature(fn, sets, sig, iters, warmup)
            except Exception as exc:  # noqa: BLE001 - a probe must not stop a run
                print(f"### PROBE FAILED {type(exc).__name__}: {exc}", flush=True)

    ops_priced = sum(counts[s] for s in priced)
    return {
        "version": 1,
        "provenance": {
            "graph": graph_path,
            "iters": iters,
            "cache": cache,
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



def _profile_signature(fn, sets, sig: str, iters: int, warmup: int) -> None:
    """Four views of one operator, printed rather than returned.

    Priced one way a kernel is a number with nothing to check it against. These
    four disagree in ways that identify the fault: a loop whose device time
    equals its host time was waiting on the host; a captured batch whose per-call
    cost falls with batch size was paying capture overhead per call; and a
    profile whose kernels do not add up to the measured time has gaps in it.
    """
    import torch

    print(f"\n### PROBE {sig[:150]}", flush=True)

    dev, host = _time_over(fn, sets, iters, warmup)
    print(f"###  loop over {iters}: device {dev*1e6:8.2f}us  "
          f"host {host*1e6:8.2f}us  ratio {dev/max(host,1e-12):5.2f}", flush=True)

    global GRAPH_BATCH
    keep = GRAPH_BATCH
    try:
        for batch in (1, 4, 16, 64):
            GRAPH_BATCH = batch
            secs, _ = _time_in_graph(fn, sets, max(iters, batch), warmup)
            print(f"###  graph B={batch:<3d}: {secs*1e6:8.2f}us per call",
                  flush=True)
    finally:
        GRAPH_BATCH = keep

    # What the captured batch actually executes. If these kernels sum to far
    # less than the measured per-call cost, the batch is not kernel-bound and
    # the price is of something other than the kernel.
    from torch.profiler import ProfilerActivity, profile

    n = len(sets)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            a, k = sets[i % n]
            fn(*a, **k)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for i in range(GRAPH_BATCH):
            a, k = sets[i % n]
            fn(*a, **k)
    graph.replay()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()

    rows = []
    for entry in prof.key_averages():
        micros = float(getattr(entry, "device_time_total", 0.0) or 0.0)
        if micros > 0.0:
            rows.append((micros, entry.count, entry.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"###  profile of one replay of {GRAPH_BATCH}: device total "
          f"{total:9.1f}us -> {total/GRAPH_BATCH:8.2f}us per call", flush=True)
    for micros, count, key in rows[:15]:
        print(f"###    {micros:9.1f}us n={count:<5d} {key[:88]}", flush=True)
