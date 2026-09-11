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
import math
import os
from typing import Any, Optional

from atom.compass.runtime.triton_trace import GENERATED

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
    # Ambient state the arguments do not carry. One decode attention reading 40
    # tokens of history and another reading 4000 are the same operator on the
    # same shapes and are not the same amount of work, and after the operator
    # stopped pretending to be a function of its arguments this is the only
    # place that difference is recorded. `block_tables` is left out: it decides
    # which blocks are walked, not how many.
    context = op.get("context") or ()
    if context:
        sig += "|" + ";".join(
            f"{k}={v}" for k, v in (tuple(x) for x in context)
            if k != "block_tables")
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
    # A Triton launch's grid is not an argument and decides how much work runs,
    # so two launches of one kernel over different grids are different prices.
    # `origin` says where the kernel came from, not what it costs, so it is not
    # part of the key.
    for key, value in (tuple(x) for x in op.get("launch") or ()):
        if key == "grid":
            sig += "|grid=" + ",".join(str(x) for x in value)
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


def _resolve_triton(op: dict):
    """A callable for a Triton kernel that is not a torch operator.

    `torch.ops` cannot find these -- they are `@triton.jit` functions reached as
    attributes of the module that defined them -- so the graph records where to
    import one from and what grid it ran on, and this puts the two back
    together. The returned callable takes the kernel's arguments and supplies
    the grid, so the caller times it exactly as it times everything else.
    """
    import importlib

    launch = {k: v for k, v in (tuple(x) for x in op.get("launch") or ())}
    origin, grid = launch.get("origin"), launch.get("grid")
    if not origin:
        return None
    # An inductor kernel computes its own grid from its arguments, so unlike a
    # hand-written one it needs no grid recorded and must not be refused for
    # lacking it.
    if origin.startswith(GENERATED):
        return _resolve_generated(origin)
    if not grid:
        return None
    # An unresolved grid is recorded as text, and guessing one would price a
    # different amount of work than ran. Numeric text is a dimension that an
    # older graph stringified, and is read back rather than thrown away.
    try:
        dims = tuple(int(x) for x in grid)
    except (TypeError, ValueError):
        return None
    module, _, name = origin.partition(":")
    try:
        kernel = getattr(importlib.import_module(module), name, None)
    except Exception:  # noqa: BLE001 - an import can fail many ways
        return None
    if kernel is None:
        return None

    def call(*args, **kwargs):
        return kernel[dims](*args, **kwargs)

    return call


def _is_this_rank(path: str) -> bool:
    """Whether a generated module belongs to the rank running this process."""
    import re

    import torch

    found = re.search(r"[/_]rank_(\d+)[/_]", path)
    if not found:
        return True
    try:
        mine = (torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized() else 0)
    except Exception:  # noqa: BLE001 - no process group is rank 0
        mine = 0
    return int(found.group(1)) == mine


def _resolve_generated(origin: str):
    """Load an inductor-generated kernel from the file it was generated into.

    The file defines the kernel at module level, so loading it and taking the
    attribute gives the autotuner inductor would have called. Loading executes
    the module, which compiles the kernel -- once, and only for a graph that
    contains one.

    Returns None if the codecache has been cleared since the trace, or if the
    module will not execute standalone. Both leave the operator unpriced, which
    is what it was before, rather than priced against something else.
    """
    import importlib.util
    import os

    import torch

    path, _, name = origin[len(GENERATED):].rpartition("::")
    if not path or not name or not os.path.exists(path) or not _load_generated():
        return None
    spec = importlib.util.spec_from_file_location(
        "compass_generated_" + os.path.basename(path).partition(".")[0], path)
    if spec is None or spec.loader is None:
        return None
    # Another rank's generated module is not this process's to execute. Under
    # tensor parallelism the bench graph is the union of every rank's, and
    # inductor caches per rank (`.../rank_0/...`); loading one belonging to
    # another rank binds this process to a device it cannot use, and every
    # later allocation dies with "invalid device ordinal" far from the cause.
    # Restoring the current device afterwards is not enough -- what the module
    # does on import reaches further than that -- so the module is not loaded
    # at all. Each rank prices its own, which is what the union was for.
    if not _is_this_rank(path):
        return None
    previous = torch.cuda.current_device()
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - generated code can fail many ways
        return None
    finally:
        if torch.cuda.current_device() != previous:
            torch.cuda.set_device(previous)
    kernel = getattr(module, name, None)
    if kernel is None:
        return None

    def call(*args, **kwargs):
        return kernel.run(
            *args, stream=torch.cuda.current_stream().cuda_stream, **kwargs)

    return call


def _make_tensor(shape, dtype_name: str, values=None, span=None):
    """A stand-in for one tensor argument, from its shape and its contents.

    ``values`` are the recorded contents of a small integer tensor, and where
    they exist they are used, because for a data-dependent kernel the numbers
    decide the work. Attention reads as much KV cache as ``context_lens`` says;
    handed zeros it measured something that is not attention, and priced one
    step of it above the cost of the whole step.

    Without recorded values an integer tensor falls back to zeros -- in range
    for anything that indexes, where a random value would not be -- and the
    price that results should be read as describing the shape only.

    **Which is why replaying them is off by default.** A recorded index is only
    in range for the tensor it was recorded against, and replaying one into a
    rebuilt tensor faults the device rather than raising:

        index_elementwise_kernel ... HSA_STATUS_ERROR_EXCEPTION

    A GPU memory fault kills the process, so one bad index loses the whole run
    instead of leaving one signature unpriced -- which is why `price_graph`'s
    per-signature `try/except` never sees it.

    Attribution took two attempts and the first was wrong, so the evidence is
    worth stating. The fault appeared while pricing prefill graphs; it was
    blamed on replayed values, replay was disabled, and it *still* faulted --
    because the device was also oversubscribed by a leaked engine process
    holding 169 GB. Killing that made pricing succeed, and the success was
    credited to the kill. It was not: replay was off in that run too. With the
    device idle and replay back on, the fault returned. Two variables, one
    conclusion drawn too early.

    What is lost by using zeros is small and was once large. Recorded values
    existed because attention priced wrongly without them -- one step costing
    more than the whole step -- but attention now takes its metadata from the
    forward context Compass installs, so the values are needed to *key* a price
    and not to rebuild one. They are still recorded and still part of the
    signature. ``COMPASS_REPLAY_INT_VALUES=1`` restores the old behaviour.
    """
    import torch

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        return None
    size = tuple(int(d) for d in shape)
    if values is not None and REPLAY_INT_VALUES and not dtype.is_floating_point:
        flat = torch.tensor(list(values), dtype=dtype, device="cuda")
        if flat.numel() == int(torch.Size(size).numel()):
            return flat.reshape(size)
    if dtype.is_floating_point:
        return torch.randn(size, dtype=dtype, device="cuda")
    if span is not None and SYNTH_INT_RANGES:
        return _spread(size, dtype, span)
    return torch.zeros(size, dtype=dtype, device="cuda")


def _spread(size, dtype, span):
    """An index tensor that reaches as far as the recorded one did.

    Not the recorded values -- those are unsafe to replay -- but the same span
    and the same direction, which is what decides how much memory the kernel
    walks. A climbing tensor is a cumulative offset (`cu_seqlens`, whose ends
    are the only entries that matter, and which this reproduces exactly at the
    two-element sizes it usually has); anything else is a scattered index like a
    block table, and is spread across the span so successive entries address
    different blocks.

    In range by construction: every tensor argument is rebuilt at its recorded
    shape, so an index that was valid against the real tensor is valid against
    the rebuilt one. That is the difference between this and replaying recorded
    values into whatever shape a signature happened to carry, which faulted.
    """
    import torch

    low, high, climbing = int(span[0]), int(span[1]), bool(span[2])
    n = int(torch.Size(size).numel())
    if high <= low:
        return torch.full(size, low, dtype=dtype, device="cuda")
    if climbing:
        steps = torch.linspace(low, high, n, device="cuda")
    else:
        steps = low + torch.arange(n, device="cuda") % (high - low + 1)
    return steps.to(dtype).reshape(size)


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


def _operand_tensors(op: dict, recorded: dict, spans: dict):
    """One fresh set of tensor arguments, as views into what held them.

    Without recorded layout every argument is rebuilt dense and alone, which is
    wrong for any tensor that was a view: the fused QKV projection hands the
    norm kernel a ``[4, 24, 256]`` q whose ``stride0`` is the 14336-element row
    of the buffer it sits in, and a dense rebuild launched with that stride
    walks off the end and faults the device. With layout, the allocation is
    rebuilt first -- once per storage, at its recorded element count -- and each
    argument is an ``as_strided`` view into it, so q and k come back as two
    windows on one buffer rather than as two unrelated tensors.

    The reconstruction is checked rather than trusted: the furthest element each
    view addresses must fall inside the storage it claims, and every argument
    sharing a storage must agree on dtype. Either failing returns ``None`` and
    the signature goes unpriced, which is the same refusal as before -- what
    changes is that a layout the graph *does* record no longer triggers it.
    """
    import torch

    layouts = {int(i): tuple(v) for i, v in (op.get("layouts") or ())}
    shapes = [tuple(s) for s in op["input_shapes"]]
    dtypes = list(op["dtypes"])
    if any(i >= len(shapes) for i in layouts):
        return None

    # Check every layout before allocating any of them: a refusal should cost
    # nothing, and half-allocating a set that is about to be thrown away is how
    # a pricing run runs out of memory for the signatures that would have worked.
    for i, (stride, offset, elements, owner) in layouts.items():
        if len(stride) != len(shapes[i]) or offset < 0:
            return None
        reach = offset + sum(abs(int(s)) * (int(d) - 1)
                             for s, d in zip(stride, shapes[i])) + 1
        if reach > elements:
            return None
        if owner not in layouts or layouts[owner][3] != owner:
            return None
        if len({dtypes[j] for j, v in layouts.items() if v[3] == owner}) != 1:
            return None

    bases: dict[int, object] = {}
    for i, (_stride, _offset, elements, owner) in layouts.items():
        if owner in bases:
            continue
        base = _make_tensor((int(elements),), dtypes[owner])
        if base is None:
            return None
        bases[owner] = base

    tensors = []
    for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        if i in layouts:
            stride, offset, _elements, owner = layouts[i]
            view = torch.as_strided(
                bases[owner], shape, tuple(int(s) for s in stride), int(offset))
            if recorded.get(i) is not None or spans.get(i) is not None:
                dense = _make_tensor(shape, dtype, recorded.get(i), spans.get(i))
                if dense is None:
                    return None
                view.copy_(dense)
            tensors.append(view)
            continue
        t = _make_tensor(shape, dtype, recorded.get(i), spans.get(i))
        if t is None:
            return None
        tensors.append(t)
    return tensors


#: Print each signature before it is touched. A device memory fault kills the
#: process outright, so a `try/except` never sees it and the artifact never gets
#: written -- the last line printed is the only evidence of which operator did
#: it. Off by default because it is one line per signature.
ANNOUNCE = os.environ.get("COMPASS_ANNOUNCE", "") == "1"


def _announce(what: str, sig: str) -> None:
    if ANNOUNCE:
        print(f"### compass {what}: {sig[:150]}", flush=True)


def _message(exc: BaseException, head: int = 160, tail: int = 260) -> str:
    """The failure text, kept short but not decapitated at the wrong end.

    A Triton ``CompilationError`` opens with the source line it choked on and
    says what was actually wrong several lines later; truncated to its first 100
    characters it reads "at 176:34: + (conv_states_output_coord * strid" and
    names no cause at all. Keeping both ends fits the whole message for almost
    every failure and the two informative ends of the ones it does not.
    """
    text = " ".join(str(exc).split())
    if len(text) <= head + tail + 5:
        return text
    return f"{text[:head]} ... {text[-tail:]}"


def _where(exc: BaseException) -> str:
    """The innermost frame of a failure, as ``file:line`` in this codebase."""
    import traceback

    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if "/atom/" in f.filename] or frames
    if not ours:
        return ""
    last = ours[-1]
    return f" at {last.filename.split('/atom/')[-1]}:{last.lineno}"


def _is_collective_op(op: dict) -> bool:
    """Whether running this operator makes the rank talk to its peers.

    The recorded group is the honest answer where there is one; the name is a
    fallback for graphs written before groups were recorded.
    """
    if op.get("group") is not None:
        return True
    name = op.get("name", "")
    return name.startswith("c10d::") or "all_reduce" in name or "all_gather" in name


def _stride_past_its_tensors(op: dict):
    """A recorded integer that can only be a stride into memory not recorded.

    A Triton kernel takes pointers, not tensors. Its strides arrive as plain
    ints alongside them, and when the tensor it was handed was a *view*, that
    stride belongs to the allocation the view looked into -- which the graph
    never saw. Rebuilt here as a dense tensor of exactly its recorded shape and
    launched with the original stride, the kernel walks off the end.

    That is not an exception. It is a GPU memory access fault, and it kills the
    pricing run and every signature after it. `_fused_qk_norm_single_kernel`
    does it: q is a view into the fused qkv buffer, its recorded shape is
    [4, 24, 256] and its `q_in_stride0` is 14336, so row 3 addresses element
    49151 of 24576.

    The test needs no knowledge of any particular kernel. For a tensor argument
    [d0, *rest] with d0 > 1, its dense row extent is prod(rest), and a stride s
    used over d0 rows fits exactly when s <= that extent -- s*(d0-1)+R > d0*R is
    just s > R. So a positional integer larger than *every* argument's row
    extent cannot be a dense stride for any of them, and the graph cannot show
    it is not a stride at all. Constexprs are excluded: they are compiled into
    the kernel, not applied to a pointer, and `BLOCKS_PER_TILE=4096` is a tile
    size that prices correctly today.

    That size test is a guess about meaning, and it is wrong in both directions.
    The same kernel at a 16k prefill chunk takes `num_tokens = 16384` beside
    arguments whose rows are 6144, and the test refuses the whole family for a
    token count that addresses nothing. Where the graph records what the kernel
    calls its arguments (`param_names`) the guess is unnecessary: only a
    parameter the kernel *declares* as a stride can be applied to a pointer, and
    only such a parameter is examined. Nothing is inferred from a value's size.

    A declared stride is still refused unless the graph records the allocation
    it belongs to, either as one of that operator's `layouts` -- in which case
    the view is rebuilt inside a storage of the recorded element count, checked
    to fit before it is used (`_operand_tensors`) -- or by fitting a dense row
    of some argument, which is what the rebuild produces anyway.

    An operator whose names were never recorded falls back to the size test
    unchanged, layouts or not, because for it there is still nothing better.
    Returns the first refusal as (name, value, largest extent), or None. Both
    paths over-refuse, which is the direction to err: the other failure takes
    the whole run.
    """
    shapes = [tuple(s) for s in (op.get("input_shapes") or ())]
    extents = [math.prod(shape[1:])
               for shape in shapes
               if len(shape) >= 2 and shape[0] > 1]
    if not extents:
        return None
    limit = max(extents)
    known = {abs(int(s))
             for _, v in (op.get("layouts") or ())
             for s in tuple(v)[0]}
    params = {int(i): str(n) for i, n in (op.get("param_names") or ())}
    for key, value in (tuple(x) for x in op.get("scalars") or ()):
        if not (key.startswith("#") and isinstance(value, int)
                and not isinstance(value, bool) and value > limit):
            continue
        if value in known:
            continue
        if params and "stride" not in params.get(int(key[1:]), ""):
            continue
        return (key, value, limit)
    return None


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

    Neither mode reaches state an operator does not take as an argument. The KV
    cache is the case that matters: attention fetches it from the forward
    context, so rotating arguments leaves it resident across the whole batch
    however many sets there are. That one is cooled separately, by rotating the
    KV region each captured call addresses -- see ``KV_VARIANTS`` and
    ``forward_ctx.install``.
    """
    import torch

    recorded = {int(i): v for i, v in (op.get("int_values") or ())}
    spans = {int(i): v for i, v in (op.get("int_ranges") or ())}

    def one():
        tensors = _operand_tensors(op, recorded, spans)
        if tensors is None:
            return None
        return _rebuild_args(op, tensors)

    first = one()
    if first is None:
        return None
    if cache == "hot":
        return [first]

    # A set costs what it allocates, not what its arguments span: an argument
    # with a recorded layout is a window into a storage that may be far larger,
    # and two such arguments may share one. Count each storage once, at its
    # recorded size, and every other argument at its own.
    layouts = {int(i): tuple(v) for i, v in (op.get("layouts") or ())}
    elements: dict[object, tuple[int, str]] = {}
    for i, (sh, d) in enumerate(zip(op["input_shapes"], op["dtypes"])):
        if getattr(torch, d, None) is None:
            continue
        if i in layouts:
            elements[("storage", layouts[i][3])] = (int(layouts[i][2]), d)
        else:
            elements[("arg", i)] = (max(1, int(torch.Size(tuple(sh)).numel())), d)
    per_set = sum(
        int(torch.empty(0, dtype=getattr(torch, d)).element_size()) * n
        for n, d in elements.values()
    ) or 1
    n = max(2, min(64, COLD_WORKING_SET_BYTES // per_set))
    sets = [first]
    for _ in range(n - 1):
        nxt = one()
        if nxt is None:
            break
        sets.append(nxt)
    return sets


#: Calls captured into one graph. A replay costs a few microseconds whatever it
#: contains, so a capture of one call charges all of that to one kernel and
#: overstates it; per-call cost falls from B=1 to B=8 and is flat thereafter, at
#: every shape from M=1 to M=2048. Attention fits `c + r/B` with r = 6.1us and
#: c = 18.2us across B = 1, 4, 16, 64 to within 1%.
#:
#: Flat is the important part, and it is what says the captured calls run one
#: after another, as stream-ordered capture implies: concurrency would keep
#: driving per-call cost down as B grew instead of letting it asymptote. So what
#: is measured is a latency and not a throughput. 64 is comfortably past the knee.
GRAPH_BATCH = int(os.environ.get("COMPASS_GRAPH_BATCH", "64"))

#: Whether to write recorded integer contents back into rebuilt tensors. Off:
#: doing so faults the device while pricing prefill graphs. See `_make_tensor`
#: for the evidence, which took two attempts to read correctly.
REPLAY_INT_VALUES = os.environ.get("COMPASS_REPLAY_INT_VALUES", "") == "1"

#: Rebuild integer tensors across the span the traced ones covered, rather than
#: as zeros. On by default: unlike replaying the recorded values, which is not
#: safe and is off above, a value inside the recorded span indexes the rebuilt
#: tensor in range because the rebuilt tensor has the recorded shape. Set
#: ``COMPASS_SYNTH_INT_RANGES=0`` to go back to zeros.
SYNTH_INT_RANGES = os.environ.get("COMPASS_SYNTH_INT_RANGES", "1") != "0"

#: Load inductor-generated kernels from the codecache to price them.
#:
#: Off under parallelism, like `PRICE_KERNELS` and for a worse reason: at TP=2
#: it faults the device mid-pricing ("Memory access fault by GPU node-2"),
#: taking the whole run with it. Skipping other ranks' modules is necessary --
#: loading one leaves the process on a device it cannot use -- and is not
#: sufficient; a rank loading only its own still faults. Since a fault kills the
#: run rather than leaving one signature unpriced, the default is off where it
#: is known to happen.
#:
#: **What is lost is not small on a decode step.** The 2.0 ms this note used to
#: cite was 0.6% of a 316 ms *prefill*. A decode step is thirty times shorter
#: and runs the same per-layer generated kernels: on the 27B at TP=4 they are
#: 112 kernels and 0.465 ms, **4.8% of the step's kernel time**, with 128 of the
#: graph's 130 generated-kernel operators unpriced. The default trades a fault
#: for a systematic 5% hole, not for a rounding error.
#:
#: Executing generated code is the one part of pricing that runs code this
#: process did not write, so it has a way off at TP=1 too:
#: ``COMPASS_LOAD_GENERATED=0``.
LOAD_GENERATED = os.environ.get("COMPASS_LOAD_GENERATED")

#: Distinct KV-cache regions a captured batch rotates over. One per captured
#: call by default, so a region is revisited only after every other has been
#: walked -- which is what evicts it. Set to 1 to price against a single region,
#: the warm case, which understated all three of attention's kernels by 15-30%.
KV_VARIANTS = int(os.environ.get("COMPASS_KV_VARIANTS", str(GRAPH_BATCH)))

#: Whether to record which kernels each operator launches. Off by default under
#: parallelism, because the breakdown costs two extra calls per signature and for
#: a *collective* those are two extra collectives. Every rank must make the same
#: calls in the same order or the group deadlocks, and a breakdown that fails on
#: one rank -- the profiler is not guaranteed to succeed -- diverges it silently.
#: Pricing at TP=4 died in a distributed recv for exactly this reason.
PRICE_KERNELS = os.environ.get("COMPASS_PRICE_KERNELS", "1") != "0"

#: Least a signature must contribute to a step, in seconds, to be worth a
#: breakdown. Each one is a profiler session and the sessions are what break
#: under parallelism, so under it the cheap majority are skipped: 313
#: signatures become a few dozen, and the ones dropped are the ones no
#: comparison would have read.
BREAKDOWN_OVER = os.environ.get("COMPASS_BREAKDOWN_OVER")


def under_parallelism() -> bool:
    """Whether this process has peers it must stay in step with.

    Not ``WORLD_SIZE``. Three gates in this file were written to read it and
    none of them ever fired: the engine spawns its ranks itself and never sets
    it, so every "off under parallelism" default was on at TP=2 and had been
    since it was written. That is how a defect known to fault the device came
    back -- the explicit override was dropped, the default was trusted, and the
    default was dead.

    The process group is the thing that actually knows, and it is only up after
    this module is imported, so this cannot be a constant.
    """
    try:
        import torch.distributed as dist

        return (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1)
    except Exception:  # noqa: BLE001 - no distributed is one rank
        return False


def _load_generated() -> bool:
    if LOAD_GENERATED is not None:
        return LOAD_GENERATED != "0"
    return not under_parallelism()


def _breakdown_over() -> float:
    if BREAKDOWN_OVER is not None:
        return float(BREAKDOWN_OVER)
    return 1e-3 if under_parallelism() else 0.0


#: Signatures whose key contains this substring get taken apart rather than just
#: priced. A price is a duration and says nothing about where the duration went;
#: this says whether it went into kernels or into the gaps between them, which is
#: the difference between "this kernel is slow" and "this measurement is wrong".
PROFILE_MATCH = os.environ.get("COMPASS_BENCH_PROFILE", "")

#: A file to name each signature in before it is priced, so a fault that kills
#: the process leaves behind which operator it died on. Empty means no trace.
#: Each rank writes its own -- `{rank}` in the path is substituted.
BENCH_TRACE = os.environ.get("COMPASS_BENCH_TRACE", "")
if BENCH_TRACE and "{rank}" in BENCH_TRACE:
    BENCH_TRACE = BENCH_TRACE.replace(
        "{rank}", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _time_in_graph(fn, sets: list, iters: int, warmup: int, before=None,
                   breakdown: bool = False, occurrences: int = 1,
                   family: str | None = None,
                   covered: set | None = None) -> tuple[float, float, dict]:
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
            if before is not None:
                before(i)
            a, k = sets[i % n]
            fn(*a, **k)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(GRAPH_BATCH):
            # Host-only: it swaps which pre-built metadata the next call reads,
            # so each captured call bakes in its own KV region. Nothing is
            # allocated here, which capture would otherwise charge to the pool.
            if before is not None:
                before(i)
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
    seconds = total / (replays * GRAPH_BATCH)
    # Only for signatures that carry real time. Each breakdown is a profiler
    # session, and it is *sessions* that are the problem: profiling the real run
    # once at TP=2 is fine, while several hundred start/stop cycles in a loop
    # faults the device partway through, with ROCTracer complaining about
    # duplicate flow starts on the way. Taking them only where the answer
    # matters cuts the cycles by an order of magnitude and loses nothing the
    # comparison uses -- on the 0.6B, 19 kernels covered 98.7% of a step.
    take = breakdown and _wants_breakdown(seconds, occurrences, family, covered)
    kernels = _kernels_of_replay(graph) if take else {}
    if kernels and covered is not None and family is not None:
        # Only on success. A family whose first signature yields nothing stays
        # uncovered, so the next one is tried rather than the whole family being
        # written off on one failure.
        covered.add(family)
    return seconds, 0.0, kernels


def _wants_breakdown(seconds: float, occurrences: int,
                     family: str | None = None,
                     covered: set | None = None) -> bool:
    """Whether this signature is worth a profiler session.

    Two ways to earn one. Carrying a millisecond of the step is the first, and
    was the only one; each breakdown is a profiler session and it is *sessions*
    that fault the device under parallelism, so they go where the answer
    matters.

    Being the first signature of an operator nobody has taken apart yet is the
    second, and it exists because a threshold on the *signature* is blind to an
    operator whose signature fragments. `linear_attention_with_output_base`
    arrives as 48 signatures of 0.093 ms -- one per layer, since its signature
    carries per-layer state -- so every one falls under a 1 ms bar while the
    family is the third largest thing in the step at 4.47 ms. It got no
    breakdown at all, and most of the quarter of a step's kernel time that
    could not be audited against a price was it.

    One per family is enough to *name* an operator's kernels, which is what an
    audit needs, and the fragments launch the same kernels as one another. It
    also bounds the cost: a 27B decode graph has 30 distinct operator names,
    against the several hundred sessions that fault.
    """
    if family is not None and covered is not None and family not in covered:
        return True
    return seconds * max(1, occurrences) >= _breakdown_over()


def _kernels_of_replay(graph) -> dict:
    """Which kernels the priced graph runs, from replaying the graph itself.

    The breakdown used to come from two *extra* eager calls of the operator,
    which is where both of its failures came from. For a collective those are
    two calls the peers do not make, so ranks that price different signatures
    wait for each other forever. And an operator that only works the way the
    price ran it may not survive being called any other way -- at TP=2
    `aiter::masked_embedding` faults the device outright, killing the run
    rather than losing one breakdown.

    Replaying the graph that was just timed has neither problem: no extra
    launches beyond the one replay, nothing run that was not already run, and
    the kernels are the ones the price is a price *of* rather than the ones a
    differently-shaped call would have launched.

    So a breakdown now exists only where there is a graph to replay. An
    operator that could not be captured -- the chunked-prefill attention, an
    `aten::item` that has to synchronise -- is priced without one, which is the
    price of never calling an operator any way but the way it was priced. The
    attribution that cost bought is worth more than the fourteen breakdowns it
    gives up: chasing the fault by hand was unreliable anyway, because an HSA
    fault surfaces at the next synchronise rather than at the kernel that
    caused it, so the operator it appeared to blame moved between runs.

    Durations are per replay, which is `GRAPH_BATCH` calls, so they are divided
    back down to one.
    """
    import json as _json
    import os as _os
    import tempfile

    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()

    handle, path = tempfile.mkstemp(suffix=".json")
    _os.close(handle)
    try:
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8") as fh:
            events = _json.load(fh).get("traceEvents", [])
    finally:
        try:
            _os.unlink(path)
        except OSError:
            pass

    kernels: dict[str, float] = {}
    for event in events:
        if event.get("cat") in ("kernel", "Kernel"):
            name = event.get("name", "")
            kernels[name] = kernels.get(name, 0.0) + float(
                event.get("dur", 0.0)) / 1e6 / GRAPH_BATCH
    return kernels


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


def _uncapturable(exc: Exception) -> bool:
    """Whether a failure means "this cannot be graph-captured" rather than
    "this operator is broken"."""
    text = str(exc)
    return ("stream is capturing" in text
            or "StreamCaptureUnsupported" in text
            or "capture_begin" in text)


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


def load_ops(graph_path: str) -> tuple[list, list, dict | None]:
    """Read the graph(s) at ``graph_path``: their operators, paths, topology.

    Shared so that whatever stands the operators up sees exactly the graphs that
    will be priced. A stand-up sized from one glob and a pricing run reading
    another is a cache too small by a factor nobody would notice in the numbers.
    """
    # A glob or a comma-separated list, because a deployment has more than one
    # kind of step and each is its own graph -- a decode one and a prefill one.
    # Their operators are pooled before pricing: signatures common to both are
    # priced once, and the ones that differ each get their own price.
    import glob as _glob

    paths = []
    for part in str(graph_path).split(","):
        part = part.strip()
        found = sorted(_glob.glob(part))
        if not found and any(c in part for c in "*?["):
            # A pattern that matches nothing is a mistake worth naming. Falling
            # through to open the pattern as a filename reports it as a missing
            # file, which reads like the graph was not written rather than like
            # the pattern was wrong -- `graph.tp*.json` matches nothing at TP=1,
            # where no rank suffix is applied.
            logger.warning("ATOMCompass WARNING: --compass-bench-graph pattern "
                           "%r matched no files; nothing from it is priced",
                           part)
            continue
        paths.extend(found or [part])
    if not paths:
        raise OSError(f"no graphs matched {graph_path!r}")
    ops = []
    # A price list that does not say which parallel width it was measured at
    # cannot be stopped from being read at another one. `aiter::all_reduce_`
    # signs identically over 2 ranks and over 4 -- same message, same dtype,
    # and `unique_name` is `tp:0` however wide the group is -- so a 4-way price
    # matches a 2-way call exactly, at full coverage, with no warning. The width
    # is not in the operator; it is in the graph the operator came from.
    topologies = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        ops.extend(blob["ops"])
        topology = ((blob.get("key") or {}).get("topology")) or []
        topologies.add(tuple(sorted(tuple(x) for x in topology)))
    # Ranks of one deployment agree; a caller who pooled graphs from two
    # different widths gets None, which the consumer treats as uncertifiable.
    measured_topology = (dict(next(iter(topologies)))
                         if len(topologies) == 1 else None)
    return ops, paths, measured_topology


def price_graph(graph_path: str, iters: int = 2000, warmup: int = 20,
                cache: str = "hot", only: str | None = None) -> dict[str, Any]:
    """Price every distinct operator signature in a captured graph.

    Returns the price list and what it could not reach. Coverage is reported by
    operator count *and* by how many of the graph's operators a priced signature
    accounts for, because the two differ enormously: a handful of signatures
    cover most of a step.

    `only` narrows the run to operator names containing that substring. A whole
    graph takes minutes and stands up a context per signature; reproducing one
    family's failure does not need the other two hundred. The coverage counts
    then describe the narrowed set, so the result carries `only` -- a partial
    run must not read back as a graph's coverage.
    """
    ops, paths, measured_topology = load_ops(graph_path)
    if only:
        ops = [op for op in ops if only in op["name"]]

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

    from atom.compass.runtime import forward_ctx

    priced: dict[str, dict] = {}
    unpriced: dict[str, str] = {}
    # Operator names that already have a breakdown somewhere. See the family
    # rule in `_time_in_graph`: without it, an operator whose signature
    # fragments per layer is never taken apart, however much of the step it is.
    covered: set = set()
    for sig, op in example.items():
        # A GPU memory fault is not an exception. It kills the process, so the
        # `unpriced` bookkeeping below never runs and the log says only that
        # the run died -- with nothing to say which of two hundred signatures
        # was in hand. Written and flushed before the attempt, this file names
        # it. Off unless asked for, and appended to rather than held, because
        # anything buffered dies with the process.
        if BENCH_TRACE:
            with open(BENCH_TRACE, "a", encoding="utf-8") as fh:
                fh.write(sig + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        # Per signature, not once: an operator that rebuilds the context from
        # its arguments leaves that context behind for whatever is priced next.
        reset_forward_context()
        triton_kernel = op["name"].partition("::")[0] in ("triton", "inductor")
        fn = _resolve_triton(op) if triton_kernel else _resolve(op["name"])
        if fn is None:
            unpriced[sig] = (
                "triton kernel with no importable origin or resolved grid"
                if triton_kernel else "operator not registered in this process")
            continue
        # Only for Triton: a registered aten/aiter operator is handed real
        # tensors and reads their strides off them, so a dense rebuild is
        # self-consistent however the original was laid out.
        stride = _stride_past_its_tensors(op) if triton_kernel else None
        if stride is not None:
            unpriced[sig] = (
                f"{stride[0]}={stride[1]} exceeds every argument's row extent "
                f"({stride[2]}), so it can only be a stride into an allocation "
                "the graph does not record")
            continue
        # An operator that reads a forward context gets the one it was recorded
        # with, or is not priced. Calling it without would price it against
        # whatever capture left installed, which is how attention came to be
        # measured at 7x its cost.
        rotate, variants = None, []
        if forward_ctx.is_context_dependent(op["name"]):
            # One context per captured call, each against its own slice of the
            # KV cache. The cache is not an argument -- it is reached through
            # the context -- so no amount of rotating arguments cools it, and
            # `hot` and `cold` alike price it warm.
            try:
                variants = forward_ctx.install(
                    op["name"], op.get("context"),
                    variants=KV_VARIANTS if cache == "graph" else 1)
            except Exception as exc:  # noqa: BLE001 - an installer can fail
                # Standing up a context is as fallible as calling the operator,
                # and this ran outside the guard below: one installer raising
                # took the whole pricing run and every signature with it.
                unpriced[sig] = (
                    f"could not install its context: {type(exc).__name__}: "
                    f"{str(exc)[:80]}")
                continue
            if not variants:
                unpriced[sig] = ("reads a forward context and the graph "
                                 "recorded none")
                continue
            variants[0]()
            if len(variants) > 1:
                def rotate(i, _variants=variants):
                    _variants[i % len(_variants)]()
        try:
            sets = _build_arg_sets(op, cache, fn)
        except Exception as exc:  # noqa: BLE001 - allocation can fail many ways
            unpriced[sig] = f"could not build inputs: {type(exc).__name__}"
            continue
        if sets is None:
            unpriced[sig] = "unknown dtype"
            continue
        used, kernels = cache, {}
        try:
            if cache == "graph":
                try:
                    seconds, host_seconds, kernels = _time_in_graph(
                        fn, sets, iters, warmup, before=rotate,
                        breakdown=PRICE_KERNELS, occurrences=counts[sig],
                        family=op["name"], covered=covered)
                except Exception as exc:  # noqa: BLE001
                    if not _uncapturable(exc):
                        raise
                    # Some operators cannot be captured at all. Chunked-prefill
                    # attention gathers cached and new KV with a
                    # `repeat_interleave` whose output size is only known on the
                    # device, so it synchronises, and a synchronise inside a
                    # capture is an error. Refusing to price those leaves the
                    # whole of chunked prefill's attention unpriced; timing them
                    # back-to-back instead prices them, at the cost of carrying
                    # per-launch overhead the graph would have amortised.
                    #
                    # `used` records which, so a reader can tell one price from
                    # the other rather than finding them silently mixed.
                    used = "over"
                    import torch
                    torch.cuda.synchronize()
                    if variants:
                        variants[0]()
                    seconds, host_seconds = _time_over(fn, sets, iters, warmup)
            else:
                timer = _time_isolated if cache == "isolated" else _time_over
                seconds, host_seconds = timer(fn, sets, iters, warmup)
        except Exception as exc:  # noqa: BLE001 - a call can fail many ways
            # Where it failed, not just what it said. An operator rebuilt from
            # a graph fails inside the engine's own code, and the message alone
            # ("'NoneType' object has no attribute 'device'") names neither the
            # field that was None nor the branch that wanted it.
            unpriced[sig] = f"{type(exc).__name__}: {_message(exc)}{_where(exc)}"
            continue
        priced[sig] = {
            "name": op["name"],
            "seconds": seconds,
            "occurrences": counts[sig],
            "cache": used,
            "arg_sets": len(sets),
            "kv_regions": len(variants) or 1,
            # Host enqueue cost per call. Where this matches `seconds`, the
            # device was idle waiting and the price is the host's, not the
            # kernel's.
            "host_seconds": host_seconds,
            # Which kernels this operator launches, so its price can be compared
            # against the same work in a step -- where a graph replay leaves
            # kernel names as the only thing the two sides share. See
            # scripts/compass/price_check.py and open problem 21.
            "kernels": kernels,
        }
        if PROFILE_MATCH and PROFILE_MATCH in sig:
            try:
                _profile_signature(fn, sets, sig, iters, warmup,
                                   before=rotate)
            except Exception as exc:  # noqa: BLE001 - a probe must not stop a run
                print(f"### PROBE FAILED {type(exc).__name__}: {exc}", flush=True)

    ops_priced = sum(counts[s] for s in priced)
    return {
        "version": 1,
        "provenance": {
            "graph": graph_path,
            "graphs": paths,
            # Which width these prices are of. A consumer at another width
            # must not spend a collective price from this list; see
            # `PricedOracle._cost`.
            "topology": measured_topology,
            "iters": iters,
            "cache": cache,
            # Set when the run was narrowed to one family. The coverage below
            # then counts only what was asked for, and reads as complete
            # unless this says otherwise.
            "only": only,
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



def _profile_signature(fn, sets, sig, iters: int, warmup: int,
                       before=None) -> None:
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
        batches = [int(b) for b in os.environ.get(
            "COMPASS_PROBE_BATCHES", "1,4,16,64").split(",")]
        for batch in batches:
            GRAPH_BATCH = batch
            secs, _host, _kernels = _time_in_graph(
                fn, sets, max(iters, batch), warmup, before=before)
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
            if before is not None:
                before(i)
            a, k = sets[i % n]
            fn(*a, **k)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for i in range(GRAPH_BATCH):
            if before is not None:
                before(i)
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
