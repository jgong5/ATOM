"""**analytical** -- a configuration's memory derived, not measured.

`RecordedMemory` reads back five readings some run took, so a configuration can
be sized on a box that could not hold it. It still needs the configuration to
have run *somewhere*. Deriving the terms removes that, which is what makes
"which configuration should I deploy" answerable over configurations that do not
yet exist.

Four terms, and only one of them is interesting:

* **weights** -- the checkpoint says how many bytes of parameters there are.
* **non-torch** -- collective buffers and the CUDA context, a per-rank constant.
* **CUDA-graph pool** -- geometry ATOM already computes.
* **activations** -- everyone else guesses this. It is a liveness question:
  not how much memory the operators touch, but how much is live at once, which
  needs to know *which tensor is which*. The traced op graph records which
  operator produced each input, so it can be walked.

**Liveness is recorded, not inferred.** The first walk guessed at deaths from
the last read of a tensor, which is a different event: a tensor lives until its
last Python reference goes, so a local held across a block outlives every read
of it. The trace now watches each output die -- a finalizer fires when the
allocator takes the memory back -- and records the death per *output*, since an
operator's outputs need not share a life. A fused add-and-norm returns the
normed activation and the new residual, and giving both the later of the two
deaths held one extra tensor per layer.

What stays analytical is the part that has to generalise. Which outputs are
fresh and how long each lives are structural facts about an operator, recorded
once at one shape; the bytes come from the shapes, so the term still answers
for a shape nobody traced.
"""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

__all__ = ["peak_activation_bytes", "activation_curve", "weight_bytes",
           "resident_bytes", "non_torch_bytes", "load_residue_bytes",
           "modelled_readings", "activation_bytes_at",
           "graph_pool_bytes", "measured_graph_pool_bytes", "ELEMENT_BYTES"]

ELEMENT_BYTES = {
    "float64": 8, "int64": 8, "double": 8,
    "float32": 4, "int32": 4, "float": 4,
    "bfloat16": 2, "float16": 2, "int16": 2, "half": 2,
    "float8_e4m3fnuz": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "uint8": 1, "bool": 1,
}


def _bytes_of(shape, dtype: str) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * ELEMENT_BYTES.get(dtype, 2)


def _canonical(ops) -> list:
    """The operator each output is really attributed to.

    An in-place operator writes into a tensor someone else allocated, so the
    tensor belongs to its allocator for the whole of its life. Following the
    alias back means a chain like `gemm -> all_reduce_ -> read` keeps one
    tensor live across all three, where treating the all-reduce as an allocator
    would count two, and treating it as a consumer would free the tensor while
    it is still being read.

    Returns an index per operator: itself when it allocated its output, the
    operator whose tensor it wrote into otherwise, and -1 when that tensor came
    from before the step and is nobody's activation.
    """
    canonical = list(range(len(ops)))
    for index, op in enumerate(ops):
        aliases = op.get("output_aliases") or ()
        # Only an operator whose outputs *all* alias is redirected. One that
        # both allocates and writes in place -- attention, which returns a
        # tensor and fills the KV cache -- still owns what it allocated, and
        # redirecting it would drop that tensor from the live set entirely.
        if not aliases or any(alias is None for alias in aliases):
            continue
        first = aliases[0]
        canonical[index] = canonical[first] if 0 <= first < index else -1
    return canonical


def _deaths(ops, canonical) -> dict:
    """After which operator each *output* was released.

    A recorded death is the allocator taking the memory back: a finalizer on
    the tensor fires the moment its last reference goes, and the operator in
    progress then is where it died. Last-read is a guess at the same event and
    is wrong in both directions -- a local held across a block outlives every
    read of it, and a producer map keyed on storage address credits a reused
    address to the tensor that used to live there, which resurrects the dead.

    Keyed per output, because an operator's outputs need not share a life. A
    fused add-and-norm returns the normed activation and the new residual: the
    first dies into the next gemm, the second carries to the end of the block.
    Giving both the later death holds one extra tensor per layer, which at TP=4
    was 36% of the term.

    Index ``d`` means the tensor was still live while operator ``d`` ran. A
    position absent from the result was never seen to die and is held to the
    end of the step -- the opposite default to the one last-read needed, where
    an output nobody read had to be dropped at once.
    """
    dies, observed = {}, False
    for index, op in enumerate(ops):
        owner = canonical[index] if canonical[index] >= 0 else index
        for position, death in enumerate(op.get("dies_at") or ()):
            if death is None or death < 0:
                continue
            observed = True
            key = (owner, position)
            dies[key] = max(dies.get(key, -1), int(death))
    if observed:
        return dies

    # No deaths recorded. Fall back to the rule this replaced, which knows
    # nothing about positions and so gives every output of an operator the
    # same life.
    per_operator = {}
    for index, op in enumerate(ops):
        for producer in op.get("inputs_from") or ():
            if 0 <= producer < len(canonical) and canonical[producer] >= 0:
                per_operator[canonical[producer]] = index
    for index, op in enumerate(ops):
        death = per_operator.get(index, index)  # unread, so it went at once
        for position in range(max(1, len(op.get("output_shapes") or ()))):
            dies[(index, position)] = death
    return dies


def activation_curve(graph) -> list:
    """How much activation memory is live at each operator.

    The activation term is a curve and its peak is one point on it. Comparing
    peaks says a model is wrong; comparing curves says *where* -- the first
    operator at which the walk and the allocator disagree is the operator whose
    liveness is modelled wrongly, and everything after it is downstream of that
    one mistake.

    Each entry is what is live *while* that operator runs: its outputs already
    exist, and anything released only after it finished is still there. That is
    the same instant at which the trace reads the allocator, so the two curves
    are comparable point for point.
    """
    ops = graph["ops"]
    canonical = _canonical(ops)
    dies = _deaths(ops, canonical)

    released = {}
    for key, death in dies.items():
        released.setdefault(death, []).append(key)

    live, held, curve = 0, {}, []
    for index, op in enumerate(ops):
        for key in released.get(index - 1, ()):
            if key in held:
                live -= held.pop(key)
        if canonical[index] == index:
            dtypes = op.get("dtypes") or ()
            dtype = dtypes[0] if dtypes else "bfloat16"
            aliases = op.get("output_aliases") or ()
            for position, shape in enumerate(op.get("output_shapes") or ()):
                if position < len(aliases) and aliases[position] is not None:
                    continue  # written into, not allocated
                size = _bytes_of(shape, dtype)
                if size:
                    held[(index, position)] = size
                    live += size
        curve.append(live)
    return curve


def peak_activation_bytes(graph) -> int:
    """The most activation memory live at once, by walking the graph.

    A tensor is live from the operator that produced it until its last
    consumer, so this is a def-use walk: add an operator's outputs as it runs,
    drop every tensor whose last reader has just run, and keep the high-water
    mark. Inputs with no producer are not counted -- a weight is not an
    activation, and counting it here would double it against the weight term.
    Nor is an in-place operator's output, which is not a new tensor.

    Two approximations, both stated rather than hidden. Output dtype is not
    recorded, so an operator's outputs are counted at its *first input's* dtype,
    which is right for the elementwise and matmul operators that hold the memory
    and wrong for a cast. And a tensor with no reader in the graph is freed
    immediately, where the engine frees it whenever the last Python reference
    goes -- so this is a lower bound on the high-water mark, not a bound on what
    the allocator reserves.
    """
    return max(activation_curve(graph) or [0])


def _safetensors_header(path: str) -> Optional[dict]:
    """The JSON header of a safetensors file, without reading its tensors.

    The format is an 8-byte little-endian header length, then that many bytes
    of JSON. So the per-tensor sizes cost two reads of a few kilobytes each,
    whatever the checkpoint weighs.
    """
    try:
        with open(path, "rb") as fh:
            length = int.from_bytes(fh.read(8), "little")
            if not 0 < length < 100 << 20:
                return None
            return json.loads(fh.read(length).decode("utf-8"))
    except (OSError, ValueError):
        return None


def _tensor_sizes(checkpoint: str) -> Optional[dict]:
    """Every parameter in the checkpoint and its size in bytes."""
    shards = []
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        try:
            with open(index, encoding="utf-8") as fh:
                weight_map = json.load(fh).get("weight_map") or {}
        except (OSError, ValueError):
            return None
        shards = sorted({os.path.join(checkpoint, f) for f in weight_map.values()})
    elif os.path.exists(os.path.join(checkpoint, "model.safetensors")):
        shards = [os.path.join(checkpoint, "model.safetensors")]
    if not shards:
        return None

    sizes = {}
    for shard in shards:
        header = _safetensors_header(shard)
        if header is None:
            return None
        for name, entry in header.items():
            if name == "__metadata__" or not isinstance(entry, dict):
                continue
            offsets = entry.get("data_offsets")
            if not offsets or len(offsets) != 2:
                continue
            sizes[name] = (int(offsets[1]) - int(offsets[0]), entry.get("shape") or [])
    return sizes or None


def _is_tied(checkpoint: str) -> bool:
    """Whether the model shares one tensor between the embedding and the head."""
    try:
        with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return False
    if config.get("tie_word_embeddings"):
        return True
    text = config.get("text_config")
    return bool(isinstance(text, dict) and text.get("tie_word_embeddings"))


def weight_bytes(checkpoint: str, tensor_parallel: int = 1) -> Optional[int]:
    """Parameter bytes resident on one rank, from the checkpoint's headers.

    Not the checkpoint's size on disk, which was the first cut and was wrong in
    two ways this fixes.

    A checkpoint that ties its embedding to its output head still stores both
    tensors; the model loads one and points the other at it, so the file is
    larger than what is resident. `config.json` says whether they are tied, and
    the head is dropped when it does.

    Tensor parallelism does not divide everything. The 2-D projections shard,
    the 1-D norms and biases are replicated on every rank -- so dividing the
    whole checkpoint by the world size under-counts by whatever is replicated,
    which is the unsafe direction for a budget. Sharding by rank on the 2-D
    tensors alone is still an approximation (a build may replicate the
    embedding, and an uneven head count does not divide exactly) but it errs
    the other way.

    Falls back to the size on disk when the headers cannot be read, which is
    the old behaviour and is flagged by returning a number rather than None.
    """
    parallel = max(1, tensor_parallel)
    sizes = _tensor_sizes(checkpoint)
    if sizes is None:
        single = os.path.join(checkpoint, "model.safetensors")
        index = os.path.join(checkpoint, "model.safetensors.index.json")
        if os.path.exists(index):
            try:
                with open(index, encoding="utf-8") as fh:
                    total = (json.load(fh).get("metadata") or {}).get("total_size")
            except (OSError, ValueError):
                total = None
            if total:
                return int(total) // parallel
        if os.path.exists(single):
            return os.path.getsize(single) // parallel
        return None

    tied = _is_tied(checkpoint)
    total = 0
    for name, (size, shape) in sizes.items():
        if tied and name.endswith("lm_head.weight"):
            continue
        total += size // parallel if len(shape) >= 2 else size
    return total


#: The engine's manual-capture pool estimate, as a fraction of peak activations
#: (`ModelRunner._estimate_cudagraph_overhead`). Mirrored rather than imported
#: because the point of a derived term is that it runs without a runner.
MANUAL_POOL_FRACTION = 0.2

#: Live tensors a captured PIECEWISE graph retains per layer per token, the
#: engine's `_LIVE_TENSORS_PER_LAYER`.
PIECEWISE_LIVE_TENSORS_PER_LAYER = 2.8


def resident_bytes(model, tied_head: bool = False) -> tuple:
    """What a built model's parameters and buffers weigh, per rank.

    Returns ``(parameters, buffers)``. Counted once per storage, so a tied head
    and its embedding are one allocation, and a meta-built model answers as
    well as a loaded one -- meta tensors carry shape and dtype and allocate
    nothing, which is what lets a configuration nobody has run be sized. Pass
    ``tied_head`` for a meta build, which has not been through the loader and
    so has not had the tie applied.

    This is what the checkpoint-derived `weight_bytes` approximates. Prefer
    this where the model can be built: it is ATOM's own sharding rather than a
    rule about it, and the rule is where the approximation lives -- exact on
    the dense 0.6B at TP=1, 2 and 4 and on the hybrid 27B at TP=2, and 3.3%
    low on the same 27B at TP=4, where something does not divide the way
    "2-D tensors shard, 1-D tensors do not" assumes.
    """
    import torch

    seen, parameters, buffers = set(), 0, 0
    for tensors, into in ((model.named_parameters(), "p"),
                          (model.named_buffers(), "b")):
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            # A model built on meta has not been through the loader, which is
            # what ties the head to the embedding, so the two are separate
            # tensors with separate storages and identity cannot see the tie.
            # It is worth exactly one embedding: 0.290 GiB on the 0.6B, which
            # is the whole of the gap between the meta build and the loaded
            # model.
            if tied_head and name.endswith("lm_head.weight"):
                continue
            # A meta tensor has no storage to ask, so its own extent is the
            # only answer -- and on meta there is no aliasing to deduplicate.
            if tensor.is_meta:
                key = id(tensor)
                size = tensor.numel() * tensor.element_size()
            else:
                storage = tensor.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                size = storage.nbytes()
            if key in seen:
                continue
            seen.add(key)
            if into == "p":
                parameters += size
            else:
                buffers += size
    return parameters, buffers


#: What a rank holds outside the torch allocator, and what the collective
#: libraries take through it, measured on this box (MiB, per rank).
#:
#: `non_torch` is `(total - free) - reserved`, and `total - free` is
#: *device-wide* -- so a neighbour's allocation is charged to this
#: configuration. That is the same defect the recorded-`free` guard already
#: refuses a record for, and nothing guarded this one. It shows: at TP=1 and
#: TP=2 every rank agreed to the byte, at TP=4 they spread 192 MiB and at TP=8
#: 640 MiB. These are the *minimum* across ranks, which is the least
#: contaminated estimate of the configuration's own share.
#:
#: A table rather than a law, because the evidence does not support a law. Over
#: the TP=1 baseline the collective term is 5980, 6196 and 9138 MiB at widths
#: 2, 4 and 8: no fixed-plus-per-peer form fits all three, and the jump at 8 is
#: most likely RCCL opening more channels. Calibrate per deployment rather than
#: trusting these -- `validate_memory.py --calibrate` writes a replacement from
#: records, the way `step_accounting.py --calibrate` does for the overhead
#: constant, and for the same reason: the constant does not transfer.
MIB = 1 << 20
DEFAULT_NON_TORCH = {
    # world size -> bytes held outside the torch allocator
    1: 926 * MIB,
    2: 6906 * MIB,
    4: 7266 * MIB,
    8: 10704 * MIB,
}
#: What the collective libraries take *through* the torch allocator, beyond the
#: model's own parameters -- AITER's CustomAllreduce registers a 1 GiB pool and
#: the two-stage kernel a second, so this is ~2 GiB from world size 2 upwards
#: and 1 MiB at world size 1. Flat in width, which the 1.1 / 2069 / 2069 /
#: 2068 MiB measured at widths 1, 2, 4 and 8 says plainly.
DEFAULT_LOAD_RESIDUE = {1: 1 * MIB, 2: 2069 * MIB}

#: How much of `non_torch` varied with the *model* rather than the topology:
#: the 27B sat exactly 266 MiB above the 0.6B at TP=2 and TP=4 alike. 3.8% of
#: the term, and two models cannot say what it is a function of, so it is
#: carried as headroom on the larger side.
MODEL_HEADROOM = 266 * MIB


def _at_width(table: Mapping, world_size: int) -> int:
    """The table's entry for this width, or the widest one at or below it."""
    if world_size in table:
        return int(table[world_size])
    below = [w for w in table if w <= world_size]
    return int(table[max(below)]) if below else int(table[min(table)])


def non_torch_bytes(world_size: int, calibration: Optional[Mapping] = None) -> int:
    """What this rank holds outside the torch allocator.

    Includes the model headroom, because under-reserving here over-allocates
    KV and the run then dies at steady state rather than at start-up.
    """
    table = (calibration or {}).get("non_torch") or DEFAULT_NON_TORCH
    return _at_width({int(k): v for k, v in table.items()}, world_size) + (
        MODEL_HEADROOM if not calibration else 0)


def load_residue_bytes(world_size: int,
                       calibration: Optional[Mapping] = None) -> int:
    """What the collectives take through the torch allocator, beyond the model."""
    table = (calibration or {}).get("load_residue") or DEFAULT_LOAD_RESIDUE
    return _at_width({int(k): v for k, v in table.items()}, world_size)


#: The engine's own per-step buffers -- `allocate_forward_vars` and the
#: attention metadata. Flat in tensor-parallel width (85.2 MiB on the 0.6B at
#: widths 1, 2, 4 and 8; 117.2 MiB on the 27B at 2 and 4) and differing only by
#: model, so it is a calibrated constant rather than a geometry term. 0.05% of
#: a 192 GB card, which is why it is not worth a campaign of its own.
DEFAULT_PERSISTENT = 118 * MIB


def activation_bytes_at(graph, tokens: int) -> int:
    """The activation peak at a token count the graph was not traced at.

    Linear in tokens, which is not an assumption but a measurement: the walk
    scaled from a 3494-token trace lands on the independently measured 4096-token
    warmup peak to +0.0% at TP=1, 2 and 4 alike.
    """
    peak = peak_activation_bytes(graph)
    traced = sum(int(n) for n in
                 ((graph.get("key") or {}).get("batch_signature") or ()))
    return int(peak * tokens / traced) if traced and tokens else peak


def modelled_readings(*, total_bytes: int, world_size: int, parameters: int,
                      buffers: int, activation_bytes: int,
                      calibration: Optional[Mapping] = None,
                      enforce_eager: bool = False) -> dict:
    """The five readings `get_num_blocks` needs, derived rather than measured.

    This is the point of the whole memory model: a configuration nobody has run
    can be sized, because none of these came off a device.

    `total` is the one thing that cannot be derived -- it is the target card's
    capacity and has to be supplied. `free` is modelled as *a clean box*: total
    minus what this process itself holds. That is deliberate. The recorded
    `free` is what the neighbours happened to leave, which is why a record in
    which it was the binding term is refused; deriving it removes the accident
    instead of preserving it, and the answer is what the configuration needs
    rather than what this afternoon allowed.
    """
    persistent = int((calibration or {}).get("persistent") or DEFAULT_PERSISTENT)
    residue = load_residue_bytes(world_size, calibration)
    peak_torch = parameters + buffers + residue + persistent + activation_bytes
    non_torch = non_torch_bytes(world_size, calibration)
    return {
        "total": int(total_bytes),
        "free": max(0, int(total_bytes) - peak_torch - non_torch),
        "peak_torch": int(peak_torch),
        "non_torch": int(non_torch),
        "cudagraph_overhead": graph_pool_bytes(
            activation_bytes, enforce_eager=enforce_eager),
    }


#: What CUDA-graph capture actually reserves, as `floor + slope x sum(captured
#: tokens)`. Fitted on six ladders on the 0.6B at TP=1 whose pools spanned 100
#: to 402 MiB, to within +/-6%; fitted on the five up to 896 tokens it predicts
#: the sixth, at 1071, to **+6.4%**.
#:
#: The floor is the larger surprise. At the shortest ladder it is 87% of the
#: pool, and nothing in the engine's estimate corresponds to it at all.
#:
#: **Only at width one.** Above it the pool does not scale with the ladder at
#: all: across three widths and three ladders -- TP=2 at 31, 512 and 1071
#: tokens, TP=4 at 512 and 1071, TP=8 at 1071 -- the *allocated* delta was
#: 79692800 bytes every single time, to the byte. A 35x change in captured
#: tokens moved it by nothing.
#:
#: Identical allocated bytes cannot come from sharded work, so the graphs are
#: not pinning sharded activations. With tensor parallelism the per-layer
#: intermediates flow through AITER's registered collective buffer, which is
#: outside the torch allocator -- and is already charged to `non_torch` and the
#: load residue. What the graphs pin in torch is a fixed set that neither
#: shards nor grows with the ladder.
#:
#: So: a line in the ladder at width one, and a constant above it. Not "a
#: quarter of width one", which was the earlier reading of two equal numbers
#: and had no mechanism behind it.
DEFAULT_POOL_FLOOR = 91.1 * MIB
DEFAULT_POOL_PER_TOKEN = 0.3033 * MIB
#: The reserved delta above width one: 80, 104, 104, 104 and 88 MiB over the
#: six runs. It is segment bookkeeping around one fixed 76.0 MiB of pinned
#: memory, so the largest is taken rather than the mean -- under-reserving buys
#: dropped capture buckets.
DEFAULT_POOL_SHARDED = 104 * MIB


def measured_graph_pool_bytes(capture_sizes, world_size: int = 1,
                              calibration: Optional[Mapping] = None,
                              enforce_eager: bool = False) -> int:
    """What capture will actually reserve -- not what the engine budgets for it.

    Two different questions, and both are needed. `graph_pool_bytes` mirrors
    the engine's estimator, which is the number that reserves the memory and so
    is what a modelled budget has to reproduce. This one predicts the cost, and
    the two disagree by 4-19x.

    The engine's estimator is `0.2 x` the peak activations, which depends on
    the *warmup* shape and not on the capture ladder at all: across five
    ladders whose pools ran 100 to 370 MiB it returned a flat 20.8 MiB. It is
    not merely low, it is blind to the variable that drives the thing it
    estimates.

    What that costs is not an out-of-memory: the capture loop re-checks free
    memory per bucket and skips what will not fit, so under-reserving buys
    silently dropped buckets and a decode cliff at those batch sizes. On a
    192 GB card nothing has ever been dropped, which is exactly why this went
    unnoticed.

    Above width one the ladder stops mattering -- see `DEFAULT_POOL_SHARDED`.
    """
    if enforce_eager:
        return 0
    sizes = [int(s) for s in (capture_sizes or ()) if int(s) > 0]
    if not sizes:
        return 0
    settings = (calibration or {}).get("graph_pool") or {}
    if world_size > 1:
        return int(settings.get("sharded", DEFAULT_POOL_SHARDED))
    floor = float(settings.get("floor", DEFAULT_POOL_FLOOR))
    per_token = float(settings.get("per_token", DEFAULT_POOL_PER_TOKEN))
    return int(floor + per_token * sum(sizes))


def graph_pool_bytes(activation_bytes: int, *, enforce_eager: bool = False,
                     piecewise: bool = False, hidden_size: int = 0,
                     num_hidden_layers: int = 0, dtype_bytes: int = 2,
                     capture_sizes=(), max_num_batched_tokens: int = 0,
                     total_bytes: int = 0, utilization: float = 0.0,
                     data_parallel: int = 1) -> int:
    """What CUDA-graph capture is expected to hold, without a device.

    This mirrors the engine's own estimator, which is the number that actually
    reserves the memory -- so agreement with it means the derivation reproduces
    the engine's decision, and says nothing yet about whether capture really
    costs that. The second question is answered by the pool the capture loop
    measures and logs, which is what `validate_memory.py` compares against.

    Manual FULL capture takes a fraction of the peak activations, so this term
    composes with the activation term and inherits its error. PIECEWISE is
    geometry: bytes per token per layer, summed over the buckets that fit
    inside a fraction of the utilization budget.
    """
    if enforce_eager:
        return 0
    if not piecewise:
        return int(activation_bytes * MANUAL_POOL_FRACTION)

    per_token = (int(hidden_size) * int(dtype_bytes) * int(num_hidden_layers)
                 * PIECEWISE_LIVE_TENSORS_PER_LAYER)
    if data_parallel > 1:
        per_token *= float(data_parallel) ** 0.6
    shapes = sorted({int(s) for s in capture_sizes})
    if max_num_batched_tokens:
        shapes = [s for s in shapes if s <= max_num_batched_tokens]
    target = 0.15 * utilization * total_bytes
    captured, acc = [], 0
    for num_tokens in shapes:
        if captured and per_token * (acc + num_tokens) > target:
            break
        captured.append(num_tokens)
        acc += num_tokens
    return int(per_token * acc)
