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
  needs to know *which tensor is which*. The traced op graph now records which
  operator produced each input, so it can be walked.
"""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

__all__ = ["peak_activation_bytes", "weight_bytes", "graph_pool_bytes",
           "ELEMENT_BYTES"]

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
    ops = graph["ops"]
    canonical = _canonical(ops)

    last_read = {}
    for index, op in enumerate(ops):
        for producer in op.get("inputs_from") or ():
            if 0 <= producer < len(canonical) and canonical[producer] >= 0:
                last_read[canonical[producer]] = index

    live, peak, held = 0, 0, {}
    for index, op in enumerate(ops):
        if canonical[index] == index:
            dtypes = op.get("dtypes") or ()
            dtype = dtypes[0] if dtypes else "bfloat16"
            aliases = op.get("output_aliases") or ()
            size = 0
            for position, shape in enumerate(op.get("output_shapes") or ()):
                if position < len(aliases) and aliases[position] is not None:
                    continue  # written into, not allocated
                size += _bytes_of(shape, dtype)
            if size:
                held[index] = size
                live += size
                peak = max(peak, live)
        # Everything whose last reader was this operator dies here, including
        # this operator's own outputs when nothing downstream reads them.
        for produced, reader in list(last_read.items()):
            if reader == index and produced in held:
                live -= held.pop(produced)
        if index in held and index not in last_read:
            live -= held.pop(index)
    return peak


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
