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

__all__ = ["peak_activation_bytes", "weight_bytes", "ELEMENT_BYTES"]

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


def peak_activation_bytes(graph: Mapping) -> int:
    """The most activation memory live at once, by walking the graph.

    A tensor is live from the operator that produced it until its last
    consumer, so this is a def-use walk: add an operator's outputs as it runs,
    drop every tensor whose last reader has just run, and keep the high-water
    mark. Inputs with no producer are not counted -- a weight is not an
    activation, and counting it here would double it against the weight term.

    Two approximations, both stated rather than hidden. Output dtype is not
    recorded, so an operator's outputs are counted at its *first input's* dtype,
    which is right for the elementwise and matmul operators that hold the memory
    and wrong for a cast. And a tensor with no reader in the graph is freed
    immediately, where the engine frees it whenever the last Python reference
    goes -- so this is a lower bound on the high-water mark, not a bound on what
    the allocator reserves.
    """
    ops = graph["ops"]
    last_read = {}
    for index, op in enumerate(ops):
        for producer in op.get("inputs_from") or ():
            if producer >= 0:
                last_read[producer] = index

    live, peak, held = 0, 0, {}
    for index, op in enumerate(ops):
        dtypes = op.get("dtypes") or ()
        dtype = dtypes[0] if dtypes else "bfloat16"
        size = sum(_bytes_of(s, dtype) for s in op.get("output_shapes") or ())
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


def weight_bytes(checkpoint: str, tensor_parallel: int = 1) -> Optional[int]:
    """Parameter bytes from the checkpoint, without loading it.

    A sharded checkpoint indexes its shards and records the total; a single-file
    one is its own size. Divided by the tensor-parallel size, which is right for
    the projections that hold almost all of it and wrong for the norms and
    embeddings that some builds replicate -- an overestimate of the shard by
    whatever is replicated, and stated because the term it feeds is a budget.
    """
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            total = (json.load(fh).get("metadata") or {}).get("total_size")
        if total:
            return int(total) // max(1, tensor_parallel)
    single = os.path.join(checkpoint, "model.safetensors")
    if os.path.exists(single):
        return os.path.getsize(single) // max(1, tensor_parallel)
    return None
