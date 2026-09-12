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

import hashlib
import json
import os
from typing import Mapping, Optional

__all__ = ["peak_activation_bytes", "activation_curve", "weight_bytes",
           "resident_bytes", "non_torch_bytes", "load_residue_bytes",
           "modelled_readings", "activation_bytes_at",
           "scratch_bytes_per_token", "liveness_is_recorded", "traced_shape",
           "liveness_instrumentation", "UNVERSIONED_LIVENESS",
           "UnfoundedActivation", "UnfoundedPrediction",
           "derived_readings", "CALIBRATED_TERMS", "traced_width",
           "graph_pool_bytes", "measured_graph_pool_bytes", "ELEMENT_BYTES",
           "capture_pinned_bytes", "CAPTURE_FIXED_PINNED",
           "AllocatorPool", "allocator_pool_bytes",
           "dtype_ambiguities", "lineage_keys", "width_classes",
           "width_coverage"]

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


_FLOAT_DTYPES = ("float64", "double", "float32", "float", "bfloat16",
                 "float16", "half", "float8_e4m3fnuz", "float8_e4m3fn",
                 "float8_e5m2")


def _promote(dtypes) -> Optional[str]:
    """The dtype PyTorch's promotion rules give these arguments together.

    Not a guess about a particular operator: promotion is defined, and the one
    part of it that matters for sizing is that a floating point argument beats
    every integer one whatever the widths -- `int32` ids and a `bfloat16`
    weight promote to `bfloat16`, never to 4 bytes. Among floats the wider
    wins, except that `float16` with `bfloat16` promotes to `float32`, neither
    being able to hold the other.

    This is still an inference about an operator that did not say, so the
    callers keep it separate from a recorded dtype and `dtype_ambiguities`
    reports every output it was applied to.
    """
    known = [d for d in dtypes if d in ELEMENT_BYTES]
    if not known:
        return None
    floats = [d for d in known if d in _FLOAT_DTYPES]
    if not floats:
        return max(known, key=lambda d: ELEMENT_BYTES[d])
    widest = max(ELEMENT_BYTES[d] for d in floats)
    at_width = {d for d in floats if ELEMENT_BYTES[d] == widest}
    if widest == 2 and {"float16", "half"} & at_width and "bfloat16" in at_width:
        return "float32"
    return sorted(at_width)[0]


def _output_dtype(op, position: int) -> tuple:
    """The dtype of an operator's output, and on what basis.

    `OpSpec.dtypes` is the dtype of each *argument*. Nothing records what an
    operator produced, so a walk that needs an output's size has been taking
    argument 0's and hoping promotion changed nothing. It does change things:
    `aiter::masked_embedding` takes int32 token ids and a bfloat16 weight and
    returns a bfloat16 hidden-width activation, and argument 0's dtype sizes
    one 16 384 x 5 120 buffer at 335 544 320 B instead of 167 772 160.

    Three bases, and the caller is told which it got:

    * `"recorded"` -- the producer wrote `output_dtypes` (O18 asks for it).
    * `"unanimous"` -- every argument has the same dtype, so promotion has
      nothing to choose between and argument 0 is not a guess.
    * `"promoted"` -- the arguments disagree and `_promote` decided. A stated
      rule rather than a per-operator correction, and reported as an ambiguity
      because the rule is not what the operator said.

    `(None, "none")` when the graph gives no argument dtypes at all.
    """
    recorded = op.get("output_dtypes") or ()
    if position < len(recorded) and recorded[position]:
        return str(recorded[position]), "recorded"
    dtypes = tuple(op.get("dtypes") or ())
    if not dtypes:
        return None, "none"
    if len(set(dtypes)) == 1:
        return dtypes[0], "unanimous"
    promoted = _promote(dtypes)
    if promoted is None:
        return None, "none"
    return promoted, "promoted"


def dtype_ambiguities(graph) -> list:
    """Every output whose dtype this graph did not record, and what it costs.

    Reported rather than quietly sized, because the difference between a rule
    and a record is exactly what a memory term is being asked about. Each entry
    carries the bytes the promotion rule gives and the bytes argument 0 would
    have given, so a reader can see whether the ambiguity matters -- most
    operators with mixed argument dtypes produce something small, and the one
    that does not is the embedding.
    """
    out = []
    for index, op in enumerate(graph.get("ops") or ()):
        aliases = op.get("output_aliases") or ()
        for position, shape in enumerate(op.get("output_shapes") or ()):
            if position < len(aliases) and aliases[position] is not None:
                continue  # written into, not allocated: never sized
            dtype, basis = _output_dtype(op, position)
            if basis in ("recorded", "unanimous"):
                continue
            dtypes = tuple(op.get("dtypes") or ())
            out.append({
                "operator": index,
                "name": op.get("name"),
                "position": position,
                "basis": basis,
                "shape": list(shape),
                "argument_dtypes": list(dtypes),
                "chosen_dtype": dtype,
                "bytes_chosen": _bytes_of(shape, dtype) if dtype else 0,
                "bytes_if_argument_0": _bytes_of(shape, dtypes[0])
                if dtypes else 0,
            })
    return out


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


def liveness_is_recorded(graph) -> bool:
    """Whether the graph says when its tensors died, or is being guessed at.

    `_deaths` falls back to last-read when no `dies_at` survives, and that
    fallback is silent -- it returns a number either way. The number it
    returned on the 27B's meta-derived prefill graphs is 570 425 344 B against
    a measured 2 956 984 320 B, 19.3% of the term, and all of it from 64
    `aten::empty.memory_format` allocations the walk had no death for. A caller
    reporting an activation figure has to be able to tell that apart from a
    walk over recorded liveness, so this is the question asked separately.

    This used to say that a graph derived on meta never has it, because
    nothing runs and no finalizer fires. That is wrong, and believing it is
    why the derivation path never stamped what it had already observed. A
    death is a Python reference going, not a device event: a meta tensor is
    refcounted like any other and its finalizer fires at the same moment in
    the same forward. Derived graphs carried no `dies_at` because nothing
    wrote `MetaOpTracer.deaths` onto the operators, not because it was empty.

    What a derivation genuinely cannot observe is the *allocator*: reuse,
    fragmentation, and any buffer a kernel takes internally, none of which
    reach a dispatch tracer. So a recorded liveness from meta is the graph's
    own reference structure, which is what the walk needs, and is still not
    the allocator's high-water mark -- see `activation_curve`, which is the
    comparison that separates the two.
    """
    return any(
        death is not None and death >= 0
        for op in (graph.get("ops") or ())
        for death in (op.get("dies_at") or ())
    )


#: What a graph that does not name its instrumentation revision is. Every
#: artifact written before 2026-09-11 predates the field, and all of them came
#: off the same broken producer, so the absent key is not "unknown" -- it is
#: version 1, and reading it as anything else would let an old template pass a
#: check it never met.
UNVERSIONED_LIVENESS = 1

#: Which revision of the tracer produced a graph's liveness fields --
#: `inputs_from`, `output_aliases` and `dies_at`, the three this walk reads.
#: Stamped into every graph's provenance by
#: `atom.compass.runtime.tracer.ModelTracer.provenance`, so an artifact on disk
#: can be told apart from a re-derived one without re-deriving it. Defined here
#: rather than beside the producer because it is a property of what the walk is
#: entitled to assume, and because this module reaches a reader with no torch.
#:
#: 1. Everything derived before 2026-09-11. `MetaOpTracer._storage_of` keyed on
#:    `data_ptr()`, which is 0 for every meta storage ever made, so on the
#:    device derivation runs on all three fields are wrong in the same
#:    direction: every input reads as produced by the operator before it, every
#:    output as an alias, and no out-variant destination as unseen. Deaths were
#:    observed and never stamped, so the walk fell back to a last-read rule
#:    over that same corrupted `inputs_from`. Captures are not affected: a
#:    device tensor has an address, so the key worked where there was one.
#: 2. Storage identity where there is no address, and the tracer stamps what it
#:    watched die. Prices are unaffected at either version: a signature is
#:    built from input shapes, dtypes, context, integer values and scalars, and
#:    no storage key reaches any of them.
LIVENESS_INSTRUMENTATION = 2


def liveness_instrumentation(graph) -> int:
    """Which revision of the tracer produced this graph's liveness fields.

    The three fields the memory walk reads -- `inputs_from`, `output_aliases`,
    `dies_at` -- were all wrong on anything derived on meta before
    2026-09-11: `_storage_of` keyed on `data_ptr()`, which is 0 for every meta
    storage, so one key stood for every tensor in the trace. The graph that
    came out reads as tidy rather than broken, so a caller cannot tell the two
    apart by looking at the fields; it has to ask which producer wrote them.

    Prices are unaffected at either revision and old artifacts are deliberately
    left as they are, so this is a fact about a graph, not a verdict on it. See
    `atom.compass.runtime.meta.LIVENESS_INSTRUMENTATION` for what each revision
    did, and `liveness_is_recorded` for the separate question of whether the
    deaths are there at all.
    """
    version = (graph.get("provenance") or {}).get("liveness_instrumentation")
    try:
        return int(version)
    except (TypeError, ValueError):
        return UNVERSIONED_LIVENESS


def traced_width(graph) -> Optional[int]:
    """The tensor-parallel width the graph was traced at, or None.

    From `key.topology`, which is how a graph records the group it ran in.
    None means the graph does not say -- not "width one". The difference
    matters: the activation peak is the one term that shards, so a graph whose
    width is unknown cannot be scaled to a target of known width, and guessing
    one is how a TP=1 peak ends up inside a TP=4 budget.
    """
    topology = dict((graph.get("key") or {}).get("topology") or ())
    width = topology.get("tp")
    return int(width) if width else None


def traced_shape(graph) -> tuple:
    """The step's shape as `(query_lens, context_lens)`, per request.

    Not the token total. Two graphs of the same total are not the same step:
    the 27B's `s27prefhead` and `s27prefdeep` are both `batch_signature
    [16384]` -- identical keys -- and differ by 98 304 tokens of history, which
    is 7x the KV to read and a different attention branch. Matching on the sum
    picks whichever was traced last.

    Read from `provenance.batch_spec` (what a derivation was asked for) or
    `provenance.shape` (what a capture observed), in that order. Falling back
    to `key.batch_signature` gives the queries and `()` for the context, which
    is "history unknown" and not "history zero" -- the caller has to decide
    what an unknown history is worth, and for warmup matching it is worth
    nothing.
    """
    provenance = graph.get("provenance") or {}
    spec = provenance.get("batch_spec") or {}
    if spec.get("query_lens"):
        return (tuple(int(n) for n in spec["query_lens"]),
                tuple(int(n) for n in (spec.get("context_lens") or ())))
    shape = provenance.get("shape") or {}
    if shape.get("num_scheduled_tokens"):
        return (tuple(int(n) for n in shape["num_scheduled_tokens"]),
                tuple(int(n) for n in (shape.get("context_lens") or ())))
    signature = (graph.get("key") or {}).get("batch_signature") or ()
    return (tuple(int(n) for n in signature), ())


def activation_curve(graph, *, strict_dtypes: bool = False) -> list:
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
            aliases = op.get("output_aliases") or ()
            for position, shape in enumerate(op.get("output_shapes") or ()):
                if position < len(aliases) and aliases[position] is not None:
                    continue  # written into, not allocated
                dtype, basis = _output_dtype(op, position)
                if basis != "recorded" and strict_dtypes:
                    # A strict walk takes the graph's word and nothing else.
                    raise UnfoundedActivation(
                        "operator %d (%s) produces an output the graph records "
                        "no dtype for; its arguments are (%s) and the walk "
                        "would size it as %s by %s. Record `output_dtypes`, or "
                        "walk with `strict_dtypes=False` and read "
                        "`dtype_ambiguities`"
                        % (index, op.get("name"),
                           ", ".join(op.get("dtypes") or ()) or "none",
                           dtype, basis))
                if dtype is None:
                    # No argument dtype at all. Two bytes is the model's own
                    # activation dtype and it is written down here as the guess
                    # it is; `dtype_ambiguities` lists the output.
                    dtype = "bfloat16"
                size = _bytes_of(shape, dtype)
                if size:
                    held[(index, position)] = size
                    live += size
        curve.append(live)
    return curve


def peak_activation_bytes(graph, *, strict_dtypes: bool = False) -> int:
    """The most activation memory live at once, by walking the graph.

    A tensor is live from the operator that produced it until its last
    consumer, so this is a def-use walk: add an operator's outputs as it runs,
    drop every tensor whose last reader has just run, and keep the high-water
    mark. Inputs with no producer are not counted -- a weight is not an
    activation, and counting it here would double it against the weight term.
    Nor is an in-place operator's output, which is not a new tensor.

    Output dtype comes from the graph where the graph records it, from the
    arguments where they all agree, and otherwise from PyTorch's promotion
    rule, with `dtype_ambiguities` listing every output that was not recorded.
    `strict_dtypes=True` refuses anything but a recorded dtype. Sizing an
    output at argument 0's dtype -- the rule until now -- is how one
    `aiter::masked_embedding` came to be counted at int32: 335 544 320 B for a
    buffer that is 167 772 160.

    One approximation remains, stated rather than hidden: a tensor with no
    reader in the graph is freed immediately, where the engine frees it whenever
    the last Python reference goes -- so this is a lower bound on the high-water
    mark, not a bound on what the allocator reserves.
    """
    return max(activation_curve(graph, strict_dtypes=strict_dtypes) or [0])


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


def scratch_bytes_per_token(graph, *, strict_dtypes: bool = False) -> float:
    """Activation memory per token that no recorded operator output explains.

    A dispatch tracer sees what crosses the dispatcher. `torch.empty` called
    inside a custom operator does not: re-entering the operator from
    `__torch_dispatch__` runs below the mode. For an out-variant the
    destination can be recovered, and is; for an operator that *returns* a
    tensor and also allocates internal scratch, nothing in the graph says the
    scratch exists. The 0.6B has almost none. The hybrid 27B's chunked-scan and
    DeltaNet kernels have enough to put the walk 37% under its own step's
    measured peak.

    **This is measured, not derived, and deliberately so.** Correcting the walk
    per operator from the recorded curve would just reproduce the recorded
    curve -- exact at the traced shape and worth nothing. What is recorded here
    is one number, the shortfall per token, on the same linear-in-tokens
    footing as the rest of the activation term. It generalises to a shape the
    trace was not taken at, which is the only thing attribution can honestly
    buy. It does not generalise to a model that was never traced.

    Zero when the graph carries no measured peak to compare against, which
    leaves the walk exactly as it was.
    """
    measured = (graph.get("provenance") or {}).get("activation_peak_bytes")
    tokens = sum(int(n) for n in
                 ((graph.get("key") or {}).get("batch_signature") or ()))
    if not measured or not tokens:
        return 0.0
    return max(0.0, (int(measured) - peak_activation_bytes(
        graph, strict_dtypes=strict_dtypes)) / tokens)


class UnfoundedPrediction(ValueError):
    """A derived prediction was asked for and a term has nothing behind it.

    Raised rather than defaulted. The caller asked what a configuration
    *would* do; a number taken from the box it was asked on answers a
    different question, and answering it silently is how a fallback comes to
    be reported as a forecast.
    """


class UnfoundedActivation(UnfoundedPrediction):
    """The graph cannot support an activation figure, at any token count.

    Raised rather than returned so that a budget is never built on a number
    nobody observed. A caller that would rather size from a device catches it;
    a caller sizing a configuration nobody has run has to hear it, because for
    that caller there is nothing else to fall back on.
    """


#: The terms `modelled_readings` would otherwise take from built-in defaults.
#: Each was fitted somewhere, on some width of some model; a prediction has to
#: say where, which is why a profile that names no calibration is refused
#: rather than quietly given these.
CALIBRATED_TERMS = ("persistent", "non_torch", "load_residue")


def derived_readings(profile: Mapping, *, warmup_tokens: int,
                     load, enforce_eager: bool = False,
                     world_size: Optional[int] = None,
                     source: str = "the profile"):
    """The five readings and the activation peak, or a refusal naming what is missing.

    Fails closed, which is the entire point of the function. Asking for a
    derived prediction and receiving a device reading is worse than receiving
    nothing: the number looks like a forecast, is a measurement of a different
    configuration, and nothing downstream can tell the two apart. So every term
    that would otherwise be defaulted, inferred from the running box or skipped
    on an exception is a refusal here instead.

    `load` reads a path and returns parsed JSON; the caller owns the file
    system so that this stays testable off a device.

    `world_size` is the width the *deployment* is actually about to run at. It
    is optional only so that device-free callers that have no deployment can
    omit it; a caller that knows should pass it. Every width-dependent term
    here is keyed off the profile's own `world_size`, and the calibration
    tables are read with `_at_width`, which answers a width it has no entry for
    from the widest one below it. So a profile written for TP=2 handed to a
    TP=4 run does not fail -- it silently sizes TP=2 and reports it as a
    forecast for TP=4. That is refused here instead.

    There are two ways to reach the activation term and the profile picks one.
    With `graph`, the device-free walk over a graph traced at the target width
    -- unchanged, and still refused if the graph was traced anywhere else. With
    `model_config`, the config-derived instant, which needs no per-width graph
    but does need `compile_mode`: the instant is witnessed in one program, and
    defaulting the mode would choose a program for the caller. `model_config`
    wins when both are given.

    Returns `(readings, activation_bytes)`. Raises `UnfoundedPrediction`, or
    `UnfoundedActivation` from the walk.
    """
    def refuse(what: str):
        raise UnfoundedPrediction(
            "%s carries %s. A prediction cannot fall back to the device it is "
            "running on: that device is a different configuration, which is "
            "the reason for modelling it. Supply the term, or ask for a "
            "measurement instead." % (source, what))

    total = int(profile.get("total") or 0)
    if total <= 0:
        refuse("no card capacity (`total`)")
    if not int(profile.get("parameters") or 0):
        refuse("no parameter bytes")
    if not int(warmup_tokens or 0):
        refuse("a configuration with no `max_num_batched_tokens` or "
               "`max_model_len`, so there is no warmup shape to evaluate the "
               "activation peak at")
    width = int(profile.get("world_size") or 1)
    if world_size is not None and int(world_size) != width:
        refuse("a `world_size` of %d for a deployment about to run at TP=%d. "
               "Every width-dependent term is keyed off the profile's width "
               "and the calibration tables answer a missing width from the "
               "widest one below it, so this would size TP=%d and report it as "
               "a forecast for TP=%d" % (width, int(world_size), width,
                                         int(world_size)))
    config_path = str(profile.get("model_config") or "").strip()
    if config_path:
        # The config-derived instant. It needs no graph at the target width,
        # because the instant is witnessed once at the source and its widths
        # come from the checkpoint -- but it is witnessed in one *program*, so
        # the profile has to say which program it is asking about.
        mode = str(profile.get("compile_mode") or "").strip()
        if not mode:
            refuse("a model config for a config-derived activation term but no "
                   "`compile_mode`. The activation instant is witnessed "
                   "per-program: an instant from an Inductor-compiled run is "
                   "not evidence about an eager one, and defaulting the mode "
                   "would pick a program on the caller's behalf")
        # The same argument one step further out: the profile names a program
        # and the deployment runs one, and `enforce_eager` is how the run says
        # which. A profile that says `inductor` for a run with graphs disabled
        # is an activation instant from a different program, and the graph-pool
        # term is already branching the other way on the same flag.
        if enforce_eager and mode != "eager":
            refuse("a `compile_mode` of %r for a run with `enforce_eager` set. "
                   "The activation instant is witnessed per-program and the "
                   "graph pool is zero under eager, so the two halves of this "
                   "prediction would describe different runs" % mode)
        # Only that direction. `enforce_eager=False` is also what a caller with
        # no opinion passes, so it is not a statement that graphs are on and
        # cannot be read as one.
        config = load(config_path)
        config = config.get("text_config", config)
        instant = activation_instant_bytes(
            config, int(warmup_tokens), width, compile_mode=mode,
            dtype_bytes=int(profile.get("dtype_bytes") or 2))
        activation = int(instant["bytes"])
    else:
        graph_path = str(profile.get("graph") or "").strip()
        if not graph_path:
            refuse("neither an operator graph nor a model config, so the "
                   "activation peak has no evidence")

        graph = load(graph_path)
        traced = traced_width(graph)
        if traced is None:
            refuse("an operator graph that does not record the tensor-parallel "
                   "width it was traced at (`key.topology`), and the activation "
                   "peak is the one term that shards")
        if traced != width:
            refuse("an operator graph traced at TP=%d for a prediction at TP=%d. "
                   "The activation peak shards and the walk cannot be re-sharded "
                   "after the fact; trace the graph at the target width"
                   % (traced, width))

        activation = activation_bytes_at(graph, int(warmup_tokens))
    calibration = _prediction_calibration(profile, load, refuse, source)
    readings = modelled_readings(
        total_bytes=total, world_size=int(profile.get("world_size") or 1),
        parameters=int(profile["parameters"]),
        buffers=int(profile.get("buffers") or 0),
        activation_bytes=activation, calibration=calibration,
        enforce_eager=enforce_eager)
    return readings, activation


def _prediction_calibration(profile: Mapping, load, refuse, source: str):
    """The calibration behind a prediction, with its provenance checked.

    Two separate demands. The terms have to be *there*: the built-in defaults
    were fitted at another width on another model, and standing in for a
    missing calibration is the same fallback in a smaller place. And the
    provenance has to say where each came from, because a term fitted on the
    target configuration turns the prediction into a restatement of the
    measurement it is meant to anticipate. That one is refused outright rather
    than reported -- there is no use for the answer.
    """
    cal_path = str(profile.get("calibration") or "").strip()
    if not cal_path:
        refuse("no calibration, so `persistent`, `non_torch` and the load "
               "residue would come from defaults fitted at another width on "
               "another model")
    calibration = load(cal_path)
    missing = [t for t in CALIBRATED_TERMS if not calibration.get(t)]
    if missing:
        refuse("a calibration with no %s" % ", ".join(missing))
    provenance = calibration.get("provenance") or {}
    if not provenance:
        refuse("a calibration with no provenance block, so nothing says "
               "whether its terms were fitted at the source or read off the "
               "target")
    unstated = [t for t in CALIBRATED_TERMS
                if not str(provenance.get(t) or "").strip()]
    if unstated:
        refuse("a calibration whose provenance does not cover %s"
               % ", ".join(unstated))
    target = sorted(t for t in CALIBRATED_TERMS
                    if str(provenance[t]).strip().upper().startswith("X"))
    if target:
        raise UnfoundedPrediction(
            "%s calibrates %s on the target configuration itself. A prediction "
            "fitted on the thing it predicts is not one."
            % (cal_path, ", ".join(target)))
    _check_calibration_conditions(calibration, refuse)
    return calibration


def _check_calibration_conditions(calibration: Mapping, refuse) -> None:
    """A calibration may only be spent in the environment it was taken in.

    Opt-in by data: only a calibration that states `conditions` is checked, so
    nothing that predates them changes behaviour. The case this exists for is
    the collective pools -- under `PYTORCH_HIP_ALLOC_CONF=expandable_segments`
    or `AITER_CUSTOM_AR_RAW_INPUT_POOL` the 1 GiB input pool per instance moves
    out of the torch allocator, so 2 GiB per rank crosses from `load_residue`
    into `non_torch`. Both terms stay plausible and both are wrong, which is
    exactly the failure a prediction cannot report on its own. Read from this
    process's environment because this process is the deployment.
    """
    conditions = ((calibration.get("topology_delta") or {}).get("conditions")
                  or calibration.get("conditions") or {})
    if not conditions:
        return
    bad = []
    for key, want in conditions.items():
        got = os.environ.get(key)
        got = None if got in (None, "") else str(got)
        want = None if want in (None, "") else str(want)
        if got != want:
            bad.append("%s is %r here, %r when the calibration was measured"
                       % (key, got, want))
    if bad:
        refuse("a calibration measured under a different environment: %s. "
               "Under expandable segments or a raw collective input pool, "
               "2 GiB per rank moves between `load_residue` and `non_torch`, "
               "so both terms would be wrong and neither would look it"
               % "; ".join(bad))


#: The instants at which the activation high-water mark has been witnessed on
#: the source config, each tagged with the program it was witnessed in.
#:
#: The two are **not the same program**, which is why they may not be combined.
#: The TP=1 warmup allocation history was recorded from an Inductor-compiled
#: run -- its frames pass through ``/tmp/torchinductor_root/...`` and
#: ``torch/_inductor/utils.py:3220`` -- while the device-free walk graphs carry
#: ``compilation_level: 0``. Inductor decides buffer reuse and lifetimes in its
#: own scheduler, so the two disagree in both directions at TP=1: the compiled
#: run holds the layer's attention buffers through that layer's MLP (+741 343
#: 232 B) and reuses hidden-sized buffers the eager walk keeps separate
#: (-503 316 480 B). Taking a maximum across them would be a maximum over two
#: different programs, not a bound on either.
#:
#: ``sharded`` widths are split across ranks; ``replicated`` ones are not;
#: ``collective`` ones exist only above one rank.
GDN_ACTIVATION_INSTANTS = {
    "linear_attn": {
        "compile_mode": "inductor",
        "witness": "TP=1 warmup allocation history (S27), matched allocation "
                   "by allocation; peak at the act_fn allocation inside layer "
                   "1's MLP, with that layer's in_proj and core-attention "
                   "buffers still live",
        "sharded": ("mlp_gate_up", "mlp_act", "in_proj_qkvzba", "attn_value"),
        "replicated": ("hidden", "hidden", "hidden"),
        # The peak falls before the down-projection, so no collective
        # destination for it exists yet; whether an earlier one is still live
        # at that point is unwitnessed, since no compiled history exists above
        # TP=1. Left out, and reported as uncounted.
        "collective": (),
        "collective_witnessed": False,
    },
    "mlp_down": {
        "compile_mode": "eager",
        "witness": "device-free walk high-water mark on the pinned recapture; "
                   "same module path and same non-collective ordinal 72 at "
                   "TP=1, 2 and 4",
        "sharded": ("mlp_gate_up", "mlp_act"),
        "replicated": ("hidden",) * 6,
        "collective": ("hidden",),
        "collective_witnessed": True,
    },
}


def gdn_activation_widths(config: Mapping) -> dict:
    """The trailing widths of the GDN-hybrid activation buffers, off a config.

    Nothing here is measured. ``in_proj_qkvzba`` is the concatenation the
    module actually projects to -- q and k at ``linear_num_key_heads`` x
    ``linear_key_head_dim``, v and z at ``linear_num_value_heads`` x
    ``linear_value_head_dim``, then b and a at one element per value head --
    and on the 27B that is 16 384 + 96, which is the width the device-free walk
    produces at ``linear_attn.in_proj_qkvzba`` and shards to 8240 and 4120.
    """
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["linear_num_value_heads"])
    key = int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
    value = heads * int(config["linear_value_head_dim"])
    return {"hidden": hidden,
            "mlp_gate_up": 2 * intermediate,
            "mlp_act": intermediate,
            "attn_value": value,
            "in_proj_qkvzba": 2 * key + 2 * value + 2 * heads}


def activation_instant_bytes(config: Mapping, tokens: int, world_size: int = 1,
                             *, compile_mode: str, dtype_bytes: int = 2,
                             instants: Optional[Mapping] = None) -> dict:
    """A candidate for the activation term, within one compile mode.

    **A candidate, not a bound.** Each instant is an approximation of the live
    set at one point of one program, and a maximum over approximations is only
    a lower bound if each input is one. Neither is established as such, so the
    result is labelled and used as a candidate.

    ``compile_mode`` is required and is not a formality. The gate the predictor
    has to match is ``peak_torch - current_torch`` of the deployment as it
    actually runs; the two witnessed instants come from two different programs
    (see ``GDN_ACTIVATION_INSTANTS``), and mixing them is refused rather than
    silently maximised.

    Within a mode the widths come from ``config.json`` via
    ``gdn_activation_widths`` and the live sets from that mode's witness.
    Nothing is fitted and no TP=2 or TP=4 measurement is opened.

    The returned ``uncounted`` names what the mode's witness cannot cover at
    this width, so a caller can see the reservation instead of inheriting it
    silently.
    """
    widths = gdn_activation_widths(config)
    chosen = {name: instant
              for name, instant in (instants or GDN_ACTIVATION_INSTANTS).items()
              if instant.get("compile_mode") == compile_mode}
    if not chosen:
        raise UnfoundedActivation(
            "no activation instant is witnessed for compile mode %r; the "
            "witnessed modes are %s, and an instant from one program is not "
            "evidence about another"
            % (compile_mode,
               sorted({i.get("compile_mode")
                       for i in (instants or GDN_ACTIVATION_INSTANTS).values()})))
    best = None
    for name, instant in sorted(chosen.items()):
        sharded = sum(widths[key] for key in instant["sharded"])
        replicated = sum(widths[key] for key in instant["replicated"])
        if world_size > 1:
            replicated += sum(widths[key]
                              for key in instant.get("collective") or ())
        if sharded % world_size:
            raise UnfoundedActivation(
                "instant %r sums to %d sharded elements, which %d ranks do not "
                "divide; that is not the split the module makes"
                % (name, sharded, world_size))
        total = tokens * dtype_bytes * (sharded // world_size + replicated)
        if best is None or total > best["bytes"]:
            uncounted = []
            if world_size > 1 and not instant.get("collective_witnessed"):
                uncounted.append(
                    "collective destinations live at this instant above one "
                    "rank are unwitnessed for %r and are not counted" % name)
            best = {"instant": name, "bytes": int(total),
                    "compile_mode": compile_mode,
                    "sharded": sharded, "replicated": replicated,
                    "witness": instant["witness"],
                    "is_candidate": True, "uncounted": tuple(uncounted)}
    return best


def activation_bytes_at(graph, tokens: int, *,
                        strict_dtypes: bool = False) -> int:
    """The activation peak at a token count the graph was not traced at.

    Linear in tokens, which is not an assumption but a measurement: the walk
    scaled from a 3494-token trace lands on the independently measured 4096-token
    warmup peak to +0.0% at TP=1, 2 and 4 alike.

    Refuses a graph that neither recorded liveness nor measured a peak. Such a
    graph still walks -- `_deaths` falls back to last-read -- and on the 27B's
    meta-derived prefill graphs the walk returns 570 425 344 B against a
    measured 2 956 984 320 B, understating the term by 2.4 GB out of
    allocations it never saw freed. A measured peak is enough on its own:
    `scratch_bytes_per_token` carries whatever the walk missed, so the answer
    at the traced shape is the measurement however poor the walk. Neither is
    not enough, and the honest failure is louder than a quiet 19%.
    """
    if not liveness_is_recorded(graph) and not (
            (graph.get("provenance") or {}).get("activation_peak_bytes")):
        raise UnfoundedActivation(
            "this graph records neither tensor deaths nor a measured "
            "activation peak, so there is no liveness in it to scale: "
            "%s" % ((graph.get("provenance") or {}).get("source") or "unknown"))
    peak = peak_activation_bytes(graph, strict_dtypes=strict_dtypes)
    traced = sum(int(n) for n in
                 ((graph.get("key") or {}).get("batch_signature") or ()))
    if not (traced and tokens):
        return peak
    # The walk scales, and so does what the walk cannot see -- both are
    # activation memory and both are linear in tokens.
    return int(peak * tokens / traced
               + scratch_bytes_per_token(
                   graph, strict_dtypes=strict_dtypes) * tokens)


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
#:
#: **Superseded as a reading, kept as a number.** The identical bytes are not
#: evidence that the graphs pin nothing sharded: at TP>1 the runner does not
#: capture the LM head at all (`logits_in_graph = world_size == 1 and not
#: is_tbo`, `model_runner.py:4104`), and the head is the only part of the
#: pinned set that scales with the ladder. So this is the *whole* pinned set
#: with its one variable term removed, which `capture_pinned_bytes` states
#: directly and which predicts TP=1 to the byte. The switch is the predicate,
#: not the width -- a TP=1 TBO run also drops the head. The AITER collective
#: buffer remains outside the torch allocator, but it is not what this
#: measures.
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

    Above width one the ladder stops mattering -- see `DEFAULT_POOL_SHARDED`,
    and `capture_pinned_bytes` for why that is the LM head leaving the graph
    rather than a property of width.

    **What it is compared against is not private-pool residency.** The number
    in the record is `memory_reserved()` differenced across the whole capture
    window (`model_runner.py:4120`, `:4346`), and that window contains a full
    eager warmup forward per bucket (`:4229`) whose segments grow the *global*
    pool, with `empty_cache` patched out inside piecewise capture
    (`cuda_graph.py`). A release anywhere else in the process lands in it too,
    and `max(..., 0)` reads a net release as a pool of zero. The allocated
    delta beside it is the sounder target, and pool-scoped residency is
    readable directly -- `torch.cuda.memory_snapshot(mempool_id)` takes the
    id that `graph.pool()` returns -- which is what a future capture probe
    should use instead of a global difference.
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


#: The capture-time *allocated* delta that is not the LM head, in bytes.
#:
#: **Source calibration, now witnessed.** The value was read off the 27B's
#: TP=1 record (S27) as the residue after the logits term below --
#: 110 981 120 - 63 x 248 320 x 2 -- and carried as a taken number. A pool-id
#: scoped capture probe on the same source config (S27, TP=1, node18 GPU0,
#: `agent_scratch/memval/pool_probe/att2_artifact.json`) has since named what
#: the residue is, by diffing the *global* segment list across the capture
#: window. Nothing was released; six segments appeared, and the residue is two
#: allocations that live **outside every capture pool**:
#:
#:   * one 79 691 776 B (76 MiB exactly) block, `requested_size == size`, in
#:     its own exactly-sized oversize `large` segment -- a fixed-size
#:     workspace request, not a tensor of any model dimension; and
#:   * two 512 B blocks of an 8 B request each, in a `small` segment.
#:
#: 79 691 776 + 2 x 512 = 79 692 800, the constant, to the byte. So it is
#: neither private-pool residency nor a size-class rounding of the logits
#: (the logits round by 0 B; see the docstring below). A request for exactly
#: 76 MiB that depends on neither model nor width is what makes the same value
#: appear on the 0.6B (C06) at TP=2, 4 and 8 over ladders from 31 to 1071
#: tokens. The TP=1 warmup allocation history already on disk
#: (`agent_scratch/memval/producer_packet/tp1_probe/out/warmup_history.959476.pickle`)
#: carries a 79 691 776 B `segment_alloc` whose frames run
#: `aiter/tuned_gemm.py:450:torch_gemm` <- `gemm_a16w16` <- the inductor region
#: of the GDN linear-attention forward. **Equal size settles nothing in either
#: direction**, and neither does factoring it: any size divides many ways, and
#: 2432 is not a width this checkpoint produces. What is witnessed about the
#: *warmup* block is its life, not its shape. It is allocated once, at the
#: second event of the window, and is still live when the window closes -- the
#: window's whole net allocated retention, 79 691 776 B, is this one block.
#: `tuned_gemm.py` contains no workspace at all (no `workspace` appears in the
#: file); `torch_gemm` ends in `F.linear(inp, weights, bias)`, so what it
#: returns is a GEMM output of shape `[M, N]` where `N` is a weight output
#: width -- of the config's widths only `in_proj_qkvz` = 16 384 divides
#: 39 845 888, which would make `M` 2432 tokens.
#:
#: The *capture-window* block is a different observation and stays
#: unattributed: it carries no frames, and all six segments new in that window
#: are on the capture stream (460554448) while the 313 pre-existing segments
#: are on stream 0 -- so it was requested on the side stream capture runs on,
#: outside the graph's private pool. A per-stream cache would explain both a
#: fixed size and a second allocation after warmup already made one; so would
#: several other things. Naming it needs allocation history recorded *inside*
#: the capture window, and no prediction waits on that: the term is
#: source-calibrated by construction.
#:
#: One thing is settled and matters more for the gate. The warmup block is
#: retained across its window, so it is in `peak` and in `current` alike and
#: **cancels in `peak - current`**. It is excluded from the activation term
#: once, there, and is not also carried as a residue anywhere else.
#: The constant therefore stays a calibrated number rather than a derived one,
#: and stays overridable via `calibration["graph_pool"]["fixed_pinned"]`: it is
#: a property of the AITER/ROCm build, not of the model. Its value must not
#: move -- frozen predictions were made with it.
CAPTURE_FIXED_PINNED = 79_692_800


def capture_pinned_bytes(capture_sizes, *, vocab_size: int = 0,
                         dtype_bytes: int = 2, q_len: int = 1,
                         world_size: int = 1, tbo: bool = False,
                         logits_in_graph: Optional[bool] = None,
                         enforce_eager: bool = False,
                         calibration: Optional[Mapping] = None) -> int:
    """What capture *pins*, as a mechanism rather than as a width constant.

    The engine captures a warmup forward per bucket and, at TP=1 only, the LM
    head with it::

        self.logits_in_graph = self.world_size == 1 and not is_tbo
        ...
        if self.logits_in_graph:
            graph_logits = self.model.compute_logits(outputs[:num_tokens])

    (`model_runner.py:4104`, `:4297`.) The logits tensor is allocated inside
    the capture, so it comes from the graph's private pool, and the runner
    keeps it in `self.graph_logits[(bs, max_q_len)]`, so it stays live. Capture
    builds decode metadata, so `ParallelLMHead.forward` takes no last-token
    index and the tensor is `[num_tokens, vocab_size]` whole -- at TP>1 it
    would also be all-gathered, but at TP>1 it is not captured at all.

    So the term is `vocab_size x dtype_bytes x sum(captured num_tokens)` when
    the head is in the graph, and nothing when it is not, over a fixed residue.
    On the 27B's TP=1 record that is 79 692 800 + 63 x 248 320 x 2 =
    110 981 120 B, which is the recorded allocated delta **to the byte**.

    That row is exact by construction -- it is where the residue came from --
    and the TP=2/TP=4 rows that also land at +0.0% were checked against records
    already on disk, so they are **retrospective evidence, not a fresh frozen
    evaluation**. All of it is the capture-time *allocated* delta. The reserved
    delta the engine records is a different quantity (see
    `measured_graph_pool_bytes`) and agreement here says nothing about it.

    The pool-id scoped probe (S27, TP=1) checks the *mechanism* of the logits
    term rather than just its total. Six captures, one private pool `(1, 0)`,
    runner keys `(bs, max_q_len)` = (32,1) (16,1) (8,1) (4,1) (2,1) (1,1), so
    `sum` = 63. Inside that pool six new live blocks appeared, one per bucket,
    of 15 892 480 / 7 946 240 / 3 973 120 / 1 986 560 / 993 280 / 496 640 B --
    each exactly `bs x 248 320 x 2` and each with `requested_size == size`, so
    the term rounds by **0 B**, not approximately. They sum to 31 288 320 B =
    `63 x 248 320 x 2`. Five separated counters for that pool: reserved
    residency 46 137 344, active allocated 31 288 320, active requested
    31 288 320, internal rounding 0, inactive capacity 14 849 024 B.

    Two things follow. The residue above is *not* in the pool -- pool active
    allocated is the logits term alone -- so a pool-scoped search could never
    have found it; it took a global before/after block diff. And the reserved
    side is a different decomposition again: the window's reserved delta was
    127 926 272 B from six new segments with none released -- four in the
    capture pool (46 137 344 B) and two outside (the 79 691 776 B oversize
    segment plus a whole 2 097 152 B small segment holding only the 1 024 B of
    8-byte scalars). The allocated-side constant is 79 692 800 B; the same
    residue costs 81 788 928 B of *reserved*. Do not use one for the other.

    All of this is source-only (S27) diagnostic evidence for how the term is
    built. It explains the global reserved gate; it does not redefine it, and
    it is not target validation -- that remains the frozen e2e cc-traces gates.

    **The switch is `logits_in_graph`, not the width.** Reading it as a width
    law -- which `measured_graph_pool_bytes` still does -- gets the right
    answer for the wrong reason at TP>1 and the wrong answer at TP=1 under TBO,
    where the head leaves the graph while the width stays one.

    The model output is not in this term: the capture writes it into the
    preallocated `forward_vars["outputs"]` (`model_runner.py:4237`), the same
    buffer as O19, which was allocated at engine init and is already inside
    `current_torch`.

    Raises `UnfoundedPrediction` when the head is in the graph and no
    vocabulary was given, rather than quietly returning the residue alone.
    """
    if enforce_eager:
        return 0
    sizes = [int(s) for s in (capture_sizes or ()) if int(s) > 0]
    if not sizes:
        return 0
    settings = (calibration or {}).get("graph_pool") or {}
    fixed = int(settings.get("fixed_pinned", CAPTURE_FIXED_PINNED))
    if logits_in_graph is None:
        logits_in_graph = (int(world_size) == 1) and not tbo
    if not logits_in_graph:
        return fixed
    if not vocab_size:
        raise UnfoundedPrediction(
            "the LM head is captured at this configuration, so the pinned "
            "pool contains vocab_size x %d x %d tokens, and no vocabulary "
            "was given" % (int(dtype_bytes), sum(sizes) * int(q_len)))
    tokens = sum(sizes) * int(q_len)
    return fixed + int(vocab_size) * int(dtype_bytes) * tokens


#: The caching allocator's own size constants, read off the shipped header
#: `torch/include/c10/core/AllocatorConfig.h` rather than inferred from a
#: measurement. They are what turns a *requested* size into mapped bytes, so
#: they are what separates the reserved side from the allocated side.
ALLOCATOR_MIN_BLOCK = 512          #: kMinBlockSize -- every request rounds up
ALLOCATOR_SMALL_SIZE = 1_048_576   #: kSmallSize -- largest "small" allocation
ALLOCATOR_SMALL_BUFFER = 2_097_152  #: kSmallBuffer -- small segment size
ALLOCATOR_MIN_LARGE_ALLOC = 10_485_760  #: kMinLargeAlloc
ALLOCATOR_ROUND_LARGE = 2_097_152  #: kRoundLarge -- oversize segments round here
ALLOCATOR_LARGE_BUFFER = 20_971_520  #: kLargeBuffer -- large segment size


def allocator_block_bytes(requested: int) -> int:
    """What a request of `requested` bytes occupies as a *block*."""
    n = int(requested)
    if n < ALLOCATOR_MIN_BLOCK:
        return ALLOCATOR_MIN_BLOCK
    return ALLOCATOR_MIN_BLOCK * (
        (n + ALLOCATOR_MIN_BLOCK - 1) // ALLOCATOR_MIN_BLOCK)


def allocator_segment_bytes(requested: int) -> int:
    """What the allocator *maps* to satisfy a fresh request of that size.

    The three cases are the allocator's, not ours: a small request takes a
    whole `kSmallBuffer` segment, a request under `kMinLargeAlloc` takes a
    whole `kLargeBuffer` segment, and anything larger gets its own segment
    rounded to `kRoundLarge`. A later request may be packed into an existing
    segment's free tail instead, which is why this is an upper bound per
    allocation and only exact for one that has to map new memory.
    """
    n = allocator_block_bytes(requested)
    if n <= ALLOCATOR_SMALL_SIZE:
        return ALLOCATOR_SMALL_BUFFER
    if n < ALLOCATOR_MIN_LARGE_ALLOC:
        return ALLOCATOR_LARGE_BUFFER
    return ALLOCATOR_ROUND_LARGE * (
        (n + ALLOCATOR_ROUND_LARGE - 1) // ALLOCATOR_ROUND_LARGE)


def allocator_charged_bytes(requested: int, *, fresh_segment: bool = True) -> dict:
    """What `allocated_bytes` charges for one request, which is not the request.

    Four different quantities get called "the size of an allocation" and the
    prediction needs them apart:

    * **requested** -- what the caller asked for. This is what a trace entry's
      `size` carries (the artifact settles it: the capture window contains
      2-byte allocations, and no block is ever smaller than `kMinBlockSize`).
    * **block** -- the request rounded up to `kMinBlockSize`.
    * **segment** -- what `hipMalloc` maps, per `allocator_segment_bytes`.
    * **charged** -- what the allocator adds to `allocated_bytes`, which is the
      *block it hands out*, and that is where the surprise lives.

    When a fresh segment is mapped, the allocator splits the remainder off into
    a free block only if it is worth splitting. In the large pool that test is
    `remaining > kSmallSize`; in the small pool it is `remaining >=
    kMinBlockSize`. A remainder that fails the test is **not** split -- it stays
    inside the handed-out block, and `allocated_bytes` charges the whole thing.
    So a request can be charged more than it asked for while nothing is wasted
    anywhere a snapshot would show as free.

    The rule is witnessed, not assumed: `should_split` lives in a `.cpp` that
    the wheel does not ship, so it was read off the shipped binary's behaviour
    in the S27 TP=1 warmup snapshot. Of 305 large segments, 213 hold a single
    active block filling the segment, and their excess over the request is
    either 0 (210 of them) or exactly 1 048 576 (3) -- never more. 49 hold an
    active block plus a split-off tail, and the smallest such tail is 1 114 112
    -- never less. The boundary sits exactly at `kSmallSize`, with no
    counterexample either side.

    `fresh_segment=False` means the request was served out of an existing
    segment's free space, where the host block's size is a property of the
    history and not of this request; the charge is then at least the block and
    this returns that lower bound.
    """
    block = allocator_block_bytes(requested)
    segment = allocator_segment_bytes(requested)
    if not fresh_segment:
        return {"requested": int(requested), "block": block,
                "segment": None, "charged": block, "retained": 0,
                "split_off": None, "fresh_segment": False}
    remaining = segment - block
    if block <= ALLOCATOR_SMALL_SIZE:
        splits = remaining >= ALLOCATOR_MIN_BLOCK
    else:
        splits = remaining > ALLOCATOR_SMALL_SIZE
    charged = block if splits else segment
    return {"requested": int(requested), "block": block, "segment": segment,
            "charged": charged, "retained": 0 if splits else remaining,
            "split_off": remaining if splits else 0, "fresh_segment": True}


class _PoolSegment:
    """One `hipMalloc`ed mapping inside a private pool."""

    __slots__ = ("size", "small", "index")

    def __init__(self, size: int, small: bool, index: int):
        self.size = size
        self.small = small
        self.index = index


class _PoolBlock:
    """One block inside a segment, in the segment's address-ordered list."""

    __slots__ = ("segment", "offset", "size", "allocated", "prev", "next")

    def __init__(self, segment: _PoolSegment, offset: int, size: int):
        self.segment = segment
        self.offset = offset
        self.size = size
        self.allocated = False
        self.prev = None
        self.next = None


class AllocatorPool:
    """The caching allocator's own bookkeeping for one private pool.

    Given a request/free *stream* this decides for itself how many segments to
    map and how big each one is. No observed segment size, address or block
    layout is an input, which is what makes the result a prediction rather than
    a restatement. `reserved` is the sum of what it mapped.

    Provenance of each rule, kept apart on purpose:

    * **Source-proven.** The size constants (`ALLOCATOR_*` above), read off the
      shipped `c10/core/AllocatorConfig.h`.
    * **Empirical.** `should_split` (`remaining > kSmallSize` in the large
      pool, `>= kMinBlockSize` in the small one), best-fit block selection and
      coalescing on free. `should_split` and `get_free_block` live in a `.cpp`
      the wheel does not ship. The split threshold is witnessed from both sides
      in the S27 warmup snapshot (O27); the selection and coalescing rules are
      *not* independently witnessed, and are this class's main unproven
      assumption.

    Validated at the 27B capture window: fed the source TP=1 capture request
    order it maps four segments for 46 137 344 B, the recorded pool residency
    to the byte, and its live blocks at the end sum to 31 288 320 B, also
    exact. See O28/O29.
    """

    def __init__(self):
        self.segments: list = []
        self.free: list = []
        self.reserved = 0
        self.maps: list = []          # (forcing request, segment size), in order

    @staticmethod
    def _round(size: int) -> int:
        return allocator_block_bytes(size)

    @staticmethod
    def _segment_for(size: int) -> int:
        return allocator_segment_bytes(size)

    @staticmethod
    def _should_split(block: _PoolBlock, size: int) -> bool:
        remaining = block.size - size
        if block.segment.small:
            return remaining >= ALLOCATOR_MIN_BLOCK
        return remaining > ALLOCATOR_SMALL_SIZE

    def _get_free_block(self, size: int, small: bool):
        best = None
        for block in self.free:
            if block.segment.small != small or block.size < size:
                continue
            key = (block.size, block.segment.index, block.offset)
            if best is None or key < (best.size, best.segment.index, best.offset):
                best = block
        return best

    def _map_segment(self, size: int, small: bool, request: int) -> _PoolBlock:
        segment = _PoolSegment(size, small, len(self.segments))
        self.segments.append(segment)
        self.reserved += size
        self.maps.append((request, size))
        block = _PoolBlock(segment, 0, size)
        self.free.append(block)
        return block

    def malloc(self, request: int) -> _PoolBlock:
        size = self._round(request)
        small = size <= ALLOCATOR_SMALL_SIZE
        block = self._get_free_block(size, small)
        if block is None:
            block = self._map_segment(self._segment_for(size), small, request)
        self.free.remove(block)
        if self._should_split(block, size):
            tail = _PoolBlock(block.segment, block.offset + size,
                              block.size - size)
            tail.prev, tail.next = block, block.next
            if block.next is not None:
                block.next.prev = tail
            block.next = tail
            block.size = size
            self.free.append(tail)
        block.allocated = True
        return block

    def free_block(self, block: _PoolBlock) -> None:
        block.allocated = False
        self.free.append(block)
        for neighbour in (block.prev, block.next):
            if neighbour is None or neighbour.allocated or neighbour not in self.free:
                continue
            first, second = ((neighbour, block)
                             if neighbour.offset < block.offset
                             else (block, neighbour))
            first.size += second.size
            first.next = second.next
            if second.next is not None:
                second.next.prev = first
            self.free.remove(second)
            if second is block:
                block = first


def allocator_pool_bytes(stream) -> dict:
    """Replay a capture pool's request/free `stream` through the allocator.

    `stream` is an ordered iterable of `(op, key, size)`, where `op` is
    ``"alloc"`` or ``"free"`` and `key` pairs a free with its allocation. It is
    a *program*, not a measurement: the sizes are requests the captured forward
    makes, and where they come from is the caller's problem -- a recorded
    history at the source width, or a source-derived transformation of one.

    Returns the reserved bytes, the segments in the order they were mapped, the
    request that forced each, and what is still live at the end.

    A free with no matching allocation is ignored rather than raising: a
    transformation may legitimately drop a source branch (for example the
    `logits_in_graph` statement above one rank) and leave its free behind.
    """
    pool = AllocatorPool()
    blocks: dict = {}
    for op, key, size in stream:
        if op == "alloc":
            blocks[key] = pool.malloc(int(size))
        elif op == "free":
            block = blocks.pop(key, None)
            if block is not None:
                pool.free_block(block)
        else:
            raise ValueError("unknown stream op %r" % (op,))
    return {
        "reserved": pool.reserved,
        "segments": [size for _, size in pool.maps],
        "forced_by": [request for request, _ in pool.maps],
        "live_at_end": sum(block.size for block in blocks.values()),
        "segments_mapped": len(pool.segments),
    }


def capture_reserved_parts(pool_reserved: Optional[int] = None, *,
                           pool_stream=None,
                           fixed_pinned: int = CAPTURE_FIXED_PINNED) -> dict:
    """Split the capture window's *reserved* delta into what is derivable.

    The allocated side has a mechanism (`capture_pinned_bytes`). The reserved
    side has two halves:

    * **Outside the capture pools.** The fixed residue is one ~76 MiB request
      plus two 512 B blocks, so the allocator maps
      `allocator_segment_bytes(76 MiB) + allocator_segment_bytes(512)`. On the
      S27 TP=1 window that is 79 691 776 + 2 097 152 = 81 788 928 B, which is
      the observed figure with **no residual** and no fitted parameter. Note
      what it says: 1 024 B of live scalars cost a whole 2 MiB segment, so the
      allocated constant (79 692 800) and its reserved cost (81 788 928) are
      different numbers.
    * **Inside the capture pools.** Not derivable from what capture *pins*: on
      the S27 window the pool holds 46 137 344 B reserved while the same rule
      over the pinned blocks alone reads 83 886 080 B -- 82% high -- and one of
      its four segments holds no live block at all. It *is* derivable from what
      capture *requests*, in order. Pass `pool_stream` and the pool half is
      replayed through `allocator_pool_bytes`; pass `pool_reserved` and the
      figure is taken as given, as it was before.

    `pool_reserved_derived` reports which of the two happened, so a caller can
    tell a prediction from a restatement. Exactly one of the two arguments is
    required.

    The stream for a width other than the one that was recorded is a
    *transformation* of the recorded one, and the transformation carries its
    own assumptions -- which statements the target program does not run
    (`logits_in_graph`), which widths shard, and above all that the execution
    order is unchanged. None of that lives here; this function replays whatever
    program it is handed.
    """
    outside = (allocator_segment_bytes(fixed_pinned - 2 * ALLOCATOR_MIN_BLOCK)
               + allocator_segment_bytes(ALLOCATOR_MIN_BLOCK))
    if (pool_stream is None) == (pool_reserved is None):
        raise UnfoundedPrediction(
            "the capture pool's reserved half needs either a recorded figure "
            "(`pool_reserved`) or the request order to replay (`pool_stream`), "
            "and exactly one of them")
    if pool_stream is not None:
        replay = allocator_pool_bytes(pool_stream)
        return {
            "outside_pools": int(outside),
            "outside_pools_derived": True,
            "pool_reserved": int(replay["reserved"]),
            "pool_reserved_derived": True,
            "pool_replay": replay,
            "total": int(outside) + int(replay["reserved"]),
        }
    return {
        "outside_pools": int(outside),
        "outside_pools_derived": True,
        "pool_reserved": int(pool_reserved),
        "pool_reserved_derived": False,
        "total": int(outside) + int(pool_reserved),
    }


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


def _is_collective(op) -> bool:
    """Whether this operator exists only because the model is sharded.

    `group` is set for every collective the tracer or the recorder produced;
    the name test catches a hand-built `OpSpec` that omitted it.
    """
    name = (op.get("name") or "").lower()
    return bool(op.get("group")) or "all_reduce" in name or "all_gather" in name


def lineage_keys(graph) -> list:
    """A width-invariant identity for each operator, from its ancestry.

    Aligning two graphs by operator index only works while the graphs have the
    same operators, and tensor parallelism is precisely the case where they do
    not: every row-parallel matmul gains an all-reduce after it, so index *i*
    at TP=2 is a different operator from index *i* at TP=1, and the further
    into the model the further the drift.

    So identity comes from ancestry instead: an operator is the one that ran
    this name, on values produced by *those* operators, which is recursive and
    unique in a feed-forward graph -- the second layer's `gemm` has a different
    chain from the first layer's because its chain contains the first layer.
    Shapes and dtypes are deliberately not in the key: they are what the
    comparison is *for*, and putting them in would make every sharded tensor a
    non-match and report nothing.

    Collectives are transparent. A collective is not a value the model computes,
    it is a value made whole, and it exists at one width and not another; making
    it pass its input's identity through is what keeps the operator after it
    aligned with the operator after the matmul at TP=1. A shape-changing
    collective (`all_gather`) is passed through for *alignment* only -- its own
    outputs have no counterpart at TP=1 and are reported as such.

    Returns one key per operator, as a digest of the ancestry itself, so
    that equal ancestry gives an equal key in two graphs that discovered
    their operators in different orders -- which two widths do, because a
    width inserts operators.
    """
    ops = graph.get("ops") or ()
    interned: dict = {}
    keys: list = []
    for op in ops:
        sources = tuple(op.get("inputs_from") or ())
        if _is_collective(op):
            # Pass the first produced input's identity through.
            through = next((keys[s] for s in sources
                            if 0 <= s < len(keys)), None)
            keys.append(through if through is not None else "external")
            continue
        parents = tuple(keys[s] if 0 <= s < len(keys) else "external"
                        for s in sources)
        structure = (op.get("name"), parents)
        key = interned.get(structure)
        if key is None:
            # A digest of the structure, not the order it was met in: an
            # ordinal would make two graphs agree whenever they happened
            # to discover the same number of ancestries first, which is
            # index alignment wearing a different name.
            key = hashlib.blake2b(repr(structure).encode("utf-8"),
                                  digest_size=8).hexdigest()
            interned[structure] = key
        keys.append(key)
    return keys


def _non_collective_positions(ops) -> list:
    """Indices of the operators that exist at every width."""
    return [index for index, op in enumerate(ops) if not _is_collective(op)]


def _resolve_through_collectives(ops, index: int) -> int:
    """Follow a source index past any collectives to the value's real producer.

    A collective is a value made whole, not a value computed, so an operator
    whose input came from an all-reduce was really fed by whatever fed the
    all-reduce. Walking through keeps the comparison with TP=1 -- where no
    collective stands in the way -- an honest one.
    """
    seen = 0
    while 0 <= index < len(ops) and _is_collective(ops[index]):
        sources = ops[index].get("inputs_from") or ()
        index = next((s for s in sources if s >= 0), -1)
        seen += 1
        if seen > len(ops):          # a cycle cannot happen, but do not hang
            return -1
    return index


def module_path_keys(graph, module_paths) -> list:
    """A width-invariant identity taken from the module tree, not the graph.

    `lineage_keys` asks the graph who produced each input, and at TP>1 the
    graph frequently does not know. In the derived 27B graphs 725 of 4378
    source edges are -1 at TP=2 and TP=4 against 596 at TP=1, because
    `_storage_of` collapses on meta tensors (O14) and the collective stand-in
    returns its own input (O15). An unknown source breaks the ancestry chain
    and every descendant inherits the break, which is why ancestry alone
    aligns 12 of 3014 outputs across the three real widths.

    A module path needs the graph to know nothing. It is recorded at trace
    time by the module hooks, the module tree is the same tree at every width
    -- only the shard sizes inside it change -- and an operator's ordinal
    among the *non-collective* operators of its own module separates siblings
    without a global index that an inserted collective would shift.

    This is ordinal alignment inside a module, and ordinal alignment is the
    thing `lineage_keys` was written to avoid. So it is not to be used
    unchecked: `alignment_integrity` is the check, and `width_classes` refuses
    these keys when that check fails.
    """
    ops = graph.get("ops") or ()
    if len(module_paths) != len(ops):
        raise ValueError("a module path is needed for every operator: got %d "
                         "paths for %d operators"
                         % (len(module_paths), len(ops)))
    keys: list = []
    seen: dict = {}
    for index, op in enumerate(ops):
        if _is_collective(op):
            # A collective consumes no ordinal, or the operator after it would
            # be renumbered at exactly the widths where it appears.
            sources = tuple(op.get("inputs_from") or ())
            keys.append(next((keys[s] for s in sources
                              if 0 <= s < len(keys)), "external"))
            continue
        path = module_paths[index] or ""
        ordinal = seen.get(path, 0)
        seen[path] = ordinal + 1
        keys.append(hashlib.blake2b(repr((path, ordinal)).encode("utf-8"),
                                    digest_size=8).hexdigest())
    return keys


def _named_externals(resolved, row) -> dict:
    """The recorded identity of each input no operator in the graph produced.

    Keyed by argument position, and "" where the trace has no name for it --
    an input whose producer the tracer lost reads -1 exactly as a weight does,
    and the two must not be allowed to look alike here.
    """
    outside = {}
    for position, source in enumerate(resolved):
        if source >= 0:
            continue
        outside[position] = (row[position] if position < len(row) else "")
    return outside


def _origin_verdict(outside: Mapping, widths, keyed: bool) -> str:
    """Whether the named externals of one aligned operator match across widths.

    Only positions unreadable at *every* width are compared: a position
    readable at one width and not another says something about the tracer's
    coverage, not about whether the two operators correspond, and an unnamed
    input says nothing at all.

    `keyed` is whether the operator has the same name at every width. When it
    does, the argument position means the same thing on both sides and the
    names are compared position by position. When it does not, the position
    means nothing across the pair and only the set of names is comparable:
    `VocabParallelEmbedding` calls `F.embedding(weight, ids)` at TP=1 and
    `masked_embedding(ids, weight)` at TP>1 (`embed_head.py:168-178`), which is
    the same two externals in the other order and is not evidence of a
    misalignment.
    """
    if len(outside) != len(widths):
        return "unresolved"
    if not keyed:
        if any(not name for row in outside.values() for name in row.values()):
            return "unresolved"
        distinct = {tuple(sorted(row.values())) for row in outside.values()}
        return "agrees" if len(distinct) == 1 else "contradicts"
    common = set.intersection(*(set(row) for row in outside.values()))
    named = [p for p in common if all(outside[w].get(p) for w in outside)]
    if not named:
        return "unresolved"
    first = widths[0]
    for position in named:
        if any(outside[w][position] != outside[first][position]
               for w in outside):
            return "contradicts"
    return "agrees" if len(named) == len(common) else "unresolved"


def alignment_integrity(graphs: Mapping, module_paths: Mapping,
                        input_origins: Optional[Mapping] = None) -> dict:
    """Whether a module-path alignment may be read at these widths.

    Structural checks, none of them fitted to any measured quantity:

    * every module path holds the same number of non-collective operators at
      every width. A path whose count differs is a module that did different
      work at width, and its ordinals then mean different things;
    * where the ancestry survives at *every* width -- no unknown source on
      either side, collectives walked through -- it must agree with the
      alignment the ordinals give. Ancestry that is present is evidence, and
      evidence that contradicts the alignment ends it;
    * where it does not survive, `input_origins` -- one recorded identity per
      input, as `capture_lifetimes.py` writes beside the graph -- can still be
      compared. An operator reading `layers.7.mlp.down_proj.weight` aligned
      against one reading the same parameter at another width is evidence of
      correspondence; one reading a different parameter is evidence against.

    Counting discipline, because these three are not the same claim:

    * `ancestry_agrees` / `ancestry_contradicts` -- the ancestry ran;
    * `origin_agrees` / `origin_contradicts` -- the ancestry could not run and
      the named externals were compared instead. Weaker: it says the aligned
      operators read the same outside tensors, not that they sit at the same
      place in the graph;
    * `ancestry_unknown` counts every case the ancestry could not examine, and
      `origin_unresolved` the ones that neither check reached. A check that
      cannot run is not a check that passed, and an alignment resting on the
      ordinals alone stays labelled as resting on the ordinals alone.
    """
    widths = sorted(int(w) for w in graphs)
    if len(widths) < 2:
        raise ValueError("alignment integrity needs graphs at two or more "
                         "tensor-parallel widths; got %r" % (widths,))

    def pick(mapping, width):
        return mapping[width] if width in mapping else mapping[str(width)]

    ops = {w: (pick(graphs, w).get("ops") or ()) for w in widths}
    paths = {w: pick(module_paths, w) for w in widths}
    for width in widths:
        if len(paths[width]) != len(ops[width]):
            raise ValueError("width %d has %d operators and %d module paths"
                             % (width, len(ops[width]), len(paths[width])))
    positions = {w: _non_collective_positions(ops[w]) for w in widths}

    counts = {}
    for width in widths:
        table: dict = {}
        for index in positions[width]:
            path = paths[width][index] or ""
            table[path] = table.get(path, 0) + 1
        counts[width] = table
    every_path = set()
    for table in counts.values():
        every_path |= set(table)
    disagreeing = sorted(path for path in every_path
                         if len({counts[w].get(path, 0) for w in widths}) > 1)

    origins = None
    if input_origins is not None:
        origins = {w: pick(input_origins, w) for w in widths}
        for width in widths:
            if len(origins[width]) != len(ops[width]):
                raise ValueError("width %d has %d operators and %d origin rows"
                                 % (width, len(ops[width]),
                                    len(origins[width])))

    base = widths[0]
    agrees = contradicts = unknown = 0
    named = misnamed = unresolved = 0
    lengths = {w: len(positions[w]) for w in widths}
    if len(set(lengths.values())) == 1 and not disagreeing:
        rank = {w: {index: n for n, index in enumerate(positions[w])}
                for w in widths}
        for n in range(lengths[base]):
            chains = {}
            outside = {}
            for width in widths:
                index = positions[width][n]
                sources = list(ops[width][index].get("inputs_from") or ())
                resolved = [_resolve_through_collectives(ops[width], s)
                            for s in sources]
                if any(s < 0 for s in resolved):
                    outside[width] = _named_externals(
                        resolved,
                        origins[width][index] if origins is not None else ())
                else:
                    chains[width] = [rank[width].get(s) for s in resolved]
            if outside:
                unknown += 1
                if origins is None:
                    continue
                keyed = len({ops[w][positions[w][n]].get("name")
                             for w in widths}) == 1
                verdict = _origin_verdict(outside, widths, keyed)
                named += verdict == "agrees"
                misnamed += verdict == "contradicts"
                unresolved += verdict == "unresolved"
            elif len({tuple(c) for c in chains.values()}) == 1:
                agrees += 1
            else:
                contradicts += 1

    safe = (not disagreeing and len(set(lengths.values())) == 1
            and contradicts == 0 and misnamed == 0)
    return {"widths": widths, "operators_per_width": lengths,
            "paths_disagreeing": disagreeing,
            "ancestry_agrees": agrees,
            "ancestry_contradicts": contradicts,
            "ancestry_unknown": unknown,
            "origin_agrees": named,
            "origin_contradicts": misnamed,
            "origin_unresolved": unresolved,
            "safe": safe}


def width_classes(graphs: Mapping, module_paths: Mapping = None,
                  input_origins: Mapping = None) -> dict:
    """How each tensor's shape actually behaves with width, read off the graphs.

    `graphs` maps a tensor-parallel width to a graph derived at that width. For
    every output that can be aligned across all of them by `lineage_keys`, this
    reports what the widths did to its shape:

    * `"replicated"` -- the same shape at every width;
    * `"sharded"` -- exactly one axis divides by the width ratio, the rest
      unchanged, and the axis is named;
    * `"unresolved"` -- anything else, including a shape that changes by a
      ratio the width does not explain. Reported, not classified.

    This is the check that the trailing-dimension rule could not make. At TP=1
    an attention output is `[tokens, heads * head_dim]`, and `heads * head_dim`
    *is* the hidden size, so it is indistinguishable by shape from the residual
    stream -- and it shards while the residual does not. Reading the width
    behaviour from graphs derived at each width asks ATOM's own sharding
    arithmetic instead of guessing from one width's shape, which is a source
    derivation and not a fit to any measured peak.

    Outputs that exist at one width and not another -- a collective's own
    result, an all-gather's widened tensor -- are absent from the result rather
    than guessed at; `width_coverage` says how many those were.

    `module_paths` is optional and maps each width to one module path per
    operator, as `capture_lifetimes.py` writes beside the graph. Given it,
    alignment comes from `module_path_keys` rather than `lineage_keys`, and is
    refused outright unless `alignment_integrity` passes. On the real graphs
    that is the only alignment available at all: ancestry reaches 12 of 3014
    outputs there, because the tracer records an unknown producer for a sixth
    of the source edges at TP>1 (O14, O15). An entry whose operator *name*
    differs across widths keeps a `names` field, so a join the module tree
    licenses but the names do not is visible instead of silent.
    """
    widths = sorted(int(w) for w in graphs)
    if len(widths) < 2:
        raise ValueError("width classification needs graphs at two or more "
                         "tensor-parallel widths; got %r" % (widths,))
    if module_paths is not None:
        integrity = alignment_integrity(graphs, module_paths, input_origins)
        if not integrity["safe"]:
            raise ValueError(
                "module-path alignment is not safe to read at these widths: "
                "%r" % (integrity,))

    base = widths[0]
    per_width = {}
    for width in widths:
        graph = graphs[width] if width in graphs else graphs[str(width)]
        if module_paths is None:
            keys = lineage_keys(graph)
        else:
            paths = (module_paths[width] if width in module_paths
                     else module_paths[str(width)])
            keys = module_path_keys(graph, paths)
        table = {}
        for index, op in enumerate(graph.get("ops") or ()):
            if _is_collective(op):
                continue
            for position, shape in enumerate(op.get("output_shapes") or ()):
                table[(keys[index], position)] = (op.get("name"),
                                                  tuple(int(d) for d in shape))
        per_width[width] = table

    out = {}
    for ident, (name, base_shape) in per_width[base].items():
        shapes = {base: list(base_shape)}
        missing = False
        for width in widths[1:]:
            entry = per_width[width].get(ident)
            if entry is None:
                missing = True
                break
            shapes[width] = list(entry[1])
        if missing:
            continue
        classification, axis = _classify_shapes(shapes, base)
        entry = {"name": name, "position": ident[1], "shapes": shapes,
                 "class": classification, "axis": axis}
        # Two widths can run the same module through different operators --
        # `VocabParallelEmbedding` emits `aten::embedding` at TP=1 and
        # `aiter::masked_embedding` above it. A module-path alignment is
        # right to join those, and wrong to do it silently.
        names = {w: per_width[w][ident][0] for w in widths}
        if len(set(names.values())) > 1:
            entry["names"] = names
        out[ident] = entry
    return out


def _classify_shapes(shapes: Mapping, base: int) -> tuple:
    base_shape = shapes[base]
    if all(list(s) == list(base_shape) for s in shapes.values()):
        return "replicated", None
    for axis, extent in enumerate(base_shape):
        ok = True
        for width, shape in shapes.items():
            if len(shape) != len(base_shape):
                ok = False
                break
            ratio = width // base
            for other, size in enumerate(shape):
                want = (extent // ratio if other == axis
                        else base_shape[other])
                if size != want or (other == axis and extent % ratio):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return "sharded", axis
    return "unresolved", None


def width_coverage(graphs: Mapping, module_paths: Mapping = None,
                   input_origins: Mapping = None) -> dict:
    """How much of each graph the width classification could align at all.

    A classification that quietly drops half the graph is worse than no
    classification, so the count is reported next to it: outputs aligned,
    outputs present only at a wider width (a collective's own result), and
    outputs at the base width with no counterpart.

    `aligned` is coverage, not correspondence. It counts the outputs the
    ordinals paired up; whether each pair is really the same operator is what
    `integrity` reports, and it reports it in grades -- ancestry checked, named
    externals checked, neither -- because those are different strengths of
    evidence and collapsing them into one number reads as a verification that
    did not happen.
    """
    widths = sorted(int(w) for w in graphs)
    classes = width_classes(graphs, module_paths, input_origins)
    counts = {}
    for width in widths:
        graph = graphs[width] if width in graphs else graphs[str(width)]
        counts[width] = sum(
            len(op.get("output_shapes") or ())
            for op in (graph.get("ops") or ()) if not _is_collective(op))
    out = {"aligned": len(classes),
            "outputs_per_width": counts,
            "renamed": sum(1 for v in classes.values() if "names" in v),
            "unaligned_at_base": counts[widths[0]] - len(classes),
            "by_class": {name: sum(1 for v in classes.values()
                                   if v["class"] == name)
                         for name in ("replicated", "sharded", "unresolved")}}
    if module_paths is not None:
        out["integrity"] = alignment_integrity(graphs, module_paths,
                                               input_origins)
    return out
