"""Ambient state an operator reads that its arguments do not describe.

Most operators are functions of their arguments, so a recorded ``(name, shapes,
dtypes, scalars)`` is enough to call one again and find out what it costs.
Attention is not. It reads its metadata -- which blocks of KV cache to walk, how
long each sequence is -- from a module-global forward context, and its arguments
say nothing about any of it.

The obvious repair was to give the operator those arguments and let it stand up
its own context when there is no live forward. That was tried, in
``c337ee3a``, and it does not work: ``torch.compile`` traces
``md = get_forward_context().attn_metadata`` and only the *tensor* reads survive
as graph inputs. ``md.max_seqlen_q`` is an int, so it is constant-folded, and
``md.block_tables`` was ``None`` in the forward that compiled -- the warmup
dummy -- so it is baked in as the constant ``None``. The recorded call then
claims ``block_tables=None, max_seqlen_q=16384`` where the live step has a
``(4, 2560)`` tensor and ``1``. Production never notices, because it ignores
those arguments and reads the context; a benchmark that honours them attends the
wrong thing or crashes.

So the context is recorded alongside the operator instead of through it. The
tracer reads it from the live forward, where it is true, and the benchmark
installs it before calling. The operator is untouched and stays exactly what
production runs.

The cost of this design is that Compass has to know something about attention
specifically, which the rest of the op graph avoids. It is confined to this
module, and to operators that genuinely read ambient state -- currently one.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["capture", "install", "is_context_dependent"]

#: Operators whose cost depends on state their arguments do not carry.
_ATTENTION = "aiter::unified_attention_with_output_base"
_LINEAR_ATTENTION = "aiter::linear_attention_with_output_base"

#: The GDN metadata fields the implementation reads, tensors first. Recorded by
#: name so a field added upstream is absent rather than wrong.
_GDN_TENSORS = (
    "has_initial_state", "spec_query_start_loc", "non_spec_query_start_loc",
    "spec_state_indices_tensor", "non_spec_state_indices_tensor",
    "non_spec_state_indices_in_tensor", "spec_sequence_masks",
    "spec_token_indx", "non_spec_token_indx", "num_accepted_tokens",
)
_GDN_COUNTS = (
    "num_prefills", "num_prefill_tokens", "num_decodes", "num_decode_tokens",
    "num_spec_decodes", "num_spec_decode_tokens", "num_actual_tokens",
)


def _values(tensor) -> list[int] | None:
    if tensor is None:
        return None
    return [int(x) for x in tensor.flatten().tolist()]


def _capture_attention() -> tuple[tuple[str, Any], ...]:
    """Attention's metadata, as the live forward has it.

    ``block_tables`` is recorded by its full shape but only the columns the
    kernel reads. The full width is ``max_model_len / block_size`` -- 2560 here
    -- and is part of the call because it is the row stride the kernel indexes
    with, but a decode at 315 tokens of context touches 20 columns of it. Keeping
    the used prefix and the shape reproduces both the addressing and the traffic
    without an artifact that grows with the model's maximum context.
    """
    from atom.config import get_current_atom_config
    from atom.utils.forward_context import get_forward_context

    fwd = get_forward_context()
    md = getattr(fwd, "attn_metadata", None)
    if md is None or fwd.context is None or md.context_lens is None:
        return ()

    recorded: list[tuple[str, Any]] = [
        ("context_lens", _values(md.context_lens)),
        ("slot_mapping", _values(md.slot_mapping)),
        ("cu_seqlens_q", _values(md.cu_seqlens_q)),
        # Prefill reads more of the metadata than decode does. The varlen fmha
        # kernel wants both cumulative-length arrays and all three extents, and
        # raises on a missing one rather than defaulting it -- recording only
        # what decode needed left every prefill attention unpriced, which is the
        # whole of prefill's attention cost.
        ("cu_seqlens_k", _values(md.cu_seqlens_k)),
        ("max_seqlen_q", int(md.max_seqlen_q)),
        ("max_seqlen_k", int(md.max_seqlen_k)),
        ("min_seqlen_q", int(md.min_seqlen_q)),
        ("has_cached", bool(md.has_cached)),
        ("state", md.state.value),
        ("is_prefill", bool(fwd.context.is_prefill)),
        ("positions", _values(fwd.context.positions)),
    ]

    # Chunked prefill reads three more fields than any other path. The prefix
    # gather packs cached and new KV into one dense tensor, taking its size
    # from `total_kv` and its per-sequence offsets from `seq_starts` -- neither
    # recoverable from the shapes, since the query length is the *new* tokens
    # and the gather is over cached+new. Without them the gather allocates
    # `torch.empty((None, ...))` and every chunked attention goes unpriced.
    #
    # Recorded only under `has_cached`, which is where they are populated and
    # read. `total_kv` is set on the unchunked path too, but nothing there
    # reads it, and recording it unconditionally would rewrite the signature of
    # every attention call already priced for a value that changes no cost.
    if md.has_cached:
        recorded += [
            ("total_kv", int(md.total_kv or 0)),
            ("seq_starts", _values(md.seq_starts)),
            ("num_cached_tokens", _values(md.num_cached_tokens)),
        ]

    table = md.block_tables
    if table is not None and table.dim() == 2:
        block_size = int(get_current_atom_config().kv_cache_block_size)
        longest = max(_values(md.context_lens) or [0])
        used = min(table.shape[1], -(-longest // max(block_size, 1)) or 1)
        recorded += [
            ("block_tables_shape", [int(d) for d in table.shape]),
            ("block_tables", _values(table[:, :used])),
        ]
    return tuple(recorded)


#: A paged KV cache is block-major and five-dimensional:
#: ``(blocks, kv_heads, block_size, head_dim, packing)``. A hybrid model's
#: linear-attention state is not -- see :func:`_kv_blocks`.
_PAGED_KV_RANK = 5


def _kv_blocks() -> int:
    """How many blocks the PAGED KV cache holds, or 0 if it is not reachable.

    Read from the persistent KV context rather than the live forward one. The
    cache is installed once at startup and kept separately, which is exactly why
    a benchmark can reset the forward context and still find it -- and why
    reading it from the live context returns nothing right after that reset.

    **Paged only.** A hybrid model's ``kv_cache_data`` holds two kinds of entry
    under the same layer keys:

        layer_0  k=(32, 3, 10240)             linear-attention state
        layer_3  k=(147456, 4, 32, 16, 8)     paged KV

    and the leading dimension means different things in the two: blocks in the
    paged pool, **sequences** in the state. Returning the first entry found
    returned 32 -- a state cache's sequence count -- for a pool of 147,456
    blocks. `_install_attention` then computed ``blocks // stride`` as 0 and
    collapsed the rotation to a single region at every requested count, with
    nothing said. Every standalone price for this model was therefore timed
    against one region however many were allocated, which is what the recorded
    ``kv_regions: 1`` has been reporting all along.

    So the rank is the discriminator, and the largest paged pool wins: there is
    one per full-attention layer and they are the same size, but taking the max
    is stable if that ever stops being true.
    """
    from atom.utils import forward_context as fc

    holders = [getattr(fc, "_forward_kv_cache_context", None),
               fc.get_forward_context()]
    for holder in holders:
        blocks = 0
        for entry in (getattr(holder, "kv_cache_data", None) or {}).values():
            cache = getattr(entry, "k_cache", None)
            if (cache is not None and cache.dim() >= _PAGED_KV_RANK
                    and cache.shape[0] > 0):
                blocks = max(blocks, int(cache.shape[0]))
        if blocks:
            return blocks
    return 0


def shift_addresses(values, offset: int):
    """Move real addresses into another KV region, leaving sentinels alone.

    A padded batch does not carry an address in every position. A bucketed
    decode rounds three real rows up to a bucket of four, and the fourth row's
    ``slot_mapping`` entry is ``-1``: not a slot, a statement that there is no
    slot. The kernel tests for the negative and skips the write.

    Adding a region offset to that ``-1`` turns it into a perfectly valid
    address -- and specifically into an address in the *previous* region, which
    is the one region guaranteed to hold live data from the call before. The
    padding row then stops being skipped and writes over another row's KV. The
    benchmark reports a time for work that is not the work being priced, and a
    correctness check on the output would not obviously fail, because the
    damage lands in a region the current call does not read.

    So the offset applies to addresses only. Negative entries are sentinels and
    are carried through unchanged, which is also what makes variant 0 --
    offset 0 -- exactly the recorded call.
    """
    return [v if v < 0 else v + offset for v in values]


def _install_attention(recorded: dict[str, Any], variants: int) -> list:
    """One installer per distinct KV region the captured batch should touch.

    A benchmark that calls attention repeatedly calls it against *one* layer's
    KV cache, so after the first call the whole working set is resident and every
    kernel that touches it -- the attention read, the RoPE-and-cache write, the
    reshape-and-cache write -- runs against warm memory. A decode step does not
    work that way: it touches each of 28 layers' regions once, separated by the
    other ~300 kernels of a model far larger than the last level of cache. That
    difference showed up as a uniform 15-30% discount across all three kernels.

    So each captured call is given its own slice of the KV cache, offset by the
    blocks the recorded call used. Rotating with period ``variants`` means a
    region is revisited only after every other one has been walked, which is what
    evicts it -- eviction reproduced rather than simulated, and at no cost, where
    scrubbing a 256 MB cache between 18 us calls would swamp what it measures.

    Variant 0 is the recorded call exactly, so a single-variant run is unchanged.
    """
    import torch

    from atom.config import get_current_atom_config
    from atom.utils.forward_context import (
        AttentionMetaData,
        AttnState,
        Context,
        set_forward_context,
    )

    def tensor(values, dtype):
        if values is None:
            return None
        return torch.tensor(values, dtype=dtype, device="cuda")

    shape = recorded.get("block_tables_shape")
    flat = recorded.get("block_tables") or []
    slots = recorded.get("slot_mapping")
    block_size = int(get_current_atom_config().kv_cache_block_size)

    # One variant's footprint, in blocks: the recorded call's own, so successive
    # variants are disjoint. Bounded by what the cache actually holds -- asking
    # for regions past the end would index out of the allocation.
    stride = (max(flat) + 1) if flat else 0
    requested = variants
    if stride and variants > 1:
        blocks = _kv_blocks()
        variants = max(1, min(variants, blocks // stride)) if blocks else 1
    else:
        variants = 1
    if requested > 1 and variants < requested:
        # Say it. A caller asked for N cold regions and is getting fewer, and
        # the only previous evidence was a `kv_regions` count in the artifact
        # that nobody was reading as a shortfall. A price measured over one
        # region while the caller believed it was rotating over sixty-four is
        # a warm price wearing a cold label.
        logger.warning(
            "ATOMCompass WARNING: %d KV regions requested, %d available -- "
            "the pool "
            "holds %d blocks and this operator's footprint is %d, so the "
            "rotation is %s. Prices from this run are over %d region(s), not "
            "%d.", requested, variants, _kv_blocks(), stride,
            "disabled" if variants == 1 else "reduced", variants, requested)

    thunks = []
    for v in range(variants):
        table = None
        if shape is not None:
            table = torch.zeros(tuple(shape), dtype=torch.int32, device="cuda")
            used = len(flat) // max(shape[0], 1)
            if used:
                table[:, :used] = torch.tensor(
                    shift_addresses(flat, v * stride),
                    dtype=torch.int32, device="cuda").reshape(shape[0], used)

        metadata = AttentionMetaData(
            block_tables=table,
            context_lens=tensor(recorded.get("context_lens"), torch.int32),
            slot_mapping=tensor(
                None if slots is None
                else shift_addresses(slots, v * stride * block_size),
                torch.int64),
            cu_seqlens_q=tensor(recorded.get("cu_seqlens_q"), torch.int32),
            cu_seqlens_k=tensor(recorded.get("cu_seqlens_k"), torch.int32),
            max_seqlen_q=int(recorded.get("max_seqlen_q", 0)),
            max_seqlen_k=int(recorded.get("max_seqlen_k", 0)),
            min_seqlen_q=int(recorded.get("min_seqlen_q", 0)),
            has_cached=bool(recorded.get("has_cached", False)),
            state=AttnState(recorded.get("state", AttnState.DECODE.value)),
            # Absent for every path but chunked prefill, where they decide the
            # size of the gathered KV rather than merely describing it.
            total_kv=(None if recorded.get("total_kv") is None
                      else int(recorded["total_kv"])),
            seq_starts=tensor(recorded.get("seq_starts"), torch.int32),
            num_cached_tokens=tensor(
                recorded.get("num_cached_tokens"), torch.int32),
        )
        context = Context(
            positions=tensor(recorded.get("positions"), torch.int64),
            is_prefill=bool(recorded.get("is_prefill", False)),
        )

        def install_this(metadata=metadata, context=context):
            set_forward_context(
                attn_metadata=metadata,
                atom_config=get_current_atom_config(),
                context=context,
            )

        thunks.append(install_this)
    return thunks


def _capture_linear_attention() -> tuple[tuple[str, Any], ...]:
    """The GDN metadata a linear-attention layer reads.

    Without it `attention_gdn.py` zeroes its output and returns, so the
    benchmark timed 48 `zero_()` calls and priced the DeltaNet half of a hybrid
    model at 1% of its cost. Unlike attention's, this metadata is not on the
    forward context's declared fields -- the backend attaches it to the
    attention metadata as an attribute -- so it is read the same way the
    implementation reads it.

    The recurrent and convolution state it walks is *not* recorded: those live
    in `kv_cache_data`, which the engine sets once at start-up and which is
    therefore already real in the process doing the pricing. Recording them
    would mean carrying a per-layer cache in a JSON artifact to rebuild
    something that is already there.
    """
    import torch

    from atom.utils.forward_context import get_forward_context

    fwd = get_forward_context()
    md = getattr(fwd, "attn_metadata", None)
    gdn = getattr(md, "gdn_metadata", None) if md is not None else None
    if gdn is None:
        return ()

    recorded: list[tuple[str, Any]] = [
        (name, int(getattr(gdn, name, 0) or 0)) for name in _GDN_COUNTS
    ]
    recorded.append(("replayssm", bool(getattr(gdn, "replayssm", False))))
    for name in _GDN_TENSORS:
        value = getattr(gdn, name, None)
        if not isinstance(value, torch.Tensor):
            continue
        # Dtype travels with the values: these are a mix of index tensors and
        # boolean masks, and a mask rebuilt as int32 selects nothing.
        recorded.append((name, [_values(value),
                                str(value.dtype).replace("torch.", "")]))
    return tuple(recorded)


def _install_linear_attention(recorded: dict[str, Any], variants: int) -> list:
    """Stand up the GDN metadata, on an attention metadata to hang it off."""
    import torch

    from atom.config import get_current_atom_config
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
    )

    fields = {name: int(recorded.get(name, 0) or 0) for name in _GDN_COUNTS}
    for name in _GDN_TENSORS:
        held = recorded.get(name)
        if not held:
            continue
        values, dtype_name = held
        dtype = getattr(torch, dtype_name, None)
        if dtype is None or values is None:
            continue
        fields[name] = torch.tensor(values, dtype=dtype, device="cuda")

    # The convolution's own metadata -- `nums_dict`, `batch_ptr`,
    # `token_chunk_offset_ptr` -- is a pure function of the query start
    # offsets, and `causal_conv1d_fn` is handed the GDN metadata as its
    # metadata, so it reads them straight off it. Recomputing them with the
    # engine's own helper is exact where recording them would be a copy, and it
    # is the difference between the operator running and dying on
    # `batch_ptr.device` with `batch_ptr` None.
    starts = fields.get("non_spec_query_start_loc")
    if starts is not None:
        from atom.model_ops.attentions.gdn_attn import (
            compute_causal_conv1d_metadata,
        )

        (fields["nums_dict"], fields["batch_ptr"],
         fields["token_chunk_offset_ptr"]) = compute_causal_conv1d_metadata(
            starts)

    gdn = GDNAttentionMetadata(**fields)
    if recorded.get("replayssm"):
        gdn.replayssm = True
    metadata = AttentionMetaData()
    metadata.gdn_metadata = gdn

    # The GDN path reads its metadata and the caches, not the positions, but a
    # forward context is not constructible without them.
    positions = torch.arange(max(1, fields.get("num_actual_tokens", 1)),
                             dtype=torch.int64, device="cuda")

    def install_this(metadata=metadata, positions=positions):
        set_forward_context(
            attn_metadata=metadata,
            atom_config=get_current_atom_config(),
            context=Context(positions=positions, is_prefill=True),
        )

    return [install_this]


_CAPTURE = {_ATTENTION: _capture_attention,
            _LINEAR_ATTENTION: _capture_linear_attention}
_INSTALL = {_ATTENTION: _install_attention,
            _LINEAR_ATTENTION: _install_linear_attention}


def is_context_dependent(name: str) -> bool:
    """Whether ``name`` cannot be called from its arguments alone."""
    return name in _INSTALL


def capture(name: str) -> tuple[tuple[str, Any], ...]:
    """What the live forward context holds for ``name``, or ``()``.

    Each read is a device-to-host copy, so this belongs to trace mode, which
    already runs eagerly and exists to produce an artifact rather than to serve.
    """
    recorder = _CAPTURE.get(name)
    if recorder is None:
        return ()
    try:
        return recorder()
    except Exception as exc:  # noqa: BLE001 - never fail a trace over this
        logger.warning("ATOMCompass WARNING: could not record the forward "
                       "context for %s (%s); it will be unpriceable", name, exc)
        return ()


def install(name: str, recorded, variants: int = 1) -> list:
    """Stand up the forward context(s) ``name`` was recorded with.

    Returns a list of thunks, one per distinct cache footprint the captured
    batch should rotate over, or an empty list if the operator cannot be given
    the context it needs. Empty means the caller must report it unpriced rather
    than price it against whatever context happened to be installed, which is the
    failure this module exists to prevent.

    The first thunk installs the recorded call exactly, so ``variants=1`` is the
    faithful single-shot case and anything above it trades exactness for a cache
    state that resembles a real step's.
    """
    installer = _INSTALL.get(name)
    if installer is None or not recorded:
        return []
    return installer({k: v for k, v in (tuple(x) for x in recorded)},
                     max(1, int(variants)))
