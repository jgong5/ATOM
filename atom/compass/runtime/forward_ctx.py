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


def _kv_blocks() -> int:
    """How many blocks the KV cache holds, or 0 if it is not reachable.

    Read from the persistent KV context rather than the live forward one. The
    cache is installed once at startup and kept separately, which is exactly why
    a benchmark can reset the forward context and still find it -- and why
    reading it from the live context returns nothing right after that reset.
    """
    from atom.utils import forward_context as fc

    holders = [getattr(fc, "_forward_kv_cache_context", None),
               fc.get_forward_context()]
    for holder in holders:
        for entry in (getattr(holder, "kv_cache_data", None) or {}).values():
            cache = getattr(entry, "k_cache", None)
            if cache is not None and cache.dim() >= 1 and cache.shape[0] > 0:
                return int(cache.shape[0])
    return 0


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
    if stride and variants > 1:
        blocks = _kv_blocks()
        variants = max(1, min(variants, blocks // stride)) if blocks else 1
    else:
        variants = 1

    thunks = []
    for v in range(variants):
        table = None
        if shape is not None:
            table = torch.zeros(tuple(shape), dtype=torch.int32, device="cuda")
            used = len(flat) // max(shape[0], 1)
            if used:
                table[:, :used] = torch.tensor(
                    [x + v * stride for x in flat],
                    dtype=torch.int32, device="cuda").reshape(shape[0], used)

        metadata = AttentionMetaData(
            block_tables=table,
            context_lens=tensor(recorded.get("context_lens"), torch.int32),
            slot_mapping=tensor(
                None if slots is None
                else [x + v * stride * block_size for x in slots], torch.int64),
            cu_seqlens_q=tensor(recorded.get("cu_seqlens_q"), torch.int32),
            cu_seqlens_k=tensor(recorded.get("cu_seqlens_k"), torch.int32),
            max_seqlen_q=int(recorded.get("max_seqlen_q", 0)),
            max_seqlen_k=int(recorded.get("max_seqlen_k", 0)),
            min_seqlen_q=int(recorded.get("min_seqlen_q", 0)),
            has_cached=bool(recorded.get("has_cached", False)),
            state=AttnState(recorded.get("state", AttnState.DECODE.value)),
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


_CAPTURE = {_ATTENTION: _capture_attention}
_INSTALL = {_ATTENTION: _install_attention}


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
