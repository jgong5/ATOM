"""Is one layer's KV view the same tensor whether the pool holds 1 layer or 16?

`aiter_attention.build_kv_cache_tensor` hands a module a *view* into one slot of
a shared pool:

    k_cache = runner.kv_cache[0, attn_idx].view(blocks, heads, head_dim // x,
                                                block_size, x)

Pricing times one attention operator at a time, and during that operator only
its own layer's slot is read -- the other fifteen are allocated and never
touched. If a pool sized for one layer produces a view of identical shape,
stride and contiguity, then materialising per layer is a sixteen-fold reduction
that changes nothing the timed call can observe about its own memory.

That "if" is the whole question, so it is tested rather than asserted. These
run on CPU tensors: the geometry is a property of the shape arithmetic, not of
the device.
"""

from __future__ import annotations

import torch

# Qwen3.8-27B at TP1, read off the graphs rather than assumed: the attention
# operands are 32,6144;32,1024;32,1024, so 4 KV heads of head_dim 256, and the
# block arithmetic gives block_size 16 over 16 full-attention layers.
N_FULL = 16
BLOCKS = 8
BLOCK_SIZE = 16
KV_HEADS = 4
HEAD_DIM = 256
DTYPE = torch.bfloat16
X = 16 // DTYPE.itemsize


def pool(n_full: int) -> torch.Tensor:
    """The pool `allocate_kv_cache_tensors` builds, at a given layer count."""
    return torch.zeros(2, n_full, BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM,
                       dtype=DTYPE)


def layer_view(kv: torch.Tensor, half: int, attn_idx: int) -> torch.Tensor:
    """The per-module view, as build_kv_cache_tensor takes it."""
    return kv[half, attn_idx].view(BLOCKS, KV_HEADS, HEAD_DIM // X,
                                   BLOCK_SIZE, X)


def test_a_layer_slice_has_the_same_geometry_in_a_small_pool_as_a_large_one():
    """Shape, stride and contiguity, for every layer of the full pool."""
    small = layer_view(pool(1), 0, 0)
    big_pool = pool(N_FULL)
    for attn_idx in range(N_FULL):
        big = layer_view(big_pool, 0, attn_idx)
        assert big.shape == small.shape, attn_idx
        assert big.stride() == small.stride(), attn_idx
        assert big.is_contiguous() == small.is_contiguous(), attn_idx


def test_the_k_and_v_halves_stay_distinct_storage_in_either_pool():
    """k and v are separate halves; a reduction must not alias them together."""
    for n_full in (1, N_FULL):
        kv = pool(n_full)
        k = layer_view(kv, 0, 0)
        v = layer_view(kv, 1, 0)
        assert k.data_ptr() != v.data_ptr()
        # Distinct in the sense that matters: writing one does not move the
        # other.
        k.fill_(1.0)
        assert float(v.abs().sum()) == 0.0


def test_a_layer_slice_is_a_contiguous_run_so_the_other_layers_are_dead_weight():
    """The layer axis is outer, so one layer's bytes are one unbroken run.

    This is what makes the reduction sound: slot `attn_idx` does not interleave
    with its neighbours, so removing them changes no offset *within* the slot.
    """
    kv = pool(N_FULL)
    per_layer_elements = BLOCKS * BLOCK_SIZE * KV_HEADS * HEAD_DIM
    for attn_idx in range(N_FULL):
        view = kv[0, attn_idx]
        assert view.is_contiguous()
        assert view.numel() == per_layer_elements
        offset = view.storage_offset() - kv[0, 0].storage_offset()
        assert offset == attn_idx * per_layer_elements


def test_the_pool_scales_with_the_layer_count_and_nothing_else():
    """Sixteen times smaller, which is the point of doing it at all."""
    one = pool(1).numel() * pool(1).element_size()
    many = pool(N_FULL).numel() * pool(N_FULL).element_size()
    assert many == one * N_FULL


def test_block_indices_within_a_layer_are_untouched_by_the_reduction():
    """A call walks block indices inside its own slot; those must not move.

    Writing through block `b` of the small pool and of the corresponding slot
    of the large pool must land at the same offset relative to the slot.
    """
    small, big = pool(1), pool(N_FULL)
    for attn_idx in (0, 7, N_FULL - 1):
        for block in (0, 3, BLOCKS - 1):
            s = layer_view(small, 0, 0)[block]
            b = layer_view(big, 0, attn_idx)[block]
            assert s.shape == b.shape
            s_off = s.storage_offset() - layer_view(small, 0, 0).storage_offset()
            b_off = b.storage_offset() - layer_view(big, 0, attn_idx).storage_offset()
            assert s_off == b_off
