# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A captured leaf's work, by op kind: the op-level extractor registry.

A leaf is one operator of an `op_graph` as its JSON holds it: `op`, its kind;
`args` by schema slot, a tensor as its shape, dtype, stride and role; `out`,
its output shapes. `work(leaf, view, tp, kv_format)` is the leaf's `Work` on a
step of `view`'s rows: F and C from the primitives its kind's extractor emits,
M the leaf's own tensor arguments read once and its outputs written once, plus
the ambient state it reads (the KV context in `kv_format`, the recurrent
state). A fused leaf is
one leaf, so its intermediates cost nothing. A tensor counts its distinct
elements, a broadcast dimension (stride 0) once. The leaf's own tensors are
priced at the shape it was captured at, padding as executed; the ambient state
at the view's rows.

The work reads nothing `bench.ops.signature` leaves out of a leaf's identity,
so one call prices every instance of a signature at a view.

Aten operators share one extractor, whose F is `torch.utils.flop_counter`'s
formula where it has one. A leaf that launches no kernel (a view, an
allocation, a wait, one whose outputs hold no element) has no work: None. A collective spans the `tp` ranks of
the tensor-parallel group unless the leaf states its group size. A kind with
no extractor, or a leaf its extractor cannot price, is refused by name.

DeepSeek-V4's MoE takes the step roofline's expectation under independent
uniform routing: the routed experts run the rows the leaf holds, under expert
parallelism the busiest rank's expected share, at the format the op quantizes
its rows to, and read each activated expert once; an exchange's group width
is read off its own shapes.
"""

from __future__ import annotations

import math

import torch
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import flop_registry

from atom.compass.backends.provenance import Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView
from atom.compass.backends.step_roofline import Module, _index_scores, _sparse_products
from atom.compass.backends.step_roofline import _work as module_work
from atom.compass.backends.work import (
    ELEMENT_BYTES,
    Collective,
    Work,
    activated_experts,
    all_to_all,
    allgather,
    allreduce,
    attention,
    gemm,
    grouped_gemm,
    max_rank_load,
    reduce_scatter,
)

SOURCE = "op-work"
#: The work primitives' operand formats, by dtype.
FORMATS = {
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e4m3fnuz": "fp8",
    "torch.float4_e2m1fn_x2": "fp4",
}
#: Aten operators that allocate and launch nothing.
ALLOCATIONS = ("aten.empty.", "aten.empty_like.")


class _Unpriced(Exception):
    """A leaf its kind's extractor cannot price; `work` refuses it."""


def _fmt(dtype: str) -> str:
    if dtype not in FORMATS:
        raise _Unpriced(f"no operand format for {dtype}; known: {', '.join(FORMATS)}")
    return FORMATS[dtype]


def _size(dtype: str) -> int:
    return getattr(torch, dtype.removeprefix("torch.")).itemsize


def _tensors(value):
    if isinstance(value, dict) and "dtype" in value:
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _tensors(item)


def _bytes(tensor) -> int:
    shape, stride = tensor["shape"], tensor["stride"]
    return math.prod(n for n, s in zip(shape, stride) if s) * _size(tensor["dtype"])


def _out_dtype(op) -> str | None:
    """The outputs' dtype: the `dtype` slot's, else the first tensor's."""
    dtype = op["args"].get("dtype")
    if isinstance(dtype, str):
        return dtype
    return next((t["dtype"] for t in _tensors(list(op["args"].values()))), None)


def _boundary(op, written=(), outs=None) -> Work:
    """Each tensor argument read once, the `written` slots written once more,
    and each output written once in its dtype in `outs`, by default
    `_out_dtype`'s."""
    args = op["args"]
    moved = sum(map(_bytes, _tensors(list(args.values()))))
    moved += sum(_bytes(t) for slot in written for t in _tensors(args[slot]))
    if outs is None:
        outs = [_out_dtype(op)] * len(op["out"])
    moved += sum(math.prod(s) * _size(d) for s, d in zip(op["out"], outs))
    return Work({}, moved)


def _gemm(a="A", b="B", out="otype", extra=("bias",)):
    """`a [.., k] @ b [n, k]^T` in the `out` slot's dtype, and the `extra`
    slots (a bias, the block scales) read once."""

    def extract(op, view, tp, kv) -> Work:
        args = op["args"]
        x, w = args[a], args[b]
        *rows, k = x["shape"]
        fmts = _fmt(x["dtype"]), _fmt(w["dtype"]), _fmt(args.get(out) or x["dtype"])
        read = sum(_bytes(t) for slot in extra for t in _tensors(args.get(slot)))
        return gemm(math.prod(rows), w["shape"][0], k, *fmts) + Work({}, read)

    return extract


def _attention(op, view, tp, kv) -> Work:
    """Causal attention over the view's rows: Q read and O written, each row's
    K and V over its context read once in the KV cache's format `kv`, the new
    tokens' among them. F and M take heads times head dim only, so each side
    counts as one head of its width."""
    q, k = op["args"]["q"], op["args"]["k"]
    rows = view.requests
    return attention(
        [r.query_tokens for r in rows],
        [r.context_tokens for r in rows],
        math.prod(q["shape"][1:]),
        1,
        math.prod(k["shape"][1:]),
        _fmt(q["dtype"]),
        kv,
    )


def _linear_attention(op, view, tp, kv) -> Work:
    """GDN's recurrence over the view's rows, as the step roofline counts it:
    value heads and width off `core_attn_out`, key heads off `mixed_qkv`, the
    state in the activations' format, where ATOM keeps it for these models."""
    qkv, out = op["args"]["mixed_qkv"], op["args"]["core_attn_out"]
    *_, heads, dv = out["shape"]
    # ponytail: the key width is taken equal to the value width, which the
    # leaf does not state apart; the module walk reads it off the layer.
    key_heads = (qkv["shape"][-1] - heads * dv) // (2 * dv)
    dims = (heads, key_heads, dv, dv)
    return module_work(
        Module("", "linear_attention", dims, _fmt(qkv["dtype"])), view, 0, 0
    )


def _gather(table):
    """A lookup of `out`'s rows in the `table` slot: the rows it writes and
    the indices. The rows it reads are not counted: a masked lookup reads only
    its shard's ids, and a repeated id reads its row once."""

    def extract(op, view, tp, kv) -> Work:
        args = op["args"]
        rows = sum(math.prod(s) for s in op["out"]) * _size(args[table]["dtype"])
        rest = [v for slot, v in args.items() if slot != table]
        return Work({}, sum(map(_bytes, _tensors(rest))) + rows)

    return extract


def _in_place(*slots):
    return lambda op, view, tp, kv: _boundary(op, slots)


def _all_reduce(op, view, tp, kv) -> Work:
    return _boundary(op) + allreduce(tp, _bytes(op["args"]["tensor"]))


def _all_gather(op, view, tp, kv) -> Work:
    gathered = math.prod(op["out"][0]) * _size(op["args"]["input"]["dtype"])
    return _boundary(op) + allgather(op["args"]["group_size"], gathered)


def _broadcast(op, view, tp, kv) -> Work:
    """The source sends the whole buffer at least once."""
    sent = _bytes(op["args"]["input"])
    return _boundary(op) + Work(comm=(Collective(tp, sent),) if tp > 1 else ())


def _all_to_all(op, view, tp, kv) -> Work:
    """A rank sends every split but its own, at least all but the largest."""
    sizes, sent = op["args"]["input_split_sizes"], _bytes(op["args"]["input"])
    if not sizes:
        raise _Unpriced("no input_split_sizes, so the group's width is not stated")
    sent *= 1 - max(sizes) / sum(sizes)
    return _boundary(op) + Work(comm=(Collective(len(sizes), sent),) if sent else ())


def _dp_exchange(op, view, tp, kv) -> Work:
    """The data-parallel group's gather of the MoE's input, or reduce-scatter
    of its output: the larger side holds every rank's rows, the smaller this
    rank's, and the group is `sizes` long or their ratio wide."""
    x, out = op["args"]["x"], op["out"][0]
    rows = sorted((x["shape"][0], out[0]))
    ranks = len(op["args"].get("sizes") or ()) or rows[1] // rows[0]
    every = rows[1] * math.prod(x["shape"][1:]) * _size(x["dtype"])
    collective = allgather if out[0] > x["shape"][0] else reduce_scatter
    return _boundary(op) + collective(ranks, every)


def _fused_moe(op, view, tp, kv) -> Work:
    """The routed experts on `hidden_states`' rows, each route through one
    expert's gate-up and down products: an expectation under uniform routing.
    Under expert parallelism `expert_mask` spans every rank's experts, of which
    this rank holds `w1`'s, and it runs the busiest rank's expected share.
    A DP gather's pad rows (`moe_tokens` real among them) all carry token 0,
    so they activate experts as one token.

    With a nonzero `quant_type` the op quantizes `hidden_states` to the
    weights' format inside it, so both products run at that format; otherwise
    at `hidden_states`'. M is the leaf's boundary: each activated expert's
    weights and scales read once, and the rows' tensors at the share of rows
    this rank runs, at least its routes over `k`."""
    args = op["args"]
    (rows, hidden), k = args["hidden_states"]["shape"], args["topk_ids"]["shape"][1]
    local, up = args["w1"]["shape"][:2]
    mask = args.get("expert_mask")
    ranks = mask["shape"][0] // local if mask else 1
    real = view.moe_tokens
    distinct = rows if real is None else min(rows, real + 1)
    routed, groups = rows * k, activated_experts(local * ranks, k, distinct)
    if ranks > 1:
        routed = max_rank_load(rows, k, local * ranks, ranks)
        groups /= ranks
    b = _fmt(args["w1"]["dtype"])
    a = b if args.get("quant_type") else _fmt(args["hidden_states"]["dtype"])
    flops = grouped_gemm(routed, up, hidden, groups, a, b)
    flops += grouped_gemm(routed, hidden, up // 2, groups, a, b)
    held = ("w1", "w2", "w1_scale", "w2_scale", "bias1", "bias2")
    weights = sum(_bytes(t) for s in held for t in _tensors(args.get(s)))
    io = _boundary(op).bytes - weights
    return Work(
        flops.flops_by_format, weights * groups / local + io * routed / (rows * k)
    )


def _mori(*payload, dispatch: bool):
    """MORI's dispatch, or its combine back: each of this rank's tokens, its
    `payload` rows, goes to every other rank holding one of its `k` experts.
    The group is `num_ep_ranks` wide, or, in a graph that does not state it,
    the received buffer's rows over the tokens. With the experts split evenly,
    a rank holds one of a token's experts with chance at least
    `1 - (1 - 1/N)^k`, whatever the expert count. The step roofline
    counts the exchange alone, and so does this."""

    def extract(op, view, tp, kv) -> Work:
        args = op["args"]
        tokens, k = args["indices"]["shape"]
        buffer = op["out"][0] if dispatch else args["input"]["shape"]
        ranks = args.get("num_ep_ranks") or buffer[0] // tokens
        row = sum(_bytes(t) / t["shape"][0] for s in payload for t in _tensors(args[s]))
        return all_to_all(ranks, tokens * activated_experts(ranks, 1, k) * row)

    return extract


def _mhc(residual):
    """A hyper-connection mix: its boundary, and the residual's widened rows
    projected by `fn`'s mixing weights."""

    def extract(op, view, tp, kv) -> Work:
        r, fn = op["args"][residual], op["args"]["fn"]
        flops = 2 * r["shape"][0] * math.prod(fn["shape"])
        return _boundary(op) + Work({_fmt(r["dtype"]): flops})

    return extract


def _launch(op, view, tp, kv) -> Work:
    """One launch at the kernel floor."""
    # ponytail: V4's index builds write each row's window and compressed
    # entries, which scale with its context through V4's geometry, and the
    # leaf does not state that geometry; the floor is their lower bound.
    return Work()


def _unstated(op, view, tp, kv) -> Work:
    raise _Unpriced(
        "the leaf states the layer's input, positions and context key only; "
        "the weights, heads, window and compression it reads live in that "
        "layer, which the capture does not record"
    )


def _kv_indices(op, view, tp, kv) -> Work:
    # ponytail: the block-table entries it reads and the indices it writes
    # scale with each row's blocks, which the leaf does not state; only the
    # row offsets are counted. Add them when the KV block size reaches here.
    return Work({}, _bytes(op["args"]["kv_indptr_convert"]))


def _only(*slots, out=None):
    """The `slots`' tensors each read or written once, and the outputs in the
    `out` dtype. What else the leaf reaches by index (a KV pool, a rotary
    cache, a state ring) is left out, which only lowers its bytes."""

    def extract(op, view, tp, kv) -> Work:
        args = op["args"]
        moved = sum(_bytes(t) for s in slots for t in _tensors(args.get(s)))
        if out is not None:
            moved += sum(math.prod(s) * _size(out) for s in op["out"])
        return Work({}, moved)

    return extract


def _quant(slot):
    """A block quant: its boundary, the values in the `slot` dtype and their
    e8m0 scales."""
    return lambda op, view, tp, kv: _boundary(
        op, outs=[op["args"][slot], "torch.float8_e8m0fnu"]
    )


def _batched_gemm(op, view, tp, kv) -> Work:
    """V4's grouped output projection: `x [m, g, k]` through each group's
    `[n, k]` weights, read once each, and both block scales read once. The
    outputs are counted at `x`'s format, which only lowers them."""
    args = op["args"]
    x, w = args["x"], args["wo_a"]
    m, groups, k = x["shape"]
    fmt = _fmt(x["dtype"])
    one = gemm(m, w["shape"][1], k, fmt, _fmt(w["dtype"]), fmt)
    scales = _bytes(args["x_scale"]) + _bytes(args["w_scale"])
    return sum([one] * groups, Work({}, scales))


def _v4_prefill(op, view, tp, kv) -> Work:
    """V4's prefill attention over the KV entries its index lists name, the
    cached prefix's and the chunk's own: each query against each of its
    entries, two products of the head width each, at Q's format. Q, the
    chunk's KV, the index lists and O cross once; the prefix entries are
    read from the pool and left out of the bytes."""
    args = op["args"]
    pairs = args["kv_indices_prefix"]["shape"][0]
    pairs += args["kv_indices_extend"]["shape"][0]
    _, heads, _ = args["q_nope"]["shape"]
    flops = 4 * heads * op["out"][0][-1] * pairs
    rest = [v for s, v in args.items() if not s.startswith("unified_kv")]
    moved = sum(map(_bytes, _tensors(rest)))
    moved += math.prod(op["out"][0]) * _size("torch.bfloat16")
    return Work({_fmt(args["q_nope"]["dtype"]): flops}, moved)


def _stated(op, *names) -> list:
    """What `op` states beside its arguments under `names`."""
    stated = op.get("stated", {})
    if missing := [name for name in names if name not in stated]:
        raise _Unpriced(
            f"the leaf states no {', '.join(missing)} of its layer, which its "
            "reads depend on: trace its graph again"
        )
    return [stated[name] for name in names]


def _v4_decode(q, *slots, out=None):
    """V4's decode attention: `_only`'s `q`, `slots` and outputs, and the KV
    entries each row reads at the window, compression ratio and top-k its
    leaf states, as the step roofline counts them: two products of `q`'s head
    width per head and entry, at the wider of Q's and the KV cache's formats,
    and each entry read once at that width in the KV format. The rotary
    part's products and bytes are left out, which only lowers both."""
    boundary = _only(q, *slots, out=out)

    def extract(op, view, tp, kv) -> Work:
        geometry = _stated(op, "window_size", "compress_ratio", "index_topk")
        x = op["args"][q]
        _, heads, dim = x["shape"]
        fmt, rows = _fmt(x["dtype"]), view.requests
        work = _sparse_products(rows, heads, dim, fmt, kv, *geometry)
        return boundary(op, view, tp, kv) + work

    return extract


def _indexer(op, view, tp, kv) -> Work:
    """V4's indexer score and top-k: its boundary, and each new query scored
    against every compressed entry before it at the ratio the leaf states, as
    the step roofline counts them: one product of the index head width per
    index head, at fp4 when the leaf takes `q_scale`, else at `q_quant`'s
    format, and each entry read once at that width and format."""
    (ratio,) = _stated(op, "compress_ratio")
    q = op["args"]["q_quant"]
    fmt = "fp4" if op["args"].get("q_scale") else _fmt(q["dtype"])
    _, heads, packed = q["shape"]
    dim = int(packed * _size(q["dtype"]) / ELEMENT_BYTES[fmt])
    return _boundary(op) + _index_scores(view.requests, ratio, heads, dim, fmt)


def _shape(value):
    if isinstance(value, dict) and "dtype" in value:
        return value["shape"]
    return [_shape(v) for v in value] if isinstance(value, list) else value


def _meta(slot, value):
    """`value` as a call on the meta device takes it: a tensor as an empty
    meta tensor of its shape, dtype and stride."""
    if isinstance(value, list):
        return [_meta(slot, v) for v in value]
    if isinstance(value, dict) and "dtype" in value:
        dtype = getattr(torch, value["dtype"].removeprefix("torch."))
        return torch.empty_strided(
            value["shape"], value["stride"], dtype=dtype, device="meta"
        )
    if slot == "device":
        return "meta"
    if isinstance(value, str) and value.startswith("torch."):
        return getattr(torch, value.removeprefix("torch."), value)
    return value


def _aten(op, view, tp, kv) -> Work | None:
    """Any aten operator: its boundary, each output in the dtype a meta-device
    call of it returns, and torch's flop formula when it has one. A view,
    which aliases its input, writes nothing and keeps the dtype, launches no
    kernel; nor does an allocation."""
    name, args = op["op"], op["args"]
    if name.startswith(ALLOCATIONS):
        return None
    _, packet, overload = name.split(".")
    target = getattr(getattr(torch.ops.aten, packet), overload)
    try:
        result = target(**{slot: _meta(slot, v) for slot, v in args.items()})
    except Exception as error:  # noqa: BLE001 - refused with what stopped it
        raise _Unpriced(f"a meta-device call raised {error!r}") from None
    outs = [str(t.dtype) for t in tree_leaves(result) if isinstance(t, torch.Tensor)]
    schema = target._schema
    first = next(_tensors(list(args.values())), None)
    aliased = all(r.alias_info for r in schema.returns)
    writes = any(a.alias_info and a.alias_info.is_write for a in schema.arguments)
    if aliased and not writes and first and set(outs) <= {first["dtype"]}:
        return None
    # An output that is an argument it writes is not written twice.
    rewritten = any(r.alias_info and r.alias_info.is_write for r in schema.returns)
    work = _boundary(op, outs=[] if rewritten else outs)
    formula = flop_registry.get(target.overloadpacket)
    if formula is None:
        return work
    out = op["out"][0] if len(op["out"]) == 1 else op["out"]
    flops = formula(*map(_shape, args.values()), out_val=out)
    return work + Work({_fmt(first["dtype"]): flops})


#: The extractor of each op kind but aten's, by the name a leaf carries.
MOE = "atom.model_ops.moe:"
MORI = "mori.ops.dispatch_combine:EpDispatchCombineOp."
V4_INDICES = "atom.model_ops.v4_kernels.paged_"
V4 = "atom.model_ops.v4_kernels."
#: What V4's fused QK norm and RoPE writes: Q and K, each as its fp8 values
#: with their scales and its bf16 rotary part.
QK_OUT = ("q_nope_scale_buff", "q_rope_buff", "k_nope_scale_buff", "k_rope_buff")
EXTRACTORS = {
    "aiter.gemm_a16w16.default": _gemm(),
    "aiter.gemm_a8w8_blockscale_preshuffle_impl.default": _gemm(
        "x", "weight", "dtype", ("x_scale", "w_scale")
    ),
    "aiter.per_1x128_e8m0_quant.default": _quant("quant_dtype"),
    "aiter.rmsnorm2d_fwd_.default": _in_place(),
    "aiter.mhc_res_repeat.default": _in_place(),
    "aiter.mhc_pre.default": _mhc("residual"),
    "aiter.mhc_post.default": _in_place(),
    "aiter.mhc_fused_post_pre.default": _mhc("residual_in"),
    "aiter.v4_attention_with_output.default": _unstated,
    "aiter._aiter_rms_quant.default": _quant("value_dtype"),
    "aiter._fused_qk_norm_rope_group_quant_kernel.default": _only(
        "q", "kv", "k_weight", "positions", *QK_OUT
    ),
    "aiter._inverse_rope_group_quant_kernel.default": _only(
        "o", "positions", "x_fp8", "x_scale"
    ),
    "aiter._rope_rotate_activation_fp4quant.default": _only(
        "input", "positions", "out", "scale"
    ),
    "aiter.batched_gemm_a8w8_mxscale.default": _batched_gemm,
    "aiter.indexer_score_topk.default": _indexer,
    "aiter.pa_sparse_prefill_fp8_opus.default": _v4_prefill,
    "aiter.mla_decode_v4_ps_asm.default": _v4_decode("Q", "qrope", "sink", "output"),
    f"{V4}paged_decode:_sparse_attn_v4_paged_decode_asm": _v4_decode(
        "q_packed_in", "q_rope_in", "attn_sink", out="torch.bfloat16"
    ),
    f"{V4}fused_compress:fused_compress_attn": _only(
        "kv_in", "score_in", "ape", "rms_weight", "state_slot_mapping"
    ),
    # The kernel reads only the kv and score rows write_plan names, at most
    # (1 + overlap) * ratio per sequence, so neither is counted.
    f"{V4}state_writes:update_compressor_states": _only("ape", "state_slot_mapping"),
    f"{V4}state_writes:swa_write": _only(
        "positions", "cu_seqlens_q", "state_slot_per_seq"
    ),
    f"{V4}csa_translate_pack:csa_translate_pack": _only(
        "topk_local", "positions", "kv_indptr_csa", "batch_id_per_q_token"
    ),
    "aiter.topk_gating_fwd.default": _in_place(),
    "atom.model_ops.triton_hash_topk:hash_topk_triton": _gather("tid2eid"),
    "aiter.fused_moe_.default": _fused_moe,
    MOE + "all_gather_with_padding": _dp_exchange,
    MOE + "all_gatherv": _dp_exchange,
    MOE + "reduce_scatter_with_unpadding": _dp_exchange,
    MOE + "reduce_scatterv": _dp_exchange,
    MORI + "dispatch": _mori("input", "weights", "scales", "indices", dispatch=True),
    MORI + "combine": _mori("input", "weights", dispatch=False),
    "_c10d_functional.all_to_all_single.default": _all_to_all,
    V4_INDICES + "decode_indices:build_v4_paged_decode_indptr": _launch,
    V4_INDICES + "decode_indices:write_v4_paged_decode_indices": _launch,
    V4_INDICES + "prefill_indices:write_v4_paged_prefill_indices": _launch,
    "atom.model_ops.attentions.deepseek_v4_attn:"
    "DeepseekV4AttentionMetadataBuilder._build_v4_indexer_meta": _launch,
    "aiter.unified_attention_with_output_base.default": _attention,
    "aiter.linear_attention_with_output_base.default": _linear_attention,
    "aiter.silu_and_mul.default": _in_place(),
    "aiter._fused_qk_rmsnorm_group_quant_kernel.default": _in_place(),
    "aiter.mixed_sample_outer_exponential.default": _in_place(),
    "aiter.masked_embedding.default": _gather("weight"),
    "aten.embedding.default": _gather("weight"),
    "aten.index.Tensor": _gather("self"),
    "aiter.all_reduce_.default": _all_reduce,
    "_c10d_functional.all_gather_into_tensor.default": _all_gather,
    "_c10d_functional.broadcast.default": _broadcast,
    "_c10d_functional.wait_tensor.default": lambda op, view, tp, kv: None,
    "atom.model_ops.layernorm:fused_qk_norm": _in_place("q", "k"),
    "atom.model_ops.triton_mrope:try_mrope_qk_fused": _in_place("q", "k"),
    "atom.model_ops.sampler:_apply_greedy_tokens": _in_place(),
    "atom.model_ops.decode_input_ids:fill_deferred_decode_ids": _in_place(),
    "atom.utils.block_convert:kv_indices_generate_triton": _kv_indices,
}


def work(op: dict, view: BatchView, tp: int, kv_format: str) -> Work | None:
    """`op`'s work on a step of `view`'s rows in a TP group of `tp` ranks
    whose KV cache holds `kv_format` (a work format, "bf16" or "fp8"), or None
    when it launches no kernel. A kind with no extractor, or a leaf its
    extractor cannot price, is refused."""
    name = op["op"]
    extract = EXTRACTORS.get(name) or (name.startswith("aten.") and _aten)
    if not extract:
        reason = f"no extractor for the op kind {name}"
        raise CostRefused(view, [Refusal(SOURCE, reason)])
    # Its outputs, an in-place leaf's target among them, hold no element: it
    # writes nothing, whatever its sources hold.
    if op["out"] and not any(map(math.prod, op["out"])):
        return None
    try:
        return extract(op, view, tp, kv_format)
    except _Unpriced as error:
        reason = f"the op kind {name}: {error}"
    raise CostRefused(view, [Refusal(SOURCE, reason)])
