# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A captured leaf's work, by op kind: the op-level extractor registry.

A leaf is one operator of an `op_graph` as its JSON holds it: `op`, its kind;
`args` by schema slot, a tensor as its shape, dtype, stride and role; `out`,
its output shapes. `work(leaf, view, tp, kv_format, ep)` is the leaf's `Work`
on a step of `view`'s rows: F and C from the primitives its kind's extractor emits,
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
its rows to, and read each activated expert once; a DP exchange's group width
is read off its own shapes. ATOM's Triton experts run as aiter's grouped
GEMMs, each priced so, at the activation format its kind names; their
expert-parallel width is the run's `ep`, the experts `ep` times a rank's.

DeepSeek-V4.1's sparse attention and indexer scorers read their entries at the
geometry their leaf states, over the queries the leaf holds (`_last`): a
replayed prefill's decoder layers run each finishing request's last rows.
"""

from __future__ import annotations

import bisect
import inspect
import math
from dataclasses import replace

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


def _pass(view, q) -> tuple:
    """The view's rows as the pass whose Q is `q` runs them: a Q of fewer
    tokens than the rows query is a drafter's later pass, one query token a
    row, as ATOM's `EagleProposer.propose` runs every pass after its first."""
    rows = view.requests
    if q["shape"][0] < sum(r.query_tokens for r in rows):
        return tuple(replace(r, query_tokens=1) for r in rows)
    return rows


def _last(view, q) -> tuple:
    """The view's rows as a leaf of Q `q` runs them: each row's last `n`
    queries, `n` the most whose sum `q`'s rows hold, every query a row has
    when they hold them all: a bounded replay's tail of a prefill, its last
    layer's last row, a drafter's later pass."""
    rows, held = view.requests, q["shape"][0]
    top = max(r.query_tokens for r in rows)
    n = bisect.bisect_right(
        range(1, top + 1), held, key=lambda n: sum(min(r.query_tokens, n) for r in rows)
    )
    return tuple(replace(r, query_tokens=min(r.query_tokens, max(n, 1))) for r in rows)


def _attention(op, view, tp, kv) -> Work:
    """Causal attention over the pass's rows (`_pass`): Q read and O written,
    each row's K and V over its context read once in the KV cache's format
    `kv`, the new tokens' among them. F and M take heads times head dim only,
    so each side counts as one head of its width."""
    q, k = op["args"]["q"], op["args"]["k"]
    rows = _pass(view, q)
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


def _fused_all_reduce_norm(op, view, tp, kv) -> Work:
    """An all-reduce of `inp` fused with the residual add and RMSNorm after
    it: the all-reduce's bytes, and each tensor read and written once."""
    return _boundary(op) + allreduce(tp, _bytes(op["args"]["inp"]))


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
    rank's, and the group is `sizes` long or their ratio wide. A gather of
    routed rows moves several tensors of one height in the one collective."""
    xs, out = op["args"]["x"], op["out"][0]
    xs = xs if isinstance(xs, (list, tuple)) else [xs]
    rows = sorted((xs[0]["shape"][0], out[0]))
    ranks = len(op["args"].get("sizes") or ()) or rows[1] // rows[0]
    row = sum(math.prod(x["shape"][1:]) * _size(x["dtype"]) for x in xs)
    every = rows[1] * row
    collective = allgather if out[0] > xs[0]["shape"][0] else reduce_scatter
    return _boundary(op) + collective(ranks, every)


def _routing(view, tokens, k, experts, ranks):
    """The routes and activated experts of `tokens` each routed to `k` of
    `experts` spread over `ranks`, an expectation under uniform routing: the
    busiest rank's expected share under expert parallelism. A DP gather's pad
    rows (`moe_tokens` real among them) all carry token 0, so they activate
    experts as one token."""
    real = view.moe_tokens
    distinct = tokens if real is None else min(tokens, real + 1)
    groups = activated_experts(experts, k, distinct)
    if ranks == 1:
        return tokens * k, groups
    return max_rank_load(tokens, k, experts, ranks), groups / ranks


def _fused_moe(op, view, tp, kv, ep) -> Work:
    """The routed experts on `hidden_states`' rows, each route through one
    expert's gate-up and down products, routed as `_routing` says. Under
    expert parallelism `w1` holds this rank's share of `ep` times its experts.

    With a nonzero `quant_type` the op quantizes `hidden_states` to the
    weights' format inside it, so both products run at that format; otherwise
    at `hidden_states`'. M is the leaf's boundary: each activated expert's
    weights and scales read once, and the rows' tensors at the share of rows
    this rank runs, at least its routes over `k`."""
    args = op["args"]
    (rows, hidden), k = args["hidden_states"]["shape"], args["topk_ids"]["shape"][1]
    local, up = args["w1"]["shape"][:2]
    routed, groups = _routing(view, rows, k, local * ep, ep)
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


def _experts(act: str):
    """One of aiter's Triton grouped GEMMs of the routed experts, `x` in `act`
    through fp4 weights `w` (each expert's `K x N` packed two to a byte),
    routed as `_routing` says: `routing_data` states the top k, the gather or
    scatter index one slot per route. Under expert parallelism over `ep` ranks
    `w` holds this rank's share of `ep` times its experts. M is the leaf's
    boundary: each activated expert's weights and scales read once, and the
    rows' tensors and its output (SwiGLU's half of the gate-up, or the tokens'
    rows a scatter reduces into, unless `y_out` holds them and so they are
    read as an argument) at the share of routes this rank runs."""

    def extract(op, view, tp, kv, ep) -> Work:
        args = op["args"]
        x, w = args["x"], args["w"]
        k = dict((args.get("routing_data") or {}).get("state", ())).get("n_expts_act")
        index = args.get("gather_indx") or args.get("scatter_indx")
        if not (k and index):
            raise _Unpriced("no top k in routing_data, or no gather or scatter index")
        routes, local = index["shape"][0], w["shape"][0]
        tokens = routes // k
        width = int(x["shape"][-1] * _size(x["dtype"]) / ELEMENT_BYTES[act])
        n = round(_bytes(w) / local / (width * ELEMENT_BYTES["fp4"]))
        routed, groups = _routing(view, tokens, k, local * ep, ep)
        flops = grouped_gemm(routed, n, width, groups, act, "fp4").flops_by_format
        held = ("w", "w_scales", "bias")
        weights = sum(_bytes(t) for s in held for t in _tensors(args.get(s)))
        rows = tokens if args.get("scatter_indx") else routes
        out = n // 2 if args.get("apply_swiglu") else n
        written = 0 if args.get("y_out") else rows * out * _size(args["out_dtype"])
        io = _boundary(op).bytes - weights + written
        return Work(flops, weights * groups / local + io * routed / routes)

    return extract


def _downcast(op, view, tp, kv) -> Work:
    """aiter's MX downcast along the last axis: its source read once, its
    values written in `out_quant_type`, and one e8m0 scale byte per 32."""
    args = op["args"]
    values = math.prod(args["src_tensor"]["shape"])
    return _boundary(op) + Work({}, values * (_size(args["out_quant_type"]) + 1 / 32))


def _mori(*payload):
    """MORI's dispatch, or its combine back: each of this rank's tokens, its
    `payload` rows, goes to every other rank holding one of its `k` experts.
    The group is the run's EP width, N `ranks`. With the experts split evenly,
    a rank holds one of a token's experts with chance at least
    `1 - (1 - 1/N)^k`, whatever the expert count. The step roofline
    counts the exchange alone, and so does this."""

    def extract(op, view, tp, kv, ranks) -> Work:
        args = op["args"]
        tokens, k = args["indices"]["shape"]
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
    # ponytail: V4's and V4.1's index builds write each row's window and
    # compressed entries, which scale with its context through the model's
    # geometry, and the leaf does not state it; the floor is their lower bound.
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


def _v4_decode(q, *slots, out=None, queries=_pass):
    """V4's decode attention: `_only`'s `q`, `slots` and outputs, and the KV
    entries each row reads at the window, compression ratio and top-k its
    leaf states, as the step roofline counts them: two products of `q`'s head
    width per head and entry, at the wider of Q's and the KV cache's formats,
    and each entry read once at that width in the KV format. The rotary
    part's products and bytes are left out, which only lowers both. The view's
    rows are as `queries` says `q` runs them."""
    boundary = _only(q, *slots, out=out)

    def extract(op, view, tp, kv) -> Work:
        geometry = _stated(op, "window_size", "compress_ratio", "index_topk")
        x = op["args"][q]
        _, heads, dim = x["shape"]
        fmt, rows = _fmt(x["dtype"]), queries(view, x)
        work = _sparse_products(rows, heads, dim, fmt, kv, *geometry)
        return boundary(op, view, tp, kv) + work

    return extract


def _indexer(op, view, tp, kv) -> Work:
    """V4's indexer score and top-k: its boundary, and each new query scored
    against every compressed entry before it at the ratio the leaf states, as
    the step roofline counts them: one product of the index head width per
    index head, at fp4 when the leaf takes `q_scale`, else at `q_quant`'s
    format, and each entry read once at that width and format. `q_quant` is
    rows first and heads by packed width last, a query axis between them or
    not."""
    (ratio,) = _stated(op, "compress_ratio")
    q = op["args"]["q_quant"]
    fmt = "fp4" if op["args"].get("q_scale") else _fmt(q["dtype"])
    heads, packed = q["shape"][-2:]
    dim = int(packed * _size(q["dtype"]) / ELEMENT_BYTES[fmt])
    return _boundary(op) + _index_scores(_pass(view, q), ratio, heads, dim, fmt)


def _v41_scores(q):
    """V4.1's indexer scorer: `q` and the head weights read once, and each
    query `q` holds (`_last`) scored at its format against the compressed
    entries before it at the ratio the leaf states, at most the candidates it
    states where an earlier layer chose them (`_index_scores`). The logits
    plane it writes is left out, which only lowers M."""

    def extract(op, view, tp, kv) -> Work:
        ratio, cap = _stated(op, "compress_ratio", "index_candidates")
        x = op["args"][q]
        *_, heads, dim = x["shape"]
        scores = _index_scores(_last(view, x), ratio, heads, dim, _fmt(x["dtype"]), cap)
        return _only(q, "weights")(op, view, tp, kv) + scores

    return extract


def _mhc_delayed(op, view, tp, kv) -> Work:
    """V4.1's delayed hyper-connection seam: each tensor argument read once,
    the outputs written at a byte an element, which only lowers M, and each
    row's widened residual projected onto the `hc_base` mixes."""
    args = op["args"]
    *rows, copies, width = args["residual"]["shape"]
    mixes = math.prod(args["hc_base"]["shape"])
    flops = 2 * math.prod(rows) * copies * width * mixes
    outs = ["torch.float8_e4m3fn"] * len(op["out"])
    return _boundary(op, outs=outs) + Work({_fmt(args["residual"]["dtype"]): flops})


def _quantize(op, view, tp, kv) -> Work:
    """ATOM's group-32 fp8 quant: `x` read once and its values written in fp8
    with an e8m0 scale per 32, or in its own format where it only rounds them
    (`dequantize`)."""
    x = op["args"]["x"]
    each = _size(x["dtype"]) if op["args"].get("dequantize") else 1 + 1 / 32
    return Work({}, _bytes(x) + math.prod(x["shape"]) * each)


def _mx(*slots, read=()):
    """Each of `slots` read once and written once in mxfp8, a byte an element
    and an e8m0 scale per 32, and `read` read once."""

    def extract(op, view, tp, kv) -> Work:
        args = op["args"]
        values = sum(math.prod(args[s]["shape"]) for s in slots)
        moved = sum(_bytes(args[s]) for s in (*slots, *read))
        return Work({}, moved + values * (1 + 1 / 32))

    return extract


def _gather_rows(op, view, tp, kv) -> Work:
    """`rows`' rows of each of `tensors` along `dim`, each read and written
    once, and the row indices."""
    args = op["args"]
    count, dim = args["rows"]["shape"][0], args["dim"]
    row = sum(
        math.prod(t["shape"][dim + 1 :]) * _size(t["dtype"])
        for t in _tensors(args["tensors"])
    )
    return Work({}, 2 * count * row + _bytes(args["rows"]))


def _index_fill(op, view, tp, kv) -> Work:
    """A fill of `self`'s slices along `dim` that `index` names: each slice
    written once, and the index read."""
    args = op["args"]
    target, index = args["self"], args["index"]
    shape = target["shape"]
    each = math.prod(shape) // shape[args["dim"] % len(shape)] * _size(target["dtype"])
    return Work({}, index["shape"][0] * each + _bytes(index))


def _draft_kv_tail(op, view, tp, kv) -> Work:
    """V4.1 DSpark's draft-KV epilogue: `kv`'s rows read once and written back
    as `_quantize` writes them, packed or rounded in place, and the stages'
    norm weights and the positions read once. The rotary rows it reads by
    position are left out, which only lowers M."""
    args = op["args"]
    rows = {"args": {"x": args["kv"], "dequantize": not args["packed"]}}
    norms = _only("norm_weight", "positions")(op, view, tp, kv)
    return _quantize(rows, view, tp, kv) + norms


def _window_rows(op, view, tp, kv) -> Work:
    """V4.1 DSpark's window gather: the ring of each of `layers` at each of
    `slots` read and written once, and the indices."""
    args = op["args"]
    window = args["window"]
    ring = math.prod(window["shape"][-2:]) * _size(window["dtype"])
    rings = args["layers"]["shape"][0] * args["slots"]["shape"][0]
    return Work({}, 2 * rings * ring) + _only("layers", "slots")(op, view, tp, kv)


def _draft_indices(op, view, tp, kv) -> Work:
    """V4.1 DSpark's block indices from each row's anchor position: the
    positions read once, and per row its `width` positions, an int32 index
    of every ring slot and block row for each of them, and each ring slot's
    position written."""
    args = op["args"]
    positions = args["positions"]
    ring, width, size = args["ring"], args["width"], _size(positions["dtype"])
    each = (width + ring) * size + width * (ring + width) * _size("torch.int32")
    return Work({}, _bytes(positions) + positions["shape"][0] * each)


def _block_state(op, view, tp, kv) -> Work:
    """V4.1 DSpark's block state: the anchors' and the noise's embeddings read
    once, and each row's `width` positions written as `hc_mult` copies of the
    embedding with an fp32 pre-mix each."""
    args = op["args"]
    (rows, hidden), dtype = args["anchor_embed"]["shape"], args["anchor_embed"]["dtype"]
    copies = rows * args["width"] * args["hc_mult"]
    written = copies * (hidden * _size(dtype) + _size("torch.float32"))
    return _only("anchor_embed", "noise_embed")(op, view, tp, kv) + Work({}, written)


def _block_attention(op, view, tp, kv) -> Work:
    """DSpark's draft attention: each query of `q` attends every entry its row
    of `topk_idxs` lists, two products of the head width per head and entry,
    each argument read once and O written as `q`."""
    args = op["args"]
    q = args["q"]
    *queries, heads, dim = q["shape"]
    entries = math.prod(queries) * args["topk_idxs"]["shape"][-1]
    products = Work({_fmt(q["dtype"]): 4 * heads * dim * entries}, _bytes(q))
    return _boundary(op) + products


def _markov(op, view, tp, kv) -> Work:
    """DSpark's Markov head over a vocabulary shard: a GEMM of each row's
    `markov_w1` row against the shard's `markov_w2` rows, whose output term is
    `base_logits`, which it adds and reduces to an id rather than writes; and
    the ids and the rows it picks and publishes read or written once."""
    args = op["args"]
    rows, shard = args["base_logits"]["shape"]
    w2 = args["markov_w2"]
    fmt, logits = _fmt(w2["dtype"]), _fmt(args["base_logits"]["dtype"])
    picks = _only("prev_ids", "out", "packed", "embed_out")(op, view, tp, kv)
    return gemm(rows, shard, w2["shape"][1], fmt, fmt, logits) + picks


def _commit_cursors(op, view, tp, kv) -> Work:
    """V4.1's tentative cursor commit: each of `count` requests' accepted
    prefix's row read from `cursors` and written to its slot of `cursor`, and
    the anchors, starts and slots read once."""
    args = op["args"]
    row = args["cursor"]["shape"][1] * _size(args["cursor"]["dtype"])
    indices = _only("anchors", "starts", "slots")(op, view, tp, kv)
    return Work({}, 2 * args["count"] * row) + indices


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
    kernel; nor does a dtype view, which reads the same bytes as another
    dtype, or an allocation. A cast, also aliased, copies when its dtype
    differs."""
    name, args = op["op"], op["args"]
    if name.startswith(ALLOCATIONS) or name == "aten.view.dtype":
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
FLYDSL = "aiter.ops.flydsl.kernels."
TRITON_MOE = "aiter.ops.triton.moe."
#: aiter's Triton grouped GEMMs of the routed experts, by kind: the format
#: their activations run in. Their extractors also take the EP width.
EXPERTS = {
    f"{TRITON_MOE}moe_op_gemm_{a}w4:moe_gemm_{a}w4": fmt
    for a, fmt in (("a16", "bf16"), ("a8", "fp8"), ("a4", "fp4"))
}
#: V4's compressor: its sources and parameters read once. The KV and score
#: state rings it reads by slot and the cache rows it writes are left out.
COMPRESS = _only("kv_in", "score_in", "ape", "rms_weight", "state_slot_mapping")
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
    FLYDSL
    + "qk_norm_rope_quant:flydsl_qk_norm_rope_quant": _only(
        "q", "kv", "kv_weight", "q_weight", "positions", "q_out", "kv_out"
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
    "aiter.mla_decode_v4_asm.default": _v4_decode("Q", "qrope", "sink", "output"),
    "aiter.mla:mla_decode_fwd_v4_nm": _v4_decode("q", "qrope", "sink", "output"),
    f"{V4}paged_decode:_sparse_attn_v4_paged_decode_triton": _v4_decode(
        "q", "attn_sink"
    ),
    FLYDSL + "fused_compress_attn:flydsl_fused_compress_attn": COMPRESS,
    FLYDSL + "fused_compress_attn_hca:flydsl_hca_compress_attn": COMPRESS,
    # gfx1250's kernels for the same work, at the same arguments.
    "aiter.batched_gemm_a8w8_mxscale_bpreshuffle.default": _batched_gemm,
    "aiter.mla_sparse_prefill_fp8_asm.default": _v4_prefill,
    FLYDSL + "fused_compress_attn_gfx1250:flydsl_fused_compress_attn_gfx1250": COMPRESS,
    FLYDSL
    + "fused_compress_attn_hca_gfx1250:flydsl_hca_compress_attn_gfx1250": COMPRESS,
    "aiter._build_tiles_raw.default": _launch,
    "aiter._build_sched_raw.default": _launch,
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
    "aiter.rocm_aiter_fused_moe.default": _fused_moe,
    "aiter._topk_softmax.default": _in_place(
        "topk_weights", "topk_indices", "token_expert_indices"
    ),
    "aiter.per_group_quant_hip.default": lambda op, view, tp, kv: _boundary(
        op, outs=[op["args"]["quant_dtype"], "torch.float32"]
    ),
    "aiter.gated_rmsnorm_fp8_group_quant.default": _in_place("out", "scale"),
    "atom.model_ops.fla_ops.replayssm:replayssm_commit": _only(
        "slot_idx", "num_accepted"
    ),
    "atom.model_ops.rejection_sampler:rejection_sample": _in_place(),
    **{kind: _experts(fmt) for kind, fmt in EXPERTS.items()},
    TRITON_MOE + "quant_moe:downcast_to_mxfp": _downcast,
    "aiter.ops.triton.quant.quant:dynamic_mxfp4_quant": _in_place(),
    "aiter.ops.triton.fusions.fused_clamp_act_mul:fused_clamp_act_mul": _in_place(),
    # A lifted leaf records no outputs: the index and weight tensors the
    # routing returns are left out, which only lowers M.
    TRITON_MOE + "moe_routing.topk:topk": _in_place(),
    TRITON_MOE + "moe_routing.routing:sort_tokens": _in_place(),
    TRITON_MOE + "moe_routing.routing:sort_tokens_fused": _in_place(),
    TRITON_MOE + "moe_routing.routing:ep_sort_routing": _in_place(),
    MOE + "all_gather_with_padding": _dp_exchange,
    MOE + "all_gatherv": _dp_exchange,
    MOE + "reduce_scatter_with_unpadding": _dp_exchange,
    MOE + "reduce_scatterv": _dp_exchange,
    MORI + "dispatch": _mori("input", "weights", "scales", "indices"),
    MORI + "combine": _mori("input", "weights"),
    "_c10d_functional.all_to_all_single.default": _all_to_all,
    V4_INDICES + "decode_indices:build_v4_paged_decode_indptr": _launch,
    V4_INDICES + "decode_indices:write_v4_paged_decode_indices": _launch,
    V4_INDICES + "prefill_indices:write_v4_paged_prefill_indices": _launch,
    FLYDSL + "mqa_logits.pa_mqa_logits_fp4_prefill:compute_prefill_schedule": _launch,
    "aiter.unified_attention_with_output_base.default": _attention,
    "aiter.linear_attention_with_output_base.default": _linear_attention,
    "aiter.silu_and_mul.default": _in_place(),
    "aiter._fused_qk_rmsnorm_group_quant_kernel.default": _in_place(),
    "aiter.mixed_sample_outer_exponential.default": _in_place(),
    "aiter.masked_embedding.default": _gather("weight"),
    "aten.embedding.default": _gather("weight"),
    "aten.index.Tensor": _gather("self"),
    "aiter.all_reduce_.default": _all_reduce,
    "aiter.fused_allreduce_rmsnorm_.default": _fused_all_reduce_norm,
    "_c10d_functional.all_gather_into_tensor.default": _all_gather,
    "_c10d_functional.broadcast.default": _broadcast,
    "_c10d_functional.wait_tensor.default": lambda op, view, tp, kv: None,
    "atom.model_ops.layernorm:fused_qk_norm": _in_place("q", "k"),
    "atom.model_ops.triton_mrope:try_mrope_qk_fused": _in_place("q", "k"),
    "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax": _in_place(),
    # The pick over a vocabulary shard's logits, which the LM head's GEMM, a
    # leaf of its own, wrote: each row's best value and index packed in fp32.
    "aiter.lm_head_argmax_pack.default": lambda op, view, tp, kv: _boundary(
        op, outs=["torch.float32"]
    ),
    "atom.model_ops.decode_input_ids:fill_deferred_decode_ids": _in_place(),
    "atom.utils.block_convert:kv_indices_generate_triton": _kv_indices,
}
#: DeepSeek-V4.1's kinds, gfx1250's and gfx950's. What a lifted leaf writes
#: past its arguments, and the tables and planes it reads by index, are left
#: out, which only lowers M.
V41 = "atom.model_ops.deepseek_v41."
V41_INDICES = "atom.model_ops.attentions.deepseek_v41.indices:"
TOPK = "aiter.ops.flydsl.topk.topk_per_row:"
EXTRACTORS |= {
    "aten.index_fill_.int_Scalar": _index_fill,
    "aiter.gemm_a8w8_blockscale.default": _gemm(
        "XQ", "WQ", "dtype", ("x_scale", "w_scale")
    ),
    "aiter.router_logits.default": _gemm("x", "weight"),
    "atom.model_ops.blockscale:quantize_fp8": _quantize,
    "aiter.ops.triton.quant.fused_mxfp8_quant:fused_dual_rmsnorm_mxfp8_quant": _mx(
        "q", "k", read=("q_weight", "k_weight")
    ),
    "aiter.dynamic_per_token_scaled_quant.default": _only("input", "out", "scales"),
    "aiter.v41_mhc_pre_delayed.default": _mhc_delayed,
    f"{V41}mhc:collapse_streams": _only("residual", "pre_mix"),
    "atom.model_ops.engram.device.gate:engram_post_wkv": _only(
        "hidden", "kv", "weight"
    ),
    "aiter.rope_cached_positions_fwd_impl.default": _only(
        "input", "output", "positions"
    ),
    f"{V41}rope_window:_launch_rope_window": _only("query", "kv", "positions"),
    f"{V4}fused_compress:_launch_fused_compress": _only(
        "kv_in", "score_in", "ape", "rms_weight", "state_slot_mapping", "latent_out"
    ),
    f"{V41}index_write:write_index_rows": _only("keys", "plan"),
    f"{V4}indexer_weights:scale_indexer_weights": _only("weights", "q_scale"),
    "aiter.pa_sparse_prefill_opus.default": _v4_decode(
        "q", "attn_sink", "out", queries=_last
    ),
    **dict.fromkeys(
        (
            FLYDSL + "mqa_logits.fp8_paged_mqa_logits_gfx1250:"
            "flydsl_fp8_paged_mqa_logits_gfx1250",
            "aiter.ops.triton.attention.pa_mqa_logits:deepgemm_fp8_paged_mqa_logits",
        ),
        _v41_scores("q_fp8"),
    ),
    f"{V41}indexer:_block_maxima": _only("visible", "maxima"),
    TOPK + "flydsl_radix_topk_one_block": _only("row_starts", "row_ends", "indices"),
    TOPK + "flydsl_top_k_per_row_decode": _only("seq_lens", "indices"),
    "aiter._top_k_per_row_decode.default": _only("seqLens", "indices"),
    "atom.model_ops.attentions.token_layout.gather:gather_rows": _gather_rows,
    **dict.fromkeys(
        (
            V41_INDICES + "_launch_indptr_scan",
            V41_INDICES + "_launch_indices",
            V41_INDICES + "_launch_window_rows",
            f"{V41}unit_table:unit_table",
            f"{V41}candidate_table:candidate_block_table",
            f"{V41}candidate_table:lift_candidate_selection",
        ),
        _launch,
    ),
}
#: DeepSeek-V4.1's DSpark drafter's kinds: its draft-KV pass and its block.
DSPARK = "atom.model_ops.deepseek_v41.dspark:"
MARKOV = "atom.model_ops.dspark_markov_sample:"
EXTRACTORS |= {
    DSPARK + "fused_draft_kv_tail": _draft_kv_tail,
    DSPARK + "gather_window_rows": _window_rows,
    DSPARK + "draft_step_indices": _draft_indices,
    DSPARK + "build_block_state": _block_state,
    "atom.model_ops.sparse_attn_v4:_sparse_attn_triton": _block_attention,
    MARKOV + "_launch": _markov,
    MARKOV + "argmax_gather": _in_place("out"),
    "atom.spec_decode.dspark_proposer:DSparkProposer._stage_anchors": _only(
        "next_token_ids", "rows"
    ),
    "atom.model_ops.attentions.deepseek_v41.speculative:_launch_commit_cursors": (
        _commit_cursors
    ),
}
#: Every kind whose extractor takes the EP width, as a fifth parameter.
SHARDED = {n for n, f in EXTRACTORS.items() if len(inspect.signature(f).parameters) > 4}


def work(
    op: dict, view: BatchView, tp: int, kv_format: str, ep: int = 1
) -> Work | None:
    """`op`'s work on a step of `view`'s rows in a TP group of `tp` ranks
    whose KV cache holds `kv_format` (a work format, "bf16" or "fp8"), its
    experts sharded over `ep` ranks, or None when it launches no kernel. A
    kind with no extractor, or a leaf its extractor cannot price, is refused."""
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
        return extract(op, view, tp, kv_format, *([ep] if name in SHARDED else []))
    except _Unpriced as error:
        reason = f"the op kind {name}: {error}"
    raise CostRefused(view, [Refusal(SOURCE, reason)])
