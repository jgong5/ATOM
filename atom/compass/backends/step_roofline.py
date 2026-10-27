# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step roofline, and the step-level fallback to it.

`StepRooflineBackend` prices a step from `Module`s, plain descriptors of one
rank's modules that `atom.compass.run.modules.walk` reads off the meta build.
Each module's work comes from its per-rank shapes and the batch (`work`), and
the module costs one roofline, `max(t0, F / peak, M / BW(M))`, plus the
collective it joins; a module whose collective runs inside its own kernel
(`fused`) takes the collective inside that max instead. The step is the sum
over modules, device busy time only: host time is the runner's host
recurrence, which composes it with this price. Nothing is fitted.

Rows. A module that runs per token runs at the rows the step's replayed graph
pads to, `capture_rung` rows of the decode rows' query length, or at the real
rows when built with `padded=False` (the run file's ``padding: none``).
Attention runs over the real rows, a pad row holding no context. The LM head
runs on the rows that sample: each prefill row's last token and every decode
token, none on a step that produces no output. Routed experts run on the
rows the data-parallel group's gather carries (`moe_rows`), and under expert
parallelism on the busiest rank's expected share of them; a graph's pad rows
route as one token.

A prefill that replays its decoder runs a module with a `replay` on each
request that finishes its prompt, on its last `replay` rows, which attend
only each other as their window; on none when the step samples nothing. A view
does not say which rows finish, so one is counted, the least of every row's:
a lower bound. A step with decode rows, a split or a data-parallel gather
replays nothing.

Drafter. A module with a `draft` width is a block drafter's: after the
target, it runs once for each request that samples, on that many tokens:
every decode row, or on a prefill one finishing row, counted as above. One
whose `draft` is `CONTEXT` runs on the target's rows, writing the drafter's
KV, and replays as the target's modules do.

Two-batch overlap. A step the data-parallel group split (`tbo_tokens`) runs
this rank's tokens as two micro-batches, cut at the first one's token count,
a cut request's second part attending the first as history. Each layer's
modules are priced per micro-batch and laid on one compute and one comm
stream in ATOM's order (`overlap`); the finish time is the `roofline.tbo`
term. Modules outside the layers run once over the whole step.

Terms. One per module kind, `roofline.<kind>`. A term is `declared`, its
note naming each input, when a spec field one of its modules read for the
bound that won, or for its collective, is neither from a datasheet nor
probed, and `analytical` otherwise. Its `by_bound` holds each module's kernel
seconds under its largest bound and its collectives' as `comm.latency` and
`comm.transfer`, a fused module's under the largest of all (`work.binding`);
``roofline.tbo``'s is empty. A drafter's modules price the step's
`draft_terms`, ``draft roofline.<kind>``, each at 0 s where it runs nothing,
so a step under a drafter always states its draft.

Refused: a step with image tokens, since nothing here prices the vision
encoder that fills them; any step, while the walk found a module it could not
classify; a step whose modules read a spec field the spec does not state; a
data-parallel MoE step with no `moe_rows`; a split step whose counts name no
single pair for this rank, or whose MoE runs MoRI's split send and receive,
whose overlap no stream carries.

`StepFallbackBackend` asks a backend first and the roofline only for a step
that backend refuses, so a law with no vector for a step still prices it. The
roofline's terms, its draft's too, carry what the first backend declined,
which shows the fall-through in the record; the step is answered unless the
roofline refuses it too. Both backends
are step-level: there is no fallback across granularities, so a first backend
of another granularity is refused at construction.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from itertools import cycle
from math import comb

import numpy as np

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_parts, fold_seconds
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import (
    IMAGES,
    BatchView,
    RequestShape,
    sum_context,
    sum_tokens,
)
from atom.compass.backends.work import (
    COMM,
    ELEMENT_BYTES,
    KERNEL,
    Work,
    _wider,
    activated_experts,
    all_to_all,
    allgather,
    allreduce,
    attention,
    binding,
    bounds,
    elementwise,
    gemm,
    grouped_gemm,
    max_rank_load,
    parts,
    reduce_scatter,
)

SOURCE = "step-roofline"
#: The activations' format: the models priced here run them in bf16.
ACT = "bf16"
#: Kinds whose work is one pass over `dims[0]` elements a token.
PASSES = ("embedding", "norm", "rotary", "conv")
UNCLASSIFIED = "unclassified"
#: The `draft` of a drafter's module that runs on the target's rows.
CONTEXT = -1
DRAFT = "draft "


@dataclasses.dataclass(frozen=True)
class Module:
    """One module of one rank, as the walk read it off the meta build.

    `kind` picks its work; `dims` are its per-rank shapes in that kind's order:
    `gemm` and `lm_head` (n, k), a pass (elements a token; a `norm`'s
    second, those of them it writes in fp8), `act` (output
    elements a token), `attention` (heads, KV heads, head dim),
    `linear_attention` (value heads, key heads, key dim, value dim), `mhc`
    (residual copies, width), `sparse_attention` (heads, head dim, window,
    compression ratio, top k, indexer heads and head dim, then optionally the
    most compressed entries the indexer scores), `mla_attention`
    (heads, the decode kernel's padded heads, latent, rope, no-rope and value
    widths), `mla_up` (heads, no-rope and value widths, latent width), `moe`
    as `run.modules` reads it, `index_keys` (index dim, head dim, compression
    ratio), `block_attention` (heads, head dim, window), `markov` (vocabulary
    shard, rank, block width). `fmt` is its weight, KV or state format, `act` the
    format its matrix products read activations in; `ranks` the group of the
    collective it joins, 1 for none; `fused` that the collective runs in the
    module's kernel; `layer` the decoder layer it sits in, -1 outside them;
    `wire` the formats a `moe`'s all-to-all dispatch and combine send;
    `index` the format a `sparse_attention`'s indexer scores at; `replay`
    the rows of each finishing request a replayed prefill runs it on, 0 for
    every row; `draft` a drafter's block width, `CONTEXT` for a drafter's
    module on the target's rows, 0 for the target's.
    An unclassified module's `name` says what it is.
    """

    name: str
    kind: str
    dims: tuple[int, ...] = ()
    fmt: str = ACT
    ranks: int = 1
    fused: bool = False
    act: str = ACT
    layer: int = -1
    wire: tuple[str, str] = (ACT, ACT)
    index: str = "fp8"
    replay: int = 0
    draft: int = 0


def _work(m: Module, view: BatchView, tokens: int, sampled: int) -> Work | None:
    """`m`'s work on a step of `tokens` executed tokens; None if it runs none."""
    d, act = m.dims, ELEMENT_BYTES[ACT]
    if m.kind == "gemm":
        return gemm(tokens, *d, m.act, m.fmt) + allreduce(m.ranks, tokens * d[0] * act)
    if m.kind == "mhc":
        # Fold the sub-layer's output into the widened residual and reduce it:
        # read and write both, and the mixing weights' fp32 projection.
        copies, width = d
        mix = (2 + copies) * copies
        moved = 2 * tokens * (copies + 1) * width * act
        moved += mix * copies * width * ELEMENT_BYTES["fp32"]
        return Work({ACT: 2 * tokens * mix * copies * width}, moved)
    if m.kind == "sparse_attention":
        return _sparse_attention(m, view.requests)
    if m.kind == "index_keys":
        return _index_keys(m, view.requests)
    if m.kind == "block_attention":
        return _block_attention(m, view.requests)
    if m.kind == "markov":
        return _markov(m, tokens)
    if m.kind == "mla_attention":
        return _mla_attention(m, view)
    if m.kind == "mla_up":
        return _mla_up(m, view, tokens - sum_tokens(view.prefill))
    if m.kind == "lm_head":
        if not sampled:
            return None
        gathered = sampled * d[0] * m.ranks * act
        return gemm(sampled, *d, b=m.fmt) + allgather(m.ranks, gathered)
    if m.kind in PASSES:
        # ponytail: one pass over the module's own width; a per-head norm or
        # rotary touches heads times that, left out of this lower bound.
        width = tokens * d[0]
        moved = elementwise(width, fmt=m.fmt).bytes
        if len(d) > 1:  # the part it writes in fp8, for the product it feeds
            moved -= tokens * d[1] * (ELEMENT_BYTES[m.fmt] - ELEMENT_BYTES["fp8"])
        return Work({}, moved) + allreduce(m.ranks, width * act)
    if m.kind == "act":
        return elementwise(tokens * d[0], reads=2)
    rows = view.requests
    if m.kind == "attention":
        q, kv = [r.query_tokens for r in rows], [r.context_tokens for r in rows]
        return attention(q, kv, d[0], d[2], d[1], kv_fmt=m.fmt)
    # linear_attention: the recurrence reads q, k, v, gate and beta and writes
    # its output once a token, and writes each row's state, reading it where
    # the row has history; two matrix-vector products a token and head.
    vh, kh, dk, dv = d
    real = sum_tokens(rows)
    held = sum(1 for r in rows if r.cached_tokens)
    moved = real * (2 * kh * dk + 2 * vh * dv + 2 * vh) * act
    moved += (len(rows) + held) * vh * dk * dv * ELEMENT_BYTES[m.fmt]
    # ponytail: the state is priced at the bf16 rate, a lower bound on an
    # fp32 accumulation the spec states no rate for.
    return Work({ACT: 4 * real * vh * dk * dv}, moved)


def _pairs(rows) -> int:
    """The query-key pairs of causal attention over `rows`."""
    return sum(
        r.query_tokens * (2 * r.context_tokens - r.query_tokens + 1) // 2 for r in rows
    )


def _mla_attention(m: Module, view: BatchView) -> Work:
    """MLA's attention kernels, as ATOM runs them.

    Decode rows run one kernel over the latent cache, Q and KV at the cache's
    format `fmt`: each of the kernel's padded heads scores a key at the latent
    and rope widths and sums values at the latent width. Q and each request's
    entries are read once, the latent output written in bf16. Prefill rows
    run bf16 attention at the rank's heads over the up-projected keys and
    values of each request's context: QK at the no-rope and rope widths, PV
    at the value width.
    """
    heads, padded, latent, rope, nope, v = m.dims
    work = Work()
    if rows := view.decode:
        q, entry = sum_tokens(rows), latent + rope
        moved = (q * padded + sum_context(rows)) * entry * ELEMENT_BYTES[m.fmt]
        moved += q * padded * latent * ELEMENT_BYTES[ACT]
        work += Work({m.fmt: 2 * padded * (entry + latent) * _pairs(rows)}, moved)
    if rows := view.prefill:
        width = nope + rope + v
        moved = (sum_tokens(rows) + sum_context(rows)) * heads * width
        work += Work(
            {ACT: 2 * heads * width * _pairs(rows)}, moved * ELEMENT_BYTES[ACT]
        )
    return work


def _mla_up(m: Module, view: BatchView, decoded: int) -> Work:
    """MLA's up-projection of the latent KV, at `decoded` decode rows.

    A decode step absorbs it into two fp8 batched GEMMs, one weight a head:
    each query's no-rope part into the latent width, and the attention's
    latent output out to the value width. A prefill step runs the linear, at
    its formats, over each request's whole context, cached part included.
    """
    heads, nope, v, latent = m.dims
    work = Work()
    if view.decode:
        rows = decoded * heads
        work += grouped_gemm(rows, latent, nope, heads, "fp8", "fp8")
        work += grouped_gemm(rows, v, latent, heads, "fp8", "fp8")
    if rows := view.prefill:
        work += gemm(sum_context(rows), heads * (nope + v), latent, m.act, m.fmt)
    return work


def _moe(m: Module, view: BatchView, tokens: int) -> tuple[Work, Work, Work]:
    """The exchange before the routed experts, the experts, and the exchange
    after, routing uniform and independent.

    The experts run on the gathered rows, or under expert parallelism on the
    busiest rank's expected share, in one kernel: it reads each expert a row
    reaches once, weights and scales, and its routes' share of the rows'
    input, output, top-k ids and weights; the gate-up output never leaves it.
    On one data-parallel rank the output is all-reduced over the module's
    `ranks`, 1 where its block leaves that to the next norm. Over a group, the
    hidden states and router logits are gathered and the output scattered
    back; under expert parallelism MoRI sends each of this rank's tokens once
    to every rank holding one of its experts, and back.
    """
    experts, local, k, inter, hidden, ranks, dp, _, held = m.dims[:9]
    shared = m.dims[9] if len(m.dims) > 9 else 0
    rows, act = view.moe_rows if dp > 1 else tokens, ELEMENT_BYTES[ACT]
    real = view.moe_tokens if dp > 1 else sum_tokens(view.requests)
    # Pad rows all carry token 0 at position 0, so they route as one token.
    distinct = rows if real is None else min(rows, real + 1)
    routed, groups = rows * k, activated_experts(experts, k, distinct)
    if local < experts:
        routed, groups = max_rank_load(rows, k, experts, ranks), groups / ranks
    # Every row also runs through each shared expert the kernel holds.
    picks, groups = routed + rows * shared, groups + shared
    flops = grouped_gemm(picks, 2 * inter, hidden, groups, m.act, m.fmt)
    flops += grouped_gemm(picks, hidden, inter, groups, m.act, m.fmt)
    ids = 2 * (k + shared) * ELEMENT_BYTES["fp32"]
    row = hidden * (ELEMENT_BYTES[m.wire[0]] + act) + ids
    work = Work(flops.flops_by_format, groups * held + routed / k * row)
    if dp == 1:
        return Work(), work, allreduce(m.ranks, tokens * hidden * act)
    if local == experts:
        gather = allgather(dp, rows * (hidden + experts) * act)
        return gather, work, reduce_scatter(dp, rows * hidden * act)
    sent = tokens * hidden * ranks * (1 - comb(experts - local, k) / comb(experts, k))
    dispatch = all_to_all(ranks, sent * ELEMENT_BYTES[m.wire[0]])
    return dispatch, work, all_to_all(ranks, sent * ELEMENT_BYTES[m.wire[1]])


def _prefix(n: np.ndarray, ratio: int = 1, cap: int | None = None) -> np.ndarray:
    """`sum(min(x // ratio, cap) for x in 1..n)`, for each `n`."""
    if cap is not None:
        edge = cap * ratio - 1
        return _prefix(np.minimum(n, edge), ratio) + cap * np.maximum(0, n - edge)
    whole = n // ratio
    return ratio * whole * (whole - 1) // 2 + whole * (n - whole * ratio + 1)


def _sparse_reads(rows, window: int, ratio: int, top: int | None = None):
    """V4's attention over `rows`: a query at position `p` reads the last
    `window` tokens and the `p // ratio` compressed entries before it, at most
    `top` of them when `top` is set. Returns the new queries' keys and the KV
    entries each request reads once, summed; then the new queries' compressed
    entries, each of which an indexer scores, and each request's, summed."""
    seen = np.array([r.context_tokens for r in rows])
    held = seen - np.array([r.query_tokens for r in rows])

    def span(**k) -> int:
        return int((_prefix(seen, **k) - _prefix(held, **k)).sum())

    keys, read = span(cap=window), int(np.minimum(seen, window).sum())
    if not ratio:
        return keys, read, 0, 0
    entries = seen // ratio
    keys += span(ratio=ratio, cap=top)
    read += int((entries if top is None else np.minimum(entries, top)).sum())
    return keys, read, span(ratio=ratio), int(entries.sum())


def _sparse_products(rows, heads, dim, q, kv, window, ratio, top=None) -> Work:
    """V4's attention products over `rows` (`_sparse_reads`): two of `dim` per
    head and key, at the wider of Q's format `q` and the KV's `kv`, and the KV
    entries each request reads, once each at `dim` in `kv`."""
    keys, read, _, _ = _sparse_reads(rows, window, ratio, top)
    return Work({_wider(q, kv): 4 * heads * dim * keys}, read * dim * ELEMENT_BYTES[kv])


def _index_scores(rows, ratio, heads, dim, fmt, cap=None) -> Work:
    """V4's indexer over `rows`: each new query scored against every
    compressed entry before it, or its last `cap` where V4.1 scores only a
    candidate set, one product of `dim` per head at `fmt`, and each request's
    entries read once at `dim` in `fmt`."""
    if cap:
        scored, entries, _, _ = _sparse_reads(rows, 0, ratio, cap)
    else:
        _, _, scored, entries = _sparse_reads(rows, 0, ratio)
    return Work({fmt: 2 * heads * dim * scored}, entries * dim * ELEMENT_BYTES[fmt])


def _sparse_attention(m: Module, rows) -> Work:
    """V4's attention (`_sparse_products`) at Q's format `act` and the KV's
    `fmt`, at most the top `k` compressed entries where it has a top k, which
    an indexer scores where it has one, every compressed entry, or at most its
    `cap`, at `index`. Q is read at `act`, O written in bf16."""
    heads, dim, window, ratio, top, index_heads, index_dim = m.dims[:7]
    cap = m.dims[7] if len(m.dims) > 7 else None
    work = _sparse_products(rows, heads, dim, m.act, m.fmt, window, ratio, top or None)
    if ratio and index_heads:
        work += _index_scores(rows, ratio, index_heads, index_dim, m.index, cap)
    q = sum_tokens(rows) * heads * dim
    return work + Work({}, q * (ELEMENT_BYTES[m.act] + ELEMENT_BYTES[ACT]))


def _index_keys(m: Module, rows) -> Work:
    """V4.1's index keys: each compressed entry the step completes projected
    from the head dim to the index dim in bf16, and written once at `fmt`.
    Their norm and rotation are left out of this lower bound."""
    index_dim, head_dim, ratio = m.dims
    made = sum(r.context_tokens // ratio - r.cached_tokens // ratio for r in rows)
    return gemm(made, index_dim, head_dim) + elementwise(
        made * index_dim, reads=0, fmt=m.fmt
    )


def _block_attention(m: Module, rows) -> Work:
    """DSpark's draft attention: each request's block of queries attends every
    key of the block and the last `window` tokens of its context before it.
    Q, O and the block's keys are bf16, the context is read once at `fmt`."""
    heads, dim, window = m.dims
    keys = held = 0
    for r in rows:
        context = min(r.cached_tokens, window)
        keys += r.query_tokens * (context + r.query_tokens)
        held += context
    q, act = sum_tokens(rows), ELEMENT_BYTES[ACT]
    moved = q * (2 * heads + 1) * dim * act + held * dim * ELEMENT_BYTES[m.fmt]
    return Work({_wider(ACT, m.fmt): 4 * heads * dim * keys}, moved)


def _markov(m: Module, tokens: int) -> Work:
    """DSpark's sampler over this rank's vocabulary shard, once for each of
    the block's positions: each request's Markov bias, the shard of a
    (vocabulary, rank) table times its previous token's row, added to the LM
    head's logits there and reduced to one (max, id) pair the TP group
    gathers, in fp32."""
    vocab, rank, width = m.dims
    rows, act = tokens // width, ELEMENT_BYTES[ACT]
    moved = (vocab * rank + rows * vocab + rows * rank) * act
    pick = rows * 2 * ELEMENT_BYTES["fp32"] * m.ranks
    position = Work({ACT: 2 * rows * vocab * rank}, moved) + allgather(m.ranks, pick)
    return sum([position] * width, Work())


def _finishing(view: BatchView) -> tuple[RequestShape, ...]:
    """The prefill rows of `view` that finish their prompts, as far as a view
    says: none when the step samples nothing, else one with the fewest tokens
    and the shortest context of any row."""
    if not view.produces_output or not view.prefill:
        return ()
    rows = view.prefill
    least = RequestShape(
        min(r.query_tokens for r in rows), min(r.context_tokens for r in rows), False
    )
    return (least,)


def overlap(layers) -> float:
    """The finish time of two micro-batches' layers on one compute and one
    comm stream, in ATOM's issue order.

    Each layer is pairs of seconds, one per micro-batch, on the compute and
    comm streams in turn from compute: for a layer of modules, the compute
    before the MoE exchange, the exchange in, the routed experts, the exchange
    out. Within a layer every segment is issued for both micro-batches before
    the next; a segment starts when its stream is free and its micro-batch's
    previous segment is done.
    """
    free = {"compute": 0.0, "comm": 0.0}
    done = [0.0, 0.0]
    for layer in layers:
        for segment, stream in zip(layer, cycle(("compute", "comm"))):
            for k in 0, 1:
                free[stream] = done[k] = max(free[stream], done[k]) + segment[k]
    return max(free.values())


def micro_batches(view: BatchView) -> tuple[BatchView, BatchView]:
    """A split step's two micro-batches on this rank, each with the group's
    rows of it as `moe_rows`; a ValueError names a split that gives none.

    This rank's pair is its `dp_rank` slot of `tbo_tokens`, or, unstated, the
    one pair that sums to its tokens. The step is cut at the first one's
    token count, a cut request's second part attending the first as history.
    """
    own = sum_tokens(view.requests)
    pairs = list(zip(*view.tbo_tokens))
    if view.dp_rank is not None:
        pairs = pairs[view.dp_rank : view.dp_rank + 1]
    pairs = list(dict.fromkeys(p for p in pairs if sum(p) == own))
    if len(pairs) != 1:
        raise ValueError(
            f"the two-batch-overlap split {view.tbo_tokens} names no single "
            f"pair of micro-batches for this rank's {own} tokens"
        )
    # ponytail: the micro-batches run their real rows; a split decode
    # graph's padding is left out of this lower bound.
    left, halves = pairs[0][0], ([], [])
    for r in view.requests:
        first = min(left, r.query_tokens)
        left -= first
        if first:
            halves[0].append(RequestShape(first, r.cached_tokens + first, r.decode))
        if r.query_tokens > first:
            rest = r.query_tokens - first
            halves[1].append(RequestShape(rest, r.context_tokens, r.decode))
    a, b = (
        BatchView(tuple(h), moe_rows=sum(t)) for h, t in zip(halves, view.tbo_tokens)
    )
    return a, b


def _reads(work: Work) -> dict[str, list[str]]:
    """The spec fields each bound of `work`'s price reads, as `work.bounds` does."""
    compute = [
        f"device.compute.{f}"
        for fmt in work.flops_by_format
        for f in (f"{fmt}_flops", f"derate.{fmt}")
    ]
    link = "interconnect.intra_node."
    comm = ["domain_devices", "topology", "link_bandwidth_bytes_per_s", "derate"]
    return {
        "floor": ["device.kernel_floor_s"],
        "compute": compute,
        "memory": ["device.memory.bandwidth_curve"],
        "comm": [link + f for f in comm + ["link_latency_s"]] if work.comm else [],
    }


class StepRooflineBackend(CostBackend):
    """A step as the sum of one roofline per module."""

    def __init__(self, machine, modules, *, padded: bool = True) -> None:
        self.machine = machine
        self.padded = padded
        modules = tuple(modules)
        self.unclassified = [m.name for m in modules if m.kind == UNCLASSIFIED]
        classified = [m for m in modules if m.kind != UNCLASSIFIED]
        target = [m for m in classified if not m.draft]
        # Modules of one shape price alike: count them, price each shape once.
        self.shapes = Counter(dataclasses.replace(m, name="", layer=-1) for m in target)
        self.outside = Counter(
            dataclasses.replace(m, name="") for m in target if m.layer < 0
        )
        self.layered = [m for m in target if m.layer >= 0]
        self.drafted = Counter(
            dataclasses.replace(m, name="", layer=-1) for m in classified if m.draft
        )
        self.count = len(modules)
        self.name = machine.values.get("name", "")
        self.domain = machine.values.get("interconnect.intra_node.domain_devices")
        self._stated_as: dict[str, str] = {}

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def _stated(self, path: str) -> str:
        """`path` and its origin when it is neither a datasheet's nor probed."""
        if path not in self._stated_as:
            origin = self.machine.origin(path)["origin"]
            spec = f"{path} {origin} in the machine spec {self.name}"
            self._stated_as[path] = "" if origin in ("datasheet", "probed") else spec
        return self._stated_as[path]

    def _unit(self, m: Module, view: BatchView, tokens: int, sampled: int):
        """`m`'s seconds as (exchange before, the rest, exchange after) with
        the notes of the inputs that decided them and its seconds by bound,
        a refusal's text, or None if `m` runs nothing on this step."""
        if m.kind == "moe" and m.dims[6] > 1 and view.moe_rows is None:
            return f"a MoE over {m.dims[6]} data-parallel ranks, and no moe_rows"
        if m.kind == "moe":
            pieces = _moe(m, view, tokens)
        else:
            pieces = (Work(), _work(m, view, tokens, sampled), Work())
            if pieces[1] is None:
                return None
        work = sum(pieces, Work())
        reads = _reads(work)
        missing = [
            p for ps in reads.values() for p in ps if p not in self.machine.values
        ]
        if missing:
            return (
                f"{m.kind} reads {', '.join(dict.fromkeys(missing))}, "
                "which the machine spec does not state"
            )
        if work.comm and m.ranks > self.domain:
            return (
                f"a {m.kind} collective over {m.ranks} ranks leaves the "
                f"scale-up domain of {self.domain} devices"
            )
        p = parts(work, self.machine)
        kernel = max(KERNEL, key=p.get)
        won = [kernel, "comm"] if p[COMM[0]] + p[COMM[1]] else [kernel]
        notes = [s for w in won for f in reads[w] if (s := self._stated(f))]
        if m.fused:
            by = binding(work, self.machine)
            return (0.0, fold_seconds(by.values()), 0.0), notes, by
        comm = [bounds(w, self.machine)["comm"] if w.comm else 0.0 for w in pieces]
        by = {kernel: p[kernel]} | {k: p[k] for k in COMM if p[k]}
        return (comm[0], p[kernel] + comm[1], comm[2]), notes, by

    def _overlapped(self, view: BatchView, refuse, priced, stated) -> str:
        """Price the layers of a split step as two micro-batches on two
        streams into `priced["tbo"]`, or add why not to `refuse`."""
        if any(m.kind == "moe" and m.dims[7] for m in self.layered):
            refuse.append(
                "a two-batch-overlap step on MoRI's split send and receive "
                "(--all2all-backend low-latency): its transfer in flight lies "
                "on no stream"
            )
            return ""
        try:
            views = micro_batches(view)
        except ValueError as exc:
            refuse.append(str(exc))
            return ""
        layers: dict[int, list[list[float]]] = {}
        units: dict = {}
        notes = stated.setdefault("tbo", [])
        for m in self.layered:
            shape = dataclasses.replace(m, name="", layer=-1)
            segments = layers.setdefault(m.layer, [[0.0, 0.0] for _ in range(4)])
            for k, half in enumerate(views):
                if (shape, k) not in units:
                    unit = self._unit(shape, half, sum_tokens(half.requests), 0)
                    units[shape, k] = unit
                    if isinstance(unit, str):
                        refuse.append(unit)
                    elif unit:
                        notes += unit[1]
                unit = units[shape, k]
                if not unit or isinstance(unit, str):
                    continue
                if m.kind == "moe":
                    for s, seconds in zip(segments[1:], unit[0]):
                        s[k] += seconds
                else:
                    segments[0][k] += sum(unit[0])
        priced["tbo"] = [overlap(layers[n] for n in sorted(layers))]
        return " and ".join(str(sum_tokens(v.requests)) for v in views)

    def price(self, batch_view: BatchView) -> StepCost:
        refuse = []
        if batch_view.image_tokens:
            refuse.append(IMAGES)
        if self.unclassified:
            refuse.append("unclassified modules: " + ", ".join(self.unclassified))
        if refuse:
            raise CostRefused(batch_view, [Refusal(SOURCE, r) for r in refuse])
        rows = batch_view.requests
        tokens = sum_tokens(rows)
        decode = batch_view.decode
        if self.padded and batch_view.capture_rung:
            pad = batch_view.capture_rung - len(decode)
            tokens += pad * max(r.query_tokens for r in decode)
        sampled = 0
        if batch_view.produces_output is not False:
            sampled = len(batch_view.prefill) + sum_tokens(decode)
        target: tuple[dict, dict, dict] = ({}, {}, {})
        split = batch_view.tbo_tokens is not None
        for m, count in (self.outside if split else self.shapes).items():
            at = self._at(m, batch_view, tokens)
            self._add(m, count, at, sampled, refuse, target)
        if split:
            split = self._overlapped(batch_view, refuse, target[0], target[2])
        drafts: tuple[dict, dict, dict] = ({}, {}, {})
        for m, count in self.drafted.items():
            self._add(m, count, self._at(m, batch_view, tokens), 0, refuse, drafts)
        if refuse:
            reasons = dict.fromkeys(refuse)
            raise CostRefused(batch_view, [Refusal(SOURCE, r) for r in reasons])
        tbo = {"tbo": f"the layers' two micro-batches of {split} tokens overlapped"}
        terms = self._terms(target, "", f"at {tokens} tokens", tbo)
        draft = self._terms(drafts, DRAFT, "of the drafter")
        return StepCost(terms, draft_terms=draft)

    def _at(self, m: Module, view: BatchView, tokens: int):
        """The batch `m` runs on in this step and its tokens, or None if it
        runs on none: a drafter's block, a replayed prefill's finishing rows,
        or the step's own."""
        if m.draft > 0:
            rows = view.decode or _finishing(view)
            if not rows:
                return None
            width = m.draft
            block = tuple(
                RequestShape(width, r.context_tokens + width, True) for r in rows
            )
            return BatchView(block), width * len(rows)
        grouped = view.tbo_tokens is not None or view.moe_rows is not None
        if not m.replay or view.decode or grouped:
            return view, tokens
        rows = _finishing(view)
        if not rows:
            return None
        tail = min(rows[0].query_tokens, m.replay)
        return BatchView((RequestShape(tail, tail, False),)), tail

    def _add(self, m: Module, count: int, at, sampled: int, refuse, sums) -> None:
        """Add `count` times `m`'s price on `at` (`_at`) to `sums`, its
        seconds, seconds by bound and notes by kind, or why not to `refuse`. A
        drafter's kind is added at 0 s where it runs nothing."""
        priced, by, stated = sums
        unit = at and self._unit(m, *at, sampled)
        if isinstance(unit, str):
            refuse.append(unit)
            return
        if m.draft:
            priced.setdefault(m.kind, [])
        if unit:
            priced.setdefault(m.kind, []).append(sum(unit[0]) * count)
            by.setdefault(m.kind, []).append({k: s * count for k, s in unit[2].items()})
            stated.setdefault(m.kind, []).extend(unit[1])

    def _terms(self, sums, prefix: str, where: str, details=None) -> list[CostTerm]:
        """One term a kind of `sums` (`_add`), named `prefix` then
        ``roofline.<kind>``, its detail the kind's in `details` if any."""
        priced, by, stated = sums
        none = "; no successor: the roofline is the last step-level source"
        terms = []
        for kind, seconds in priced.items():
            note = "; ".join(dict.fromkeys(stated.get(kind, ())))
            method = Method.DECLARED if note else Method.ANALYTICAL
            detail = (details or {}).get(kind) or f"sum over {kind} modules {where}"
            provenance = Provenance(method, detail, SOURCE, note=note and note + none)
            terms.append(
                CostTerm(
                    f"{prefix}roofline.{kind}",
                    fold_seconds(seconds),
                    provenance,
                    by_bound=fold_parts(by.get(kind, ())),
                )
            )
        return terms

    def describe(self) -> str:
        return f"step roofline; {self.count} modules"


class StepFallbackBackend(CostBackend):
    """`first`, and the roofline for a step `first` refuses."""

    def __init__(self, first: CostBackend, roofline: StepRooflineBackend) -> None:
        if first.granularity is not roofline.granularity:
            raise ValueError(
                f"a {first.granularity.value}-level backend cannot fall back to "
                f"a {roofline.granularity.value}-level one: fallback runs only "
                "inside one granularity"
            )
        self.first = first
        self.roofline = roofline

    @property
    def granularity(self) -> Granularity:
        return self.first.granularity

    def price(self, batch_view: BatchView) -> StepCost:
        try:
            return self.first.price(batch_view)
        except CostRefused as refused:
            declined, cost = refused.declined, self.roofline.price(batch_view)
        fell = [
            [
                dataclasses.replace(
                    t, provenance=dataclasses.replace(t.provenance, refusals=declined)
                )
                for t in terms
            ]
            for terms in (cost.terms, cost.draft_terms)
        ]
        return StepCost(fell[0], draft_terms=fell[1])

    def describe(self) -> str:
        return f"{self.first.describe()}; below it, {self.roofline.describe()}"
