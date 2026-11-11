# SPDX-License-Identifier: MIT
"""`compass measure ops`: each leaf of an `op_graph` timed alone on one device, cold.

    compass measure ops (--width W | --prefill T B)... [--context C] --llc-mib M
        --gpu G --store S --out DIR -- <ATOM args>
    compass measure ops --plan RECORD [--plan RECORD]... --llc-mib M --gpu G
        --store S --out DIR -- <ATOM args>
    compass measure ops --coverage RECORD [--coverage RECORD]... --store S
        --out DIR -- <ATOM args>

Each shape names the `op_graph` `compass trace ops` published in S for the
model and TP width ATOM's arguments resolve, and under a speculative drafter
each graph of the drafter's part of a step S holds at that shape (`DRAFTED`,
`leaves_at`), timed on the target's batch. A point is one leaf of those graphs,
its instances grouped (`leaves`), at the shape and `--context`: a decode
step's rows each hold `--context` tokens, under a drafter verifying as many
more as it drafts, and a prefill's rows, split as the capture split them,
follow `--context` cached tokens each, none when it is not given; there
the planes of DeepSeek-V4.1's FP8 indexer, which the capture sized to its own
rows' KV blocks, are as wide as the point's rows' (`PAGE_WIDE`). Rows the KV
pool cannot hold at once are ragged (`point_rows`), and the point records its
rows; under `--plan` over ragged decode steps a decode point
the pool cannot hold refuses instead, past the first round (`RAGGED`).

ATOM's runner is built from ATOM's arguments on each GPU `--gpu` names, one
rank each of the layout the arguments resolve (`ranks`), its parameters on the
meta device and with no warmup (`_runner`): no forward of the model, its LM
head or its sampler runs. Each point reaches the runner as a batch
(`ReplayRule`) that `ModelRunner.prepare_model` prepares, each row's state
first seated at its cached context (`replay.seat_states`), so ATOM builds its
metadata and forward context; for DeepSeek-V4, the CSA top-k section its
forward writes is written from ids spread over each token's compressed rows
(`Stub._top_k`); for DeepSeek-V4.1's FP8 index plane, each table an earlier
leaf of its forward writes (`_WRITTEN`) is written for the reading leaf's pass
as that leaf writes it (`Stub._index`), and the logits plane its scorer
writes, which a top-k or the pick's block maxima ranks (`PAGE_WIDE`), is
each copy's own, drawn for a top-k or block maxima as an activation is,
below each row's bound (`_scores`). A `Stub` calls each leaf from its
contract: a tensor the capture states on the host is built there on every
copy, as an activation is, pinned (`_host`); an
activation is random at its shape, dtype and stride, except such
a table, the placement's, a router's
top-k expert ids, drawn uniformly on every copy, past top-k the shared
experts' ids ATOM fuses into them (`_route`), the `NM_OUTPUT` slot at
one KV split, a view at the start of twice its span, and aiter's top-k
scratch (`TOPK_WORKSPACE`), one byte in the graph, built as aiter builds it
on the device for the call's rows, row stride, k and path
(`_topk_workspace`); a weight is random at its
shape, dtype and stride, one tensor per occurrence of the leaf in the step;
a metadata tensor is a
view of ATOM's live metadata (`meta`), whose name the point records, or of
the token ids ATOM prepared (`TOKEN_IDS`), or of the field of DeepSeek-V4's
HCA decode workspace an `HCA_DECODE` slot names, or of the buffer
`V4_METADATA` names for a DeepSeek-V4 slot, and on a V4 prefill's step
`V4_SHARED` for one of an operator V4.1 runs too, or beside a DeepSeek-V4
CSA layer `V4_CSA_SHARED` for one of an operator other models run too, or
ATOM's batch-id build's repeats (`BATCH_IDS`), refused when it is not live,
a prefill's index list whole (`V4_LISTS`), or,
when live, `V41_NAMES`' for a DeepSeek-V4.1 slot at its ratio (`_ratio`),
whose step and scorer workspace are live too; a leaf of the drafter's alone
takes what ATOM passes a drafter (`DRAFTER_NAMES`) and, of a pass of its own,
the entry of ATOM's `batch_info` stating its real rows; a leaf of a replaying
prefill's tail rows that runs more than once takes, on a copy past its first
occurrence (copy c mod its occurrences), what layer 39 passes a `QUERIED`
slot; an integer the capture
found affine in a decode step's context is taken at the point's, and the
batch-id build's length is the point's KV tokens (`_summed`); a module or
context key is the one each instance names, its module's meta parameters
moved to the device and filled as `--load_dummy xavier` fills them (`_fill`),
and a runner attribute this runner's own; a KV tensor is a view of a tensor
ATOM's KV allocation bound, on the runner or a metadata builder (`_bound`),
but no view a builder cut from the runner's (`_cut_from`),
of its layout or `V41_NAMES`' (`_kv`), refused when none is, whose
name the point records, the same on every copy; an object is rebuilt from its class
and state as the capture's replay rebuilds one, its tensors built as any other
argument's, its activations fresh on every copy, a lifted call's sequence a tuple.
A leaf whose operator cannot run, or that copies to the host, refuses by name.
On one GPU at a DP-attention layout, the runner is one DP rank's own, TP 1 and DP 1, the
graphs read and the keys written keep the layout, and a collective leaf
refuses: it is measured across the layout's ranks, where a DP all-gather's
`x` holds each rank's own `sizes` entry's rows, the capture's being rank 0's
(`_own_rows`).
A collective leaf is one family over its message bytes, decode and prefill
alike (`exchange`), which the plan ladders, densifies and covers.

Cold, by copies. A call's bytes are its tensor arguments' spans, a KV
argument's as many entries of its widest-strided dimension, on a tie the
first of size above one, each the span of the others, as the batch's rows,
or its blocks when the model keeps no
per-request state, or, of a decode attention leaf that states its layer's
window and compress ratio, the plane rows the batch's blocks hold at that
ratio, a pointer table's (`_CACHE`) its entries, a view operator's none, a
RoPE table's (`TABLES`), a weight or a tensor of a module it names, a row a
position its call takes, else a token, a table a leaf gathers by token ids
(`GATHERED`) as many rows as its ids, aiter's top-k scratch (`TOPK_WORKSPACE`)
the bytes it launches with, but none of the multi-block path's one cached
buffer, which every copy launches and the point's footprint counts once, the
other tensors of each module it names,
of one whose first dimension is the pool's blocks the batch's blocks, and, for
a leaf that names a context key, the KV and
per-request state its layer holds for the batch, a state slot a row, but at a
decode its recurrent state (`SNAPSHOTS`) at each slot a row holds, which under
a drafter is one a token. One graph runs the leaf K
times, K `CALLS` or the calls that fill `GRAPH_NS` at the device time of the
kernels of one eager call of copy 0, the fewer, at least one, cycling over its
copies: N built ones: M, the least count whose bytes pass `COLD` times the
last-level cache, raised for a context key to its I instances (at most K),
then, up to the leaf's occurrences when it takes a weight other than a
gathered table (at most K), copy
c mod N with the c-th weights on that copy's other arguments.
Copy c < N takes fresh activations below M and copy c mod M's from there,
weight c mod the more of M and the leaf's occurrences, a gathered table's one
weight, its ids rows c n to (c + 1) n of one seeded permutation of the table's
rows, n the ids' count, and
instance c mod I at placement (c div I) mod P, its metadata and module
tensors cloned past the first round: a placement is the point's batch on KV
blocks and state slots of its own, as many as the pool and a KV registry
entry of per-request state hold, past what the state slots hold on the first
placements' slots again, for which ATOM prepares the forward context
again, it and the metadata views cloned once each, and only where they differ
from the last placement's; a view an operator writes (`_writes`) has clones
apart, which only that operator's leaves share. A KV argument
places a leaf's copies as a context key does. A second graph runs
copy 0 K times, warm; with one copy, or K under `CALLS`, there is none.
Before each capture, and once the runner sized its KV pool, the memory the
bench freed goes back to the driver, and the bench waits, at most `CLEAR_S`,
until the card's VRAM in use falls by it (`_release`); a point records the
seconds waited since the last point and how many waits the cap ended.
Each graph is replayed `WARMUPS` times, then `REPEATS` times timed, the cold
one each time after a read of `COLD` times the last-level cache outside the
timing, the host idle `REST` times a replay's device time after each replay;
a repeat is the replay's device time over K, and a graph that captured no
kernel times 0. The kernels one eager call of copy 0 launches, memory copies
aside, are the point's kernel identity, read from two profiles in a row that
name the same; a profile that recorded nothing of a call whose cold graph
timed device work missed it, and the call is profiled again, refused when no
two profiles name its kernels. When `ReplayRule` refuses a batch, its
leaves refuse; an error in ATOM's preparation fails the run, naming the point.
The points run in a process of their own on each GPU, which dies when its
device faulted (`_alive`); one that dies ends every rank, keeps the points
every rank finished, the point it was measuring refuses, and new ones measure
the rest; two in a row that die having finished no point fail the run.

DIR gets run.json, the run's identity, and ops.jsonl, one line per point. S
gets each measured point as an `op_measurement`, keyed by the model, the
parallel layout, the digest of the machine spec of the device, stack and PTL
state run.json reads, the source root and the point.

`--plan` measures, in rounds, the points the steps of record directories call
for (`record`'s or `trace steps`' output), under `--enable-tbo` a step its DP
group split as its two micro-batches, as `trace ops --population` traces it
(`population_steps`), capturing each `op_graph` S lacks on the run's first GPU
as it reads, a prefill as one forward (`_capture`), or, before any point,
refusing one only that call traces:
every leaf at every rung of ATOM's decode ladder, and under a speculative
drafter each leaf of its own decode shape past a step's (`parts`) that the
graphs at its width, its B, do not hold, at its rows padded to that width,
at the steps' lowest and highest row context (`_rows`); then each leaf of a
kind those show depends on context (`_dependent`) over a ladder (`_ladder`)
of the steps' row contexts, and, when some step's rows are ragged, at a
shape of more than one row on ragged rows too (`_means`): the first at the
context, the rest sharing a mean, each `GRID`th of it from the least the rows
hold; and each prefill family (`_family`), under a speculative drafter each
leaf of its own prefill shape past a step's that the step's graphs do not
hold too, at its rows padded to its W, over a ladder of their token counts,
chunk remainders among them,
its shapes that state the rows a DP group gathers apart, as `fit ops` prices them,
one of a kind that depends on context at each at the most cached tokens of
steps of its rows, and over a ladder of them once those points and the fresh
ones show the kind still does; then the point halfway between two neighbours
of a family, or of a prefill leaf's token count over cached tokens, that
launched different kernels or, at a decode width, either side of a point off
the line through theirs, or, on rows alike, whose cold times differ where the
leaf's kind steps under one kernel between two adjacent contexts at some width
(`_densify`), until they are adjacent; a collective's message bytes bend as a
decode width does; and then, at the rows of each prefill family of a kind that
depends on context, each token count its leaf's families at other rows hold
between its own least and most, at each of its cached-token counts
(`_filled`), densifying again after. DIR gets each round's ops.jsonl in `round<n>`, a run
resumed into DIR numbering on from the last round there, and after each round
every point held, a measured one with its store token, in plan.jsonl.

`--coverage` counts the steps of record directories, read as `--plan` reads
them, that DIR's plan.jsonl covers (`coverage`), with the `op_graph`s of the
arch DIR's run.json records, a step only when `fit ops`' price list of those
points prices each leaf, and prints the count by phase, by part of the step
(its target's graph and its drafter's, `parts`) and why each leaf kind left
the rest out, a drafter's kinds named as a run names their terms.
"""

import argparse
import bisect
import contextlib
import copy
import dataclasses
import datetime
import functools
import gc
import hashlib
import importlib
import inspect
import itertools
import json
import multiprocessing
import os
import re
import shlex
import statistics
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from atom.compass import ir
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    member_name,
    roots_for,
)
from atom.compass.backends.op_graph import DRAFT, DRAFT_KV, KV, PERMILLE
from atom.compass.bench.routing import uniform_ids
from atom.compass.drive.gpu import add_gpu_flag
from atom.utils.arg_parser import FlexibleArgumentParser

PROG = "compass measure ops"
#: The timing protocol's version; it changes whenever a count below, the
#: copies a graph reads, what a capture waits for or what a replay follows
#: change.
PROTOCOL = 28
WARMUPS = 3
REPEATS = 20
#: Calls a graph holds, and so the copies it cycles over at most. One count for
#: every leaf a graph of which fits `GRAPH_NS`: a replay's fixed cost adds the
#: same share to every such price.
CALLS = 256
#: Device time a graph's calls fill at most: a longer call takes fewer calls,
#: at least one, so a capped graph holds at least 5 ms: a replay's fixed cost,
#: about 19 us on MI308X, is under 0.4% of it.
GRAPH_NS = 10_000_000
#: Profiles of one call taken at most until two in a row name the same kernels:
#: a profile can drop a call's kernel or keep a kernel launched before it.
PROFILES = 4
#: How the profiler's name of a memory copy's device activity starts.
COPY = "Memcpy"
#: Times over the last-level cache the copies' bytes pass, and the bytes read
#: before each cold replay: just past it, part of them stays there, and an
#: MI308X GEMM read up to 7% fast.
COLD = 2
#: Host idle after each replay, as a share of its device time. Replayed back to
#: back on an MI308X, the lm_head GEMM ran at a 1100 MHz memory and 1550 MHz
#: fabric clock, against 1300 and 1800 MHz with an idle after each call, at the
#: same socket power, and read up to 2.55% over ATOM's decode step by block
#: median. A power limit is the suspected cause. With this idle after a replay
#: `GRAPH_NS` bounds, it read 0.22% to 0.58% under the step.
REST = 0.1
#: Seconds a release waits at most for the driver to clear what it returned
#: (`_release`), and the VRAM in use over the release's target that ends it:
#: on MI308X a release of 78 to 208 MiB left VRAM in use where it was for 3 s,
#: while 2.4 to 51 GiB fell by the release's bytes.
CLEAR_S, CLEAR_SLACK = 3.0, 256 << 20
#: The byte boundary each weight a leaf cycles starts on: PyTorch's caching
#: allocator's block size, where an ATOM parameter allocated alone starts.
#: aiter's Mega-MHC launcher aborts the process on an `hc_scale` that is not
#: 16-byte aligned, as three in four three-float slices packed back to back are.
WEIGHT_ALIGN = 512
#: The release waits since a point last recorded them: seconds, and how many
#: ended at `CLEAR_S`.
_waited: Counter = Counter()
#: Metadata arguments, by operator and slot, that are the token ids ATOM passes
#: the model's forward; `_meta`'s name rule binds an embedding's `indices` to
#: another index buffer.
TOKEN_IDS = {
    ("aten.embedding.default", "indices"),
    ("aiter.masked_embedding.default", "x"),
}
#: ATOM's batch-id build (`build_batch_ids_device`), the one repeat its input
#: preparation runs, past cached tokens or under TBO, by operator and slot, and
#: what it repeats each row's id by: the row's context, `context_lens` widened
#: to int64 as the build widens it, into their sum (`_summed`). Bound by size,
#: a point past cached tokens repeated another int64 buffer into the length of
#: the capture, which caches none, and its measuring process died.
BATCH_IDS = {("aten.repeat_interleave.self_Tensor", "repeats"): "context_lens.long"}
#: Weight slots, and a named module's tensors, a leaf reads a row a token of:
#: RoPE's cos and sin tables, indexed by each position its call takes, at most
#: one a token, or without them each token's, never read whole.
TABLES = {"cos_cache", "sin_cache", "cos", "sin"}
#: A weight slot a leaf reads a row a token id of, and the slot of the ids, by
#: operator: the embedding and DeepSeek-V4's hash routing table.
GATHERED = {
    "aten.embedding.default": ("weight", "indices"),
    "atom.model_ops.triton_hash_topk:hash_topk_triton": ("tid2eid", "ids"),
}
#: The operator whose metadata slots are named as the fields of DeepSeek-V4's
#: HCA decode workspace: aiter defines its schema and the workspace together.
HCA_DECODE = "aiter.mla_decode_v4_ps_asm.default"
#: DeepSeek-V4's metadata arguments, by operator and slot, a dict argument's
#: item by its key (`slot.<key>`), and the live buffer ATOM passes each, at the
#: compress ratio the leaf takes or states (`{ratio}`), that ratio's
#: attention kind (`{kind}`) and its destination rows' class (`{dest}`):
#: `_meta`'s rules bind them to a tighter buffer of their dtype. A leaf whose
#: buffer is not live refuses.
#: A decode's `split_indptr` is its split plan's, `split_plan_<kind>.1`. The
#: paged-decode index leaves write their indptrs into the clone that the
#: leaves timed after them at a placement read. A prefill's index build takes
#: each row's cached tokens, which ATOM uploads apart each step and holds
#: nowhere: `num_cached_tokens` is the batch's (`Stub._prepare`).
_NM = "aiter.mla:mla_decode_fwd_v4_nm"
_INDICES = "atom.model_ops.v4_kernels.paged_decode_indices"
_WRITE = f"{_INDICES}:write_v4_paged_decode_indices"
_CSA = "aiter.ops.flydsl.kernels.fused_compress_attn:flydsl_fused_compress_attn"
_HCA = "aiter.ops.flydsl.kernels.fused_compress_attn_hca:flydsl_hca_compress_attn"
_STATE = "atom.model_ops.v4_kernels.state_writes:update_compressor_states"
_QKN = "aiter._fused_qk_norm_rope_group_quant_kernel.default"
_QKN_BF16 = "aiter.ops.flydsl.kernels.qk_norm_rope_quant:flydsl_qk_norm_rope_quant"
_BF16 = "atom.model_ops.v4_kernels.paged_decode:_sparse_attn_v4_paged_decode_triton"
_FILL = "atom.model_ops.v4_kernels.paged_prefill_indices:write_v4_paged_prefill_indices"
_PREFILL = "aiter.pa_sparse_prefill_fp8_opus.default"
_DESTS = {0: "dense", 4: "csa", 128: "hca"}
V4_METADATA = {
    (_NM, "kv_indptr"): "kv_indptr_{kind}",
    (_NM, "split_indptr"): "split_plan_{kind}.1",
    (_WRITE, "state_slot_per_seq"): "v4_meta_state_slot_out",
    **{
        (_WRITE, f"dest_rows.{ratio}"): f"v4_swa_dest_{dest}"
        for ratio, dest in _DESTS.items()
    },
    **{
        (f"{_INDICES}:{fn}", f"{kind}_indptr"): f"kv_indptr_{kind}"
        for fn in ("build_v4_paged_decode_indptr", "write_v4_paged_decode_indices")
        for kind in ("swa", "csa", "hca")
    },
    (HCA_DECODE, "kv_indptr"): "kv_indptr_{kind}",
    (_BF16, "kv_indptr"): "kv_indptr_{kind}",
    (_CSA, "state_slot_mapping"): "state_slot_in",
    (_CSA, "plan_gpu"): "v4_compress_plan_{ratio}",
    (_HCA, "state_slot_mapping"): "state_slot_in",
    (_HCA, "plan_gpu"): "v4_compress_plan_{ratio}",
    (_STATE, "state_slot_mapping"): "state_slot_out",
    (_STATE, "write_plan"): "v4_write_plan_{ratio}",
    (_QKN, "batch_id_per_token"): "batch_id_per_q_token",
    (_QKN, "swa_dest_row"): "v4_swa_dest_{dest}",
    (_QKN_BF16, "batch_id_per_token"): "batch_id_per_q_token",
    (_QKN_BF16, "swa_dest_rows"): "v4_swa_dest_{dest}",
    (_FILL, "bid_per_token"): "batch_id_per_q_token",
    (_FILL, "chunk_start_per_seq"): "num_cached_tokens",
    (_FILL, "state_slot_per_seq"): "state_slot_out",
    **{
        (_FILL, f"{kind}_{part}"): f"kv_{part}_{kind}"
        for kind in ("extend", "prefix_swa", "prefix_csa", "prefix_hca")
        for part in ("indptr", "indices")
    },
    (_PREFILL, "kv_indptr_prefix"): "kv_indptr_prefix_{kind}",
    (_PREFILL, "kv_indices_prefix"): "kv_indices_prefix_{kind}",
    (_PREFILL, "kv_indptr_extend"): "kv_indptr_extend",
    (_PREFILL, "kv_indices_extend"): "kv_indices_extend",
}
V4_KINDS = {0: "swa", 4: "csa", 128: "hca"}
#: DeepSeek-V4's arguments, keyed as `V4_METADATA`'s, of an operator
#: DeepSeek-V4.1 runs too on buffers of its own (`V41_NAMES`): its prefill
#: attention under the bf16 KV cache, passed the fp8 one's buffers. They bind
#: as `V4_METADATA`'s on a step whose metadata holds V4's prefill lists.
V4_SHARED = {
    ("aiter.pa_sparse_prefill_opus.default", slot): name
    for (op, slot), name in V4_METADATA.items()
    if op == _PREFILL
}
#: DeepSeek-V4's prefill index lists, which ATOM allocates each step at the
#: entries its rows read (`_build_paged_prefill_meta`). A capture's prefill
#: caches nothing, so past cached tokens each is longer than the leaf's
#: captured one: a slot `V4_METADATA` binds to one takes it whole, as ATOM
#: passes it, and aiter's prefill attention splits its KV by its length. Its
#: bytes still count at the captured span.
V4_LISTS = {
    f"kv_indices_{kind}"
    for kind in ("extend", "prefix_swa", "prefix_csa", "prefix_hca")
}
#: DeepSeek-V4.1's arguments, keyed as `V4_METADATA`'s, and the buffers ATOM
#: passes each (`_named`), at the leaf's ratio (`_ratio`), as `_bound` names
#: them: the paged cache's, the pass's step (`step.`), the decoder plan
#: (`plan.`), the scorer workspace (`workspace.`), the forward context.
_CACHE = "attn_metadata_builder.cache.indptr_buffers.{ratio}"
_LOGITS = (
    "aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx1250:"
    "flydsl_fp8_paged_mqa_logits_gfx1250"
)
_PLAN = "step.plans.{ratio}.compress_plan_gpu"
_TOPK = "aiter.ops.flydsl.topk.topk_per_row:flydsl_radix_topk_one_block"
_INDPTRS = "atom.model_ops.attentions.deepseek_v41.indices"
_SCAN = f"{_INDPTRS}:_launch_indptr_scan"
_INDICES = f"{_INDPTRS}:_launch_indices"
_FUSED = "atom.model_ops.v4_kernels.fused_compress:_launch_fused_compress"
_ROPE = "aiter.rope_cached_positions_fwd_impl.default"
_FROM = ("replay_start", "cached_until")
_SWA = "atom.model_ops.v4_kernels.state_writes:swa_write"
_GATHER = "atom.model_ops.attentions.token_layout.gather:gather_rows"
V41_NAMES = {
    ("aiter.pa_sparse_prefill_opus.default", "kv_indptr_prefix"): f"{_CACHE}.0",
    ("aiter.pa_sparse_prefill_opus.default", "kv_indptr_extend"): f"{_CACHE}.1",
    (_SCAN, "pptrs"): f"{_CACHE}.0",
    (_SCAN, "eptrs"): f"{_CACHE}.1",
    (_LOGITS, "out_logits"): "workspace._logits",
    (_LOGITS, "context_lens"): "step.visible.{ratio}",
    (_LOGITS, "kv_indices"): "workspace._tiles.{ratio}",
    (_TOPK, "row_starts"): "step.visible.{ratio}",
    (_TOPK, "row_ends"): "step.visible.{ratio}",
    (_FUSED, "plan_gpu"): _PLAN,
    ("atom.model_ops.deepseek_v41.index_write:write_index_rows", "plan"): _PLAN,
    (_INDICES, "pptr"): f"{_CACHE}.0",
    (_INDICES, "eptr"): f"{_CACHE}.1",
    **{(f, s): f"step.swa_{s}" for f in (_INDICES, _SCAN) for s in _FROM},
    (f"{_INDPTRS}:_launch_window_rows", "eptr"): f"{_CACHE.format(ratio=0)}.1",
    (_FUSED, "state_slot_mapping"): "step.slots",
    (_SWA, "state_slot_per_seq"): "step.slots",
    (_ROPE, "positions", "query"): "step.positions",
    (_ROPE, "positions", "key"): "step.plans.{ratio}.key_rope_positions_gpu",
    ("aten.index_select.default", "index", "kv"): "step.slots",
    ("aten.index_select.default", "index", "metadata"): "plan.last",
    ("aten.index.Tensor", "indices"): "plan.query_ptrs",
    ("aten.index_put_.default", "indices"): ("plan.tail_rows", "plan.last_rows"),
    (_GATHER, "rows"): ("plan.tail_rows", "plan.last"),
    **{
        (fn, f"{field}_state"): f"attn_metadata_builder.cache.state.compress_{field}"
        for fn in (_FUSED, _STATE)
        for field in ("kv", "score")
    },
    ("aiter.fused_moe_.default", "num_local_tokens"): "scheduled_tokens_gpu",
    ("aiter.topk_gating_fwd.default", "num_valid_rows"): "scheduled_tokens_gpu",
}
#: DeepSeek-V4's arguments, keyed and bound as `V4_METADATA`'s, to operators
#: other models run too, which a runner binds so only beside a DeepSeek-V4 CSA
#: layer (`Stub.csa`): its window write's state slots, which DeepSeek-V4.1
#: passes its step's (`V41_NAMES`), and its FP8 indexer's prefill row starts,
#: `cu_committed[batch_id_per_q_token]`. Bound by size, the slots were a count
#: and the gather's indices another index list, past the slots and the rows.
V4_CSA_SHARED = {
    (_SWA, "state_slot_per_seq"): "state_slot_out",
    ("aten.index.Tensor", "indices"): "batch_id_per_q_token",
}
#: A drafter's arguments, keyed as `V4_METADATA`'s, that ATOM passes it from
#: buffers no leaf of a target takes: each row's anchor among its sampled
#: tokens on a step that scored no draft, at the runner's zero rejections, and
#: the rows of DeepSeek-V4.1's replayed tail, which its draft-KV pass reads.
DRAFTER_NAMES = {
    ("aten.gather.default", "index"): "zero_spec_counts",
    ("aten.index.Tensor", "indices"): "plan.tail_rows",
}
#: DeepSeek-V4.1's arguments, keyed as `V4_METADATA`'s, that a replaying
#: prefill's layer 39 passes, at its tail's rows, a buffer of its own: its
#: indptr scan and index group's build, which the tail's layers run once a
#: forward, memoized on the tail's step, and layer 39 runs again over its
#: queries (`QueryRows.indices`), whose batch ids are -1 but at each
#: finishing request's last row. A leaf of the tail's rows that runs more
#: than once binds its later occurrences the buffer named (`Stub.later`).
QUERIED = {(op, "batch_ids"): "plan.query_ids" for op in (_SCAN, _INDICES)}
#: DeepSeek-V4.1's FP8 prefill leaves whose logits plane holds the compressed rows
#: of each KV block the step's longest row reaches (`PagedAttentionCache.
#: unit_tiles`), and the tile table's build, which reads as many block-table
#: columns, by operator: the slot of its row bounds, then each slot and
#: integer that wide. A capture's prefill caches nothing, so a point past
#: cached tokens takes them at its own rows' blocks (`_widened`). A leaf whose
#: bounds are a candidate list's (`_written`) scores an earlier layer's
#: candidate blocks, on a plane the checkpoint sizes.
_MAXIMA = "atom.model_ops.deepseek_v41.indexer:_block_maxima"
_PICK = "aiter._top_k_per_row_decode.default"
_V41 = "atom.model_ops.deepseek_v41"
PAGE_WIDE = {
    _LOGITS: ("context_lens", "out_logits", "kv_indices", "max_model_len"),
    _TOPK: ("row_ends", "logits", "stride0"),
    _PICK: ("seqLens", "logits", "stride0"),
    _MAXIMA: ("visible", "logits", "maxima", "blocks"),
    f"{_V41}.unit_table:unit_table": ("block_tables", "block_tables"),
}
#: DeepSeek-V4.1's FP8 index tables an earlier leaf of its forward writes, by
#: the operator and slot that read them (`_written`): the page-wide scorer's
#: tile table (`tiles`); the blocks each row keeps of the candidate pick
#: (`ids`) and a candidate list's tile table (`table`) and row bounds
#: (`context`), its scorer's and its top-k's; and the pick's top-k's row
#: bounds (`ends`), there where its logits, the block maxima, are an activation.
_WRITTEN = {
    (_LOGITS, "kv_indices"): "tiles",
    (_LOGITS, "context_lens"): "context",
    (_TOPK, "row_starts"): "context",
    (_TOPK, "row_ends"): "context",
    (_PICK, "seqLens"): "context",
    **{
        (f"{_V41}.candidate_table:{f}", "candidates"): "ids"
        for f in ("candidate_block_table", "lift_candidate_selection")
    },
}
#: The one leaf and slot built past its span: aiter's `mla_decode_fwd_v4_nm`
#: at one KV split writes its BF16 `output` at the FP32 split stride, twice
#: the span, and past an exact buffer kills the process (aiter defect, #1977).
NM_OUTPUT = (_NM, "output")
#: aiter's top-k leaves, each with whether it is the decode one, whose
#: `workspace` aiter sizes by host queries into a module built for the
#: device, which the capture answers 0 (`capture.forward._patches`), so a
#: graph holds one byte. A point records the bytes it launched.
TOPK_WORKSPACE = {
    "aiter._top_k_per_row_decode.default": True,
    "aiter._top_k_per_row_prefill.default": False,
}
#: The field of a per-request state layer of ATOM's KV registry that a decode
#: row's leaf reads and writes at every state slot the row holds: GDN's and
#: KDA's recurrent state, which their spec-decode kernel takes with each row's
#: slots and writes once a token, 1 + K under a drafter of K drafts without
#: ReplaySSM. Their conv state (`k_cache`) takes the row's first slot alone.
SNAPSHOTS = "v_cache"
#: The seed of a copy's drawn expert ids, with the copy's index.
SEED = 0
#: An integer dtype of each element size, which compares a tensor's bits.
BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
#: The rank `trace ops` writes an `op_graph`'s members at, which `graph` reads.
RANK = Topology().ranks()[0]
#: The runner's metadata builders the KV allocation may bind on: DeepSeek-V4.1's
#: and its DSpark drafter's.
BUILDERS = ("attn_metadata_builder", "draft_kv_builder")
#: The variable naming the file a measuring process appends each point's
#: record to as it finishes it.
JOURNAL = "COMPASS_OPS_JOURNAL"
#: The graphs of a drafter's part of a step at a shape, beside its target's, by
#: the `(output, draft)` `forward.structure` keys them by: its work past the
#: target's on a step that samples, a block drafter's draft-KV pass there
#: (`KV`), and its work past the target's forward on a middle chunk.
DRAFTED = ((True, True), (True, KV), (False, True))
#: The variable `--plan` sets for its measuring processes past its first round
#: when some step's decode rows are ragged: there a uniform decode point the KV
#: pool cannot hold refuses, as a ragged one does, unless every leaf it times is
#: a collective. A cut point's rows hold a mean its context does not name.
RAGGED = "COMPASS_OPS_RAGGED"


def main(argv: list[str]) -> None:
    from atom.compass.bench import ranks
    from atom.compass.bench.measure import atom_config
    from atom.compass.capture.forward import decode, prefill
    from atom.compass.drive.gpu import hip_visible, kfd_pid, quiet
    from atom.compass.drive.identity import identity
    from atom.compass.run.step_grid import keyed
    from atom.model_engine.arg_utils import EngineArgs

    cut = argv.index("--") if "--" in argv else len(argv)
    if "--coverage" in argv[:cut]:
        return print(json.dumps(_coverage_main(argv[:cut], argv[cut + 1 :]), indent=1))
    args = _parser().parse_args(argv[:cut])
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    if not shapes and not args.plan:
        raise ValueError("name a shape: --width or --prefill, or --plan")
    if shapes and args.plan:
        raise ValueError("--plan places every point; name no --width or --prefill")
    if args.context is None and any(s.phase == "decode" for s in shapes):
        raise ValueError("a decode shape needs --context, each row's tokens")
    engine = EngineArgs.add_cli_args(FlexibleArgumentParser(prog=f"{PROG} --"))
    engine = engine.parse_args(argv[cut + 1 :])
    world = engine.tensor_parallel_size * engine.data_parallel_size
    # DP attention makes each TP rank a DP rank, and one GPU can run one alone
    alone = engine.enable_dp_attention and len(args.gpu) == 1
    if world != len(args.gpu) and not alone:
        raise ValueError(
            f"--gpu names {len(args.gpu)} GPUs and ATOM's arguments resolve "
            f"{world} ranks; the op bench runs one on each, or one DP-attention "
            "rank on one GPU"
        )
    os.environ["HIP_VISIBLE_DEVICES"] = hip_visible(args.gpu)
    kfd_pid()  # before atom_config opens a GPU, so quiet leaves this process out
    config = atom_config(argv[cut + 1 :])
    ranks.layout(config)
    if config.compilation_config.level:
        raise ValueError(
            f"--level {config.compilation_config.level} compiles the model, whose "
            "forward the bench never runs; measure at --level 0"
        )
    _eager(config)
    store = ArtifactStore(args.store)
    todo = [(s, args.context, leaves_at(store, config, s)) for s in shapes]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "engine.json").write_text(json.dumps(keyed(config)))
    command = [*PROG.split(), *argv]
    with quiet(args.gpu, out):
        run = identity(out / "engine.json", config.model, command, args.gpu)
        (out / "run.json").write_text(json.dumps(run))
        key, held = keys(config, run), 0
        if args.plan:
            command, atom_args = " ".join(command), argv[cut + 1 :]
            return plan(store, key, config, run, args, command, out, atom_args)
        for shape, context, found in todo:
            for sig in list(found):
                if store.directory_for(key(shape, context, sig)).exists():
                    del found[sig]
                    held += 1
        print(f"{held} points already in the store", flush=True)
        todo = [point for point in todo if point[2]]
        results = _measured(config, todo, args.llc_mib << 20, len(args.gpu))
    publish(store, key, config, run, " ".join(command), results, out)
    refused = [
        f"{r['op']} at {r['shape']}: {r['refused']}" for r in results if "refused" in r
    ]
    print(f"{len(results) - len(refused)} of {len(results)} points measured")
    if refused:
        raise ValueError(f"{len(refused)} points refused:\n" + "\n".join(refused))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--width", type=int, action="append", default=[], help="a decode width"
    )
    p.add_argument(
        "--prefill",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("T", "B"),
        help="a prefill of T tokens over B sequences",
    )
    p.add_argument(
        "--context",
        type=int,
        help="each decode row's tokens, or the tokens cached before each prefill row",
    )
    p.add_argument(
        "--plan",
        action="append",
        default=[],
        metavar="RECORD",
        help="a record directory whose steps the planned points cover",
    )
    p.add_argument(
        "--llc-mib",
        type=int,
        required=True,
        help="the device's last-level cache in MiB: 256 on MI300X, MI308X and "
        "MI355X (AMD Infinity Cache, CDNA 3 and CDNA 4 white papers); 512 on "
        "MI455, an upper bound confirmed cold at four times it (#2154's card "
        "runs 7 and 8)",
    )
    add_gpu_flag(p)
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument(
        "--out",
        required=True,
        help="where run.json and ops.jsonl go; with --plan, each round's "
        "ops.jsonl goes in round<n> and every point held in plan.jsonl",
    )
    return p


def graph(store, config, shape, arch=None, output=True, draft=False) -> dict:
    """The `op_graph` `trace ops` published for `config` at `shape` on `arch`,
    this GPU's when it is None, as JSON: of a step that produces output, or
    one that does not (`output`), of its target or of its drafter's part
    (`draft`, as `forward.structure` keys it). A prefill that produces none
    runs its target's leaves less the LM head and the sampler, so the
    target's points at output cover both. A miss where the store holds the
    shape under another engine config, or traced under another ATOM source or
    torch (`capture.graph_key`), refuses naming what differs. A
    two-batch-overlap micro-batch's `shape` reads its own graph."""
    from atom.compass.artifacts import ArtifactRefusal, Rule
    from atom.compass.capture import graph_key
    from atom.compass.capture.forward import missed, structure

    at = structure(
        config,
        shape.phase,
        device=arch,
        output=output,
        draft=draft,
        micro_batch=shape.micro_batch,
    )
    key = graph_key(at, str(shape))
    if not store.directory_for(key).exists() and (why := missed(store, key)):
        part = " that produces no output" * (not output)
        part += " as a two-batch-overlap micro-batch" * shape.micro_batch
        part += " for its drafter" * bool(draft) + "'s draft-KV pass" * (draft == KV)
        raise ArtifactRefusal(
            Rule.RESOLUTION,
            f"no op_graph holds {shape}{part} under this engine config, ATOM "
            f"source and torch{why}",
            "compass trace ops captures one under this process's environment",
        )
    return json.loads(store.read(key).read_member("graph", RANK, "json"))


def leaves_at(store, config, shape, arch=None, own=()) -> dict[str, tuple[dict, list]]:
    """Each leaf (`leaves`) of `shape`'s `graph` and, under a speculative
    drafter, of each graph of the drafter's (`DRAFTED`) the store holds there,
    the capture having kept no other; one of a drafter's graphs alone is
    marked ``drafted``. With `own`, `shape` is a drafter's own shape past its
    step's (`parts`), where a capture keeps a decode's drafter's graph alone:
    the leaves of the drafter's graphs there at each output (`graph`'s
    `output`) `own` holds, as its steps read them."""
    from atom.compass.artifacts import ArtifactRefusal

    if own:
        found, target = {}, set()
        for output in sorted(own):
            region = graph(store, config, shape, arch, output, True)["region"]
            leaves(region, None, found)
    else:
        found = leaves(graph(store, config, shape, arch)["region"])
        target, drafter = set(found), getattr(config, "speculative_config", None)
        for output, draft in DRAFTED if drafter else ():
            try:
                region = graph(store, config, shape, arch, output, draft)["region"]
            except ArtifactRefusal:
                continue
            leaves(region, None, found)
    for sig, (op, _) in found.items():
        if sig not in target:
            op["drafted"] = True
    return found


def _ref(value) -> bool:
    return isinstance(value, dict) and value.keys() in ({"context"}, {"module"})


def signature(op: dict) -> str:
    """A leaf's identity across instances: its operator, arguments and outputs,
    a context key or module path by its kind alone, and what the capture
    states beside its arguments when it states anything."""
    args = {s: {k: None for k in v} if _ref(v) else v for s, v in op["args"].items()}
    leaf = {"op": op["op"], "args": args, "out": op["out"]}
    leaf |= {"stated": op["stated"]} if "stated" in op else {}
    return json.dumps(leaf, sort_keys=True, separators=(",", ":"))


def leaves(region: dict, env=None, found=None) -> dict[str, tuple[dict, list]]:
    """Each leaf of a graph's JSON region by `signature`: one of its ops, its
    times in the step under ``occurrences``, and its instances in run order,
    each the key or path every reference slot binds."""
    env, found = env or {}, {} if found is None else found
    if "op" in region:
        bound = {}
        for slot, value in _values(region["args"]):
            if _ref(value):
                ref = ir.ContextRef(next(iter(value.values())))
                bound[slot] = ref.bind(**{n: env[n] for n in ref.index_names}).key
        leaf = region | {"occurrences": 0}
        op, instances = found.setdefault(signature(region), (leaf, []))
        op["occurrences"] += 1
        instances += [bound] if bound not in instances else []
    elif "repeat" in region:
        for i in range(region["count"]):
            index = {region["index"]: region["start"] + region["step"] * i}
            leaves(region["repeat"], {**env, **index}, found)
    else:
        for item in region.get("seq", region.get("par", ())):
            leaves(item, env, found)
    return found


def measure(config, todo, llc: int, rank: int = 0) -> list[dict]:
    """Each point's record, shape by shape: its timing, or why it refused; at
    `rank` of `config`'s layout, over every rank's points (`ranks.union`);
    each also appended to the `JOURNAL` file as it is finished, when one is
    named, suffixed `.<rank>` over several ranks."""
    from atom.compass.bench import ranks
    from atom.compass.bench.replay import ReplayRule

    runner, journal = _runner(config, rank), os.environ.get(JOURNAL)
    try:
        group = ranks.cpu_group()
        if journal and group is not None:
            journal += f".{torch.distributed.get_rank()}"
        stub, rule, got = Stub(runner, llc, journal), ReplayRule(runner.config), []
        todo = ranks.union(todo, group)
        with torch.inference_mode():
            for shape, context, found in todo:
                got += stub.point(rule, shape, context, found)
        return got
    finally:
        runner.exit()


def _runner(config, rank: int):
    """ATOM's runner at TP `rank` of `config`'s DP rank, its parameters built on
    the meta device and no warmup run, its KV pool allocated at the fewest
    blocks a rank has room for, sized by ATOM with the parameters' bytes held
    on the device and then freed for the weights each leaf builds; the runner
    attributes the allocation set are in its `kv_held`, and by path each
    tensor it bound on a metadata builder (`BUILDERS`, `_bound`) but a view the
    builder cut from one of the runner's (`_cut_from`); and, when
    the runner defers its output, one sampled-id handoff to the host on ATOM's
    copy stream, as ATOM's postprocess makes one after every forward. One DP rank
    of several run alone is built as ATOM's simulated DP attention builds it:
    DP 1, its experts cut as across the ranks."""
    from torch.distributed import TCPStore

    from atom.compass.bench.ranks import agreed, cpu_group
    from atom.model_engine.model_runner import ModelRunner, RapidServeModelRunner

    class MetaRunner(ModelRunner):
        def _build_and_load_model(self, model_class):
            with RapidServeModelRunner._init_weight_params_on_meta():
                self.model = model_class(self.config)
            torch.set_default_device(None)

        def _maybe_warmup(self):
            pass

    pc, store = config.parallel_config, None
    if not pc._managed_distributed_store:  # one rank: the store is its own
        if pc.data_parallel_size > 1:
            # ponytail: ATOM's top-k buffers then hold one rank's batched
            # tokens, not the rows the layout gathers: a leaf past them refuses
            config = copy.deepcopy(config)
            config.dp_logical_size = pc.data_parallel_size
            pc = config.parallel_config
            pc.data_parallel_size = pc.data_parallel_size_local = 1
        store = TCPStore(
            pc.data_parallel_master_ip,
            pc.data_parallel_base_port,
            is_master=True,
            wait_for_workers=False,
        )
        pc.data_parallel_base_port, pc._managed_distributed_store = store.port, True
    runner = MetaRunner(rank, config)
    runner.store = store
    nbytes = sum(p.nbytes for p in runner.model.parameters())
    held = torch.empty(nbytes, dtype=torch.uint8, device=runner.device)
    blocks = runner.get_num_blocks()["num_kvcache_blocks"]
    del held
    _release()  # the pool takes the room the hold had
    config.num_kvcache_blocks = agreed(blocks, cpu_group())
    before = dict(vars(runner))
    builders = {n: b for n in BUILDERS if (b := getattr(runner, n, None)) is not None}
    old = {id(v) for b in [runner, *builders.values()] for v in [b, *vars(b).values()]}
    runner.allocate_kv_cache(config.num_kvcache_blocks)
    # what the allocation bound on the runner: `trace ops` reads a tensor
    # made there as KV
    runner.kv_held = {k: v for k, v in vars(runner).items() if before.get(k) is not v}
    old |= set(map(id, runner.kv_held.values()))
    own = _held(runner, None).values()
    for name, builder in builders.items():
        for key, value in vars(builder).items():
            bound = _bound(f"{name}.{key}", value, old, runner.device)
            runner.kv_held |= {n: t for n, t in bound.items() if not _cut_from(t, own)}
    # with no handoff made, a Qwen3.8-27B GDN decode leaf read 5% under a
    # runner that had made one
    ids, done = runner.tokenID_processor, torch.cuda.Event()
    if ids.is_deferred_out:
        done.record()
        zero = torch.zeros(1, dtype=torch.int32, device=runner.device)
        ids.send_to_cpu_async(zero, ids.token_ids_cpu, done)
        ids.recv_async_output(ids.token_ids_cpu)
    return runner


class Stub:
    """The leaves of a point timed beside ATOM's runner, on the forward context
    and live metadata ATOM prepared for each placement of its batch. `alone`
    is the layout's ranks when the runner runs one of them by itself, and then
    a collective leaf refuses."""

    def __init__(self, runner, llc: int, journal=None) -> None:
        from atom.compass.bench.replay import state_slots_per_request
        from atom.model_engine.kv_block import STATE_SLOT_CLASS

        self.runner, self.llc, self.journal = runner, llc, journal
        # read before each cold replay, so no call reads what the last left
        # in the cache: under twice the cache, 256 copies alone leave a leaf warm
        self.flush = None
        self.alone = getattr(runner.config, "dp_logical_size", 0)
        self.slots = runner.pool_plan.entries.get(STATE_SLOT_CLASS, 0)
        self.per_row = state_slots_per_request(runner.config)
        self.built, self.leaves, self.placed = {}, {}, []
        # by placement, leaf and slot, what a leaf's later occurrences take
        # where ATOM passes them a buffer of their own (`QUERIED`)
        self.later = {}
        self.cold_copies, self.clones, self.kv, self.state = {}, {}, {}, {}
        # `once`: by leaf, the bytes of a buffer every copy launches
        self.written, self.unplanned, self.once = {}, {}, {}
        self.modules = dict(runner.model.named_modules())
        self.contexts = runner.config.compilation_config.static_forward_context
        drafter = getattr(runner, "drafter", None)
        self.moe = _moe(runner.model)
        self.draft_moe = _moe(drafter.model) if drafter else []
        # ATOM's top-k weights and ids buffers when its MoE layers fuse shared
        # experts into top-k (`init_aiter_topK_meta_data`), live metadata a
        # leaf may read: past top-k, the shared experts' ids, under EP one
        # more, and the fake expert's on rows another rank owns
        topk = sys.modules.get("atom.model_ops.topK")
        meta = getattr(topk, "aiter_topK_meta_data", None)
        self.topk = dict(
            zip(
                ("topk_weights", "topk_ids"),
                meta if meta and (self.moe or self.draft_moe) else (),
            )
        )
        self.shared = self.topk.get("topk_ids")
        self.loads = []
        # DeepSeek-V4.1's compress ratios by the rows a page holds at each
        geometry = getattr(runner, "attn_metadata_builder", None)
        geometry = getattr(geometry, "geometry", None)
        self.known = self.pages = {
            ("rows_per_page", geometry.rows_per_page(r)): r
            for r in getattr(geometry, "layer_ratios", ())
            if r and hasattr(geometry, "rows_per_page")
        }
        # and the rows a request's window ring holds at a layer
        self.ring = getattr(geometry, "ring_slots", 0)
        # V4.1's FP8 index plane, whose prefill logits are as wide as the
        # step's KV blocks (`PAGE_WIDE`); the FP4 plane packs them over the
        # whole block table, which no context widens
        self.paged = bool(self.pages) and not getattr(geometry, "index_fp4", False)
        # its index layer that picks its successors' candidate blocks
        spec = {m: getattr(m, "spec", None) for m in self.modules.values()}
        picks = [m for m, s in spec.items() if getattr(s, "produces_candidates", 0)]
        self.picker = next((m for m in picks if hasattr(m, "topk_blocks")), None)
        # a DeepSeek-V4 CSA layer: its forward writes the top-k section of the
        # CSA index buffer, and `_prepare` runs none
        self.csa = next(
            (
                m
                for m in self.modules.values()
                if getattr(m, "indexer", None) is not None
                and hasattr(m, "_fill_csa_paged_compress")
            ),
            None,
        )
        # the buffers ATOM passes a slot by name, DeepSeek-V4's shared ones
        # beside its CSA layer alone
        self.names = V4_METADATA | BATCH_IDS
        self.names |= V4_CSA_SHARED if self.csa is not None else {}

    def _prepare(self, batch) -> None:
        """`batch` prepared by ATOM's runner as its forward prepares one, no
        forward run, and a snapshot of the forward context it built and the
        live metadata each leaf names, beside the KV `point` found for it."""
        from atom.compass.bench.replay import seat_states
        from atom.utils.forward_context import get_forward_context

        seat_states(getattr(self.runner, "attn_metadata_builder", None), batch)
        self.runner._gate_staging_reuse()
        with _uncommitted():
            input_ids = self.runner.prepare_model(batch)[0]
        self.runner._mark_staging_h2d_enqueued()
        fc, live = get_forward_context(), {"input_ids": input_ids, **self.topk}
        if self.csa is not None:
            self._top_k(fc, len(input_ids))
        for name, buffer in self.runner.forward_vars.items():
            live[name] = getattr(buffer, "gpu", buffer)
        for owner in (fc.attn_metadata, fc.context):
            live |= {k: v for k, v in vars(owner).items() if torch.is_tensor(v)}
        device, md = self.runner.device, fc.attn_metadata
        live = {
            k: v
            for k, v in live.items()
            if torch.is_tensor(v) and v.device.type == device.type
        }
        # DeepSeek-V4.1's scorer workspace and decoder plan, and each pass's
        # step first, so a tie takes its own, its real rows' count the context's
        scorer = getattr(getattr(md, "cache", None), "workspace", None)
        plan = getattr(getattr(md, "step", None), "decoder", None)
        live |= _bound("workspace", scorer, set(), device)
        live |= _bound("plan", plan, set(), device)
        passes, running = _passes(fc), getattr(fc.context, "running_tokens", None)
        if running in passes or len(passes) < (1 if plan is None else 3):
            rows = sorted(k for k in passes if k is not None)
            raise PassRows(f"its passes run {rows} rows and its forward {running}")
        lives = {
            rows: _bound("step", step, set(), device)
            | live
            | ({} if count is None else {"scheduled_tokens_gpu": count})
            for rows, (step, count) in passes.items()
        }
        # DeepSeek-V4's persistent HCA decode workspace, one per device: a slot
        # of `HCA_DECODE` named as one of its fields is a view of that field
        hca = sys.modules.get("atom.model_ops.v4_kernels.hca_persist")
        work = getattr(hca, "_workspaces", {}).get(self.runner.device.index)
        work = work._asdict() if work is not None else {}
        # a decode split plan, `(splits, split_indptr)`: only a slot
        # `V4_METADATA` names takes it, so `_meta`'s rules never offer it
        named = lives[None] | {
            f"{k}.1": v[1]
            for k, v in vars(fc.attn_metadata).items()
            if k.startswith("split_plan_") and v
        }
        if rows := getattr(batch, "total_seqs_num_prefill", 0):
            cached = batch.num_cached_tokens[:rows]
            named["num_cached_tokens"] = torch.tensor(
                cached, dtype=torch.int32, device=device
            )
            if (lens := named.get("context_lens")) is not None:
                named["context_lens.long"] = lens.long()
        # a drafter's pass of rows of its own counts its real rows at the entry
        # of ATOM's `batch_info` stating them, as the drafter publishes its shape
        info = self.runner.forward_vars.get("batch_info")
        counts = {} if info is None else {n: i for i, n in enumerate(info.np.tolist())}
        tokens = getattr(self.runner, "tokenID_processor", None)
        zeros = getattr(tokens, "zero_spec_counts", None)
        drafter = {} if zeros is None else {"zero_spec_counts": zeros}
        meta = dict(self.kv)
        for sig, (op, _) in self.leaves.items():
            writes, rows = _writes(op["op"]), _pass_rows(op)
            at = None if rows == running else rows
            live = lives.get(at, lives[None])
            if drafted := op.get("drafted", False):
                live = live | drafter
                if at not in lives and rows in counts:
                    i = counts[rows]
                    live = live | {"scheduled_tokens_gpu": info.gpu[i : i + 1]}
            for path, spec in _specs(op["args"]):
                index = self.paged and _written(op, path)
                if spec.get("device") == "cpu":  # `_copy` builds it on the host
                    continue
                if spec["role"] == "metadata" or index:
                    token = (op["op"], path) in TOKEN_IDS
                    pool = {"input_ids": input_ids} if token else live
                    what, field = "metadata", path.split(".")[0]
                    if op["op"] == HCA_DECODE and field in work:
                        pool, what = {field: work[field]}, "HCA workspace"
                    ratio = _ratio(op, path, self.known)
                    key = op["op"], _keyed(op["args"], path)
                    wanted = self.names.get(key)
                    if "kv_indptr_extend" in named:  # a V4 prefill's step
                        wanted = wanted or V4_SHARED.get(key)
                    if wanted:
                        what = wanted.format(
                            ratio=ratio,
                            kind=V4_KINDS.get(ratio),
                            dest=_DESTS.get(ratio),
                        )
                        pool = {what: named[what]} if what in named else {}
                    elif drafted and (want := DRAFTER_NAMES.get(key)) in live:
                        pool = {want: live[want]}
                    elif v41 := _named(op, path, spec, live, self.known):
                        pool = v41
                    try:
                        if index:  # as the forward's first reader writes it
                            step = passes.get(at, passes[None])[0]
                            pool = self._index(md.cache, step, index, ratio)
                        if not pool:
                            raise ValueError(f"ATOM passes {path} {what}: not live")
                        name, view = _meta(spec, path, pool, what)
                        if what in V4_LISTS:  # at the point's rows, not the capture's
                            view = pool[what]
                        # leaves share a view's clone: DeepSeek-V4's decode
                        # leaves view 2.7 GB of it at a placement. What an
                        # operator writes is cloned apart, or a leaf timed
                        # after it reads what it wrote (a `copy_` bound by
                        # shape); its leaves share that clone, as ATOM's
                        # layers share the HCA decode workspace
                        clones = self.clones
                        if field in writes:
                            clones = self.written.setdefault(op["op"], {})
                        meta[sig, path] = name, _deep(view, {}, clones)
                        later = QUERIED.get(key)  # live where a decoder plan is
                        if (
                            later in live
                            and op["occurrences"] > 1
                            and at == plan.tail_rows.numel()
                        ):
                            name, view = _meta(spec, path, {later: live[later]})
                            here = len(self.placed), sig, path
                            self.later[here] = name, _deep(view, {}, clones)
                    except ValueError as error:  # the leaf refuses, not the step
                        meta[sig, path] = str(error), None
        # one clone across placements of what none changed: a leaf may cache
        # what it prepares on the host by its identity (linear attention's
        # chunk indices), and a graph over placements replays all of it
        memo: dict = {}
        snap = copy.copy(fc)
        snap.attn_metadata = _deep(fc.attn_metadata, memo, self.clones)
        snap.context = _deep(fc.context, memo, self.clones)
        self.placed.append((snap, meta))

    def _top_k(self, fc, tokens) -> None:
        """The CSA top-k section of the placement's index buffer written as
        ATOM's forward writes it, from top-k ids over the compressed rows each
        of its `tokens` sees: every row when it sees no more than top-k, else
        one drawn from each of top-k equal runs of them, seeded by `SEED` and
        the placement."""
        # ponytail: spread over the rows, ascending; the indexer picks where
        # its scores fall, which no placement has
        positions, k = fc.context.positions[:tokens], self.csa.indexer.index_topk
        seen = (positions.long() + 1) // self.csa.compress_ratio
        run = torch.arange(k, device=positions.device)
        lo, hi = run * seen[:, None] // k, (run + 1) * seen[:, None] // k
        seed = torch.Generator(positions.device)
        seed.manual_seed(SEED * CALLS + len(self.placed))
        draw = torch.rand(lo.shape, generator=seed, device=positions.device)
        ids = torch.where(seen[:, None] > k, lo + (draw * (hi - lo)).long(), run)
        self.csa._fill_csa_paged_compress(
            fc.attn_metadata, ids.int(), fc.context.positions, tokens
        )

    def _index(self, cache, step, kind, ratio) -> dict[str, torch.Tensor]:
        """DeepSeek-V4.1's index table `kind` (`_written`) for the pass `step`,
        by name, as its forward writes it: the tile table at `ratio`, into the
        scorer workspace's (`unit_tiles`); or, off the blocks each row keeps of
        the candidate pick (`_candidates`), those, a candidate list's tile
        table or row bounds (`candidate_blocks`), or each row's visible blocks,
        the pick's top-k's row bounds (`_block_maxima`'s `ends`)."""
        if kind == "tiles":  # rewritten: another pass may have written it since
            step.tiles.pop(ratio, None)
            return {f"workspace._tiles.{ratio}": cache.unit_tiles(step, ratio)}
        if self.picker is None:
            raise ValueError("no layer of the model picks candidate blocks")
        spec, block = self.picker.spec, self.picker.block_size
        ends = (step.visible[spec.ratio] + block - 1) // block
        if kind == "ends":
            return {f"step.visible.{spec.ratio}.blocks": ends}
        if spec.layer_id not in step.candidates:
            step.candidates[spec.layer_id] = _candidates(ends, self.picker.topk_blocks)
        bound = cache.candidate_blocks(step, spec.layer_id, spec.ratio)
        return {f"step.candidate_blocks.{spec.layer_id}.{kind}": getattr(bound, kind)}

    def point(self, rule, shape, context, found) -> list[dict]:
        """Each leaf of `found` timed at `shape` on `point_rows`, or every leaf
        refused when `rule` refuses the batch."""
        from atom.utils.forward_context import _forward_kv_cache_context

        def refuse(why):
            return _journal(
                self.journal,
                [
                    {"shape": str(shape), "context": context, "op": op["op"]}
                    | {"leaf": sig, "refused": why}
                    for sig, (op, _) in found.items()
                ],
            )

        pool = self.runner.config.num_kvcache_blocks
        uncut = os.environ.get(RAGGED) and not all(
            exchange(o) for o, _ in found.values()
        )
        drafts = getattr(rule, "drafts", 0)  # a rule of no drafter states none
        rows = point_rows(shape, context, pool, rule.block_size, bool(uncut), drafts)
        if rows is None:
            return refuse(f"its rows need more than the pool's {pool} KV blocks")
        if shape.phase != "decode" and self.paged:
            wide = [-(-max(r[i] for r in rows) // rule.block_size) for i in (0, 1)]
            found = {s: (_widened(op, wide), at) for s, (op, at) in found.items()}
        found = {s: (_own_rows(_summed(op, rows)), at) for s, (op, at) in found.items()}
        view = {"rows": rows, "produces_output": True}
        blocks = sum(-(-row[1] // rule.block_size) for row in rows)
        kv = _forward_kv_cache_context.kv_cache_data
        held = _held(self.runner, kv)
        self.state = _state(self.runner, held)
        # V4.1's ratios also by the width of the logits each scorer writes, but
        # a candidate list's, whose width a widened plane can take
        self.known = self.pages | {
            ("stride0", op["args"]["out_logits"]["shape"][1]): ratio
            for op, _ in found.values()
            if (ratio := op.get("stated", {}).get("compress_ratio")) is not None
            and "out_logits" in op["args"]
            and (op["args"].get("context_lens") or {}).get("role") != "activation"
        }
        # and by the rows each compress plan holds, as many as its keys' RoPE's
        self.known |= {
            ("compressed", op["args"]["plan_gpu"]["shape"][0]): op["args"]["ratio"]
            for op, _ in found.values()
            if op["op"] == _FUSED
        }
        bind = {
            (s, path): (spec, _named(op, path, spec, held, self.known) or held)
            for s, (op, _) in found.items()
            for path, spec in _specs(op["args"])
            if spec["role"] == "kv"
        }
        self.kv = {k: _kv(spec, k[1], live) for k, (spec, live) in bind.items()}
        # a view of a pool holding the state arena past its pages, narrower than
        # the widest of it its leaf takes, is a window ring, as `_plan` reads one
        span = {k: _span(v["shape"], v["stride"]) for k, (v, _) in bind.items()}
        named = {k: (k[0], self.kv[k][0]) for k in bind}
        for k, (spec, live) in bind.items():
            pooled = self.state.get(named[k][1])
            if pooled and span[k] < max(span[j] for j in bind if named[j] == named[k]):
                self.kv[k] = _kv(spec, k[1], live, ring=True)
        taken = len(rows) * self.per_row  # the batch's state slots
        room = [pool // blocks]
        room += [t.k_cache.shape[0] // taken for t in kv.values() if _stateful(t)]
        self.context, self.unplanned, self.once = _split(context)[0], {}, {}
        tokens = sum(row[0] for row in rows)  # a drafter's rows verify its drafts
        plans = {
            s: self._plan(
                op, i, kv, blocks, len(rows), tokens, s, shape.phase == "decode"
            )
            for s, (op, i) in found.items()
        }
        need = max(p for _, _, p, _ in plans.values())
        self.leaves, self.placed, self.clones = found, [], {}
        self.written, self.later = {}, {}
        self.cold_copies = {s: self._cold(size) for s, (_, size, *_) in plans.items()}
        ladder = contextlib.nullcontext
        if shape.width or shape.phase == "decode" and shape.tokens < shape.seqs:
            # a step that pads past its rows is prepared padded, as the
            # capture's runner pads it: to its W, where a prefill's LM head
            # and its drafter's later passes run, or, a drafter's own decode
            # shape, to its B, where a block drafter's passes run
            width = np.array([shape.width or shape.seqs], dtype=np.int32)
            ladder = functools.partial(
                patch.object, self.runner, "capture_sizes_np", width
            )
        for p in range(max(min(need, *room), 1)):
            slot = p * taken
            if self.slots >= taken:  # past the state slots, take them again
                slot %= self.slots - self.slots % taken
            try:
                batch = rule.batch(view, p * blocks, slot)
            except ValueError as error:  # the rule refuses, before any preparation
                return refuse(f"ValueError: {error}")
            try:
                with ladder():
                    self._prepare(batch)
            except PassRows as error:  # no leaf's rows tell which pass it runs
                return refuse(f"ValueError: {error}: a leaf's rows name no pass")
            except Exception as error:
                error.add_note(f"measuring {shape} at context {context}")
                raise
        got = []
        for sig, (op, inst) in found.items():
            n, size, _, (by_block, by_slot) = plans[sig]
            names = {p: m[0] for (s, p), m in self.placed[0][1].items() if s == sig}
            later = {k[2]: m[0] for k, m in self.later.items() if k[:2] == (0, sig)}
            record = {"shape": str(shape), "context": context, "op": op["op"]}
            record |= {"leaf": sig, "metadata": names, "rows": rows}
            record |= {"later": later} if later else {}
            record |= {"exchange": ex} if (ex := exchange(op)) else {}
            try:
                record |= self._time(op, sig, inst, n)
                # a KV argument is one view every copy reads at its placement:
                # its bytes count once a placement, a slot's once however many
                # placements wrap onto it; a copy past the cold ones reads copy
                # `c mod m`'s activations
                built = min(n, record.get("copies", n))
                at = min(len(self.placed), -(-built // len(inst)))
                wrap = self.slots // taken if self.slots >= taken else at
                # ponytail: routed expert ids and an object's activations are
                # drawn a copy each yet count once a cold copy here; split them
                # out when a leaf with more instances than cold copies holds one
                acts = sum(
                    _span(s["shape"], s["stride"]) * _dtype(s["dtype"]).itemsize
                    for _, s in _specs(op["args"])
                    if s["role"] == "activation"
                )
                record["footprint_bytes"] = (
                    built * (size - by_block - by_slot - acts)
                    + min(built, self.cold_copies[sig]) * acts
                    + at * by_block
                    + min(at, wrap) * by_slot
                    + self.once.get(sig, 0)
                )
            except Exception as error:  # noqa: BLE001 - a refusal names what stopped it
                record["refused"] = f"{type(error).__name__}: {error}"
                _alive(self.runner.device)
            copies = record.get("copies", n)
            print(f"{shape}: {op['op']} over {copies} copies {names or ''}", flush=True)
            record |= {"instances": len(inst), "placements": len(self.placed)}
            got += _journal(self.journal, [record])
        self.placed = []
        return got

    def _plan(
        self, op, instances, kv, blocks, rows, tokens=None, sig=None, decode=False
    ):
        """A leaf's copies, its bytes per call, the placements its context
        keys and KV arguments take, and its KV arguments' bytes per call read
        by blocks and by state slots; `tokens` defaults to `rows`, a
        decode's; at a `decode`, a context key's recurrent state counts each
        state slot a row holds."""
        need, size, shared = 1, 0, [0, 0]
        fn, cia = _overload(op["op"]), "CompositeImplicitAutograd"
        if fn and fn.is_view and not fn.has_kernel_for_dispatch_key(cia):
            return 1, size, need, shared  # reads none; a composite (`reshape`) may copy
        table, ids = GATHERED.get(op["op"], (None, None))
        acts = [s for _, s in _specs(op["args"]) if s["role"] == "activation"]
        pos = op["args"].get("positions")
        pos = pos.get("shape") if isinstance(pos, dict) else None
        rope = pos[-1] if pos else tokens or rows  # a RoPE table's rows
        for slot, s in _specs(op["args"]):
            itemsize = _dtype(s["dtype"]).itemsize
            nbytes = _span(s["shape"], s["stride"]) * itemsize
            if s["role"] == "weight" and (slot in TABLES or slot == table):
                read = rope
                if slot == table:  # a row an id
                    read = torch.Size(op["args"][ids]["shape"]).numel()
                entry = _span(s["shape"][1:], s["stride"][1:]) * itemsize
                nbytes = entry * min(read, s["shape"][0])
            if s["role"] == "kv":
                # ponytail: its widest-strided dim taken as blocks or state slots,
                # or plane rows when the leaf states its layer's attention geometry;
                # beside state slots a block view is one block a request, the
                # one a decode compressor writes its row to, and a row view
                # holding a state arena past its pages (`_state`) one row a
                # request
                unit, history = rows if self.slots else blocks, 0
                name = self.kv.get((sig, slot), ("",))[0]
                slotted = int(name in self.state)
                stated = op.get("stated", {})
                window = stated.get("window_size")
                if window and (ratio := stated["compress_ratio"]):
                    # the rows the batch's blocks hold at its layer: a row
                    # per `ratio` tokens; its window rows are left out
                    unit = blocks * self.runner.config.kv_cache_block_size // ratio
                    slotted = 0
                    # of which a request reads at most its indexer's top-k
                    if top := stated.get("index_topk"):
                        unit = min(unit, rows * top)
                elif window and self.state.get(name):
                    unit = 0  # its window is read from the ring alone
                if self.state.get(name):
                    # its window rings: a ring cut from it (`ring_view`), smaller
                    # than the widest of it a leaf stating its window takes, read
                    # a window a token, or the view a leaf given the ring's
                    # `window` writes a row a token; a ring a request at most
                    widest = max(
                        _span(v["shape"], v["stride"]) * itemsize
                        for p, v in _specs(op["args"])
                        if self.kv.get((sig, p), ("",))[0] == name
                    )
                    given = op["args"].get("window")
                    writes = isinstance(given, dict) and "object" in given
                    if nbytes < widest if window else writes:
                        reach = acts[0]["shape"][0] + rows * ((window or 1) - 1)
                        unit, slotted = min(reach, rows * self.ring), 1
                    elif window and not decode:
                        # ponytail: a prefill's prefix reads each request's
                        # history through it too, the window less one at most
                        # its cached tokens, by slot; a leaf of fewer rows, a
                        # bounded-replay tail, none: its window starts there
                        if acts[0]["shape"][0] >= (tokens or rows):
                            history = rows * min(window - 1, self.context or 0)
                # an entry is the span of its other dims: a view may be strided
                # by a whole state slot or by every layer's block, and a field
                # laid out layers first, `[layers, slots, ...]`, is read a slot
                # once a layer; on a tie, the first dim of size above one: a
                # field of one layer filling its entry strides layer and slot alike
                shape, stride = s["shape"], s["stride"]
                cut = _widest(shape, stride)
                rest = shape[:cut] + shape[cut + 1 :], stride[:cut] + stride[cut + 1 :]
                entry = _span(*rest) * itemsize
                if name.startswith(_CACHE.format(ratio="")):
                    unit, history = shape[cut], 0  # a pointer table, at its entries
                nbytes, need = entry * (unit + history), None
                shared[slotted] += entry * unit
                shared[1] += entry * history
            if slot == "workspace" and op["op"] in TOPK_WORKSPACE:
                try:  # not the byte captured; a query that raises refuses the leaf
                    scratch, cached = self._workspace(op)
                    nbytes = scratch.nbytes
                    if cached:  # no copy's own: one buffer, held once
                        self.once[sig], nbytes = nbytes, 0
                except Exception as error:  # noqa: BLE001 - raised at its copy
                    self.unplanned[sig] = error
            size += nbytes
        for key in instances[0].values():
            module = self.contexts.get(key)
            if hasattr(module, "layer_num"):  # a layer of ATOM's KV registry
                held = kv.get(f"layer_{module.layer_num}")
                unit = rows if _stateful(held) else blocks
                # ponytail: a tensor's first dim is its blocks or its state slots
                for name, t in _fields(held):
                    at = unit
                    if name == SNAPSHOTS and decode and _stateful(held):
                        at *= self.per_row  # each slot a row holds
                    size += t.nbytes // t.shape[0] * at
                need = None
            elif (module := self.modules.get(key, module)) is not None:
                pool = self.runner.config.num_kvcache_blocks
                tables = {id(getattr(module, name, None)) for name in sorted(TABLES)}
                for t in _own(module):
                    if id(t) in tables:  # a RoPE table, a row a token
                        size += t.nbytes // t.shape[0] * min(rope, len(t))
                    # ponytail: a first dim of the pool's blocks is a view of
                    # the KV pool (DeepSeek-V4's Indexer), read at the batch's
                    elif t.dim() and t.shape[0] == pool:
                        size += t.nbytes // pool * blocks
                        need = None
                    else:
                        size += t.nbytes
        n = self._cold(size)
        if need is None:  # cycle over every instance's state, which is held anyway
            n = min(CALLS, max(n, len(instances)))
        return n, size, need or -(-n // len(instances)), shared

    def _cold(self, size) -> int:
        """Copies of `size` bytes a call whose bytes pass `COLD` times the cache."""
        return min(CALLS, COLD * self.llc // size + 1) if size else 1

    def _time(self, op, sig, instances, n) -> dict:
        """`op` timed in a graph of K calls, `CALLS` or as many as fill
        `GRAPH_NS`, cold over at most `n` copies, past the first `m` its point
        sized to the cache (`cold_copies`) each on copy `c mod m`'s
        activations, then copy `c mod n` with the c-th weight up to the
        leaf's occurrences when it takes one, at most K, each replay after one
        read of `flush`; and, when K is `CALLS`, warm over one, with no read;
        with its kernels, profiled again on every rank when any rank's named
        none of a call its cold graph timed. Every rank's graph holds the most K
        any rank's call gives, and a leaf any rank refuses, every rank does."""
        from atom.compass.bench.ranks import agreed, cpu_group, refused_together
        from atom.utils import forward_context

        if self.alone and exchange(op):
            raise ValueError(
                f"a collective over the layout's {self.alone} ranks, measured "
                "across them, not on a runner that runs one alone"
            )
        if op["args"].get("device") == "cpu":
            raise ValueError(
                "a copy to pageable host memory, which ATOM allocates and no "
                "CUDA graph captures"
            )
        fn, m = _function(op["op"]), self.cold_copies.get(sig, n)
        self.loads, self.built = [], {}
        if self.flush is None:
            self.flush = torch.empty(
                COLD * self.llc, dtype=torch.uint8, device=self.runner.device
            )
        with refused_together():
            calls = [self._copy(op, sig, instances, 0)]
        # one weight can read a few percent off another of its spec: cycle one
        # per occurrence of the leaf, on the n copies' other arguments; a
        # gathered table is one, its copies reading rows of their own
        table = GATHERED.get(op["op"], ("",))[0]
        weighs = any(
            s["role"] == "weight" and p != table for p, s in _specs(op["args"])
        )
        held = [op["occurrences"]] if weighs else []
        try:
            forward_context._forward_context = calls[0][1]
            with refused_together():
                fn(**calls[0][0])  # compiles and tunes outside the graphs
            kernels, ns = _identity(fn, calls[0][0])
            k = min(CALLS, max(int(GRAPH_NS // ns), 1)) if ns else CALLS
            # a collective's call holds its wait on the last rank to arrive,
            # whose own call is the shortest: every graph takes its calls
            k = agreed(k, cpu_group(), most=True)
            n = min(n, k)
            with refused_together():
                calls += [
                    self._copy(op, sig, instances, c) for c in range(1, min(n, m))
                ]
                calls += [
                    self._copy(op, sig, instances, c, calls[c % m]) for c in range(m, n)
                ]
                for c in range(n, min(k, max([n, *held]))):
                    calls.append(self._reweigh(op, sig, calls[c % n], c))
            n = len(calls)
            cycle = [calls[i % n] for i in range(k)]
            cold = _replay(fn, cycle, self.flush)
            warm = _replay(fn, calls[:1] * k) if n > 1 and k == CALLS else None
            # two profiles that named nothing of a call its graph timed missed it
            if agreed(int(not kernels and any(cold)), cpu_group(), most=True):
                forward_context._forward_context = calls[0][1]
                kernels = _identity(fn, calls[0][0], timed=any(cold))[0]
        finally:
            forward_context._forward_context = forward_context.ForwardContext()
        medians = statistics.median(cold), warm and statistics.median(warm)
        scratch = {}
        if op["op"] in TOPK_WORKSPACE:
            scratch["workspace_bytes"] = calls[0][0]["workspace"].nbytes
        return scratch | {
            "kernels": kernels,
            "cold_ns": medians[0],
            "warm_ns": medians[1],
            "warm_over_cold": medians[1] / medians[0] if warm and medians[0] else None,
            "cold_repeats": cold,
            "warm_repeats": warm,
            "copies": n,
            "calls": len(cycle),
            "llc_bytes": self.llc,
            "protocol": PROTOCOL,
            "release_wait_s": _waited.pop("s", 0.0),
            "release_capped": _waited.pop("capped", 0),
            "routing": _routing(self.loads) if self.loads else None,
        }

    def _copy(self, op, sig, instances, c, like=None):
        """Copy `c` of `op`'s call: its keyword arguments and forward context,
        on the activations of copy `like` when given."""
        from atom.compass.capture.forward import rebuild_object, referenced

        instance = instances[c % len(instances)]
        p = c // len(instances) % len(self.placed)
        snap, meta = self.placed[p]
        fresh = c >= len(self.placed) * len(instances)
        table, ids = GATHERED.get(op["op"], (None, None))
        drafted = op.get("drafted", False)

        seq = tuple if ":" in op["op"] else list  # a lifted call's, as ATOM's

        def build(path, value, old):
            if isinstance(value, list):
                return seq(
                    build(f"{path}.{i}", v, old and old[i]) for i, v in enumerate(value)
                )
            if isinstance(value, dict) and value.get("device") == "cpu":
                return _host(value, self.runner.device.type != "cpu")
            if path == ids:  # copy c's own rows of the one table
                out = self._tensor(value, meta.get((sig, path)), True)
                n, rows = out.numel(), op["args"][table]["shape"][0]
                if "rows" not in self.built:
                    seed = torch.Generator(out.device).manual_seed(SEED)
                    self.built["rows"] = torch.randperm(
                        rows, generator=seed, device=out.device
                    )
                at = torch.arange(c * n, (c + 1) * n, device=out.device) % rows
                return out.copy_(self.built["rows"][at].view(out.shape))
            if isinstance(value, dict) and self._routed(value, sig, path, drafted):
                return self._route(value, c, drafted)
            if isinstance(value, dict) and value.get("role") == "weight":
                # each cache-sized copy takes a weight of its own, as it does
                # activations; a gathered table's copies share one
                count = max(op["occurrences"], self.cold_copies.get(sig, 1))
                return self._weight(value, c, 1 if path == table else count)
            if isinstance(value, dict) and "object" in value:
                return rebuild_object(
                    value["object"], build(path, value["state"], None)
                )
            if (
                isinstance(value, dict)
                and value.get("role") == "activation"
                and old is not None
                and (sig, path) not in meta  # one the placement wrote
            ):
                return old
            if path == "workspace" and op["op"] in TOPK_WORKSPACE:
                if sig in self.unplanned:  # its size query raised at planning
                    raise self.unplanned[sig]
                return self._workspace(op)[0]
            if _ref(value):
                key, module = instance[path], "module" in value
                held = (
                    referenced(self.runner, key) if module else self.contexts.get(key)
                )
                if isinstance(held, torch.nn.Module):
                    _fill(held, self.runner.device)
                return (_module_copy(held) if fresh else held) if module else key
            if isinstance(value, dict) and "affine" in value:
                return ir.Affine(*value["affine"]).at(self.context)
            if (op["op"], path) == NM_OUTPUT and op["args"].get("num_kv_splits") == 1:
                # the captured view at the start of the kernel's write extent
                n = _span(value["shape"], value["stride"])
                device = self.runner.device
                whole = _random({**value, "shape": [2 * n], "stride": [1]}, device)
                return whole.as_strided(value["shape"], value["stride"])
            if isinstance(value, dict):
                taken = meta.get((sig, path))
                if c % op.get("occurrences", 1):  # a later one's own (`QUERIED`)
                    taken = self.later.get((p, sig, path), taken)
                out = self._tensor(value, taken, fresh)
                # V4.1's FP8 plane a top-k or block maxima ranks: ATOM's scorer
                # writes it first, which here no leaf has, or a scorer timed
                # before has, off its own random inputs. Placements share one
                # clone of it, which ATOM's preparation leaves unchanged, so
                # each copy draws on a plane of its own
                if self.paged and path == "logits" and value["role"] == "metadata":
                    bound = PAGE_WIDE.get(op["op"], ("",))[0]
                    if (ends := meta.get((sig, bound), ("", None))[1]) is not None:
                        out = out if fresh else _clone(out)
                        _scores(out, ends, value)
                # and each copy of the scorer writes a plane of its own
                written = self.paged and value["role"] == "metadata"
                if written and (op["op"], path) == (_LOGITS, "out_logits"):
                    out = out if fresh else _clone(out)
                return out
            if isinstance(value, str) and value.startswith("torch."):
                return getattr(torch, value[6:], value)
            if isinstance(value, str) and value.split(":")[0] == "cuda":
                return self.runner.device  # the capture ran on device 0
            return value

        args = op["args"].items()
        return {s: build(s, v, like[0][s] if like else None) for s, v in args}, snap

    def _workspace(self, op) -> tuple[torch.Tensor, bool]:
        """`_topk_workspace` for the top-k leaf `op` at the point's context."""
        rows, stride, k = (
            ir.Affine(*v["affine"]).at(self.context) if isinstance(v, dict) else v
            for v in map(op["args"].get, ("numRows", "stride0", "k"))
        )
        # a capture leaves out a trailing argument at its default
        stable = op["args"].get("stable", False)
        return _topk_workspace(op["op"], rows, stride, k, stable, self.runner.device)

    def _routed(self, spec, sig, path, drafted=False) -> bool:
        """Whether `spec` at `sig`'s `path` is a router's top-k expert ids: an
        integer activation, or a metadata view `_meta` bound to ATOM's ids
        buffer, a row per token, a column per expert an MoE layer of the model
        takes, of its drafter's for a leaf of the drafter's alone (`drafted`),
        or, where it holds one, per column of ATOM's ids buffer with its fused
        shared experts."""
        moe = self.draft_moe if drafted else self.moe
        widths = {k for _, k in moe}
        if moe and self.shared is not None:
            widths.add(self.shared.shape[1])
        bound = self.placed[0][1].get((sig, path), ("",))[0]
        return (
            (spec.get("role") == "activation" or bound == "topk_ids")
            and len(spec["shape"]) == 2
            and not _dtype(spec["dtype"]).is_floating_point
            and spec["shape"][1] in widths
        )

    def _route(self, spec, c, drafted=False):
        """Copy `c`'s expert ids for `spec`: each row's experts drawn
        independently and uniformly, distinct, from the model's routed experts,
        its drafter's for a leaf of the drafter's alone (`drafted`), seeded by
        `SEED` and `c`, and past top-k the row's ids in ATOM's buffer; the
        copy's expert loads go to `loads`."""
        moe = self.draft_moe if drafted else self.moe
        if len(moe) > 1:
            raise ValueError(
                f"the model's MoE layers route (experts, top-k) {moe}: "
                f"which one draws {_text(spec)} is not known"
            )
        [(experts, k)] = moe
        device = self.runner.device
        ids = uniform_ids(spec["shape"][0], experts, k, SEED * CALLS + c, device)
        self.loads.append(torch.bincount(ids.flatten(), minlength=experts).tolist())
        out = _random(spec, device)
        out[:, :k] = ids
        if out.shape[1] > k:
            out[:, k:] = self.shared[: len(out), k:]
        return out

    def _reweigh(self, op, sig, like, c):
        """Copy `like` of `op`'s call with the c-th weight of each weight
        argument's spec and expert ids of its own, its other arguments and
        forward context shared."""
        drafted = op.get("drafted", False)

        def build(path, value, old):
            if isinstance(value, list):
                return type(old)(
                    build(f"{path}.{i}", v, o)
                    for i, (v, o) in enumerate(zip(value, old))
                )
            if isinstance(value, dict) and value.get("device") == "cpu":
                return old  # the copy's own, built on the host
            if isinstance(value, dict) and value.get("role") == "weight":
                return self._weight(value, c, op["occurrences"])
            if isinstance(value, dict) and self._routed(value, sig, path, drafted):
                return self._route(value, c, drafted)
            return old

        kwargs, snap = like
        return {s: build(s, v, kwargs[s]) for s, v in op["args"].items()}, snap

    def _tensor(self, spec, meta, fresh):
        role = spec["role"]
        if role == "activation" and meta is None:
            return _random(spec, self.runner.device)
        if role in ("metadata", "kv", "activation"):
            if meta[1] is None:
                raise ValueError(meta[0])
            return _clone(meta[1]) if fresh and role != "kv" else meta[1]
        raise ValueError(f"a {role} tensor argument {_text(spec)} is not built")

    def _weight(self, spec, c, count):
        """Weight `c mod count` of `spec`, random: the `count` of a leaf are
        built once, as slices of one tensor each starting on a `WEIGHT_ALIGN`
        boundary, each marked preshuffled when ATOM marked the captured one, so
        the operator picks the kernel it picked."""
        if (_spec_of(spec), 0) not in self.built:
            n = _span(spec["shape"], spec["stride"])
            dtype, device = _dtype(spec["dtype"]), self.runner.device
            room = -(-n * dtype.itemsize // WEIGHT_ALIGN) * WEIGHT_ALIGN
            step = room // dtype.itemsize
            whole = torch.empty(count * step, dtype=dtype, device=device)
            for i in range(count):
                w = whole[i * step : i * step + n]
                w.copy_(_random({**spec, "shape": [n], "stride": [1]}, device))
                view = w.as_strided(spec["shape"], spec["stride"])
                if spec.get("is_shuffled"):
                    view.is_shuffled = True
                self.built[_spec_of(spec), i] = view
        return self.built[_spec_of(spec), c % count]


def _alive(device) -> None:
    """Ends this process when `device` faulted: the fault is sticky, so every
    later call there fails, and a peer rank would wait on this one's next
    collective until it times out. `_measured` refuses the leaf it died on."""
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:  # noqa: BLE001 - any error here is the device's
        os._exit(1)


def _routing(loads) -> dict:
    """The drawn copies' imbalance, recorded and not priced: each copy's
    experts that took a token, and the most tokens one expert took."""
    return {
        "seed": SEED,
        "experts": len(loads[0]),
        "activated": [sum(n > 0 for n in load) for load in loads],
        "max_load": [max(load) for load in loads],
    }


@contextlib.contextmanager
def _uncommitted():
    """DeepSeek-V4.1's cache, when it is loaded, letting a batch begin over the
    tentative state of the verify step prepared before it, which no sampler
    here commits: that state is dropped rather than refused
    (`PagedAttentionCache.require_committed`), its verdict being the
    sampler's, which the bench never runs."""
    v41 = sys.modules.get("atom.model_ops.attentions.deepseek_v41.cache")
    if v41 is None:
        yield
        return

    def dropped(cache):
        cache.pending = None

    with patch.object(v41.PagedAttentionCache, "require_committed", dropped):
        yield


def point_rows(
    shape, context, pool: int, block_size: int, uncut: bool = False, drafts: int = 0
) -> list[list] | None:
    """A point's rows, `[query, context, decodes]` each, longest first: a decode
    row holds `context` tokens and, of a shape that names its rows alone, under
    a drafter of `drafts` verifies as many more, as the capture's rows do; a
    decode of fewer rows than its B, a drafter's own shape past its step's,
    holds those rows alone, its step padded to B (`Stub.point`); a
    prefill row its query tokens, the capture's split, past `context` cached
    ones, its step padded to its W where it states one. When they need more
    than the `pool`'s KV blocks, the first keeps its context and each other
    shares what it leaves, as the rows of a ragged step do. A ragged decode
    point's rows (`_split`) are its first row's context and, past it, rows of
    at least a token each that hold its mean; None when they need more than
    the pool, as a decode point's are when `uncut`."""
    decode = shape.phase == "decode"
    context, mean = _split(context)
    seqs = min(shape.tokens, shape.seqs) if decode else shape.seqs
    per, extra = divmod(shape.tokens, seqs)
    per += drafts if decode and shape.tokens <= shape.seqs else 0
    past = context - 1 if decode else context or 0
    rows = [[q, q + past, decode] for q in (per + (i < extra) for i in range(seqs))]
    if mean:
        share = (mean * context * len(rows) // PERMILLE - context) // (len(rows) - 1)
        for row in rows[1:]:
            row[1] = max(share, 1)
    blocks = [-(-row[1] // block_size) for row in rows]
    if sum(blocks) > pool and (mean or uncut and decode):
        return None
    if sum(blocks) > pool and len(rows) > 1:
        share = (pool - blocks[0]) // (len(rows) - 1) * block_size
        for row in rows[1:]:
            row[1] = min(row[1], max(share, row[0]))
    return rows


def _split(context) -> tuple:
    """A point's context and its rows' mean context in permille of it, None
    when its rows hold it alike: a ragged decode point's context is the
    pair."""
    return tuple(context) if isinstance(context, (tuple, list)) else (context, None)


def _specs(args):
    """Each tensor argument in `args`, by its slot path."""
    return [(p, v) for p, v in _values(args) if isinstance(v, dict) and "role" in v]


def _writes(name: str) -> set[str]:
    """The slots the operator `name` writes, as its schema marks them: an
    in-place operator's or an `out` argument; none for a lifted function."""
    # ponytail: a lifted function states no writes, so its clones are shared:
    # ATOM's lifted metadata writers (`kv_indices_generate_triton`,
    # `fill_deferred_decode_ids`, DeepSeek-V4's index builders) write what
    # ATOM's forward writes from the prepared metadata, which later leaves
    # read; one that writes other values needs its writes named here
    args = target._schema.arguments if (target := _overload(name)) else []
    return {a.name for a in args if a.alias_info and a.alias_info.is_write}


def _overload(name: str):
    """The `torch.ops` overload `name` names; None for a lifted function."""
    target = torch.ops
    for attr in [] if ":" in name else name.split("."):
        target = getattr(target, attr, None)
    return target if isinstance(target, torch._ops.OpOverload) else None


def _keyed(args, path) -> str:
    """A slot path's argument, a dict argument's item `slot.i.1` named by its
    key, `slot.<key>`."""
    slot, *rest = path.split(".")
    value = args.get(slot)
    if rest and isinstance(value, dict) and value.get("object") == "builtins:dict":
        return f"{slot}.{value['state'][int(rest[0])][0]}"
    return slot


def _values(args, path=""):
    """Each argument value in `args` by its slot path, `slot`, `slot.i` in a
    list, `slot.i.1` for an object's i-th attribute or item."""
    items = args.items() if isinstance(args, dict) else enumerate(args)
    for slot, value in items:
        where = f"{path}.{slot}" if path else str(slot)
        if isinstance(value, dict) and "object" in value:
            value = value["state"]
        if isinstance(value, list):
            yield from _values(value, where)
        else:
            yield where, value


class PassRows(ValueError):
    """A step whose passes run rows of one count: a leaf's rows tell no pass."""


def _passes(fc) -> dict:
    """A forward's passes by the rows each runs, each its step and real rows'
    count (`use_pass_rows`): DeepSeek-V4.1's decoder over its tail's rows and
    its finishing requests' last ones; the step's own under the key None."""
    step = getattr(fc.attn_metadata, "step", None)
    plan, passes = getattr(step, "decoder", None), {}
    if plan is not None:
        passes[plan.tail_rows.numel()] = plan.tail, plan.filled_gpu
        queries = dataclasses.replace(plan.tail, batch_ids=plan.query_ids)
        passes[plan.last.numel()] = queries, plan.finishing_gpu
    passes[None] = step, getattr(fc.context, "scheduled_tokens_gpu", None)
    return passes


def _pass_rows(op) -> int | None:
    """The rows a leaf runs: its positions', its batch ids' or first activation's."""
    specs = dict(_specs(op["args"]))
    for slot in ("positions", "batch_ids"):
        if slot in specs:
            return specs[slot]["shape"][-1]
    shape = next((s["shape"] for s in specs.values() if s["role"] == "activation"), [])
    return shape[1] if shape[:1] == [1] and len(shape) > 2 else (shape or [None])[0]


def _ratio(op, path, known=None):
    """The compress ratio `op` runs at for its slot `path`: its `ratio`
    argument, else the one it states; a list slot's i-th item the i-th of its
    `ratios` (DeepSeek-V4.1's indptr scan, a table a ratio); else the ratio
    `known` gives the rows a page holds at it (`rows_per_page`, V4.1's index
    build, 0 when it takes no selection) or the width of the logits it reads
    (`stride0`, V4.1's top-k, its scorer's), or a RoPE's positions the rows its
    compress plan holds (`compressed`)."""
    args, (slot, *rest) = op["args"], path.split(".")
    listed = args.get("ratios")
    if rest and isinstance(args.get(slot), list) and isinstance(listed, list):
        return listed[int(rest[0])]
    ratio = args.get("ratio", op.get("stated", {}).get("compress_ratio"))
    if ratio is None and "rows_per_page" in args and "selected" in args:
        ratio = (known or {}).get(("rows_per_page", args["rows_per_page"]))
        ratio = ratio if args["selected"] else 0
    if ratio is None and known and "stride0" in args:
        ratio = known.get(("stride0", args["stride0"]))
    if ratio is None and known and (op["op"], path) == (_ROPE, "positions"):
        ratio = known.get(("compressed", args[path]["shape"][-1]))
    return ratio


def _named(op, path, spec, pool, known=None) -> dict | None:
    """`pool`'s buffers `V41_NAMES` names for `op`'s `path`, at its ratio or each
    `known` one, of `spec`'s dtype, its shape and stride first if any; or None."""
    key = op["op"], _keyed(op["args"], path)
    role = (op["args"].get("self") or {}).get("role")
    if op["op"] == _ROPE:  # over one head a ratio's index keys, over more a query
        role = "key" if op["args"]["input"]["shape"][-2] == 1 else "query"
    wanted = V41_NAMES.get((*key, role), V41_NAMES.get(key, ()))
    wanted = wanted if isinstance(wanted, tuple) else (wanted,)
    ratio = _ratio(op, path, known)
    ratios = [ratio] if ratio is not None else sorted(set((known or {}).values()))
    names = [w.format(ratio=r) for w in wanted for r in ratios or [None]]
    held = {
        n: pool[n] for n in names if n in pool and str(pool[n].dtype) == spec["dtype"]
    }
    exact = {n: t for n, t in held.items() if _spec(t) == _spec_of(spec)}
    return exact or held or None


def _widened(op, blocks) -> dict:
    """`op` with the slots and integers `PAGE_WIDE` names for it as wide as
    `blocks[1]` KV blocks rather than the capture's `blocks[0]`: an integer,
    a tensor's columns, and its row stride where that is its columns."""
    bound, *wide = PAGE_WIDE.get(op["op"], (None,))
    if blocks[0] == blocks[1] or bound is None or _written(op, bound) == "context":
        return op
    op = copy.deepcopy(op)
    for slot in wide:
        value = op["args"][slot]
        if isinstance(value, int):
            op["args"][slot] = value // blocks[0] * blocks[1]
            continue
        shape, stride = value["shape"], value["stride"]
        if stride[0] == shape[1]:
            stride[0] = stride[0] // blocks[0] * blocks[1]
        shape[1] = shape[1] // blocks[0] * blocks[1]
    return op


def _summed(op, rows) -> dict:
    """`op`, when it is the batch-id build (`BATCH_IDS`), with its length the
    sum of `rows`' contexts, which it repeats: the point's KV tokens."""
    repeats = op["args"].get("repeats")
    if (op["op"], "repeats") not in BATCH_IDS or repeats.get("role") != "metadata":
        return op
    return {**op, "args": op["args"] | {"output_size": sum(r[1] for r in rows)}}


def _own_rows(op) -> dict:
    """`op`, when it is the DP all-gather (`GATHERV`), with each tensor of its
    `x` holding this DP rank's `sizes` entry's rows: the capture records rank
    0's call, and aiter fails a rank whose rows are not its entry."""
    if op["op"] != GATHERV:
        return op
    from aiter.dist.parallel_state import get_dp_group

    rows = op["args"]["sizes"][get_dp_group().rank_in_group]

    def own(t):
        return {**t, "shape": [rows, *t["shape"][1:]]}

    x = op["args"]["x"]
    x = [own(t) for t in x] if isinstance(x, list) else own(x)
    return {**op, "args": op["args"] | {"x": x}}


def _written(op, path) -> str | None:
    """Which index table of `_WRITTEN` `op`'s `path` takes: a scorer's on a
    plane its row bounds ATOM stages bound, the page-wide one, its tile table
    (`tiles`), else a candidate list's (`table`); an activation row bound the
    pick's (`ends`) under a top-k of activation logits, else a candidate
    list's (`context`); None for a row bound ATOM stages, or another slot."""
    kind = _WRITTEN.get((op["op"], path))
    role = {s: v.get("role") for s, v in op["args"].items() if isinstance(v, dict)}
    if kind == "tiles" and role.get("context_lens") != "metadata":
        return "table"
    if kind == "context" and role.get(path) != "activation":
        return None
    return "ends" if kind == "context" and role.get("logits") == "activation" else kind


def _candidates(ends, k) -> torch.Tensor:
    """`[rows, k]` int32 blocks each row of `ends` visible blocks keeps, as V4.1's
    candidate pick keeps them, ascending: every one when it sees no more
    than `k`, -1 past them; else `k` spread evenly from block 0 up to the
    newest, which the pick pins, last."""
    # ponytail: the pick keeps the blocks its scores peak in, which no placement
    # has. Block 0 first: the candidate lift rewrites its selection, zeros
    # (`_random`), in place through the first block, and a copy called again
    # reads what it wrote; through any other block it walks past the list
    run, seen = torch.arange(k, device=ends.device), ends.long()[:, None]
    kept = torch.where(run < seen, run, -1)
    spread = seen - 1 - (k - 1 - run) * (seen - 1) // max(k - 1, 1)
    return torch.where(seen > k, spread, kept).int()


def _meta(spec, slot, live, what="metadata") -> tuple[str, torch.Tensor]:
    """The live metadata tensor a metadata argument is a view of, and its name:
    one of its dtype with room for its span: one whose name the slot holds or
    that holds the slot's first, then one of its shape and stride, then the
    tightest."""
    need = _span(spec["shape"], spec["stride"])
    fits = [
        (name, t)
        for name, t in live.items()
        if str(t.dtype) == spec["dtype"] and _room(t) >= need
    ]
    if not fits:
        raise ValueError(f"ATOM's live {what} has no room for {slot} {_text(spec)}")
    slot = slot.split(".")[0]
    name, t = min(
        fits,
        key=lambda f: (
            not (f[0] in slot or slot in f[0]),
            _spec(f[1]) != _spec_of(spec),
            _room(f[1]),
        ),
    )
    return name, t.as_strided(spec["shape"], spec["stride"], t.storage_offset())


def _kv(spec, slot, live, ring=False) -> tuple[str, torch.Tensor | None]:
    """The live KV tensor a KV argument is a view of, and its name, picked as
    `_meta` picks among those whose layout past its widest-strided dim's size
    is its own (`_layout`), when any is: the capture's pool sized that one;
    else among the last layers, `name.i`, of tensors stacked by layer whose
    layer has that layout: ATOM passes a layer of DeepSeek-V4's compressor
    state, and another field's base at this one's slot stride reaches past
    the pool; else among the buffers of arenas, as its dtype views them
    (`_bound`), whose fields' strided views ATOM carves out of them, the
    capture's pool sizing a dimension other than the widest-strided;
    or why none fits, and None: a tensor of another layout is another buffer
    than the one ATOM passes. A flat one takes first an arena's buffer, of its
    own dtype first, whose whole entries (`_entry`) it fills, as the capture's
    pool sized it. Before an arena's buffer, one takes a tensor in one, of its
    strides and dtype, that it lies inside, cut from that tensor's end in
    every dim but the widest-strided (V4.1's cursor history). A `ring` (V4.1's
    `ring_view`) binds the arena buffer ending its tensor, past its start by the
    bytes it falls short of whole entries, or refuses past that buffer's."""
    shape, stride, dtype = spec["shape"], spec["stride"], _dtype(spec["dtype"])
    nbytes = _span(shape, stride) * dtype.itemsize
    bufs = {n: (t, _entry(n, live)) for n, t in live.items() if ".buf." in n}
    own = {
        n: t.view(dtype)
        for n, (t, e) in sorted(bufs.items(), key=lambda b: b[1][0].dtype != dtype)
        if stride == [1] and 0 < nbytes <= t.nbytes and not nbytes % e
    }
    want = _layout(spec["shape"], spec["stride"])
    own = own or {n: t for n, t in live.items() if _layout(t.shape, t.stride()) == want}
    own = own or {
        f"{n}.{len(t) - 1}": t[-1]
        for n, t in live.items()
        if t.dim() > 2 and len(t) and _layout(t[-1].shape, t[-1].stride()) == want
    }
    cut, dims = _widest(shape, stride), lambda t: enumerate(zip(t.shape, shape))
    own = own or {
        n: t[tuple(slice(0 if i == cut else m - k, None) for i, (m, k) in dims(t))]
        for n, t in live.items()
        if (t.stride(), t.dtype) == (tuple(stride), dtype)
        and all(m >= k for _, (m, k) in dims(t))
        and any(_inside(t, b) for b, _ in bufs.values())
    }
    flat = f".buf.{spec['dtype']}"
    own = own or {n: t for n, t in live.items() if n.endswith(flat)}
    if not own:
        return f"ATOM's KV holds no tensor laid out as {slot} {_text(spec)}", None
    try:
        name, view = _meta(spec, slot, own, "KV")
    except ValueError as error:  # the leaf refuses, not the point
        return str(error), None
    t, end = own[name], lambda x: x.storage_offset() + _span(x.shape, x.stride())
    for b, e in bufs.values() if ring else ():
        if end(b) * b.itemsize == end(t) * t.itemsize and _inside(b, t):
            n, room = -(-nbytes // e), b.nbytes // e
            if n > room:
                why = f"a window ring over {n} state slots; the bench holds {room}"
                return f"{slot} {_text(spec)}: {why}", None
            at = b.storage_offset() * b.itemsize + n * e - nbytes
            view = t.as_strided(shape, stride, at // t.itemsize)
    return name, view


def _entry(name, live) -> int:
    """The bytes an entry takes of the arena buffer `name` (`_bound`): its bytes
    over the entries of its fields' views, `[layers, entries, ...]`."""
    arena = name.split(".buf.")[0] + "."
    held = [t.shape[1] for n, t in live.items() if n.startswith(arena) and t.dim() > 1]
    return live[name].nbytes // (held or [1])[0]


def _layout(shape, stride) -> list:
    """Each dimension's size and stride, but the widest-strided one's size,
    which a pool sizes (`_widest`), and any other dimension of one past the first."""
    cut = _widest(shape, stride)
    dims = zip(range(len(shape)), shape, stride)
    return [(None if i == cut else n, s) for i, n, s in dims if n != 1 or i in (0, cut)]


def _widest(shape, stride) -> int:
    """The dimension of the widest stride, on a tie the first of size above one."""
    dims = [(s, n > 1, -i) for i, (n, s) in enumerate(zip(shape, stride))]
    return -max(dims, default=(0, 0, 0))[2]


def _held(runner, registry) -> dict[str, torch.Tensor]:
    """Each tensor ATOM's KV allocation bound, by name: the runner attributes
    it set (`kv_held`), a list's items as `name.i`, each field of a KV
    `registry` entry as `layer_<n>.field`, the block views of the KV planes
    it bound on DeepSeek-V4's attention compressors, which no runner
    attribute holds, as `<module>.kv_cache` and `<module>.kv_cache_rope` (an
    indexer's compressor writes a layer of a pool the runner holds), and the
    scales it bound on any V4 compressor, an indexer's included, as
    `<module>.cache_scale`: under an FP8 indexer cache, a strided view of
    the scale region of the pool layer the compressor writes."""
    named = [
        (f"{key}.{field}", t)
        for key, entry in (registry or {}).items()
        for field, t in vars(entry).items()
    ]
    model = getattr(runner, "model", None)
    for path, module in model.named_modules() if model is not None else ():
        if path.endswith("attn.compressor"):
            named += [
                (f"{path}.{a}", getattr(module, a, None))
                for a in ("kv_cache", "kv_cache_rope")
            ]
        if path.endswith("compressor"):
            named.append((f"{path}.cache_scale", getattr(module, "cache_scale", None)))
    for name, value in runner.kv_held.items():
        if isinstance(value, (list, tuple)):
            named += [(f"{name}.{i}", t) for i, t in enumerate(value)]
        else:
            named.append((name, value))
    return {name: t for name, t in named if torch.is_tensor(t)}


def _bound(name, value, old, device) -> dict[str, torch.Tensor]:
    """Each tensor on `device` `value` holds, by each path from `name`, the
    shortest first, through items, attributes, an arena's fields (`view`) and
    its buffer in each field's dtype; none `old` holds by id, which gains each
    object entered, nor a module."""
    out, queue = {}, [(name, value)]
    for name, value in queue:  # breadth first: the walk appends as it goes
        if torch.is_tensor(value):
            if value.device.type == device.type and id(value) not in old:
                out[name] = value
        elif id(value) not in old and not isinstance(value, torch.nn.Module):
            old.add(id(value))
            items = ()
            if isinstance(value, dict):
                items = value.items()
            elif isinstance(value, (list, tuple)):
                items = enumerate(value)
            elif hasattr(value, "fields") and callable(getattr(value, "view", None)):
                dtypes = sorted({f.dtype for f in value.fields}, key=str)
                items = [(f.name, value.view(f.name)) for f in value.fields]
                items += [(f"buf.{t}", value.buf.view(t)) for t in dtypes]
            elif hasattr(value, "__dict__"):
                items = vars(value).items()
            queue += [(f"{name}.{key}", item) for key, item in items]
    return out


def _cut_from(t, held) -> bool:
    """Whether `t` is a view cut from one of the `held` tensors: inside it, of
    its dtype and layout (`_layout`). `_kv`, taking the tightest of a layout,
    would bind the cut at its own base, where ATOM passes its leaves the tensors
    it was cut from (DeepSeek-V4's slot views of its KV planes, one a state slot)."""
    like = t.dtype, _layout(t.shape, t.stride())
    return any(
        (h.dtype, _layout(h.shape, h.stride())) == like and _inside(t, h) for h in held
    )


def _state(runner, held) -> dict[str, bool]:
    """The names of the `held` tensors that hold per-request state, each also
    as `_kv` names its last layer: the fields of an arena of a request's state
    an entry that the runner holds (DeepSeek-V4's compressor state), and each
    tensor inside the state arena of a metadata builder's cache (`BUILDERS`,
    DeepSeek-V4.1's `cache.state`: its fields, a layer of one, the cursor),
    each to False; and each that holds such an arena past its pages as one
    view (V4.1's `cache.pool`, whose window rings lie in it) to True.
    A KV registry entry of it caps a point's placements at its slots instead."""
    from atom.model_ops.attentions.pool_layout.entry_arena import SplitEntryMajorArena

    state = []
    for arena in runner.kv_held.values():
        if isinstance(arena, SplitEntryMajorArena):
            state += [p.view(f.name) for p in arena.arenas for f in p.fields]
    at = {(t.data_ptr(), t.shape, t.stride()) for t in state}
    caches = [getattr(getattr(runner, b, None), "cache", None) for b in BUILDERS]
    bufs = [s.buf for c in caches if hasattr(s := getattr(c, "state", None), "buf")]
    named = {
        n: False
        for name, t in held.items()
        if (t.data_ptr(), t.shape, t.stride()) in at
        or any(_inside(t, buf) for buf in bufs)
        for n in (name, f"{name}.{len(t) - 1}" if t.dim() else name)
    }
    return {n: True for n, t in held.items() for b in bufs if _inside(b, t)} | named


def _inside(t, buf) -> bool:
    """Whether `t`'s bytes lie within `buf`'s, of one storage."""
    from torch.multiprocessing.reductions import StorageWeakRef

    if StorageWeakRef(t.untyped_storage()) != StorageWeakRef(buf.untyped_storage()):
        return False
    at = buf.storage_offset() * buf.element_size()
    start = t.storage_offset() * t.element_size()
    end = start + _span(t.shape, t.stride()) * t.element_size()
    return at <= start and end <= at + buf.nbytes


def _function(name: str):
    """The callable a leaf names: `ns.op.overload`, or a lifted `module:qualname`;
    a functional collective waited on, as the in-place one ATOM calls is, on
    this rank's group of the coordinator it was captured on (`_on_coordinator`);
    a lifted collective that takes a group on the one ATOM calls it on
    (`DP_GROUP`), refused when ATOM's group is not known; varargs by position."""
    if ":" in name:
        module, qualname = name.split(":")
        fn, path = importlib.import_module(module), qualname.split(".")
    else:
        namespace, rest = name.split(".", 1)
        fn, path = getattr(torch.ops, namespace), rest.split(".")
    for attr in path:
        fn = getattr(fn, attr)
    if name.startswith(COLLECTIVES):
        if name in DP_GROUP:  # the capture drops the group
            from aiter.dist.parallel_state import get_dp_group

            return functools.partial(fn, group=get_dp_group())
        if ":" in name and "group" in inspect.signature(fn).parameters:
            raise ValueError(f"{name} takes a group no leaf name maps to")
        if name.startswith("_c10d_functional."):
            wait, fn = torch.ops._c10d_functional.wait_tensor, _on_coordinator(name, fn)
            return lambda **kwargs: wait(fn(**kwargs))
    params = list(inspect.signature(fn).parameters) if inspect.isfunction(fn) else []
    kinds = [inspect.signature(fn).parameters[p].kind for p in params]
    if (star := inspect.Parameter.VAR_POSITIONAL) in kinds:
        lead, slot = params[: kinds.index(star)], params[kinds.index(star)]
        return lambda **kw: fn(*map(kw.pop, lead), *kw.pop(slot, ()), **kw)
    return fn


def _on_coordinator(name: str, fn):
    """`fn` on this rank's device group of the aiter coordinator a call's
    `group_name` holds (`tp:0`), whatever torch named that group here; refused
    naming the coordinator when this rank has none by that name."""

    def call(**kwargs):
        from aiter.dist.parallel_state import _groups

        ref = _groups.get(coordinator := kwargs["group_name"])
        if (group := ref and ref()) is None:
            raise ValueError(
                f"{name} on coordinator {coordinator!r}, which rank "
                f"{torch.distributed.get_rank()} does not hold: {sorted(_groups)}"
            )
        return fn(**kwargs | {"group_name": group.device_group.group_name})

    return call


def _identity(fn, kwargs, timed=False) -> tuple[list[str], float]:
    """`_kernels` of one call of `fn`, memory copies aside, once two profiles
    in a row name the same kernels on every rank, at most `PROFILES`; refused
    on every rank together when no two do. Of a call whose graph `timed`
    device work, a profile that recorded none, kernel or copy, missed the
    call and agrees with no other."""
    from atom.compass.bench.ranks import agreed, cpu_group

    last = None
    for _ in range(PROFILES):
        names, ns = _kernels(fn, kwargs)
        kernels = [n for n in names if not n.startswith(COPY)]
        seen = None if timed and not names else kernels
        if agreed(int(seen is not None and seen == last), cpu_group()):
            return seen, ns
        last = seen
    raise ValueError(
        f"no two profiles in a row of {PROFILES} named the same kernels; "
        f"the last named {names}"
    )


def _kernels(fn, kwargs) -> tuple[list[str], float]:
    """The kernels one call of `fn` launches, in order, or its memory copies
    when it launches no kernel, and the kernels' device time in ns: beside a
    kernel its memory copies are left out, which the profiler records at
    some calls of a leaf and misses at others; started on every rank
    together (`_together`), once the device has finished `_together`'s
    all-reduce, which would otherwise join the call's kernels."""
    from torch.profiler import ProfilerActivity, profile

    _together()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(**kwargs)
        torch.cuda.synchronize()
    device = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    device.sort(key=lambda e: e.time_range.start)
    kernels = [e for e in device if not e.name.startswith(COPY)]
    ns = 1e3 * sum(e.time_range.elapsed_us() for e in kernels)
    return [e.name for e in kernels or device], ns


def _replay(fn, calls, flush=None) -> list[float]:
    """Each timed replay of one graph of `calls`, in ns per call, each started
    on every rank together (`_together`) after one read of `flush` when
    given, outside the timed events, the capture refused on every rank
    when any rank's refused, and nothing it left read by a later one
    (`_contained`), captured once the memory earlier leaves freed is cleared
    (`_release`). Over several ranks, a decode step's graph is captured as
    ATOM captures one, each collective on the buffers it registered. The
    graph registers ATOM's forced-acceptance generators (`_synthetic`) before
    its capture, as the capture registers the default one. Before the
    capture, the first call runs once on the stream it captures on, as ATOM's
    runner warms a graph's stream before it captures: aiter's Mega-MHC grows
    a scratch per stream, and refuses to while that stream captures."""
    from atom.compass.bench.ranks import cpu_group, refused_together
    from atom.utils import forward_context

    registered = cpu_group() is not None and not calls[0][1].context.is_prefill
    if registered:
        from aiter.dist.parallel_state import graph_capture
    graph = torch.cuda.CUDAGraph()
    forward_context._forward_context = calls[0][1]
    with refused_together():
        capture = torch.cuda.graph(graph, capture_error_mode="thread_local")
        stream, current = capture.capture_stream, torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            fn(**calls[0][0])
        current.wait_stream(stream)
    _release()  # the capture's own release then has nothing to return
    for generator in _synthetic():
        graph.register_generator_state(generator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with (
            refused_together(),
            graph_capture() if registered else contextlib.nullcontext(),
            _contained(capture),
        ):
            for kwargs, snap in calls:
                forward_context._forward_context = snap
                fn(**kwargs)
    if any("Graph is empty" in str(w.message) for w in caught):
        return [0.0] * REPEATS
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for i in range(WARMUPS + REPEATS):
        if flush is not None:
            # read, not written, so the replay writes back no line of it; either
            # way it pushes the graph's kernel arguments, cached between
            # replays, out to HBM, and on MI355X each kernel then waits about
            # 0.3 us longer for them
            flush.max()
        _together()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if i >= WARMUPS:
            times.append(ms * 1e6 / len(calls))
        time.sleep(ms * REST / 1e3)
    return times


def _release() -> None:
    """The caching allocator's free memory returned to the driver, then a wait,
    at most `CLEAR_S`, until the card's VRAM in use falls by it, added to
    `_waited`. The driver clears VRAM a process returns in the background,
    about 31 GiB/s on MI308X, and a memory-bound leaf timed during the clear
    read up to 1.20 of its time after it."""
    used, held = _vram_used(), torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    if used is None:
        return
    target = used - (held - torch.cuda.memory_reserved()) + CLEAR_SLACK
    start = time.monotonic()
    while (over := _vram_used() > target) and time.monotonic() - start < CLEAR_S:
        time.sleep(0.005)
    _waited["s"] += time.monotonic() - start
    _waited["capped"] += over


def _vram_used() -> int | None:
    """The bytes in use on this process's GPU as its driver counts them, or
    None when the process has no GPU in use."""
    from atom.compass.drive.identity import PCI

    if not torch.cuda.is_initialized():
        return None
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return int(Path(PCI, bus, "mem_info_vram_used").read_text())


@contextlib.contextmanager
def _contained(capture):
    """`capture` entered with no garbage collection, and when it fails,
    undone what it leaves behind for later leaves: its stream current, each
    generator in capture, and the capture HIP leaves open on its stream when
    a call invalidated it. A failed capture's graph can sit in a reference
    cycle, and a graph freed during a capture synchronizes the device, which
    ends the process."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        with capture:
            yield
    except BaseException:
        if torch.cuda.current_stream() == capture.capture_stream:
            capture.stream_ctx.__exit__(None, None, None)
        with torch.cuda.stream(capture.capture_stream):
            stuck = torch.cuda.is_current_stream_capturing()
        if stuck:
            _end_capture(capture.capture_stream)
        default = torch.cuda.default_generators[torch.cuda.current_device()]
        for generator in (default, *_synthetic()):
            generator.graphsafe_set_state(generator.clone_state())
        raise
    finally:
        if collecting:
            gc.enable()


def _synthetic() -> list[torch.Generator]:
    """The generators ATOM's forced acceptance draws from, one a device it has
    drawn on (`rejection_sampler._synthetic_generator`): in the bench, the
    leaf's, made by the leaf's eager call before its capture. None before
    ATOM imports its sampler, which has made none then: read where it is
    loaded, not imported, as importing it imports aiter."""
    sampler = sys.modules.get("atom.model_ops.rejection_sampler")
    return list(sampler._SYNTHETIC_GENERATORS.values()) if sampler else []


def _end_capture(stream) -> None:
    """End the capture HIP keeps open on `stream` after its end refused it, by
    beginning and ending an empty one on it, through the HIP runtime this
    process loaded."""
    import ctypes

    maps = Path("/proc/self/maps").read_text().splitlines()
    hip = ctypes.CDLL(next(m.split()[-1] for m in maps if "libamdhip64" in m))
    handle, graph = ctypes.c_void_p(stream.cuda_stream), ctypes.c_void_p()
    if hip.hipStreamBeginCapture(handle, 2) or hip.hipStreamEndCapture(  # relaxed
        handle, ctypes.byref(graph)
    ):
        raise RuntimeError(f"the capture HIP left open on {stream} does not end")
    hip.hipGraphDestroy(graph)


def _together() -> None:
    """Over several ranks, a barrier on the host, then one all-reduce each
    rank's stream waits on, so what is launched next starts on every rank's
    device as on the others': a collective's time holds no peer's lateness."""
    from atom.compass.bench.ranks import cpu_group

    if (group := cpu_group()) is None:
        return
    from aiter.dist.parallel_state import get_world_group

    torch.distributed.barrier(group=group)
    device, flag = get_world_group().device_group, torch.zeros(1, device="cuda")
    torch.distributed.all_reduce(flag, group=device)


def _topk_workspace(name, rows, stride, k, stable, device) -> tuple[torch.Tensor, bool]:
    """The `workspace` aiter's top-k leaf `name` launches with on `device`
    for `rows` rows at row stride `stride` and `k`, as aiter's wrappers build
    it: on the multi-block path, which a prefill takes unless `stable` where
    the device's `topk_use_mulblocks` says so, aiter's cached buffer, zeroed
    once and reset by its kernel after each launch; on the one-block path,
    fresh scratch. With it, whether it is that cached buffer."""
    from aiter.ops import topk

    decode = TOPK_WORKSPACE[name]
    if not (decode or stable) and topk.topk_use_mulblocks(rows, stride):
        size = topk.topk_mb_workspace_size(rows, stride, k, False)
        return topk.get_topk_mb_workspace(device, size), True
    size = topk.topk_ob_workspace_size(rows, stride, k, decode)
    return topk.get_topk_scratch_workspace(device, size), False


def _span(shape, stride) -> int:
    """Elements a view of `shape` and `stride` reaches from its offset."""
    if 0 in shape:
        return 0
    return 1 + sum((n - 1) * s for n, s in zip(shape, stride))


def _room(t) -> int:
    return t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()


def _dtype(text: str) -> torch.dtype:
    return getattr(torch, text.removeprefix("torch."))


def _spec(t) -> tuple:
    return tuple(t.shape), str(t.dtype), tuple(t.stride())


def _spec_of(spec: dict) -> tuple:
    return tuple(spec["shape"]), spec["dtype"], tuple(spec["stride"])


def _text(spec: dict) -> str:
    return f"{spec['shape']} {spec['dtype']} stride {spec['stride']}"


def _random(spec, device) -> torch.Tensor:
    """A tensor of `spec`'s shape, dtype and stride, random over its span when
    floating, else zero: an integer one may index a dimension of one. Packed
    fp4, which no cast reaches, takes random bytes: every e2m1 code is finite."""
    dtype = _dtype(spec["dtype"])
    n = _span(spec["shape"], spec["stride"])
    if dtype == torch.float4_e2m1fn_x2:
        flat = torch.randint(256, (n,), dtype=torch.uint8, device=device).view(dtype)
    elif dtype.is_floating_point:
        flat = torch.randn(n, device=device).mul_(0.1).to(dtype)
    else:
        flat = torch.zeros(n, device=device, dtype=dtype)
    return flat.as_strided(spec["shape"], spec["stride"])


def _host(spec, pinned) -> torch.Tensor:
    """`spec` on the host, drawn as `_random` draws it, in pinned memory when
    `pinned`: a CUDA graph replays a copy from pinned memory only, and a
    capture cannot tell pinned from pageable."""
    drawn = _random(spec, "cpu")
    out = torch.empty_strided(
        drawn.shape, drawn.stride(), dtype=drawn.dtype, device="cpu", pin_memory=pinned
    )
    return out.copy_(drawn)


def _scores(plane, ends, spec) -> None:
    """`plane`'s columns below each row's bound in `ends` drawn as an
    activation of `spec` is (`_random`); the rest as they were."""
    drawn = _random(spec, plane.device)
    below = torch.arange(plane.shape[-1], device=plane.device) < ends.reshape(-1, 1)
    plane.copy_(torch.where(below, drawn, plane))


def _clone(t) -> torch.Tensor:
    """`t` on storage of its own, its span copied, its shape and stride kept."""
    n = _span(t.shape, t.stride())
    flat = t.as_strided((n,), (1,), t.storage_offset()).clone()
    return flat.as_strided(t.shape, t.stride())


def _deep(x, memo, clones=None):
    """`x` with every tensor it reaches through dataclasses, ATOM objects,
    dicts and lists cloned, aliases kept; with `clones`, a tensor whose bits
    have not changed since the last clone of its live view takes that clone:
    an uninitialised buffer's NaNs equal nothing by value."""
    if id(x) in memo:
        return memo[id(x)]
    if torch.is_tensor(x):
        key = x.data_ptr(), x.shape, x.stride(), x.dtype
        twin = None if clones is None else clones.get(key)
        bits = BITS.get(x.element_size(), x.dtype)
        if twin is None or not torch.equal(twin.view(bits), x.view(bits)):
            twin = _clone(x)
        memo[id(x)] = twin
        if clones is not None:
            clones[key] = twin
    elif isinstance(x, dict):
        memo[id(x)] = {k: _deep(v, memo, clones) for k, v in x.items()}
    elif isinstance(x, list):
        memo[id(x)] = [_deep(v, memo, clones) for v in x]
    elif dataclasses.is_dataclass(x) or type(x).__module__.startswith("atom.utils"):
        if not hasattr(x, "__dict__"):
            return x
        memo[id(x)] = twin = copy.copy(x)
        vars(twin).update({k: _deep(v, memo, clones) for k, v in vars(x).items()})
    else:
        return x
    return memo[id(x)]


def _moe(model) -> list:
    """The `(experts, top-k)` of each MoE layer `model` holds."""
    return sorted(
        {
            (m.global_num_experts, m.top_k)
            for m in model.modules()
            if hasattr(m, "global_num_experts")
        }
    )


def _own(module):
    """The tensors `module` holds itself: attributes, parameters and buffers."""
    held = [*vars(module).values(), *module._parameters.values()]
    return [t for t in [*held, *module._buffers.values()] if torch.is_tensor(t)]


def _fill(module, device) -> None:
    """`module`'s meta parameters moved to `device` and filled as ATOM's
    `--load_dummy xavier` fills a model's."""
    meta = [
        (m, n, p)
        for m in module.modules()
        for n, p in m._parameters.items()
        if p is not None and p.is_meta
    ]
    if not meta:
        return
    from atom.model_loader.loader import initialize_dummy_weights

    for m, n, p in meta:
        t = torch.empty_like(p, device=device)
        m._parameters[n] = torch.nn.Parameter(t, requires_grad=False)
    initialize_dummy_weights(module, "xavier")


def _module_copy(module):
    """`module` with its own tensors, parameters and buffers cloned."""
    twin = copy.copy(module)
    for name, value in vars(module).items():
        if torch.is_tensor(value):
            vars(twin)[name] = _clone(value)
        elif name in ("_parameters", "_buffers"):
            vars(twin)[name] = {
                k: v if v is None else _clone(v) for k, v in value.items()
            }
    return twin


def _stateful(held) -> bool:
    return bool(getattr(held, "per_request_state", False))


def _fields(held):
    for name, value in vars(held).items() if held is not None else ():
        if torch.is_tensor(value) and value.dim():
            yield name, value


def keys(config, run: dict):
    """The `op_measurement` key of a point, `(shape, context, leaf)`, measured
    under `config` on the platform `run` reads."""
    from atom.compass.artifacts.keys import LAYOUT, platform_key
    from atom.compass.fit.samples import run_fields
    from atom.compass.run.memory import key_fields

    fields = run_fields(run)
    platform = platform_key(fields["device"], run["stack"], run["ptl"]).canonical()
    layout = {k: v for k, v in key_fields(config).items() if k in LAYOUT}

    def key(shape, context, leaf) -> Key:
        point = json.dumps([str(shape), context, leaf]).encode()
        return Key.of(
            Kind.OP_MEASUREMENT,
            model=fields["model"],
            **layout,
            machine_spec="sha256:" + hashlib.sha256(platform.encode()).hexdigest(),
            source_root=fields["source_root"],
            point="sha256:" + hashlib.sha256(point).hexdigest(),
        )

    return key


def publish(store, key, config, run: dict, command: str, results, out: Path):
    """Write `results` to ``ops.jsonl`` in `out` and each measured one to
    `store` at its `key`."""
    from atom.compass.fit.publish import conditions, ranks
    from atom.compass.fit.samples import run_fields
    from atom.compass.run.step_grid import keyed

    readings = conditions(
        {**run_fields(run), "engine_config": keyed(config)}
        | {k: run[k] for k in ("stack", "host_key", "ptl")}
    )
    roots = roots_for({"atom": "atom", "aiter": "aiter"})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out / "ops.jsonl").open("w") as lines:
        for record in results:
            at = key(record["shape"], record["context"], record["leaf"])
            lines.write(json.dumps(record | {"token": at.token()}) + "\n")
            if "refused" not in record:
                body, every = json.dumps(record).encode(), ranks(at)
                store.publish(
                    at,
                    provenance=Provenance(command, now, roots),
                    topology=every[0].topology,
                    conditions=readings,
                    gates=GateState.of(),
                    members={member_name("point", r, "json"): body for r in every},
                )


#: A ladder keeps a value only this many times the last it kept, and the largest.
LADDER_STEP = 1.25
#: A ragged decode point's rows' mean is a whole number of these parts of their
#: max. At 16, MI308X decode attention at 6 rows and 212k tokens read 5.5% off
#: its neighbours at a mean of 438 permille, past `fit ops`'s 5%.
GRID = 32
#: The spread of a leaf's cold times across contexts, or a prefill leaf's
#: across rows, over the largest, past which its kind depends on them; a
#: repeat's spread stays within 4%.
CONTEXT_MOVES = 0.10
#: How far the line through a decode point's two neighbours may price it off its
#: measured cold time, over that time as `fit ops` reads a held-out error, when
#: the three launched one kernel, before the plan halves the gaps either side;
#: and how far a gap's ends' cold times may differ, over the lower, before the
#: plan halves it where the leaf's kind steps (`_apart`). A collective's point
#: over its message bytes is read as a decode one.
BENDS = 0.05
#: The spread of a leaf's cold times across contexts, or a prefill leaf's
#: across rows, in ns, at or under which its kind does not depend on them
#: whatever its share: one 3.5 us elementwise kernel on MI308X reads in two
#: modes about 0.5 us apart, either one at any shape.
CONTEXT_FLOOR_NS = 1000


def plan(store, key, config, run, args, command: str, out: Path, atom_args=()) -> None:
    """Measure the points the steps of `--plan`'s record directories call for,
    round by round, numbered on from the last round `out` holds, and after
    each round write every point held, a measured one with its store token, to
    plan.jsonl in `out`."""
    from atom.compass.capture.forward import decode
    from atom.compass.runner.overrides import schedulable_ladder

    steps = parts(args.plan, config)
    # a drafter's own shapes, past their steps', and the outputs its graph is
    # read at: a leaf a decode's width's graphs hold is that width's point,
    # and a prefill's family (`_family`) states no W, so a leaf its step's
    # graphs hold is the step's. G and W are None or a count
    own = defaultdict(set)
    for _, reads in steps:
        for part, shape, output, _ in reads[1:]:
            if part == DRAFT and shape != reads[0][1]:
                own[shape].add(output)
    steps = [(reads[0][1], line) for line, reads in steps]
    contexts = [
        c for s, line in steps if s.phase == "decode" for c in _rows(line, s.seqs)
    ]
    if not contexts:
        raise ValueError("the --plan records hold no decode step to read contexts of")
    ragged = any(
        _mean(line, s.seqs) < PERMILLE for s, line in steps if s.phase == "decode"
    )
    rungs = [decode(r) for r in schedulable_ladder(config)]
    graphs, done, tried = {}, {}, set()
    prefills, cached = set(), defaultdict(list)
    for s, line in steps:
        if s.phase == "prefill":
            prefills.add(s)
            cached[s.seqs] += _cached(line)
    # one that is a step's shape too is measured whole, as that step's
    own = {s: outputs for s, outputs in own.items() if s not in prefills}
    # first, so a graph only `trace ops --population` traces refuses before any point
    widths = {decode(s.seqs) for s in own if s.phase == "decode"}
    shapes = prefills | set(own) | widths
    _load(store, config, args.store, shapes, graphs, None, atom_args, args.plan, own)
    for s in own:
        if s.phase == "prefill":
            prefills.add(s)
            continue
        width = graphs[decode(s.seqs)]
        graphs[s] = {sig: leaf for sig, leaf in graphs[s].items() if sig not in width}
    # a run resumed into `out` keeps the rounds an earlier one wrote there
    rounds = [int(d.name[5:]) for d in out.glob("round*") if d.name[5:].isdigit()]
    first = max(rounds, default=-1) + 1

    def measure_round(want, n):
        dependent = _dependent(done.values())
        _load(store, config, args.store, {s for s, _ in want}, graphs, None, atom_args)
        todo = []
        for (shape, context), families in want.items():
            found = {}
            for sig, (op, instances) in graphs[shape].items():
                fam = _family(shape, context, op["op"], sig, dependent)[0]
                if (
                    (str(shape), context, sig) in done
                    or families
                    and fam not in families
                ):
                    continue
                if store.directory_for(at := key(shape, context, sig)).exists():
                    entry = store.read(at)
                    held = entry.read_member("point", entry.topology.ranks()[0], "json")
                    done[str(shape), context, sig] = json.loads(held)
                else:
                    found[sig] = op, instances
            todo += [(shape, context, found)] if found else []
        print(f"round {first + n}: {sum(len(f) for *_, f in todo)} points to measure")
        gpus = len(run["device"])
        # round 0 cuts: `_dependent` reads its points for which kinds move
        with patch.dict(os.environ, {RAGGED: "1"} if ragged and n else {}):
            results = _measured(config, todo, args.llc_mib << 20, gpus)
        (into := out / f"round{first + n}").mkdir()
        publish(store, key, config, run, command, results, into)
        done.update((_point(r), r) for r in results)
        with (out / "plan.jsonl").open("w") as lines:
            for r in done.values():
                if "refused" not in r:  # a refused point is not in the store
                    r = r | {"token": key(r["shape"], r["context"], r["leaf"]).token()}
                lines.write(json.dumps(r) + "\n")

    decodes = [*rungs, *sorted((s for s in own if s.phase == "decode"), key=str)]
    measure_round(
        {(s, c): None for s in decodes for c in (min(contexts), max(contexts))}, 0
    )
    dependent, want = _dependent(done.values()), defaultdict(set)
    for shape in decodes:
        for sig, (op, _) in graphs[shape].items():
            if dependent.get(op["op"], True) and not exchange(op):
                for c in _ladder(contexts):
                    means = _means(shape.tokens) if ragged else []
                    for at in [c, *((c, m) for m in means)]:
                        fam = _family(shape, at, op["op"], sig, dependent)[0]
                        want[shape, at].add(fam)
    seen = defaultdict(dict)
    for shape in sorted(prefills, key=str):  # G and W are None or a count
        for sig, (op, _) in graphs[shape].items():
            fam, tokens = _family(shape, None, op["op"], sig, dependent)
            # `fit ops` keeps a shape that states G apart from one that does not
            seen[fam, not shape.gathered].setdefault(tokens, (shape, op["op"], sig))

    def cached_ladder(top_only: bool) -> dict:
        """The points of each prefill family over its rows' cached tokens, of
        a kind `_dependent` reads as depending on context: only the most, its
        reading's far end, while the kind has no cached point to read. None whose
        first row, the cached tokens and its share of the token count, holds
        more than `max_model_len` tokens, a row ATOM's runner cannot take."""
        dependent, want = _dependent(done.values()), defaultdict(set)
        for at in seen.values():
            for tokens in _ladder(at):
                shape, op, sig = at[tokens]
                ladder = _ladder(cached[shape.seqs])
                room = config.max_model_len - -(-shape.tokens // shape.seqs)
                for c in ladder[-1:] if top_only else ladder:
                    fam = _family(shape, c, op, sig, dependent)[0]
                    if c and c <= room and _cut(fam):
                        want[shape, c].add(fam)
        return want

    for (fam, _), at in seen.items():
        for tokens in _ladder(at):
            want[at[tokens][0], None].add(fam)
    for point, fams in cached_ladder(top_only=True).items():
        want[point] |= fams
    n = 1
    while want:
        measure_round(want, n)
        n += 1
        want = cached_ladder(top_only=False) if n == 2 else {}
        families = _families(done.values(), _dependent(done.values()))
        want = want or _densify(families, tried)
        want = want or _filled(families, tried, config.max_model_len)
    kinds = Counter(r["op"] for r in done.values() if "refused" not in r)
    print(json.dumps({"points per leaf kind": dict(kinds.most_common())}, indent=1))
    print(f"{sum(kinds.values())} of {len(done)} points measured in {n} rounds")


def _coverage_main(argv: list[str], atom_args: list[str]) -> dict:
    from atom.compass.bench import ranks
    from atom.compass.bench.measure import atom_config
    from atom.compass.capture import driverless

    p = argparse.ArgumentParser(prog=f"{PROG} --coverage")
    p.add_argument("--coverage", action="append", required=True, metavar="RECORD")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="the --plan run's --out")
    args = p.parse_args(argv)
    out = Path(args.out)
    lines = (out / "plan.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    store = ArtifactStore(args.store)
    arch = json.loads((out / "run.json").read_text())["device"][0]["arch"]
    if arch not in driverless.DEVICES:
        declared = sorted(driverless.DEVICES)
        raise ValueError(f"no declared device for {arch}; declared: {declared}")
    with tempfile.TemporaryDirectory() as tmpdir:
        # aiter asks rocminfo for an arch at import, which the config reaches.
        driverless._declare_arch(tmpdir, arch)
        config = atom_config(atom_args)
        ranks.layout(config)
        _eager(config)
        names = args.coverage
        return coverage(store, args.store, config, names, records, arch, atom_args)


def _eager(config) -> None:
    """Refuse `config` unless ATOM runs it eager: `trace ops` captures and keys
    every `op_graph` eager, so a graph-mode key names no graph it publishes."""
    if not config.enforce_eager:
        raise ValueError(
            "ATOM's arguments run the model in CUDA graphs, and trace ops "
            "publishes eager op_graphs only; measure with --enforce-eager"
        )


def coverage(store, root, config, names, records, arch, atom_args=()) -> dict:
    """How the measured points among `records` cover the steps of the record
    directories `names`, each micro-batch of a step its DP group split under
    two-batch overlap a step: the steps and those covered by phase and by part
    (`parts`), and per leaf kind, a drafter's prefixed as a run names its
    terms, the steps it left out by why. A step is covered when each leaf of
    each of its parts' `op_graph`s lies on a point or between two of one
    kernel (`_bracket`): a
    decode leaf of a kind that depends on context over its rows' lowest to
    highest context (`_rows`), any other decode leaf at its width, a prefill
    leaf at its tokens, and one of a kind that depends on context over its
    rows' cached tokens too (`_bracket_cached`); and `fit ops`' price list of
    `records` prices it at the step, as `compass run --coverage` asks it. A
    decode leaf is read at the rows it ran, its shape's T: of a drafter's own
    decode shape past its step's, a leaf its width's graphs hold at that
    width, as `--plan` measures it. The graphs are `arch`'s; a part's graph
    the store does not hold leaves the step out, by its refusal."""
    from atom.compass.artifacts import ArtifactRefusal
    from atom.compass.backends.op_graph import band
    from atom.compass.backends.provenance import Refusal
    from atom.compass.capture.forward import decode
    from atom.compass.fit.ops import PriceList, fit, held_out

    steps = parts(names, config)
    dependent = _dependent(records)
    held_out(params := fit([r for r in records if "refused" not in r]))
    listed = PriceList(params)
    axes, cuts = {}, defaultdict(dict)
    for fam, points in _families(records, dependent).items():
        axes[fam] = [p[0] for p in points], [p[1] for p in points]
        if cut := _cut(fam):
            cuts[cut[0]][cut[1]] = axes[fam]
    report, out, by_part = defaultdict(Counter), defaultdict(Counter), {}
    targets = {shape for _, reads in steps for part, shape, *_ in reads if not part}
    _load(store, config, root, targets, {}, arch, atom_args, names)

    @functools.cache
    def families(shape, output, draft, own):
        try:  # a target's graph at output as `graph` reads it by default
            which = [output, draft] if draft else []
            region = graph(store, config, shape, arch, *which)["region"]
            width, held = shape, {}
            if own:  # a drafter's own decode shape, past its step's
                width = decode(shape.seqs)
                held = leaves_at(store, config, width, arch)
        except ArtifactRefusal as refused:
            return refused.what
        return [
            (op["op"], *_family(at, None, op["op"], sig, dependent), op, at)
            for sig, (op, _) in leaves(region).items()
            for at in [width if sig in held else shape]
        ]

    for line, reads in steps:
        whys = {}
        for part, shape, output, draft in reads:
            by_part.setdefault(part, Counter())["steps"] += 1
            decoding, before = shape.phase == "decode", len(whys)
            mean, top = band(line["rows"], decoding)
            own = part == DRAFT and decoding and shape != reads[0][1]
            if isinstance(found := families(shape, output, draft, own), str):
                whys[f"{part}op_graph"], found = found, []
            for name, fam, tokens, op, at in found:
                bound = (*at[:4], mean, top)
                if decoding:  # its max, then its mean over T, as `OpGraphBackend` asks
                    total = sum(r[1] for r in line["rows"])
                    bound = (*at[:2], at.tokens, at.gathered, top, total / at.tokens)
                moves = dependent.get(name, True)
                cut = _cut(fam)
                if cut and cut[0] in cuts:
                    at, (lo, hi) = cuts[cut[0]], _cached(line)
                    why = (
                        "a cached row, measured on fresh rows only"
                        if hi and at.keys() == {0}
                        else _bracket_cached(at, tokens, lo, hi)
                    )
                elif fam not in axes:
                    why = "no measured point"
                elif decoding and not _exchanged(fam):
                    why = _bracket(*axes[fam], *_rows(line, at.tokens)) if moves else ""
                else:
                    why = _bracket(*axes[fam], tokens, tokens)
                if not why and isinstance(
                    got := listed.price((op, bound, None)), Refusal
                ):
                    why = re.sub(r"\d+", "#", got.reason.split(": ")[1])
                if why:
                    whys[part + name] = why
            by_part[part]["covered"] += len(whys) == before
        report[reads[0][1].phase].update(steps=1, covered=not whys)
        for name, why in whys.items():
            out[name][why] += 1
    total = sum(report.values(), Counter())
    return {
        "steps": total["steps"],
        "covered": total["covered"],
        "share": total["covered"] / max(total["steps"], 1),
        "by phase": {phase: dict(c) for phase, c in report.items()},
        "by part": {part.strip() or "target": dict(c) for part, c in by_part.items()},
        "left out, by leaf kind": {k: dict(v) for k, v in sorted(out.items())},
    }


def parts(names, config) -> list:
    """Each step of the record directories `names`, as `--plan` reads them, and
    the graphs a run prices it from, `(part, shape, output, draft)` each, its
    part as the run prefixes its terms: its target's; under a speculative
    drafter its drafter's at the shape `trace ops --population` traces it, of
    a step that samples or not; and a block drafter's draft-KV pass's on a
    step that samples, at the step's shape."""
    from atom.compass.capture import population_steps

    spec, steps = getattr(config, "speculative_config", None), []
    for shape, line in population_steps(names, config.enable_tbo, drafters=True):
        if steps and line is steps[-1][0]:  # the drafter's own shape, past its step's
            reads = steps[-1][1]
            reads[1:2] = [(DRAFT, shape, *reads[1][2:])] if spec else []
            continue
        samples = shape.phase == "decode" or bool(line.get("produces_output"))
        reads = [("", shape, True, False)]
        if spec:
            reads.append((DRAFT, shape, samples, True))
            reads += [(DRAFT_KV, shape, True, KV)] * (samples and spec.use_dspark())
        steps.append((line, reads))
    return steps


def _rows(line, rung: int) -> tuple[int, int]:
    """A step's lowest and highest row context: at least 1, the least over
    `k`, the powers of two up to the rung, of `rowmod(k)` per row of the
    `rung` a class `i mod k` holds (`k = 1`: the mean over the rung's rows,
    padding included); and the max."""
    from atom.compass.backends.step_linear import rowmod

    contexts = [row[1] for row in line["rows"]]
    ks = [1 << i for i in range(rung.bit_length())]
    lo = min(rowmod(contexts, k) // -(-rung // k) for k in ks)
    return max(lo, 1), max(contexts)


def _mean(line, rung: int) -> int:
    """A decode step's rows' mean context over its rung, a pad row holding
    none, in `PERMILLE` of their max."""
    contexts = [row[1] for row in line["rows"]]
    return round(PERMILLE * sum(contexts) / (rung * max(contexts)))


def _means(rung: int) -> list:
    """The means, in `PERMILLE` of the max, a rung's ragged points take: the
    least its rows hold, a token past the first each, and each `GRID`th of
    the max past it; none at one row."""
    least = round(PERMILLE / rung)
    grid = (round(PERMILLE * k / GRID) for k in range(1, GRID))
    return [least, *(m for m in grid if m > least)] if rung > 1 else []


def _point(record) -> tuple:
    """A point record's shape, context and leaf, hashable."""
    context = record["context"]
    context = tuple(context) if isinstance(context, list) else context
    return record["shape"], context, record["leaf"]


def _cached(line) -> tuple[int, int]:
    """A step's rows' mean and max cached tokens."""
    cached = [row[1] - row[0] for row in line["rows"]]
    return sum(cached) // len(cached), max(cached)


def _ladder(values) -> list:
    """`values` in order, each kept only `LADDER_STEP` past the last kept,
    and the largest."""
    kept = []
    for v in sorted(set(values)):
        if not kept or v >= kept[-1] * LADDER_STEP:
            kept.append(v)
    return kept + [v] if kept[-1] != v else kept


def _shape(text: str):
    """The `Shape` whose `str` is `text`, each count read by its name."""
    from atom.compass.capture.forward import decode, prefill
    from atom.compass.fit.ops import shape_counts

    n = shape_counts(text)
    if text.startswith("prefill"):
        return prefill(n["T"], n["B"], n.get("G"), n.get("W"))
    return decode(n["B"], n.get("G"), n.get("T"))._replace(width=n.get("W"))


def _abstract(value, n: int):
    """`value` with every int equal to `n` replaced by ``"T"``."""
    # ponytail: an unrelated int equal to n (a stride of 1 at T = 1) is
    # abstracted too, so that shape's leaf joins another family and its
    # points bracket nothing; symbolic slots in the contract would end it.
    if isinstance(value, dict):
        return {k: _abstract(v, n) for k, v in value.items()}
    if isinstance(value, list):
        return [_abstract(v, n) for v in value]
    return "T" if type(value) is int and value == n else value


#: Leaf kinds that exchange rows between ranks (`exchange`).
COLLECTIVES = (
    "atom.model_ops.moe:",
    "mori.ops.dispatch_combine:",
    "aiter.all_reduce_.",
    "_c10d_functional.all_",
    "_c10d_functional.broadcast.",
    "_c10d_functional.reduce_scatter",
)
#: The DP all-gather, whose `x` holds the calling rank's rows (`_own_rows`).
GATHERV = "atom.model_ops.moe:all_gatherv"
#: Lifted collectives every ATOM caller passes aiter's DP group.
DP_GROUP = (GATHERV, "atom.model_ops.moe:reduce_scatterv")


def exchange(op: dict) -> list | None:
    """A collective leaf's family, decode and prefill alike, and its message
    bytes, the axis it is priced over: the family is its kind, each tensor
    argument's dtype and row width, its output rows' widths and its arguments
    neither tensors nor numbers; the bytes the largest of its tensors and its
    outputs, at its first tensor's dtype. None for any other leaf."""
    import math

    if not op["op"].startswith(COLLECTIVES):
        return None
    tensors = [s for _, s in _specs(op["args"])]
    size = _dtype(tensors[0]["dtype"]).itemsize
    nbytes = max(
        [math.prod(t["shape"]) * _dtype(t["dtype"]).itemsize for t in tensors]
        + [math.prod(o) * size for o in op["out"]]
    )
    fam = {
        "op": op["op"],
        "tensors": [[t["dtype"], t["shape"][1:], t["stride"][1:]] for t in tensors],
        "out": [o[1:] for o in op["out"]],
        "flags": {s: v for s, v in op["args"].items() if isinstance(v, (bool, str))},
    }
    return [json.dumps(["exchange", fam], sort_keys=True), nbytes]


def _exchanged(fam: str) -> bool:
    return json.loads(fam)[0] == "exchange"


def _family(shape, context, op: str, sig: str, dependent: dict):
    """The family a leaf's point belongs to, and where on its axis: a decode
    leaf at its width, and its rows' mean when they are ragged, over context;
    a prefill leaf over its tokens, at its rows unless its kind's prefill
    points show it does not move with them, and at its rows and cached tokens
    when its kind depends on context, as its prefill points read it, else its
    decode points, else by default; a collective over its message bytes
    (`exchange`)."""
    if op.startswith(COLLECTIVES):
        return tuple(exchange(json.loads(sig)))
    if shape.phase == "decode":
        context, mean = _split(context)
        return json.dumps([str(shape), sig] + ([mean] if mean else [])), context
    leaf = _abstract(json.loads(sig), shape.tokens)
    if not dependent.get(("prefill", op), dependent.get(op, True)):
        rows = shape.seqs if dependent.get(("rows", op), True) else None
        return json.dumps(["prefill", rows, leaf], sort_keys=True), shape.tokens
    fam = ["prefill", shape.seqs, leaf, context or 0]
    return json.dumps(fam, sort_keys=True), shape.tokens


def _cut(fam: str):
    """A prefill family of a kind that depends on context as the family over
    every cached-token count, and its count; None for any other."""
    *base, cached = json.loads(fam)
    return (json.dumps(base, sort_keys=True), cached) if len(base) == 3 else None


def _dependent(records) -> dict:
    """Whether each leaf kind timed at two contexts of one shape depends on
    context: its kernels differ there, or its cold times by more than
    `CONTEXT_MOVES` of the largest and `CONTEXT_FLOOR_NS`. A decode kind, read
    at two row contexts, is keyed by its name; a prefill kind, read fresh and
    over cached tokens, by ``("prefill", name)``. Keyed ``("rows", name)``,
    whether a prefill kind timed at two row counts of one token count and
    cached tokens moves with its rows."""
    by = defaultdict(list)
    for r in records:
        if "refused" in r:
            continue
        if r["shape"].startswith("decode"):
            by[r["op"], r["shape"], r["leaf"]].append(r)
        else:
            by[("prefill", r["op"]), r["shape"], r["leaf"]].append(r)
            at = _shape(r["shape"]).tokens, r["context"]
            by[("rows", r["op"]), at, r["leaf"]].append(r)
    kinds = {}
    for (kind, *_), points in by.items():
        if len(points) > 1:
            cold = [p["cold_ns"] for p in points]
            kernels = {tuple(p["kernels"]) for p in points}
            spread = max(cold) - min(cold)
            moved = spread > max(CONTEXT_MOVES * max(cold), CONTEXT_FLOOR_NS)
            kinds[kind] = kinds.get(kind, False) or moved or len(kernels) > 1
    return kinds


def _families(records, dependent) -> dict:
    """Each family's measured points, `(at, kernels, record)` in `at` order."""
    found = defaultdict(list)
    for r in records:
        if "refused" not in r:
            shape = _shape(r["shape"])
            fam, at = _family(shape, r["context"], r["op"], r["leaf"], dependent)
            found[fam].append((at, tuple(r["kernels"]), r))
    return {fam: sorted(points, key=lambda p: p[0]) for fam, points in found.items()}


def _bracket(ats, kernels, lo, hi) -> str:
    """'' if `lo` to `hi` lies on or between two points, `ats` in order, that
    launched the same `kernels` as every point between them; else why not."""
    band = _band(ats, lo, hi)
    if band is None:
        return "outside the measured range"
    return "between two kernels" if len(set(kernels[band])) > 1 else ""


def _bracket_cached(cuts: dict, tokens, lo, hi) -> str:
    """`_bracket` over two axes: '' if `lo` to `hi` cached tokens lie on or
    between two of `cuts`' cached-token counts, and `tokens` on or between two
    points of each count from the one to the other, every point of one kernel;
    else why not."""
    cached = sorted(cuts)
    if (band := _band(cached, lo, hi)) is None:
        return "outside the measured range"
    kernels = set()
    for ats, ks in (cuts[c] for c in cached[band]):
        if (inner := _band(ats, tokens, tokens)) is None:
            return "outside the measured range"
        kernels.update(ks[inner])
    return "between two kernels" if len(kernels) > 1 else ""


def _band(ats, lo, hi) -> slice | None:
    """The points of `ats`, in order, from the last at or below `lo` to the
    first at or above `hi`; None when either is missing."""
    i, j = bisect.bisect_right(ats, lo) - 1, bisect.bisect_left(ats, hi)
    return None if i < 0 or j == len(ats) else slice(min(i, j), max(i, j) + 1)


def _densify(families, tried: set) -> dict:
    """The point halfway between each two neighbours of a family, or of a
    prefill leaf's token count over cached tokens (`_cut`), more than one
    apart, unless `tried`, that launched different kernels or, in a decode
    family, lie either side of a point `_bent` off the line through theirs or,
    on rows alike, are `_apart` where the leaf's kind steps: some decode family
    of the kind holds two adjacent contexts of one kernel that are `_apart`;
    between two
    points of a collective's family, the step of the rows halfway, a decode
    when the farther is one, else a prefill of one sequence, and none where
    the farther's step holds fewer rows (a drafter's collective over its
    anchors or its block, whose bytes do not follow its step's). A collective's
    family is `_bent` too, each message size at the mean of its readings, as
    `fit ops` merges them."""
    from atom.compass.capture.forward import decode, prefill

    steps = {
        p[2]["op"]
        for points in families.values()
        for p, q in itertools.pairwise(points)
        if p[2]["shape"].startswith("decode") and q[0] - p[0] == 1
        if p[1] == q[1] and _apart(p, q)
    }
    across = defaultdict(list)
    for fam, points in families.items():
        if cut := _cut(fam):
            for tokens, kernels, record in points:
                across[cut[0], tokens].append((cut[1], kernels, record))
    want = defaultdict(set)
    for fam, points in [*families.items(), *across.items()]:
        points = sorted(points, key=lambda p: p[0])
        decoding = points[0][2]["shape"].startswith("decode")
        # a ragged decode line is halved where it bends, not where its kind steps
        ragged = decoding and json.loads(fam)[2:]
        if isinstance(fam, str) and _exchanged(fam):
            sizes = [[*g] for _, g in itertools.groupby(points, key=lambda p: p[0])]
            mean = (statistics.mean(p[2]["cold_ns"] for p in g) for g in sizes)
            at = ((*g[0][:2], {"cold_ns": ns}) for g, ns in zip(sizes, mean))
            bent = _bent([*at]) if len(sizes) > 2 else ()  # no line under three
        else:
            bent = _bent(points, () if ragged else steps) if decoding else ()
        for (a, ka, record), (b, kb, other) in itertools.pairwise(points):
            mid = (a + b) // 2
            if ka == kb and a not in bent or b - a < 2 or (fam, mid) in tried:
                continue
            tried.add((fam, mid))
            shape = _shape(record["shape"])
            if isinstance(fam, tuple):  # over cached tokens at one token count
                new = json.dumps([*json.loads(fam[0]), mid], sort_keys=True)
                want[shape, mid].add(new)
            elif _exchanged(fam):
                far = _shape(other["shape"])
                if far.tokens < shape.tokens:  # no step's rows lie between
                    continue
                rows = (shape.tokens + far.tokens) // 2
                step = decode(rows) if far.phase == "decode" else prefill(rows, 1)
                want[step, other["context"]].add(fam)
            elif shape.phase == "decode":
                want[shape, (mid, *ragged) if ragged else mid].add(fam)
            else:
                want[prefill(mid, shape.seqs), record["context"]].add(fam)
    return want


def _filled(families, tried: set, max_model_len: int) -> dict:
    """The points of each prefill family over cached tokens (`_cut`) at its
    rows that its leaf's families at any rows hold the token count of, between
    its own least and most, at each cached-token count its own points hold
    that its first row has room for under `max_model_len`, unless `tried`.
    `fit ops` reads such a leaf over T, then its rows: a step's rows are
    bracketed only where the token counts either side of it hold them."""
    from atom.compass.capture.forward import prefill
    from atom.compass.fit.ops import skeleton

    held, skeletons, axis = defaultdict(set), {}, defaultdict(set)
    for fam, points in families.items():
        if cut := _cut(fam):
            for tokens, _, record in points:
                held[cut[0]].add((tokens, cut[1]))
                skeletons[cut[0]] = skeleton(record["shape"], record["leaf"])
                axis[skeletons[cut[0]]].add(tokens)
    want = defaultdict(set)
    for base, at in held.items():
        rows, tokens = json.loads(base)[1], [t for t, _ in at]
        for t in sorted(axis[skeletons[base]]):
            for c in sorted({c for _, c in at}):
                fam = json.dumps([*json.loads(base), c], sort_keys=True)
                if (
                    not min(tokens) < t < max(tokens)
                    or (t, c) in at
                    or (fam, t) in tried
                    or c > max_model_len - -(-t // rows)
                ):
                    continue
                tried.add((fam, t))
                want[prefill(t, rows), c or None].add(fam)
    return want


def _bent(points, steps=()) -> set:
    """The lower ends of the gaps either side of each middle point of three
    neighbours that launched one kernel and whose cold time the line through
    the outer two's prices more than `BENDS` off, over that cold time; and,
    for a leaf of a kind in `steps`, of each gap whose ends are `_apart`."""
    ends = set()
    if steps and points[0][2]["op"] in steps:
        ends = {p[0] for p, q in itertools.pairwise(points) if _apart(p, q)}
    for p, q, r in zip(points, points[1:], points[2:]):
        y = [point[2]["cold_ns"] for point in (p, q, r)]
        line = y[0] + (y[2] - y[0]) * (q[0] - p[0]) / (r[0] - p[0])
        if p[1] == q[1] == r[1] and abs(line - y[1]) > BENDS * y[1]:
            ends |= {p[0], q[0]}
    return ends


def _apart(p, q) -> bool:
    """Whether two points' cold times differ by more than `BENDS` of the
    lower: the line across them may then price a point between them more
    than `BENDS` off its cold time."""
    lo, hi = sorted(point[2]["cold_ns"] for point in (p, q))
    return hi - lo > BENDS * lo


def _load(
    store,
    config,
    root,
    shapes,
    graphs: dict,
    arch=None,
    atom_args=(),
    records=(),
    own=None,
) -> None:
    """Each of `shapes`' leaves (`leaves_at`, a drafter's own at a shape `own`
    maps to its outputs) into `graphs`, its op_graph captured first when the
    store has none (`_capture`, naming the record directories `records` the
    shapes are of)."""
    from atom.compass.artifacts import ArtifactRefusal

    own, missing = own or {}, []
    for shape in sorted(shapes - graphs.keys(), key=str):  # G and W: None or a count
        try:
            graphs[shape] = leaves_at(store, config, shape, arch, own.get(shape, ()))
        except ArtifactRefusal:
            missing.append(shape)
    if missing:
        _capture(config, missing, root, arch, atom_args, records)
    for shape in missing:
        graphs[shape] = leaves_at(store, config, shape, arch, own.get(shape, ()))


def _capture(config, shapes, root, arch=None, atom_args=(), records=()) -> None:
    """`trace ops` of `shapes` under ATOM's arguments `atom_args` into the store
    at `root`, in a process of its own that sees no GPU, on a declared `arch`,
    or with none on this run's first GPU as it reads (`trace ops --gpu`): its
    stepping and compute units pick kernels as they do on the card. A shape
    its `--prefill` and `--width` cannot name, a DP group's gathered rows or a
    padded width, refuses before any is traced, naming the `trace ops
    --population` of the record directories `records` that traces it. Under
    `--enable-tbo` a prefill is traced as one forward (`--whole`), where ATOM
    would split a step of its tokens: a point times a leaf at its own shape."""
    from atom.compass import capture
    from atom.compass.artifacts import ArtifactRefusal, Rule
    from atom.compass.capture.forward import decode, device_arch, prefill
    from atom.model_engine.arg_utils import EngineArgs

    engine = EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args(atom_args)
    argv = ["--model", config.model, "--tp", str(engine.tensor_parallel_size)]
    argv += ["--arch", arch or device_arch()]
    if not arch:  # the card read by its HIP ordinal, which ROCm indexes alike
        argv += ["--gpu", os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]]
    for s in shapes:
        if s != (prefill(s.tokens, s.seqs) if s.phase == "prefill" else decode(s.seqs)):
            population = [a for r in records for a in ("--population", r)]
            call = ["compass", "trace", "ops", *argv, *population, "--store", str(root)]
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"no op_graph holds {s}, a shape trace ops takes only from the "
                "steps of a record",
                f"trace it with {shlex.join([*call, '--', *atom_args])}",
            )
    for s in shapes:
        argv += ["--prefill", str(s.tokens)] if s.phase == "prefill" else ["--width"]
        argv.append(str(s.seqs))
    if engine.enable_tbo:
        argv.append("--whole")
    # The capture sets HIP_VISIBLE_DEVICES itself, and HIP then holds memory on
    # the GPU it names; ROCm's own mask, below HIP's, hides every GPU.
    with patch.dict(os.environ, ROCR_VISIBLE_DEVICES=""):
        _child(capture.main, [*argv, "--store", str(root), "--", *atom_args])


def _journal(path, records: list[dict]) -> list[dict]:
    """`records`, each first appended to the file at `path` when one is named."""
    if path:
        with open(path, "a") as lines:
            lines.writelines(json.dumps(r) + "\n" for r in records)
    return records


def _measured(config, todo, llc: int, gpus: int = 1) -> list[dict]:
    """`measure` of `todo` on `gpus` GPUs (`ranks.measure`), a child of its own
    on each. A launch whose child dies keeps the points every rank finished;
    the first it left, the one it died measuring, refuses, and another launch
    measures the rest: the pool ends every rank once one dies. Two launches in
    a row that die having finished no point raise: a child itself is failing,
    not a point."""
    from atom.compass.bench import ranks

    got, empty = [], False
    with tempfile.TemporaryDirectory() as tmp:
        journal = Path(tmp, "points.jsonl")
        paths = (
            [journal] if gpus == 1 else [Path(f"{journal}.{r}") for r in range(gpus)]
        )
        while todo:
            for path in paths:
                path.write_text("")
            try:
                with patch.dict(os.environ, {JOURNAL: str(journal)}):
                    return got + ranks.measure(config, todo, llc, gpus)
            except BrokenProcessPool:
                new = ranks.merge(
                    [[json.loads(x) for x in p.read_text().splitlines()] for p in paths]
                )
                if not new and empty:
                    raise
                got, empty = got + new, not new
            done = {_point(r) for r in got}
            todo = [
                (s, c, {k: v for k, v in found.items() if (str(s), c, k) not in done})
                for s, c, found in todo
            ]
            todo = [point for point in todo if point[2]]
            if todo:  # the first point it left is the one it died measuring
                shape, context, found = todo[0]
                sig = next(iter(found))
                op = found.pop(sig)[0]["op"]
                print(f"{shape}: {op}: the measuring process died", flush=True)
                got.append(
                    {"shape": str(shape), "context": context, "op": op}
                    | {"leaf": sig, "refused": "the measuring process died"}
                )
                todo = todo if found else todo[1:]
    return got


def _child(fn, *args):
    """`fn(*args)` in a spawned process of its own: the device is free again
    at its end, and a capture's process-wide patches end with it."""
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn) as child:
        return child.submit(fn, *args).result()
