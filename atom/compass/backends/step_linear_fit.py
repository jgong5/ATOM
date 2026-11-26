# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law fitted from real step records.

    python -m atom.compass.backends.step_linear_fit RECORD... --out LAW --platform JSON [--form FORM] [--moe-routing E,K] [--engine FILE]
    python -m atom.compass.backends.step_linear_fit offload LOG... --out ENTRY [--master-flags FLAGS --capacity-bytes BYTES]
    python -m atom.compass.backends.step_linear_fit figures RECORD... --law LAW [--source NAME] [--form FORM] [--moe-routing E,K] [--engine FILE]
    python -m atom.compass.backends.step_linear_fit pack PATH...

The second form fits a run file's ``lmcache_offload`` entry, beside the law of
an arm whose records ran the connector, on those runs' server logs
(`fit_offload`), or a shared Mooncake Store's ``mooncake_store`` entry on its
workers' logs. The third writes the held-out figures (below) of the records
into an existing law fitted on them (`figures_main`). The fourth stores law
files xz-compressed, as the checked-in ones are (`pack_main`). A law is read
and written by `json_file`, xz-compressed when its name ends in ``.xz``.

Each RECORD is a real run's step record (`steps/dp0.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`), labelled by `step_labels`, whose docstring
states the labels; the command writes each record's per-step table beside it.
A record refuses, before any table is written, when the run.json beside the
`steps` directory its path resolves into names JIT compiles its server ran
(`step_labels.refusal`).
A step whose device label is cut is counted under its reason and not fitted.
Prefill labels come from trace runs alone (`step_labels`); given none, the
prefill vector is left out and named under `unfitted`. A trace run gives no
decode or host label, so its record fits only prefill.
`fit` and `fit_host` take the labelled samples, grouped into runs by their
provenance's `run`, and each returns its laws and a report. `fit`'s report
holds the runs it was fitted on, each vector's fit figures and hull, and, once
`mark_held_out` has run, the held-out errors per run; its `seed` fixes the
`rowmod` shuffles, so the same samples and arguments give the same bytes.

The law states the records' `engine_config`, read from the `.engine.json`
beside each (`engine_config`) or from `--engine FILE` in their place, a record's
over the compilation level its run.json names (`recorded_level`), and a run
under another engine config refuses it, records whose ladders are prefixes of
the longest read at it (`step_grid_plan.pooled`). Records at a decode
context-parallel width state it there (`run.step_grid.keyed`), so their law is
keyed by it and a run at another width refuses it; records stating a prefill
context-parallel width refuse. At its `data_parallel_size` N above 1
(EP off) the records come in runs of
N, each run's `dp0.jsonl` to `dp<N-1>.jsonl`. A step is its ranks' lines
joined on step index, labelled with the smallest rank label (the others wait
for the last rank to enter), and a prefill row adds the line's `moe_rows` and,
when some prefill step of the samples the group split for two-batch overlap,
`serial_moe_rows` (`StepLinearLaw`), and when asked (`compass fit steps
--cold-peer-rows`), `cold_peer_rows`, read off the view's `prefill_duty`, the
vector then stating its `duty_tau`. A
rank that ran a decode step as a dummy forward adds no row; one that ran a
prefill step so adds its dummy's row (`step_linear.dummy_view`), labelled as
the step, which a run prices as a rank of the group the step waits for, and
when some such dummy ran eagerly, the prefill vector adds `EAGER_DUMMY`
(`StepLinearLaw`); a draft law takes no dummy row. A step that some rank
lacks otherwise or cut, or whose ranks' rows fall to different vectors, is
counted under `ranks` and not fitted. The law reports each step's rank label
spread, `max / min - 1`, as its p50 and p90 under `spread`. The group's step
is the max over its ranks, so each vector is fitted by max-affine regression
(`_fit_form`): from the fit on every rank's row and from each feature's axis,
the lowest after a first round going on, each pass refits on the row
each step's price picks, a step line-searched until it lowers the squared
relative error, until no step changes rank, the step's ranks tie or the passes
reach `FIT_PASSES`, reweighted until no price moves; MAD drops rerun it, short of that limit. The vector's solve, warm steps and hull read each step's row at the
rank it prices the step as (`priced_rank`); so does each form a decode rung
chooses among, MoE column and `rowmod` (below), and each held-out fold
that chooses, which prices a step as the max over its ranks. Each
vector reports, under `fit.all_rows`, the centre and scaled MAD of the fit on
every rank's row, each step priced as its max over ranks, and under
`fit.passes` the start, passes, how they stopped, `settled`, `tie` or `limit`,
and the squared error `F`; a fit at the limit keeps the vector it reached, the
lowest `F` its passes found. A held-out
step is priced and banded as its costliest rank.

A calibration ladder's samples (`Sample.ladder`, the rungs its client run
calibrates) join the decode vectors of those rungs only (`ladder_cut`): a
ladder runs a decode instance under a load no deployment serves, so at a
rung the other client runs fill, its steps would set the rung's price at the
ladder's. On #3720's cell B the long-context ladder (`ldl` and `ldl2`: 252
subagents on 18 shared prefixes, admitted at once and decoded down with no
arrival) ran D's rung-16 steps 5-6% under the same deployment's c=64 point at
the same summed context and MoE tokens; pooled with it, rung 16's vector
priced the c=128 record's steps 7% low, and its TPOT 6.5% low. A ladder's
prefills, final and middle chunks, join no vector: a ladder admits its
requests at once on a few shared prefixes, a prefill load no deployment
serves. Cell B's `ldl` / `ldl2` prefills (162 steps of 8192+ tokens, 113 s)
ran 11-14% under the law fitted on its live points; pooled into the one
prefill vector they priced r256's c=256 prefill steps 4.1% low (1.2% without
them), and its simulated TTFT p50 3 points lower.

Records with a host key (`dp0.key.json`, from the real runner) also give the
host laws as a run file's `host` entry: `key`, and `laws[kind][segment]`,
seconds on `(1, tokens, seqs)`, fitted on the samples' host labels. They take
the step linear law's solver and hulls, and drop no sample (`_host_law`); `seqs` in a fixed
proportion to `tokens` on every sample (equal, or a decode under MTP, where
each row computes its drafts too) gets zero. Two keys refuse, ladders pooled
first. The law, as
`platform`, and the host key state the platform fields `--platform` gives for
their kinds, `step_linear_law` and `host_laws`, which a run's spec must match,
and the host key's `engine` is the law's `engine_config`. A segment outside `HOST_SEGMENTS` is
fitted for a kind whose samples carry it: `W_inputs` and `L_mtp`, on an MTP
record, and `W_recv`, on a later pipeline stage's. A forward whose `W_launch` its kind's
solve drops above the law paid a first-use JIT compile (`_jit`): it and the
forward after it, whose period takes up the stall, are left out of every host
law; the report lists the JIT forwards per run under `jit_steps`. A `W_launch`
label leaves out the time its launch spent in HIP's blocking calls, which the
report states per run and kind under `blocked` (`_blocked`); it names under
`unstamped` the runs whose records state none, so their `W_launch` labels are
wall times.
A DP rank's dummy forward at a step its group prefills gives no sample, but its
`W_pre`, `W_inputs`, `W_launch` and `W_post` (`step_labels.Sample.dummy_host`)
fit the host laws' `dummy` kind (`_fit_dummy`), every launch feature a column,
which a run prices such a dummy by, its other segments by `decode`'s
(`HostDeviceRecurrence`).
A forward's `E` that admits requests (`step_labels.admissions`) is left out of
its kind's law: the law then prices the engine loop alone, and `admit`, seconds
on `step_labels.admit_x`, prices what the forwards of every kind that admit spend
beyond it, fitted on their `E` with their kind's law as a fixed part. Records
on too few such forwards give `admit` null, so a run refuses a forward that
admits a request rather than charge it nothing, and the report says why under
`unfitted`.

Under a launch provider that declares features (`launch_features`), a step it
predicts otherwise than its line observed refuses, naming it; each host law
adopts features as columns on their held-out error per launch path
(`_launch_columns`), their coefficients under the entry's
`columns[kind][segment]`, and `_jit` solves `W_launch` with every one. A fit of
one run holds out `RUN_BLOCKS` contiguous spans of its steps in place of runs.

The same solver fits the host entry's ``tokenizer`` term on a recorded
server's tokenizer calls (`fit_tokenizer`), and its ``http_intake`` term on its
HTTP intake stamps (`fit_intake`): ``per_byte_s`` and ``idle_s`` on the lone
requests, ``fixed_s`` from the bursts or else ``idle_s``, and ``bytes`` over every
request's body. On a prefill-decode deployment's two sides' stamps it fits the
prefill host entry's ``relay`` term (`fit_relay`): what carrying a prefill's
response through the router to decode's app call costs per request and per
byte. On a streaming server's stamps it fits the ``http_stream`` term
(`fit_stream`): its API loop's CPU per SSE event its streamed responses send,
from the loop thread's CPU clock over windows of the record.

Each equation is divided by its price, from its label until no price moves
(`_fit_form`), and the coefficients are bound to be
non-negative. Steps whose relative residual, price less label over price, is more than `MAD_K` scaled MADs
from the median are dropped and the rest refitted, until the kept set settles.
The dropped steps that lead a record's steps of one vector, each slower than
the law, are its warm steps (shape compiles and first touches), and so is a
dropped step anywhere that ran more than `WARM_FACTOR` times its price, such as
the first short chunk on a deep history, or more than `PAUSE_S` seconds past
it, a whole-group pause (`validate.stalls`) on a step of any price: counted
with their seconds apart from
the dropped ones, never charged, and listed per run under `warm_steps`, in the
order the law's `runs` names the runs. A feature with no variance gets coefficient
zero and is named. Each vector reports its samples,
warm and dropped counts, the centre and scaled MAD of its relative residuals,
the condition number of its column-scaled design and
the convex hull of its fitted steps, which `StepLinearLaw` reads: each step's
row at the rank the vector, with any column it adopted, prices highest
(`priced_rank`), the rank a run prices the step as. A vector with
no more fitted steps than coefficients, or whose steps span no hull, is left
out and named under `unfitted`; `StepLinearBackend` refuses its steps.

On an MTP record a step also carries a draft label (`step_labels`). `fit` fits
the steps that carry one on their draft labels the same way, and writes that law
under the law's `draft`; it holds no `held_out` unless its records time a
EAGLE3 drafter's passes (below) or run at `data_parallel_size` above 1. A step
with a propose label (`step_labels`) is fitted there on its draft label less
that label, the draft-KV pass a middle chunk runs too, and left out when
nothing remains; the propose labels fit the draft law's `propose` vector, seconds on `[1, seqs]`,
whatever lag the device ran at.
A decode's draft-KV label fits the draft law's `draft_kv` vector alike. At
`data_parallel_size` above 1 `propose`, whose drafter forward waits on its
group, is fitted as the law's vectors are: a step is its ranks' labelled rows,
labelled with the smallest, and priced as their max (`_pass_steps`); a step
some rank neither labels nor ran as a dummy forward is left out, counted under
the vector's `fit.ranks`. `draft_kv` is fitted there on each rank's own label,
as at width 1: each rank writes its own draft context and waits on no peer
(`GROUPED`). There `mark_held_out` gives the draft law's vectors `held_out` as
it gives the law's, on the steps each is fitted on, `propose` priced as the max
over each step's ranks and `draft_kv` at each rank's own (`_mark_dp_passes`). An
output prefill's fits the draft law's prefill vector in place of its draft
label, which, all after `forward_done_event`, fits the `propose` vector, and
sets the draft law's `output_prefill_kv`. The target's vectors fit the device
label less the draft-KV label.
On an MTP record both prefill vectors, the target's and the draft's, add
`seqs` and `out` (`StepLinearLaw`).
Records whose decodes time an EAGLE3 drafter's passes (`passes_s`) fit the
draft law's `forward` vector instead, each pass a sample at its own features
(`serial_passes`), with `passes` the count every such step ran, and its
`propose` vector on what each decode's draft label holds past its passes; the
draft law's prefill vector then fits the middle chunks alone, whose draft is
the draft-KV pass, in the `base` form, and it has no decode or `draft_kv`
vector. `mark_held_out`
gives each of that draft law's vectors `held_out` as it gives the law's, on
the steps each is fitted on.

Records whose engine files state the prefill token-graph buckets their runner
recorded (`PREFILL_BUCKETS`) fit a prefill vector that adds `padding` and
states them (`StepLinearLaw`); every record states the same buckets, or none
does. A record whose engine file predates the field states them in its host
key only where its runner wrote graph sets there (`stated_buckets`); `--engine`
states them for one that states none. Records stating them nowhere fit no
`padding`, named under `unfitted`.
The draft law adds none: #2976 measured the target's token graph alone.

Forms (`FORMS`): `base`, the features `StepLinearLaw` always prices; `history`,
prefill plus `Sum (N_KV - N_Q)`, the KV a short chunk reads on a deep history;
`shared`, decode with the context and padding slopes shared by every rung and
an intercept per rung; `cached_seqs`, prefill plus its rows with cached
context, each a cost whatever its history's length.

With `moe_routing`, the model's routed experts and top-k (`--moe-routing E,K`),
the law states `moe_routing` and, outside `shared`, a decode rung may adopt one
MoE column (`StepLinearLaw`), `moe_tokens` or `moe_experts`, by the rule
`_moe_column` states; the vector records under `fit.moe_column` the figures the
rule read.
An MTP record's draft law adopts none at `data_parallel_size` 1; above it,
given the drafter's own routed experts and top-k (`compass fit steps --model`
reads a block drafter's), its decode rungs adopt one by the same rule, as a
drafter's routed experts run on its group's rows in the arm's layout, as the
target's do.

Opt-in columns and rules (`env_options`, each off unless asked for, so a fit
without them writes the law it wrote before them), each column counted by the
`step_linear` function the backend prices it with: `row_padding` adds the decode
column of that name to every rung (`ROW_PADDING_ENV`); `gather_rows` and
`moe_imbalance` add the prefill columns of those names above `dp` 1, the vector
stating `ranks` and `moe_imbalance`'s `min_spread`, `gather_rows` ``pad`` the
column `gather_pad_rows` in its place, the gather's pad rows alone, and
`uneven_rows` the prefill column of that name at its `min_spread`
(`GATHER_ROWS_ENV` at ``1`` or ``pad``,
`MOE_IMBALANCE_ENV`, `UNEVEN_ROWS_ENV`, `MOE_MIN_SPREAD_ENV`); `graph_tokens` and
`graph_query_cached` add those prefill columns (`GRAPH_COLUMNS`) after
`padding`, at any `dp`, to records stating their prefill buckets
(`GRAPH_TOKENS_ENV`, `GRAPH_QUERY_CACHED_ENV`); `short_hinge` at a knee K adds
the prefill column of that name, max(0, K - Sum N_Q), after the form's columns,
the vector stating K as `short_knee` (`SHORT_HINGE_ENV`); `moe_select` ``insample`` makes each
target decode rung adopt the MoE column its fit on all its steps prices nearest
(`_moe_column`, `MOE_SELECT_ENV`), and ``fallback`` only each rung its records
cannot hold out, which adopts none under the default rule; `moe_both` adds to
the forms each target decode rung picks among the one carrying both
`MOE_COLUMNS`, `MOE_BOTH` (`_moe_column`, `MOE_BOTH_ENV`); `moe_folds`
``covered`` makes a decode rung of several records, the draft law's too, hold
out each record's group compositions and grade each form only on the held-out
steps inside the hull of the others' steps, adopting by in-sample MAD where
none is (`_moe_column`, `MOE_FOLDS_ENV`); `peer_tokens` adds that decode column, after
the MoE column, to the rungs it names or, as ``saturated``, to each rung whose
adopted `moe_experts` column its median step holds at `SATURATED` of the
experts (`_peer_rung`, `PEER_TOKENS_ENV`); `draft_peer_tokens` adds it to the
draft law's decode rungs above `dp` 1, every one as ``all``
(`DRAFT_PEER_TOKENS_ENV`): a draft's MoE runs on its group's rows too. The draft
law takes none of the others but `moe_folds`. The `shared` form, whose one decode solve shares
the base columns over rungs, refuses `row_padding` and `peer_tokens`. A vector carrying one of these columns states,
under `fit.vif`, each one's variance inflation factor over the rows it was
fitted on (`_vif`), how far the other columns already hold it.

Outside `shared`, a decode rung may adopt a `rowmod(k)` column (`StepLinearLaw`),
read from its rows in scheduled order, by the rule `_rowmod` states; the
vector then names `k` and records, under `fit.rowmod`, the MAD without the
column, the mean MAD with each step's rows shuffled, and per held-out record
the p50 error without and with it. `rowmod_vif` (`ROWMOD_VIF_ENV`) also gates
each `k` on the column's variance inflation factor over the steps its fit
keeps, at most `ROWMOD_VIF`; the vector then states under `fit.rowmod_vif`
the factor at each `k` read, adopted or not.

With two or more runs it also prints, per form, the error on each run of a
law fitted on the others, warm steps left out: per kind and rung, or prefill
token band, the p10, p50 and p90 of `predicted / measured - 1`; and the same
per host law. The law's own form's errors are written into it: under each
vector's `held_out`, the convex hull of the rows they were priced by, each
step's costliest rank's, and their percentiles per band, which `StepLinearLaw`
reads to mark a term validated. A cheaper rank's row is not hulled: no error was
measured at it. A vector whose held-out steps span no hull gets none.
The law and its report also state, under `figures`, per held-out run and kind
(`prefill`, `decode`, and `draft` for a drafter's draft), `FIGURES` of that run's
own steps priced by the law fitted on the others, as `validate.cost` grades a
run, a step that law has no vector for entering none; under `prefill` and
`decode`, `refused` counts those steps, their drafts refused with them
(`fold_figures`). The command prints them. A law fitted
on records with a host key states the ATOM digest they ran at, the key's
``atom``, as `records_atom`, which re-keying the host key leaves as it is.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import re
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from atom.compass.backends import json_file, launch_features, step_labels
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.shape import (
    DUTY_TAU,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.step_grid_plan import PREFILL_BUCKETS, keyed_of, pooled
from atom.compass.backends.step_labels import CUTS, HOST_SEGMENTS, host_kind
from atom.compass.backends.step_linear import (
    DECODE,
    EAGER,
    EAGER_DUMMY,
    GATHER_COLUMNS,
    GRAPH_COLUMNS,
    HISTORY,
    KDA,
    KDA_CHUNK,
    MOE_COLUMNS,
    PEER_SHARE,
    PREFILL,
    PROPOSE,
    SPREAD_COLUMNS,
    cached_seqs,
    cold_peer_rows,
    dummy_view,
    eager_dummy,
    gather_column,
    graph_columns,
    hull_axes,
    kda_columns,
    moe_column,
    moe_tokens,
    peer_share,
    peer_tokens,
    row_padding,
    rowmod,
    serial_passes,
    token_padding,
)

FORMS = ("base", "history", "shared", "cached_seqs")
#: The context-parallel widths a record's engine file states above one (`parity`).
CONTEXT_PARALLEL = ("prefill_context_parallel_size", "decode_context_parallel_size")
#: Those a law cannot price at: the decode width is keyed in the law's engine
#: config (`run.step_grid.keyed`), so a run at another width refuses it there.
UNKEYED_CONTEXT_PARALLEL = ("prefill_context_parallel_size",)
MAD_K = 3.5
#: A relative residual this small is never an outlier, however tight the rest.
MAD_FLOOR = 0.01
MAD_PASSES = 20
#: Passes after which a max-affine fit whose error still falls ends, at stop
#: `limit`, keeping the vector it reached: every pass lowers the error, so that
#: is the best it found. Near a tie between ranks each line-searched pass gains
#: little, so a fit that ends at a tie takes far more passes than one that
#: settles.
FIT_PASSES = 10_000
#: A step that ran this many times its price is a one-off first touch, not scatter.
WARM_FACTOR = 5.0
#: A step that ran this many seconds past its price held a whole-group pause, the
#: least `validate.stalls` names (#3778): every rank's device stopped, which no
#: law prices, so it is warm too, on a prefill of a second or more as well,
#: whose pause is less than `WARM_FACTOR` times its price.
PAUSE_S = 3.0
TOKEN_BANDS = (512, 2048, 8192)
#: Steps at its rung a record needs before a `rowmod` choice holds it out.
ROWMOD_CELL_STEPS = 100
#: The fraction of a rung's scaled MAD a `rowmod` column must cut: a smaller cut is
#: within the sampling error of a MAD over a few hundred steps.
ROWMOD_GAIN = 0.1
#: Seeded shuffles of each step's rows whose mean MAD the `rowmod` column must not beat.
ROWMOD_SHUFFLES = 8
#: The MoE-token bands, quantiles of a decode rung's real MoE tokens, a record
#: held out of the rung is graded on, a p50 each, when the rung picks its MoE
#: column (`_moe_column`).
MOE_BANDS = 4
#: The free features up to which `_hull` is qhull's exact hull; past them it
#: is the intersection of each pair's 2-D hull, whose facets the pairs' edges
#: bound (#3720: real_pw_c128's prefill points, 8 features, hulled exactly to
#: 430k facets; pooled with real_pw_c1's, joggled, to 2.9M, a 1.8 GB law
#: whose runs advanced about 3x slower).
HULL_EXACT_FREE = 6
HOST_KINDS = ("decode", "prefill", "middle")
HOST_FEATURES = ("step", "tokens", "seqs")
#: What a lone HTTP request's intake costs: per request, and per body byte.
INTAKE_FEATURES = ("idle", "bytes")
#: What a prefill-decode relay costs a request: per request, and per byte of
#: the request decode takes in.
RELAY_FEATURES = ("fixed", "bytes")
#: Output rows one output thread takes in within this of the row before are
#: one burst's, the rows of the requests one prefill step finished.
RELAY_BURST_S = 0.05
#: What a streaming server's API loop spends in a window: the window's seconds,
#: and per SSE event, per request taken in and per body byte read there.
STREAM_FEATURES = ("seconds", "events", "requests", "bytes")
#: The windows `fit_stream` reads the loop's CPU over, and the share of each
#: the loop thread's stamps must span for it to be read.
STREAM_WINDOW_S, STREAM_SPAN = 10.0, 0.8
#: The intake stamps the loop thread makes (`atom.compass.run._stamp_intake`).
LOOP_KINDS = frozenset(("app", "body", "send", "recv", "wait", "head"))
#: What a lone tokenizer call costs: per call, per token, and for a segment
#: tokenizer call per character its stored-prefix search compared.
TOKENIZER_FEATURES = ("fixed", "tokens", "chars")
#: The `admit` form (opt-in; unset: the solver's fit): ``median``, the median
#: of the admitting forwards' `E` past their kind's law, a fixed cost per
#: admitting forward with no token terms (`_admit_median`).
ADMIT_FORM_ENV = "COMPASS_FIT_ADMIT"
#: The host segments fitted median-centred (opt-in; unset: none), a comma list
#: of ``kind.segment`` (e.g. ``middle.E,middle.W_launch``): the solver's law
#: (which prices a tailed class at its mean) scaled so the median of label over
#: price is 1, for a latency that follows the typical forward, not one that
#: spreads a rare long one over every forward (`_median_centred`).
HOST_MEDIAN_ENV = "COMPASS_FIT_HOST_MEDIAN"
#: The host segments that take a queue column (opt-in; unset: none),
#: ``Q0:kind.segment,...`` (e.g. ``4:middle.E,middle.W_launch``): each listed
#: segment's law gains ``queue_sat<Q0>`` (`step_labels.QUEUE_COLUMN`), what each
#: request in the prefill engine's queue past none adds, up to Q0 of them
#: (`step_labels.queue_depths` from the records' intake stamps; a run's
#: `step_labels.QueueDepth`), a host cost that grows with the queue (P's on the
#: CAL: middle E 1.5 ms mean at a queue of 1, 11.9 ms at 64 and more).
HOST_QUEUE_ENV = "COMPASS_FIT_HOST_QUEUE"


def host_queue_spec() -> tuple[int, frozenset] | None:
    """`HOST_QUEUE_ENV`'s ``(Q0, segments)``; None unset. Raises when malformed."""
    value = os.environ.get(HOST_QUEUE_ENV, "")
    if not value:
        return None
    q0, _, segments = value.partition(":")
    segments = frozenset(x for x in segments.split(",") if x)
    if not q0.isdigit() or int(q0) < 1 or not segments:
        raise ValueError(f"{HOST_QUEUE_ENV}={value} is not Q0:kind.segment,...")
    return int(q0), segments


#: What a forward that admits requests spends once, per prompt token past the
#: prefix-cache hit and per token within it (`step_labels.admit_x`).
ADMIT_FEATURES = ("admits", "uncached", "cached")
#: The figures `validate.cost` grades a kind by that a held-out run's are written
#: under: the steps graded, the error on their totals, the step that contributes
#: most to it and the error holding that step out, and the median per step.
FIGURES = ("steps", "on_totals", "held_out_step", "held_out", "median")
#: Why a law fitted on records stating their prefill buckets nowhere has no `padding`.
NO_BUCKETS = "no record states prefill buckets, so the prefill vector adds no padding"
#: The engine config's field for the compilation level, and ATOM's ``--level``
#: default (`EngineArgs.level`), which `run.step_grid.keyed` leaves unnamed;
#: stated here, as this package imports nothing from the engine.
LEVEL = "level"
DEFAULT_LEVEL = 3
#: The switches of the opt-in columns (`env_options`), each on at ``1``, off at
#: ``0`` or unset, and refused at any other value (`_switch`): decode
#: `row_padding`, prefill `gather_rows`, `moe_imbalance` and `uneven_rows`;
#: `GATHER_ROWS_ENV` also takes ``pad``, `gather_pad_rows` in place of
#: `gather_rows`.
ROW_PADDING_ENV = "COMPASS_FIT_ROW_PADDING"
GATHER_ROWS_ENV = "COMPASS_FIT_GATHER_ROWS"
MOE_IMBALANCE_ENV = "COMPASS_FIT_MOE_IMBALANCE"
UNEVEN_ROWS_ENV = "COMPASS_FIT_UNEVEN_ROWS"
#: Prefill `GRAPH_COLUMNS`: `tokens` and `query_cached` on a step replaying a
#: prefill token graph.
GRAPH_TOKENS_ENV = "COMPASS_FIT_GRAPH_TOKENS"
GRAPH_QUERY_CACHED_ENV = "COMPASS_FIT_GRAPH_QUERY_CACHED"
#: The prefill column `short_hinge` at this knee, in tokens (a positive whole
#: number; unset or 0, none): a MoE prefill's floor below the chunk whose
#: experts' rows fill a tile, its knee set from that roofline, not fitted.
SHORT_HINGE_ENV = "COMPASS_FIT_SHORT_HINGE"
#: `moe_imbalance`'s and `uneven_rows`' `min_spread`, `MOE_MIN_SPREAD` when unset.
MOE_MIN_SPREAD_ENV = "COMPASS_FIT_MOE_MIN_SPREAD"
MOE_MIN_SPREAD = 0.2
#: How a decode rung picks its MoE column (`_moe_column`), one of `MOE_SELECTS`,
#: the first when unset.
MOE_SELECT_ENV = "COMPASS_FIT_MOE_SELECT"
MOE_SELECTS = ("held_out", "insample", "fallback")
#: Whether a target decode rung may also pick the form carrying both
#: `MOE_COLUMNS`, named `MOE_BOTH` (`_moe_column`): ``1`` on.
MOE_BOTH_ENV = "COMPASS_FIT_MOE_BOTH"
MOE_BOTH = "moe_tokens+moe_experts"
#: The largest variance inflation factor `MOE_BOTH`'s two columns may take over
#: a rung's fitted steps for the rung to pick among it: a calibration design's
#: acceptance bound (`calib.deploy`'s default). Past it, at a rung whose steps
#: all reach nearly every expert, `moe_experts` is the intercept again.
MOE_BOTH_VIF = 10.0
#: Whether a decode rung's `rowmod` choice (`_rowmod`) is also gated on the
#: column's variance inflation factor, `ROWMOD_VIF`: ``1`` on.
ROWMOD_VIF_ENV = "COMPASS_FIT_ROWMOD_VIF"
#: The largest variance inflation factor `rowmod` may take over the steps its
#: fit keeps for a rung to adopt it under `ROWMOD_VIF_ENV`, `MOE_BOTH`'s bound.
#: Past it the other columns already hold the column, and the MAD it cuts is
#: theirs to cut (#752 cell A v2, its calibration record and addendum fitted
#: together: rung 32 adopted `rowmod(16)` at VIF 57, `graph_padding` then 49,
#: for a 24% MAD cut that moved neither record's held-out p50 by 0.05 points).
ROWMOD_VIF = MOE_BOTH_VIF
#: How a decode rung of several records holds them out to pick its MoE column
#: (`_moe_column`), one of `MOE_FOLDS`, the first when unset: each record
#: whole, or each record's group compositions, graded only on the steps the
#: other cells' steps cover (`_covered`).
MOE_FOLDS_ENV = "COMPASS_FIT_MOE_FOLDS"
MOE_FOLDS = ("record", "covered")
#: The quantities over whose convex hull the other cells' steps must hold a
#: held-out step for a MoE form's fold fit to grade it (`_covered`).
COVER_AXES = ("context", "moe_tokens", "group_dummies")
#: The decode rungs that add `peer_tokens` (`_peer_rung`): ``1`` for each
#: rung whose `moe_experts` column is `SATURATED`, else the rungs, comma separated.
PEER_TOKENS_ENV = "COMPASS_FIT_PEER_TOKENS"
#: The draft law's decode rungs that add `peer_tokens` above `dp` 1: ``1`` for
#: every rung, else the rungs, comma separated.
DRAFT_PEER_TOKENS_ENV = "COMPASS_FIT_DRAFT_PEER_TOKENS"
#: The share of the routed experts past which a rung's median step's
#: `moe_experts` count is saturated: past it the column barely moves with the
#: group's tokens.
SATURATED = 0.9
#: The opt-in columns whose variance inflation factor a vector states (`_vif`).
OPT_IN = ("row_padding", "peer_tokens", *GATHER_COLUMNS, *GRAPH_COLUMNS, "short_hinge")


def _rung_key(rung: int | None) -> str:
    return EAGER if rung is None else str(rung)


def _prefill_counts(
    view,
    where: str,
    form: str,
    dp: int,
    serial: bool,
    out: bool,
    buckets,
    eager,
    cold: bool = False,
    share: bool = False,
    kda: bool = False,
    cols=None,
) -> tuple:
    """The prefill vector's features of `view`'s step, `where` naming it in a
    refusal, as `_features` states them; under `eager` its `EAGER_DUMMY`,
    under `cold` its `cold_peer_rows`, the `GATHER_COLUMNS` `cols` turns on
    (`_gathered`), under `share` its `PEER_SHARE`, under `kda` its `KDA`
    columns, and the `GRAPH_COLUMNS` `cols` holds."""
    r = view.prefill
    counts = (1, sum_tokens(r), sum_query_square(r), sum_query_cached(r))
    if form == "history":
        counts += (sum_context(r) - sum_tokens(r),)
    if form == "cached_seqs":
        counts += (cached_seqs(r),)
    if cols and cols.get("short_hinge"):
        counts += (max(0, cols["short_hinge"] - sum_tokens(r)),)
    if dp > 1:
        if view.moe_rows is None:
            raise ValueError(
                f"{where} has no moe_rows, which a data-parallel law prices "
                "prefill from"
            )
        counts += (view.moe_rows,)
        if serial:
            counts += (0 if view.tbo_tokens else view.moe_rows,)
        if cold:
            if (peers := cold_peer_rows(view)) is None:
                raise ValueError(
                    f"{where} has no prefill_duty, which its peers' MoE rows "
                    "are priced by"
                )
            counts += (peers,)
        for feature in _gathered(cols):
            got = gather_column(feature, view, buckets, dp, _min_spread(cols))
            if got is None:
                raise ValueError(
                    f"{where} has no group_max_tokens, which its {feature} is "
                    "counted from"
                )
            counts += (got,)
    if out:
        output = bool(view.produces_output)
        counts += (len(r) if output else 0, int(output))
    if share:
        counts += peer_share(view)
    if kda:
        if (columns := kda_columns(view)) is None:
            raise ValueError(
                f"{where} states no kda counts, which a KDA model's law prices "
                "prefill from; its record predates them"
            )
        counts += columns
    if buckets:
        if (padding := token_padding(view, buckets)) is None:
            raise ValueError(
                f"{where} has no group_max_tokens, which its prefill token "
                "graph's bucket holds"
            )
        counts += (padding,)
    if graph := [f for f in GRAPH_COLUMNS if cols and cols.get(f)]:
        got = dict(zip(GRAPH_COLUMNS, graph_columns(view, buckets)))
        counts += tuple(got[f] for f in graph)
    if eager:
        if (got := eager_dummy(view, buckets)) is None:
            raise ValueError(
                f"{where} has no group_max_tokens, which says whether its group "
                "runs it eagerly"
            )
        counts += got
    return counts


def _cold(options) -> bool:
    """Whether the prefill vector adds `cold_peer_rows`: as `options`' `cold`
    states, False by default. It prices a data-parallel group's peers' rows,
    so `dp` 1 refuses it; a fitted prefill step whose view states no
    `prefill_duty` refuses (`_prefill_counts`)."""
    if not (options or {}).get("cold"):
        return False
    if _options(options)[1] == 1:
        raise ValueError(
            "cold_peer_rows prices the rows a data-parallel group's peers bring "
            "to its MoE gather, and these records run one rank"
        )
    return True


def _share(options) -> bool:
    """Whether the prefill vector adds `PEER_SHARE`: as `options`'
    `peer_share` states, False by default. Its columns mark a rank whose
    data-parallel peers run real rows, so `dp` 1 refuses them."""
    if not (options or {}).get("peer_share"):
        return False
    if _options(options)[1] == 1:
        raise ValueError(
            "peer_small and peer_large mark a prefill beside a data-parallel "
            "peer's real rows, and these records run one rank"
        )
    return True


def _keep_peer_tail(options) -> bool:
    """Whether the prefill vector's solve keeps the slower steps a
    `PEER_SHARE` column marks (`_fit_form`'s `protect`): as `options`'
    `keep_peer_tail` states, False by default; without `peer_share` there
    is no such column, which refuses it."""
    if not (options or {}).get("keep_peer_tail"):
        return False
    if not _share(options):
        raise ValueError(
            "keep_peer_tail keeps the steps a peer_small or peer_large column "
            "marks, and the fit adds neither without peer_share"
        )
    return True


def _kda(options) -> bool:
    """Whether the prefill vector adds `KDA`'s columns: as `options`' `kda`
    states, False by default; `compass fit steps --model` sets it for a model
    with KDA layers (`step_linear.kda_layers`). A fitted prefill step whose
    view states no `kda` counts refuses (`_prefill_counts`)."""
    return bool((options or {}).get("kda"))


def _min_spread(cols) -> float | None:
    """The `min_spread` `cols`' `moe_imbalance` and `uneven_rows` share
    (`_columns` refuses two), or None with neither."""
    spread = [cols[f] for f in SPREAD_COLUMNS if cols and cols[f] is not None]
    return spread[0] if spread else None


def _columns(options, draft: bool = False) -> dict:
    """The opt-in columns `options` asks a law's vectors to add (`env_options`):
    `row_padding` on decode, True or False, and on prefill `gather_rows` and
    `gather_pad_rows`, True or None (the `gather_rows` option True, or
    ``pad`` for `gather_pad_rows`), `moe_imbalance` and `uneven_rows`, their
    one `min_spread` or None, and `GRAPH_COLUMNS`, True or False; none on a
    draft law. Each gather column prices a data-parallel group's gather, so
    `dp` 1 refuses it, and `gather_rows`, `gather_pad_rows` and
    `GRAPH_COLUMNS`, which read which steps replay a token graph, refuse
    records stating no prefill buckets; a `min_spread` below 0, which
    `StepLinearLaw` refuses, is refused, as are two; the `shared` form
    refuses `row_padding`, whose decode solve shares the base columns over
    rungs."""
    options = options or {}
    gather = options.get("gather_rows") or None
    if gather not in (None, True, "pad"):
        raise ValueError(f"gather_rows is {gather!r}, not True or 'pad'")
    cols = {
        "row_padding": bool(options.get("row_padding")),
        "gather_rows": True if gather is True else None,
        "gather_pad_rows": True if gather == "pad" else None,
        "moe_imbalance": options.get("moe_imbalance"),
        "uneven_rows": options.get("uneven_rows"),
        **{f: bool(options.get(f)) for f in GRAPH_COLUMNS},
        "short_hinge": options.get("short_hinge"),
    }
    knee = cols["short_hinge"]
    if knee is not None and (type(knee) is not int or knee < 1):
        raise ValueError(f"short_hinge's knee is {knee!r}, not a positive token count")
    if draft:
        return {f: None for f in cols} | {
            "row_padding": False,
            **dict.fromkeys(GRAPH_COLUMNS, False),
        }
    form, dp = _options(options)
    for f in SPREAD_COLUMNS:
        if (spread := cols[f]) is not None and not spread >= 0:
            raise ValueError(
                f"{f}'s min_spread is a share of moe_rows, 0 or more, not "
                f"{spread!r}"
            )
    if len({cols[f] for f in SPREAD_COLUMNS} - {None}) > 1:
        raise ValueError(
            f"moe_imbalance at min_spread {cols['moe_imbalance']} and uneven_rows "
            f"at {cols['uneven_rows']}: a vector states one min_spread"
        )
    gathered = _gathered(cols)
    if gathered and dp == 1:
        raise ValueError(
            f"{' and '.join(gathered)} price a data-parallel group's MoE gather, "
            "and these records run one rank"
        )
    if gather and not _buckets(options):
        raise ValueError(
            f"{gathered[0]} reads which prefill steps replay a token graph from "
            "the records' prefill buckets, and they state none"
        )
    for name in GRAPH_COLUMNS:
        if cols[name] and not _buckets(options):
            raise ValueError(
                f"{name} reads which prefill steps replay a token graph from the "
                "records' prefill buckets, and they state none"
            )
    if cols["row_padding"] and form == "shared":
        raise ValueError("the shared form fits no row_padding column")
    return cols


def _gathered(cols) -> list[str]:
    """The prefill `GATHER_COLUMNS` that `cols` (`_columns`) turns on, each
    one whose entry is not None, so `moe_imbalance` at a `min_spread` of 0
    is on; none when `cols` is None."""
    return [f for f in GATHER_COLUMNS if cols and cols[f] is not None]


def _switch(env, name: str) -> bool:
    """Whether the switch `name` in `env` is on: ``1`` on, ``0`` or unset
    off, and any other value refused."""
    value = env.get(name) or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{name}={value} is not 0 or 1")
    return value == "1"


def env_options(environ=None) -> dict:
    """The `fit` options the opt-in column switches in `environ`, the process's
    environment by default, set: `ROW_PADDING_ENV`, `GATHER_ROWS_ENV` (or at
    ``pad``), `MOE_IMBALANCE_ENV` and `UNEVEN_ROWS_ENV` at ``1`` (the last two
    with `MOE_MIN_SPREAD_ENV`'s share), `GRAPH_TOKENS_ENV` and
    `GRAPH_QUERY_CACHED_ENV` at ``1``, the
    `MOE_SELECT_ENV` rule, `MOE_BOTH_ENV` and `ROWMOD_VIF_ENV` at ``1``, the
    `MOE_FOLDS_ENV` cells, and the `PEER_TOKENS_ENV` and
    `DRAFT_PEER_TOKENS_ENV` rungs; none set, none.
    A switch set to anything but ``0`` or ``1`` is refused (`_switch`)."""
    env = os.environ if environ is None else environ
    out = {}
    if _switch(env, ROW_PADDING_ENV):
        out["row_padding"] = True
    if (knee := env.get(SHORT_HINGE_ENV) or "0") != "0":
        if not knee.isdigit():
            raise ValueError(f"{SHORT_HINGE_ENV}={knee} is not a whole number of tokens")
        out["short_hinge"] = int(knee)
    if (gather := env.get(GATHER_ROWS_ENV)) == "pad":
        out["gather_rows"] = "pad"
    elif gather not in (None, "", "0", "1"):
        raise ValueError(f"{GATHER_ROWS_ENV}={gather} is not 0 or 1 (or pad)")
    elif _switch(env, GATHER_ROWS_ENV):
        out["gather_rows"] = True
    for key, name in (
        ("graph_tokens", GRAPH_TOKENS_ENV),
        ("graph_query_cached", GRAPH_QUERY_CACHED_ENV),
    ):
        if _switch(env, name):
            out[key] = True
    spread = env.get(MOE_MIN_SPREAD_ENV)
    for key, name in (
        ("moe_imbalance", MOE_IMBALANCE_ENV),
        ("uneven_rows", UNEVEN_ROWS_ENV),
    ):
        if _switch(env, name):
            out[key] = MOE_MIN_SPREAD if spread is None else float(spread)
    if spread is not None and not {"moe_imbalance", "uneven_rows"} & set(out):
        raise ValueError(
            f"{MOE_MIN_SPREAD_ENV} is moe_imbalance's and uneven_rows'; set "
            f"{MOE_IMBALANCE_ENV}=1 or {UNEVEN_ROWS_ENV}=1"
        )
    if select := env.get(MOE_SELECT_ENV):
        if select not in MOE_SELECTS:
            raise ValueError(f"{MOE_SELECT_ENV}={select} is not one of {MOE_SELECTS}")
        out["moe_select"] = select
    if _switch(env, MOE_BOTH_ENV):
        out["moe_both"] = True
    if _switch(env, ROWMOD_VIF_ENV):
        out["rowmod_vif"] = True
    if folds := env.get(MOE_FOLDS_ENV):
        if folds not in MOE_FOLDS:
            raise ValueError(f"{MOE_FOLDS_ENV}={folds} is not one of {MOE_FOLDS}")
        out["moe_folds"] = folds
    if peers := env.get(PEER_TOKENS_ENV):
        out["peer_tokens"] = (
            "saturated" if peers == "1" else tuple(int(r) for r in peers.split(","))
        )
    if peers := env.get(DRAFT_PEER_TOKENS_ENV):
        out["draft_peer_tokens"] = (
            "all" if peers == "1" else tuple(int(r) for r in peers.split(","))
        )
    return out


def _peer_rung(steps, rung: int, vector: dict, routing, peers) -> bool:
    """Whether decode rung `rung`, its `steps` and the `vector` it adopted, adds
    `peer_tokens` under `peers`, the `peer_tokens` option or a draft law's
    `draft_peer_tokens`: a rung it names, every rung under ``all``, or under
    ``saturated`` a rung that adopted `moe_experts` and whose median step
    reaches `SATURATED` of the routed experts or more."""
    if not peers:
        return False
    if peers == "all":
        return True
    if peers != "saturated":
        return rung in peers
    if not routing:
        return False
    if "moe_experts" not in vector["coefficients"]:
        return False
    reached = [moe_column("moe_experts", s[0][6], routing, s[0][7]) for s in steps]
    return float(np.median(reached)) >= SATURATED * routing[0]


def _vif(points, features, columns) -> dict:
    """Each of `columns`' variance inflation factor over `points`, the rows a
    vector of `features`, the intercept first, was fitted on: the column's
    variance over its residual's on the other columns and an intercept,
    `1 / (1 - R^2)`, infinite where they hold it exactly. A column that does
    not vary has none."""
    x = np.asarray(points, dtype=np.float64)[:, 1:]
    out = {}
    for f in columns:
        j = features.index(f) - 1
        others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        fitted = others @ np.linalg.lstsq(others, x[:, j], rcond=None)[0]
        spread = float(((x[:, j] - x[:, j].mean()) ** 2).sum())
        residual = float(((x[:, j] - fitted) ** 2).sum())
        if spread > 0:
            out[f] = spread / residual if residual > 0 else math.inf
    return out


def _kept_vif(solved, vector: dict, routing, columns) -> dict:
    """`_vif` of `columns` over the steps `solved`, `vector`'s `_fit_vector`,
    keeps, each its rows as `_counts` counts them."""
    points = [
        _counts(vector, routing, r[2], *r[5:])
        for r, keep in zip(solved["rows"], solved["keep"])
        if keep
    ]
    return _vif(points, vector["coefficients"], columns)


def _eager_dummies(samples, options) -> bool:
    """Whether the prefill vector adds `EAGER_DUMMY`: as `options` states, else
    above `dp` 1 when some prefill step of `samples` had a dummy peer and ran
    eagerly (`eager_dummy` of its dummy's view), under `options`' buckets."""
    if "eager_dummies" in (options or {}):
        return options["eager_dummies"]
    buckets = _buckets(options)
    return _options(options)[1] > 1 and any(
        s.view.prefill
        and s.view.group_dummies
        and s.view.group_max_tokens is not None
        and eager_dummy(dummy_view(s.view), buckets)[0]
        for s in samples
        if not s.cut
    )


def _features(
    samples,
    form: str,
    dp: int,
    draft: bool = False,
    serial: bool = False,
    routing: tuple[int, int] | None = None,
    buckets: tuple[int, ...] = (),
    eager: bool = False,
    cold: bool = False,
    share: bool = False,
    kda: bool = False,
    cols=None,
) -> tuple[list[tuple], dict]:
    """One rank's fittable samples as (vector, features, seconds, step, context,
    tokens, dummies, peers, dummy), and how many were cut, per reason.

    `vector` is `prefill` or a decode rung's key; features are in `StepLinearLaw`'s
    order, plus `history` or `cached_seqs` in its form, above `dp` 1 `moe_rows` on prefill and
    under `serial` `serial_moe_rows`, under `cold` `cold_peer_rows`, and under `draft` or on a step with a draft
    label `seqs` and `out` on prefill, under `share` its `PEER_SHARE`, under `kda` its `KDA` columns, with
    `buckets` its `padding`, and under `eager` its `EAGER_DUMMY`, as `StepLinearBackend` counts them;
    `cols` (`_columns`) adds `row_padding` on decode and `GATHER_COLUMNS` on prefill;
    `context` is the decode rows' N_KV in scheduled order, empty for prefill;
    `tokens` a decode step's real MoE tokens under `routing`, the model's
    (experts, top-k), which its MoE column reads, else None; `dummies` its
    group's dummy forwards (`BatchView.group_dummies`), which the column's
    `moe_experts` counts as one token; `peers` its `peer_tokens`, None where
    its view states no group tokens; `dummy`, on a prefill step above `dp` 1
    and not under `draft`, the features of a peer's dummy forward at that step
    (`dummy_view`), as the backend prices that rank, else None.
    """
    cut, rows = dict.fromkeys(CUTS, 0), []
    for s in samples:
        view = s.view
        if s.cut:
            cut[s.cut] = cut.get(s.cut, 0) + 1
        else:
            tokens = dummy = None
            peers = 0
            if view.prefill:
                key = "prefill"
                where = f"{s.provenance['run']} step {s.step}"
                shape = (form, dp, serial, draft or s.draft_s is not None, buckets)
                flags = (eager, cold, share, kda, cols)
                counts = _prefill_counts(view, where, *shape, *flags)
                if dp > 1 and not draft:
                    dummy = _prefill_counts(dummy_view(view), where, *shape, *flags)
            else:
                key = _rung_key(view.capture_rung)
                counts = (1, sum_context(view.decode), view.graph_padding)
                if cols and cols["row_padding"]:
                    counts += (row_padding(view.decode),)
                peers = peer_tokens(view)
                if routing and (tokens := moe_tokens(view)) is None:
                    raise ValueError(
                        f"{s.provenance['run']} step {s.step} has no moe_tokens, "
                        "which a data-parallel law's MoE column reads"
                    )
            context = tuple(r.context_tokens for r in view.decode)
            dummies = view.group_dummies
            rows.append(
                (
                    key,
                    counts,
                    s.device_s,
                    s.step,
                    context,
                    tokens,
                    dummies,
                    peers,
                    dummy,
                )
            )
    return rows, cut


def _host_x(s) -> tuple[int, int, int]:
    return 1, sum_tokens(s.view.requests), len(s.view.requests)


#: The spans of its steps a host fit on one run holds out in turn to choose
#: launch columns (`_launch_columns`), each a contiguous part of its time.
RUN_BLOCKS = 5


def _host_rows(samples, blocks: int = 0, launch_of=None) -> list[tuple]:
    """(kind, segment, (1, tokens, seqs), seconds, run, launch) per host label
    of each sample, but the `E` of one that admits requests. With `blocks`,
    `run` is ``(run, block)``, the sample's step's block of `blocks` contiguous
    spans of its run's steps. `launch_of` (default: its `launch`) gives a
    sample's launch features."""
    rows, last = [], {}
    for s in samples:
        run = s.provenance["run"]
        last[run] = max(last.get(run, 0), s.step + 1)
    for s in samples:
        run = s.provenance["run"]
        if blocks:
            run = (run, s.step * blocks // last[run])
        launch = s.launch if launch_of is None else launch_of(s)
        rows += [
            (host_kind(s.view), k, _host_x(s), v, run, launch)
            for k, v in s.host.items()
            if k != "E" or not s.admitted[0]
        ]
    return rows


def _host_price(host, kind, segment, x, launch) -> float | None:
    """What the host entry `host` prices a `kind` forward's `segment` at `x`
    that launched `launch`, its columns included; None with no law."""
    c = host["laws"].get(kind, {}).get(segment)
    columns = host.get("columns", {}).get(kind, {}).get(segment, {})
    priced = sum(v * launch[f] for f, v in columns.items())
    return None if c is None else float(np.dot(c, x)) + priced


def _median_centred(coef, fitted) -> tuple[list, dict]:
    """`coef` (a host segment's law, its columns' coefficients too) scaled by
    1 - `fitted`'s centre, the median of 1 - label / price over its samples, so
    that median is 0; and `fitted` with the scale and the solver's centre."""
    scale = 1.0 - fitted["centre"]
    if not scale > 0:
        raise ValueError(f"centre {fitted['centre']} leaves no positive scale")
    return [v * scale for v in coef], fitted | {
        "median_centred": {"scale": scale, "solver_centre": fitted["centre"]},
        "centre": 0.0,
    }


def _admit_median(base, labels) -> tuple[list, dict]:
    """`admit` as the median of `labels` (the admitting forwards' `E`) less
    `base` (what their kind's `E` law prices), 0 or more, per admitting forward,
    no token terms; and its report. Raises on no sample."""
    extra = np.maximum(np.array(labels, float) - np.array(base, float), 0.0)
    if not len(extra):
        raise ValueError("no forward that admits a request has an E label")
    p10, p50, p90 = np.percentile(extra, (10, 50, 90)).tolist()
    return [p50, 0.0, 0.0], {"form": "median", "samples": len(extra), "p10": p10, "p50": p50, "p90": p90}


def _admit_rows(samples, host, launch_of=None) -> list[tuple]:
    """(`admit_x` of admitted, `E` its kind's law prices, `E`) per sample
    that admits requests, has a positive `E` and a kind with an `E` law;
    `launch_of` as `_host_rows` takes it."""
    of = launch_of or (lambda s: s.launch)
    return [
        (step_labels.admit_x(s.admitted), base, s.host["E"])
        for s in samples
        if s.admitted[0]
        and s.host.get("E", 0) > 0
        and (base := _host_price(host, host_kind(s.view), "E", _host_x(s), of(s)))
        is not None
    ]


def ladder_cut(samples) -> tuple[list, dict]:
    """`samples` less each of a calibration ladder's client run (`Sample.ladder`)
    but its decodes at the rungs it calibrates: its decodes at any other rung
    and its prefills, middle chunks too; and the report of what was left out:
    per run, rung (`EAGER` for an eager step) or `host_kind` of a step that
    is no decode, to the samples left out."""
    kept, cut = [], {}
    for s in samples:
        rung, kind = s.view.capture_rung, host_kind(s.view)
        if s.ladder is not None and (kind != "decode" or rung not in s.ladder):
            at = cut.setdefault(s.provenance["run"], {})
            key = _rung_key(rung) if kind == "decode" else kind
            at[key] = at.get(key, 0) + 1
            continue
        kept.append(s)
    return kept, cut


def records_atom(samples) -> str | None:
    """The ATOM digest `samples`' records ran at, their host keys' ``atom``;
    None when no record has a host key. Records at two refuse."""
    atoms = sorted(
        {s.provenance["config"]["atom"] for s in samples if s.provenance["config"]}
    )
    if len(atoms) > 1:
        raise ValueError(f"the records ran at {len(atoms)} ATOM digests: {atoms}")
    return atoms[0] if atoms else None


def label_records(records, dp: int = 1) -> tuple[list, dict]:
    """`records`, runs of `dp` ranks in rank order, as `step_labels` samples, each
    run named by its records' digests; and each record's table rows."""
    samples, tables = [], {}
    for i in range(0, len(records), dp):
        run = records[i : i + dp]
        name = ", ".join(map(step_labels.digest, run))
        for record, (got, table) in zip(run, step_labels.label_run(run, name)):
            samples += got
            tables[record] = table
    return samples, tables


def _runs(samples, dp: int) -> dict[str, list[list]]:
    """`samples` per run, in the order runs first appear, as each rank's samples."""
    runs: dict[str, dict[int, list]] = {}
    for s in samples:
        runs.setdefault(s.provenance["run"], {}).setdefault(s.rank, []).append(s)
    if any(len(ranks) != dp for ranks in runs.values()):
        n = sum(map(len, runs.values()))
        raise ValueError(f"{n} records are not runs of {dp} ranks")
    return {run: [ranks[r] for r in sorted(ranks)] for run, ranks in runs.items()}


def fit_host(samples, queue=None) -> tuple[dict, dict]:
    """The host laws over `samples`, as a run file's `host` entry, and their
    report. Under `HOST_QUEUE_ENV`, `queue` holds each sample's queue depth by
    run and step (`step_labels.queue_depths`), the listed segments' column."""
    from scipy.spatial import QhullError

    held = {json.dumps(s.provenance["config"], sort_keys=True) for s in samples}
    held = {k: json.loads(k) for k in held}
    at = pooled({k: c["engine"] for k, c in held.items() if c})
    keys = {
        json.dumps(c and {**c, "engine": at[k]}, sort_keys=True)
        for k, c in held.items()
    }
    keys = sorted(keys)
    if len(keys) != 1:
        raise ValueError(
            f"the records carry {len(keys)} host keys ({' / '.join(keys)}); "
            "host laws are fitted under one"
        )
    launch, graphs = launch_features.stated(json.loads(keys[0]))
    for s in samples:
        if (predicted := launch.predict(s.view, graphs)) != s.launch:
            raise ValueError(
                f"{s.provenance['run']} step {s.step} launched {s.launch}, and "
                f"its launch provider predicts {predicted}"
            )
    jit = _jit(samples, launch.features)
    after = {(run, rank, n + 1) for run, rank, n in sorted(jit)}
    kept = [
        s for s in samples if (s.provenance["run"], s.rank, s.step) not in jit | after
    ]
    queued = host_queue_spec()
    launch_of = column = None
    if queued is not None:
        q0, _ = queued
        column = f"{step_labels.QUEUE_COLUMN}{q0}"

        def launch_of(s):
            depth = (queue or {}).get(s.provenance["run"], {}).get(s.step)
            if depth is None:
                raise ValueError(
                    f"{HOST_QUEUE_ENV}: {s.provenance['run']} step {s.step} has no "
                    "queue depth (its record's intake stamps)"
                )
            return {**s.launch, column: min(depth, q0)}

    # One run holds out spans of its own steps to choose its launch columns.
    one = len({s.provenance["run"] for s in kept}) == 1
    rows = _host_rows(kept, RUN_BLOCKS if one else 0, launch_of)
    host = {"key": json.loads(keys[0]), "laws": {}}
    report = {"cut": {"idle": sum(s.idle for s in samples)}, "unfitted": {}}
    report["blocked"], report["unstamped"] = _blocked(samples)
    report["jit_steps"] = {}
    for run, _, n in sorted(jit):
        report["jit_steps"].setdefault(run, []).append(n)
    centred = {x for x in os.environ.get(HOST_MEDIAN_ENV, "").split(",") if x}
    for kind in HOST_KINDS:
        extra = sorted({r[1] for r in rows if r[0] == kind} - set(HOST_SEGMENTS))
        for segment in HOST_SEGMENTS[: 3 if kind == "middle" else 4] + tuple(extra):
            members = [r for r in rows if r[:2] == (kind, segment) and r[3] > 0]
            try:
                adopted, figures = _launch_columns(members, launch.features)
                if column is not None and f"{kind}.{segment}" in queued[1]:
                    adopted = [*adopted, column]
                    figures = figures | {"queue": {"column": column, "Q0": queued[0]}}
                columns = [(f, [r[5][f] for r in members]) for f in adopted]
                coef, fitted = _host_law(
                    [r[2] for r in members],
                    [r[3] for r in members],
                    HOST_FEATURES,
                    columns=columns,
                )
                if f"{kind}.{segment}" in centred:
                    coef, fitted = _median_centred(list(coef), fitted)
            except (ValueError, QhullError) as e:
                report["unfitted"][f"{kind}.{segment}"] = str(e).splitlines()[0]
                continue
            host["laws"].setdefault(kind, {})[segment] = coef[: len(HOST_FEATURES)]
            if adopted:
                c = dict(zip(adopted, coef[len(HOST_FEATURES) :]))
                host.setdefault("columns", {}).setdefault(kind, {})[segment] = c
            report[f"{kind}.{segment}"] = fitted | figures
    _fit_dummy(samples, launch, graphs, host, report)
    a, base, labels = list(zip(*_admit_rows(kept, host, launch_of))) or [(), (), ()]
    try:
        if (form := os.environ.get(ADMIT_FORM_ENV, "")) not in ("", "median"):
            raise ValueError(f"{ADMIT_FORM_ENV}={form} is not median")
        if form == "median":
            host["admit"], report["admit"] = _admit_median(base, labels)
        else:
            host["admit"], report["admit"] = _host_law(a, labels, ADMIT_FEATURES, base)
    except (ValueError, QhullError) as e:
        host["admit"] = None  # a run refuses a forward that admits a request
        report["unfitted"]["admit"] = str(e).splitlines()[0]
    return host, report


#: The host-law kind of a DP rank's dummy forward at a step its group
#: prefills (`step_labels.Sample.dummy_host`).
DUMMY = "dummy"


def _fit_dummy(samples, launch, graphs, host, report) -> None:
    """Fit `DUMMY`'s laws into `host`, each of `step_labels.DUMMY_SEGMENTS` on
    the dummy forwards' labels the samples' `dummy_host` holds, one set per
    run and rank, as `_host_law` fits a kind's, its `report` entry under
    ``dummy.<segment>``, and every launch feature of `launch` a column: a
    dummy runs its group's launch path, which sets its `W_launch` (DeepSeek-V4.1:
    about 2 ms replayed, 27 ms past the largest prefill token graph or on an
    eager tail, 51 ms on both), so no held-out choice is made. A dummy whose
    features `launch` predicts otherwise than its line states refuses, as a
    sample does; a segment too few dummies give is named under `unfitted`."""
    from scipy.spatial import QhullError

    held = {}
    for s in samples:
        held.setdefault((s.provenance["run"], s.rank), s.dummy_host)
    rows = [row for hosts in held.values() for row in hosts]
    if not rows:
        return
    for step, view, _, observed in rows:
        if (predicted := launch.predict(view, graphs)) != observed:
            raise ValueError(
                f"a dummy forward at step {step} launched {observed}, and its "
                f"launch provider predicts {predicted}"
            )
    for segment in step_labels.DUMMY_SEGMENTS:
        members = [
            (_host_x_view(view), h[segment], observed)
            for _, view, h, observed in rows
            if h.get(segment, 0) > 0
        ]
        columns = [(f, [m[2][f] for m in members]) for f in launch.features]
        try:
            coef, fitted = _host_law(
                [m[0] for m in members],
                [m[1] for m in members],
                HOST_FEATURES,
                columns=columns,
            )
        except (ValueError, QhullError) as e:
            report["unfitted"][f"{DUMMY}.{segment}"] = str(e).splitlines()[0]
            continue
        host["laws"].setdefault(DUMMY, {})[segment] = coef[: len(HOST_FEATURES)]
        if columns:
            c = dict(zip(launch.features, coef[len(HOST_FEATURES) :]))
            host.setdefault("columns", {}).setdefault(DUMMY, {})[segment] = c
        report[f"{DUMMY}.{segment}"] = fitted | {"forwards": len(members)}


def _host_x_view(view) -> tuple[int, int, int]:
    return 1, sum_tokens(view.requests), len(view.requests)


def _blocked(samples) -> tuple[dict, list[str]]:
    """Per run whose lines state `blocked_ns` and kind, its launches, those
    that blocked in a HIP call, their blocked ms at p50 and p90, and each
    call's blocked ms in all; and the runs whose lines state none, so their
    `W_launch` labels are wall times."""
    runs: dict = {}
    for s in samples:
        if "W_launch" in s.host:
            runs.setdefault(s.provenance["run"], []).append(s)
    blocked = {}
    for run, launches in runs.items():
        if all(s.blocked is None for s in launches):
            continue
        for kind in HOST_KINDS:
            mine = [s.blocked or {} for s in launches if host_kind(s.view) == kind]
            ms = [sum(b.values()) * 1e3 for b in mine if b]
            calls: dict = {}
            for b in mine:
                for call, seconds in b.items():
                    calls[call] = calls.get(call, 0.0) + seconds * 1e3
            blocked.setdefault(run, {})[kind] = {
                "launches": len(mine),
                "blocked": len(ms),
                "ms": np.percentile(ms, (50, 90)).tolist() if ms else None,
                "calls": calls,
            }
    return blocked, sorted(runs.keys() - blocked.keys())


def fit_tokenizer(paths, intakes, windows=None) -> tuple[dict, dict]:
    """The ``tokenizer`` term over the lone calls (`_lone_calls`) each
    tokenizer record in `paths` holds, beside the intake stamps in `intakes`,
    and its report: the records' backend and fingerprint, and per kind of
    call, ``fixed_s + tokens / tokens_per_s`` by the host-law solver on the
    calls' wall seconds, a segment tokenizer call's less its calls', over
    their ids, plus ``chars / chars_per_s`` over the characters its
    stored-prefix search compared where every call's record holds them
    (`_own_work`).
    One-item ``encode_batch`` calls are ``encode``'s and several-item ones
    their own kind. A ``decode`` with no positive slope is ``fixed_s`` alone
    over ``tokens``, the ``[lo, hi]`` token counts its calls span. A kind the
    calls cannot give is left out and its reason listed under the report's
    ``unfitted``, a ``segment`` without ``chars_per_s`` as
    ``segment.chars_per_s``; the report gives each record's p10, p50 and p90
    relative residual per kind. Raises on records of two tokenizers. With
    `windows`, per record its host-clock spans ``(start ns, end ns)``, or None
    for all of it, only the calls that start within one are fitted
    (`within`): each is still judged lone, and a segment call's own work
    found, among all of the record's calls."""
    from scipy.spatial import QhullError

    heads, rows = {}, []
    for i, (path, intake) in enumerate(zip(paths, intakes)):
        head, *lines = Path(path).read_text().splitlines()
        heads[json.dumps(json.loads(head), sort_keys=True)] = None
        got = [json.loads(line) for line in lines]
        span = windows[i] if windows else None
        rows += [(i, *row) for row in tokenizer_rows(got, intake, span)]
    if len(heads) != 1:
        raise ValueError(f"the records name {len(heads)} tokenizers: {sorted(heads)}")
    term, report = json.loads(next(iter(heads))), {"unfitted": {}}
    for kind in ("encode", "encode_batch", "decode", "segment"):
        got = np.array([r[2:4] for r in rows if r[1] == kind and r[3] > 0])
        chars = [r[4] for r in rows if r[1] == kind and r[3] > 0 and len(r) > 4]
        record = np.array([r[0] for r in rows if r[1] == kind and r[3] > 0])
        try:
            if not len(got):
                raise ValueError(f"the records hold no lone {kind} call")
            x = np.column_stack([np.ones(len(got)), got[:, 0]])
            if len(chars) == len(got) and None not in chars:
                x = np.column_stack([x, chars])
            coef, fitted = _host_law(x, got[:, 1], TOKENIZER_FEATURES[: x.shape[1]])
        except (ValueError, QhullError) as e:
            report["unfitted"][kind] = str(e)
            continue
        if coef[1] > 0:
            term[kind] = {"fixed_s": coef[0], "tokens_per_s": 1 / coef[1]}
            if len(coef) > 2 and coef[2] > 0:
                term[kind]["chars_per_s"] = 1 / coef[2]
            elif kind == "segment":
                lack = chars.count(None)
                report["unfitted"]["segment.chars_per_s"] = (
                    f"{lack} of {len(chars)} lone segment calls' records hold no "
                    "compared_chars"
                    if lack
                    else "the records give no time per compared character"
                )
        elif kind == "decode":
            term[kind] = {"fixed_s": coef[0], "tokens": _span(fitted, "tokens")}
        else:
            report["unfitted"][kind] = f"{kind}'s calls take no time per token"
            continue
        error = 1 - got[:, 1] / (x @ coef)
        fitted["records"] = {
            str(p): np.percentile(error[record == i], [10, 50, 90]).tolist()
            for i, p in enumerate(paths)
            if (record == i).any()
        }
        report[kind] = fitted
    return term, report


def tokenizer_rows(calls: list[dict], intake, windows=None) -> list[tuple]:
    """The rows `fit_tokenizer` fits of one record's tokenizer `calls`,
    beside the intake stamps in `intake`: ``(kind, tokens, seconds)`` per
    lone call (`_lone_calls`) but a segment call, and ``("segment", ids,
    seconds, chars)`` per segment call's own work (`_own_work`); with
    `windows`, host-clock spans, only the calls that start within one
    (`within`), each still judged lone among all of `calls`."""
    rows = []
    for c in _lone_calls([c for c in calls if c["call"] != "segment"], intake):
        if within(c["t_start_ns"], windows):
            kind = c["call"] if c["items"] > 1 or c["call"] == "decode" else "encode"
            rows.append((kind, c["tokens"], (c["t_end_ns"] - c["t_start_ns"]) * 1e-9))
    work = _own_work(calls, intake, windows)
    return rows + [("segment", n, s, chars) for n, chars, s in work]


def within(ns: int, windows) -> bool:
    """Whether the host-clock time `ns` falls in one of `windows`, ``(start
    ns, end ns)`` spans; any time does where `windows` is None."""
    return windows is None or any(a <= ns < b for a, b in windows)


def _span(fitted: dict, name: str) -> list:
    """The ``[lo, hi]`` of the one feature `name` of `_host_law`'s report."""
    if name in fitted["hull"]["equal"]:
        return [fitted["hull"]["equal"][name]] * 2
    (_, lo), (_, hi) = fitted["hull"]["facets"]
    return [lo, -hi]


def _stamps(path) -> list[tuple]:
    """The HTTP intake stamps `path` holds (`atom.compass.run.INTAKE_RECORD`),
    each ``(start ns, end ns, kind, size, start cpu ns, end cpu ns, thread)``,
    in order of start."""
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    return sorted(
        (int(r[3]), int(r[0]), r[2], int(r[4]), int(r[6]), int(r[5]), int(r[1]))
        for r in rows
    )


def _requests(ev) -> tuple[list, np.ndarray]:
    """Each request's app call among the stamps `ev`, in order, and every
    engine send's end, sorted."""
    # An array, searched once per body: a list is converted on every search.
    apps = np.array([e[0] for e in ev if e[2] == "app"], dtype=np.int64)
    # A request's app call is the last before its body's first read; any
    # other, such as a health probe's, has no body.
    calls = sorted(
        {int(apps[np.searchsorted(apps, e[0]) - 1]) for e in ev if e[2] == "body"}
    )
    return calls, np.array(sorted(e[1] for e in ev if e[2] == "send"))


def _lone_calls(calls: list[dict], intake) -> list[dict]:
    """The tokenizer calls among `calls` that ran lone: no other call, and no
    request's intake (`_requests` over the stamps in `intake`) in flight from
    the call's start to its end but, for an encode, its own."""
    apps, sends = _requests(_stamps(intake))
    calls = sorted(calls, key=lambda c: c["t_start_ns"])
    start = np.array([c["t_start_ns"] for c in calls])
    end = np.array([c["t_end_ns"] for c in calls])
    before = np.maximum.accumulate(np.r_[-math.inf, end[:-1]])
    after = np.r_[start[1:], math.inf]
    arrived = np.searchsorted(apps, end, "right") - np.searchsorted(apps, start)
    flight = np.searchsorted(apps, start) - np.searchsorted(sends, start)
    own = np.array([c["call"] != "decode" for c in calls])
    lone = (before < start) & (after > end) & (arrived == 0) & (flight <= own)
    return [c for c, k in zip(calls, lone) if k]


def _own_work(
    calls: list[dict], intake, windows=None
) -> list[tuple[int, int | None, float]]:
    """Each ``segment`` call among `calls` lone (`_lone_calls`) among the calls it
    did not make, its thread's that start in its span, as the ids those returned,
    the characters its stored-prefix search compared if its record holds them,
    and its wall seconds less theirs; with `windows`, only those that start
    within one (`within`)."""
    own, ids, made, last = {}, {}, set(), None
    for c in sorted(calls, key=lambda c: (c["thread"], c["t_start_ns"])):
        start, end = c["t_start_ns"], c["t_end_ns"]
        if last and last["thread"] == c["thread"] and start < last["t_end_ns"]:
            own[id(last)] -= end - start
            ids[id(last)] += c["tokens"]
            made.add(id(c))
        elif c["call"] == "segment":
            own[id(c)], ids[id(c)], last = end - start, 0, c
    lone = _lone_calls([c for c in calls if id(c) not in made], intake)
    return [
        (ids[id(c)], c.get("compared_chars"), own[id(c)] * 1e-9)
        for c in lone
        if c["call"] == "segment" and within(c["t_start_ns"], windows)
    ]


def _tokenizing(calls) -> dict:
    """Each thread's tokenizer calls other than a decode, as sorted start and
    end ns arrays, from the tokenizer record at `calls`; none when it names no
    file."""
    if calls is None or not Path(calls).is_file():
        return {}
    spans: dict = {}
    _, *lines = Path(calls).read_text().splitlines()
    for c in map(json.loads, lines):
        if c["call"] != "decode":
            spans.setdefault(c["thread"], []).append((c["t_start_ns"], c["t_end_ns"]))
    return {t: np.array(sorted(v), dtype=np.int64).T for t, v in spans.items()}


def _intakes(path, calls=None, windows=None) -> tuple[list, list, list]:
    """The HTTP intake stamps `path` holds as each lone request's ``(body
    bytes, CPU seconds)``, each burst's ``(body bytes, mark ns, mark CPU
    ns)`` per request, and every request's body bytes.

    A request is its app call and the body, template, pool job and engine send
    that follow it; one given its prompt's token ids, a prefill-decode decode
    side's, renders no template. A lone one found every request whose body was read before
    its app call sent, and its own sent before the next app call, with nothing
    else stamped between: its seconds are the CPU time the loop's thread spent
    from the app call to the send's end, less what it spent from its pool job's
    submit to its resume (``wait``), the job's and other requests' work; and,
    where its template was rendered on another thread, that render's wall
    time, which the loop awaits and the template job, charging only its encode
    calls, does not price. The pool job of a request given its token ids,
    during which its thread made no tokenizer call other than a decode by the
    tokenizer record at `calls` (none where there is none), is priced by no
    tokenizer law: its wall time, which the loop awaits, joins the request's
    seconds. A request's mark is its template's start or, on a server that
    renders none, its pool job's submit: either follows its body's parse. A
    burst is a run of requests each of whose app call came before the mark of
    the one ahead of it, the loop taking them in one after another, its marks
    on the loop's thread, the only one whose CPU clock their gaps read.
    With `windows`, host-clock spans (`within`), only the lone requests whose
    app call, the bursts whose first mark and the bodies whose read starts
    within one are given: each is still read among all of the record's stamps."""
    ev = _stamps(path)
    tokenized = _tokenizing(calls)
    # Arrays, searched once per request: np.searchsorted converts a list on
    # every call, which over a c=128 agentic record's two million stamps (an
    # `out` per streamed row) made this loop O(requests x stamps).
    at = np.array([e[0] for e in ev], dtype=np.int64)
    cpu = {e[0]: (e[4], e[6]) for e in ev if e[2] == "app"}
    calls, sends = _requests(ev)
    called = np.array(calls, dtype=np.int64)
    # Requests are counted by their bodies: two whose app calls both precede
    # either's body share one in `calls`, so a count of `calls` falls behind
    # the sends for the rest of the record and no later request is lone.
    read = np.array(sorted(e[0] for e in ev if e[2] == "body"), dtype=np.int64)
    lone = []
    for call, after in zip(calls, calls[1:] + [math.inf]):
        span = ev[np.searchsorted(at, call, "right") : np.searchsorted(at, after)]
        # An output taken in or a head sent is the output's or the response's.
        skip = ("app", "recv", "wait", "out", "head")
        between = [e for e in span if e[2] not in skip]
        waits = sum(e[5] - e[4] for e in span if e[2] == "wait")
        kinds = [e[2] for e in between]
        if np.searchsorted(sends, call) != np.searchsorted(read, call) or kinds not in (
            ["body", "tmpl", "prep", "send"],
            ["body", "prep", "send"],
        ):
            continue
        body, *rendered, prep, send = between
        start, loop = cpu[call]
        seconds = send[5] - start - waits
        for tmpl in rendered:
            seconds += (tmpl[1] - tmpl[0]) * (tmpl[6] != loop)
        starts = np.asarray(tokenized.get(prep[6], ((), ()))[0])
        if not rendered and not np.any((prep[0] <= starts) & (starts <= prep[1])):
            seconds += prep[1] - prep[0]
        if within(call, windows):
            lone.append((body[3], seconds * 1e-9))
    # The loop parses each request's body as soon as it is read, then renders
    # its template, if the server renders one, and submits its pool job; since
    # the API server hands a template to a thread, no gap of theirs is the loop's.
    loops = {e[6] for e in ev if e[2] == "app"}
    mark = "tmpl" if any(e[2] == "tmpl" for e in ev) else "wait"
    marks = [e for e in ev if e[2] == mark and e[6] in loops]
    bodies = [e for e in sorted(ev, key=lambda e: e[1]) if e[2] == "body"]
    sizes = [e[3] for e in bodies]
    # A mark that finds a later request arrived and not yet marked.
    queued = [np.searchsorted(called, m[0]) > k + 1 for k, m in enumerate(marks)]
    bursts, run = [], []
    for k, m in enumerate(marks):
        run.append((sizes[k], m[0], m[4]))
        if not queued[k]:
            if len(run) > 1 and within(run[0][1], windows):
                bursts.append(run)
            run = []
    sizes = [n for n, e in zip(sizes, bodies) if within(e[0], windows)]
    return lone, bursts, sizes


def fit_intake(paths, calls=None, windows=None) -> tuple[dict, dict]:
    """The ``http_intake`` term over the intake stamps each record in `paths`
    holds (`_intakes`), each beside the tokenizer record `calls` names in the
    same order (none where `calls` is None), and its report: ``idle_s`` and ``per_byte_s`` by the
    host-law solver on the lone requests' CPU seconds over their body bytes;
    ``bytes`` spans every request's body; ``fixed_s`` the mean, over every
    burst's later requests, of the loop's CPU time from the mark before each
    to its own, less ``per_byte_s`` times its body, the report's
    ``burst`` giving the bursts, their requests and those values' p10 and p90;
    or ``idle_s`` where that mean is negative or there is no burst, the
    report's ``fixed_s_is_idle_s`` saying which. Raises on too few lone
    requests. With `windows`, per record its host-clock spans or None, each
    record's requests within them (`_intakes`)."""
    lone, bursts, bodies = [], [], []
    spans = windows or [None] * len(paths)
    for path, made, span in zip(paths, calls or [None] * len(paths), spans):
        got = _intakes(path, made, span)
        lone += got[0]
        bursts += got[1]
        bodies += got[2]
    nbytes, seconds = np.array(lone, dtype=np.float64).reshape(-1, 2).T
    x = np.column_stack([np.ones(len(nbytes)), nbytes])
    (idle, per_byte), report = _host_law(x, seconds, INTAKE_FEATURES)
    term = {"per_byte_s": per_byte, "idle_s": idle, "bytes": [min(bodies), max(bodies)]}
    if not bursts:
        report["fixed_s_is_idle_s"] = f"no burst among {len(lone)} lone requests"
        return {"fixed_s": idle, **term}, report
    # Each burst's (body bytes, mark ns, its CPU ns) per request.
    runs = [np.array(b, dtype=np.float64).T for b in bursts]
    gaps = np.concatenate([np.diff(cpu) * 1e-9 for _, _, cpu in runs])
    sizes = np.concatenate([body[1:] for body, _, _ in runs])
    values = gaps - per_byte * sizes
    fixed = values.mean()
    if fixed < 0:
        report["fixed_s_is_idle_s"] = (
            f"the bursts' {len(gaps)} later requests take {gaps.mean():.3g} s "
            f"of the loop's CPU each, less than per_byte_s {per_byte:.3g} times "
            f"their mean body of {sizes.mean():.0f} B, so fixed_s would be "
            f"{fixed:.3g} s"
        )
        return {"fixed_s": idle, **term}, report
    p10, p90 = np.percentile(values, [10, 90]).tolist()
    report["burst"] = {
        "bursts": len(bursts),
        "requests": sum(map(len, bursts)),
        "p10_s": p10,
        "p90_s": p90,
    }
    return {"fixed_s": fixed, **term}, report


def fit_relay(pairs) -> tuple[dict, dict]:
    """The prefill host entry's ``relay`` term over `pairs`, each a
    prefill-decode deployment's prefill and decode intake stamps
    (`atom.compass.run.INTAKE_RECORD`), stamped on one host's clock, and its
    report.

    A relayed request leaves prefill's server once its output thread has
    taken in the last of its output rows, its response carrying the prompt's
    token ids, and reaches decode's server at the app call of the request the
    router builds from it, the body that call reads holding those ids. A
    request's rows come in a burst, its thread taking each within
    `RELAY_BURST_S` of the one before, and a burst holds every request one
    prefill step finished: it is sent from its last row's return. On each
    request alone on the relay (`_alone`), one burst's, the host-law solver
    fits ``fixed_s + body bytes x per_byte_s`` on the time from that return
    to its app call over that body's bytes, which ``bytes`` spans. Raises on
    too few requests alone on it."""
    took, sizes = [], []
    for prefill, decode in pairs:
        threads: dict = {}
        for e in _stamps(prefill):
            if e[2] == "out":
                threads.setdefault(e[6], []).append(e)
        # Each burst's last row, (start, end), thread by thread.
        lasts = sorted(
            (a[0], a[1])
            for rows in threads.values()
            for a, b in zip(rows, rows[1:] + [None])
            if b is None or b[0] - a[1] > RELAY_BURST_S * 1e9
        )
        ev = _stamps(decode)
        calls, _ = _requests(ev)
        bodies = sorted((e[0], e[3]) for e in ev if e[2] == "body")
        starts = np.array([b[0] for b in bodies], dtype=np.int64)
        if not lasts or not calls:
            continue
        got, seconds = _alone([a for a, _ in lasts], calls, [b for _, b in lasts])
        for arrival, ns in zip(got, seconds):
            # The body its app call reads, the first read after the call.
            k = int(np.searchsorted(starts, arrival))
            if k < len(bodies):
                took.append(ns * 1e-9)
                sizes.append(bodies[k][1])
    x = np.column_stack([np.ones(len(sizes)), sizes])
    (fixed, per_byte), report = _host_law(x, np.array(took), RELAY_FEATURES)
    term = {"fixed_s": fixed, "per_byte_s": per_byte, "bytes": [min(sizes), max(sizes)]}
    return term, report


def streams(path) -> bool:
    """Whether the server whose intake stamps `path` holds sent an SSE
    response head (``head``): it streamed its responses."""
    with open(path) as lines:
        return any(line.split(maxsplit=3)[2:3] == ["head"] for line in lines)


def _loop_windows(path) -> list[tuple]:
    """The intake stamps at `path` (`atom.compass.run.INTAKE_RECORD`) as one row
    per `STREAM_WINDOW_S` window: ``(seconds, events, requests, bytes, CPU
    seconds)``. CPU seconds are the API loop thread's, the one stamping app
    calls, read off its stamps' thread CPU clock between its first and last
    stamp in the window and scaled to the window, which they must span
    `STREAM_SPAN` of; events are the output rows its output threads took in
    (``out``), one SSE event each on a streamed response; requests and bytes
    are the bodies read and their sizes. None for a server that sent no SSE
    response head (``head``): its outputs stream nothing."""
    if not streams(path):
        return []
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    loop = next((r[1] for r in rows if len(r) >= 7 and r[2] == "app"), None)
    width = STREAM_WINDOW_S * 1e9
    counts: dict = {}
    clock = []
    for r in rows:
        if len(r) < 7:
            continue
        end, kind = int(r[0]), r[2]
        at = counts.setdefault(int(end // width), [0, 0, 0])
        if kind == "out":
            at[0] += 1
        elif kind == "body":
            at[1] += 1
            at[2] += int(r[4])
        if kind in LOOP_KINDS and r[1] == loop:
            clock.append((end, int(r[5])))
    clock.sort()
    ends = np.array([e for e, _ in clock], dtype=np.float64)
    cpu = np.array([c for _, c in clock], dtype=np.float64)
    out = []
    for w, (events, requests, nbytes) in sorted(counts.items()):
        i = int(np.searchsorted(ends, w * width))
        j = int(np.searchsorted(ends, (w + 1) * width)) - 1
        if j <= i or ends[j] - ends[i] < STREAM_SPAN * width:
            continue
        busy = (cpu[j] - cpu[i]) / (ends[j] - ends[i]) * STREAM_WINDOW_S
        out.append((STREAM_WINDOW_S, events, requests, nbytes, busy))
    return out


def fit_stream(paths) -> tuple[dict, dict]:
    """The ``http_stream`` term over the intake stamps each record in `paths`
    holds, and its report: ``per_event_s``, the API loop's CPU seconds per SSE
    event its streamed responses send, by the host-law solver on the loop's
    CPU over every window of the streaming records (`_loop_windows`) on
    `STREAM_FEATURES`. The window's seconds and its requests and bytes stand
    for the loop's idle polling and its intakes, which ``http_intake`` prices,
    and stay in the report. Raises where no record streamed."""
    rows = [w for path in paths for w in _loop_windows(path)]
    if not rows:
        raise ValueError("no record streamed an SSE response, so none times its events")
    x = np.array([w[:4] for w in rows], dtype=np.float64)
    y = np.array([w[4] for w in rows], dtype=np.float64)
    coef, report = _host_law(x, y, STREAM_FEATURES)
    report["coefficients"] = dict(zip(STREAM_FEATURES, coef))
    report["windows"] = len(rows)
    report["loop_busy"] = float(y.sum() / x[:, 0].sum())
    return {"per_event_s": coef[1]}, report


#: The host entry's ``hops`` `fit_hops` fits (`atom.compass.run.HOPS`).
HOPS = ("admission_s", "ipc_s", "ipc_out_s", "stream_s")


def fit_hops(intakes, clients) -> tuple[dict, dict]:
    """The ``hops`` term over the records whose intake stamps `intakes` names,
    each with the engine's queue handoffs and step lines beside them
    (``dp<r>.handoff.jsonl``, ``dp<r>.jsonl``) and its client requests'
    ``sent`` and ``acked``, CLOCK_MONOTONIC ns, in `clients`, or the reason
    there are none as ``missing``; and its report.

    Each hop is the p10 of its messages alone on it (`_alone`) in every
    record: ``admission_s``, a client send to its request's app call;
    ``ipc_s``, an engine send call's return to the engine input thread's receipt,
    inside a forward of that DP rank, since an idle engine's loop holds the GIL
    the thread waits on; ``ipc_out_s``, an output's put on the engine's output
    queue to the server's output thread taking its first row; ``stream_s``,
    an SSE head's send call's return to the client's ack. A hop the records cannot give is
    left out and its reason listed under the report's ``unfitted``."""
    took, report = {h: [] for h in HOPS}, {"unfitted": {}}
    for intake, client in zip(intakes, clients):
        for hop, messages in _hop_messages(Path(intake), client).items():
            try:
                took[hop] += messages()
            except ValueError as e:
                report["unfitted"][hop] = f"{intake}: {e}"
    term = {}
    for hop, s in took.items():
        if hop in report["unfitted"]:
            continue
        if not s:
            report["unfitted"][hop] = "no message ran alone on it"
            continue
        p10, p50, p90 = np.percentile(np.array(s) * 1e-9, [10, 50, 90]).tolist()
        term[hop] = p10
        report[hop] = {"p10_s": p10, "p50_s": p50, "p90_s": p90, "messages": len(s)}
    return term, report


#: ATOM's offload worker line naming its rank and a chunk's tokens and bytes.
OFFLOAD_WORKER = re.compile(
    r"offload worker rank=(\d+):.*? chunk=(\d+) gpu_staging_chunk_bytes=(\d+)"
)
#: ATOM's per-copy line under `OFFLOAD_PROFILE`: the chunks it copied and the
#: milliseconds the worker thread spent on it.
OFFLOAD_COPY = re.compile(
    r"\[OFFLOAD-(SAVE|LOAD)-PROF\] .*? chunks=(\d+) .*? total_ms=([\d.]+)"
)
OFFLOAD_PERF = "[OFFLOAD-PERF] "
#: The Mooncake Store worker's start line, with the fields its entry matches
#: against a deployment; ``nic`` names the protocol, or under rdma the NIC.
STORE_WORKER = re.compile(
    r"Mooncake Store offload worker rank=(?P<rank>\d+) world=(?P<world>\d+): .*? "
    r"nic=(?P<nic>\S+) .*? chunk=(?P<chunk_tokens>\d+) chunk_bytes=(?P<chunk_bytes>\d+) "
    r".*? save_slots=(?P<save_slots>\d+) load_slots=(?P<load_slots>\d+) "
    r".*? save_workers=(?P<save_workers>\d+) load_workers=(?P<load_workers>\d+)"
)
#: The Store worker's per-copy lines under `OFFLOAD_PROFILE` and its
#: scheduler's per-lookup line at DEBUG, each of ``name=value`` fields.
STORE_LINE = re.compile(r"\[OFFLOAD-(SAVE-PROF|LOAD-PROF|LOOKUP)\] (.*)")
#: Per line of `STORE_LINE`: the law it fits, its count, its milliseconds and
#: the parts of them the report keeps apart.
STORE_LAWS = {
    "SAVE-PROF": ("store_s", "chunks", "total_ms", ("pack_ms", "put_ms")),
    "LOAD-PROF": ("load_s", "chunks", "total_ms", ("get_ms", "unpack_ms")),
    "LOOKUP": ("lookup_s", "keys", "elapsed_ms", ()),
}


def fit_offload(logs, master=None, capacity_bytes=None) -> tuple[dict, dict]:
    """The run file's ``lmcache_offload`` entry, fitted on the server logs of
    real runs `logs`, each at `OFFLOAD_PROFILE` (`atom.compass.kv.lmcache_offload`),
    and its report.

    ``chunk_size`` and ``chunk_bytes`` are what each worker logs at its start,
    and ``tensor_parallel_size`` how many workers logged; logs that differ in
    any refuse. ``store_s`` and ``load_s`` are fitted by the host-law solver on
    each copy that moved a chunk: the worker thread's seconds for it on
    ``(1, chunks)``. ``lookup_s`` is fitted on the scheduler's counters, which
    it logs a minute apart: per minute with a lookup, the seconds its lookups
    took on ``(lookups, chunks)``, chunks counted as the prompt tokens they
    asked over the chunk size. The entry's ``fit`` holds the logs and the
    report.

    Logs of the Mooncake Store's workers give its ``mooncake_store`` entry
    instead (`_fit_store`), with the master's flags `master`, as argv, and the
    owners' mounted bytes `capacity_bytes`, which no log states."""
    if any(STORE_WORKER.search(Path(log).read_text(errors="replace")) for log in logs):
        return _fit_store(logs, master, capacity_bytes)
    if master is not None or capacity_bytes is not None:
        raise ValueError("lmcache_offload's logs state no Store master or owners")
    workers, copies, lookups = {}, {"SAVE": [], "LOAD": []}, []
    for log in logs:
        perf = []
        for line in Path(log).read_text(errors="replace").splitlines():
            if m := OFFLOAD_WORKER.search(line):
                workers.setdefault(log, {})[int(m[1])] = (int(m[2]), int(m[3]))
            elif (m := OFFLOAD_COPY.search(line)) and int(m[2]):
                copies[m[1]].append((1, int(m[2]), float(m[3]) * 1e-3))
            elif OFFLOAD_PERF in line:
                said = line.partition(OFFLOAD_PERF)[2].partition(" | ")[0]
                perf.append(json.loads(said))
        for a, b in itertools.pairwise(perf):
            d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
            if d.get("lookup_n", 0) > 0:
                lookups.append(
                    (d["lookup_n"], d["lookup_prompt_tokens"], d["lookup_us"])
                )
    stated = {
        (len(ranks), *geometry)
        for ranks in workers.values()
        for geometry in ranks.values()
    }
    if len(stated) != 1 or len(workers) != len(logs):
        raise ValueError(
            f"the logs state (workers, chunk, chunk bytes) {sorted(stated)}, and "
            f"{len(logs) - len(workers)} state none; an entry is fitted on one"
        )
    [(tp, chunk, chunk_bytes)] = sorted(stated)
    entry = {
        "chunk_size": chunk,
        "tensor_parallel_size": tp,
        "chunk_bytes": chunk_bytes,
    }
    report = {}
    for name, kind in (("store_s", "SAVE"), ("load_s", "LOAD")):
        rows = np.array(copies[kind], dtype=np.float64).reshape(-1, 3)
        entry[name], report[name] = _host_law(
            rows[:, :2], rows[:, 2], ["fixed", "chunks"]
        )
    rows = np.array(lookups, dtype=np.float64).reshape(-1, 3)
    x = np.column_stack([rows[:, 0], rows[:, 1] / chunk])
    entry["lookup_s"], report["lookup_s"] = _host_law(
        x, rows[:, 2] * 1e-6, ["lookups", "chunks"]
    )
    entry["fit"] = {"logs": [str(log) for log in logs], **report}
    return entry, report


def _fit_store(logs, master, capacity_bytes) -> tuple[dict, dict]:
    """`fit_offload`'s ``mooncake_store`` entry and report on the Store logs
    `logs`, at `OFFLOAD_PROFILE` and ``ATOM_LOG_LEVEL=DEBUG``.

    The fields `STORE_WORKER` matches are what each worker logs at its start,
    ``protocol`` ``rdma`` where it names a NIC; logs that differ in any refuse.
    Each law of `STORE_LAWS` is fitted by the host-law solver on its lines'
    seconds over ``(1, count)``: ``store_s`` and ``load_s`` on each copy with
    ``status=ok``, ``lookup_s`` on each lookup's call, per key it asked. The
    report keeps each copy's parts apart, their p50 and p90 milliseconds.

    A pipelined prefill's ranks (``world`` of them, every stage's) differ in
    the fields `mooncake_store_server.PER_RANK` names, as each stage holds
    other layers: where they do, the entry states each such field a rank, in
    rank order, and ``store_s`` and ``load_s`` each rank's own, fitted on its
    copies' lines (``rank=``). Ranks that agree give the entry as before."""
    if master is None or capacity_bytes is None:
        raise ValueError(
            "no log states the Store master's flags or its owners' bytes: give "
            "--master-flags and --capacity-bytes"
        )
    from atom.compass.kv.mooncake_store_server import PER_RANK

    stated, rows, parts, ranked = {}, {}, {}, {}
    for log in logs:
        for line in Path(log).read_text(errors="replace").splitlines():
            if m := STORE_WORKER.search(line):
                fields = {k: int(v) for k, v in m.groupdict().items() if k != "nic"}
                fields["protocol"] = "tcp" if m["nic"] == "tcp" else "rdma"
                stated.setdefault(log, set()).add(tuple(sorted(fields.items())))
            elif m := STORE_LINE.search(line):
                law, count, took, kept = STORE_LAWS[m[1]]
                said = dict(re.findall(r"(\w+)=(\S+)", m[2]))
                # A failed copy is not priced; other lookup lines time no call.
                if said.get("status", "ok") == "ok" and took in said:
                    seconds = float(said[took]) * 1e-3
                    row = (1, int(said[count]), seconds)
                    rows.setdefault(law, []).append(row)
                    if "rank" in said:
                        ranked.setdefault((law, int(said["rank"])), []).append(row)
                    for part in kept:
                        parts.setdefault(part, []).append(float(said[part]))
    # Each rank's fields, which every log must state alike.
    by_rank = {}
    for log, lines in stated.items():
        for line in lines:
            fields = dict(line)
            by_rank.setdefault(fields.pop("rank"), set()).add(
                tuple(sorted(fields.items()))
            )
    shared = sorted({f for fs in by_rank.values() for line in fs for f, _ in line})
    common = {
        f: {dict(line)[f] for fs in by_rank.values() for line in fs} for f in shared
    }
    apart = sorted(f for f, values in common.items() if len(values) > 1)
    world = common.get("world", set())
    ranks = sorted(by_rank)
    if (
        len(stated) != len(logs)
        or any(len(fs) != 1 for fs in by_rank.values())
        or len(world) != 1
        or set(apart) - set(PER_RANK)
        or (apart and ranks != list(range(next(iter(world)))))
    ):
        matched = sorted({line for fs in by_rank.values() for line in fs})
        raise ValueError(
            f"the logs' Store workers state {[dict(s) for s in matched]}, "
            f"and {len(logs) - len(stated)} state none; an entry is fitted on one"
        )
    entry = {f: next(iter(v)) for f, v in common.items() if f not in apart}
    for f in apart:  # a pipelined prefill's: one a rank
        entry[f] = [dict(next(iter(by_rank[r])))[f] for r in ranks]
    report = {}
    for line, (law, count, _, kept) in STORE_LAWS.items():
        x = np.array(rows.get(law, []), dtype=np.float64).reshape(-1, 3)
        if not len(x):
            # No line fits it: the scheduler's lookup lines are DEBUG-only, so a
            # run at INFO states none. The entry leaves the law out, and a run
            # refuses it until it is stated.
            report[law] = {"samples": 0, "missing": f"no [OFFLOAD-{line}] line in the logs"}
            continue
        entry[law], report[law] = _host_law(x[:, :2], x[:, 2], ["fixed", count])
        if apart and law in PER_RANK:
            each = [
                _host_law(
                    np.array(ranked.get((law, r), []), dtype=np.float64).reshape(-1, 3)[
                        :, :2
                    ],
                    np.array(ranked.get((law, r), []), dtype=np.float64).reshape(-1, 3)[
                        :, 2
                    ],
                    ["fixed", count],
                )
                for r in ranks
            ]
            entry[law] = [term for term, _ in each]
            report[law]["ranks"] = [rep for _, rep in each]
        for part in kept:
            p50, p90 = np.percentile(parts[part], [50, 90]).tolist()
            report[law][part] = {"p50": p50, "p90": p90}
    entry |= {"master": master, "capacity_bytes": capacity_bytes}
    entry["fit"] = {"logs": [str(log) for log in logs], **report}
    return entry, report


#: Per copy line of `STORE_LINE`, for `fit_stream_floors`: the floor it fits,
#: its count, the copy it stages on its stage's stream, and the windows that
#: enqueue one there (a save's packed segments, a load's windows).
STREAM_COPIES = {
    "SAVE-PROF": ("store_floor_s", "chunks", "pack_ms", ("segments",)),
    "LOAD-PROF": ("load_floor_s", "chunks", "unpack_ms", ("in_place_windows", "windows")),
}


def fit_stream_floors(logs, ranks, stream_wait: float = 1.0) -> tuple[dict, dict]:
    """The ``mooncake_store`` entry's `STREAM_FIELDS`
    (`atom.compass.kv.mooncake_store_server`), fitted on the Store logs
    `logs`, each rank of `ranks` its own floors: what each copy takes off its
    stage's stream, which ``COMPASS_STORE_ON_STREAM=1`` prices with the wait
    apart.

    ATOM's worker stages each window's pack or unpack on its stage's forward
    stream and waits for it there, so a line's ``pack_ms`` or ``unpack_ms``
    is that copy and the device work queued ahead of it. A rank's copy floor
    a window is the 10th percentile of its lines' copy milliseconds over
    their windows; a line's floor seconds are its ``total_ms`` with its copy
    replaced by that floor times its windows, fitted by the host-law solver on
    ``(1, chunks)`` as ``store_s`` and ``load_s`` are. ``stream_wait``, the
    share of a stage's backlog a window waits for, is stated, not fitted: all
    of it, as the copy is ordered after every kernel its host enqueued."""
    if not 0.0 <= stream_wait <= 1.0:
        raise ValueError(f"stream_wait is a fraction, got {stream_wait}")
    lines: dict = {}
    for log in logs:
        for line in Path(log).read_text(errors="replace").splitlines():
            m = STORE_LINE.search(line)
            if not m or m[1] not in STREAM_COPIES:
                continue
            law, count, copy, windows = STREAM_COPIES[m[1]]
            said = dict(re.findall(r"(\w+)=(\S+)", m[2].split(" draft_")[0]))
            if said.get("status", "ok") != "ok" or "rank" not in said:
                continue
            n = next((int(said[w]) for w in windows if int(said.get(w, 0))), 0)
            lines.setdefault((law, int(said["rank"])), []).append(
                (int(said[count]), n, float(said[copy]), float(said["total_ms"]))
            )
    entry, report = {}, {}
    for law in ("store_floor_s", "load_floor_s"):
        each = []
        for r in ranks:
            rows = np.array(lines.get((law, r), []), dtype=np.float64).reshape(-1, 4)
            staged = rows[rows[:, 1] > 0]
            if not len(staged):
                raise ValueError(f"no copy line of rank {r} stages a window for {law}")
            floor = float(np.percentile(staged[:, 2] / staged[:, 1], 10))
            seconds = (rows[:, 3] - rows[:, 2] + floor * rows[:, 1]) * 1e-3
            x = np.column_stack([np.ones(len(rows)), rows[:, 0]])
            term, rep = _host_law(x, seconds, ["fixed", "chunks"])
            rep["copy_floor_ms_a_window"] = floor
            each.append((term, rep))
        entry[law] = [t for t, _ in each] if len(ranks) > 1 else each[0][0]
        report[law] = {"ranks": [rep for _, rep in each]}
    entry["stream_wait"] = float(stream_wait)
    return entry, report


def _alone(sent, got, returned=None) -> tuple[np.ndarray, np.ndarray]:
    """The arrival and ``got - sent`` of each message alone on its hop: sent
    once every message ahead of it arrived, and arrived before the next was
    sent. One side may miss messages the other saw, as a client misses those
    in flight at its stop (`_paired`). With `returned`, the return of each send
    call, made one at a time, a message is timed from it, and is not alone when
    it arrived first. Raises on a side with no message."""
    sent, got = np.sort(np.array(sent, np.int64)), np.sort(np.array(got, np.int64))
    if not len(sent) or not len(got):
        raise ValueError(f"{len(sent)} sends and {len(got)} arrivals")
    if len(got) >= len(sent):
        k, i = _paired(sent, got)
    else:  # in reverse time each arrival is a send, and each send an arrival
        i, k = _paired(-got[::-1], -sent[::-1])
        k, i = len(sent) - 1 - k, len(got) - 1 - i
    if returned is not None:
        sent = np.sort(np.array(returned, np.int64))
    took = got[i] - sent[k]
    return got[i][took > 0], took[took > 0]


def _paired(sent, got) -> tuple[np.ndarray, np.ndarray]:
    """The indices into `sent` and `got`, both sorted, of each message alone:
    exactly one arrival between its send and the next, and as many before its
    send as at any send before it. `got` may hold arrivals whose sends `sent`
    misses, each raising that count for good."""
    before = np.searchsorted(got, sent, "right")
    upto = np.searchsorted(got, np.r_[sent[1:], np.iinfo(np.int64).max], "right")
    ahead = before - np.arange(len(sent))
    alone = (upto - before == 1) & (ahead == np.maximum.accumulate(ahead))
    return np.flatnonzero(alone), before[alone]


def _hop_messages(intake: Path, client) -> dict:
    """Each hop's alone messages' ns in the record whose intake stamps are at
    `intake`, as a call per hop that raises on what the record cannot give;
    `client` is as `fit_hops` takes it."""
    ev = _stamps(intake)
    calls, sends = _requests(ev)
    recvs, puts, forwards = {}, [], {}
    for path in intake.parent.glob("dp*.handoff.jsonl"):
        rank = path.name.split(".")[0]
        for h in json_file.read_jsonl(path):
            if "t_recv_ns" in h:
                recvs[h["t_recv_ns"]] = rank
            elif h["rows"]:
                puts.append((h["t_out_put_ns"], h["rows"]))
        steps = json_file.read_jsonl(path.with_name(f"{rank}.jsonl"))
        forwards[rank] = np.array(
            sorted((s["t_enter_ns"], s["t_exit_ns"]) for s in steps)
        )

    def stepping(t, rank) -> bool:
        enter, exit_ = forwards[rank].reshape(-1, 2).T
        i = np.searchsorted(enter, t, "right") - 1
        return i >= 0 and t <= exit_[i]

    def of_client(field: str) -> list:
        if "missing" in client:
            raise ValueError(client["missing"])
        return client[field]

    def handed() -> None:
        if not forwards:
            raise ValueError("no engine's dp<r>.handoff.jsonl is beside it")

    def ipc():
        handed()
        called = [e[0] for e in ev if e[2] == "send"]
        got, took = _alone(called, list(recvs), returned=sends)
        return [d for g, d in zip(got, took) if stepping(g, recvs[g])]

    def ipc_out():
        # From the first request on, each output timed to its first row: the
        # server stamps each row it takes in, and the message came with its first.
        handed()
        if not len(sends):
            raise ValueError("no request was sent")
        put = sorted(p for p in puts if p[0] > sends[0])
        rows = sorted(e[0] for e in ev if e[2] == "out" and e[0] > sends[0])
        firsts = np.cumsum([0] + [n for _, n in put])
        if firsts[-1] != len(rows):
            raise ValueError(f"{len(rows)} rows taken in, not the {firsts[-1]} put")
        return _alone([t for t, _ in put], [rows[i] for i in firsts[:-1]])[1].tolist()

    heads = [e for e in ev if e[2] == "head"]
    return {
        "admission_s": lambda: _alone(of_client("sent"), calls)[1].tolist(),
        "ipc_s": ipc,
        "ipc_out_s": ipc_out,
        "stream_s": lambda: _alone(
            [e[0] for e in heads], of_client("acked"), [e[1] for e in heads]
        )[1].tolist(),
    }


def _host_law(x, y, names, offset=0.0, columns=()) -> tuple[list, dict]:
    """The host law on `names` over the forwards `x` labelled `y`, the part
    `offset` of each label fixed, and its report. Every forward stays in the
    fit: one slowed by an idle CPU is no outlier, and a MAD test cannot tell it
    from scatter. Raises on too few forwards or no hull; with the last of
    `names` in a fixed proportion to the one before on every forward, it gets
    zero. Each of `columns`, (feature, values), follows."""
    x = np.array(x, dtype=np.float64).reshape(-1, len(names))
    if len(x) <= (k := len(names) + len(columns)):
        raise ValueError(f"{len(x)} forwards for {k} coefficients")
    width = len(names)
    if np.linalg.matrix_rank(x[:, 1:]) < width - 1:
        width -= 1
        x[:, width] = 0.0
    wide = np.column_stack([x, *(v for _, v in columns)])
    names = [*names, *(f for f, _ in columns)]
    s = _solve(wide, np.array(y), names, np.array(offset), names[0], drops=False)
    hull = _hull(x[s["keep"]][:, 1:width], names[1:width])
    return s["coef"].tolist(), {
        "hull": hull,
        "samples": int(s["keep"].sum()),
        "dropped": int((~s["keep"]).sum()),
        **{k: s[k] for k in ("condition", "centre", "mad", "zero")},
    }


def _jit(samples, features=()) -> set[tuple]:
    """(run, rank, step) of each forward whose `W_launch` the fit of its kind
    drops above its law, a first-use JIT compile; on a decode, whose launch
    of a millisecond or so drops on host noise, only past `WARM_FACTOR` times
    its price. Each launch feature of `features` is a column of that fit."""
    jit = set()
    for kind in HOST_KINDS:
        members = [
            s
            for s in samples
            if host_kind(s.view) == kind and s.host.get("W_launch", 0) > 0
        ]
        if len(members) <= len(HOST_FEATURES) + len(features):
            continue
        x = np.array(
            [(*_host_x(m), *(m.launch[f] for f in features)) for m in members],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(x[:, 1:3]) < 2:
            x[:, 2] = 0.0
        y = np.array([m.host["W_launch"] for m in members])
        solved = _solve(x, y, [*HOST_FEATURES, *features])
        floor = 1 - WARM_FACTOR if kind == "decode" else 0.0
        jit |= {
            (m.provenance["run"], m.rank, m.step)
            for m, keep, r in zip(members, solved["keep"], solved["residual"])
            if not keep and r < floor
        }
    return jit


def _launch_columns(members, features) -> tuple[list[str], dict]:
    """The `features` host-law `members` (`_host_rows`) adopt as columns, in
    order, and the report's `launch_columns`: each round's held-out error per
    design tried (`_held_out_error`, each run held out, its p50 taken per
    launch path, its members of one value of every feature). One p50 over all
    of a run's members is its commonest path's, which a column a rarer path
    needs barely moves. A round adds the feature of least error while it is
    below the last design's. A member's run is its `_host_rows` cell: on a
    fit of one run, a span of its steps (`RUN_BLOCKS`), so a single record,
    whose launch-path mix moves with its load over time, adopts the columns
    its spans need too."""
    if not features:
        return [], {}
    x = np.array([r[2] for r in members], dtype=np.float64).reshape(-1, 3)
    if np.linalg.matrix_rank(x[:, 1:]) < 2:
        x[:, 2] = 0.0
    y = np.array([r[3] for r in members])
    cell = np.unique([repr(r[4]) for r in members], return_inverse=True)[1]
    paths = [[r[5][f] for f in features] for r in members]
    path = np.unique(paths, axis=0, return_inverse=True)[1]

    def error(names):
        values = [[r[5][f] for r in members] for f in names]
        design = np.column_stack([x, *values])
        return _held_out_error(_folds(design, y, cell, [*HOST_FEATURES, *names], path))

    adopted, rounds = [], [{"none": error([])}]
    while rounds[0]["none"] is not None and len(adopted) < len(features):
        last = min(rounds[-1].values())
        rounds.append({f: error([*adopted, f]) for f in features if f not in adopted})
        if not min(rounds[-1].values()) < last:
            break
        adopted.append(min(rounds[-1], key=rounds[-1].get))
    return adopted, {"launch_columns": {"adopted": adopted, "held_out": rounds}}


def _solve(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    offset=0.0,
    kept: str = "step",
    drops: bool = True,
) -> dict:
    """Non-negative least squares on relative error, MAD-dropped until settled
    unless `drops` is false; `offset` is a fixed part of each label's price,
    outside the coefficients: `_fit_form` on one row per step."""
    return _fit_form(
        x[:, None], np.ones((len(y), 1), dtype=bool), y, names, offset, kept, drops
    )


def _fit_form(
    x: np.ndarray,
    real: np.ndarray,
    y: np.ndarray,
    names: list[str],
    offset=0.0,
    kept: str = "step",
    drops: bool = True,
    protect: np.ndarray | None = None,
) -> dict:
    """One form fitted by max-affine regression: step `i`'s label `y[i]` is the
    max over its ranks' rows `x[i, r]`, those `real[i, r]` holds, of their
    price, plus `offset`, a fixed part of it outside the coefficients.

    `F(c) = sum_i ((max_r x_ir.c + offset_i - y_i) / s_i)^2` over the kept steps,
    each equation divided by its scale `s_i`, its label in the first round.
    The start is the non-negative least-squares solve of every rank's row, each
    equation divided by its label. Each pass solves on the row each step's
    current price picks, a Gauss-Newton step on `F`, and halves it until `F`
    falls. The passes end settled, the solve picking the rows it was solved
    on, or at a tie between ranks, when no step that moves some price by more
    than the square root of the float epsilon makes `F` fall, or at the
    limit, `FIT_PASSES` passes in all, keeping the vector they reached. Each
    feature that differs between some step's ranks is a start too, its axis
    scaled to the labels: the first round runs from each start, every step
    kept, and the lowest `F` goes on, a start at the limit included. Each kept
    step's scale then becomes its price under the vector reached and the passes
    rerun, until no kept step's price moves by that epsilon of its scale:
    iteratively reweighted least squares for a Gamma model with identity link,
    so the law prices its steps' mean. A step's residual is its price less its
    label, over its price. Steps whose residual is more than
    `MAD_K` scaled MADs from the median are then dropped and the passes rerun
    from there, until the kept set settles, for `MAD_PASSES` rounds, or until
    the passes reach the limit; with `drops` false none is dropped. A step
    `protect` marks is not dropped for running slower than its price, the
    median's side of the scatter, unless it ran `WARM_FACTOR` times its price
    or more: a class whose cost a one-sided tail carries is then priced at its
    mean, not its median. A kept step
    priced at no time, which no equation can be divided by, ends the
    reweighting, and a MAD round drops it.

    Returns the solve's figures: `pick`, each step's rank, at a tie the rank
    of larger features (`priced_rank`, prices folded as a run folds them);
    `passes`, its start, the passes over every round, the last stop and `F`;
    and `all_rows`, the centre and scaled MAD of the first start's residuals.
    """
    from scipy.optimize import nnls

    n, width, k = x.shape
    flat = x.reshape(-1, k)
    scale = np.array(y, dtype=np.float64)
    # An intercept, a column whose name starts with `kept`, is kept whatever
    # its variance; any other constant column is not.
    vary = [
        j for j in range(k) if names[j].startswith(kept) or np.ptp(x[real][:, j]) > 0
    ]
    tol = np.finfo(np.float64).eps ** 0.5
    steps = np.arange(n)

    def solve(rows, labels):
        norms = np.linalg.norm(rows[:, vary], axis=0)
        # A column with no non-zero kept row (a rung this pass dropped every
        # step of) sits out of this pass's solve, at coefficient 0.
        solved = norms > 0
        cols = rows[:, vary][:, solved] / norms[solved]
        coef = np.zeros(k)
        coef[np.array(vary)[solved]] = nnls(cols, labels)[0] / norms[solved]
        return coef, cols

    def price(c):
        return np.where(real, (flat @ c).reshape(n, width), -np.inf)

    def priced(c):
        return price(c).max(axis=1) + offset

    def relative(c):
        with np.errstate(divide="ignore"):
            return 1.0 - y / priced(c)

    def cost(c):
        return float((((priced(c) - y) / scale)[keep] ** 2).sum())

    keep = np.ones(n, dtype=bool)
    c, cols = solve(
        (x / scale[:, None, None])[real],
        np.broadcast_to(((y - offset) / scale)[:, None], real.shape)[real],
    )
    start = relative(c)

    def alternate(c, cols, on, passes):
        """Passes from `c` over the kept steps until settled, at a tie or at
        the limit."""
        while True:
            pick = price(c).argmax(axis=1)
            picked = np.zeros_like(real)
            picked[steps, pick] = True
            if on is not None and (on[keep] == picked[keep]).all():
                return c, cols, on, passes, "settled"
            if passes == FIT_PASSES:
                return c, cols, on, passes, "limit"
            passes += 1
            rows = x[steps, pick][keep] / scale[keep, None]
            full, solved = solve(rows, ((y - offset) / scale)[keep])
            if (price(full).argmax(axis=1)[keep] == pick[keep]).all():
                c, cols, on = full, solved, picked
            else:
                d, f, t = full - c, cost(c), 1.0
                moved = price(full - c) / scale[:, None]
                moves = np.abs(moved[real & keep[:, None]]).max()
                while cost(c + t * d) >= f and t * moves > tol:
                    t /= 2
                if t * moves <= tol:
                    return c, cols, on, passes, "tie"
                c, cols, on = c + t * d, solved, picked if t == 1 else None

    def reweight(c, cols, on, passes, stop):
        """Each kept step's scale set to its price under `c` and the passes
        rerun, until no kept step's price moves by `tol` of its scale, the
        passes reach the limit or `c` prices a kept step at no time."""
        while stop != "limit":
            got, now = priced(c)[keep], scale[keep]
            if (got <= 0).any() or (np.abs(got - now) <= tol * now).all():
                break
            scale[keep] = got
            c, cols, on, passes, stop = alternate(c, cols, None, passes)
        return c, cols, on, passes, stop

    # Each feature that differs between some step's ranks starts from its axis,
    # scaled by the median of each step's label over its largest value of it.
    starts = {"all_rows": (c, real)}
    top = np.where(real[..., None], x, -np.inf).max(axis=1)
    low = np.where(real[..., None], x, np.inf).min(axis=1)
    for j in np.flatnonzero((top > low).any(axis=0)):
        lit = top[:, j] > 0
        axis = np.zeros(k)
        axis[j] = np.median((y - offset)[lit] / top[lit, j])
        starts[names[j]] = (axis, None)
    # Each start's first round keeps every step; the lowest `F` wins.
    ends = {s: alternate(c0, cols, on, 0) for s, (c0, on) in starts.items()}
    first = min(ends, key=lambda s: cost(ends[s][0]))
    c, cols, on, passes, stop = ends[first]
    # Refitted until the kept set stops moving, so a step an outlier's pull
    # dropped on the first pass comes back once the outlier is gone.
    for rounds in range(1, MAD_PASSES + 1):
        if rounds > 1:
            c, cols, on, passes, stop = alternate(c, cols, on, passes)
        c, cols, on, passes, stop = reweight(c, cols, on, passes, stop)
        residual = relative(c)
        centre = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - centre))
        moved = np.abs(residual - centre) <= max(MAD_K * mad, MAD_FLOOR)
        if protect is not None:
            # A residual is (price - label) / price: a slower step's is lower.
            moved |= protect & (residual < centre) & (residual > 1.0 - WARM_FACTOR)
        if (
            not drops
            or stop == "limit"
            or (moved == keep).all()
            or rounds == MAD_PASSES
        ):
            break
        keep, on = moved, None
    pick = np.zeros(n, dtype=int)
    if width > 1:
        pick = [
            priced_rank(
                {
                    r: (fold_seconds(v * w for v, w in zip(x[i, r], c)), tuple(x[i, r]))
                    for r in np.flatnonzero(real[i])
                }
            )
            for i in steps
        ]
    return {
        "coef": c,
        "keep": keep,
        "residual": residual,
        "centre": float(centre),
        "mad": float(mad),
        "condition": float(np.linalg.cond(cols)),
        "zero": [names[j] for j in range(k) if j not in vary],
        "pick": np.asarray(pick),
        "passes": {"start": first, "count": passes, "stop": stop, "F": cost(c)},
        "all_rows": {
            "centre": float(np.median(start)),
            "mad": float(1.4826 * np.median(np.abs(start - np.median(start)))),
        },
    }


def _fit_vector(steps, vector: dict, routing) -> dict:
    """`_fit_steps` of `steps` on the features `vector` names under
    `coefficients`, as `_counts` counts them."""
    return _fit_steps(
        steps,
        list(vector["coefficients"]),
        lambda r: _counts(vector, routing, r[2], *r[5:]),
    )


def _folds(design, y, cell, names, path=None) -> dict:
    """Per record of `cell` beside which another holds rows, (its rows, the p50
    error on them of the host-law solve of `design` on the other records'
    rows, which drops none); given
    `path`, a value per row, the same per record and value, keyed by both."""
    out = {}
    for c in sorted(set(cell.tolist())):
        other = cell != c
        if other.any():
            coef = _solve(design[other], y[other], names, drops=False)["coef"]
            error = design[~other] @ coef / y[~other] - 1.0
            if path is None:
                out[c] = (int((~other).sum()), float(np.median(error)))
                continue
            held = path[~other]
            for p in sorted(set(held.tolist())):
                mine = error[held == p]
                out[c, p] = (len(mine), float(np.median(mine)))
    return out


def _max_affine_errors(steps, vector: dict, routing, cell=None) -> dict:
    """Per cell beside which another holds steps, (the positions of its steps
    in `steps`, the error on each, priced as the max over its ranks, of
    `_fit_vector` of `vector` on the other cells' steps). `cell` is each
    step's cell, its record when None."""
    if cell is None:
        cell = np.array([s[0][0] for s in steps])
    out = {}
    for c in sorted(set(cell.tolist())):
        other = cell != c
        if other.any():
            fold = [s for s, o in zip(steps, other) if o]
            coef = _fit_vector(fold, vector, routing)["coef"]
            at = np.flatnonzero(~other)
            error = np.array(
                [
                    max(
                        np.dot(coef, _counts(vector, routing, r[2], *r[5:]))
                        for r in steps[i]
                    )
                    / steps[i][0][3]
                    - 1.0
                    for i in at
                ]
            )
            out[c] = (at, error)
    return out


def _p50s(errors: dict, band=None) -> dict:
    """Per held-out cell of `errors`, `_max_affine_errors`', (its steps, the
    p50 error on them); given `band`, a value per step, the same per cell and
    value, keyed by both, as `_folds` takes a path."""
    out = {}
    for c, (at, error) in errors.items():
        if band is None:
            out[c] = (len(error), float(np.median(error)))
            continue
        held = band[at]
        for b in sorted(set(held.tolist())):
            mine = error[held == b]
            out[c, b] = (len(mine), float(np.median(mine)))
    return out


def _max_affine_folds(steps, vector: dict, routing, cell=None) -> dict:
    """Per cell beside which another holds steps, (its steps, the p50 error
    on them, each priced as the max over its ranks, of `_fit_vector` of
    `vector` on the other cells' steps). `cell` is each step's cell, its
    record when None."""
    return _p50s(_max_affine_errors(steps, vector, routing, cell))


def _token_bands(steps) -> tuple[np.ndarray, list[float]]:
    """Each of `steps`' MoE-token band, the `MOE_BANDS` quantile of the steps'
    real MoE tokens it lies in, counted from 0, and the quantiles' edges."""
    tokens = np.array([s[0][6] for s in steps], dtype=np.float64)
    edges = np.quantile(tokens, np.arange(1, MOE_BANDS) / MOE_BANDS)
    return np.searchsorted(edges, tokens, side="right"), edges.tolist()


def _held_out_error(folds: dict) -> float | None:
    """The held-out error of `folds`, `_folds`' or `_max_affine_folds`': the
    mean of each held-out p50 error's magnitude, a record's or a record's
    path's, weighted by its rows or steps; None with no record held out."""
    rows = sum(n for n, _ in folds.values())
    return sum(n * abs(p) for n, p in folds.values()) / rows if rows else None


def _covered(steps, cell) -> np.ndarray:
    """Whether each of `steps`, each its ranks' `fit` rows, lies inside the
    convex hull (`_hull`, as a law's vector bounds the steps it validates) of
    the steps of every other `cell` over `COVER_AXES`: the largest context any
    of its ranks holds (Sum N_KV, which a context slope multiplies), its
    group's real MoE tokens and its group's dummy forwards (which, with the
    tokens, set the experts a `moe_experts` column counts). Every candidate
    form prices a step by these and the rung's constants, so a fold's fit
    interpolates at a step inside that hull. Other cells no qhull hull holds
    cover no step."""
    from scipy.spatial import QhullError

    from atom.compass.backends.hull import Hull

    point = np.array(
        [(max(int(r[2][1]) for r in s), int(s[0][6]), int(s[0][7])) for s in steps]
    )
    out = np.zeros(len(steps), dtype=bool)
    for c in set(cell.tolist()):
        mine, other = cell == c, cell != c
        if not other.any():
            continue
        try:
            hull = Hull("fold", COVER_AXES, _hull(point[other], COVER_AXES))
        except (QhullError, ValueError):
            continue
        out[mine] = [hull.contains(tuple(p)) for p in point[mine]]
    return out


def _moe_column(
    steps,
    routing,
    base=DECODE,
    select: str = "held_out",
    both: bool = False,
    folds: str = MOE_FOLDS[0],
) -> tuple:
    """The vector decode rung `steps` adopts with a MoE column, or None, the
    figures the rule read, as the vector records them under `fit.moe_column`,
    and the held-out folds of the form adopted, a column or none (None under
    ``insample``). Each form adds its column to `base`, the rung's columns.

    `steps` are the rung's steps, each its ranks' `fit` rows. The forms are no
    column and each of `MOE_COLUMNS` over the steps' real MoE tokens,
    `moe_experts` under `routing` and with each step's group dummies as one
    token (`moe_column`). Each form is fitted with each record held out of the
    rung's steps (`_max_affine_errors`), its held-out error `_held_out_error`'s
    over each held-out record's p50 error per MoE-token band (`_token_bands`:
    `MOE_BANDS` quantiles of the rung's real MoE tokens), as a host law's
    columns are chosen per launch path (`_launch_columns`). One p50 over all
    of a record's steps is its commonest token count's, which a column its
    fullest steps need barely moves (#3720 rung 16 of cell B's D, 384 experts
    top-6: `moe_experts`, which nearly every expert reached past ~200 group
    tokens holds flat, held out nearest by record, 0.25% against
    `moe_tokens`' 0.35%, yet priced the rung's fullest steps 2-5% low out of
    sample, the load the requests making TPOT p90 decode at, where
    `moe_tokens`' error stays level across the rows). With `both` the forms
    add `MOE_BOTH`, the rung's columns with both `MOE_COLUMNS`: the experts'
    weights each step reads, which `moe_experts` counts and which stop growing
    once nearly every expert is reached, and the rows the gather and the
    experts' GEMMs carry, which `moe_tokens` counts and which keep growing
    (#752 cell A v2, 384 experts top-6 over 4 data-parallel ranks: rung 24
    adopted `moe_experts`, whose fit dropped as outliers all 21 calibration
    steps of 400 group tokens or more, 10% slower than it, and priced the c=128
    target's rung-24 steps, all four ranks busy at 300-560 tokens, 8.6% low;
    both columns, VIF 3, kept those steps and priced them 0.2% low). The form
    is a candidate only where both its MoE columns' variance inflation factor
    over the steps its fit on all of them keeps is `MOE_BOTH_VIF` or less,
    stated under `both_vif` either way. Each record's own p50
    stays under `folds`, the p50s graded under `bands`, keyed "record:band",
    and the bands' edges under `token_bands`.
    A rung of one record holds out each group composition instead, the steps
    with as many dummy forwards beside them (`BatchView.group_dummies`): what a
    run at another concurrency changes at a rung is how many of its ranks are
    busy, which the column states and the context column would otherwise take
    up (#3720 c=1: a c=128 record's rung 1, its steps at 1-4 busy ranks, fitted
    context at 25x its slope). The form of smallest held-out error is picked,
    no column on a tie, and a picked column is adopted, returned as the vector
    `_counts` reads. A rung of one record and one composition has no held-out
    error and adopts none.

    Under `folds` ``covered`` (`MOE_FOLDS_ENV`) a rung of several records
    holds out each record's compositions, a cell keyed "record:dummies", each
    priced by a fit on every other cell, its own record's other compositions
    too, and grades each form only on the held-out steps the other cells'
    steps cover (`_covered`): inside their convex hull over `COVER_AXES`, the
    largest rank context, the group's MoE tokens and its dummies. A fold's
    fit says how a column prices steps like the ones it was fitted on; graded
    on steps past them it says how far the fit extrapolates, and weighted by
    steps the largest cell beyond the others' reach decides the rung. Held out
    whole, a record whose MoE tokens the others do not reach grades each form
    on that extrapolation (#752 cell A v2, its calibration record and addendum
    fitted together: at rung 48 the calibration's long-row steps at ~520 group
    tokens and the addendum's at 264 and 693; `moe_experts` won, 4.42% against
    `MOE_BOTH`'s 4.63%, though its MAD was 5.1% against 1.9%, and priced the
    c=128 target's rung-48 steps 4.7% low; held out by composition `MOE_BOTH`
    won and priced them 0.7% high), and so does one whose context the others
    do not reach (#752 cell B's D, cal and hi records: rung 24's cal steps at
    0.1-0.27M tokens of context on the fullest rank, hi's at 3.5-5.5M, every
    step of both with all four ranks busy, one composition each; the fold
    fitted on cal's 793 steps priced hi's 11013 with the two MoE columns 115%
    high, with `moe_experts` 13% low, so the rule adopted `moe_experts` with a
    zero intercept at an in-sample MAD of 2.0%, against 0.63% for both
    columns, whose fold fitted on hi priced cal 0.4% high, `moe_experts`' 5.7%
    low). Each held-out cell weighs its covered steps alone, the p50 per band
    over them, and one with none drops out; where no cell has one the rung
    adopts as under ``insample``, its figures stating ``fallback:
    uncovered``. The vector states, per held-out cell, its steps and how many
    are covered under `fold_cover`. A hull, not the box of the ranges: a
    record whose steps lie along a line in these quantities spans a box it
    does not cover (cell A v2's calibration record and addendum, rung 8: 299
    of the calibration record's 475 steps inside the addendum's box, 53 inside
    its hull; graded on the 299, the addendum's fit with `moe_tokens` priced
    them 73% high). Its folds are no record's, and none is returned for
    `_rowmod`. A rung of one record is held out by composition, uncovered
    steps graded too, either way: how a column carries a step across
    compositions is what that hold-out tests.

    Under `select` ``insample`` (`MOE_SELECT_ENV`) the rung adopts one of
    `MOE_COLUMNS` whatever its records: the one whose fit on all its steps has
    the smaller scaled MAD of its relative residuals, the first on a tie, the
    figures stating each under `in_sample`. Under ``fallback`` a rung picks by
    its held-out error where it has one, and where it has none, one record of
    one composition, as under ``insample``, its figures stating `fallback`:
    without a MoE column such a rung's context slope takes up its group's MoE
    work, which grows with the rows it decodes (#3720 cell A's e3v2 record:
    rung 16 fitted context at 3x rung 8's and 24's slope, its MAD 1.99% against
    `moe_tokens`' 1.22%, and priced as the max over its ranks it charged a
    target's step its busiest rank's context at that slope, its c=64 steps
    3.7% high).
    """
    forms = {"none": {"coefficients": list(base)}}
    forms |= {f: {"coefficients": [*base, f]} for f in MOE_COLUMNS}
    vif = None
    if both:
        wide = {"coefficients": [*base, *MOE_COLUMNS]}
        solved = _fit_vector(steps, wide, routing)
        vif = _kept_vif(solved, wide, routing, list(MOE_COLUMNS))
        if len(vif) == len(MOE_COLUMNS) and max(vif.values()) <= MOE_BOTH_VIF:
            forms[MOE_BOTH] = wide
    columns = [f for f in forms if f != "none"]
    if select not in MOE_SELECTS:
        raise ValueError(f"MoE column rule {select!r} is not one of {MOE_SELECTS}")
    if folds not in MOE_FOLDS:
        raise ValueError(f"MoE folds {folds!r} are not one of {MOE_FOLDS}")
    cell, by = np.array([s[0][0] for s in steps]), "record"
    if len(set(cell.tolist())) == 1:
        cell, by = np.array([s[0][7] for s in steps]), "group_dummies"
    elif folds == "covered":
        cell = np.array([f"{s[0][0]}:{s[0][7]}" for s in steps])
        by = "record:group_dummies"
    fallback = select == "fallback" and len(set(cell.tolist())) == 1
    if select == "insample" or fallback:
        fits = {f: _fit_vector(steps, forms[f], routing) for f in columns}
        error = {f: fits[f]["mad"] for f in columns}
        picked = min(error, key=error.get)
        figures = {
            "picked": picked,
            "adopted": picked,
            "selected_by": "insample",
            **({"fallback": by} if fallback else {}),
            **({"both_vif": vif} if vif is not None else {}),
            "in_sample": {
                f: {"mad": fits[f]["mad"], "F": fits[f]["passes"]["F"]} for f in columns
            },
        }
        return forms[picked], figures, None
    covered = cover = None
    if by == "record:group_dummies":
        covered = _covered(steps, cell)
        cover = {
            str(c): [int((cell == c).sum()), int((covered & (cell == c)).sum())]
            for c in sorted(set(cell.tolist()))
        }
        if not covered.any():
            fits = {f: _fit_vector(steps, forms[f], routing) for f in columns}
            error = {f: fits[f]["mad"] for f in columns}
            picked = min(error, key=error.get)
            figures = {
                "picked": picked,
                "adopted": picked,
                "selected_by": "insample",
                "fallback": "uncovered",
                "fold_cover": cover,
                **({"both_vif": vif} if vif is not None else {}),
                "in_sample": {
                    f: {"mad": fits[f]["mad"], "F": fits[f]["passes"]["F"]}
                    for f in columns
                },
            }
            return forms[picked], figures, None
    band, edges = _token_bands(steps)
    held, banded, error, graded = {}, {}, {}, {}
    for form, vector in forms.items():
        errors = _max_affine_errors(steps, vector, routing, cell)
        held[form], banded[form] = _p50s(errors), _p50s(errors, band)
        if covered is None:
            error[form] = _held_out_error(banded[form])
            continue
        kept = {
            c: (at[covered[at]], e[covered[at]])
            for c, (at, e) in errors.items()
            if covered[at].any()
        }
        graded[form] = _p50s(kept, band)
        error[form] = _held_out_error(graded[form])
    picked = min(error, key=error.get) if held["none"] else None
    adopted = None if picked in (None, "none") else picked
    figures = {
        "picked": picked,
        "adopted": adopted,
        "held_out_by": by,
        **({"fold_cover": cover} if cover is not None else {}),
        **({"both_vif": vif} if vif is not None else {}),
        "token_bands": edges,
        "held_out": {
            form: {
                "error": error[form],
                "folds": {str(c): list(f) for c, f in held[form].items()},
                "bands": {f"{c}:{b}": list(f) for (c, b), f in banded[form].items()},
                **(
                    {
                        "covered_bands": {
                            f"{c}:{b}": list(f) for (c, b), f in graded[form].items()
                        }
                    }
                    if cover is not None
                    else {}
                ),
            }
            for form in forms
        },
    }
    vector = None if adopted is None else forms[adopted]
    return vector, figures, held[adopted or "none"] if by == "record" else None


def _rowmod(
    steps, rung: int, vector: dict, base: dict, seed: int, routing, folds, vifs=None
):
    """The `rowmod(k)` column decode rung `rung` adopts, or None.

    `steps` are the rung's steps, each its ranks' `fit` rows, `vector` the
    rung's form without the column, `base` its `_fit_vector` and `folds` its
    `_max_affine_folds`, fitted here when None.
    `k` runs over the powers of two up to the rung. A `k` qualifies when its
    fit cuts `base`'s scaled MAD by `ROWMOD_GAIN`, the mean MAD of the same
    fit over `ROWMOD_SHUFFLES` shuffles of each rank's rows, by `seed`, does
    not, and no record (a cell) priced by a fit on the others
    (`_max_affine_folds`) has a p50 error further from zero than without the
    column. Every cell
    needs `ROWMOD_CELL_STEPS` steps at the rung, and a rung needs two cells.
    The qualifying `k` of smallest MAD is adopted: returned with its fit and
    the evidence the law records. Given `vifs`, a dict (`ROWMOD_VIF_ENV`), a
    `k` whose fit cuts the MAD qualifies only where the column's variance
    inflation factor over the steps that fit keeps (`_kept_vif`) is
    `ROWMOD_VIF` or less, a column the other columns hold being no evidence
    of order; the factor is stated in `vifs` under `k`, adopted or not.
    """
    cell = np.array([s[0][0] for s in steps])
    cells = sorted(set(cell.tolist()))
    if len(cells) < 2 or min((cell == c).sum() for c in cells) < ROWMOD_CELL_STEPS:
        return None

    def wide(k):
        return {"coefficients": [*vector["coefficients"], "rowmod"], "rowmod": k}

    def shuffled_mad(k):
        # The same shuffles at every `k`.
        rng, mads = np.random.default_rng(seed), []
        for _ in range(ROWMOD_SHUFFLES):
            moved = [
                [(*r[:5], rng.permutation(r[5]), *r[6:]) for r in s] for s in steps
            ]
            mads.append(_fit_vector(moved, wide(k), routing)["mad"])
        return float(np.mean(mads))

    def p50(folds):
        return [p for _, p in folds.values()]

    if folds is None:
        folds = _max_affine_folds(steps, vector, routing)
    without, best, k = p50(folds), None, 2
    target = (1.0 - ROWMOD_GAIN) * base["mad"]
    while k <= rung:
        solved = _fit_vector(steps, wide(k), routing)
        cuts = solved["mad"] < min(target, best[1]["mad"] if best else target)
        if cuts and vifs is not None:
            vif = _kept_vif(solved, wide(k), routing, ["rowmod"])
            vifs[str(k)] = vif.get("rowmod", math.inf)
            cuts = vifs[str(k)] <= ROWMOD_VIF
        if cuts:
            shuffled = shuffled_mad(k)
            held = p50(_max_affine_folds(steps, wide(k), routing))
            if shuffled >= target and all(
                abs(w) <= abs(o) for w, o in zip(held, without)
            ):
                evidence = {
                    "mad_without": base["mad"],
                    "shuffled_mad": shuffled,
                    "held_out_p50": {
                        str(c): [o, w] for c, o, w in zip(cells, without, held)
                    },
                }
                best = (k, solved, evidence)
        k *= 2
    return best


def _hull(points: np.ndarray, names: tuple[str, ...], affine: bool = True) -> dict:
    """`StepLinearLaw`'s hull over the non-intercept features of the fitted steps.
    qhull's precision checks depend on the order the points come in: where it
    refuses them, it hulls their distinct points in sorted order, the same hull,
    and where it refuses those too and they span every free feature, it hulls
    them joggled (`QJ`), each facet then moved out to hold every point. A set
    that spans none still raises. Past `HULL_EXACT_FREE` free features it is
    no qhull hull of the set but `_pair_hull`'s, bounded in facets, which
    holds every point and may hold more. A feature affine in an earlier one on
    the set is held to it by two opposed facets, as a constant one is by
    `equal`, unless `affine` is false: then qhull refuses that flat set."""
    from scipy.spatial import ConvexHull, QhullError

    equal = {
        names[j]: int(points[0, j])
        for j in range(points.shape[1])
        if np.ptp(points[:, j]) == 0
    }
    vary = [j for j in range(points.shape[1]) if names[j] not in equal]
    x = points[:, vary]
    lo, hi = x.min(axis=0), x.max(axis=0)
    if len(vary) == 1:
        facets = [[-1.0, float(lo[0])], [1.0, -float(hi[0])]]
    elif vary:
        # Hulled in the unit box, where qhull's tolerances suit every feature.
        unit = (x - lo) / (hi - lo)
        mirror = (hi - x) / (hi - lo)  # an affine feature equals another's unit or this
        tied = {}  # a feature: the earlier one it is held to, and the sign
        for i, j in itertools.combinations(range(len(vary)), 2) if affine else ():
            for sign, other in ((1.0, unit), (-1.0, mirror)):
                if j not in tied and np.array_equal(unit[:, j], other[:, i]):
                    tied[j] = (i, sign)
        free = [j for j in range(len(vary)) if j not in tied]
        if len(free) == 1:
            hulled = np.array([[-1.0, 0.0], [1.0, -1.0]])
        elif len(free) > HULL_EXACT_FREE:
            hulled = _pair_hull(unit[:, free])
        else:
            try:
                hulled = ConvexHull(unit[:, free]).equations
            except QhullError:
                distinct = np.unique(unit[:, free], axis=0)
                try:
                    hulled = ConvexHull(distinct).equations
                except QhullError:
                    # #3720: real_pw_c128 + real_pw_c1's prefill points, 8
                    # features, failed both orders on a precision check.
                    spans = distinct - distinct.mean(axis=0)
                    if np.linalg.matrix_rank(spans) < len(free):
                        raise
                    hulled = ConvexHull(distinct, qhull_options="QJ").equations
                    # A joggled facet may cut a point by the joggle: each moves
                    # out by its largest cut, so the hull holds every point.
                    cut = (distinct @ hulled[:, :-1].T + hulled[:, -1]).max(axis=0)
                    hulled[:, -1] -= np.maximum(cut, 0.0)
        eq = np.zeros((len(hulled), len(vary) + 1))
        eq[:, free], eq[:, -1] = hulled[:, :-1], hulled[:, -1]
        for j, (i, sign) in tied.items():  # u_j = u_i, or u_j = 1 - u_i
            tie = np.zeros(len(vary) + 1)
            tie[[j, i, -1]] = np.array([1.0, -sign, min(sign, 0.0)]) / 2**0.5
            eq = np.vstack([eq, tie, -tie])
        normals = eq[:, :-1] / (hi - lo)
        facets = np.column_stack([normals, eq[:, -1] - normals @ lo]).tolist()
    else:
        facets = []
    return {"equal": equal, "facets": facets}


def _pair_hull(unit: np.ndarray) -> np.ndarray:
    """The facets, as qhull's `equations`, of the intersection of the unit box
    and each pair of `unit`'s columns' 2-D hull: a convex set that holds every
    point of `unit`, points in the unit box, and is no smaller than their hull,
    with at most the box's faces and the pairs' hull edges. A pair whose
    distinct points qhull refuses (on a line, or one point) adds no facet: the
    box bounds it. Each facet moves out by its largest cut of a point, so the
    set holds every point under rounding too."""
    from scipy.spatial import ConvexHull, QhullError

    d = unit.shape[1]
    box = np.zeros((2 * d, d + 1))
    box[np.arange(d), np.arange(d)], box[d + np.arange(d), np.arange(d)] = -1.0, 1.0
    box[d:, -1] = -1.0
    rows = [box]
    for i, j in itertools.combinations(range(d), 2):
        pair = np.unique(unit[:, [i, j]], axis=0)
        try:
            edges = ConvexHull(pair).equations
        except QhullError:
            continue
        eq = np.zeros((len(edges), d + 1))
        eq[:, [i, j, d]] = edges
        rows.append(eq)
    hulled = np.vstack(rows)
    cut = (unit @ hulled[:, :-1].T + hulled[:, -1]).max(axis=0)
    hulled[:, -1] -= np.maximum(cut, 0.0)
    return hulled


def _groups(
    run,
    form: str,
    draft: bool = False,
    steps=None,
    serial: bool = False,
    routing: tuple[int, int] | None = None,
    dummies: list[frozenset] | None = None,
    buckets: tuple[int, ...] = (),
    eager: bool = False,
    cold: bool = False,
    share: bool = False,
    kda: bool = False,
    cols=None,
) -> tuple[list[dict[int, tuple]], dict, list[float]]:
    """One run's steps, each {rank: its `_features` row} labelled with the
    smallest rank label; what was cut; and each step's rank label spread. A
    rank that ran a prefill step as a dummy forward holds the row of its
    dummy (`_features`' `dummy`), labelled the same, since the backend prices
    it as a rank of the group and the step waits for it; one that ran a decode
    step or, under `draft`, any step so holds none. With `steps`, only the
    steps numbered in it; `serial`, `routing`, `buckets`, `eager`, `cold`,
    `share`, `kda` and `cols` as `_features` reads them;
    `dummies`, each rank's steps run as dummy forwards, read from its samples
    when None: a rank that gave no sample needs them given."""
    if dummies is None:
        dummies = [rank[0].dummies for rank in run]
    run = [[s for s in rank if steps is None or s.step in steps] for rank in run]
    got = [
        _features(
            rank,
            form,
            len(run),
            draft,
            serial,
            routing,
            buckets,
            eager,
            cold,
            share,
            kda,
            cols,
        )
        for rank in run
    ]
    cut = {
        k: sum(c.get(k, 0) for _, c in got)
        for k in dict.fromkeys(k for _, c in got for k in c)
    }
    if len(run) == 1:
        return [{0: r[:8]} for r in got[0][0]], cut, []
    ranks = [{r[3]: r for r in rows} for rows, _ in got]
    groups, spread, cut["ranks"] = [], [], 0
    for step in sorted(set().union(*ranks)):
        group = {i: rank[step] for i, rank in enumerate(ranks) if step in rank}
        idle = [i for i, d in enumerate(dummies) if step in d]
        if (
            len(group) + len(idle) < len(ranks)
            or len({r[0] for r in group.values()}) > 1
        ):
            cut["ranks"] += 1
            continue
        labels = [r[2] for r in group.values()]
        spread.append(max(labels) / min(labels) - 1.0)
        rows = {i: (*r[:2], min(labels), *r[3:8]) for i, r in group.items()}
        peer = next(iter(group.values()))
        if peer[8] is not None:
            rows |= {i: (peer[0], peer[8], min(labels), *peer[3:8]) for i in idle}
        groups.append(dict(sorted(rows.items())))
    return groups, cut, spread


def _fit_steps(
    steps: list[list[tuple]], names: list[str], design, protect=None
) -> dict:
    """`_fit_form` of `steps`, each its ranks' `fit` rows, on `design(row)`, the
    form's features of a row, with `protect`, a step's mark, as `_fit_form`
    takes it; with `rows`, each step's row at the rank picked.
    A rank that ran the step as a dummy forward holds its dummy's row, or none
    (`_groups`)."""
    width = max(map(len, steps))
    x = np.zeros((len(steps), width, len(names)))
    real = np.zeros((len(steps), width), dtype=bool)
    for i, step in enumerate(steps):
        for r, row in enumerate(step):
            x[i, r], real[i, r] = design(row), True
    marks = None if protect is None else np.array([protect(s) for s in steps])
    y = np.array([s[0][3] for s in steps])
    solved = _fit_form(x, real, y, names, protect=marks)
    solved["rows"] = [s[p] for s, p in zip(steps, solved["pick"])]
    return solved


def _serial(samples, options) -> bool:
    """Whether the prefill vector adds `serial_moe_rows`: as `options` states,
    else when the group split some data-parallel prefill step of `samples`;
    with none split the column would repeat `moe_rows`."""
    split = any(s.view.prefill and s.view.tbo_tokens for s in samples)
    return (options or {}).get("serial", _options(options)[1] > 1 and split)


def _options(options) -> tuple[str, int]:
    options = {"form": "base", "dp": 1, **(options or {})}
    if options["form"] not in FORMS:
        raise ValueError(f"form {options['form']!r} is not one of {FORMS}")
    return options["form"], options["dp"]


def _routing(options) -> tuple[int, int] | None:
    """`options`' `moe_routing`, (routed experts, top-k), or None; refused under
    `shared`, whose decode solve has no MoE column."""
    routing = (options or {}).get("moe_routing")
    if routing and _options(options)[0] == "shared":
        raise ValueError("the shared form fits no MoE column")
    return tuple(routing) if routing else None


def _buckets(options) -> tuple[int, ...]:
    """`options`' `buckets`, the records' prefill token-graph buckets, ascending."""
    return tuple(sorted((options or {}).get("buckets") or ()))


def _law_routing(law: dict) -> tuple[int, int] | None:
    """`law`'s `moe_routing` as (experts, top-k), or None."""
    routing = law.get("moe_routing")
    return routing and (routing["experts"], routing["top_k"])


def fit(samples, options=None, seed: int = 0, draft: bool = False) -> tuple[dict, dict]:
    """The law over `samples`, as the mapping `StepLinearLaw` reads, and its report.

    `options` takes `form` (`base` by default), `dp`, the ranks each run
    holds (1 by default), `moe_routing`, the model's (routed experts,
    top-k), under which each decode rung but the draft law's may adopt a MoE
    column (`_moe_column`), `draft_moe_routing`, the drafter's, under which
    above `dp` 1 the draft law's may (`_draft_options`), and `buckets`, the records' prefill token-graph
    buckets, under which the prefill vector adds `padding`, `cold`, under
    which above `dp` 1 it adds `cold_peer_rows` (`_cold`), and `kda`, under
    which it adds `KDA`'s columns and states `kda_chunk` (`_kda`); `seed` fixes the
    `rowmod` shuffles, and `rowmod_vif` gates each `rowmod` choice on the
    column's variance inflation factor (`_rowmod`), and `moe_folds`, one of
    `MOE_FOLDS`, sets how each decode rung of several records, the draft
    law's too, holds them out to pick its MoE column (`_moe_column`); `draft`, or a sample with
    a draft label, adds `seqs` and `out` to the prefill vector. `peer_share`,
    false by default, adds `PEER_SHARE` to it above `dp` 1 (`_share`), and
    `keep_peer_tail`, false by default and only with `peer_share`, keeps in
    its solve the slower steps a
    `PEER_SHARE` column marks on some rank (`_fit_form`'s `protect`); its
    `fit` then counts under `peer_tail` those kept past the MAD band.
    """
    from scipy.spatial import QhullError

    form, dp = _options(options)
    routing = _routing(options)
    serial = _serial(samples, options)
    buckets = _buckets(options)
    eager = not draft and _eager_dummies(samples, options)
    cold = not draft and _cold(options)
    share = not draft and _share(options)
    keep_tail = not draft and _keep_peer_tail(options)
    kda = not draft and _kda(options)
    cols = _columns(options, draft)
    select = (not draft and (options or {}).get("moe_select")) or MOE_SELECTS[0]
    both = not draft and bool((options or {}).get("moe_both"))
    gated = bool((options or {}).get("rowmod_vif"))
    moe_folds = (options or {}).get("moe_folds") or MOE_FOLDS[0]
    peers = (options or {}).get("draft_peer_tokens" if draft else "peer_tokens")
    if peers and dp == 1:
        raise ValueError(
            "peer_tokens counts a data-parallel group's peers' tokens, and these "
            "records run one rank"
        )
    if peers and form == "shared":
        raise ValueError("the shared form fits no peer_tokens column")
    samples = [
        (
            s
            if s.draft_kv_s is None
            else dataclasses.replace(s, device_s=s.device_s - s.draft_kv_s)
        )
        for s in samples
    ]
    runs = _runs(samples, dp)
    groups, cut, spread = [], {"warm": 0, "warm_s": 0.0}, []
    for i, run in enumerate(runs.values()):
        got, c, s = _groups(
            run,
            form,
            draft,
            serial=serial,
            routing=routing,
            buckets=buckets,
            eager=eager,
            cold=cold,
            share=share,
            kda=kda,
            cols=cols,
        )
        groups += [[(i, *r) for r in g.values()] for g in got]
        spread += s
        for k, v in c.items():
            cut[k] = cut.get(k, 0) + v
    prefill = list(HISTORY if form == "history" else PREFILL)
    prefill += ["cached_seqs"] * (form == "cached_seqs")
    prefill += ["short_hinge"] * bool(cols["short_hinge"]) + ["moe_rows"] * (dp > 1)
    prefill += ["serial_moe_rows"] * serial + ["cold_peer_rows"] * cold
    gathered = _gathered(cols)
    prefill += gathered
    prefill += ["seqs", "out"] * (draft or any(s.draft_s is not None for s in samples))
    prefill += list(PEER_SHARE) * share
    prefill += list(KDA) * kda
    prefill += ["padding"] * bool(buckets)
    prefill += [f for f in GRAPH_COLUMNS if cols[f]]
    prefill += list(EAGER_DUMMY) * eager
    law = {
        "provenance": f"step_linear_fit {form} on {len(runs) * dp} real records "
        f"({', '.join(runs)})",
        "form": form,
        "cut": cut,
        "runs": list(runs),
        "warm_steps": [[] for _ in runs],
        "unfitted": {},
        "decode": {},
    }
    if routing:
        law["moe_routing"] = {"experts": routing[0], "top_k": routing[1]}
    if spread:
        law["spread"] = np.percentile(spread, (50, 90)).tolist()
    keys = sorted(
        {g[0][1] for g in groups},
        key=lambda k: (k != "prefill", k != EAGER, k.isdigit() and int(k)),
    )
    rungs = [k for k in keys if k != "prefill"]
    # Each solve: its column names, its rows, and their design matrix.
    solves = [("prefill", prefill)]
    if form == "shared" and rungs:
        solves.append(
            (None, ["context", "graph_padding"] + [f"step@{k}" for k in rungs])
        )
    else:
        base = list(DECODE) + ["row_padding"] * cols["row_padding"]
        solves += [(k, base) for k in rungs]
    for key, names in solves:
        steps = [
            g
            for g in groups
            if g[0][1] == key or (key is None and g[0][1] != "prefill")
        ]
        if not steps:
            law["unfitted"][key] = (
                "no prefill step carries a trace run's busy time"
                if key == "prefill"
                else f"0 steps for {len(names)} coefficients"
            )
            continue
        if key is None:
            solved = _fit_steps(
                steps, names, lambda r: (*r[2][1:], *(r[1] == k for k in rungs))
            )
        elif key == "prefill" and keep_tail:
            marks = [names.index(f) for f in PEER_SHARE]
            solved = _fit_steps(
                steps,
                names,
                lambda r: r[2],
                lambda step, marks=marks: any(r[2][j] for r in step for j in marks),
            )
        else:
            solved = _fit_steps(steps, names, lambda r: r[2])
        rowmod_k = figures = folds = None
        rowmod_vif = {} if gated else None
        if key and key.isdigit():
            vector = {"coefficients": names}
            if routing:
                chosen, figures, folds = _moe_column(
                    steps, routing, names, select, both, moe_folds
                )
                if chosen:
                    vector, solved = chosen, _fit_vector(steps, chosen, routing)
            if _peer_rung(steps, int(key), vector, routing, peers):
                vector = {"coefficients": [*vector["coefficients"], "peer_tokens"]}
                solved, folds = _fit_vector(steps, vector, routing), None
            chosen = _rowmod(
                steps, int(key), vector, solved, seed, routing, folds, vifs=rowmod_vif
            )
            if chosen:
                rowmod_k, solved, evidence = chosen
            names = [*vector["coefficients"], *["rowmod"] * bool(rowmod_k)]
        members = solved["rows"]
        slow = ~solved["keep"] & (solved["residual"] < 0)
        # A residual r = (price - label) / price, so price = label / (1 - r).
        label = np.array([r[3] for r in members], dtype=np.float64)
        past = label - label / np.maximum(1 - solved["residual"], 1e-12)
        once = (solved["residual"] < 1 - WARM_FACTOR) | (past > PAUSE_S)
        warm = np.zeros(len(members), dtype=bool)
        for rec in range(len(runs)):
            leading = True
            for j in (j for j, r in enumerate(members) if r[0] == rec):
                leading = leading and slow[j]
                if not (leading or slow[j] and once[j]):
                    continue
                warm[j] = True
                law["warm_steps"][rec].append(members[j][4])
                cut["warm"] += 1
                cut["warm_s"] += members[j][3]
        for k in rungs if key is None else [key]:
            mine = np.array([r[1] == k for r in members])
            coef = solved["coef"]
            if key is None:
                coef = (coef[2 + rungs.index(k)], *coef[:2])
            features = names if key else DECODE
            coefficients = dict(zip(features, map(float, coef)))
            priced = {"coefficients": coefficients}
            if rowmod_k:
                priced["rowmod"] = rowmod_k
            kept = mine & solved["keep"]
            try:
                if kept.sum() <= len(features):
                    raise ValueError(
                        f"{int(kept.sum())} steps for {len(features)} coefficients"
                    )
                points = [
                    _counts(priced, routing, r[2], *r[5:])
                    for r, keep in zip(members, kept)
                    if keep
                ]
                axes = list(hull_axes(features))
                hull = _hull(
                    np.array(points, dtype=np.float64)[:, axes],
                    tuple(features[i] for i in axes),
                )
            except (ValueError, QhullError) as e:
                law["unfitted"][k] = str(e).splitlines()[0]
                continue
            vector = {
                "coefficients": coefficients,
                "hull": hull,
                "fit": {
                    "samples": int(kept.sum()),
                    "warm": int((mine & warm).sum()),
                    "dropped": int((mine & ~solved["keep"] & ~warm).sum()),
                    "condition": solved["condition"],
                    "centre": solved["centre"],
                    "mad": solved["mad"],
                    "zero": solved["zero"],
                },
            }
            if max(map(len, steps)) > 1:
                vector["fit"]["all_rows"] = solved["all_rows"]
                vector["fit"]["passes"] = solved["passes"]
            if figures:
                vector["fit"]["moe_column"] = figures
            carried = [f for f in OPT_IN if f in features]
            carried += list(MOE_COLUMNS) * set(MOE_COLUMNS).issubset(features)
            if carried:
                vector["fit"]["vif"] = _vif(points, list(features), carried)
            if rowmod_k:
                vector["rowmod"] = rowmod_k
                vector["fit"]["rowmod"] = evidence
            if rowmod_vif:
                vector["fit"]["rowmod_vif"] = rowmod_vif
            if k == "prefill":
                if buckets:
                    vector["buckets"] = list(buckets)
                if cold:
                    vector["duty_tau"] = DUTY_TAU
                if keep_tail:
                    band = max(MAD_K * solved["mad"], MAD_FLOOR)
                    past = np.abs(solved["residual"] - solved["centre"]) > band
                    vector["fit"]["peer_tail"] = int((solved["keep"] & past).sum())
                if kda:
                    vector["kda_chunk"] = KDA_CHUNK
                if cols["short_hinge"]:
                    vector["short_knee"] = cols["short_hinge"]
                if gathered:
                    vector["ranks"] = dp
                if (spread := _min_spread(cols)) is not None:
                    vector["min_spread"] = spread
                law["prefill"] = vector
            else:
                law["decode"][k] = vector
    vectors = [("prefill", law.get("prefill")), *law["decode"].items()]
    report = {
        "provenance": [
            {k: v for k, v in run[0][0].provenance.items() if k != "requests"}
            for run in runs.values()
        ],
        "vectors": {k: {**v["fit"], "hull": v["hull"]} for k, v in vectors if v},
        "held_out": {},
    }
    drafts, serial = _draft_samples(samples), [s for s in samples if s.passes_s]
    if serial and not drafts:
        law["draft"] = {
            "provenance": f"step_linear_fit draft passes on {len(runs) * dp} "
            f"real records ({', '.join(runs)})",
            "decode": {},
        }
    elif drafts:
        drafted = _draft_options(options, bool(serial))
        law["draft"] = fit(drafts, drafted, seed, draft=True)[0]
    if serial:
        law["draft"]["forward"] = _forward(serial)
        law["draft"]["propose"] = _propose(_rest(serial))
    elif drafts:
        for name, mine in _pass_samples(samples).items():
            width = dp if name in GROUPED else 1
            vector = mine and _propose(mine, PASSES[name], width, samples)
            if vector:
                law["draft"][name] = vector
            elif mine:
                why = "no step every rank of which ran it or a dummy forward"
                law["draft"]["unfitted"][name] = why
        if any(s.view.prefill and s.draft_kv_s is not None for s in samples):
            law["draft"]["output_prefill_kv"] = True
    return law, report


def _draft_options(options, serial: bool) -> dict:
    """`options` for the draft law, which adds no padding, and adopts no MoE
    column at `dp` 1; above it, over `draft_moe_routing`, the drafter's own
    routed experts and top-k where `options` state them, its decode rungs pick
    one as the target's do.
    Under an EAGLE3 drafter whose records time its passes, the `base` form: its
    prefill vector fits the middle chunks alone, each one request's chunk at
    the step's token budget, whose `history` is its `query_cached` over it."""
    dp, options = _options(options)[1], options or {}
    routing = options.get("draft_moe_routing") if dp > 1 else None
    drafted = {**options, "moe_routing": routing, "buckets": None}
    return {**drafted, "form": "base"} if serial else drafted


def _draft_samples(samples) -> list:
    """The samples the draft law's prefill and decode vectors are fitted on,
    each labelled with its draft label (`_draft_kv`). Under an EAGLE3 drafter
    whose records time its passes, only the middle chunks: its decodes' and
    output prefills' drafts are its passes and the rest of their draft."""
    serial = any(s.passes_s for s in samples)
    return [
        dataclasses.replace(
            s,
            device_s=_draft_kv(s),
            draft_s=None,
            propose_s=None,
            draft_kv_s=None,
            passes_s=None,
        )
        for s in samples
        if s.draft_s is not None
        and _draft_kv(s) > 0
        and not (serial and (s.passes_s or s.view.produces_output))
    ]


def _rest(samples) -> list:
    """`samples` labelled, as propose labels, with what their draft label
    holds past their passes; those whose passes hold it all left out."""
    rest = [
        dataclasses.replace(s, propose_s=s.draft_s - sum(s.passes_s)) for s in samples
    ]
    return [s for s in rest if s.propose_s > 0]


def _forward(samples) -> dict:
    """The draft law's `forward` vector over `samples`' passes, each a sample
    at its features (`serial_passes`), solved and hulled as a step vector is,
    with `passes`, how many each holds; samples holding other counts refuse."""
    passes = sorted({len(s.passes_s) for s in samples})
    if len(passes) != 1:
        raise ValueError(f"the records' steps ran {passes} draft passes")
    (k,) = passes
    x = [c for s in samples for c in serial_passes(s.view, k)]
    y = [p for s in samples for p in s.passes_s]
    return {**_vector(np.array(x, dtype=np.float64), np.array(y), PREFILL), "passes": k}


def _draft_kv(s) -> float:
    """The draft prefill vector's label of `s`: an output prefill's draft-KV
    label, else its draft label less any propose label."""
    if s.view.prefill and s.draft_kv_s is not None:
        return s.draft_kv_s
    return s.draft_s - (s.propose_s or 0.0)


#: The draft law's vectors on `PROPOSE`, each with the label it is fitted on.
PASSES = {"propose": "propose_s", "draft_kv": "draft_kv_s"}
#: Those of `PASSES` whose pass waits on the rank's data-parallel peers, fitted
#: over the group's steps (`_pass_steps`); the others on each rank's own label.
GROUPED = ("propose",)


def _pass_samples(samples) -> dict:
    """The samples each of `PASSES` is fitted on: under `propose` those with a
    propose label, an output prefill's being its whole draft label, all after
    `forward_done_event`; under `draft_kv` the decodes with a draft-KV label."""
    return {
        "propose": [
            s if s.draft_kv_s is None else dataclasses.replace(s, propose_s=s.draft_s)
            for s in samples
            if s.propose_s is not None
        ],
        "draft_kv": [
            s for s in samples if s.draft_kv_s is not None and not s.view.prefill
        ],
    }


def _pass_steps(samples, label: str, dp: int, every) -> tuple[list, int]:
    """`samples` of `dp` ranks joined on run and step, each step its ranks'
    rows `(run, None, (1, seqs), seconds, step)`, `seconds` the smallest of
    their `label`s, in the order `every`'s runs first appear, then by step;
    and how many steps were left out, at which a rank gave no sample here and
    ran no dummy forward, as its samples in `every` state."""
    runs = list(dict.fromkeys(s.provenance["run"] for s in every))
    dummies = {(s.provenance["run"], s.rank): s.dummies for s in every}
    joined: dict = {}
    for s in samples:
        joined.setdefault((runs.index(s.provenance["run"]), s.step), {})[s.rank] = s
    steps, cut = [], 0
    for (i, step), ranks in sorted(joined.items()):
        idle = [step in dummies.get((runs[i], r), ()) for r in range(dp)]
        if sum(r in ranks or idle[r] for r in range(dp)) < dp:
            cut += 1
            continue
        seconds = min(getattr(s, label) for s in ranks.values())
        rows = [(1, len(s.view.requests)) for _, s in sorted(ranks.items())]
        steps.append([(i, None, x, seconds, step) for x in rows])
    return steps, cut


def _propose(samples, label: str = "propose_s", dp: int = 1, every=()) -> dict:
    """The draft law's `propose` vector over `samples`' propose labels, or
    another `label`, on `PROPOSE`, solved and hulled as a step vector is; above
    `dp` 1 by max-affine regression over `_pass_steps` of them, whose dummy
    forwards `every` states, hulled at each step's priced rank, the steps it
    left out under its fit's `ranks`; None when it left out every step."""
    if dp > 1:
        steps, cut = _pass_steps(samples, label, dp, every)
        if not steps:
            return None
        s = _fit_steps(steps, list(PROPOSE), lambda r: r[2])
        x = np.array([r[2] for r in s["rows"]], dtype=np.float64)
        return {
            "coefficients": dict(zip(PROPOSE, map(float, s["coef"]))),
            "hull": _hull(x[s["keep"]][:, 1:], PROPOSE[1:]),
            "fit": {
                "samples": int(s["keep"].sum()),
                "dropped": int((~s["keep"]).sum()),
                "ranks": cut,
                **{k: s[k] for k in ("condition", "centre", "mad", "zero")},
                "all_rows": s["all_rows"],
                "passes": s["passes"],
            },
        }
    x = np.array([(1, len(s.view.requests)) for s in samples], dtype=np.float64)
    return _vector(x, np.array([getattr(s, label) for s in samples]), PROPOSE)


def _vector(x, y, names) -> dict:
    """A vector on `names` over the rows `x` labelled `y`, solved and hulled."""
    s = _solve(x, y, list(names))
    fitted = {k: s[k] for k in ("condition", "centre", "mad", "zero")}
    return {
        "coefficients": dict(zip(names, map(float, s["coef"]))),
        "hull": _hull(x[s["keep"]][:, 1:], names[1:]),
        "fit": {
            "samples": int(s["keep"].sum()),
            "dropped": int((~s["keep"]).sum()),
            **fitted,
        },
    }


def token_band(tokens: int) -> str:
    """The band of `TOKEN_BANDS` `tokens` falls in: `lo-hi`, or `lo+` past the last."""
    lo = max([0] + [b for b in TOKEN_BANDS if b <= tokens])
    hi = min([b for b in TOKEN_BANDS if b > tokens], default=None)
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def _band(key: str, counts) -> str:
    if key != "prefill":
        return f"decode rung {key}"
    return f"prefill tokens {token_band(counts[1])}"


def priced_rank(ranks: dict):
    """The rank a DP step is priced and banded as, of `ranks`, {rank: (price,
    features)}: its costliest, a tie going to the rank of larger features, which a
    law blind to them prices the same and the step still waits for."""
    return max(ranks, key=ranks.get)


def _fold_laws(samples, options, seed, draft: bool = False):
    """Each run of `samples`, in the order runs first appear, with its ranks'
    samples and the law `fit` gives on the other runs, under `draft` as `fit`
    fits a draft law."""
    options = {
        **(options or {}),
        "serial": _serial(samples, options),
        "eager_dummies": _eager_dummies(samples, options),
        "cold": _cold(options),
        "peer_share": _share(options),
        "kda": _kda(options),
    }
    for run, ranks in _runs(samples, _options(options)[1]).items():
        others = [s for s in samples if s.provenance["run"] != run]
        yield run, ranks, fit(others, options, seed, draft)[0]


def _held_out(samples, options, seed, warm=None, draft: bool = False, folds=None):
    """Each step of each run, warm steps left out, priced as its costliest rank
    by the law fitted on the other runs, `folds`' when given, else
    `_fold_laws`', under `draft` as `fit` fits a draft law: (run, vector,
    band, that rank's (counts, context, tokens, dummies), relative error), the
    error nan where that law has no vector; `counts` with its `peer_tokens`."""
    form = _options(options)[0]
    options = {
        **(options or {}),
        "serial": _serial(samples, options),
        "eager_dummies": _eager_dummies(samples, options),
        "cold": _cold(options),
        "peer_share": _share(options),
        "kda": _kda(options),
    }
    if warm is None:
        warm = fit(samples, options, seed, draft)[0]["warm_steps"]
    folds = _fold_laws(samples, options, seed, draft) if folds is None else folds
    for i, (run, ranks, law) in enumerate(folds):
        routing = _law_routing(law)
        for group in _groups(
            ranks,
            form,
            draft,
            serial=options["serial"],
            routing=_routing(options),
            buckets=_buckets(options),
            eager=not draft and options["eager_dummies"],
            cold=not draft and options["cold"],
            share=not draft and options["peer_share"],
            kda=not draft and options["kda"],
            cols=_columns(options, draft),
        )[0]:
            key, _, seconds, step, *_ = next(iter(group.values()))
            if step in warm[i]:
                continue
            vector = law["decode"].get(key) if key != "prefill" else law.get(key)
            mine = [(r[1], *r[4:8]) for r in group.values()]
            price, error = [c[1] for c, *_ in mine], float("nan")
            if vector is not None:
                c = list(vector["coefficients"].values())
                price = [float(np.dot(c, _counts(vector, routing, *m))) for m in mine]
                error = max(price) / seconds - 1.0
            top = mine[priced_rank({j: (p, mine[j][0]) for j, p in enumerate(price)})]
            yield run, key, _band(key, top[0]), top, error


def _counts(vector: dict, routing, counts, context, tokens, dummies, peers=0) -> tuple:
    """A step's `counts` with the MoE column, `peer_tokens` and `rowmod`
    count `vector` adds, over its rows' `context`, its real MoE `tokens`,
    `dummies` of them its group's dummy forwards', under `routing`, and its
    `peers`, its `peer_tokens`."""
    moe = [f for f in MOE_COLUMNS if f in vector["coefficients"]]
    counts = (*counts, *(moe_column(f, tokens, routing, dummies) for f in moe))
    if "peer_tokens" in vector["coefficients"]:
        if peers is None:
            raise ValueError("a step stating no group tokens has no peer_tokens")
        counts += (peers,)
    return counts + ((rowmod(context, vector["rowmod"]),) if "rowmod" in vector else ())


def held_out(samples, options=None, seed: int = 0) -> dict:
    """Per band, the relative errors on each run of the law fitted on the others."""
    errors: dict[str, list[float]] = {}
    for _, _, band, _, error in _held_out(samples, options, seed):
        band = band if not math.isnan(error) else f"{band} (no vector)"
        errors.setdefault(band, []).append(error)
    return errors


def _bands(rows) -> dict:
    """Per band of `_held_out`'s rows, their steps and error percentiles."""
    bands = {}
    for band in sorted({r[2] for r in rows}):
        e = [r[4] for r in rows if r[2] == band]
        p10, p50, p90 = np.percentile(e, (10, 50, 90)).tolist()
        bands[band] = {"steps": len(e), "p10": p10, "p50": p50, "p90": p90}
    return bands


def _vector_of(view) -> str:
    """The name a law's `sources` gives the vector that prices `view`'s step."""
    return "prefill" if view.prefill else f"decode {_rung_key(view.capture_rung)}"


def _figures(ranks, law: dict, vectors=None) -> dict:
    """Per kind, `FIGURES` of the steps of `ranks`, one run's ranks' samples,
    priced by `law` as `validate.cost` grades a run (`accuracy.run_accuracy`),
    and under `prefill` and `decode`, as `refused`, the steps of the kind `law`
    refused, a kind with none graded stating 0 steps and no other figure; with
    `vectors`, only the steps a vector it names prices (`_vector_of`), each
    with its draft."""
    from atom.compass.backends.accuracy import run_accuracy

    steps = None
    if vectors is not None:
        steps = {s.step for r in ranks for s in r if _vector_of(s.view) in vectors}
    report = run_accuracy(ranks, law, steps)
    figures = {k: {f: g[f] for f in FIGURES} for k, g in report["graded"].items()}
    refused = dict.fromkeys(("prefill", "decode"), 0)
    # A band names its kind first: "prefill tokens ..." or "decode rung ...".
    for band, r in report["refused"].items():
        refused[band.split()[0]] += r["steps"]
    none = {**dict.fromkeys(FIGURES), "steps": 0}
    for kind, n in refused.items():
        if kind in figures or n:
            figures.setdefault(kind, dict(none))["refused"] = n
    return figures


def fold_figures(folds, vectors=None) -> dict:
    """Per run of `folds`, `_fold_laws`', `_figures` of its steps priced by the
    law fitted on the other runs."""
    return {run: _figures(ranks, law, vectors) for run, ranks, law in folds}


#: A filled vector's hull: it bounds the first hull coordinate to x >= 1 and
#: x <= 0, a set holding no point, so every step it prices is extrapolated.
NO_POINT = ((-1.0, 1.0), (1.0, 0.0))


def fill_rungs(law: dict, ladder) -> dict:
    """Give each rung of `ladder`, the captured decode widths, that `law` (and
    its draft law) holds no decode vector for a declared one, and return the
    rungs filled, each to its rule. No record reaching a rung leaves it with no
    vector, and a run's step at it would be refused for want of one; a filled vector
    prices it instead, `declared` and never validated, and outside its hull.

    The rule: between two fitted rungs, coefficient by coefficient, a
    coefficient a vector lacks read as 0, linear in the rung between the
    nearest fitted rungs on either side; a step's cost at a rung is its
    graph's, whose padded width the rung is, so a rung between two fitted ones
    costs between theirs. Past the fitted rungs, the nearest fitted rung's
    vector as it is: the widest fitted one's above them, the narrowest's below.
    A line through the two nearest rungs on one side multiplies their
    slopes' noise by the distance (#3796: rung 64 from rungs 24 and 32 priced
    the steps it filled +56%), and a copy is off by the rungs' own spread only.
    Two vectors under different `rowmod` moduli combine to no vector: the
    nearer one's is taken, ties to the wider. The law names the rungs filled
    under `filled`, each to its rule, and under `FILLED_RUNGS` by law
    (`filled_rungs`)."""
    filled = {}
    for of in (law, law.get("draft")):
        if not of or "decode" not in of:
            continue
        decode = of["decode"]
        fitted = sorted(int(k) for k in decode if k.isdigit())
        for rung in sorted(int(r) for r in ladder or ()):
            if str(rung) in decode or not fitted:
                continue
            below = [r for r in fitted if r < rung]
            above = [r for r in fitted if r > rung]
            if below and above:
                pair = (below[-1], above[0])
            else:
                pair = ((below[-1:] or above[:1])[0],) * 2
            a, b = (decode[str(r)] for r in pair)
            w = 0.0 if pair[0] == pair[1] else (rung - pair[0]) / (pair[1] - pair[0])
            if a.get("rowmod") != b.get("rowmod"):
                near = min(pair, key=lambda r: (abs(r - rung), -r))
                a = b = decode[str(near)]
                pair, w = (near, near), 0.0
            names = [*a["coefficients"]]
            names += [f for f in b["coefficients"] if f not in names]
            coefficients = {
                f: max(
                    0.0,
                    (1 - w) * a["coefficients"].get(f, 0.0)
                    + w * b["coefficients"].get(f, 0.0),
                )
                for f in names
            }
            free = len(hull_axes(tuple(coefficients)))
            lo, hi = pair
            if lo != hi:
                how = f"interpolated in the rung between rungs {lo} and {hi}"
                how += f" (weight {w:.4g} on rung {hi}), clipped at 0"
            elif not (below and above):
                side = "widest" if lo < rung else "narrowest"
                how = f"rung {lo}'s vector, the {side} fitted rung"
            else:
                how = f"rung {lo}'s vector"
            rule = f"rung {rung} filled, no record reached it: {how}"
            vector = {
                "coefficients": coefficients,
                "hull": {"facets": [[x, *[0.0] * (free - 1), c] for x, c in NO_POINT]},
                "declared": rule,
                "filled": {"from": list(pair), "weight": w},
            }
            if a.get("rowmod"):
                vector["rowmod"] = a["rowmod"]
            decode[str(rung)] = vector
            filled[("draft " if of is not law else "") + str(rung)] = rule
    if filled:
        law["filled"] = {k: v for k, v in sorted(filled.items(), key=lambda kv: kv[0])}
        law[FILLED_RUNGS] = filled_rungs(law)
    return filled


#: The law's key naming its filled rungs by law, `{"decode": [...], "draft": [...]}`.
FILLED_RUNGS = "filled_rungs"


def filled_rungs(law: dict) -> dict[str, list[int]]:
    """The decode rungs `fill_rungs` filled in `law`, by law: the target's under
    ``decode`` and its draft law's under ``draft``, each ascending and each
    empty when none was; a law written before `FILLED_RUNGS` is read from its
    `filled` names. A preflight compares them with the rungs a target record's
    steps ran at (`filled_share`)."""
    out: dict[str, list[int]] = {"decode": [], "draft": []}
    for name in law.get("filled") or {}:
        of, _, rung = name.rpartition(" ")
        out["draft" if of == "draft" else "decode"].append(int(rung))
    return {k: sorted(v) for k, v in out.items()}


def filled_share(law: dict, rungs: Mapping) -> dict:
    """The part of `rungs`, a target record's decode steps or seconds by the
    rung they ran at, that `law` prices on a filled vector: its ``steps`` (or
    seconds) there, the ``share`` of the whole and the filled ``rungs`` it ran
    at, ascending; a share of 0 when `rungs` holds none."""
    filled = set(filled_rungs(law)["decode"])
    on = {int(r): v for r, v in rungs.items() if str(r).isdigit() and int(r) in filled}
    total = sum(rungs.values())
    return {
        "steps": sum(on.values()),
        "share": sum(on.values()) / total if total else 0.0,
        "rungs": sorted(r for r, v in on.items() if v),
    }


def filled_warning(law: dict) -> str | None:
    """A warning naming the rungs `fill_rungs` filled in `law`, or None: a step
    a run takes at one is priced by a vector no record's step was fitted on."""
    named = filled_rungs(law)
    if not any(named.values()):
        return None
    parts = [f"{k} rungs {', '.join(map(str, v))}" for k, v in named.items() if v]
    return (
        f"WARNING: no record reached {'; '.join(parts)}: each is filled from the "
        "fitted rungs beside it, declared and never validated, so a run whose "
        "steps reach them is priced off its records; compare a target record's "
        "rung histogram with the law's filled_rungs (step_linear_fit.filled_share)"
    )


def mark_held_out(
    law: dict, report: dict, samples, options=None, seed: int = 0, draft=False
):
    """Write the held-out errors: under each of `law`'s vectors, as `held_out`,
    the hull of the rank rows `_held_out` priced its steps by, their errors per
    band; under `report`'s `held_out`, each run's errors per band; and, unless
    `draft`, under both's `figures`, `fold_figures`' on the same folds. A draft
    law fitted on an EAGLE3 drafter's passes (`_mark_passes`), or above `dp` 1
    (`_mark_dp_passes`), its vectors alike, on the steps each is fitted on,
    their bands named `draft`; `draft` says `law` is a draft law."""
    from scipy.spatial import QhullError

    folds = list(_fold_laws(samples, options, seed, draft))
    rows = [
        r
        for r in _held_out(samples, options, seed, law["warm_steps"], draft, folds)
        if not math.isnan(r[4])
    ]
    if not draft:
        law["figures"] = report["figures"] = fold_figures(folds)
    for run in dict.fromkeys(r[0] for r in rows):
        report["held_out"][run] = _bands([r for r in rows if r[0] == run])
    routing = _law_routing(law)
    for key, vector in [("prefill", law.get("prefill")), *law["decode"].items()]:
        mine = [r for r in rows if r[1] == key]
        if vector is None or not mine:
            continue
        points = [_counts(vector, routing, *r[3]) for r in mine]
        names = tuple(vector["coefficients"])
        axes = list(hull_axes(names))
        try:
            hull = _hull(
                np.array(points, dtype=np.float64)[:, axes],
                tuple(names[i] for i in axes),
            )
        except (ValueError, QhullError):
            continue
        vector["held_out"] = {"hull": hull, "bands": _bands(mine)}
    dp, serial = _options(options)[1], "forward" in law.get("draft", {})
    if "draft" not in law or not (serial or dp > 1):
        return
    passes = [s for s in samples if s.passes_s]
    drafts, mine = _draft_samples(samples), {"held_out": {}}
    if len({s.provenance["run"] for s in drafts}) > 1:
        drafted = _draft_options(options, serial)
        mark_held_out(law["draft"], mine, drafts, drafted, seed, True)
    if serial and len({s.provenance["run"] for s in passes}) > 1:
        _mark_passes(law["draft"], mine, passes)
    elif not serial:
        _mark_dp_passes(law["draft"], mine, samples, dp)
    for run, bands in mine["held_out"].items():
        named = {f"draft {b}": v for b, v in bands.items()}
        report["held_out"].setdefault(run, {}).update(named)


def _mark_passes(law: dict, report: dict, samples) -> None:
    """`mark_held_out` for a draft law's `forward` and `propose` vectors:
    each run's passes and propose rests (`_rest`) priced by the vectors fitted
    on the other runs, banded by pass."""
    from scipy.spatial import QhullError

    rows = {"forward": [], "propose": []}
    for run in dict.fromkeys(s.provenance["run"] for s in samples):
        others = [s for s in samples if s.provenance["run"] != run]
        mine = [s for s in samples if s.provenance["run"] == run]
        forward = _forward(others)
        for s in mine:
            passes = serial_passes(s.view, forward["passes"])
            for i, (x, seconds) in enumerate(zip(passes, s.passes_s)):
                price = np.dot(list(forward["coefficients"].values()), x)
                rows["forward"].append((run, "", f"pass {i}", x, price / seconds - 1))
        propose = _propose(_rest(others))
        for s in _rest(mine):
            x = (1, len(s.view.requests))
            price = np.dot(list(propose["coefficients"].values()), x)
            rows["propose"].append((run, "", "propose", x, price / s.propose_s - 1))
    for key, got in rows.items():
        for run in dict.fromkeys(r[0] for r in got):
            mine = _bands([r for r in got if r[0] == run])
            report["held_out"].setdefault(run, {}).update(mine)
        names = tuple(law[key]["coefficients"])[1:]
        try:
            hull = _hull(np.array([r[3] for r in got], dtype=np.float64)[:, 1:], names)
        except (ValueError, QhullError):
            continue
        law[key]["held_out"] = {"hull": hull, "bands": _bands(got)}


def _mark_dp_passes(law: dict, report: dict, samples, dp: int) -> None:
    """`mark_held_out` for a data-parallel draft law's `PASSES` vectors: each
    run's steps priced by the vector fitted on the other runs, banded by
    vector; a `GROUPED` vector's step is its ranks' (`_pass_steps`), priced as
    the max over them, another's each rank's sample at its own label."""
    from scipy.spatial import QhullError

    for name, mine in _pass_samples(samples).items():
        runs = list(dict.fromkeys(s.provenance["run"] for s in mine))
        if name not in law or len(runs) < 2:
            continue
        label, width, rows = PASSES[name], dp if name in GROUPED else 1, []
        for run in runs:
            others = [s for s in mine if s.provenance["run"] != run]
            fold = _propose(others, label, width, samples)
            if fold is None:
                continue
            c = list(fold["coefficients"].values())
            held = [s for s in mine if s.provenance["run"] == run]
            if width > 1:
                steps = _pass_steps(held, label, dp, samples)[0]
            else:
                steps = [
                    [(run, None, (1, len(s.view.requests)), getattr(s, label))]
                    for s in held
                ]
            for step in steps:
                price = {j: (float(np.dot(c, r[2])), r[2]) for j, r in enumerate(step)}
                top = priced_rank(price)
                error = price[top][0] / step[top][3] - 1.0
                rows.append((run, "", name, step[top][2], error))
        for run in dict.fromkeys(r[0] for r in rows):
            bands = _bands([r for r in rows if r[0] == run])
            report["held_out"].setdefault(run, {}).update(bands)
        if not rows:
            continue
        try:
            points = np.array([r[3] for r in rows], dtype=np.float64)
            hull = _hull(points[:, 1:], PROPOSE[1:])
        except (ValueError, QhullError):
            continue
        law[name]["held_out"] = {"hull": hull, "bands": _bands(rows)}


def host_held_out(samples) -> dict:
    """Per host law, the relative errors on each run of the laws fitted on the
    others; under `admit E`, those of the `E` of the forwards that admit
    requests, priced by their kind's law and `admit`."""
    errors: dict[str, list[float]] = {}
    for run in dict.fromkeys(s.provenance["run"] for s in samples):
        others = [s for s in samples if s.provenance["run"] != run]
        host = fit_host(others)[0]
        admit = host.get("admit")
        mine = [s for s in samples if s.provenance["run"] == run]
        rows = [
            (f"{kind} {segment}", _host_price(host, kind, segment, x, launch), seconds)
            for kind, segment, x, seconds, _, launch in _host_rows(mine)
        ]
        rows += [
            ("admit E", admit and base + float(np.dot(admit, a)), e)
            for a, base, e in _admit_rows(mine, host)
        ]
        for band, price, seconds in rows:
            if seconds > 0:
                errors.setdefault(
                    f"host {band}" + ("" if price is not None else " (no law)"), []
                ).append(price / seconds - 1.0 if price is not None else float("nan"))
    return errors


def _table(errors: dict) -> str:
    lines = ["band | steps | p10 | p50 | p90"]
    for band in sorted(errors):
        e = np.array(errors[band])
        p = ["-"] * 3
        if not np.isnan(e).any():
            p = [f"{v:+.1%}" for v in np.percentile(e, (10, 50, 90))]
        lines.append(f"{band} | {len(e)} | " + " | ".join(p))
    return "\n".join(lines)


def engine_config(records) -> dict:
    """The engine config `records` ran under: `keyed_of` the `.engine.json`
    beside each, at the longest ladder (`pooled`). A record without one, or
    records under two, refuse, naming the fields that differ."""
    held = []
    for record in records:
        engine = Path(record).with_suffix(".engine.json")
        if not engine.exists():
            raise ValueError(
                f"{record} has no {engine.name}, the engine config it ran under"
            )
        held.append(recorded_level(record) | keyed_of(json.loads(engine.read_text())))
    held = list(pooled(dict(enumerate(held))).values())
    names = sorted({k for e in held for k in e})
    if differ := [k for k in names if any(e.get(k) != held[0].get(k) for e in held)]:
        raise ValueError(
            f"the records ran under two engine configs, differing in {differ}"
        )
    return held[0]


def stated_buckets(record) -> list | None:
    """The prefill token-graph buckets the record `record` (`dp<r>.jsonl`) states:
    its engine file's, else its host key's graph sets'; None where neither does."""
    engine = Path(record).with_suffix(".engine.json")
    raw = json.loads(engine.read_text()) if engine.exists() else {}
    if PREFILL_BUCKETS in raw:
        return raw[PREFILL_BUCKETS]
    key = Path(record).with_suffix(".key.json")
    graphs = launch_features.stated(key.exists() and json.loads(key.read_text()))[1]
    return None if graphs is None else graphs["token"]["prefill"]


def recorded_level(record) -> dict:
    """`keyed`'s compilation level of the record `record` (`dp<r>.jsonl`): the
    ``--level`` of the ATOM arguments after -- in the run.json beside its
    `steps`, off `DEFAULT_LEVEL`, for engine and host key files, which named
    none at any level before `keyed` did."""
    run = Path(record).resolve().parent.parent / "run.json"
    argv = json.loads(run.read_text()).get("argv", []) if run.is_file() else []
    if "--" not in argv:
        return {}
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    level = parser.parse_known_args(argv[argv.index("--") + 1 :])[0].level
    return {} if level == DEFAULT_LEVEL else {LEVEL: level}


def recorded_buckets(stated: dict) -> tuple[int, ...] | None:
    """The prefill token-graph buckets `stated`, {where: buckets or None}, states,
    None when none states any. Two sets, or some stating them and others not,
    refuse, naming each."""
    held: dict = {}
    for where, buckets in stated.items():
        stated_as = None if buckets is None else tuple(buckets)
        held.setdefault(stated_as, []).append(where)
    if len(held) > 1:
        raise ValueError(
            "the records state other prefill buckets: "
            + "; ".join(
                f"{'none' if b is None else list(b)} in {', '.join(f)}"
                for b, f in held.items()
            )
        )
    return next(iter(held), None)


def refuse_contradicting(own, held: dict, stated, engine: dict) -> None:
    """Refuse `engine`, the engine file `stated`'s keyed fields, where it omits
    or states otherwise a field `held`, the record's own file `own`'s, holds."""
    if differ := [k for k, v in held.items() if k not in engine or engine[k] != v]:
        raise ValueError(f"{own} contradicts {stated} in {differ}")


def refuse_context_parallel(engine: dict, where: str) -> None:
    """Refuse an engine config that states a prefill context-parallel width: a
    step linear law states none, so a run at width 1 would price with it. A
    decode context-parallel width is the law's engine config's, which a run
    holds to its own (`run.refuse_unfitted_law`)."""
    if widths := [f"{k} {engine[k]}" for k in UNKEYED_CONTEXT_PARALLEL if k in engine]:
        raise ValueError(
            f"{where} states {' and '.join(widths)}, and a step linear law states "
            "no prefill context-parallel width, so a run at width 1 would price "
            "with it"
        )


def join_width(engine: dict) -> int:
    """The ranks a run's records join at: `engine`'s `data_parallel_size`.
    An engine config stating a prefill context-parallel width refuses."""
    refuse_context_parallel(engine, "the records' engine config")
    if "data_parallel_size" not in engine:
        raise ValueError(
            "the records' engine config states no data_parallel_size, the ranks "
            "a step joins; it predates the field"
        )
    return engine["data_parallel_size"]


def _routing_arg(value: str) -> tuple[int, int]:
    """`--moe-routing`'s `E,K`: two positive integers, `K` at most `E`."""
    try:
        experts, top_k = map(int, value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not E,K") from None
    if not 0 < top_k <= experts:
        raise argparse.ArgumentTypeError(f"{value!r} routes to {top_k} of {experts}")
    return experts, top_k


#: The kinds `--platform` states platform fields for: the law's and its host laws'.
PLATFORM_KINDS = ("step_linear_law", "host_laws")


def _platform_arg(value: str) -> dict:
    """`--platform`'s JSON object, an object of platform fields per kind."""
    try:
        given = json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"{value!r} is not JSON") from None
    if not isinstance(given, dict) or not all(
        isinstance(given.get(k), dict) for k in PLATFORM_KINDS
    ):
        raise argparse.ArgumentTypeError(
            f"{value!r} states no object for each of {', '.join(PLATFORM_KINDS)}"
        )
    return given


def offload_main(argv) -> None:
    """``offload LOG... --out ENTRY [--master-flags FLAGS --capacity-bytes
    BYTES]``: write `fit_offload`'s entry, and print each term with its fit."""
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.backends.step_linear_fit offload",
        description=fit_offload.__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="real server logs at OFFLOAD_PROFILE, a Mooncake Store's also at "
        "ATOM_LOG_LEVEL=DEBUG",
    )
    parser.add_argument("--out", required=True, help="where the entry is written")
    parser.add_argument(
        "--master-flags",
        type=shlex.split,
        help="a Mooncake Store's mooncake_master flags, as one string; the entry "
        "keeps them as argv",
    )
    parser.add_argument(
        "--capacity-bytes",
        type=int,
        help="the bytes a Mooncake Store's owners mounted, summed",
    )
    parser.add_argument(
        "--stream-floors",
        action="store_true",
        help="also fit a Mooncake Store entry's store_floor_s and load_floor_s "
        "and state its stream_wait (`fit_stream_floors`), which a run reads "
        "only under COMPASS_STORE_ON_STREAM=1",
    )
    parser.add_argument(
        "--stream-wait",
        type=float,
        default=1.0,
        help="with --stream-floors, the share of its stage's backlog a window waits for",
    )
    args = parser.parse_args(argv)
    entry, report = fit_offload(args.logs, args.master_flags, args.capacity_bytes)
    if args.stream_floors:
        world = entry.get("world", 1)
        floors, more = fit_stream_floors(args.logs, list(range(world)), args.stream_wait)
        entry |= floors
        entry["fit"] |= more
        report |= more
    Path(args.out).write_text(json.dumps(entry) + "\n")
    for name, fit in report.items():
        print(f"{name}: {json.dumps(entry.get(name))} {json.dumps(fit)}")


def _records_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """A parser of the records and the options `_labelled` reads with them."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("records", nargs="+", help="real step records, steps/dp0.jsonl")
    parser.add_argument("--form", choices=FORMS, default="base")
    parser.add_argument(
        "--moe-routing",
        metavar="E,K",
        type=_routing_arg,
        help="the model's routed experts and top-k: each decode rung may adopt its "
        "steps' real MoE tokens, or the experts they reach, as a column",
    )
    parser.add_argument(
        "--engine",
        help="the engine file the records ran under, stated in place of the "
        ".engine.json beside each, its prefill buckets for a record that states "
        "none; refused where one contradicts it",
    )
    return parser


def _labelled(args) -> tuple[list, dict, dict, dict]:
    """`args.records`' samples, each record's table, the options `fit` takes
    on them and the engine config they ran under. A record whose server built
    an aiter JIT module refuses, and so does an engine file that contradicts
    `--engine` at the records' longest ladder."""
    for record in args.records:
        run = Path(record).resolve().parent.parent / "run.json"
        if run.is_file() and (
            built := step_labels.refusal(json.loads(run.read_text()))
        ):
            raise ValueError(f"{record} is not fitted: {built}")
    owns = {Path(r).with_suffix(".engine.json"): r for r in args.records}
    stated = {r: stated_buckets(r) for r in args.records}
    if args.engine:
        engine = keyed_of(json.loads(Path(args.engine).read_text()))
        raw = {own: json.loads(own.read_text()) if own.exists() else {} for own in owns}
        ran = {o: recorded_level(r) | keyed_of(raw[o]) for o, r in owns.items()}
        for own, held in pooled(ran).items():
            refuse_context_parallel(held, str(own))
            refuse_contradicting(own, held, args.engine, engine)
        given = json.loads(Path(args.engine).read_text()).get(PREFILL_BUCKETS)
        stated = {args.engine: given} | {
            r: b for r, b in stated.items() if b is not None
        }
    else:
        engine = engine_config(args.records)
    dp = join_width(engine)
    options = {"form": args.form, "dp": dp, "moe_routing": args.moe_routing}
    options["buckets"] = recorded_buckets(stated)
    samples, tables = label_records(args.records, dp)
    return samples, tables, options, engine


def _figures_table(figures: dict) -> str:
    """`fold_figures`' figures, a line per held-out run and kind, `-` where a
    kind states no figure."""

    def show(value, percent=False) -> str:
        if value is None:
            return "-"
        return f"{value:+.2%}" if percent else str(value)

    head = "run | kind | steps | on totals | held-out step | holding it out | median"
    lines = [f"{head} | refused"]
    for run, kinds in figures.items():
        for kind, f in kinds.items():
            lines.append(
                f"{run} | {kind} | {f['steps']} | {show(f['on_totals'], True)} | "
                f"{show(f['held_out_step'])} | {show(f['held_out'], True)} | "
                f"{show(f['median'], True)} | {show(f.get('refused'))}"
            )
    return "\n".join(lines)


def figures_main(argv) -> None:
    """``figures RECORD... --law LAW [--source NAME]``: write `fold_figures`'
    figures on the records into the law file LAW, and print them.

    The records must be the runs LAW states under `runs`, and are folded under
    their engine config as the command fits them, and under the `form` and
    `moe_routing` LAW states, a `--form` or `--moe-routing` that contradicts
    them refusing, with `KDA`'s columns where its prefill vector has them. A law whose `sources` splits its vectors takes one source's
    at a time: `--source` names it, the records must be its `runs`, they are
    folded under `--form` and `--moe-routing`, as its provenance states them,
    and the figures, of the steps its `vectors` price, go under it; a source
    whose vectors price none of the records' uncut steps refuses. Records whose
    prefill buckets, `--engine`'s included, are not those LAW's prefill
    vector was fitted under refuse where the figures grade that vector, as
    do any under a law with no `sources` that names `padding` unfitted."""
    parser = _records_parser(
        "python -m atom.compass.backends.step_linear_fit figures",
        figures_main.__doc__.split("\n\n")[0],
    )
    parser.add_argument("--law", required=True, help="the law file written into")
    parser.add_argument("--source", help="the entry of the law's sources folded")
    parser.set_defaults(form=None)
    args = parser.parse_args(argv)
    law = json_file.read_json(args.law)
    into, vectors, runs = law, None, law.get("runs")
    sources = law.get("sources", {})
    if args.source is not None or sources:
        if args.source not in sources:
            raise ValueError(
                f"{args.law} splits its vectors over the sources {sorted(sources)}, "
                f"and --source names {args.source}; each is folded with its own "
                "records, form and options"
            )
        into = sources[args.source]
        vectors = into["vectors"]
        runs = runs and [runs[i] for i in into["runs"]]
        args.form = args.form or "base"
    else:
        stated = {"form": law["form"], "moe_routing": _law_routing(law)}
        for name, value in stated.items():
            if (flag := getattr(args, name)) not in (None, value):
                raise ValueError(
                    f"{args.law} states {name} {value}, and --"
                    f"{name.replace('_', '-')} gives {flag}: a law with no "
                    "sources is folded as it was fitted"
                )
        args.form, args.moe_routing = stated["form"], stated["moe_routing"]
    if runs is None:
        raise ValueError(f"{args.law} states no runs to hold the records to")
    samples, _, options, _ = _labelled(args)
    options["draft_moe_routing"] = _law_routing(law.get("draft", {}))
    fitted_with = (law.get("prefill") or {}).get("coefficients", {})
    options["kda"] = any(f in fitted_with for f in KDA)
    given = {s.provenance["run"] for s in samples}
    if differ := sorted(given ^ set(runs)):
        raise ValueError(
            f"the records' runs are not the {len(runs)} runs {args.law} states "
            f"for them; these are in one and not the other: {differ}"
        )
    prefill = law.get("prefill") if vectors is None or "prefill" in vectors else None
    if prefill or (vectors is None and "padding" in law.get("unfitted", {})):
        fitted, stated = tuple((prefill or {}).get("buckets", ())), _buckets(options)
        if stated != fitted:
            raise ValueError(
                f"{args.law} was fitted under the prefill buckets "
                f"{list(fitted) or 'none'}, and the records"
                f"{' with ' + args.engine if args.engine else ''} state "
                f"{list(stated) or 'none'}: its folds are fitted under its own"
            )
    cuts = [s.cut for s in samples if vectors and _vector_of(s.view) in vectors]
    if vectors is not None and all(cuts):
        why = {c: cuts.count(c) for c in sorted(set(cuts))}
        raise ValueError(
            f"{args.law}'s source {args.source} prices none of the records' "
            f"uncut steps: its vectors are {vectors}"
            + (f", and the fit cuts each step they price: {why}" if why else "")
        )
    into["figures"] = fold_figures(_fold_laws(samples, options, 0), vectors)
    json_file.write_json(law, args.law)
    print(_figures_table(into["figures"]))


def pack_main(argv) -> None:
    """``pack PATH...``: store each law file PATH, or each ``*.json`` in a
    directory PATH, xz-compressed beside it under its name and ``.xz``, as
    `json_file.write_json` writes it, and remove it, printing both sizes. A law
    whose document did not change is stored in the same bytes. A PATH whose
    name does not end in ``.json`` refuses, before any law is stored."""
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.backends.step_linear_fit pack",
        description=pack_main.__doc__.split("\n\n")[0],
    )
    parser.add_argument("paths", nargs="+", type=Path, help="law files or directories")
    paths = parser.parse_args(argv).paths
    laws = [x for p in paths for x in (sorted(p.glob("*.json")) if p.is_dir() else [p])]
    if named := [str(law) for law in laws if law.suffix != ".json"]:
        parser.error(f"a law file packed is named *.json, and these are not: {named}")
    for law in laws:
        into = law.with_name(law.name + json_file.XZ)
        json_file.write_json(json_file.read_json(law), into)
        print(f"{into}: {law.stat().st_size} -> {into.stat().st_size} bytes")
        law.unlink()


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["offload"]:
        offload_main(argv[1:])
        return
    if argv[:1] == ["figures"]:
        figures_main(argv[1:])
        return
    if argv[:1] == ["pack"]:
        pack_main(argv[1:])
        return
    parser = _records_parser(
        "python -m atom.compass.backends.step_linear_fit", __doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--out",
        required=True,
        help="where the law is written, as JSON, xz-compressed when it ends in .xz",
    )
    parser.add_argument(
        "--platform",
        required=True,
        type=_platform_arg,
        help="the platform the records ran on, as the law's and its host laws' "
        'keys state it: {"step_linear_law": {...}, "host_laws": {...}}',
    )
    args = parser.parse_args(argv)
    samples, tables, options, engine = _labelled(args)
    dp = options["dp"]
    for record, table in tables.items():
        step_labels.write_table(record, table)
    law, report = fit(samples, options)
    if options["buckets"] is None:
        law["unfitted"]["padding"] = NO_BUCKETS
    if any(s.provenance["config"] for s in samples):
        law["host"], law["host_fit"] = fit_host(samples)
    law["platform"] = args.platform["step_linear_law"]
    law["engine_config"] = engine
    if atom := records_atom(samples):
        law["records_atom"] = atom
    if "host" in law:
        stated = law["host"]["key"]["engine"]  # its graph sets stay
        graphs = {k: v for k, v in stated.items() if k == launch_features.GRAPHS}
        law["host"]["key"] |= {
            **args.platform["host_laws"],
            "engine": engine | graphs,
        }
    if len(args.records) > dp:
        mark_held_out(law, report, samples, options)
    fill_rungs(law, engine.get("ladder"))
    if warning := filled_warning(law):
        print(warning, file=sys.stderr)
    json_file.write_json(law, args.out)
    print(f"cut: {json.dumps(law['cut'])}")
    print(f"unfitted: {json.dumps(law['unfitted'])}")
    print(f"filled: {json.dumps(law.get('filled', {}))}")
    for name, vector in [("prefill", law.get("prefill"))] + list(law["decode"].items()):
        if vector:
            fitted = vector.get("fit", vector.get("declared"))
            print(f"{name}: {json.dumps(vector['coefficients'])} {json.dumps(fitted)}")
    for name, vector in law.get("draft", {}).items():
        if isinstance(vector, dict) and "coefficients" in vector:
            held = vector.get("held_out", {}).get("bands", {})
            print(f"draft {name}: {json.dumps(vector['coefficients'])} held out {held}")
    if "host" in law:
        print(f"host: {json.dumps(law['host']['laws'])} {json.dumps(law['host_fit'])}")
    if len(args.records) > dp:
        for form in FORMS:
            routing = None if form == "shared" else args.moe_routing
            errors = held_out(
                samples, {**options, "form": form, "moe_routing": routing}
            )
            print(f"\nheld out, form {form}\n{_table(errors)}")
        if "host" in law:
            print(f"\nheld out, host laws\n{_table(host_held_out(samples))}")
        print(f"\nfigures, form {options['form']}\n{_figures_table(law['figures'])}")


if __name__ == "__main__":
    main()
