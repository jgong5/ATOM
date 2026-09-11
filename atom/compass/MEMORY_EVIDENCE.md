<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved. -->

# The memory terms, and where each one's evidence comes from

Every number in the memory budget has two sides: a **derived** side, which the
model computes, and a **recorded** side, which a run observed. A row agrees or
it does not, and the percentage is reported either way -- but the percentage is
only worth reading once you know where both sides came from. Two of these rows
used to agree because they were the same arithmetic twice, and one of the
constants that makes a third row agree was fitted on the very model it is now
being checked against.

This file is the provenance ledger. It is written from the artifacts, not from
the commit messages, and where it contradicts an older claim the older claim is
named.

## Provenance classes

The boundary is not the model identity, it is the *configuration under
evaluation*. Calibrating a per-model constant on the 27B at the declared TP=1
source configuration is authorized; feeding a measurement taken at the target
configuration (TP=2, TP=4, or the utilization being predicted) back into its own
prediction is not.

| class | meaning | may a target claim rest on it? |
|---|---|---|
| **S** | source: `config.json`, the checkpoint header, a meta build, the deployment's own flags, the card's spec capacity | yes |
| **C06** | a constant fitted on the **0.6B** campaign | yes -- the target had no part in fitting it |
| **S27** | a constant calibrated by a full-engine run on the 27B **at the declared TP=1 source configuration**, recorded with model + TP + full config + run id + role | yes, as an input -- see the recording rule below |
| **X27** | a measurement taken at a **target** configuration: TP=2, TP=4, or a changed utilization | **no** -- using it makes the prediction self-referential |
| **T** | a reading the target run recorded on the device | only as the thing being predicted, never as an input to the prediction |

**The S27 recording rule.** An S27 constant is only S27 if its origin is written
down: model, TP, the full config it was measured under, the run it came from,
and its role (which term it supplies). A 27B-fitted number whose configuration
is unknown is not S27 and cannot be promoted to one -- absence of provenance is
not evidence of source provenance. `MODEL_HEADROOM` is exactly that case: it is
a 27B fit of unrecorded TP and configuration, so it stays disallowed until its
origin is established, and it is never to be relabelled as source.

**What S27 does and does not prove.** A term calibrated at TP=1 and then checked
against the same TP=1 run has a *calibration residual*, not a validation error;
quoting it as agreement is quoting the fit back to itself. Validation comes from
somewhere the fit did not see: an independent rerun at TP=1, or the TP=2 / TP=4 /
changed-utilization targets. Prefer a mechanistic derivation where one exists;
calibrate at the source width only where it does not.

## The terms

Errors are TP=1 / TP=2 / TP=4, rank 0, from the three records in
`tests/compass/memory_records/` (`scripts/compass/validate_memory.py`).

| term | derived side | class | recorded side | error | status |
|---|---|---|---|---|---|
| weights | `resident_bytes` on a meta build | S | `parameter_bytes` | **0 B at TP=1, 2 and 4, on all 7 records** | **closed, exact** |
| model buffers | `resident_bytes` on a meta build | S | `buffer_bytes` = 33 554 432 B (rotary cos/sin tables) | **0 B, every record** | **closed, exact** |
| load residue | `DEFAULT_LOAD_RESIDUE` | C06 | `weights_torch - parameter_bytes` | -93.0% / -3.3% / -6.7% | fails at TP=1 on a 14 MiB term |
| persistent | `DEFAULT_PERSISTENT` = 118 MiB | C06 | `current_torch - weights_torch` = 240.6 MiB | -51.0% at all three widths | **fails**, consistently, and is flat in width as claimed |
| activations | `peak_activation_bytes(graph)` | S (needs a trace) | `peak_torch - current_torch` | not run here (no 27B graph on this box) | **open** |
| non-torch | `DEFAULT_NON_TORCH[w] + MODEL_HEADROOM` | C06 **+ an unattributed 27B fit (provenance OPEN, not S27)** | `non_torch` | +4.9% / +1.2% / +1.3% | passes **only with a 27B-fitted constant** -- see below |
| pool estimate | `graph_pool_bytes(peak_torch - current_torch)` | T | `cudagraph_overhead` | +0.0% / +0.0% / +0.0% | **identity**, not a term; kept and labelled as a mirror check |
| graph pool | `measured_graph_pool_bytes(capture_sizes, w)` | C06 | `graph_pool.reserved` | -9.7% / +2.0% / **+26.8%** | **fails at TP=4** |
| kv blocks | `kv_geometry` + ATOM's `plan_pools` | S, given the budget | `blocks.num_kvcache_blocks` | +0.00% at every width and rank | exact -- but see "what exact means" |

## Five structural findings

### 1. The recorded decomposition telescopes, so its sum proves nothing

Each recorded component is the difference of two adjacent readings:

```
parameter_bytes            (params + buffers, deduplicated by storage)
load residue   = weights_torch  - parameter_bytes
persistent     = current_torch  - weights_torch
activations    = peak_torch     - current_torch
```

Their sum is `peak_torch` by construction, and it is `EQUAL` to the byte on all
four records -- which is arithmetic, not agreement. The components are disjoint
and exhaustive, so **there is no double counting on the recorded side**, and
equally **no evidence in the fact that they add up**. Only the derived side of
each row is independent, and each must be judged on its own.

The modelled side sums the same four, from `modelled_readings`:
`parameters + buffers + residue + persistent + activation`. Its `parameters`
and `buffers` come from `resident_bytes`, which returns them **disjoint**, so
the buffers are not counted twice there either. Note the definitional skew
between the two sides: the record's `parameter_bytes` *includes* buffers, which
is why `validate_memory.py` subtracts them before comparing, and why the
recorded `load residue` at TP=1 (14.2 MiB) is smaller than the buffers it sits
beside.

### 2. The graph pool sits beside `peak_torch`, not inside it

`get_num_blocks` subtracts `peak_torch + non_torch + cudagraph_overhead`, and
capture happens *after* `get_num_blocks` has run -- the KV cache must exist
before the graphs can be captured. So the pool cannot be inside the recorded
`peak_torch`, and adding it as a separate term does not double count.

Two pool numbers are recorded, and they are different quantities:

* **`reserved`** -- the delta in `torch.cuda.memory_reserved()` across capture.
  This is the term: a captured graph pins its intermediates for replay, so the
  segments the allocator had to create are what the deployment must budget.
* **`allocated`** -- the delta in `memory_allocated()`. Smaller, by segment
  bookkeeping and fragmentation (105.8 of 122 MiB at TP=1).

`measured_graph_pool_bytes` models `reserved`, and is compared against
`reserved`. The `allocated` side is reported beside it and is *not* a second
term.

### 3. `non_torch` passes because of a constant fitted at the target widths

`non_torch_bytes` returns `DEFAULT_NON_TORCH[w] + MODEL_HEADROOM`. The comment
at `memory_model.py:405` says `MODEL_HEADROOM = 266 MiB` is where "the 27B sat
exactly 266 MiB above the 0.6B at TP=2 and TP=4 alike". That comment is the
whole of its provenance: no run id, no utilization, no `max_num_seqs`, no
capture ladder, no date. So two things are true of it, and they are separate.

First, **its provenance is incomplete** -- it does not meet the S27 recording
rule, and no amount of confidence in the number substitutes for the record.
Second, **the widths named in the comment are the target widths**. TP=2 and
TP=4 are what this audit evaluates; a constant fitted there and used to predict
there is class **X27** regardless of how well its origin is documented. Closing
the provenance gap would not make this constant usable at TP=2/4. What it would
make usable is a *re-derivation* at the authorized TP=1 source configuration.

With it and without it, against these records (MiB):

| width | table (C06) | + headroom (X27) | recorded | error with | error without |
|---|---|---|---|---|---|
| 1 | 926 | 1192 | 1136 | **+4.9%** | **-18.5%** |
| 2 | 6906 | 7172 | 7084 | +1.2% | -2.5% |
| 4 | 7266 | 7532 | 7290 | +3.3% | -0.3% |

The headroom is 22.3% of the derived term at TP=1 and 3.5-3.7% at TP=2 and
TP=4. It was fitted at TP=2/4, where removing it costs little, and it is doing
its real work at TP=1, where it was not fitted -- and it makes TP=4 *worse*.
That is the shape of a constant absorbing something it does not explain.

Consequence, stated plainly: **the non-torch row's +4.9% at TP=1 is not an
authorized-input result.** Either the row is reported at -18.5% with C06
constants alone, or it is reported as passing with a constant fitted at the
target widths. It cannot be both, and POC_STATUS should not carry it as the
former. The way out is neither: calibrate the model offset at the declared TP=1
source configuration, record it as S27, freeze it, and let TP=2 and TP=4 judge
it.

`non_torch` is also `(total - free) - reserved`, and `total - free` is
device-wide: a co-tenant's allocation is charged to this configuration. The
TP=4 ranks differ by exactly 48 MiB here, which is the size of that doubt.

### 4. `deployment_constants.json` is not a calibration input

`validate_memory.py --calibrate` wrote
`agent_scratch/mem/evidence/stage0/deployment_constants.json` **from the 27B
records themselves** (`non_torch` 1191182336 / 7428112384 / 7644119040,
`load_residue` 14924832 / 2244419104 / 2324113184, `persistent` 252339712).
Every one of those is a class-**T** number.

Passing that file to `modelled_readings(calibration=...)` would make the
modelled budget a restatement of the target's own readings, and every term
downstream of it circular. Nothing does so today: `holdout.py` calibrates
through `step_accounting.py`, a different path. But `runtime/runner.py` loads
`profile["calibration"]` straight into `modelled_readings`, so the circular
wiring is one profile field away and nothing refuses it. **Dependency for the
lead** (that file is not mine to edit): `_modelled_readings` should refuse, or
at minimum record, a calibration whose provenance is the model under
evaluation.

### 5. The weights term is source-exact, and that is the one clean transfer

`meta_probe.py --weights-only` builds the model on the meta device -- shapes and
dtypes from `config.json` and the checkpoint header, nothing allocated, nothing
measured -- and `resident_bytes` sums it. Against every record on hand:

| tp | meta `parameters` | meta `buffers` | sum | recorded `parameter_bytes` | error |
|---|---|---|---|---|---|
| 1 | 54 713 457 120 | 33 554 432 | 54 747 011 552 | 54 747 011 552 | **0 B** |
| 2 | 27 818 133 472 | 33 554 432 | 27 851 687 904 | 27 851 687 904 | **0 B** |
| 4 | 14 370 471 648 | 33 554 432 | 14 404 026 080 | 14 404 026 080 | **0 B** |

Seven records -- TP=1 at utilizations 0.33, 0.40, 0.41 and 0.90, TP=2 rank 0,
TP=4 ranks 0 and 1 -- and the error is zero bytes on every one. The recorded
`parameter_bytes` includes buffers, which is why `validate_memory.py` subtracts
them before comparing against a parameters-only derivation; compared as the sum,
the two sides are the same number.

This closes O1, and it is worth being precise about why it is the strongest row
in the table: nothing about it was fitted. The derivation is arithmetic over the
checkpoint, it crosses tensor-parallel widths without a per-width constant, and
it was never shown a target reading. Where the historical ledger recorded
weights errors of +1.6 / -0.1 / -3.3%, those came from a different derived side;
the meta build has no error to report.

It also sharpens what is wrong with the rest of the budget. Weights are 94% of
`peak_torch` and they are exact, so the whole of the budget's uncertainty lives
in the three small terms beside them -- load residue, persistent, activations --
plus `non_torch` outside the allocator. Those are the terms to fix, and their
combined size is what bounds how well any of this can do.

## What "exact" means on the kv blocks row

`kv_geometry` reproduces 112 740 / 265 520 / 585 071 / 584 880 / 585 642 blocks
to the block. The inputs are `config.json`, the deployment's flags, **and the
five device readings the target run recorded** -- class T.

So the row establishes the *native pool geometry*: that one paged block is
1 056 768 B at TP=1, that one in-flight request's recurrent state is 74.8 MiB,
and that `plan_pools`' arithmetic over those two numbers is reproduced off the
device. Given a byte budget, the block count follows exactly.

It does **not** establish that the budget can be predicted. The budget was an
input. This must not be counted as G3b transfer evidence while that remains
true. The prediction that would count is the next section, and it is open.

## The frozen candidate budget (source-only), and what is missing

The rule: a candidate budget is frozen **before** any target reading is opened,
and may draw on classes S, C06 and S27 -- the last being a constant calibrated
by a full-engine run at the declared TP=1 source configuration, recorded with
model, TP, full config, run id and role. Freezing on S+C06 alone, which is what
an earlier draft of this ledger did, is stricter than the goal: it leaves
authorized TP=1 source calibration unused and reports gaps that are closable.
What stays out is class X27 -- anything measured at TP=2, TP=4, or at the
utilization being predicted.

```
available_for_kv = min(total * utilization
                       - (peak_torch + non_torch + graph_pool + 0.02 * total)
                       - extra_reserve,
                       free)
peak_torch = parameters + buffers + load_residue + persistent + activations
free       = total - peak_torch - non_torch          (a clean box, by model)
```

Input ledger at TP=1, the width full-engine source calibration is allowed at:

| input | value | class | source |
|---|---|---|---|
| `total` | 206 141 652 992 | S | MI308X spec capacity; the records agree, and are not the source |
| `utilization` | 0.90 | S | deployment flag |
| `max_num_seqs` | 32 | S | deployment flag |
| `max_model_len` | 262 144 | S | deployment flag |
| `block_size` | 16 | S | deployment flag |
| `kv dtype` | bf16, 2 B | S | deployment flag |
| paged block | 1 056 768 B | S | `config.json` geometry |
| state slot | 78 446 592 B | S | `config.json` geometry |
| `parameters` | 54 713 457 120 | S | meta build, `meta_probe.py --weights-only --tp 1`; exact against 4 records |
| `buffers` | 33 554 432 | S | same; exact against 4 records |
| `load_residue` | 1 MiB | C06 | `DEFAULT_LOAD_RESIDUE[1]` -- 93% low here; recalibratable as S27 at this width |
| `persistent` | 118 MiB | C06 | `DEFAULT_PERSISTENT` -- known 51% low against this model; recalibratable as S27 at this width, though a mechanism is preferable |
| `non_torch` | 926 MiB | C06 | `DEFAULT_NON_TORCH[1]`, **headroom excluded** as X27; the model offset is recalibratable as S27 at this width |
| `activations` | **OPEN** | S | `peak_activation_bytes` of a TP=1 meta trace at the warmup token count |
| `graph_pool` | `floor + slope x sum(ladder)` | C06 | `measured_graph_pool_bytes`, ladder (1,2,4,8,16,32) is a deployment flag |

Two of the five inputs that were open are now closed and exact (finding 5).
One remains: `activations`, which needs a TP=1 meta trace at the warmup token
count. Beside it sit three terms that are present but wrong -- `load_residue`
at -93%, `persistent` at -51%, and `non_torch`, whose C06 value is -18.5% here
and whose device reading is itself unstable (phase A, 0.33 rung). Those four
are the whole remaining gap, and together they are 6% of `peak_torch`.

Under the corrected calibration boundary, three of the four are addressable at
the declared TP=1, util 0.90 source configuration: see "Source calibration at
TP=1" below.

## Source calibration at TP=1: three terms fixed, and what each is worth

`atom/compass/core/memory_calibration.py` carries the constants measured at the
27B's declared source configuration -- TP=1, utilization 0.90, `max_num_seqs`
32, the cc-traces capture ladder, prefix caching off. Each one records model,
width, the full config, the record it came from and that record's hash, and
which term it supplies, which is the recording rule the `MODEL_HEADROOM` case
exists to justify.

| term | C06 default | error | calibrated | fitted on | validated at | error there |
|---|---|---|---|---|---|---|
| `persistent` | 118 MiB | **-51.0%** | 252 339 712 B | `27b.tp1.memory.json` | TP=1 util 0.33 / 0.40 / 0.41 | **0 B** |
| | | | | | TP=2 rank 0 | -0.003% |
| | | | | | TP=4 ranks 0, 1 | -0.004% |
| `load_residue` | 1 MiB at TP=1 | **-93.0%** | 14 924 832 B | `27b.tp1.memory.json` | TP=1 util 0.33 / 0.40 / 0.41 | **0 B** |
| `non_torch` | 1 104 MiB + headroom | -2.8% vs the busy record | 1 157 627 904 B | `27b.tp1.exclusive.memory.json` | TP=1 util 0.33 settled / 0.40 / 0.41 | **0 B**, and one miss of **+45.1%** |

Source records: `27b.tp1.memory.json` sha256 `62332900...`,
`27b.tp1.exclusive.memory.json` sha256 `4ff60279...`.

### Three answers, not two

`classify` used to answer `residual` or `validation`. It now answers three
ways, because two runs of one configuration are neither:

* **residual** -- the record the term was fitted on. Reproducing it
  demonstrates arithmetic and nothing else.
* **repeat** -- a *different* run of the same configuration. Real
  reproducibility evidence; no evidence of transfer, because nothing the
  constant was fitted against has changed.
* **validation** -- a run at a configuration the fit never saw.

Telling `repeat` from `residual` needs the record's hash, so
`validate_memory.py --source-calibration` hashes each record it reads and the
row says which of the three it is. Without a hash the stricter answer stands.
The distinction is not cosmetic: on `27b.tp1.exclusive.memory.json`,
`persistent` is a repeat and `non_torch` is a residual, and on
`27b.tp1.memory.json` it is the other way round.

### What transferred and what did not

`persistent` transfers across both axes -- flat to within 11 KiB on a 240 MiB
term, over TP=1, 2 and 4 and four utilizations -- which is the claim the 0.6B
constant was making and getting wrong by half.

`load_residue` is offered at TP=1 only. It is 14 MiB there and 2.1 GB at TP=2,
because at width 1 there are no collective pools to register; a constant fitted
at TP=1 says nothing about TP=2, so the C06 table keeps the wider entries
(-3.3% and -6.7%) rather than being overwritten with a number that does not
apply. This is the same width-specificity `DEFAULT_LOAD_RESIDUE` already
encodes, honoured rather than flattened.

`non_torch` is offered at TP=1 only for a different reason: it is
`(total - free) - reserved`, a whole-device reading, and at TP=1 one rank *is*
the device. At TP=4 it is four ranks' worth of a shared card and the source run
says nothing about it, so the C06 table keeps those widths.

### `non_torch`: which record it may come from, and the run it cannot predict

The TP=1 source record reads 1 191 182 336 B; the same configuration on an
exclusively-held device reads 1 157 627 904 B (phase C, three byte-identical
runs). The 33 554 432 B between them is, as far as this evidence goes, a
neighbour, so the calibrated value is the exclusive one and the busy record is
not a calibration input. That is the whole reason phase C was run.

Substituting the constant for the reading predicts the frozen ladder exactly:
1583 blocks at util 0.33 settled, 15 238 at 0.40, 17 188 at 0.41, each on the
nose. It does **not** predict phase A's first 0.33 run, which read
1 677 721 600 B and got 1091 blocks: the constant says 1583, **+45.1%**. Three
controls have failed to reproduce that excursion and nobody has explained it,
so it is recorded in the term's own `validated_against` as a KNOWN RISK, with a
test that fails if the warning is removed. A single constant cannot bound a
486 MiB excursion, and this one does not claim to.

Nothing here mutates the campaign defaults: `DEFAULT_PERSISTENT`,
`DEFAULT_LOAD_RESIDUE` and `DEFAULT_NON_TORCH` are untouched, the 0.6B path is
unchanged, and the calibration is opt-in per model. `for_model` returns None
for a model nobody has measured rather than handing back a neighbouring model's
constants.

One sharp edge, now handled rather than flagged: `non_torch_bytes` drops
`MODEL_HEADROOM` whenever *any* calibration mapping is passed, including one
that says nothing about `non_torch`. That is wanted where the calibrated run
already contains the headroom and wrong everywhere else, so the validator hands
the mapping over only at a width the term was actually measured at. At TP=2 and
TP=4 the non-torch row is unchanged from before this work.
## G3 phase A: the frozen ladder, run on the device

Predictions were frozen in `tests/compass/memory_records/frozen_util_predictions.json`
(sha256 `520a40b0...`) before the device was touched. Outcomes are in
`g3_util_phasea.json`. One device, exclusively leased, TP=1, `max_num_seqs 32`,
capture ladder (1,2,4,8,16,32), prefix caching off; 2026-09-11T08:31:14Z to
08:39:20Z, shell exit 0.

### The refusal at util 0.32 happened, as predicted, for the predicted reason

The engine did not start. The chain is
`InsufficientPoolBudget` -> `RuntimeError`, and the numbers it carries are:

| quantity | predicted | actual | error |
|---|---|---|---|
| state floor, 32 slots | 2 510 290 944 B | 2 510 290 944 B | **0 B, exact** |
| KV budget at 0.32 | 2 088 656 282 B | 2 122 210 714 B | +33 554 432 B (+1.6%) |
| engine's `min_util` hint | >= 0.33 | >= 0.33 | equal |

The state floor is the geometry, and the geometry is exact to the byte. The
budget is 32 MiB out, and that 32 MiB is the whole of the disagreement -- it is
`non_torch`, which read 32 MiB lower on the device than the frozen input did.
The rounded strings in the engine's message differ (1.98GB where the prediction
said 1.95GB) because the byte counts differ by that 32 MiB; the strings are
reported, not asserted on.

### The three starting rungs: the geometry contributes no error at all

| util | predicted blocks | actual | error | `non_torch` drift | drift / block bytes | blocks from the device's own `non_torch` |
|---|---|---|---|---|---|---|
| 0.33 | 1 551 | 1 091 | +42.2% | +486 539 264 B | -460.4 | **1 091, exact** |
| 0.40 | 15 206 | 15 238 | -0.2% | -33 554 432 B | +31.8 | **15 238, exact** |
| 0.41 | 17 157 | 17 188 | -0.2% | -33 554 432 B | +31.8 | **17 188, exact** |

The last column is the same geometry fed the `non_torch` the device itself
recorded, everything else unchanged. It reproduces the engine's block count at
every rung, exactly. So the sizing chain decomposes cleanly:

> **every block of prediction error at TP=1 is the `non_torch` reading, divided
> by the paged block size. The geometry contributes zero.**

That is the useful result of the ladder, and it is worth more than the error
percentages: it says which term to spend the next measurement on, and it says
the other terms are not hiding compensating errors.

`peak_torch`, `weights_torch`, `parameter_bytes`, `buffer_bytes`,
`current_torch` and `cudagraph_overhead` are **bit-identical across all four
utilizations**. Nothing in the non-KV budget except `non_torch` moved when
utilization moved, which is what the axis was chosen for.

### The 0.33 rung was the odd one out, and phase B tested it

The 486 MiB by which `non_torch` sat high at 0.33 is 45% of the term, and it
appears once: 0.40 and 0.41 agree with each other to the byte
(1 157 627 904 B) on the same exclusively-held device. The 0.33 run started
96 s after the 0.32 run died, so "the failed process was still releasing VRAM"
fitted as well as "the term depends on utilization" did.
`agent_scratch/mem/g3_util_phaseb.sh` ran the 2x2 that separates them, and
refuted the teardown theory -- see phase B below.

**The +42.2% stands as the observed error.** It is what the frozen prediction
missed that run by, and it is reported in the rung table above without
qualification. What is unresolved is its *cause*, and whether a run with an
unexplained 486 MiB excursion should count toward an acceptance verdict. Those
are separate questions from the measurement, and the measurement is not
withdrawn pending either. If isolation evidence later invalidates this run, the
reason will be a concrete observed one -- named owner, timing -- recorded
beside it; disagreeing with the other six runs is not itself a reason.

### Phase B: the 0.33 excursion did not reproduce, and no theory survives it

The obvious explanation for the 486 MiB was that the 0.33 run started 96 s
after a run that died, and inherited its teardown. Phase B ran the 2x2 that
would show it -- each utilization once from a settled device, once straight
after a failure -- on the same exclusively-held device, 08:41:49Z to 08:52:28Z,
shell exit 0.

| run | preceded by | blocks | `non_torch` |
|---|---|---|---|
| `settled_u0.33` | 120 s idle | 1 583 | 1 157 627 904 |
| `settled_u0.40` | 120 s idle | 15 238 | 1 157 627 904 |
| `afterfail_pre` (0.32) | -- | refused | -- |
| `afterfail_u0.40` | a failure, 93 s earlier | 15 238 | 1 157 627 904 |

**The teardown theory is refuted, not confirmed.** Reproducing the sequence as
closely as the harness allows -- a 0.32 failure, then a run 93 s later, against
phase A's 96 s -- produced the same `non_torch` as a settled device, to the
byte. So the honest account of the excursion is: it happened once, in seven
TP=1 runs on an exclusively-held device, and the nearest thing to a repeat did
not reproduce it. Cause unknown.

That is worth more than a tidy explanation would have been, because of what the
other six runs say. `non_torch` read **1 157 627 904 B, bit-identical, in six
runs** spanning utilizations 0.33, 0.40, 0.41 and both sequence positions. A
term that is bit-stable six times out of seven and 42% out on the seventh
cannot be characterised by its median: the error bar that matters is the
excursion, and one sample of it is one sample.

Two consequences follow, and they point in opposite directions:

* `settled_u0.33` gives the clean reading of that rung: **1 583 blocks against
  1 551 predicted, -2.0%**. So all three starting rungs now carry the same
  single-cause error, 32 blocks, from the frozen input's `non_torch` sitting
  32 MiB above what an exclusive device reads. The ladder is accurate to 0.2%
  at 0.40 and 0.41 and 2.0% at 0.33, and the difference between those is only
  that the same 32 MiB buys more blocks when there are fewer of them.
* Nothing in the model bounds the excursion. A deployment sized at 0.33 on the
  settled reading would have sized 1 091 blocks on the other one, 31% fewer.
  Phase C answered O7 with three byte-identical exclusive-device runs, so the
  calibrated constant is no longer a single sample -- but three samples of a
  term that was stable six times in seven still say nothing about the seventh.
  The constant carries the excursion in its own record as a known risk.

### What the isolation evidence actually covers

Phases A and B recorded `rocm-smi --showmeminfo vram --showpids` **before** the
first run and the device's used bytes before each subsequent one. They did not
sample during or after. So the isolation claim they support is narrower than
"the device was exclusively held": it is "the device was idle at 297 689 088 B
each time a run started, and one foreign KFD entry (PID 1685833, name UNKNOWN)
was present holding 0 B throughout".

That is not enough to attribute the 0.33 excursion to a neighbour, and it is
not enough to rule one out either. A process that allocated 486 MiB and exited
between two samples leaves no trace in this evidence. Phase C samples ownership every ~35 s across
its whole window (`tests/compass/memory_records/27b.tp1.exclusive.ownership.txt`)
so that the source calibration, at least, carries sampled evidence through the
run rather than two endpoints -- at that granularity, not continuously.

### Phase C: three repeats of the source configuration, with ownership sampled

08:53:49Z to 09:03:38Z, shell exit 0, same exclusively-leased device, TP=1,
utilization 0.90, `max_num_seqs` 32. All three runs produced **byte-identical
records** -- sha256 `4ff60279...` for each -- with 112 772 blocks,
`non_torch` 1 157 627 904 B, `free` 149 866 676 224 B, `peak_torch`
57 971 260 416 B, graph pool reserved 127 926 272 B. Frozen in
`g3_util_phasebc.json` together with phase B.

Ownership was sampled every ~35 s from 08:55:09Z to 09:08:32Z, 22 samples,
kept in `27b.tp1.exclusive.ownership.txt`. What the samples show, exactly:
every KFD process attributed VRAM in any sample belongs to this agent's own
container; one foreign KFD entry (PID 1685833, UNKNOWN) is present in every
sample holding 0 B; the device's idle floor is 297 689 088 B in every sample.
The coverage is 35-second granularity over the phase C window only -- it does
not cover phases A or B, and an allocation that began and ended between two
samples would not appear in it.

The historical TP=1 source record reads 1 191 182 336 B against these runs'
1 157 627 904 B. The difference is 33 554 432 B, exactly 32 MiB. That is the
observation; attributing it to a specific neighbour is not something this
evidence does.

Two of the three runs started while the box still showed 21.5 GB and 170 GB of
residual occupancy from this agent's own previous engine, and read the same
`non_torch` as the run that started from an idle device. That is a statement
about the readings at those three run-starts and nothing more: it does not
establish what the device held at each run's measurement point, and it does not
refute an intermittent teardown interaction as the cause of the phase A
excursion. **The phase A +42.2% remains an observed error with an unknown
cause.**

### What this is and is not

It is a genuine startup rejection, predicted from a frozen artifact and then
observed: the configuration was chosen on the model's own numbers, not fitted
to a failure that had already happened. The 0.32 rung needs no request length
to reject, so it stands whatever the final workload turns out to be.

It is **not** a model-free memory proof, and it is not closure of the memory
gate -- it is narrow startup evidence for one deployment on one axis. The held-fixed non-KV readings come
from the TP=1, util 0.90 run: they are *calibrated inputs* to this utilization
transfer, and what the ladder validates is the transfer across utilization with
those inputs held, not the inputs themselves. And the 0.33 and 0.40 admission
verdicts are not verdicts on the final workload -- see below.

## The feasibility scenarios are conditional, not verified

`atom/compass/core/feasibility.py` reports, at TP=1 on the 27B:

* `--max-num-seqs 1551`: the state floor (1551 x 74.8 MiB = 113.31 GB) exceeds
  the KV budget (113.30 GB) and ATOM's own `plan_pools` raises
  `InsufficientPoolBudget`;
* `--max-num-seqs 1400`: sizing yields 11 190 blocks and reports healthy, while
  the stress request (249 344 in + 5 690 out = 255 034 tokens = 15 940 blocks)
  can never be admitted. Against the provisional window bound (6 725 blocks) a
  1 400-seq pool is ample, so this scenario is a diagnostic about the stress
  length, not a finding about the acceptance workload.

Both reuse the non-KV readings from a run at `max_num_seqs 32`. **Persistent
forward buffers, warmup activations and the capture pool may each depend on
`max_num_seqs` or on the capture ladder**, and none of the three has been
measured at another concurrency. So the correct statement is conditional: *if*
those three terms are flat in `max_num_seqs`, the refusal is at 1551 and the
silent-failure window opens at 1400.

The admission half is checked against `Scheduler._unschedulable_reason`, called
unbound over a real `BlockManager`. That is narrow semantic evidence that the
mirrored rule decides as ATOM's rule decides. It is **not** evidence about
engine startup, about the admission path end to end, or about integrated
replay.

`Scheduler.__init__` did not return in this container -- it opens a KV-event
publisher and a connector -- and `tests/test_scheduler.py` did not either. That
is an environment limit on a box whose driver is wedged, pending proof, and is
not a claim about ATOM's code.

### The 255 034-token request is a stress diagnostic, not the workload

`cc_pilot.jsonl` (62 requests) has a longest of 249 344 in + 5 690 out =
255 034 tokens, 15 940 blocks, and earlier drafts of this ledger treated it as
the request the deployment has to admit. It is not. It comes from the old full
trace; CC's provisional long acceptance window caps input at 107 328 tokens,
and **46 of the 62 requests in that slice are longer than the window allows**.
A slice that is 74% out of window is not the workload, it is the evidence the
workload was selected from.

So the lengths divide, and the code and tests now name them apart:

| name | lengths | blocks | what it is for |
|---|---|---|---|
| `STRESS_LONGEST` | 249 344 + 5 690 | 15 940 | a prompt longer than anything the window sends; shows where the admission gate bites |
| `CC_LONG` (registered) | 107 328 + 2 413 | 6 859 | the long arm of the registered CC protocol -- the request the deployment must admit |
| `CC_SHORT` (registered) | 2 560 + 21 | 162 | the short arm |
| slice bound (superseded) | 107 328 + 260 | 6 725 | what `window_upper_bound` derived from `cc_pilot.jsonl` before the manifest existed |
| longest actually inside the window | 96 960 + 260 | 6 077 | the largest request `cc_pilot.jsonl` contributes to the window |

The CC protocol was registered as `47917ade` with those two arms, which
replaces the provisional bound this ledger carried. The slice-derived numbers
stay in the table because they are what the acceptance argument rested on until
09:20Z today, and the gap between the two -- 260 output tokens assumed against
2 413 registered, 6 725 blocks against 6 859 -- is the measure of what a
slice-derived bound was worth. `within_window` and `window_upper_bound` in
`feasibility.py` still compute the slice figures; they are the fallback for a
workload whose manifest has not been locked, and CC no longer needs them.

Consequences for the phase A rungs, stated exactly:

* **util 0.32 is a verdict on the deployment.** It never starts, so no request
  length rescues it and none is needed to condemn it. It stands whatever the
  manifest says.
* **util 0.33 refuses the registered long arm.** 6 859 blocks are needed;
  0.33 sizes 1 583 on a settled device and 1 091 on the phase A reading. Both
  refuse, so this rung fails against the registered workload and not merely
  against a stress diagnostic.
* **util 0.40 admits it.** 15 238 blocks against 6 859 needed, 2.2x over, and
  the short arm needs 162. That is now a statement about the registered
  workload rather than a provisional one -- subject to everything else in this
  ledger, in particular that all three rungs hold the non-KV readings fixed
  from the TP=1 util 0.90 run and that `max_num_seqs` has never been varied.

## Open items

| # | item | needs | status |
|---|---|---|---|
| O1 | `parameters` / `buffers` for the 27B at TP=1/2/4 | meta build | **closed** -- exact at all three widths, finding 5 |
| O2 | 27B activation trace at TP=1, prefill-shaped | meta trace, no device | open; the lead's CPU cardinality tracer derives 27B graphs in a device-free container, so this does not need the GPU -- coordinate reuse of that import bootstrap |
| O3 | source-only candidate budget, frozen, vs the recorded budget | O2 | open; the other four inputs are now closed or calibrated |
| O4 | `persistent` -- was 51% low | -- | **closed by calibration**, exact at TP=1, -0.004% at TP=4; a mechanism would still be better than a constant |
| O5 | `persistent` / activations / pool as functions of `max_num_seqs` | GPU, TP=1 | open, and now the main conditionality left; all three are proven flat in *utilization* (phase A) but untested in concurrency |
| O6 | physical start-up at `--max-num-seqs 1551` and 1400 | GPU, TP=1 | open; superseded as the acceptance gate by the utilization axis, kept as a diagnostic |
| O7 | `non_torch` from an exclusive-device source run at util 0.90 | GPU, exclusive | **closed by phase C** -- three byte-identical runs, calibrated at TP=1, exact at three unseen utilizations; the +42.2% excursion it cannot bound is carried with it |
| O8 | graph pool at TP=4, where the model reads +26.8% | GPU, 4 devices | open |
| O9 | `MODEL_HEADROOM` provenance: which run, which config | lead / history | open; until then it stays disallowed and is not to be relabelled as source |
| O10 | manifest-derived acceptance lengths | final CC workload | **closed** -- CC protocol `47917ade`: long 107 328 + 2 413 (6 859 blocks), short 2 560 + 21 (162) |
| O11 | what the 486 MiB `non_torch` excursion was | unknown; three controls failed to reproduce it | open, and the one thing the calibrated `non_torch` does not bound |
