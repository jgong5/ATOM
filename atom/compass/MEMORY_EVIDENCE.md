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
through `step_accounting.py`, a different path. `runtime/runner.py` used to load
`profile["calibration"]` straight into `modelled_readings`, leaving the
circular wiring one profile field away with nothing to refuse it. **Closed**
(2026-09-11, with the lead's authorisation for this one method): a derived
prediction now refuses a calibration that carries no provenance block, or whose
provenance for `persistent`, `non_torch` or the load residue names a class in
the target family. The numbers above are class **T** and would be rejected by
name.

### 4a. What a derived prediction now refuses

`_modelled_readings` was three fallbacks in one method. A profile naming no
graph defaulted the activation peak to **zero** -- the largest derived term in
the budget, 2.96 GB at the 27B source config, so the KV pool grew by that much
and the engine died at steady state instead of at start-up. A profile naming no
`total` read `torch.cuda.mem_get_info()` and sized the prediction to whichever
card the modelling run happened to land on. And a bare `except Exception`
turned every failure above into device sizing with a warning. All three
produced a budget that read as a forecast and was a measurement of something
else.

The judgement now lives in `memory_model.derived_readings`, which returns the
five readings and the activation peak or raises `UnfoundedPrediction` naming
the term that has nothing behind it. `runtime/runner.py` owns only the file
system. The refusals: no `total`, no `parameters`, no warmup shape, no graph,
a graph with neither recorded deaths nor a measured peak (`UnfoundedActivation`,
which is now a subclass and so propagates the same way), no calibration, a
calibration missing any of the three terms it supplies, a calibration with no
provenance block or with a term the provenance does not cover, and a
target-class fit.

One more, and it is the one that matters for the target: a graph traced at a
**different tensor-parallel width** than the prediction asks for, or one that
does not record its width at all. The activation peak is the only term in the
budget that shards -- everything else is either flat in width or carries its
own per-width table -- and a walk cannot be re-sharded after the fact. Silence
is refused rather than read as width one, because that reading is exactly the
failure: a TP=1 peak, four ranks too wide, sitting inside a TP=4 budget with
nothing in the five readings to show for it. This is the gate a TP=2 or TP=4
prediction stops at today, and stopping is the correct answer until a graph
exists at that width.

The two diagnostic modes are untouched and stay distinct: a run with no flag
measures this device, and `--compass-memory-in` replays what a device recorded.
Both are labelled as measurements because that is what they are.

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
| `activations` | 2 956 984 320 | S27 | `peak_torch - current_torch` at the TP=1 source configuration, which is the warmup prefill and nothing else; measured, not walked -- see O2 below |
| `graph_pool` | `floor + slope x sum(ladder)` | C06 | `measured_graph_pool_bytes`, ladder (1,2,4,8,16,32) is a deployment flag |

**No input in this ledger is open any more.** Two of the five were closed
exactly by the meta build (finding 5), three by source calibration at TP=1, and
the last -- `activations` -- is closed empirically rather than analytically
(O2). The three terms that were present but wrong are replaced: `load_residue`
was -93%, `persistent` -51%, `non_torch` -18.5%.

Running the ledger through `modelled_readings` and `blocks_from_readings` gives
`peak_torch` **57 971 260 416**, `non_torch` **1 157 627 904**, pool
**591 396 864**, and a plan of **112 772 KV blocks and 32 state slots** -- which
is what the source run recorded, to the block.

**That agreement is a residual and nothing more.** Four of the five terms in
`peak_torch` were read off this very record, so the arithmetic reconstructing
it demonstrates the arithmetic and not the model. What it does establish is
narrower and still worth stating: the ledger is complete, every input carries a
class, and the budget closes end to end without a single class X27 number. The
one place the model and the record genuinely differ is `free` -- 147 012 764 672
modelled against 149 866 676 224 recorded, because the model assumes a clean box
and the record has the neighbours in it. It changes no answer here: the
utilization budget binds well below `free` on both sides, which is exactly the
condition a record has to meet before it may be replayed at all.

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
runs). The 33 554 432 B between them is *observed* and not
explained: the exclusive record's ownership sample covers its own run and shows
one foreign KFD attachment holding 0 B throughout, and no sample was taken
during the other run at all, so nothing here identifies what held those bytes.
What makes the exclusive record the calibration input is therefore the sampled
ownership of the device it was taken on, not a diagnosis of the gap. That is
the whole reason phase C was run.

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
## O2: what the existing prefill graphs can and cannot say about activations

The term was open because it wanted a TP=1 trace at the warmup shape. Two
things had to be settled: *which* graph is the warmup step, and whether that
graph carries liveness at all.

**Which graph.** `warmup_model` resets the peak, runs one dummy prefill, reads
the peak back -- so the reading belongs to that step and no other. At the
source configuration the arithmetic (`model_runner.py`, `warmup_model`) is
`num_seqs = max(1, min(16384 // 262144, 32)) = 1` and `seq_len = min(262144,
16384 // 1) = 16384`: **one request, 16 384 query tokens, no history**, since
the sequences it builds are fresh and nothing is cached.

Exactly one derived graph has that shape -- `s27prefhead.tp1.r0.json`, from
`pref_chunk16k_head.json` (`query_lens [16384]`, `context_lens [16384]`). Its
neighbour `s27prefdeep.tp1.r0.json` is the trap: same 16 384 tokens, but
`context_lens [114688]`, so 98 304 tokens of history and 7x the KV to read.
**The two graphs have identical keys** -- `batch_signature [16384]` both -- so
`graph_tokens` cannot separate them, and anything matching on the token total
takes whichever it is handed. The distinguishing shape is in
`provenance.batch_spec`, which is what `traced_shape` now reads and
`warmup_mismatch` now checks. A different token *count* is not a mismatch:
scaling across counts is the claim the row exists to test.

**Whether it carries liveness. It does not.** Both graphs are `source:
"derivation", device: "meta"`, and across 2 439 operators **not one** records a
`dies_at`. Nothing runs on meta, so no finalizer fires; liveness at a shape is
a device observation and a derivation cannot have made it. The alias
distribution says the same thing from the other side: the meta graphs read
2 067 in-place against 193 allocated, where an on-device capture of the same
model reads 648 allocated against 83 in-place.

Walking `s27prefhead.tp1` anyway returns **570 425 344 B**, all of it from 64
`aten::empty.memory_format (16384, 17408)` allocations with no recorded death.
Against the measured term of 2 956 984 320 B that is 19.3% -- a 2.4 GB
understatement, in the direction that sizes a pool too large and starts an
engine that cannot start. It is an artifact of the last-read fallback, not a
model of anything, and it is now refused: `activation_bytes_at` raises
`UnfoundedActivation` for a graph with neither recorded deaths nor a measured
peak, and `validate_memory` prints no derived figure for one.

**So the term is closed empirically rather than analytically.** At the source
configuration `peak_torch - current_torch` is **2 956 984 320 B**, and the two
TP=1 records -- the historical one and the exclusive phase C capture, separate
engine starts -- agree **to the byte**. That is class S27: a full-engine run at
the declared TP=1 source configuration, and it carries its shape with it.

What it does not close:

* **Width.** TP=2 reads 1 730 150 400 B and TP=4 1 191 969 280 B. Both are
  class X27 -- measurements at the configuration being predicted -- so neither
  is an input, and `mapping` does not offer this term above width one.
  Activations at TP=2 and TP=4 remain **underived**, and that is now the last
  open input in the candidate budget at those widths.
* **Shape.** The constant is the peak at 1x16384 cold. A deployment that
  changes `max_num_batched_tokens`, `max_model_len` or `max_num_seqs` changes
  the warmup shape, so `SourceRun.matches` now holds all three and the
  constant is withheld rather than stretched.
* **Mechanism.** A constant is not a model. Closing this analytically needs a
  device capture at the warmup shape -- one prefill, 16 384 tokens, no history,
  with deaths recorded -- which is an empirical need for a later lease. The
  only on-device 27B captures that exist are decode-shaped (4 requests x 1
  token at context 66, `activation_peak_bytes` 3 104 256).

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

## Identity is not integrity: what tells a repeat from a residual

Phase C settled a question the classifier had been answering with a hash. The
three repeats wrote records that are equal **byte for byte** -- same sha256 as
each other and as the committed `27b.tp1.exclusive.memory.json`. So:

* a hash **match** is not one execution. Here it is three.
* a hash **mismatch** is not a second execution. Re-serialising one record --
  re-indenting the JSON, round-tripping it through a tool -- changes the hash
  while no run has occurred.

A content hash answers *are these the bytes the constant was read off*. It
cannot answer *was this the run the constant was read off*, and only the second
question separates a residual (the fit reported back to itself, worth nothing
as agreement) from a repeat (reproducibility evidence, worth something).

The two questions are now two calls. `classify(term, config, producer=...)`
takes a **producer**, and the identity is CC's rather than a second scheme:
`compass.execution/1`, an `execution_id` minted in `cc_traces_run.py` at the
instant the server process is launched, derived from the launch facts recorded
beside it so that any reader can re-derive it. `integrity(term, sha)` takes the
hash and says `intact` / `altered` / `unknown`.

`atom/compass/core/execution_id.py` holds that definition once, stdlib-only and
with no engine imports, because the harness that mints an id and the classifier
that reads one must not drift apart -- and `cc_traces_run.py` cannot be
imported for its two functions alone, since it loads sibling campaign modules
at import time. The rule is pinned against a vector taken from CC's own
implementation at `f4e06b0c`, with a cross-check that re-derives it from that
file wherever the harness is in the tree.

It is the **canonical** copy, not one of two: `scripts/compass/execution_id.py`
delegates its constants and both functions here and keeps only the script and
file helpers. Two byte-compatible copies would pass on the day they landed and
diverge afterwards without failing loudly, which is the same silent-caution
failure the schema exists to prevent, so the suite checks delegation by
function identity rather than by answer. The module carries both names for the
field order -- `ID_FIELDS` and `ID_INPUTS`, the same tuple object -- so neither
caller had to be edited on the commit that merged them, and it exports the
stamp shape (`STAMP_FIELDS`, `stamp_of`, `read_stamp`) that lets an artifact
carry enough to re-verify its own id after it has been copied. Importing it as
a package module pulls `atom/__init__.py` and `atom/compass/__init__.py` and
nothing heavier -- no torch, no AITER, checked on the device-free box -- and a
caller that wants not even that can load the file by path, a recipe the
docstring offers and the suite exercises.

An id is **never inferred**. A block naming a host and a pid is not an
execution identity and is not promoted into one; an id that does not follow
from its own recorded inputs is damaged or transplanted, which is worse than
unidentified rather than better; an id under an unknown schema cannot be
checked. All three read as "cannot tell". Reading an identity out of a payload
hash would assert a producer nobody observed, which is the specific error the
hash-keyed first cut made.

Two conservative rules, both of which cost the calibration credit rather than
grant it:

1. **An unidentified producer is classified `residual`** -- the weaker claim.
   Calling an unknown run a repeat would credit the constant with
   reproducibility nobody observed; calling it a residual credits it with
   nothing.
2. **Integrity is only reported where the producer says this *is* the fitted
   run** (`identifies`). The first cut printed `bytes altered` on any record
   whose hash differed from the fitted one, which fired on every row of every
   other record: the exclusive capture is not the historical record and never
   claimed to be. `altered` has to mean the bytes moved under a run.

What this costs today, stated plainly: **no shipped record carries a run
block**, so `producers` is empty on both source runs, every row reads
`residual (run unidentified)`, and no integrity note can fire. The three phase
C executions are witnessed -- engine pids 695009, 751439, 775417 across
08:55:09Z-09:03:34Z in the ownership log -- but that witness lives in a
gitignored sampler log beside the records, not inside them, so it cannot be
machine-checked and is not asserted as identity. Those records also predate
`compass.execution/1` entirely, and nothing here back-fills them: a legacy
record stays unidentified. That is O12.

## O13: tensor lifetime at the source width, on no device at all

O13 asked for the wrong thing. It asked for the 27B warmup prefill traced at
TP=2 and TP=4 with deaths recorded, and called that the single blocker to a
wider budget. Those are the target configurations. A tensor lifetime read off
a TP=4 run is a TP=4 measurement whoever captures it and whatever the capture
is called; feeding it to a TP=4 prediction is class X27 with a source-side
label on it. The item is restated here: **capture lifetime at the declared
TP=1 source, derive the sharded and replicated components for TP=2 and TP=4.**

### The premise underneath it was false

The stated reason a TP=1 walk was impossible was that the only TP=1 graphs at
the warmup shape are meta derivations, and that "a derivation cannot record
liveness because nothing runs and no finalizer fires". That sentence was in
`liveness_is_recorded`'s docstring and in O2's row, and it is wrong.

`MetaOpTracer._watch` puts a `weakref.finalize` on every output it sees and
`_died` writes the index at which it fired, on meta as on a device. Nothing
about meta suppresses it: a meta tensor is a Python object with a refcount,
and lifetime under refcounting is a property of the code that runs, not of the
device the code runs on. `agent_scratch/memval/probe_deaths.py` runs the same
five-operator body on meta and on cpu, under `inference_mode` and under
`no_grad`, through the real tracer:

```
meta  inference ops= 5 deaths= 5 producers=0 seen=0 input_storages=[0]
cpu   inference ops= 5 deaths= 5 producers=0 seen=3 input_storages=[207317824, ...]
```

Five operators, five observed deaths, on both devices, in both grad modes. The
recorder holds no strong reference that would defer them. What was missing was
never the observation -- it was the *writing*: only `runner.py::_stamp_deaths`
copies `tracer.deaths` onto the graph, and it runs on the device capture path
alone. `scripts/compass/graph_diff.py` builds its own tracer and discards them.

### Three instrumentation defects, none in a file this worker owns

| # | where | what | consequence |
|---|---|---|---|
| D1 | `atom/compass/runtime/meta.py::_storage_of` | asks a meta tensor for `data_ptr()`, which is always 0 | every output looks like the same buffer: `_canonical` folds the graph into one alias chain, `inputs_from` credits every input to whichever operator ran last, and the walk finds **2 live tensors in 2999 operators**. `untyped_storage()._cdata` is a working identity (`probe_meta_storage.py`: views share it, separate buffers do not) |
| D2 | `atom/compass/runtime/derive.py::record_collectives` | hand-builds the `aiter::all_reduce_` `OpSpec`, returns the simulated passthrough, and never watches an output | the walk reads 128 immortal residual-stream allocations at TP=2: **21.9 and 21.5 GiB** against a TP=1 walk of 2.5. The first fix written here was `output_aliases=(-1,)`, and it was **wrong** -- see the correction below. The defect is that no output is watched, so nothing can die. **It sits under any derived-graph walk at TP>1, which includes the graph-pool figure in O8** |
| D3 | `scripts/compass/graph_diff.py::_trace` | never stamps deaths | the actual reason meta graphs carry no `dies_at`. `_stamp_deaths` belongs in `MetaOpTracer`, where both paths reach it |

All three are patched locally, each documented as the upstream change it
stands in for, in `agent_scratch/memval/lifetime/capture_lifetimes.py`. They
are the lead's to place.

### What the source capture gives

`capture_lifetimes.py --tp 1 --batch-spec warmup_tp1.spec.json`, one CPU
process, no device, no weights, no data, 11 s: **2999 operators, 2790 with an
observed death.** The batch spec is the warmup step as `warmup_model` builds
it -- 1 request, 16 384 query tokens, no history -- recorded in
`agent_scratch/memval/lifetime/warmup_tp1.spec.json`.

The peak is **not in the GDN**. It is in the MLP, at 2 717 908 992 B live:
`aiter::gemm_a16w16` [16384, 34816] at 1 140 850 688, its silu [16384, 17408]
at 570 425 344, and six hidden-width [16384, 5120] buffers at 167 772 160
each. The separately derived gated-delta-rule workspace total
(`memory_activation.py`, 2 684 878 848 B at TP=1) sits *below* that, so the
opaque region does not set the high-water mark at any width -- which is worth
saying plainly, because that derivation was built on the assumption it did.

### The width mechanism is read off the shapes, not fitted

> **Withdrawn, and replaced.** The per-tensor rule below -- trailing dimension
> equal to hidden means replicated -- cannot distinguish the residual stream
> from an attention output, which is `num_heads * head_dim` and therefore also
> 5120 at TP=1. It is superseded by the lineage-aligned cross-width
> classification in *A tensor's width class is not readable from its shape at
> one width*. The **bytes** below do not rest on it: each width's number is the
> walk over that width's own derived graph, and the flag was an annotation on
> the live set. The paragraph is kept as written so the correction has
> something to be a correction of.

Each live tensor is classed by its trailing dimension: `== hidden (5120)` is
the residual stream, replicated at every width; wider is a column-parallel
projection, and the graph derived at that width already carries the narrower
shape. The derivation confirms it operator by operator -- gate_up [16384,
34816] -> [16384, 17408] -> [16384, 8704], silu [16384, 17408] -> [16384,
8704] -> [16384, 4352], hidden-width buffers unchanged at all three.

The three gate_up and silu chains above are the part that survives: they are
column-parallel and the lineage classification agrees. What the rule had no
right to say is anything about the six hidden-width buffers, and that is
exactly where an attention output would hide.

One named bias, now removed: at TP>1 `aiter::masked_embedding` was counted at
its int32 first-input dtype, 335 544 320 B instead of 167 772 160. The walk
takes PyTorch's promotion rule instead of the first argument's dtype, so the
operator is sized correctly by rule rather than by name, and the frozen
candidate's `walk_bytes` / `visible_peak_bytes` gap is now that rule's
arithmetic rather than a named subtraction (O18).

### Frozen, then checked exactly once

`tests/compass/memory_records/frozen_activation_candidates.json`
(`compass.activation.candidate/1`) was written before any comparison, with
`frozen_before_any_comparison_with` naming the TP=2/TP=4 measured peaks:

| width | candidate | X27 actual (evaluation-only) | error |
|---|---|---|---|
| 1 | 2 956 984 320 | 2 956 984 320 | 0, **by construction** -- the TP=1 shortfall of 239 075 328 B between the walk and the source measurement is carried as a declared `unexplained_residue` term, replicated across widths. It is a residual, not a prediction |
| 2 | 2 101 346 304 | 1 730 150 400 | **+21.5%** |
| 4 | 1 673 527 296 | 1 191 969 280 | **+40.4%** |

Per token: 165 888 / 113 664 / 87 552 B modelled against 180 480 / 105 600 /
72 752 B measured. Not a shape artifact -- `max_num_batched_tokens` is 16 384
in all three memory records, and `peak_torch - current_torch` reproduces each
recorded activation peak exactly, so the three widths are being compared at
one warmup shape.

The direction matters for what the term is for: over-stating activations
under-sizes the KV budget, which is the conservative failure. It is still
wrong, and it is wrong in a way that grows with width, which is exactly the
axis being transferred.

### What is actually open here

The walk holds six hidden-width buffers live at the peak, three of them from
`aten::empty_like`. The measured term is `peak_torch - current_torch`, so any
buffer allocated *before* the warmup forward -- a preallocated forward
variable -- is already inside `current_torch` and cannot appear in the
difference, however alive it is. The walk counts them anyway. That is a
mechanism that over-counts, it points the right way, and it is decidable
entirely at the source: a TP=1 device capture of the allocation curve across
the warmup step says which of those buffers are fresh. Two of the six would
close most of the TP=4 gap, which is precisely why it must be *measured* at
TP=1 and not chosen to fit -- no term here is to be selected by the size of
the error it removes.

Recorded and not used: a peak taken in the GDN region instead of the MLP
tracks the three measured widths more closely (residues 16 608 / 13 424 /
16 424 B per token). That is a post-hoc observation over three points, it
contradicts the operator walk about where the high-water mark sits, and it is
written down only so that a later mechanism cannot be mistaken for it.
## The collective is out-of-place: a correction, and what it changes

The fix recorded above as D2 -- mark `aiter::all_reduce_` in-place, because the
trailing underscore says so -- was wrong, and the naming was the only evidence
for it. Read from the live group instead (`aiter/dist/parallel_state.py` and
`dist/device_communicators/`, in the container image):

* `GroupCoordinator.all_reduce`: *"PyTorch custom ops do not support mutation or
  returning a new tensor in the same op. So we always make the all-reduce
  operation out-of-place."* It returns `input_` unchanged only when
  `world_size == 1`.
* every live path allocates its own output -- quick reduce and custom
  all-reduce `out = torch.empty_like(inp)`, the non-capturing warmup branch
  `torch.zeros_like(input)` ("to mimic the allocation pattern since custom
  allreduce is out-of-place"), pynccl `out_tensor = torch.empty_like(in_tensor)`,
  the torch.distributed fallback `input_.clone()`.
* the IPC-registered pool is on the **input** side (`reg_inp = self._pool
  ["input"].data_ptr`); `registered_input` says whether the *input* is already
  registered. The output is not a registered buffer and is not input 0.

So the operator allocates, and `output_aliases` for it must be `None`. What is
actually broken is liveness: the hand-built `OpSpec` has no output tensor of
its own, nothing is watched, and nothing can die -- 128 immortal buffers, not
128 aliased ones. The meta dispatch path has the mirror-image defect:
`_collective_stand_in` returns `tensors[0]`, the input object itself, so a
collective that dispatches on meta is recorded as writing into its own input
and the input's death at the call site disappears. `atom/model_ops/linear.py`
line 1115 is `y = tensor_model_parallel_all_reduce(y)`: the row-parallel
matmul's output loses its last reference *at* the collective. One hidden-width
buffer live across the call is the right answer; zero and one-per-call-forever
are the two wrong ones, and the derivation has been making both.

`DESIGN_NOTES`'s 0.6B table reports the TP=2 activation term moving from +12.6%
to -0.7% when in-place all-reduces were marked. If that came from a captured
graph, it was compensating for an input death that was never seen, and it needs
re-deriving on the C06 artifacts before it is trusted in either direction.

Two consequences beyond the activation term. Under CUDA-graph capture the
collective's `empty_like` is served from the capture's private pool, so a
collective inside a captured region is **graph-pool** bytes and the same
operator in eager is activation bytes -- O8 and O17 both need to say which they
are counting. And the packet's P2 is now a different change from the one first
proposed: allocate a fresh stand-in, register it, watch it.

## Why a replicated over-count cannot be the width error

`forward_vars["outputs"]` is `torch.empty(max_num_batched_tokens, hidden_size,
bf16)` in `ModelRunner.allocate_forward_vars` (`model_runner.py:1290`) --
167 772 160 B, allocated at engine init, before `warmup_model` runs, and
therefore inside `current_torch`. It cannot appear in `peak_torch -
current_torch` however alive it is, and `hidden_size` comes from the HF config,
so it is the same size at every width. The walk credits a write into it as an
allocation. That is a real over-count, found from source ownership rather than
from the size of any residual.

It is also not the answer, and the decomposition says why. The candidate is
`sharded(tp) + replicated + residue`, and the residue is *defined* at TP=1 as
`measured - derived`. Remove a replicated term and the residue grows by exactly
as much; the totals at TP=2 and TP=4 do not move. **Any error in a replicated
term cancels at every width.** The +21.5% / +40.4% therefore lives in the
sharded fraction.

Solving the form against the two source-legal anchors: exactness at TP=1 and
TP=2 needs about 2.45 GB that divides by the width, against the walk's 1.71 GB.
Roughly 0.74 GB of what the walk holds replicated -- or never sees at all --
must really shard. The allocations inside the opaque custom operators are the
obvious place for it to be hiding, and no dispatch trace on any device can see
them. This is arithmetic on the open error, not a term: nothing has been
changed to match it, and the frozen candidate stands where it was.
## A tensor's width class is not readable from its shape at one width

The frozen candidate annotates each tensor live at the peak with whether a
wider group makes it smaller, and the rule was: trailing dimension equal to the
hidden size means the residual stream, therefore replicated. That rule cannot
be right. At TP=1 an attention projection's input is `num_heads * head_dim`,
and `num_heads * head_dim` **is** the hidden size -- 5120 here -- so the
residual stream and a tensor that halves at TP=2 are the same shape and the
rule calls both replicated. Reshapes and views of a head-indexed tensor have
the same problem, and a fused operator's outputs need not agree with each
other.

What replaces it asks ATOM's own sharding arithmetic instead of guessing from
one width. `lineage_keys` gives each operator a width-invariant identity from
its ancestry -- the operator that ran this name, on values produced by *those*
operators, recursively -- and makes collectives transparent, passing their
input's identity through, so the operator after an all-reduce at TP=2 still
aligns with the operator after the matmul at TP=1. Index alignment cannot do
this: every row-parallel matmul gains a collective, so index *i* drifts further
from its counterpart the deeper into the model it sits. `width_classes` then
reads each aligned output's shape at TP=1, 2 and 4 and reports `replicated`,
`sharded` with the axis named, or `unresolved` -- a ratio the width does not
explain is reported, not rounded. `width_coverage` says how much of the graph
aligned at all, because a classification that silently drops half a graph is
worse than none.

**What this does not do is explain the width error.** The candidate's bytes at
each width are the walk over *that width's own derived graph*, not a projection
from TP=1 through the flag; the flag was an annotation on the live set. So
+21.5% and +40.4% stand exactly where they were. Said plainly because the
convenient reading -- "the width rule was wrong, that was the bug" -- is
available here and is not true.

What could still move those numbers is the next audit, which is the same method
applied to computation rather than communication: whether any operator's meta
or derived behaviour disagrees with its native one about *mutation*. A fused
add-and-norm or an activation that writes in place on the device, but whose
meta path returns a fresh tensor, invents a live buffer per call -- replicated,
hidden-width, exactly the shape that would inflate a walk at every width. The
collective audit found precisely this class of disagreement in both directions,
so the fused add/RMSNorm, the silu/MLP destinations and the attention output
handling are to be read the same way: the registered schema and the
implementation, not the trailing underscore.

## The two graphs differ by step kind, and nothing else

The TP=1 lifetime trace records 2999 operators; the body graph the cost side
prices records 2439. Until that gap has a cause, neither can be called the
native warmup allocation graph, so the cause was read off the two graphs'
own provenance and operator names (`agent_scratch/memval/lifetime/parity.py`).

Everything that could have made them incomparable is identical: `region: body`,
`device: meta`, `compilation_level: 0`, 129 device factories redirected, and
the same `includes` / `excludes` (`model forward`; not `compute_logits`, not
the sampler, not input preparation). The one difference is the batch: one
prefill of 16 384 tokens against four decodes at context 66.

The names say the same thing, and the arithmetic closes exactly. 416 operators
appear only in the prefill graph -- `aten::cat` 64, `aten::chunk` 32,
`aten::index.Tensor` 32, `aten::slice.Tensor` 64, `aten::squeeze` 64,
`aten::sub.Tensor` 32, `aten::to.dtype` 64, `aten::unsqueeze` 64, plus 176
operators of shared-name drift (`aten::mul.Tensor` -128, `aten::add.Tensor`
-32, `aten::view` -32, `aten::reshape` -16, `aten::empty_like` +32) -- which is
the gated-delta-rule's chunked-prefill path. 32 appear only in the decode graph:
`triton::_mrope_qk_kernel` 16 and `aten::min` 16, decode's rotary path.
2999 - 416 - 144 = 2439, with no remainder.

So the gap is not scope, not compile mode and not a meta substitution: it is
two different step kinds, and the 2439 graph is a cost artifact that was never
a candidate for the warmup allocation graph. The prefill graph is the one
shaped like `warmup_model`'s dummy run -- one request, `max_num_batched_tokens`
query tokens, no history -- which is what the memory work needs. Two caveats
stay attached to it and are **not** closed by this: it is traced at
`compilation_level: 0`, where the native warmup runs the compiled region, and
its scope excludes `compute_logits` and the sampler, which the native peak may
not.

## What the fused operators actually promise

The collective audit's method -- registered schema and implementation, never
the name -- applied to computation. The schemas below are this process's own
dispatcher entries, read on CPU
(`agent_scratch/memval/lifetime/schema_audit.py`), against the operators the
TP=1 graph actually records.

| operator | schema | what it means |
|---|---|---|
| `aiter::silu_and_mul` | `(Tensor(a0!) out, Tensor(a1!) input, float limit) -> ()` | destination-passing, returns nothing. The graph records `output_shapes: []` and the destination in `inputs_from`: counted once, correct |
| `aiter::_fused_qk_rmsnorm_group_quant_kernel` | ten optional `Tensor(aN!)` destinations `-> ()` | same shape, same correct record |
| `aiter::gemm_a16w16` | `(Tensor(a0!) A, Tensor(a1!) B, ...) -> Tensor` | **unannotated return**: the output is fresh, not an alias. Counted once, correct |
| `aiter::linear_attention_with_output_base` | `(Tensor mixed_qkv, Tensor b, Tensor a, Tensor core_attn_out, str layer_name) -> Tensor` | `mutates_args=[]`, and the implementation is `ret = torch.empty_like(core_attn_out)` (`atom/model_ops/base_attention.py:403`). The name promises destination-passing and the source refuses it: `core_attn_out` and the return are two live buffers, on the device as much as in the graph |
| `aiter::unified_attention_with_output_base` | same shape, no alias annotations | same |

**No double count was found.** Every custom operator in the TP=1 graph either
records no output and writes into a buffer the trace already counted, or
records a fresh output the implementation really does allocate. The walk's
treatment of the fused operators is right, which is worth stating as plainly as
a defect would have been.

Two things did fall out of reading the schemas. First, `mutates_args="unknown"`
-- the default in `aiter/jit/utils/torch_guard.py` -- marks *every* tensor
argument `Tensor(a!)`, including `gemm_a16w16`'s weight matrix `B` and
`fused_allreduce_rmsnorm_`'s norm weight `w`. A producer that read mutability
off these schemas would conclude the model mutates its own weights. **In this
registry `(a!)` is not evidence of mutation.** Second, the asymmetry that *is*
usable: an operator returning a tensor that aliases an argument would have to
say so in the return annotation, and none of them do. `aiter::all_reduce_
(Tensor(a0!) tensor, str group_name, ...) -> Tensor` declares an unaliased
return, which is the schema agreeing with the implementation's own comment that
the all-reduce is out-of-place -- the collective correction now rests on the
schema as well as the source. `aiter::fused_allreduce_rmsnorm_(Tensor(a0!) inp,
Tensor(a1!) res_inp, Tensor(a2!) w, ...) -> (Tensor, Tensor)` returns two
unaliased tensors; it does not appear in the 27B's graph at any width, so it is
recorded here and not modelled.

## The lineage classification does not yet align these graphs

Run over the real TP=1/2/4 derived graphs, `width_coverage` reports 71 of 3014
outputs aligned across all three widths -- 2.4% -- and 2943 unaligned at the
base width. Every tensor live at the TP=2 and TP=4 peaks is `unaligned`. The
classification is therefore **not usable yet**, and the honest reading of the
`width_class` field in the current candidate is that it is populated at TP=1
and empty of meaning at width.

The first run of this check reported 98, from a defect in the key it was
checking. `lineage_keys` interned each distinct ancestry to `L0`, `L1`, `L2`
-- the *order* the ancestry was first met in -- so two graphs agreed whenever
they happened to meet the same number of distinct ancestries first, which is
index alignment wearing a different name and exactly what the function was
written to avoid. The key is now a digest of the ancestry itself. The
correction lowered the coverage rather than raising it, which is the direction
worth noticing: a defect that flatters a number is the kind that survives.

The cause is in the key, and the source names it. `VocabParallelEmbedding.
forward` (`atom/model_ops/embed_head.py:168-178`) branches on width: `tp_size >
1` takes `masked_embedding` followed by an all-reduce, and `tp_size == 1` takes
`F.embedding`. So operator 0 is `aten::embedding` at TP=1 and
`aiter::masked_embedding` at TP=2 and 4 -- the same module, two operator names
-- and `lineage_keys` interns the name into the ancestry key, so the divergence
propagates to every descendant. One width-conditional branch at the root
poisons the whole graph. It also produces a wrong answer where it does align:
the silu destination `aten::empty.memory_format [16384, 17408]`, which visibly
becomes `[16384, 8704]` at TP=2, is classed `replicated`, meaning it aligned
against something that is not its counterpart.

The fix is not a bigger equivalence table. Operator names are the wrong
identity because ATOM's width behaviour is a property of *modules*: the module
path -- `model.layers.31.mlp.gate_up_proj` -- is the same string at every
width, and the module decides both the branch and the sharding. `OpSpec`
carries no module field today, and `@mark_trace` (`atom/utils/decorators.py`)
already wraps every module forward with a named region, so the information
exists at trace time and is thrown away. Stamping a module path per operator,
aligning within a module's span, and dropping collectives from the ordinal --
collectives being exactly what a width adds -- is a derivation over ATOM's own
structure and needs no comparison with any measured peak.

Nothing above changes a byte of the candidate: `candidate_bytes` is still
{1: 2 956 984 320, 2: 2 101 346 304, 4: 1 673 527 296}, each width's own walk.

## The TP=1 target was preserved, and it is not any record we hold

O24 asked how a TP=1 target cell could be assembled honestly. It need not be:
node 18 preserves two genuine TP=1 target/table pairs for this configuration,
`agent_scratch/g4/cap_subspan/` and `agent_scratch/poc/g5_27b/`, read-only and
the lead's, and CC has the first of them. Nothing has to be synthesised, and
the rest of this section is about a mistake this worker made while checking
that, because the mistake is the reusable part.

The pairs carry what the committed memory records lack:

    run.server_code_sha256  dafd70f14ee5eac5e37b75ce9df9186f679423223c074ba44...
    run.model_revision      hf:1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    hardware.arch           gfx942:sramecc+:xnack-
    hardware.device_name    AMD Instinct MI308X   device_count 1
    hardware.torch_version  2.10.0+rocm7.2.4.git3d3aa833
    server.visible_devices  5

and lack what the records carry: `target.json` has `config`, `hardware`,
`blocks` and `graph` and no `readings`, and `capture.json` names no memory term
at all. Those runs' memory numbers survive only as the engine's budget line in
`capture.server.log`, at two decimals -- `peak_torch=53.99GB, non_torch=1.08GB,
cudagraph_est=0.55GB, total_gpu=191.98GB, block_bytes=1056768,
num_kvcache_blocks=112772`.

### Eight of eight is not an identity, and this document already said so

`27b.tp1.exclusive.memory.json` agrees with that target on every field the two
share: exactly on `num_kvcache_blocks` 112 772, `pool_entries {state: 32, kv:
112772}`, graph pool 127 926 272 B and capture sizes `[1,2,4,8,16,32]`, and to
the log's two decimals on `total`, `peak_torch`, `non_torch` and
`cudagraph_overhead`. Eight of eight, including a block count derived from free
memory after every other term.

This worker read that as a join -- "the record *is* that run" -- and it is not
one. Numeric agreement cannot establish run identity, and the evidence against
it is two sections above: *Identity is not integrity* is the finding that the
three phase C repeats produced **byte-identical payloads from demonstrably
distinct runs**. A reading that repeats exactly is what a stable configuration
on a quiet device looks like; it is not a fingerprint, and treating a
sensitive-looking derived field as one is the same error in a more flattering
costume.

Timestamps settle it, and they settle it the other way:

| artifact | when it ran | on what |
|---|---|---|
| `poc/g5_27b` | 2026-09-10T13:12:42Z (`compass_config.epoch`) | `visible_devices "5"` |
| `g4/cap_subspan` | 2026-09-11T04:37:55Z (`compass_config.epoch`) | `visible_devices "5"` |
| phase C repeats -> `27b.tp1.exclusive` | 2026-09-11T08:53:49Z..09:03:38Z (`ownership.txt`) | this agent's own container |

Disjoint: four hours after the one, a day after the other. The exclusive record
is **not** either target run, and the right label for the relationship is
*compatible independent evidence at the same configuration* -- two captures of
the same knobs on the same chip that agree, which is worth having and is not an
identity. `27b.tp1.memory.json` has no provenance at all and is unproven in
both directions.

### The 32 MiB is an observation, not a neighbour

The two committed TP=1 records differ by `non_torch` +33 554 432 B (32.0 MiB)
and `num_kvcache_blocks` -32, and 32 MiB / 1 056 768 B per block is 31.75,
floored to 32 -- so the block difference is *arithmetically downstream* of the
`non_torch` difference. That is all that is established. This worker wrote that
a neighbour's allocation caused it; nothing here shows that. The ownership
sample belongs to the phase C run and shows a foreign KFD attachment holding
0 B throughout, so it cannot speak for whatever the other record's device held,
and no sample was taken during that run at all. The cause is open, and O11's
486 MiB `non_torch` excursion is the older form of the same unanswered
question.

### One physical executor is a mechanism, not a TP=1 ceiling

An earlier note here treated `replay/local_proc.py:43` -- one rank in one
process -- as fixing device-free replay at TP=1. It does not: one physical CPU
executor is the intended emulation mechanism, and a *logical* TP=2 or TP=4
prediction may run on it provided the target topology, the per-rank geometry
and pools, the per-rank cost semantics and the provenance of each are
explicitly right. What it must not do is perform or present itself as real
distributed GPU work, and it must not carry TP=1's capacity -- 112 772 blocks
for a rank that owned a whole MI308X -- into a wider cell. Where that line
falls is the lead's to set, not this guard's.

## The allocation history: what `peak - current` is made of, by address

The activation gate quantity is unchanged and stays `peak_torch - current_torch`
= 2 956 984 320 B. Nothing below replaces it. What the bounded TP=1 probe adds
is the *composition* of that number: which storages are live when the allocator
reaches its high-water mark, identified by address rather than by a count of
buffers assumed to be preallocated.

The probe wrapped `ModelRunner.warmup_model` in the engine worker, recorded the
live set immediately before the forward, ran `_record_memory_history` across it,
and read the allocator's own counters afterwards. Reader:
`agent_scratch/memval/producer_packet/tp1_probe/read_alloc_probe.py`.

### The replay reconciles at both endpoints, and the second one cost a correction

Replaying 1012 allocations and 1011 frees on top of the 1013 blocks that
predate the forward gives a running total. Two points on that curve are known
independently, from `memory_stats()`:

| | replay | allocator |
|---|---|---|
| end | 55 014 276 096 | 55 014 276 096 |
| peak | 57 971 260 416 | 57 971 260 416 |

The first attempt reconciled the end exactly and missed the peak by 1 048 576 B,
and the reason is worth keeping. The trace records the size *requested*; the
allocator counts `block->size`. Those differ when a fresh large segment cannot
be split, because the large pool only splits when the remainder exceeds
`kSmallSize` (1 MiB) -- a remainder of exactly 1 MiB is absorbed into the block
and charged. Two segments in this run are in that state: a 541 065 216 B segment
serving a 540 016 640 B request, and a 2 097 152 B segment serving a 1 048 576 B
request. Charging the segment in exactly those two cases moves the replayed peak
from event #132 to event #174 and onto the allocator's figure.

The consequence for the model is small and real: **`peak_torch` includes 1 048 576 B
that corresponds to no tensor.** A walk that sums tensor bytes cannot reproduce
the gate quantity exactly. It is 0.035% here, and it is a floor on how close any
shape-derived walk can get without modelling segment rounding.

### The named preallocated buffers are all outside the gate, and that is now an address match, not a subtraction

All 29 entries of `forward_vars` are live at the peak, and **every one of them
has an address in the pre-forward live set**. Together they are 172 704 716 B,
and none of it is in `peak - current`. This is what O19 asked for: the seed set
for a walk's seen-storages is not a guessed buffer count, it is these addresses.

The sharpest case is `outputs`. It is 167 772 160 B, `[16384, 5120]` bfloat16, at
address 139699547013120, preallocated and live throughout. During the forward
`embed_head.py:177` allocates *another* 167 772 160 B block, at address
139693404454912, and that one is in the gate. Two buffers of identical size and
shape, one inside `current_torch` and one inside the gap, distinguishable only by
address. A rule that recognises the preallocated output by its shape would
suppress the wrong one.

### Eight blocks at the peak, and three of them are allocated inside an opaque operator

| bytes | allocating frame |
|---:|---|
| 1 140 850 688 | `aiter/tuned_gemm.py:450 torch_gemm` |
| 570 425 344 | inductor `c5fb7q7n...py:1227 call` |
| 541 065 216 | `aiter/tuned_gemm.py:450 torch_gemm` |
| 201 326 592 | `atom/model_ops/base_attention.py:403 linear_attention_with_output_base` |
| 167 772 160 | `atom/model_ops/embed_head.py:177 forward` |
| 167 772 160 | inductor `cwfrpmhu...py:322 call` |
| 167 772 160 | inductor `c5fb7q7n...py:1212 call` |
| 79 691 776 | `aiter/tuned_gemm.py:450 torch_gemm` |

Total 3 036 676 096 B across 8 blocks; no block that predates the forward is
freed during it. 1 761 607 680 B of the peak -- 58% -- is allocated inside
`aiter::gemm_a16w16`'s implementation, which is the place a dispatch trace
cannot see into. That is the region O16 named as the candidate for the sharded
fraction, and it is now located rather than suspected.

### The two components, and why neither is the gate

One block survives the step: 79 691 776 B, allocated at `tuned_gemm.py:450` and
still live when `current_torch` is read. So

    3 036 676 096  allocated in the forward and live at the peak
      - 79 691 776  still live at the end, and therefore inside current_torch
    = 2 956 984 320  peak - current

exactly. Peak-before-baseline and retained-after are the two explanatory
components of that identity. They are not acceptance metrics and no gate is to
be restated in terms of them; they say where the gate's bytes come from and
where the 79 691 776 B the gate does not see has gone.

### A 32 MiB segment is released inside warmup, and the size is a coincidence until it is not

The first event in the trace is `empty_cache` at `model_runner.py:1203`
releasing a 33 554 432 B segment -- warmup's own bracketing, untouched by the
probe. A record taken after that release, with the device reading unchanged,
would show `non_torch` 32 MiB higher and the block count correspondingly lower,
which is the exact shape of the gap between the two committed TP=1 records
(O24: `non_torch` +33 554 432, blocks -32). This is a mechanism of the right
size in the right phase. It is not a demonstration that it is *the* mechanism,
and the two records still have no run identity to join them on. O11 and O24 stay
open on the same terms as before.

### What this run does and does not establish about perturbation

The record this run wrote reproduces the exclusive S27 record term for term, and
the `after` counters match the record the same run emitted. The second of those
is internal consistency and proves nothing about the probe. The first says the
probe perturbed no *budgeted* quantity -- the same weights, `non_torch`,
cudagraph overhead and 112 772 blocks came out. It is not a proof that all
behaviour was unperturbed, and it is a fourth byte-identical payload, which is
one more reason not to read byte-identity as identity.

## The module path aligns the widths; the ancestry never could

O23 said `lineage_keys` aligned 71 of 3014 outputs across the three real
widths. Re-measured on freshly captured graphs it is 12 of 3014, and the
diagnosis in O23 was only half right. `VocabParallelEmbedding` emitting two
operator names is real, but it is not the reason the alignment is this bad.

The reason is that at width the graph does not know its own ancestry. Source
edges recorded as `-1`:

| width | source edges | unknown | operators with an unknown source |
|---|---:|---:|---:|
| 1 | 4378 | 596 (13.6%) | 563 |
| 2 | 4378 | 725 (16.6%) | 692 |
| 4 | 4378 | 725 (16.6%) | 692 |

An ancestry key is recursive, so one unknown producer is not one lost
operator -- every descendant inherits the break. The extra 129 unknowns at
TP>1 come from the collective stand-in returning its own input (O15) on top of
`_storage_of` collapsing on meta tensors (O14), both in files this worker does
not own. No key built on `inputs_from` can align these graphs until those are
fixed.

### What replaced it, and what stops it being index alignment in disguise

`capture_lifetimes.py` now stamps each operator with the module it ran inside,
via global `nn.Module` forward hooks, and writes it as a sidecar
(`<graph>.modules.json`) rather than a new `OpSpec` field -- the graph schema
is shared and not this worker's to change. 772 spans, and 2999 of 2999
operators attributed at TP=1, 3128 of 3128 at each of TP=2 and TP=4.

`module_path_keys` keys an operator on its module path and its ordinal among
the *non-collective* operators of that module. A collective consumes no
ordinal, or every operator after the first all-reduce would be renumbered at
exactly the widths where all-reduces exist.

That is ordinal alignment inside a module, which is the thing `lineage_keys`
was written to avoid, so it does not ship unguarded. `alignment_integrity`
runs two structural checks and `width_classes` refuses the keys unless both
pass:

* every module path holds the same count of non-collective operators at every
  width. Here: 2999 at all three, and **no path disagrees**;
* where ancestry survives at every width -- no unknown source on either side,
  collectives walked through -- it must agree with the ordinal alignment.
  Here: **2307 agree, 0 contradict, 692 not checkable.** The unknowns are
  counted apart from the agreements, because a check that could not run is not
  a check that passed.

**3014 aligned outputs is coverage, not correspondence.** It is the count of
outputs the ordinals paired up. Whether each pair is the same operator is a
separate question, and 692 of the 2999 operators behind those outputs had no
readable ancestry to answer it with. Zero contradictions among the 2307 that
could be checked is evidence about those 2307 and about nothing else.

| | ancestry key | module-path key |
|---|---:|---:|
| aligned | 12 | **3014** |
| unaligned at base | 3002 | 0 |
| replicated / sharded / unresolved | 11 / 1 / 0 | 1382 / 1632 / **0** |

Exactly one aligned pair joins operators with different names, and it is the
one O23 predicted: `aten::embedding` at TP=1 against `aiter::masked_embedding`
above it, `[16384, 5120]` at every width, classified replicated. A join the
module tree licenses but the names do not is recorded on the entry as `names`
and counted in coverage as `renamed`, so it is visible rather than silent.

### What the 692 unreadable alignments actually are

"Not checkable" was a placeholder, not a finding. `-1` in `inputs_from` is
documented as *"a weight, an embedding table, a buffer allocated before the
forward"* -- an input from outside the graph, which is a real thing two
aligned operators can be compared on -- but the tracer writes the same `-1`
when it loses a producer it should have recorded, and those two are not the
same evidence. The artifact kept no way to tell them apart, so the first pass
guessed from the shape, which collides: a `(5120,)` input is a norm weight in
one operator and a hidden-state row in another.

`capture_lifetimes.py` now records the *identity* of every input whose storage
belonged to a parameter, a registered buffer, or a tensor the runner built
before the forward, and writes it as a second sidecar
(`<graph>.origins.json`). The storage is in hand at dispatch, so the name is
too. Over the three real captures, every `-1` edge:

| | TP=1 | TP=2 | TP=4 | what it is |
|---|---:|---:|---:|---|
| parameter | 434 | 434 | 434 | named weight, e.g. `layers.7.mlp.down_proj.weight` |
| buffer | 32 | 32 | 32 | the rotary cos/sin cache, `[262144, 1, 1, 32]` |
| forward-input | 33 | 33 | 33 | token ids, positions, attention metadata |
| unnamed | **97** | **226** | **226** | a producer the trace did not record |
| total | 596 | 725 | 725 | |

The unnamed are not a residue of unknown character. Every one of them carries
the token dimension -- none is parameter-shaped -- and they fall into four
groups, each with its operator and the operator that ran immediately before:

| edges (TP=1 / TP>1) | operator | ran after | reading |
|---:|---|---|---|
| 0 / 130 | `aten::view` | `aiter::all_reduce_` | the collective's own output |
| 32 / 32 | `aten::cat` | `aten::add.Tensor`, `aten::cat` | rotary, in `self_attn.rotary_emb` |
| 32 / 32 | `aten::squeeze` | `aten::slice.Tensor` | rotary, same module |
| 16 / 16 | `unified_attention_with_output_base` | `aten::cat`, `aten::reshape` | attention's output base |
| 16 / 16 | `aten::sigmoid` | attention | storage the trace *had* seen |

The 130 at TP>1 are O15 with a name on it: `record_collectives` builds the
all-reduce's `OpSpec` by hand, outside the dispatch tracer, so the collective's
output never enters the producer map and its consumer reads `-1`. That is the
entire TP=1-to-TP>1 rise, 129 edges, and the fix belongs in the shared tracer.
The 16 `sigmoid` edges are a different defect: the storage *was* seen earlier,
so a producer entry existed and was dropped -- the finalizer in `_died` forgets
an address while a view of it is still alive. Both are tracer bookkeeping, not
model structure.

### Grading the alignment with the names

Names that survive are evidence. An operator reading
`layers.7.mlp.down_proj.weight` aligned against one reading the same parameter
at another width did not have to match, and a pair reading different
parameters would be evidence against the alignment. `alignment_integrity` now
takes the origins and reports three grades instead of two buckets:

| grade | count | what it rests on |
|---|---:|---|
| `ancestry_agrees` | 2307 | the producer chain agrees at all three widths |
| `origin_agrees` | **466** | the ancestry could not run; the named externals match |
| `origin_unresolved` | **226** | neither check reached it; the ordinals alone |
| `ancestry_contradicts` / `origin_contradicts` | 0 / 0 | |

2307 + 466 + 226 = 2999. So two thirds of the previously-unreadable 692 are now
checked by something, and **226 operators remain aligned on the module tree and
the ordinal alone** -- exactly the 226 whose inputs the tracer lost. That is the
honest confidence: not "the alignment is verified", but 77% ancestry-checked,
16% name-checked, 7% unchecked, 0 contradictions anywhere.

One comparison had to be made signature-aware rather than positional, and the
reason is the case O23 named. `VocabParallelEmbedding` calls
`F.embedding(weight, ids)` at TP=1 and `masked_embedding(ids, weight)` above
it: the same two externals, the argument positions swapped. Compared by
position that reads as a contradiction and would have ended the alignment
(`safe: false`) on an operator-signature difference. Positions are compared
only while the operator name is the same at every width; where the name
differs, only the set of names is comparable. A swap under an *unchanged*
operator name is still a contradiction, and `test_a_swapped_argument_order_is_
not_a_misalignment` pins both halves.

Equal per-path operator counts remain a guard, not a proof. A module that ran
the same number of operators at two widths can still have run different ones;
what the counts rule out is the specific failure where an inserted or removed
operator renumbers every ordinal after it.

### The withdrawn trailing-dimension rule got nothing wrong at the peak

With the widths aligned, the rule that was withdrawn can finally be scored
instead of argued about. Over the eight tensors live at the TP=1 walk's
high-water mark:

    the rule agreed    : 8 tensors, 2 717 908 992 B -- 100% of the peak
    the rule was wrong : 0 tensors, 0 B

and the derived width totals are identical to the rule's at TP=2 and TP=4,
to the byte. The rule *is* wrong elsewhere -- it would shard 224 aligned
outputs that do not shard, mostly rope and index tensors of shape
`[16384, 1, 1, 32]` -- but none of them is live at the peak, so none of them
is in the activation term. **The withdrawn rule is not the cause of O16.**
Removing it was right on its own terms, and it buys no accuracy.

### The walk's eight tensors and the allocator's eight blocks are not the same eight

The derived TP=1 walk peaks at 2 717 908 992 B over 8 tensors. The measured
allocator peak holds 8 blocks allocated in the forward, 3 036 676 096 B. Both
are eight. Matched by size:

| size | walk | allocator |
|---:|---:|---:|
| 1 140 850 688 | 1 | 1 |
| 570 425 344 | 1 | 1 |
| 541 065 216 | 0 | 1 |
| 201 326 592 | 0 | 1 |
| 167 772 160 | **6** | **3** |
| 79 691 776 | 0 | 1 |

    in both                                2 214 592 512 B
    walk counts, allocator does not          503 316 480 B   over-count
    allocator holds, walk cannot see          822 083 584 B   invisible

This answers O2's standing question directly. Six hidden-width buffers are
live at the walk's peak and only three are live on the device: the walk counts
three that are not there, 503 316 480 B, and `forward_vars["outputs"]` is one
of them by address (O19). Against that, 822 083 584 B is allocated where a
dispatch trace cannot look -- 541 065 216 B and 79 691 776 B inside
`tuned_gemm.py:450`, 201 326 592 B in `linear_attention_with_output_base`.

The walk lands 239 075 328 B under the gate quantity, -8.1%. That number is
two errors of opposite sign partly cancelling, 503 MB against 822 MB. Any
correction fitted to the -8.1% would be fitting a difference of two unrelated
mistakes, which is the specific thing O16 says not to do.

## The graph pool: what capture pins is the LM head, and the width was never the rule

O8 reads the pool term +26.8% high at TP=4. The term it reads is a line fitted
to six 0.6B ladders at width one and a flat 104 MiB above it, and the flat part
was justified by an observation with no mechanism under it: the *allocated*
delta at TP=2, 4 and 8 was 79 692 800 B over ladders from 31 to 1071 tokens,
identical to the byte. The reading taken from that -- "the graphs are not
pinning sharded activations" -- was the wrong half of the story.

### The runner captures the LM head only at width one

    self.logits_in_graph = self.world_size == 1 and not is_tbo   # :4104
    ...
    model_output = self.model(input_ids[:num_tokens], model_positions)
    outputs[:num_tokens] = model_output                          # :4237
    if self.logits_in_graph:
        graph_logits = self.model.compute_logits(outputs[:num_tokens])  # :4297
    ...
    self.graph_logits[(bs, max_q_len)] = graph_logits            # :4305

(`atom/model_engine/model_runner.py`.) Three things follow, and all three are
readable from source rather than from a device.

* The logits tensor is allocated *inside* `torch.cuda.graph(...)`, so it comes
  from the graph's private pool, and the runner keeps it in a dict, so it stays
  live after capture. It is pinned, per bucket.
* Capture builds decode metadata, so `ParallelLMHead.forward`
  (`atom/model_ops/embed_head.py:243`) takes no last-token index: the tensor is
  `[num_tokens, vocab_size]` whole. At TP>1 it would also be all-gathered to
  full vocabulary -- but at TP>1 it is not captured at all.
* The model *output* is not in the pool. `outputs[:num_tokens] = model_output`
  writes into the preallocated `forward_vars["outputs"]`, the same buffer as
  O19, which was allocated at engine init and is already inside `current_torch`.

So the pinned set is a residue plus `vocab_size x dtype_bytes x Σ(captured
num_tokens)` when the head is in the graph, and the residue alone when it is
not. On the 27B's TP=1 record, with the vocabulary from the checkpoint's own
`config.json` (248 320, under `text_config`):

    110 981 120 - 63 x 248 320 x 2 = 79 692 800

The residue is **the same 79 692 800 B** that the 0.6B showed at TP=2, 4 and 8.
Two models that differ by 45x in parameters, three widths, ladders from 31 to
1071 tokens: the same number to the byte. It is not the model, not the width and
not the ladder.

What it *is* has not been witnessed. Invariance is a strong constraint on the
explanation, and it is not the explanation: 76 MiB + 1 KiB allocated once inside
the capture window could be a reusable workspace, an allocator size-class
rounding, or a block the pool keeps per capture. So the number is carried as
**source calibration** -- taken from the TP=1 source record (S27), labelled as
taken rather than derived -- and it stays that way until a pool-scoped
allocation probe names the mechanism. `CAPTURE_FIXED_PINNED` is a calibratable
argument for exactly that reason: a caller holding the mechanism passes its own.

### What that buys, with no constant fitted to a target

`capture_pinned_bytes` takes the residue from the TP=1 record (S27) and the
vocabulary from the checkpoint, and has no free parameter left. Against the
recorded allocated deltas:

| record | predicted | recorded | error |
|---|---:|---:|---:|
| 27B TP=1 | 110 981 120 | 110 981 120 | **+0.0%** |
| 27B TP=1 exclusive | 110 981 120 | 110 981 120 | **+0.0%** |
| 27B TP=2 rank0 | 79 692 800 | 79 692 800 | **+0.0%** |
| 27B TP=4 rank0/1 | 79 692 800 | 79 692 800 | **+0.0%** |

The first row is exact by construction -- it is where the residue came from --
and the second is an independent engine start of the same configuration. The
TP=2 and TP=4 rows are class X27 and are read here as **evaluation, not input**:
nothing in the derivation saw them, and the prediction at those widths is
whatever `logits_in_graph` says, which is the residue.

Two limits on how much those four rows are worth. They were checked against
records that were already on disk when the term was written, so the agreement is
**retrospective evidence, not a fresh frozen evaluation** -- the term was not
registered and then met by a run made afterwards, and only the second kind
closes a prediction. And all four are the capture-time **allocated** delta,
which is not the quantity O8 reads at +26.8%: that is `graph_pool.reserved`.
Agreement at +0.0% on the allocated side leaves the reserved-pool gate exactly
as open as it was.

The switch is the predicate, not the width. A TP=1 run with TBO enabled also
drops the head, and a model keyed on `world_size == 1` would over-read it by the
whole ladder term -- 31 288 320 B on this ladder. That is why the new term takes
`tbo` and refuses (`UnfoundedPrediction`) rather than guessing when the head is
captured and no vocabulary was supplied.

**For the cost model, not just the memory model**: at TP>1 `compute_logits` is
outside the graph, so every decode step pays an eager LM-head GEMM plus an
all-gather that the TP=1 replay does not. That is a per-step cost difference
that follows from the same predicate.

### Why the number it was being checked against is not the pool

The recorded `graph_pool.reserved` is a **global** difference:

    _rsv_before_capture = torch.cuda.memory_reserved()            # :4120
    ...
    _pool_bytes = max(torch.cuda.memory_reserved() - _rsv_before_capture, 0)  # :4346

and the window between them contains a full **eager** warmup forward per bucket
(`:4229`), whose segments grow the ordinary pool; in piecewise capture
`torch.cuda.empty_cache` is patched to a no-op inside it, and `pause_gc`
disables the collector across the whole loop. Anything released elsewhere in the
process lands in it too, and `max(..., 0)` reads a net release as a pool of
zero. Across the four 27B records the gap between reserved and allocated is
16.2, 26.0 and 6.0 MiB at TP=1, 2 and 4 -- it does not scale with the ladder,
the width or the pinned set, which is what a bookkeeping term looks like.

Pool-scoped residency does not have to be inferred from a global delta:
`torch.cuda.memory_snapshot(mempool_id)` and `MemPool.snapshot()` take exactly
the id that `graph.pool()` returns. That is the probe that would identify the
79 692 800 B residue -- 76 MiB plus 1 KiB, allocated once inside the capture
window, model- and width-independent, and still unidentified. It is a probe to
coordinate, not a constant to widen.

### Two capture modes, two pool topologies, one term

The reserved side also has a structural reason not to be one number.
PIECEWISE capture takes **one private pool per `num_tokens` bucket** by default
(`ATOM_PER_BUCKET_POOL=1`, `atom/utils/cuda_graph.py`), because sharing one pool
across buckets corrupts DeepSeek-V4 decode -- the module header carries the
accuracy measurements. FULL capture takes the opposite topology: the first
graph's pool becomes `self.graph_pool` and every later bucket captures into it
(`model_runner.py:4301`). Per-bucket pools cannot reuse each other's freed
blocks; one shared pool can.

The header also records what that costs on DSV4 TP8: 1.11 GB per rank of
*reserved*, with the capture-time **allocated** delta identical at 14.71 GB
either way. Topology moves the bookkeeping and leaves the pinned set alone,
which is the second reason to model the pinned set and report the reserved
delta rather than predict it.

## Open items

| # | item | needs | status |
|---|---|---|---|
| O1 | `parameters` / `buffers` for the 27B at TP=1/2/4 | meta build | **closed** -- exact at all three widths, finding 5 |
| O2 | 27B activation trace at TP=1, prefill-shaped | GPU, TP=1, one prefill | **closed as a constant, closed as an accounting, open as a mechanism** -- the term is 2 956 984 320 B, exact across two independent engine starts. The allocation curve asked for here has now been read: 8 blocks live at the allocator peak, of which three are the hidden-width buffers where the walk holds six, so 503 316 480 B of the walk is preallocated or never allocated and `forward_vars["outputs"]` is one by address (O19). The reverse error is larger: 822 083 584 B is live on the device and invisible to the walk, allocated inside `tuned_gemm.py:450` and `linear_attention_with_output_base`. Net -239 075 328 B, -8.1%, and it is two errors of opposite sign, so no correction is to be fitted to it (O16) |
| O3 | source-only candidate budget, frozen, vs the recorded budget | O2 | **closed at TP=1** -- every input classed S, C06 or S27, and the budget reproduces the source run's 112 772 blocks. A residual by construction: it is not evidence of transfer, and the widths where transfer would be tested need O2's activation term at TP=2 and TP=4, which is class X27 today and therefore underived |
| O4 | `persistent` -- was 51% low | -- | **closed by calibration**, exact at TP=1, -0.004% at TP=4; a mechanism would still be better than a constant |
| O5 | `persistent` / activations / pool as functions of `max_num_seqs` | GPU, TP=1 | open, and now the main conditionality left; all three are proven flat in *utilization* (phase A) but untested in concurrency |
| O6 | physical start-up at `--max-num-seqs 1551` and 1400 | GPU, TP=1 | open; superseded as the acceptance gate by the utilization axis, kept as a diagnostic |
| O7 | `non_torch` from an exclusive-device source run at util 0.90 | GPU, exclusive | **closed by phase C** -- three byte-identical runs, calibrated at TP=1, exact at three unseen utilizations; the +42.2% excursion it cannot bound is carried with it |
| O8 | graph pool at TP=4, where the model reads +26.8% | GPU, 4 devices | **open for the reserved delta, closed for the pinned set.** The +26.8% is against `graph_pool.reserved`, which is a global `memory_reserved()` difference across a window containing an eager warmup forward per bucket and a patched-out `empty_cache` -- not private-pool residency. The pinned set has a mechanism now: `capture_pinned_bytes` = fixed residue + `vocab x dtype x Σ(captured tokens)` when `logits_in_graph` (`world_size == 1 and not is_tbo`, `model_runner.py:4104`), which reproduces the recorded allocated delta at **+0.0% on all four 27B records**, the TP=2/TP=4 rows being evaluation (X27) against a derivation that never saw them. The old candidate cause -- a derived-graph walk counting each `all_reduce_` as an immortal allocation (O15) -- is not in this path at all: the recorded pool is measured, not walked. The four +0.0% rows are **retrospective evidence, not a fresh frozen evaluation** -- the records predate the term -- and they are the *allocated* delta, not the reserved one this item reads. The residue is now **witnessed** on the source config: a pool-id scoped capture probe (S27, TP=1, node18 GPU0, `agent_scratch/memval/pool_probe/att2_artifact.json`, att2 rc=0) diffed the *global* segment list across the capture window. Nothing was released; six segments appeared. Private pool `(1, 0)`, five counters kept separate: reserved residency 46 137 344 B, active allocated 31 288 320 B, active requested 31 288 320 B, internal rounding **0 B**, inactive capacity 14 849 024 B. The pool active bytes are six blocks, one per bucket, each exactly `bs x 248 320 x 2` for keys (32,1) (16,1) (8,1) (4,1) (2,1) (1,1) -- so the logits term is witnessed per bucket and rounds by nothing. The 79 692 800 B residue is **not pool residency**: it is one 79 691 776 B (76 MiB) block with `requested_size == size` in its own oversize segment, plus two 512 B blocks of an 8 B request each, all **outside every capture pool** -- an exactly-76-MiB fixed request is why the same number appears on the 0.6B at TP=2/4/8 across ladders. The TP=1 warmup history already on disk carries a 79 691 776 B `segment_alloc` framed `aiter/tuned_gemm.py:450:torch_gemm` under the GDN linear-attention forward -- but that settles nothing in either direction, and neither does factoring the size -- 2432 is not a width this checkpoint produces. What is witnessed of the *warmup* block is its life: allocated at the second event of the window and still live when it closes, it is the window's entire net allocated retention (79 691 776 B). `tuned_gemm.py` has no workspace in it at all and `torch_gemm` ends in `F.linear`, so what it returns is a GEMM output `[M, N]` with `N` a weight width -- of the config widths only `in_proj_qkvz` = 16 384 divides 39 845 888. The *capture-window* block is a separate observation: no frames, and all six segments new in that window sit on the capture stream 460554448 against 313 pre-existing segments on stream 0, i.e. requested on the side stream capture runs on and outside the graph pool. A per-stream cache would explain a fixed size allocated again after warmup; so would other things. It stays unattributed, and nothing waits on it -- the term is source-calibrated by construction. **What is settled and matters for the gate**: the warmup block is retained, so it sits in `peak` and in `current` alike and cancels in `peak - current`; it is excluded there once and carried nowhere else. Reserved side, separately: the window reserved delta 127 926 272 B = 46 137 344 in the pool + 81 788 928 outside (the 76 MiB oversize segment plus a whole 2 MiB small segment holding only 1 024 B). So the allocated constant 79 692 800 and its reserved cost 81 788 928 are different numbers. This is source-only diagnostic evidence that *explains* the global reserved gate; it does not redefine it, and it is not target validation. The reserved delta this item reads at +26.8% remains open. **The pool's contents are now named in config widths, and the missing input is named too** (`agent_scratch/memval/pool_probe/pool_layout.py`). Every one of the nine distinct request sizes the pool holds, live and dead alike, resolves to `bs x <a config width> x 2 B` with no fitted parameter: the six live blocks are 496 640/993 280/1 986 560/3 973 120/7 946 240/15 892 480 B = `bs x vocab_size` at bs = 1/2/4/8/16/32, one logits tensor per capture, which is what `capture_pinned_bytes` already models; the dead ones are 1 114 112 = `32 x intermediate_size`, 458 752 = `32 x (hidden_size + key)` and 10 240 = `1 x hidden_size`, decode-forward transients freed before the window closed. So the transient residency's *widths* are source-derivable. What is not derivable from a final snapshot is **which request forced the allocator to map each segment**, and the artifact shows the gap directly: a 20 971 520 B segment holds a 15 892 480 B block whose own sizing rule gives `kRoundLarge x ceil(n / kRoundLarge)` = 16 777 216, so that segment was mapped for an earlier, smaller request and this block landed in it after a free. Reserved is set by the interleaving of requests and frees, and a snapshot reports only the set that survived -- which is why the same rule over the pinned blocks alone reads 83 886 080 B against 46 137 344 B observed. The ordering is exactly what an allocation history is, so a bounded **TP=1-only** capture-history probe is prepared in `agent_scratch/memval/producer_packet/capture_probe/` (`REQUEST.md` md5 `d3bdc3205b3c8da42585d71a632a02d7`, `capture_hook.py` md5 `d61dbe24fe53e97e6c2c64634c7fdef2`): `_record_memory_history` enabled immediately before the capture loop and disabled immediately after, frames on, and a hard refusal to record at `world_size != 1` so the artifact cannot become class X27. It is a source-side calibration input, class S27, and explicitly **not** a request for a TP=2 or TP=4 full-engine calibration. It also closes the one thing the pool probe left open -- that probe recorded no frames, so the capture-window 76 MiB block has no stack of its own and its identification with the `tuned_gemm` workspace is inference by exact size from a different window. The reserved delta this item reads at +26.8% remains open until that replay either reproduces 46 137 344 B from these widths or names the segment it fails on. |
| O9 | `MODEL_HEADROOM` provenance: which run, which config | lead / history | open; until then it stays disallowed and is not to be relabelled as source |
| O10 | manifest-derived acceptance lengths | final CC workload | **closed** -- CC protocol `47917ade`: long 107 328 + 2 413 (6 859 blocks), short 2 560 + 21 (162) |
| O11 | what the 486 MiB `non_torch` excursion was | unknown; three controls failed to reproduce it | open, and the one thing the calibrated `non_torch` does not bound |
| O12 | a `run.execution` block in the memory record, carrying CC's `compass.execution/1` `execution_id` and its `id_inputs`, written by `_write_memory` | **lead** -- `_write_memory` is shared | open; until it lands, every calibrated row reads `residual (run unidentified)` and the phase C repeats cannot be machine-checked. The identity is CC's, not a second scheme: `atom/compass/core/execution_id.py` holds the one definition, stdlib-only, and `producer_key` reads it. `_write_replay_target` already writes a `hardware` block in the same neighbourhood, so the shape is precedented |
| O13 | tensor lifetime at the **source** width, and the sharded/replicated derivation of it for TP=2 and TP=4 | CPU only -- done | **restated and partly closed.** The original item asked for a TP=2/TP=4 warmup capture; that is a target measurement under a source label and is withdrawn. Lifetime is now captured at TP=1 on no device (2999 ops, 2790 deaths) and the width mechanism is read off the shapes. The frozen candidate reads +21.5% at TP=2 and +40.4% at TP=4 against the class-X27 peaks, so the *mechanism* is open: see O16. Three instrumentation defects found on the way (D1-D3), all in files this worker does not own  **Width alignment is no longer the blocker**: the module-path key pairs up all 3014 outputs and the integrity check grades the correspondence -- 2307 ancestry-checked, 466 name-checked, 226 on the module tree and ordinal alone, 0 contradictions (O23) -- 2436 / 466 / 97 when the same grading is re-derived on the lead runtime `cd718ef6`; and the withdrawn trailing-dimension rule, now scorable, gets all 8 tensors at the walk peak right -- 100% of 2 717 908 992 B, 0 wrong, +0.0% at TP=2 and TP=4. It is wrong on 224 aligned outputs elsewhere, none of them live at the peak, so it is eliminated as a cause of O16 rather than reinstated. |
| O14 | `_storage_of` returns 0 for every meta tensor (`runtime/meta.py`), so alias and provenance tracking collapse on any derived graph | **lead** -- shared runtime | open; fix is `untyped_storage()._cdata` when `data_ptr()` is 0, negated so it cannot collide with a device address. Patched locally in `agent_scratch/memval/lifetime/capture_lifetimes.py` |
| O15 | the collective the derivation records has no output tensor of its own: nothing is watched, so it can never die, and the meta stand-in (`_collective_stand_in`) returns the *input object*, which is the opposite error | **lead** -- shared runtime | open. Cost: 21.9 GiB against a true 2.5 GiB at TP=2, and **it sits under every derived-graph memory walk at TP>1, O8's graph pool included**. The in-place reading is withdrawn: the live implementation allocates a fresh output on every path (packet P2). **Now witnessed by name rather than inferred**: at TP>1 the 130 aligned `aten::view` operators that read the all-reduce output have no recorded producer and no external identity, which is the entire 596 -> 725 rise in unresolved source edges and the largest of the four unnamed groups in O23. **That group is now gone on the lead runtime.** Re-deriving on an immutable `cd718ef6` snapshot, unpatched, the unresolved source edges are 596 at TP=1, TP=2 *and* TP=4 -- the 129-edge width rise is fixed by `6b4bfe8b` (`fresh_like` collective outputs) plus `note_operator` producer registration, already on lead. No duplicate fix is wanted here; what is left of this item is the walk cost, not the producer gap |
| O16 | why the derived activation term over-reads at width: +21.5% at TP=2, +40.4% at TP=4 | GPU, TP=1 source only -- the allocation history across `warmup_model`'s step, requested in `agent_scratch/memval/producer_packet/tp1_probe/REQUEST.md` | open, and **narrowed**: the residue is defined at TP=1 by difference, so a replicated over-count is absorbed by it and cancels at every width. The error is in the sharded fraction -- being exact at TP=1 and TP=2 needs ~2.45 GB that divides by width against the walk's 1.71 GB. Allocations made inside opaque custom operators are where a dispatch trace cannot look, and the allocation history can. **No term is to be chosen by the size of the error it removes**. **Narrowed by the allocation history**: 1 761 607 680 B of the 3 036 676 096 B live at the peak -- 58% -- is allocated inside `aiter/tuned_gemm.py:450`, i.e. inside `aiter::gemm_a16w16`, which is exactly where the dispatch trace cannot look. The region is now located rather than suspected  **The region is now measured, not suspected**: at the walk peak 822 083 584 B lives on the device that the walk cannot see -- 541 065 216 B and 79 691 776 B inside `tuned_gemm.py:450`, 201 326 592 B inside `linear_attention_with_output_base` -- against a 503 316 480 B over-count of replicated hidden-width buffers. Opposite signs, so the -8.1% residual is not a coefficient. **The peak live set is now named, block by block, from source alone** (`agent_scratch/memval/producer_packet/tp1_probe/peak_sites.py`). Replaying the TP=1 history over the baseline gives a peak of 57 970 211 840 B, of which the window allocated 3 035 627 520 B in **eight** blocks at four sites: `tuned_gemm.py:450` (1 140 850 688 + 540 016 640 + 79 691 776), `qwen3_5.py:329` (570 425 344 + 2 x 167 772 160), `base_attention.py:403` (201 326 592) and `embed_head.py:177` (167 772 160). Dividing each by the run's own `max_num_batched_tokens` = 16 384 (`out/probe.memory.json`, read, not fitted) gives widths that the checkpoint config names exactly: 34 816 = `2 x intermediate_size`, 17 408 = `intermediate_size`, 16 480 = `in_proj_qkvz` 16 384 + `in_proj_ba` 96, 6 144 = 48 value heads x 128, and 5 120 = `hidden_size` three times. Classing them by what the source shards -- column/row-parallel GDN heads and MLP projections shard, `hidden_size` is replicated -- gives **2 452 619 264 B sharded, 503 316 480 B replicated, 79 691 776 B unnamed** (the 2432 block; see O8 -- token-shaped, and ambiguous between 16 384 x 2432 and 2432 x 16 384). No free parameter is involved: the blocks are measured, the widths come from `config.json`, the sharding comes from the module classes. That the sharded part lands on the ~2.45 GB this row says the width behaviour needs is a **consistency check against a number that was already known**, not a frozen evaluation -- the TP=2/TP=4 readings are X27 and stay out of the derivation. What is still owed is the term itself: this names the bytes, it does not yet compute them from a config without a history. **A candidate is now frozen** in `agent_scratch/memval/lifetime/FROZEN_ACTIVATION_CANDIDATE.md` (md5 `3cbf89c917327e004127c3c21ef64e74`), written before any TP=2/TP=4 reading was consulted: `activation(T, W) = T x dtype_bytes x (S / W + R)` with `S` = 74 848 sharded width and `R` = 15 360 replicated, giving 2 955 935 744 / 1 729 626 112 / 1 116 471 296 B at W = 1/2/4. The retained 79 691 776 B block is removed **once**, in the gate: it is live when the window closes, so it sits in `current` as well as `peak` and cancels in `peak - current`; that leaves 2 955 935 744 B against the allocator's 2 956 984 320 B `peak - current`, a 1 048 576 B remainder which is the known replay-vs-allocator gap and is reported rather than absorbed. **The candidate's assumption is already known to fail in part**: it transports the set live at the TP=1 peak, and the peak instant moves with width -- on the lead-runtime captures the device-free peak is at operator 71 at TP=1 but 76 at TP=2 and TP=4, with the live count going 8 to 9, which is what replicated bytes not shrinking while sharded ones do would produce. So these are a candidate under a stated assumption, not a prediction to grade. Making it one needs per-width liveness for the four opaque-operator blocks so the maximum is taken over instants at each width -- device-free work, no probe. **Now a maximum over instants, on the pinned recapture.** The correction above stands but its diagnosis was wrong twice over. First, raw operator 71 vs 76 is not a moved peak: TP>1 inserts five collectives ahead of it, and at the **non-collective ordinal** the peak is at 72 at TP=1, 2 and 4 alike, at the same module path `layers.1.mlp.down_proj`. Second, a first alignment run used the older graph set (`lifetime/out`, md5 `1204688df2c8`/`9093cb6cdee9`/`660362b9505d`) and appeared to confirm that the live set is identical at every width; that is withdrawn as a confirming result from stale inputs. On the pinned current-runtime recapture (`recapture/out_lead`, md5 `8f79e1d8e2e3`/`21fdda744ee0`/`756a47aa2461`, runtime `leadrt` at `cd718ef6` -- the 97-edge audit's dataset) the site is the same but its contents are not: TP=1 holds 8 tensors there, TP>1 holds 9, the extra one being `aiter::all_reduce_` at `layers.1.mlp.down_proj`, a storage of its own alive alongside the GEMM output it reduces, worth 167 772 160 B. So two instants are witnessed on the source config -- linear attention from the TP=1 history (74 848 sharded, 15 360 replicated) and MLP down-projection from the walk (52 224 sharded, 30 720 replicated plus one hidden-sized collective destination above one rank) -- and because the sharded side falls as 1/W while the replicated side does not, the more sharded instant is overtaken at W=2. `activation_instant_bytes` takes the maximum: 2 955 935 744 / 2 030 043 136 / 1 602 224 128 B at W=1/2/4, against 1 729 626 112 / 1 116 471 296 for transporting the TP=1 set, which would understate TP=2 by 14.8% and TP=4 by 30.3%. Frozen before comparison as `agent_scratch/memval/lifetime/FROZEN_ACTIVATION_CANDIDATE.md` md5 `8b1c7e6f4e55d1389ed1ba47ac45bb29` (v1, md5 `3cbf89c917327e004127c3c21ef64e74`, preserved inside it as historical). Reservations carried rather than resolved: the maximum runs over the instants that happen to be witnessed, so it is a lower bound; `linear_attn` has no history above TP=1 so any collective destination it holds there is uncounted, which can only raise it; and the walk and the history disagree about `linear_attn` itself -- the walk's curve at that site is 1 378 877 440 B at TP=1 against 3 035 627 520 B live in the history, because the walk's death rule releases the MLP buffers before the next layer's attention while the runtime does not. The walk does allocate those blocks and shards them correctly (16 480 -> 8240 -> 4120, 6144 -> 3072 -> 1536), so that is a lifetime disagreement, not a visibility one, and it is why `linear_attn` is taken from the history. No TP=2 or TP=4 measurement is opened; both remain class X27. **The maximum is now withdrawn: it ran over two different programs.** The TP=1 allocation history's frames pass through `/tmp/torchinductor_root/...py` and `torch/_inductor/utils.py:3220 run` -- it is an **Inductor-compiled** run -- while every walk graph carries `compilation_level: 0`. Inductor chooses its own buffer reuse, so the two disagree at TP=1 in **both** directions and the disagreement closes exactly: eager walk 2 717 908 992 + 741 343 232 held longer (`in_proj` 16 480 and value 6 144, alive through the same layer's MLP: alloc@106 freed@153 and alloc@114 freed@151) - 503 316 480 reused (the walk keeps 6 hidden buffers where the history keeps 3) + 79 691 776 retained = 3 035 627 520 B live at the history peak. A maximum across the two is therefore a maximum over two programs, not a bound on either. Accounting was checked and cleared first (`producer_packet/tp1_probe/reconcile.py`, `reconcile2.py`): the trace has 1012 `alloc` against 1011 `free_requested` and 1011 `free_completed`, and replaying under **either** free semantics gives the identical peak 57 970 211 840 B at event 132 and the identical end 55 014 276 096 B, which equals `current_torch` exactly -- a 0 B difference between the two semantics, so the 1 048 576 B gap to `peak_torch` is the whole of the replay residual and free accounting is not the disagreement's cause. Storage identities had to be matched allocation-by-allocation rather than by address, since addresses are reused; keying by address reported frees preceding their own allocations. With lifetimes matched correctly there is **one instant, not two**: the history's peak is the `act_fn` allocation inside layer 1's MLP with that same layer's attention buffers still live, a few events before the walk's down-projection in the same MLP. `activation_instant_bytes` now takes a required `compile_mode` and raises `UnfoundedActivation` rather than borrowing an instant across modes; it returns `is_candidate: True` and an `uncounted` tuple. Per width at T=16 384: inductor 2 955 935 744 / 1 729 626 112 / 1 116 471 296, eager 2 717 908 992 / 2 030 043 136 / 1 602 224 128. **v2's TP=2/TP=4 figures are withdrawn as predictions of the real term** -- they describe the eager program; the deployed term is the compiled one, so the candidate returns to v1's numbers for a different reason than v1 gave. No compiled history exists above TP=1, so the compiled instant's collective destinations at W>1 are unwitnessed and reported in `uncounted` rather than folded in; the eager walk cannot fill that gap now that it is known to be wrong in both directions. Frozen as v3, `agent_scratch/memval/lifetime/FROZEN_ACTIVATION_CANDIDATE.md` md5 `29fd7e9da62365bb72dcff3d94b1eb1c`, with v1 and v2 preserved inside it. Still no TP=2 or TP=4 measurement is opened; both remain class X27. **Diagnostic evaluation, one-way, held-out at TP=2/TP=4.** The implementation, its inputs, the program identity and the TP=1/2/4 predictions were frozen by digest first -- `agent_scratch/memval/lifetime/frozen/frozen_prediction.json` md5 `4713bc073b179c02660462d3445ae975`, emitted by `lifetime/freeze_manifest.py` from `memory_model.py` md5 `11bf531329df406c49241944131cf3fa` at commit `5e685260` -- and only then did `lifetime/diagnostic_eval.py`, which verifies that digest and refuses any other, open the records. On `peak_torch - current_torch` at T=16 384: TP=1 predicted 2 955 935 744 against 2 956 984 320 (-1 048 576, -0.035%, and **not held out** -- S27 calibrated it); TP=2 rank0 1 729 626 112 against 1 730 150 400 (**-524 288, -0.030%, held out**); TP=4 rank0 and rank1 alike 1 116 471 296 against 1 191 969 280 (**-75 497 984, -6.334%, held out**). Every error is negative, which is the only direction the uncounted collective term can move the prediction, so nothing here contradicts the stated assumptions in sign -- and nothing here shows that term explains them either. The residuals are two different things: 1 MiB at W=1 and 0.5 MiB at W=2 are exactly 32 and 16 elements per token, the known replay gap halving with width, while W=4's 72 MiB + 512 B is 288x what that halving predicts, is not token-aligned (75 497 984 / 32 768 = 2304.03) and is identical on both ranks, so it is a distinct term appearing only at W=4 rather than the gap growing. The eager walk cannot arbitrate it: it over-reads W=4 by +34% (1 602 224 128) as badly as it under-reads the compiled TP=1 instant. **No coefficient, peak choice or residual was changed as a result of any of these numbers**, and no TP=2 compiled history was requested -- collecting one as calibration would cross the source-only boundary. Two mechanisms are ruled out from source in parallel: `aiter`'s `bf16_tuned_gemm.csv` has no gfx942 row at M=16 384 and `is_skinny_default_shape` requires M <= 8, so unquantised bf16 at this M dispatches `libtype: torch` -> `F.linear` -> hipBLASLt with no aiter split-K path and no `_alloc_splitk_workspace` (which also settles the earlier caution: at this M the `F.linear` frame *is* the dispatch, not a wrapper hiding an aiter workspace); and the gfx942/gfx950 a16w16 split-K workspace, were it ever reached, comes from the `opus_gemm_workspace_init`/`opus_splitk_ws_get` hipMalloc registry rather than the torch caching allocator, so its bytes land in `non_torch` and cannot appear in `peak - current` at all. Recorded in `agent_scratch/memval/lifetime/frozen/DIAGNOSTIC_RESULT.md` md5 `9789f52b819c38ea949ff0567bc80b7f`. The records do not state their compile mode; a graph pool and a non-zero `cudagraph_overhead` are consistent with compilation but are not a statement of it. This is a historical diagnostic and does not replace the e2e cc-traces gates. |
| O17 | the graph pool budget *estimate* and the pool the engine actually reserves are different quantities and are not to be compared as one | -- | **open, and now three quantities rather than two.** The engine's estimator (`graph_pool_bytes`, 0.2 x peak activations, 4.6x over on the 27B at TP=1) is kept as its own row and is untouched. The recorded reserved delta is bookkeeping-contaminated by construction (see O8), so it is reported rather than predicted. The third is what capture *pins*, which is the one with a mechanism and the one a budget should carry. Capture mode is a fourth thing the reserved side depends on and the pinned side does not: PIECEWISE takes one private pool per bucket (`ATOM_PER_BUCKET_POOL=1`), FULL shares one pool across buckets (`model_runner.py:4301`), and on DSV4 TP8 the two differ by 1.11 GB reserved per rank at an identical 14.71 GB allocated. **The reserved side now splits into a derivable half and an undeclared one** (`capture_reserved_parts`): with the allocator's own constants from `c10/core/AllocatorConfig.h`, the residue outside every capture pool maps `allocator_segment_bytes(76 MiB) + allocator_segment_bytes(512)` = 81 788 928 B, which is the S27 TP=1 figure with no residual -- while the same rule over what capture *pins* reads 83 886 080 B against a pool that reserves 46 137 344 B, 82% high, and one of that pool's four segments holds no live block at all. So the pool half is the high-water mark of the whole captured forward, not a function of the pinned set, and it is taken as an argument rather than predicted. |
| O18 | `OpSpec` records the dtype of each *argument* and never of an output, so every consumer that needs an output's size reads `dtypes[0]` and assumes promotion changed nothing | **lead** -- shared schema (`core/graph.py`, `runtime/meta.py`) | open. `aiter::masked_embedding` takes int32 ids and returns bfloat16: `dtypes[0]` sizes one hidden-width buffer at 335 544 320 B instead of 167 772 160, which is the whole `walk_bytes` / `visible_peak_bytes` gap in the frozen candidate. Fix is an `output_dtypes` field filled from the real outputs and a schema bump (packet P4). Until then the walk on this branch sizes an output by PyTorch's own promotion rule when the graph records no dtype -- float beats int, and float16 with bfloat16 gives float32 -- labels the basis `recorded`, `unanimous` or `promoted`, and reports every non-`recorded` output through `dtype_ambiguities`. The masked_embedding case is now right by rule rather than by name, and the ad-hoc correction that was subtracting 167 772 160 B is deleted. Refusal is available but not the default: `strict_dtypes=True` raises `UnfoundedActivation` on the first output the graph does not record, which is what a consumer that must not guess should pass |
| O19 | the tracer's "unseen destination is a fresh allocation" rule cannot tell a buffer allocated before the traced region from one allocated invisibly inside a custom operator | **lead** -- shared runtime | open. `forward_vars["outputs"]` (`model_runner.py:1290`) is 167 772 160 B allocated at engine init, so it is inside `current_torch` and cannot be part of `peak - current`; the walk counts a write into it as an allocation. Fix is to seed the seen-set with the storages that exist when the region opens. Note this is a *replicated* over-count and therefore cancels at width -- it is a TP=1 accuracy item, not the cause of O16. The seed set is no longer hypothetical: all 29 `forward_vars` are live at the peak and every one has a pre-forward address, 172 704 716 B in total. `outputs` is the sharp case -- `embed_head.py:177` allocates a second block of the identical 167 772 160 B and shape during the forward, so the two are separable by address and by nothing else |
| O20 | mutability, alias and output-dtype contracts of the fused computation operators, not just the collectives | source + registered schema, CPU only -- done | **closed, and it found nothing wrong with the walk.** `silu_and_mul` and `_fused_qk_rmsnorm_group_quant_kernel` are destination-passing and record no output; `gemm_a16w16`, `linear_attention_with_output_base` and `unified_attention_with_output_base` return genuinely fresh tensors, by schema and by implementation (`base_attention.py:403`). No double count. Two by-products: `mutates_args="unknown"` marks weight arguments mutable, so `(a!)` in this registry is not evidence of mutation (O22), and `fused_allreduce_rmsnorm_` is absent from the 27B's graph at every width |
| O21 | the 2999-operator TP=1 lifetime trace against the 2439-operator body graph | the two artifacts, CPU only -- done | **closed.** Same region, device, compilation level, redirections and scope; the only difference is step kind, and the operator arithmetic closes with no remainder: 2999 - 416 (GDN chunked-prefill path) - 144 (drift) + 32 (decode's mrope and `aten::min`) = 2439. The prefill graph is the one shaped like `warmup_model`; the decode graph never was a candidate. The two caveats that remain on the prefill graph are its own: `compilation_level: 0`, and a scope that excludes `compute_logits` and the sampler |
| O22 | whether `torch_compile_guard(mutates_args="unknown")` declares mutation the implementation does not perform | one schema read, CPU only -- done | **closed, and it does.** `aiter::gemm_a16w16(Tensor(a0!) A, Tensor(a1!) B, ...)` marks the weight matrix mutable; `fused_allreduce_rmsnorm_` marks the norm weight mutable. The usable half of the schema is the return annotation: an aliasing return must be declared, and `all_reduce_ -> Tensor` is unannotated, so the schema independently confirms the collective is out-of-place |
| O23 | `lineage_keys` aligns 12 of 3014 outputs across the three real widths | CPU only; a module path per operator | **closed for coverage, graded for correspondence.** Re-measured on fresh captures the ancestry key aligns 12, not 71. The `VocabParallelEmbedding` name split is real but is not the cause: the graph does not know its own ancestry at width -- 725 of 4378 source edges are `-1` at TP=2/4 against 596 at TP=1 (O14, O15), and an ancestry key is recursive, so every descendant inherits the break. `module_path_keys` keys on the module path plus an ordinal among the non-collective operators of that module, from a `.modules.json` sidecar rather than a shared-schema field, and pairs up **3014 of 3014** outputs -- 1382 replicated, 1632 sharded, 0 unresolved. **That 3014 is coverage, not correspondence**: it says the ordinals paired everything, not that each pair is the same operator. Correspondence is graded, and the three grades do not add up to one number: 2307 checked by surviving ancestry, 466 checked by the recorded identity of their external inputs (parameter, buffer or forward-input name, compared positionally when the operator name is identical at every width and as a multiset when it is not -- `F.embedding(weight, ids)` becomes `masked_embedding(ids, weight)`), and **226 resting on the module tree and the ordinal alone**. 0 contradictions in either checkable grade, which is evidence about the 2773 that could be checked and about nothing else. The 226 were audited rather than assumed: every one has an input whose producer the trace *lost*, not an external it legitimately read -- 130 `aten::view` on the all-reduce output at TP>1 (O15: `record_collectives` builds the OpSpec outside the dispatch tracer, so the output never enters `_producers`; this is the whole 596 -> 725 rise), 32 `aten::cat` + 32 `aten::squeeze` in `rotary_emb`, 16 `unified_attention_with_output_base`, 16 `aten::sigmoid` whose storage *was* seen earlier (a `_died` finalizer dropping an address while a view is still alive). None is parameter-shaped, so the name evidence and the shape heuristic agree. `alignment_integrity` reports the grades separately and `width_classes` raises when either checkable grade contradicts; equal per-path non-collective counts (2999 / 2999 / 2999, no path disagreeing) remain a guard against a missing operator, not a proof of correspondence. The one renamed join is the predicted `aten::embedding` / `aiter::masked_embedding` pair, replicated, reported as `renamed` rather than joined silently. **Re-graded on the lead runtime.** The 226 were measured on my own instrumented tree; on an immutable `cd718ef6` snapshot run unpatched the same grading reads **2436 ancestry-checked / 466 name-checked / 97 neither, 0 contradictions**, and the 97 are the same set at every width (0 width-only, so no all-reduce residue). The surviving unnamed groups are 32 `aten::cat` + 32 `aten::squeeze` in `self_attn.rotary_emb`, 16 `aiter::unified_attention_with_output_base`, 16 `aten::sigmoid` (storage seen earlier), 1 `aten::view`; the 130-edge `aten::view` group has collapsed to 1. So 97 is the **remaining** producer-tracking defect and 129 of the old 226 were old graphs. **None of the 97 reaches the activation gate**: 0 of them produce a tensor live at the peak at any width, and the only never-seen bytes consumed at or before the peak are one `aten::view` of 167 772 160 B = 16 384 x 5 120 x bf16 -- the runner's persistent `forward_vars["outputs"]` buffer, which must *not* enter `peak_torch - current_torch` (`lifetime/gate_impact.py`). The 16 `aten::sigmoid` edges now have a minimal device-free reproducer (`lifetime/repro_alias_death.py`, packet P5): `_died`'s guard misses two aliases of one storage where the one registered last is released first, so a live alias loses its producer. It costs ancestry, not bytes -- the aliasing output is skipped by the walk -- so it is a lineage defect, not a gate. The candidate bytes still do not depend on this |
| O24 | a TP=1 *target* cell for a device-free replay: the committed TP=1 records carry readings, blocks and config but no run identity and no hardware identity | CPU only, read-only | **closed as a question, and a correction**. No cell need be assembled: node 18 preserves two genuine TP=1 target/table pairs (`g4/cap_subspan/`, `poc/g5_27b/`) and CC has one. `27b.tp1.exclusive.memory.json` agrees with the target on all eight shared fields, but agreement is not identity -- the phase C repeats already showed byte-identical payloads from distinct runs, and the epochs here are disjoint (2026-09-11T04:37:55Z and 2026-09-10T13:12:42Z on `visible_devices 5`, vs phase C at 08:53:49Z..09:03:38Z). Compatible independent evidence, identity unproven. The 32.0 MiB `non_torch` gap between the two committed records is observed and unexplained; the earlier neighbour attribution is withdrawn |
| O25 | which *program* a deployment runs, now that `derived_readings` can reach the activation term from the model config instead of a per-width graph | the caller -- the engine's own compile settings, not a measurement | **open by construction, and deliberately so.** The config-derived route (`model_config` + `compile_mode`) passes the gate a TP>1 prediction used to stop at, because the instant is witnessed once at the source and its widths come from the checkpoint. What it cannot derive is the program: `linear_attn`/inductor and `mlp_down`/eager differ by 238 MB at the 27B source config, so a defaulted mode would silently pick one. A profile naming a config and no `compile_mode` is refused, in the same style as every other missing term. `model_config` wins when both it and `graph` are given -- one term, one derivation. Errors carried in from the frozen diagnostic and unchanged here: -0.035% TP1 (S27, not held out), -0.030% TP2, -6.334% TP4 both ranks (held out, X27, evaluator-only); the TP4 residual is 72 MiB + 512 B, is not token-aligned, and is not chased. Uncounted above one rank: the collective destination the compiled witness never showed |
| O26 | the capture pool's 46 137 344 B, replayed rather than bounded | GPU, TP=1 only -- **done**, probe `cap1`, snapshot `3e04c166`, 47 805 events, cap 200 000, not truncated | **closed for the source width.** The replay (`agent_scratch/memval/capture_replay/`) reproduces the pool exactly: 46 137 344 B total and all four segments' allocated bytes (9 932 800 / 19 865 600 / 1 489 920 / 0), with 0 allocations landing outside an open segment and 0 frees with no live allocation. Keyed on (address, generation), which was necessary: `140076166152192` is mapped, freed and mapped again, and only the second mapping is in the pool. **No segment was forced by the tensor that survives in it** -- every one was forced by a transient, which is exactly why the rule over live blocks reads 83 886 080. Survivors are `bs × vocab_size × 2 B` for the six captured buckets, 63 × 496 640 = 31 288 320; the other 14 849 024 is reserved-but-dead, including one 2 MiB segment that ends empty and still mapped. 37 of 62 distinct transient sizes resolve to `bs × config width × 2 B`; the remaining 25 are small or unidentified and stay **uncounted**. Independent cross-check: the two still-mapped non-pool segments sum to 79 691 776 + 2 097 152 = 81 788 928, the `outside_pools` quantity `capture_reserved_parts` derives by arithmetic. Open: the 1 024 B between `CAPTURE_FIXED_PINNED` (79 692 800) and the segment observed here (79 691 776), and the *order* -- which decides that bs=32 and bs=8 share one buffer while bs=16 and bs=4 share another. Order is recorded at TP=1 now; its stability across widths is not something one history can answer, and no TP>1 history was collected |
| O27 | the source prediction's 1 048 576 B gap: is it an unsplit block tail rather than a missing tensor? | existing TP=1 artifacts only -- **done**, no new run | **closed, and witnessed from both sides.** Four quantities were being conflated: *requested* (what a trace entry's `size` carries -- settled by the capture artifact, which contains 2 B allocations while no block is ever under `kMinBlockSize`), *block* (request rounded to 512), *segment* (what is mapped), and *charged* (what `allocated_bytes` adds, which is the block handed out). `should_split` lives in a `.cpp` the wheel does not ship (torch 2.10.0+rocm7.2.4), so the rule was read off the shipped binary's behaviour in the S27 TP=1 warmup snapshot: of 305 large segments, 213 hold one active block filling the segment and their excess over the request is 0 (210) or exactly 1 048 576 (3) and **never more**; 49 hold an active block plus a split-off tail and the smallest tail is 1 114 112 and **never less**. The boundary is exactly `kSmallSize`, with no counterexample either side. The in_proj generation is witnessed directly in the warmup trace: `segment_alloc` 541 065 216 at event 26, the 540 016 640 B request served at the *same address* at event 27, no allocation ever at the tail address, `segment_free` of the same 541 065 216 at event 3051. Its block record is *not* witnessed -- the segment was freed before the snapshot -- so the block is inferred from the rule, not read. 16 384 × 16 480 × 2 = 540 016 640 rounds to 258 × `kRoundLarge` = 541 065 216, leaving exactly 1 048 576, which fails `remaining > kSmallSize` and is retained inside the block. Every other tensor in the instant rounds exactly (34 816 / 17 408 / 6 144 / 5 120 elements all land on `kRoundLarge` multiples), and the warmup window's large segment sizes are exactly that set, so **one allocation accounts for the whole source gap**. `allocator_charged_bytes` states the rule. **It does not halve with width**: TP=2 retains 524 288, TP=4 retains **nothing** (remainder 1 310 720 is over the threshold and is split off), so this explains the -0.035% and -0.030% residuals and predicts no help at all for the -6.334% at TP=4. Assumption, unverified and load-bearing at other widths: that the allocation maps a fresh segment there too, as it is witnessed to do at TP=1 |
| O28 | whether a request-driven allocator candidate, built from the source order and rules alone, carries to other widths | existing artifacts only -- **done**, CPU, no new run | **open, and the candidate is refuted at TP>1 for a stated reason.** `pool_simulator.py` takes only the cap1 request/free order and the c10 rules and maps four segments -- 2 097 152, 20 971 520, 2 097 152, 20 971 520 -- for **46 137 344 B**, the source residency exactly, with no observed segment size or address as input; it recovers the same forcing requests the address replay found (327 680 / 1 054 720 / 393 216 / 7 946 240). Pool membership is *source-proven*: `model_runner.py:4290` is the `with torch.cuda.graph(...)` statement and 4293/4298 are its body, a predicate that selects 7 896 allocations -- the same count the independent address attribution reached. The split rule, best-fit selection and coalescing remain *empirical* (the wheel omits the defining `.cpp`). `width_transform.py` carried the stream to TP=2/TP=4 by sharding the config widths and adding the `ParallelLMHead` all-gather, and the result was frozen (`frozen/pool_candidate.json`, md5 `55e1228fa18fbe5e073884c2e60f07cf`) **before** any comparison. The one-way diagnostic (`pool_diagnostic.py`, X27 opened only there) then reads, against `graph_pool.reserved` with the derived 81 788 928 B outside half added: TP=1 +0.000% (S27, the source -- not evidence), TP=2 **+15.686%**, TP=4 **+48.780%**, both *over*. The cause is named in the candidate itself and not fitted afterwards: `self.logits_in_graph = self.world_size == 1 and not is_tbo` (`model_runner.py:4104`) means the LM head is **not captured at all** at TP>1, so the width transform modelled a captured head that the target program does not run, and the all-gather branch it added is unreachable during capture. The residual is one-sided and structural, in kLargeBuffer units, which is what a whole missing family of vocabulary-width tensors looks like. Still unmodelled and carried, not folded in: the assumption that execution order is width-stable, 33 distinct request sizes (2 118 requests) that do not resolve to `bs x config width x 2 B`, the width-independence of the outside half, and the 1 024 B between `CAPTURE_FIXED_PINNED` and the observed oversize segment. Nothing was refit; the frozen manifest stands as written and refuted |
