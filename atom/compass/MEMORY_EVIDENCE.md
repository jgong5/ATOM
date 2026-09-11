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

## Open items

| # | item | needs | status |
|---|---|---|---|
| O1 | `parameters` / `buffers` for the 27B at TP=1/2/4 | meta build | **closed** -- exact at all three widths, finding 5 |
| O2 | 27B activation trace at TP=1, prefill-shaped | GPU, TP=1, one prefill | **closed as a constant, open as a mechanism** -- the term is 2 956 984 320 B, measured at the source configuration and exact across two independent engine starts. The walk is now possible after all: the claim that a derivation cannot record liveness was false (O13), and a device-free TP=1 lifetime capture gives 2999 operators, 2790 deaths, peak 2 717 908 992 B in the MLP. What is still wanted on a device is the *allocation curve* across the warmup step -- which of the six hidden-width buffers live at the peak were preallocated, and so already inside `current_torch` |
| O3 | source-only candidate budget, frozen, vs the recorded budget | O2 | **closed at TP=1** -- every input classed S, C06 or S27, and the budget reproduces the source run's 112 772 blocks. A residual by construction: it is not evidence of transfer, and the widths where transfer would be tested need O2's activation term at TP=2 and TP=4, which is class X27 today and therefore underived |
| O4 | `persistent` -- was 51% low | -- | **closed by calibration**, exact at TP=1, -0.004% at TP=4; a mechanism would still be better than a constant |
| O5 | `persistent` / activations / pool as functions of `max_num_seqs` | GPU, TP=1 | open, and now the main conditionality left; all three are proven flat in *utilization* (phase A) but untested in concurrency |
| O6 | physical start-up at `--max-num-seqs 1551` and 1400 | GPU, TP=1 | open; superseded as the acceptance gate by the utilization axis, kept as a diagnostic |
| O7 | `non_torch` from an exclusive-device source run at util 0.90 | GPU, exclusive | **closed by phase C** -- three byte-identical runs, calibrated at TP=1, exact at three unseen utilizations; the +42.2% excursion it cannot bound is carried with it |
| O8 | graph pool at TP=4, where the model reads +26.8% | GPU, 4 devices | open, and now with a candidate cause that is not the model: any derived-graph walk at TP>1 counts each in-place `all_reduce_` as a fresh immortal allocation (O15). Whether the pool figure goes through that walk is the first thing to check, before anything in the pool model is changed. See also O17: the estimate and the reserved pool are two quantities |
| O9 | `MODEL_HEADROOM` provenance: which run, which config | lead / history | open; until then it stays disallowed and is not to be relabelled as source |
| O10 | manifest-derived acceptance lengths | final CC workload | **closed** -- CC protocol `47917ade`: long 107 328 + 2 413 (6 859 blocks), short 2 560 + 21 (162) |
| O11 | what the 486 MiB `non_torch` excursion was | unknown; three controls failed to reproduce it | open, and the one thing the calibrated `non_torch` does not bound |
| O12 | a `run.execution` block in the memory record, carrying CC's `compass.execution/1` `execution_id` and its `id_inputs`, written by `_write_memory` | **lead** -- `_write_memory` is shared | open; until it lands, every calibrated row reads `residual (run unidentified)` and the phase C repeats cannot be machine-checked. The identity is CC's, not a second scheme: `atom/compass/core/execution_id.py` holds the one definition, stdlib-only, and `producer_key` reads it. `_write_replay_target` already writes a `hardware` block in the same neighbourhood, so the shape is precedented |
| O13 | tensor lifetime at the **source** width, and the sharded/replicated derivation of it for TP=2 and TP=4 | CPU only -- done | **restated and partly closed.** The original item asked for a TP=2/TP=4 warmup capture; that is a target measurement under a source label and is withdrawn. Lifetime is now captured at TP=1 on no device (2999 ops, 2790 deaths) and the width mechanism is read off the shapes. The frozen candidate reads +21.5% at TP=2 and +40.4% at TP=4 against the class-X27 peaks, so the *mechanism* is open: see O16. Three instrumentation defects found on the way (D1-D3), all in files this worker does not own |
| O14 | `_storage_of` returns 0 for every meta tensor (`runtime/meta.py`), so alias and provenance tracking collapse on any derived graph | **lead** -- shared runtime | open; fix is `untyped_storage()._cdata` when `data_ptr()` is 0, negated so it cannot collide with a device address. Patched locally in `agent_scratch/memval/lifetime/capture_lifetimes.py` |
| O15 | the collective the derivation records has no output tensor of its own: nothing is watched, so it can never die, and the meta stand-in (`_collective_stand_in`) returns the *input object*, which is the opposite error | **lead** -- shared runtime | open. Cost: 21.9 GiB against a true 2.5 GiB at TP=2, and **it sits under every derived-graph memory walk at TP>1, O8's graph pool included**. The in-place reading is withdrawn: the live implementation allocates a fresh output on every path (packet P2) |
| O16 | why the derived activation term over-reads at width: +21.5% at TP=2, +40.4% at TP=4 | GPU, TP=1 source only -- the allocation history across `warmup_model`'s step, requested in `agent_scratch/memval/producer_packet/tp1_probe/REQUEST.md` | open, and **narrowed**: the residue is defined at TP=1 by difference, so a replicated over-count is absorbed by it and cancels at every width. The error is in the sharded fraction -- being exact at TP=1 and TP=2 needs ~2.45 GB that divides by width against the walk's 1.71 GB. Allocations made inside opaque custom operators are where a dispatch trace cannot look, and the allocation history can. **No term is to be chosen by the size of the error it removes** |
| O17 | the graph pool budget *estimate* and the pool the engine actually reserves are different quantities and are not to be compared as one | -- | open, and separate from O8. O8 is the +26.8% error in the predicted pool at TP=4; this is the prior question of which two numbers that percentage is between |
| O18 | `OpSpec` records the dtype of each *argument* and never of an output, so every consumer that needs an output's size reads `dtypes[0]` and assumes promotion changed nothing | **lead** -- shared schema (`core/graph.py`, `runtime/meta.py`) | open. `aiter::masked_embedding` takes int32 ids and returns bfloat16: `dtypes[0]` sizes one hidden-width buffer at 335 544 320 B instead of 167 772 160, which is the whole `walk_bytes` / `visible_peak_bytes` gap in the frozen candidate. Fix is an `output_dtypes` field filled from the real outputs and a schema bump (packet P4). Until then the walk on this branch sizes an output by PyTorch's own promotion rule when the graph records no dtype -- float beats int, and float16 with bfloat16 gives float32 -- labels the basis `recorded`, `unanimous` or `promoted`, and reports every non-`recorded` output through `dtype_ambiguities`. The masked_embedding case is now right by rule rather than by name, and the ad-hoc correction that was subtracting 167 772 160 B is deleted. Refusal is available but not the default: `strict_dtypes=True` raises `UnfoundedActivation` on the first output the graph does not record, which is what a consumer that must not guess should pass |
| O19 | the tracer's "unseen destination is a fresh allocation" rule cannot tell a buffer allocated before the traced region from one allocated invisibly inside a custom operator | **lead** -- shared runtime | open. `forward_vars["outputs"]` (`model_runner.py:1290`) is 167 772 160 B allocated at engine init, so it is inside `current_torch` and cannot be part of `peak - current`; the walk counts a write into it as an allocation. Fix is to seed the seen-set with the storages that exist when the region opens. Note this is a *replicated* over-count and therefore cancels at width -- it is a TP=1 accuracy item, not the cause of O16 |
| O20 | mutability, alias and output-dtype contracts of the fused computation operators, not just the collectives | source + registered schema, CPU only -- done | **closed, and it found nothing wrong with the walk.** `silu_and_mul` and `_fused_qk_rmsnorm_group_quant_kernel` are destination-passing and record no output; `gemm_a16w16`, `linear_attention_with_output_base` and `unified_attention_with_output_base` return genuinely fresh tensors, by schema and by implementation (`base_attention.py:403`). No double count. Two by-products: `mutates_args="unknown"` marks weight arguments mutable, so `(a!)` in this registry is not evidence of mutation (O22), and `fused_allreduce_rmsnorm_` is absent from the 27B's graph at every width |
| O21 | the 2999-operator TP=1 lifetime trace against the 2439-operator body graph | the two artifacts, CPU only -- done | **closed.** Same region, device, compilation level, redirections and scope; the only difference is step kind, and the operator arithmetic closes with no remainder: 2999 - 416 (GDN chunked-prefill path) - 144 (drift) + 32 (decode's mrope and `aten::min`) = 2439. The prefill graph is the one shaped like `warmup_model`; the decode graph never was a candidate. The two caveats that remain on the prefill graph are its own: `compilation_level: 0`, and a scope that excludes `compute_logits` and the sampler |
| O22 | whether `torch_compile_guard(mutates_args="unknown")` declares mutation the implementation does not perform | one schema read, CPU only -- done | **closed, and it does.** `aiter::gemm_a16w16(Tensor(a0!) A, Tensor(a1!) B, ...)` marks the weight matrix mutable; `fused_allreduce_rmsnorm_` marks the norm weight mutable. The usable half of the schema is the return annotation: an aliasing return must be declared, and `all_reduce_ -> Tensor` is unannotated, so the schema independently confirms the collective is out-of-place |
| O23 | `lineage_keys` aligns 71 of 3014 outputs across the three real widths, and misclassifies one that it does align | CPU only; a module path per operator | open, and it blocks any use of `width_class` at width. Cause is read from source: `VocabParallelEmbedding.forward` (`embed_head.py:168-178`) emits `aiter::masked_embedding` at TP>1 and `aten::embedding` at TP=1, and an ancestry key that interns operator names propagates that one branch through the whole graph. Fix is a module path stamped per operator -- width-invariant by construction, already named by `@mark_trace` at trace time, and absent from `OpSpec`. The candidate's bytes do not depend on this |
| O24 | a TP=1 *target* cell for a device-free replay: the committed TP=1 records carry readings, blocks and config but no run identity and no hardware identity | **lead** + CC | open. The memory-side TP=1 source evidence exists and is committed (`27b.tp1.memory.json`, `27b.tp1.exclusive.memory.json` with its ownership sample, `g3_util_phasea/phasebc.json`, `qwen3_5_27b.config.json`); what is missing is the identity O12 would supply. A diagnostic-only cell assembled from them must label every term with where it came from, must not be described as a capture of a run it cannot name, and must not carry TP=1's 112 740 blocks into a TP=2 or TP=4 cell -- the pool is a per-rank quantity and a one-process CPU replay can only execute rank 0 (`replay/local_proc.py:43`) |
