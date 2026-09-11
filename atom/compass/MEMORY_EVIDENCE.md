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

| class | meaning | may a target claim rest on it? |
|---|---|---|
| **S** | source: `config.json`, the checkpoint header, a meta build, the deployment's own flags, the card's spec capacity | yes |
| **C06** | a constant fitted on the **0.6B** campaign | yes -- the target had no part in fitting it |
| **C27** | a constant fitted on the **27B**, i.e. on the evaluation target itself | **no** -- using it makes the row partly self-referential |
| **T** | a reading the target run recorded on the device | only as the thing being predicted, never as an input to the prediction |

## The terms

Errors are TP=1 / TP=2 / TP=4, rank 0, from the three records in
`tests/compass/memory_records/` (`scripts/compass/validate_memory.py`).

| term | derived side | class | recorded side | error | status |
|---|---|---|---|---|---|
| weights | `weight_bytes(checkpoint, tp)` or `resident_bytes` on a meta build | S | `parameter_bytes - buffer_bytes` | not run here (no checkpoint on this box) | **open** |
| model buffers | not modelled | -- | `buffer_bytes` = 32 MiB (rotary tables) | -- | **open**, small |
| load residue | `DEFAULT_LOAD_RESIDUE` | C06 | `weights_torch - parameter_bytes` | -93.0% / -3.3% / -6.7% | fails at TP=1 on a 14 MiB term |
| persistent | `DEFAULT_PERSISTENT` = 118 MiB | C06 | `current_torch - weights_torch` = 240.6 MiB | -51.0% at all three widths | **fails**, consistently, and is flat in width as claimed |
| activations | `peak_activation_bytes(graph)` | S (needs a trace) | `peak_torch - current_torch` | not run here (no 27B graph on this box) | **open** |
| non-torch | `DEFAULT_NON_TORCH[w] + MODEL_HEADROOM` | C06 **+ C27** | `non_torch` | +4.9% / +1.2% / +1.3% | passes **only with a 27B-fitted constant** -- see below |
| pool estimate | `graph_pool_bytes(peak_torch - current_torch)` | T | `cudagraph_overhead` | +0.0% / +0.0% / +0.0% | **identity**, not a term; kept and labelled as a mirror check |
| graph pool | `measured_graph_pool_bytes(capture_sizes, w)` | C06 | `graph_pool.reserved` | -9.7% / +2.0% / **+26.8%** | **fails at TP=4** |
| kv blocks | `kv_geometry` + ATOM's `plan_pools` | S, given the budget | `blocks.num_kvcache_blocks` | +0.00% at every width and rank | exact -- but see "what exact means" |

## Four structural findings

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

### 3. `non_torch` passes because of a constant fitted on the target

`non_torch_bytes` returns `DEFAULT_NON_TORCH[w] + MODEL_HEADROOM`, and
`MODEL_HEADROOM = 266 MiB` was measured as the 27B's own offset above the 0.6B
at TP=2 and TP=4 -- class **C27**. With it and without it, against these
records (MiB):

| width | table (C06) | + headroom (C27) | recorded | error with | error without |
|---|---|---|---|---|---|
| 1 | 926 | 1192 | 1136 | **+4.9%** | **-18.5%** |
| 2 | 6906 | 7172 | 7084 | +1.2% | -2.5% |
| 4 | 7266 | 7532 | 7290 | +3.3% | -0.3% |

The headroom is 22.3% of the derived term at TP=1 and 3.5-3.7% at TP=2 and
TP=4. It was fitted at TP=2/4, where removing it costs little, and it is doing
its real work at TP=1, where it was not fitted -- and it makes TP=4 *worse*.
That is the shape of a constant absorbing something it does not explain.

Consequence, stated plainly: **the non-torch row's +4.9% at TP=1 is not a
source-only result.** Either the row is reported at -18.5% with C06 constants
alone, or it is reported as passing with a constant that saw the target. It
cannot be both, and POC_STATUS should not carry it as the former.

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
and may draw only on classes S and C06.

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
| `parameters` | **OPEN** | S | meta build, `meta_probe.py --weights-only --tp 1` |
| `buffers` | **OPEN** | S | same |
| `load_residue` | 1 MiB | C06 | `DEFAULT_LOAD_RESIDUE[1]` |
| `persistent` | 118 MiB | C06 | `DEFAULT_PERSISTENT` -- known 51% low against this model |
| `non_torch` | 926 MiB | C06 | `DEFAULT_NON_TORCH[1]`, **headroom excluded** as C27 |
| `activations` | **OPEN** | S | `peak_activation_bytes` of a TP=1 meta trace at the warmup token count |
| `graph_pool` | `floor + slope x sum(ladder)` | C06 | `measured_graph_pool_bytes`, ladder (1,2,4,8,16,32) is a deployment flag |

Three inputs are open, and two of them -- `persistent` and `activations` -- are
the terms whose derived side is weakest. Until they are closed, a source-only
budget cannot be quoted, and the honest statement of coverage is: the geometry
half of the sizing chain is source-only and exact; the budget half is not yet
derived at all for this model.

## The feasibility scenarios are conditional, not verified

`atom/compass/core/feasibility.py` reports, at TP=1 on the 27B:

* `--max-num-seqs 1551`: the state floor (1551 x 74.8 MiB = 113.31 GB) exceeds
  the KV budget (113.30 GB) and ATOM's own `plan_pools` raises
  `InsufficientPoolBudget`;
* `--max-num-seqs 1400`: sizing yields 11 190 blocks and reports healthy, while
  the longest cc-traces request (249 344 in + 5 690 out = 255 034 tokens =
  15 940 blocks) can never be admitted.

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

## Open items

| # | item | needs |
|---|---|---|
| O1 | `parameters` / `buffers` for the 27B at TP=1/2/4 | meta build, CPU only |
| O2 | 27B activation trace at TP=1, prefill-shaped | meta trace, CPU only |
| O3 | source-only candidate budget, frozen, vs the recorded budget | O1 + O2 |
| O4 | `persistent` -- 51% low, flat in width; needs a mechanism, not a refit | instrumentation |
| O5 | `persistent` / activations / pool as functions of `max_num_seqs` | GPU, TP=1 |
| O6 | physical start-up at `--max-num-seqs 1551` (expect `InsufficientPoolBudget`) and at 1400 (expect start, then refusal on the long request) | GPU, TP=1 |
| O7 | `non_torch` on an exclusively-held device, to separate the configuration's share from the neighbours' | GPU, exclusive |
| O8 | graph pool at TP=4, where the model reads +26.8% | GPU, 4 devices |
