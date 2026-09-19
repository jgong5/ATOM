# ATOM Compass — Design Point 2: Model Runner Seam and Cost Backend

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Depends on:** `01_execution_and_time_model.md` (the clock protocol supplies the
`advance_to` that consumes this document's output).

**Scope.** Where Compass attaches to ATOM, what replaces the forward pass, and the
interface between that replacement and whatever predicts a duration. It does **not**
cover how a *real* cost is derived (operator capture, pricing, analytical models) — that
is design point 3 — nor memory sizing, which is design point 4.

---

## D10. The attachment point

### Problem

Compass replaces only the forward pass. The attachment must be narrow enough that the
scheduler, block manager, admission logic and prefix cache all run unchanged, and cheap
enough that it does not become a maintenance burden against upstream ATOM.

### Candidate cut points

| Cut | Where | Replaces | Verdict |
|---|---|---|---|
| `ModelRunner.run_model()` | `model_runner.py:2840-3060` | `model(input_ids, positions)` + `compute_logits` | **Rejected.** Returns `(logits, hidden_states)` as real tensors that `postprocess` indexes, samples and gathers logprobs from. Faking it still requires a `[bs, vocab]` device allocation. |
| `ModelRunner.forward()` | `model_runner.py:3233-3320` | the whole forward, sampling included | **Chosen.** Its contract is `ScheduledBatch` in, `ScheduledBatchOutput` out — both pure numpy/list/int, already pickled across the worker boundary. |
| whole-runner replacement via `Config.runner_qualname` | `config.py:1595` | weights, KV tensors, CUDA graphs, sampling | **Chosen as the delivery mechanism** for the above. |

### Decision

**Subclass `ModelRunner` and inject it with `--runner-qualname`.**

`Config.runner_qualname` (`atom/config.py:1595`) is consumed at `engine_core.py:125-130`
and `async_proc.py:166-169`. It already has two in-tree users —
`atom/rollout/async_engine.py:26-32` injects `RLHFModelRunner`, and `config.py:1729-1736`
swaps in `RapidServeModelRunner` automatically. **The injection itself requires no ATOM
change.**

`RapidServeModelRunner` (`model_runner.py:4168-4298`) is a working template for a
non-allocating runner already in the tree. It overrides exactly the memory-owning
methods: `_build_and_load_model` (`:4218`), `_maybe_warmup` (`:4232`),
`_kv_budget_extra_reserve` (`:4239`), `get_num_blocks` (`:4245`),
`allocate_kv_cache` (`:4261`), `forward` (`:4269`) — and constructs parameters on meta
through `_init_weight_params_on_meta` (`:4188-4211`).

### The RPC surface that must be honoured

`AsyncIOProc.busy_loop` dispatches by name (`async_proc.py:236-250`). A replacement
runner must answer all of: `get_num_blocks`, `allocate_kv_cache`, `capture_cudagraph`,
`forward`, `dummy_execution`, `exit`, `freeze_gc_heap`, `process_kvconnector_output`,
`async_proc_aggregation`, `start_profiler`, `stop_profiler`, `flush_pp_send`.

**Return contracts are load-bearing across a process boundary.** `engine_core` calls
`capture_cudagraph` with `wait_out=True` and unpacks three values; a stub that returned
`None` killed the worker on an unpacking error while the parent waited forever. **Across
a process boundary a breached contract becomes a hang, not a traceback.**

### Three semantics `forward()` must reproduce

1. **Deferred output.** `tokenIDProcessor.is_deferred_out` is True by default: the tokens
   returned for step *N* belong to step *N-1*. `Scheduler.postprocess` depends on it
   (`scheduler.py:2437, 2467-2480`). Returning the current batch's tokens with
   `is_deferred_out=False` offers them one step early to a loop that cannot yet see the
   sequence, and **never offers them again until a decode batch happens to include the
   request** — on the prior 27B run every request whose prefill completed inside a
   36-step prefill streak waited for step 37.
2. **The deferral unit is one *meaningful* step, not one step.** `ModelRunner.forward`
   returns early for a pure middle chunk of a chunked prefill (`produces_output()` false,
   nothing sampled) with `is_deferred_out` unset. On the 27B that is **4,354 of 4,440
   `postprocess` calls** that defer. Mirroring `is_pure_middle_chunk` reproduced the lag
   step for step (mean 8.99 s vs a real 9.00 s).
3. **`produces_output()`** (`scheduler.py:823-840`): a pure-middle-chunk prefill batch
   must return an **empty** `token_ids` list with the same `req_ids`, mirroring
   `model_runner.py:3298-3305`.

Speculative decoding adds `num_rejected` / `num_bonus` sized `batch.total_seqs_num` and
`draft_token_ids` shaped `[bs, mtp_k]`. ATOM's existing `synthetic_acceptance_rates` path
(`config.py:1064-1190`, `rejection_sampler.py:20-224`) is the model to copy — it already
forces a chosen acceptance curve instead of computing one.

### Open issues

- `--runner-qualname` exists on `Config` but has **no `EngineArgs` field and no CLI
  flag**. Adding one is the 3-touch recipe (`arg_utils.py` field, `add_cli_args`,
  `Config` field); `_get_engine_kwargs` forwards by name and `LLMEngine.__init__` filters
  by `fields(Config)`, so name-matching suffices.
- `ModelRunner.__init__` warms the model before returning, and warmup drives a forward,
  so anything mode-dependent is consulted **before** a subclass's `__init__` body has run.
  Config must resolve lazily from `self.config`.
- The deferral *unit* is empirical. "One meaningful step" reproduced the 27B exactly, but
  the underlying rule is not derived. Prior work left this open explicitly.

---

## D11. No modes on the runner; the algorithm comes from a backend

### Problem

An earlier design gave the runner three modes (`trace`, `measure`, `predict`), which put
a mode check in front of every override and made "which mode was this artifact produced
under" a question that had to be tracked out-of-band.

### Decision

Per the project's own design note: **the runner has no modes. Every run is a simulation.
The algorithm that produces a duration comes from a modelling backend.**

Two orthogonal flags sit beside it, neither a mode of the runner:

- **`measure`** — additionally perform a real GPU run and cache the measurements for
  later simulation.
- **`trace`** — performed on demand on first use and cached; not something a user selects.

### The backend interface

```
CostBackend:
    estimate(batch_view) -> StepCost          # seconds, plus a breakdown
    describe() -> str                          # provenance, for the artifact
```

`batch_view` is a projection of `ScheduledBatch`, not the object itself, so the backend
stays free of ATOM imports and is testable on a laptop. The prior work enforced exactly
this separation (`atom.compass.core` was forbidden from importing anything from ATOM) and
it paid for itself.

`StepCost` carries `seconds` **and a breakdown**. The breakdown is not decoration: every
serious error in the prior effort was found by decomposing a total, and at least six
times an aggregate hid compensating errors — a +13.8% memory sum that was three errors
two of which cancelled; a cc-traces latency within 5% that was +52% TTFT against −30%
decode; a prefill total of +0.04% that was −4.01% pure against +27.28% mixed.

### Provenance vocabulary, carried on every backend

Adopted from the prior work because it prevented a recurring category error:

- `analytical` — computed without measuring the subject. **Nothing was analytical in the
  prior PoC.** The word is reserved, not aspirational.
- `empirical/measured` — the unit itself was timed
- `empirical/fitted` — a form chosen, coefficients regressed over measured steps
- `empirical/interpolated` — no form assumed, nearby measurements looked up
- `empirical/extrapolated` — asked outside the measured range

Two rules: **"priced" is not a species** (a price is `measured`, differing only in unit —
say `measured (op-level)` vs `measured (step-level)`); and **reading a measurement back
from a file is still `measured`** — what changes is whether the key matched exactly
(measured), nearest (interpolated) or outside (extrapolated).

### Open issues

- Whether `measure` and `trace` are CLI flags, env vars, or both.
- A backend asked about a step kind it has no samples of must **raise**, not fall back.
  The prior fallback was the mean of an empty list, i.e. zero — "a confident, precise,
  entirely fictional answer", a TTFT of 0 ms against a real 7.6 s.

---

## D12. The milestone-1 fake model

### Problem

Milestone 1 requires "fake models that can simulate all a real model can do — prefill,
decode, KV cache needs, TP/DP/PP/EP". Read as a modelling requirement this is a large
task. Read correctly it is a **test double**: the cheapest thing that produces a plausible
duration and a correctly-shaped `ScheduledBatchOutput`, so that the time model, the
simulated KV management and the scheduler integration become observable **before any real
cost model exists**.

### What "simulate all a real model can do" actually requires

From the engine's point of view the fake model must be *interface-indistinguishable*, not
*accurate*. Concretely it must supply:

1. **KV geometry** — enough that `get_num_blocks` returns a real number, `plan_pools`
   over `SubPoolSpec` runs for real, and `BlockManager.__init__`'s `assert num_blocks > 0`
   (`block_manager.py:77`) passes. Derived from the HF config: layer count, KV head count,
   head dim, KV dtype, block size.
2. **A step duration** from the cost backend.
3. **A correctly-shaped output**, honouring the three semantics of D10.

Parallelism support then collapses to arithmetic on the geometry — which dimension
divides by what:

| Axis | Effect on the fake model |
|---|---|
| TP | KV heads shard (GQA-bounded) so KV bytes/token divides; weights shard; collectives appear as a cost term |
| DP | N independent engines, each holding full KV |
| PP | layers split per stage, so KV per stage; and it creates the LP structure of design point 1 |
| EP | experts shard; KV unaffected; all-to-all appears as a cost term |

This is on the order of thirty lines, and the point is that **the rest of the system then
does genuine work**: the real `BlockManager` gets a genuinely different block count per
TP width, and the Clock Authority gets a genuinely different LP count per PP depth.

### Options for the cost backend at M1

**A. Two constants** (`prefill_seconds`, `decode_seconds`). take2's `ConstantCostOracle`,
about 40 lines.

- *Pros:* absolutely minimal; proves the plumbing.
- *Cons:* a fixed-shape oracle hides every shape-dependent bug, and you cannot tell
  whether the scheduler is reacting to the predictions at all. Measured directly in the
  prior work: an offline workload has one shape per kind, so a one-graph-per-kind oracle
  looked exact — and then cost **+47.9% TTFT** the moment shapes varied under real
  serving (3 prefill steps of 16/256/16128 tokens costing 38.6/151.0/177.6 ms, all
  answered as ~47 ms).

**B. Shape-analytic stub.** A declared linear form over fields `ScheduledBatch` already
carries.

- *Pros:* about twenty lines more than A, and it buys a **testable integration property**:
  double the chunk size, the step time doubles, the prefill streak breaks at a predicted
  place. That closed loop — step cost to queueing to a different batch — is what the whole
  design rests on, and constants cannot exercise it.
- *Cons:* its coefficients mean nothing physically, so it must never be quoted as
  accuracy.

**C. Roofline from the geometry** — FLOPs/compute + bytes/bandwidth.

- *Pros:* physically motivated; uses the configurable compute, bandwidth and interconnect
  values the project requires anyway.
- *Cons:* this is the milestone-2+ analytical backend arriving early. More work than M1
  needs, and it invites judging M1 on accuracy when M1 is about plumbing.

### Decision

**Option B, with a constant mode retained for first bring-up.**

The form, including the quadratic query term:

```
prefill:  a  + b ·tokens + c ·Sum_req N_Q^2 + d ·Sum_req (N_Q * N_KV_cached)
decode:   a' + b'·batch  + c'·Sum ctx       + e'·( rung * max(ctx) - Sum ctx )
```

**ATOM already computes the attention terms.** `ScheduledBatch.detailed_sqsq`,
`detailed_sqsk`, `detailed_sk` (`scheduler.py:790-792`) are Sum N_Q^2, Sum N_Q*N_KV and
Sum N_KV, produced by `Scheduler.compute_detailed_aggregates` (`:2788-2842`), currently
gated on `self.profile_active and ATOM_ENABLE_DETAILED_ANNOTATION` (`:2820-2821`).
Ungating them is a flag, not code.

Two properties of this form are free here and cost the prior effort real time to
discover, so they are baked in from the start even though M1 is a stub:

- **The attention terms are summed per request, not computed from batch-collapsed
  scalars.** Collapsing the batch to `tokens x history` and multiplying was a rank
  deficiency; the per-request sum was the repair.
- **Decode is fitted per CUDA-graph rung**, with both coefficients per rung — shared-slope
  gave 8.09% median error against **0.93%** for per-rung intercept and slope. And the
  padding feature is **the rung's rectangle** `rung*max(ctx) - Sum ctx`, not the batch's
  `len(ctx)*max - Sum ctx`. The wrong one read as **-22.6% at rung 16** and survived two
  rounds of evidence-widening, because a wrongly-computed feature is indistinguishable
  from an unconstrained one. Error tracked the ratio exactly: rung 2 ratio 1.00 gave
  -0.34%, rung 16 ratio 5.59 gave -22.64%.

### Geometry source

**Default: the real HF config** (Qwen3.8-27B's actual layer, head and dtype numbers), so
M1 already exercises real geometry and M2 is a backend swap rather than a rebuild.
**Additionally: a synthetic dial-able config as a test fixture**, for stressing the
scheduler at shapes no real model has.

### Open issues

- **A single global quadratic coefficient is wrong for a hybrid.** The 27B target is 48
  gated-DeltaNet layers plus 16 full-attention layers; a DeltaNet decode does not grow
  with history at all. Harmless for a stub; **must not be carried into design point 3
  unexamined.**
- The stub's coefficients are declared, so every M1 number is a plumbing result and must
  be labelled as such. Nothing from M1 may be presented as accuracy evidence.
- `compute_detailed_aggregates` is currently gated behind profiling. Ungating it in the
  simulated path is trivial; whether it should also be ungated on the real path (it is
  cheap, and having it on both sides makes real-vs-simulated feature comparison possible)
  is open.
- CUDA-graph rung selection: `StepShape.capture_bucket` must be `None` when nothing was
  replayed. A prior bug derived the bucket from batch size alone and so assigned a rung to
  prefill steps too, charging a four-sequence prefill 2 us per launch instead of 67 us per
  operator. Also note ATOM's own padding site takes the **largest** rung where its comment
  says otherwise (`reversed(capture_sizes)` is correct only on a descending list, and
  `capture_cudagraph` re-sorts ascending when it finishes); the rule that actually selects
  the graph is `ForwardMode.decide`.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D10 | Attach at `ModelRunner.forward`, delivered by a `--runner-qualname` subclass; no ATOM change for the injection | 2026-09-18 |
| D11 | No modes on the runner. Every run simulates; the algorithm comes from a pluggable cost backend. `measure` and `trace` are orthogonal flags. | 2026-09-18 |
| D12 | M1 fake model = KV/weight geometry from the HF config + a shape-analytic cost stub including the quadratic query term; constant mode retained for bring-up | 2026-09-18 |
