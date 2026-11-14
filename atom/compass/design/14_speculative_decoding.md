# ATOM Compass — Design Topic 14: Speculative Decoding and MTP

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `02` (the cost form), `03` (KV and memory), `04` (structures), `07`
(calibration), `08` (validation). This topic changes something in each of them.

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** acceptance is a stated input carrying
a method, reported in the run summary and propagated into no cost term (D83, D84, D87).

**Revised 2026-10-27 by the owner's rulings on
[#2138](https://github.com/jgong5/ATOM/issues/2138):** stage 4 prices DeepSeek-V4.1-Flash's
DSpark5 on MI455, a second draft path beside V4-Pro's serial MTP (*Milestone placement*).

**Revised 2026-11-02 by the owner's ruling on
[#2683](https://github.com/jgong5/ATOM/pull/2683#issuecomment-6578751421):** stage 4's exit
cells state `--spec-decode-acceptance-length 3` on their real and simulated runs alike, in
place of rates measured from the real run; contract 3 stays unbuilt (D84, *Milestone
placement*).

**Revised 2026-11-12 by the owner's ruling on
[#3257](https://github.com/jgong5/ATOM/issues/3257):** the M3-Agentic stage prices
MiniMax-M3's EAGLE3, a Llama draft stack with its own KV pool, at step level by the
drafter's own vectors, its acceptance forced on both runs (D87.1).

**Revised 2026-11-13 by the owner's ruling on
[#3345](https://github.com/jgong5/ATOM/issues/3345#issuecomment-6849660042):** no cell grades
its mean acceptance length or accepted-count histogram; tests check once that a simulated
run draws the acceptance it is fed (D87).

**Revised 2026-11-14 by the owner's ruling on
[#3423](https://github.com/jgong5/ATOM/issues/3423):** the DSv41-DPEP stage prices
DeepSeek-V4.1-Flash's DSpark5 under DP attention, its acceptance forced at 3.51 on both
runs (D87.2).

**Scope.** What speculative decoding — Eagle3, DSpark, and MTP heads — changes for a
simulator, and why one of those changes is different in kind from everything else in this
design. The draft paths it prices are DeepSeek-V4-Pro's serial MTP, in M2's MI355X cells,
DeepSeek-V4.1-Flash's DSpark5, in stage 4's MI455 cells (D87, *Milestone placement*), and
MiniMax-M3's EAGLE3, in the M3-Agentic stage's MI355X cells (D87.1); the DSv41-DPEP
stage prices DSpark5 again, under DP attention (D87.2). Eagle3 is not on M2's path.

---

## D82. What spec decode actually changes, and the one hard part

### The four changes

| # | What changes | Difficulty |
|---|---|---|
| 1 | **Step structure.** A decode step becomes *draft* (one or more small forwards, or `mtp_k` reuses of one layer) plus *verify* (one target forward carrying `K+1` tokens per sequence) plus a rejection sample. | **Routine.** More structures, discovered by `07` Phase 0 exactly like any other. |
| 2 | **Batch shape.** `num_tokens` per sequence is `K+1`, not 1. A decode batch stops being one-token-per-row. | **Routine.** The cost form already generalises — D85. |
| 3 | **Memory.** Draft model weights, and draft KV layers on top of the target's. | **Routine.** ATOM already has the single source of truth — D86. |
| 4 | **Acceptance.** How many draft tokens are accepted decides how many steps a request needs, and therefore TPOT and throughput directly. | **This is the hard part.** |

### Why acceptance is different in kind

Everything else Compass predicts is a *duration*. Acceptance is a **behaviour** — it comes
out of comparing draft logits with target logits, and Compass computes neither, because it
does not run the model. There is no cost model that can produce it and no amount of
calibration that would help.

This is the first thing in the whole design that Compass cannot derive and cannot measure
its way to. It has to be **stated**, like device bandwidth is stated, and it carries a method like any input (D84).

The good news, and it is substantial: **ATOM already built the mechanism**, for its own
reasons, before Compass existed.

---

## D83. Acceptance is a stated input, and ATOM's existing synthetic path is the mechanism

### What is already in the tree

`SpeculativeConfig` (`atom/config.py`) carries a synthetic-acceptance path added
for benchmarking against a published acceptance figure while a draft head is still
training (ROCm/ATOM#555):

| Field / flag | Meaning |
|---|---|
| `--spec-decode-acceptance-length` (`synthetic_acceptance_length`) | mean acceptance length in `[1, K+1]`, **counting the target's own guaranteed token** — the same unit as vLLM's `synthetic_acceptance_length` and SGLang's `SGLANG_SIMULATE_ACC_LEN` |
| `--spec-decode-acceptance-rate` (`synthetic_acceptance_rate`) | the same target as a mean rate in `[0,1]`, i.e. `(length − 1) / K`. Mutually exclusive with the above. |
| `synthetic_acceptance_rates` | resolved by `__post_init__` into **per-position unconditional** rates: entry `i` is the marginal probability that the first `i+1` draft tokens are all accepted |

The rejection sampler consumes them (`atom/model_ops/rejection_sampler.py::rejection_sample`, `atom/model_ops/rejection_sampler.py::_get_synthetic_cond_rates`),
converting unconditional to conditional rates because the kernel walks positions
sequentially, and force-accepting accordingly.

### Decision

> **Compass does not build an acceptance model. It states acceptance through ATOM's
> existing `--spec-decode-acceptance-*` flags, and applies ATOM's acceptance *policy*.**

### The sampler is a cost and a value, and only one of them can be traced

`rejection_synthetic_sample_kernel` is a **Triton kernel**
(`rejection_sampler.py`). Under Compass no kernel runs. So the sampler has to be
split into two halves that are handled differently:

| Half | What it is | Treatment |
|---|---|---|
| **cost** | the kernel's duration | **Traced and priced like any other operator** — the sampler is dispatcher-visible and goes through `07` Phase 1b with everything else. Nothing special. |
| **value** | `num_bonus_tokens` per sequence, written by `tl.store(num_bonus_tokens_ptr + req_idx, …)` (`rejection_sampler.py::rejection_synthetic_sample_kernel`) | **Cannot be traced.** Compass needs the number itself — the scheduler updates `num_computed_tokens` from it, decides `max_tokens` completion from it, and sizes the next step from it. Without the value the simulation cannot advance. |

So there is exactly one thing Compass must reimplement, and it is small:

> **A device-free draw reproducing the synthetic kernel's semantics: walk positions,
> compare a uniform against the conditional acceptance rate, stop at the first reject.**

That is the whole of `rejection_synthetic_sample_kernel`'s logic
— a sequential walk, no reduction, no tensor algebra. The conditional-rate
conversion it consumes is *already host Python*
(`_get_synthetic_cond_rates`, and `acceptance_length_to_rates`), so only the
walk moves. Estimated at ~15 lines.

**The rank-consistency machinery does not need reproducing, and this is the one place the
simulator is simpler than the engine.** The real kernel needs a dedicated device generator
re-seeded per step (`_SYNTHETIC_RNG_BASE_SEED`) because `sampled_tokens` is broadcast from
rank 0 while `num_bonus_tokens` stays local, and a per-rank `torch.rand` would desync the
two — *"the anchor gather reads a rejected (−1) column → the draft model then embeds an
invalid id (HSA out-of-bounds)."* Compass single-sources the clock and the step from rank 0
already (`01`), so there is one draw, not `world_size` draws that must agree. Seed it from
the step counter and a run is reproducible, which is what `12` M-e asks for elsewhere.

**As built ([#1090](https://github.com/jgong5/ATOM/issues/1090)).** `DeferredTokenStream`
(`atom/compass/runner/step_output.py`) draws one uniform `u` per decode row of a verify
step, from one stream per DP rank seeded by its rank, and accepts `a = #{i : u <
rates[i]}` drafts. Over non-increasing unconditional rates that is the kernel's
distribution, `P(a >= i+1) = rates[i]`, in one draw per row. The row reports `a + 1`
tokens, `num_rejected = K - a` and `num_bonus = a`, and every row reports `K` draft ids,
as ATOM's runner does. A run whose speculative config declares no acceptance is refused by
name. ATOM refuses forced acceptance beside DSpark's confidence schedule
(`Config.__post_init__`), so a DSpark run verifies its full draft block; its
confidence-scheduled truncation is not modelled. The drafter is built on the meta device
and reads no checkpoint; a drafter hook ATOM renames is refused by name.

The realisations differ: the kernel draws per position from a generator re-seeded each
step, the host one uniform per row from a per-rank stream, so a paired run below full
acceptance parts at the first differing draw. Only the distribution is shared; it is
checked against the kernel at K = 1, and at K > 1 by
[#1546](https://github.com/jgong5/ATOM/issues/1546). A speculative step whose price tags
no draft term is refused by name, `cost:<backend>: prices no draft`; step-level backends
tag one with [#1541](https://github.com/jgong5/ATOM/issues/1541). An eager DP dummy priced
0 s under the `eager DP dummy` treatment (`09`) needs none
([#1976](https://github.com/jgong5/ATOM/issues/1976)).

### What still flows through ATOM's real code

Everything downstream of the value: `EngineStats` accumulation, the bonus-token bookkeeping,
the anchor gather, and the scheduler's `max_tokens` interaction
(`tests/test_scheduler_mtp_max_tokens.py`). The stated acceptance enters at the same
point the kernel's output would, so the *consequences* of an acceptance are ATOM's, not a
reimplementation.

### Why declaring through ATOM's flag is still right

**It is ATOM's flag, not a Compass one** (`13` D80's non-flag audit). A real run can be
driven with the identical argument, which is what makes a paired comparison a pairing —
and it means the *policy* being applied is the same object in both runs even though the
draw happens in different places.

**Recorded as T62:** the host draw and the Triton kernel must agree. Same stated rates,
same seed, same accepted-count distribution over a few thousand draws. A cheap test, and
without it the two halves of the pairing can drift silently.

**What this makes the prediction:** conditional. Compass predicts *"at this acceptance
length, this throughput"*. That is a weaker claim than for a dense model and it must be
reported as one — D87.

---

## D84. The mean is not enough: feed the measured per-position distribution

### The trap

`acceptance_length_to_rates` builds **the minimum-variance schedule**: accept
`floor(length − 1)` positions with certainty and put the remainder on the next one. It is
the right choice for its original purpose — replay a published mean AL — and it is a trap
for Compass.

A real run's acceptance is *ragged*: at a mean of 2.4, some sequences accept 0 and some
accept `K`. The minimum-variance synthetic schedule at the same mean produces a much
narrower distribution. The two agree on total tokens and **disagree on the shape of every
subsequent batch**, because a sequence that accepted 0 needs a step that one which accepted
`K` does not.

This is the same failure mode as the design's central finding, in a new place: matching an
aggregate does not match the schedule. A prior run was within 1.0% on prefill seconds and
90% wrong on median TTFT.

### What a distribution can and cannot buy

Feeding measured *rates* and re-drawing from them reproduces the acceptance
**distribution**, in expectation. It does **not** reproduce:

- **the per-step realisation** — which sequence accepted how much on step *k*;
- **temporal correlation within a sequence** — a request whose continuation is genuinely
  hard to predict has persistently low acceptance, and independent per-step draws erase
  that;
- **cross-sequence correlation** at a step.

All three change batch composition, and batch composition is what the design's central
finding says aggregates do not bound. So a rate-based feed is a **deliberate trade**, not
an equivalence.

### Three possible contracts

| # | Contract | Cost | Reproduces |
|---|---|---|---|
| **1** | The user states acceptance rates from prior knowledge and passes them to Compass | **cheapest** — no real run needed | the stated distribution |
| **2** | The user collects statistical acceptance rates from a real run (via Compass tooling) and passes them | one real run | the *measured* distribution, marginals only |
| **3** | Each request carries a per-token acceptance list, passed like any other HTTP attribute | **heaviest** — per-request payload, and the trace has to exist | the realisation, including correlation |

### Decision

**Contracts 1 and 2 are the design. Contract 3 is recorded as available and is not
built now.**

Rationale: per-step accepted-token counts are too heavy to carry for a 300-second,
multi-thousand-request replay, and the statistics are a good trade for what they cost.
Contract 2 is the one to use for M2's MTP acceptance cells; contract 1 is what makes
*"what if the draft head reaches AL 3.0"* answerable at all, and stage 4's exit cells use
it, one stated length on the real and simulated runs alike (D87, *Milestone placement*).

Contract 3 stays on the shelf rather than being rejected: it is the only way to close the
correlation gap above, and if a result ever turns out to depend on it, the transport is
the same additive-field mechanism `06` D28 already defines.

A graded result did come to depend on it. On stage 4's c=16 smoke cell, the simulated run
drawing per step from rates measured on one real run read TPOT 10% to 15% below both real
runs, because a real request keeps its own acceptance level and per-step draws lose it;
with each request's real decode-row count it read 2% above
([#2574](https://github.com/jgong5/ATOM/issues/2574)). Contract 3 was built to close that
gap ([#2683](https://github.com/jgong5/ATOM/pull/2683)) and closed unmerged: a graded cell
instead fixes the acceptance length on both runs through ATOM's flag, so the real run
accepts by the same synthetic schedule the simulated run draws, and no request has a level
of its own to lose (*Milestone placement*).

For contract 2, the flow:

```
  real run  --->  EngineStats.distribution     (dict: accepted_count -> occurrences,
                  engine_stats.py)              maintained per sequence per step)
                        |
                        v
            per-position unconditional rates    P(first i+1 all accepted)
                        |
                        v
  simulated run  --->  speculative_config.synthetic_acceptance_rates
```

`EngineStats` already tracks exactly the histogram this needs — `distribution` is keyed
`0..mtp_k` — so the conversion is a cumulative sum and nothing more.

### The transport for a list does not exist yet

`synthetic_acceptance_rates` is a `list[float]` on `SpeculativeConfig`
(`config.py`), but it is **internal and derived** — filled by `__post_init__` from
`acceptance_length_to_rates(length, n)` (`config.py::SpeculativeConfig._resolve_synthetic_acceptance`). The CLI exposes only the two
**scalars** — `--spec-decode-acceptance-length` and
`--spec-decode-acceptance-rate` (`arg_utils.py::EngineArgs.add_cli_args`). There is no input path for a
list, so contract 2 has nowhere to put its measurement.

**The fix is an ATOM flag, not a Compass one**, by `13` D78's own test: a per-position
acceptance curve is meaningful in a real ATOM run — it is the *same* benchmarking use case
as the scalar, only more faithful, letting a published AL curve be replayed shape and all
rather than collapsed to its mean.

```
--spec-decode-acceptance-rates 0.92,0.71,0.48,0.19
```

- mutually exclusive with the two existing scalars, which already exclude each other;
- validated as non-increasing and in `[0,1]` — they are marginals of a prefix event, so a
  rising entry is a malformed input, not a preference;
- bypasses `acceptance_length_to_rates` and sets `synthetic_acceptance_rates` directly.

Small, useful to ATOM independently of Compass, and it is the same shape of ask as
`--preprocess-pool-width` in `05` D24. Recorded as **T63**.

**Three acceptance inputs, ordered, each with its method** (#952):

| # | Input | When | Method |
|---|---|---|---|
| 1 | measured per-position rates from a paired real run | M2's MTP acceptance cells | `measured` |
| 2 | measured mean AL, expanded by the min-variance schedule | a real run exists but only its mean was recorded | `estimated` |
| 3 | a stated AL from a datasheet or a published figure | exploration, day zero, a draft head that does not exist yet; stage 4's exit cells, whose real runs state the same AL | `declared` |

The run summary reports the input's method. It propagates into no cost term: a step's
terms carry the methods their own inputs give them, whatever the acceptance input's.

Input 3 is legitimate and is what makes *"what if the draft head reaches AL 3.0"* a question
Compass can answer. It is **not** legitimate in an acceptance run whose real side drafted
without it, and `08`'s hygiene refusals gain a row saying so; D87 gives the row and when a
stated AL is evidence.

---

## D85. Cost: no new form is needed

### Why the existing form already covers it

The prefill form of `02` D12 is

```
a + b·tokens + c·Σ_req N_Q² + d·Σ_req (N_Q · N_KV_cached)
```

and a speculative *verify* step is precisely this with `N_Q = K+1` per sequence rather than
`N_Q = 1`. The quadratic term covers attention among the `K+1` new tokens; the product term
covers them attending over context. Nothing about the algebra is new — what changes is that
decode can no longer be priced by a form that assumes one token per row.

**So the decision is a simplification, not an addition:**

> **Drop the separate decode form. Price every step with the general form, and let a
> non-speculative decode be the `N_Q = 1` case.**

The features are already on the batch: `detailed_sqsq`, `detailed_sqsk` and `detailed_sk`
(`scheduler.py::ScheduledBatch`) are `Σ N_Q²`, `Σ N_Q·N_KV` and `Σ N_KV`, computed by
`compute_detailed_aggregates`.

### What *is* new

| Item | Treatment |
|---|---|
| **draft forwards** | their own structures. Eagle3 and standalone DSpark run a real layer stack per drafting step; serial MTP reuses one layer `mtp_k` times. `07` Phase 0 discovers both. |
| **rejection sampler** | an ordinary operator, priced by `07` Phase 1b. Small, but it is per step and it is not free. |
| **`num_spec_steps`** | on `SpecDecodeMetadata` (`forward_context.py`); part of the structure key, because `K` changes the graph. |
| **CUDA-graph rungs** | the capture ladder is over `running_bs`, and a spec step's token count is `running_bs × (K+1)`. Rung padding is computed from the real ladder, as ever — but the rung set a spec run captures differs from a dense one, so `07` Phase 2's memory readings must be taken **with spec enabled**. |

---

## D86. Memory: two extra terms, and ATOM already owns the layer count

Draft models add:

| Term | Source |
|---|---|
| **draft weights** | Class A, exact from the draft's own HF config. `_MTP_CONFIG` (`config.py`) names the architecture and the `n_predict` attribute per model type. |
| **draft KV** | `model_runner.py::ModelRunner._sub_pool_specs`, the declarations `get_num_blocks` sizes the pools from: the target builder's `sub_pool_specs()`, plus `draft_kv_builder.sub_pool_specs()` when the draft owns a pool. No `ModelRunner` method counts draft layers; each builder declares what it holds. A builder on the module walk (`atom/model_ops/attentions/pool_layout/pool_rows.py::PoolRowsMixin`) buys one row per draft attention module it holds, so a real layer stack (Eagle3, standalone DSpark) gets one per layer and serial MTP one per MTP layer. DeepSeek-V4's and V4.1's builders take the draft layers from the config instead (*Milestone placement*). |

Two properties inherited rather than designed, both consequences of `03` D13's decision to
run ATOM's real block accounting:

1. **Whether draft KV shares the target's pool or owns its own is already decided by
   ATOM**: only an Eagle3 draft and a standalone DSpark draft ask
   (`atom/spec_decode/draft_kv.py::draft_kv_builder`, from `EagleProposer._build_draft_model`
   and `DSparkProposer._build_draft_model`), and for them the backend the draft's own config
   resolves to answers through `DRAFT_OWNS_KV_POOL`. Serial MTP and a DSpark draft inside
   the target's checkpoint always join the target's pool. `ModelRunner.draft_shares_kv_pool`
   reports the outcome. Compass runs that choice as ATOM makes it; it does not model the
   policy.
2. **A draft's own pool rides the target's block ids**: `DraftKvBuilder.sub_pool_specs`
   declares it in the target's paged class (`page_pool`), and specs sharing a name merge in
   `plan_pools`. That is `03`'s statement of Eagle3 draft KV merged by name, unchanged.

**The one thing to be careful about:** two config-derived spellings of the draft's layer
count once silently disagreed for the standalone DSpark draft, sizing 1 slot while
allocating 5. ATOM's walk-based builders now size and bind from one assignment,
`PoolRowsMixin.pool_rows`, so a module the sizing did not count raises `KeyError` when
bound. Compass runs ATOM's sizing and never re-derives the count — which is `07` Phase 0's
"never reimplement" rule applying to memory.

---

## D87. Validation: the claim is conditional, and must be reported that way

### The rule

> **A throughput or TPOT result for a speculative configuration is a statement
> *conditional on the acceptance input*, and the run summary names that input's method.**

Concretely, for acceptance cells:

1. Run the real side. Record `EngineStats.distribution`.
2. Give the simulated side the real side's acceptance: for M2's MTP cells, the measured
   per-position rates (D84 input 1); for stage 4's, the same
   `--spec-decode-acceptance-length` the real side ran with (D84 input 3,
   *Milestone placement*).
3. Report the three results of `08` D44 as usual. Acceptance agreement between the two
   sides is weaker than exact, for the reason D84 gives, and no cell grades it:

   | Quantity | Expectation | Status |
   |---|---|---|
   | mean acceptance length | agrees to sampling error over the run | **checked by tests, not graded per cell** |
   | the accepted-count histogram (`EngineStats.distribution`) | agrees in distribution | **checked by tests, not graded per cell** |
   | per-step realisation — which sequence accepted what on step *k* | **does not agree, and is not expected to** | not a result |
   | within-sequence temporal correlation | **not reproduced** by a rate-based feed | not a result |

4. **The two aggregate rows are plumbing, and tests check it once for every cell.** A
   simulated run draws the acceptance it is fed: `tests/compass/test_run_measured_acceptance.py`
   turns a record's histograms into the rates ATOM is handed and has the run summary name
   the record, and `tests/compass/test_speculative_step.py` holds the histogram ATOM's
   scheduler counts from `DeferredTokenStream`'s draws within four standard errors of the
   stated rates, the draws fixed by a seed. A cell's run summary names its acceptance input
   with its method, and its record when the input is measured (`run._acceptance`).
5. **Schedule agreement (`08` D44's second result) is graded more loosely for speculative
   cells than for dense ones**, and the artifact says so. Two runs whose acceptance
   histograms match can still diverge in *which* request accepted what, and that
   divergence is an input property, not a modelling error. Exactly how much looser is
   set by the same noise-floor procedure as everything else (`08` D45) — measure
   real-vs-real on a speculative cell first, and if that spread swamps the target, the
   cell is ungradeable and must be reported as such.

### What this does and does not demonstrate

- **Demonstrated:** that Compass predicts the *time* consumed by a speculative
  configuration whose acceptance is known. That is the question the acceptance targets
  actually ask.
- **Not demonstrated, and not claimed:** that Compass predicts what acceptance a draft head
  will achieve on a workload. Nothing in this design attempts that, and a result that
  appeared to show it would be an artifact of the input.

A new hygiene refusal for `08` D49: **a speculative acceptance cell whose simulated run
takes input 3 (declared acceptance) while its real run drafts unforced is not acceptance
evidence.** It is exploration. A cell whose two runs state the same length, as stage 4's
do, is evidence for the claim above.

### Milestone placement

**M2, on DeepSeek-V4-Pro** (`16` D95). MTP runs in two of the nightly configurations M2
grades, TP + MTP3 and DP-attention + MTP1 (`recipes/mesh/DeepSeek-V4.md`), so it enters
with the model and is not a milestone of its own. The earlier placement, a mechanism check
on Qwen3.8-27B and the claim on Kimi-K3, was retired with those milestones.

**The draft path priced is DeepSeek-V4-Pro's serial MTP**, as ATOM builds it:

| Item | In ATOM | For this topic |
|---|---|---|
| draft model | `deepseek_v4` maps to `deepseek_v4_mtp` (`SpeculativeConfig._MTP_TYPE_MAP`), built as `DeepseekV4MTPModel` (`atom/models/deepseek_v4_mtp.py`). A DSpark checkpoint is routed elsewhere by `SpeculativeConfig.hf_config_override` | serial MTP, not a draft stack |
| draft steps | `hf_config_override` sets `num_nextn_predict_layers` to 1; the one `MTPBlock` runs `mtp_k` times, `K = --num-speculative-tokens`, 3 or 1 in the nightly cells | one draft structure per drafting step (D85); T60 is the serial case |
| draft forward | `MTPBlock` subclasses V4's `Block` (the target's attention and MoE classes), adds `e_proj`, `h_proj`, three norms and its own `hc_head` parameters, and reads the target's un-reduced mHC residual | traced and priced like any structure (D85) |
| draft weights | the MTP block only: `DeepseekV4MTP.share_with_target` binds the embedding and LM head to the target's | D86's weight term excludes both |
| draft KV | in the target's pool, since serial MTP never asks for one of its own (`EagleProposer._build_draft_model` calls `draft_kv_builder` for Eagle3 only). `DeepseekV4AttentionMetadataBuilder` lays its pool out from `hf_config.compress_ratios`, one entry per layer, those past `num_hidden_layers` the MTP blocks: V4-Pro's single MTP entry is ratio 0, so its KV is a sliding window in each request's slot (the STATE class), bound at layer id `n_layers` (`DeepseekV4MTPModel.mtp_start_layer_idx`) | D86: the target builder's `sub_pool_specs()` declares it |

**Every M2 MTP cell is an acceptance cell.** Its per-position rates are measured from the
paired real run, input 1 of D84 (`16` W4.7), and never declared. Input 1 needs T63's
transport, so T63 is on M2's path. For MTP1 the curve has one position and the mean
carries it whole; MTP3 is where its shape matters.

**Stage 4 prices DeepSeek-V4.1-Flash's DSpark5** on MI455 (`16` *Stage 4*), the only
draft path ATOM serves V4.1 with: `deepseek_v41/config.py::validate_speculative_config`
refuses any method but DSpark with five draft tokens, a draft checkpoint other than the
target's, and relaxed verification.

| Item | In ATOM | For this topic |
|---|---|---|
| draft model | `DeepseekV41DSpark` (`atom/models/deepseek_v41/dspark.py`), its `mtp.*` stages loaded from the target checkpoint | a block drafter, not serial MTP |
| draft steps | one pass through the draft stages drafts all `K = 5` positions; a Markov head then samples them in turn, and a confidence head scores them | one draft structure per step (D85); its terms are priced apart from the target's, at op level by [#2345](https://github.com/jgong5/ATOM/issues/2345) and at step level by [#2346](https://github.com/jgong5/ATOM/issues/2346) |
| draft forward | `DraftBlock` and `DraftAttention` subclass V4.1's `Block` and `Attention`, with window-only attention and the draft's own routed-expert count (`dspark_n_routed_experts`) | traced and priced like any structure (D85) |
| draft weights | the draft stages only: `share_with_target` binds the embedding and LM head to the target's | D86's weight term excludes both |
| draft KV | in the target's pool: `validate_speculative_config` requires the target checkpoint, so `DSparkProposer._build_draft_model` builds the draft from it and never calls `draft_kv_builder`. Under a speculative config `DeepseekV41MetadataBuilder` adds `num_nextn_predict_layers` window-only layers to the backbone's, each a sliding window in the request's STATE entry (`V41PoolGeometry.state_fields`) | D86: the target builder's `sub_pool_specs()` declares it |
| verification | the cells set no confidence schedule (`DSparkConfig.confidence_schedule` is off by default), so every verify step carries the full block, real and simulated (D83) | no truncation to model |

**Stage 4's exit cells state their acceptance**: `--spec-decode-acceptance-length 3`, near
the 3.13 measured on the c=16 smoke cell, in the ATOM arguments of the real runs
(`compass record`) and the simulated runs (`compass run`) alike
([ruling](https://github.com/jgong5/ATOM/pull/2683#issuecomment-6578751421)). At length 3
the minimum-variance schedule (D84) accepts two of the five drafts on every verify row, on
both sides, so the cell grades time at one acceptance and the summary names its input
`declared`. Another length is set the same way, one value of the flag passed to both
commands. `compass run --acceptance-record` stays a tool: it feeds a speculative record's
measured rates (input 1) to a simulated run whose ATOM arguments state none.

---

## D87.1. EAGLE3, as MiniMax-M3 serves it

Owner ruling on [#3257](https://github.com/jgong5/ATOM/issues/3257), 2026-11-12, for the
M3-Agentic stage (`16` D95.1), whose every arm serves MiniMax-M3 with `--method eagle3
--draft-model Inferact/MiniMax-M3-EAGLE3-GQA --num-speculative-tokens 3`. It is the first
Eagle3 path a stage prices.

| Item | In ATOM | For this topic |
|---|---|---|
| draft model | the checkpoint's `LlamaForCausalLMEagle3`, which `SpeculativeConfig.hf_config_override` maps to `Eagle3LlamaModel` (`atom/models/eagle3_llama.py`): one Llama decoder layer behind an `fc` that folds three of the target's hidden states, the layers MiniMax-M3's `get_eagle3_aux_hidden_state_layers` names (`atom/models/minimax_m3/model.py`) | a Llama draft stack, not serial MTP |
| draft steps | `EagleProposer.propose` runs `K` serial passes: the first over the step's whole token stream, `K+1` rows a sequence on a decode, each later one over one row a sequence through its `DraftGraph`; every pass ends in the drafter's LM head and argmax | one draft forward per serial pass (D85) |
| draft weights | the whole draft checkpoint: its embedding and LM head are its own, not the target's | D86's weight term counts all of it |
| draft KV | its own pool: `EagleProposer._build_draft_model` asks `draft_kv_builder`, whose answer for the draft's MHA backend is a pool of its own (`DRAFT_OWNS_KV_POOL`), and `DraftKvBuilder.sub_pool_specs` declares a row for its layer, merged by name into the target's paged class on the target's block ids (D86) | memory readings are taken with it, and the simulated draft KV block count is ATOM's |

**Priced at step level, from records, by the drafter's own vectors:** a draft forward,
charged once per serial pass at that pass's rows; propose; and draft KV, the pass
`EagleProposer.compute_draft_kv` makes over a step of middle chunks only. The draft
forward replaces the draft law's per-rung decode vector, which prices a whole draft at once
(`backends/step_linear.py`), so a record carries each serial pass apart, where today all of
a step's passes land in one `d_propose_ns` (`parity/__init__.py`).

**Acceptance is forced** through `--spec-decode-acceptance-length 2.78` on the real and the
simulated runs alike, D84's input 3 as stage 4's exit cells state theirs. At 2.78 and
`K = 3` the minimum-variance schedule accepts one draft on every verify row and a second
with probability 0.78, never the third, on both sides; the summary names the input
`declared`. Stage 4 forced only integers, so
[#3260](https://github.com/jgong5/ATOM/issues/3260) shows a fractional length holds through
record, fit and simulated run.

**It runs at level 0 by scenario**, the level
[#2712](https://github.com/jgong5/ATOM/issues/2712) defines: a MiniMax-M3 scenario with its
EAGLE3 drafter built on the meta device runs under the step stub and passes `check_run`.
That closes the Eagle case of [#2672](https://github.com/jgong5/ATOM/issues/2672), which
asked each such form to run or be refused. The step roofline lays out no Eagle3 pass, so an
Eagle3 step it prices is refused by name (`cost:<backend>: prices no draft`, D83).

A law and a readings entry name the draft by method and `K`
(`run/step_grid.py::keyed`'s `speculative`) beside the target's `config.json` digest, and
by its own `config.json` digest where that is not the target's (`draft_model`,
[#3276](https://github.com/jgong5/ATOM/issues/3276)), so two Eagle3 drafts of one target
key apart. A draft read from the target's checkpoint keys by method and `K` alone.

---

## D87.2. DSpark5 under DP attention, as the DSv41-DPEP stage serves it

Owner ruling on [#3423](https://github.com/jgong5/ATOM/issues/3423), 2026-11-14, for the
DSv41-DPEP stage (`16` D95.2), whose arms serve DeepSeek-V4.1-Flash with `--method dspark
--num-speculative-tokens 5` at DTP2 and DEP2. Stage 4 priced DSpark5 at TP4 alone
(*Milestone placement*).

- **One draft pass a step, on every rank.** `DSparkProposer.propose` drafts the whole
  block in one backbone pass and samples it through the Markov head. A block drafter pads
  its pass to the group as the target does, so `runner/projection.py` gives it no gather
  of its own (`draft_moe_rows` stays None), and an idle rank's dummy batch drafts too
  (`15` D90, *Idle ranks*). The later passes of a serial drafter under DP
  ([#3372](https://github.com/jgong5/ATOM/issues/3372),
  [#3206](https://github.com/jgong5/ATOM/issues/3206)) are not on this path.
- **Its routed experts take the arm's layout**, over TP at DTP2 and over EP2 at DEP2
  (`15` D92.1).
- **Its KV is in the target's pool**, as at stage 4: at these arms an MXFP8 pool with an
  FP4 index, its window-only layers in each request's STATE entry. Each rank's memory
  reading is taken with it.
- **Acceptance is forced at 3.51** through `--spec-decode-acceptance-length` on both runs,
  InferenceX's golden length for DSpark5 with thinking on. At 3.51 and `K = 5` the
  minimum-variance schedule (D84) accepts two drafts on every verify row and a third with
  probability 0.51; [#3260](https://github.com/jgong5/ATOM/issues/3260) showed a
  fractional length holds through record, fit and run.
- **The draft keys by method and `K` alone** (D87.1): it is read from the target's
  checkpoint.

---

## Open issues

- **`ATOM_ENABLE_RELAXED_MTP` changes acceptance semantics**, not just its rate —
  `RELAXED_TOP_N` goes 1 → 10 and `RELAXED_DELTA` 0 → 0.6
  (`rejection_sampler.py`). It is an environment variable, so it is invisible to
  every artifact key today. It must be captured in the run fingerprint or two runs with
  different acceptance semantics will compare as one. Recorded as **T59**.
- **The DSpark hang of `04` T52 was observed in this subsystem** (`dspark_scheduler.py::schedule_prefix_lengths_tensor`,
  while the function was written; the landed function builds that value with `torch.full_like`, which avoids the call that hung).
  Root-causing it is already a gating task (`16` P0.7). Stage 4 prices DSpark5
  (`16` *Stage 4*), so DSpark is on this topic's path to M2; whether the hang gates any of
  stage 4's work is T52's question.
- **Whether a draft forward's cost transfers across `K`** is untested. A serial MTP reusing
  one layer `mtp_k` times should be linear in `K`; a real draft stack need not be. One
  measurement at two values of `K` settles it. Recorded as **T60**.
- **Chunked prefill interacting with spec decode** — whether a partially-prefilled sequence
  drafts, and what structure that produces, is not covered here. Recorded as **T61**.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D82 | Spec decode changes four things; three are routine and acceptance is different in kind, because it is a behaviour Compass cannot compute. | 2026-09-19 |
| D83 | Acceptance is a **stated input**, carrying a method, supplied through ATOM's existing `--spec-decode-acceptance-*` flags. The sampler splits: its **cost** is traced and priced like any op, its **value** (`num_bonus_tokens`) needs a ~15-line device-free host draw, because the synthetic sampler is a Triton kernel and no kernel runs. Compass builds no acceptance *model*. | 2026-09-19 |
| D84 | Feed the **measured per-position distribution**, not the mean - the minimum-variance schedule matches an aggregate and not a schedule. Contracts 1 and 2 (stated or measured statistical rates) are the design; contract 3 (per-request per-token lists) is recorded and not built. Transport for a list does not exist today and needs an ATOM flag (T63). Three acceptance inputs, measured, estimated and declared; the summary reports the method, which propagates into no cost term; input 3 is not acceptance evidence unless the real run states the same AL. Contract 3 stays unbuilt after #2574 measured the per-request gap; stage 4's exit cells state AL 3 on both runs instead. | 2026-09-19; revised 2026-10-14 (#952), 2026-11-02 (#2683) |
| D85 | No new cost form. Drop the separate decode form and price every step with the general one; non-speculative decode is the `N_Q = 1` case. | 2026-09-19 |
| D86 | Draft weights are Class A; draft KV comes from ATOM's own pool declarations (`ModelRunner._sub_pool_specs`), and its layer count is never re-derived. | 2026-09-19; revised 2026-10-29 (#2437) |
| D87 | A speculative throughput result is **conditional on its acceptance input**, and the run summary names its method. Mean AL and the accepted-count histogram are checked by tests of the plumbing, not graded per cell; per-step realisation and within-sequence correlation are **not reproduced and not claimed**. Schedule agreement is graded more loosely for speculative cells. Predicting acceptance itself is explicitly not attempted. A cell whose two runs state the same AL is acceptance evidence. | 2026-09-19; revised 2026-11-02 (#2683), 2026-11-13 (#3345) |
| D87.1 | MiniMax-M3's EAGLE3, ruled by the owner: a Llama draft stack with its own draft KV pool, declared by `draft_kv_builder`; priced at step level from records by the drafter's own vectors, a draft forward per serial pass, propose and draft KV; acceptance forced through `--spec-decode-acceptance-length` on both runs; at level 0 by scenario, closing #2672's Eagle case. | 2026-11-12 (#3257) |
| D87.2 | DSpark5 under DP attention, ruled by the owner for the DSv41-DPEP stage: one draft pass a step on every rank, padded to the group with no gather of its own; its routed experts in the arm's layout; its KV in the target's MXFP8 pool; acceptance forced at 3.51 on both runs. | 2026-11-14 (#3423) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T59 | Capture `ATOM_ENABLE_RELAXED_MTP` in the run fingerprint — it changes acceptance semantics and is invisible today | needs the artifact fingerprint to be implemented |
| T60 | Test whether a draft forward's cost is linear in `K` | one measurement at two values of `K` |
| T61 | Decide how chunked prefill and drafting interact, and what structure that produces | needs Phase 0 discovery on a spec-enabled run |
| T62 | Assert the host acceptance draw and the Triton kernel agree over a few thousand draws | needs the host draw to exist |
| T63 | Add ATOM flag `--spec-decode-acceptance-rates` (list); contract 2 has no transport today | an ATOM change, sized and specified in D84 |
