# ATOM Compass — Open Items: TODO Register, Assumptions, Gaps

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**What this is.** Everything across the sixteen design topics that is *not settled*, in one
place. Split out of `README.md` so the front page stays a bird's-eye view rather than a
backlog. Nothing here is a decision; every decision lives in its topic's decision log.

**How to read it.** Four sections, in decreasing order of how much rests on them:

1. **Load-bearing assumptions** — hold up large parts of the design; each has a check plan
2. **Missing topics** — design points nobody has written yet, with a recommendation
3. **TODO register** — T1–T72, per topic
4. **Cross-cutting issues and pending amendments**

---

## 1. Load-bearing assumptions, and how each gets checked

Five assumptions hold up large parts of the design. **None has been tested.** Each row
names the check, where it runs, and roughly what it costs — so these are schedulable work
in the execution plan rather than caveats in a document.

| # | Assumption | If false | The check | Cost |
|---|---|---|---|---|
| **T21** | The in-situ calibration transfers across TP width | the recipe's "calibrate at TP1, predict TP2/4/8" collapses and the campaign multiplies by the number of widths | **Doc `07` Phase 1c's one extra run.** Calibrate at TP1, predict a TP2 full-engine run, compare. Already a designed step of the recommended flow — it is step 6 — so the check is not extra work, it is the reason that step exists. | one TP2 engine run, ~1 h GPU |
| **T25** | The real-vs-real noise floor stays narrow under closed-loop replay at high client count | those cells become ungradeable — not failed, *ungradeable*, which is worse because nothing is proven either way | **Doc `08` D45's step 1, run before any simulated comparison.** N≥3 spaced real repeats at the 64- and 256-client cells; report the pairwise spread. If it swamps 10%, say so and re-scope the acceptance cells. | 3 real cc-traces runs per cell, ~3 h GPU |
| **T5** | ATOM's model classes trace cleanly under `FakeTensorMode` at TP>1 | tier b has no IR, and docs `04`, `07` and `09` rest on it | **Trace the 27B at TP2 under the doc `04` D18 mechanism and diff the captured structure against TP1.** Run this first and cheaply. A fake-tensor trace should be GPU- and collective-free, so the known mode hang should not be reachable from it; T5 is the experiment that settles whether that reasoning holds. Needs a non-wedged node (`rocminfo` under `timeout` before starting). | half a day, one node |
| ~~**T10**~~ | ~~`AgenticReplayStrategy` can be subclassed rather than vendored~~ — **resolved 2026-09-20 by P0.3.** Yes, and it is not needed on its own: the strategy is built by the plugin factory, so an out-of-tree subclass displaces it with no upstream edit — but a subclass reaches only four of the seven pacing sites. All seven share one `LoopScheduler` resolved as a module global, so the adapter **rebinds that global** instead (`06` D34). No vendoring; W1.9 stands at 450–650 lines. Evidence: six executed claims, `agent_scratch/compass_dev/p0_3/spike_t10.py`, zero edits to agentx-harness. | done |
| **T52** | `TorchDispatchMode` instrumentation does not hang ATOM at width | `--measure` runs and any mode-based instrumentation of a REAL execution are unusable. **Probably does not gate Phase 1a tracing** - the hazard is a `__torch_function__` guard on a real device, not a fake-tensor trace. | **Root-cause the known hang** at `atom/spec_decode/dspark_scheduler.py:264`. `rocgdb` attach, `info dispatches` per rank, identify which rank diverges. | <1 day, quiet node |

**Ordering.** T10 first: no hardware, largest swing per hour. Then **T5** — a
`FakeTensorMode` trace should be GPU-free and collective-free, so it is both the cheaper
experiment and the one that tells us whether T52 gates anything on the critical path.
**T52** follows, at ordinary priority unless T5 actually hits the hang. T21 and T25 need
the calibration and harness to exist, so they land later — but both are *designed-in
steps*, not add-ons, and neither should slip to the end.

**What each one costs if it fails.** T21 and T25 can each invalidate a whole acceptance
claim. T5 can invalidate a whole tier. T52 invalidates `--measure` and mode-based
instrumentation of a real run — narrower, but a designed path. Finding out late is the
expensive outcome in every case, which is the argument for putting them early rather than
where they naturally fall.

---

## 2. Missing topics

Design topics identified but not yet written up. Listed with a recommendation rather than
silently carried as gaps.

| # | Topic | Why it is missing, and why it matters | Recommendation |
|---|---|---|---|
| ~~M-a~~ | ~~Configuration surface and CLI~~ | Flags were scattered across five topics with no owner and no precedence rule. | **DONE** - topic `13_configuration_surface.md`, D78-D81. |
| ~~M-b~~ | ~~Refusal semantics end to end~~ | Seven documents emitted refusals and none said what one *does to the run*. | **DONE** - `08` D50.1: mark and continue, priced by the next answerable rung, with a >5%-of-predicted-seconds admissibility gate. |
| ~~M-c~~ | ~~Model loading without a GPU~~ | 关键技术点 1.4.2. Doc `03` covers memory *sizing*; nothing covers how the module tree comes into existence to be traced. ATOM has the pieces — `--load_dummy {empty,zero,xavier}`, `RapidServeModelRunner._init_weight_params_on_meta` — but no doc names the path or says whether weights are read at all. | **DONE** - `02` D10.1: HF-config geometry where only geometry is needed, construction inside `FakeTensorMode` with `--load_dummy empty` where a module tree is. |
| ~~M-d~~ | ~~TP / DP / PP / EP specifics~~ | M1 names all four, and DP couples *scheduling decisions* across ranks through a per-forward collective that rewrites the batch - so the LP structure is an M1 deliverable, not an M7 one. | **DONE** - topic `15_parallelism_support.md`, D88-D94, with an explicit M1/M7 split. |
| ~~M-e~~ | ~~Determinism and reproducibility~~ | Doc `08` **T26** asks for bit-reproducibility as a test, but nothing designs for it. Under a distributed CA, grant ordering is a function of real-time message arrival unless something pins it. Two runs of one configuration disagreeing would undermine every paired comparison. | **DONE** - `01` D3.4: the `(LP, virtual time, event)` sequence is what must reproduce; CA grants tie-break by LP identity. |
| ~~M-f~~ | ~~Speculative decoding / MTP~~ | Acceptance is a *behaviour* Compass cannot compute - the first quantity in the design that is neither derivable nor measurable. | **IN SCOPE** by decision 2026-09-19; topic `14_speculative_decoding.md`, D82-D87. Placed as **M3.5** (mechanism on Qwen3.8-27B), real claim at M5/M6. |
| ~~M-g~~ | ~~Simulated-run observability~~ | Doc `01` D3.1's open issue says the CA should own the global timeline log and the deadlock dump, and that "its output format is part of the acceptance evidence and should be designed, not improvised". Doc `11` covers Prometheus metrics, which is a different thing. Still improvised. | **DONE** - `01` D3.5: timeline log, deadlock dump, and an always-written run summary. |

**All seven are now closed.** M-a `13`; M-b `08` D50.1; M-c `02` D10.1; M-d `15`; M-e `01` D3.4;
M-f `14`; M-g `01` D3.5.

---

## 3. TODO register

### Topic 04 — model capture and cost IR

| # | Item |
|---|---|
| T1 | Inductor fusion correction (~4.8% of a decode step) |
| T2 | Enumerate the structure set for Qwen3.8-27B |
| T3 | Build the per-leaf parameter-extractor table (~20 entries) |
| T4 | Establish scratch constants per leaf for the 27B |
| T5 | Verify ATOM's model classes trace cleanly under FakeTensorMode at TP>1 |
| T6 | Validate that `Repeat` grouping reproduces the flat cost |
| T7 | Validate `Par` reconstruction from stream ids |
| T8 | Decide whether tier (a) is fitted independently or derived from tier (b) |
| T9 | Declare a row-ordering treatment for decode attention |
| **T51** | Enumerate the layer-pattern shapes for Qwen3.8-27B and Kimi-K3; confirm the nested-`Repeat` detector reaches the hierarchical form on both |
| **T52** | Root-cause the `TorchDispatchMode` 8-rank hang at `dspark_scheduler.py:264` — gates T5 |

### Topic 06 — workload harness contract

| # | Item |
|---|---|
| ~~T10~~ | ~~Verify `AgenticReplayStrategy` can be subclassed rather than vendored~~ — **done**: yes, but the adapter rebinds the shared `LoopScheduler` global instead, which covers all seven pacing sites |
| T11 | Build the per-tokenizer vetted filler-token set |
| T12 | Chase the 32 `asyncio.wait_for` sites under virtual time |
| T13 | Decide the simulated KV connector's completion semantic |
| T14 | Build the client-count matrix given only 144 fan-out-capable sessions |
| ~~T15~~ | ~~Warmup handling in the harness contract~~ — **done**: warmup requests are ordinary requests; the rule is an exclusion window agreed by request id |
| **T54** | Detect warmth that *recurs* mid-run (a new shape reaching autotune at minute 10); D62 measures leading warmth only |

### Topic 07 — calibration toolchain

| # | Item |
|---|---|
| T16 | Calibrate `compass plan`'s GPU-time estimates |
| T17 | Draw the boundary of the standalone `ModelRunner` bench |
| T18 | Verify a replayed step table reproduces the forward context faithfully |
| T19 | Decide the artifact store's physical form |
| T20 | Declared node + standalone benchmark for invisible collectives |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) |
| ~~T22~~ | ~~Analytic laws as their own design topic~~ — **done**, now `10` |
| **T55** | Decide the treatment of `c10d::broadcast_`, which takes a `ProcessGroup` no artifact can hold — working answer is to fold it into the Phase 1c host floor |

### Topic 08 — validation protocol

| # | Item |
|---|---|
| T23 | Choose the family-2 distance function |
| T24 | Define "structural event" for family 3 beyond prefill streaks |
| T25 | Measure the real-vs-real noise floor under closed-loop replay at high client count |
| T26 | Assert simulator bit-reproducibility as a test — mechanism now `01` D3.4; this item is the CI wiring |
| T27 | Decide the 256-client cell's construction |
| T28 | Establish whether ranking/regret becomes an explicit acceptance gate |
| ~~T48~~ | ~~What a refusal does to a run~~ - **DONE**: `08` D50.1, mark-and-continue with a 5%-of-seconds admissibility gate |

### Topic 09 — fitting and law selection

| # | Item |
|---|---|
| T29 | Choose the hull implementation (convex hull vs k-NN threshold) and its threshold |
| T30 | Enumerate candidate laws per leaf family, with held-out validation shapes |
| T31 | Test raggedness, cached fraction and chunk-position for treatment status |
| T32 | Decide whether tier (a) is derived from tier (b) or fitted independently |
| T33 | Establish `warmup_seconds` for Qwen3.8-27B under the current stack |

### Topic 10 — analytic laws

| # | Item |
|---|---|
| T34 | Test whether the activation coefficient is derivable from geometry |
| T35 | Derive FLOPs and bytes-moved expressions for the ~20 opaque leaves |
| T36 | Name the collective algorithm per code path in the machine spec schema |
| T37 | Decide whether the host floor is derivable or stays a per-model constant |
| T38 | Build the analytic-vs-measured ratio report as part of the empirical campaign |
| T39 | Establish a dispatch-band table from geometry where no measured bands exist |
| **T56** | Record the observed tier-0 error per measured device in the artifact, so an unmeasured-device user sees a range rather than a promise |

### Topic 11 — metrics support

| # | Item |
|---|---|
| T40 | Confirm the new histograms are **classic**, not native |
| T41 | Audit every `observe()` site; extend the AST test to observation arguments |
| T42 | Measure `collect_metrics()` per-step cost on the real side; decide decimation |
| T43 | Verify the backfill end to end — one block, loaded, visible in Grafana |
| T44 | Sanity-check histogram bucket ranges against simulated latencies |
| T45 | Tag ATOM's existing twenty metrics with their D77 class |
| T46 | Decide the DP-aggregation rule per class; refuse summaries there |

### Topic 13 — configuration surface

| # | Item |
|---|---|
| T57 | Generate the `ATOM_COMPASS_*` environment twins from the flag table rather than hand-writing them |
| T58 | Decide whether a per-leaf tier override is worth the reproducibility cost |

### Topic 14 — speculative decoding and MTP

| # | Item |
|---|---|
| T59 | Capture `ATOM_ENABLE_RELAXED_MTP` in the run fingerprint - it changes acceptance *semantics* (`RELAXED_TOP_N` 1 to 10, `RELAXED_DELTA` 0 to 0.6) and is invisible to every artifact key today |
| T60 | Test whether a draft forward's cost is linear in `K` - serial MTP should be, a real draft stack need not be |
| T61 | Decide how chunked prefill and drafting interact, and what structure that produces |
| T62 | Assert the host acceptance draw and the Triton kernel agree: same declared rates, same seed, same accepted-count distribution over a few thousand draws |
| T63 | Add ATOM flag `--spec-decode-acceptance-rates` (list) - contract 2 has no transport today; the CLI exposes only the two scalars |

### Topic 16 — execution plan

| # | Item |
|---|---|
| T71 | Add Wave 4+ detail as Phase 0 and T21 answers arrive |
| T72 | Decide whether reviewer agents use ATOM's `review-pr` skill or a Compass-specific checklist |

### Topics 02, 01 — gaps now closed

| # | Item |
|---|---|
| T68 | Enumerate buffer allocations in the two target models and confirm none escapes `FakeTensorMode` - `--load_dummy` and the meta wrapper act on parameters only |
| T69 | Whether a quantized checkpoint's geometry is derivable without reading it; header derivation was 3.3% low at TP=4 on the 27B |
| T70 | Estimate the timeline log's volume at PP degree > 1 - grants scale with stages and microsecond lookahead, so it is largest exactly where it is most wanted |

### Topic 15 — parallelism support

| # | Item |
|---|---|
| T64 | Establish whether ATOM microbatches PP - changes the LP event rate and the bubble model |
| T65 | Establish EP's group membership per supported configuration; if EP spans DP, the LP collapse does not hold |
| T66 | Measure whether the Class-C runtime constants move with PP degree |
| T67 | Measure the step-duration spread across DP ranks, and what padding to `unified_bs` costs |

### Topics 01, 03, 05 — newly opened

| # | Item | Topic |
|---|---|---|
| **T47** | A lookahead that is wrong but never exercised by the workload is not detected by the straggler check | `01` |
| **T49** | The prefix-index *lookup* cost is charged to nobody — ~1,387 blocks hashed and probed per request at the cc-traces p50, magnitude unmeasured | `03` |
| **T50** | Whether runtime memory constants transfer across dies (the working assumption says yes within a software generation) | `03`, `05` |
| **T53** | Whether tokenizer throughput transfers across CPU classes (the working assumption says yes, adjusted by derate) | `05` |

---

## 4. Cross-cutting issues

Beyond the per-topic TODOs.

1. **Silent failure is the dominant risk mode.** The always-on causality detectors
   (`01` D3.2), loud deadlock aborts, and the AST clock-site test are the design, not
   decoration.
2. **Simulation speed is unmeasured under this architecture.** The prior design ran
   **0.30×** under saturation — slower than the system it simulates. Measure a saturated
   cell early, not at the end.
3. **Per-step replay CPU cost** was 4.3 ms against a 32.7 ms modelled step, and only with
   the bound allocation in the cache key; a shape-only key is unsound.
4. **Schedule agreement must be reported separately from latency**, and it was established
   two changes *before* the latency numbers were right.
5. **Scheduling fidelity is unobservable at saturation.** Some cell needs deliberate slack.
6. **Multi-node DP is implemented but not hardware-validated** — `docs/distributed_guide.md`
   §9 carries the banner. If paired evidence is needed there, the real side may not exist.
7. **A user guide is a deliverable, not documentation debt.** `compass plan` is designed
   so the tool tells the user what to measure, which only works if the flows, the
   recommended calibration sequence and the flag surface are written down for a reader
   who was not in these design conversations. Requested during review; belongs in the
   execution plan as its own task, not as a trailing chore.
8. 7. **ATOM's `main` moves while Compass is built.** The seam (`Config.runner_qualname`) has
   two in-tree users so it is unlikely to vanish, but the ~55 synchronization sites of
   `01` D4 and the clock-read sites of `11` D72 are ordinary code that upstream will
   touch. The CI clock-source lint is the detector; a rebase cadence is an execution-plan
   question.

---

## 5. Pending amendments

Corrections identified while writing later documents, not yet applied to earlier ones.

| Document | Amendment |
|---|---|
| `02`, `04` | "Two tiers" becomes **three** — analytic/roofline is a tier in its own right (`07` D36), not merely rung 4 of the resolver ladder |
| `02` | Extend `CostBackend` with the resolver ladder and compositional `provenance_mix` |
| `05` | D26's `transfer` probe is **deferred**: no cross-hardware transfer of empirical data (`07` D36) |
