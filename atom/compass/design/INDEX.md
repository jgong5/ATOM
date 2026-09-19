# ATOM Compass — Design Index

**Status:** all documents are drafts for review. Drafted by an AI assistant during a design
interview; none has been reviewed or approved, and no code has been written against them.

File numbers reflect the order documents were written, **not** the order to read them. This
index gives the reading order, the decision map, and the open work.

---

## Start here

| If you are… | Read |
|---|---|
| new to the project | `00_initial_prompt.md`, then Part I below |
| reviewing a specific decision | find it in the decision map, go straight to that document |
| about to implement | Part I → the document owning your area → the TODO register |
| wondering what is *not* settled | the TODO register and the cross-cutting issues, both below |

---

## Reading order

### Part I — Architecture: what a simulated run *is*

| Doc | Title | What it settles |
|---|---|---|
| `01` | Execution and Time Model | Keep ATOM's multi-process topology. A central **Clock Authority** grants virtual time; logical processes collapse onto ATOM's existing hardware barriers. Which waits are rewritten, annotated, disabled or ignored. KV transfer simulated; Atomesh untouched; arrivals via a next-arrival bound. |
| `02` | Model Runner Seam and Cost Backend | Attach at `ModelRunner.forward`, delivered by a `--runner-qualname` subclass — **no ATOM change for the injection**. The runner has no modes; the algorithm comes from a pluggable backend. The milestone-1 fake model. |

### Part II — What is modelled

| Doc | Title | What it settles |
|---|---|---|
| `05` | Machine Specification and its Probes | The one input artifact describing device, host, interconnect and the software stack it is pinned to. Read this before `03` and `04`, both of which consume it. Probe tools and their contamination refusals. |
| `03` | Memory Model and the KV Pool | **Do not simulate the KV pool** — ATOM's real one is pure arithmetic. Substitute the five device readings, never the arithmetic. Validate per term, never as a sum. |
| `04` | Model Capture and the Cost IR | Three tiers of cost model. Capture with `TorchDispatchMode` + `FakeTensorMode` + `ShapeEnv`. A hierarchical, symbolic, stream-annotated IR. **Opaque leaves** are priced, not decomposed. |
| `10` | Analytic Laws (Tier 0) | Cost and memory from device parameters and model geometry, with no measurement of the subject. Three classes: exact from geometry, device-parameterised, policy-determined. **The most speculative document here.** |

### Part III — How the data is made

| Doc | Title | What it settles |
|---|---|---|
| `07` | Calibration and Benchmarking Toolchain | `compass plan` as the single entry point. The minimal calibration recipe: one full-engine run at TP1, plus one TP2 transfer test, plus a startup per width. The artifact store, its keys and its invalidation matrix. |
| `09` | Fitting and Law Selection | How measurements become a model: fit relative error, per-rung decode, hull guards not bounding boxes, and why leave-one-out cannot choose a family. |

### Part IV — How it is driven and judged

| Doc | Title | What it settles |
|---|---|---|
| `06` | Workload Harness Contract | A three-part contract, not a bespoke client. agentx-harness reused with **zero edits** via an out-of-tree plugin. Timeline piggybacked on `kv_transfer_params` so Atomesh needs no change. Tokenizer cost is a queue, not a constant. |
| `08` | Validation Protocol | Three separable results, never one number. **The real-vs-real spread is the tolerance.** A metric is admissible only if stable *and* sensitive. Three metric families ordered by robustness. |
| `11` | Engine Metrics under Virtual Time | ATOM's Prometheus exporter under a virtual clock. Metrics are classified by the **provenance of their value**, not their type. Sample once per engine step &mdash; virtual time is discrete-event. Both metrics clock reads stay real. |

### Not yet written

| Doc | Title | Status |
|---|---|---|
| — | Execution Plan | next, and last |

---

## Five findings that shaped everything

Anyone who reads only one section should read this one.

1. **Aggregate cost accuracy does not bound schedule accuracy.** A prior run was within
   **1.0% on prefill seconds, 1.1% on decode, 1.0% on run length — and 90% wrong on median
   TTFT**, because TTFT hung on two comparisons with 1.5 s and 1.2 s of slack. Hence `08`
   D44's three separable results.
2. **Aggregates hide compensating errors — at least six times.** A +13.8% memory sum was
   three errors two of which cancelled, the largest being 25% of one term. A cc-traces
   latency within 5% was **+52% TTFT against −30% decode**. Hence per-term validation
   everywhere.
3. **Every failure mode in this design is silent.** A causality violation, a specialized
   trace, a dead gate, a contaminated reading — none of them crash. They produce a
   plausible table. Hence assertions that are always on, loud aborts, and gates whose state
   is written into the artifact rather than inferred from a flag.
4. **Concentration rescues the cost model.** Sixteen operators cover a step, half the time
   in two, twelve kinds cover 99.7%. 60–75% of step time sits in 8–20 **opaque leaves**. So
   the job is pricing ~20 leaves well, not decomposing everything.
5. **The instrument has a floor.** Pricing repeats to 0.96% summed, 1.18% median per
   signature, **p90 32%**. *Any residual quoted below about 2% is quoting the instrument,
   not the model.*

---

## Decision map

| Decisions | Document |
|---|---|
| D0 – D9 | `01` Execution and Time Model |
| D10 – D12 | `02` Model Runner Seam and Cost Backend |
| D13 – D16 | `03` Memory Model and the KV Pool |
| D17 – D23 | `04` Model Capture and the Cost IR |
| D24 – D26 | `05` Machine Specification and its Probes |
| D27 – D35 | `06` Workload Harness Contract |
| D36 – D43 | `07` Calibration and Benchmarking Toolchain |
| D44 – D52 | `08` Validation Protocol |
| D53 – D62 | `09` Fitting and Law Selection |
| D63 – D70 | `10` Analytic Laws (Tier 0) |
| D71 – D77 | `11` Engine Metrics under Virtual Time |

### Headline decisions

| # | Decision |
|---|---|
| D1 | Keep ATOM's multi-process and multi-thread topology; every change is additive |
| D3 | A central **Clock Authority**; grant rule `min_j(now[j] + L[j→i])`; logical processes collapse onto existing hardware barriers |
| D4 | Four-category wait contract: ~6 rewritten, ~10 annotated, ~20 disabled, ~20 ignored |
| D10 | Attach at `ModelRunner.forward` via `--runner-qualname`; no ATOM change for the injection |
| D13 | Do not simulate the KV pool; run ATOM's real one |
| D14 | Substitute the five device readings, never the arithmetic |
| D18 | Capture with dispatch mode + FakeTensor + ShapeEnv; `_EnablePythonDispatcher()` is mandatory |
| D19 | Hierarchical, symbolic, stream-annotated IR; no branches in the IR — applicability is a discrete key plus an evaluated guard domain |
| D20 | Opaque leaves are priced, not decomposed; each carries a declared parameter extractor |
| D28 | Additive optional fields on the **real** endpoint, both directions |
| D30 | Piggyback the simulated timeline on `kv_transfer_params`; Atomesh needs zero changes |
| D36 | Three tiers: analytic, coarse empirical, op-level empirical. Compass models a device it has been measured on |
| D37 | `compass plan` — the tool tells the user what to measure |
| D45 | The real-vs-real spread is the tolerance; a metric must be stable **and** sensitive |

---

## Open TODO register

| # | Item | Doc |
|---|---|---|
| T1 | Inductor fusion correction (~4.8% of a decode step) | `04` |
| T2 | Enumerate the structure set for Qwen3.8-27B | `04` |
| T3 | Build the per-leaf parameter-extractor table (~20 entries) | `04` |
| T4 | Establish scratch constants per leaf for the 27B | `04` |
| T5 | Verify ATOM's model classes trace cleanly under FakeTensorMode at TP>1 | `04` |
| T6 | Validate that `Repeat` grouping reproduces the flat cost | `04` |
| T7 | Validate `Par` reconstruction from stream ids | `04` |
| T8 | Decide whether tier (a) is fitted independently or derived from tier (b) | `04` |
| T9 | Declare a row-ordering treatment for decode attention | `04` |
| T10 | Verify `AgenticReplayStrategy` can be subclassed rather than vendored | `06` |
| T11 | Build the per-tokenizer vetted filler-token set | `06` |
| T12 | Chase the 32 `asyncio.wait_for` sites under virtual time | `06` |
| T13 | Decide the simulated KV connector's completion semantic | `06` |
| T14 | Build the client-count matrix given only 144 fan-out-capable sessions | `06` |
| T15 | Warmup handling in the harness contract | `06` |
| T16 | Calibrate `compass plan`'s GPU-time estimates | `07` |
| T17 | Draw the boundary of the standalone `ModelRunner` bench | `07` |
| T18 | Verify a replayed step table reproduces the forward context faithfully | `07` |
| T19 | Decide the artifact store's physical form | `07` |
| T20 | Declared node + standalone benchmark for invisible collectives | `07` |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) | `07` |
| ~~T22~~ | ~~Analytic laws as their own design point~~ — **done**, now `10` | `07` |
| T23 | Choose the family-2 distance function | `08` |
| T24 | Define "structural event" for family 3 beyond prefill streaks | `08` |
| T25 | Measure the real-vs-real noise floor under closed-loop replay at high client count | `08` |
| T26 | Assert simulator bit-reproducibility as a test | `08` |
| T27 | Decide the 256-client cell's construction | `08` |
| T28 | Establish whether ranking/regret becomes an explicit acceptance gate | `08` |
| T29 | Choose the hull implementation and its threshold | `09` |
| T30 | Enumerate candidate laws per leaf family, with held-out validation shapes | `09` |
| T31 | Test raggedness, cached fraction and chunk-position for treatment status | `09` |
| T32 | Decide whether tier (a) is derived from tier (b) or fitted independently | `09` |
| T33 | Establish `warmup_seconds` for Qwen3.8-27B under the current stack | `09` |
| T34 | Test whether the activation coefficient is derivable from geometry | `10` |
| T35 | Derive FLOPs and bytes-moved expressions for the ~20 opaque leaves | `10` |
| T36 | Name the collective algorithm per code path in the machine spec schema | `10` |
| T37 | Decide whether the host floor is derivable or stays a per-model constant | `10` |
| T38 | Build the analytic-vs-measured ratio report as part of the empirical campaign | `10` |
| T39 | Establish a dispatch-band table from geometry where no measured bands exist | `10` |
| T40 | Confirm the new histograms are **classic**, not native | `11` |
| T41 | Audit every `observe()` site; extend the AST test to observation arguments | `11` |
| T42 | Measure `collect_metrics()` per-step cost on the real side; decide decimation | `11` |
| T43 | Verify the backfill end to end &mdash; one block, loaded, visible in Grafana | `11` |
| T44 | Sanity-check histogram bucket ranges against simulated latencies | `11` |
| T45 | Tag ATOM's existing twenty metrics with their D77 class | `11` |
| T46 | Decide the DP-aggregation rule per class; refuse summaries there | `11` |

### The four I would settle first

- **T21** — the calibration recipe's one unproven assumption. Overhead constants have
  already been measured moving **12% and 7% in opposite directions** between TP1 and TP2 on
  the same GPUs.
- **T25** — the noise floor under closed-loop replay. All prior data is 20 requests, one
  session, declared arrivals. If the floor is wide at 256 clients, those cells are
  ungradeable.
- **T10** — an hour of work that swings the adapter estimate by 2,000 lines.
- **T5** — everything in `04` rests on it, and it needs a non-wedged node.

---

## Pending amendments

Corrections identified while writing later documents, not yet applied to earlier ones.

| Document | Amendment |
|---|---|
| `02`, `04` | "Two tiers" becomes **three** — analytic/roofline is a tier in its own right (`07` D36), not merely rung 4 of the resolver ladder |
| `02` | Extend `CostBackend` with the resolver ladder and compositional `provenance_mix` |
| `05` | D26's `transfer` probe is **deferred**: no cross-hardware transfer of empirical data (`07` D36) |

---

## Cross-cutting issues

Beyond the per-document TODOs, from `01`'s register.

1. **Silent failure is the dominant risk mode.** The always-on assertions, loud deadlock
   aborts and the AST clock-site test are the design, not decoration.
2. **Simulation speed is unmeasured under this architecture.** The prior design ran
   **0.30×** under saturation — slower than the system it simulates. Measure a saturated
   cell early.
3. **Per-step replay CPU cost** was 4.3 ms against a 32.7 ms modelled step, and only with
   the bound allocation in the cache key; a shape-only key is unsound.
4. **Schedule agreement must be reported separately from latency**, and it was established
   two changes *before* the latency numbers were right.
5. **Scheduling fidelity is unobservable at saturation.** Some cell needs deliberate slack.
6. **Multi-node DP is implemented but not hardware-validated** — `docs/distributed_guide.md`
   §9 carries the banner. If paired evidence is needed there, the real side may not exist.
