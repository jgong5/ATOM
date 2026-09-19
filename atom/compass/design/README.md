# ATOM Compass — Design

**Status: design only.** Nothing here has been implemented, and every document carries a
header marking it as an unreviewed draft. 78 decisions (D0–D77) and 46 open TODOs (T1–T46)
are tracked at the end of this file.

---

## What Compass is

A performance simulator for ATOM. It answers one question:

> **For this model, this parallelism, and this workload — what latency and throughput, and
> does it fit in memory?**

Memory is part of the question, not an afterthought: it decides which configurations exist
at all, and predicting the speed of a configuration that cannot start is answering the
wrong question.

## The one idea that explains all the others

> **Compass replaces only the forward pass.**

ATOM's real scheduler, real block manager, real admission logic and real prefix cache all
run unchanged. A simulated run therefore makes **the same scheduling decisions** as a real
one; it only substitutes a predicted duration for the work.

Almost every decision in these documents follows from taking that literally. Compass does
not model serving *decisions* — only the time they consume.

### What is modelled

| | |
|---|---|
| **time** | what a step costs, and what serving adds around it |
| **memory** | what a configuration consumes, so whether it fits and how many KV blocks it gets |
| **KV cache pool** | not simulated — ATOM's real one runs, because it is pure arithmetic over integers |

### What is not

Serving decisions. Fragmentation. Neighbour contention. A device that has never been
measured (except at analytic fidelity). See **Scope and non-goals** below.

---

## Architecture

### A. Where Compass sits

Everything above the dashed line is ATOM's own code, running unmodified.

```
   harness (any vendor)            ATOM, unmodified
   +--------------+         +-----------------------------+
   |  agentx /    |  HTTP   |  api_server                 |
   |  aiperf +    |-------->|  LLMEngine / CoreManager    |
   |  adapter     |<--------|  Scheduler                  |
   +--------------+         |  BlockManager / BlockPool   |
          ^                 |  prefix cache, admission    |
          |                 +--------------+--------------+
          |                                |
          |                    ScheduledBatch |  ScheduledBatchOutput
   - - - -|- - - - - - - - - - - - - - - - -|- - - - - - - - - - - - -
          |                                v
          |                 +-----------------------------+
          |                 |  CompassModelRunner         |   <-- THE SEAM
          |                 |  (--runner-qualname)        |
          |                 |                             |
          |                 |  no weights, no KV tensors, |
          |                 |  no GPU. Returns a PREDICTED|
          |                 |  duration + shaped output.  |
          |                 +--------------+--------------+
          |                                |
          |                                v
          |                 +-----------------------------+
          +---------------- |  Clock Authority            |
            grants,         |  grants virtual time to     |
            blocked/running |  each logical process       |
                            +-----------------------------+
```

The seam needs **no ATOM change**: `Config.runner_qualname` already exists and already has
two in-tree users.

### B. A simulated step, end to end

```
  harness                api_server        Scheduler         CompassModelRunner
     |                       |                 |                     |
     |--- POST /v1/... ----->|                 |                     |
     |                       |-- tokenize ---->|                     |   modelled as a
     |                       |   (real)        |                     |   bounded-width queue
     |                       |--- add_request->|                     |
     |                       |                 |-- schedule() ------>|   REAL decisions:
     |                       |                 |   ScheduledBatch    |   admission, chunking,
     |                       |                 |                     |   prefix hits, preemption
     |                       |                 |                     |
     |                       |                 |                     |-- cost model
     |                       |                 |                     |   -> predicted seconds
     |                       |                 |<-- BatchOutput -----|
     |                       |                 |    + predicted dt   |
     |                       |                 |                     |
     |                       |          advance_to(now + dt) --------------> Clock Authority
     |                       |                 |                     |
     |                       |<-- postprocess -|                     |
     |<-- SSE chunk ---------|                 |                     |
     |    + sim_arrive       |                 |                     |
     |      sim_first_token  |                 |                     |
     |      sim_finish       |                 |                     |
```

Wall-clock time passes while that HTTP round trip happens. **Virtual time does not** — it
advances only for durations the cost model produced.

### C. Offline / online split

This is what makes "no GPU at simulation time" work.

```
  OFFLINE, needs a GPU, run rarely          ONLINE, device-free, every run
  ================================          ==============================

   spec probes      Phase 1b price
   Phase 1c in-situ Phase 2 memory
        |                  |
        v                  v
   +--------------------------------+       +---------------------------+
   |        ARTIFACT STORE          |------>|  Phase 0  discovery       |
   |  key + digest + fingerprint    |       |  Phase 1a trace (JIT)     |
   |                                |<------|  cost model -> seconds    |
   |  machine_spec   price_list     |  JIT  |  memory model -> blocks   |
   |  op_graph       region_terms   | trace |                           |
   |  memory_readings coverage_hull |       |  -> simulated run         |
   +--------------------------------+       +---------------------------+
```

Artifacts are keyed, digested and fingerprinted; a stack or source change **refuses**
rather than answering from stale data.

### D. The three cost tiers

```
  +-------------------------------------------------------------+
  |  tier b   op-level empirical                                 |
  |           symbolic IR + priced leaves + region terms         |
  |           needs: a calibration campaign                      |
  |           used for: acceptance cells                         |
  +-------------------------------------------------------------+
  |  tier a   coarse empirical                                   |
  |           fitted over features ScheduledBatch already carries|
  |           needs: measured steps                              |
  |           used for: structure discovery, sweeps, plumbing    |
  +-------------------------------------------------------------+
  |  tier 0   analytic / roofline                                |
  |           machine spec + model geometry ONLY                 |
  |           needs: nothing                                     |
  |           used for: day one; a device never measured         |
  +-------------------------------------------------------------+
```

Orthogonal to the tier, every individual answer carries its **provenance** — measured,
fitted, interpolated, extrapolated or analytical — and a step reports the mixture.

---

## Acceptance targets

| Requirement | Target |
|---|---|
| Throughput prediction | error ≤ 10% |
| Time per output token (TPOT) | error ≤ 10% |
| Time to first token (TTFT) | error ≤ 10% |
| Each non-KV memory term | error ≤ 10% |
| KV capacity / block count | **error ≤ 5%** |
| Generalization | predict beyond the configurations used for calibration |
| Simulation speed | aim ≥ 5×; negotiable, but **faster than a real run** is the floor |

Final proof is **paired simulated and real execution of cc-traces proper**. Milestones M1–M7
are in `00_initial_prompt.md`.

---

## Scope and non-goals

Stated here so they are not discovered at review.

| Not in scope | Where it is recorded |
|---|---|
| A device that has never been measured — except at tier-0 fidelity, which is derate-dominated | `10` D69 |
| Neighbour contention. Compass models a **dedicated** device, so it will not predict an OOM a shared box produces | `03` D14 |
| Memory fragmentation — not modelled, not planned, and nobody models it | `03` D16 |
| **Cancellation** — `status` is `"completed"` on all 1,697 subagent wrappers in both corpora. There is nothing to replay. | `06` D35 |
| Closed-loop *arrival generation*. The corpus records no parent/child completion dependency; the harness reconstructs joins. | `06` D35 |
| Asymmetric parallelism beyond what the milestones name | `01` D2 |

---

## Load-bearing assumptions — what would change our mind

Four assumptions hold up large parts of the design. Each is cheap to test and none has been.

| # | Assumption | If false |
|---|---|---|
| **T21** | The in-situ calibration transfers across TP width | the recipe's "calibrate at TP1, predict TP2/4/8" collapses and the campaign multiplies. The overhead constants have **already** been measured moving 12% and 7% *in opposite directions* between TP1 and TP2 on the same GPUs. |
| **T25** | The real-vs-real noise floor stays narrow under closed-loop replay at high client count | those cells become ungradeable. All prior data is 20 requests, one session, declared arrivals. |
| **T5** | ATOM's model classes trace cleanly under `FakeTensorMode` at TP>1 | tier b has no IR, and docs `04`, `07` and `09` rest on it |
| **T10** | `AgenticReplayStrategy` can be subclassed rather than vendored | the harness adapter grows by ~2,000 lines to keep in sync with upstream |

---

## Five findings that shaped everything

Anyone reading one section should read this one.

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
   plausible table. Hence always-on assertions, loud aborts, and gates whose state is
   written into the artifact rather than inferred from a flag.
4. **Concentration rescues the cost model.** Sixteen operators cover a step, half the time
   in two, twelve kinds cover 99.7%. 60–75% of step time sits in 8–20 **opaque leaves**. So
   the job is pricing ~20 leaves well, not decomposing everything.
5. **The instrument has a floor.** Pricing repeats to 0.96% summed, 1.18% median per
   signature, **p90 32%**. *Any residual quoted below about 2% is quoting the instrument,
   not the model.*

---

## Glossary

The documents use these precisely; a reader will bounce off without them.

| Term | Meaning |
|---|---|
| **provenance** | how an answer was obtained. `analytical` (computed without measuring the subject) / `measured` (the unit itself was timed) / `fitted` (a form chosen, coefficients regressed) / `interpolated` (nearby measurements looked up) / `extrapolated` (asked outside the measured range). *"Priced" is not a species* — a price is `measured`, differing only in unit. |
| **opaque leaf** | an operator the tracer records as one node whose internal kernels are invisible, because ATOM deliberately registers a whole subsystem as a single dispatcher op. ~20 of them hold 60–75% of step time. |
| **logical process (LP)** | the unit the Clock Authority coordinates. *Not* an OS process — a TP group is one LP, a DP group is one LP, because both already have a hardware barrier. |
| **structure** | a distinct operator list, independent of shape. Prefill has two (chunked, unchunked); decode has one across all CUDA-graph rungs. Traces are cached per structure, not per shape. |
| **treatment** | a factor that moves cost and is invisible to every feature in the model. Decode **row order** is one, worth 1.77× at a fixed context multiset. |
| **tier** | which cost model was asked (0 analytic / a coarse / b op-level). Orthogonal to provenance. |
| **refusal** | a declined answer with a named reason. Refusals are results, not errors, and are preferred to fallbacks everywhere. |

---

## Conventions

- Every document opens with a **status header**. None has been reviewed.
- Decisions are numbered **`D*`**, continuous across documents, with a decision log at the
  end of each. TODOs are **`T*`**, likewise continuous.
- Documents cite each other by number and decision — *"`07` D41"*.
- Claims carry their measurement. A number without a source is a defect.
- **Refuse rather than fall back.** The archetypal failure was an oracle asked about a step
  kind it had no samples of falling back to the mean of an empty list — zero — producing
  *"a confident, precise, entirely fictional answer"*: a TTFT of 0 ms against a real 7.6 s.

---
## Start here

| If you are… | Read |
|---|---|
| new to the project | the top of this file, then `00_initial_prompt.md` for the original task, then Part I below |
| reviewing a specific decision | the decision map below, then straight to that document |
| about to implement | Part I, then the document owning your area, then the TODO register |
| wondering what is *not* settled | **Load-bearing assumptions** above, then the TODO register and cross-cutting issues below |

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

This file replaces the earlier `INDEX.md`.

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
