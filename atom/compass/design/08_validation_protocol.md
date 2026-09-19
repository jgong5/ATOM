# ATOM Compass — Design Point 8: The Validation Protocol

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Depends on:** `07_calibration_toolchain.md` (the artifacts being validated),
`06_workload_harness_contract.md` (the workload driving both sides).

**Scope.** What counts as evidence: which results are reported, how each is measured, what
tolerance applies and where that tolerance comes from, and what invalidates a run.
Acceptance is paired simulated and real execution of cc-traces proper, so every rule here
applies symmetrically to both sides.

---

## D44. Three separable results, never one number

### The finding this exists to respect

The prior effort reached, on one run: **prefill seconds within 1.0%, decode within 1.1%,
run length within 1.0% — and median TTFT wrong by 90%.**

> *"Aggregate cost accuracy does not bound schedule accuracy when the scheduler has a
> discontinuity in it."*

TTFT there was decided by two comparisons with **1.5 s and 1.2 s of slack**. Prefill-first
means a decode window opens only in the gap between one request's last prefill chunk and
the next request's arrival; the simulated run reached the same point 3.9 s later, both
margins went negative, both windows closed, and ten requests got a first token at 271.6 s
instead of 213.1 and 231.2.

### Decision

**Three results, reported side by side, always:**

| Result | Question | Measured on |
|---|---|---|
| **cost accuracy** | is a step priced correctly? | the **real** run's own step sequence |
| **schedule agreement** | does the engine do the same things? | both step tables |
| **end-to-end** | TTFT / TPOT / throughput | the engine's own readings |

And the standing prohibition:

> **Aggregate latency is the one number this benchmark must never be judged on.** It would
> have called the first cc-traces pilot a success at −5.0% and the second a regression at
> −48.4%, when the second was the more truthful model.

That −5.0% was **+52% TTFT cancelling −30% decode**. Compensating errors inside an
aggregate have happened here at least six times.

---

## D45. Noise-floor first, and a two-sided admissibility test

### Problem

Every schedule metric needs a tolerance, and an arbitrary one is indefensible. Two *real*
runs do not agree exactly either — so the question is not "how close must simulation be"
but "how close is real to itself".

### Decision

**The real-vs-real spread is the tolerance. Measure it; do not assume it.**

```
  1. run the REAL side N>=3 times, SPACED (not back to back)
  2. metric(real_i, real_j) for all pairs        -> the NOISE FLOOR
  3. metric(sim, real_i) for each i              -> the RESULT
  4. GATE: the result lies inside the noise floor
```

Self-calibrating per cell, and it makes an ungradeable cell visible instead of passed or
failed: **if a cell's real-vs-real spread swamps the 10% acceptance target, that cell
demonstrates nothing** and must be reported as such.

**Repeats must be spaced or interleaved.** Consecutive repeats reported **−2.2% ± 0.1**
where five separated runs of the same command spanned **−6.1 / −14.0 / −14.7 / −14.4 /
−0.5%, mean −9.9 ± 6.4 — a standard deviation larger than the mean**. *"A tight sd from
consecutive repeats is a lower bound and not a measurement."*

### What the noise floor looked like before, and why it may not hold

| | spread |
|---|---|
| 27B, 4 real runs: streak structure | **identical** (36, 6, 42, 7, 15), breaks at the same two places, margins varying by at most **0.1 s** |
| 27B, 4 real runs: TTFT | 27.54 / 27.57 / 27.61 / 27.62 s — **0%** |
| 27B, 4 real runs: latency | 74.50–74.89 s — **1%** |
| 0.6B, 5 real runs: TTFT | 3.55–3.89 s — **9%** |

The conclusion then was *"the real machine is not on the knife edge; only the simulator
is."* **That was 20 requests from one session under declared arrivals.** Under closed-loop
replay at 256 clients, request *k+1* arrives at `finish(k) + delay`, so step-time jitter
feeds back into arrival times and *which* requests coexist depends on arrival ordering
across independent sessions. Mitigating: within-session inter-arrival p50 is **5.61 s** and
`think_time` p50 is **1.17 s**, against ~1% latency jitter. But that is an argument, not a
measurement, and this protocol replaces it with one.

### A metric is admissible only if it is both stable and sensitive

- **Stable** — small real-vs-real spread. Otherwise it cannot discriminate.
- **Sensitive** — it moves under a known perturbation. Otherwise it cannot grade a model.

**The sensitivity control already exists and is measured.** Re-price prefill at **×0.97**
and re-simulate:

| prefill price | TTFT median | prefill streaks | longest |
|---|---|---|---|
| real | 27.63 s | 5 | 42 |
| ×1.00 | 52.40 s | 6 | 63 |
| **×0.97** | **37.31 s** | **8** | **42** |
| ×1.03 | 54.33 s | 6 | 63 |

At ×0.97 the 63-chunk streak splits back into the real run's exact **42 + 7 + 15**,
breaking where the real run breaks (213.7 vs 213.1 s; 230.6 vs 231.2 s). **Any candidate
metric is run against this counterfactual. One that does not respond to a 3% price change
cannot grade a cost model.**

A metric failing either test is reported **inadmissible**, not silently dropped.

---

## D46. Three metric families, ordered by robustness

An extensible registry. Each metric declares a **family**, an **extractor** (step table →
value) and a **comparator** (value, value → distance). Admission is measured by D45's
two-sided test, not asserted. New metrics are expected; the families and the admission
procedure are the stable part.

### Family 1 — counting invariants. Exact match expected; a mismatch is a defect, not noise.

| Metric | Prior observation |
|---|---|
| prefill step count | 106 on both sides |
| decode step count | 4,346 vs 4,333 |
| total prefill tokens, total decode tokens | pure workload arithmetic — should be exact |
| preemption count | 0 in the runs measured |
| prefill streak count | identical across four real runs |
| requests admitted / completed | — |

### Family 2 — distributional. The family to weight most.

Compare **histograms**, never step-*k*-to-step-*k*: batch-width distribution,
scheduled-token distribution, per-step context-length distribution, cached-fraction
distribution. Graded by a distance (KS or Wasserstein) against the real-vs-real distance
for the same quantity.

Two reasons this family carries the most weight:

1. **It sidesteps alignment drift entirely** — no step has to correspond to a step, which
   is exactly the failure mode a per-step alignment rate suffers from. A per-step rate is
   dominated by long uniform decode runs, so it reads ~99% while the handful of steps that
   decide TTFT all disagree.
2. **The batch-width histogram is literally what determines throughput**, so it is
   meaningful rather than merely computable.

### Family 3 — event timing. Most diagnostic, least robust.

Structural events — prefill streak boundaries, admission points, preemptions — matched in
order, reporting the offset distribution of matched boundaries. Graded against the
real-vs-real offsets, which on the 27B were **0.1 s**.

### What does not need measuring

Scheduler **policy** agreement. **0 of 4,378 real decode steps and 0 of 4,621 simulated
ones** took a decode step while an arrived request still owed prefill. *"There is no
scheduling disagreement to find."* Keep the decision-record diff as a **regression guard**,
not a discriminator.

### Open issues

- The distance function for family 2 is unchosen (KS vs Wasserstein vs something
  workload-aware).
- Family 3's "structural event" needs a definition that survives past prefill-first
  scheduling; streaks may be an artefact of that policy.

---

## D47. Cost accuracy, on a fixed real step sequence

### Method

Apply the cost model to the **real** run's own steps. A within-run comparison that
separates cost error from schedule error by construction. *"Use the real sequence to assess
cost predictions first; use controlled durations to assess the scheduler separately."*

### Reporting rule: never the total alone

The tool reports **three numbers side by side** — on totals, holding out the largest single
contributor, and the median per step:

```
prefill  106 steps   measured 235.56 s   priced 228.20 s
  on totals                 -3.13 %
  holding out step 0        -0.27 %      <- the unmodelled cold start
  median per step           -0.09 %

decode  4346 steps   measured  73.17 s   priced  73.10 s
  on totals                 -0.09 %
  holding out step 148      -0.07 %
  median per step           +0.09 %
```

One step moved the prefill total by **3.0 points**. *The total was describing that step,
not the model.*

### The sign trap, and why the running sum is what matters

Prefill priced against a real run's own steps came to **+0.10% overall** but **+3.06%
excluding step 0** — a single cold-start error of **−6.75 s** almost exactly offsetting
**+6.99 s** spread over the other 105 chunks.

> *"The scheduler never sees the total. It sees the running sum, which is 3% high from the
> second step onward and never gets the 6.75 s back."*

So cost accuracy must be reported **with outliers held out**, and the cold start priced or
excluded deliberately rather than left to cancel a real bias.

---

## D48. Memory: per term, never as a sum

### The rule

A summed check reported **+13.8%**. It was three errors, two of which cancelled: weights
over-counted by +0.280 GB (a tied head never resident), activations compared at the wrong
shape (−0.015 GB), and −0.084 GB of *"a resident term nobody had noticed was there at
all"*. **The largest single error was 25% of a term and the sum said 13.8%.**

`peak_torch` was consequently split at source into three non-subtractive readings —
`parameter_bytes`, `weights_torch`, `current_torch` — so a sum can never hide a term again.

Validation costs **no GPU time**: every hardware run already prints the real breakdown.

### The decision the byte error exists to produce

| | modelled | measured | error |
|---|---|---|---|
| 0.6B TP=1 | 96,033 | 96,051 | **−0.02%** |
| 0.6B TP=2 | 183,709 | 183,745 | **−0.02%** |
| 0.6B TP=4 | 367,614 | 367,364 | **+0.07%** |
| 27B TP=2 | 37,825 | 37,858 | **−0.09%** |

TP=1 and TP=2 land **under**, which is the safe direction. TP=4's +250 blocks is
contamination — the calibration takes the minimum `non_torch` across ranks and the compared
run had 144 MiB more on rank 0. **+0.07% is inside the contamination band and its sign
carries no information.**

### The gate that actually matters

> **The gate is not the byte error, it is whether the top-1 configuration choice survives.
> A byte error of a few percent that never changes which configuration wins is a better
> outcome than a tighter one that does.**

---

## D49. Measurement hygiene, as refusals

Each of these voided real percentages before.

1. **Never mix profiled and unprofiled numbers.** Being profiled costs ~**0.7–1.05 µs per
   kernel** — **8.1% of a 27B decode step** (951 kernels, +1.002 ms) and 8.5% on the 0.6B;
   a profiled run is ~16% slower end to end (TPOT 3.619 vs 3.115 ms). *The cost model is
   validated against the measure table (CUDA events, no profiler). A profile is for
   attribution within a step and never for an absolute.*
2. **Never mix machines.** The same unprofiled 27B TP=4 decode step is **9.774 ms on one
   box and 12.365 ms on another — 26%**, larger than the difference between TP=4 and TP=8
   on one box.
3. **Read the spread before the mean**, and report an interval or report nothing. A
   difference smaller than ~5% needs repeats before it means anything.
4. **Any residual below about 2% is quoting the instrument.** Pricing repeatability across
   two back-to-back runs of one graph: summed **0.96%**, median per signature **1.18%**,
   **p90 32%**.
5. **Check the machine before *and* after, not only before.** Three of the last five prior
   pilot attempts were lost or degraded by other tenants — two refused to start with a
   negative KV budget at 141 GB of neighbour, one ran **64% slow**. A check that only runs
   first cannot see a tenant that arrived mid-run.
6. **Coverage is a hull, not a bounding box.** Rung 16 was covered on both axes
   *separately* and still came out **22.6% low**, because it held 64 samples at raggedness
   exactly 1.00 against a run at 1.18–1.32. And the axes are workload-dependent: 0.6B
   batches ran at raggedness 2.82–3.85, 27B at 1.11–1.42.
7. **Leave-one-out is not cross-validation of a *family*.** An attention power law scored
   LOO median **0.75%** and then missed two independently traced graphs by **−6.2% and
   −6.7%, in the same direction**. *"Where a sample is small and geometrically spaced, LOO
   reports how stable the fit is, not whether the family is right."*

---

## D50. Registration and the cell matrix

### Registration

Everything is frozen **before** evaluation and hashed into the artifact: dataset version,
scope, request identities and order, token lengths, arrival pacing, the session split, the
machine spec, every artifact fingerprint, and the gate states of D43.

**The case set may never shrink after errors are seen.** And a development workload may
never be called held-out: the prior pilot corpus was iterated against during development,
so it could not be — *"the held-out axis the user actually wants is configuration."*

### The cell matrix

Given doc 07 D38's calibration recipe — full-engine calibration at TP1 only, plus one TP2
transfer test — the acceptance cells span what calibration did **not**:

| Axis | Cells |
|---|---|
| parallel width | TP1, TP2, TP4 (and TP8 for Kimi-K3) |
| client count | 1, 4, 16, 64, 256 |
| workload class | short, long |
| PD topology | aggregated; disaggregated from M4 |

Client count means **agent session trees**, not requests; in-flight exceeds it during
fan-out. Each cell is one `aiperf profile` invocation — the scenario rejects
comma-separated sweeps.

**One corpus limit to design around:** only **175 of 393** sessions contain any subagent
and only **144** offer a multi-request episode containing a descendant. A 256-client cell
requiring genuine fan-out in every root is not constructible without reusing sessions.
Whatever is done there must be declared, not discovered.

### Repeats

The simulator is **deterministic by construction** under doc 01 D3 (Clock Authority, ties
broken by LP id) — so: **one simulated run plus an asserted reproducibility test**, and
N≥3 **spaced** real runs per cell to establish the noise floor.

---

## D51. Simulation speed

### What was measured before

| workload | real | simulated | ratio |
|---|---|---|---|
| 64 requests, Poisson 8/s, 0.6B TP1 | 9,426 ms | 639 ms | **14.8×** |
| cc-traces, 20 requests, 27B TP4 | 309 s | 3 s | **~103×** |
| **saturated, 300 requests, 0.6B** | **36 s** | **122 s** | **0.30× — slower** |

The discrete-event jump only pays when there is idle to skip. Partly explained by the
arrival barrier spinning **89,336 scheduling ticks** before the first step — machinery doc
01 D8 removes.

### Rules

- **Measure a saturated cell early**, before the architecture is load-bearing. The
  ≥5× target is negotiable; *faster than real* is not.
- **Account cold costs once.** Factory build is 9.5–11 s; first `graph_for` on a new shape
  is 0.27 s. Journal them rather than folding them into a per-step average.
- **Beware the cold-loop artefact** — an average over a loop whose first iteration is a
  cache miss reads **3–7× high**. Measure a second, warm loop.
- Report the simulator's own CPU cost per step alongside the ratio. Previously **4.3 ms**
  per step with the allocation carried in the cache key, against a 32.7 ms modelled step.

---

## D52. What invalidates a run

Fail closed. Each of these has a prior incident behind it.

| Condition | Why |
|---|---|
| a **coverage hull violation** on any acceptance step | the step was priced outside what was measured |
| **machine contamination** before or after | a tenant compute-bound in 0.7 GB was caught by a health check and would have passed any free-memory test |
| a **`--measure` flag** on an acceptance run | the instrument changes what it measures — ~11 ms of TTFT on the 27B |
| **analytic fraction** above the declared threshold | doc 07 D36's resolver ladder rung 4 is a permission, not a fallback |
| an **artifact fingerprint mismatch** | doc 07 D43 |
| a **gate state** in the artifact disagreeing with the declared configuration | *"a dead gate is worse than no gate"* — a `PRICE_KERNELS` gate read `WORLD_SIZE`, which the engine never sets, so it was false in every worker and the proof was in the artifact: breakdowns present for 164 of 237 entries |
| **saturation** where scheduling fidelity is being claimed | see below |
| any **refusal** raised by the cost model | it refused for a reason; name it |

### Saturation deserves its own note

**Scheduling fidelity is unobservable on a saturated workload.** Every loaded prior
experiment ran at essentially 100% utilisation, *"where the queue term swamps everything
and no scheduling property is observable."* Concretely: both runs executed the same **1,028
steps**, the simulated ones cost **3.8% more**, the engine was **99.6% utilised real and
100.0% simulated** — and *"adding 3.8% to every step is enough to take 99.6% to 100%, and
the wait follows."*

A previously-reported *"constant quarter-second admission delay"* was **withdrawn** on
exactly this basis: per quartile the simulator admits the **first** requests *faster*
(0.000 s vs 0.046 s) and then falls progressively behind. The question as posed —
*"find why the simulator admits requests late"* — had no answer, because it does not.

So a scheduling claim needs a workload **with slack**, or controlled step durations that
remove cost error by construction. Throughput claims at saturation are fine; scheduling
claims there are not.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D44 | Three separable results — cost accuracy, schedule agreement, end-to-end — reported side by side. Aggregate latency is never the judgement. | 2026-09-18 |
| D45 | Noise-floor first: the real-vs-real spread is the tolerance. A metric is admissible only if stable **and** sensitive, the latter tested against the ×0.97 counterfactual. | 2026-09-18 |
| D46 | Three metric families ordered by robustness, as an extensible registry: counting invariants (exact), distributional (weighted most), event timing (most diagnostic). Policy agreement is a regression guard, not a discriminator. | 2026-09-18 |
| D47 | Cost accuracy on the real run's own step sequence; report totals, held-out-worst, and median per step together. | 2026-09-18 |
| D48 | Memory per term, never as a sum. The gate is whether the top-1 configuration choice survives, not the byte error. | 2026-09-18 |
| D49 | Seven hygiene refusals, each with a prior incident behind it. | 2026-09-18 |
| D50 | Everything registered and hashed before evaluation; the case set never shrinks. One simulated run plus a reproducibility assertion; N≥3 spaced real runs. | 2026-09-18 |
| D51 | Measure a saturated cell early. Account cold costs once; report the simulator's own per-step CPU cost alongside the ratio. | 2026-09-18 |
| D52 | Eight fail-closed invalidation conditions. Scheduling claims require slack; throughput claims at saturation are fine. | 2026-09-18 |

---

## TODO register

| # | Item | Why deferred |
|---|---|---|
| T23 | Choose the family-2 distance function | needs one cell's real-vs-real data to compare candidates |
| T24 | Define "structural event" for family 3 beyond prefill streaks | streaks may be an artefact of prefill-first scheduling |
| T25 | Measure the real-vs-real noise floor under **closed-loop** replay at high client count | the only prior data is 20 requests, one session, declared arrivals |
| T26 | Assert simulator bit-reproducibility as a test | depends on the Clock Authority existing |
| T27 | Decide the 256-client cell's construction given only 144 fan-out-capable sessions | doc 06 T14, surfaces again here |
| T28 | Establish whether ranking/regret becomes an explicit acceptance gate | the prior effort called it the gate that matters; the current acceptance table does not list it |
