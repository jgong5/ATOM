# ATOM Compass — Design Point 11: Engine Metrics under Virtual Time

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Depends on:** `01_execution_and_time_model.md` (what a clock read means),
`08_validation_protocol.md` (what the numbers are for).

**Scope — deliberately narrow.** ATOM's existing **Prometheus-based engine metrics**, and
what it takes to make them correct under a virtual clock. The goal is to reuse that
infrastructure and add nothing to it.

**Explicitly out of scope, with pointers:**

| | Where it lives |
|---|---|
| harness-side latency metrics | `06` D34 — aiperf stamps them in its own transport, so our transport controls them |
| per-request `sim_*` response fields | `06` D28 |
| simulator observability (speed ratio, provenance mix, refusal counts, coverage distance, Clock Authority stats) | **fields in the run artifact**, written once at the end. Not a metrics subsystem, not an exporter, no cadence. |
| the graded product outputs | `08` D78 |

---

## D71. Current state, and what it means

Verified on `feature/atomcompass_new`:

| | |
|---|---|
| metric families | **10 `GaugeMetricFamily`, 10 `CounterMetricFamily`** |
| histograms / summaries | **none today** |
| `.observe()` call sites | **none** (the `buckets=` hits in the tree are CUDA-graph query buckets, unrelated) |
| API style | collector-style `prometheus_client.core.*MetricFamily` (`metrics.py:11-13`) |
| rate metrics (anything ÷ time) | **none.** The three that look like rates are not: `queued_prefill_tokens_per_rank` is per *rank*, `mtp_acceptance_rate` is a ratio of counts, `mtp_average_tokens_per_forward` is per *forward* |

The pipeline: `EngineUtilityHandler.collect_metrics()` (`engine_utility.py:336-392`) →
`push_metrics()` (`:322`) on the EngineCore busy loop → DP aggregation in
`LLMEngine.get_metrics_statistics()` (`llm_engine.py:438-562`) → `_metrics_refresh_loop`
(`api_server.py:1504-1526`) → `_AtomMetricsCollector` / `AtomMetricsExporter`
(`atom/entrypoints/openai/metrics.py`) → `/metrics` (`api_server.py:2336-2344`).

**Histograms are being added.** They are not present yet, which means this is the moment to
influence how they land rather than retrofit them. See D73.

---

## D72. Counters and gauges are valid by construction; their clocks stay real

### Why they are valid

`collect_metrics()` reads `scheduler.get_request_counts()`, `scheduler.block_manager.kv`,
`spec_stats` and `cache_stats` — **all state of the real scheduler and block manager, which
run unmodified under simulation** (`03` D13). Nothing there is a timing read, and there are
no rate denominators to correct.

### The two clock reads, and why both stay on the real clock

```python
# engine_core.py:317-320   <- inside EngineCore.busy_loop, the process that OWNS the clock
now = time.monotonic()
if now >= next_metrics_push:
    next_metrics_push = now + METRICS_PUSH_INTERVAL_S      # 5.0, engine_core.py:50
    self.utility_handler.push_metrics()

# metrics.py:408           <- in the API-server process
self._last_refresh = time.time()    # "when was this snapshot taken", returned by read()
```

**Decision: both stay real, and both go on the allowlist of `01` D9's AST test.**

*"Leave them alone"* is not sufficient, because both sit inside processes whose clocks are
virtualized:

- `engine_core.py:317` is in the **clock owner's hot loop**. Redirect `time.monotonic()`
  wholesale there and the metrics gate goes virtual whether or not that was intended.
- `metrics.py:408` is in the API-server process, which installs a virtual clock for arrival
  stamping. Redirect `time.time()` broadly and `_last_refresh` silently reports a snapshot
  *"taken"* at virtual t=180 s while wall time is t=4 s — a scraper's staleness check then
  reads nonsense.

The allowlist is the mechanism; these are two of its entries; the AST test is what stops a
later refactor from quietly virtualizing them.

### This corrects `01` D5

That document lists `METRICS_PUSH_INTERVAL_S` among the things that *"become a virtual
timer, not disabled — metrics should be on the virtual timeline"*. **That is wrong.** A
virtual cadence is harmful in both directions:

| virtual clock | 5 virtual s becomes | consequence |
|---|---|---|
| 100× fast (idle-skipping) | ~50 ms wall | **100× more pushes** over the ZMQ output socket and through `output_queue.get()` — real load, perturbing the measurement (D77 of the previous draft, now D75) |
| 0.3× (saturated) | ~16.7 s wall | metrics stale past the API server's own 5 s refresh, which re-reads the same values |

A wall cadence avoids both. The simulated *timeline* comes from D74's sampling, not from the
push cadence.

---

## D73. Histograms observe durations — the problem counters do not have

### The qualitative difference

> **Counters and gauges read *state*. Histograms observe *values* — and for latency
> histograms those values are *durations computed from clock deltas*. A duration is exactly
> what virtual time changes.**

So D72's "valid by construction" argument does **not** extend to histograms.
`histogram.observe(t1 − t0)` is correct only if `t1` and `t0` are simulated.

### Decision: the clock audit extends from reads to observation arguments

`01` D9's AST test currently fails on any un-allowlisted `time.*` **read** in the serving
path. It must also cover **the arguments to histogram observations**: any `observe()` whose
value derives from a clock delta is a site that must be on the virtual clock.

### The placement hazard

The obvious place to feed a TTFT histogram is `api_server.py:936-943`, where ttft and tpot
are already computed — from `time.time()` deltas in the API-server process. Under
simulation those stamp **wall** time for an event that happened at a **simulated** instant.

A histogram fed from there would be confidently, precisely wrong, and would look
authoritative **because it came out of Prometheus**. It must instead be fed from the
engine's own readings (the `sim_*` values of `06` D28) or from an explicitly clock-correct
source.

### Classic versus native, and why it decides D74

| | Representation | Backfillable? |
|---|---|---|
| **classic** histogram | `_bucket{le="…"}` + `_sum` + `_count` — counters with labels | **yes**, ordinary OpenMetrics text |
| **native** histogram (Prometheus 2.40+) | one sample carrying a sparse exponential encoding | **no** — cannot be expressed in OpenMetrics text at all |

`prometheus_client`'s `HistogramMetricFamily` emits **classic**, so if the new histograms go
through the same collector-style API as the existing twenty, D74's backfill route survives
untouched. If ATOM ever adopts **native** histograms, that route stops working and the only
option is remote-write into a looser TSDB (VictoriaMetrics, Mimir).

### Two requests to make while the code is being written

Both are cheap now and expensive later:

1. **Classic histograms, via `HistogramMetricFamily`** — the same collector-style API as
   everything else in the exporter.
2. **The instrumentation site observes a duration it is *handed*, not one it computes from
   an inline `time.*` delta.** `record_ttft(seconds)` rather than `record_ttft(t_end −
   t_start)` inside the metric code. That single shape choice is what lets the caller supply
   either a wall or a simulated duration, and it is what keeps the site auditable.

### The gap this closes

ATOM currently exports **no latency histograms at all** — TTFT and TPOT exist only as
per-request values and never reach Prometheus, so a Prometheus-only observer cannot grade
the acceptance metrics on **either** side.

If the new histograms observe simulated durations, **the acceptance metrics come out of
ATOM's own exporter**, and the `sim_*` response fields become a cross-check rather than the
sole source. That is more faithful to "reuse ATOM's infrastructure and nothing more", not
less.

---

## D74. A simulated-time series, without fighting the pull model

### Why the naive path is blocked

Prometheus is **pull-based and stamps at ingestion** — a scraping server labels each sample
with *its own* wall clock, so `/metrics` alone can never carry a simulated timeline. The
escape is explicit timestamps in the exposition format, but a **live** server rejects
samples far in the future or out of order, and virtual time is 100x ahead or 0.3x behind by
construction.

### The pairing that must be avoided

**Do not sample on the wall clock and label with virtual time.** At a 100x speed ratio, two
samples 5 real seconds apart are **500 virtual seconds** apart, and every transition between
them is lost:

```
  wall-timer sampling + virtual timestamps
  ----------------------------------------
  virtual time >  0----------------500s----------------1000s
  true gauge      ~~~\__/~~\____/~~~~\__/~\___/~~~\__/~~
  samples         *                    *                    *
                  +--- 5 real seconds -+
                      = 500 virtual seconds at 100x

  -> three points describing a signal with hundreds of transitions
```

### The decoupling: sampling is cheap, transport is expensive

`collect_metrics()` is a pure read returning a dict of ints. The ZMQ push is what costs. So
the two are driven separately — and the sampling trigger is **the engine step**, not a
timer.

### Sample per step, because virtual time is discrete-event

The engine's clock does not flow. It **advances at step boundaries**, by the predicted
duration of the step just taken, or it jumps to the next arrival when idle. Every gauge in
`collect_metrics()` changes at exactly those moments:

| Gauge | Changes in |
|---|---|
| `kv_blocks_used` / `free` / `indexed` | `Scheduler.schedule()` (allocate) and `postprocess()` (deallocate) |
| running and waiting counts | admission and completion, both per step |
| preemption count | `_preempt_one_running`, per step |
| prompt and generation token totals | per step |

So the true series is a **step function whose breakpoints are engine steps**. A timer at N
virtual seconds either oversamples (firing repeatedly on identical state) or undersamples
(missing breakpoints). Sampling once per step is neither — **the series is the ground truth,
not an approximation of it**.

```
  per-step sampling
  -----------------
  virtual time >  0----------------500s----------------1000s
  true gauge      ~~~\__/~~\____/~~~~\__/~\___/~~~\__/~~
  samples         **************************************
                  one per step = every breakpoint
```

### And it is cheap

The prior 27B cc-traces run was **106 prefill + 4,346 decode = ~4,450 steps over 267 s of
virtual time**. At ~40 series that is ~178k data points for a whole run — trivial to buffer
and to write. The transport still drains on the wall-clock cadence of D72, so the ZMQ
message **count** is unchanged; only the payload gets wider.

```
  EngineCore (owns the virtual clock)
  +----------------------------------------------+
  |  per ENGINE STEP                              |
  |     snapshot = collect_metrics()    <- cheap   |
  |     buffer.append((virtual_ts, snapshot))      |
  |                                                |
  |  wall timer, every 5 REAL seconds  (D72)       |
  |     push_metrics(buffer.drain())    <- costly  |
  +----------------------+-------------------------+
                         | same ZMQ, same message COUNT, bigger payload
                         v
   +---------------------------------------------+
   | API server                                  |
   |   /metrics -> current values, wall scrape   |  <- liveness, unchanged
   |   run end  -> OpenMetrics text w/ virtual ts|  <- the simulated timeline
   +---------------------+-----------------------+
                         v
        promtool tsdb create-blocks-from openmetrics
                         v
                   TSDB blocks -> Grafana
```

### Two things to get right

**The per-step cost on the REAL side must be measured, not assumed.**
`collect_metrics()` calls `get_request_counts()`, reads `block_manager.kv`, and calls
`spec_stats.get_statistics()` and `cache_stats` — none of which has been verified O(1). At
~33 decode steps per second on a real run it is almost certainly free, but if it is not, the
answer is **decimation by a declared K** (every K-th step). Decimation loses information in
a *known* way; timer aliasing does not.

**Arrivals between steps are the one genuine gap.** A request arriving changes the
waiting-queue depth before the next step samples it. But nothing *consumes* that change
until the next step — the scheduler only acts at steps — so the step-resolution series is
the decision-relevant one. Stated here rather than left implicit.

### A subtlety this dissolves

An earlier draft of this design used a virtual timer and needed a rule: the timer must be
**re-armed, not back-filled**, across an idle jump, or a 100 ms timer would owe 550 firings
of identical state when the clock jumps from t=5 s to t=60 s. **With per-step sampling that
case does not arise** — the jump is one event, it produces one sample, and Prometheus's own
staleness handling draws the gap. The rule is unnecessary rather than merely satisfied.

### Backfill constraints, and how ATOM lands against them

| Constraint | ATOM |
|---|---|
| native histograms and staleness markers unsupported | fine **if** D73's request is honoured — classic only |
| do not backfill the last 3 hours (overlaps the mutable head block) | **virtual time's origin is ours to choose.** `CompassConfig.epoch` is pinned once and carried to every process — place the run safely in the past |
| timestamps are the third field, in **seconds** (floats allowed); promtool multiplies by 1000 | a writer detail, and a silent one: a malformed timestamp lands the whole block at **1970-01-01** |
| `# EOF` terminator mandatory | writer detail |
| backfilled data is subject to retention | a frequent cause of *"the metric appears but has no data points"* |
| 2 h blocks by default; `--max-block-duration` for longer | a simulated run is minutes of virtual time — one block |
| requires promtool v2.24.0+ | check the deployment |

### The overlay, which is the reason to do this at all

**Treat the real run the same way.** Its wall timestamps *are* its timeline, and its state
changes at its own step boundaries, so the identical per-step sampler produces a comparable
block. Real and simulated series then overlay in one Grafana view — KV pool occupancy,
waiting-queue depth, preemptions, running count, side by side over the run.

That is a much stronger validation view than terminal counters, and it is most of `08`
D46's family-2 intuition made visible.

### What stays untouched

`collect_metrics()`, every metric definition, `_AtomMetricsCollector`,
`AtomMetricsExporter`, the `/metrics` endpoint, the registry. The additions are a per-step
sampling hook, a bounded buffer, a wider push payload, and a writer — **none of it in the
exporter**.

## D75. What is invalid, and is refused rather than reported

| Invalid | Why |
|---|---|
| any metric with a **wall-clock denominator** presented as a simulated rate | there are none today; this is a rule about future additions |
| any **histogram observation of an un-virtualized duration** | D73 — the failure looks authoritative because it came from Prometheus |
| GPU utilisation, power, temperature, and the `server_metrics/` Prometheus scraper | there is no GPU |
| metric emission that **advances the virtual clock** | emission is simulator overhead, not modelled work. Under `01` D4 virtual time advances only for durations the cost model produced, so this is satisfied by construction — but it must be **asserted**, because a hook that accidentally sat inside a Category-A path would be invisible |

The standing reminder behind the last row: the instrument changes what it measures. Measure
mode cost ~**11 ms of TTFT on the 27B (4%)**, and an earlier version that synchronised
around each forward made the run **33% slower** (TPOT 3.26 → 4.33 ms).

---

## D76. Three validation results this yields for free

1. **The gauges *are* `08` D46's family 1.** KV blocks used/free/indexed, preemption count,
   running and waiting counts, prompt and generation token totals — already exported on both
   the real and simulated sides, in the same format, through the same code. **The
   counting-invariant comparison needs no new instrumentation** — a scrape on each side and
   a diff.
2. **A histogram's `_count` is itself a counting invariant.** Same workload, same requests,
   so real and simulated must produce the same number of observations, exactly.
3. **`_sum / _count` is a mean** comparable against the real-vs-real noise floor of `08`
   D45, with no new instrumentation.

And classic histograms are cumulative counters, so `rate()` and `histogram_quantile()`
behave normally over backfilled blocks provided timestamps are monotone in virtual time —
which the Clock Authority guarantees.

---

## D77. A taxonomy that covers every metric type, present and future

### The classification principle

> **A Prometheus *type* does not determine the treatment. The *provenance of the value*
> does.**

A `Counter` may be an event tally (safe by construction) or the `_sum` of a latency
histogram (a duration, and therefore a clock read). A `Gauge` may be a queue depth (safe) or
a last-observed latency (a duration) or a process start time (a timestamp). So the rule must
attach to where the number came from, not to how it is exposed.

### Six classes

| Class | Value derived from | Treatment under a virtual clock |
|---|---|---|
| **S — state** | engine state read at sample time: queue depths, pool occupancy, indexed blocks | **valid by construction.** Sample per step (D74). |
| **E — event tally** | a count incremented on an occurrence: requests finished, preemptions, tokens, histogram `_bucket` and `_count` | **valid by construction.** Monotone; sample per step. |
| **D — duration** | a clock delta: TTFT, TPOT, step time, queue wait, histogram `_sum` over durations | **must be a simulated duration.** Audit the observation argument (D73). |
| **R — rate** | a count divided by elapsed time | **do not export.** Export the underlying counter and let PromQL `rate()` compute it over virtual timestamps. If a rate must be exported, its denominator is virtual elapsed. |
| **T — timestamp** | a wall-clock instant exported as a value: `process_start_time_seconds`, `_created`, `_last_refresh`, exemplar timestamps | **must declare its clock.** Usually real, because it describes the *process*; virtual if it describes the *run*. Never left implicit. |
| **X — external** | measured outside the engine: GPU telemetry, host stats, the `server_metrics/` scraper | **invalid under simulation.** Refuse. |

ATOM's current twenty metrics are all **S** or **E**, which is why D72's "valid by
construction" holds today. The classes exist so that the next metric is classified rather
than assumed.

### How the Prometheus types map — note that one type spans several classes

| Type | Can be | Notes |
|---|---|---|
| Counter | **E**, or **D** if it is a `_sum` of durations | the ambiguity is the whole reason for this taxonomy |
| Gauge | **S**, **D**, or **T** | |
| Histogram (classic) | `_bucket` and `_count` are **E**; `_sum` is **D** when the observed value is a duration | backfill-compatible |
| Summary | same split, **plus client-computed quantiles** | see the hazards below |
| Native histogram | as classic, but **kills the backfill route** (D73) | |
| Info / StateSet | constant, no clock | always safe |
| Exemplar | attached to a sample, **carries its own timestamp** → **T** | |
| Untyped | unclassifiable by definition | must be declared explicitly |

### Enforcement: declare the class, or refuse

Convention and code review will not hold across future additions. The mechanism:

1. **Every metric family declares its class where it is constructed** — ~20 tags today in
   `metrics.py`, one per metric thereafter.
2. **An unclassified metric refuses to export under simulation.** On a real run it is
   unaffected, so the cost of the rule falls only where correctness depends on it.
3. `01` D9's AST test covers the other half — any `observe()` whose argument derives from a
   clock delta, and any un-allowlisted `time.*` read.

The failure mode is the right one: a new metric added without a tag **refuses**, rather than
silently emitting a wall-clock-derived number that looks authoritative because it came from
Prometheus.

### Type-specific hazards

| Hazard | Detail |
|---|---|
| **Summaries cannot be aggregated across DP ranks** | client-computed quantiles do not average. `LLMEngine.get_metrics_statistics()` (`llm_engine.py:438-562`) merges per-rank snapshots; summing histogram `_bucket` is valid, averaging summary quantiles is not. This is a general Prometheus truth that would bite ATOM whether or not Compass existed. |
| **`_created` series** | opt-in — `CounterMetricFamily.add_metric(..., created=...)`. ATOM does not pass it today. If it starts, `created` is class **T** and must not be a wall-clock value inside a virtual-timeline block. |
| **Exemplars** | `add_metric(..., exemplar=...)` exists and an `Exemplar` carries a timestamp. If trace linking is ever added, that timestamp is class **T**. |
| **Counter resets** | PromQL `rate()`/`increase()` detect decreases as resets. Normal handling; no special treatment needed under backfill, provided timestamps are monotone in virtual time. |
| **`le` / `quantile` labels** | bucket boundaries must be identical across real and simulated runs, or the series are not comparable. Same metric definitions on both sides guarantees it. |

### The exposition format, which is not the one ATOM uses

Verified in the container:

```
CONTENT_TYPE_LATEST      : text/plain; version=1.0.0; charset=utf-8     <- what ATOM emits
openmetrics CONTENT_TYPE : application/openmetrics-text; version=1.0.0  <- what promtool wants
```

`prometheus_client.openmetrics.exposition.generate_latest` exists. So D74's run-end writer
uses **that**, not the `generate_latest` ATOM imports at `metrics.py:11`. The `/metrics`
endpoint keeps the text format it has.

### And the timestamp injection point already exists

Every family's `add_metric` takes a `timestamp=` argument:

```python
CounterMetricFamily.add_metric(labels, value, created=None, timestamp=None, exemplar=None)
GaugeMetricFamily.add_metric(labels, value, timestamp=None)
HistogramMetricFamily.add_metric(labels, buckets, sum_value, timestamp=None)
SummaryMetricFamily.add_metric(labels, count_value, sum_value, timestamp=None)
```

So the backfill export **reuses ATOM's exact collector code** and passes the virtual
timestamp; the live `/metrics` path passes nothing and lets the scraper stamp. One
collector, two serializations, no custom format code — which is what "reuse the
infrastructure and add nothing" should mean in practice.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D71 | Scope is ATOM's Prometheus engine metrics only. Harness metrics belong to `06`; simulator observability is run-artifact fields, not a metrics subsystem. | 2026-09-19 |
| D72 | Counters and gauges are valid by construction. **Both metrics clock reads stay on the real clock** and go on `01` D9's allowlist. This corrects `01` D5, which made the push cadence virtual. | 2026-09-19 |
| D73 | Histograms observe durations, so the clock audit extends from clock *reads* to observation *arguments*. Ask for classic histograms and for instrumentation that is handed a duration rather than computing one inline. | 2026-09-19 |
| D74 | **Sample once per engine step** — virtual time is discrete-event, so state changes only at step boundaries and the per-step series is the ground truth. Transport on the wall clock; write OpenMetrics with virtual timestamps; backfill into TSDB. Treat the real run identically, for overlay. | 2026-09-19 |
| D75 | Four classes of invalid metric, refused rather than reported. Emission never advances the virtual clock, and that is asserted. | 2026-09-19 |
| D76 | The gauges supply `08`'s counting invariants for free; a histogram's `_count` is one too, and `_sum/_count` is a mean comparable at the noise floor. | 2026-09-19 |
| D77 | Classify every metric by the **provenance of its value** (state, event tally, duration, rate, timestamp, external), not by its Prometheus type. Declare the class at construction; an unclassified metric refuses to export under simulation. The backfill writer uses the OpenMetrics serializer and the `timestamp=` argument that `add_metric` already provides. | 2026-09-19 |

---

## TODO register

| # | Item | Why deferred |
|---|---|---|
| T40 | Confirm which histograms are landing, and that they are **classic** rather than native | decides whether D74's backfill route survives |
| T41 | Audit every `observe()` site for the provenance of its value; extend the AST test to cover observation arguments | the sites do not exist yet |
| T42 | Measure `collect_metrics()` cost per step on the **real** side; decide whether decimation by a declared K is needed | `get_statistics()` and the cache/pool reads are not verified O(1) |
| T43 | Verify the backfill end to end — produce one block, load it, see the series in Grafana | the documented gotchas are silent ones (epoch-dated blocks, retention) |
| T44 | Sanity-check histogram bucket ranges against simulated latencies | a quantile pinned to `+Inf` is worth catching once rather than discovering |
| T45 | Tag ATOM's existing twenty metrics with their D77 class | ~20 tags; mechanical, but it is the gate for everything after |
| T46 | Decide the DP-aggregation rule per class, and refuse summaries there | `llm_engine.py:438-562` merges per-rank snapshots; quantiles do not average |
