# ATOM Compass — Design Topic 11: Engine Metrics under Virtual Time

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.
**Amended 2026-11-16** by [#3584](https://github.com/jgong5/ATOM/issues/3584): D71 and D73
state the metrics ATOM has now, D74 is revised for its first build, the series a
simulated run writes for CI's Prometheus report (`19` D115), D72 and D75 say what that
leaves of the scrape and of the GPU panels, D76 what it leaves of the GPU counts, and D77
where the first build declares each family's class. *The first build*, before the decision
log, walks it. **Amended 2026-11-17** by [#3611](https://github.com/jgong5/ATOM/issues/3611):
D74's samplers sample once more as they hand in and the report's query reaches past the last
instant (*The final sample*), the first version's run origin is named, and (b)'s cost is the
one measured.

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
| simulator observability (speed ratio, provenance mix, refusal counts, coverage distance, Clock Authority stats) | **fields in the run artifact**, written once at the end, always, an early end included (`01` D3.5). Not a metrics subsystem, not an exporter, no cadence. |
| the graded product outputs | `08` D44 |

---

## D71. Current state, and what it means

Verified on `feature/atomcompass_new`:

| | |
|---|---|
| snapshot families | **`GaugeMetricFamily` and `CounterMetricFamily`**, most built in loops, from the engine snapshot; the inventory is `atom/metrics/exporter.py::_AtomMetricsCollector.collect` |
| instruments | prometheus_client `Histogram` and `Gauge`, each owned by the component that observes it: the API server's TTFT and inter-token latency (`atom/metrics/request.py`), each scheduler's queue-time, batch, token, context and PD-transfer histograms and its request-context gauges (`atom/metrics/scheduler.py::SchedulerMetrics`), each worker's forward device time (`atom/metrics/gpu.py::GPUForwardMetrics`) |
| histograms / summaries | **classic** histograms (`Histogram`, and `atom/metrics/histogram.py::WeightedHistogram`); no summaries |
| where instruments live | prometheus_client's multiprocess files, in the `PROMETHEUS_MULTIPROC_DIR` each API server creates (`atom/metrics/prometheus.py::initialize_metrics`) and every engine and worker it spawns inherits; `/metrics` renders every process's file through `MultiProcessCollector` (`atom/entrypoints/openai/metrics_setup.py::create_metrics_exporter`) |
| rate metrics (anything ÷ time) | **none.** The three that look like rates are not: `queued_prefill_tokens_per_rank` is per *rank*, `mtp_acceptance_rate` is a ratio of counts, `mtp_average_tokens_per_forward` is per *forward* |

The snapshot pipeline: `EngineUtilityHandler.collect_metrics()` (`engine_utility.py`) →
`push_metrics()` on the EngineCore busy loop → DP aggregation in
`LLMEngine.get_metrics_statistics()` (`llm_engine.py`) → `_periodic(metrics_interval, _refresh_metrics_once)`
(`api_server.py`) → `_AtomMetricsCollector` / `AtomMetricsExporter`
(`atom/metrics/exporter.py`) → `/metrics` (`api_server.py::metrics`). An instrument's
value reaches `/metrics` through no pipeline: the render reads its file.

The histograms landed after this topic was first written; D73 records how.

---

## D72. Counters and gauges are valid by construction; their cadence is simulated time

### Why they are valid

`collect_metrics()` reads `scheduler.get_request_counts()`, `scheduler.block_manager.kv`,
`spec_stats` and `cache_stats` — **all state of the real scheduler and block manager, which
run unmodified under simulation** (`03` D13). Nothing there is a timing read, and there are
no rate denominators to correct.

### The two timers and the stamp, all on the LP clock

```python
# EngineCore.busy_loop and DPEngineCoreProc.busy_loop (atom/model_engine/engine_core.py)
now = clock.now(time.monotonic)
if now >= next_metrics_push:
    next_metrics_push = now + metrics_interval    # ATOM_METRICS_UPDATE_INTERVAL_S
    self.utility_handler.push_metrics()

# _periodic(metrics_interval, _refresh_metrics_once) (atom/entrypoints/openai/api_server.py), in the API-server process
await asyncio.sleep(interval)     # ATOM_METRICS_UPDATE_INTERVAL_S
await step()                      # _refresh_metrics_once

# AtomMetricsExporter.update (atom/metrics/exporter.py)
self._last_refresh = time.time()   # through atom.utils.time_shim; "when was this snapshot taken", returned by read()
```

**Decision (owner ruling, 2026-10-02): the push and the refresh are timers on simulated
time, marked daemon.** They follow `01` D5's rule like every other timer the CA can
reach. The push reads the engine LP's clock through `atom.utils.clock.now`. The refresh
sleeps on the frontend's event loop, whose `time()` is the frontend LP's clock (`01` D5.1),
and `_last_refresh` stamps the simulated instant of the snapshot. As daemon deadlines they
fire as usual but never keep a run alive: the run finishes when no essential work is left
(`01` D3, #533). The frontend names its daemon timers in `DAEMON_TIMERS`
(`atom/utils/compass_loop.py`).

### The observer: Prometheus's scrape, on simulated time

A real deployment's Prometheus `GET`s `/metrics` every `scrape_interval`, and the
simulation does the same: the traffic LP scrapes every `scrape_interval` of simulated time
(`01` D9, item 13), with `scrape_interval` taken from the real deployment's configuration,
so the scrape, the refresh, the push and the stamp share one clock. The scrape is Compass code in the traffic LP, whose clock owner is already
Compass's own, so it adds no LP. It adds no channel either, because `/metrics` is a real
HTTP endpoint: in a single deployment the request rides `traffic->frontend:http` and the
response `frontend->traffic:stream` (`01` D3, the channel table). The frontend serves it
at its arrival time from the exporter's current snapshot, so it sees the lag of the push
and the refresh, from the same sources as on a real system. ATOM renders it on the API
server's thread pool (`AtomMetricsExporter.render_async`); in a simulated run that render
is an uncharged job on the tokenizer station (`SERVICE_JOBS`, `atom/utils/compass_loop.py`),
so a scrape waits past its arrival only while tokenization holds every thread. Each sample carries its
simulated scrape time, and `atom:metrics_last_refresh_timestamp_seconds`
(`_AtomMetricsCollector.collect`) exports `_last_refresh`, so a staleness check compares
two simulated instants. The scrape timer is a daemon deadline, granted only once it is at
most the essential horizon `H` (`01` D3), but its `GET` and the response are registered
arrivals, which raise `H` (`atom/compass/clock/authority.py`): a scrape at `H` extends it
by one round trip. While `scrape_interval` exceeds that round trip, the next scrape's
deadline lies beyond the raised `H`, so no scrape chains into the next and the scrapes
alone never keep a run alive.

**The scrape is not built (2026-11-16, #3584).** The paragraph above predates ATOM's
histograms. Since they live in multiprocess files (D71), a render of `/metrics` reads every
engine's and worker's file as well as the snapshot, so under simulation it reads other
LPs' state off the channels, which `01` D3 rules out (LPs interact only through
timestamped channels); the harness refuses aiperf's server metrics
(`compass_harness/strategy.py::CompassAgenticReplay`), and #522 holds the 1P1D channels.
The report a simulated run writes reads D74's series instead.

### This revises the first version of this decision

The first version kept both clock reads on the real clock. Its reasons do not survive the
ruling:

| reason it gave | why it no longer holds |
|---|---|
| at 100× (idle-skipping) a 5 s virtual push is ~50 ms wall: **100× more pushes**, real load perturbing the measurement | emission never advances the virtual clock (D75), so the extra pushes cost wall time and change no simulated result |
| at 0.3× (saturated) a 5 s virtual push is ~16.7 s wall, staler than the API server's real 5 s refresh | the refresh is on simulated time as well, so push and refresh keep their ratio at every speed |
| a virtual `_last_refresh` read against wall time makes a scraper's staleness check nonsense | the scraper reads simulated time too (above) |
| a redirected clock read in the engine's hot loop virtualizes the push whether or not that was intended | it is intended; the clock-source lint (`01` D1, detector (3)) treats both reads as substituted |

The simulated *timeline* comes from D74's series, which each engine and frontend LP
samples at this push or refresh (D74, *Revised 2026-11-16*).

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

The obvious place to feed a TTFT histogram is `api_server.py::generate_async`, where ttft and tpot
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
through the same collector-style API as the existing metrics, D74's backfill route survives
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

### How they landed (recorded 2026-11-16, #3584)

Classic, through prometheus_client `Histogram` instruments in multiprocess files rather
than `HistogramMetricFamily` (D71), so the backfill route survives. Each site is handed its
value (`RequestMetrics.observe_time_to_first_token(interval, streaming)`), and every
duration the report reads is a delta of `atom.utils.time_shim` reads, which take the LP
clock in a simulated run: TTFT in `atom/entrypoints/openai/request_timing.py`, inter-token
latency in `atom/entrypoints/openai/streaming_dispatch.py`, queue time and the PD KV
transfer in `atom/metrics/scheduler.py`. The placement hazard above did not happen. The one
exception is a worker's forward device time, from device events, which a simulated run
does not create; D75 says what it observes there.

---

## D74. A simulated-time series, without fighting the pull model

The per-step sampling below is this decision's first version; its first build samples at
ATOM's metrics timer instead, as [*Revised 2026-11-16 (#3584): the first
build*](#revised-2026-11-16-3584-the-first-build), at this decision's end, states.

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

This series coexists with D72's observer scrape, and the two serve different purposes: the
per-step series is the simulated timeline's ground truth, for analysis; the scrape series is
what an operator's Prometheus would see, comparable with a real run's scrape at the same
`scrape_interval`.

### And it is cheap

The prior 27B cc-traces run was **106 prefill + 4,346 decode = ~4,450 steps over 267 s of
virtual time**. At ~40 series that is ~178k data points for a whole run — trivial to buffer
and to write. The transport still drains on D72's push cadence, so the ZMQ message
**count** is unchanged; only the payload gets wider.

```
  EngineCore (owns the virtual clock)
  +----------------------------------------------+
  |  per ENGINE STEP                              |
  |     snapshot = collect_metrics()    <- cheap   |
  |     buffer.append((virtual_ts, snapshot))      |
  |                                                |
  |  daemon timer, every 5 SIMULATED s  (D72)      |
  |     push_metrics(buffer.drain())    <- costly  |
  +----------------------+-------------------------+
                         | same ZMQ, same message COUNT, bigger payload
                         v
   +---------------------------------------------+
   | API server                                  |
   |   /metrics -> values, simulated-time scrape |  <- the traffic LP's daemon scrape (D72)
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
answer is **decimation by a stated K** (every K-th step). Decimation loses information in
a *known* way; timer aliasing does not.

**Arrivals between steps are the one genuine gap.** A request arriving changes the
waiting-queue depth before the next step samples it. But nothing *consumes* that change
until the next step — the scheduler only acts at steps — so the step-resolution series is
the decision-relevant one. Stated here rather than left implicit.

### A subtlety per-step sampling dissolves

A cadence-driven sampler has to decide what an idle jump owes it: a 100 ms timer spanning
a clock jump from t=5 s to t=60 s is 550 firings of identical state, which must be either
re-armed or back-filled, and the choice is load-bearing.

**Per-step sampling never faces the question.** The jump is one event, it produces one
sample, and Prometheus's own staleness handling draws the gap. The question is dissolved
rather than answered.

### Backfill constraints, and how ATOM lands against them

| Constraint | ATOM |
|---|---|
| native histograms and staleness markers unsupported | fine **if** D73's request is honoured — classic only |
| do not backfill the last 3 hours (overlaps the mutable head block) | moot: the report's Prometheus serves only the backfilled blocks and scrapes nothing (revised below) |
| timestamps are the third field, in **seconds** (floats allowed); promtool multiplies by 1000 | the series keeps the LP clock, seconds from the run's start, so its blocks lie in 1970 by design (revised below) |
| `# EOF` terminator mandatory | writer detail |
| backfilled data is subject to retention | time retention is measured from the newest block's end, not the wall clock (Prometheus v3.5.0, [`tsdb/db.go::BeyondTimeRetention`](https://github.com/prometheus/prometheus/blob/v3.5.0/tsdb/db.go)), so a run's 1970 blocks are kept |
| 2 h blocks by default; `--max-block-duration` for longer | a run's first 2 h of simulated time are one block, aligned from 0 |
| requires promtool v2.24.0+ | promtool 3.5.0, from the archive `collect_metrics.py::ensure_prometheus` verifies (`19` D115) |

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

### Revised 2026-11-16 (#3584): the first build

The first build feeds one view, the Prometheus report CI writes for a real run, drawn for
every simulated run beside its `report.html` (`19` D115). It replaces the sampling trigger,
the transport and the real-run overlay above, and the first version's run origin
(`CompassConfig.epoch`, Rejected below); what stays untouched stays, and the additions are a
sampler and a writer outside the exporter.

**Rule.**

- **Who samples, and when.** The samplers are the engine and frontend LPs, the LPs that
  hold ATOM's instruments; each samples the instruments its own processes hold, on its
  own LP clock, at ATOM's metrics timer (`ATOM_METRICS_UPDATE_INTERVAL_S`, 1 s by default,
  CI's scrape interval too): an engine, each DP member alone, when its loop calls
  `utility_handler.push_metrics`, just before the call, from its own process's and its
  workers' multiprocess files (`os.getpid()`, `runner_mgr.procs`); a frontend when its
  refresh calls `AtomMetricsExporter.update`, just after it, from the families
  `_AtomMetricsCollector` renders and its own process's files. Both are daemon deadlines
  (D72), so through an idle stretch they fire every interval while the essential horizon
  reaches them, as ATOM's own do. No sample reads another LP's process, and sampling makes
  no clock call. LP `lmcache`, the stand-in lmcache server of an `lmcache_mp` run
  (`atom.compass.kv.lmcache_server.LMCacheLP`), holds no ATOM instrument and samples
  nothing.
- **The final sample (#3611).** At its last hand-in, the one that writes its file (a
  finish, a stop or an engine that raised; a client run's end writes none), each sampler
  samples once more, with its timer's read, at its LP's last finite clock: its clock, or,
  once the `+inf` grant that ends a run has reached it, the clock it held before, which its
  runtime keeps (`atom/utils/clock.py::LPRuntime`). The authority never fires the daemon
  deadlines it holds at the finish (`atom/compass/clock/authority.py`), so a timer's last
  sample can fall up to an interval short of that clock, where CI scrapes once more
  (`collect_metrics.py::wait_for_final_scrape`). A final sample at the instant of the LP's
  last sample replaces it; one that raises is skipped and named among the hand-in's
  reasons, so the hand-in and the file still land.
- **The workers' GPU series.** Of an engine's workers, TP rank 0 alone observes the GPU
  forward histograms: it is the one `_group_step_seconds` prices on, every other rank
  returning 0 s, and the one whose `forward` reply the engine waits for
  (`AsyncIOProcManager.call_func`). It observes step k inside that `forward`, before it
  replies, so the observation is in its file before the engine's loop goes on, and the
  sample at a push holds exactly the steps whose `forward` returned before it. The other
  ranks' histograms keep zero observations, which add nothing to the panels' sums over
  workers (`export_report.py::query_for`). What it observes is D75's.
- **What.** The families the sampler's class table holds, and no other: the ones
  `export_report.py::panels_for` queries and the cache counters its cache queries read,
  each with its D77 class (D77, *The first build*). Each series at each instant, as a 1 s
  scrape holds it; the request-context gauges (`atom:prefill_request_context_tokens`,
  `atom:decode_request_context_tokens`) once per series, at the first instant that holds
  it, since `SchedulerMetrics._record_request_context` sets each once and the report reads
  them as one `max_over_time` over the run.
- **Labels.** On top of the instruments' own, the labels CI's scrape adds
  (`collect_metrics.py::scrape_config`): `job="atom"`, `observer="api"`, `role` (`prefill`,
  `decode`, or `standalone` for an instance) and `instance`, the side's frontend LP name
  (`frontend-P`, `frontend-D`, `frontend`) on every LP of the side.
- **The file.** `metrics-<lp>.om.gz` per sampler, named as its hand-in (an engine member
  `engine-D.dp0`, a frontend `frontend-P`), in the run's `out_dir` beside its
  `lp-<lp>.json`: OpenMetrics 1.0 from
  `prometheus_client.openmetrics.exposition.generate_latest` with `escaping=ALLOWUTF8`, each
  family's series contiguous, each series' points in time order, a histogram point's buckets
  ascending, then `_count`, then `_sum`; gzip. The LP buffers its samples and writes the
  file at its last hand-in, before its `lp-<lp>.json`: an engine on either of
  `engine_loop`'s hand-in paths, a frontend in `frontend_done`. It writes the file whole or
  not at all, under a temporary name then renamed, as `_write_whole` writes the hand-in, so
  the wait for every hand-in (`_read_hand_ins`) covers it and no reader meets it
  part-written. A run that ends early keeps every sample its samplers took; a
  sampler killed before it hands in leaves no file, and the report names it.
- **The time axis.** The LP clock, seconds from the run's start, with no epoch: the blocks
  lie in 1970, and the `started_at` labels, which the scheduler takes from the same clock,
  stay consistent with the samples.
- **Real runs.** No sampler; CI's collector serves them, and the two reports sit side by
  side in place of the overlay above (the owner at #3584's plan review; #3552 measured
  aiperf's scrape slowing a real server's long-prompt encode 10 to 25%).

**Reason.** Each sample holds its LP's state at the instant it carries, so the series keep
`01` D3's rule that LPs interact only through timestamped channels (*The move that makes
this small*, its third rule) and repeat run to run, since a daemon deadline fires under
every request order or none (`atom/compass/clock/authority.py`), and a worker's
observations reach its file before the engine reads it. The cadence is CI's, which is what
the report's 5 s steps and 60 s windows read. On #3498's run (the template's 1P1D DPA8
configuration at c=256, 2577 simulated s; #3584's named result), a side's panel families hold 1,819 series at an instant (8 engines
x 167, 8 workers x 44, the frontend 131) plus one per request-context series:

| Cadence and who samples | Samples, both sides | OpenMetrics at 166 bytes a sample |
|---|---|---|
| (a) per engine step, as this decision first had it | 312.9 M: 949,560 steps x 211, and 858,888 engine-to-frontend outputs x 131 for TTFT and ITL | 52 GB |
| (b) each engine and frontend LP at ATOM's timer, this rule | 5.68 M to 9.38 M: 23,728 pushes at steps alone, up to 41,248 with idle pushes, x 211, and 2 x 2,578 refreshes x 131; measured with no worker series, 5.61 M (#3585) | 1.56 GB at most, 41 MB in gzip; measured, 22.5 MB |
| (c) the frontend renders its whole registry each simulated second | 9.38 M: 2 x 2,578 x 1,819 | 1.56 GB |
| request-context gauges at every instant, on top of (b) or (c) | 4.84 M, against 4,020 written once | 0.8 GB |

An engine samples before `push_metrics` runs because that call sends its workers
`poll_forward_metrics` without waiting (`EngineUtilityHandler.push_metrics`,
`atom/compass/runner/overrides.py::RPC_SURFACE`), so nothing it sets off can reach the
sample; under the rule above rank 0 has nothing left for it to poll. promtool 3.5.0
backfilled a 200-instant file of one side in this layout (365,816 samples) into one 1.1 MB block from 0.5 s,
Prometheus 3.5.0 served it with no scrape configured, and `export_report.collect_report`
returned its prefill panels; promtool reads the file's raw bytes and refuses gzip, so the
report decompresses each file first. prometheus_client's default escaping writes
`atom_..._bucket` sample names under `# TYPE atom:...`, series stored under names
`panels_for` never queries, hence `ALLOWUTF8`: measured with 0.23.1, the gate image's, and
the same in 0.25.0, ATOM's pin (`pyproject.toml`, `>=0.25,<0.26`), whose
`openmetrics/exposition.py::generate_latest` keeps the `UNDERSCORES` default and the
`ALLOWUTF8` branch.

The final sample is the simulated run's counterpart of the scrape CI waits for after a
benchmark. On #3498's configuration (#3587's run, whose final clocks are #3498's), the
timers' last samples fell 0.19 s short of the decode engines' last clock, 0.24 and 0.30 s
of the frontends' and 0 s of the prefill engines', and about 24 SSE events followed
frontend-D's; a final sample adds 1,598 samples and replaces 1,336, against the run's
5,608,087. promtool 3.5.0 keeps the first of two points at one instant and drops the
second without error, hence the replacement.

**Rejected.**

- (c), the frontend rendering its whole registry each second: one sampler instead of one
  per LP, but its render reads every engine's and worker's file (D71), so its content is the
  other LPs' state at whatever instant each has reached, up to a lookahead from the sample's
  (`atom/compass/clock/authority.py`), off the channels `01` D3 requires, and it
  can differ between two runs of the same inputs; and its render costs 11 ms per instant with no request-context gauges and
  28 ms with 2,016, 28 to 72 s on one side's critical path of #3498's 761 wall seconds,
  against this rule's 7.3 s on the prefill side and 17.4 s on the decode side of the
  27.5 s it added to 780 wall seconds in #3585's timed run of #3498's configuration on
  one MI355X of a shared node (#3616's named result): an engine's sample takes 3.1 to
  3.6 ms (median), and its DP group's iteration waits for the slowest member's.
- (a), per engine step: 33 times (b)'s samples for a report that reads 5 s steps, and TTFT
  and inter-token latency, observed in the frontend, need a trigger of their own.
- D72's scrape from the traffic LP: (c)'s read, plus the harness's refusal and #522's
  missing 1P1D channels.
- A run origin at the run's wall start, as the first version's `CompassConfig.epoch`, a
  value pinned once and carried to every process to place the run in the past, never
  built: every timestamp and every `started_at` label rewritten in two places, and a run
  faster than real time (#3498: 2577 simulated s in 761 wall s) stamped in the wall
  clock's future.
- A series only where a value changed: a 60 s window with one sample gives no rate where
  CI's gives 0, and a gauge steady for 5 minutes leaves instant queries.
- A worker's pair kept until the next poll, as a real worker keeps an unfinished one: the
  push's poll is not waited for, so after an idle push rank 0's last pair lands in one
  sample or the next by wall time, and ranks whose replies nobody reads observe at any
  wall time, so the series would not repeat.
- Every TP rank observing its group's price: a broadcast of rank 0's price each step, for
  copies that change no quantile or mean the panels draw from the pooled observations. The
  one thing the copies would give, a TP group's `_count` equal to a real run's, a
  comparison gets from the real run's `tp_rank="0"` series (D76).
- The API server's `host:port` from `server.json` as `instance`: the launcher writes it
  once the server answers, so the engines cannot know it, and its port is a wall-clock
  allocation with no meaning on the simulated timeline.
- One file per side, merged at the end: a merge across processes that only the backfill
  needs, which Prometheus 3.5.0 does itself over overlapping blocks, and a dead LP would
  cost its side's whole file.
- Appending each instant to the file as it is taken: an OpenMetrics family must be
  contiguous, so the file would need a second pass to regroup, for the one case, a sampler
  killed, that hands in nothing either.
- The authority firing at the finish the daemon deadlines it holds, in place of the final
  sample: LP clocks then pass the last essential time by up to an interval, moving
  `final_clocks` and the run's `simulated_seconds` for a metrics file, against D72's rule
  that a daemon deadline fires only once essential work reaches it.
- The final sample at the run's end, one instant for every LP: only the authority holds it
  (`ClockAuthority.final_clocks`), so an LP would read another's clock and stamp a time it
  never reached.
- No final sample where the LP's last point is at its instant: what the LP observed there
  after its timer's sample, such as a frontend's output after its refresh, is lost.

## D75. What is invalid, and is refused rather than reported

| Invalid | Why |
|---|---|
| any metric with a **wall-clock denominator** presented as a simulated rate | there are none today; this is a rule about future additions |
| any **histogram observation of an un-virtualized duration** | D73 — the failure looks authoritative because it came from Prometheus |
| GPU utilisation, power, temperature, and the `server_metrics/` Prometheus scraper | there is no GPU |
| metric emission that **advances the virtual clock** | emission is simulator overhead, not modelled work. Under `01` D4 virtual time advances only for durations the cost model produced, so this is satisfied by construction — but it must be **asserted**, because a hook that accidentally sat inside a K1 (event cost, `advance_to`) path would be invisible |

The standing reminder behind the last row: the instrument changes what it measures. Measure
mode cost ~**11 ms of TTFT on the 27B (4%)**, and a variant that synchronised around each
forward made the run **33% slower** (TPOT 3.26 → 4.33 ms).

**The GPU forward histograms are not telemetry (2026-11-16, #3584).** In a simulated run
`atom:gpu_forward_seconds` observes, for each step `GPUForwardMetrics.measure` times, its
`StepCost.target_seconds`. `measure` times a batch with requests and skips a DP dummy run
and an empty batch, as on a real run, though `_group_step_seconds` prices a dummy run too.
`StepCost.target_seconds` is the target forward's device seconds as the cost backend
prices the rank's batch, with no host time, no draft and no DP group maximum, which
`_group_step_seconds` folds into the step's `predicted_s`. CI's panel reads the same span,
a forward that excludes input preparation, sampling and drafting, and includes its
stream's communication waits (`GPUForwardMetrics.steps`' description). *Revised
(2026-11-18, #3720 c=64):* under DP with a `HostDeviceRecurrence`, each rank's
observation adds its lockstep wait, from its own target end to the group's last, which
the `all_reduce(MAX)` that already ends every rank's target at the group's latest
reports: a rank prefilling a short chunk beside a long one times the long one, as its
device events do. Before it, cell A's c=64 runs read the prefill request series' p50
-22.5% and its mean -10.5% against the real run's, c=32's p50 -21.4%.
`atom:prefill_request_gpu_forward_seconds` sums it over a request's prefill chunks, as
`GPUForwardMetrics.poll` does. TP rank 0 observes it through ATOM's
`GPUForwardMetrics.measure` and `poll` inside its `forward`, before it replies (#3586; D74,
*Revised*); a step the backend refuses observes nothing, and a request with a refused chunk
publishes no sum, since `GPUForwardMetrics` publishes a request's sum only once it has
timed every chunk from the first to the final in order. A duration on the simulated clock
(class D, D77), which the report labels predicted (`19` D115); the telemetry row above
stands. Rejected: `predicted_s`, a step's whole charge, which holds host time and the draft
the panel excludes; the DP group's largest `target_seconds` alone, which drops each
rank's own host-paced start (the wait runs from the rank's own launch to the slowest
rank's arrival at each collective; the revision above reads both ends off the run's own
recurrence, whose last collective it takes as the forward's). `compass run` starts
a simulated server with
`ATOM_ENABLE_METRICS_DEVICE_TIMER=1` unless its environment sets the variable, as CI's
`pd_server_atom.sh` starts both sides; no readings or law key holds it
(`atom/benchmarks/results/metadata.py::PERFORMANCE_KEYS`). The last row's assertion is the
samplers' (D74, *Revised*): a sample changes no LP's clock.

---

## D76. Three validation results this yields for free

1. **The gauges *are* `08` D46's family 1.** KV blocks used/free/indexed, preemption count,
   running and waiting counts, prompt and generation token totals — already exported on both
   the real and simulated sides, in the same format, through the same code. **The
   counting-invariant comparison needs no new instrumentation** — a scrape on each side and
   a diff.
2. **A histogram's `_count` is itself a counting invariant.** Same workload, same requests,
   so real and simulated observation counts compare directly; like every count but
   preemptions and prefill tokens (`08` D46), it is reported, not gated. *The exception
   (2026-11-16, #3584):* the two GPU forward families, which every worker of a real run
   observes (`atom/metrics/gpu.py::record_gpu_forward`) and only TP rank 0 of a simulated
   run (D74, *Revised*), so a TP group's simulated `_count` compares with the real run's
   `tp_rank="0"` series, not with its sum over workers, TP times larger. Their means and
   quantiles compare as they are: copying rank 0's observation to the other ranks, the
   rival D74 rejects, would move neither.
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
| **S — state** | engine state read at sample time: queue depths, pool occupancy, indexed blocks | **valid by construction.** Sampled at its LP's metrics timer (D74, *Revised*). |
| **E — event tally** | a count incremented on an occurrence: requests finished, preemptions, tokens, histogram `_bucket` and `_count` | **valid by construction.** Monotone; sampled at its LP's metrics timer. |
| **D — duration** | a clock delta: TTFT, TPOT, step time, queue wait, histogram `_sum` over durations | **must be a simulated duration.** Audit the observation argument (D73). |
| **R — rate** | a count divided by elapsed time | **do not export.** Export the underlying counter and let PromQL `rate()` compute it over virtual timestamps. If a rate must be exported, its denominator is virtual elapsed. |
| **T — timestamp** | an instant exported as a value: `process_start_time_seconds`, `_created`, exemplar timestamps; `_last_refresh`, which stamps simulated time (D72) | **must declare its clock.** Usually real, because it describes the *process*; virtual if it describes the *run*. Never left implicit. |
| **X — external** | measured outside the engine: GPU telemetry, host stats, the `server_metrics/` scraper | **invalid under simulation.** Refuse. |

ATOM's current metrics (`_AtomMetricsCollector.collect`) are **S** or **E**, which is why
D72's "valid by construction" holds today, except two frontend clock reads, both on the LP
clock under simulation. `atom:metrics_last_refresh_timestamp_seconds` is **T**: it exports
`_last_refresh`, a simulated instant (D72). `atom:stream_longest_silence_seconds` is **D**:
`longest_silence_seconds` (`atom/entrypoints/openai/streaming_dispatch.py`) subtracts two
`time.monotonic()` reads through `atom.utils.time_shim`, so it is a simulated duration (`01` D5). The classes
exist so that the next metric is classified rather than assumed.

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
| Untyped | unclassifiable by definition | must be stated explicitly |

### Enforcement: declare the class, or refuse

Convention and code review will not hold across future additions. The mechanism:

1. **Every metric family declares its class where it is constructed** — one tag per family
   in `metrics.py`, today's and every one added after.
2. **An unclassified metric refuses to export under simulation.** On a real run it is
   unaffected, so the cost of the rule falls only where correctness depends on it.
3. `01` D9's AST test covers the other half — any `observe()` whose argument derives from a
   clock delta, and any un-allowlisted `time.*` read.

The failure mode is the right one: a new metric added without a tag **refuses**, rather than
silently emitting a wall-clock-derived number that looks authoritative because it came from
Prometheus.

### The first build: the class declared where the family is exported (2026-11-16, #3584)

**Rule.** A simulated run's one export is its series files (D74, *Revised*); `/metrics`
has no simulated reader (D72). So the sampler (`atom/compass/run/metrics.py`, #3585) holds a
class table of the families it writes, each with its class, and writes no other: items 1
and 2 above, at the one place a simulated run exports. The table admits S and E, and D
only where the observation argument is audited as a simulated duration: TTFT, inter-token
latency, queue time and the PD KV transfer (D73, *How they landed*) and the two GPU forward
families (D75). The scheduler's request and KV-block gauges and the request-context gauges
are S; the prefix-cache counters and the batch-size, token-count and context-length
histograms, whose `_sum`s count, are E. It holds no R, T or X family. A family it lacks is
refused: no sampler writes it, and the report names each family `panels_for` queries for
the run's deployment kind that the table lacks, its panel empty (`19` D115). The router's
`mesh_router_ttft_seconds`, which no LP holds, keeps its own note (*The first build*, S2).
ATOM's families stay untagged; T45 waits for a reader of `/metrics` under simulation (D72's
scrape), which renders every family.

**Reason.** Principle 6: a family nobody classified is refused by name, not written as a
number that may be wall-clock time. Principle 1: ATOM's metric files change only where
discrete time requires it, and the export that needs the classes is Compass's. Principle 3:
one table where the export happens, which is also the list of what is sampled, so an
engine process need not import CI's script.

**Rejected.**

- T45 first, a tag at each family's construction in ATOM: prometheus_client's families
  carry no field for it (`prometheus_client.metrics_core.Metric(name, documentation, typ,
  unit)`), so `atom/metrics/exporter.py`, `scheduler.py`, `request.py` and `gpu.py` would
  each gain a Compass-only registry beside their families, for a value only the sampler
  reads.
- The sampler selecting by `panels_for` at run time, with no class: item 2's refusal
  unmet, and CI's script imported into every engine process.

### Type-specific hazards

| Hazard | Detail |
|---|---|
| **Summaries cannot be aggregated across DP ranks** | client-computed quantiles do not average. `LLMEngine.get_metrics_statistics()` (`llm_engine.py`) merges per-rank snapshots; summing histogram `_bucket` is valid, averaging summary quantiles is not. This is a general Prometheus truth that would bite ATOM whether or not Compass existed. |
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
uses **that**, not the `generate_latest` ATOM imports in `metrics.py`. The `/metrics`
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
infrastructure and add nothing" should mean in practice. *Since the instruments landed
(D73):* their samples come from `MultiProcessCollector.merge` over an LP's own files, which
the writer stamps, and prometheus_client's OpenMetrics serializer, 0.25.0 as ATOM pins it,
keeps the colon of an `atom:` sample name only with `escaping=ALLOWUTF8` (D74, *Revised*).

---

## The first build (#3584)

D74's revision and `19` D115's Prometheus report, walked through the runs that write them.
#3585 builds the series, #3586 the predicted GPU panels, #3587 the report.

### Scenarios

**S1. An instance's simulated run ends.** `client.sh --stop` on a `compass run` server.

```
 compass run --out OUT ...      ATOM_ENABLE_METRICS_DEVICE_TIMER=1 unless set (D75)
   engine LP    every step with requests: forward -> TP rank 0 prices it, observes its
                  target_seconds (GPUForwardMetrics.measure, poll), then replies;
                  other ranks, a dummy run and an empty batch observe none
                every push_metrics, 1 s of its clock: sample own + workers' files
   frontend LP  every refresh,      1 s of its clock: sample snapshot families + own files
                (labels job=atom observer=api role=standalone instance=frontend)
                each sampler: only the families its class table holds (D77)
 client.sh --stop -> server stops -> engine_loop and frontend_done:
   each sampler once more at its last finite clock, replacing a point there (#3611)
   write OUT/metrics-engine.om.gz, OUT/metrics-frontend.om.gz whole   (D74, Revised)
   -> then hand in: OUT/lp-engine.json, OUT/lp-frontend.json
 -> python -m atom.compass.report.split OUT
 -> python -m atom.compass.report.prometheus OUT                 (19 D115)
      each file decompressed (promtool reads no gzip)
      promtool tsdb create-blocks-from openmetrics, each file -> one TSDB
      prometheus on it, no scrape -> export_report.collect_report(standalone,
        first instant .. first + ceil((last - first) / 5) x 5, step 5,
        window max(60, 4 x interval), interval: the shortest step between
        a file's instants, its last step, the final sample's, left out)
      -> title, notes, GPU labels marked -> export_report.write_report
   -> OUT/prometheus.html, OUT/prometheus.json
 -> python -m atom.compass.report.html OUT    links prometheus.html
```

**S2. A 1P1D deployment's simulated run ends.** `deploy.sh run`, one clock authority.

```
 prefill side: frontend-P, engine-P.dp0..dp7     role=prefill instance=frontend-P
 decode side:  frontend-D, engine-D.dp0..dp7     role=decode  instance=frontend-D
   each of them samples itself at its own timer, on its own clock, as in S1
   LP lmcache, on an lmcache_mp run: no instrument, no sampler
 deploy.sh stops both sides -> every LP hands in
   -> OUT/metrics-<lp>.om.gz for each frontend and engine member
 -> report.prometheus OUT: OUT holds prefill/ and decode/ -> deployment pd
      every file into one TSDB -> collect_report(pd, ...)
      Overall TTFT (mesh_router_ttft_seconds): no series; the notes say the router
        times it on the wall clock, and report.html carries the client's TTFT
   -> OUT/prometheus.html: Prefill and Decode panels, KV transfer wait on decode
```

**S3. A run that ends early, a sampler that dies, a family with no class, a node with no
tool.**

```
 an LP hands in before the finish, or a stop ends the run
   -> every sampler that hands in writes what it sampled, a final sample at its
      last finite clock included (#3611), or that sample's error among its reasons;
      the report covers it
 a frontend or engine member killed before its hand-in -> no file
   -> the report's notes name it, of the run file's hand-ins less LP lmcache
      (run/__init__.py: _hand_ins)
 a family panels_for queries that the samplers' class table lacks (D77)
   -> no file holds it; the report's notes name it, its panel empty
 no series at all | no prometheus or promtool and no download
   | no .github/scripts/atomesh/observability/ in the tree | backfill or query fails
   -> OUT/prometheus.txt says which; report.prometheus exits 0
   -> client.sh / deploy.sh keep their exit status; report.html quotes the reason
```

**S4. A real run.** `compass record` or `deploy.sh record`: no sampler runs (D74,
*Revised*), so `report.prometheus` finds no series and writes only `prometheus.txt`, which
names CI's collector (`collect_metrics.py`) as a real run's report.

**S5. Compass off.** ATOM served without `--compass-run`.

```
 openai_server ... (no --compass-run) -> the run file variable unset -> spec() is None
   engine_core.py -> compass_run.engine_loop(engine): no run -> push_metrics unwrapped
   llm_engine.py  -> compass_run.frontend(config):    no run -> update unwrapped
   ATOM's ModelRunner: GPUForwardMetrics times real device events, if the user set
     ATOM_ENABLE_METRICS_DEVICE_TIMER (drive/run.py does not run)
 -> no sampler, no metrics-<lp>.om.gz, no report
 -> /metrics: exporter, MultiProcessCollector and instruments untouched, byte-identical
```

The samplers install only inside Compass's existing hooks, `engine_loop` and `frontend`,
when `spec()` returns a run file; no ATOM engine or frontend file changes.

### Simpler alternative

One sampler per side in the frontend, rendering its whole registry at each refresh, as
CI's Prometheus scrapes it: one hook and one file per side instead of one per sampler. It
breaks `01` D3's rule that LPs interact only through timestamped channels, since the render
reads the engines' and workers' files off the channels, so a sample's content is not the
state at its instant and differs between two runs of the same inputs; and on #3498's run it adds 28 to 72 s to a side's 761 wall seconds (D74,
*Revised*, Rejected), against the samplers' measured 7.3 to 17.4 s.

For #3611's final sample, the query's end rounded up alone: the grid then reaches each
LP's last timer sample, but the LP's state after it reaches no sample, on #3498's
configuration up to 0.30 s and about 24 SSE events on the decode side.

### Low confidence

None.

### Dependencies

No spike and no sync issue. The cadence's figures were measured on #3498's run, and the backfill
end to end on a synthetic side of the same shape (#3584's named result). #522 keeps the 1P1D scrape channels, unneeded by this build; #3552's
measurement is why real runs keep CI's collector.

### Summary

**Deliverable.** Every simulated run, an instance or a 1P1D deployment, writes its engine
and frontend LPs' Prometheus series on the simulated clock and, when it ends, CI's
Prometheus report from them beside its `report.html`.

**Scope and non-goals.** Simulated runs only. Out: a real run's series or report (CI's
collector), a per-step series, the traffic LP's scrape (#522), the router's TTFT, Grafana,
and any change to `export_report.py` or the page it renders.

**Changes.**

| Where | Change | Task |
|---|---|---|
| `atom/compass/run/metrics.py` (new) | the class table, each family the sampler writes with its D77 class (D77, *The first build*); the sampler: an LP's own files merged, the frontend's snapshot families, the table's families only, CI's labels, the buffer; the writer: OpenMetrics with `ALLOWUTF8`, the order above, gzip, written whole | #3585 |
| `atom/compass/run/__init__.py` | `engine_loop` samples before `utility_handler.push_metrics`, `frontend` after `AtomMetricsExporter.update`; each writes its file where it hands in, before its `lp-<lp>.json` | #3585 |
| `atom/compass/runner/overrides.py` | on TP rank 0, `NonAllocatingRunner.forward` through `GPUForwardMetrics.measure`, its pair's elapsed time the step's `StepCost.target_seconds`, and `GPUForwardMetrics.poll` before the reply; a refused step not measured, and a dummy run or empty batch not timed, as `measure` skips them | #3586 |
| `atom/compass/drive/run.py` | a simulated server started with `ATOM_ENABLE_METRICS_DEVICE_TIMER=1` unless set | #3586 |
| `.github/scripts/atomesh/observability/collect_metrics.py` | `ensure_prometheus(directory, tool="prometheus")`: `ATOMESH_PROMTOOL_BIN` beside `ATOMESH_PROMETHEUS_BIN`, a tool already extracted into `directory` reused; the archive and each tool written under a per-process `.part` name and renamed whole, the archive once its checksum holds, so parallel reports never run another's part-written tool; CI's call unchanged | #3587 |
| `atom/utils/clock.py`, `atom/compass/run/metrics.py`, `atom/compass/run/__init__.py` | `LPRuntime` keeps the clock it held before a `+inf` grant; `Sampler` keeps its timer's read, and `_hand_in`, where it writes the sampler's file, first takes a final sample with it at the LP's clock while finite, else at that clock, replacing a sample at its instant; one that raises skipped and named among the hand-in's reasons | cut from #3611 |
| `atom/compass/report/prometheus.py` | the query's end `first + ceil((last - first) / 5) x 5`, the first grid point at or past the last instant, still at least `first + 1`; the window's interval each file's shortest step but its last, the final sample's, which lands under one interval after the timer's last sample | cut from #3611 |
| `atom/compass/report/prometheus.py` (new) | `python -m atom.compass.report.prometheus OUT`: each `metrics-<lp>.om.gz` decompressed, since promtool reads no gzip, then backfill, serve, `collect_report`, mark (each family `panels_for` queries that the class table lacks named), `write_report`; or `prometheus.txt` | #3587 |
| `atom/compass/report/html.py` | a line linking `prometheus.html` or quoting `prometheus.txt` | #3587 |
| `scripts/compass/client.sh`, `scripts/compass/deploy.sh` | the report after the split and before `report.html`; its failure reported, their exit status kept | #3587 |

**Acceptance items**, each walked by the scenario named:

1. Each engine member and frontend LP of a simulated run writes `metrics-<lp>.om.gz`
   holding the families of the sampler's class table and no other, at each push or refresh
   on its own clock, each request-context series once, with CI's scrape labels, written
   whole before its hand-in; LP `lmcache` writes none; sampling changes no clock; a real
   run and Compass off write none, and `/metrics` is unchanged (S1, S2, S4, S5).
2. Each instant of a file, read back on its own, holds the values the instruments had; the
   whole file backfills with promtool 3.5.0 (S1).
3. With the timer on, TP rank 0 of each engine observes its step's `StepCost.target_seconds`
   once per step `GPUForwardMetrics.measure` times, as on a real run, before its `forward`
   replies, and a prefill request the sum over its chunks; the other ranks, a refused step,
   a dummy run and an empty batch observe nothing; with the timer off, nothing changes
   (S1).
4. Every simulated run's end writes `prometheus.html` and `prometheus.json`: `panels_for`'s
   panels for its deployment kind from its first instant to the first 5 s grid point at or
   past its last, titled simulated, the GPU panels labelled predicted, a deployment's
   Overall TTFT empty with its reason, each family
   `panels_for` queries that the class table lacks named; `report.html` links it (S1, S2,
   S3).
5. With no series, no tool or no `export_report.py`, `prometheus.txt` says why, the stop's
   exit status is unchanged, and `report.html` quotes it; a frontend or engine member with
   no file is named (S3, S4).
6. Each sampler's file ends with a point at its LP's last finite clock holding what its
   instruments held then, one point per series per instant, at a finish, a stop or an
   engine's raise, and at no client run's end; a final sample that raises is skipped and
   named in the hand-in; the report's window stays `max(60, 4 x interval)` at any metrics
   interval (S1, S3; #3611).

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D71 | Scope is ATOM's Prometheus engine metrics only. Harness metrics belong to `06`; simulator observability is run-artifact fields, not a metrics subsystem. | 2026-09-19 |
| D72 | Counters and gauges are valid by construction. **The metrics push and refresh are timers on simulated time, marked daemon**, so they never keep a run alive (`01` D3, D5); `_last_refresh` stamps simulated time. The traffic LP's scrape is not built: its render would read other LPs' files off the channels. | 2026-09-19; revised 2026-10-02; the scrape 2026-11-16 ([#3584](https://github.com/jgong5/ATOM/issues/3584)) |
| D73 | Histograms observe durations, so the clock audit extends from clock *reads* to observation *arguments*. Ask for classic histograms and for instrumentation that is handed a duration rather than computing one inline. They landed classic, handed `time_shim` durations. | 2026-09-19; how they landed 2026-11-16 (#3584) |
| D74 | First: **sample once per engine step**, transport on D72's push cadence, OpenMetrics with virtual timestamps backfilled into a TSDB, the real run treated identically. **Revised:** each engine member and frontend LP samples its own processes' instruments on its own clock at ATOM's metrics push or refresh, of an engine's workers TP rank 0 alone observing, inside its `forward`, the families of its class table, the ones `export_report.py` queries, with CI's scrape labels, each request-context series once; it writes `metrics-<lp>.om.gz` (OpenMetrics, `ALLOWUTF8`, gzip) whole at its last hand-in, before `lp-<lp>.json`, on the LP clock from 0 with no epoch; no sampler on a real run. Each sampler samples once more at its last hand-in, at its LP's last finite clock, replacing a point at that instant. | 2026-09-19; revised 2026-11-16 (#3584); the final sample 2026-11-17 ([#3611](https://github.com/jgong5/ATOM/issues/3611)) |
| D75 | Four classes of invalid metric, refused rather than reported. Emission never advances the virtual clock, and that is asserted. A simulated run's GPU forward histograms observe the `StepCost.target_seconds` of each step `GPUForwardMetrics.measure` times, under DP with a host recurrence plus the rank's lockstep wait for the group's last target end, a simulated duration, not telemetry. | 2026-09-19; the GPU forward histograms 2026-11-16 (#3584); the lockstep wait 2026-11-18 (#3720) |
| D76 | The gauges supply `08`'s counting invariants for free; a histogram's `_count` is one too, and `_sum/_count` is a mean comparable at the noise floor. A simulated run's GPU forward `_count`, TP rank 0's alone, compares with a real run's `tp_rank="0"` series. | 2026-09-19; the GPU forward `_count` 2026-11-16 (#3584) |
| D77 | Classify every metric by the **provenance of its value** (state, event tally, duration, rate, timestamp, external), not by its Prometheus type. Declare the class at construction; an unclassified metric refuses to export under simulation. The backfill writer uses the OpenMetrics serializer and the `timestamp=` argument that `add_metric` already provides. The first build declares the class where a simulated run exports, in its sampler's class table, which writes no family it lacks, and the report names each one it queries that is missing. | 2026-09-19; the first build 2026-11-16 (#3584) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T40~~ | ~~Confirm which histograms are landing, and that they are **classic** rather than native~~ — settled 2026-11-16 by [#3584](https://github.com/jgong5/ATOM/issues/3584): classic, prometheus_client `Histogram` and `WeightedHistogram` (D73) | — |
| T41 | Audit every `observe()` site for the provenance of its value; extend the AST test to cover observation arguments | the sites the report reads are audited (D73, *How they landed*); the AST test is not extended |
| ~~T42~~ | ~~Measure `collect_metrics()` cost per step on the **real** side; decide whether decimation by a stated K is needed~~ — retired 2026-11-16 by [#3584](https://github.com/jgong5/ATOM/issues/3584): no per-step sampler is built, on either side (D74, *Revised*) | — |
| ~~T43~~ | ~~Verify the backfill end to end — produce one block, load it, see the series in Grafana~~ — settled 2026-11-16 by [#3584](https://github.com/jgong5/ATOM/issues/3584): promtool 3.5.0 backfilled one side's series from 0.5 s, Prometheus 3.5.0 served it and `export_report.collect_report` read its panels (D74, *Revised*) | — |
| T44 | Sanity-check histogram bucket ranges against simulated latencies | a quantile pinned to `+Inf` is worth catching once rather than discovering |
| T45 | Tag ATOM's existing metrics with their D77 class | one tag per family in `_AtomMetricsCollector.collect`, mechanical; the first build classifies the families it exports in its sampler's class table instead (D77, *The first build*), so the tags wait for a reader of `/metrics` under simulation (D72's scrape), which renders every family |
| T46 | Decide the DP-aggregation rule per class, and refuse summaries there | `llm_engine.py::LLMEngine.get_metrics_statistics` merges per-rank snapshots; quantiles do not average |
