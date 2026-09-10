# ATOMCompass retrospective — updated 10 September 2026

Reviewed through `87838198` on `feature/atomcompass`, including the original
Claude session's continuation through tasks #56–63. The
[9 September review](RETROSPECTIVE_2026-09-09.md) is preserved separately.
This update checks the new source, reanalyses saved node-18 repeat artifacts,
and exercises remaining validation gaps with CPU-only probes. No new GPU
experiment was run for this review.

**The priority has moved from the known deadlock to the 27B first-token
delay, with unfinished validity checks alongside it.** The submission fix,
shared-clock timestamps, richer calibration and independent real-run repeats
are substantive progress. The saved 27B workload still predicts median TTFT
at 52.56 s against 27.54–27.62 s measured. Its large discrepancy is concentrated
after the final prefill dispatch. The next experiment should locate the first
different request-state transition or scheduling decision; it should not fit
another admission constant or assume a cause from aggregate agreement.

The current logs are much more informative. They still do not justify declaring
all of P0 complete, declaring the remaining error independent of costs, or
claiming that the hardware was continuously uncontended.

**What changed since the first review**

| Area | New evidence | Current assessment |
| --- | --- | --- |
| Submission deadlock | `dcf3e679`: one connection/thread per request, explicit limit of 1,024. Subsequent 300-request runs complete. | The identified 64-request dependency is fixed. Bulk submission remains the scalable replacement. |
| Timing reconstruction | `d8e81715`, `b8aabc9c`: `started_at` comes from the engine core and travels with the batch. | The original inferred start times and worker clock mismatch are fixed for the measured GPU path. This records dispatch, not GPU completion. |
| Run identification | `9f1fdfe5`: scaled arrivals are saved, and a run manifest records count, trace hash, model and time scale. | Improved, but acceptance checks and source/calibration provenance remain incomplete. |
| Reused calibration | `small.sh` preserves its sweep table. | The specific accidental deletion is fixed. Other scripts still overwrite fixed filenames; `big.sh` tests the unsuffixed sweep path although TP writes suffixed files. |
| Admission hypothesis, #56 | Request-position analysis refuted a constant quarter-second delay; the small model accumulated cost error under load. | Useful rejection of that hypothesis. It does not prove that every scheduling contract is equivalent. |
| Decode cost, #57–59 | Unequal-context calibration batches plus a padding feature improve fixed-sequence prediction. Mildly unequal batches improve the 27B's low buckets. | Strong evidence for a missing predictor and insufficient variation in calibration. Accuracy remains configuration- and workload-dependent. |
| Real-run repeats, #60 | Five 0.6B and four 27B runs; all expected artifacts are available in the reviewed packet. | The large 27B TTFT mismatch survives these repeats. General irreproducibility is not established. |
| Localization, #61–62 | Prefill work/cost largely agree; long prefill streaks and delayed first-token timestamps differ. | A useful location and candidate mechanism, not a complete causal explanation. |
| Decision records, #63 | `87838198` carries queue counts, branch, budget and tick on each produced batch; the session reports a successful small-model smoke run. | Instrumentation exists. It is a summary taken after selection and needs stronger semantics before it can explain why a request was skipped. The 27B has not yet been compared using these records. |

**What the preserved measurements establish**

I recomputed these numbers from `rep06_real1`–`real5`, `rep06_modelled`,
`rep27_real1`–`real4`, and `rep27_modelled` on node 18. Workloads match within
each campaign. Every required request joins to its engine record and first step;
input and output lengths match; the checked ordering
`arrival ≤ first dispatch ≤ first token ≤ finish` holds for all requests.
These checks validate the specific artifacts, even though the general harness
does not yet enforce all of them.

| Campaign | Real run medians | Simulation median | Error against median of real run medians |
| --- | --- | ---: | ---: |
| 0.6B TTFT, 300 requests, 5 real repeats | 3.550–3.893 s | 3.623 s | −1.13% |
| 0.6B latency | 4.455–4.718 s | 4.870 s | +7.76% |
| 27B TP=4 TTFT, 20 requests, 4 real repeats | 27.543–27.622 s | 52.557 s | +90.51% |
| 27B TP=4 latency | 74.498–74.890 s | 78.663 s | +5.42% |

The 27B TTFT range is **0.287%** of its median, not literally zero; latency's
range is **0.525%**. These ranges describe four runs, not confidence intervals.
Only one simulated run appears in each repeat campaign. The error definition
here is median-based; it must not be mixed with earlier “on totals” errors.

The new calibration is independently checkable. Applying the current fit to
the first real repeat's recorded steps gives:

| Group | 0.6B error in summed step time | 27B error in summed step time |
| --- | ---: | ---: |
| Prefill | −8.53% | +0.10% |
| Decode bucket 1 | −7.41% | −0.10% |
| Decode bucket 2 | −4.16% | −2.46% |
| Decode bucket 4 | −3.18% | −4.76% |
| Decode bucket 8 | −10.23% | −1.68% |
| Decode bucket 16 | −10.03% | −22.70% |
| Decode bucket 32 | +5.95% | Not exercised |

The preserved calibration and current fit reproduce every saved simulated step
duration in these two campaigns exactly. This supports their correspondence
despite the missing calibration hash in the run manifests. It does not create
an independent generalisation test. “Every bucket within 8%” describes the
earlier #58 experiment; it does not hold on these later repeats. The 27B still
has a material bucket-16 error and no bucket-32 evidence in this campaign.

For the 27B, the first real repeat has 106 prefill steps and 235.411 s of
measured prefill work; simulation has 105 steps and 237.820 s. Both process
1,681,024 prefill tokens. The prefill fit applied to the real steps gives about
235.65 s, +0.10%. The largest unbroken prefill streak is 42 steps/93.947 s real
and 63 steps/152.567 s simulated. These observations were reproduced from the
saved tables.

The request-level decomposition gives approximately 6.25 s versus 4.84 s
from arrival to first dispatch, and 20.56 s versus 45.73 s from first dispatch
to first token. Its estimated final-prefill-to-first-token interval is 9.19 s
versus 32.21 s, with maxima 16.55 s versus 139.03 s. The short-model analogue
is about 0.010 s versus zero at the median. This is a much sharper diagnostic
than comparing total latency alone.

**How the latest interpretation should be tightened**

The final-prefill endpoint is still estimated as
`started_at + seconds` in `after_prefill.py`. `started_at` is stamped before
worker dispatch; `seconds` on the real side is a CUDA-event duration. Their
sum is not a directly measured completion timestamp. It omits dispatch/queue
offsets and does not identify when output becomes visible to the scheduler.
The tens-of-seconds discrepancy is worth pursuing, but confirm the exact
interval with completion/output events before calling it entirely a delay in
scheduling the first decode step.

The first generated token normally originates from final prefill. Record
whether that token was produced, deferred, returned under a previous batch's
IDs, appended to the sequence, and then timestamped. A later first-token
timestamp does not prove that an additional decode computation was required.
The real runner's deferred-output path and the predictor's synthesized
current-batch output remain an important contract to check.

Matching total prefill cost and work rules out a large uniform prefill-cost
bias on this sample. It does not establish equivalent request ordering, cost
for every relevant step, or state at an arrival. Bucket-16 decode error and
real host work can still change which arrivals are visible before the next
prefill. The scheduler does not compare oracle prices as competing bids; its
prefill-first rule sees readiness and resource state, which cost-driven clock
advancement can change.

The streak analysis needs a matched-request calculation. A ratio of longest
streaks (1.6×) cannot be compared with a ratio of median waits (3.5×) to decide
how much is explained: they are different statistics over different sets.
Count the work between each request's actual final-prefill completion and
first-token publication. Equal peak/mean concurrent-prefill counts likewise do
not rule out different ordering. The subsequent “admission→concurrency chain
refuted” observation is narrower than proof that arrival timing is irrelevant.

The repeated runs refute the claim that the 27B must fluctuate by 2×. They do
not identify why the earlier 56.4 s run differed. Autotuning is plausible, but
needs a warm/cold intervention or recorded compile activity. Keep that run
separate until explained. Similarly, a deliberately compressed burst workload
is a valid stress test when its transformation is declared; it does not
validate the original trace's arrival process.

**What remains unfinished in P0**

The repairs support useful measurements, but “P0 complete” is too broad:

- `arrival_barrier_timed_out` is set on the scheduler, but no source reference
  exports or consumes it in the result path. A stored attribute is not yet an
  end-to-end validity flag.
- `cc_compare.py` accepts incomplete request joins and merely warns about
  workload/time-scale mismatches. A CPU probe declaring two requests with only
  one matching engine record exits successfully and prints metrics.
- `queue_wait.py` checks first dispatch against first token, but accepts first
  dispatch before arrival. Its failure path prints a message rather than
  returning a failing process status. It does not establish full trace coverage.
- `spread.py` silently skips missing repeats. A CPU probe requesting four with
  only one present still reports a distribution; with no GPU snapshots it
  prints “quiet at every sample.” This contradicts the session's assertion that
  it refuses incomplete distributions. All expected runs are present in the
  reviewed packet; this is a remaining acceptance defect.
- The new replay/barrier tests do not exercise the dependency. Their `_workers`
  helper returns `len(range(count))`, independent of the client's pool choice;
  the refusal test searches source text. Replace them with the actual client
  driving a blocking test transport and observed acceptance/refusal.
- The real runner can still leave final pending CUDA-event records unwritten.
  Its CPU fallback stamps `perf_counter()` into `started_at`, while engine
  timestamps use another domain. These do not invalidate the checked GPU
  first-dispatch records, but prevent a universal timing contract.

The original tautological bucket assertion was removed, and its underlying
premise was correctly fixed: sequences in one graph bucket can still differ
in total history. Keep that improvement and apply the same test scrutiny at
the new client/server boundary.

Manifests identify effective arrivals and the input trace, but the reviewed
remote manifests have `revision: null` and no calibration hash, model revision,
server build, device mapping or software fingerprint. The replay client's
local Git revision would not identify the remote server anyway. Return server
provenance with results and preserve unique run directories. Required analysis
programs and campaign scripts remain under ignored `agent_scratch`; promote
the harness into maintained source when fixing its acceptance logic.

**The GPU evidence needs a correction**

The continuation says the 98% activity was only on unused cards. The saved
arrays do not support that dismissal. The launch command sets `GPUS=0,1,2,3`;
`rep27_gpu_before_4.json` has 98% at array position 0, and
`rep27_gpu_before_model.json` has 98% at position 1. Old snapshots lack explicit
device IDs and the `visible` field, so preserve the mapping ambiguity rather
than treating this as proven off-target activity. The corrected probe was
added after these snapshots.

All reviewed `sclk_mhz` arrays are empty. There are before/after snapshots for
real repeats and a before-model snapshot, but no continuous observation and no
after-model snapshot. The timing distribution is observed and tight, while
uninterrupted isolation and clock stability are unproven. Do not discard the
timings solely on this basis, and do not claim contention explains the 90%
residual. Sample explicit physical GPU IDs during the next run, distinguish
missing telemetry from zero activity, and identify the server's own activity.

**The new decision trace needs one more refinement**

`Scheduler._decision_record()` records queue summaries near batch construction,
after selection has already removed or moved requests. It describes remaining
state, not a full snapshot of alternatives considered. It has no per-request
skip reason, selected/eligible identity comparison, partial-prefill state in
running requests, or record for a call returning no batch. `kind` records which
branch won, not why it won.

Keep the compact summary, but for the divergent interval record pre-selection
state, selected IDs, reason codes for excluded candidates, remaining token/KV
budgets, pending outputs and completed final chunks. Record empty-call reasons
and a return/output timestamp on the engine clock. The narrow question is the
first point at which equivalent requests cease to be eligible, chosen, or
completed in the same way.

The smoke run's first tick of 89,337 demonstrates many scheduling calls before
the first recorded step. Without empty-call reason records, attributing every
one of the preceding 89,336 calls to the barrier remains an inference. Frozen
virtual time means they do not directly add simulated seconds; it does not
prove zero effect through CPU contention or receipt order. Measure elapsed CPU
cost before ranking bulk-submit optimisation against fidelity work.

The trace scans the waiting queue twice and runs on ordinary real scheduling
paths. Measure its overhead on the small model; a short output record does not
imply a free observation. Missing required fields should invalidate the
analysis even if serving correctly continues without them.

**Revised order of work**

| Order | Work | Evidence needed to finish |
| --- | --- | --- |
| 1 | Complete validity alongside the 27B investigation. | Timeout state reaches results; missing requests/repeats, mismatched workloads, missing telemetry and bad time ordering produce explicit verdicts. Integration tests drive the real client and analysis commands. |
| 2 | Run the proposed #64 comparison, using and refining #63's trace for the 27B post-prefill interval. | A valid comparison with decision and completion/output records; matched requests; the first differing eligibility, selection or publication event. Inspect deferred-output semantics before attributing the interval to decode scheduling. |
| 3 | Intervene at that first divergence. | Hold arrivals, tie ordering, budget and completion semantics fixed. Change one cost/timing/state factor and reproduce its predicted effect. A small deterministic scheduler case and a real-run check agree. |
| 4 | Close remaining cost gaps. | Address 27B bucket 16, validate bucket 32 separately, and test small-model predictions on held-out workloads. Keep per-sequence prefill features and mixed batches on the list. |
| 5 | Establish repeatability and generalisation separately. | Repeat both modes sufficiently to quantify variation; establish warmup and instrumentation costs; evaluate unused sessions, shapes and configurations with frozen calibration. |
| 6 | Restore decision-quality and speed gates. | Feasibility, configuration ranking/regret and SLO accuracy across TP/workload choices; valid startup/calibration/replay accounting; explicit GPU-free execution and cache-reuse tests. |

Operator-level pricing, generated-kernel coverage, #51 instability and compiled
graph derivation remain relevant to cross-configuration prediction. They do not
block isolating this fitted-oracle serving mismatch. EP/MoE and asymmetric
parallelism remain later work; the 27B is not the model for an EP validation.

Repeated calibration changes make the current cc-traces slices development
sets. Repeats of the same slice measure repeatability, not unseen-workload
accuracy. Batch unevenness is a useful empirical predictor; improvement does
not prove every attention kernel reads a padded rectangle. Dropping constant
columns addresses that degeneracy, not general rank deficiency or conditioning;
the prediction guard still checks total context rather than joint support
including padding.

The earlier review's product concerns still apply: per-term memory validation
near feasibility limits, prefix-preserving workloads, open-loop versus
dependency-driven arrivals, fixed-schedule versus deployment-counterfactual
gap analysis, and configuration choice under uncertainty. Saturated workloads
matter and can be diagnosed with controlled durations and matched event traces;
saturation does not make scheduler behaviour inherently unobservable.

**Evidence and review limits**

The git-ignored packet `agent_scratch/retrospective_20260910/` contains
`node18_repeats.tgz`, hashes in `manifest.json`, results in `audit.json`, and
the scripts below. It preserves both campaigns' request records, step tables,
calibration and telemetry, plus relevant logs and scripts. Source inspection
covered `4572a36a` through `87838198`. The #63 smoke tick observation comes
from the saved session and commit; the repeated-run numbers above were
independently recomputed from raw artifacts.

Relevant source: [replay client](../../scripts/compass/replay.py),
[scheduler](../model_engine/scheduler.py), and [measurement writer](runtime/runner.py).
The [computed audit](../../agent_scratch/retrospective_20260910/audit.json) and
[validity probes](../../agent_scratch/retrospective_20260910/probe_validity.py)
are preserved locally with the evidence packet.

From `/md1/users/jgong5/gpu_docker`:

```bash
./shell.sh python /workspace/ATOM/agent_scratch/retrospective_20260910/audit_continuation.py
./shell.sh python /workspace/ATOM/agent_scratch/retrospective_20260910/probe_validity.py
```

Both scripts ran successfully. The second deliberately demonstrates acceptance
defects, not that the harness is correct. This update changes the retrospective
and handover pointers; production fixes and new GPU campaigns remain work for
the implementation continuation.
