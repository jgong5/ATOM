# ATOMCompass retrospective — 9 September 2026

**Historical review.** The continuation and current priorities are assessed in
[RETROSPECTIVE.md](RETROSPECTIVE.md). This document preserves the evidence and
conclusions at `4572a36a`; subsequent fixes change which items remain open.
Its reproduction scripts import source from the checkout, so reproducing the
original defects requires the reviewed source revision, not the latest branch.

Reviewed at `4572a36a` on `feature/atomcompass`, using the original Claude
session, the scope and handover documents, the relevant implementation, and
the saved 300-request experiment on node 18. New checks in this review ran on
CPU inside `gpu_docker`; no new GPU benchmark was run.

**The next priority is to repair the evidence chain before tuning admission.**
The loaded experiment used to close Claude task #55 timed out waiting for its
workload and explicitly marked its latencies invalid. Its queue-wait analysis
also reconstructs an invalid timeline. Neither “20% late admission” nor “the
simulator is slower under saturation” is an established result. Task #56 should
start by obtaining a valid comparison, then separating cost error from event
ordering. This corrects the context handoff as well as the previous analysis.

This review distinguishes findings reproduced here, historical results that
were inspected but not rerun, and proposed experiments. The older notes remain
useful as an experiment history; their latest conclusion is not automatically
the strongest evidence.

**What the work got right**

Reusing ATOM's scheduler and block manager is a useful architectural choice:
it reduces the amount of serving logic that Compass must maintain. The separate
cost-oracle interface also makes it possible to investigate serving semantics
with an inexpensive fitted model while developing operator-level prediction.

Several methodological improvements were substantial: separating prefill and
decode; respecting CUDA-graph buckets; replacing per-step synchronisation with
deferred CUDA-event readings; measuring memory terms separately; capturing
ambient attention context; and checking errors per request rather than accepting
cancelling totals. The history openly records failed hypotheses, including the
per-launch boundary explanation. Preserve that evidence and those corrections.

The fixed-workload 27B and memory results are useful demonstrations within their
stated configurations. They do not yet establish unseen-configuration prediction,
production serving fidelity, or the original configuration-selection gate.

**1. Findings that change the current diagnosis**

| Finding | Evidence checked here | Consequence |
| --- | --- | --- |
| The loaded simulated run could not submit its full workload before the server's barrier expired. | `replay.py` uses at most 64 workers for unpaced simulation; each worker waits for its completion. The scheduler waits for all 300 requests, with a 120 s timeout. The saved server log says only 64/300 arrived and explicitly invalidates latency. | Withdraw the loaded run as an accuracy baseline. The 122 s elapsed time includes the 120 s wait and cannot establish intrinsic simulator throughput. |
| The queue-wait reconstruction is not a timestamp measurement. | `first_step_at()` adds a row's preceding host gap after assigning its start, starts the first forward at the first request's arrival, and has no record of virtual idle jumps. GPU durations and host gaps can overlap. | Withdraw the 4.0 s versus 4.8 s comparison as evidence of admission delay. Record actual events in a common clock domain. |
| The invalid reconstruction fails on the original real-run data. | For **207/300 requests**, reconstructed first-step time is later than the recorded first token. The largest violation is **0.987 s**. Summed device durations plus host gaps give **37.545 s**, while first arrival to last finish is **35.533 s**. | The problem is present in the artifact used for the conclusion, not only in a synthetic example. |
| The light-to-heavy comparison did not reuse its calibration. | `small.sh` deletes `sm_*.jsonl`, including the sweep table, before testing `SKIP_SWEEP`. The loaded log says `### sweep`, not `### sweep reused`. | The claim “same calibration” is false. Both workload size and arrival compression also changed; a controlled load comparison should hold the requests and calibration fixed. |
| Cost error remains independently of simulated scheduling. | Applying the existing fitted coefficients to each saved **real** step gives the table below. This holds the real step sequence fixed. | Do not attribute the whole error to scheduling or admission, even after the transport problem is fixed. |

The replay/barrier interaction was reproduced with the actual `replay.main`
and the actual scheduler barrier method, substituting a blocking in-memory
transport and a 0.25 s timeout. With 64 requests, all arrived before release.
With 65 and 300, the barrier timed out at exactly 64 received requests. This is
a test of the client/server dependency; it does not require model execution.

The saved node-18 server log contains:

> only 64 of 300 declared requests arrived within 120s; running anyway.
> Virtual time may now advance past an arrival still in flight, which makes
> that request retroactively late -- treat this run's latencies as invalid.

Subtracting 120 s and advertising the remainder as a speedup would also be
wrong: after the barrier opened, later requests entered as earlier ones
completed. That execution no longer had the declared arrival semantics.

The cost check uses `100 × (sum(predictions) / sum(measurements) − 1)` within
each group. These are diagnostics from one real run, not uncertainty bounds or
a new validated baseline.

| Real step group | Recorded steps | Error in summed step time | Coverage observation |
| --- | ---: | ---: | --- |
| Prefill | 171 | −0.33% | Inside the existing marginal feature bounds; aggregate agreement does not establish accuracy for every batch. |
| Decode bucket 32 | 1,615 | −8.06% | Inside recorded context bounds. |
| Decode bucket 16 | 481 | −28.90% | Inside recorded context bounds. |
| Decode bucket 8 | 237 | −40.55% | Inside recorded context bounds. |
| Decode bucket 4 | 69 | −5.09% | All 69 outside recorded context bounds. |
| Decode bucket 2 | 359 | +6.48% | All 359 outside recorded context bounds. |
| Decode bucket 1 | 34 | −0.54% | Inside recorded context bounds. |

The simulated log independently warns about buckets 4 and 2. Both sides did
complete 300 requests with the requested output lengths: **54,997 output
tokens each**. That rules out an output-length mismatch for this experiment;
it does not repair its arrival semantics. Local and remote SHA-256 hashes
matched for the replay client, scheduler, runner, queue analysis, and campaign
script, so these code findings apply to the preserved run.

Sources: [replay client](../../scripts/compass/replay.py),
[scheduler](../model_engine/scheduler.py),
[measurement writer](runtime/runner.py), and the local evidence packet
`agent_scratch/retrospective_20260909/` described at the end.

**2. Analysis habits to change**

**Separate observation, explanation, and closure.** “TTFT high, decode low”
is an observation. “Admission caused both” requires an intervention or event
trace showing that mechanism. Similar signs on two workloads do not prove a
shared cause. “Resolved” should name the criterion that passed. #55 is an
unresolved attribution question after this audit; #53 improved coverage but
did not establish that all decode costs were accurate; #54 added a useful
history feature but did not establish complete schedule fidelity.

**Do not eliminate cost error from aggregate agreement.** A mean over a broad
context bin can hide errors at particular buckets, batch compositions, or
history lengths. Under saturation, even a small service-time bias can change
queueing considerably. Use the real sequence to assess cost predictions first;
use controlled durations to assess the scheduler separately. Then assess their
combined behaviour.

**Scheduler reuse is conditional fidelity.** The same scheduler code produces
the same decisions only with equivalent inputs and state: ordered arrivals,
ready times, token completion events, KV capacity, cached prefixes, preemption,
and clock progression. Compass already changes several of these interfaces.
`Scheduler.add()` appends in receipt order, while the client posts concurrently.
The barrier establishes completeness, not canonical event order. The real
runner can also return deferred outputs, whereas prediction synthesises current
batch tokens. These are contracts to test, not newly proven causes of the
27B discrepancy. `_advance_to_next_arrival()` runs before admission in the same
`schedule()` call; a one-tick delay has not been demonstrated.

**A residual does not identify a physical mechanism.** Keep the 2.25 µs
correction explicitly empirical and limited to its calibration domain. The
top-level explanation in `priced.py` still contradicts its later correction.
Similarly, “no visible microsecond gaps in these traces” supports rejection of
that gap model; it does not prove zero hardware dependency cost. Nanosecond
claims require timestamp-resolution and event-boundary evidence. Profiling
perturbation need not cancel equally between kernels and gaps on every workload.

For general timeline accounting, clip intervals to the window and compute the
union separately from summed kernel work. `residual.py` and
`step_accounting.py` sum durations of events whose starts fall in a window.
That needs a verified serial-stream assumption. Two overlapping 4 ms events
in a 10 ms window can have 6 ms of union coverage: summing them gives 2 ms idle
where the union gives 4 ms. A check that the sum is below the window does not
detect this case. The previous fixed serial traces may still support their
conclusions; the calculation is not a general multi-stream proof.

**3. Measurement and experiment design**

Make run validity a structured result, enforced by the comparison program.
The current scripts can print “0 failed” after a barrier timeout; the replay
client also returns success after individual request failures. A metric should
not become a headline unless required requests joined, lengths matched, the
arrival protocol completed, clocks were consistent, and no fatal validity
warning occurred. Coverage warnings should produce counts and an explicit
extrapolation status, rather than disappearing in a server log.

Use one immutable directory per run. Record code and model revisions, image
digest and library versions, device/topology, compilation mode, graph ladder,
KV budget, workload and calibration hashes, transformation parameters, seeds,
warnings, and completion status. `replay.py` saves the workload before applying
`time_scale` and does not save that setting, so the JSON alone cannot reproduce
the loaded arrival process. Preserve raw evidence before the next campaign.

Define each time boundary. Current engine arrival is stamped after tokenisation
and sequence construction in `LLMEngine.preprocess`; it is not HTTP ingress.
Report engine TTFT as such, and measure or model the additional stages before
claiming client-visible SLOs. Record request arrival, ready/enqueue, first
scheduled step, first token, and finish directly using a consistent run clock.
Record step start/end and virtual jumps, keeping CUDA duration as a separate
quantity. Drain final pending event records after the measured window: the
runner currently allows terminal steps to go unwritten, acceptable for some
calibration sampling but unsuitable for exact event reconstruction.

Useful validity invariants include:

- Every required workload ID maps exactly once through completion ID and
  internal sequence ID; no comparison silently uses only an intersection.
- Arrival ≤ first scheduled step ≤ first token ≤ finish, under the declared
  timestamp definitions. Calculate phase differences per request before
  aggregating them; never subtract medians to obtain a median phase duration.
- Every declared arrival is present before virtual time may pass it; ties have
  an explicit order. Completion and output counts reconcile with the workload.
- All required step records and final events are present. Device busy time,
  host activity, and elapsed time are not assumed additive when they overlap.

Repeat independent runs, with alternating or randomised measurement order and
calibration checks before and after. Record GPU utilisation, clocks and power,
as well as free VRAM: empty memory alone does not establish an uncontended
timing environment. Resample by run or session, not by thousands of correlated
decode tokens. “Mean error smaller than standard deviation” is not an
equivalence test or proof that further improvement is impossible. Compare an
uncertainty interval with an explicit acceptable error range.

Lock calibration before evaluating a held-out workload. `validate.py` normally
derives admission from the evaluation run's TTFT and prefill time. That is
useful for a diagnostic with measured admission, but it is not a fully held-out
serving prediction. Report that distinction and test the frozen admission model
on a separate run. Split calibration and evaluation by shape family, run,
workload/session, and eventually configuration; adjacent decode rows from one
request are weak independent evidence.

**4. Modelling gaps and product goals**

The history feature was a useful correction, but the current prefill features
still collapse a batch too early. With per-sequence new tokens `q_i` and prior
history `h_i`, attention-work features should distinguish `sum(q_i²)` and
`sum(q_i × h_i)` from `(sum q_i)²` and `(sum q_i) × (sum h_i)`. The latter
introduce interactions between unrelated requests. Keep GEMM work, padding,
kernel regimes, and any mixed decode work distinct; these are candidate
features to validate, not a claim that FLOPs alone predict time.

For example, new-token lengths `[15000, 1000]` and post-chunk contexts
`[16000, 64000]` have the same current features as contexts `[64000, 16000]`.
Their `sum(q_i × h_i)` differs from **78 million to 750 million**. The existing
`StepShape` retains the information needed to distinguish them. Multi-request
prefill occurs in 72 of the 171 real prefill steps in the saved loaded run.

Coverage must describe joint support, kernel regime, and hardware/configuration
identity. A feature bounding box is necessary but insufficient. Check design
matrix rank and conditioning as well as row count; many nearly identical rows
do not identify a slope. Inspect outlier rejection per bucket: the saved fit
discards 10/40 rows at both buckets 2 and 4. A large residual may be real kernel
behaviour rather than contamination. Keep reasons for exclusions and compare
against fits without rejection. Choose loss functions for both relative error
and the service-time bias that drives queueing; neither relative nor absolute
error alone is universally the right objective.

Memory is comparatively mature, but small KV-block percentage errors on a large
pool do not establish accuracy near feasibility thresholds. Preserve the
per-term checks; add low-headroom cases, state-slot limits, preemption and
prefix reuse, and at least one correctly rejected infeasible configuration.
The node-18 real and simulated runs used **90,346 versus 90,787 KV blocks**.
This is not evidence that KV capacity bound that run; it is evidence that
“same scheduler” did not mean identical initial memory state. Hold the budget
fixed while isolating scheduling, then validate the memory model separately.

The vocabulary needs a small correction. There is no general SOL time oracle,
but built-model byte counts and liveness arithmetic are mechanistic components
combined with empirical scratch/pool constants. “Nothing is analytical” is too
broad. A simulator can also use measured costs and still be a simulator; the
missing capability is generalisation to unmeasured configurations. State that
capability and its evidence directly.

The original [PoC scope](../../../llm_infer_deploy_study/perf_modeling/presentation/POC_SCOPE.md)
made TP configuration selection, extrapolation from limited capture,
GPU-free replay, and ≥5× speed primary gates. They are not demonstrated by the
current serving results. Meta derivation and memory arithmetic are building
blocks, not proof of an end-to-end GPU-free serving path: the predictor still
inherits model initialisation and live memory setup unless those paths are
explicitly substituted. An exact graph cache also needs a measured hit rate;
context grows every decode step, so exact shape keys need not repeat.

Restore decision-quality validation: best feasible configuration, selection
regret, and SLO attainment, alongside timing error. Compare configurations
within the same workload and objective. Treat statistically tied configurations
as ties rather than demanding a noisy top-1 ordering. Correct ranking alone is
insufficient if systematic bias misclassifies an SLO or memory limit.

The cc-traces campaign is currently a transformed arrival/length workload:
synthetic prompts, prefix caching off, and sometimes filtered lengths, overlaid
sessions, clipped gaps, or compressed arrivals. That is useful if named
precisely. It does not validate agentic prefix reuse. Moreover, a closed-loop
agent's next request may depend on the previous completion; fixed recorded
timestamps are an open-loop stress workload when changing engine speed.
Introduce prefix-preserving and dependency-aware workloads after a valid
open-loop baseline, before claiming representative production coverage.

Distinguish two uses of gap analysis. Costing a **fixed real step sequence**
with two oracles helps attribute cost differences. Running both oracles through
the scheduler measures a deployment counterfactual: faster steps can change
arrivals seen, batching, KV state and later shapes. Those totals cannot be
subtracted into per-operator causes without accounting for the changed work.

**5. Revised priorities and completion criteria**

The numbers below are priorities for this retrospective, not new GitHub issues.

| Priority | Work | Completion evidence |
| --- | --- | --- |
| P0 | Fix workload submission/barrier compatibility and enforce validity. Preserve calibration and immutable run artifacts. | Replays below and above the 64-request boundary, including 300, complete with all arrivals known, no timeout, no silent partial success, and reproducible ordering. Prefer a bulk schedule upload with acknowledgement for scale; one thread per request is only a bounded interim option. |
| P0 | Replace inferred queue timestamps with direct event records and complete the final trace. | Per-request temporal and count invariants pass; queueing, prefill execution, and post-first-token waiting reconcile on one timeline. The existing 207 violations disappear for a demonstrated reason. |
| P1 | Re-establish the small-model baseline and separate costs from scheduling. | Freeze workload, calibration, budget and event ordering. Cost the real steps independently; run deterministic scheduler cases with controlled completions; locate the first state/decision divergence; then run the valid full comparison. |
| P1 | Correct and validate cost behaviour in the exercised regimes. | Address bucket-8/16 residuals and bucket-2/4 coverage; test per-sequence prefill features. Hold out whole shapes/runs, report per-group and total service-time error, and verify resulting TTFT/TPOT effects. |
| P2 | Return to the original TP × short/long-workload decision gate. | Frozen calibration, independent repeats, feasibility checks, configuration ranking/regret, throughput and latency/SLO error. Include a genuinely unmeasured configuration and show what data were transferred. |
| P2 | Establish simulator cost and practical reuse. | Separate calibration, startup, workload loading, simulation, and reporting time; measure valid light and saturated workloads, steps/s, requests/s, CPU/GPU resources, and amortisation across configuration sweeps. Test the actual GPU-free path explicitly. |
| P2 | Complete realistic serving semantics. | Prefix-preserving replay, clearly labelled open-loop versus dependency-driven arrivals, and cache/preemption/state-pressure cases. |
| P3 | Improve the operator-cost path according to decision impact. | Quantify generated-kernel bias, compile/derive equivalence, isolated-versus-in-situ error, and #51 instability on the configurations they affect. Each correction has an independent matched-step check. |
| Deferred | EP/MoE, asymmetric parallelism, and a broad SOL oracle. | Start after the above baseline or as a separately scoped product milestone. EP needs a suitable MoE model and visible communication; equal FLOPs at EP=TP do not by themselves establish a no-op. |

Operator-level work remains necessary for the original cross-configuration
ambition. It should not block diagnosing a calibrated-oracle serving failure:
`silu_and_mul` microbenchmark instability is a different path from the fitted
step oracle used in this campaign. Conversely, good fitted serving results
would not close the operator-level extrapolation goal.

Distributed virtual time is a real design challenge, not evidence that PP/DP/PD
simulation is impossible. A central event model and coordinated rank executors
are possible design directions with different costs. Keep claims such as
“nobody has published a solution” out of conclusions without a dated, scoped
literature check. The current one-rank surrogate's limits should be stated as
implementation limits. A small SOL oracle can also be checked for arithmetic,
units and bound assumptions even though closeness to measured latency is not
its acceptance criterion.

**6. Documentation and testing discipline**

`POC_SUMMARY.md` still recommends the decode-coverage work already completed
in #53 and repeats its rejected causal explanation. It also says prefills are
untraced, although `trace_prefill` exists. `DESIGN_NOTES.md` contains useful
corrections after older claims; it cannot be read as a flat list of current
facts. The scope document also retains the superseded dense-model assumption.

Use one current status table with a revision/date, validity, calibration domain,
evidence link, and next discriminating experiment for each claim. Retain the
chronology separately. Update the status when evidence changes instead of
appending a confident conclusion that leaves the old one authoritative.

Unit-test counts are not measurement validation. Add tests at the boundaries
that failed: replay-to-barrier, timestamp-to-request join, and fatal
warning-to-run verdict. Ensure an assertion can fail for the behaviour its name
promises: the bucket test currently permits `at_12 == approx(at_16) or at_12 > 0`,
so any positive incorrect result passes. Small deterministic scenarios should
precede multi-minute GPU campaigns. A GPU experiment then tests the residual
question that those scenarios cannot answer.

**Evidence and reproduction**

Local evidence is preserved in the git-ignored directory
`agent_scratch/retrospective_20260909/`. It includes the eight source artifacts
in `node18_loaded.tgz`, hashes in `manifest.json`, extracted server logs,
`artifact_audit.json`, `cost_audit.json`, and two reproduction scripts. The
archive is a local evidence packet, not a source-controlled fixture.

From `/md1/users/jgong5/gpu_docker`:

```bash
./shell.sh python /workspace/ATOM/agent_scratch/retrospective_20260909/probe_harness.py
./shell.sh python /workspace/ATOM/agent_scratch/retrospective_20260909/audit_artifacts.py
```

The first reproduces the barrier dependency and two queue-timeline
counterexamples. The second recomputes the temporal violations and evaluates
the existing cost features/fit on the preserved step sequences. The scripts
were run successfully during this review. Production code was not changed;
historical accuracy claims outside these artifacts were not independently
remeasured.
