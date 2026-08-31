# ATOMCompass — design notes and open problems

What is settled and why, and what is not. Recorded as each was established, so
the reasoning survives the context it was found in, including the things that
were wrong first.

The target is to simulate ATOM **as it is deployed**: compilation on
(`--level 3`), CUDA graphs on, prefix caching on, chunked prefill on. Anything
that only works under `--enforce-eager` or `--level 0` is a step towards that,
not an instance of it. `--enforce-eager` does **not** disable compilation — it
disables CUDA graphs; compilation is `--level`, and its default is on.

Effort estimates are rough: **S** hours, **M** days, **L** weeks or unknown.

---

# Status

A simulated serving run predicts a real one on Qwen3-0.6B on MI308X, calibrated
on a sweep of shapes and evaluated on a workload neither saw, so the error is a
generalisation error rather than a fit residual. Five runs of the identical
command at TP=1, at roughly 6× wall clock:

| run | TTFT | TPOT | latency |
| --- | --- | --- | --- |
| 1 | −6.1% | −1.8% | −3.3% |
| 2 | −14.0% | −5.4% | −8.7% |
| 3 | −14.7% | −5.3% | −8.8% |
| 4 | −14.4% | −1.9% | −6.7% |
| 5 | −0.5% | +2.4% | +1.4% |
| **mean ± sd** | **−9.9 ± 6.4** | **−2.4 ± 3.2** | **−5.2 ± 4.3** |

Reproduce with `scripts/compass/validate.py`.

**Read the spread before the mean.** This document reported run 1 alone for some
time, as −6.1/−1.8/−3.3, and treated it as the project's result. It is one draw
from a distribution whose TPOT runs from −5.4% to +2.4% — the reported error was
smaller than the interval it came from. Nothing was wrong with that run; what was
wrong was quoting it. The machine is shared with about twenty other containers
and the GPUs are not partitioned, so a run is a sample of the box as much as of
the model.

The same comparison at **TP=2**, on the same model and the same hardware.
One run, so read it with the caveat above — and note that calling it
significant below borrows the TP=1 spread as an estimate of this one's,
which is an assumption and not a measurement:

| metric | real | modelled | error |
| --- | --- | --- | --- |
| TTFT | 69.10 ms | 46.80 ms | **−32.3%** |
| TPOT | 3.54 ms | 3.06 ms | **−13.6%** |
| latency | 178.90 ms | 141.71 ms | **−20.8%** |

TP=2 is worse than TP=1 by more than the noise: −13.6% TPOT sits 3.5 standard
deviations below the TP=1 mean, so it is a real effect and not another draw. Two
things are under it. The linear fit under-predicts the region both evaluations
land in, at both widths — that is item 10, and it is in-sample. And run 1 of the
TP=1 series happened to land at the optimistic end of the band, which made the
gap between the two widths look larger than the −2.4% mean justifies.

How it got there, because the shape of the sequence matters more than the
endpoint:

| iteration | TTFT | TPOT | latency |
| --- | --- | --- | --- |
| first end-to-end run | +107% | +800% | +560% |
| CUDA graphs captured in measure mode | +39.6% | +13.1% | +22.9% |
| prefill calibration widened | −23.9% | +22.5% | +4.7% |
| batch calibration widened | −11.9% | +17.5% | +7.1% |
| timing switched to CUDA events | −6.1% | −1.8% | −3.3% |
| ...the same command, four more times | −0.5 to −14.7% | +2.4 to −5.4% | +1.4 to −8.8% |
| the same, at TP=2 | −32.3% | −13.6% | −20.8% |

Every step up to the last was a defect in how Compass measured or calibrated,
never in the cost model's form; three of the four were configurations or methods
that looked entirely reasonable and were quietly wrong. **The last row is the
first exception.** Widening to a second rank did not introduce an error so much
as stop concealing one: the linear fit cannot reproduce its own training data in
the region both evaluations land in, by −3.8% at TP=1 and −8.3% at TP=2, and at
TP=1 that was cancelled by run-to-run variation pointing the other way.

---

# Open problems

Ranked by what is known about their size. Anything without a measured magnitude
is ranked by argument, which is weaker — the whole reason the end-to-end
comparison was built first is that structural findings cannot rank each other.

## The product gap

### 1. No op-graph work feeds the cost model — **L**

Both oracles predict from a step's **shape** alone: token counts, batch size,
context lengths. The graphs, the meta derivation, the TP validation — none of it
contributes to a single prediction.

This is the difference between the F1 "calibrated" point, which is where the
project is, and "empirical", which is per-operator cost attributed from a
captured graph. It also decides whether Compass can predict a configuration
nobody has measured, or only interpolate between ones that were. Everything
under *Settled* about graphs is groundwork for this and is not yet paying for
itself.

Precondition, from the event-timing finding below: per-operator attribution must
be checked for the same observer effect — run the workload with and without it
and compare the engine's own numbers.

### 2. Most of this project's numbers are inside their own noise — **S**, done

Five runs of one command, no change between them, gave a TPOT error ranging from
−5.4% to +2.4% and a TTFT error from −14.7% to −0.5% — see the spread in Status.
Mean ± sd is −2.4 ± 3.2 (TPOT) and −9.9 ± 6.4 (TTFT): **the standard deviation is
larger than the mean in both**. At a fixed shape, without the calibration and
scheduling variance on top, repeats still move ±3–4% and the sign is not stable.

Every row of the trajectory table is therefore a difference of two single draws.
The early rows are large enough to survive it — +800% is not noise — but the last
two steps, from +17.5% to −1.8% TPOT, are not distinguishable from a lucky pair
by the evidence that was recorded for them.

Some of the spread is the box: about twenty containers share these GPUs and they
are not partitioned, so a run samples the neighbours as much as the model. That
is not a defect to fix, it is a condition to measure under.

And the variance has **time structure**, so repeats are not independent draws. A
`--repeats 2` run taken back-to-back reported −2.2% ± 0.1 on TPOT — a hundredth
of the spread the five separated runs showed. Adjacent runs share whatever the
machine was doing, so a tight sd from consecutive repeats is a lower bound and
not a measurement. Spacing repeats, or interleaving the configurations being
compared, would be the honest design; the script says so rather than pretending
otherwise.

`validate.py --repeats N` now runs the whole pipeline N times and reports mean,
sd, range and each draw, and says so plainly when the spread exceeds the mean.
The remaining work is judgement, not code: nothing should be quoted as an
accuracy figure from a single run again, and a difference smaller than ~5% needs
repeats before it means anything.

### 3. No collective cost model — **S/M**, and narrower than it looked

Collectives are recorded with their group and their bytes, which is what a cost
model needs, and nothing consumes it.

It does **not** block multi-GPU prediction the way this once claimed. The
collective runs inside the forward, and the forward is timed with CUDA events, so
a calibrated oracle at a width it was measured at has already absorbed the
collective's cost without naming it. The TP=2 error above is not a missing
collective — it is item 10.

What the gap actually costs is prediction at a width **nobody measured**:
calibrate at TP=2 and ask about TP=4, and there is nothing to scale. That is a
narrower and later problem than "blocks any credible multi-GPU prediction".

### 4. An HTTP benchmark cannot measure a simulated engine — **L**

Attempted, and the result is structural rather than numerical. The server takes
the Compass flags for free (it builds `EngineArgs` like everything else), so
predict mode serves over HTTP with no new plumbing. Qwen3-0.6B, TP=1, random
256-in/64-out:

| | real | predict | |
| --- | --- | --- | --- |
| **32 requests at 4/s** | | | |
| duration | 9.54 s | 9.55 s | arrival-bound |
| TTFT | 48.5 ms | 11.2 ms | |
| TPOT | 3.27 ms | 0.32 ms | |
| concurrency | 0.85 | 0.11 | |
| **64 requests, unpaced** | | | |
| duration | 0.71 s | 0.22 s | server-bound |
| TTFT | 371.9 ms | 89.5 ms | |
| TPOT | 5.13 ms | 1.78 ms | |
| output tok/s | 5 795 | 18 943 | |

**Nothing in the predict column is a prediction.** `benchmark_serving` times
requests with `perf_counter` around an HTTP stream and divides every metric by
its own wall-clock duration, so it measures the process that produced the
tokens. Predict mode skips the forward, so what it reports is how fast the
simulator ran. The oracle's estimate influences the run — it advances the
virtual clock, which the scheduler reads — but it is never reported to anybody.

Three separate defects sit under that, and only the first is about measurement.

**The simulated numbers are computed nowhere in serving.** Offline, `postprocess`
derives per-request TTFT and TPOT from `arrive_time` and `first_token_time`,
which under a virtual clock are the simulated values — that is what `run.py`
compares and what every number in Status rests on. `api_server.py` never calls
`postprocess` and never touches `first_token_time`. So in serving the quantity
this project exists to produce is not merely unexported, it is never calculated.

**Arrival is stamped on a clock that does not move, and it costs everything.**
`seq.arrive_time = get_clock().time()` in the engine process, and that process
installs a `VirtualClock` which by design never advances — correct for an offline
batch submitted at once, wrong for a server. Measured over 65 requests:

| | real | simulated |
| --- | --- | --- |
| arrival spread | 310.6 ms | **0.000 ms** |
| distinct arrival stamps | 65 | **1** |
| mean TTFT | 349.9 ms | 643.2 ms |

Every request is stamped as having arrived at the same instant, so its TTFT is
measured from the start of the run rather than from when it turned up. The TTFT
gap is 293 ms and the collapsed arrival spread is 311 ms: **the whole of the
engine-side TTFT error is this artifact**, and none of it is the cost model. The
cost model's own errors here point the other way -- decode at batch 63 predicts
2.47 ms against a real 3.72 ms.

Worth stating plainly, because it is the second time on this page that a headline
error turned out not to be about the thing being modelled. A simulator returns a
number for every request whether or not the question it answers is the one that
was asked.

**A paced workload cannot tell the two apart.** At 4 requests/s the duration and
throughput matched to within 0.1% — not accuracy, but the client pacing both
runs: 32 requests at 4/s takes 8 s regardless of the server. Only the unpaced run
separated them. Any future serving comparison has to saturate, or it measures the
client. Concurrency looked preserved in the unpaced run (62.9 against 59.6) for
the same hollow reason — with no arrival process there is no queue behaviour left
to get wrong.

What *did* work: the extrapolation guard fired correctly through the server,
catching 16-token prefill steps below the calibrated floor of 64. Chunked prefill
produces shapes the offline sweep never generated, and the safeguard saw them.

**What was built, and what it settled.** Both halves of the virtual-clock route
are now in place. The engine's arrival, first-token and finish readings ride out
on `RequestOutput` and are served by `GET /compass/requests`, reporting which
clock they came from. Requests carry a declared arrival (`compass_arrival`, an
offset into the run), the scheduler defers one whose time has not come, and when
nothing is runnable the clock jumps to the next arrival — the discrete-event
step, and the reason a simulation can be faster than the thing it simulates.
`scripts/compass/serve_bench.py` computes one arrival schedule and applies it two
ways: paced in real time against a real server, declared against a simulated one.

64 requests, Poisson at 8/s, Qwen3-0.6B TP=1:

| | real | simulated |
| --- | --- | --- |
| TTFT median | 35.84 ms | **35.85 ms** |
| TTFT mean | 42.23 ms | 213.50 ms |
| TTFT max | 82.10 ms | 1653.09 ms |
| latency median | 262.35 ms | 306.40 ms |
| wall clock | 9 417 ms | 495 ms |

The median is exact and the mean is not, and the residual is one specific thing:
the first 13 requests share **two** distinct first-token instants where the real
run has 13. They were held and released in two batches.

**Which turns out to be a causality constraint, not a bug.** The client submits
concurrently, so requests reach the engine in a different order from the one they
were declared in. If the first request *received* is declared for t=0.9 s, the
idle engine jumps virtual time to 0.9 s — and a request declared for t=0 that
lands on the socket a moment later is retroactively late. A discrete-event clock
may only advance when it knows no earlier event will still turn up, and an HTTP
client submitting concurrently cannot promise that. Serialising submission would
fix the order and reintroduce the wall-clock pacing the whole design exists to
escape.

So the remaining step is forced, and it is the one the routes below already
pointed at: **the engine has to be given the whole arrival schedule up front.**
Then it can advance to the next arrival knowing what the next arrival is. The
workload becomes an input file, submission stops being an event, and HTTP becomes
a way to fetch results rather than the thing being timed.

That also reframes the original goal. "Validate against `benchmark_serving`" is
partly mis-specified: an HTTP benchmark is the right yardstick for the *real*
engine and cannot drive a simulated one. The three routes, for the record:

* *Export the simulated metrics* — done, and necessary, but measurement only.
* *Virtual arrival* — half done: declared arrivals and a clock that skips idle
  time work; the schedule must move from per-request to up-front.
* *Real-time pacing* — sleep the difference between virtual and wall time. HTTP
  would work unchanged and queueing would be right, but the speedup that
  motivates the project is gone, and only systems slower than real time could be
  simulated. Rejected.

## Graph fidelity against a production launch

### 5. Derivation is uncompiled; production is not — **L**

Derivation runs the model eagerly on meta. At `--level 3` inductor fuses
operators and eliminates views and allocations, so a derived graph and a captured
one differ by construction at the default level: 386 operators against 330 on
Qwen3-0.6B. Neither is wrong; they cannot be compared, and the sweep story
depends on comparing them.

Two routes, neither cheap. Run dynamo and inductor during derivation, which means
compiling for a device the derivation does not have. Or model the fusion —
predict which operators collapse into one kernel and what that kernel costs —
which is a research problem, not an implementation one.

Interim: derive and capture at matched levels, and be explicit that a level-0
validation does not transfer to level 3.

### 6. Trace mode does not observe the CUDA-graph path — **M**

Trace mode skips `capture_cudagraph` deliberately, so its graph is the eager
operator sequence. A replay is a single opaque submission: there are no
per-operator events to record, even in principle. So the operators must come from
eager execution and the replay's cost must be carried as a term the oracle
applies, not as operators in the graph.

Measure mode does capture, so timings already come from the replayed path. What
is missing is the bridge between the two artifacts.

### 7. Only decode steps are traced, and only one — **M**

`trace_step` records exactly one step, in practice a small decode. A deployment's
cost is dominated by shapes never captured: prefill, chunked prefill, mixed
prefill/decode batches, and long-context decode where attention stops being
cheap. Speculative decoding and MTP are entirely untraced.

### 8. A custom op's inner Triton kernel is recorded only on hardware — **S**

On a real device `aiter::masked_embedding` both dispatches and launches
`triton::_masked_embedding_kernel`, so the capture holds both. On meta the kernel
never launches, so derivation holds only the outer operator. Harmless for cost —
the outer operator carries the shapes — but a systematic difference between the
two graphs that will confuse anyone diffing them.

### 9. Simulated TP's `all_gather` is not the real one — **S**

At a physical width of one it builds a zero-padded buffer (`movedim`, `reshape`,
`view`, `zeros`) where the real path uses `view.dtype`. Seven operators out of
451. Possibly worth simply accepting — but as a decision, not a residue nobody
looked at.

### 10. Decode cost falls as batch grows; the model says it rises — **M**

Not "not linear". The **sign is wrong**. At matched total context, mean cost per
step over a sweep of 1662 decode steps:

| batch | 1 | 2 | 4 | 8 | 12 |
| --- | --- | --- | --- | --- | --- |
| TP=1 | 3.640 ms | 3.564 | 3.393 | 3.317 | 3.532 |
| TP=2 | 3.636 ms | 3.563 | 3.403 | 3.351 | 3.310 |

Twelve sequences cost *less* per step than one, because with CUDA graphs the
replay runs a padded batch-size bucket and the step is bound by launch and replay
rather than by the work inside it. Per-sequence cost falls more than twelvefold
across that range, and at TP=1 the curve is not even monotonic. The decode model carries batch size as a positive linear term.

The consequence is in-sample, which is what makes it a statement about the
model's form rather than about coverage. Asked about 49 of **its own training
rows** at batch 8 and short context:

| | measured | linear fit | k-NN |
| --- | --- | --- | --- |
| TP=1 | 3.282 ms | −3.8% | +0.0% |
| TP=2 | 3.303 ms | −8.3% | +0.0% |

(k-NN scoring zero on training rows is memorisation and proves nothing about
k-NN. The linear column is the finding.) Held out on an evaluation workload the
same contrast survives: −10.3% against −3.4% at TP=2.

This item was **demoted once already**, on the grounds that a nearest-neighbour
oracle assuming no functional form agreed with the linear fit to within 0.3%.
That agreement was over a whole workload; it did not hold in the region the
evaluation actually occupies, and averaging concealed it. Two methods agreeing is
evidence only where they were both asked the same question — and "the same
question" has to mean the same *region*, not the same table.

Featuring the padded bucket is the obvious repair; `Config.capture_sizes`
declares the ladder. Whether a per-bucket constant is enough, or the KV term
needs to stop being linear too, is not yet known.

## Configurations that cannot be modelled at all

### 11. Asymmetric parallelism — **L**

`simulated_tp.py`'s `_reject_unsupported` refuses pipeline parallel, prefill and
decode context parallel, data parallel and DP-attention, TBO, EPLB, and
disaggregated prefill, because ranks diverge and absent ranks cannot be faked.
Compass inherits every one of those limits, and the line it draws is exactly
where this project drew it independently: TP is symmetric and simulable on fewer
devices, the rest are not, because virtual time would have to be coordinated
across processes.

Nothing in the field has shipped a solution — Revati claims a multi-process
timekeeper and released no code, LLM-Emu avoids the problem by hooking a
single-process executor. Expert parallelism matters most: it is where the
interesting models are going, and where a rank's graph genuinely depends on which
experts its tokens chose.

### 12. Shape-changing collectives have no meta stand-in — **S/M**

`all_reduce` and `broadcast` preserve shape, so meta can hand back the input.
`all_gather` grows and `reduce_scatter` shrinks, so they raise rather than guess
— deliberately, since assuming "same shape" would corrupt every downstream shape
while appearing to work. TP alone does not need them; sequence and expert
parallelism do.

### 13. One communication group is resolvable; several are not — **M**

A collective names its group by elimination: with exactly one group of size above
one, there is nothing else it could have run on. With TP and EP together the
ambiguity is real and is recorded as `"?"`.

Settling it means intercepting at the group object rather than the dispatcher —
`get_tp_group()` and its siblings know their own identity. That is a replacement
for the current resolver, not an addition, and item 11 needs it too.

### 14. Qwen3.8-27B cannot be captured here — **?** (environmental)

Its decode path JIT-builds AITER's *gluon* paged-attention kernel and the build
fails:

    subprocess.CalledProcessError: Command '['make', 'build', '-j1']'
    returned non-zero exit status 2

Narrower than it first appeared: Qwen3-0.6B takes the **ASM** decode path, which
ships as a prebuilt `.co`, so it captures without touching the failing build. One
kernel path, not capture as a mechanism — which is why validation was possible on
a smaller model while the stated target model stays blocked. Needs the `make`
stderr captured to tell a toolchain problem from an image defect.

## Measurement and performance

### 15. The extrapolation warning is per-feature, so it cannot see a hole — **S**

The oracle warns when a query falls outside the range of a feature it was
calibrated over, and that safeguard has caught two real errors. It checks each
feature *separately*, so it is blind to a gap in their joint distribution.

The TP=2 evaluation asked about batch 8 with a total context of ~2650. Batch 8
was covered (1–16 seen) and 2650 was covered (57–41052 seen), so nothing warned.
The sweep's batch-8 steps actually run 1776–2288 and then jump to 5360: the query
sits in a hole, and the guard reported it as interpolation.

Not the cause of the TP=2 error — item 10 is, and it is in-sample — but the guard
claims a property it does not have. A convex hull, or a nearest-neighbour
distance with a threshold, would say what the bounding box cannot. The k-NN
oracle already computes the distance this needs.

### 16. No way to tell the runner when to start measuring — **M**

Throwaway warmup requests were served to get Triton autotuning out of the way,
and the runner measured them anyway: a runner sees steps and has no idea which
request a step belongs to, or that some were meant to be ignored. The result was
a 7.5 s prefill sample sitting in a table beside a 0.03 s one.

Worked around by discarding the first step of each kind, which is blunt — it
drops a real sample when a workload has few and keeps warmup steps when it has
many. What is missing is a measurement window the engine can open and close
across the process boundary. The same gap makes it hard to calibrate one phase of
a deployment without the others polluting the table.

### 17. Derivation is too slow to run per step — **M**

A full meta forward of Qwen3.8-27B (64 layers, 2999 operators) takes **~0.16 s**
against a modelled decode step of ~1 ms, so deriving inside the serving loop
would dominate what it measures.

The graph only changes when the batch shape changes, so it should be derived once
per distinct shape and reused, keyed on the **exact** `GraphKey` rather than a
quantised bucket. Bucketing trades away precisely the batch-skew sensitivity this
design keeps: a decode batch mixing short and long histories does not cost what
its mean history suggests.

Open whether an exact-key cache hits often enough. Decode at a stable batch size
should repeat shapes constantly; prefill with varied prompt lengths will not. If
the miss rate is high, the options are incremental re-derivation for the part of
the batch that changed, or a bucketing scheme with a **measured** error budget
rather than an assumed one. Not on the critical path for correctness, only for
speed.

### 18. Capture pays for Triton autotuning — **S**

The first launch of each kernel benchmarks every candidate configuration, which is
why `trace_step` is never 1 and why capture takes minutes. Fine as it is; worth
knowing before anyone tries to capture a sweep.

---

# Settled

## Tracing

### Triton kernels need interception, not shape inference

Triton launches bypass the dispatcher, so no meta kernel can stand in for them,
and on meta they fail outright: there is no storage behind the pointers.

They do not need shape inference. AITER's Triton kernels take their destination
as an argument — `run_pa_decode_gluon(output, q, k_cache, ...)` — so the caller
has already allocated every output with the right shape before the launch.
Tracing one is therefore: record what it was asked to do, and skip it.

Worth re-checking if a kernel ever traces wrongly: the assumption that every
intercepted kernel is out-parameter style. One that allocates internally and
returns a tensor would need real handling.

### Inductor kernels launch through a different path

`torch.compile` does not use `JITFunction.run`; it calls generated kernels
through `CachingAutotuner`, which is intercepted alongside it. Without that a
compiled capture silently omits whatever inductor generated.

An earlier version of this note claimed compilation dropped 57 of 386 operators.
That was wrong in substance, and the correction matters because the wrong version
argued for capturing at `--level 0`, which nobody deploys. Inductor accounted for
exactly **one**: `aten::embedding` becomes `inductor::triton_poi_fused_embedding_0`.
The other 56 are `split_with_sizes` and `empty` — views and allocations, which
inductor resolves into offsets and a buffer plan rather than executing. They do
not run at level 3. Compute totals confirm it: 283 operators either way.

So a compiled and an uncompiled capture are both correct and describe different
configurations. Compilation is the default, so it is the one worth modelling, and
a derivation may only be compared against a capture at its own level.

### Never trace the first forward — Triton autotunes on it

Tracing step one recorded **90,838** operators. Step two recorded **101**. On a
kernel's first launch Triton benchmarks every candidate configuration, so step one
contains tens of thousands of launches steady-state serving never performs —
`chunk_fwd_kernel_o` alone appeared 34,269 times.

`trace_step` therefore defaults to 2. Anything that captures a graph, times a
step, or calibrates a cost model has to step past warmup first, and a tool that
quietly recorded step one would produce numbers that look precise and describe
nothing that happens in production.

Meta never revealed this, because skipped launches never autotune — a case where
hardware told us something derivation could not, which is an argument for keeping
the comparison even after derivation is trusted.

### A failed forward writes no graph

A 101-operator capture of a 64-layer model looked like truncation. The forward had
**crashed**, and the recording was written anyway from a `finally` block: a
well-formed artifact from a failed run, structurally valid and merely wrong, which
would have costed out at a fraction of the model.

Now a failed forward writes nothing and says so, and what does get written is
checked against the model's depth — attention runs once per layer, so a graph with
fewer attention operators than the model has layers is reported as truncated.

The lesson generalises: for a tool whose output is an artifact, a plausible
artifact from a broken run is worse than a crash, because it propagates silently
into everything fitted against it.

### The meta-kernel worklist is empty for this model

Every AITER operator Qwen3.8-27B reaches already runs on meta. An earlier estimate
of "22 meta kernels to write" was wrong in both directions: fewer are missing, and
the real barrier was a different mechanism entirely (Triton interception).

Model-specific. AITER registers operators lazily through JIT, so the worklist for
another architecture — a MoE model especially — has to be discovered by running
the probe, not predicted.

## Derivation and comparison

### Derivation reproduces hardware, and the check is repeatable

For Qwen3-0.6B at TP=1 on a one-token decode, all **338** derived operators appear
in the capture in order, with identical shapes *and* dtypes. At TP=2, all **395**
do, including all 57 all-reduces.

`graph_diff.py compare` asks **containment, not equality**. A positional
diff is right between two graphs of the same kind and wrong here: a derivation is
the model body, a capture is the body plus the runner around it, so compared
position by position they disagree from the first operator while in fact agreeing
about everything a cost model is fitted to. Matching is greedy and fails closed —
on divergence it reports every later derived operator as unfound rather than
hunting for a match that might be coincidence.

Two conditions must hold, both enforced rather than remembered: the two sides must
be at the same compilation level, and must describe the same batch. The tool warns
when the batches disagree.

### Derivation must use the engine's input dtypes

`input_ids` is `int32` and `positions` is `int64` (`model_runner.py`, lines 189
and 1277). Easy to get wrong in the same way, and the consequence is out of all
proportion: PyTorch's `int64` default diverges at the embedding, `int32` for both
diverges at the first attention operator. Because matching fails closed, either
mistake rejects every operator from that point on — a correct derivation reporting
as a total structural disagreement. Both live in `derived_inputs()` now.

### Tracing goes through the runner, not the model

Driving `model(input_ids, positions)` directly traces *a* forward, but not the one
ATOM would run. On hardware it fails at `fwd_ctx.context.is_dummy_run`: attention
reads a forward context only the runner establishes. Building that by hand means
reimplementing `prepare_inputs`, which is the re-implementation this design exists
to avoid. So capture enters through `ModelRunner.forward`.

Derivation still uses a bare model call, which is why the comparison asks
containment. That is a known limitation rather than a defect — see the division of
labour below.

A related trap: built at the default fp32 rather than the model's `torch_dtype`,
meta traces happily while hardware refuses — AITER's fused qk-rmsnorm takes
fp16/bf16 only. Meta accepts kernels real devices reject, so dtype is pinned
deliberately on both sides. The diff caught this, which is some evidence it earns
its keep.

## Parallelism

### The op graph knows world sizes and ranks, nothing else

A collective names the communication group it ran on; the shapes around it already
reflect whatever sharding produced them. That is what lets one representation
serve every parallel strategy, including combinations, without Compass being
taught what any of them mean.

Collectives were being recorded with the literal group `"unknown"`, which hollowed
that out — a graph in which every collective is indistinguishable cannot tell an
all-reduce over tensor ranks from one over expert ranks. Resolved by elimination
now where the rank belongs to a single non-trivial group; genuine ambiguity is
recorded as `"?"` rather than guessed.

### TP=2 shows the abstraction working

Captured at TP=2, with nothing in Compass that knows what tensor parallelism is:

* the qkv GEMM narrows from `[4096, 1024]` to `[2048, 1024]` — sharding visible in
  the shapes, exactly as intended
* 57 `all_reduce_` collectives appear where TP=1 has none
* the embedding switches to `aiter::masked_embedding` plus its Triton kernel, the
  vocab-parallel path
* the two ranks' graphs are identical, correct for symmetric TP — which is why
  rank attribution comes from the filename and key rather than the contents

### A sharded rank can be derived from one process

TP>1 derivation used to hang, and only on the derivation side: capture at TP=2
works because it really starts two processes, while one process asking gloo for a
world of two waits forever for a peer nobody launched.

Fixed with ATOM's own simulated TP, which reports a logical group width over a
smaller real group. One correction was needed: `apply_simulated_tp` takes the
physical width from `torch.cuda.device_count()`, right for a worker holding a
device and wrong for derivation, whose real group is one rank however many GPUs
exist. On an 8-GPU box it concluded a TP2 derivation needed no simulation, and the
model built **unsharded, silently**. `simulate_group_width()` states the widths
outright.

That surfaced a second problem. At a physical width of one there is no
communicator, so simulated TP replaces `all_reduce` with a passthrough — right for
benchmarking kernels, wrong here, because the derived graph then holds no
communication and tensor parallelism costs out as free. Collectives are recorded
rather than performed, under the name the dispatcher uses on hardware.

### The whole step, at any TP width, on one device

Compass models the serving flow, not the model body, so the runner's own work —
batch-metadata preparation, the LM head, sampling, the transfer home — is inside
the scope. Calling those "not part of the model body" excused the gap rather than
closing it.

The route that closes it is not a meta runner (62 `torch.cuda.` sites in
`model_runner.py`). It is capture under simulated TP: the runner runs for real, so
everything it does is recorded, while `--fake-eplb` makes one device stand in for
a TP-N deployment. A TP2 capture on a single GPU gives **451** operators against
**448** for the real two-GPU capture.

The residual seven are simulated TP's own zero-padded `all_gather` — item 9.

**The division of labour this settles.** *Capture* on one device gives the
complete step for any symmetric TP width, and is what a cost model should be
fitted against. *Derivation* on meta gives the model body with no device at all,
about a thousand times faster, and is what a sweep should use once the two are
known to agree.

### Each rank writes its own graph

Every rank traces and under any parallelism their graphs differ — that difference
is what records how the model is sharded. A single `--compass-graph-out` made the
ranks race for it and left one file naming no rank: the survivor could not be
attributed and the rest were lost. Paths carry the rank's coordinates in every
group it belongs to (`g.json` → `g.tp1.json`, or `g.dp3-tp1.json`).

### ...and must read it back the same way

Writing per-rank artifacts is only half a convention. The measure side suffixed
the timing table whenever any group was wider than one; the predicting side asked
for the path it was given. At TP=2 that meant a calibration wrote
`steps.tp0.jsonl` and `steps.tp1.jsonl` and the run fitted against them looked
for `steps.jsonl`.

**A single rank cannot expose this.** At width one no suffix is applied, so the
two sides agree by accident, and every test and every validation run to that
point had been at width one.

What reached the terminal was worse than the bug: both workers died on a bare
`FileNotFoundError`, and the engine manager reported *"Received unexpected
SHUTDOWN signal from DP rank 0 during initialization"* — no DP in the run, and
nothing to do with initialization. The same shape as the `capture_cudagraph`
unpacking hang: a worker dies of a specific, nameable cause and the parent
announces something else entirely.

The convention now lives in `atom/compass/core/artifacts.py` and both sides go
through it. A rank prefers its own file, falls back to a shared one with a
warning, and a missing table raises a message that names the path it wanted and
how to produce one — because that message is the only thing that escapes a
worker process.

### Ranks of a symmetric group are interchangeable, measured

The engine core owns the clock and advances it by the duration the forward
reported; that reply comes from rank 0's output channel, so every other rank's
prediction is discarded. Whether that is sound had never been checked.

Over 1727 steps at TP=2, rank 1 against rank 0: identical shapes on every step,
median difference **0.03%**, worst **0.82%**, and rank 1 was the slower one on
**51%** of steps — a coin flip, so there is no systematic skew to correct for.
Rank-0-only accounting discards 0.01% of the total.

So single-sourcing the clock on rank 0 is right for symmetric TP, and the concern
belongs entirely to the asymmetric strategies of item 11, where ranks genuinely
diverge and no rank stands in for another.

### No fixed ports

The rendezvous store was hardcoded to a port. The container runs with host
networking on a machine shared with about twenty others, so a fixed number
collides with whatever holds it — including an earlier run of the same script —
and fails with an `EADDRINUSE` that says nothing about tracing. The OS picks it.

## Measurement

### The step period is the forward, and TP does not change that

A simulated run advances its clock by the predicted forward alone, so everything
the engine does *between* forwards — broadcasting the batch to each worker,
collecting output, scheduling, detokenising — is implicitly zero. With two
workers instead of one there is more of it, which made a tidy explanation for why
TP=2 predicted 13.6% fast.

It is wrong. Measure mode reports the real TPOT and records each forward's device
time in the same run, so the difference between them is exactly that unmodelled
remainder:

| | real TPOT | forward | remainder |
| --- | --- | --- | --- |
| TP=1 | 3.139 ms | 3.150 ms | −0.4% |
| TP=2 | 3.387 ms | 3.414 ms | −0.8% |

Nil at both widths, and not growing. For offline batch serving the step period
*is* the forward, so modelling the forward is not the approximation it looked
like. Whether that survives real request arrival over HTTP — where queueing,
admission and detokenisation are not overlapped with a saturated engine — is
item 4 and is untested.

Worth recording as a refutation rather than deleting: the hypothesis was
plausible, cheap to test, and wrong, and the test is two runs.

### The instrument was changing what it measured — a 33% error

`measure` timed each forward between two `torch.cuda.synchronize()` calls. Same
workload, same engine:

| | TPOT |
| --- | --- |
| plain run | **3.26 ms** |
| under measure (host sync) | **4.33 ms** |

A serving loop overlaps host and device — while the device runs one step the host
prepares the next — and a sync on every step destroys that overlap. So what was
recorded was each step's *isolated latency*, and a model fitted to isolated
latencies then predicts a pipelined run and over-estimates it by whatever the
overlap was worth. That was most of the residual TPOT error.

Timed with CUDA events now, recorded on the stream and read back later, drained by
`query()` rather than `synchronize()` so the host never waits. Perturbation fell
from 1.33× to 0.97×, recorded decode mean from 4.05 ms to 3.43 ms against a real
3.26 ms.

Found by disagreement between two methods that should have disagreed: a
nearest-neighbour oracle assuming no functional form predicted 3.84 ms where the
linear fit predicted 3.83 ms. Two methods with opposite failure modes agreeing
meant the fit was faithful to its data and the data was wrong.

Generalises: **a profiler that serialises what it profiles measures a machine that
only exists while being profiled.**

### CUDA graphs must be captured when the forward is real

`capture_cudagraph` and `warmup_model` were stubbed unconditionally, on the
reasoning that neither means anything when no kernels run. True for `predict`,
false for the two modes that perform the real forward — where skipping them skips
them from a *real* run.

That single assumption produced a **+800%** TPOT error: measure ran eager while
production replayed a graph. Measured directly, Qwen3-0.6B decode: **28.78 ms**
eager against **3.24 ms** replayed, 8.9×.

Now `measure` captures for real, `predict` skips having nothing to capture, and
`trace` skips for the opposite reason to `predict` — a replay is one opaque
submission, so a traced step would record no operators at all.

### A stub must honour its caller's return contract

`engine_core` calls `capture_cudagraph` across the worker boundary with
`wait_out=True` and unpacks three values. The override returned `None`, which
killed the worker on an unpacking error while the parent was still waiting — so
the failure surfaced as a hang on a shared-memory broadcast that never arrived,
naming neither CUDA graphs nor Compass.

Across a process boundary a breached contract becomes a hang, not a traceback.

### A subclass's `__init__` body runs too late

`ModelRunner.__init__` warms the model up before returning, and warmup drives a
forward, so anything mode-dependent is consulted before `CompassModelRunner`'s own
`__init__` has assigned its config. `_compass_config` resolves lazily from
`self.config`, which the base sets well before warmup.

Together with the two above: **a subclass that stubs out work the base class
depends on is wrong in proportion to how much of the base class it never lets
run.**

### Warmup batches are steps, and must not be counted

`warmup_model` drives synthetic batches through `forward` with `is_dummy_run=True`.
They must run — they are what autotunes Triton — but they are not steps a
deployment performs, and counted they spend the trace budget on a dummy shape and
put dummy rows in the table a cost model is fitted to.

### A real run must not use the virtual clock

`trace` and `measure` perform the real forward, so the wall clock is the truthful
one. The virtual clock was installed for any Compass run, so every request came
back with a TTFT of zero — which reads as a broken measurement rather than a
misconfigured clock.

## Calibration

### Coverage must bracket every dimension the model uses

The same error twice, in two dimensions, an hour apart.

ATOM batches several requests' prefill into one step, so a sweep of large prompts
produced seven samples spanning 1753–16370 tokens while the evaluation batched to
~520. The model extrapolated below everything it had seen, where the intercept
dominates and the slope does no work, and predicted 66.8 ms for a 64-token
prefill.

Widening the prompt lengths fixed that and left the sweep varying *only* length,
so it produced decode steps at batch sizes 1–4 — and the evaluation ran 8
concurrent requests. TTFT improved and TPOT got worse, from +13.1% to +22.5%.

Both times coverage was designed against the dimension that had just caused a
problem rather than against the model's feature set. The sweep now varies length
and concurrency together.

The durable fix is not a better sweep: **the oracle says when it is
extrapolating.** A fitted model answers anything, confidently, including questions
its evidence does not cover, and neither of these errors announced itself. Warned
once per kind and direction, since a serving run asks thousands of times.

### Warmup counted in steps destroys the data it protects

`measure_warmup_steps` defaulted to 2, and prefill happens twice in a whole run,
so it discarded every prefill sample. The oracle had nothing to fit, fell back to a
mean of zero samples, and predicted a TTFT of 0 ms against a real 7.6 s. Counted
per kind now, defaulting to zero.

### An oracle asked outside its evidence must refuse

The fallback for "no samples of this kind" was the mean of an empty list, which is
zero: a confident, precise, entirely fictional answer. It raises now.

### The fit resists one-off contamination

**Triton autotunes per shape, not once per process**, so a calibration sweep built
from deliberately varied shapes pays a benchmarking cost on many of its own
samples — one prefill row sat at 0.13 s where its neighbour at a larger size took
0.036 s.

Least squares, then discard points whose residual is far outside the spread of the
rest, then refit. Spread by median absolute deviation, since the contaminating
points would otherwise inflate the very quantity used to detect them. The number
dropped is reported in `describe()`: a fit that discarded half its evidence should
not describe itself like one that kept it.

### Prefill and decode are fitted separately

They are not two regimes of one function. Prefill is compute-bound in new tokens;
decode is bandwidth-bound in the KV history it must read. Fitting them together
produces a model that is wrong about both.

Total context is summed across the batch rather than averaged: a decode batch
mixing short and long histories does not cost what its mean history suggests, and
the sum is what the hardware actually moves.

---

# Process notes

* Every defect in this project surfaced by **running** something, never by reading
  code. The meta-kernel worklist, the autotuning blowup, the passthrough
  all-reduce, the unpacking hang, the observer effect — all of them.
* Validating against a convenient configuration (`--enforce-eager`, `--level 0`,
  one GPU) reads as validation and is not. Three separate bugs hid behind that.
* Structural validation cannot rank its own findings. Nothing here had a
  defensible priority until something predicted time and was wrong by a measurable
  amount.
* A fitted model is confident everywhere, including outside its evidence. Twice the
  error was not the fit but the range it was fitted over, and neither time did
  anything complain.
* A profiler that serialises what it profiles measures a machine that only exists
  while being profiled.
* For a tool whose output is an artifact, a plausible artifact from a broken run is
  worse than a crash.
* Two methods agreeing is only evidence when they could have disagreed. The
  nearest-neighbour oracle earned its place by matching the linear fit, not by
  beating it. But they have to be asked the same *question*, and a whole-workload
  average is not one question: the two agreed to within 0.3% over a table while
  disagreeing by 7 points in the region that decided the answer. An agreement
  computed over an average can only rule out disagreements that survive
  averaging.
* A good end-to-end number can be two errors cancelling. The TP=1 result stood
  for weeks as evidence the cost model was sound; widening to a second rank
  showed the fit had been under-predicting the whole time and run-to-run
  variation had been quietly paying the difference back.
* Report an interval or report nothing. Five runs of one unchanged command gave
  TPOT errors from −5.4% to +2.4%: the standard deviation is larger than the
  mean, and the number this document led with for weeks was simply the luckiest
  draw of the five. Every single-run comparison here, the trajectory table
  included, is a difference of two samples from that spread.
* The thing that finally measured the noise was running the same command five
  times, which cost twenty minutes and nothing else. It was not done earlier
  because each individual run had always looked reasonable.
* A harness can measure itself instead of the system and still return a full
  table of plausible numbers. `benchmark_serving` reported TTFT, TPOT, ITL and
  throughput for a simulated engine; every one of them described the simulator.
  Nothing failed, nothing warned, and the numbers were in the range a reader
  would expect.
* Widening a configuration is a cheaper way to find defects than deepening one.
  One flag — `--tp 2` — produced a broken artifact convention, a misattributed
  worker death, a refuted hypothesis, a measured non-issue, and the first
  functional-form failure in the project.
