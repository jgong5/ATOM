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

Defects in ATOM itself, rather than in Compass, are collected separately in
[ATOM_DEFECTS.md](ATOM_DEFECTS.md).

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

Since then: decode fitted per CUDA-graph rung, every fit moved to relative error,
and admission time modelled (items 2, 3 and 11). Three repeats each:

| | TTFT | TPOT | latency |
| --- | --- | --- | --- |
| before any of it | −9.9 ± 6.4 | −2.4 ± 3.2 | −5.2 ± 4.3 |
| rungs + relative fit | −20.7 ± 4.5 | −0.4 ± 4.1 | −8.0 ± 1.7 |
| **+ admission (13 ms)** | **+2.0 ± 7.0** | **−1.0 ± 3.3** | **+0.0 ± 1.8** |

All three are now **inside the noise** — the harness says so itself, because the
spread across repeats exceeds the error. That is the point at which further
modelling stops being measurable on this workload, and it is the honest ceiling
for a single-GPU 0.6B model on a shared machine.

The middle row is worth keeping. TTFT looked worse there, and not because prefill
got worse — prefill improved from +18.3% to +3.9% at the shape this workload
actually uses. Correcting the fit removed an over-prediction that had been paying
for the admission cost, and made the missing term visible. See "A quarter of TTFT
happens outside any forward".

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
land in, at both widths — that is item 12, and it is in-sample. And run 1 of the
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

**Checked, and the obvious route is closed.** `OpTimingTracer` brackets every
dispatched operator with its own pair of CUDA events, and the region containing
them with one more, in the same forward — so the comparison carries no
run-to-run variance. On a decode step at batch 4, compiled at level 3:

| | |
| --- | --- |
| 327 operators, summed | 45.664 ms |
| the region containing them | 68.949 ms |
| covered | **66.2%** |
| the same step, replayed | **3.946 ms** |
| its 113 gemms alone, timed per-op | **15.368 ms** |

Two separate problems, and the second is fatal. A third of the eager region
belongs to no operator, so the parts do not account for the whole. And the parts
are themselves inflated by roughly an order of magnitude: the gemms alone cost
nearly four times the entire replayed step, and every operator together costs
nearly twelve times it. A gemm reads 0.121 ms where its true cost must be tens of
microseconds. **The instrumentation costs several times the kernel it measures**,
which is the same observer effect that made the first step timer report a machine
33% slower than the real one, at a scale where it cannot be tuned away: these
kernels are simply too small to bracket individually.

So in-line per-operator events are not a cost source. What survives is more
encouraging than that sounds:

* The concentration is extreme. Sixteen distinct operators, **half the time in
  two of them** — `aiter::gemm_a16w16` at 33.7% and
  `aiter::unified_attention_with_output_base` at 23.8% — and twelve kinds cover
  99.7%. Pricing a handful of kernels would cover almost all of a step, rather
  than needing a model per operator across hundreds.
* That makes **offline microbenchmarking** the indicated route: run each
  `(kernel, shapes, dtypes)` many times back to back and take the steady state.
  No per-operator instrumentation, no observer effect, and a price list that is
  reusable across every configuration rather than remeasured per deployment —
  which is also the answer to the calibration-economics problem.

**Where such a harness can live is not a free choice.** `aiter` registers its
operators lazily, through a JIT that fires on first call: importing the module
that defines `gemm_a16w16` does not put it in `torch.ops.aiter`. So a benchmark
cannot look a kernel up cold. Worse, running a whole model first does not help
either — a process that created an engine and generated tokens still reports
`torch.ops.aiter.gemm_a16w16` missing, because ATOM runs the model in a **worker
subprocess** and the registration happens there. The traced graph is the proof
from the other side: it holds `aiter::`, `triton::` and `inductor::` operators,
recorded inside the runner.

So the price list has to be produced **inside the model-runner process, after
warmup** — a Compass mode rather than a standalone script. That is not a
concession: the kernels priced are then the deployment's own, already autotuned
for the shapes it uses, rather than a fresh JIT build tuned differently.

The shape of it: after warmup, read a captured graph, take each distinct
`(name, input_shapes, dtypes)`, allocate tensors to match, call it a few thousand
times inside one pair of events, and divide.

**Built and run.** `--compass-bench-graph` / `--compass-bench-out`, priced after
warmup inside the runner. On the batch-4 decode graph:

| | |
| --- | --- |
| signatures priced | 24 of 36 |
| operators covered | 181 of 330 (54.8%) |
| `aiter::gemm_a16w16` | **31.4 µs** each |
| the same gemm, in-line events | **121 µs** |

The instrumentation overhead is confirmed quantitatively at **3.9x**, which is
what the in-line route was rejected for.

**But summed prices overshoot the step.** 113 gemms at 31.4 µs is 3.552 ms and
the priced total is 4.235 ms, against a replayed step of **3.946 ms** — and that
total does not include attention or rmsnorm, the second and third largest
consumers, which are unpriced. So the parts already exceed the whole while a
third of the whole is missing.

Two candidates, neither yet separated: a standalone call allocates its output
every iteration where the model reuses a buffer, so the price carries an
allocation the real step does not; and a replay overlaps kernels that a
back-to-back loop serialises. Both inflate a sum of standalone prices. This is
the same shape of question as "does Σ ops explain a step", asked of better
numbers, and it is what phase C has to answer.

**Scalars recorded, and coverage nearly doubled.** Twelve signatures failed at
first, nine of them wanting a non-tensor argument the graph did not keep --
`aiter::rmsnorm2d_fwd_` wants `eps`. `OpSpec` now carries them, positional ones
named by index so the call can be interleaved back together, and coverage went
from **54.8% to 90.3%** of operators. Graph comparison is untouched: `_signature`
in `diff.py` never looked at scalars.

The three that remain are `triton::` and `inductor::` kernels, recorded by the
Triton launch tracer patching `JITFunction.run` rather than by the dispatcher, so
`torch.ops` can never resolve them. Those need the launch path, which is a
different mechanism rather than a missing field.

### A step is not the sum of its kernels

With 90% of the operators priced:

| | |
| --- | --- |
| priced total, hot | 9.239 ms |
| priced total, cold | 8.640 ms |
| the replayed step | **3.946 ms** |

**Summed prices are 2.2-2.3x the step they are meant to explain.** The earlier
agreement to 7% was an artifact of pricing only 55% of the operators: adding
rmsnorm's 113 doubled the sum. A replay overlaps its kernels heavily, and a
back-to-back loop measures each one alone.

**The root cause of the 2.3x, found.** A per-call benchmark cannot price a
kernel smaller than its own call overhead. Pricing one gemm at twelve shapes:

| M | per-call loop | in a CUDA graph |
| --- | --- | --- |
| 1 | 30.9 µs | 9.09 µs |
| 8 | 30.2 µs | 9.24 µs |
| 64 | 30.0 µs | 12.65 µs |
| 256 | 30.4 µs | 29.29 µs |
| 1024 | 135.3 µs | 137.58 µs |

**A flat 30 µs floor from M=1 to M=256** — 256 times the work for the same price.
The kernel only becomes visible past M=512, and the two methods converge there,
which is what a launch-overhead explanation predicts. The arithmetic closes it:
298 priced operators times 30 µs is 8.9 ms, against a priced total of 9.24 ms.
**The price list was the operator count times the floor.**

**And the floor is the host, measured directly.** A CUDA event is timestamped
when the *device* reaches it, so the elapsed time between two of them is
device-side wall clock across the loop — not a sum of kernel durations. That
leaves two possibilities, and timing the enqueue loop's own wall clock separates
them:

| M | device (event) | host (enqueue) | host/device | in graph |
| --- | --- | --- | --- | --- |
| 1 | 31.9 µs | 31.9 µs | **1.00** | 9.09 µs |
| 8 | 31.4 µs | 31.4 µs | **1.00** | 9.24 µs |
| 64 | 31.3 µs | 31.2 µs | **1.00** | 12.65 µs |
| 256 | 30.7 µs | 30.7 µs | **1.00** | 29.29 µs |
| 512 | 43.3 µs | 31.7 µs | 0.73 | 48.44 µs |
| 1024 | 136.0 µs | 30.7 µs | **0.23** | 137.58 µs |

They are the same number up to M=256. The device was idle waiting for the host
for the whole loop, so the price is not the kernel plus an overhead — it is the
host's cost, with the kernel finishing early inside the wait. At M=4 that is 9 µs
of work inside a 34 µs wait.

The host figure is flat across every shape, which is what per-call work that
cannot depend on tensor size looks like: the Python call, the dispatcher, aiter's
ctypes wrapper, the operator's output allocation, the HIP launch. Not decomposed
further here.

The crossover falls where it must. At M=256 the kernel is 29.3 µs against 31 µs
of host and the two are neck and neck; by M=1024 the kernel is 138 µs and the
host is irrelevant. Past that point a per-call loop measures the kernel correctly,
which is exactly where loop and graph pricing agree.

A third measurement settles it, and its surprise is the useful part. Bracketing
each call in its own pair of events -- meant to time one kernel alone -- returns
the *largest* number of the three:

| M | loop | host | own event pair | in graph |
| --- | --- | --- | --- | --- |
| 4 | 33.9 µs | 33.9 µs | **52.20 µs** | 9.22 µs |
| 64 | 31.3 µs | 31.2 µs | 52.08 µs | 12.65 µs |
| 256 | 30.7 µs | 30.7 µs | 65.72 µs | 29.29 µs |
| 1024 | 136.0 µs | 30.7 µs | 174.46 µs | 137.58 µs |

Because `began` is recorded before the call: the stream is empty, the device
timestamps it immediately, and then idles through the entire host dispatch before
the kernel arrives. So it measures dispatch **plus** kernel with no overlap --
174 ≈ 31 + 138 at M=1024, and 52 ≈ 34 + 9 plus a synchronise at M=4. There is no
way with this API to start a clock after dispatch and before execution, so a
single call cannot be timed in isolation at all.

The loop, by contrast, pipelines: while the device runs kernel *i* the host
dispatches *i+1*, so it returns `max(host, kernel)`. That model reproduces every
row — 31.2 against 31.3 at M=64, 30.7 against 30.7 at M=256, 137.6 against 136.0
at M=1024.

**Three measurements, one model:** host dispatch about 31 µs per call, kernel
from 9 µs at M=4 to 138 µs at M=1024, the loop host-bound below M≈256 and
device-bound above it. **The loop number is the host's cost and the in-graph
number is the kernel's**, and `host_seconds` sits beside every price so a reader
can tell which one they are holding.

Hoisting the event construction out of the loop is worth doing and worth very
little: `torch.cuda.Event` builds its CUDA event lazily on first `record()`, so
constructing `ended` inside the loop put that construction between the two
timestamps. Removing it saves **0.95 µs** per call at the median.

**A claim retracted.** It looked for a while as though graph mode understated a
kernel by 1.5-1.8x, on the grounds that it captures 64 independent calls the
device could overlap, with `own pair − host` as the honest figure. That
subtraction is invalid: part of the host's 31 µs happens *after* the kernel is
enqueued and runs concurrently with it, so removing the whole host cost removes
work that overlapped the kernel and inflates what is left.

Varying the capture size settles it, because the two stories predict opposite
things -- overlap keeps falling with more captured calls, serialisation does not:

| M | B=1 | B=2 | B=8 | B=32 | B=128 |
| --- | --- | --- | --- | --- | --- |
| 1 | 14.78 µs | 11.56 | 9.38 | 8.87 | **9.04** |
| 64 | 18.12 µs | 15.18 | 12.68 | 12.31 | **12.63** |
| 1024 | 143.07 µs | 139.91 | 137.39 | 138.23 | **137.88** |

Falls to B=8, flat after. That is a fixed per-replay cost being amortised --
`kernel + overhead/B`, with the overhead measuring 5.7 µs at M=1 and 5.2 µs at
M=1024, the same constant regardless of kernel size, which is what a graph launch
must look like. The model reproduces the intermediate points (M=1, B=2 predicts
11.91 against 11.56 measured).

**Flat across B means the captured calls serialise**, as stream-ordered capture
implies. So the in-graph figure is a latency, not a throughput, and it is the
kernel's cost.

### So graph pricing is the right measurement

It removes the ~31 µs of host dispatch that swamps a decode-shaped kernel, and it
measures the kernel in a replayed graph -- the context production runs it in. The
only requirement is a capture past the knee, since a one-call capture charges the
whole 5.5 µs replay cost to one kernel; B=8 is enough and 64 is used.

Which means the gap between the priced total of 1.765 ms and the 3.946 ms step is
**not** a measurement error waiting to be corrected. It is mostly the 28 unpriced
attention operators.

### Attention cannot be priced from a shape signature at all

Not a missing scalar. `unified_attention_with_output_base(q, q_scale, k, v,
positions, layer_name, use_mla, qkv)` takes a `layer_name` and a `use_mla` that
the tracer does record — but the body looks the layer up in
`static_forward_context` and calls into its implementation, which begins
(`attention_mha.py:175`):

```python
fwd_ctx: ForwardContext = get_forward_context()
if fwd_ctx.context.is_dummy_run:      # .context is None outside a forward
```

and goes on to need `attn_metadata`: block tables, sequence lengths, a populated
paged KV cache. Rebuilding the call means rebuilding the runner's per-step state.

So **attention's cost is not a function of its tensor arguments.** It depends on
paged-KV state — how many blocks the sequences occupy and how scattered they are
— which a signature of `(name, input_shapes, dtypes)` cannot express and which no
amount of extra recorded arguments would fix. A price list keyed that way has a
hole in it exactly where a quarter of the work is.

**Fixed by moving the boundary, not the granularity.** Pricing a whole layer
would have worked and was the wrong answer: a layer is a model-structure concept,
and Compass does not have those -- its graph knows group sizes and ranks and
nothing else. Putting layers in it to serve a benchmark would trade the property
that makes the representation general.

The op boundary moved instead. `unified_attention_with_output_base` now takes
`block_tables`, `context_lens`, `slot_mapping`, `cu_seqlens_q`, `max_seqlen_q`
and `max_seqlen_k` as arguments, resolved by the caller from the same context the
backends read. Production takes the same path it always did and ignores them; off
the serving path, with no live forward, the operator stands up a context from
them and the backends work unchanged. Eight attention backends were left
untouched. The KV cache needs no argument -- it lives in its own context,
installed once at startup, and `set_forward_context` picks it up, which is also
why the benchmark had to move from after warmup to after CUDA graph capture: at
warmup the runner is still being built and there is no cache to walk.

Coverage went from 90.3% to **98.8%** of operators, and the graph is better for
it independently of pricing: attention used to be an opaque node whose real
inputs were invisible.

### Callable is not the same as priced

The number that came back is not credible, and the way it fails is the finding:

| kernel | n | each | total |
| --- | --- | --- | --- |
| `unified_attention_with_output_base` | 28 | 163.6 µs | **4.582 ms** |
| `gemm_a16w16` | 113 | 11.3 µs | 1.275 ms |
| priced total | | | 6.347 ms |
| the replayed step | | | **3.946 ms** |

Attention alone exceeds the whole step. The harness fills integer tensors with
zeros, so `context_lens` is all zero, while the recorded scalars carry
`max_seqlen_k = 16384` -- the padded bound, not the actual context. The kernel
sizes its work from those and walks 16K of KV per sequence where the real step
walks a few dozen tokens.

So a signature of `(name, shapes, dtypes, scalars)` is not enough for a
**data-dependent** kernel: its cost turns on argument *values*.

`OpSpec` now carries `int_values` -- the contents of small integer tensor
arguments, recorded during trace and used to rebuild the call. The signature
includes them, so the same kernel at two context lengths prices as two entries
rather than one, which took the graph from 36 signatures to 44. That is correct
and was needed regardless of what follows.

**It did not move attention's price.** With `context_lens` recorded truthfully as
`[155, 155, 155, 155]` instead of zeros, attention costs 163.7 µs against 163.6 µs
before. So the zero-filled metadata was never the cause, and the hypothesis that
motivated the work was wrong.

What attention must cost in situ can be bounded: the step is 3.946 ms and the
other priced kernels account for about 1.6 ms of it, leaving roughly 84 µs per
call. The benchmark says 163.7 µs -- about twice too slow, and indifferent to the
metadata values.

**The reconstruction is incomplete in a way arguments cannot fix.**
`AttentionMetaData` has 29 fields and the synthetic one fills 7. The backend
reads `has_cached` and `cu_seqlens_k`, which are not among them, and it also
reaches into runner-owned buffers -- `var["slot_mapping"].gpu[:running_bs]` --
that are not arguments to the operator at all. So the benchmark is very likely
executing a different branch from production, and passing more arguments does not
close that: some of what the kernel consumes belongs to the runner, not the call.

Which sharpens the earlier rule. An operator is priceable when it is a pure
function of its arguments; attention was made *callable* by supplying its
metadata, but it is still not *pure*, because the backend sources state from
elsewhere.

### What attention actually costs, measured rather than inferred

Three hypotheses about the initialisation were tested and all three were wrong.
Recording `context_lens` truthfully as `[155, 155, 155, 155]` instead of zeros
left the price at 163.7 µs against 163.6 µs. Correcting the extents from the
configured bound to the decode-actual values -- `max_qlen` 16384 to 1,
`max_klen` 16384 to 155 -- gave 163.73 µs against 163.61 µs. And the
work-partitioning fields (`kv_indices`, `work_info_set`) are not on this path at
all: `use_pa_decode_bf16_asm()` requires `gfx1250` and this is `gfx942`, so the
backend is `paged_attention_asm`, which reads only `block_tables`,
`context_lens`, `max_qlen` and `qo_indptr`.

The benchmark's figure is invariant to every attention parameter there is to
vary, which is itself the finding: it is not measuring attention's work.

So the cost was measured by **ablation** instead -- return the right shape
without doing the work, and difference the step:

| | |
| --- | --- |
| decode step with attention | 3.041 ms |
| decode step without | 2.396 ms |
| attention, per call | **23.0 µs** |
| the microbenchmark | 163.7 µs |

**The benchmark overstates attention by 7.1x.** And the earlier estimate of
84 µs, arrived at by subtracting the other priced kernels from the step, was
also wrong -- which says those prices do not sum correctly either. Substituting
the measured 23 µs, the priced total becomes 2.41 ms against a 3.946 ms step:
the sum now *under*-counts by 1.5 ms, where with the benchmark's attention it
over-counted by 2.4 ms.

Two lessons worth keeping. Ablation is the ground truth this line of work needed
and was reached for only after three rounds of guessing at inputs; differencing a
step with and without an operator answers "what does this cost here" directly,
where a microbenchmark answers "what does this cost somewhere". And a benchmark
figure that does not move when its inputs do is not a measurement of those
inputs -- that invariance should have been checked first, since it is one run.

`ATOM_ABLATE_ATTN` is kept for the purpose, read once at import because it is
consulted 28 times per decode step.

(A false lead worth recording so nobody follows it twice: the implementation
returns `torch.empty_like(q)` when `is_dummy_run` is set, with a comment saying
attention is skipped during CUDA graph capture — which would mean the captured
graph omits attention entirely. It does not. `is_dummy_run=True` is set only in
`dummy_execution` and `warmup_model`, never in `capture_cudagraph`, so the skip
applies to the dummy runs *around* capture and attention is captured normally.)

Capturing the calls into a CUDA graph removes what production removes — one
submission, no host in the loop — and takes ~21 µs off. A ~9 µs floor survives,
and that one is real: the gemm reads an 8 MB weight whatever M is.

### And the fix over-corrects, which is the more interesting result

| | |
| --- | --- |
| priced total, per-call loop | 9.239 ms — **2.34x** the step |
| priced total, in a graph | 1.765 ms — **0.45x** the step |
| the replayed step | 3.946 ms |

Two reasons, and the second is structural.

*Attention is still not priced.* `aiter::unified_attention_with_output_base` takes
a runtime object, not a scalar — it fails with `'NoneType' object has no attribute
'is_dummy_run'`, so it wants a forward-context argument the graph cannot record
as a value. That is 28 operators and about a quarter of the work, missing from
the total.

*The graph benchmark lets kernels overlap that never overlap.* It captures 64
copies of one kernel on one set of inputs, which are mutually independent, so the
device runs them concurrently. In a real step, layer N+1 consumes layer N's
output and the chain is serial. `aten::slice` falling **315x** to essentially zero
is the signature: not a cheap kernel, an elided one.

**So neither mode measures a kernel in the dependency context it runs in.** The
loop adds a host launch the replay does not pay; the graph removes a
serialisation the model cannot avoid. They bracket the answer —
1.765 ms < 3.946 ms < 9.239 ms — and neither converges to it.

That is a limit of per-operator pricing itself, not of this harness: **a step's
cost is a property of the sequence, not of the multiset of kernels in it.**
Pricing kernels independently discards exactly the information that decides how
they compose.

The route that survives is to benchmark a *chain* rather than a kernel — capture
one transformer layer's operator sequence, price that, and multiply by depth.
It keeps the dependency structure inside the measurement, stays far cheaper than
a shape sweep, and should transfer across batch size and width. Pricing the whole
step that way would just be measuring the step again, which is what the
calibrated oracle already does; a layer is the largest unit that is not circular.

### Hot and cold do not isolate what they were meant to

Cache state is per argument, not per kernel: in a real decode step a gemm's
activation is hot because the previous operator wrote it, while its weight is
cold and every one of the 113 gemms uses a different one. So the harness prices
either way — `--compass-bench-cache hot|cold`, the latter rotating over enough
input sets to overflow the cache.

The expectation was that cold would be dearer, since streaming a weight from
memory costs more than rereading a resident one. **Cold came out cheaper**:
0.94x overall, 0.94x on gemm, 0.87x on rmsnorm. The likely reason is that hot
mode reuses one output buffer, so consecutive calls serialise on a
write-after-write dependency, while rotating buffers lets them pipeline. The two
modes therefore conflate cache residency with output-buffer serialisation, and
neither isolates cache by itself.

Worth keeping — they bracket a 6% band and cost nothing to run — but the honest
reading is that at this kernel size overlap dominates cache, and the 2.2x gap
above is not something cache realism will close.

The tracer is kept, because the question it answers is the right one and the
answer needs to stay reproducible. `scripts/compass/op_cost.py` reports it.

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

### 3. Admission time is not modelled, and it is all of the TTFT error — **M**

A request's first token arrives about 13 ms later than the forward that produced
it can account for. That time is spent getting the request from `preprocess` to
the engine core and from there to a worker -- two process hops through polling
loops, with an idle engine on the far side so nothing overlaps it. Measured
across six runs: 8.25, 11.52, 13.71, 13.91, 17.88, 18.25 ms, unrelated to request
count or prompt length.

A simulated run advances its clock by predicted *forward* durations, so none of
it exists, and TTFT comes back short by roughly that amount every time --
46.09/57.68 predicts −20.1% against an observed −20.7%. It is the whole of the
TTFT error and no work on the cost model can touch it. See "A quarter of TTFT
happens outside any forward" for the decomposition.

Modelling it means advancing the clock when a request is **admitted** rather than
when a step runs, which is `engine_core`'s loop rather than the runner seam this
project has stayed behind so far. The term's own spread is ±5 ms on 13 ms, so
expect TTFT accurate to about ±8% afterwards, not better -- worth having against
−20%, and worth knowing before starting.

Per-*step* engine work needs no term: it measures ~0.5 ms and is overlapped with
the device, so adding it would make decode worse.

### 4. TP=4 calibrates and evaluates 19% apart on prefill — **M**

The cost model generalises to four ranks; the calibration does not. Three
repeats at TP=4, against three at TP=1, both with 13 ms admission:

| | TP=1 | TP=4 |
| --- | --- | --- |
| TPOT | −1.0 ± 3.3 | −2.1 ± 6.9 |
| TTFT | +2.0 ± 7.0 | **−26.1 ± 11.4** |
| latency | +0.0 ± 1.8 | −12.1 ± 9.4 |

TPOT is inside its noise at both widths, so the per-rung decode model transfers
untouched. TTFT is negative on every draw and its best is −16.3%, so it is
systematic.

**Not admission**, which was the obvious guess and is wrong: measured the same
way at both widths it is 17.88 ms at TP=1 and 17.94 ms at TP=4. It does not scale
with worker count, so it is not made of process hops in the way the two-hop
description suggests.

**Not the model's form either.** At TP=4 the oracle predicts 39.49 ms for the
evaluation's 2512-token prefill, and its own sweep rows between 1500 and 4000
tokens average 40.69 ms — faithful to about 3%. The evaluation measured 48.44 ms.

So the sweep and the evaluation disagree by **19%** at a matched shape, where at
TP=1 the same comparison agrees to 4%. Whatever differs between a calibration run
and an evaluation run is four times larger at four ranks. Run-to-run spread is
larger there too — sd 11.4 against 7.0 on TTFT — so some of it is simply variance
across four contending processes, but a gap that big on a single shape is worth
separating from noise with repeated calibration rather than repeated evaluation.
Nothing yet distinguishes "the sweep's prefill steps are warm and the
evaluation's is cold" from "four processes vary more".

### 5. No collective cost model — **S/M**, and narrower than it looked

Collectives are recorded with their group and their bytes, which is what a cost
model needs, and nothing consumes it.

It does **not** block multi-GPU prediction the way this once claimed. The
collective runs inside the forward, and the forward is timed with CUDA events, so
a calibrated oracle at a width it was measured at has already absorbed the
collective's cost without naming it. The TP=2 error above is not a missing
collective — it is item 12.

What the gap actually costs is prediction at a width **nobody measured**:
calibrate at TP=2 and ask about TP=4, and there is nothing to scale. That is a
narrower and later problem than "blocks any credible multi-GPU prediction".

### 6. An HTTP benchmark cannot measure a simulated engine — **L**

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

The median was exact and the mean was not: the first 13 requests shared **two**
distinct first-token instants where the real run had 13.

**That was a causality constraint, not a bug.** The client submits concurrently,
so requests reach the engine in a different order from the one they were declared
in. If the first request *received* is declared for t=0.9 s, the idle engine jumps
virtual time to 0.9 s — and a request declared for t=0 landing on the socket a
moment later is retroactively late. A discrete-event clock may only advance when
it knows no earlier event will still turn up.

**Fixed, for a closed workload, without pacing.** The client says how many
requests are coming (`compass_workload_size`) and the engine runs nothing until
it has them all; after that every arrival time is known, so every jump is safe.
The wait is bounded by how long submission takes — a one-off startup cost, not a
per-step sleep, which is what makes real-time pacing unacceptable. A timeout
releases the barrier if a client dies mid-submission, and says outright that the
run's latencies are then invalid rather than letting a plausible table through.

| | real | declared arrivals | + barrier |
| --- | --- | --- | --- |
| TTFT median | 35.84 ms | 35.85 ms | **35.39 ms** |
| TTFT mean | 42.23 ms | 213.50 ms | **37.39 ms** |
| TTFT max | 82.10 ms | 1653.09 ms | **67.63 ms** |
| latency median | 262.35 ms | 306.40 ms | **294.05 ms** |
| wall clock | 9 417 ms | 495 ms | 847 ms |

64 distinct first-token instants out of 64 requests, and none finishing before it
arrived. The tail is gone and the maximum is now *below* the real one.

**Re-measured once decode was fitted per rung and every fit moved to relative
error**, same workload, 64 requests at 8/s:

| | real | simulated | error |
| --- | --- | --- | --- |
| **latency mean** | 280.28 ms | 269.00 ms | **−4.0%** |
| latency median | 271.14 ms | 262.58 ms | −3.2% |
| TTFT mean | 40.97 ms | 31.79 ms | −22.4% |
| TTFT median | 35.37 ms | 31.05 ms | −12.2% |
| wall clock | 9 426 ms | 639 ms | 14.8× faster |

Against +83.8% TTFT and +37.5% latency the first time this ran. End-to-end
latency at −4% is the closest this project has come to reproducing a served
workload, and it is on the configuration the whole thing is for.

The remaining TTFT shortfall was 9.18 ms, inside the 8-18 ms admission cost
measured offline, and the same single cause: item 3. One extrapolation warning
fired, correctly, on a context below rung 8's calibrated floor.

**With admission modelled** (`--compass-admission-seconds`):

| | real | none | 13 ms | 9 ms |
| --- | --- | --- | --- | --- |
| TTFT mean | 40.97 ms | −22.4% | +9.3% | **−0.4%** |
| latency mean | 280.28 ms | −4.0% | +0.6% | **−0.8%** |
| latency median | 271.14 ms | −3.2% | +1.6% | **+0.2%** |
| TTFT median | 35.37 ms | −12.2% | +24.5% | +13.2% |

**The constant is path-specific, not machine-specific.** 13 ms is what the
offline batch path measured and it overshoots serving by 9%; 9 ms is what the
serving path measured. Same machine, same model, same process layout — different
entry code, so a different cost. Calibrate it against the path being simulated,
not against whichever run was convenient.

**Read the mean, not the median.** A single constant shifts the whole
distribution, and the real one is right-skewed — mean 40.97 against median 35.37.
So the means land near-exact while the median overshoots by 13%. Reproducing the
shape needs a distribution rather than a constant, which is more machinery than
this is worth until something depends on tail TTFT.

**And this number is in-sample for that term.** 9 ms was derived from the
previous run of this same workload, so −0.4% is a fit, not a generalisation. It
is a different run — the spread between runs is real — but a held-out figure
needs a workload the constant was not measured on.

**What this does not solve.** There is no count to wait for in open-ended
serving, and a simulator cannot know whether another request is about to arrive.
That case still needs the schedule handed over up front — the workload as an
input file, with HTTP demoted to fetching results. The barrier buys a correct
closed-workload validation harness, which is what validation needs, and defers
the general problem rather than answering it.

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

### 7. Derivation is uncompiled; production is not — **L**

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

### 8. Trace mode does not observe the CUDA-graph path — **M**

Trace mode skips `capture_cudagraph` deliberately, so its graph is the eager
operator sequence. A replay is a single opaque submission: there are no
per-operator events to record, even in principle. So the operators must come from
eager execution and the replay's cost must be carried as a term the oracle
applies, not as operators in the graph.

Measure mode does capture, so timings already come from the replayed path. What
is missing is the bridge between the two artifacts.

### 9. Only decode steps are traced, and only one — **M**

`trace_step` records exactly one step, in practice a small decode. A deployment's
cost is dominated by shapes never captured: prefill, chunked prefill, mixed
prefill/decode batches, and long-context decode where attention stops being
cheap. Speculative decoding and MTP are entirely untraced.

### 10. A custom op's inner Triton kernel is recorded only on hardware — **S**

On a real device `aiter::masked_embedding` both dispatches and launches
`triton::_masked_embedding_kernel`, so the capture holds both. On meta the kernel
never launches, so derivation holds only the outer operator. Harmless for cost —
the outer operator carries the shapes — but a systematic difference between the
two graphs that will confuse anyone diffing them.

### 11. Simulated TP's `all_gather` is not the real one — **S**

At a physical width of one it builds a zero-padded buffer (`movedim`, `reshape`,
`view`, `zeros`) where the real path uses `view.dtype`. Seven operators out of
451. Possibly worth simply accepting — but as a decision, not a residue nobody
looked at.

### 12. Decode cost falls as batch grows; the model said it rises — **M**, done

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

**Repaired by fitting one small model per rung.** `StepShape` now carries
`capture_bucket` — which rung the replay padded up to — and decode is fitted
separately within each, as `intercept + slope x total_context`. Batch size is
dropped inside a rung because the rung already carries it.

Held out on 499 rows none of the models had seen:

| decode model | median \|err\| | RMSE | the bad region |
| --- | --- | --- | --- |
| `[1, batch, total_context]` (was) | 5.04% | 0.4216 ms | −3.7% |
| per-rung intercept, shared slope | 8.09% | 0.3952 ms | −12.6% |
| **per-rung intercept and slope** | **0.93%** | **0.1259 ms** | **−1.3%** |

A shared slope across rungs is *worse* than what it replaced, which is the
informative part: a replay at rung 16 reads sixteen padded rows and one at rung 1
reads one, so cost per unit of history is not the same number at both. The rung
has to own both coefficients.

In place, over 2174 measured decode steps the fitted oracle sits at 0.73% median
error, per rung: 0.46 / 0.45 / 0.69 / 1.67 / 2.11 / 1.57 / 0.25 / 0.37% for rungs
1 to 64.

**It does not explain everything.** The batch-8 short-context region went from
−3.8% to −1.7%, not to zero, and error is worst at rungs 8 and 16. Something
still varies with batch *inside* a rung: at matched total context, one long
sequence is not the same cost as many short ones, and neither the rung nor the
summed history can see the difference. That is the next feature to look for, and
it is a smaller effect than the one just removed.

**Getting the rung right was harder than fitting it.** The first implementation
mirrored `ModelRunner`'s input-buffer padding, which scans `reversed(capture_
sizes)` — correct only on a descending list. `capture_cudagraph` sorts descending
for the capture loop and ascending again when it finishes, so at runtime that
expression returns the *largest* rung, not the smallest: every step in a 1662-row
sweep came back as bucket 512 and the diagnostic compared three models that had
all collapsed to one. The rule that actually selects the graph is in
`ForwardMode.decide`, and Compass now mirrors that instead, written as a `min` so
it cannot care which way the list is sorted.

That leaves a real defect in ATOM: the padding site takes the largest rung where
its own comment says a 65-request batch should replay the 128 graph. It
over-zeroes a buffer rather than mis-selecting a graph, so it costs a little
bandwidth and nothing else — but it is wrong, and it is not Compass's to fix.

The sweep now reaches rungs 32, 48 and 64, each at two prompt lengths so its
slope is identifiable. Stopping at 16 is what left the serving run at batch 63
asking about a rung nothing had measured.

## Configurations that cannot be modelled at all

### 13. Asymmetric parallelism — **L**

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

### 14. Shape-changing collectives have no meta stand-in — **S/M**

`all_reduce` and `broadcast` preserve shape, so meta can hand back the input.
`all_gather` grows and `reduce_scatter` shrinks, so they raise rather than guess
— deliberately, since assuming "same shape" would corrupt every downstream shape
while appearing to work. TP alone does not need them; sequence and expert
parallelism do.

### 15. One communication group is resolvable; several are not — **M**

A collective names its group by elimination: with exactly one group of size above
one, there is nothing else it could have run on. With TP and EP together the
ambiguity is real and is recorded as `"?"`.

Settling it means intercepting at the group object rather than the dispatcher —
`get_tp_group()` and its siblings know their own identity. That is a replacement
for the current resolver, not an addition, and item 13 needs it too.

### 16. Qwen3.8-27B cannot be captured here — **?** (environmental)

Its decode path JIT-builds AITER's *gluon* paged-attention kernel and the build
fails:

    subprocess.CalledProcessError: Command '['make', 'build', '-j1']'
    returned non-zero exit status 2

Narrower than it first appeared: Qwen3-0.6B takes the **ASM** decode path, which
ships as a prebuilt `.co`, so it captures without touching the failing build. One
kernel path, not capture as a mechanism — which is why validation was possible on
a smaller model while the stated target model stayed blocked.

**Diagnosed: it is an image defect, and the fix is one package.** The `make`
stderr had never been looked at, because `aiter` passes
`capture_output=AITER_LOG_MORE < 2` and nobody had set that to 2. Running the
leftover build by hand shows it:

    clang++: warning: ... '/usr/lib/gcc/x86_64-linux-gnu/13' would be chosen
                          over '/usr/lib/gcc/x86_64-linux-gnu/14'
    error: "Could not find standard C++ header 'cmath'..."
    fatal error: 'cstdlib' file not found

`/usr/include/c++/` holds only `13`. The image has gcc-14's runtime directory
(`crtbegin.o` and friends) but not its C++ headers, and ROCm 7.2.4's clang prefers
the highest version it finds. So every JIT build of a C++ kernel fails, while
everything prebuilt is unaffected — which is exactly the observed split.

Verified: adding `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13` compiles
the failing translation unit cleanly, exit 0. `libstdc++-14-dev` is installable
from the configured apt sources and would fix it at the root, since clang's own
preference then finds headers where it looks.

Not applied — it changes the shared container rather than this repository, and
the container's writable layer does not survive a teardown, so it belongs in
`gpu_docker`'s setup rather than in an ad-hoc `apt install`.

## Measurement and performance

### 17. The extrapolation warning is per-feature, so it cannot see a hole — **S**

The oracle warns when a query falls outside the range of a feature it was
calibrated over, and that safeguard has caught two real errors. It checks each
feature *separately*, so it is blind to a gap in their joint distribution.

The TP=2 evaluation asked about batch 8 with a total context of ~2650. Batch 8
was covered (1–16 seen) and 2650 was covered (57–41052 seen), so nothing warned.
The sweep's batch-8 steps actually run 1776–2288 and then jump to 5360: the query
sits in a hole, and the guard reported it as interpolation.

Not the cause of the TP=2 error — item 12 is, and it is in-sample — but the guard
claims a property it does not have. A convex hull, or a nearest-neighbour
distance with a threshold, would say what the bounding box cannot. The k-NN
oracle already computes the distance this needs.

### 18. No way to tell the runner when to start measuring — **M**

Throwaway warmup requests were served to get Triton autotuning out of the way,
and the runner measured them anyway: a runner sees steps and has no idea which
request a step belongs to, or that some were meant to be ignored. The result was
a 7.5 s prefill sample sitting in a table beside a 0.03 s one.

Worked around by discarding the first step of each kind, which is blunt — it
drops a real sample when a workload has few and keeps warmup steps when it has
many. What is missing is a measurement window the engine can open and close
across the process boundary. The same gap makes it hard to calibrate one phase of
a deployment without the others polluting the table.

### 19. Derivation is too slow to run per step — **M**

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

### 20. Capture pays for Triton autotuning — **S**

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

The residual seven are simulated TP's own zero-padded `all_gather` — item 11.

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
belongs entirely to the asymmetric strategies of item 13, where ranks genuinely
diverge and no rank stands in for another.

**It holds at four ranks too**, which was not obvious — four processes contending
could have skewed where two did not. Over 2295 steps at TP=4:

| rank | total | vs rank 0 |
| --- | --- | --- |
| 0 | 16527.3 ms | — |
| 1 | 16524.7 ms | −0.02% |
| 2 | 16530.3 ms | +0.02% |
| 3 | 16527.8 ms | +0.00% |

Charging every step to its *slowest* rank rather than to rank 0 — which is what a
barrier actually costs — adds **0.06%**. And the slowest rank was 0/1/2/3 on
615/555/567/558 steps, near-uniform, so there is no straggler to correct for.
Rank 0 is a fair proxy at both widths tried.

Everything else structural survived the widening untouched: four per-rank tables
written and read back, all eight rungs fitted, the arrival barrier and the
admission delay unchanged. TP=4 needed no code.

### No fixed ports

The rendezvous store was hardcoded to a port. The container runs with host
networking on a machine shared with about twenty others, so a fixed number
collides with whatever holds it — including an earlier run of the same script —
and fails with an `EADDRINUSE` that says nothing about tracing. The OS picks it.

## Measurement

### A quarter of TTFT happens outside any forward

TTFT has been wrong by 15-25% through every version of the cost model, and no
amount of work on prefill moved it. Measured on the evaluation workload, with
every step timed:

| | |
| --- | --- |
| prefill steps in the whole run | **1** |
| that step, measured | 43.78 ms |
| that step, predicted | 46.09 ms (+5.3%) |
| mean TTFT reported by the engine | 57.68 ms |
| **TTFT not spent inside any forward** | **13.91 ms — 24%** |

The prefill step is predicted to within 5%. The error is the 14 ms either side of
it: scheduling, block allocation, sampling, detokenising the first token,
returning it. A simulated run advances its clock by predicted *forward*
durations, so none of that time exists — TTFT is short by almost exactly the
amount that is missing, and 46.09/57.68 is −20.1% against an observed −20.7%.

**This is not the same as the decode finding above, and both are true.** In
steady-state decode the step period *is* the forward to within 0.4%, because
steps run back to back and the engine's own work overlaps the device. TTFT is a
different quantity: it spans one request's arrival to its first token, and
crosses the scheduling and sampling path once, unoverlapped.

Which explains why two rounds of cost-model work could not fix it, and why the
prefill fix appeared to make things worse: a fit over-predicting prefill by 18%
was paying for the missing 14 ms. Correcting the fit to +3.9% removed the
compensation and exposed the gap. **Two errors cancelling, for the third time on
this page** — and this is the argument for checking components against their own
measurements rather than reading a system-level number and calling it accuracy.

**Measured, not subtracted.** The runner now records the wall time between one
forward returning and the next starting, which is every non-forward thing the
engine does. That decomposes the gap:

| | 8 x 64 | 16 x 64 |
| --- | --- | --- |
| inter-step gap, median | 0.477 ms | 0.563 ms |
| TTFT − prefill forward | 17.88 ms | 8.46 ms |
| share one gap explains | 2.7% | 6.7% |
| decode forward + gap | 4.02 ms | 4.08 ms |
| decode TPOT | 3.25 ms | 3.44 ms |

Two conclusions, and the second is the useful one.

*Per-step engine work is real but free.* It runs about half a millisecond, and
forward + gap **overshoots** TPOT rather than matching it — so it is overlapped
with the device, not added to it. That is why the step period equals the forward
to within 0.4%: not because the engine does nothing between steps, but because
what it does happens while the GPU is busy. Adding a per-step term would make
decode worse.

*The TTFT gap is almost all before the first forward.* One inter-step gap
accounts for 3-7% of it. The rest is admission: preprocess hands the request to
the engine core, which hands the batch to a worker — two process hops, each
through a polling loop, with an idle engine on the far side and therefore nothing
to overlap against. Across six runs it measured 8.25, 11.52, 13.71, 13.91, 17.88
and 18.25 ms, with no relationship to request count or prompt length. It looks
like a fixed cost of order 13 ms whose spread is a polling artifact.

**Implemented, and the seam was not the one first proposed.** The initial reading
was that this needed the clock advanced when a request is admitted, in
`engine_core`, reaching past the runner into ATOM's scheduling loop. That was
wrong twice over. Advancing a global clock per admission double-counts — two
requests arriving together would pay it twice, when in reality they are admitted
concurrently. And the machinery already existed: the declared-arrival deferral
holds a request until `arrive_time`, so holding it until `arrive_time +
admission` is the same hook with a different threshold.

So it is modelled as a delay on the *request*, not as time the engine consumes:
`--compass-admission-seconds`, defaulting to zero, which is exactly the previous
behaviour. It has to be measured per deployment, being a property of the machine
and the process layout rather than of the model.

The quantity's own spread is ±5 ms on 13 ms, so expect TTFT accurate to roughly
±8% afterwards and no better.

What it is *not* is a cost-model problem, which is the thing worth recording: TTFT
was wrong by 15-25% through four separate rounds of work on prefill and decode,
and none of them could have fixed it.

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
item 6 and is untested.

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

### A fit minimising seconds is decided by its largest samples

Widening the sweep to cover decode rungs 32-64 made TTFT worse: −9.9% before,
−17.3% after, and the standard deviation collapsed from 6.4 to 0.6, so it was
systematic rather than a draw. Nothing about prefill had changed.

The added rounds run 24-64 concurrent requests, and ATOM batches their prefill
into one step, so they are large prefill samples. The share of prefill steps
under 1024 tokens fell from 17% to 9% and the median token count doubled to
13 008. Ordinary least squares minimises squared *seconds*, so a 250 ms sample
counts about sixty times a 32 ms one: the fit followed the new mass to the large
end and the prediction at 512 tokens moved from −13% to −17% against a
measurement that did not move at all (32.51 ms then, 32.53 ms after).

Every number this project reports is a percentage, so the residual being
minimised should be a percentage too. Each equation is divided by its own target
before fitting, which makes every sample worth the same fraction of itself, and
the outlier test runs on the same relative residuals so it is not dominated by
the largest samples either. Decode is unaffected: 0.79% median against 0.73%.

| prefill prediction | absolute fit | relative fit | measured |
| --- | --- | --- | --- |
| at 512 tokens | −11.8% | −5.3% | 32.53 ms |
| **at 2512 tokens** | **+18.3%** | **+3.9%** | 43.78 ms |
| median over 121 rows | 9.51% | 9.03% | |

2512 is the number that matters, and finding that out took longer than the fix.
The change was first judged at 512 tokens, on the reasoning that eight prompts of
64 tokens make a 512-token prefill step. They do not: `--prompt-tokens 64` builds
64 *words*, which tokenise to about 314 each, so the evaluation's one prefill step
carries 2512. **A model checked at a shape the workload never visits has not been
checked.** The fix was right anyway, and by a wider margin there than at the point
used to argue for it.

The general form of the trap: **coverage is not only about range**. The range
here always bracketed the evaluation — 64 to 16 000 tokens. What changed was the
*density*, and an unweighted fit is a weighted average whose weights are the
sample values. Adding evidence in one region degraded prediction in another,
which is not something a coverage check can catch.

**And making it more accurate made the end-to-end number worse.** TTFT went from
−17.3% to −20.7% as prefill went from +18.3% to +3.9%, because the
over-prediction had been paying for something else that is missing — see below.

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
* Every engine start calls the HuggingFace API to resolve the repo, even with
  the weights already in the local cache. A day of runs plus one attempt at a
  27B download exhausted the shared IP's quota, and the next validation died at
  engine init with `429 Too Many Requests` — surfaced, of course, as the engine
  manager reporting a DP-rank shutdown. `HF_HUB_OFFLINE=1` avoids it entirely
  and should be the default for any repeated run.
* Adding evidence can make a model worse somewhere else. Widening a sweep to
  cover decode rungs degraded prefill by 4 points, because an unweighted fit is
  a weighted average whose weights are the sample values. Nothing was removed
  and no range stopped being covered.
* Mirroring an engine's rule means finding the rule that runs, not the first one
  that looks right. The bucket rule was copied from a padding site whose
  expression only holds on a descending list, against a list that is ascending
  by then -- so every step in a 1662-row sweep reported the top rung and a
  three-way model comparison silently became a one-way one. It was caught by a
  value being impossible, not by the comparison looking wrong.
* A harness can measure itself instead of the system and still return a full
  table of plausible numbers. `benchmark_serving` reported TTFT, TPOT, ITL and
  throughput for a simulated engine; every one of them described the simulator.
  Nothing failed, nothing warned, and the numbers were in the range a reader
  would expect.
* Widening a configuration is a cheaper way to find defects than deepening one.
  One flag — `--tp 2` — produced a broken artifact convention, a misattributed
  worker death, a refuted hypothesis, a measured non-issue, and the first
  functional-form failure in the project.
