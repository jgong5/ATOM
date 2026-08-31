# ATOMCompass — open problems

**Where the error stands**, Qwen3-0.6B TP=1, calibrated on a sweep and evaluated
on a workload it did not see:

| iteration | TTFT | TPOT | latency |
| --- | --- | --- | --- |
| first end-to-end run | +107% | +800% | +560% |
| A1 fixed (CUDA graphs) | +39.6% | +13.1% | +22.9% |
| prefill coverage widened | −23.9% | +22.5% | +4.7% |
| batch coverage widened | **−11.9%** | **+17.5%** | **+7.1%** |

**Ranked next: D4, the decode model's form.** TPOT is now the largest error and
it grew when the sweep widened, which is the signature of a model that cannot
fit its whole range rather than one short of data.

The target is to simulate ATOM **as it is deployed**: compilation on
(`--level 3`), CUDA graphs on, prefix caching on, chunked prefill on. Anything
that only works with `--enforce-eager` or `--level 0` is a step towards that,
not an instance of it.

Finding these was the point of the PoC, so this list is meant to grow. Each
entry says what is wrong, why it matters, and what would settle it. Effort is
rough: **S** hours, **M** days, **L** weeks or unknown.

See `DESIGN_NOTES.md` for what is settled and why.

---

## A. The graph does not yet match a production launch

### A1. CUDA-graph replay — **fixed**, was an 8.9x error

Trace and measure both stubbed `capture_cudagraph`, so a Compass run executed
eagerly while the deployment replayed a captured graph. Measured on Qwen3-0.6B
decode, TP=1: **3.24 ms** replayed against **28.78 ms** eager — 8.9x, which
reached the end-to-end comparison as +800% TPOT.

`measure` now captures for real; `predict` skips because no kernels run; `trace`
skips because a replay is one opaque submission and would record no operators
at all. Operator sequence from eager execution, cost from a measure run.

    TTFT   +107%  ->  +39.6%
    TPOT   +800%  ->  +13.1%

Still open in this area: trace mode's graph is the eager operator sequence, so
it does not describe the replayed launch. That is inherent — a replay has no
per-operator events — and the answer is to carry replay overhead as a term the
oracle applies rather than as operators in the graph.

### A2. Derivation is uncompiled; production is not — **L**

Derivation runs the model eagerly on meta. At `--level 3` inductor fuses
operators and eliminates views and allocations, so a derived graph and a
captured one differ by construction at the default level: 386 operators against
330 on Qwen3-0.6B.

Neither is wrong, but they cannot be compared, and the sweep story depends on
being able to compare them.

Two routes, neither cheap. Run dynamo and inductor during derivation, which
means compiling for a device the derivation does not have. Or model the fusion —
predict which operators collapse into one kernel and what that kernel costs —
which is a research problem, not an implementation one.

Interim: derive and capture at matched levels, and be explicit that a level-0
validation does not transfer to level 3.

### A3. A custom op's inner Triton kernel is recorded only on hardware — **S**

On a real device `aiter::masked_embedding` both dispatches and launches
`triton::_masked_embedding_kernel`, so the capture holds both. On meta the
kernel never launches, so derivation holds only the outer operator. Harmless for
cost (the outer operator carries the shapes), but it is a systematic difference
between the two graphs and it will confuse anyone diffing them.

### A4. Simulated TP's `all_gather` is not the real one — **S**

At a physical width of one, simulated TP builds a zero-padded buffer
(`movedim`, `reshape`, `view`, `zeros`) where the real path uses `view.dtype`.
Seven operators out of 451. Bookkeeping either way, and possibly worth simply
accepting — but it should be a decision rather than a residue nobody looked at.

---

## B. Configurations that cannot be modelled at all

### B1. Asymmetric parallelism — **L**

Simulated TP refuses pipeline parallel, prefill and decode context parallel,
data parallel, DP-attention, TBO, EPLB and disaggregated prefill, because ranks
diverge and absent ranks cannot be faked. Compass inherits every one of those
limits.

This is the standing open risk, not a new one, and nothing in the field has
shipped a solution. Expert parallelism matters most: it is where the interesting
models are going, and where a rank's graph genuinely depends on which experts
its tokens chose.

### B2. Shape-changing collectives have no meta stand-in — **S/M**

`all_reduce` and `broadcast` preserve shape, so meta can hand back the input.
`all_gather` grows and `reduce_scatter` shrinks, so they raise rather than guess
— deliberately, since guessing "same shape" would corrupt every downstream shape
while appearing to work. TP alone does not need them; sequence parallelism and
expert parallelism do.

### B3. One group is resolvable; several are not — **M**

A collective names its group by elimination: with exactly one group of size
above one, there is nothing else it could have run on. With TP and EP together
the ambiguity is real and is recorded as `"?"`.

Settles it: intercept at the group object rather than the dispatcher.
`get_tp_group()` and its siblings know their own identity. This is also what B1
needs, so the two should probably be done together.

### B4. Only one step, and only a decode step — **M**

`trace_step` records exactly one step, in practice a small decode. A deployment's
cost is dominated by shapes never captured: prefill, chunked prefill, mixed
prefill/decode batches, and the long-context decode where attention stops being
cheap. Speculative decoding and MTP are entirely untraced.

### B5. Qwen3.8-27B still cannot be captured — **?** (environmental)

Its decode path JIT-builds AITER's gluon kernel and the build fails in this
image. The stated PoC target model is therefore unreachable here, while
Qwen3-0.6B works because it takes the prebuilt ASM path. Needs the `make` stderr
captured to tell a toolchain problem from an image defect.

---

## C. Performance

### C1. Derivation is too slow to run per step — **M**

A full meta forward is ~0.16 s against a modelled decode step of ~1 ms, so
deriving inside the serving loop would dominate what it measures. Needs an
exact-`GraphKey` cache. Open whether the hit rate is good enough in practice:
decode at a stable batch size should repeat shapes constantly, varied-length
prefill will not.

### C2. Capture pays for Triton autotuning — **S**

The first launch of each kernel benchmarks every candidate configuration, which
is why `trace_step` is never 1 and why capture takes minutes. Fine as it is;
worth knowing before anyone tries to capture a sweep.

---

## D. The product itself is not built yet

Everything above is about the graph being right. None of it is the cost model,
and the cost model is what Compass is for.

### D1. A calibrated oracle exists; analytical and empirical do not — **L**

`CalibratedCostOracle` fits prefill and decode separately against measured steps
and reproduces a serving run's latency to within about 10-20%. That is the F1
"calibrated" point. Analytical (cost from first principles) and empirical (cost
attributed per operator from a captured graph) are both unimplemented, and
nothing has yet been fitted to an **op graph** — the calibrated oracle uses only
the step's shape, so all the structural work on graphs is not yet feeding the
cost model at all.

### D4. Decode cost is not linear in batch size — **M**, ranked next

TPOT sits at +17.5% and it *rose* when the sweep widened, from +13.1%. A model
short of data improves when given more; a model whose form cannot fit its range
gets worse, because the extra range pulls the fit away from wherever it is being
asked about.

The physical reason is visible in the table: with CUDA graphs the replay is for a
**padded batch-size bucket**, not the actual batch, so cost steps at the capture
sizes (1, 2, 4, 8, 16, 32, ...) rather than rising smoothly. Measured means at
batch 1/2/3/4 were 4.0/4.6/5.0/4.1 ms — not monotonic, which a linear term
cannot represent.

Settles it: feature the padded capture bucket rather than the raw batch size,
and keep context linear within a bucket. `Config.capture_sizes` already declares
the ladder.

### D2. No collective cost model — **M**

Collectives are now recorded with their group and their bytes, which is what a
cost model needs, and there is no cost model consuming it.

### D3. Never validated against `benchmark_serving` — **L**

The PoC's own success criterion is end-to-end TTFT and TPOT agreement against a
real serving run. That comparison has not been attempted once. Everything
claimed so far is structural: graphs matching graphs, not time matching time.

---

## F. Found while closing the loop

Building the first end-to-end comparison surfaced these. Most are fixed; the
ones that are not are the interesting ones.

### F1. No way to tell the runner when to start measuring — **M**

`--warmup-prompts` served throwaway requests to get Triton autotuning out of
the way, and the runner measured them anyway, because a runner sees steps and
has no idea which request a step belongs to or that some of them were meant to
be ignored. The result was a 7.5 s prefill sample sitting in a table next to a
0.03 s one, and a cost model that split the difference.

Worked around by discarding the first step of each kind, which is blunt: it
drops a real sample when the workload has few, and keeps warmup steps when it
has many. What is missing is a measurement window the engine can open and close
across the process boundary — the same gap that makes it hard to calibrate one
phase of a deployment without the others polluting the table.

### F6. Calibration must bracket what it will be asked about — **S**, mitigated

ATOM batches several requests' prefill into one step, so what lands in the table
is the *batched* token count. A sweep of large prompts therefore produced only
large samples: seven of them, spanning 1753 to 16370 tokens. Asked about an
evaluation workload batching to ~520, the model extrapolated below everything it
had seen, where the intercept dominates and the slope does no work — it
predicted 66.8 ms for a 64-token prefill.

The sweep now runs twenty sizes from 8 to 1024, twice through. Twice because
**Triton autotunes per shape, not once per process**: the first visit to a shape
pays a benchmarking cost steady-state serving never pays again, and having both
visits lets outlier rejection see the difference rather than guess at it.

### F8. Calibration coverage has to bracket every dimension, not one — **S**, done

The same error twice, in two dimensions, an hour apart.

First prefill: ATOM batches several requests' prefill into one step, so a sweep
of large prompts produced seven samples spanning 1753-16370 tokens, and the
evaluation batched to ~520. The model extrapolated below everything it had seen
and predicted 66.8 ms for a 64-token prefill.

Widening the prompt lengths fixed that and left the sweep varying *only* length,
so it produced decode steps at batch sizes 1-4 — and the evaluation ran 8
concurrent requests. TTFT improved and TPOT got worse, from +13.1% to +22.5%,
because the decode model was now extrapolating in a dimension nobody had
thought to check.

Both are the same mistake: coverage was designed against the dimension that had
just caused a problem rather than against the model's feature set. The sweep now
varies length and concurrency together.

The general fix is not a better sweep, it is that **the oracle now says when it
is extrapolating**. A fitted model answers anything, confidently, including
questions its evidence does not cover, and a linear extrapolation is worst
exactly where the intercept starts to dominate. Warned once per kind and
direction, since a serving run asks thousands of times.

### F7. The fit is now resistant to one-off contamination — **S**, done

Least squares, then discard points whose residual is far outside the spread of
the rest, then refit. Spread measured by median absolute deviation, since the
contaminating points would otherwise inflate the very quantity used to detect
them. The number dropped is reported in `describe()` rather than logged and
forgotten: a fit that discarded half its evidence should not describe itself the
same way as one that kept it.

### F2. Prefill samples are scarce by nature — **S**, mitigated

A fixed workload prefills everything in one or two steps, so a table from it
holds two prefill rows and cannot support fitting three coefficients. Fixed by
calibrating on a sweep of prompt lengths and batch sizes rather than on the
evaluation workload — which also means the reported error is a generalisation
error rather than a fit residual.

Still thin for long contexts: nothing in the sweep goes beyond a few thousand
tokens, and that is where attention stops being cheap.

### F3. Warmup counted in steps destroys the data it was meant to protect —
fixed

`measure_warmup_steps` defaulted to 2 and prefill happens twice in a whole run,
so it discarded every prefill sample. The oracle then had nothing to fit, fell
back to a mean of zero samples, and predicted a TTFT of 0 ms against a real
7.6 s. Now counted per kind and defaulting to zero.

### F4. An oracle asked outside its evidence must refuse — fixed

The fallback for "no samples of this kind" was the mean of an empty list, which
is zero: a confident, precise, entirely fictional answer, and exactly the class
of failure this project keeps meeting. It raises now.

### F5. A real run under a virtual clock reports zeros — fixed

`measure` and `trace` perform the real forward, so the wall clock is the
truthful one, but the virtual clock was installed for any Compass run. Every
request came back with a TTFT of zero, which reads as a broken measurement
rather than a misconfigured clock. Real modes now keep the real clock.

---

## E. Process notes worth keeping

* Every defect in this project so far surfaced by running something, never by
  reading code. The meta-kernel worklist, the autotuning blowup, the passthrough
  all-reduce, the unpacking hang — all of them.
* A stub must honour its caller's return contract. Across a process boundary the
  breach becomes a hang, not a traceback.
* Validating against a convenient configuration (`--enforce-eager`, `--level 0`,
  one GPU) reads as validation and is not. Three separate bugs hid behind that.
* For a tool whose output is an artifact, a plausible artifact from a broken run
  is worse than a crash.
* Structural validation cannot rank its own findings. Nothing in this file had a
  defensible priority until something predicted time and was wrong by a
  measurable amount.
* A fitted model is confident everywhere, including outside its evidence. Twice
  now the error was not the fit but the range it was fitted over, and neither
  time did anything complain.
