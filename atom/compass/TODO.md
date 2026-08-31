# ATOMCompass — open problems

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

### A1. CUDA-graph replay is never observed — **M**

Trace mode stubs `capture_cudagraph`, so a traced run executes eagerly even when
CUDA graphs are enabled. The default configuration replays a captured graph
instead, and the whole reason it exists is to remove per-launch overhead.

So the op graph is faithful about *what* runs and silently wrong about *how* it
is launched. A cost model fitted to eager traces will overstate a replayed
step's launch cost, and the error grows as the batch gets smaller — which is
exactly the decode regime the 5x speed goal lives in.

Worth noting the shape of the problem: a replay is one opaque submission, so
tracing it operator by operator is not possible even in principle. The graph has
to come from the capture phase, and the *cost* has to account for replay
separately.

Settles it: trace during `capture_cudagraph` rather than during a steady-state
step, and measure replay overhead against eager for the same batch so the oracle
can carry the difference as a term.

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

### D1. No real cost oracle — **L**

`ConstantCostOracle` returns a fixed number per prefill and per decode. The whole
F1 axis — analytical, calibrated, empirical — is unimplemented. Nothing has been
fitted to any captured graph.

### D2. No collective cost model — **M**

Collectives are now recorded with their group and their bytes, which is what a
cost model needs, and there is no cost model consuming it.

### D3. Never validated against `benchmark_serving` — **L**

The PoC's own success criterion is end-to-end TTFT and TPOT agreement against a
real serving run. That comparison has not been attempted once. Everything
claimed so far is structural: graphs matching graphs, not time matching time.

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
