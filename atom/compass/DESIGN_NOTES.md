# ATOMCompass — findings and deferred work

Recorded as they were established, so the reasoning survives the context it was
found in. Each item says what is known, what is not, and what would settle it.

## Deferred: graph derivation is too slow to run per step

A full meta forward of Qwen3.8-27B (64 layers, 2999 operators) takes **~0.16 s**.
A modelled decode step is on the order of a millisecond, so deriving the graph
inside the serving loop would dominate the very thing being measured and put the
5× speed goal out of reach.

The graph only changes when the batch shape changes, so it should be derived
once per distinct shape and reused. Per the earlier decision, the cache key is
the **exact** `GraphKey` — model, topology, rank coordinates, batch signature —
not a quantised bucket. Bucketing trades accuracy for reuse, and the accuracy it
trades is precisely the batch-skew sensitivity that this design keeps: a decode
batch mixing short and long histories does not cost what its mean history would
suggest.

Open: whether an exact-key cache hits often enough in practice. Decode steps at
a stable batch size should repeat shapes constantly; prefill with varied prompt
lengths will not. If the miss rate turns out high, the options are incremental
re-derivation for the part of the batch that changed, or admitting a bucketing
scheme with a **measured** error budget rather than an assumed one.

Not on the critical path for correctness — only for speed. Deferred until the
empirical oracle exists and there is something worth timing.

## Settled: Triton kernels need interception, not shape inference

Triton launches bypass the dispatcher, so no meta kernel can stand in for them,
and on meta they fail outright: there is no storage behind the pointers.

They do not need shape inference. AITER's Triton kernels take their destination
as an argument — `run_pa_decode_gluon(output, q, k_cache, ...)` — so the caller
has already allocated every output with the right shape before the launch.
Tracing one is therefore: record what it was asked to do, and skip it.

Assumption worth re-checking if a kernel ever traces wrongly: that every
intercepted kernel is out-parameter style. One that allocates internally and
returns a tensor would need real handling.

## Settled: the meta-kernel worklist is empty for this model

Every AITER operator Qwen3.8-27B reaches already runs on meta —
`gemm_a16w16`, `_fused_qk_rmsnorm_group_quant_kernel`,
`linear_attention_with_output_base`, `silu_and_mul`,
`unified_attention_with_output_base`. The earlier estimate of "22 meta kernels
to write" was wrong in both directions: fewer are missing, and the real barrier
was a different mechanism entirely.

This is model-specific. AITER registers operators lazily through JIT, so the
worklist for a different architecture — a MoE model especially — has to be
discovered by running the probe against it, not predicted.

## Open risk: asymmetric parallelism

ATOM already simulates a **TP** width wider than the physical device count
(`atom/distributed/simulated_tp.py`): the group reports the logical width so
layers shard that many ways, while collectives cover only the ranks that exist.
Its own warning says the model output is then meaningless — which costs Compass
nothing, since Compass never uses the output and models collective cost from
bytes, group size and interconnect rather than performing collectives.

`_reject_unsupported` in that module refuses pipeline parallel, prefill and
decode context parallel, data parallel and DP-attention, TBO, EPLB, and
disaggregated prefill. That line is drawn in exactly the same place this project
drew it independently: TP is symmetric and simulable on fewer devices; expert
parallelism, DP-attention and prefill/decode disaggregation are not, because
ranks diverge and virtual time would have to be coordinated across processes.

The symmetric half looks close to solved. The asymmetric half remains the real
open risk, and nothing in the field has shipped a solution: Revati claims a
multi-process timekeeper and released no code, and LLM-Emu avoids the problem
entirely by hooking a single-process executor.

## Settled by failure: tracing must go through the runner, not the model

Driving `model(input_ids, positions)` directly traces *a* forward, but not the
forward ATOM would run. On real hardware it fails at
`fwd_ctx.context.is_dummy_run`: attention reads a forward context that only the
runner establishes, carrying attention metadata, KV-cache state and the
dummy-run flag. Building those by hand means reimplementing `prepare_inputs`,
which is exactly the re-implementation this design exists to avoid.

So both sides of the comparison have to enter through `ModelRunner.forward`.
`dummy_execution()` already does this: it assembles a `ScheduledBatch` with
`is_dummy_run=True` and calls `self.forward(...)`, letting ATOM set up
everything. Capture is then a real runner on hardware; derivation is the same
runner with `_build_and_load_model` overridden to construct on meta and skip
weight loading — an override the base method's own docstring invites.

**The 2999-operator meta graph recorded so far is therefore provisional.** It is
reproducible and it proved the tracing machinery works end to end — dispatcher
and Triton interception, persistence, comparison — but it came from a bare model
call, so it is not yet the graph a served batch produces. It should not be
treated as a reference artifact until it is re-derived through the runner.

A related trap, worth keeping: built at the default fp32 rather than the model's
`torch_dtype`, meta traces happily while hardware refuses — AITER's fused
qk-rmsnorm takes fp16/bf16 only. Meta accepts kernels real devices reject, so
dtype has to be pinned deliberately on both sides. The diff caught this, which
is some evidence it is worth having.

## Settled: never trace the first forward — Triton autotunes on it

Tracing step one of a real run recorded **90,838 operators**. Tracing step two
recorded **101**. The difference is Triton autotuning: on a kernel's first
launch it benchmarks every candidate configuration, so the first step contains
tens of thousands of launches that steady-state serving never performs —
`chunk_fwd_kernel_o` alone appeared 34,269 times, `chunk_scaled_dot_kkt_fwd_kernel`
26,528. It is also slow, taking minutes of wall time.

`CompassConfig.trace_step` therefore defaults to 2 rather than 1. Anything that
captures a graph, times a step, or calibrates a cost model has to step past
warmup first, and a tool that quietly recorded step one would produce numbers
that look precise and describe nothing that happens in production.

Meta never revealed this, because skipped launches never autotune. It is a case
where hardware capture told us something derivation could not — which is an
argument for keeping the comparison even after meta is trusted.

## Resolved: the small captured graph was an artifact of a crash

The 101-operator capture recorded about three layers of a 64-layer model. It was
not truncation by `torch.compile` — compilation is off at level 0 under
`--enforce-eager`. The traced forward **crashed**, and the recording was written
anyway from a `finally` block, producing a well-formed artifact from a failed
run. Nothing downstream would have noticed: it is structurally valid and merely
wrong, and would have costed out at a fraction of the model.

Two fixes. A failed forward now writes no graph at all and says so loudly. And
what does get written is checked against the model's depth first — attention
runs once per layer, so a graph holding fewer attention operators than the model
has layers is reported as truncated rather than trusted.

The general lesson is worth keeping: for a tool whose output is an artifact, a
plausible artifact from a broken run is a worse failure than a crash, because it
propagates silently into everything fitted against it.

## Blocked, but only for one attention path: the AITER gluon JIT build

Running Qwen3.8-27B's real forward reaches AITER's *gluon* paged-attention
decode kernel, which JIT-builds on first use, and the build fails:

    subprocess.CalledProcessError: Command '['make', 'build', '-j1']'
    returned non-zero exit status 2

This is environmental and narrower than it first appeared. Qwen3-0.6B takes the
**ASM** decode path instead, which ships as a prebuilt `.co`, so it captures
without touching the failing build. The blocker is one kernel path, not capture
as a mechanism — which is why validation was possible on a smaller model while
the 27B path stays blocked.

Worth noting what capture costs even when it works: Triton autotunes the prefill
path for several minutes before the first traced step. That is an argument for
meta derivation rather than against it — capture is the expensive side.

## Settled: derivation reproduces hardware, and the check is now repeatable

For Qwen3-0.6B at TP=1, on a decode step of one token, all **338** derived
operators appear in the capture in order, with identical shapes *and* dtypes.
The 48 operators the capture holds in addition are the runner's own work —
batch-metadata slicing and copies, two hand-written Triton metadata kernels, the
LM-head GEMM, sampling, and the transfer home. None of them belong to the model
body, and the model body is what a cost model is fitted against.

The check is `compass_graph_diff.py compare`, and its question is **containment,
not equality**. A positional diff is right between two graphs of the same kind
and wrong here: a derivation is the model body alone, a capture is the body plus
the runner around it, so compared position by position they disagree from the
first operator while in fact agreeing about everything that matters. Matching is
greedy and fails closed — on divergence it reports every subsequent derived
operator as unfound rather than hunting for a later match that might be a
coincidence.

Two conditions have to hold or the comparison is meaningless, and both are now
enforced rather than remembered:

* **The capture must be uncompiled.** See below.
* **Both sides must describe the same batch.** Derive at the token count the
  capture used; otherwise shapes differ for reasons that carry no information.
  The tool warns when the batches disagree.

## Corrected: a compiled capture is a different graph, not an incomplete one

An earlier version of this note claimed compilation silently dropped 57 of 386
operators. That was wrong in substance, and the correction matters because the
wrong version argued for capturing at `--level 0`, which is not the
configuration anybody deploys.

Inductor kernels **are** traced now: `CachingAutotuner.run` is intercepted
alongside `JITFunction.run`, because inductor launches its generated kernels
through the former and a graph built only from the dispatcher and `JITFunction`
would miss them. That interception was the one real gap, and it accounted for
exactly one operator: `aten::embedding` becomes
`inductor::triton_poi_fused_embedding_0`.

The other 56 are `split_with_sizes` and `empty` — views and allocations, which
inductor resolves into offsets and a buffer plan rather than executing. They do
not run at level 3. Their absence is the compiled configuration being faithfully
recorded, not a hole in the trace. The compute totals confirm it: 283 operators
either way.

So a compiled and an uncompiled capture are both correct and describe different
configurations. Compilation is the default, so it is the one worth modelling,
and a derivation may only be compared against a capture taken at its own level.
`--enforce-eager` does not change this; it disables CUDA graphs, and compilation
is `--level`.

Open: derivation still runs uncompiled. Comparing it to a compiled capture means
either running inductor during derivation or modelling the fusion itself —
predicting which operators collapse into one kernel and what that kernel costs.
Unsolved, and the largest remaining gap between derivation and production.

## Settled: the derivation must use the engine's input dtypes

`input_ids` is `int32` and `positions` is `int64` (`model_runner.py`, lines 189
and 1277). They are easy to get wrong in the same way, and the consequence is
out of all proportion to the cause: a derivation using PyTorch's `int64` default
differs from the capture at the embedding, and one using `int32` for both
diverges at the first attention operator. Because matching fails closed, either
mistake rejects every operator from that point on — a correct derivation
reporting as a total structural disagreement.

Both live in `derived_inputs()` now, so there is one definition rather than a
literal at each call site.

## Settled: each rank writes its own graph

Every rank traces, and under any parallelism their graphs differ — that
difference is precisely what records how the model is sharded. A single
`--compass-graph-out` path made the ranks race for it and left one file naming
no rank: the survivor could not be attributed, and the others were lost without
a trace. The path now carries the rank's coordinates in every group it belongs
to (`g.json` becomes `g.tp1.json`, or `g.dp3-tp1.json`).

## Settled: collectives now name their group, where the group is determinable

The op graph's one concession to parallelism is that a collective names the
group it ran on. It was being recorded as the literal string `"unknown"` for
every collective, which quietly hollowed that out: a graph in which every
collective is indistinguishable cannot tell an all-reduce over tensor ranks from
one over expert ranks, and that distinction is the reason the representation is
shaped this way.

The dispatcher does not hand us the group, but it does not need to when the rank
belongs to only one group of size greater than one — there is nothing else the
collective could have been. At TP=2 all 57 all-reduces and the one broadcast now
resolve to `tp`. With several non-trivial groups the ambiguity is real and is
recorded as `"?"` rather than guessed.

Open: resolving the ambiguous case means intercepting at the group object rather
than the dispatcher. ATOM routes collectives through `get_tp_group()` and its
siblings, which know their own identity. That is the replacement for the current
resolver, not an addition to it, and it is what asymmetric parallelism (EP, DP
attention, P/D) will require.

## Settled: TP=2 shows the abstraction doing its job

Captured at TP=2, with nothing in Compass that knows what tensor parallelism is:

* the qkv GEMM narrows from `[4096, 1024]` to `[2048, 1024]` — the sharding is
  visible in the shapes, exactly as the design intends
* 57 `all_reduce_` collectives appear where TP=1 has none
* the embedding switches from `aten::embedding` to `aiter::masked_embedding`
  plus its Triton kernel, which is the vocab-parallel path
* the two ranks' graphs are identical, which is correct for symmetric TP and is
  the reason rank attribution has to come from the filename and key rather than
  from the contents

## Open: deriving a sharded graph from a single process

Derivation is validated at TP=1. At TP=2 it hangs: `init_dist_env(2, ...)` from
one process waits forever for a rank that will never arrive. This is the gap
between "derive a graph for a configuration nobody has run" and what the tool
does today, and it is the whole point of derivation — sweeping TP without a GPU
per point.

ATOM already has the mechanism — see *Open risk: asymmetric parallelism* above:
`atom/distributed/simulated_tp.py` reports a logical group width wider than the
devices present, so layers shard that many ways without the ranks existing. That
is exactly what a single-process derivation needs, so this is wiring rather than
invention, and it is bounded by the same line that module already draws: TP is
symmetric and simulable, the asymmetric strategies are not.

## Settled: no fixed ports

The rendezvous store was hardcoded to a port. The container runs with host
networking on a machine shared with about twenty others, so a fixed number
collides with whatever holds it — including an earlier run of the same script —
and fails with an `EADDRINUSE` that says nothing about tracing. The OS picks the
port now.


## Settled: a sharded rank can be derived from one process

TP>1 derivation used to hang, and the hang was only ever on the derivation side:
capture at TP=2 works fine, because it really does start two processes. A single
process asking gloo for a world of two waits forever for a peer nobody launched.

The fix is ATOM's own simulated TP, which reports a logical group width over a
smaller real group. One correction was needed to use it: `apply_simulated_tp`
decides the physical width from `torch.cuda.device_count()`, which is the right
question for a worker holding a device and the wrong one for derivation, whose
real group is always exactly one rank however many GPUs exist. On an 8-GPU box
that heuristic concluded a TP2 derivation needed no simulation, and the model
then built unsharded — silently, and looking normal.
`simulate_group_width()` states the widths outright instead.

That surfaced a second problem. At a physical width of one there is no
communicator, so simulated TP replaces `all_reduce` with a passthrough — right
for benchmarking kernels, wrong here, because the derived graph then contains no
communication at all and tensor parallelism costs out as free. Collectives are
now recorded rather than performed, under the same name the dispatcher records
on hardware, so the two graphs compare operator for operator.

With both in place, a TP=2 rank derived on meta in one process reproduces the
TP=2 capture: all 395 operators, including all 57 all-reduces. Deriving a
configuration nobody has run — the point of the whole exercise — works for
symmetric TP.

## Settled: the whole step, at any TP width, on one device

Compass models the serving flow, not the model body, so the runner's own work —
batch-metadata preparation, the LM head, sampling, the transfer home — is inside
the scope, not outside it. Calling those operators "not part of the model body"
excused the gap rather than closing it.

The route that closes it is not a meta runner. It is capture under simulated TP:
the runner runs for real, so every operator it performs is recorded, while
`--fake-eplb` makes one device stand in for a TP-N deployment. A TP2 capture on
a single GPU produces **451** operators against **448** for the real two-GPU
capture, including all 57 all-reduces and every runner operator.

One fix was needed to get there, the same one derivation needed: simulated TP
replaces `all_reduce` with a passthrough at a physical width of one, so a graph
captured on a single device showed no communication whatsoever. Collectives are
now recorded in the capture path too — a no-op on a real multi-device run, where
the collective dispatches and is recorded once already.

The residual seven operators are simulated TP's own `all_gather`, which builds a
zero-padded buffer (`movedim`, `reshape`, `view`, `zeros`) where the real path
uses `view.dtype`. That is the module's documented behaviour for absent ranks
rather than a tracing gap, and it is bookkeeping either way.

What this leaves is a division of labour worth stating plainly. **Capture** on
one device gives the complete step for any symmetric TP width, and is what a
cost model should be fitted against. **Derivation** on meta gives the model body
with no device at all, is roughly a thousand times faster, and is what a sweep
should use once the two are known to agree.

Still open: derivation does not produce the runner's operators, so the
derivation-vs-capture check remains containment rather than equality. Since
capture now covers the configurations a sweep would want, this is a performance
question rather than a coverage one.

A smaller instance of the same gap: on hardware a custom op both dispatches and
launches its inner Triton kernel, so the capture records
`aiter::masked_embedding` *and* `triton::_masked_embedding_kernel`. On meta only
the outer operator is recorded, since the kernel never launches.

## Open: predict mode has only been validated without CUDA graphs

The step-1 gate passes with `--enforce-eager` and hangs without it, on an AITER
shared-memory broadcast. CUDA graphs are the default, so every Compass run to
date has been in a configuration that is not the deployed one.
`capture_cudagraph` is a no-op in Compass, and the likely cause is the engine
waiting on a capture that never happens. Same class of problem as the
compilation one: validated against a convenient configuration rather than the
real one.
