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

## Open: the captured graph looks too small for the model

The steady-state capture holds 101 operators with 3
`linear_attention_with_output_base` and 13 `gemm_a16w16`, which is on the order
of three layers, not the 64 this model has. Either the trace is being cut short,
or most layers execute inside a region that does not dispatch operator by
operator — `torch.compile(..., backend="eager")` is applied to the model in
`ModelRunner`, and a compiled region would explain both the low count and why
meta, which is not compiled the same way, saw 2999.

Unresolved. It matters, because a graph that silently omits most of the model
would cost out at a fraction of the truth while looking well-formed. Worth
checking before any cost model is fitted against these graphs: compare the
per-layer operator counts against the model's layer count, and fail loudly when
they disagree rather than trusting the artifact.
