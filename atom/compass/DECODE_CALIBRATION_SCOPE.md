# Native decode: what the graphs actually dispatch, and the vectors that would price it

Scope of this document. It is an audit of the decode path against the derived
graphs, and a proposal for the exact vectors that would calibrate it. Nothing
here is fitted, nothing here was measured on a GPU for it, and no frozen
artifact is touched. H1-H8 and every calibration artifact stand as they are.
Where a number below comes from an existing full-run step regime it is
development evidence for *what to collect*, never a fit to the final target.

Read-only evidence throughout. The one thing executed was a CPU probe
(`agent_scratch/cc_head/.tmp/probe_decode_regime.py`, git-ignored) that loads a
recorded decode graph and asks the attention family what it answers; its output
is quoted in §2.1.


## 1. Which decode kernel the acceptance configuration dispatches

**Retraction.** An earlier revision of this section read the kernel off the
operator census of `agent_scratch/g4/s27decode4.tp1.r0.json` -- 16
`aiter::unified_attention_with_output_base` calls for 16 full-attention layers
-- and concluded that aiter's Triton `unified_attention` ran. That does not
follow, and the census cannot support it.

`aiter::unified_attention_with_output_base` is not a kernel. It is the generic
per-layer graph-splitting boundary, registered at
`atom/model_ops/base_attention.py:347` with `@mark_spliting_op(is_custom=True)`,
and its whole body is

    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.impl.forward(query=q, key=k, value=v, ...)

Every `PagedAttentionImpl` call goes through it -- rope cache, KV insert,
dispatch, kernel -- and Dynamo is told explicitly not to look inside
(`base_attention.py:344-346`). So the name identifies the *layer wrapper*, and
one census entry per full-attention layer is what that wrapper produces on
every branch equally. No serialized operator record can name the callee,
because the callee is not in the graph. Pricing's independent resolution --
`AiterBackend` / `PagedAttentionImpl`, bf16 packed K/V,
`ATOM_USE_UNIFIED_ATTN=False` -- is consistent with this and with nothing the
census said.

**What the predicate resolves to, traced forwards.** `attention_mha.py:864
_dispatch_decode` reads four facts, in order:

| fact | value here | source |
| --- | --- | --- |
| `self.sliding_window` | `-1` | no `sliding_window` key in the model config |
| `envs.ATOM_USE_UNIFIED_ATTN` | `False` | `envs.py:342`, default `"0"`; not set by the plan |
| `envs.ATOM_FORCE_ATTN_TRITON` | `False` | `envs.py:345`, default `"0"` |
| `self.head_dim` | **256** | `config.json` `text_config.head_dim` |
| `self.use_flash_layout` | `False` | `attention_mha.py:116`; assigned `False` at every write site in the tree and `True` at none |

`use_triton_attn = ATOM_FORCE_ATTN_TRITON or sliding_window != -1 or head_dim
!= 128` (`:228-233`) is therefore **True**, and for the third reason, not the
first two: this is a 256-wide head. `_dispatch_decode` returns
`self.paged_attention_triton` before it ever reaches
`use_pa_decode_bf16_asm()`.

Inside `paged_attention_triton` there is a second fork (`:515`). With
`ATOM_USE_UNIFIED_ATTN` False and `use_flash_layout` False, the
`unified_attention` call at `:523` is *not* taken. The else branch at `:545`
runs, and it calls `run_pa_decode_gluon` (`:585`) ->
`torch.ops.aiter.pa_decode_gluon` (`base_attention.py:104`).

So the predicate resolves to the **Gluon paged decode**, and the earlier
conclusion was wrong about which fork as well as about the evidence for it.

**Status: predicate-resolved, callee not yet observed.** Every input above is
read off a config file, an env default or a static assignment, which is enough
to say what the branch must be and not enough to say what ran. ASM, unified and
Gluon all stay UNPROVEN in
`atom/compass/core/cost/families/attention.py`'s table until a run evidences
one. The cheap way to evidence it, when a GPU run is next authorised: log
`type(impl)`, `impl._dispatch_decode().__name__`, `impl.head_dim`,
`impl.sliding_window`, `impl.use_flash_layout` and `envs.ATOM_USE_UNIFIED_ATTN`
once per full-attention layer at bind time. That is an observation of the bound
callee, and it costs one line in the startup path.

Two consequences for the model, both already acted on and neither dependent on
the branch being settled:

* `DECODE_KERNELS` maps `paged_gluon -> unified.decode.paged_gluon`, which is
  the regime whose features the acquisition in §5 would fill. The
  `unified.decode.unified_attn` regime is not the acceptance path on this
  reading and its `grid_pad_rows` law (§2.3) is not what the PoC needs first.
* `max_seqlen_k` is **not read at all** on the Gluon branch. `:545-607` passes
  `context_lens`, `block_tables`, `max_seqlen_q`, `max_context_partition_num`
  and `context_partition_size`, and no maximum history. The FULL/eager
  divergence in §2.2 is therefore a *graph-matching* defect -- `max_seqlen_k` is
  in the operator identity key (`core/cost/identity.py:38`) -- and not a
  mispriced kernel. It is fixed as such.

Two configuration inputs are inherited silently rather than declared, and
neither is in the plan's engine arguments.
`scripts/compass/cc_traces_plan.py:143-151` passes only
`--gpu-memory-utilization 0.90 --max-model-len 262144
--no-enable_prefix_caching --max-num-seqs 32`. It does not pass `--block-size`
and it does not pass `--cudagraph-mode`:

* **`--block-size`.** `atom/config.py:1543` defaults `kv_cache_block_size = 16`.
  It does **not** move the decode dispatch here -- the `== 256` test is inside
  the `ATOM_USE_UNIFIED_ATTN` arm, which is off, and `head_dim != 128` takes
  the branch before it either way. (An earlier revision claimed 256 would
  dispatch `paged_attention_persistent_asm`; that followed from the retracted
  reading above and does not hold.) What it does move is the KV geometry and
  the host work: at 256 or 1024 `aiter_attention.py:1135` and `:1371`
  additionally call `set_aiter_persistent_worker_buffers(bs)` on every decode
  step *and* at capture, building `work_indptr` / `work_info_set` /
  `reduce_indptr` / `reduce_final_map` / `reduce_partial_map` through
  `aiter.get_pa_metadata_v1` -- host work that happens whether or not the
  dispatch then uses it.
* **`--cudagraph-mode`.** The engine CLI defaults it to **`FULL`**
  (`atom/model_engine/arg_utils.py:456-465`; the plan launches `python -m
  atom.entrypoints.openai.api_server` at `cc_traces_plan.py:238`, and that entry
  point builds its parser from `EngineArgs.add_cli_args`, `api_server.py:2930`).
  The `PIECEWISE` default at `atom/config.py:1957-1958` applies only when
  `compilation_config.cudagraph_mode` is left `None`, which the CLI path never
  does. So the acceptance run captures and replays **whole-forward FULL decode
  graphs**, with attention inside the graph. That is the case where §2.2 and
  §2.3 bite, and it is the current path, not a hypothetical one. Corroborated
  independently: the CLI's `--cudagraph-capture-sizes` default
  `[1,2,4,8,16,32,48,64,128,256]`, narrowed by `--max-num-seqs 32`, gives
  exactly the `capture_sizes [1,2,4,8,16,32]` recorded in `target.json`.

Recommendation: declare both explicitly in `ENGINE_ARGS` rather than inherit
them. This is a plan input, not a code change, and it is the same class of gap
already recorded for the harness in `POC_STATUS.md`.

### 1.1 The launch extent is not `max(request contexts)`

Confirmed, and it is worth stating precisely because it is true for two
different reasons on two different branches.

**On the Gluon branch §1 resolves to, the extent is the row count and the split
count, and neither is `max(contexts)`.** `pa_decode_gluon.py:5342` takes
`batch_size = query.shape[0] // query_length` and `:5356` launches

    grid = (batch_size, num_kv_heads, max_context_partition_num)

with a second reduce launch at `:5580`, `grid = (batch_size, num_kv_heads, 1)`,
whenever `max_context_partition_num > 1` (`:5393`, `one_shot`). Three things
follow, and the first is the one the lead's warning names:

* **`batch_size` is the padded bucket under a FULL replay, not the batch.** It
  comes off the query operand, which the runner allocated at `running_bs *
  max_q_len`. Thirty-one requests replaying a bucket of 32 launch 32 rows.
* **`max_context_partition_num` is a function of that row count**, through
  `get_recommended_splits(num_seqs, num_kv_heads)` (`attention_mha.py:552`,
  `pa_decode_gluon.py:111`): `min(8, cdiv(num_sm * occupancy, num_seqs *
  num_kv_heads))`, occupancy 2. With `num_kv_heads = 4` that is `min(8,
  cdiv(2 * num_sm / 4, num_seqs))` -- a *step* function of the row count, so
  the padded row does not merely add a row of work, it can change the split
  count for every row. At 31 rows against 32 the two differ.
* **`context_partition_size` is a constant 256** on this deployment
  (`attention_mha.py:554`; the 128 case is sliding-window only), so the
  per-sequence tile count is `ceil(context / 256)` and the family's existing
  `context_tiles` term is the right shape for it.

`num_sm` is a device property read at dispatch time
(`torch.cuda.get_device_properties()`), so the split count is not derivable on
a CPU host without declaring it. It is a static per-device fact, not a
per-step one; the proposal in §5 treats it as a declared scope input in the
same class as `topology`, and no step in this document depends on its value.

On the unified/flash branch -- which this deployment does **not** take, and
which is retained here because a `--block-size` or env change would reach it --
the extent is inside aiter's `unified_attention` and is driven by
`max_seqlen_k`:

* `unified_attention.py:331` -- `use_2d_kernel(...)` returns true when
  `max_seqlen_k <= 512`. Below that threshold a single 2D kernel runs and there
  is no reduce; above it the 3D split kernel runs and a reduce follows. **These
  are different kernels, not the same kernel at a different size.**
* `:211` and `:246` -- `MAX_SEGMENTS = ceil(max_seqlen_k / TILE_SIZE)`, capped
  at 128 on the 3D path, and `num_segments` is clamped to it and rounded up to a
  power of two. At `block_size=16`, `TILE_SIZE = min(64, next_pow2(16)) = 16`.
* `:910` -- `grid = (NUM_KV_HEADS, total_query_blocks)`, with the segment count
  entering as the third launch dimension on the split path.

On the persistent branch, which this deployment does not take but which a
`--block-size 256` would, the extent is not an argument at all:
`attention_mha.py:648 aiter.pa_persistent_fwd(...)` takes no grid and no
partition count. The whole decomposition lives in the `work_*` arrays that
`get_pa_metadata_v1` computed from the *distribution* of per-request KV blocks.
Two batches with the same summed context and different raggedness get different
`work_indptr`, and no function of `max(contexts)` recovers it.


## 2. Where the derived graph and the native metadata disagree

Four concrete items. §2.1 blocks the decode family outright today. §2.2 and
§2.3 follow from the mode the acceptance run actually gets, `FULL`, so they are
live rather than hypothetical. §2.4 is an operand-shape difference that is
probably benign and is listed so it is not rediscovered later.

### 2.1 `capture_bucket` is not in the operator key, so the decode family refuses every existing graph

`attention.py:304` builds `Structure(bucket=ctx.get("capture_bucket"))` from the
operator's recorded `context`, and `features_for` refuses without it
(`:700-706`): *"the replay bucket this call ran at was not recorded, so how many
padded rows the kernel executed is unknown; that is a question for the padding
owner, not a zero"*. That refusal is right. But nothing puts `capture_bucket`
into an operator context. It exists on `StepShape`
(`atom/compass/core/cost/base.py:61`) and on `BatchSpec`
(`runtime/batch_spec.py:121`); the tracer's per-operator context
(`runtime/forward_ctx.py:93`) does not carry it, and neither does
`BatchSpec.attention_context` (`batch_spec.py:419-465`).

Run against the real recorded graph, with the kernel declared:

    == aiter::unified_attention_with_output_base
       structure: seqs=4 bucket=None contexts=(66, 66, 66, 66)
       regime_of {'attention_backend': 'unified_attention'}
           -> Regime('unified.decode.unified_attn')
       features: Refusal('the replay bucket this call ran at was not
           recorded, so how many padded rows the kernel executed is unknown')

    == aiter::linear_attention_with_output_base
       structure: seqs=4 bucket=None contexts=() actual=4 out_rows=4
       regime_of -> Regime('gdn.decode')
       features: [1.0, 4.0, 0.0]

So: **GDN decode prices; full-attention decode does not, on any graph we hold.**
`unified.decode.unified_attn` has a declared law and has never been able to emit
a feature vector. That is the first thing to fix, and it is a declared-field
change on both sides (native trace and derived context) rather than a new
measurement. It belongs to the padding owner, which after `d3afcbd5` is this
worktree's lineage -- but the field lands in Attention's key, so it needs
Attention's and Pricing's agreement on the name before either side emits it.
Proposed name `capture_bucket`, `None` for an eager step, matching `StepShape`.

Note also `actual=None out_rows=None` on the unified-attention row above. The
`unified.decode.unified_attn` law does not currently use either, so this is not
blocking; it is recorded because `bucket_pad` and a future `tail_pad_rows`-style
term would both want them and only GDN's wrapper records them today.

### 2.2 `max_seqlen_k`: the eager value and the captured value are different numbers

    aiter_attention.py:1100,1139   prepare_decode:            max_seqlen_k = context_lens.max()
    aiter_attention.py:1367        build_for_cudagraph_capture: max_seqlen_k = config.max_model_len

The derived side takes the eager value in both places it computes it --
`batch_spec.py:436` (`("max_seqlen_k", max(self.context_lens))`) and
`templates.py:488-492`, which cites `model_runner.py:3187` for it. That citation
is correct for an eager step and for a `PIECEWISE` one.

It is not correct for a `FULL` one, **and `FULL` is what the acceptance run
gets** (§1). `model_runner.py:4019-4031 capture_cudagraph` logs *"PIECEWISE
cudagraph: capturing per-piece graphs (attention eager); manual FULL
whole-forward capture disabled"* and the piecewise branch `continue`s past the
whole-forward `torch.cuda.graph` capture at `:4290`. Under `PIECEWISE`
attention is outside the graph and runs eagerly every step, so
`max(context_lens)` is what the kernel sees. Under `FULL` attention is inside
the graph, and the kernel that got recorded was selected by
`max_seqlen_k = max_model_len`. A replay re-runs the recorded kernel whatever
the refilled buffers say.

Concretely, at the declared `--max-model-len 262144` and `block_size 16`: every
`FULL`-captured decode graph took the 3D split path with
`MAX_SEGMENTS = min(128, 262144/16) = 128` and a reduce, including at buckets
whose real contexts are a few hundred tokens -- where an eager step takes the 2D
kernel and no reduce at all (`use_2d_kernel`, `max_seqlen_k <= 512`). **The
derived graph declares the eager kernel for a step that replays the captured
one.** That is the largest single divergence found.

How often the *kernel family* differs is a separate question and is not
answered here. The 2D/3D threshold reads `max(context_lens)` over the whole
batch, so a step falls below 512 only when every concurrent request does. On
the frozen short workload the per-request inputs are 256-2 560 with p90 768, so
a wide batch will usually exceed the threshold on its longest member alone and
the two paths would agree on family while still disagreeing on the segment cap
(128 captured, against `ceil(max(contexts)/16)` eager). A single-request or
narrow all-short step is where the family itself flips. Counting which steps
land where needs the replay schedule, not the manifest, and is left to the
acquisition rather than estimated now. Below 2 048 tokens of context the eager
cap `ceil(max(contexts)/16)` is itself under 128, so the captured graph can
only ever have more segments than an eager step would choose -- but both are
then clamped by an occupancy term, so whether the counts actually differ is a
measurement, not an inference from the cap.

The existing decode graphs are not invalidated by this -- they record
`max_seqlen_k = 66` against contexts `(66,66,66,66)`, so they were traced eager
or piecewise, consistently, and they remain valid development evidence for what
an eager decode costs. What follows is that they are not evidence for what the
acceptance run's decode step costs, and §5's vectors must be collected under a
declared mode.

### 2.3 `grid_pad_rows` is built on the eager grid

`attention.py:642-647`:

    grid_pad_rows = max(contexts) * len(contexts) - sum(contexts)

The module's own comment is explicit that this is *"a candidate law and nothing
more until a holdout at a structure it was not fitted on says otherwise"*, and
the development evidence behind it is striking -- a 32-sequence mixed batch
summing 394 164 context rows at ~3.70 ms against a balanced 32x16384 batch
summing 524 288 at ~0.998 ms. More rows, under a third the time. No law in
summed rows produces that.

But `max(contexts)` is the eager grid, and the acceptance run is `FULL` (§1).
Under a `FULL` capture the grid every sequence is covered by is forced by
`max_model_len`, not by the longest sequence in the batch, so the ragged term
degenerates on exactly the path that matters: every batch at the same bucket
gets the same `grid_pad_rows` regardless of its shape, and the term that carries
the 3.7x spread in the development evidence carries nothing.

Two ways out, and the choice is Attention's, not this document's: make the
extent capture-mode-aware (the batch already knows -- `BatchSpec` carries both
`max_model_len` and `capture_bucket`), or state that the law covers eager and
piecewise decode only and refuse a `FULL` step. What this document commits to is
that §5's vectors are collected under a declared `--cudagraph-mode` so the
question is decidable from measurement rather than settled by assumption.

A caution on the development numbers behind the law. The 394 164-row mixed batch
at ~3.70 ms against the balanced 32x16384 batch summing 524 288 at ~0.998 ms is
the observation the ragged term exists to explain. If those two were measured
eager, they say nothing about the `FULL` case; if captured, `grid_pad_rows`
could not have been the variable that separated them. Which they were is not
recorded in the regime comment and was not resolved here. **Resolving it is a
precondition for reusing that pair as evidence**, and it is cheap -- the
`max_seqlen_k` in their own keys answers it.

### 2.4 `block_tables` width

Native copies `bs` rows of a host buffer whose width is `max_model_len //
block_size` (`aiter_attention.py:1129`, `copy_to_gpu(bs)`); at the declared
262144/16 that is 16 384 columns, or 2 MiB of H2D per step at bucket 32,
independent of context. The derived side records `block_tables_shape = [bs,
width]` with the same `width` but flattens only `used = ceil(max(context) /
block_size)` columns of data (`batch_spec.py:450-465`). The *shape* agrees, so
the key agrees; only the recorded payload is narrower. Listed for completeness:
no refusal or mispricing follows from it today, and the shape is what the cost
identity reads.


## 3. What is already right, and stays untouched

* **Active vs bucket rows.** Settled by `d3afcbd5` and not reopened here.
  `attention_mha.py:670` takes `batch_size = context_lens.shape[0]`, the bucket
  width, and padded rows carry `context_len == 0`, `slot_mapping == -1`,
  `cu_seqlens_q` repeating the last real offset, GDN state indices `-1`. The
  derived side reproduces each of those tails separately rather than with one
  padding value.
* **GDN decode carries no full-history term, and the traced operands show why.**
  The recorded `linear_attention_with_output_base` call at bucket 4 has
  `input_shapes [[4, 10240], [4, 48], [4, 48], [4, 48, 128]]` and
  `output_shapes [[4, 48, 128]]`. Every dimension is rows, heads or head_dim;
  none is context. A DeltaNet decode reads the held recurrent state and writes
  it back, so its work is O(rows) and the `gdn.decode` law
  `("calls", "active", "tail_pad_rows")` is the right shape. This is the one
  decode regime that already emits a feature vector on real data -- `[1.0, 4.0,
  0.0]` above. The `tail_pad_rows` term is not idle padding-accounting: the
  wrapper slices to `num_actual_tokens` but then zeroes
  `core_attn_out[num_actual_tokens:]`, which is work over exactly those rows.
* **TP2/4 head placement.** `runtime/tracer.py:178-215` already refuses rather
  than guesses: only `decode AND cudagraph_mode == FULL AND
  tensor_parallel_size == 1 AND TBO off` projects the padded bucket inside the
  body graph; every other path slices hidden states to the scheduled rows first.
  So at TP2 and TP4 the head is *always* at scheduled rows, whatever the bucket.
  No vector below needs to establish that; the vectors need only avoid assuming
  the TP1 answer at TP2/4.


## 4. Feasible total KV footprints, from the deployment and the frozen workload

Both bounds are needed, and the tighter one is the pool, not the workload.

From `poc/g5_27b/target.json`: `num_kvcache_blocks = 112772`,
`max_num_seqs = 32`, `max_model_len = 262144`. At the default
`block_size = 16` the KV pool holds **1 804 352 tokens** in total. A single
request may occupy up to 16 384 blocks, so 32 concurrent max-length requests
would need 524 288 blocks -- 4.6x the pool. **This deployment can never run its
own `max_num_seqs` at its own `max_model_len`**, and any calibration vector that
assumes it will not schedule.

From the frozen manifests (`cc_traces_long.manifest.json`,
`cc_traces_short.manifest.json`), the decode contexts the final workload
actually reaches:

| | requests | input min / median / max | output min / median / max |
|---|---|---|---|
| long | 20 | 448 / 91 008 / 107 328 | 24 / 608 / 2 413 |
| short | 64 | 256 / 448 / 2 560 | 14 / 21 / 39 |

So a long request's decode context runs from its input length up to at most
107 328 + 2 413 = **109 741**, and a short one's from 256 to at most 2 599.
Twenty long requests over a 262.8 s arrival span, sixty-four short ones all at
t=0.

The binding combination: **balanced at bucket 32, the largest feasible
per-request context is 1 804 352 / 32 = 56 386 tokens** (56 320 rounding down to
a block multiple) -- roughly half the workload's longest request. A batch of one
109 741-token request plus 31 short ones needs 109 741 + 31x2 048 ~= 173 229
tokens, comfortably inside the pool. That asymmetry is not an inconvenience, it
is the structure the final workload has: the long class is never 32-wide, and
the ragged mixed batch is the one that occurs.


## 5. Proposed exact-vector set

Thirteen training vectors and four holdouts. Small on purpose: every vector
below isolates one term of an already-declared law, and none of them expands the
registered benchmark matrix -- this is a proposal for a *separate* decode
acquisition, to be registered as such if the Lead accepts it.

All decode, TP1 unless stated, one model, `block_size 16`,
`max_model_len 262144`, `--cudagraph-mode` declared explicitly and held fixed
across the set. Contexts are block multiples. The laws under test:

    unified.decode.unified_attn : context_rows, grid_pad_rows, active, bucket_pad
    gdn.decode                  : calls, active, tail_pad_rows

### Training

| # | batch | bucket | contexts | isolates |
|---|---|---|---|---|
| T1-T6 | 1, 2, 4, 8, 16, 32 | = batch | 1 024 each | `active` at every rung, `bucket_pad = 0`, `grid_pad_rows = 0` |
| T7 | 3 | 4 | 1 024 each | `bucket_pad = 1` at the commonest padded step |
| T8 | 31 | 32 | 1 024 each | `bucket_pad = 1` at the top rung |
| T9-T11 | 32 | 32 | 1 024 / 16 384 / 56 320 each | `context_rows` slope, balanced, to the feasible ceiling of §4 |
| T12 | 32 | 32 | one 56 320, thirty-one 1 024 | `grid_pad_rows` at its extreme |
| T13 | 32 | 32 | graded 1 024 x k, k = 1..32 | `grid_pad_rows` at an intermediate raggedness |

T1-T6 give `active` and the intercept with both ragged terms pinned to zero.
T7-T8 give `bucket_pad` against T2/T3 and T6 respectively, at both ends of the
ladder, which is what "3->4 and 31->32" asks for. T9-T11 give `context_rows`
with raggedness still zero. T12-T13 are the only vectors where
`grid_pad_rows != 0`, so the term is identified from two points and is the one
most in need of the holdout below.

Each vector prices the GDN regime at the same time at no extra cost: T1-T6 sweep
`active` with `tail_pad_rows = 0`, T7/T8 give `tail_pad_rows` directly.

### Holdout, frozen before measurement

Predicted with input hashes and frozen first, then measured -- the E7/E8
protocol, and for the same reason: a residual on the step it was fitted from is
not a prediction.

* **H-a. 3 -> 4 at context 56 320.** Padding crossed with a long context, a
  combination no training vector contains (T7 is padded and short, T9-T11 are
  long and unpadded). Tests that `bucket_pad` and `context_rows` are separable.
* **H-b. 31 -> 32, ragged, contexts drawn from the frozen long manifest's own
  histogram** (p10 48 960 / median 91 008 / p90 105 920, padded down to the pool
  bound of §4). Padding crossed with raggedness. This is the vector closest to
  what the acceptance run actually executes, and it is deliberately held out.
* **H-c. 8, balanced, context 32 768.** Between the trained knots on both axes:
  an interpolation check, and the cheapest of the four.
* **H-d. GDN decode at 31 -> 32, context 56 320.** The `gdn.decode` law claims
  no history term (§3). If that is right, H-d's GDN cost must match T8's to
  within noise despite 55x the context. If it does not, the claim is wrong and
  the law needs a term -- which is exactly what a holdout is for.

Reporting rule, stated in advance: each holdout is reported with its frozen
prediction, its measurement, and the signed error, **whether or not it passes**.
A miss is kept, as E8's was. No vector is re-run to replace a value that came
out badly.

### What this set deliberately does not cover

Said plainly so that nothing here reads as broader than it is.

* **TP2 and TP4.** Not in the set. The set is TP1 because the two open TP2/TP4
  questions -- the region term's width transfer, and the unstable rank-1 body
  price behind E8's +26.2% -- are not decode-kernel questions and would not be
  answered by decode vectors. What §3 records is that the *head placement* rule
  is already decided for TP2/4 and needs no vector. If the Lead wants the decode
  law itself checked at width, the cheapest addition is T6 and H-b repeated at
  TP2, and that should be a separate decision with its own cost.
* **Speculative decode**, `ReplaySSM`, mixed prefill/decode batches, and sliding
  windows. All four are named refusals in `regime_of` today and stay that way.
* **The ASM decode kernels.** They remain `UNPROVEN_DECODE_KERNELS`. This set
  does not price them and does not alias them onto the unified law.
* **Any refit of an existing artifact.** The set adds decode coverage; it does
  not touch a prefill fit, a region profile, or a frozen prediction.


## 6. Dependencies, in the order they block

1. **`capture_bucket` in the operator key** (§2.1). Until this lands on both the
   native trace and the derived context, `unified.decode.unified_attn` refuses
   every call and no vector in §5 can be turned into a feature row. Blocks
   everything else here. Needs Attention's and Pricing's agreement on the field
   name; the emission on the derived side is small.
2. **`--cudagraph-mode` and `--block-size` written into the plan's engine
   arguments** (§1). Both are inherited defaults today -- `FULL` and `16`. The
   values are fine; the silence is not, because a later default change would
   move the decode kernel and the `max_seqlen_k` without moving anything that
   gets reviewed. A plan input, not code.
3. **Attention's ruling on `grid_pad_rows` under `FULL`** (§2.3) -- either a
   capture-aware extent, or a stated eager/piecewise-only domain that refuses a
   `FULL` step. Blocked behind reading the `max_seqlen_k` in the two development
   measurements' own keys, which decides whether they remain usable evidence.
4. Then the §5 acquisition, training first, holdouts frozen before measurement.


## 7. What the launch-extent change implements

Dependencies 1 and 3 of §6 are closed by the change this section documents.
Dependency 2 remains a plan input and dependency 4 remains unstarted. No GPU
measurement was taken, no law was fitted, and no frozen artifact was touched.

**The extent is derived, not declared.** §2.1's refusal asked for
`capture_bucket` in the operator context, and adding it there was the obvious
fix and the wrong one: an operator's context is part of its identity key
(`core/cost/identity.py`), so a new field reindexes every price already on disk.
The rows are on the call instead, exactly as the kernel reads them --
`_output_rows` now answers for the unified wrapper from `q.shape[0] //
max_seqlen_q`, the expression at `pa_decode_gluon.py:5342`, and
`Structure.active_sequences` counts the rows whose query length is non-zero.
`bucket_pad` is their difference. A declared `capture_bucket`, where one is
present, is used only to *contradict* the rows and never to supply them.

Consequence for the family's contract, for Attention to confirm: `active`
continues to mean every launched row, including the padded ones, which is what
`len(queries)` has always returned on this path; the padding is carried
separately by `bucket_pad`. Two tests in `test_attention_family.py` encoded the
old contract and are rewritten -- one asserted a refusal that fired on every
decode call this family can build, and one declared a bucket of 8 over two
recorded rows, which is now the contradiction refusal.

**`max_seqlen_k` follows the mode.** `BatchSpec` gains the declared
`cudagraph_mode` -- `ShapeDeriver` already held it and put it on every
`TraceRequest` -- and three properties: `replays_captured_metadata`,
`launch_max_seqlen_k` and `launch_extent_scope`. A FULL decode replay records
the capture's `max_model_len` (`aiter_attention.py:1367`), an eager or
PIECEWISE step records the batch's longest history (`:1100`, `:1139`), and a
bucket with no declared mode records the batch value under the scope
`"undeclared"`, which says it is not evidence. Same convention as
`tracer.head_rows_padded`: `None` rather than a plausible number.

**The warm binder keeps it.** `bind_cohort` takes `extent_scope`, defaulting to
`"batch"` -- every eager and PIECEWISE step -- and `_bind` returns a captured
extent unchanged rather than recomputing it from the cohort. This is the same
failure `pad_rows` exists for: a correctly derived graph coming back narrowed
one cohort later. `"undeclared"` refuses.

Validated in `tests/compass/test_decode_launch_extent.py` at 3 -> 4 and
31 -> 32 under FULL, PIECEWISE and eager, on specs built by the real
`ShapeDeriver`; through `_bind` and `bind_cohort` for the warm path; on
serialized operator fixtures shaped as `forward_ctx` writes them, for the
executed rows, the padded-row count and the bucket contradiction; and on GDN
decode fixtures, which must not move -- the recurrence still prices `[calls,
active, tail_pad_rows]` off its own output operand and `num_actual_tokens`, and
still carries no full-history term.
