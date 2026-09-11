# G4: predicting TP=2 and TP=4 from measurements that are not of TP=2 or TP=4

The PoC's central claim is that limited measurement plus ATOM's own serving
logic picks the right configuration *without running every candidate*. The
same-TP matrix cannot demonstrate that: each of its cells is predicted with a
step table swept on that very configuration, so it shows the oracle reproduces
what it was fitted to. Useful as a diagnostic ceiling; not the claim.

This file fixes, before the predictor is built, what it is allowed to read.
The list is the experiment. A transfer result whose inputs were decided
afterwards is not a held-out result.

## 1. The ledger

Three kinds of input, and the boundaries between them are the whole point.

### A. Source-configuration measurement -- ALLOWED

Measured on TP=1, the configuration being transferred *from*.

| artifact | what it is |
| --- | --- |
| `matrix/sweep_tp1_short.jsonl` | 379 timed steps at TP=1, short shapes |
| `matrix/tp1_short*/real_steps.jsonl` | TP=1 step tables |
| the per-launch overhead constant | fitted at TP=1 (2.02 us, see `priced.py`) |

There is no 27B TP=1 operator-price list. `compass_ops/silu_*` (without the
`27`) keys as `{'model_id': 'Qwen/Qwen3-0.6B', 'topology': [['tp', 1]]}` -- it
is the development model, not this one, and pricing the derived 27B TP=1 graph
against it matches 0 of 2999. Only `silu27_*` is the 27B, and those are TP=4.

### B. Hardware-primitive calibration -- ALLOWED, AND DECLARED AS SUCH

A price for one kernel at one shape, measured away from any serving run
(`atom/compass/runtime/microbench.py`: a few thousand calls inside one event
pair). It is a property of the chip and the kernel, reusable by any model that
issues that call. It is *not* a property of this deployment.

| artifact | what it is |
| --- | --- |
| `compass_ops/silu27_i1000_r*_prices.tp*.json` | 191 priced signatures per rank, 3 repeats |

Two things must be said plainly about this, not buried:

1. **Collecting it required standing the target width up on GPUs.** The price
   list for a TP=4 rank was taken inside a TP=4 model-runner process, because
   `aiter` registers its operators lazily and the tuned kernel for a shape is
   only selected there. So this is not free, and G5c must carry its cost. What
   it is *not* is a serving measurement: no scheduler, no batching, no
   workload, nothing end-to-end.
2. **It includes communication.** `aiter::all_reduce_` appears 129 times in the
   TP=4 rank graph and is priced at
   `aiter::all_reduce_|4,5120|bfloat16|#1=tp:0;#2=True;#3=False`. A TP=1 run
   contains no all-reduce at all, so this term cannot come from the source
   configuration by any route. It is an interconnect primitive -- message size
   and group width -- and it is declared here as one.

3. **What it does not cover, measured 2026-09-11.** The list was collected in a
   TP=4 process, so it holds TP=4 shard widths and nothing else. Pricing each
   derived rank-0 graph against `silu27_i1000_r1_prices.tp0.json`:

   | derived | priced | of | `gemm_a16w16` widths needed | in the list? |
   | --- | --- | --- | --- | --- |
   | TP=1 | 129 | 2999 | 5120, 14336, 16480, 34816 | no |
   | TP=2 | 258 | 3128 | 5120, 7168, 8240, 17408 | no |
   | TP=4 | 578 | 3128 | 3584, 4120, 5120, 8704 | yes (3584, 4120, 8704) |

   The list's widths are `[3584, 4120, 8704, 62080]`. So all 256 `gemm_a16w16`
   and all 64 `silu_and_mul` calls are unpriced at TP=1 and at TP=2 -- the
   matmul, which is the bulk of a step. **A TP=2 prediction cannot be made from
   the inputs declared above.** Recorded here before any TP=2 number was
   produced, so that the gap is a fact about the ledger and not a result being
   worked around.

   What transfers today without a new measurement is the width-independent
   part: `_fused_qk_rmsnorm_group_quant_kernel` matches all 129 at every TP,
   because its signature carries no shard dimension.

4. **A defect that would have made TP=2 look like TP=4.** `all_reduce_` signs
   identically at both widths:

   ```
   TP=2  aiter::all_reduce_|4,5120|bfloat16|#1=tp:0;#2=True;#3=False
   TP=4  aiter::all_reduce_|4,5120|bfloat16|#1=tp:0;#2=True;#3=False
   ```

   `signature_of` keys on shapes, dtypes, context, scalars and Triton grid. The
   group width appears in none of them, and `#1` is `group.unique_name`, which
   is `tp:0` however many ranks are in the group. So all 129 TP=2 all-reduces
   match the TP=4 entry and take the 4-way price -- silently, with full
   coverage, no warning. Point 2 above calls this primitive "message size *and*
   group width"; the key only carries the first. Until the key carries the
   width, an all-reduce price may not be transferred across TP.

   **Closed 2026-09-11, by refusing rather than by re-keying.** Putting the
   width into the signature would have invalidated every collective key already
   measured. The width is not a property of the operator anyway -- it is a
   property of the graph the operator came from and of the list the price came
   from -- so the check lives where both are in hand,
   `PricedGraphCostOracle._cost`: a collective is priced only when the graph's
   group widths and the price list's are the same, and otherwise is left
   unpriced, counted in `untransferable_collectives`, and warned about.

   `price_graph` now records the width it measured at. Lists written before it
   did are not thereby useless at their own width: they name the graphs they
   were priced from, a graph has always carried its topology, and
   `_declared_topology` reads it back from there -- a statement the artifact
   already makes, so no existing artifact was edited. When those graphs are
   missing or disagree, the list certifies nothing and its collectives are
   refused.

   Measured against `silu27_i1000_r1_prices.tp0.json`, which declares no width
   of its own and resolves to `{'tp': 4}`:

   | derived | priced | refused collectives | priced kernel seconds |
   | --- | --- | --- | --- |
   | TP=1 | 129 | 0 (the graph has none) | 3.49e-4 |
   | TP=2 | 129 | 129 | 3.49e-4 |
   | TP=4 | 578 | 0 | 7.50e-3 |

   TP=4 is unchanged; TP=2 lost the 129 all-reduces it should never have
   matched. The order-of-magnitude gap in the last column is the missing matmul
   of point 3, now visible in the number rather than hidden behind coverage.

### C. Target-configuration serving measurement -- FORBIDDEN AS INPUT

Evaluation only. Nothing below may reach the predictor.

| artifact | why it is out |
| --- | --- |
| `matrix/sweep_tp2_short.*.jsonl`, `sweep_tp4_short.*.jsonl` | step tables of the target width -- this is exactly the fitting the transfer claim excludes |
| `matrix/tp2_short*/`, `tp4_short*/` results | end-to-end at the target width |
| any TTFT/TPOT/throughput measured at TP=2 or TP=4 | the quantities being predicted |

The same-TP matrix stays, labelled as reference and diagnostic. It answers "how
well could this oracle do if it were allowed to cheat", which is worth knowing
and is not G4.

## 2. What carries the prediction across the width

Not a scaling law. The structure:

1. **The graph changes, and is derived without devices.**
   `atom/compass/runtime/derive.py` builds a sharded rank's model in one CPU
   process via `atom/distributed/simulated_tp.py`, so a TP=4 rank's op graph
   exists for a machine that never ran TP=4. This is the seam the whole claim
   rests on: at TP=4 each rank's gemms are narrower and 129 all-reduces appear
   that TP=1 does not have. The graph says so; nothing has to be assumed.
2. **Each operator in that graph is priced from B.**
   `PricedGraphCostOracle` looks up `signature_of(op)` exactly -- no
   interpolation across shapes, deliberately, because a neighbouring shape can
   dispatch a differently-tuned kernel and cost 2.4x more.
3. **The launch overhead comes from A**, unchanged, which is a real
   assumption and is the one `scripts/compass/holdout.py` already tests.
4. **ATOM's own scheduler turns per-step costs into a serving result**, on the
   CPU replay seam (G5), with no GPU.

So the predicted quantity is end-to-end serving behaviour at a width whose
serving behaviour was never measured, from a graph derived on CPU and prices
taken per kernel.

## 3. What this will not cover, stated in advance

- Unpriced operators. Coverage is 98.8% of operators by count and the priced
  kernels sum to ~0.74 of a step; the remainder is carried by the launch
  constant from A. If that constant does not hold at another width, this shows
  up as a systematic error, and it should be reported as that rather than
  absorbed.
- Rungs that were never traced. `_for_rung` falls back to the largest measured
  decode graph and warns once. A fallback that fires during a G4 run is a
  limitation of that run and goes in its record.
- The long workload, initially. The short workload is the smallest thing that
  can answer the question; long-context transfer is a separate claim and its
  calibration is still blocked on the sweep fault (tasks #5, #6).

## 4. Acceptance

Same tolerances as everywhere else, against the held-out TP=2 and TP=4 warmed
cells: throughput and TPOT/ITL within 10%, TTFT within 15%, and -- the part
that actually matters for the PoC -- the *ranking* of TP=1/2/4 on the workload
must come out right, with Spearman rho >= 0.90 within the comparable group.

The G1 ranking should be reported from this predictor. The same-TP fitted
matrix is quoted beside it as the diagnostic ceiling, and labelled.

## 5. Evidence: the TP=4 rank-0 graph, derived on a machine with no GPU

Run in `xiaobizh_n18_cpu`, a container with no `/dev/kfd` and no `/dev/dri`:

    python scripts/compass/graph_diff.py trace --device meta \
        --model Qwen/Qwen3.8-27B --tokens 4 --tp 4 --rank 0 \
        --replay-target agent_scratch/poc/g5_27b/target.json \
        -o agent_scratch/g4/derived_tp4_r0_t4.json

    operators : 3128 (29 distinct)
    built in  : 0.22s | traced in 0.462s
    redirected: 129 cuda factory calls to meta

Against the captured `compass_ops/silu27_graph.tp0.json`, every AITER operator
appears at the same count:

| operator | derived (no GPU) | captured (GPU) |
| --- | --- | --- |
| `aiter::all_reduce_` | 129 | 129 |
| `aiter::_fused_qk_rmsnorm_group_quant_kernel` | 129 | 129 |
| `aiter::gemm_a16w16` | 256 | 257 |
| `aiter::silu_and_mul` | 64 | 64 |
| `aiter::linear_attention_with_output_base` | 48 | 48 |
| `aiter::unified_attention_with_output_base` | 16 | 16 |
| `aiter::masked_embedding` | 1 | 1 |

The gemm difference is the LM head, which the capture runs and the model body
does not contain. The `gemm_a16w16` and `silu_and_mul` *signatures* are equal
character for character, `4120` shard width included -- which is the claim
under test, because that width exists only at TP=4 and no TP=4 forward was run
to obtain it.

### Three seams this needed, each recorded rather than assumed

1. **Architecture.** The already-authorised replay bootstrap, reached by
   `graph_diff.py --replay-target`. AITER asks `rocminfo` for the chip at
   import; the captured target answers instead.
2. **Device-typed factories.** `redirect_device_factories` sends
   `torch.empty(1, device="cuda")` -- AITER's dummy for selecting a dispatch
   key -- to meta, for an allowlist of factory functions only. Movement
   (`.to("cuda")`, `.cuda()`) is deliberately not covered and still fails. The
   count is written into the graph's provenance as
   `device_factories_redirected`.
3. **`attention_mha`'s build device.** It asked `torch.cuda.current_device()`
   while being constructed under `torch.device("meta")`, and put a scale tensor
   on a real card. It now follows the ambient build device; off meta the string
   is unchanged.

### Two pricing defects this found, both now fixed

Coverage against `silu27_i1000_r1_prices.tp0.json` was 320/3128 before and
578/3128 after. The two ops that moved are the two that matter most:

- `aiter::all_reduce_` was derived **without the dispatcher's scalars**, so it
  signed as `aiter::all_reduce_|4,5120|bfloat16` against the captured
  `...|#1=tp:0;#2=True;#3=False`. Exact lookup missed all 129, and
  tensor-parallel communication would have been priced at zero -- the one error
  that would have made TP look free. `record_collectives` now binds against
  `GroupCoordinator.all_reduce` (the class, since the instance attribute is
  already simulated TP's passthrough) and records `unique_name`, `ca_use_new`
  and `ca_fp8_quant` as the custom op does.
- The trace ran at the **fp32 default dtype** while the build ran at the
  model's, so AITER's dispatch dummy was `float32` and all 129 fused
  qk-rmsnorms missed their price by dtype alone. The dtype now spans the trace.

### What is still not priced, and why

`linear_attention_with_output_base` (48), `unified_attention_with_output_base`
(16), `triton::_fused_qk_norm_single_kernel` (16) and `masked_embedding` (1).
These are shape-dependent on context length: the derivation traced four tokens
of one sequence, the capture is four requests of one token each at context 66.
That is the untraced-rung limitation of §3, not a derivation defect, and it is
the next thing to close.

### What this section does not claim

`graph_diff.py compare` **fails** on this pair, and the tool says why itself:
the capture was recorded at compilation level 3, so inductor-fused operators
reach neither tracer and ordered containment cannot be evaluated. Only 2 of
3128 aligned. The histogram and signature agreement above is real; the ordered
alignment is untested until a level-0 capture exists. No step time has been
predicted yet.

## 6. The attention family is not a shape gap. It is a context gap.

Taking §5's four unpriced operators one at a time, the *shapes and dtypes are
already identical*. What differs is everything after them:

    derived   ...|4,1536;4,256;4,256|bf16,bf16,bf16|#1=None;#4=None;
                 #5=language_model.model.layers.3.self_attn;#6=False;#7=None
    captured  ...|4,1536;4,256;4,256|bf16,bf16,bf16|context_lens=[66,66,66,66];
                 slot_mapping=[257,273,289,305];cu_seqlens_q=[0,1,2,3,4];
                 max_seqlen_q=1;max_seqlen_k=66;...

The capture ran inside ATOM's runner, so `forward_ctx.capture(name)` had a
populated forward context to record. The derivation calls `model(input_ids,
positions)` bare, so there is none, and the raw keyword arguments are recorded
in its place. Deriving at a different token count would not close this; there
is no token count at which a bare model call acquires a `slot_mapping`.

Two consequences, and the second is the one that matters.

**`masked_embedding` cannot be closed at all.** Its captured signature carries
`0:279,279,279,279`, the value range of the index tensor. A meta tensor has no
values, so a derived graph will never have that entry. It is one operator; it
is named here so its absence is not later read as a defect.

**Attention price is context-dependent, but on this deployment it is a small
term, and that is measured rather than assumed.** `priced.py` records the
experiment: four decode graphs spanning a run's context range moved held-out
error from 3.7% to 1.5-2.3%, and covering the range rather than extrapolating
over it made no further difference. The conclusion drawn there -- "context is
not what dominates" -- was the reason interpolation within a kind was tried and
dropped. So this is not the `all_reduce_` case. Pricing all-reduce at zero
removed a whole term; pricing attention at one representative context costs a
few percent, and the project already decided that was not worth the machinery.

That decides the next piece, and against building new machinery for it:

- structure, per configuration, derived once with no GPU -- **done** (§5);
- non-attention operator prices, exact by signature -- **done** (§5), 578 ops;
- attention cost, per rung, from a captured decode graph at a representative
  context, exactly as the same-TP path already does it -- **the next piece**,
  carrying the 1.5-2.3% context residual as a stated limitation rather than
  trying to remove it.

The graph a derivation produces is therefore not the whole input to the cost
model, and was never going to be: the derived graph supplies structure and
non-attention shapes, and the attention term stays keyed on the rung. What the
derivation has to be trusted for is precisely what §5 tested.

## 7. The whole TP ladder, derived on CPU: the shapes actually move

TP=1 and TP=2 derived the same way as §5 (`agent_scratch/g4/derived_tp{1,2,4}_r0_t4.json`),
each in about half a second on the device-free container. Per-rank
`gemm_a16w16` widths, and the collective count:

| | TP=1 | TP=2 | TP=4 |
| --- | --- | --- | --- |
| operators | 2999 (28 distinct) | 3128 (29) | 3128 (29) |
| `aiter::all_reduce_` | **0** | 129 | 129 |
| MLP up | 34816 | 17408 | 8704 |
| MLP down (in) | 17408 | 8704 | 4352 |
| attention in | 6144 | 3072 | 1536 |
| qkv | 16480 | 8240 | 4120 |
| gate/up | 14336 | 7168 | 3584 |

Every width halves and halves again, which is what a correct shard derivation
must do and what a cost model priced only at TP=1 widths could never see.

The zero is the part worth reading twice. TP=1 derives **no all-reduces at
all** -- not because the recorder was disabled, but because
`GroupCoordinator.all_reduce` returns early at `world_size == 1`, and the
derivation runs the real method. So the derivation reproduces the real code's
behaviour at each width rather than adding a collective per layer because the
topology says so. That is also why §5's `all_reduce_` scalar defect mattered
only from TP=2 upward, and why it could not have been caught at TP=1.

## 8. A third pricing defect: a stride into memory the graph never saw

Collecting the TP=2 price list of §1B did not fail for want of GPUs. It faulted
the device:

```
Memory access fault by GPU node-3 (Agent handle: 0x...) on address 0x7f... Reason: Unknown.
[ModelRunner1/2] proc died unexpectedly (exitcode=-6)
```

No price list was written, and the log named no operator. A fault is not an
exception -- the process dies, so the `unpriced` bookkeeping never runs and
nothing says which of two hundred signatures was in hand.

**The witness.** `COMPASS_BENCH_TRACE` now names each signature in a file,
flushed and `fsync`ed, *before* it is priced. Anything buffered dies with the
process; this does not. Re-run at TP=1 on one card, which is the smaller
question -- is this TP=2's fault, or the derived graph's? -- it answered in 87
seconds:

```
--- signatures attempted: 37
--- last attempted: triton::_fused_qk_norm_single_kernel|4,24,256;4,4,256;...
                    |#9=14336;#10=14336;#11=6144;#12=1024;#13=24;#14=4;...|grid=112
Memory access fault by GPU node-2 ... Reason: Unknown.
```

So it is **not TP=2's**. It reproduces at TP=1, with `COMPASS_LOAD_GENERATED=0`,
after 36 signatures priced cleanly.

**The defect.** Read the kernel's own signature in
`atom/model_ops/layernorm.py`: positional arguments 9 to 12 are
`q_in_stride0, k_in_stride0, q_out_stride0, k_out_stride0`. At TP=1
`q_in_stride0` is 14336 -- the *fused qkv buffer's* row stride -- while the q
the graph records is `[4, 24, 256]`, 6144 elements a row. In the real forward q
is a **view** into that buffer. Rebuilt here as a dense tensor of exactly its
recorded shape and launched with the original stride, row 3 addresses element
`3*14336 + 23*256 + 255 = 49151` of 24576. Twice past the end.

A Triton kernel takes pointers, not tensors. Its strides arrive as plain ints
beside them, and when the tensor it was handed was a view, the stride belongs to
an allocation the graph never saw. Shapes do not record it and cannot.

**Closed 2026-09-11, by refusing.** `_stride_past_its_tensors` in
`microbench.py` needs no knowledge of any particular kernel: for an argument
`[d0, *rest]` with `d0 > 1`, a stride `s` over `d0` rows fits exactly when
`s <= prod(rest)`, since `s*(d0-1)+R > d0*R` reduces to `s > R`. A **positional**
integer larger than *every* argument's row extent therefore cannot be a dense
stride for any of them, and the graph cannot show it is not a stride at all.
Constexprs are excluded -- they are compiled into the kernel rather than applied
to a pointer, and `BLOCKS_PER_TILE=4096` is a tile size whose kernel prices
correctly today. Measured against the *widest* argument, not the narrowest, or
q's own contiguous out-stride 6144 would read as a fault beside k's row of 1024.

What this refuses, across every real graph in hand:

| graph | triton/inductor ops | refused | which |
| --- | --- | --- | --- |
| `silu27_graph.tp0` (capture, level 3) | 130 | 80 | 5 `inductor::` families, all **already** unpriced for want of an importable origin |
| `derived_tp1_r0_t4` | 16 | 16 | `_fused_qk_norm_single_kernel` |
| `derived_tp2_r0_t4` | 16 | 16 | same |
| `derived_tp4_r0_t4` | 16 | 16 | same |

`triton::kv_indices_generate_kernel` and `triton::_fill_deferred_decode_ids_kernel`
-- the only two raw Triton kernels any existing price list prices -- are not
refused. So no price that works today is lost.

It over-refuses by construction: a kernel taking a genuinely large extent loses
its price. That is the direction to err, because the other failure takes the
whole run and every signature after it.

**What this guard is not.** It is a conservative refusal, and its success on one
known kernel is not evidence that generic argument reconstruction is safe. Three
limits, stated so no later section can quietly borrow a stronger claim:

* A positional integer need not be a stride at all. The test asks only whether a
  value is too large to be a dense stride for any recorded argument, which is a
  necessary condition for the fault and nowhere near a sufficient one.
* A *bad* stride can be smaller than the widest argument's row extent. A kernel
  handed a view whose stride is 2048 where the widest recorded row is 6144 walks
  off its own tensor and passes this test. The guard catches the case that
  faulted; it does not characterise the case that faults.
* Refusal is the right answer wherever layout is unknown, and it stays. The
  durable fix is not a better guess at what a scalar means -- it is recording
  shape, stride, storage offset and extent (and aliasing, where a kernel writes
  through more than one view of one allocation) at trace time, so a rebuild is
  reconstruction rather than inference. That is not built.

**An asymmetry worth stating.** The captured graph is at compilation level 3 and
holds this kernel only as `inductor::_fused_qk_norm_single_kernel_0`; the raw
`triton::` launch appears only in the level-0 derivations. So the capture path
never priced this operator -- it refused it for want of an importable origin --
and the derivation path is the first to try. The hazard was always there. Only
derivation reached it.

**The re-run, 2026-09-11.** Same script, same card, `rc=0`. The derived TP=1
graph priced to completion: **126 signatures attempted, 126 handled** -- where
the faulting run stopped at 37 -- 56 priced, 70 unpriced, no fault. The refusal
fired exactly once, with its reason recorded in the list itself:

```
#9=14336 exceeds every argument's row extent (6144), so it can only be
a stride into an allocation the graph does not record
```

and `provenance.topology` reads `{'tp': 1}`, which is the cross-width guard of
§1B.4 writing its declaration for the first time.

The 70 unpriced break down as 64 attention-family signatures that "read a
forward context and the graph recorded none" -- §6's context gap, and expected,
because a derivation on `meta` records no context -- 1 refused stride, and 5
argument-rebuild failures:

| signature | why |
| --- | --- |
| `aten::to.dtype\|4,1,32` | the `dtype` argument was not recorded, so the call has no value for it |
| `aten::cat\|4,24,32;4,24,32` (and 3 more) | `cat` takes a *list* of tensors; the rebuild passes them as separate positional arguments |

Both are rebuild gaps in the same family as the scalar defect of §5, not
transfer defects, and both are small and independent of width. They are recorded
here rather than fixed, because neither is on the path to a TP=2 price.

## 9. The TP=2 price list exists. The width matrix, measured 2026-09-11.

`g4_price_tp2.sh` completed, `rc=0`, both ranks writing a list that declares its
own width: `provenance.topology = {'tp': 2}`. `aiter::all_reduce_` is priced at
**9.124e-06 s** on rank 0 and 9.128e-06 s on rank 1 -- a two-way collective
measured two-way, which is the thing §1B.4 refused to take from the four-way
list. §1B.3's "**a TP=2 prediction cannot be made from the inputs declared
above**" is closed for the matmul and collective terms.

Every derived graph priced against every list:

| graph | list | priced | unpriced | refused collectives | seconds |
| --- | --- | --- | --- | --- | --- |
| TP=1 | TP=1 | **2791** / 2999 | 208 | 0 | **2.179838e-02** |
| TP=1 | TP=2 | 1318 | 1681 | 0 | 4.479612e-04 |
| TP=1 | TP=4 | 129 | 2870 | 0 | 3.493040e-04 |
| TP=2 | TP=1 | 1318 | 1810 | 129 | 4.438718e-04 |
| TP=2 | TP=2 | **2920** / 3128 | 208 | 0 | **1.376608e-02** |
| TP=2 | TP=4 | 129 | 2999 | 129 | 3.493040e-04 |
| TP=4 | TP=1 | 1318 | 1810 | 129 | 4.438718e-04 |
| TP=4 | TP=2 | 1318 | 1810 | 129 | 4.479612e-04 |
| TP=4 | TP=4 | 578 / 3128 | 2550 | 0 | 7.496273e-03 |

Three things to read off it.

**The off-diagonal is 1318 in four cells and 129 in two.** 1318 is the
width-invariant part of the graph -- norms, elementwise, the rope kernels -- and
it is the same 1318 operators in every cell that pairs a graph with the TP=1 or
TP=2 list across a width. Everything shard-dependent misses, because its shapes
carry the width. So a cost model calibrated at one width and read at another
covers 42-44% of the graph and contributes about 2% of its seconds.

The two cells that read 129 are the ones that price a TP=1 or TP=2 graph against
the **TP=4** list. That is not a width effect: the TP=4 list was priced from a
level-3 *captured* graph, so it holds inductor-fused signatures a level-0
derivation never emits, and 129 is all of it that matches anything. The TP=4
list's own diagonal reads 578 for the same reason. Corrected 2026-09-11.

**The refusals are exactly where they should be.** 129 in every cell whose graph
has collectives and whose list was measured at another width; 0 in the first
three rows, because a TP=1 *graph* derives no collective at all
(`GroupCoordinator.all_reduce` returns early at `world_size == 1`, and the
derivation runs the real method). The zero belongs to the graph's width, not to
the list's: the TP=1 list still refuses to price the TP=2 and TP=4 graphs'
collectives, which is the 129 in rows four and seven. Without the §1B.4 guard,
a four-way price would have been spent on a two-way call at full coverage and in
silence.

**The TP=4 diagonal is the odd one: 578 of 3128.** That list was priced from the
*captured* level-3 graph, whose operators are inductor-fused, so its signatures
do not meet the level-0 derivation's. The TP=1 and TP=2 lists were priced from
the derived graphs themselves and cover ~93%. This is a statement about
compilation level, not about width -- and it is why §1B.3's coverage table
looked so much worse at TP=4 than the two new lists do.

### What this is not yet

**Not a step time.** 208 operators are still unpriced at both widths, and 64 of
them are the attention family: §6's context gap, which no amount of pricing
closes because the missing thing is the forward context, not a shape. A step
predicted from these lists is predicted *low* by whatever attention costs, and
the oracle says so on every load. Until that closes, these are calibrated
per-operator costs and a coverage matrix -- not a G4 result against the ≤10%
gate.

**Not free of the target width.** §1B.1 already declares it: collecting each
list required standing that width up on GPUs. What has *not* touched TP=2 is the
capture, the workload, the scheduler and the serving measurement -- the graph
was derived on a machine with no GPU. That is the boundary this ledger claims,
and it is narrower than "predicted without ever running TP=2".

## 10. The context gap is closed. A priced body, at two widths, 2026-09-11.

§6 said the attention family was a context gap, not a shape gap, and that no
amount of pricing would close it. That was right, and the fix was to stop
tracing a bare forward pass. `atom/compass/runtime/batch_spec.py` states the
batch instead of inferring it -- kind, per-request query and context lengths,
block size, `max_model_len`, capture bucket, block policy, prompt lengths,
position rows -- and derives from it exactly what
`forward_ctx._capture_attention` and `_capture_linear_attention` would have
recorded from a real runner. `graph_diff.py trace --batch-spec` installs it for
the duration of the trace. Before this, `_trace` installed *no* forward context
at all, `forward_ctx.capture()` returned `()`, and all 64 attention operators
were recorded unpriceable.

**The derived context equals the captured one.** Against the level-3 27B capture
`compass_ops/silu27_graph.tp0.json`, all 13 full-attention fields and all 11
DeltaNet fields match field for field -- capture at TP=4, derivation at TP=1 on
meta. Nothing was read from a meta tensor and nothing was borrowed from another
configuration; the spec is written from quantities a scheduler holds. The batch
is `tests/compass/batch_specs/decode4_c66.json`: four requests admitted at 64
prompt tokens, each two tokens into generation, context 66.

**The spec changes which code path runs, not only the metadata.** The bare
four-token body pass records 2999 operators; the spec'd decode records 2439.
`triton::_mrope_qk_kernel` (16) and `aten::min` (16) appear only with the spec.
`aten::cat` (64), `aten::chunk` (32), `aten::index.Tensor` (32), `aten::slice`
(64), `aten::squeeze` (64), `aten::sub` (32), `aten::to.dtype` (64) and
`aten::unsqueeze` (64) appear only without it. A four-token single-sequence body
trace is therefore not interchangeable with four decodes at context 66 merely
because Q/K/V shapes agree.

**Priced, at the graph's own width.** `agent_scratch/g4_price_spec.sh`, `rc=0`:

| graph | signatures priced | operators priced | body, priced operators |
| --- | --- | --- | --- |
| TP=1 rank 0 | 106 / 107 | 2423 / 2439 (99.3%) | **23.122 ms** |
| TP=2 rank 0 | 107 / 108 | 2552 / 2568 (99.4%) | **15.467 ms** |

At TP=1 the largest terms are `aiter::gemm_a16w16` 19.07 ms (82.5%, 256 ops),
`aiter::linear_attention_with_output_base` 1.66 ms (7.2%, 48 ops) and
`aiter::unified_attention_with_output_base` 0.43 ms (1.9%, 16 ops). At TP=2 the
same three read 10.55 ms, 1.29 ms and 0.26 ms, and `aiter::all_reduce_` enters
at 1.18 ms (7.6%, 129 ops) -- the collective that TP=1 does not emit. The one
family still unpriced at both widths is `triton::_fused_qk_norm_single_kernel`,
16 operators, refused by §8's stride guard.

### What this is and is not

It is a **predicted body cost with its coverage**, not an accepted step time,
and it is not yet compared to any measured step. Four things stand between the
two, and none is closed:

* **The 16 refused operators are unpriced and unbounded.** §8's guard refuses
  them; a refusal is not a zero and not a bound. The body number is low by
  whatever they cost.
* **The LM head and the rest of the runner's step are not in the graph.**
  `graph_diff.py trace` traces the model body. Sampling, the logits projection
  and the runner's own per-step work are outside it and are neither priced nor
  independently bounded here.
* **Level 0 against level 3.** The derivation is an eager trace; production runs
  compiled and fused. §9's TP=4 diagonal (578 of 3128) is what that mismatch
  looks like from the other direction. The size of the effect on a *step time*
  is unmeasured.
* **One point in the domain.** Decode, batch 4, context 66. Nothing here speaks
  to long contexts or to chunked prefill, where the attention family's share is
  not small. §6's old "attention is small at a representative context" does not
  generalise and is not being relied on.

So: the operator that could not be priced at all now prices, at two widths, with
99.3-99.4% operator coverage. The G4 ≤10% gate is still open, and closing it
needs the step-level terms above plus a held-out comparison against measured
TP=2 and TP=4 production steps.

## 11. Pricing without standing the deployment up, 2026-09-11

Everything above was priced by `scripts/compass/run.py`, which starts a real
ATOM server at the target width -- weights loaded, KV pool allocated, scheduler
running, a throwaway workload served -- and only then prices the graph. That is
a fair diagnostic and it is what §9 and §10 used, but it is not the workflow
this PoC promises: if pricing a candidate requires standing that candidate up,
nothing has been avoided.

`scripts/compass/primitives.py --layers attention` prices the same graph with a
much narrower thing behind it: the attention modules materialised with random
parameters, a KV region sized from the graph's own block tables, a real process
group of the target width for the collectives, and no model. What the two lists
agree on is what the server was not contributing.

### What received storage and what did not

Per rank, read from the process rather than asserted:

| width | attention tensors on device | target tensors left on meta | stand-up peak | process peak |
| --- | --- | --- | --- | --- |
| TP1 | 352, 1.25 GiB | 898, 51.0 GiB not materialised | 1.62 GiB | 6513 MiB |
| TP2 | 352, 0.63 GiB | 898, 25.9 GiB not materialised | 2.86 GiB | 5352 MiB |
| TP4 | 352, 0.31 GiB | 898, 13.4 GiB not materialised | 2.47 GiB | 4029 MiB |

`model_runner_initialized`, `weights_loaded` and `served_workload` are recorded
`false` in every artifact. Times separate: distributed init 0.07-3.01 s, model
build 0.62-0.81 s, pricing 9.2-17.1 s. Source and graph sha256 digests are in
`provenance.collector.hashes`.

The on-device attention storage is the primitive operands and the attention
state -- `conv1d.weight`, `A_log`, `dt_bias`, the decode scale table, `kv_scale`
and the bound `k_cache`/`v_cache` views. Those are expected; the requirement was
never that attention runs on nothing, it was that the other 27 billion
parameters are not loaded.

### A placement bug that inflated the first TP2/TP4 answer

`init_dist_env(..., local_rank=r)` builds the process group but does not move
the process's current CUDA device. Under one shared `HIP_VISIBLE_DEVICES` mask
every rank therefore allocated and ran on logical device 0 -- one physical card
running four ranks' worth of work. It does not fail. It contends, and the first
TP2/TP4 prices came out 1.43-1.47x and 2.18-2.81x high against the reference,
with nothing in the artifact to say why.

The fix is `torch.cuda.set_device(args.rank)` before anything allocates, and
then a refusal: `_require_distinct_devices` gathers every rank's UUID and PCI
address and raises before benchmarking if two ranks report the same card, or if
any card cannot be identified. A logical index alone is not accepted as the
answer -- `dev 0` on every rank is correct when each rank has its own mask -- so
the mask, the logical index and the physical identity are all recorded:

    TP2  rank0 dev0 vis 0,1      rank1 dev1 vis 0,1     pci 10, 128
    TP4  rank0..3 dev0..3 vis 0,1,2,3                   pci 10, 128, 164, 200

The superseded runs are kept as diagnostics, marked INVALID, and excluded from
calibration. No correction factor was fitted to the discrepancy.

### Agreement, single-rank multiplicity on both sides

The reference collector was pointed at every rank's graph at once, so its raw
operator counts are 2x at TP2 and 4x at TP4. Both sides are weighted by the
standalone (single-rank) counts and the multiplicity is printed, rather than
divided out afterwards.

| width | reference rank | body ratio (standalone / reference) |
| --- | --- | --- |
| TP1 | r0 | 0.9949 |
| TP2 | r0 | 0.9486 |
| TP2 | r1 | 0.9736 |
| TP4 | r0 | 0.9943 |

Coverage is 107/108 signatures and 2552/2568 operators (99.4%) on both sides at
every width; the refused signature is §8's `_fused_qk_norm_single_kernel`.

Both TP2 reference ranks are retained, because they disagree with each other at
component level and neither is the one to keep:

| signature | std r0 | std r1 | ref r0 | ref r1 |
| --- | --- | --- | --- | --- |
| `aiter::silu_and_mul\|4,8704;4,17408` | 3.561 us | 3.598 us | 10.009 us | 3.556 us |
| `aiter::gemm_a16w16\|4,3072;5120,3072` | 17.619 us | 17.673 us | 22.760 us | 17.739 us |
| `aten::mul.Tensor\|4,24,128;4,24,128` | 1.988 us | 2.153 us | 2.604 us | 9.613 us |

Rank-to-rank spread of |ratio - 1| over the 107 shared signatures: standalone
p50 0.65% / p90 8.60% / max 57.19% (a 0.026 us allocation), reference p50 0.82%
/ p90 28.31% / max 181.46% (the silu above). The TP2 residual is concentrated in
signatures where the reference disagrees with its own sibling rank by 3-5x.
**That is an observed asymmetry in the reference, not an explanation of it.**
The reference run is not reproducible from these artifacts, so its cause --
contention, clock state, allocator layout, something else -- is unknown, and is
recorded as unknown rather than called noise.

The only same-collector repeat available is at TP1 (p50 0.5%, p90 16.2%, max
87.7%). **That band is TP1's**, and is labelled as such wherever it is printed;
the comparison tool scores a signature against it only when the signature
appears verbatim in both runs, so a TP2-only shape is never scored against a
TP1 number.

### What this section does not claim

It is collector validation: the standalone numbers and the deployment numbers
are the same numbers. It is not full-step accuracy, not a G4 result, and not an
acceptance test. Every limit in §10 still stands -- the 16 refused operators,
the LM head and runner work outside the traced body, level 0 against level 3,
and a domain of one decode point. The long-context and chunked-prefill graphs
are derived but not yet priced.

## 12. A frozen TP2 forward step, then measured, 2026-09-11

The first prediction in this document written down *before* the thing it
predicts was run. A TP2 decode step was predicted from the ledger's allowed
inputs alone, the number and its report were frozen with their input hashes at
**2026-09-11T06:21:54Z**, and only then was a TP2 deployment stood up and
captured.

    frozen   20.970 ms   measured   19.845 ms   +5.7%

### What the frozen number is a prediction of

`ModelRunner.forward` for one decode step: prepare_model + run_model +
postprocess, which is exactly the span a capture row records as `seconds`. The
shape is written out rather than implied, because a comparison against a
neighbouring shape is a different claim:

    kind=decode, tp=2, bucket=32, cohort=32, tokens_each=1, context=1151

### Inputs, against the §1 ledger

Fifteen files, all hashed into `dec32/tp2_frozen.sha256` at the freeze:

| what | files | ledger class |
|---|---|---|
| per-rank body/head graphs, TP2 | `b27dec32.tp2.r{0,1}.json`, `h27dec32.tp2.r{0,1}.json` | structure, derived on CPU |
| per-rank price lists | `p27bdec32.tp2.r{0,1}.json`, `p27hdec32.tp2.r{0,1}.json` | B: standalone TP2 primitives |
| registered/unregistered all-reduce | `ar_prices_tp2_graph_{capture,plain}.json` | B: real group, real communicator |
| head all-gather | `ag_prices_tp2.json` | B |
| region model | `regions.py` (`SOURCE_27B_TP1`) | A: TP1 source capture, applied unchanged |
| composition code | `tp2_frozen.py`, `library.py` | -- |

No TP2 step time and no TP2 serving time entered the prediction. Class C stayed
out, which is the whole point of freezing it first.

### Reproducibility

`checkpoint_tp2.py` re-executes the frozen script in-process, hashes every
`sys.modules` entry whose file lives under the working tree, and compares its
output to the frozen text. **20 modules** hashed; the regenerated text is
byte-identical; torch 2.10.0+rocm7.2.4.git3d3aa833, Python 3.12.3. It never
writes `tp2_frozen.txt` -- a prediction that can be rewritten afterwards is not
frozen.

### The measurement, matched on the full key

Captured 06:26-06:28 the same day, devices 4 and 5, 64 requests. Two steps match
the key, at scheduler ticks 130 and 260 (cohorts 0-31 and 32-63), each seen by
both ranks -- four rows, all four retained and reported, none dropped.

Rank alignment is **read, not assumed**: every row carries the `rank_coords` the
runner recorded, each per-rank file contains exactly its own rank, and the
comparison exits if that does not hold. What remains assumed is only that those
coords index the same tp group the graphs were derived per-rank against -- both
come from `get_tp_group().rank_in_group`.

| | rank 0 | rank 1 | step (max over ranks) |
|---|---|---|---|
| **full forward, frozen** | **20.970 ms** | **20.814 ms** | **20.970 ms** |
| full forward, measured | 19.845 ms | 19.845 ms | 19.845 ms |
| | **+5.7%** | **+4.9%** | **+5.7%** |
| run_model predicted / measured | 20.708 / 19.593 | 20.552 / 19.588 | +5.7% |
| postprocess predicted / measured | 0.131 / 0.140 | 0.131 / 0.142 | -6.3% / -7.5% |
| prepare predicted / measured | 0.131 / 0.112 | 0.131 / 0.116 | +16.8% / +12.5% |

The step is the max over ranks because it ends when the last rank does; the
per-rank numbers are reported beside it because a per-rank error is not a step
error. The two rows agree to 5 us on each rank.

The one thing the comparison changes is where it *charges* the TP broadcast:
into the `postprocess` span, which is the span that physically contains it
(`regions.py:26`). That moves 0.029 ms between two held-out lines and leaves
every predicted total exactly as frozen. It is an attribution fix, not a refit.

### What this result is, and what it is not

It is a `ModelRunner.forward` decode-step check on one covered shape, judged
against a 10% criterion, which it meets at +5.7%.

It is **not** a passed serving TPOT gate. TPOT, throughput, TTFT and the
configuration ranking are end-to-end quantities measured over a whole serving
run; none of them is claimed or tested here, and all of them still require the
complete short/long serving matrix of §4. Nothing in G1, G2 or G3 moves on this.

Within that scope: the kernel side is the transfer -- graphs are structure,
prices are standalone TP2 primitives, and that composition lands within 5.7% of
a real TP2 production step it never saw. The region model is the held-out part:
`SOURCE_27B_TP1` was calibrated on a TP1 capture and applied unchanged, carrying
the falsifiable claim that postprocess runs on TP-invariant shapes. It survives
at -6.3% / -7.5% on a term worth 0.7% of the step; `prepare` transfers less well
in relative terms but misses by 0.019 ms absolute.

### Method sensitivity -- separate from the result above

The frozen text named one sensitivity in advance: the two copy-path measurement
methods of the body all-reduce disagree by 13.5%, worth 0.223 ms across the
body's 129 reductions. Applied on its own it gives 21.193 ms, +6.8% -- it would
worsen this comparison. That is the whole of the statement. It does not identify
any component of the +1.115 ms residual and does not rule any component out. The
residual is recorded here and not attributed; the sensitivity stays an open
question about the two methods, to be settled by a registered-path measurement
taken the second way.

### Archive

`agent_scratch/` is git-ignored, so the archive is the durable copy and this
section cites it by hash. `agent_scratch/g4/archive/tp2_2026-09-11/` holds the
frozen text and its timestamp, the 15-input manifest **and the fifteen input
files themselves**, the 20-module reproducibility record, the four paired
capture rows (`paired_rows.jsonl`), both complete per-rank step files, the
server and replay logs, the comparison as produced, and the four scripts. 33
files, hashed in `MANIFEST.sha256`.

    tp2_2026-09-11.tar.gz
      40c1b3caf9e021314efe9e543c1150a2caba1c30bbc1c1028280d02b1eb54134

Held on node 18's `xiaobizh_n18` and on the host under the same path; the
tarball hash above was verified equal on both, and `sha256sum -c` over all 33
files passes on the host copy. Built by `agent_scratch/g4/archive_tp2.sh`
(`5cc465f185e64409e837231e133fa9e584f1cc2ea757d2111bb064eeae9d6272`).

---

## 13. A frozen TP4 forward step, then measured — and it misses, 2026-09-11

**Result: frozen 16.157 ms against a measured 12.807 ms step, +26.2%, outside
the 10% criterion.** The prediction fails. This section records why, and the
"why" turns out to be one bad input rather than the transfer, which is a
different problem with a different fix — but the fix is a *new* prediction
against a *new* capture, not a revision of this one.

### The freeze

Frozen `2026-09-11T07:22:04Z`, before any TP4 full step was measured, by
`agent_scratch/g4/tpN_frozen.py` — `tp2_frozen.py` generalised to either width.
The generalisation is not asserted, it is checked: run at TP=2 the new script
reproduces `dec32/tp2_frozen.txt` **byte for byte**, and `freeze_tp4.sh` runs
that diff as a precondition and refuses to freeze if it fails
(`tpN_equivalence_at_tp2.txt` in the archive). So the TP4 number comes out of
the same composition as the TP2 one, not a second model that resembles it. The
TP2 artifacts were not touched.

Twenty-three inputs, all hashed into `dec32/tp4_frozen.sha256`: four per-rank
body graphs, four head graphs, eight generic price lists, the two all-reduce
probe files, the all-gather probe file, the frozen script, `library.py`,
`regions.py` and the frozen text itself. `checkpoint_tpN.py` then re-executed
the composition in-process and recorded the 21 tree modules Python actually
imported: **regenerated text byte-identical to the frozen text, all 23 inputs
unchanged**, torch 2.10.0+rocm7.2.4.

Predicted, per rank: rank 0 13.510 ms, **rank 1 16.157 ms**, rank 2 13.459 ms,
rank 3 13.452 ms. The maximum is the step, because the step ends when the last
rank does. That 21% rank asymmetry was visible in the frozen text and was
flagged before the capture ran; it is the whole of the failure.

### The measurement

`capture_dec32.sh` at TP=4 on devices 0–3, server up in 85 s, 64 requests, 0
failed, 260 rows per rank. Eight rows match the frozen shape — two scheduler
ticks seen by all four ranks — matched on the full key (kind, tp, bucket 32,
cohort 32, one token each, context 1151), with rank alignment verified from
each row's own `rank_coords` rather than from the filename.

| rank | measured forward | predicted | error |
| --- | --- | --- | --- |
| 0 | 12.807 ms | 13.510 ms | **+5.5%** |
| 1 | 12.804 ms | 16.157 ms | **+26.2%** |
| 2 | 12.805 ms | 13.459 ms | **+5.1%** |
| 3 | 12.805 ms | 13.452 ms | **+5.1%** |
| **max over ranks** | **12.807 ms** | **16.157 ms** | **+26.2% — THE CHECK** |

The hardware shows no rank asymmetry at all: 12.792–12.807 ms across four ranks
and two ticks, a spread of 0.1%. The prediction claimed 21%.

Held-out terms, on every rank: postprocess −9.0% to −11.4% (predicted 0.131 ms,
measured 0.143–0.148), prepare + remainder −22.5% to −23.7% (predicted 0.131 ms,
measured 0.166–0.172). Both are TP1-calibrated and applied unchanged at TP4.
They are small in absolute terms — together about 0.06 ms of a 12.8 ms step —
and they under-predict, so they are not what made this fail. They are a real
residual and are recorded as one.

### Which input, and the evidence that it is the input

Per-operator, rank 1's frozen body price list is 15.818 ms against 13.112–13.126
ms on the other three, and **2.671 ms of the 2.692 ms difference is
`aiter::gemm_a16w16`** — 6.398 ms at rank 0, 9.070 ms at rank 1, +42%. The
dense GEMM shapes are identical across ranks at TP4; there is no structural
reason for one rank to differ.

So the body pricing was repeated, at the same width on the same four devices,
into a separate directory (`repeat_price_tp4.sh` → `dec32/repeat_2026-09-11/`,
rc=0, `2026-09-11T07:33:52Z`). It writes nowhere near the frozen inputs.

| rank | frozen body | repeat body | change | gemm frozen → repeat |
| --- | --- | --- | --- | --- |
| 0 | 13.126 ms | 13.113 ms | −0.1% | 6.398 → 6.311 |
| 1 | **15.818 ms** | **13.106 ms** | **−17.1%** | **9.070 → 6.369** |
| 2 | 13.112 ms | 13.077 ms | −0.3% | 6.305 → 6.367 |
| 3 | 13.121 ms | 13.056 ms | −0.5% | 6.276 → 6.352 |

Three ranks reproduce to within 0.5%. Rank 1 does not reproduce, and on repeat
it agrees with the others. The frozen prediction was built on a one-off
measurement that was 21% high on one rank, and the maximum-over-ranks rule then
made that one rank the whole answer.

### What this does and does not license

It **does not** license a corrected number. The capture has now been seen;
re-running the composition against it would be fitting, which is the one thing
this whole construction exists to avoid. For the record, and as arithmetic on
numbers already in this document rather than as a result: the three ranks whose
prices did reproduce predicted +5.5%, +5.1% and +5.1%, close to TP2's +5.7%. A
claim that TP4 transfers at that accuracy needs a new freeze on new prices and
a new capture, and until that exists, **G4 at TP4 is failed, not pending**.

It **does** identify a gap in the method. Nothing in the freeze path checks a
price against a second measurement of itself. `PriceLibrary` already refuses to
silently resolve same-signature records that disagree by more than 5% — the
mechanism exists and is used for the two all-reduce measurement methods — but
it only fires when two files are loaded. Pricing twice into two files and
refusing to freeze while any signature disagrees would have caught this before
the GPU-hours of a capture were spent. At TP4 the repeat cost about five
minutes.

### One real TP2/TP4 difference, separate from the failure

The frozen TP4 text names **no** method sensitivity, where the TP2 text named a
13.5% one. This is a result, not an omission: at TP4 the two copy-path
measurements of the body's all-reduce agree to 0.83% (11.618 µs from the probe,
11.715 µs from the generic pricing run), inside the library's 5% threshold, so
no conflict is recorded and there is no gap to carry onto the registered price.
At TP2 the same two methods differed by 13.5%. Which method matches production
is still open at both widths; at TP4 the two answers are close enough that it
would not move the step.

A second observation, recorded and not chased: re-pricing the all-gather after
the stage guard moved the 1-row `all_gather_unreg` shape by −59.7% (81.881 →
33.000 µs) while the 32-row shape the head actually uses at bucket 32 moved
−0.5% (118.440 → 117.840 µs). The shape in use is stable; the 1-row shape is
not.

### Archive

`agent_scratch/g4/archive/tp4_2026-09-11/`, 53 files, 7.0 M. Tarball
`834732b042c372b1699f87b6454c2469d092e43edeb3ff2cd74601e0f2edb7a0`, verified
equal on node 18 and on the host, with `sha256sum -c MANIFEST.sha256` passing
over all 53 files on the host copy. It holds the frozen text and both
manifests, the TP2 equivalence record, all 23 inputs under their manifest
names, the scripts, all four per-rank capture files with the workload and
server logs, the comparison as produced, the eight paired rows, and — filed
separately under `diagnostic/` so it can never be mistaken for a frozen input —
the repeat pricing run. Built by `agent_scratch/g4/archive_tp4.sh`.

## 14. What this ledger is now for — registered 2026-09-11

**G4 is finally proved on an end-to-end cc-traces replay, and everything in
§§1–13 is a diagnostic.** Registered 2026-09-11 at the user's direction and
recorded in `POC_STATUS.md` under "Final acceptance is end-to-end cc-traces".
This file does not become history: §1's ledger — what a transfer prediction may
and may not read — is exactly as binding on a cc-traces prediction as it was on
E7 and E8, and those boundaries are the experiment either way. What changes is
what counts as a *result*.

Three things follow.

**A forward-step check can no longer move the G4 row, passing or failing.** E7
(+5.7% at TP2) and E8 (+26.2% at TP4) are the two step-level evaluations this
ledger has produced, and the row they justified is back to UNPROVEN. That is not
a demotion of the work: the freeze discipline, the byte-identical equivalence
precondition and the input manifests are the machinery a cc-traces transfer claim
will also need, and they exist because these two ran. It is a statement about
what a single decode step evidences, which is one decode step.

**The held-out axis is configuration, and it is named per prediction.** The
source is TP=1 full-engine measurement; TP=2 and TP=4 are evaluation targets and
their full-engine step or serving timings never enter a fit. Standalone primitive
measurements at the target width remain allowed hardware-library inputs under
§1.B and are declared as such. `cc_pilot.jsonl` is a development and regression
workload — the cost model was iterated against it — so a cell built on it is not
held out on workload, and no prediction in this file has ever claimed to be.

**E8's failure carries one requirement forward.** Nothing in the freeze path
checked a price against a second measurement of itself, and that is what let a
17.1%-high body price be frozen. `PriceLibrary` already has the mechanism — the
5% conflict band at `library.py:373` — but it fires only when a price is loaded
twice. **A price measured once may not be frozen again.** The repeat run cost
about five minutes at TP4; the capture it would have saved cost four cards and
the better part of an hour.
