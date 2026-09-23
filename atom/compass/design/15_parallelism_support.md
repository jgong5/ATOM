# ATOM Compass — Design Topic 15: Parallelism Support (TP, DP, PP, EP)

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `01` (the LP structure and the wait contract this applies per strategy),
`03` / `05` (width-keyed memory), `07` D40 (communication pricing), `04` (structures that
exist only at width).

**Scope.** What each parallelism strategy changes for a simulator, answered in one frame
so the four are comparable. It **applies** `01`'s clock protocol per strategy; it does not
re-decide it.

**Why this is not an M7 topic.** M1 requires *"fake models covering prefill, decode, KV
need and TP/DP/PP/EP"*. So the LP structure, the clock topology and the memory-accounting
shape for all four are **M1** deliverables. Only cost *accuracy* waits for M7 — D94 splits
them.

---

## D88. The frame: four questions, asked of each strategy

A parallelism strategy can affect a simulated run in exactly four ways. Asking the same
four questions of each is what makes them comparable, and three of the four turn out to
have the same answer for most strategies.

| # | Question | Why it matters |
|---|---|---|
| **Q1** | **Does it add logical processes, and at what lookahead?** | LP count and lookahead set the cost of the clock protocol. A microsecond lookahead means many grants; a barriered group means none. |
| **Q2** | **Does it couple *scheduling decisions* across ranks?** | This is the one that can break Compass's central claim. A strategy that only changes cost is routine; one that changes *which batch runs* touches the part Compass reuses rather than models. |
| **Q3** | **What does it change about cost?** | Which structures exist, which collectives appear, how they scale with width. |
| **Q4** | **What does it change about memory?** | Which terms shard, which replicate, which are keyed by width. |

**The headline result, and it is the reason this topic is short:**

> **Only PP adds logical processes.** TP, DP and EP each sit behind a barrier that already
> exists in ATOM, so `01` D3's collapse rule absorbs all three. The clock protocol's cost
> is therefore a function of PP degree alone, not of total GPU count.

---

## D89. TP — the settled case

Nothing here is new; it is gathered so the frame has a worked instance and so the
width-keyed material has one index.

| | |
|---|---|
| **Q1 LPs** | **None added.** A TP group is one LP (`01` D3) — it already barriers on every collective. Rank-0 single-sourcing of the clock is measured correct: TP=2 over 1,727 steps, per-step rank difference median **0.03%**, worst 0.82%; TP=4 over 2,295 steps, rank totals within **±0.02%**, and charging every step to its *slowest* rank adds **0.06%**. |
| **Q2 scheduling** | **None.** One scheduler, one batch. |
| **Q3 cost** | Collectives priced per width, never scaled by a law — measured `cross_device_reduce_1stage` **9.08 / 10.40 / 10.95 µs** at TP 2/4/8 (+20.6%) while `allgather_lastdim` went **29.00 / 23.73 / 16.39 µs (−43.5%)** (`07` D40, `10` D66). Some structures exist only at width. |
| **Q4 memory** | Weights shard; `runtime_constants` are **tabulated per width and do not fit a law** — 926 / 6906 / 7266 / 10704 MiB at 1/2/4/8 (`03` D14, `05` D25). |
| **Open** | **T21** — whether the in-situ calibration transfers across width. The one unproven assumption in the recipe. |

**What TP supplies to the others:** the per-width discipline. Every strategy below inherits
the rule that *width is a key, not a parameter* — a number measured at one degree is not
evidence about another until it is shown to transfer.

---

## D90. DP — the step shape is decided by a collective, and that is the design problem

### Q2 first, because it is the one with teeth

DP is the only strategy that couples **scheduling decisions** across ranks. Under DP, the
shape a rank runs is not a function of what that rank scheduled.

`ForwardMode.decide` (`forward_context.py:227-300`) runs a DP collective *before* the
step's shape is settled, and settles the shape **from the result**:

```
   rank A schedules its batch  ---+
   rank B schedules its batch  ---+--> sync_dp_metadata()  --> unified_bs
   rank C schedules its batch  ---+    (one packed all_gather    any_rank_has_prefill
                                        of int32 per rank)       max_seqlen_q_across_dp
                                              |
                                              v
                                   the batch is REWRITTEN from the group's
                                   answer, then the ladder, the graph key,
                                   the draft pass and the attention plan are
                                   all read off it
```

The class docstring states the property and the hazard together:

> `running_bs` is agreed on **EVERY** step — it is the reduction on a ladder every rank
> shares — because a captured graph holds its collective at that batch and **a rank
> arriving with another one hangs the group**.

and, on why the chain is not separable:

> `decide` owns the whole chain … because the steps are not separable: the reduced query
> length rewrites the batch, and everything else is read off it afterwards.

`running_tokens` is agreed only when `running_tokens_are_unified`, i.e. when no rank has a
prefill; otherwise each rank runs its own count through the variable-length gather, whose
contract is that the rows it was handed are the rows *this* rank scheduled.

### There are two DP collectives, not one

`01` D3 cites the liveness one and is right about its conclusion, but the shape one is the
load-bearing one for this topic:

| | Where | What it carries | Cadence |
|---|---|---|---|
| **liveness / lockstep** | `EngineCore._sync_dp_state` (`engine_core.py:751-781`) | Gloo CPU `all_reduce(MAX)` over three booleans: `has_unfinished`, `shutdown`, `offloaded` | engine-core loop |
| **step shape** | `sync_dp_metadata` (`tbo/ubatching.py:199-224`), called from `ForwardMode.decide` | one packed `all_gather` of `n_fields` int32 per rank: DP token padding, the prefill fan-out, the cross-DP TBO gate, and the DSpark graph-shape MAX | **every forward** |

The second is cheap by construction — *"one all_gather of n_fields int32 values per rank
suffices"*, down from up to three `all_reduce`s, and with TBO off only the first three
fields are exchanged.

### Decision

> **Both DP collectives are *exactly reproducible* under simulation, because both reduce
> over scheduling metadata rather than over model outputs.**

This is the same argument as `03` D13's for the KV pool: the quantity is arithmetic over
integers Compass already has — `scheduled_tokens`, `scheduled_bs`, `is_prefill`, and the
TBO/DSpark scalars — so **running the real reduction is more faithful than modelling it,
and free.** Nothing in either collective depends on a logit, a hidden state, or anything a
device would have produced.

Consequences:

1. **Do not stub `sync_dp_metadata`.** Let it run. The simulated DP group reaches the same
   `unified_bs`, the same `any_rank_has_prefill`, the same ladder rung — by computation,
   not by assumption.
2. **The DP group stays one LP** (`01` D3), and the shape collective is a *second* barrier
   confirming it: with a blocking all-gather at the head of every forward, DP ranks cannot
   drift by more than one step.
3. **It is a category-A or category-B wait** in `01` D4's taxonomy, and which one depends
   on where the LP boundary falls. Since the DP group is one LP, the collective is
   **internal to an LP** — so by `01` D4's own rule it is **ignored**: its duration is not
   observable in the simulated result as a cross-LP wait. What *is* charged is its cost,
   as a priced collective (Q3 below).

### Q1, Q3, Q4

| | |
|---|---|
| **Q1 LPs** | **None added.** One LP per DP group, barriered twice per step. |
| **Q3 cost** | Two collectives per step, both tiny and both on the **CPU/Gloo** path for the liveness one. Priced like any other collective (`07` D40), per width. The all-gather payload is `n_fields × dp_size` int32 — bandwidth-irrelevant, latency-dominated, which is the regime `10` D66 already models. |
| **Q4 memory** | Weights and KV **replicate** per DP rank; nothing shards. So the memory model is per-replica and the width key is DP-independent — DP is the one strategy that does *not* add a `runtime_constants` dimension. |

### The rule that bites harder under DP than under TP

`01` charges a step to its **slowest rank**, and measured that this costs 0.06% at TP=4
because TP ranks are near-identical. **DP ranks are not.** Different DP ranks hold
different requests and therefore genuinely different batches, so "slowest rank" is not a
rounding correction here — it is the definition of the step, and it is what the
`unified_bs` padding exists to express.

Two things follow:

- **The LP's step duration is `max` over DP ranks, and it must be computed, not
  approximated by rank 0.** Rank-0 single-sourcing is a TP result and does not transfer to
  DP.
- **Idle DP ranks still cost a step.** `EngineCore._execute_dummy_batch` runs
  `dummy_execution` on ranks with nothing to do (`engine_core.py:748-749`), so a rank with
  no work is not free and must be priced as the dummy shape rather than as zero.

---

## D91. PP — the only strategy that adds logical processes

### Q1: N LPs at microsecond lookahead

Each PP stage is its own `EngineCore` (`01` D1: `dp_size × pp_size` engine cores), and
stages communicate by point-to-point transfer rather than by a barrier — so `01` D3's
collapse rule does **not** absorb them. PP degree `P` multiplies the LP count by `P`.

The lookahead between adjacent stages is the p2p latency, which is **microseconds**. That
is the expensive case for a conservative protocol: small lookahead means frequent grants.
`01` D3.1 already anticipated this and gives the partition rule —

> PP stages have microsecond lookahead and **must stay under one node-local CA**; PD role
> boundaries have millisecond lookahead and are the cheap links to cross a node.

**Decision: adopt that rule as a constraint, not a preference.** A PP stage boundary is
never the cut point for a hierarchical CA, and for M4/M6 the standalone CA sits at the PD
boundary with each role's PP stages under one local authority.

### Q2: no scheduling coupling

Unlike DP, PP does not reduce anything into the batch. The batch is decided once, by the
first stage's scheduler; later stages execute a shape they are handed. So PP adds LPs
without adding a decision coupling — the opposite trade to DP.

### Q3: cost — the transfer is a size, not a tensor

ATOM's PP transport (`atom/distributed/pp_comm.py`):

| Call | Nature |
|---|---|
| `send_intermediate_tensors` (`:96-101`) | **blocking** send of `hidden_states` / `residual` to the next stage |
| `async_send_intermediate_tensors` (`:127-157`) | non-blocking `isend`, metadata then buffers |
| `commit_pp_send_work` (`:159-162`) | blocks until in-flight `isend`s complete |
| `pp_send_allgather_group` (`:28-38`) | TP group for PP send-allgather, `None` if disabled or tp=1 |

`flush_pp_send` is already in the RPC surface a replacement runner must answer (`02` D10).

**The payload is `hidden_states` + `residual` — real tensors that a simulated run never
materialises.** So PP transfer is treated exactly like KV transfer (`01` D6): a **size
computed from geometry** (`tokens × hidden × dtype`, plus residual) divided by the
interconnect bandwidth from the machine spec, plus its latency. Priced from the spec, not
measured, which is what keeps interconnect configurable.

The metadata `isend`s are latency-only. The `pp_send_allgather_group` path adds a TP-wide
all-gather before the send when enabled, which is an ordinary priced collective.

### Q4: memory — layers split, and the split is ATOM's

`get_pp_indices(num_hidden_layers, pp_rank, pp_size)` (`atom/models/utils.py`, used at
`kv_transfer/offload/config.py:332,439` and in the Mooncake connector) owns the layer
range per stage. `ModelRunner._get_total_num_layers` already consults `get_pp_indices`
when `pp_group.world_size > 1`.

**Compass calls it; it never re-derives a split.** Same rule as `14` D86 for draft KV
layers, and for the same reason: two spellings of one count drift.

**Weights shard by layer range. KV does not.** What separates them is homogeneity, not
per-layer-ness: KV is per-layer too. *Every* layer carries weights, so a stage's share of
the sharded weight terms is `layers_in_stage / total_layers`; only the layers that cache
the whole history carry paged KV, and inside a span of a hybrid stack that count is not
proportional to the span's length. A stage's share of the KV term is
`paged_layers_in_stage / total_paged_layers`, and on a hybrid the two fractions differ.

The weights ratio is itself exact only where the layer kinds are equally sized, and on this
hybrid they are merely close. The two kinds are different modules — `Qwen3NextAttention`
against `Qwen3_5GatedDeltaNet` — and at TP1, counted from the vendored config over their
projection shapes with norms and biases omitted (under 0.01% of a layer), a
`full_attention` layer is **372,244,480** parameters against **383,262,720** for a
`linear_attention` one, **+2.96%**. That moves a stage's true weight share off the layer
ratio by at most **0.242%**, at pp = 7; 0.197% at pp = 6 and 0.104% at pp = 3. Two orders
of magnitude inside the ≤10% target a non-KV memory term carries, so the layer range stays
the weights key here — but it stays one because the two kinds happen to be nearly the same
size, not because weights are per-layer, and on a hybrid whose kinds differ more it would
not. How close that ratio has to be is a weights-term question rather than this decision's.

**The paged count is read, not computed.** It comes from the same two sources as the split
itself — `get_pp_indices` for the span, and the model's own `layer_types` for which layers
inside it are paged (`atom/compass/backends/geometry.py`'s `paged_layers`, which refuses a
kind it does not recognise rather than counting it as ordinary attention). An earlier form
of this paragraph gave the KV share as the layer ratio. That is exactly the re-derivation
the rule above forbids: it recomputed from the layer count something ATOM already holds,
and the two spellings disagree on every hybrid.

**A second spelling of this count already exists in the engine, and it is not the one to
read.** `GDNStateMixin._init_gdn_state` (`atom/model_ops/attentions/gdn_attn.py`) sets
`num_full_attn` by dividing `num_hidden_layers` by `full_attention_interval`, and the
attention sizing in the same file consumes it. That count is **global**, so it cannot
answer what a stage holds at all. It agrees at 16 on this config and not by luck:
`Qwen3_5TextConfig.__init__` (`atom/model_config/qwen3_5.py`) fills `layer_types` *from*
that interval when a config omits it, so anything built through ATOM's own config class is
consistent by construction. A config carrying an explicit non-periodic `layer_types` would
separate the two, and `layer_types` is the one that stays right.

Derived at `feature/atomcompass_new` `92f1fdafe` from `get_pp_indices`, with
`VLLM_PP_LAYER_PARTITION` cleared — left set it overrides the partitioner and a layout out
of the environment reads as ATOM's — on the 64-layer hybrid vendored at
`tests/compass/qwen3_5_27b_config.json`: one `full_attention` layer in four, so 16 of the
64 are paged. The layers that do not divide evenly are added walking back from the
*second-to-last* partition, so the **last stage never takes one** and the rest fill in from
the right; at pp = 5 that reaches stage 0, the first. *The middle stages* is true of 3, 6
and 7 here and false of 5, which is why no row below is the split a reader would write
out:

| PP | `get_pp_indices` spans | layers held | paged layers | stages where `held/64` = `paged/16` |
|---|---|---|---|---|
| 2 | 0-32, 32-64 | 32, 32 | 8, 8 | all |
| 3 | 0-21, 21-43, 43-64 | 21, 22, 21 | 5, 5, 6 | none |
| 4 | 0-16, 16-32, 32-48, 48-64 | 16, 16, 16, 16 | 4, 4, 4, 4 | all |
| 5 | 0-13, 13-26, 26-39, 39-52, 52-64 | 13, 13, 13, 13, 12 | 3, 3, 3, 4, 3 | stage 4 |
| 6 | 0-10, 10-21, 21-32, 32-43, 43-54, 54-64 | 10, 11, 11, 11, 11, 10 | 2, 3, 3, 2, 3, 3 | none |
| 7 | 0-9, 9-18, 18-27, 27-36, 36-45, 45-55, 55-64 | 9, 9, 9, 9, 9, 10, 9 | 2, 2, 2, 3, 2, 2, 3 | none |
| 8 | 0-8, 8-16, 16-24, 24-32, 32-40, 40-48, 48-56, 56-64 | 8, 8, 8, 8, 8, 8, 8, 8 | 2, 2, 2, 2, 2, 2, 2, 2 | all |

**The pp = 6 row falsifies the layer ratio in one line: stages 1 and 3 each hold 11 layers
and carry 3 and 2 paged ones.** `layers_in_stage / total_layers` is 11/64 for both, and
their shares of the KV are 3/16 and 2/16. The ratio understates stage 1 by 8.3% and
overstates stage 3 by 37.5% — against the **≤ 5%** target the KV term carries, the
tightest in the project, where every other memory term is gated at 10%. The sign of the
error changes between two stages of one deployment, so no single correction factor absorbs
it. At pp = 3 the stage holding the *most* layers holds the *fewest* paged ones of the
three: 22 layers carry 5 where 21 carry 6.

**Uniform stacks are unaffected.** Where every layer is paged the two fractions are equal
by construction and the layer ratio is correct — that is the case anyone checks first, and
it is why this stood. The widths above where they agree at every stage, 2, 4 and 8, are
the ones where the paged period of this model, 4, divides every span.

The Class-C `runtime_constants` scale by neither key, since HIP context and collective
buffers are per-process; that half of the sentence this replaces was already right.
**Memory readings must be keyed by PP degree as well as TP width**, and that is a new
dimension on `05` D25's table rather than a derivation.

The per-stage *block count* these paged counts produce, and the `all_reduce(MIN)` that
picks one of them for the whole deployment, are `03` D14's question and not this one's.

### What is not established

**Whether ATOM microbatches PP.** A grep for `microbatch` / `micro_batch` / `num_micro`
across `pp_comm.py` and `engine_core.py` returns nothing, which suggests stage-to-stage
execution of whole batches rather than a 1F1B-style pipeline. That matters a lot — without
microbatching, PP's bubble is the whole of a stage's idle time and the LP structure is
simple; with it, each microbatch is a separate event and the grant count rises by the
microbatch factor. **Recorded as T64**, and it is the first thing to settle before any PP
work.

---

## D92. EP — invisible communication and an exclusive-occupancy kernel

### The EP group, per supported configuration

**Measured 2026-09-21 by P0.6, answering T65.** ATOM at `7fc7a5ddd`; aiter at
`23f83724f` (`v0.1.20-103-g23f83724f`) in container `jgong5_vllm`, re-checked against
`f4e7c7509` (`v0.1.21.dev0-49-gf4e7c7509`) in node 18's `xiaobizh_n18` and
`xiaobizh_n18_cpu`, where every aiter line quoted below is byte-identical at the same
numbers. Two aiter versions are in circulation on this project and they agree here; that
they agree is a reading, not a guarantee, and no artifact key currently records which one
answered — **T86**.

**ATOM does not construct the EP group — aiter does.** No ATOM file assigns `_EP`, and
every use site imports `get_ep_group` from `aiter.dist.parallel_state` (`moe.py:599`,
`fused_moe/mori_v2_prepare_finalize.py:153,636`, `fused_moe/flydsl_mega_experts.py:186`,
`eplb.py:1768`, `models/glm4_moe.py:96`, `models/qwen3_next.py:171`,
`model_runner.py:3580`). Both of ATOM's distributed-init paths —
`init_pp_aware_dist_env` (`distributed/pp_comm.py:46`) when `pp_size > 1`, aiter's
`init_dist_env` (`aiter/ops/communication.py:22`) otherwise, chosen in
`_setup_device_and_distributed` (`model_runner.py:946`, the branch at `:981`) — end in
aiter's `initialize_model_parallel`, which builds the
group at `aiter/dist/parallel_state.py:1926-1945` out of

```python
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(-1, data_parallel_size
                 * prefill_context_model_parallel_size
                 * tensor_model_parallel_size)
    .unbind(0)
)
```

over `all_ranks = torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)`
(`parallel_state.py:1833-1839`). The `transpose(1, 2)` swaps DP and PP, so **the EP group
is every rank of one PP stage — it spans DP, PCP and TP, its size is `dp × pcp × tp`, and
there is one group per PP stage.** aiter says so itself:
*"all2all lives in ep group, which is merged from dp and tp group"*
(`dist/device_communicators/base_device_communicator.py:42`).

Membership, from re-executing those expressions verbatim at each configuration ATOM
ships or documents:

| Configuration | world | dp/pp/pcp/tp | EP groups | size |
|---|---|---|---|---|
| `-tp 8` (`recipes/Qwen3-235b.md:24`) | 8 | 1/1/1/8 | `[0…7]` | 8 |
| `-tp 2 --enable-dp-attention` (`recipes/GPT-OSS.md:34`) | 2 | 2/1/1/1 | `[0,1]` | 2 |
| `-tp 8 --enable-dp-attention` (`recipes/TBO.md:63`, `recipes/GLM-5.md:205`, `recipes/DeepSeek-V4.md:126`) | 8 | 8/1/1/1 | `[0…7]` | 8 |
| `-tp 4 -dp 2` (`docs/distributed_guide.md:20`) | 8 | 2/1/1/4 | `[0…7]` | 8 |
| `-tp 2 -pp 2` | 4 | 1/2/1/2 | `[0,1]`, `[2,3]` | 2 |
| `-tp 4 -pcp 2` | 8 | 1/1/2/4 | `[0…7]` | 8 |
| `-tp 2 -pp 2 -dp 2` — **refused** at `engine_core_mgr.py:297-300` | 8 | 2/2/1/2 | `[0,1,4,5]`, `[2,3,6,7]` | 4 |

Under DP-attention `CoreManager` rewrites `dp := dp × tp, tp := 1` before any of this
(`engine_core_mgr.py:281-295`), which is why those rows carry `tp 1`. The torch world is
`dp × pcp × tp` either way: `init_dist_env` passes `world_size = pp × tp × pcp` with `pp`
pinned to 1 (`aiter/ops/communication.py:33-40`) and `init_distributed_environment`
multiplies DP back in (`parallel_state.py:1726-1729`); the PP branch computes the same
index itself at `model_runner.py:986-988`.

**Ranks are contiguous whenever `pp == 1`**, which is every reachable EP configuration.
The strided last row is the only non-contiguous case, and ATOM refuses it
(*"Pipeline parallel combined with data parallel is not supported yet."*).

**Membership is not topology-aware.** It is index arithmetic on
`torch.arange(world_size)`; nothing reads the interconnect. *Kernel selection* is:
`All2AllManagerBase` sets
`self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))`
(`base_device_communicator.py:55`), MoRI picks `IntraNode` against `InterNodeV1` from it,
and ATOM deliberately shares that one probe rather than re-deriving it from a width
(`moe.py:692-698`).

**Two configurations where ATOM does not own the answer at all.** Under the vLLM plugin
ATOM adopts vLLM's group wholesale — `aiter_ps._EP = getattr(vllm_ps, "_EP", None)`
(`plugin/vllm/tp_group_reuse.py:150-152`) — so membership is vLLM's decision there, and a
vLLM configuration with no `_EP` installs `None` and fails at the first `get_ep_group()`
(`parallel_state.py:1586`). And EP width is never requested directly: ATOM has no
`--ep-size` flag (`model_engine/arg_utils.py:271-275` offers only the boolean
`--enable-expert-parallel`), and the SGLang and rtp-llm frontends, which do carry an
`ep_size`, have it reduced to that boolean (`plugin/config.py:612,690`) — so `ep_size=2`
under `-tp 8` is accepted and silently means 8.

### `moe_parallel_config.ep_size` is a second number, and it is not the group size

`FusedMoEParallelConfig.make` computes its own: `ep_size = tp_size; ep_rank = tp_rank`
(`moe.py:299-300`), where `tp_size` has been flattened across DP **only if**
`enable_dp_attention or moe_ep_flatten_tp_across_dp` (`moe.py:240-242`, `252-256`) and
folded with PCP only under `ATOM_PCP_MOE_MERGE` (`moe.py:272-280`). The two numbers agree
in every configuration ATOM ships a recipe for, and disagree in one it documents:

| Configuration | group size (aiter) | `moe_parallel_config.ep_size` | agree |
|---|---|---|---|
| `-tp N`, DP 1 | `N` | `N` | yes |
| `-tp N --enable-dp-attention` | `N` | `N` | yes |
| vLLM plugin `--enable-expert-parallel` (`plugin/config.py:361`) | `dp × tp` | `dp × tp` | yes |
| **`-tp 4 -dp 2 --enable-expert-parallel`, no DP-attention** | **8** | **4** | **no** |
| `-pcp P` without `ATOM_PCP_MOE_MERGE` | `P × tp` | `tp` | **no** |

MoRI v1 is handed `num_ep_ranks` from the group and `num_local_experts` from the config
number (`moe.py:654,664`); MoRI v2 derives both from the group
(`mori_v2_prepare_finalize.py:640,666`). The two paths therefore disagree exactly where
the two numbers do. **Compass must carry both and assert they agree**, because ATOM
asserts nothing here — recorded as **T83**, which also covers `local_ep_size`
(`moe.py:313-314`, MoRI's `gpu_per_node`) omitting PCP while the group includes it.

### EP without DP runs no all-to-all at all

`use_all2all_kernels` requires `dp_size > 1` (`moe.py:201-211`) and is the sole gate on
building `MoriPrepareAndFinalize` (`moe.py:736-742`); without it `self.fused_experts`
stays `None` (`moe.py:756-759`) and the layer falls through to a plain
`fused_moe(…, expert_mask=…)` (`moe.py:907-915`). So under
`-tp 8 --enable-expert-parallel` — `recipes/Qwen3-235b.md:24`, the flagship EP recipe —
the MoE is masked local-expert compute plus the ordinary TP all-reduce
(`moe.py:4370-4374`), and moves **zero all-to-all bytes**. That is the mechanism behind
`04`'s *"at `ep_size == tp_size` EP is close to a no-op"*, and it is stronger than close.

**A cost model must refuse to price a MoRI all-to-all at `dp_size == 1` rather than price
zero bytes** — a confident, precise, fictional number is the archetypal failure `README`
names. The gate to mirror is `moe.py:201-211` in full, including `dp_logical_ratio == 1`
and `_has_module("mori")`. One exception: `--moe-backend mega` installs `MegaFusedExperts`
unconditionally (`moe.py:1738-1753`) and reads the group directly for its rank and world
(`flydsl_mega_experts.py:186-193`), so it does run peer-to-peer at `dp_size == 1`;
`config.py:1677-1681` refuses `mega` without EP.

### Q1: no LPs added — the conclusion holds, the reason under it did not

**The conclusion stands, and D93's formula is unchanged.** The EP group is exactly the set
of ranks of one PP stage, and PP is the only LP-adding dimension (D88, D93). The one case
where EP's membership could cut across an LP boundary is `pp > 1` together with `dp > 1`,
and `engine_core_mgr.py:297-300` refuses it.

**The reason previously given here was wrong, and is recorded rather than quietly
dropped**, because a future reader who lifts that refusal will need to know which half
survived. It read that EP *"inherits the TP group"*, citing `moe.py:265`; that line is a
comment inside the PCP-merge block explaining that the *integers* `ep_size`/`ep_rank`
inherit `tp_size`/`tp_rank`, and says nothing about the communicator. The caveat below it
was stated conditionally — *"if a deployment configures EP to span the DP dimension"* —
and EP spans DP in every configuration where DP exists. The second cross-DP
synchronisation that caveat predicted **does** appear: MoRI dispatch/combine, over a
membership different from `sync_dp_metadata`'s. ATOM confirms the two groups are distinct
objects in code — `eplb.py:1784-1793` compares the DP group's global ranks against the EP
group's to decide `_dp_is_migration_group`, a comparison with no purpose if they were the
same group.

`get_max_tokens_across_dispatchers` (`moe.py:495`) was cited here as hinting at a
cross-dispatcher reduction. It is `def …(input): return input.item()` — no collective —
and a tree-wide `grep -rn` returns the definition and this document. It has no callers.

**D92's decision-log row below is left unamended**: its parentheticals *"inherits the TP
group"* and *"remainder included"* are the two claims this section corrects, and rewriting
a decision is the project owner's call. Registered as a pending amendment in
[`12_open_items.md`](12_open_items.md) §5.

### Q2: no scheduling coupling

Expert assignment is a function of the routing computed inside the layer, not of a
scheduler decision — and **real routing is data-dependent, so it is not derivable from
geometry**. The closed-form flat ring
`expert_ids = (p % ep_size) × L + (p // ep_size) % L` is `init_balance_router_logits`
(`moe.py:137-152`), the **synthetic** router built only under `--fake-eplb`
(`moe.py:2974-2989`: *"if atom_config.fake_eplb else None"*). Class A under `--fake-eplb`,
and nothing outside it.

### Q3: cost — the hard part, and it is already characterised

From `07` D40 class (c): MoE all-to-all is **MORI, called inside `moe_forward`, invisible
at any degree**. Zero dispatcher events, so it needs `04` D21's *declared node* plus the
declared-entry-point benchmark of `07` D40.1 — the same tool class (b) uses.

And the property no generic cost model would capture:

> `_get_dispatch_config` caps `block_num` at `get_cu_num()` because dispatch and combine
> use a **grid-wide spin barrier requiring all blocks co-resident**, so the kernel occupies
> the entire device. On an 80-CU MI308X, launching 128 blocks **deadlocks**.

That is `04` D19's `exclusive` join policy, and it is a **measured fact rather than a
modelling choice** — an EP all-to-all cannot overlap with anything, so the IR must not
place it in a `Par`. The cap is two numbers and not one —
`min(128, CU)` blocks at 16 warps for prefill against `min(64, CU)` at 4 for decode
(`fused_moe/mori_prepare_finalize.py:257-261`), which on the 80-CU MI308X is 80 and 64 —
recorded as **T87**, since `07`'s price-list table states it as a single cell.

### Q4: memory — experts shard contiguously, and an indivisible count is refused

`determine_expert_map` (`fused_moe/expert_layout.py:111-152`) gives rank `r` the
contiguous run `[r×L, (r+1)×L)` with `L = E // ep_size`, and gives any remainder to the
**last** rank (`expert_layout.py:147-152`) — it is not left unused. Expert weight bytes per
rank are `L × bytes_per_expert`, exact from geometry (Class A).

**There is no remainder for a memory model to reproduce**, because a configuration whose
expert count does not divide by `ep_size` is refused outright:
`assert self.global_num_experts % self.ep_size == 0` whenever `use_ep`
(`moe.py:2758-2763`). MoRI derives a token's destination as
`expert_id // num_experts_per_rank` (`distributed/simulated_tp.py:105-107`,
`moe.py:203-206`) and cannot represent an uneven last rank, so ATOM refuses rather than
pads. The *"a remainder is left unused"* comment this section used to quote is
`moe.py:151`, inside the `--fake-eplb` synthetic router, not the real path.

---

## D93. Composition, and what the LP count actually is

Strategies compose, and the LP count is what the clock protocol pays for:

```
  LPs  =  (number of PD roles)                     1 for aggregated, 2 for disaggregated
          x (dp_size)          <- collapses to 1 per group; a DP GROUP is one LP
          x (pp_size)          <- DOES NOT collapse: one LP per stage
          x 1                  <- tp_size and ep_size collapse into their group
        + 1                    the harness
        + 1                    the API server
```

So a TP8/EP8 single-node aggregated deployment is **~3 LPs**, the same as TP1 — and a
PP4 deployment is four times that. **The protocol's cost tracks PP degree and PD roles,
not GPU count**, which is the property that makes `01`'s single-CA decision hold as the
milestones widen.

**The composition worth watching** is DP × PP: `dp_size × pp_size` engine cores means the
shape collective of D90 runs *per stage*, across DP, at microsecond-lookahead boundaries.
That is the densest grant traffic any milestone produces and it lands at M7. Sizing it
before M7 starts is cheaper than discovering it.

---

## D94. What M1 needs, and what waits for M7

The split, because "parallelism support" is otherwise read as one late lump:

| | **M1 — plumbing, with fake models** | **M7 — accuracy, with real ones** |
|---|---|---|
| **TP** | LP collapse; width as an artifact key | priced collectives per width; T21 |
| **DP** | both collectives run for real; `max`-over-ranks step duration; dummy-batch pricing for idle ranks | the collectives' own cost |
| **PP** | **one LP per stage**; transfer as a size from the spec; layer split via `get_pp_indices`; memory keyed by PP degree | bubble fidelity; microbatching if it exists (T64) |
| **EP** | group membership established — it is `dp × pcp × tp` within one PP stage, built in aiter (D92); `exclusive` occupancy honoured in the IR | MORI all-to-all priced via declared nodes |

**The M1 test that matters** is not "does it produce plausible numbers" — the fake model
guarantees it will. It is: **does a fake-model run at each of TP2 / DP2 / PP2 / EP2 reach
the same scheduling decisions as the real engine at the same configuration?** *Which*
EP2 is now an open question rather than a detail: `-tp 2 --enable-expert-parallel` at
DP 1 runs no all-to-all at all (D92), so that leg would pass while exercising nothing,
and the only non-degenerate EP2 is `-tp 2 --enable-dp-attention --enable-expert-parallel`.
**T84.** That aside, the test is
checkable without a cost model, it exercises exactly the couplings this topic is about,
and ATOM's own `test_dp_load_balance.py`, `test_dp_metadata.py`, `test_dp_sync_layout.py`
and `test_forward_mode.py` already cover the pieces on the CPU-only path (`08` D43.1).

---

## Open issues

- **T64 — does ATOM microbatch PP?** Unestablished, and it changes both the LP event rate
  and the bubble model. Settle before any PP work.
- **T65 — EP group membership** per supported configuration. **Answered 2026-09-21 by
  P0.6**, in D92: the group is `dp × pcp × tp` within one PP stage, built in aiter rather
  than in ATOM. EP does span DP wherever DP exists, and the second cross-DP barrier does
  appear — but the LP collapse survives anyway, because the group is exactly one PP
  stage's ranks and PP>1 with DP>1 is refused. Three successors are open: **T83** (the
  group size and `moe_parallel_config.ep_size` disagree in two configurations, one of
  them documented), **T84** (which EP2 D94's M1 test means) and **T85** (multi-node EP
  rank-to-node mapping, unverified).
- **PP degree is a new key on the memory readings table** (`05` D25), and nothing has
  measured whether the Class-C constants move with it. One startup per PP degree settles
  it; recorded as **T66**.
- **The DP `max`-over-ranks rule is unmeasured.** `01`'s 0.06% figure is a TP result. What
  the spread across DP ranks actually is — and therefore how much the padding to
  `unified_bs` costs — has not been measured. **T67**.
- Nothing here covers **PCP / DCP** (`pcp_size`, `dcp_world_size`), which appear in the
  topology and in `03`'s note that `hash_block_size = block_size × dcp_world_size`. They
  are out of scope for M1–M7 as written, and recorded so they are not discovered late.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D88 | One frame of four questions per strategy — LPs and lookahead, scheduling coupling, cost, memory. **Only PP adds logical processes**; TP, DP and EP each sit behind an existing barrier. | 2026-09-19 |
| D89 | TP is the settled instance and supplies the per-width discipline: width is a key, not a parameter. | 2026-09-19 |
| D90 | DP's two collectives **run for real** — both reduce over scheduling metadata, never over model outputs, so the real reduction is more faithful than a model and free. The DP group stays one LP. Step duration is `max` over ranks, computed not rank-0-sourced, and idle ranks cost a dummy batch. | 2026-09-19 |
| D91 | PP is one LP per stage at microsecond lookahead, and PP boundaries are never a hierarchical-CA cut point. The inter-stage transfer is a **size from the machine spec**, like KV transfer. Layer split comes from `get_pp_indices`, never re-derived; weights shard by that range but **KV shards by the paged-layer count inside it**, which on a hybrid is not proportional to it. Memory readings gain a PP-degree key. | 2026-09-19 |
| D92 | EP adds no LPs (inherits the TP group) but its all-to-all is invisible and must be a declared node, and its `exclusive` occupancy forbids placing it in a `Par`. Expert sharding is Class A, remainder included. | 2026-09-19 |
| D93 | LP count = PD roles × PP stages (+2), independent of GPU count. The clock protocol's cost tracks PP degree, not width. | 2026-09-19 |
| D94 | Parallelism splits across milestones: LP structure, couplings and memory shape at **M1**; cost accuracy at **M7**. M1's test is scheduling-decision agreement at TP2/DP2/PP2/EP2, which needs no cost model. | 2026-09-19 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T64 | Establish whether ATOM microbatches PP — changes the LP event rate and the bubble model | a grep finds nothing; needs reading `pp_transport.py` and one PP2 run |
| ~~T65~~ | ~~Establish EP's group membership per supported configuration; if EP spans DP, D92's collapse does not hold~~ — **answered 2026-09-21 by P0.6**, in D92 above; successors T83, T84, T85 | — |
| T83 | The EP group's size and `moe_parallel_config.ep_size` disagree at `-tp N -dp M --enable-expert-parallel` without DP-attention (group `N×M`, config `N`) and at `-pcp P` without `ATOM_PCP_MOE_MERGE`; `local_ep_size` also omits PCP while the group includes it | one 8-GPU startup logging `all2all_manager.world_size` against `moe.num_local_experts`; or an owner statement that the combination is unsupported |
| T84 | Decide which EP2 D94's M1 scheduling-agreement test means — `-tp 2 --enable-expert-parallel` at DP 1 runs no all-to-all, so that leg exercises nothing | an owner decision, then one line in D94 |
| T85 | Multi-node EP rank-to-node mapping is assumed, not verified: MoRI infers node identity as `ep_rank // gpu_per_node`, which needs consecutive EP ranks to be physically consecutive GPUs | a 2-node DP+EP run logging `all2all_manager.internode` and each rank's EP group; M7-era |
| T66 | Measure whether the Class-C runtime constants move with PP degree | one engine startup per PP degree |
| T67 | Measure the step-duration spread across DP ranks, and what padding to `unified_bs` costs | needs a DP2 run with per-rank step timing |
| T78 | `qwen3_5.py:427` and `glm4_moe.py:426` declare `"intermediate_tensors": 0`, so neither model can run PP at compilation level >= 2 | upstream ATOM fix; `15` D94's PP2 test is fake-model and CPU-only, so this is not on M1's path |
| T79 | `gdn_attn.py:1329-1331` mixes a PP-local layer count with a global one; KV sizing sign-flips at PP2 | upstream ATOM fix; blocks real-model PP measurement, not M1 |
| T82 | D91 Q2's "no scheduling coupling" is contradicted by `scheduler.py:1761-1763`, which skips `_pp_inflight_token_block` seqs inside the decode admission loop | the amendment to D91 and its LP consequence is its own task; this PR registers the contradiction rather than rewriting the decision |
