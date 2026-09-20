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

Weights and KV both shard by layer range, so per-stage memory is `layers_in_stage /
total_layers` of the sharded terms — but the Class-C `runtime_constants` do **not** scale
that way, since HIP context and collective buffers are per-process. **Memory readings must
be keyed by PP degree as well as TP width**, and that is a new dimension on `05` D25's
table rather than a derivation.

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

### Q1: no LPs added, with one caveat

EP shards experts within a group that **inherits the TP group** — *"ep_size/ep_rank below
inherit tp_size/tp_rank"* (`moe.py:265`). So EP lives inside a group that is already one
LP, and adds none.

**The caveat, and it is a real one:** if a deployment configures EP to span the DP
dimension as well, its all-to-all becomes a second cross-DP synchronisation with a
different membership from `sync_dp_metadata`'s. `get_max_tokens_across_dispatchers`
(`moe.py:495`) hints at a cross-dispatcher reduction whose group is not established here.
**Recorded as T65**: establish EP's group membership per supported configuration before
assuming the collapse holds.

### Q2: no scheduling coupling

Expert assignment is a function of the routing computed inside the layer, not of a
scheduler decision. The flat-ring assignment is closed form —
`expert_ids = (p % ep_size) * L + (p // ep_size) % L` where `L = E // ep_size` and
`p = t * topk + j` (`moe.py:139-174`) — so it is Class-A derivable from geometry if a cost
model ever needs it.

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
place it in a `Par`.

### Q4: memory — experts shard, and the remainder is dropped

`L = E // ep_size` experts per rank, *"a remainder is left unused"* (`moe.py:151`). So
expert weight bytes per rank are `L × bytes_per_expert`, exact from geometry (Class A),
and a configuration whose expert count does not divide by `ep_size` wastes the remainder —
which a memory model must reproduce rather than round.

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
| **EP** | group membership established (T65); `exclusive` occupancy honoured in the IR | MORI all-to-all priced via declared nodes |

**The M1 test that matters** is not "does it produce plausible numbers" — the fake model
guarantees it will. It is: **does a fake-model run at each of TP2 / DP2 / PP2 / EP2 reach
the same scheduling decisions as the real engine at the same configuration?** That is
checkable without a cost model, it exercises exactly the couplings this topic is about,
and ATOM's own `test_dp_load_balance.py`, `test_dp_metadata.py`, `test_dp_sync_layout.py`
and `test_forward_mode.py` already cover the pieces on the CPU-only path (`08` D43.1).

---

## Open issues

- **T64 — does ATOM microbatch PP?** Unestablished, and it changes both the LP event rate
  and the bubble model. Settle before any PP work.
- **T65 — EP group membership** per supported configuration. If EP ever spans DP, the
  collapse in D92 Q1 does not hold and a second cross-DP barrier appears.
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
| D91 | PP is one LP per stage at microsecond lookahead, and PP boundaries are never a hierarchical-CA cut point. The inter-stage transfer is a **size from the machine spec**, like KV transfer. Layer split comes from `get_pp_indices`, never re-derived. Memory readings gain a PP-degree key. | 2026-09-19 |
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
| T65 | Establish EP's group membership per supported configuration; if EP spans DP, D92's collapse does not hold | needs a deployed EP configuration to inspect |
| T66 | Measure whether the Class-C runtime constants move with PP degree | one engine startup per PP degree |
| T67 | Measure the step-duration spread across DP ranks, and what padding to `unified_bs` costs | needs a DP2 run with per-rank step timing |
