# ATOM Compass — Design Topic 16: The Execution Plan

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** all of `01`–`15`. This document turns 108 decisions and 70 open items into
work that can be allocated.

**Scope.** How the work is organised, allocated and gated; what happens first; and what
each stage must produce. It is **detailed for Phase 0 through Wave 3 and deliberately
coarse for Wave 4 onward** — the later waves depend on answers Phase 0 has not produced
yet, and a detailed plan written against unknown answers is a fiction. Detail is added as
the answers arrive.

---

## D95. The operating model

### Tasks are a pool, not a track assignment

Work is a **DAG of tasks**. A task becomes claimable when its dependencies land; any free
agent claims it. There are no permanent per-track ownerships.

The cold-start problem this creates — an agent picking up a task without the accumulated
context of the ones before it — is solved by making context **durable in the task** rather
than resident in an agent. See D96.

**Concurrency is capped at 5 tasks in flight**: five developer agents and five reviewer
agents. The cap is review throughput, not the DAG — Wave 1 holds about ten independent
tasks and will run in two batches rather than all at once.

### Conflicts are tolerated, and are a signal

Tasks are cut so that each touches one module plus its tests, which makes most of them
disjoint. They are **not** guaranteed disjoint, and no allocation-time file locking is
imposed — a merge conflict is cheaper than the machinery to prevent it.

> **Frequent conflicts mean the task decomposition is wrong, not that coordination
> failed.** The response is to re-cut the tasks, not to add a scheduler.

### Developer and reviewer are separate agents

Both are per-task. They share the project context and the task record; their **briefs
differ in what they optimise for**:

| | Optimises for |
|---|---|
| **developer** | does this work, and does it match the design decision it cites |
| **reviewer** | what does this break, what does the design actually say, and what here is untested |

Same context, opposed objectives. The reviewer is a separate agent specifically so the
review is not performed by the context that produced the code.

### The halt rule

> **When something does not work as expected, stop and discuss. Do not work around it.**

This is the agent-facing form of the design's own standing principle — *refuse rather than
fall back* (`README` principle 6). A workaround improvised under build pressure is exactly
the class of decision that never gets written down, and the prior effort has recorded
instances of well-formed, wrong artifacts surviving review.

"Not as expected" includes: a design document that contradicts the code; a test that fails
for a reason the task did not predict; a measurement outside its stated range; an
interface that cannot be implemented as specified.

---

## D96. The task record, and why context lives in it

A task carries four sections. The first is written before the task is claimable; the rest
are written as it runs.

| Section | Written by | Contains |
|---|---|---|
| **Brief** | the planner | what to build; the governing decisions by number; the interfaces it **implements** and **consumes**; its file set; its exit criteria; its effort estimate |
| **Dev record** | the developer | what was found, what was decided that the design did not cover, what surprised it, what was left undone |
| **Review record** | the reviewer | what was checked, what was accepted with reservation, what the next task in this area should watch |
| **Handoff** | both | what a successor needs to know that is not in the code |

**Every task's brief links to its predecessors' records.** That is what makes continuity
survive agent turnover: the context is in the graph, not in a context window.

**A brief that cannot name its file set is under-specified** and is not claimable. This is
the same discipline as `04` D21's declared nodes — the declaration is the contract.

---

## D97. Where work lands

| | |
|---|---|
| **Integration branch** | `feature/atomcompass_new` — already the PR #3 branch |
| **Per-task isolation** | a git worktree per in-flight task, under `atomcompass-worktrees/<task-id>` |
| **Landing** | one PR per task into the integration branch, reviewed by that task's reviewer agent |
| **ATOM's `main`** | untouched until the milestone the project agrees to upstream |

**Four setup rules, from failures already recorded on this hardware.** None was caused by
worktrees; all were caused by a shared mutable non-git source tree that things silently
resolved against.

1. **No shared mutable source root exists.** Every tree is a worktree or a `git archive`
   snapshot.
2. **Containers mount the worktree parent**, so every task's tree is reachable at a stable
   path. The CPU container previously mounted only one tree, which made `pytest` on
   worktree code fail with a path error rather than a test failure.
3. **Every command sets `PYTHONPATH` to its own worktree and verifies it** —
   `python -c "import atom; print(atom.__file__)"` **before** trusting a result. A prior
   run resolved `atom` to a non-git snapshot of a different branch, 72 files divergent,
   and failed silently wherever both trees had the symbol.
4. **Snapshots use `git archive`, never `rsync`** — so a snapshot names a commit and
   cannot be a mixture of generations. One shared tree held two files from two different
   generations, producing a `TypeError` that named the callee and read as a code bug.

---

## D98. What gates a task

Four things, all required:

1. **ATOM's tests pass unmodified, in two tiers.** Revised 2026-09-20 by P0.2, which
   measured what this gate, `08` D43.1 and ATOM's own `CLAUDE.md` all assert — that the
   187-file suite runs GPU-free and green — and found both halves false. Re-derived the same
   day after review: the first derivation iterated *whole-suite collection* to a fixed point,
   which measures a property of the collection order, not of the files.

   **Derivation — one `pytest <file>` per fresh process,** over all 157 non-plugin files under
   `tests/`, in container `xiaobizh_n18_cpu` on hjbog-srdc-18, against a `git archive` snapshot
   of `b963c9411` with `PYTHONPATH` asserted to resolve `atom` under that root (Python 3.12.3,
   pytest 9.0.3):

   | run alone, the file… | files |
   |---|---|
   | fails collection — `RuntimeError: Get GPU arch from rocminfo failed` | **30** |
   | collects, fails 2 tests `RuntimeError: hipHostMalloc failed: 100`, then **never exits** (killed at 600 s) | 1 — `test_lmcache_offload_disk_integration.py` |
   | collects **zero** tests (module-level skip) | 20 |
   | runs at least one test | 106 |

   One cause for all 30 — but only per file. Collected as a whole suite the same tree gives 37
   errors, 7 under `tests/plugin/` and 30 elsewhere, and those 30 wear four faces: 25
   `rocminfo`, one `KeyError: 'aiter'` (`test_dcp_topk.py`), one `AttributeError` on
   `atom.model_ops.attentions.gdn_attn` (`test_kda_checkpoint_slot_copy.py`), and three with no
   exception recorded. Alone, all five resolve: the two "import defects" are `rocminfo` reached
   through a half-initialised `aiter` left by an earlier file, not defects in ATOM's mocks; and
   `test_dp_metadata`, `test_dp_sync_layout`, `test_forward_mode` are **CPU-green** — 31 passed
   in 0.18 s — and were excluded on that non-evidence. The sentence that stood here, "32 files
   reach the driver, 28 of them at collection time via `rocminfo`", joined two real numbers
   wrongly: the 28 counts `rocminfo` across the *whole* suite, `tests/plugin/` included (25 + 3),
   and was attached to a non-plugin set of 32.

   - **Per task — the CPU gate.** `tests/` minus `tests/plugin/` (30 files) minus the 29
     driver-dependent files below: **128 files handed, 3956 passed, 0 failed, 149 skipped,
     3 xfailed, rc=0, 26.1 s**, measured 2026-09-20 at `b963c9411` in `xiaobizh_n18_cpu`,
     pytest's own exit status captured before any pipe. The identical run at the base commit
     `83daf636d` gives the same four counts — which is what makes a change gate-neutral by
     measurement rather than by assertion. `tests/plugin/` is dropped whole because it needs
     sglang and vllm, in neither image — but that is not the measured reason for all of it:
     7 of its 30 files fail collection in the CPU container and **3 of those 7 are `rocminfo`**,
     not a missing package.
   - **Green is the bar, but green is not "exercised".** Of the 128 files handed, **106 collect
     at least one test and 22 collect none**: 16 declare a device dependency, 3 need PyAV, 2 are
     dead since ATOM #690 split `kv_transfer_engine` into `moriio`
     (`test_kv_connector_scheduler.py`, `test_transfer_engine.py`), and
     `test_prefix_cache_accuracy.py` has **no test function at all** — it is an `argparse` script
     that drives a live server on `localhost:8000`. The 149 skips are 68 distinct reasons: 66
     skipped tests name a device, 83 do not.
   - **Per wave — the GPU superset.** `tests/ --ignore=tests/plugin` in the GPU container,
     judged as a **delta** against the P0.2 baseline at `83daf636d`:
     `5 failed, 4730 passed, 105 skipped, 3 xfailed, 18 warnings in 137.76s`. A bare count
     cannot be checked — a regression that swaps one failure for another passes it — so the five
     are named:

     ```
     tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise
     tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend0-context0-cut+whole]
     tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend1-context1-whole+cut]
     tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend2-context2-resume+fresh]
     tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend4-context4-tiny-then-long]
     ```

     Each is one bf16 ULP (`max|diff| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3`) and
     pre-existing. Stack, so the delta is comparable: `xiaobizh_n18` on hjbog-srdc-18 (MI308X),
     torch `2.10.0+rocm7.2.4.git3d3aa833`, HIP `7.2.53211`, ROCm `7.2.4`, `aiter` at `f4e7c7509`,
     Python 3.12.3, pytest 9.0.3 — read from that same container on 2026-09-20, because the run
     itself did not record them. A review record that claims "green" instead of citing the delta
     has not read the baseline.

   **The exclusion list, and where it lives.** At this commit, here: `scripts/compass/` is empty
   on this branch and on its base, and a gate may not name a file its own tree does not contain.
   28 files fail collection when `tests/` is collected as one batch — the configuration the gate
   runs — iterated until the set stops growing; `test_decode_input_ids.py` appears only in the
   second iteration, which is why the derivation must be iterated and not read off one pass:

   ```
   tests/model_ops/test_balance_router_logits.py   tests/test_eplb_metadata.py
   tests/model_ops/test_shared_expert_dispatch.py  tests/test_eplb_module_{a,b,c,d,e}.py
   tests/test_block_table_marshal.py               tests/test_gdn_state_relocation.py
   tests/test_cudagraph_capture_bounds.py          tests/test_kda_checkpoint_slot_copy.py
   tests/test_dcp_merge_ops.py                     tests/test_kda_layout_id.py
   tests/test_dcp_sparse_filter.py                 tests/test_lm_head_argmax.py
   tests/test_dcp_topk.py                          tests/test_mega_mxfp4_method.py
   tests/test_decode_input_ids.py                  tests/test_merge_attn_states.py
   tests/test_deepseek_v4_wo_a_dequant.py          tests/test_moe_online_quant_batch.py
   tests/test_dspark.py                            tests/test_mori_dispatch_trim_bound.py
   tests/test_dspark_swa_fp8_2buff.py              tests/test_mtp_deferred_status_queue.py
   tests/test_dummy_weight_init.py                 tests/test_mxfp4_moe_has_bias.py
   ```

   — plus `test_lmcache_offload_disk_integration.py`, which collects and then fails at run time:
   **29**. P0.1 (PR #6, based on this branch) mechanises this same list as
   `scripts/compass/cpu_gate_exclude.txt`, with `gate_cpu.sh` to run it and
   `regen_cpu_gate_exclude.sh` to rewrite the generated half; from that commit the file is the
   source of truth, is regenerated rather than edited, and this paragraph is superseded.

   The list is regenerated under the *batch* the gate runs, because that is the configuration
   whose green is claimed — and the per-file census above is a separate question, not a
   replacement for it. Both are needed: `test_postprocess_width.py` and
   `test_v4_checkpoint_slot_copy.py` sit inside the gate, module-skip in the batch
   ("model_runner imports aiter at module load", "the V4 builder's module imports aiter at
   load"), and fail collection with `rocminfo` when run alone; the three files above go the
   other way. That pair is also the whole difference between the two counts of silent files —
   20 per file, 22 in the gate. Recording only the batch is how three CPU-green files stayed
   excluded and how one attributed cause replaced four.

   **The blind spot is bigger than the exclusion list, and it is not random.** 51 of the 157
   non-plugin files run nothing here: 29 excluded and 22 exercising nothing. The excluded 29
   include all five `test_eplb_module_*`, `test_cudagraph_capture_bounds` and
   `test_block_table_marshal`; the silent 22 include `test_pool_index.py`,
   `test_prefill_indices_paged.py`, `test_decode_indices_paged.py`, `test_postprocess_width.py`
   and `test_prefill_prefix_vs_native.py` — the paged-index and prefix-cache areas `03` D13 and
   `01` D6 lean on. **A task touching EPLB, CUDA-graph capture bounds, block tables, the paged
   index builders or the prefix-cache kernels runs the GPU superset as part of its own gate, not
   at wave end.** `test_dp_metadata` and `test_dp_sync_layout` are **not** on that list: they are
   CPU-green, measured above, and booking a GPU for them spends the resource D101 protects.

   **Three files are covered by neither tier**, measured in `xiaobizh_n18` at `83daf636d`:
   `test_prefix_cache_accuracy.py` (`no tests ran`), `test_kv_connector_scheduler.py` and
   `test_transfer_engine.py` (`1 skipped`, ATOM #690). The first and the second are cited by
   `08` D43.1 as coverage Compass keeps. See T73.

   Needing to edit an ATOM test still means the change altered ATOM's behaviour and must be
   justified on its own terms, not absorbed.
2. **New CPU-only tests** for what the task added, in `tests/compass/`, in ATOM's style.
3. **One named result**, stated in the brief and not chosen afterwards — what this task
   now makes possible that was not possible before.
4. **Review by the task's reviewer agent**, against its brief and the cited decisions.

**Baseline first — done.** P0.2 measured it at `83daf636d` on node 18: the CPU tier as above;
GPU superset 4730 passed / 5 failed; and in `xiaobizh_n18_cpu`, `ruff check .` **1003 errors,
640 fixable** and `black --check .` clean over **660 files** (ruff 0.16.7, black 26.5.1 — the
GPU container has neither, so lint and the GPU superset cannot be run in one place). Both
reproduce at `83daf636d` on 2026-09-20. So the lint gate is "**no new** ruff error, and black
stays clean" — never "ruff is clean", which it has never been.

Every number in this decision is stated here rather than cited. By D96 the task record lives
outside the tree, and the raw pytest output lives only in the containers that produced it; a
citation to either is not resolvable from a checkout, which is the same defect as naming a
script that is not in the tree. The commits carrying these measurements are `9c8df1328`,
`b963c9411` and this one, on PR #4.

---

## D99. Effort is estimated in lines of code

Not in time. Agents do not have hours; they have output volume, and LOC is estimable from
the design — `06` D34 already sizes the harness adapter at 450–650 lines.

**Wall-clock appears only for machine time with a measured basis**: a TP2 engine run is
~1 h because engine runs take that long, and the long sweep is six hours because it was
measured at six hours. Those are hardware constraints, not effort.

Estimates are ranges and are expected to be wrong. A task that overruns its estimate by
more than ~2x is a **halt-and-discuss** event, not a reason to keep going — the usual
cause is that the task was mis-cut.

---

## D100. The module layout, which is also the task boundary

Tasks are cut so each touches one module plus its tests.

```
atom/compass/
  clock/      CA core, grant rule, lookahead matrix, LP registry, both transports
  detect/     straggler check, watchdog, CI clock-source lint
  runner/     CompassModelRunner - the seam, RPC surface, forward semantics
  backends/   CostBackend interface, StepCost, provenance; tier 0 / a / b
  memory/     device-reading substitution, per-term model
  ir/         Cost IR (Seq/Repeat/Par), nested-Repeat detector, opaque leaves
  capture/    FakeTensorMode capture, guard domains
  spec/       machine-spec schema, merge / validate / explain
  artifacts/  store, keys, digests, invalidation matrix
  cli/        compass plan / discover / trace / measure / validate
  metrics/    per-step sampling, backfill
  kv/         simulated KV connector
tests/compass/          mirrors the above
<out-of-tree package>   the aiperf adapter (never inside ATOM)
```

Edits to ATOM outside `atom/compass/` are deliberately few and are enumerated per task —
clock-read substitutions, blocked/running annotations, the config flags of `13` D80, and
the two ATOM-owned additions (`--preprocess-pool-width`, `--spec-decode-acceptance-rates`).

---

## Phase 0 — de-risking, everything parallel

Nothing here depends on anything else. All seven can be claimed at once, subject to the
five-slot cap.

| ID | Task | Module | Effort | Hardware |
|---|---|---|---|---|
| **P0.1** | Environment: worktree layout, container mounts covering the worktree parent, `PYTHONPATH` wrapper with verification, GPU pre-flight script | tooling | ~150 LOC | — |
| **P0.2** | Baselines: record ATOM suite and ruff pass/fail state | — | ~0 | — |
| **P0.3** | **T10** — can `AgenticReplayStrategy` be subclassed rather than vendored? | spike | ~50 LOC | laptop |
| **P0.4** | **T5** — trace the 27B under `FakeTensorMode` at TP1 **and TP2**; diff the captured structures | spike → `capture/` | ~200 LOC | clean node |
| **P0.5** | **T64** — does ATOM microbatch PP? | reading + 1 run | ~0 | clean node |
| **P0.6** | **T65** — EP group membership per supported configuration | reading | ~0 | — |
| **P0.7** | **T52** — root-cause the dispatch-mode 8-rank hang. **Only if P0.4 hits it.** | spike | — | quiet node |

**Why these seven.** Each can invalidate work that would otherwise be built on top of it.
P0.3 swings an estimate by ~2,000 lines for one hour's work. P0.4 decides whether tier b
has an IR at all, and **it is the task most likely to reshape Waves 2–4** — if tracing
does not work at TP>1, `capture/` and `ir/` change shape. The plan marks that edge rather
than pretending the DAG is stable across it.

**P0.7 is conditional.** `15` and `04` both argue a fake-tensor trace should be GPU-free
and collective-free, so the known hang should not be reachable from P0.4. If P0.4 confirms
that, T52 drops to ordinary priority and gates nothing on the critical path.

**Each of P0.3–P0.7 ends in an escalation, not a decision.** The result plus its options
and their costs go to the project owner; the scope call is theirs (D103).

---

## Wave 1 — foundations

Ten independent tasks. No task here depends on another in this wave; each depends only on
Phase 0's environment. All CPU-only.

| ID | Task | Module | Effort | Implements / consumes |
|---|---|---|---|---|
| **W1.1** | CA core: grant rule `min_j(now[j] + L[j→i])`, lookahead matrix addressed by LP identity, LP registry, **both transports** (in-process and socket) from one implementation | `clock/` | 600–900 | implements the clock client API |
| **W1.2** | Synthetic-LP harness modelling ATOM's **verified** topology — the ZMQ/shm/Gloo links of `01` D1 and the four wait categories of D4 — plus adversarial scenarios: wrong lookahead, missing annotation, induced straggler | `tests/compass/clock/` | 400–600 | consumes W1.1 |
| **W1.3** | Causality detectors: receive-side straggler check (fails the run), annotation watchdog (warns), CI clock-source lint | `detect/` | 250–400 | consumes W1.1 |
| **W1.4** | `CompassModelRunner` skeleton: the `--runner-qualname` subclass, the full RPC surface, and the three forward semantics of `02` D10 (deferred output, the meaningful-step unit, `produces_output`) | `runner/` | 400–600 | implements the runner seam |
| **W1.5** | `CostBackend` interface, `StepCost` with its breakdown, the provenance vocabulary and the resolver ladder | `backends/` | 200–300 | implements the backend API |
| **W1.6** | M1 fake model: HF-config geometry plus the shape-analytic cost stub of `02` D12, including the quadratic query term | `backends/` | 300–450 | consumes W1.5 |
| **W1.7** | Machine-spec schema, `merge` / `validate` / `explain`. No probes yet | `spec/` | 400–600 | implements the spec artifact |
| **W1.8** | Wire-contract fields on ATOM's real endpoint: one `compass` object each direction (`06` D28) | ATOM entrypoints | 100–200 | implements the wire contract |
| **W1.9** | aiperf adapter package, out of tree: clock client, pacing redirect, latency anchors from response fields | out-of-tree | 450–650 | consumes W1.8 |
| **W1.10** | Cost IR: `Seq` / `Repeat` / `Par`, the **nested** Repeat detector with its index binding, and the provably-free grouping rule (`04` D19) | `ir/` | 500–700 | implements the IR |

**W1.2 is the one that earns its keep.** It is where the distributed CA — the riskiest
component, and the one whose failures are silent — gets exercised without ATOM and without
a GPU, so it runs in CI on every change thereafter.

**W1.9's estimate is conditional on P0.3.** If `AgenticReplayStrategy` cannot be
subclassed, this becomes ~2,000 lines of vendored code that must track upstream, and that
is an escalation rather than a bigger task.

---

## Wave 2 — integration

Depends on Wave 1. The first three are the vertical slice.

| ID | Task | Module | Effort | Depends on |
|---|---|---|---|---|
| **W2.1** | **The vertical slice**: one simulated request end to end at TP1 with the co-hosted CA — runner + clock + fake backend + ATOM's **real** scheduler, block manager and prefix cache | `runner/`, wiring | 300–500 | W1.1, W1.4, W1.6 |
| **W2.2** | Memory: substitute the five device readings, reuse ATOM's budget arithmetic and `plan_pools`; per-term validation, never a sum | `memory/` | 400–600 | W1.7 |
| **W2.3** | Simulated KV connector registered into ATOM's existing factory; `latency + bytes/bandwidth` from the spec | `kv/` | 250–400 | W1.7 |
| **W2.4** | Artifact store: keys, digests, source-root fingerprints, the invalidation matrix | `artifacts/` | 500–700 | W1.7 |
| **W2.5** | `FakeTensorMode` capture with the three mandatory disciplines and post-trace assertions; guard-domain evaluation | `capture/` | 500–800 | W1.10, **P0.4** |
| **W2.6** | Metrics under virtual time: per-step sampling, the class taxonomy, OpenMetrics backfill | `metrics/` | 300–500 | W2.1 |
| **W2.7** | Determinism test: byte-diff of two step tables from one configuration, in CI (`01` D3.4) | `tests/compass/` | ~100 | W2.1 |
| **W2.8** | Model construction under `FakeTensorMode` with `--load_dummy empty` (`02` D10.1), including the buffer audit of T68 | `runner/` | 200–350 | W2.5 |

---

## Wave 3 — M1 completion

| ID | Task | Effort | Notes |
|---|---|---|---|
| **W3.1** | LP structure for DP: both collectives run for real; `max`-over-ranks step duration; dummy-batch pricing for idle ranks (`15` D90) | 300–500 | |
| **W3.2** | LP structure for PP: one LP per stage; transfer as a size from the spec; layer split via `get_pp_indices` | 400–600 | shaped by **P0.5** |
| **W3.3** | EP: `exclusive` occupancy honoured in the IR; group membership per P0.6 | 200–350 | shaped by **P0.6** |
| **W3.4** | Standalone CA exercised against the synthetic-LP harness at multi-container topology | 200–300 | consumes W1.2 |
| **W3.5** | PD aggregation and disaggregation, two containers on one node | 400–600 | |
| **W3.6** | cc-traces harness driving a simulated run end to end | 300–500 | consumes W1.9 |
| **W3.7** | CA observability: timeline log, deadlock dump, run summary (`01` D3.5) | 300–450 | |

**M1's exit criterion**, per `15` D94, needs no cost model: **does a fake-model run at
TP2 / DP2 / PP2 / EP2 reach the same scheduling decisions as the real engine at the same
configuration?** ATOM's own `test_dp_load_balance.py`, `test_dp_metadata.py`,
`test_dp_sync_layout.py` and `test_forward_mode.py` already cover the pieces, CPU-only.

---

## Wave 4 onward — deliberately coarse

M2 and beyond are sketched, not planned. Their shape depends on Phase 0's answers and on
T21, and detail written now against unknown answers would be fiction. Detail is added wave
by wave as the answers arrive.

| Wave | Covers | Gated on |
|---|---|---|
| **W4** | `compass plan` and the CLI; calibration Phase 0 discovery and Phase 1a trace; the standalone `ModelRunner` bench | W2.5, W2.4 |
| **W5** | Calibration Phases 1b / 1c / 2 on GPU; fitting and law selection; **the first paired comparison** (M2) | GPU queue, **T21** |
| **W6** | Analytic laws (tier 0), authored **during** the empirical campaign, not after (`10` D68) | W5 |
| **W7** | M3 (TP2/TP4), **M3.5** (spec decode / MTP), M4 (PD disagg two nodes) | T21's answer |
| **W8** | M5–M7 (Kimi-K3, TP8, PD disagg, DP/PP/EP) | W7 |

**One thing to schedule early despite being a Wave 5 concern:** measure a **saturated**
cell as soon as a cost model exists. The prior design ran **0.30x** under saturation —
slower than the system it simulates. If the ≥5x target fails structurally, that is worth
knowing while the fake model is still the only thing in play.

---

## D101. The GPU booking queue

GPU is the scarce resource; almost everything else is CPU-only by design principle 2.
Only calibration Phases 1b/1c/2, `--measure` runs, the real side of pairings, and the
T52 root-cause need one.

**One queue.** A GPU task declares, before it is claimable: what it measures, which width,
how long it needs, and which artifact it writes. A quiet window is never spent deciding
what to run in it.

### The pre-flight gate — three checks, not one

```
  1.  timeout 25 rocminfo            -> hangs?  node is WEDGED. Do not use.
  2.  rocm-smi --showuse             -> compute busy?
  3.  rocm-smi --showmemuse          -> VRAM held?     <- the one that gets missed
```

**Check 3 is not optional.** A node can show 0% utilisation and ~99% VRAM — a loaded,
idle model. A pre-flight testing only utilisation calls that node free, and `non_torch` is
a **device-wide** reading, so a neighbour's allocation is indistinguishable from ours.

**Check before *and* after.** Three of the last five prior pilot attempts were lost or
degraded by other tenants; a check that only runs first cannot see a tenant that arrived
mid-run (`08` D49.5).

### Node status at time of writing — a snapshot, not a fact

| node | compute | VRAM | D-state procs | usable |
|---|---|---|---|---|
| 18 | GPU0 100%, rest idle | GPU0 90% | 0 | 7 GPUs |
| 19 | idle | **0%** | 0 | **yes** |
| 20 | idle | **0%** | 0 | **yes** |
| 21 | idle | 99% | 0 | no — VRAM held |
| 22 | idle | 98% | 0 | no — VRAM held |
| 39 | idle | 71% | **2183** | **no — wedged** |

Node 39's 2,183 D-state processes are the ROCm wedge signature: `rocm-smi` answers
normally, which is precisely why the node looks healthy. The durable artifact here is the
pre-flight script, not this table.

---

## D102. Escalation points

Five checks can each reshape the plan. Each ends in a decision that belongs to the project
owner, not to the agent that ran it.

| Trigger | What the escalation carries |
|---|---|
| **T10** fails | adapter cost rises to ~2,000 vendored lines — vendor, fork, or restrict the harness |
| **T5** fails | tier b has no IR at TP>1 — options and their effect on M2 onward |
| **T21** fails | calibration does not transfer across width — per-width campaign, a reduced acceptance set, or generalisation reported as within-width only |
| **T25** fails | the noise floor swamps 10% at high client count — those cells are **ungradeable**, and the acceptance set needs re-scoping |
| **T64** answers "yes" | ATOM microbatches PP — grant count rises by the microbatch factor and W3.2 changes shape |

**Each escalation is prepared, not improvised.** When a check completes, its result
arrives with the options and their costs already worked out, so the decision is one round
trip rather than a fresh analysis under time pressure.

The halt rule (D95) is the general case: any surprise stops and is discussed.

---

## What this plan does not contain

Stated so it is not mistaken for an omission.

- **Dates.** Effort is LOC; wall-clock appears only where it is machine time with a
  measured basis (D99).
- **Detailed Wave 4+.** Deliberate — see above.
- **Agent prompts.** Task briefs are written to be close to a prompt and generated from at
  launch, because embedded prompts go stale as tasks move.
- **A test plan separate from `08`.** Validation is `08`; this document schedules it.
- **Upstreaming to ATOM's `main`.** Out of scope until the project agrees a milestone is
  ready; the integration branch is the destination until then.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D95 | Tasks are a **pool**, not a track assignment; 5 dev + 5 reviewer agents cap concurrency at 5 in flight. Conflicts are tolerated and are a **decomposition signal**. Developer and reviewer are separate agents with opposed objectives. **Halt and discuss on any surprise.** | 2026-09-20 |
| D96 | Context is durable **in the task record** — brief, dev record, review record, handoff — with each brief linking to its predecessors'. A brief that cannot name its file set is not claimable. | 2026-09-20 |
| D97 | `feature/atomcompass_new` is the integration branch; one worktree and one PR per task. Four setup rules from recorded failures: no shared mutable source root, containers mount the worktree parent, `PYTHONPATH` verified before trusting a result, `git archive` never `rsync`. | 2026-09-20 |
| D98 | Four gates per task: ATOM's tests pass **unmodified** — the CPU tier per task, green at the exclusion list stated in the body, and the GPU superset per wave as a **delta** against a baseline whose failing node-ids are named — plus new CPU-only tests, one named result stated in advance, and review by a separate agent. Counts live in the body, with their tree and container, because they move with both. Amended 2026-09-20 after P0.2 disproved "187 files, no GPU needed", and again the same day after the exclusion set was re-derived per file. | 2026-09-20 |
| D99 | Effort in **lines of code**. Wall-clock only for machine time with a measured basis. A 2x overrun is a halt-and-discuss event. | 2026-09-20 |
| D100 | Twelve modules under `atom/compass/`; tasks are cut so each touches one plus its tests. ATOM edits outside that tree are enumerated per task. | 2026-09-20 |
| D101 | One GPU queue; tasks declare their measurement before becoming claimable. Pre-flight is **three** checks — wedge, compute, **VRAM** — run before *and* after. | 2026-09-20 |
| D102 | Five named escalation points, each prepared in advance so the decision is one round trip. | 2026-09-20 |

---

## TODO register

This topic's items only. The consolidated register is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T71 | Add Wave 4+ detail as Phase 0 and T21 answers arrive | by design — see the Wave 4 note |
| T72 | Decide whether reviewer agents use ATOM's existing `review-pr` skill or a Compass-specific checklist | needs one review cycle to tell |
| T73 | Raise with ATOM's owners: `tests/test_prefix_cache_accuracy.py` has no test function — it is an `argparse` script driving a live server — and `test_kv_connector_scheduler.py` / `test_transfer_engine.py` have been dead since #690. Measured: all three run nothing in **either** tier | not a Compass change; needs the disaggregation and prefix-cache owners |
