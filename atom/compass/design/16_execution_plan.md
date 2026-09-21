# ATOM Compass — Design Topic 16: The Execution Plan

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** all of `01`–`15`. This document turns 109 decisions and 76 open items into
work that can be allocated.

**Scope.** How the work is organised, allocated and gated; what happens first; and what
each stage must produce. It is **detailed for Phase 0 through Wave 3 and deliberately
coarse for Wave 4 onward** — the later waves depend on answers Phase 0 has not produced
yet, and a detailed plan written against unknown answers is a fiction. Detail is added as
the answers arrive.

---

## The module layout, which is also the task boundary

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
P0.3 swung an estimate by ~2,000 lines for one hour's work; it has since run and the swing
did not happen. P0.4 decides whether tier b has an IR at all, and **it is the task most
likely to reshape Waves 2–4** — if tracing does not work at TP>1, `capture/` and `ir/`
change shape. The plan marks that edge rather than pretending the DAG is stable across it.

**P0.7 is conditional.** `15` and `04` both argue a fake-tensor trace should be GPU-free
and collective-free, so the known hang should not be reachable from P0.4. If P0.4 confirms
that, T52 drops to ordinary priority and gates nothing on the critical path.

**Each of P0.3–P0.7 ends in an escalation, not a decision.** The result plus its options
and their costs go to the project owner; the scope call is theirs (see Escalation points below).

---

## The measured test and lint baselines

P0.2's result, and the reason the test gate in `AI_DEV_RULES.md` cannot be run as written.
That gate, `08` D43.1 and ATOM's own `CLAUDE.md` all assert the same thing — that the
187-file suite under `tests/` runs GPU-free and green — and P0.2 measured both halves
false. What is GPU-free is a **tier**, run per task; the rest is a GPU superset, run per
wave as a delta. Re-derived on 2026-09-20 after review: the first derivation iterated
*whole-suite collection* to a fixed point, which measures a property of the collection
order, not of the files.

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
in 0.18 s — and were excluded on that non-evidence. The sentence that `08` and this document
both carried, "32 files reach the driver, 28 of them at collection time via `rocminfo`",
joined two real numbers wrongly: the 28 counts `rocminfo` across the *whole* suite,
`tests/plugin/` included (25 + 3), and was attached to a non-plugin set of 32.

- **Per task — the CPU tier.** `tests/` minus `tests/plugin/` (30 files) minus the 29
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
28 files fail collection when `tests/` is collected as one batch — the configuration the CPU
tier runs — iterated until the set stops growing; `test_decode_input_ids.py` appears only in
the second iteration, which is why the derivation must be iterated and not read off one pass:

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

The list is regenerated under the *batch* the tier runs, because that is the configuration
whose green is claimed — and the per-file census above is a separate question, not a
replacement for it. Both are needed: `test_postprocess_width.py` and
`test_v4_checkpoint_slot_copy.py` sit inside the tier, module-skip in the batch
("model_runner imports aiter at module load", "the V4 builder's module imports aiter at
load"), and fail collection with `rocminfo` when run alone; the three files above go the
other way. That pair is also the whole difference between the two counts of silent files —
20 per file, 22 in the tier. Recording only the batch is how three CPU-green files stayed
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
CPU-green, measured above, and booking a GPU for them spends the resource the booking queue
below exists to ration.

**Three files are covered by neither tier**, measured in `xiaobizh_n18` at `83daf636d`:
`test_prefix_cache_accuracy.py` (`no tests ran`), `test_kv_connector_scheduler.py` and
`test_transfer_engine.py` (`1 skipped`, ATOM #690). The first and the second are cited by
`08` D43.1 as coverage Compass keeps. See T80 in [`12_open_items.md`](12_open_items.md).

**Lint, same day.** In `xiaobizh_n18_cpu`, `ruff check .` gives **1003 errors, 640 fixable**
and `black --check .` is clean over **660 files** (ruff 0.16.7, black 26.5.1 — the GPU
container has neither, so lint and the GPU superset cannot be run in one place). Both
reproduce at `83daf636d` on 2026-09-20. So the lint bar is "**no new** ruff error, and black
stays clean" — never "ruff is clean", which it has never been.

Every number above is stated here rather than cited. The task record lives outside the tree
(`AI_DEV_RULES.md`), and the raw pytest output lives only in the containers that produced it;
a citation to either is not resolvable from a checkout, which is the same defect as naming a
script that is not in the tree. The commits carrying these measurements are `9c8df1328`,
`b963c9411` and the two that follow them, on PR #4.

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
| **W1.9** | aiperf adapter package, out of tree: clock client, pacing redirect, latency anchors from response fields | out-of-tree | 450–650, reopened | consumes W1.8 |
| **W1.10** | Cost IR: `Seq` / `Repeat` / `Par`, the **nested** Repeat detector with its index binding, and the provably-free grouping rule (`04` D19) | `ir/` | 500–700 | implements the IR |

**W1.2 is the one that earns its keep.** It is where the distributed CA — the riskiest
component, and the one whose failures are silent — gets exercised without ATOM and without
a GPU, so it runs in CI on every change thereafter.

**W1.9's estimate is no longer conditional on vendoring. P0.3 resolved T10 on
2026-09-20:** subclassing works and nothing is vendored. The spike also found that a
strategy subclass alone is **not sufficient** — further pacing sites live in
`BranchOrchestrator` and `ReplayBarrierCoordinator`, which a subclass never sees.

**Owner decision, 2026-09-20 — rebind *and* assert (option C).** W1.9 rebinds the shared
`LoopScheduler` module global from a Compass bootstrap, **and** registers a strategy
subclass that raises unless the scheduler it receives is already clock-paced (`06` D34).
The subclass does no wrapping; it exists because the rebind's failure mode is silent, and
a silently wall-clocked run is the worst result this design can produce.

**Review of P0.3 then showed the tripwire is load-bearing, not belt-and-braces**, and that
the 450–650 range needs re-costing before W1.9 starts:

| Correction | Effect on W1.9 |
|---|---|
| A bootstrap that precedes the first `PhaseRunner` **does** exist. `discover_plugins()` resolves each entry point with `importlib.util.find_spec`, which on a *dotted* value imports the parent package, so `compass_harness.plugin:plugins.yaml` executes `compass_harness/__init__.py` during discovery. The rebind is deferred from there by a stdlib import hook, because that bootstrap runs inside `aiperf.plugin.plugins`' own module body and cannot import `aiperf` — executed, five combinations, 7/7 in the working shape (`06` D34, T73) | **T73 is a packaging choice plus ~20 lines of import hook, inside the existing "plugin manifest, bootstrap, config glue" row — not a precondition of the seam and not W1.9's first deliverable.** The bootstrap shape that *would* be too late is a side effect of the strategy module: phase 0 on the real clock, phase 1 rebound, smoke test passes. That is why the tripwire ships regardless |
| `runner.py:191` calls `LoopScheduler()` with no arguments, so the rebound class must be no-arg constructible — `ClockPacedLoopScheduler(LoopScheduler)`, a subclass, not the spike's `ClockPacedScheduler(inner)` wrapper | The spike validated option A's shape. "About five lines" was costed against the wrong object, so that component row is **open** |
| One scheduler per `PhaseRunner`, and `seamless=True` keeps two live | T76 — reconcile, or assert `seamless=False` |
| Two `loop.call_later` idle-cap timers the rebind cannot reach, one of which upstream deliberately keeps outside the scheduler | T75 — override the two `_arm_*` methods, or declare the feature unsupported and assert both caps are `None` |

**W1.9's acceptance is a list, not a count:** every one of the nine enumerated pacing calls
in `06` D34 is observed to advance on the virtual clock — plus a stated disposition for the
two timers in T75. Those line numbers are pinned to agentx-harness
`56a0cf70f4c0359454ee4bd15a17770b541a3e3e`; re-verify them against that revision, or
re-derive them from the whole-tree grep in `06` D34, before building the test to them.
"None of the seven advances on the real clock" was unimplementable as written; the seven
added four *methods* to three *call sites*.

It does **not** cover the 32 `asyncio.wait_for` timeout sites, which bypass `LoopScheduler`
entirely (T74).

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

## The GPU booking queue

GPU is the scarce resource; almost everything else is CPU-only by design principle 2.
Only calibration Phases 1b/1c/2, `--measure` runs, the real side of pairings, and the
T52 root-cause need one.

**One queue.** A GPU task declares in its issue body, before it is claimable: what it
measures, which width, how long it needs, and which artifact it writes. A quiet window is
never spent deciding what to run in it.

**Wall-clock is quoted only where it was actually measured.** A TP2 engine run is priced
at ~1 h (`12` T21) because that is what engine runs take, and the long calibration sweep
is priced at six hours because it was measured at six hours — not estimated from
throughput. Effort elsewhere is sized in lines of code, not wall-clock, precisely because
most tasks have no such measurement to quote (`AI_DEV_RULES.md`).

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

## Escalation points

Five checks can each reshape the plan; T10 has been run and did not fire. Each ends in a
decision that belongs to the project owner, not to the agent that ran it.

| Trigger | What the escalation carries |
|---|---|
| ~~**T10** fails~~ | ~~adapter cost rises to ~2,000 vendored lines — vendor, fork, or restrict the harness~~ — **did not fire.** P0.3, 2026-09-20: the subclass works and nothing is vendored. The escalation this row prepared for never arose |
| **T5** fails | tier b has no IR at TP>1 — options and their effect on M2 onward |
| **T21** fails | calibration does not transfer across width — per-width campaign, a reduced acceptance set, or generalisation reported as within-width only |
| **T25** fails | the noise floor swamps 10% at high client count — those cells are **ungradeable**, and the acceptance set needs re-scoping |
| **T64** answers "yes" | ATOM microbatches PP — grant count rises by the microbatch factor and W3.2 changes shape |

**Each escalation is prepared, not improvised.** When a check completes, its result
arrives with the options and their costs already worked out, so the decision is one round
trip rather than a fresh analysis under time pressure.

The halt rule (`AI_DEV_RULES.md`) is the general case: any surprise stops and is discussed.

Reaching any of these five, like the loop's halt above, is an escalation — so it applies
`need human` too (`AI_DEV_RULES.md`), for the same reason: the stop should be visible on
GitHub, not only inside an agent's report.

---

## What this plan does not contain

Stated so it is not mistaken for an omission.

- **Dates.** Effort is LOC; wall-clock appears only where it is machine time with a
  measured basis (`AI_DEV_RULES.md`).
- **Detailed Wave 4+.** Deliberate — see above.
- **Agent prompts.** Task briefs are written to be close to a prompt and generated from at
  launch, because embedded prompts go stale as tasks move.
- **A test plan separate from `08`.** Validation is `08`; this document schedules it.
- **Upstreaming to ATOM's `main`.** Out of scope until the project agrees a milestone is
  ready; the integration branch is the destination until then.

---

## TODO register

This topic's items only. The consolidated register is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T71 | Add Wave 4+ detail as Phase 0 and T21 answers arrive | by design — see the Wave 4 note |
| T72 | Decide whether reviewer agents use ATOM's existing `review-pr` skill or a Compass-specific checklist | needs one review cycle to tell |
| T80 | Raise with ATOM's owners: `tests/test_prefix_cache_accuracy.py` has no test function — it is an `argparse` script driving a live server — and `test_kv_connector_scheduler.py` / `test_transfer_engine.py` have been dead since #690. Measured: all three run nothing in **either** tier | not a Compass change; needs the disaggregation and prefix-cache owners |
