# ATOM Compass — Design Topic 16: The Execution Plan

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** all of `01`–`15`. This document turns those topics' decisions, and the
register of open items in [`12_open_items.md`](12_open_items.md), into work that can be
allocated. Neither is counted here. The register is owned by `12`, restated once on
`README.md`'s front page, and it moves as tasks land; a third copy in a document that does
not own it has been stale before, and "open items" and "registered items" are two different
numbers that a single figure here cannot distinguish.

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
P0.3 swings an estimate by ~2,000 lines for one hour's work. P0.4 decides whether tier b
has an IR at all, and **it is the task most likely to reshape Waves 2–4** — if tracing
does not work at TP>1, `capture/` and `ir/` change shape. The plan marks that edge rather
than pretending the DAG is stable across it.

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

- **Per task — the CPU tier, re-measured by P0.1 and superseding the counts above.**
  `tests/` minus `tests/plugin/` (30 files) minus the 29 driver-dependent files in
  `scripts/compass/cpu_gate_exclude.txt`, driven by `scripts/compass/gate_cpu.sh`:
  **130 of this tree's 189 test files, 4030 passed, 0 failed, 149 skipped, 3 xfailed,
  rc=0** — identical in all ten runs taken on 2026-09-21 in `xiaobizh_n18_cpu` on
  hjbog-srdc-18, where the clock read **25.4-31.7 s of pytest inside 31.1-37.8 s of
  wall (`time` real)**, which is a measured spread rather than a bound: it tracks
  what else is on the node. Run against a `git archive` snapshot of the tree, with
  `PYTHONPATH` asserted to resolve `atom` under that root and pytest's own exit status
  captured before any pipe. The 4030 is **3956 ATOM tests + 74 `tests/compass/` tests**,
  stated as its parts because a single total cannot show which half moved (principle 7). The
  file count moves 128 → 130 and the test count 3956 → 4030 because this tree adds
  `tests/compass/test_cpu_gate_exclude.py` and `tests/compass/test_gate_gpu_surplus.py`; the
  P0.2 readings above are the same suite without them. The other totals in circulation are
  the same suite under a different exclusion list or a different `tests/compass`, not
  discrepancies: **3925** was 32 exclusions with
  `tests/compass` at 35 tests, **3956** is the ATOM-only half at 29 exclusions, and **4005**
  was this gate at `3afcb4880` with `tests/compass` at 49, and **4022** was it at 66. The
  49 → 66 step is mechanical: `tests/compass/test_cpu_gate_exclude.py` parametrises one case
  per entry of `gpu_gate_triggers.txt`, and correcting that file's derivation took it from 13
  entries to 30. The 66 → 74 step is `tests/compass/test_gate_gpu_surplus.py`, added here.
- **Green is the bar, but green is not "exercised".** Of the 130 files handed, **22 collect
  no test at all**: 16 declare a device dependency, 3 need PyAV, 2 are dead since ATOM #690
  split `kv_transfer_engine` into `moriio` (`test_kv_connector_scheduler.py`,
  `test_transfer_engine.py`), and `test_prefix_cache_accuracy.py` has **no test function at
  all** — it is an `argparse` script that drives a live server on `localhost:8000`. The 149
  skips are 68 distinct reasons: 66 skipped tests name a device, 83 do not. That
  decomposition was measured per file by P0.2 against an earlier tree, where the same 22 sat
  inside a 128-file gate; the gate's own counts above supersede that file count, and the
  split of the 22 stands because nothing since has changed which files hold runnable tests.
- **Per wave — the GPU superset, re-measured by P0.1.** `tests/ --ignore=tests/plugin` in the
  GPU container, driven by `scripts/compass/gate_gpu.sh`, judged as a **delta**, never as
  "green". The baseline is **4779 passed / 5 failed**, 0 errors, 105 skipped, 3 xfailed,
  72.6 s, measured 2026-09-20 at `fe9ea043c` on node 18 in `xiaobizh_n18` with
  `HIP_VISIBLE_DEVICES=1` — torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip`
  **7.2.53211**, ROCm **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509** — over two runs with
  byte-identical failing sets. It supersedes P0.2's **4730 / 5** at `83daf636d`, whose torch,
  AITER and ROCm versions were unrecorded and which therefore could not be reproduced as
  recorded; **T77**, which tracked that gap, is closed by this measurement. The five are
  pre-existing and unrelated to Compass, and they are **four ULP comparisons plus one bitwise
  check**, not "five bf16 ULP failures": four `allclose` cases in
  `tests/test_fused_compress_ragged.py` off by one bf16 ULP
  (`max|diff| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3`), plus
  `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise`, a
  `torch.equal` with **no tolerance at all** — a tolerance bump would not move it. All five
  node-ids are on file verbatim in `scripts/compass/gpu_gate_known_failures.txt` and compared
  by name, so "5 failed" is checked against "the *same* 5 failed". A review record that
  claims "green" instead of citing the delta has not read the baseline. The pass count moves
  by construction — the superset includes `tests/compass/`, so every test this phase adds
  raises it above 4779, and `gate_gpu.sh` therefore expects
  `4779 + (this tree's tests/compass count - 49)` and refuses a bare "no worse than".

**The exclusion list lives in the tree.** P0.2 derived it here, in prose, because
`scripts/compass/` did not yet exist on this branch and a gate may not name a file its own
tree does not contain. P0.1 mechanises it: `scripts/compass/cpu_gate_exclude.txt`, with
`gate_cpu.sh` to run it and `regen_cpu_gate_exclude.sh` to rewrite the generated half. From
this commit that file is the source of truth and is regenerated rather than edited, and the
prose derivation above is superseded. Its 29 entries are **28 GENERATED + 1 MANUAL**: the
generator iterates batch collection to a fixed point and produces 28, and the single manual
entry — `test_lmcache_offload_disk_integration.py`, which collects cleanly and then fails on
`hipHostMalloc failed: 100` — carries the observed failure text above it, which is the only
evidence such an entry can have. Both reviewers found that the earlier 32-entry list could
not be reproduced by its own generator: four entries had been hand-added to a file whose
header read "never hand-edit", and three of those four — `test_dp_metadata.py`,
`test_dp_sync_layout.py` and `test_forward_mode.py` — collect *and pass* in a driverless
container. Prose saying the list is 32, or that those three need the driver, is stale.

The list is regenerated under the *batch* the tier runs, because that is the configuration
whose green is claimed — and the per-file census above is a separate question, not a
replacement for it. Both are needed: `test_postprocess_width.py` and
`test_v4_checkpoint_slot_copy.py` sit inside the tier, module-skip in the batch
("model_runner imports aiter at module load", "the V4 builder's module imports aiter at
load"), and fail collection with `rocminfo` when run alone; the three CPU-green files above
go the other way. That pair is also the whole difference between the two counts of silent
files — 20 per file, 22 in the tier. Recording only the batch is how three CPU-green files
stayed excluded and how one attributed cause replaced four.

**The CPU tier's blind spot is a file, not a sentence.** P0.2 stated it as prose naming four
areas — EPLB, DP metadata, cudagraph bounds, block tables — which nothing read, and which
named two files (`test_dp_metadata`, `test_dp_sync_layout`) the CPU tier does in fact cover.
It is now `scripts/compass/gpu_gate_triggers.txt`, **30 paths** in the tree committed here,
generated by `regen_gpu_gate_triggers.sh` and never hand-edited — the counts in its own
header included (189 test files, 130 CPU tier, 108 of them collecting, 47 candidates, 99
covered). The rule it applies: an `atom` module named by an excluded test is a blind spot
**unless a CPU-tier test that actually runs names it too**. A module both tiers import is
covered when the CPU gate runs, so triggering on it would make the gate cry wolf. A trailing
`/` matches a subtree. `gate_cpu.sh` matches the diff against that file and **exits 98 unless
`COMPASS_GPU_GATE_DONE` names this tree's own HEAD**; where it can compute neither a diff nor
a supplied file list — a `git archive` snapshot with no `.git` — it refuses rather than
reporting "not required", per principle 6.

The rule has three parts. One was forced by a counter-example measured in this tree, one is
what keeps the headline example, and one currently changes no path:

- *Imports are read at any indentation on the excluded side.* 190 of this tree's
  `import atom.*` lines are indented — inside a function, a `try`, or a
  `skipif(not torch.cuda.is_available())` guard. An earlier draft anchored the match at
  column 0 on both sides and silently dropped `atom/model_ops/topK.py`, whose only
  excluded-side reference is `test_moe_dp_token_capacity.py:39`, indented under exactly such
  a guard.
- *Coverage is credited only for a module-level import.* An indented import in a CPU-tier
  file is not proof that the CPU tier executes it, and crediting it would let a never-taken
  branch suppress a trigger. **This is the part that keeps
  `atom/model_engine/model_runner.py`** — the module Compass's runner seam replaces.
  `tests/test_mla_index_cache.py` imports `ModelRunner` at
  `tests/test_mla_index_cache.py:99`, indented four spaces inside a test function, so that
  import is never credited whatever the file collects. Measured at `236abfd9a` in
  `xiaobizh_n18_cpu`: crediting coverage at any indentation drops the set 30 → 29, losing
  `topK.py`; doing that *and* crediting non-collecting files drops it 30 → 27, losing
  `model_runner.py`, `aiter_mla.py` and `topK.py`.
- *Coverage is credited only from a CPU-tier file that collects at least one test.* 22 of the
  130 CPU-tier files collect none. On this tree that probe **removes no path**: 30 triggers
  with it, 30 without, difference empty — measured at `236abfd9a` in `xiaobizh_n18_cpu`,
  where it withholds 15 coverage paths, and the only one of those that is also a candidate,
  `atom/model_ops/v4_kernels/state_writes.py`, is absorbed either way by the candidate
  subtree entry above it. It is kept as a forward guard for trees this one does not
  represent, and it is not free: a full `pytest --collect-only` over the CPU tier and one
  more refusal path (exit 97). Whether that guard earns its cost is an open call
  (principle 3), not a settled one.

**That file errs in both directions, so do not cite it as a floor.** Toward *firing*: an
indented import in a CPU-tier test that does run is not credited, so a module can be listed
although the CPU tier reaches it, and coverage is subtracted by exact string, so a candidate
subtree entry is never cancelled by coverage of a file under it. Toward *silence*: imports
are read as text, not resolved as a graph, so a transitive import, an `importlib` call or a
re-export is invisible. A path *absent* from `gpu_gate_triggers.txt` is not a claim that the
CPU tier covers it, and a path *present* is not proof that it does not. Of the two mistakes
this gate prefers the first, because running the GPU tier when in doubt is never the wrong
one. Judging that a change sits outside the blind spot remains the task's own call.

**A task touching EPLB, CUDA-graph capture bounds, block tables, the paged index builders or
the prefix-cache kernels runs the GPU superset as part of its own gate, not at wave end.**
That is where the files running nothing here concentrate — 29 excluded plus 22 exercising
nothing, 51 in all; the silent 22 include `test_pool_index.py`,
`test_prefill_indices_paged.py`, `test_decode_indices_paged.py`, `test_postprocess_width.py`
and `test_prefill_prefix_vs_native.py`, the paged-index and prefix-cache areas `03` D13 and
`01` D6 lean on. `test_dp_metadata` and `test_dp_sync_layout` are **not** among them: they
are CPU-green, measured above, and booking a GPU for them spends the resource the booking
queue below exists to ration.

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
and the raw pytest output lives only in the containers that produced it; a citation to either
is not resolvable from a checkout, which is the same defect as naming a script that is not in
the tree. The commits carrying P0.2's measurements are `9c8df1328` and `b963c9411`, landed as
`947d5b282`; the ones that supersede them are on this branch. P0.1's GPU re-measurement has
an **in-tree** record, because a plan of record cannot cite one that is not: the five failing
node-ids verbatim in `scripts/compass/gpu_gate_known_failures.txt`, and the pair, the tree and
the toolchain as the `BASE_*` constants in `scripts/compass/gate_gpu.sh`. P0.2's own task
record predates that measurement and contains neither `4779` nor `fe9ea043c`; it is a source
for the superseded 4730 / 5 above and for nothing else.

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
`test_dp_sync_layout.py` and `test_forward_mode.py` already cover the pieces, CPU-only —
all four are in the CPU tier at `3afcb4880`, which was not true when this was written: the
last three sat in a 32-entry exclusion list until they were re-measured and found green.

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

### The pre-flight gate — four checks, not one

```
  0.  D-state census               -> thousands in D state? node is WEDGED.
  1.  timeout 25 rocminfo          -> hangs?  node is WEDGED. Do not use.
  2.  rocm-smi --showuse           -> compute busy?
  3.  rocm-smi --showmemuse        -> VRAM held?     <- the one that gets missed
```

**Check 3 is not optional.** A node can show 0% utilisation and ~99% VRAM — a loaded,
idle model. A pre-flight testing only utilisation calls that node free, and `non_torch` is
a **device-wide** reading, so a neighbour's allocation is indistinguishable from ours.

**Check 0 states its own scope, because in a container it cannot answer the question.** It
was added by P0.2 on the reasoning that `timeout` cannot kill a probe already in D state.
But a container's PID namespace is private, so the census counts the container, not the
node. Measured 2026-09-20T08:45Z on hjbog-srdc-39, the same second from both sides: the host
had **10840 processes of which 2236 were in D state**, while the `jgong5_vllm` container on
it saw **3171 and 103** — 4.6% of the D-state processes that were actually there, against a
threshold of 20 calibrated on host-wide readings. `docker inspect
-f '{{.HostConfig.PidMode}}'` is empty and no host procfs is bind-mounted, so there is
nothing to look through. `preflight.sh` therefore reads a host procfs when one *is* mounted
(`/host/proc`, `/hostfs/proc`, `/rootfs/proc`), detects the container positively
(`/.dockerenv` or `/proc/1/cgroup`) rather than inferring it from a low count, prints
`scope:` on every run, and when the scope is container-local and under threshold reports
**PARTIAL** — "nothing is wedged inside this container, the node was not examined" — instead
of a clear (principle 6: an unanswered question is not a "no"). Check 1 is what still sees a
host wedge from inside a container, because `rocminfo` goes through the driver.

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
