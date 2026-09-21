# ATOM Compass — Open Items: TODO Register, Assumptions, Gaps

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**What this is.** Everything across the sixteen design topics that is *not settled*, in one
place. Split out of `README.md` so the front page stays a bird's-eye view rather than a
backlog. Nothing here is a decision; every decision lives in its topic's decision log.

**How to read it.** Four sections, in decreasing order of how much rests on them:

1. **Load-bearing assumptions** — hold up large parts of the design; each has a check plan
2. **Missing topics** — design points nobody has written yet, with a recommendation
3. **TODO register** — 86 rows, **T1–T80 and T82–T87**, per topic, of which **80 are open**:
   T10, T15, T22, T48 and T65 are struck through as done, and T77 was opened and closed by
   P0.1.
   Both figures are the rows of section 3 below, counted as
   `grep -oE '^\| *~*\**T[0-9]+'` over that section and nothing else — prose elsewhere in
   this file names T-numbers that belong to other branches, and counting those tokens is
   what made two earlier counts disagree. The register is **not contiguous and is not a
   range**: T-numbers are allocated across parallel task branches and arrive when those
   branches land. T73–T76 arrived with P0.3 and T83–T87 with P0.6; at the time of writing
   T81 is still open on P0.4's branch
4. **Cross-cutting issues and pending amendments**

---

## 1. Load-bearing assumptions, and how each gets checked

Five assumptions hold up large parts of the design. **One has been tested** — T10, resolved
by P0.3 on 2026-09-20; the other four have not. Each row names the check, where it runs,
and roughly what it costs — so these are schedulable work in the execution plan rather than
caveats in a document.

| # | Assumption | If false | The check | Cost |
|---|---|---|---|---|
| **T21** | The in-situ calibration transfers across TP width | the recipe's "calibrate at TP1, predict TP2/4/8" collapses and the campaign multiplies by the number of widths | **Doc `07` Phase 1c's one extra run.** Calibrate at TP1, predict a TP2 full-engine run, compare. Already a designed step of the recommended flow — it is step 6 — so the check is not extra work, it is the reason that step exists. | one TP2 engine run, ~1 h GPU |
| **T25** | The real-vs-real noise floor stays narrow under closed-loop replay at high client count | those cells become ungradeable — not failed, *ungradeable*, which is worse because nothing is proven either way | **Doc `08` D45's step 1, run before any simulated comparison.** N≥3 spaced real repeats at the 64- and 256-client cells; report the pairwise spread. If it swamps 10%, say so and re-scope the acceptance cells. | 3 real cc-traces runs per cell, ~3 h GPU |
| **T5** | ATOM's model classes trace cleanly under `FakeTensorMode` at TP>1 | tier b has no IR, and docs `04`, `07` and `09` rest on it | **Trace the 27B at TP2 under the doc `04` D18 mechanism and diff the captured structure against TP1.** Run this first and cheaply. A fake-tensor trace should be GPU- and collective-free, so the known mode hang should not be reachable from it; T5 is the experiment that settles whether that reasoning holds. Needs a non-wedged node (`rocminfo` under `timeout` before starting). **Answered yes**, on two models at both widths, with the collectives recorded rather than substituted away, and the reasoning holds for the collectives that go through a registered custom op but not for the four raw `torch.distributed` call sites -- see `04` D18, *Collectives at TP>1*. The structure diff is there too. | half a day, one node |
| ~~**T10**~~ | ~~`AgenticReplayStrategy` can be subclassed rather than vendored~~ — **resolved 2026-09-20 by P0.3.** Yes, and it is not needed on its own: the strategy is built by the plugin factory, so an out-of-tree subclass displaces it with no upstream edit — but a subclass reaches only four of the nine pacing sites. All nine share one `LoopScheduler` resolved as a module global, so the adapter **rebinds that global** instead (`06` D34). No vendoring. **W1.9's 450–650 total is reopened, not settled** — see `06` D34's component table: one row was costed against the option-A wrapper, and T75 and T76 are new, uncosted scope. Evidence: six executed claims, `agent_scratch/compass_dev/p0_3/spike_t10.py`, zero edits to agentx-harness. | done |
| **T52** | `TorchDispatchMode` instrumentation does not hang ATOM at width | `--measure` runs and any mode-based instrumentation of a REAL execution are unusable. **Probably does not gate Phase 1a tracing** - the hazard is a `__torch_function__` guard on a real device, not a fake-tensor trace. | **Root-cause the known hang** at `atom/spec_decode/dspark_scheduler.py:264`. `rocgdb` attach, `info dispatches` per rank, identify which rank diverges. | <1 day, quiet node |

**Ordering.** T10 came first — no hardware, largest swing per hour — and is done. Then
**T5** — a `FakeTensorMode` trace should be GPU-free and collective-free, so it is both the
cheaper experiment and the one that tells us whether T52 gates anything on the critical
path.
**T52** follows, at ordinary priority unless T5 actually hits the hang. T21 and T25 need
the calibration and harness to exist, so they land later — but both are *designed-in
steps*, not add-ons, and neither should slip to the end.

**What each one costs if it fails.** T21 and T25 can each invalidate a whole acceptance
claim. T5 can invalidate a whole tier. T52 invalidates `--measure` and mode-based
instrumentation of a real run — narrower, but a designed path. Finding out late is the
expensive outcome in every case, which is the argument for putting them early rather than
where they naturally fall.

---

## 2. Missing topics

Design topics identified but not yet written up. Listed with a recommendation rather than
silently carried as gaps.

| # | Topic | Why it is missing, and why it matters | Recommendation |
|---|---|---|---|
| ~~M-a~~ | ~~Configuration surface and CLI~~ | Flags were scattered across five topics with no owner and no precedence rule. | **DONE** - topic `13_configuration_surface.md`, D78-D81. |
| ~~M-b~~ | ~~Refusal semantics end to end~~ | Seven documents emitted refusals and none said what one *does to the run*. | **DONE** - `08` D50.1: mark and continue, priced by the next answerable rung, with a >5%-of-predicted-seconds admissibility gate. |
| ~~M-c~~ | ~~Model loading without a GPU~~ | 关键技术点 1.4.2. Doc `03` covers memory *sizing*; nothing covers how the module tree comes into existence to be traced. ATOM has the pieces — `--load_dummy {empty,zero,xavier}`, `RapidServeModelRunner._init_weight_params_on_meta` — but no doc names the path or says whether weights are read at all. | **DONE** - `02` D10.1: HF-config geometry where only geometry is needed, construction inside `FakeTensorMode` with `--load_dummy empty` where a module tree is. |
| ~~M-d~~ | ~~TP / DP / PP / EP specifics~~ | M1 names all four, and DP couples *scheduling decisions* across ranks through a per-forward collective that rewrites the batch - so the LP structure is an M1 deliverable, not an M7 one. | **DONE** - topic `15_parallelism_support.md`, D88-D94, with an explicit M1/M7 split. |
| ~~M-e~~ | ~~Determinism and reproducibility~~ | Doc `08` **T26** asks for bit-reproducibility as a test, but nothing designs for it. Under a distributed CA, grant ordering is a function of real-time message arrival unless something pins it. Two runs of one configuration disagreeing would undermine every paired comparison. | **DONE** - `01` D3.4: the `(LP, virtual time, event)` sequence is what must reproduce; CA grants tie-break by LP identity. |
| ~~M-f~~ | ~~Speculative decoding / MTP~~ | Acceptance is a *behaviour* Compass cannot compute - the first quantity in the design that is neither derivable nor measurable. | **IN SCOPE** by decision 2026-09-19; topic `14_speculative_decoding.md`, D82-D87. Placed as **M3.5** (mechanism on Qwen3.8-27B), real claim at M5/M6. |
| ~~M-g~~ | ~~Simulated-run observability~~ | Doc `01` D3.1's open issue says the CA should own the global timeline log and the deadlock dump, and that "its output format is part of the acceptance evidence and should be designed, not improvised". Doc `11` covers Prometheus metrics, which is a different thing. Still improvised. | **DONE** - `01` D3.5: timeline log, deadlock dump, and an always-written run summary. |

**All seven are now closed.** M-a `13`; M-b `08` D50.1; M-c `02` D10.1; M-d `15`; M-e `01` D3.4;
M-f `14`; M-g `01` D3.5.

---

## 3. TODO register

### Topic 04 — model capture and cost IR

| # | Item |
|---|---|
| T1 | Inductor fusion correction (~4.8% of a decode step) |
| T2 | Enumerate the structure set for Qwen3.8-27B |
| T3 | Build the per-leaf parameter-extractor table (~20 entries) |
| T4 | Establish scratch constants per leaf for the 27B |
| T5 | Verify ATOM's model classes trace cleanly under FakeTensorMode at TP>1 |
| T6 | Validate that `Repeat` grouping reproduces the flat cost |
| T7 | Validate `Par` reconstruction from stream ids |
| T8 | Decide whether tier (a) is fitted independently or derived from tier (b) |
| T9 | Declare a row-ordering treatment for decode attention |
| **T51** | Enumerate the layer-pattern shapes for Qwen3.8-27B and Kimi-K3; confirm the nested-`Repeat` detector reaches the hierarchical form on both |
| **T52** | Root-cause the `TorchDispatchMode` 8-rank hang at `dspark_scheduler.py:264` — gates T5 |
| **T81** | Make the D18 capture *symbolic* on ATOM's real forward, or record that it cannot be. **Two independent specialisation sites are measured, and only one of them is reachable from the caller.** The capture is concrete — 0 non-numeric shape entries of 13,047 at TP2 — and the tracing mechanism is not the limitation: under the mode a GEMM and a softmax keep their free symbol with `shape_env.replacements` empty. **Site one, the bound.** `prepare_decode` derives one count per staged buffer and uses it to fill the buffer's numpy view before handing it to `copy_to_gpu`; anything that needs an `int` takes `__index__` of a `SymInt` and gets its hint, recording the symbol as a constant with no error and no warning. Measured on ATOM's path as `s56 -> 2` through `aiter_attention.py:1115 in prepare_decode`, and it is not a numpy behaviour — a bare `__index__()` and a plain list slice do the same. Passing the symbol as the bound closes this one. **Site two, the copy.** `copy_to_gpu` is `self.gpu[:n].copy_(self.cpu[:n])`, and `self.cpu` is a real numpy-backed tensor with constant dimensions, so `copy_` solves every symbolic dimension of the destination that the slice does not cover. Measured as `s64 -> 16384` through `aiter_attention.py:1142` → `atom/utils/__init__.py:725 in copy_to_gpu`, where `s64` is `block_tables: ['512', 's64']` — a dimension no bound controls. `A1_tp1.json` reaches `{s64: 16384}` with **no injected SymInt at all**, so this site is not an artefact of the injection attempts. **Consequences.** `CpuGpuBuffer` is *not* unchanged by a repair: it contains site two. Supplying a symbolic bound from `prepare_decode` is therefore **not shown sufficient**, and nothing here shows it is. Of the two repair routes originally recorded, "a symbolic `CpuGpuBuffer`" is the one site two points at, and "a capture entry point below `prepare_inputs`" is **untested** — neither closed nor shown unreachable. No claim is made here that a symbolic capture requires changing ATOM's serving path; that would be inference, and the experiment that would settle it has not been run. Pinned by `tests/compass/test_capture_symbolic_shapes.py`, including a check that ATOM still shares the one bound, so the conclusion is retaken rather than inherited if that changes. Gates T5 alongside T52. |

### Topic 06 — workload harness contract

| # | Item |
|---|---|
| ~~T10~~ | ~~Verify `AgenticReplayStrategy` can be subclassed rather than vendored~~ — **done**: yes, but the adapter rebinds the shared `LoopScheduler` global instead, which covers all nine pacing sites |
| T11 | Build the per-tokenizer vetted filler-token set |
| T12 | Chase the 32 `asyncio.wait_for` sites under virtual time |
| T13 | Decide the simulated KV connector's completion semantic |
| T14 | Build the client-count matrix given only 144 fan-out-capable sessions |
| ~~T15~~ | ~~Warmup handling in the harness contract~~ — **done**: warmup requests are ordinary requests; the rule is an exclusion window agreed by request id |
| **T54** | Detect warmth that *recurs* mid-run (a new shape reaching autotune at minute 10); D62 measures leading warmth only |

### Topic 07 — calibration toolchain

| # | Item |
|---|---|
| T16 | Calibrate `compass plan`'s GPU-time estimates |
| T17 | Draw the boundary of the standalone `ModelRunner` bench |
| T18 | Verify a replayed step table reproduces the forward context faithfully |
| T19 | Decide the artifact store's physical form |
| T20 | Declared node + standalone benchmark for invisible collectives |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) |
| ~~T22~~ | ~~Analytic laws as their own design topic~~ — **done**, now `10` |
| **T55** | Decide the treatment of `c10d::broadcast_`, which takes a `ProcessGroup` no artifact can hold — working answer is to fold it into the Phase 1c host floor |
| T86 | **aiter is a second executed source root, and no artifact key names it.** Opened 2026-09-21 by P0.6. D41 gives every artifact a provenance stanza naming *the* executed source root, and keys `price_list` on `(model, width, source-root digest)` — one root, ATOM's. But the EP group is constructed entirely in aiter (`15` D92), the MoE kernels and every collective are aiter's, and `aiter` is installed in the container's writable layer at `/app/aiter-test`, outside `/workspace` and discarded by `teardown.sh`. D43's matrix already invalidates `price_list`, `region_terms` and `memory_readings` on an AITER bump — but a fingerprint can only refuse on a version something recorded, and nothing records one. It is **not** a recoverability problem: that checkout is a clean `https://github.com/ROCm/aiter.git` at a commit on `origin/main`, so the hash resolves. It is a provenance gap, and the machinery already exists — `scripts/compass/gate_gpu.sh:153-159` resolves aiter's checkout from `aiter.__file__` and takes `git describe --tags --always --dirty`, comparing it against `BASE_AITER`. W2.4 adopts the same call. **Two aiter versions are in circulation on this project right now**: `v0.1.20-103-g23f83724f` in `jgong5_vllm` and `v0.1.21.dev0-49-gf4e7c7509` on node 18, so this is an observed divergence rather than a hypothetical one. The decision half — whether an aiter bump invalidates an EP artifact or only warns — is the owner's, and it is cheap to take before the first artifact exists rather than after. |
| T87 | **`07`'s price-list table states the MoE all-to-all's block-count cap as one number, and it is two.** Opened 2026-09-21 by P0.6. `_get_dispatch_config` returns `min(128, CU)` blocks at 16 warps for **prefill** and `min(64, CU)` at 4 warps for **decode** (`atom/model_ops/fused_moe/mori_prepare_finalize.py:257-261`), so on node 18's 80-CU MI308X the cap is 80 for prefill and 64 for decode, not one figure. aiter's own config-time default is a hard-coded 80 for IntraNode and 32 plus 16 RDMA for InterNodeV1 (`aiter/dist/device_communicators/all2all.py:64-75`), overridden per call. The cell that hides the split is the `exclusive` row of `07`'s calibration table; `04`'s `exclusive` join-policy text has the same single-number phrasing. Reporting one number for two is the aggregate-without-decomposition failure principle 7 names. |

### Topic 08 — validation protocol

| # | Item |
|---|---|
| T23 | Choose the family-2 distance function |
| T24 | Define "structural event" for family 3 beyond prefill streaks |
| T25 | Measure the real-vs-real noise floor under closed-loop replay at high client count |
| T26 | Assert simulator bit-reproducibility as a test — mechanism now `01` D3.4; this item is the CI wiring |
| T27 | Decide the 256-client cell's construction |
| T28 | Establish whether ranking/regret becomes an explicit acceptance gate |
| ~~T48~~ | ~~What a refusal does to a run~~ - **DONE**: `08` D50.1, mark-and-continue with a 5%-of-seconds admissibility gate |

### Topic 09 — fitting and law selection

| # | Item |
|---|---|
| T29 | Choose the hull implementation (convex hull vs k-NN threshold) and its threshold |
| T30 | Enumerate candidate laws per leaf family, with held-out validation shapes |
| T31 | Test raggedness, cached fraction and chunk-position for treatment status |
| T32 | Decide whether tier (a) is derived from tier (b) or fitted independently |
| T33 | Establish `warmup_seconds` for Qwen3.8-27B under the current stack |

### Topic 10 — analytic laws

| # | Item |
|---|---|
| T34 | Test whether the activation coefficient is derivable from geometry |
| T35 | Derive FLOPs and bytes-moved expressions for the ~20 opaque leaves |
| T36 | Name the collective algorithm per code path in the machine spec schema |
| T37 | Decide whether the host floor is derivable or stays a per-model constant |
| T38 | Build the analytic-vs-measured ratio report as part of the empirical campaign |
| T39 | Establish a dispatch-band table from geometry where no measured bands exist |
| **T56** | Record the observed tier-0 error per measured device in the artifact, so an unmeasured-device user sees a range rather than a promise |

### Topic 11 — metrics support

| # | Item |
|---|---|
| T40 | Confirm the new histograms are **classic**, not native |
| T41 | Audit every `observe()` site; extend the AST test to observation arguments |
| T42 | Measure `collect_metrics()` per-step cost on the real side; decide decimation |
| T43 | Verify the backfill end to end — one block, loaded, visible in Grafana |
| T44 | Sanity-check histogram bucket ranges against simulated latencies |
| T45 | Tag ATOM's existing twenty metrics with their D77 class |
| T46 | Decide the DP-aggregation rule per class; refuse summaries there |

### Topic 13 — configuration surface

| # | Item |
|---|---|
| T57 | Generate the `ATOM_COMPASS_*` environment twins from the flag table rather than hand-writing them |
| T58 | Decide whether a per-leaf tier override is worth the reproducibility cost |

### Topic 14 — speculative decoding and MTP

| # | Item |
|---|---|
| T59 | Capture `ATOM_ENABLE_RELAXED_MTP` in the run fingerprint - it changes acceptance *semantics* (`RELAXED_TOP_N` 1 to 10, `RELAXED_DELTA` 0 to 0.6) and is invisible to every artifact key today |
| T60 | Test whether a draft forward's cost is linear in `K` - serial MTP should be, a real draft stack need not be |
| T61 | Decide how chunked prefill and drafting interact, and what structure that produces |
| T62 | Assert the host acceptance draw and the Triton kernel agree: same declared rates, same seed, same accepted-count distribution over a few thousand draws |
| T63 | Add ATOM flag `--spec-decode-acceptance-rates` (list) - contract 2 has no transport today; the CLI exposes only the two scalars |

### Topic 16 — execution plan

| # | Item |
|---|---|
| T71 | Add Wave 4+ detail as Phase 0 and T21 answers arrive |
| T72 | Decide whether reviewer agents use ATOM's `review-pr` skill or a Compass-specific checklist |
| T73 | **Declare the plugin entry point with a dotted module path, and defer the rebind from that package's `__init__.py` with a `sys.meta_path` hook.** Successor to T10, opened 2026-09-20; re-scoped 2026-09-20 after review refuted its premise by execution, and re-measured 2026-09-21. `plugins.py:210` calls `importlib.util.find_spec` on the entry-point value, and on a **dotted** value that imports the parent package — so `compass_harness.plugin:plugins.yaml` executes `compass_harness/__init__.py` inside `discover_plugins()`, which runs at import of `aiperf.plugin.plugins` and therefore before any `PhaseRunner` is constructed, with the manifest still resolving and zero edits to agentx-harness. The rebind cannot run *inline* there: the bootstrap fires at `plugins.py:1115`, inside that module's own body, so any `import aiperf.…` re-enters `aiperf/plugin/enums.py:21` and raises `AttributeError: partially initialized module`. The bootstrap therefore installs a stdlib-only import hook that rebinds `LoopScheduler` on `aiperf.timing.phase.runner` after that module executes. Executed against `56a0cf70f` over five entry-point/bootstrap combinations, 7/7 in the working shape; decomposed in `06` D34. **A packaging decision plus ~20 lines, not a precondition of the seam** — the earlier "no such bootstrap exists today" was an asserted negative and is false. The tripwire ships regardless, because a bootstrap that silently fails to run has no other detector — but it covers only that half: an inline rebind de-registers the plugin the tripwire lives in, so W1.9 also needs a positive check that the Compass plugin registered at all (`06` D34, `16` W1.9) |
| T74 | **Whether the 32 `asyncio.wait_for(..., timeout=T)` sites need virtual time.** Successor to T10, opened 2026-09-20. They bypass `LoopScheduler`, so the rebind does not reach them; under virtual time they may fire instantly. P0.3 did not examine them (`06` D34, second risk) |
| T75 | **Decide the two `loop.call_later` idle-cap timers.** Opened 2026-09-20. `replay_dependencies.py:307` and `agentic_replay.py:592` arm real-clock timers the `LoopScheduler` rebind cannot reach; upstream's own docstring says the second deliberately lives outside the scheduler. Either override the two `_arm_*_idle_watchdog` methods from the Compass subclass, or declare the idle-cap feature unsupported under virtual time and assert both caps are `None`. Related: `agentic_replay.py:531` derives the virtual-time skip from `time.monotonic()` (`06` D34) |
| T76 | **Reconcile `seamless=True`, which keeps two `PhaseRunner`s and two schedulers live at once.** Opened 2026-09-20. `phase_orchestrator.py:267` builds one runner per phase and tracks `_active_runners`. W1.9 either reconciles two concurrent schedulers against one virtual clock or asserts `seamless=False` (`06` D34) |
| T80 | Raise with ATOM's owners: `tests/test_prefix_cache_accuracy.py` has no test function — it is an `argparse` script driving a live server — and `test_kv_connector_scheduler.py` / `test_transfer_engine.py` have been dead since #690. Measured: all three run nothing in **either** tier |

### Topics 02, 01 — gaps now closed

| # | Item |
|---|---|
| **T77** — closed 2026-09-20 by P0.1 | Re-measure the GPU-tier baseline and record it in full. As stated it was unverifiable: `gate_gpu.sh` judged the superset against `BASE_PASSED=4730 / BASE_FAILED=5` measured by P0.2 at `83daf636d`, with `BASE_TORCH=UNRECORDED`, `BASE_ROCM=UNRECORDED`, no AITER version, a toolchain-drift warning guarded on `BASE_TORCH != UNRECORDED` so the one check that would catch drift could not fire, and — sharper — no failing node-ids on file, so "5 failed" could not be checked against "the *same* 5 failed". **Closed by** a re-measurement on node 18 in `xiaobizh_n18`, `HIP_VISIBLE_DEVICES=1`, 2026-09-20, at `fe9ea043c`: **4779 passed / 5 failed**, 0 errors, 105 skipped, 3 xfailed, 72.6 s, two runs with byte-identical failing sets, under torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509**. All five node-ids are written out verbatim in `scripts/compass/gpu_gate_known_failures.txt` and compared by name, and the drift guard now fires because every version field is recorded. The five are pre-existing and unrelated to Compass, and are **four ULP comparisons plus one bitwise check**, not five ULP failures: four `allclose` cases in `tests/test_fused_compress_ragged.py` off by one bf16 ULP (`max\|diff\| = 0.001953125`, exactly 2^-9, against `atol=rtol=1e-3`), plus `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise`, a `torch.equal` with no tolerance at all. What remains open is not T77: the baseline is a *pass count*, so it moves whenever `tests/compass/` grows, and `gate_gpu.sh` handles that by re-deriving the expectation per tree (`4779 + (this tree's tests/compass count - 49)`) rather than by re-measuring. |
| T68 | Enumerate buffer allocations in the two target models and confirm none escapes `FakeTensorMode` - `--load_dummy` and the meta wrapper act on parameters only |
| T69 | Whether a quantized checkpoint's geometry is derivable without reading it; header derivation was 3.3% low at TP=4 on the 27B |
| T70 | Estimate the timeline log's volume at PP degree > 1 - grants scale with stages and microsecond lookahead, so it is largest exactly where it is most wanted |

### Topic 15 — parallelism support

| # | Item |
|---|---|
| T64 | Establish whether ATOM microbatches PP - changes the LP event rate and the bubble model |
| ~~T65~~ | ~~Establish EP's group membership per supported configuration; if EP spans DP, the LP collapse does not hold~~ — **answered 2026-09-21 by P0.6**, stated in `15` D92. The EP group is built in **aiter**, not in ATOM (`aiter/dist/parallel_state.py:1926-1945`; no ATOM file assigns `_EP`), and is `dp × pcp × tp` within one PP stage — so EP does span DP wherever DP exists, and the caveat's second cross-DP barrier (MoRI dispatch/combine) does appear. **The LP collapse survives anyway**: the group is exactly one PP stage's ranks, and `pp > 1` with `dp > 1` is refused at `engine_core_mgr.py:297-300`, so D93's formula is unchanged and only D92 Q1's stated *reason* was wrong. Three findings were carried out as successors — **T83**, **T84**, **T85** — plus **T86** and **T87** under topic 07. Out of scope and left untouched: PD disaggregation crossed with EP, and what `MORI_SHMEM_MODE=ISOLATION` does to it (`01` D6 already carries a "confirm" on that variable). |
| T83 | **The EP group's size and `moe_parallel_config.ep_size` are two different numbers, and they disagree in two configurations.** Opened 2026-09-21 by P0.6. The group is `dp × pcp × tp` (aiter); the config number is `ep_size = tp_size` (`moe.py:299-300`), flattened across DP only when `enable_dp_attention or moe_ep_flatten_tp_across_dp` (`moe.py:240-242`) and folded with PCP only under `ATOM_PCP_MOE_MERGE` (`moe.py:272-280`). At **`-tp 4 -dp 2 --enable-expert-parallel` without DP-attention** — the command `docs/distributed_guide.md:20` advertises, and which no recipe uses — the group is 8 while the config number is 4, so MoRI v1 is built over 8 ranks (`num_ep_ranks`, `moe.py:654`) with `num_local_experts` sized for 4 (`moe.py:664`), and its destination rule `expert_id // num_experts_per_rank` then only ever addresses ranks 0-3. MoRI v2 takes both from the group (`mori_v2_prepare_finalize.py:640,666`), so the two transports disagree exactly here. A second instance: `local_ep_size = data_parallel_size_local * tp_size_` (`moe.py:313-314`), which reaches MoRI as `gpu_per_node`, omits PCP while the group includes it — unreachable today only because `use_all2all_kernels` needs `dp_size > 1` and PCP with DP-attention is refused (`llm_engine.py:75-81`), but PCP with plain DP and EP was not found refused anywhere. **Settled by** one 8-GPU startup logging the two values, or by an owner statement that the combination is unsupported — in which case the fix is a refusal at `FusedMoEParallelConfig.make` and a correction to `docs/distributed_guide.md:20`. Compass's own consequence either way: carry both numbers and assert they agree, because ATOM does not. |
| T84 | **`15` D94's M1 EP2 leg is degenerate as written.** Opened 2026-09-21 by P0.6. At `dp_size == 1` ATOM builds no all-to-all: `use_all2all_kernels` requires `dp_size > 1` (`moe.py:201-211`), it is the only gate on `MoriPrepareAndFinalize` (`moe.py:736-742`), and without it the layer falls through to `fused_moe(..., expert_mask=...)` (`moe.py:907-915`). So `-tp 2 --enable-expert-parallel` would pass a scheduling-agreement test while exercising nothing EP-specific. The only non-degenerate EP2 is `-tp 2 --enable-dp-attention --enable-expert-parallel`, which `CoreManager` rewrites to `dp=2, tp=1` (`engine_core_mgr.py:281-295`). An owner decision plus one line in D94. |
| T85 | **Multi-node EP rank-to-node mapping is assumed, not verified.** Opened 2026-09-21 by P0.6. MoRI infers node identity as `ep_rank // gpu_per_node`, which is correct only if consecutive EP ranks are physically consecutive GPUs on a node. At `pp == 1` the group is `[0..world-1]` in order, and `ParallelConfig` enforces that a node owns a contiguous DP slice (`config.py:949-960`), so it holds **on paper**; nothing here measured more than one node. **Settled by** a 2-node DP+EP run logging `all2all_manager.internode` and each rank's EP group. Needs two nodes; M7-era, not on P0's path. |
| T66 | Measure whether the Class-C runtime constants move with PP degree |
| T67 | Measure the step-duration spread across DP ranks, and what padding to `unified_bs` costs |
| T78 | **`atom/models/qwen3_5.py:427` and `atom/models/glm4_moe.py:426` block PP at compilation level >= 2.** Opened 2026-09-20 by P0.5. Both declare `"intermediate_tensors": 0` in `dynamic_arg_dims`. `atom/utils/decorators.py:525` raises `ValueError("Unsupported dynamic dimensions ...")` for any non-Tensor argument, and that raise sits **before** the decorator's own `IntermediateTensors` handling at `:533-537`, which marks the token dim of each contained tensor. A PP non-first stage receives its activations inside an `IntermediateTensors` container, so the declaration can never be satisfied. Levels 0 and 1 escape it: `decorators.py:485-488` sets `do_not_compile` for `NO_COMPILATION` and `DYNAMO_AS_IS` - 0 and 1 per `atom/config.py:113-116` - and `:505` returns before the raise is reached. Observed at the default level 3: `-pp 2 --enforce-eager` on Qwen3.8-27B raised `ValueError: Unsupported dynamic dimensions [0] for argument intermediate_tensors with type <class 'atom.models.utils.IntermediateTensors'>` (node 18, container `xiaobizh_n18`, `/workspace/p05-t64/pp2_27b.log:126`, 2026-09-20); the same invocation with `--level 0` reached KV sizing instead. **Two instances, one class:** `grep -rn '"intermediate_tensors": 0' atom/models/` returns exactly these two files, so an upstream fix scoped to the M2/M3 target model alone leaves `glm4_moe.py` broken under PP. Other PP-capable models use a bare `@support_torch_compile` and reach the working path. **Not on M1's path**: `15` D94's PP2 test runs a fake model - `tests/test_pp.py` mocks `aiter.dist.parallel_state` (`:126-128`) and builds a `MockConfig` (`:463-468`), loads no model file, and is in the CPU tier. It blocks any **real-model** PP measurement, so it is a prerequisite for M7 cost accuracy and for T66. Upstream ATOM fix, not Compass; owner unassigned. |
| T79 | **`atom/model_ops/attentions/gdn_attn.py:1329-1331` mixes a PP-local layer count with a global one.** Opened 2026-09-20 by P0.5. `total = runner._get_total_num_layers()` returns the **PP-local** slice under PP>1 (`atom/model_engine/model_runner.py:1509-1515`, via `get_pp_indices`), but line 1330's `num_draft = total - hf_config.num_hidden_layers` subtracts the **global** count, and `runner.num_full_attn` (`gdn_attn.py:159-161`) is global as well. Measured on Qwen3.8-27B (64 layers, `full_attention_interval` 4, `num_full_attn` 16) on node 18, container `xiaobizh_n18`, 2026-09-20, with `python -m atom.examples.simple_inference --model <Qwen3.8-27B snapshot> -pp {1,2} --enforce-eager --level 0 --max-num-batched-tokens 512`: PP1 gives `num_draft=0`, `n_full=16`, logs `sub-pool kv: entries=80746, entry_bytes=1056768` and exits `EXIT=0` (`/workspace/p05-t64/ctrl_pp1.log:43` and `:410`); PP2 gives `num_draft=-32`, `n_full=-16`, logs `entries=0, entry_bytes=-1056768, num_kvcache_blocks=0` and exits `EXIT=1` (`/workspace/p05-t64/pp2_27b_l0.log:77-80` and `:160`). An exact sign flip at an identical magnitude, so the mechanism is determined rather than inferred. The two invocations also differ in the visible device set (1 vs 1,2) and in `--max-tokens` (8 vs 64); neither enters the `entry_bytes` computation, which is fixed at startup by the layer counts above. Same M1/M7 split as T78: the fake-model path never reaches this code. Upstream ATOM fix. |
| T82 | **`15` D91 Q2's "no scheduling coupling" is contradicted by the head's decode admission loop.** Opened 2026-09-21 by P0.5. `atom/model_engine/scheduler.py:1761-1763` skips any running sequence whose id is in `_pp_inflight_token_block` while composing a decode batch, and `mark_pp_inflight` (`:2309-2316`) puts exactly the sequences whose token is in flight into that set - its docstring at `:2310-2313` reads "Head: block re-scheduling of seqs whose token is now in flight." So the first stage's batch composition at step N is a function of pipeline in-flight state, which is the coupling D91 Q2 says PP does not add. The mechanism is T64's own finding: the head keeps up to `pp_size` independently scheduled batches in flight, and the block is what stops a sequence decoding against a stale token. The item is the amendment to D91 Q2 and whatever follows for the PP LP model; this row registers it and does not amend the decision. |

### Topics 01, 03, 05 — newly opened

| # | Item | Topic |
|---|---|---|
| **T47** | A lookahead that is wrong but never exercised by the workload is not detected by the straggler check | `01` |
| **T49** | The prefix-index *lookup* cost is charged to nobody — ~1,387 blocks hashed and probed per request at the cc-traces p50, magnitude unmeasured | `03` |
| **T50** | Whether runtime memory constants transfer across dies (the working assumption says yes within a software generation) | `03`, `05` |
| **T53** | Whether tokenizer throughput transfers across CPU classes (the working assumption says yes, adjusted by derate) | `05` |

---

## 4. Cross-cutting issues

Beyond the per-topic TODOs.

1. **Silent failure is the dominant risk mode.** The always-on causality detectors
   (`01` D3.2), loud deadlock aborts, and the AST clock-site test are the design, not
   decoration.
2. **Simulation speed is unmeasured under this architecture.** The prior design ran
   **0.30×** under saturation — slower than the system it simulates. Measure a saturated
   cell early, not at the end.
3. **Per-step replay CPU cost** was 4.3 ms against a 32.7 ms modelled step, and only with
   the bound allocation in the cache key; a shape-only key is unsound.
4. **Schedule agreement must be reported separately from latency**, and it was established
   two changes *before* the latency numbers were right.
5. **Scheduling fidelity is unobservable at saturation.** Some cell needs deliberate slack.
6. **Multi-node DP is implemented but not hardware-validated** — `docs/distributed_guide.md`
   §9 carries the banner. If paired evidence is needed there, the real side may not exist.
7. **A user guide is a deliverable, not documentation debt.** `compass plan` is designed
   so the tool tells the user what to measure, which only works if the flows, the
   recommended calibration sequence and the flag surface are written down for a reader
   who was not in these design conversations. Requested during review; belongs in the
   execution plan as its own task, not as a trailing chore.
8. 7. **ATOM's `main` moves while Compass is built.** The seam (`Config.runner_qualname`) has
   two in-tree users so it is unlikely to vanish, but the ~55 synchronization sites of
   `01` D4 and the clock-read sites of `11` D72 are ordinary code that upstream will
   touch. The CI clock-source lint is the detector; a rebase cadence is an execution-plan
   question.

---

## 5. Pending amendments

Corrections identified while writing later documents, not yet applied to earlier ones.

| Document | Amendment |
|---|---|
| `02`, `04` | "Two tiers" becomes **three** — analytic/roofline is a tier in its own right (`07` D36), not merely rung 4 of the resolver ladder |
| `02` | Extend `CostBackend` with the resolver ladder and compositional `provenance_mix` |
| `05` | D26's `transfer` probe is **deferred**: no cross-hardware transfer of empirical data (`07` D36) |
| `04` | **Applied.** D18's open issue *"whether ATOM's real model classes trace cleanly under this mode at TP>1"* is answered yes and the measurement is now in `04` itself. What remains an amendment is the other half: the trace is **concrete, not symbolic** — 0 non-numeric shape entries of 13,047 — so it does not carry D18's argument against route D, and D18 should say so where it makes that argument (T81) |
| `04` | D18's five `torch.cuda` stubs are not enough to import ATOM on this stack. Construction needs three more (`get_device_properties`, `current_device`, `get_device_capability` — the first is read at *import* by aiter's Triton attention configs) and running `ModelRunner.__init__` needs fourteen more, tagged `needed_for: "model_runner"` by `install_runner_stubs` in `atom/compass/capture/fake_trace.py` (the import-path set carries `needed_for: "import"`) and pinned by name in `tests/compass/test_capture_fake_trace.py`. `get_device_properties` and `mem_get_info` are *declared readings* in the `03` D14 sense and must be passed in, never read from a host |
| `04` | D18 does not cover raw `@triton.jit` launches. They bypass the dispatcher entirely, so `FakeTensorMode` cannot fake them and the first one reached kills the trace in `triton/backends/amd/driver.py:369`. Any inventory taken with them skipped is a **diagnostic**, not a capture, and must be labelled so wherever it is reported |
| `04` | A TP>1 capture taken through `apply_simulated_tp` at one physical rank both **erases** and **fabricates**: 129 `all_reduce` per forward become the identity and appear nowhere in the inventory, while one `all_gather` becomes six real dispatched ops over a half-zeros tensor. Neither direction is visible in the operator list itself. **Measured since: the substitution is not needed.** At an honest width the same forward completes and records all 129 (`04` D18, *Collectives at TP>1*), where the substituted one refused at 2,477 ops with none. Removing it from the capture path is a separate task; the module still takes the substituted route |
| `15` | **D92's decision row still reads *"inherits the TP group"* and *"remainder included"***, and P0.6 measured both false — the group is built in aiter and spans DP, and an indivisible expert count is refused rather than rounded (`15` D92, T65). D92's **conclusion** and D93's formula survive unchanged, so this is a correction to the decision's stated reasons, not to the decision. Rewriting a decision is the owner's call; the body of D92 carries the measurement in the meantime. |
| `04`, `07` | The MoE all-to-all's block-count cap is stated as a single number in `04`'s `exclusive` join-policy text and in `07`'s calibration table; it is `min(128, CU)` for prefill and `min(64, CU)` for decode (**T87**) |
