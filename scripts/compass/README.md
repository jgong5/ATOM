<!-- SPDX-License-Identifier: MIT -->
# Compass gate and pre-flight scripts

The scripts every task runs around its work: two test gates, a GPU pre-flight, the
snapshot builder a gate runs against, and the two list regenerators. Everything
here runs **inside a container**, which mounts the worktree parent so every task's
tree is reachable, and each script derives its tree from **its own path**, not from
your `$PWD`, so the copy in a worktree exercises that worktree. Run **your**
tree's copy: running another checkout's copy is refused rather than silently acted
on, because until 2026-09-20 it was not — see "Which tree a script acts on" below.

| Script | Where | What |
|---|---|---|
| `gate_cpu.sh` | CPU container | The CPU test tier, run per task. 130 of 189 test files, no driver, ~31 s. **Must be green.** Also exits 98 when the diff is in the blind spot below and the GPU tier has not run on *this* tree. Refuses a caller-supplied `-r`. |
| `gate_gpu.sh` | GPU container | The GPU test tier, run per wave — and per task for a blind-spot diff. Superset (`--ignore=tests/plugin`), judged as a **delta** against **4779 passed / 5 failed** at `fe9ea043c`, with torch, HIP, ROCm and AITER recorded and compared. The five failing node-ids are on file in `gpu_gate_known_failures.txt` and compared **by name**. Calls `preflight.sh` itself, before and after. |
| `preflight.sh` | GPU container | `rocminfo` reachability, compute use and VRAM use (checks 1–3), plus a D-state census (check 0), and a list of bookable devices. Run **before and after**. The census states its own scope: in a container it can only see this PID namespace, and says so rather than reporting a clear node. Also worth running for a CPU-only task: on a wedged node ATOM's *import* hangs, because aiter shells out to `rocminfo`. |
| `regen_cpu_gate_exclude.sh` | CPU container | Regenerate the **GENERATED** section of the exclusion list by iterating `--collect-only` to a fixed point (three passes at `83daf636d`: 37 errors, 1, clean). Preserves the MANUAL section verbatim. |
| `regen_gpu_gate_triggers.sh` | CPU container | Re-derive `gpu_gate_triggers.txt` from the exclusion list and the tree. Run after **any** change to `cpu_gate_exclude.txt` — the two files are derived from the same tree and are wrong separately. Writes the whole file, **header counts included**; nothing in it is maintained by hand. |
| `snapshot.sh` | either | Build the `git archive` tarball a gate runs against (never `rsync`). Stamps `.compass-commit` (so output names its tree, and a `COMPASS_GPU_GATE_DONE` attestation can be checked) and `.compass-changed` (so the blind-spot question is answerable without `.git`). Refuses a dirty tree. |
| `cpu_gate_exclude.txt` | — | **29** excluded test files in two marked sections: **28 GENERATED** (`# BEGIN GENERATED`, driver-dependent at *collection* time; never hand-edit — `regen_cpu_gate_exclude.sh` rewrites it wholesale) + **1 MANUAL** (`# BEGIN MANUAL`, collects cleanly then fails on a driver call, so the regenerator cannot see it). The MANUAL section **is** hand-edited; every entry must carry its observed failure above it. |
| `gpu_gate_triggers.txt` | — | **30** source paths that no *running* CPU-tier test names. Generated, not hand-written; matched by `gate_cpu.sh`. A trailing `/` matches a subtree. |
| `gpu_gate_known_failures.txt` | — | The five known-failing GPU node-ids at `fe9ea043c`, verbatim. Its line count and `BASE_FAILED` are two statements of one fact; `gate_gpu.sh` refuses to run if they disagree. |
| `_lib.sh` | — | Tree resolution, `PYTHONPATH`, the `import atom` assertion, commit stamp, and the `tests/compass` pass count the GPU gate derives its allowed surplus from. |

## Baselines — two tiers, two commits, two provenances

Not one baseline. The rows below were measured at different commits by different
tasks, so each row names its own commit and the tier it was measured in.

| Tier | Result | Measured |
|---|---|---|
| CPU gate (130 files) | **4030 passed, 0 failed**, 149 skipped, 3 xfailed, rc=0, **identical in ten runs on 2026-09-21, the clock 25.4-31.7 s of pytest inside 31.1-37.8 s of wall (`time` real) — a measured spread, not a bound** — decomposing as **3956 ATOM + 74 `tests/compass`** | node 18, container `xiaobizh_n18_cpu`, 2026-09-21, against a `git archive` snapshot with `PYTHONPATH` asserted and pytest's own rc captured before any pipe |
| GPU superset (`--ignore=tests/plugin`) | **4779 passed, 5 failed**, 0 errors, 105 skipped, 3 xfailed, **72.6 s**; two runs, byte-identical failing sets | `fe9ea043c`, node 18, container `xiaobizh_n18`, `HIP_VISIBLE_DEVICES=1`, 2026-09-20, torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm release **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509** (`git describe`) |
| `ruff check .` | 1003 errors, 640 fixable — the gate is *no new* error, not zero | `83daf636d` |
| `black --check .` | clean, 660 files | `83daf636d` |

**The CPU total moves when `tests/compass/` grows, and one of the things that grows
it is this directory.** `tests/compass/test_cpu_gate_exclude.py` parametrises
`test_every_trigger_path_still_exists` over the entries of `gpu_gate_triggers.txt`,
one case each, so regenerating that file changes the pass count by exactly the
change in the number of entries. The readings in circulation are the same gate
under different exclusion lists and a different `tests/compass`, not discrepancies:
**3925** at 32 exclusions; **3956** is the ATOM-only half; **3988** with
`tests/compass` at 32; **4005** with it at 49; **4022** with it at 66 — 49 plus the 17
extra trigger paths the corrected derivation below produces — and **4030** with it at 74,
the 8 cases `test_gate_gpu_surplus.py` adds. The
exclusion list went 32 → 29 because `test_dp_metadata.py`, `test_dp_sync_layout.py`
and `test_forward_mode.py` were re-measured **CPU-green**.

**The five GPU failures are four ULP comparisons and one bitwise check** — not
"five bf16 ULP failures", which is what this file and three design documents said
until the review that produced this paragraph:

| Node-id | Character |
|---|---|
| `tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend0-context0-cut+whole]` | `allclose`, off by one bf16 ULP: `max\|diff\| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3` |
| `…[extend1-context1-whole+cut]` | same |
| `…[extend2-context2-resume+fresh]` | same |
| `…[extend4-context4-tiny-then-long]` | same |
| `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise` | `torch.equal` — **bitwise, no tolerance at all**, so "one ULP" does not describe it and a tolerance bump would not move it |

All five are pre-existing and unrelated to Compass: no importable `atom` module
differs between `fe9ea043c` and the integration base `83daf636d` — the only delta
under `atom/` is markdown documentation, which no test imports. Cite the delta;
do not claim green.

## Which tree a script acts on

Each script resolves its tree from `$BASH_SOURCE`, so it acts on the checkout it
lives in. That is deliberate, and it is not the same question as "the tree the
caller meant".

**EXECUTED 2026-09-20.** From a checkout at `83daf636d` on `feature/atomcompass_new`,
running another worktree's copy —
`bash …/compass-worktrees/p0/scripts/compass/snapshot.sh <outdir>` — wrote
`compass-d78f3bbd3.tar` stamped `commit: d78f3bbd3…` and exited 0, with no warning.
The stamp was truthful about the tree it archived and silent about the tree the
caller was standing in. All five callers of `compass_tree_root` had the same
exposure; for a gate the consequence is worse, since a pytest result would be
attributed to a commit it did not come from, and a `COMPASS_GPU_GATE_DONE`
attestation would then agree with a stamp that describes the wrong branch.

`compass_tree_root` now **refuses** (exit 99) when `$PWD` is inside a *different*
checkout. Resolving from `$PWD` instead was rejected: it would let a script act on
a tree that is not its own. Being invoked by path from outside any checkout is
still allowed — there nothing contradicts the caller, and it is how
`snapshot.sh`'s own extract instructions say to run a gate.

## The CPU tier's blind spot — a file, and a gate

`gpu_gate_triggers.txt` is the blind spot stated as paths a script can match:
**30 source paths**, generated by `regen_gpu_gate_triggers.sh`. The rule it applies:

> an `atom` module named by an excluded test is a blind spot **unless a CPU-tier
> test that actually runs names it too**.

Four decisions make that sentence operational. Three were forced by a defect or a
counter-example this tree measured; the fourth, collection, currently changes no
path and is kept as a forward guard:

1. **Indentation.** The parser anchored on `^`, and **190** of this tree's
   `import atom.*` lines in non-plugin test files are indented — function-local and
   guarded imports are the dominant idiom here. It was reading about a third of the
   lines it claimed to read. The excluded side now reads any indentation.
2. **Symbols.** `from atom.model_ops import eplb` parsed to the bare package
   `atom.model_ops`, which resolved to the whole 106-file subtree and swallowed
   every sibling entry. `from X import y` is now read as `X.y`, resolved to
   `X/y.py`, else `X/y/`, else `X.py`, else `X/__init__.py`. An `atom.*` reference
   that resolves to none of those is a **refusal**, not a warning: either the parser
   misread a line or a test imports something absent, and both make the derived sets
   incomplete by an unknown amount.
3. **Skips.** An indented import is not proof that anything executed it. The
   excluded side reads any indentation, but coverage is credited **only for a
   module-level import**, and that half carries both counter-examples:
   `tests/model_ops/test_moe_dp_token_capacity.py:39` imports `topK` inside a test
   marked `skipif(not torch.cuda.is_available())`, and `tests/test_mla_index_cache.py`
   imports `ModelRunner` at **lines 99–100, indented four spaces inside a test
   function** — not at module level, as this file and three others said until
   2026-09-20. Reading coverage at any indentation drops the set 30 → 29, losing
   `topK.py`; doing that *and* crediting files that collect nothing drops it 30 → 27,
   losing `atom/model_engine/model_runner.py` — the module Compass's runner seam
   replaces — plus `aiter_mla.py` and `topK.py`. Measured at `236abfd9a` in
   `xiaobizh_n18_cpu`.
4. **Collection.** A file that collects nothing covers nothing, so coverage is
   credited only from a CPU-tier file `--collect-only` shows collecting at least one
   test. On **this** tree that probe removes no path: **30 triggers with it, 30
   without, difference empty** (measured at `236abfd9a` in `xiaobizh_n18_cpu`; it
   withholds 15 coverage paths, of which the only candidate,
   `atom/model_ops/v4_kernels/state_writes.py`, is absorbed either way by the
   candidate subtree entry `atom/model_ops/v4_kernels/`). It is a forward guard for
   trees this one does not represent, not the thing that keeps `model_runner.py`, and
   it costs a full `pytest --collect-only` over the CPU tier plus a refusal path
   (exit 97).

So the two sides are **asymmetric on purpose**: candidates come from any import in
an excluded test; coverage is subtracted only for a **module-level** import in a
CPU-tier file that collects at least one test, because those are the only imports
that provably execute (collection imports the module). Making both sides
indentation-tolerant would undo point 3 and re-delete `topK.py`.

**It errs in both directions, so it is not a floor.** Toward *firing*: an indented
import in a CPU-tier test that does run is not credited, so its module can be listed
although the CPU tier reaches it; and coverage subtraction is exact-string, so a
candidate **subtree** entry is never cancelled by coverage of the files beneath it,
nor a candidate file by coverage of its package. Toward *silence*: imports are read as text, not
resolved as a graph, so a module reached only transitively is invisible to both
sides. A path **absent** from the file is not a claim that the CPU tier covers it,
and a path **present** is not proof that it does not. Of the two mistakes, firing
spuriously is the one to prefer: running the GPU tier when in doubt is never the
wrong one, and the asymmetry above is that preference written down.

`gate_cpu.sh` **enforces** that preference rather than restating it. It takes the
changed-file list from `COMPASS_CHANGED_FILES`, else from `git diff` against the
merge-base with `feature/atomcompass_new` (`COMPASS_INTEGRATION_REF`); with neither
it exits **98** rather than answering "no". When a changed file matches a trigger,
the run ends 98 unless `COMPASS_GPU_GATE_DONE` discharges it:

| `COMPASS_GPU_GATE_DONE` | Outcome |
|---|---|
| unset | **98** — run `gate_gpu.sh`, then re-run with that tree's HEAD |
| set, but this tree has no commit (no `.git`, no `.compass-commit`) | **98** — an attestation names a tree; accepting an uncheckable one makes the rule bypassable in exactly the snapshot where the gate normally runs. Rebuild with `snapshot.sh`. |
| set to a sha other than this tree's HEAD | **98** — the GPU tier passed on a different tree |
| equal to this tree's HEAD | discharged |

The trigger match is *reported* before pytest and *enforced* after it, so one run
yields both answers instead of trading one for the other. Green at the CPU tier is
not green at the test gate when the diff is in the blind spot.

## What the GPU gate expects, and on a tree that carries no Compass tests

`gate_gpu.sh` judges an **equality**, not a floor: `BASE_PASSED` plus whatever
`tests/compass/` contributes on the tree in front of it, minus what it contributed
at the baseline (`BASE_COMPASS_TESTS=49`). A floor would be loosened by exactly the
tests each task adds, so a task adding 30 tests while silently losing a 20-test file
would still clear it.

| this tree's `tests/compass/` | expected passes |
|---|---|
| absent | `4779 + 0 - 49` = **4730** — the figure a tree without this phase's tests measures |
| present, N passing | `4779 + N - 49` |
| present, unreadable | **93** — a surplus with no source is not a measurement |

The absent case is the common one: every tree except a Compass task's own has no
such directory, the integration branch included. Deriving the surplus with
`pytest tests/compass --collect-only` made that case exit 4 (`file or directory not
found`), left the count empty and refused with `GATE_GPU_RC=93` before running a
test — reproduced on `4da2f3a2d` and on PR #9's branch, so it was a property of the
script and not of any change. An absent directory and an unreadable one are told
apart by asking the filesystem, not by parsing pytest's error text.

Both sides of the arithmetic are **pass** counts. The tree side used to be a
*collected* count, which is the same number only while `tests/compass/` holds no
skip and no xfail; the first Compass test to skip on a GPU host would have made a
green tree read `unaccounted -1` and blamed tests outside `tests/compass/`. Counting
passes costs one pytest run over a CPU-only directory and removes the condition.
`compass_compass_pass_count` in `_lib.sh` holds the derivation, and
`tests/compass/test_gate_gpu_surplus.py` covers each of its branches with a stubbed
`python`, so the absent-directory path is tested without a driver.

## Why the gates refuse a caller-supplied `-r`

Both gates forward `"$@"` to pytest. pytest's `-r` is **store-last-wins**, so a
caller's `-rE` replaces the gate's own `-rf`/`-rfE` and no `FAILED ` lines are
printed at all. `gate_gpu.sh` builds its verdict out of those lines: **EXECUTED on
`d78f3bbd3`, `gate_gpu.sh -rE` compared the five baseline failures against an empty
observed set, reported all five as "no longer failing", and exited
`GATE_GPU_RC=0` while they had in fact failed.** Both gates now refuse the flag
(exit 95) rather than absorb it, and `gate_gpu.sh` treats "baseline failures no
longer failing" as a refusal in its own right — it has three causes (genuinely
fixed, stopped running, could not be parsed) and only the first is a pass.

## Why `import atom` is asserted before every run

A prior run resolved `atom` to a non-git snapshot of a different branch, 72 files
divergent, and failed silently wherever both trees defined the symbol. `_lib.sh`
sets `PYTHONPATH` to the tree and **nothing else** — inherited entries are the
hazard, not a convenience — then asserts the resolved path is under it, and exits
92 if not. The probe runs from `/` on purpose: run in place, Python puts the cwd at
the head of `sys.path` and the assertion passes whatever `PYTHONPATH` says.
