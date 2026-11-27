---
name: compass-advice
description: Run a Compass `compass advice` parallel-configuration search on pit2 (MI355X, pit2-p03-gXX) end to end - a space file from a recipe, the search on CPU, the calibrations it prints run on GPUs under gpool, the results and dashboard, held-out real runs (one_real.py) and H1 send-order seeds. Use when asked to 并行寻优, 寻优, 推荐并行配置, 用 advice 比较并行配置, compass advice, advise a parallel configuration, compare TP/DP/EP/PP or P/D configurations by simulation on pit2, or to rerun or extend the MiniMax-M3.1 1P1D search.
version: 0.1.0
scope: jgong5/ATOM pit2 line (pit2/m31pd), scripts/compass/pit2/, scripts/compass/ADVICE.md, atom/compass/advice/
last_updated: 2026-11-27
---

# compass-advice (pit2)

This skill lives on the pit2 line by the owner's request. Design 19's D108
(`atom/compass/design/19_adoption.md`) decides "a user guide, no skill" for
`feature/atomcompass_new`, and #2722 removed `compass-onboard`; moving this skill
there needs its own ruling first.

Read `scripts/compass/ADVICE.md` first. It is the general guide: the space
file's fields, each candidate's state, what a calibration prints, the results.
This skill adds what pit2 needs on top: the containers, gpool, the Store, the
scripts in `scripts/compass/pit2/` (each says its usage in its header), and the
traps hit on the M3.1 search (LLM-Booster #781, jgong5/ATOM #4256).

## 0. What it costs, and what is already there

- **Search:** CPU only, in a container with no GPU. Each point is one
  simulated deployment and one client run. A 900 s M3.1 point takes 20 to 80 CPU
  minutes.
- **A configuration whose laws the store lacks:** one calibration on a whole
  node. M3.1 took 4.9 node-hours for decode A and 6.4 for B (calibrate.sh records
  of 900 s each).
- **Platform readings:** once per image and machine, about 0.75 node-hours.
- **Reusable store:** `/share_nfs/xingjche/m31pd/store` holds MiniMax-M3.1-Flash's
  laws for a PP4×TP1 prefill on the Mooncake Store with decode A (TP4,
  `ATOM_M3_INDEXER_CP=1`) or B (DEP4). It also holds pit2's platform readings
  under `rocm/atom-dev:nightly_202611240650`.
  - A variant of `scripts/compass/spaces/minimax-m31-pd-pit2.json` that changes
    only the concurrencies, the duration, the SLO or the trace selection runs
    with no GPU.
  - Any other engine argument or performance variable re-keys the laws and
    needs a calibration.

## 1. Setup, once

On the host, with a clone or worktree of the pit2 line at its head (`git fetch
origin && git checkout --detach origin/pit2/m31pd`):

```bash
export PIT2_DATA=/share_nfs/<you>/compass-pit2      # your shared directory: logs, real runs, Store runs
export PIT2_MOUNTS=/share_nfs/xingjche/m31pd        # the shared store, traces, network and Store entry the space file names
scripts/compass/pit2/containers.sh cpu <you>_advice_cpu 0-63
docker exec -it <you>_advice_cpu bash
# inside:
export PIT2_DATA=/share_nfs/<you>/compass-pit2
. <tree>/scripts/compass/pit2/env.sh
<tree>/scripts/compass/pit2/bootstrap.sh            # prints "venv-sim ok", atomesh's commit, BOOTSTRAP_DONE
```

- `bootstrap.sh` builds `venv-sim` (HARNESS_PYTHON) and atomesh (ATOMESH, about
  10 minutes) under `PIT2_WORK`.
- An atomesh already built at the tree's commit may be passed instead:
  `SKIP_ATOMESH=1 ATOMESH=<binary>`.
- Source `env.sh` in every shell a step runs from. It sets `HARNESS_PYTHON`,
  `AIPERF_PYTHON`, `ATOMESH`, `AITER_LOG_LEVEL=WARNING` and
  `COMPASS_INTEGRATION_REF` (default `origin/pit2/m31pd`; Traps), and defines
  `compass`.

## 2. The space file

- Write it from a recipe as ADVICE.md "1. Fill the space file" says.
- The worked example is `scripts/compass/spaces/minimax-m31-pd-pit2.json`, the
  recipe `recipes/mesh/MiniMax-M3.1-PD-PP4-MooncakeStore.md`. It holds:
  - `machine`: `mi355x`, PTL disabled, and the network file
    `/share_nfs/xingjche/m31pd/net/pit2-link.json`, probed on pit2 (#3720
    cell B);
  - `store`;
  - `commit`, the frozen ATOM;
  - `workload`: agentx traces `/share_nfs/xingjche/m31pd/inputs/traces.jsonl`,
    `select: agentx`, the concurrencies, `duration_s`;
  - `slo`: `min_interactivity_p90`, `max_ttft_p90_s`, empty for none;
  - `common`: the 1P1D basis, the prefill's arguments and variables, and the
    Store entry `/share_nfs/xingjche/m31pd/store_entry/mooncake_store.json`;
  - `dimensions`: `{"decode": {"A": ..., "B": ...}}`, one candidate per value.
- A variant whose laws exist:

```bash
python3 - <<'EOF'
import json
d = json.load(open("<tree>/scripts/compass/spaces/minimax-m31-pd-pit2.json"))
d["workload"].update(concurrency=[16, 32], duration_s=300)
d["slo"] = {"min_interactivity_p90": 150, "max_ttft_p90_s": 3}
json.dump(d, open("/share_nfs/<you>/compass-pit2/space.json", "w"), indent=1)
EOF
```

Keep the shared files read-only.
- Put the space file and OUT under `PIT2_DATA`: both containers mount it, and a
  calibration writes into `OUT/cal/`.
- A calibration publishes into the space file's `store`. Before calibrating a
  new configuration, copy the store, 56 MB, and name the copy in your space file:
  `cp -a /share_nfs/xingjche/m31pd/store $PIT2_DATA/store`. The copy keeps the
  M3.1 laws and pit2's readings.
- Likewise a new Store entry or network file goes in your own directory.

## 3. Run the search

```bash
cd / && taskset -c 0-11 nice -n 19 compass advice SPACE OUT > OUT.log 2>&1
```

- It checks, simulates and ranks in one run, and prints each step once that step
  has ended (ADVICE.md "2. Run the search").
- It runs `cpus // the widest point's ranks` points at once. An M3.1 point is 5
  ranks, so `taskset -c 0-11` runs 2 at once.
- A rerun into the same OUT skips each finished point.
- Each candidate's state is in `OUT/states.json`:

```bash
python3 -c "import json; s = json.load(open('OUT/states.json')); [print(c, v['state'], {p: q['state'] for p, q in v.get('points', {}).items()}) for c, v in s.items()]"
```

A worked run, the variant above (A and B at c=16 and 32, 300 s, SLO 150
tok/s/user and 3 s, `taskset -c 0-11`):
- The check took 6 minutes and the 4 points 18 more, 2797 CPU s.
- It printed, in order:
  - `the reuse check at f1230b51a... (origin/pit2/m31pd-advice)`;
  - for each candidate, `law prefill: reuse: step_linear_law 61eb9c870c2e in
    /share_nfs/xingjche/m31pd/store` and `state: searchable, 2 of 2 points
    searchable`;
  - `A-c16: finished in OUT/runs/A-c16, 493 CPU s`;
  - the table;
  - `Recommended: A at concurrency 32: 17030.4 tok/s/GPU ...`;
  - `Runner-up: B at concurrency 32: 15224.3 tok/s/GPU ...`.
- Compare figures only between runs of the same duration: the client's warmup
  is a larger share of a short run, so the same point's throughput is lower at
  300 s than at 900 s (A c=32: 17,030 against 28,551 tok/s/GPU).

| state | next |
|---|---|
| `searchable` | it simulates; read the results (5) |
| `missing a law` | save its `made by:` command and calibrate (4) |
| `calibrating` | wait; run the search again later |
| `missing readings` | run the `compass measure memory` command it prints on the GPUs (4) |
| `refused` | read the reason (Traps, ADVICE.md). A refusal is a result, not a step to work around |

## 4. GPU steps under gpool

Every GPU step holds a whole-node gpool lease of its own and releases it when
the step ends (read the gpu-pool skill once). `gpool_steps.sh` does both, from
the host, in the background:

```bash
export PIT2_DATA=/share_nfs/<you>/compass-pit2
setsid nohup <tree>/scripts/compass/pit2/gpool_steps.sh g40,g52 STEP... > /dev/null 2>&1 &
tail -f $PIT2_DATA/logs/gpool_steps.log        # "granted <lease> on <node>", "released <lease>: <exit line>", ALL_DONE
```

- **NODES** is one node, a comma list (the pool grants one of them) or `any`.
  Leave out the node whose CPUs your search uses.
- Each step goes through these stages, in order:
  1. it queues with `--wait --timeout 3h`;
  2. it starts the GPU container on the granted node;
  3. it checks that the GPUs are idle;
  4. it runs the step;
  5. it releases the lease with the step's exit line.
- A failed calibration or platform step stops the list.

| STEP | what | out |
|---|---|---|
| `platform:STORE` | platform readings into STORE (`platform.sh`), when the store has none at the image's stack | `PLATFORM_EXIT 0` |
| `cal:OUT/cal/A.cmd` | the calibration `compass advice` printed (`cal_step.sh`) | `CAL_EXIT A 0` |
| `real:OUT/cal/A.cmd:16` | a held-out real run of A at c=16 (`real_step.sh`) | `REAL_EXIT A 16 client=0 deploy=0` |

**Calibrate:**
1. Save each candidate's `made by:` command, its first line, as
   `OUT/cal/<candidate>.cmd`.
2. Run `RECORDS_ONLY=1 gpool_steps.sh NODES cal:OUT/cal/A.cmd cal:OUT/cal/B.cmd`.
   `cal_step.sh` moves the command's Store address to the node, starts a fresh
   Store there (`store.sh`), and runs each record for 900 s (`DURATION_S`).
3. With `RECORDS_ONLY=1` it stops calibrate.sh at its first fit, so the lease is
   not held through CPU work. Then run the fits in the CPU container:
   `scripts/compass/pit2/cal_fits.sh OUT/cal/A`. It prints `CAL_FITS_EXIT 0`, and
   `OUT/cal/A/status` ends with `end calibration <h> node-hours`.
4. Run `compass advice SPACE OUT` again.

**Machine inputs a new configuration may need:**
- **Platform readings:** the `platform:` step, when `compass advice` refuses a
  candidate for missing readings at the image's stack.
- **The Store entry:** `fit_store.sh ENTRY OUT/cal/A OUT/cal/B` in the CPU
  container, after the calibrations. It needs `OFFLOAD_PROFILE=1` on the prefill
  (the M3.1 space sets it). It also needs one calibration's prefill at
  `ATOM_LOG_LEVEL=DEBUG`, for the lookup lines; without them `lookup_s` stays
  unfitted.
- **The network file:** reuse `pit2-link.json` on pit2. A new fabric takes
  `scripts/compass/rdma_link.py` on two nodes.

## 5. Read the results

- **`OUT/results.json`**: its `text` starts `Recommended:` or `No
  recommendation:`, followed by its rows. `OUT.log` ends with the same table and
  `Left out:` lines.
- **`OUT/dashboard.html`**: throughput per GPU against p90 interactivity, the
  frontier and the recommendation, colored by configuration. Serve it from the
  CPU host: `python3 -m http.server 8767 --bind 0.0.0.0 --directory OUT`, then
  open `http://<host address>:8767/dashboard.html`. Stop the server when done.

What a row's state means:
- **ungraded**: the point simulated with no input outside its laws' fitted
  ranges. Advice never compares a point with a real run, so every clean point is
  ungraded. That says nothing bad about it; grade it yourself (6).
- **coverage report**: some input fell outside the range a law was fitted on, so
  the simulator extrapolated. One such input excludes the point from the
  recommendation, however small its effect.
  - M3.1's A c=32 and c=64 were excluded for one or two requests whose bodies
    were just under `http_intake`'s fitted minimum. Each moved its intake time by
    about 1 µs of 1.5 ms, and both points' throughput was within 1.3% of the real
    runs.
  - The row names what the law lacks; calibrating with such inputs, or a
    tolerance in the rule, brings the point back.
- **refused / failed**: the point did not run, or did not finish (`Left out:`
  with each try's reason).
  - A finished point keeps `OUT/runs/<point>/point.json`; a failed point has
    none, and is tried again from a fresh directory on the next search.
- **The recommendation** is the most throughput per GPU among the points that
  meet the SLO and are not excluded, or among all of them when there is no SLO.
  An excluded point may still be the real best: say so beside the
  recommendation.

## 6. Optional: grade the answer

- **Held-out real runs:** run `gpool_steps.sh NODES real:OUT/cal/A.cmd:16
  real:OUT/cal/B.cmd:16 ...`, into `PIT2_DATA/real/<candidate>-c<K>`. Then, in
  the CPU container:
  `PYTHONPATH=$PIT2_TREE python3 $PIT2_TREE/scripts/compass/one_real.py REAL OUT/runs/A-c16`.
  - It prints each metric's error and the real run's own 95% interval.
  - It refuses a real run whose server compiled while it served. If the compile
    stalled nothing graded, `COMPASS_ACCEPT_JIT_BUILDS=1` accepts it; say so.
  - Report throughput per GPU with every GPU counted, the PP stages included.
- **H1, three send orders (LLM-Booster #778):** one real run against the mean of
  three simulated runs with seeds 0, 1, 2:
  `PYTHONPATH=$PIT2_TREE AT_ONCE=3 nice -n 19 python3 $PIT2_TREE/scripts/compass/pit2/seed_sims.py OUT_SEEDS OUT/runs/A-c16 ... --seeds 0,1,2`,
  then `one_real.py REAL OUT_SEEDS/runs/A-c16-s0 ...-s1 ...-s2`.
  - Report each seed's error and the mean, TTFT p90, TPOT p90 and throughput per
    GPU first.
  - The unseeded run is not seed 0: do not count it.
- **A simulator fix on points a search left failed:**
  `rerun_points.py OUT_FIX NAME=OUT/runs/<finished point of the same candidate>:C ...`.
  For advice's own answer afterwards, search again.
- **A candidate searched apart**, its calibration ending late:
  `merge_results.py OUT_MERGED OUT_FULL OUT_PART B`. Searching the whole space
  again once every calibration has ended gives the same answer.

## Traps

- **`the reuse check refuses: the frozen commit ... is not one of <ref>'s`**:
  `COMPASS_INTEGRATION_REF` must name a branch holding the space file's `commit`.
  `env.sh` sets `origin/pit2/m31pd`, which holds f1230b51a since #4256 merged;
  `git fetch origin` first, as the check reads the ref as last fetched. Left at
  `compass`'s default `origin/feature/atomcompass_new`, the pit2 tree's digest is
  refused.
- **`OMP_NUM_THREADS`, `MKL_NUM_THREADS` or `OPENBLAS_NUM_THREADS` set in a
  search's or calibration's shell:** a law's key holds the performance
  variables, so every law is looked up at another key and found missing. Use
  `nice -n 19` and the cpuset for courtesy, never thread caps.
- **calibrate.sh refuses a resume (`started with other inputs`):** its `inputs`
  hold the commit's SHA as `tree`, `DURATION_S`, the traces, the arguments and
  the store.
  - After a commit that touched only `atom/compass` (the laws' ATOM digest
    leaves it out), copy `inputs` aside and move its first line to the new
    commit (`cal_fits.sh`'s header).
  - After any other commit, calibrate anew.
- **A record taken while a kernel compiled, or while the tree had uncommitted
  files:** take it again.
  - Move its directory aside (`mv OUT/cal/B/cal3-c16 OUT/cal/B/cal3-c16.dirty`)
    and resume; calibrate.sh takes the missing step again.
  - Never edit the tree while a record runs.
- **Never edit, commit or check out over a script a run executes:** bash reads
  it as it runs. A commit to calibrate.sh mid-run killed a calibration ("Stale
  file handle" over NFS). Commit in another worktree, or wait.
- **Files a container wrote are root's:** `docker exec NAME chown -R
  $(id -u):$(id -g) DIR` before moving or deleting them from the host. A failed
  point's retry that could not clear its directory fails at once with `holds an
  earlier deployment's files`.
- **pit2-p03-g27's CPUs are shared:**
  - another session's sim-speed timing runs on 0-63 and owns 64-127;
  - keep searches and fits at nice 19, within 0-63;
  - run no GPU step on g27.
- **Killing your own run:**
  - `pgrep -f <pattern>` and `pkill -f` match the shell running them when the
    pattern is in its own command line;
  - kill by process-group number (`kill -TERM -- -PGID`) after listing with `ps`;
  - only your own processes, never another user's.
- **`docker exec NAME python3 - <<EOF` runs nothing:** without `-i` the heredoc
  never reaches the container. Use `docker exec -i`.
- **`from atom.utils import envs` at a module's top in `atom/compass/kv`**
  loads torch, which `test_kv_driverless_import` refuses. Import it where it is
  used.
- **A high-concurrency 1P1D point stalls with the clock stuck** (`PD
  backpressure: parked=N, waiting=M, running=0`, the client silent): chunked P/D
  transfer with `pd_admission` under overload. Fixed on the pit2 line by
  9fa2fa45e, 2c11a9979 and 96d7c7e93; a tree without them stalls A c=256 and
  B c=64 to 256. A `BackdatedEvent` on `engine-P-pp0->mooncake_store:lookup` is
  fixed by 719738582.
- **Deployments:** a space file states `single` or `1p1d` only, so there is no
  xPyD (1P2D) and no multi-node. Each parallel configuration is a block you write;
  advice does not expand TP/DP/EP sizes itself.
