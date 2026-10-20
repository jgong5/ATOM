<!-- SPDX-License-Identifier: MIT -->
# Compass quickstart: simulate a model and grade it against real runs

This page takes one model from an empty store to a graded simulation. The example
is the cell Compass is graded on: Qwen3.8-27B, TP1, on one MI308X, driven by
cc-traces agentic sessions at concurrency 1, 4 and 16. Not at 64 or 256: there
aiperf's own warmup of these sessions cannot finish within a capture's window, so
the owner dropped them for this model and machine, in calibration and grading, as
not a sustainable configuration
([ruling on #1590](https://github.com/jgong5/ATOM/issues/1590#issuecomment-6281930268)).
`smoke_cell.sh` runs the same path at one concurrency in about an hour, so start
there (below), then run the full cell.

## Concepts

Compass replaces only ATOM's forward pass. The scheduler, the KV block manager,
the prefix cache and the API server run unchanged, on a simulated clock, and each
forward step costs what a fitted law predicts. No GPU is used.

| Term | Meaning |
|---|---|
| record | `compass record`: a real ATOM server on a GPU that writes a step record, each forward step's batch and timing, until it is stopped. |
| client run | One `client.sh` invocation against a server: aiperf replays cc-traces sessions, then the client writes `artifacts/client_run.json`, which names every request it sent. One client run per server. |
| calibration | The records the cost laws are fitted on. They replay traces 256 to 392, never the traces the cell is graded on (0 to 255). |
| fit | `compass fit steps` fits the step cost law (here the step linear law). `compass fit host` fits the host laws, the CPU time around each step. |
| store | The artifact store (`--store`). A command publishes what another reads, and prints its token: `published ... as <token>`. Records and logs stay in `--out`. |
| memory readings | What an engine start reads off the device. `compass measure memory` takes them once per configuration, and a simulated run sizes its KV pool from them. |
| run | `compass run`: the same server, simulated. Its forward steps are priced by the fitted laws. |
| cell | One configuration and workload in one directory: two or more real runs (`real1/`, `real2/`) and one simulated run (`step_linear/`). |
| grading | `python -m atom.compass.validate CELL`: the simulated run against the real runs, each metric against the real runs' own spread. A developer check, outside the `compass` CLI. |

## Environment

Blocks marked `# host` run on the node; every other block runs in the GPU
container, entered with `docker exec -it ${USER}_compass_gpu bash`.

- **Image.** `rocm/atom-dev:vllm-v0.31.0-nightly_20261013`. Node 18 holds it as
  `compass-next:nightly-20261013`, the same image: name that tag there, or Docker
  pulls. The machine specs were measured in it, and a fit refuses records from
  another stack than its machine spec states.
- **One GPU container**, named for its owner and purpose, with the GPU devices,
  host networking, the inputs mounted read-only, the data directory (Disk) at
  its own path, and a tmpfs over each cache the runs fill:

  ```bash
  # host
  mkdir -p <data dir>
  docker run -d --init --name ${USER}_compass_gpu --network host \
    --device /dev/kfd --device /dev/dri \
    --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
    --tmpfs /root/.cache:exec --tmpfs /root/.aiter:exec --tmpfs /tmp:exec \
    -v <model repo>:/models/qwen38:ro -v <traces dir>:/cc:ro -v <data dir>:<data dir> \
    <image> sleep infinity
  ```

  `--init` reaps what a stopped server leaves; without it, processes stay
  defunct. A tmpfs needs `exec`, since the engine loads the shared objects built
  there, and it empties when the container stops: treat a restarted container as
  fresh (step 0).
- **GPUs are shared.** Use a GPU that `rocm-smi --showuse --showmemuse` reads at 0%
  use and 0% VRAM, with no process in `rocm-smi --showpids --showpidgpus`. Every
  GPU command checks this before and after it runs and refuses a busy GPU. Never
  stop another user's process or container. Real runs time the host too, so run
  at most two real runs at once, and step 2's calibration alone on the node:
  start it when `rocm-smi --showpids` prints `No KFD PIDs currently running`. On
  a busy node, wait for that before step 2, since a calibration beside other
  work fits their load into its laws. The smoke cell checks the path, not the
  laws, so it may run beside them.
- **One tree.** Snapshot the integration branch's tip from a clone of
  `https://github.com/jgong5/ATOM` (its `origin`), with no uncommitted change,
  and extract it into the data directory:

  ```bash
  # host, in the clone
  git fetch origin
  git checkout --detach origin/feature/atomcompass_new
  mkdir -p <data dir>/tree
  COMPASS_INTEGRATION_REF=origin/feature/atomcompass_new scripts/compass/snapshot.sh <data dir>/tree
  mkdir -p <data dir>/tree/<sha> && tar -x -C <data dir>/tree/<sha> -f <data dir>/tree/compass-<sha>.tar
  ```

  `snapshot.sh` archives HEAD, which `git fetch` does not move: without the
  checkout it snapshots a stale commit and says nothing. It names the tarball,
  `<sha>` being the commit's first 9 characters;
  its lines on commits behind, changed files and `gate_cpu.sh` are for the gates.
  `<data dir>/tree/<sha>/ATOM` is `$T`. Run every step from it. Records, fits and
  runs keyed by another tree's source refuse each other. A git worktree used in a
  root container fails git's ownership check, so a fit refuses it as "neither a
  git checkout nor a stamped snapshot"; a snapshot avoids that. After switching
  trees, clear ATOM's compile cache in the container (`rm -rf /root/.cache/atom/*`)
  while no server runs.
- **The machine spec** of your node's GPUs, from `scripts/compass/machines/`,
  where each file is named for its device and host (`mi308x-hjbog18.json` is node
  18's). Its `device.ptl` must be the cards' PTL state, which
  `cat /sys/bus/pci/devices/*/ptl/ptl_enable` prints; `measure memory` refuses a
  spec that states another. A node with no spec of its own needs a developer to
  write one with its per-machine script first.
- **Inputs you supply:** a model directory; a cc-traces `traces.jsonl` (sha256
  `e39cd2ff3eba...`, 568,864,747 bytes); and two client Pythons (Variables). On
  node 18 the model repo is `/data/xingjche/compass/inputs/models--Qwen--Qwen3.8-27B`,
  mounted at `/models/qwen38`, and the traces directory
  `/data/xingjche/compass/inputs/cc-traces-weka-062126-256k`, mounted at `/cc`.
- **Disk.** A run takes about 400 MB, since the raw export keeps every prompt; the
  full cell about 8 GB, the smoke cell 180 MB. A client of 256 traces maps about
  1.4 GB more in its `OUT/client` while it runs. The container's `/root/.cache`
  tmpfs grows by about 1 GB in the smoke cell and 3.5 GB on the full path, 2.3 GB
  of it aiperf's dataset maps, and `/root/.aiter`, where aiter builds its
  templated modules, by about 230 MB: about 1.5 GB in all for the smoke cell.
  aiter's JIT directory, `/app/aiter-test/aiter/jit`, holds the image's prebuilt
  modules, so it stays in the container's layer on Docker's storage disk
  (`docker info -f '{{.DockerRootDir}}'`), gaining each module a server builds,
  a few MB. Keep data off the root disk. When no disk has room for the data
  directory, as on node 18 while `/data` reads full, use the host's tmpfs, such
  as `/tmp/<user>_compass`: memory, lost at reboot, its files written as root,
  so copy out what should stay and remove the rest from inside the container.

## Variables

Paste this into every container shell, with your data directory, tree and GPU:

```bash
DATA=<data dir>                  # the data directory (Disk)
T=$DATA/tree/<sha>/ATOM          # the snapshot tree (One tree)
R=$DATA/qwen38-tp1               # this cell's data directory
G=<gpu>                          # the GPU, as rocm-smi numbers it
MODEL=/models/qwen38/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0  # client.sh reads it too
TRACES=/cc/traces.jsonl          # the cc-traces file; client.sh reads it
ATOM="--model $MODEL --max-model-len 262144"   # ATOM's arguments, identical for every step
SPEC=$T/scripts/compass/machines/mi308x-hjbog18.json   # this node's machine spec (Environment)
M="--machine $SPEC"
S="--store $R/store"
V=$DATA/venv-sim                 # the compass-harness venv (below)
REAL_PY=/opt/venv/bin/python3    # a real server's client: the image's, with agentx-harness's aiperf
SIM_PY=$V/bin/python             # a simulated server's: also compass-harness
compass() { $T/scripts/compass/compass "$@"; }   # the CLI, from its own tree; the image does not install it
export MODEL TRACES AITER_LOG_LEVEL=WARNING
cd /                             # so no checkout's atom shadows $T's
```

Its `AITER_LOG_LEVEL` counts: a law refuses a run whose
environment differs from its records', and `smoke_cell.sh` sets that level too.
Every `python3 -m atom.compass...` below runs with `PYTHONPATH=$T`: without it,
`atom` is the image's own copy in `/app/ATOM`.

The simulated client's venv, once:

```bash
rm -rf /tmp/ch_src && cp -r $T/compass_harness /tmp/ch_src
/opt/venv/bin/python3 -m venv --system-site-packages $V
echo 'import site; site.addsitedir("/opt/venv/lib/python3.12/site-packages")' \
  > $V/lib/python3.12/site-packages/zz_opt_venv.pth
$V/bin/pip install --no-deps --no-build-isolation /tmp/ch_src
```

One server and its client run, used by steps 0, 2, 3 and 5:

```bash
# serve OUT TRACES C PYTHON COMMAND...: start a compass server writing to OUT,
# send it one client run of D seconds (default 1800), and stop it. The server
# logs to OUT/server.log; the compass command's own output, empty unless it
# refuses, goes to $R/logs/. It refuses an OUT that exists, and fails when the
# server dies or does not answer within 30 min, or the client run fails.
serve() {
  local out=$1 traces=$2 c=$3 py=$4 log=$R/logs/${1#$R/}.log pid end=$((SECONDS + 1800))
  shift 4
  [ ! -e $out ] || { echo "serve: $out exists; rm -rf it to take the step again" >&2; return 1; }
  mkdir -p ${log%/*}
  $T/scripts/compass/compass "$@" --out $out -- $ATOM > $log 2>&1 &
  pid=$!
  until [ -s $out/server.json ]; do
    kill -0 $pid 2>/dev/null && ((SECONDS < end)) ||
      { kill $pid 2>/dev/null; echo "serve: no server; see $log" >&2; return 1; }
    sleep 5
  done
  HARNESS_PYTHON=$py $T/scripts/compass/client.sh $out --traces $traces \
    --concurrency $c --duration ${D:-1800} --stop || { kill $pid 2>/dev/null; wait $pid; return 1; }
  wait $pid
}
```

To stop a step, press Ctrl-C until the prompt returns: a Ctrl-C can end a
client run early and let its loop go on to the next. Then run `jobs`; if it
lists a compass command, `kill %%; wait` stops its server and waits for it to
leave. Before taking a stopped step again, `rm -rf` its `--out`: it holds a
partial or a short run.

`client.sh`'s flags and exit codes are in its header. It sends one warmup request
per lane before the graded window; keep it, since grading refuses a run whose
warmup is missing or overlaps its graded steps.

## The path

Times were measured on node 18 for the cell in
[PR #1336](https://github.com/jgong5/ATOM/pull/1336), at concurrencies 1, 4, 16,
64 and 256 with a 1800 s benchmark each.

**0. Warm a fresh container** (GPU, once per container; 16 min).

```bash
D=600 serve $R/warm 0-7 1 $REAL_PY record --gpu $G
```

A server builds an aiter kernel the first time it meets a variant its image did
not prebuild, such as `mha_varlen_fwd`'s for a prefill step of one new token a
row, about 60 s inside the run, and the container keeps it for every later
server. A record whose server built one names it in run.json's
`aiter_jit_builds`, and grading refuses that record. This record takes those
builds; nothing reads it. `smoke_cell.sh` serves the same record before its
calibration, so a container it ran in is warm. Success: `serve` returns 0.

**1. Memory readings** (GPU, 2 to 3 min after step 0; 5 min as a fresh
container's first start, as in the smoke cell).

```bash
compass measure memory --gpu $G $M $S --out $R/memory -- $ATOM
```

Writes a `memory_readings` entry to the store. Success: exit 0 and a `published
memory_readings(...)` line.

**2. Calibration records** (GPU, alone on the node; per concurrency, its warmup
plus 1800 s, about 35 min with the server's start and stop: 1 h 45 min in all).

```bash
for C in 1 4 16; do serve $R/cal/c$C 256-392 $C $REAL_PY record --gpu $G || break; done
```

Writes per record `steps/dp0.jsonl` (the step record), `run.json` (with the
engine's bounds under `engine`) and `artifacts/client_run.json`. Success:
`client.sh` exits 0, and `run.json` and `artifacts/client_run.json` exist.

**3. Real runs** (GPU; per run about 35 min, as in step 2: 3 h 30 min on one GPU,
1 h 45 min on two).

```bash
for C in 1 4 16; do serve $R/cell/c$C/real1 0-255 $C $REAL_PY record --gpu $G || break; done
for C in 4 16 1; do serve $R/cell/c$C/real2 0-255 $C $REAL_PY record --gpu $G || break; done
```

Each graded run is a fresh server: a second client run would meet the prefix
cache the first left. The second loop takes the concurrencies in another order,
so on one GPU at least one run comes between the two runs of each concurrency,
and on two GPUs at once the two start at least one run apart; repeats run back
to back or together understate the real spread every metric is judged against.
On two GPUs, run the two loops at once, each with its own `$G`; no more than two
real runs at once, since real runs time the host. Success as in step 2.

**4. Fit** (CPU, under a minute each).

```bash
set -o pipefail   # a fit's exit status, not tee's
RUNS=$(for d in $R/cal/c*; do echo --client-run $d $d/artifacts/client_run.json; done)
compass fit steps --backend step-linear --form history $R/cal/c* $RUNS $M $S --out $R/fit/steps \
  2>&1 | tee $R/logs/fit_steps.log
compass fit host $R/cal/c* $RUNS $M $S --out $R/fit/host 2>&1 | tee $R/logs/fit_host.log
token() { awk '/^published / {t = $NF} END {print t}' "$1"; }
LAW=$(token $R/logs/fit_steps.log) HOST=$(token $R/logs/fit_host.log)
```

Each prints its token last, `published step_linear_law(...) as <token>` and
`published host_laws(...) as <token>`, which `LAW` and `HOST` keep. Success: exit
0, and `$R/fit/steps/step_linear_law.json` reads `"unfitted": {}`; a vector named
there refuses every step it prices.

**5. Simulated runs** (CPU, 2 to 4 min per concurrency).

```bash
for C in 1 4 16; do
  serve $R/cell/c$C/step_linear 0-255 $C $SIM_PY run --step-linear $LAW --host-laws $HOST $M $S || break
done
```

Writes the step record, `step_table.txt`, `run.json` (what grading reads) and
`client-0/summary.json`. Success: that summary reads `"coverage_report": false`.

**6. Grading** (developer check, CPU, seconds per cell, outside the `compass` CLI).

```bash
PYTHONPATH=$T python3 -m atom.compass.validate $R/cell/c1 $R/cell/c4 $R/cell/c16
```

Grade a cell once its `real1`, `real2` and `step_linear` runs are complete; a
cell short of one refuses in one line, such as `<cell> holds 1 of real*/; a cell
holds two or more real runs`. It prints one table per cell, a header
row and then the verdict: `graded`, or `coverage report` when the simulated run
refused any step; a run that cannot be graded at all, such as one fitted on the
traces it grades, refuses with the reason. `graded` says the cell could be
graded, not that every metric met its target; an `outside` row is a result.
The rows hold three results, read side by side and never as one number: cost
accuracy (`cost` rows, on each real run's own steps), schedule agreement
(`counts`, `distributions`) and end to end (`ttft`, `tpot`, throughput). Each
metric's verdict is `within target` (15% for TTFT, 10% otherwise), `inside the
real spread`, `outside`, or `ungradeable` when the real runs disagree by more than
the target. A graded row's value is the `spread`, the largest distance between
two real runs, and the `result`, the simulated run's distance to each real run:
`|a / b - 1|` end to end, the KS statistic for distributions, `|a - b|` for
counts. The `warmup` rows state each run's warmup requests and steps. The
protocol is [08_validation_protocol.md](../../atom/compass/design/08_validation_protocol.md).

Two more developer checks, the recurrence gate on each calibration record and
the step-for-step comparison with `cctraces_sim.sh`, are in the README's
[stage-2 recipe](README.md#the-stage-2-cell-through-the-compass-cli-from-an-empty-store).

Reference results: the graded tables of
[PR #1336](https://github.com/jgong5/ATOM/pull/1336) (this path) and
[PR #1547](https://github.com/jgong5/ATOM/pull/1547) (the step grid, below).

## The smoke cell

`smoke_cell.sh MODEL OUT` runs steps 0 to 6 at c=1 on one GPU, step 1 first:
step 0's record, one calibration record of traces 256 to 263, two real runs and
one simulated run of traces 0 to 7, each client run 600 s. It fits before the
real runs, so a failed fit costs no more GPU time. Its variables are in its
header. It may run on a busy node (GPUs are shared), and its real runs count
toward the two at once.

```bash
GPU=$G MACHINE=$SPEC HARNESS_PYTHON=$SIM_PY AIPERF_PYTHON=$REAL_PY \
  $T/scripts/compass/smoke_cell.sh $MODEL $R/smoke
```

It prints each step as it starts, then the grading table, then one line with the
verdict, the wall time and each run's warmup. It refuses a GPU that is not idle,
names the step that failed, and exits 1 unless the cell is graded. On node 18,
in a fresh container, it took 60 min: 5 min for step 1, 13 to 14 min per real
server, step 0's included, 2 min for the simulated run.

On these traces at c=1, no step in the first 600 s batches two requests, in
calibration or in the cell, so the law has a vector for every step the cell
takes. Its cost rows all read `unvalidated`, though: `fit steps` marks the
region it validates by holding out each calibration record in turn, so it needs
two or more, and the smoke cell has one. The smoke cell checks the path, not the
law's accuracy.

## Traps

| Symptom | Cause | Fix |
|---|---|---|
| `<tree>/atom is neither a git checkout nor a stamped snapshot` | a git worktree in a root container fails git's ownership check | use a `snapshot.sh` tree (this page), or `git config --global safe.directory <tree>` in the container |
| grading or a check imports `/app/ATOM` | `python -m` without `PYTHONPATH=$T`, or run from inside a checkout | `PYTHONPATH=$T`, from `/` |
| `measure memory`: the cards read ptl `enabled`, and the spec states `disabled` | another node's machine spec, such as `mi308x-hjbog39.json` on node 18 | this node's spec (Environment) |
| a GPU command refuses: `GPU N has PID P on it before the run` | the GPU is not idle | another GPU; never stop another user's work |
| `run` refuses its law or host laws | records, fits and runs from two trees, two `$ATOM`s, two specs or two environments | one tree, one `$ATOM`, one spec and one environment (Variables) for every step |
| `... holds an earlier run's files`, or `client.sh` refuses its OUT | one server per `--out`, one client run per server | a fresh `--out` for every server |
| `fit` refuses `... is not a record directory (no run.json)` | a log file matched `$R/cal/c*` | keep logs outside the record directories |
| `fit` refuses: `[Errno 2] No such file or directory: '<record>/artifacts/client_run.json'` | a stopped calibration record under `$R/cal/` | `rm -rf` it and take it again |
| `kill: %%: no such job` | `serve` already stopped that server, when its client run failed or ended | nothing to stop; run `jobs` first |
| `serve: OUT exists; rm -rf it to take the step again` | a stopped step's `--out` | `rm -rf` it, as the line says |
| `Triton cache error`, or `failed to map segment from shared object` | a `--tmpfs` without `exec` | the container as Environment starts it |
| grading, a fit or `validate.cost` refuses a record: `aiter ran a JIT build of [...] while its server served` | the container met that kernel variant for the first time inside the record | take the record again, since the container now holds the build; warm a fresh container first (step 0) |
| verdict `coverage report`, `no decode vector for rung N` | the cell decoded more requests at once than any calibration step did (a rung is the batch's captured-graph size) | calibrate at that concurrency or above; #1336's calibration reached no rung 48, so c=256 stayed a coverage report |
| grading refuses a run: warmup | the client ran with `--no-warmup`, or its warmup overlapped its graded steps | keep `client.sh`'s default warmup |
| grading refuses a run: fitted on the traces it grades | calibration and cell traces overlap | calibration 256 to 392, cell 0 to 255 |
| disk full | raw exports keep every prompt, and caches with no tmpfs fill the container's layer | the data directory where Disk puts it, and the container's tmpfs mounts (Environment) |

## Appendix: the step grid

The step grid prices each step from points measured on a grid of batch shapes
instead of a linear law; the README's
[step grid section](README.md#the-same-cell-under-the-step-grid) takes this cell
through it.
