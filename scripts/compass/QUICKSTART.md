<!-- SPDX-License-Identifier: MIT -->
# Compass quickstart: simulate a model and grade it against real runs

This page takes one model from an empty store to a graded simulation. The example
is the cell Compass is graded on: Qwen3.8-27B, TP1, on one MI308X, driven by
cc-traces agentic sessions. `smoke_cell.sh` runs the same path at one concurrency
in about an hour, so start there (below), then run the full cell.

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

- **Image.** `rocm/atom-dev:vllm-v0.31.0-nightly_20261013` (tagged
  `compass-next:nightly-20261013` on node 18). The machine specs were measured in
  it, and a fit refuses records from another stack than its machine spec states.
- **One GPU container**, named for its owner and purpose, with the GPU devices,
  host networking, the inputs mounted read-only and a data directory with room:

  ```bash
  docker run -d --name <user>_<issue>_gpu --network host \
    --device /dev/kfd --device /dev/dri \
    --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
    -v <model repo>:/models/qwen38:ro -v <traces dir>:/cc:ro -v <data dir>:<data dir> \
    rocm/atom-dev:vllm-v0.31.0-nightly_20261013 sleep infinity
  ```

- **GPUs are shared.** Use a GPU that `rocm-smi --showuse --showmemuse` reads at 0%
  use and 0% VRAM, with no process in `rocm-smi --showpids --showpidgpus`. Every
  GPU command checks this before and after it runs and refuses a busy GPU. Never
  stop another user's process or container. Real runs time the host too, so run
  calibration alone on the node and at most two real runs at once.
- **One tree.** Stage it on the host with `scripts/compass/snapshot.sh DIR`, then
  extract the tarball into the data directory: that extracted tree is `$T`. Run
  every step from it. Records, fits and runs keyed by another tree's source refuse
  each other. A git worktree used in a root container fails git's ownership check,
  so a fit refuses it as "neither a git checkout nor a stamped snapshot"; a
  snapshot avoids that.
- **Inputs you supply:** a model directory; a cc-traces `traces.jsonl` (sha256
  `e39cd2ff3eba...`, 568,864,747 bytes); and two client Pythons. A real server's
  client is the image's `/opt/venv/bin/python3` (agentx-harness's aiperf). A
  simulated server's client also needs compass-harness:

  ```bash
  /opt/venv/bin/python3 -m venv --system-site-packages $V
  echo 'import site; site.addsitedir("/opt/venv/lib/python3.12/site-packages")' \
    > $V/lib/python3.12/site-packages/zz_opt_venv.pth
  cp -r $T/compass_harness /tmp/ch_src && $V/bin/pip install --no-deps --no-build-isolation /tmp/ch_src
  ```

  On node 18, the Inputs section of `.claude/agent-team.local.md` in the owner's
  checkout lists where each input lives.
- **Disk.** A run takes about 400 MB, since the raw export keeps every prompt; the
  full cell about 8 GB. Keep data off the root disk.

## Variables

Run from `/` in the container, so no checkout's `atom` shadows `$T`'s.

| Name | What | Example |
|---|---|---|
| `T` | the snapshot tree | `/data/<user>/compass/tree/<sha>/ATOM` |
| `R` | this cell's data directory | `/data/<user>/compass/qwen38-tp1` |
| `G` | the GPU, as `rocm-smi` numbers it | `7` |
| `MODEL` | the model directory; `client.sh` reads it too | `/models/qwen38/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `TRACES` | the cc-traces file; `client.sh` reads it | `/cc/traces.jsonl` |
| `ATOM` | ATOM's arguments, identical for every step | `--model $MODEL --max-model-len 262144` |
| `M` | the machine spec of this node's GPUs | `--machine $T/scripts/compass/machines/mi308x-hjbog18.json` |
| `S` | the store | `--store $R/store` |
| `REAL_PY`, `SIM_PY` | the client Pythons, as above | `/opt/venv/bin/python3`, `$V/bin/python` |
| `compass` | the CLI, run from its own tree; the image does not install it | `compass() { $T/scripts/compass/compass "$@"; }` |

`export MODEL TRACES`. Every `python3 -m atom.compass...` below runs with
`PYTHONPATH=$T`: without it, `atom` is the image's own copy in `/app/ATOM`.

One server and its client run, used by steps 2, 3 and 5:

```bash
# serve OUT TRACES C PYTHON COMMAND...: start a compass server writing to OUT,
# send it one client run, and stop it. Its log goes under $R/logs.
serve() {
  local out=$1 traces=$2 c=$3 py=$4 log=$R/logs/${1#$R/}.log; shift 4
  mkdir -p ${log%/*} && compass "$@" --out $out -- $ATOM > $log 2>&1 &
  until [ -s $out/server.json ]; do sleep 5; done   # if it never appears, read $log
  HARNESS_PYTHON=$py $T/scripts/compass/client.sh $out --traces $traces \
    --concurrency $c --duration 1800 --stop
}
```

`client.sh`'s flags and exit codes are in its header. It sends one warmup request
per lane before the graded window; keep it, since grading refuses a run whose
warmup is missing or overlaps its graded steps.

## The path

Times were measured on node 18 for the cell in
[PR #1336](https://github.com/jgong5/ATOM/pull/1336), at concurrencies 1, 4, 16,
64 and 256 with a 1800 s benchmark each.

**1. Memory readings** (GPU, about 3 min).

```bash
compass measure memory --gpu $G $M $S --out $R/memory -- $ATOM
```

Writes a `memory_readings` entry to the store. Success: exit 0 and a `published
memory_readings(...)` line.

**2. Calibration records** (GPU, alone on the node; 3 h 45 min for all five).

```bash
for C in 1 4 16 64 256; do serve $R/cal/c$C 256-392 $C $REAL_PY record --gpu $G; done
```

Writes per record `steps/dp0.jsonl` (the step record), `run.json` (with the
engine's bounds under `engine`) and `artifacts/client_run.json`. Success:
`client.sh` exits 0 and both files exist. Each concurrency spends its warmup
plus 1800 s; the warmup grows to about 36 min at c=256.

**3. Real runs** (GPU, at most two at once; 3 h 20 min for ten on two GPUs).

```bash
for C in 1 4 16 64 256; do for k in 1 2; do
  serve $R/cell/c$C/real$k 0-255 $C $REAL_PY record --gpu $G
done; done
```

Each graded run is a fresh server: a second client run would meet the prefix
cache the first left. Space the two runs of one concurrency apart; back-to-back
repeats understate the real spread every metric is judged against. Success as
in step 2.

**4. Fit** (CPU, under a minute each).

```bash
RUNS=$(for d in $R/cal/c*; do echo --client-run $d $d/artifacts/client_run.json; done)
compass fit steps --backend step-linear --form history $R/cal/c* $RUNS $M $S --out $R/fit/steps
compass fit host $R/cal/c* $RUNS $M $S --out $R/fit/host
```

Each prints its token: `published step_linear_law(...) as <LAW>` and `published
host_laws(...) as <HOST>`. Success: exit 0, and `$R/fit/steps/step_linear_law.json`
reads `"unfitted": {}`; a vector named there refuses every step it prices.

**5. Simulated runs** (CPU, 2 to 4 min per concurrency).

```bash
for C in 1 4 16 64 256; do
  serve $R/cell/c$C/step_linear 0-255 $C $SIM_PY run --step-linear <LAW> --host-laws <HOST> $M $S
done
```

Writes the step record, `step_table.txt`, `run.json` (what grading reads) and
`client-0/summary.json`. Success: that summary reads `"coverage_report": false`.

**6. Grading** (developer check, CPU, outside the `compass` CLI).

```bash
PYTHONPATH=$T python3 -m atom.compass.validate $R/cell/c1 $R/cell/c4 ...
```

Prints one table per cell. Its first row is the verdict: `graded`, or `coverage
report` when the simulated run refused any step; a run that cannot be graded at
all, such as one fitted on the traces it grades, refuses with the reason.
The rows hold three results, read side by side and never as one number: cost
accuracy (`cost` rows, on each real run's own steps), schedule agreement
(`counts`, `distributions`) and end to end (`ttft`, `tpot`, throughput). Each
metric's verdict is `within target` (15% for TTFT, 10% otherwise), `inside the
real spread`, `outside`, or `ungradeable` when the real runs disagree by more than
the target. The `warmup` rows state each run's warmup requests and steps. The
protocol is [08_validation_protocol.md](../../atom/compass/design/08_validation_protocol.md).

Two more developer checks, the recurrence gate on each calibration record and
the step-for-step comparison with `cctraces_sim.sh`, are in the README's
[stage-2 recipe](README.md#the-stage-2-cell-through-the-compass-cli-from-an-empty-store).

Reference results: the graded tables of
[PR #1336](https://github.com/jgong5/ATOM/pull/1336) (this path) and
[PR #1547](https://github.com/jgong5/ATOM/pull/1547) (the step grid, below).

## The smoke cell

`smoke_cell.sh MODEL OUT` runs steps 1 to 6 at c=1 on one GPU: one calibration
record of traces 256 to 263, two real runs and one simulated run of traces 0 to
7, each client run 600 s. It fits before the real runs, so a failed fit costs no
more GPU time. Its variables are in its header.

```bash
HIP_VISIBLE_DEVICES=$G MACHINE=$T/scripts/compass/machines/mi308x-hjbog18.json \
  HARNESS_PYTHON=$SIM_PY AIPERF_PYTHON=$REAL_PY \
  $T/scripts/compass/smoke_cell.sh $MODEL $R/smoke
```

It prints each step as it starts, then the grading table, then one line with the
verdict, the wall time and each run's warmup. It refuses a GPU that is not idle,
names the step that failed, and exits 1 unless the cell is graded. On these
traces at c=1, no step in the first 600 s batches two requests, in calibration or
in the cell, so the law covers every step the cell takes.

## Traps

| Symptom | Cause | Fix |
|---|---|---|
| `<tree>/atom is neither a git checkout nor a stamped snapshot` | a git worktree in a root container fails git's ownership check | use a `snapshot.sh` tree (this page), or `git config --global safe.directory <tree>` in the container |
| grading or a check imports `/app/ATOM` | `python -m` without `PYTHONPATH=$T`, or run from inside a checkout | `PYTHONPATH=$T`, from `/` |
| `measure memory`: the cards read ptl `enabled`, and the spec states `disabled` | a machine spec for other cards | the spec of this node's cards, in `scripts/compass/machines/` |
| a GPU command refuses: `GPU N has PID P on it before the run` | the GPU is not idle | another GPU; never stop another user's work |
| `run` refuses its law or host laws | records, fits and runs from two trees, two `$ATOM`s or two specs | one tree, one `$ATOM`, one spec for every step |
| `... holds an earlier run's files`, or `client.sh` refuses its OUT | one server per `--out`, one client run per server | a fresh `--out` for every server |
| `fit` refuses `... is not a record directory (no run.json)` | a log file matched `$R/cal/c*` | keep logs outside the record directories |
| a step of about 60 s early in a real run | aiter builds a kernel on the container's first short prefill over a long history ([#1319](https://github.com/jgong5/ATOM/issues/1319)) | take graded records in a container that has served one before |
| verdict `coverage report`, `no decode vector for rung N` | the cell decoded more requests at once than any calibration step did (a rung is the batch's captured-graph size) | calibrate at that concurrency or above; #1336's calibration reached no rung 48, so c=256 stayed a coverage report |
| grading refuses a run: warmup | the client ran with `--no-warmup`, or its warmup overlapped its graded steps | keep `client.sh`'s default warmup |
| grading refuses a run: fitted on the traces it grades | calibration and cell traces overlap | calibration 256 to 392, cell 0 to 255 |
| disk full | raw exports keep every prompt | data under a large data disk |

## Appendix: the step grid

The step grid prices a step from measured points on a grid of batch shapes
instead of a linear law. It reuses steps 1 to 3 and the host laws of step 4;
the README's
[step grid section](README.md#the-same-cell-under-the-step-grid) has every
argument.

1. **Shape populations** (CPU). Per concurrency, `compass trace steps
   --step-linear <LAW> --host-laws <HOST> $M $S --out $R/pop/<traces>/c$C --
   $ATOM`, served one client run like a `run` server (`$SIM_PY`), once over
   traces 256 to 392 and once over 0 to 255. Its stop publishes a
   `shape_population` token. Plan from populations, not records: two real starts
   differ by a few KV blocks, which the planner refuses
   ([#1349](https://github.com/jgong5/ATOM/issues/1349)).
2. **Measure the grid** (GPU, hours: 412 points took 10 h on three GPUs).
   `compass measure steps --grid <POP>... --grid-definition <G.json> --client-run
   ... --sampling '{"ignore_eos": true}' --gpu $G $S --out $R/grid -- $ATOM`. The
   grid definition holds the nodes, powers of two, and a margin of 0.
3. **Fit** (CPU). `compass fit steps --backend step-grid --grid <POP>...
   --grid-definition <G.json> $R/cal/c* $RUNS $M $S --out $R/fit-grid` prints the
   bundle's token. Refit into a fresh store: a second fit of one definition into
   one store is refused as an overwrite.
4. **Run** (CPU). `compass run --step-grid <BUNDLE> --host-laws <HOST> --coverage
   $R/cell/c$C/real* $M $S --out <cell>/step_grid -- $ATOM`, served as in step 5.
   A cell holds one simulated run, so put it in its own cell directory beside
   links to `real1` and `real2`. Grading takes only a run whose coverage check ran.
5. **Grade** that cell as in step 6.
