<!-- SPDX-License-Identifier: MIT -->
# Compass for users: predict one configuration with no GPU

This page predicts how one configuration serves an agentic workload, on a host with
no GPU. ATOM's scheduler, KV block manager and API server run unchanged on a
simulated clock, and each forward step costs what a committed law file predicts.
Nothing is recorded, fitted or graded. The configuration is DeepSeek-V4-Pro served
1P1D on two nodes of eight MI355X each, crsuse2-m2m-v3-005 (prefill) and -013
(decode), each side DP attention over its eight cards: `P3` of the template
[spaces/deepseek-v4-pro-mi355x-crs005.json](spaces/deepseek-v4-pro-mi355x-crs005.json).

[QUICKSTART.md](QUICKSTART.md) is the developer's path, which fits such laws on a GPU
and grades a simulation against real runs; [ADVICE.md](ADVICE.md) searches several
configurations for the one to deploy.

## What the prediction rests on

| Input | Where | What it holds |
|---|---|---|
| step and host laws | `laws/deepseek-v4-pro-dpa8-mi355x-crs005-{prefill,decode}.json.xz` | each side's forward-step cost and the host CPU time around each step, fitted on real runs of that side ([README](README.md#the-step-linear-law-from-real-cells--step_linear_fit)) |
| memory readings | the same files, as `memory_readings` | what each side's engine start read off its cards, which sizes its KV pool |
| platform readings | `laws/platform_readings/` | the cards' measured peaks at the laws' software stack |
| datasheet and link | `atom/compass/devices/mi355x.json`, `atom/compass/networks/crs005-013-ionic.json` | the card, and the inter-node link the KV moves over |

You choose the traces, the concurrency and the duration. The model, ATOM's arguments
and environment, the device and the tree are fixed: the laws and readings are keyed
by them, and a run under another value is refused, naming it. Another configuration
needs its own laws, which take its GPUs (ADVICE.md, step 3).

## Environment

- **A container with no GPU device**, from the image the laws' stack was measured in,
  with your data directory mounted at its own path and an empty directory over
  `/sys/class/kfd`, so a host's GPUs stay out of it as well
  ([README](README.md#the-cpu-gate-container)):

  ```bash
  # host
  mkdir -p <data dir>/empty_kfd
  docker run -d --init --name ${USER}_compass --network host \
    -v <data dir>:<data dir> -v <data dir>/empty_kfd:/sys/class/kfd:ro \
    rocm/atom-dev:vllm-v0.31.0-nightly_20261026 sleep infinity
  docker exec -it ${USER}_compass bash
  ```

  Every other block runs in that container, as root.
- **CPUs.** The deployment runs two processes per DP rank, its engine core
  (`ATOM::EngineCore_DP<n>`) and its runner (`ATOM::DP<n>TP0`), 32 in all, beside the
  clock authority, two API servers, the router and the client. The times below were
  measured in a container of 16 cores of an AMD EPYC 9575F host, on 2026-11-19.
- **Disk.** About 700 MB for the tree, the inputs and the client's Python, and 1.6 GB
  for the run below, 330 MB of it the client's copy of the traces.

## Set up, once

```bash
DATA=<data dir>
git clone --depth 1 --branch feature/atomcompass_new https://github.com/jgong5/ATOM $DATA/ATOM
hf download deepseek-ai/DeepSeek-V4-Pro --revision b5968e9190ef611bbf34a7229255be88a0e937c1 \
  --exclude '*.safetensors' --local-dir $DATA/DeepSeek-V4-Pro
hf download semianalysisai/cc-traces-weka-062126-256k traces.jsonl --repo-type dataset \
  --revision 8fecd2fc56694469f758f0afbbb6335ad3043740 --local-dir $DATA/cc-traces
rm -rf /tmp/ch_src && cp -r $DATA/ATOM/compass_harness /tmp/ch_src
/opt/venv/bin/python3 -m venv --system-site-packages $DATA/venv
echo 'import site; site.addsitedir("/opt/venv/lib/python3.12/site-packages")' \
  > $DATA/venv/lib/python3.12/site-packages/zz_opt_venv.pth
$DATA/venv/bin/pip install --no-deps --no-build-isolation /tmp/ch_src
```

- **The tree** is a clone of the integration branch, made in the container so that
  git there owns it. Keep it clean: the laws load only at the ATOM source their host
  key names, which `tests/compass/test_platform_keys.py` holds every commit of the
  branch to.
- **The model** is its configuration and tokenizer, no weights: a simulated engine
  builds the model on meta tensors and reads no checkpoint. The revision is the one
  whose `config.json` the laws and readings are keyed by.
- **The traces** are cc-traces' agentic sessions, 569 MB.
- **The client's Python** is the image's, with agentx-harness's aiperf, plus the
  tree's `compass-harness`, which paces aiperf on the simulated clock.

## Predict

In each shell that predicts:

```bash
DATA=<data dir>
T=$DATA/ATOM
OUT=$DATA/run-c16                        # the run's directory, absent or empty
LAW=$T/scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005
RUN=(--step-linear $LAW-prefill.json.xz --step-linear $LAW-decode.json.xz
  --host-laws $LAW-prefill.json.xz --host-laws $LAW-decode.json.xz
  --device mi355x --ptl disabled --network crs005-013-ionic --store $DATA/store)
ARGS=(--model $DATA/DeepSeek-V4-Pro --tensor-parallel-size 8 --kv-cache-dtype fp8
  --enable-prefix-caching --max-num-seqs 512 --trust-remote-code --enable-dp-attention)
PREFILL=("${ARGS[@]}" --gpu-memory-utilization 0.72 --enable-tbo)
DECODE=("${ARGS[@]}" --gpu-memory-utilization 0.75 --cudagraph-capture-sizes "[1,2,$(seq -s, 4 4 256)]")
export AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1
export MODEL=$DATA/DeepSeek-V4-Pro TRACES=$DATA/cc-traces/traces.jsonl HARNESS_PYTHON=$DATA/venv/bin/python
cd /
```

`RUN` is `compass run`'s flags. `--store` names a store that holds nothing and need
not exist: the run reads the platform readings from `laws/platform_readings/` and
each side's memory readings from its law file, where a store holds none. Each side's
ATOM arguments and variables are `P3`'s as `compass advice` expands the template;
`GPU_MAX_HW_QUEUES` and `ATOM_NUMA_BIND`, below, are the prefill side's alone.

**1. Check each side** (seconds; no server starts):

```bash
GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1 $T/scripts/compass/compass run "${RUN[@]}" --check prefill -- "${PREFILL[@]}"
$T/scripts/compass/compass run "${RUN[@]}" --check decode -- "${DECODE[@]}"
```

Each lists what its side's run lacks, with the command that makes it, and exits 1
when anything is lacking. The prefill side lacks nothing: a prefill-decode producer
schedules no decode rung (`lacking.uncovered_rungs`). The decode side lacks one thing,
`step linear law: stale (coverage)`: no decode vector for rungs 32 up and eager,
each a captured batch size or eager, that this configuration schedules. Its
calibration recorded no decode step at those rungs but eager ones, which the fit
leaves out, as each ran beside a DP peer's prefill. A step at such a rung is
refused, which the run reports; at concurrency 16 none is. Any other line is a trap
(below).

**2. Run** (11 min: a minute to start, then 1875 simulated seconds in 490 wall seconds):

```bash
GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1 $T/scripts/compass/deploy.sh $OUT run "${RUN[@]}" \
  --prefill -- "${PREFILL[@]}" \
  --decode --on "env -u GPU_MAX_HW_QUEUES -u ATOM_NUMA_BIND" -- "${DECODE[@]}" \
  --router -- --backend atom --disable-health-check --disable-circuit-breaker --disable-retries \
  > $OUT.log 2>&1 &
DEPLOY=$!
until [ -s $OUT/server.json ]; do kill -0 $DEPLOY || break; sleep 5; done
$T/scripts/compass/client.sh $OUT --traces 0-255 --concurrency 16 --stop
wait $DEPLOY; echo "deploy.sh exit $?"
```

- **`deploy.sh run`** starts each side as a `compass run`, both on one simulated
  clock, and the router between them.
- **The decode's `--on` prefix** starts that side apart, without the prefill side's
  `GPU_MAX_HW_QUEUES` and `ATOM_NUMA_BIND`, as its law is keyed, and makes the KV link
  `inter_node`, priced by `--network`, as the real pair spans two nodes.
- **The router's flags** switch its wall-clock detectors off, which a simulated
  router needs (`deploy.sh`'s header).
- **`client.sh`** replays traces 0 to 255 at concurrency 16 for 1800 s on the
  simulated clock, its default `--duration`, after aiperf's scenario warmup, then
  stops the deployment. Its flags are in its header.

It ends `deploy.sh exit 0` once the deployment has written its report.

## Read the prediction

Open `$OUT/report.html` in a browser, from the data directory on the host: the run's
arguments and law sources, the headline metrics (throughput per GPU, interactivity,
TTFT, TPOT, hit rates), the trust table, and each request's split into queue,
prefill, KV transfer and decode.
The trust table's figures are in `$OUT/summary.json`:

```bash
jq '{coverage_report} + (.schedule | {seconds_by_method, treatments, refusals: .refusals.by_source})' $OUT/summary.json
jq -r '(.memory.readings[][], .memory.predicts[]?) | [.name, .method, .nbytes] | @tsv' $OUT/lp-engine-P.dp0.json
```

- **`seconds_by_method`** splits the predicted seconds by how they were priced:
  `estimated`, by a fitted vector inside the region its fit validated by holding out
  each record; `unvalidated`, by a fitted vector outside that region; `declared`,
  below. A step outside a vector's fitted hull is priced by it all the same, and
  counted among the trust table's steps extrapolated. The prefill law's records
  ran before ATOM's upstream merge of `b7b5ec3f2`
  ([#3936](https://github.com/jgong5/ATOM/pull/3936)). Since it, a DP rank with no
  request forwards an empty batch at its group's phase, which no longer vetoes a
  two-batch-overlap split, so most prefill steps split, a shape few records held.
  That idle forward is priced by the prefill vector beside a peer's prefill. Both
  read `unvalidated` until the law is refitted on records of this ATOM.
- **`treatments`** names what the steps assume: `uniform routing`, every step's MoE
  priced as if routing spread tokens evenly over the experts, and the treatments the
  table below names.
- **Refusals.** A host call outside the range its law was fitted on, such as an HTTP
  body larger than any the calibration sent, is charged 0 s and named in the report's
  refusal table, as is a step at a rung its law has no vector for. Any refusal makes
  the run a coverage report (`coverage_report: true`).

### What the run reports `declared`, and why

A `declared` term is stated, not fitted on this configuration's records.

| Term | Where | Why |
|---|---|---|
| `decode_beside_prefill`, treatment `eager decode beside prefill` | the prefill side's steps that prefill nothing, 0 s | a decode that replays no graph while a DP peer prefills ends before that peer; unchecked |
| `load residue`, `persistent`, `activations`, `driver and collective reserve`, the graph `pool` | each engine's `memory`, in `$OUT/lp-engine-<P or D>.dp<rank>.json` | the readings the law file carries, stating `assumed`: `compass measure memory` on crsuse2-m2m-v3-005 for [#3498](https://github.com/jgong5/ATOM/issues/3498), at an earlier ATOM digest; a host with no GPU cannot start the engine to take its own. `prefill/run.json` and `decode/run.json` state that note at `.provenance.memory_readings_assumed` |
| `0.2 x peak activations` | each engine's `memory`, as `cudagraph_overhead` | ATOM's coefficient for the CUDA graph overhead it reserves (`ModelRunner._estimate_cudagraph_overhead`), times the `activations` reading above, whose method it takes (`atom/compass/memory/graph_pool.py`: `reserves`); the law carries that reading `assumed` |

A prefill side that decodes at rung 2 also prices it by the decode law's rung-2
vector, `declared`, since the prefill side's records held too few such steps
([#2660](https://github.com/jgong5/ATOM/issues/2660)).

## Traps

| Symptom | Cause | Fix |
|---|---|---|
| a check's `lacking host laws: stale`, `refused: the run file's host laws were fitted under another ['atom']` | the tree's ATOM source is not the one the laws are keyed at, as after an edit to its Python | a clean clone of the branch (Set up) |
| a check's `lacking step linear law: stale` and `lacking host laws: stale`, `refused: no ... entry was fitted under this engine config`, naming fields such as `env.GPU_MAX_HW_QUEUES` | an argument or variable not as above; for those two, the decode side given the prefill side's | the variables and arrays as above, the decode started through the `--on` prefix |
| a check's `lacking memory readings: missing`, `refused: ... no entry for memory_readings(...)` | an argument the readings are keyed by, such as `--max-num-seqs`, not as above | `P3`'s arguments; another value takes its own readings, on the target's GPUs, by the `made by:` command |
| `$OUT.log`: `deploy: OUT holds an earlier deployment's files` | `$OUT` holds a finished or stopped run | a fresh `OUT` |
