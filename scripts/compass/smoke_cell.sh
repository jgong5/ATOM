#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# QUICKSTART.md's path end to end at concurrency 1 on one GPU, in about an hour:
# `compass measure memory`; a calibration `compass record` serving cc-traces 353
# on; `compass fit steps` (step linear, history form) and `compass fit host` on
# it; two real `compass record` runs and one `compass run`, serving traces 0 on;
# then the cell graded by `python -m atom.compass.validate`. Each server serves
# one `client.sh` run, with its warmup, which stops it. The fits run before the
# real runs, so a failed fit costs no more GPU time.
# Before calibration, a record nothing reads, warm/, serves the cell's traces
# once, so that it takes the aiter JIT builds a fresh container meets while
# serving, which grading, the fits and `validate.cost` refuse in a record
# (run.json's `aiter_jit_builds`).
# Calibration serves traces 353 on: at client.sh's seed, its sampled start reaches
# subagents and batches 1 to 7 requests a step, so the law has a decode vector for
# each batch size the cell takes; traces 256 on start at a first request.
#
# Usage: smoke_cell.sh MODEL OUT
#   MODEL           the model directory: config.json, tokenizer and weights
#   OUT             an absent or empty directory, on a disk with room: the raw
#                   exports keep every prompt
#   GPU             the one GPU, as rocm-smi and `--gpu` number it (required);
#                   it must read 0% use and 0% VRAM
#   MACHINE         the machine spec of this node's GPUs (required)
#   TRACES          a cc-traces traces.jsonl (required, as client.sh takes it)
#   HARNESS_PYTHON  the Python with agentx-harness and compass-harness, for the
#                   simulated run's client (required)
#   AIPERF_PYTHON   the Python with agentx-harness and no compass-harness, for
#                   the real runs' clients (required)
#   N_TRACES        traces each client is given (default 8): 353 on for
#                   calibration, 0 on for the cell
#   DURATION_S      each client run's benchmark duration (default 600)
#   SERVER_ARGS     ATOM's arguments after --model (default: --max-model-len
#                   262144)
#   WAIT_S          seconds a server has to start (default 1800)
#
# OUT gets each step's directory and log (<step>.log, and <step>.client.log for
# a client run), store/, and cell/ with real1/, real2/ and step_linear/. The
# grading table, grade.log, is printed last, then a line with the cell's
# verdict, each run's warmup and the wall time. It exits 1 naming the step that
# failed, or when the cell is not graded.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
compass_describe "$ROOT"
MODEL=$(realpath "${1:?usage: smoke_cell.sh MODEL OUT}")
OUT=${2:?usage: smoke_cell.sh MODEL OUT}
MACHINE=$(realpath "${MACHINE:?MACHINE names the machine spec of the GPUs}")
: "${TRACES:?TRACES names a cc-traces traces.jsonl}"
: "${HARNESS_PYTHON:?HARNESS_PYTHON names the Python with compass-harness}"
: "${AIPERF_PYTHON:?AIPERF_PYTHON names the Python without compass-harness}"
N=${N_TRACES:-8} DURATION_S=${DURATION_S:-600} WAIT_S=${WAIT_S:-1800}
# shellcheck disable=SC2206 # SERVER_ARGS is a word list
ATOM_ARGS=(--model "$MODEL" ${SERVER_ARGS:---max-model-len 262144})
CLI='from atom.compass.cli import main; main()'
export MODEL TRACES
mkdir -p "$OUT" && OUT=$(realpath "$OUT")
[[ -z $(ls -A "$OUT") ]] || { echo "smoke_cell: $OUT is not empty" >&2; exit 2; }
# Every path is absolute from here, and nothing resolves `atom` from $PWD.
cd / || exit 1
start=$SECONDS
# Any exit, a failed step's or a SIGTERM's, stops what is still running: each
# step and client run runs in the background, as a job this trap reaches.
trap 'kill -TERM $(jobs -p) 2>/dev/null; wait' EXIT
fail() { echo "smoke_cell: step $1 failed: $2" >&2; exit 1; }
say() { echo "smoke_cell: $1 $(date -u +%FT%TZ)"; }

# The compass commands check each start for processes on the GPU; this checks
# its use and VRAM too, once, before the first.
[[ ${GPU-} =~ ^[0-9]+$ ]] || fail gpu "GPU names one GPU as rocm-smi numbers it, not '${GPU-}'"
busy=$(rocm-smi --showuse --showmemuse | awk -v g="GPU[$GPU]" 'index($0, g) == 1 && /use \(%\)|VRAM%/ && $NF != "0"')
[[ -z $busy ]] || fail gpu "rocm-smi GPU $GPU is not idle: $busy"
compass_head_sha "$ROOT" >/dev/null ||
    fail tree "$ROOT is neither a git checkout nor a snapshot.sh tree, and a fit refuses it"

# step NAME CMD...: run CMD, its output in OUT/NAME.log.
step() {
    local name=$1
    shift
    say "$name"
    "$@" >"$OUT/$name.log" 2>&1 &
    wait $! || fail "$name" "exit $?; see $OUT/$name.log"
}

# serve NAME DIR A-B PYTHON COMMAND...: one compass server writing to DIR, one
# client run of traces A to B under PYTHON, which stops the server.
serve() {
    local name=$1 dir=$2 traces=$3 python=$4 pid deadline
    shift 4
    say "$name"
    python3 -c "$CLI" "$@" --out "$dir" -- "${ATOM_ARGS[@]}" >"$OUT/$name.log" 2>&1 &
    pid=$!
    deadline=$((SECONDS + WAIT_S))
    until [[ -s $dir/server.json ]]; do
        kill -0 $pid 2>/dev/null || fail "$name" "its server did not start; see $OUT/$name.log"
        ((SECONDS < deadline)) || fail "$name" "no server after $WAIT_S s"
        sleep 1
    done
    HARNESS_PYTHON=$python "$ROOT/scripts/compass/client.sh" "$dir" --traces "$traces" \
        --concurrency 1 --duration "$DURATION_S" --stop >"$OUT/$name.client.log" 2>&1 &
    wait $! || fail "$name" "client.sh exit $?; see $OUT/$name.client.log"
    wait $pid || fail "$name" "its server exit $?; see $OUT/$name.log"
}

# token LOG: the store token a fit's LOG ends with.
token() { awk '/^published / {t = $NF} END {print t}' "$1"; }

M=(--machine "$MACHINE" --store "$OUT/store")
CAL=353-$((352 + N)) CELL=0-$((N - 1))
step memory python3 -c "$CLI" measure memory --gpu "$GPU" "${M[@]}" --out "$OUT/memory" -- "${ATOM_ARGS[@]}"
serve warm "$OUT/warm" "$CELL" "$AIPERF_PYTHON" record --gpu "$GPU"
serve calibration "$OUT/calibration" "$CAL" "$AIPERF_PYTHON" record --gpu "$GPU"
RUNS=(--client-run "$OUT/calibration" "$OUT/calibration/artifacts/client_run.json")
step fit_steps python3 -c "$CLI" fit steps --backend step-linear --form history \
    "$OUT/calibration" "${RUNS[@]}" "${M[@]}" --out "$OUT/fit_steps"
step fit_host python3 -c "$CLI" fit host "$OUT/calibration" "${RUNS[@]}" "${M[@]}" --out "$OUT/fit_host"
LAW=$(token "$OUT/fit_steps.log") HOST=$(token "$OUT/fit_host.log")
for k in 1 2; do
    serve real$k "$OUT/cell/real$k" "$CELL" "$AIPERF_PYTHON" record --gpu "$GPU"
done
serve step_linear "$OUT/cell/step_linear" "$CELL" "$HARNESS_PYTHON" \
    run --step-linear "$LAW" --host-laws "$HOST" "${M[@]}"
step grade python3 -m atom.compass.validate "$OUT/cell"

cat "$OUT/grade.log"
verdict=$(awk -F' [|] ' '$3 == "verdict" {print $5}' "$OUT/grade.log")
warmup=$(awk -F' [|] ' '$3 == "warmup" {sub(/^warmup /, "", $2); printf "%s%s %s", s, $2, $4; s = "; "}' "$OUT/grade.log")
echo "smoke_cell: $verdict in $(((SECONDS - start) / 60)) min; warmup: $warmup"
[[ $verdict == graded ]] || fail grade "the cell is $verdict, not graded"
