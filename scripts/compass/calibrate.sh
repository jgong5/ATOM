#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# QUICKSTART.md's calibration in one command, on one GPU: a warm record nothing
# reads, as its step 0 serves, so the aiter JIT builds a fresh container meets
# fall outside the records the fits read; `compass measure memory`; one
# `compass record` per RECORD, each serving one `client.sh` run of the
# calibration traces, which stops it; then `compass fit steps` (step linear,
# history form) and `compass fit host` on those records. It plans nothing: what
# to record is its input. A decode rung N comes from a record at concurrency N
# that starts every trajectory at its first turn
# (https://github.com/jgong5/ATOM/issues/1840).
#
# Usage: calibrate.sh OUT RECORD...
#   OUT             an absent or empty directory, or one an earlier run given
#                   the same inputs left, which this run resumes
#   RECORD          one record: its concurrency, then the client's scenario,
#                   aiperf arguments client.sh takes as AIPERF_EXTRA_ARGS, if
#                   any, such as "8 --trajectory-start-min-ratio 0
#                   --trajectory-start-max-ratio 0"
#   MODEL           the model directory: config.json, tokenizer and weights
#                   (required)
#   GPU             the one GPU, as rocm-smi and `--gpu` number it (required);
#                   each compass command refuses it while a process is on it
#   MACHINE         the machine spec of this node's GPUs (required)
#   TRACES          a cc-traces traces.jsonl (required, as client.sh takes it)
#   AIPERF_PYTHON   the Python with agentx-harness and no compass-harness, for
#                   the records' clients (required)
#   CAL_TRACES      the traces each record's client is given (default 256-392)
#   DURATION_S      each record's client run's benchmark duration (default 1800)
#   SERVER_ARGS     ATOM's arguments after --model (default: --max-model-len
#                   262144)
#   WAIT_S          seconds a server has to start (default 1800)
#
# OUT gets each step's directory and log (<step>.log, and <step>.client.log for
# a record): warm/, memory/, cal<k>-c<C>/ for the k-th RECORD, fit_steps/ and
# fit_host/; store/, the store; and inputs, what the first run was given, which
# a run given other inputs refuses. A step whose output exists is skipped: a
# record's client export once `client.sh --stop` has marked it
# `server_stopped`, measure memory's or a fit's readable copy. A step stopped
# before its output exists is taken again, its directory removed first. A fit
# that refuses a record, such as one whose server built a JIT kernel, names it:
# remove that record's directory and run again. It prints
# the tokens, the law's held-out errors per band, the `compass run` line that
# prices by the law, and the wall time. It exits 2 on bad arguments or inputs,
# 90 to 99 as _lib.sh's tree checks refuse, else 1 naming the step that failed.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

bad() { echo "calibrate: $1" >&2; exit 2; }
ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
compass_describe "$ROOT"
(($# > 1)) || bad "no OUT or no RECORD; usage: calibrate.sh OUT RECORD..."
OUT=$1
shift
for r; do [[ $r =~ ^[1-9][0-9]*( |$) ]] || bad "RECORD '$r' does not start with a concurrency"; done
for v in MODEL GPU MACHINE TRACES AIPERF_PYTHON; do [[ -n ${!v-} ]] || bad "$v is not set"; done
MODEL=$(realpath "$MODEL") MACHINE=$(realpath "$MACHINE")
CAL=${CAL_TRACES:-256-392} DURATION_S=${DURATION_S:-1800} WAIT_S=${WAIT_S:-1800}
# shellcheck disable=SC2206 # SERVER_ARGS is a word list
ATOM_ARGS=(--model "$MODEL" ${SERVER_ARGS:---max-model-len 262144})
CLI='from atom.compass.cli import main; main()'
export MODEL TRACES
sha=$(compass_head_sha "$ROOT") ||
    bad "$ROOT is neither a git checkout nor a snapshot.sh tree, and a fit refuses it"
inputs=$(printf '%s\n' "tree $sha" "atom ${ATOM_ARGS[*]}" "machine $MACHINE" \
    "traces $TRACES $CAL" "duration $DURATION_S" "${@/#/record }")
mkdir -p "$OUT" && OUT=$(realpath "$OUT")
if [[ -e $OUT/inputs ]]; then
    diff "$OUT/inputs" <(echo "$inputs") >&2 || bad "$OUT was started with other inputs (<) than these (>)"
else
    [[ -z $(ls -A "$OUT") ]] || bad "$OUT holds files and no inputs"
    echo "$inputs" >"$OUT/inputs"
fi
# Every path is absolute from here, and nothing resolves `atom` from $PWD.
cd / || exit 1
start=$SECONDS
# Any exit, a failed step's or a SIGTERM's, stops what is still running: each
# step and client run runs in the background, as a job this trap reaches.
trap 'kill -TERM $(jobs -p) 2>/dev/null; wait' EXIT
fail() { echo "calibrate: step $1 failed: $2" >&2; exit 1; }
say() { echo "calibrate: $1 $(date -u +%FT%TZ)"; }

# todo NAME FILE [TEXT]: whether step NAME is left to take, any directory a
# stopped take left removed; it is not once OUT/NAME/FILE, its output, holds TEXT.
todo() {
    if [[ -s $OUT/$1/$2 ]] && grep -qF -- "${3-}" "$OUT/$1/$2"; then
        say "$1 is done; skipped"
        return 1
    fi
    [[ -e $OUT/$1 ]] && say "$1 was stopped; taking it again"
    rm -rf "${OUT:?}/$1" "$OUT/$1.cold"
}

# step NAME CMD...: run CMD, its output in OUT/NAME.log.
step() {
    local name=$1
    shift
    say "$name"
    "$@" >"$OUT/$name.log" 2>&1 &
    wait $! || fail "$name" "exit $?; see $OUT/$name.log"
}

# serve NAME A-B C S SCENARIO: one compass record server writing to OUT/NAME,
# and one client run of traces A to B at concurrency C for S seconds under
# SCENARIO, which stops it.
serve() {
    local name=$1 dir=$OUT/$1 pid deadline
    say "$name"
    python3 -c "$CLI" record --gpu "$GPU" --out "$dir" -- "${ATOM_ARGS[@]}" >"$OUT/$name.log" 2>&1 &
    pid=$!
    deadline=$((SECONDS + WAIT_S))
    until [[ -s $dir/server.json ]]; do
        kill -0 $pid 2>/dev/null || fail "$name" "its server did not start; see $OUT/$name.log"
        ((SECONDS < deadline)) || fail "$name" "no server after $WAIT_S s"
        sleep 1
    done
    HARNESS_PYTHON=$AIPERF_PYTHON AIPERF_EXTRA_ARGS=$5 "$ROOT/scripts/compass/client.sh" "$dir" \
        --traces "$2" --concurrency "$3" --duration "$4" --stop >"$OUT/$name.client.log" 2>&1 &
    wait $! || fail "$name" "client.sh exit $?; see $OUT/$name.client.log"
    wait $pid || fail "$name" "its server exit $?; see $OUT/$name.log"
}

M=(--machine "$MACHINE" --store "$OUT/store")
# client.sh --stop marks its export `server_stopped` once the server has
# stopped and written run.json.
STOPPED=(artifacts/client_run.json '"server_stopped": true')
todo warm "${STOPPED[@]}" && serve warm 0-7 1 600 ""
todo memory memory_readings.json &&
    step memory python3 -c "$CLI" measure memory --gpu "$GPU" "${M[@]}" --out "$OUT/memory" -- "${ATOM_ARGS[@]}"
k=0 dirs=() runs=()
for r; do
    read -r c scenario <<<"$r"
    name=cal$((++k))-c$c
    # A record taken after the fits leaves them to be taken again: the store
    # then refuses their entries' overwrite rather than keep a stale law.
    todo "$name" "${STOPPED[@]}" && serve "$name" "$CAL" "$c" "$DURATION_S" "$scenario" &&
        rm -rf "$OUT/fit_steps" "$OUT/fit_host"
    dirs+=("$OUT/$name") runs+=(--client-run "$OUT/$name" "$OUT/$name/artifacts/client_run.json")
done
todo fit_steps step_linear_law.json && step fit_steps python3 -c "$CLI" fit steps \
    --backend step-linear --form history "${dirs[@]}" "${runs[@]}" "${M[@]}" --out "$OUT/fit_steps"
todo fit_host host_laws.json &&
    step fit_host python3 -c "$CLI" fit host "${dirs[@]}" "${runs[@]}" "${M[@]}" --out "$OUT/fit_host"

token() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["token"])' "$OUT/$1"; }
law=$(token fit_steps/step_linear_law.json) host=$(token fit_host/host_laws.json)
echo "calibrate: memory_readings $(token memory/memory_readings.json), step_linear_law $law, host_laws $host"
python3 - "$OUT/fit_steps/step_linear_law.json" <<'EOF' || fail held_out "cannot read $OUT/fit_steps/step_linear_law.json"
import json, sys

law = json.load(open(sys.argv[1]))["params"]
if law["unfitted"]:
    print(f"calibrate: unfitted, so refusing every step they price: {law['unfitted']}")
rows = [
    f"{band} | {e['steps']} | " + " | ".join(f"{e[p]:+.1%}" for p in ("p10", "p50", "p90"))
    for vector in (law.get("prefill"), *law["decode"].values())
    for band, e in (vector or {}).get("held_out", {}).get("bands", {}).items()
]
print("\n".join(["held out | steps | p10 | p50 | p90", *rows]) if rows else
      "calibrate: no held-out errors: fit steps holds out each of two or more records")
EOF
run=("$ROOT/scripts/compass/compass" run --step-linear "$law" --host-laws "$host" "${M[@]}" --out DIR -- "${ATOM_ARGS[@]}")
echo "calibrate: a run priced by the law: $(printf '%q ' "${run[@]}")"
echo "calibrate: calibrated in $(((SECONDS - start) / 60)) min"
