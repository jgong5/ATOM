#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# QUICKSTART.md's calibration in one command, on the GPUs GPU names: a warm
# record nothing reads, as its step 0 serves, so the aiter JIT builds a fresh
# container meets fall outside the records the fits read; `compass measure
# memory`; one `compass record` per RECORD, each serving one `client.sh` run of
# the calibration traces, which stops it; then `compass fit steps` (step linear,
# history form, MODEL's MoE routing) and `compass fit host` on those records.
# It plans nothing: what to record is its input. A decode rung N comes from a
# record at concurrency N that starts every trajectory at its first turn
# (https://github.com/jgong5/ATOM/issues/1840). The prefill law fits from trace
# runs alone: with none, the law's `unfitted` names prefill. `compass fit host`
# leaves them out, since a trace run's host segments ran under the profiler.
# Every server but a trace run's starts with the blocking-call tool, which it
# builds into OUT/tool (atom/compass/parity/README.md).
#
# Usage: calibrate.sh OUT RECORD...
#   OUT             an absent or empty directory, or one an earlier run given
#                   the same inputs left, which this run resumes
#   RECORD          one record: its concurrency, then the client's scenario,
#                   aiperf arguments client.sh takes as AIPERF_EXTRA_ARGS, if
#                   any, such as "8 --trajectory-start-min-ratio 0
#                   --trajectory-start-max-ratio 0"; or `trace` and such a
#                   record, as "trace 16", a `compass record --trace` of it
#   MODEL           the model directory: config.json, tokenizer and weights
#                   (required)
#   GPU             every GPU the configuration uses, comma-separated, as
#                   rocm-smi numbers them and `--gpu` takes them (required);
#                   each compass command refuses one while a process is on it
#   STORE           the store the readings and laws are published into, an
#                   existing directory, such as the shared store a search
#                   reads (default OUT/store)
#   TRACES          a cc-traces traces.jsonl (required, as client.sh takes it)
#   AIPERF_PYTHON   the Python with agentx-harness and no compass-harness, for
#                   the records' clients (required)
#   CAL_TRACES      the traces each record's client is given (default 256-392)
#   DURATION_S      each record's client run's benchmark duration (default 1800)
#   TRACE_S         each trace run's client run's benchmark duration (default 120)
#   SERVER_ARGS     ATOM's arguments after --model (default: --max-model-len
#                   262144)
#   WAIT_S          seconds a server has to start (default 1800); on 1P1D,
#                   each deploy.sh instance to answer and the deployment to
#                   leave once stopped, as deploy.sh and client.sh wait, a trace
#                   run serve's TRACE_STOP_S more, as its prefill side writes
#                   its trace first
#
# A 1P1D configuration takes, in place of GPU and SERVER_ARGS:
#   PREFILL_GPU DECODE_GPU
#                   each side's cards, as GPU takes them (required)
#   PREFILL_ARGS DECODE_ARGS
#                   each side's ATOM arguments after --model, as SERVER_ARGS
#                   takes them (required)
#   PREFILL_ENV DECODE_ENV
#                   NAME=VALUE words, split on whitespace, set in that side
#                   alone, as deploy.sh takes them
#   FIT             the sides whose laws it fits: prefill, decode, or "prefill
#                   decode" (default)
#   ROUTER_ARGS     the router's arguments, split on whitespace, as deploy.sh
#                   takes them after --router -- (required)
# Each server is then a `deploy.sh record` of both sides, the prefill side on
# the cards PREFILL_GPU names and the decode side on DECODE_GPU's, which one
# client run stops; a trace run traces its prefill side, which FIT must fit.
# `compass measure memory` measures each side FIT names, on its cards, under its
# variables and the connector a real start of it takes, and `compass fit steps`
# and `compass fit host` fit each such side on its records: the prefill side on
# the trace runs too.
#
# OUT gets each step's directory and log (<step>.log, and <step>.client.log for
# a record): warm/, memory/, cal<k>-c<C>/ for the k-th RECORD, fit_steps/ and
# fit_host/, on 1P1D memory_<side>/, fit_steps_<side>/ and fit_host_<side>/
# for each side FIT names; store/, the store, unless STORE names one; inputs,
# what the first run was given, which a run given other inputs refuses; and
# status, a line per step's start, end with its wall seconds, or failure with
# its reason, each stamped with its UTC time and appended by every run, and once
# a run has fitted every law `end calibration` with the node-hours of every
# step's every take.
# A step whose output exists is skipped: a record's client export once
# `client.sh --stop` has marked it `server_stopped`, measure memory's or a fit's
# readable copy. A step stopped before its output exists is taken again, its
# directory removed first. A fit that refuses a record, such as one whose server
# built a JIT kernel, names it: remove that record's directory and run again. A
# record left to take once a fit has published refuses, before it serves: the
# fit's key names no record, so the store refuses that fit again, and the
# calibration needs a new OUT. It prints the tokens, the law's held-out errors
# per band, the `compass run` line that prices by the law, its DATASHEET and PTL
# left to name and its store to hold the platform's readings (`compass fit
# device`), each step's wall seconds as status last records them, the
# node-hours, and this run's wall time; on 1P1D, each fitted side's tokens and
# held-out errors, and no run line, as the store holds each side's law.
# It exits 2 on bad arguments or inputs, or while another calibrate.sh, or a
# step one started, runs in OUT; 90 to 99 as _lib.sh's tree checks refuse, else
# 1 naming the step that failed, a SIGTERM, SIGINT or SIGHUP included. Under
# `setsid nohup` it outlives the shell that started it.
# A trace run's directory is trace<k>-c<C>/, done once it, or on 1P1D its
# prefill/, holds busy lines, which one refused at its stop lacks.
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
for r; do [[ $r =~ ^(trace )?[1-9][0-9]*( |$) ]] || bad "RECORD '$r' does not start with a concurrency"; done
# A 1P1D configuration: its sides, each one `deploy.sh` instance.
SIDES=()
[[ -z ${PREFILL_ARGS-}${DECODE_ARGS-} ]] || SIDES=(prefill decode)
cards=(GPU) needed=(GPU)
((${#SIDES[@]} == 0)) || cards=(PREFILL_GPU DECODE_GPU) needed=(PREFILL_GPU DECODE_GPU PREFILL_ARGS DECODE_ARGS ROUTER_ARGS)
for v in MODEL "${needed[@]}" TRACES AIPERF_PYTHON; do [[ -n ${!v-} ]] || bad "$v is not set"; done
for v in "${cards[@]}"; do
    [[ ${!v} =~ ^[0-9]+(,[0-9]+)*$ ]] || bad "$v '${!v}' is not a comma-separated list of GPU indices"
done
[[ -z ${STORE-} || -d $STORE ]] || bad "STORE $STORE is not a directory"
MODEL=$(realpath "$MODEL")
CAL=${CAL_TRACES:-256-392} DURATION_S=${DURATION_S:-1800} WAIT_S=${WAIT_S:-1800} TRACE_S=${TRACE_S:-120}
# shellcheck disable=SC2206 # SERVER_ARGS is a word list
ATOM_ARGS=(--model "$MODEL" ${SERVER_ARGS:---max-model-len 262144})
if ((${#SIDES[@]})); then
    [[ -z ${GPU-}${SERVER_ARGS-} ]] || bad "a 1P1D configuration takes PREFILL_GPU, DECODE_GPU, PREFILL_ARGS and DECODE_ARGS in place of GPU and SERVER_ARGS"
    read -ra FIT <<<"${FIT-prefill decode}"
    [[ " ${FIT[*]} " =~ ^\ (prefill\ )?(decode\ )?$ ]] || bad "FIT '${FIT[*]}' is not prefill, decode or \"prefill decode\""
    for r; do
        [[ $r != trace\ * || ${FIT[0]} == prefill ]] || bad "RECORD '$r' traces the prefill side, which FIT does not fit"
    done
    GPU=$PREFILL_GPU,$DECODE_GPU
    # shellcheck disable=SC2206 # each is a word list
    PREFILL_ATOM=(--model "$MODEL" $PREFILL_ARGS) DECODE_ATOM=(--model "$MODEL" $DECODE_ARGS) ROUTER=($ROUTER_ARGS)
    read -ra PREFILL_VARS <<<"${PREFILL_ENV-}"
    read -ra DECODE_VARS <<<"${DECODE_ENV-}"
    ATOM_ARGS=()
    TRACE_STOP_S=$(cd / && python3 -c 'from atom.compass.drive.serve import TRACE_STOP_S as s; print(s)') ||
        bad "cannot read serve's TRACE_STOP_S"
fi
CLI='from atom.compass.cli import main; main()'
export MODEL TRACES
sha=$(compass_head_sha "$ROOT") ||
    bad "$ROOT is neither a git checkout nor a snapshot.sh tree, and a fit refuses it"
# ATOM's arguments, or a 1P1D configuration's sides', its router's and the sides it fits.
atom=("atom ${ATOM_ARGS[*]}")
((${#SIDES[@]} == 0)) || atom=("atom prefill ${PREFILL_ATOM[*]}" "atom decode ${DECODE_ATOM[*]}" \
    "router ${ROUTER[*]}" "fit ${FIT[*]}")
inputs=$(printf '%s\n' "tree $sha" "${atom[@]}" \
    "traces $TRACES $CAL" "duration $DURATION_S" "trace duration $TRACE_S" "${@/#/record }")
# A store named is an input: a resume publishing elsewhere would split the
# calibration across two stores.
[[ -n ${STORE-} ]] && STORE=$(realpath "$STORE") && inputs+=$'\n'"store $STORE"
mkdir -p "$OUT" && OUT=$(realpath "$OUT")
if [[ -e $OUT/inputs ]]; then
    diff "$OUT/inputs" <(echo "$inputs") >&2 || bad "$OUT was started with other inputs (<) than these (>)"
else
    [[ -z $(ls -A "$OUT") ]] || bad "$OUT holds files and no inputs"
    echo "$inputs" >"$OUT/inputs"
fi
# One calibrate.sh into OUT at a time: a second would remove the step the first
# runs. Each step inherits the lock, so one left running holds OUT too.
exec {lock}>>"$OUT/inputs"
flock -n "$lock" || bad "$OUT is held by another calibrate.sh, or a step one started, still running"
# Every path is absolute from here, and nothing resolves `atom` from $PWD.
cd / || exit 1
start=$SECONDS
# Any exit, a failed step's or a SIGTERM's, stops what is still running: each
# step and client run runs in the background, as a job this trap reaches.
trap 'kill -TERM $(jobs -p) 2>/dev/null; wait' EXIT
# The step running, and the SECONDS it began at.
current= began=0
# status EVENT STEP [TEXT]: one line of OUT/status.
status() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/status"; }
# fail STEP REASON: STEP failed, after the seconds it ran if it is the one running.
fail() {
    local s=0
    [[ $1 == "$current" ]] && s=$((SECONDS - began))
    status fail "$1" "$s s: $2"
    echo "calibrate: step $1 failed: $2" >&2
    exit 1
}
trap 'fail "${current:-calibration}" "stopped by a signal"' TERM INT HUP
say() { echo "calibrate: $1 $(date -u +%FT%TZ)"; }
# begin STEP, then ended: STEP's start and its end, each in OUT/status.
begin() {
    say "$1"
    current=$1 began=$SECONDS
    status start "$1"
}
ended() {
    status end "$current" "$((SECONDS - began)) s"
    current=
}

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
    begin "$name"
    "$@" >"$OUT/$name.log" 2>&1 &
    wait $! || fail "$name" "exit $?; see $OUT/$name.log"
    ended
}

# serve NAME A-B C S SCENARIO [FLAG]: one compass record server, given FLAG,
# writing to OUT/NAME, or on 1P1D one deploy.sh record deployment, its prefill
# side given FLAG, and one client run of traces A to B at concurrency C for S
# seconds under SCENARIO, which stops it. A trace run's server (FLAG) loads no
# tool.
serve() {
    local name=$1 dir=$OUT/$1 pid deadline tool=(ROCP_TOOL_LIBRARIES="$TOOL") waits=()
    local server=(python3 -c "$CLI" record --gpu "$GPU" ${6:+"$6"} --out "$dir" -- "${ATOM_ARGS[@]}")
    if ((${#SIDES[@]})); then
        server=("$ROOT/scripts/compass/deploy.sh" "$dir" record
            --prefill --gpu "$PREFILL_GPU" ${6:+"$6"} -- "${PREFILL_ATOM[@]}"
            --decode --gpu "$DECODE_GPU" -- "${DECODE_ATOM[@]}" --router -- "${ROUTER[@]}")
        waits=(WAIT_S="$WAIT_S")
    fi
    # A trace run's server writes its trace as it stops.
    [[ -n ${6-} ]] && tool=() && ((${#waits[@]})) && waits=(WAIT_S=$((WAIT_S + TRACE_STOP_S)))
    begin "$name"
    env "${waits[@]}" "${tool[@]}" "${server[@]}" >"$OUT/$name.log" 2>&1 &
    pid=$!
    deadline=$((SECONDS + WAIT_S))
    until [[ -s $dir/server.json ]]; do
        kill -0 $pid 2>/dev/null || fail "$name" "its server did not start; see $OUT/$name.log"
        ((SECONDS < deadline)) || fail "$name" "no server after $WAIT_S s"
        sleep 1
    done
    env "${waits[@]}" HARNESS_PYTHON="$AIPERF_PYTHON" AIPERF_EXTRA_ARGS="$5" "$ROOT/scripts/compass/client.sh" \
        "$dir" --traces "$2" --concurrency "$3" --duration "$4" --stop >"$OUT/$name.client.log" 2>&1 &
    wait $! || fail "$name" "client.sh exit $?; see $OUT/$name.client.log"
    wait $pid || fail "$name" "its server exit $?; see $OUT/$name.log"
    ended
}

# measure [SIDE]: `compass measure memory` into OUT/memory, of the server; or
# into OUT/memory_SIDE, of 1P1D side SIDE on its cards, under its variables and
# the connector a real start of it takes, as deploy.sh record starts it.
measure() {
    local as=${1:+_$1} cards=$GPU vars=() atom=("${ATOM_ARGS[@]}") role=kv_producer
    if [[ -n ${1-} ]]; then
        local -n side_cards=${1^^}_GPU side_vars=${1^^}_VARS side_atom=${1^^}_ATOM
        cards=$side_cards vars=("${side_vars[@]}") atom=("${side_atom[@]}")
        [[ $1 == decode ]] && role=kv_consumer
        atom+=(--kv-transfer-config "{\"kv_role\": \"$role\", \"kv_connector\": \"mooncake\"}")
    fi
    todo "memory$as" memory_readings.json &&
        step "memory$as" env "${vars[@]}" python3 -c "$CLI" measure memory --gpu "$cards" \
            "${M[@]}" --out "$OUT/memory$as" -- "${atom[@]}"
}

# fit [SIDE]: `compass fit steps` into OUT/fit_steps and `compass fit host` into
# OUT/fit_host, on the records; or into OUT/fit_steps_SIDE and OUT/fit_host_SIDE,
# on 1P1D side SIDE of each, each export its deployment's. `compass fit host`
# takes no trace run, nor a decode side's fit any.
fit() {
    local as=${1:+_$1} at=${1:+/$1} dirs=() runs=() cal=() cal_runs=() i d
    for i in "${!names[@]}"; do
        d=$OUT/${names[i]}
        [[ -n ${traced[i]} && ${1-} == decode ]] && continue
        dirs+=("$d$at") runs+=(--client-run "$d$at" "$d/artifacts/client_run.json")
        [[ -z ${traced[i]} ]] && cal+=("$d$at") cal_runs+=("${runs[@]: -3}")
    done
    todo "fit_steps$as" step_linear_law.json && step "fit_steps$as" python3 -c "$CLI" fit steps \
        --backend step-linear --form history --model "$MODEL" "${dirs[@]}" "${runs[@]}" "${M[@]}" \
        --out "$OUT/fit_steps$as"
    todo "fit_host$as" host_laws.json && step "fit_host$as" python3 -c "$CLI" fit host \
        "${cal[@]}" "${cal_runs[@]}" "${M[@]}" --out "$OUT/fit_host$as"
}

M=(--store "${STORE:-$OUT/store}")
# What it measures and fits: the server, or each side FIT names.
units=("")
((${#SIDES[@]} == 0)) || units=("${FIT[@]}")
# client.sh --stop marks its export `server_stopped` once the server has
# stopped and written run.json.
STOPPED=(artifacts/client_run.json '"server_stopped": true')
BUSY=steps/dp0.busy.jsonl
((${#SIDES[@]} == 0)) || BUSY=prefill/$BUSY
TOOL=$OUT/tool/libcompass_blocked.so
todo tool "${TOOL##*/}" && step tool python3 -m atom.compass.parity.blocked "$OUT/tool"
todo warm "${STOPPED[@]}" && serve warm 0-7 1 600 ""
for u in "${units[@]}"; do measure ${u:+"$u"}; done
k=0 names=() traced=()
for r; do
    read -r c scenario <<<"${r#trace }"
    name=cal$((++k))-c$c done_file=("${STOPPED[@]}") flag= d=$DURATION_S
    [[ $r == trace\ * ]] && name=trace$k-c$c done_file=("$BUSY") flag=--trace d=$TRACE_S
    if todo "$name" "${done_file[@]}"; then
        for f in "$OUT"/fit_steps*/step_linear_law.json "$OUT"/fit_host*/host_laws.json; do
            [[ -e $f ]] && fail "$name" "$f was fitted without it, and the store refuses a second fit of its key; calibrate into a new OUT"
        done
        serve "$name" "$CAL" "$c" "$d" "$scenario" ${flag:+"$flag"}
    fi
    names+=("$name") traced+=("$flag")
done
for u in "${units[@]}"; do fit ${u:+"$u"}; done

token() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["token"])' "$OUT/$1"; }
laws=()
for u in "${units[@]}"; do
    as=${u:+_$u}
    law=$(token "fit_steps$as/step_linear_law.json") host=$(token "fit_host$as/host_laws.json")
    echo "calibrate: ${u:+$u: }memory_readings $(token "memory$as/memory_readings.json"), step_linear_law $law, host_laws $host"
    laws+=("$u" "$OUT/fit_steps$as/step_linear_law.json")
done
python3 - "$OUT/status" "$GPU" "${laws[@]}" <<'EOF' || fail held_out "cannot read $OUT/status or a law under $OUT"
import json, sys, time

status, gpus, *laws = sys.argv[1:]
for side, path in zip(laws[::2], laws[1::2]):
    law = json.load(open(path))["params"]
    if side:
        print(f"calibrate: the {side} side's law")
    # A pipelined side's law holds a part a stage (`stages`), each fitted alone.
    parts = law.get("stages") or [law]
    rows = []
    for k, part in enumerate(parts):
        at = f"stage {k} " if len(parts) > 1 else ""
        if part.get("unfitted"):
            print(f"calibrate: {at}unfitted, so refusing every step they price: {part['unfitted']}")
        rows += [
            f"{at}{band} | {e['steps']} | " + " | ".join(f"{e[p]:+.1%}" for p in ("p10", "p50", "p90"))
            for vector in (part.get("prefill"), *(part.get("decode") or {}).values())
            for band, e in (vector or {}).get("held_out", {}).get("bands", {}).items()
        ]
    print("\n".join(["held out | steps | p10 | p50 | p90", *rows]) if rows else
          "calibrate: no held-out errors: fit steps holds out each of two or more records")
# Each step's wall seconds as its last end states them; the node-hours, every
# take's seconds to its end or its failure.
wall, held = {}, 0
for line in open(status):
    _, event, step, *said = line.split()
    if event in ("end", "fail") and step != "calibration":
        held += int(said[0])
        if event == "end":
            wall[step] = int(said[0])
print("\n".join(["step | wall s", *(f"{k} | {s}" for k, s in wall.items())]))
hours = f"{held / 3600:.2f} node-hours"
print(f"calibrate: {hours} over every take of every step, on GPUs {gpus}")
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
open(status, "a").write(f"{now} end calibration {hours}\n")
EOF
# On 1P1D, compass advice runs the deployment, each side priced by its law in the store.
if ((${#SIDES[@]} == 0)); then
    run=("$ROOT/scripts/compass/compass" run --step-linear "$law" --host-laws "$host" --device DATASHEET --ptl PTL "${M[@]}" --out DIR -- "${ATOM_ARGS[@]}")
    echo "calibrate: a run priced by the law: $(printf '%q ' "${run[@]}")"
fi
echo "calibrate: calibrated in $(((SECONDS - start) / 60)) min"
