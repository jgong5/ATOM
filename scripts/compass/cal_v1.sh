#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One calibration record of a v1.1 design (#739 section 3): ONE server, the
# plan's segments in turn, each started by an event, never by a predicted time.
#
# Usage: cal_v1.sh DESIGN OUT record|run COMPASS_ARGS... -- SERVE_ARGS...
#   DESIGN        gen_v1.py's OUT, with plan_v1.py's plan.json
#   OUT           an absent directory: the server's --out is OUT/real, its
#                 record OUT/real/steps; each lane's client run OUT/<segment>/<lane>
#   record|run    `compass record` (GPUs) or `compass run` (a simulated server,
#                 the CPU dry run), with its own arguments up to `--` (record:
#                 --gpu G --store S; run: --step-linear LAW --host-laws HOST
#                 --device D --ptl P --store S), then the serve arguments
#   MODEL, HARNESS_PYTHON   as client.sh takes them
#   CAL_AIPERF_EXTRA_ARGS   more aiperf arguments for every lane, before the
#                 lane's own (client.sh's AIPERF_EXTRA_ARGS, which this script
#                 sets per lane): a model whose tokenizer runs its own code
#                 (Kimi-K3.1) needs --tokenizer-trust-remote-code (default none)
#   DRAIN_TRIES   polls of the server's in-flight requests, 5 s apart, before a
#                 segment counts as drained anyway (default 180; a simulated
#                 server reports its last requests running until the next
#                 client run moves its clock, so `run` defaults to 2)
#   SETTLE_S      seconds after the drain before the next segment (default 15)
#
# The server runs under plan.json's server_env (ATOM_DP_SESSION_AFFINITY=1,
# ATOM_DP_LB_REQ_EQUIV=1000000000). Per segment every lane starts at once, a
# client.sh run each over its trace file: every trace sent once
# (--num-conversations), trajectory start ratios 0 (a session's first turn at
# the lane's start), its seed, the chat template, and for a pinned lane
# `--extra-inputs data_parallel_rank:R`. The next segment starts once every
# lane has exited and the server holds no request (atom:requests_running plus
# atom:requests_waiting at 0), then SETTLE_S. After the last, the command that
# started the server is stopped and waited for. Exits 1 naming the segment
# whose client failed, 2 on bad arguments. Check the record against its plan
# with COMPASS_PREFILL_LABEL=record check_v1.py DESIGN OUT --law LAW --host
# HOST, which labels its prefill steps as the record's fit does.
set -euo pipefail
here=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
USAGE="usage: cal_v1.sh DESIGN OUT record|run COMPASS_ARGS... -- SERVE_ARGS..."
(($# >= 3)) || { echo "$USAGE" >&2; exit 2; }
D=$(readlink -f -- "$1") OUT=$2 MODE=$3
shift 3
[[ $MODE == record || $MODE == run ]] || { echo "cal_v1: mode $MODE; $USAGE" >&2; exit 2; }
[[ -s $D/plan.json ]] || { echo "cal_v1: no $D/plan.json (plan_v1.py)" >&2; exit 2; }
[[ ! -e $OUT ]] || { echo "cal_v1: $OUT exists" >&2; exit 2; }
: "${MODEL:?}" "${HARNESS_PYTHON:?}"
TRIES=${DRAIN_TRIES:-$([[ $MODE == run ]] && echo 2 || echo 180)}
mkdir -p "$OUT/logs"
OUT=$(readlink -f -- "$OUT")
plan() { python3 -c "import json,sys; p=json.load(open('$D/plan.json')); $1" "${@:2}"; }
while read -r kv; do export "${kv?}"; done < <(plan 'print("\n".join(f"{k}={v}" for k, v in p["server_env"].items()))')
cp "$D/plan.json" "$OUT/"

"$here/compass" "$MODE" --out "$OUT/real" "$@" > "$OUT/logs/server.log" 2>&1 &
pid=$!
until [[ -s $OUT/real/server.json ]]; do
    kill -0 $pid 2> /dev/null || { tail -5 "$OUT/logs/server.log"; echo "cal_v1: the server died" >&2; exit 1; }
    sleep 10
done
read -r URL SPID < <(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["url"], s["pid"])' "$OUT/real/server.json")
CMD=$(awk '{print $4}' "/proc/$SPID/stat")
stop() { kill -TERM "$CMD" 2> /dev/null || true; wait $pid; local rc=$?; [[ -s $OUT/real/run.json ]] || echo "cal_v1: no run.json"; return $rc; }
inflight() { curl -s -m 10 "$URL/metrics" | awk '/^atom:requests_(running|waiting)[ {]/ {s += $NF} END {print s + 0}'; }
echo "$(date +%T) server $URL pid $SPID"

n=$(plan 'print(len(p["sequence"]))')
for ((k = 0; k < n; k++)); do
    SEG=$(plan 'print(p["sequence"][int(sys.argv[1])]["segment"])' $k)
    echo "$(date +%T) $SEG start (in flight $(inflight))"
    pids=()
    while read -r lane file pin c traces seed; do
        L=$OUT/$SEG/$lane
        mkdir -p "$L"
        cp "$OUT/real/server.json" "$L/"
        extra="${CAL_AIPERF_EXTRA_ARGS:+$CAL_AIPERF_EXTRA_ARGS }--apply-chat-template"
        extra+=" --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0"
        extra+=" --random-seed $seed --num-conversations $traces"
        [[ $pin == None ]] || extra+=" --extra-inputs data_parallel_rank:$pin"
        TRACES=$D/$file AIPERF_EXTRA_ARGS=$extra \
            "$here/client.sh" "$L" --traces "0-$((traces - 1))" --concurrency "$c" --duration 3600 \
            > "$OUT/logs/client_${SEG}_$lane.log" 2>&1 &
        pids+=($!)
    done < <(plan 'print("\n".join(" ".join(str(x[f]) for f in ("name", "file", "pin", "concurrency", "traces", "seed")) for x in p["sequence"][int(sys.argv[1])]["lanes"]))' $k)
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=$?; done
    if ((rc)); then
        tail -5 "$OUT"/logs/client_"${SEG}"_*.log
        stop || true
        echo "cal_v1: a client of $SEG failed ($rc)" >&2
        exit 1
    fi
    m=-1
    for ((i = 0; i < TRIES; i++)); do m=$(inflight); [[ $m == 0 ]] && break; sleep 5; done
    echo "$(date +%T) $SEG drained, $m in flight"
    sleep "${SETTLE_S:-15}"
done
stop
echo "$(date +%T) stopped: record $OUT/real"
