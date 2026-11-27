#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One held-out real run of a candidate at concurrency K, inside the GPU
# container under a gpool lease (gpool_steps.sh real:CMD_FILE:K): the
# deployment its calibration records ran, as a `deploy.sh record`, served one
# `client.sh --agentx` run of REAL_DURATION_S seconds (default 900, the
# search's), which stops it, against a fresh Store on this node. Compare it with
# the search's simulated point by scripts/compass/one_real.py.
#
# Usage: real_step.sh CMD_FILE K [OUTD]
#   CMD_FILE  the candidate's calibration command as `compass advice` printed it:
#             its variables (arguments, sides' variables, router, traces) are
#             the deployment's, its Store moved to this node
#   K         the concurrency
#   OUTD      the run's directory, absent (default PIT2_DATA/real/<name>-c<K>)
#
# A container that never served the candidate (no PIT2_WORK/warmed_<name>)
# first serves one throwaway deployment of it for 120 s, so the kernels it
# compiles on first use are built before the measured run: a run whose server
# compiled while it served is refused by one_real.py unless
# COMPASS_ACCEPT_JIT_BUILDS=1 says the compile stalled nothing graded.
# While the client runs, the decode's admission and KV gauges are sampled each
# minute into OUTD.admission.log.
# Log: PIT2_DATA/logs/real_<name>_c<K>.log. Ends with
# `REAL_EXIT <name> <K> client=<code> deploy=<code>`.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/env.sh"
CMD_FILE=${1:?real_step.sh CMD_FILE K [OUTD]}
K=${2:?real_step.sh CMD_FILE K [OUTD]}
name=$(basename "$CMD_FILE" .cmd)
IP=$(pit2_node_ip)
OUTD=${3:-$PIT2_DATA/real/$name-c$K}
mkdir -p "$PIT2_DATA/logs"
L=$PIT2_DATA/logs/real_${name}_c$K.log
say() { echo "$*" | tee -a "$L"; }
rocm-smi --showpids 2>&1 | grep -q "No KFD PIDs" || { say "GPUs busy"; say "REAL_EXIT $name $K busy 1"; exit 1; }
[[ ! -e $OUTD ]] || { say "$OUTD exists"; say "REAL_EXIT $name $K exists 1"; exit 1; }
prefix=$(python3 "$here/printed.py" prefix "$CMD_FILE" "$IP") || { say "REAL_EXIT $name $K cmd 2"; exit 2; }
eval "export $prefix"
MODEL=$(realpath "$MODEL")
# shellcheck disable=SC2206  # each side's arguments split as the printed command quotes them
PREFILL_ATOM=(--model "$MODEL" $PREFILL_ARGS) DECODE_ATOM=(--model "$MODEL" $DECODE_ARGS) ROUTER=($ROUTER_ARGS)
deploy() {
    WAIT_S=1800 "$PIT2_TREE/scripts/compass/deploy.sh" "$1" record \
        --prefill --gpu "$PREFILL_GPU" -- "${PREFILL_ATOM[@]}" \
        --decode --gpu "$DECODE_GPU" -- "${DECODE_ATOM[@]}" --router -- "${ROUTER[@]}"
}
cd /
if [[ ! -f $PIT2_WORK/warmed_$name ]]; then
    "$here/store.sh" start "warm-$name" >>"$L" 2>&1 || { say "REAL_EXIT $name $K store 1"; exit 1; }
    W=$PIT2_DATA/real/warm-$name-$(hostname)-$(date -u +%H%M)
    echo "$(date -u +%FT%TZ) warm $name into $W" >>"$L"
    deploy "$W" >>"$L" 2>&1 &
    wpid=$!
    until [[ -s $W/server.json ]] || ! kill -0 $wpid 2>/dev/null; do sleep 5; done
    WAIT_S=1800 HARNESS_PYTHON=$AIPERF_PYTHON "$PIT2_TREE/scripts/compass/client.sh" "$W" --agentx \
        --concurrency "$K" --duration 120 --stop >"$W.client.log" 2>&1
    wait $wpid
    "$here/store.sh" stop >>"$L" 2>&1
    touch "$PIT2_WORK/warmed_$name"
fi
"$here/store.sh" start "real-$name-c$K" >>"$L" 2>&1 || { say "REAL_EXIT $name $K store 1"; exit 1; }
echo "$(date -u +%FT%TZ) real $name c=$K on $(hostname) $IP" >>"$L"
deploy "$OUTD" >>"$L" 2>&1 &
pid=$!
deadline=$((SECONDS + 1800))
until [[ -s $OUTD/server.json ]]; do
    if ! kill -0 $pid 2>/dev/null || ((SECONDS >= deadline)); then
        kill $pid 2>/dev/null
        "$here/store.sh" stop >>"$L" 2>&1
        say "REAL_EXIT $name $K no-server 1"
        exit 1
    fi
    sleep 5
done
durl=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['url'])" "$OUTD/decode/server.json" 2>/dev/null)
(while kill -0 $pid 2>/dev/null; do
    echo "# $(date -u +%FT%TZ)"
    curl -s --max-time 5 "$durl/metrics" | grep -E '^atom:(pd_admission|kv_cache_usage|num_requests)'
    sleep 60
done) >"$OUTD.admission.log" 2>&1 &
WAIT_S=1800 HARNESS_PYTHON=$AIPERF_PYTHON "$PIT2_TREE/scripts/compass/client.sh" "$OUTD" --agentx \
    --concurrency "$K" --duration "${REAL_DURATION_S:-900}" --stop >"$OUTD.client.log" 2>&1
crc=$?
wait $pid
drc=$?
"$here/store.sh" stop >>"$L" 2>&1
say "REAL_EXIT $name $K client=$crc deploy=$drc"
