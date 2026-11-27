#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# GPU steps of a compass advice search on pit2, from the host, each under a
# whole-node gpool lease of its own, released the moment it ends (pass or
# fail). Run it in the background: setsid nohup gpool_steps.sh ... &.
#
# Usage: gpool_steps.sh NODES STEP...
#   NODES   one node (g52), a comma list the pool grants one of (g40,g52), or
#           any. Leave out nodes whose CPUs a CPU job owns.
#   STEP    platform:STORE          platform.sh STORE
#           cal:CMD_FILE            cal_step.sh CMD_FILE (RECORDS_ONLY passes on)
#           real:CMD_FILE:K         real_step.sh CMD_FILE K
#
# For each step: `gpool acquire -g 8 --wait --timeout 3h` (gpool's own queue;
# an error that is not its timeout, such as a stale NFS handle on its ledger,
# is retried 5 times a minute apart), the GPU container on the granted node
# (containers.sh gpu, unless PIT2_GPU_CONTAINER already runs there), a check
# that the GPUs are idle, the step inside the container, a wait for the GPU
# memory to clear, and `gpool release <lease> --note <the step's exit line>`.
# A failed calibration or platform step stops the run; a failed real run is
# logged and the next goes on.
#
# Reads env.sh's variables, GPOOL (default /share_nfs/yajizhan/gpu-pool/bin/gpool)
# and PIT2_GPU_CONTAINER (default compass-pit2-gpu-<uid>).
# Log: PIT2_DATA/logs/gpool_steps.log. Ends with `ALL_DONE`.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/env.sh"
NODES=${1:?gpool_steps.sh NODES STEP...}
shift
GP=${GPOOL:-/share_nfs/yajizhan/gpu-pool/bin/gpool}
C=${PIT2_GPU_CONTAINER:-compass-pit2-gpu-$(id -u)}
mkdir -p "$PIT2_DATA/logs"
L=$PIT2_DATA/logs/gpool_steps.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
pin=(--node "$NODES")
[[ $NODES == any ]] && pin=()
idle() { ssh "$1" "rocm-smi --showpids 2>&1 | grep -q 'No KFD PIDs'"; }
for s in "$@"; do
    IFS=: read -r kind a b <<<"$s"
    case $kind in
    platform) cmd=("$here/platform.sh" "$a") what="platform readings into $a" stop=1 ;;
    cal) cmd=(env ${RECORDS_ONLY:+RECORDS_ONLY=1} "$here/cal_step.sh" "$a") what="calibration $(basename "$a" .cmd)" stop=1 ;;
    real) cmd=("$here/real_step.sh" "$a" "$b") what="real run $(basename "$a" .cmd) c=$b" stop=0 ;;
    *) log "$s: unknown step"; exit 2 ;;
    esac
    log "$s: queueing for $NODES"
    lease=
    for try in 1 2 3 4 5; do
        out=$("$GP" acquire -g 8 ${pin[@]+"${pin[@]}"} --wait --timeout 3h \
            -p "compass advice on pit2 ($(id -un 2>/dev/null || id -u)): $what" 2>&1)
        code=$?
        lease=$(sed -n 's/^export GPOOL_LEASE=//p' <<<"$out")
        [[ -n $lease || $code == 2 ]] && break
        log "$s: acquire error (try $try): $(tail -1 <<<"$out")"
        sleep 60
    done
    [[ -n $lease ]] || { log "$s: not granted: $(tail -2 <<<"$out" | tr '\n' ' ')"; exit 1; }
    H=$(grep -oE 'pit2-p03-g[0-9]+' <<<"$out" | head -1)
    log "$s: granted $lease on $H"
    ssh "$H" "docker ps --format '{{.Names}}' | grep -qx $C" ||
        "$here/containers.sh" gpu "${H#pit2-p03-}" "$C" >>"$L" 2>&1
    if ! idle "$H"; then
        "$GP" release "$lease" --note "$s not started: GPUs busy" >>"$L" 2>&1
        log "$s: GPUs busy at grant; released $lease"
        exit 1
    fi
    ssh "$H" "docker exec -e PIT2_DATA='$PIT2_DATA' $C ${cmd[*]@Q}" >>"$L" 2>&1
    for _ in $(seq 30); do idle "$H" && break; sleep 10; done
    result=$(grep -hE "^(CAL|REAL|PLATFORM)_EXIT" "$L" | tail -1)
    "$GP" release "$lease" --note "$s: ${result:-no exit line}" >>"$L" 2>&1
    log "$s: released $lease: $result"
    if ((stop)) && ! grep -qE "_EXIT( [^ ]+)? 0$" <<<"$result"; then
        log "$s failed; stopping"
        exit 1
    fi
done
log ALL_DONE
