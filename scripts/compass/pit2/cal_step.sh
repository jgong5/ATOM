#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One calibration `compass advice` printed (its `made by:` line, saved to
# CMD_FILE), inside the GPU container under a gpool lease (gpool_steps.sh
# cal:CMD_FILE), against a fresh Store on this node.
#
# Usage: cal_step.sh CMD_FILE
#
# - The printed command names the Store of the host the search ran on
#   (`mooncake_store.local_hostname` in the prefill's offload config); every
#   occurrence of that address becomes this node's. The command run is saved
#   beside the calibration's directory as <OUT>.cmd, for cal_fits.sh.
# - Each record runs DURATION_S seconds (default 900, the search's duration;
#   calibrate.sh's own default is 1800). DURATION_S is one of calibrate.sh's
#   inputs: a resume must give the same, which <OUT>.cmd does.
# - RECORDS_ONLY=1 stops calibrate.sh once its records are taken, as its first
#   fit starts (the status file's last line `start fit_`), so the lease is not
#   held through CPU-only fits: run cal_fits.sh in the CPU container after.
# - The GPUs must be idle at the start: a record beside other work fits that
#   load into its laws.
#
# Log: PIT2_DATA/logs/cal_<name>.log, <name> CMD_FILE's base name less .cmd.
# Ends with `CAL_EXIT <name> <code>`, 0 also when RECORDS_ONLY stopped it at its
# fits.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/env.sh"
CMD_FILE=${1:?cal_step.sh CMD_FILE}
name=$(basename "$CMD_FILE" .cmd)
mkdir -p "$PIT2_DATA/logs"
L=$PIT2_DATA/logs/cal_$name.log
say() { echo "$*" | tee -a "$L"; }
rocm-smi --showpids 2>&1 | grep -q "No KFD PIDs" || { say "GPUs busy"; say "CAL_EXIT $name 1"; exit 1; }
cmd=$(python3 "$here/printed.py" localize "$CMD_FILE" "$(pit2_node_ip)") || { say "CAL_EXIT $name 2"; exit 2; }
out=$(python3 "$here/printed.py" out "$CMD_FILE") || { say "CAL_EXIT $name 2"; exit 2; }
cmd="DURATION_S=${DURATION_S:-900} $cmd"
echo "$cmd" >"$out.cmd"
"$here/store.sh" start "cal-$name" >>"$L" 2>&1 || { say "CAL_EXIT $name store 1"; exit 1; }
echo "$(date -u +%FT%TZ) cal $name on $(hostname): $cmd" >>"$L"
export CMD="$cmd"
setsid bash -c 'cd / && eval "$CMD"' >>"$L" 2>&1 </dev/null &
pid=$!
stopped=0
while kill -0 $pid 2>/dev/null; do
    if [[ -n ${RECORDS_ONLY-} ]] && tail -1 "$out/status" 2>/dev/null | grep -q "start fit_"; then
        kill -TERM -- -$pid 2>/dev/null
        sleep 5
        kill -KILL -- -$pid 2>/dev/null
        echo "$(date -u +%FT%TZ) records taken; calibrate.sh stopped at its fits" >>"$L"
        stopped=1
        break
    fi
    sleep 10
done
wait $pid
rc=$?
((stopped)) && rc=0
"$here/store.sh" stop >>"$L" 2>&1
say "CAL_EXIT $name $rc"
