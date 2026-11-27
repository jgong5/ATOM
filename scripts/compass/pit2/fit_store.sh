#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A space file's `mooncake_store` entry, the simulated Store's laws, fitted on
# the prefill server logs of calibration records, inside the CPU container at
# nice 19 (`step_linear_fit offload`).
#
# Usage: fit_store.sh ENTRY CAL_DIR...
#   ENTRY     the entry written, a JSON file the space file's common block
#             names as "mooncake_store"
#   CAL_DIR   a calibration's directory, OUT/cal/<candidate>: every record
#             under it (cal*-c*) whose client run stopped its server, a record
#             compiled twice (*.jit1) left out
#
# The records' prefill must have run with OFFLOAD_PROFILE=1, which the M3.1
# space file sets, for the copy lines. The lookup lines print at
# ATOM_LOG_LEVEL=DEBUG only: give at least one calibration's prefill that level
# (its PREFILL_ENV) or the entry's lookup_s stays unfitted. The master's flags
# are the recipe's store.sh's, the owners' bytes its default 4 x 128 GiB;
# override with MASTER_FLAGS and CAPACITY_BYTES for another Store.
# Prints the logs it read and FIT_STORE_EXIT <code>.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ENTRY=${1:?fit_store.sh ENTRY CAL_DIR...}
shift
logs=()
for cal in "$@"; do
    for f in "$cal"/cal*-c*/prefill/server.log; do
        [[ -f $f && $f != *.jit1/* ]] || continue
        # A record still serving writes its log yet.
        grep -qs '"server_stopped": true' "${f%/prefill/server.log}/artifacts/client_run.json" && logs+=("$f")
    done
done
echo "fit_store: ${#logs[@]} logs"
printf '  %s\n' "${logs[@]}"
((${#logs[@]})) || { echo "FIT_STORE_EXIT 1"; exit 1; }
MASTER=${MASTER_FLAGS:-"--rpc_port=26051 --rpc_thread_num=32 --enable_http_metadata_server=true \
--http_metadata_server_host=0.0.0.0 --http_metadata_server_port=26080 --metrics_port=26090 \
--default_kv_lease_ttl=10000 --eviction_high_watermark_ratio=0.88 --eviction_ratio=0.02 \
--allocation_strategy=random --memory_allocator=offset --client_ttl=10"}
cd / && PYTHONPATH=$PIT2_TREE nice -n 19 python3 -m atom.compass.backends.step_linear_fit offload "${logs[@]}" \
    --out "$ENTRY" --master-flags "$MASTER" --capacity-bytes "${CAPACITY_BYTES:-$((4 * 128 * 1024 * 1024 * 1024))}"
echo "FIT_STORE_EXIT $?"
