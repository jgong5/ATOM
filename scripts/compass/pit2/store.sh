#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The M3.1 recipe's Mooncake Store on this node, for a calibration's records or
# a real run, inside the GPU container: the recipe's own store.sh
# (recipes/mesh/minimax_m31_pd_store), run from a copy with #4164's two fixes:
# its atomesh scripts read from PIT2_TREE rather than /workspace/ATOM, and its
# capacity poll tolerates a metrics port that answers late. The image has no
# iproute2, so `ip`, which the recipe's env.sh calls, is answered by bin/ip from
# this node's address.
#
# Usage: store.sh start|stop|status [TAG]
#   TAG   names the run's log directory, PIT2_DATA/store/<TAG> (default manual)
#
# A calibration or real run takes a fresh Store each (cal_step.sh,
# real_step.sh): a Store left over from an earlier one holds its keys, and a
# record would read them as prefix hits. The master listens on 26051, its
# metadata on 26080 and its metrics on 26090, the ports the space file's
# kv-transfer config names.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/env.sh"
NODE_IP=$(pit2_node_ip)
export NODE_IP NODE_IF=${NODE_IF:-eno0} PATH=$here/bin:$PATH
export RUN=$PIT2_DATA/store/${2:-manual}
recipe=$(mktemp -d)
trap 'rm -rf "$recipe"' EXIT
cp "$PIT2_TREE"/recipes/mesh/minimax_m31_pd_store/{env.sh,store.sh} "$recipe/"
sed -i -e "s|^S=/workspace/ATOM/|S=$PIT2_TREE/|" \
    -e "s|{print int(\$2)}')\$|{print int(\$2)}' \|\| true)|" "$recipe/store.sh"
grep -q "^S=$PIT2_TREE/" "$recipe/store.sh" && grep -q "int(\$2)}' || true)" "$recipe/store.sh" ||
    { echo "store.sh: the recipe's store.sh changed; update the patch" >&2; exit 1; }
bash "$recipe/store.sh" "${1:-status}"
