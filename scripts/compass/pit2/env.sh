# SPDX-License-Identifier: MIT
#
# Sourced in a pit2 container, the search's CPU one or a calibration's GPU one,
# before any step of a `compass advice` search on pit2 (the compass-advice
# skill). It sets what a search, a calibration and a real run read, and defines
# `compass`, the tree's own console script.
#
# Each of these may be set before sourcing; the default otherwise:
#   PIT2_TREE    the ATOM clone this file belongs to
#   PIT2_WORK    a node-local directory holding the simulated client's Python
#                (venv-sim) and the atomesh build (mesh-target), by bootstrap.sh:
#                /var/tmp/compass-pit2-<uid of the tree's owner>
#   PIT2_DATA    a shared directory of your own for logs, real runs, Store runs
#                and saved calibration commands; no default, as it is yours
#   MODEL        /share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4
#   COMPASS_INTEGRATION_REF  origin/pit2/m31pd: the head whose ATOM digest the
#                reuse check compares this tree's with; at its default,
#                origin/feature/atomcompass_new, the pit2 line's tree is refused
#
# It exports HARNESS_PYTHON (a simulated point's client), AIPERF_PYTHON (a real
# client), ATOMESH (the router a deployment runs) and AITER_LOG_LEVEL=WARNING.
#
# It never sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS or MKL_NUM_THREADS, and
# no shell a search or a calibration runs from may: a law's key holds the
# engine's performance variables, so under them a search looks every law up at
# another key and finds none. Courtesy on a shared CPU node is `nice -n 19` and
# the container's cpuset (containers.sh), not thread caps.

PIT2_TREE=${PIT2_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PIT2_WORK=${PIT2_WORK:-/var/tmp/compass-pit2-$(stat -c %u "$PIT2_TREE")}
: "${PIT2_DATA:?set PIT2_DATA to a shared directory of your own, such as /share_nfs/<you>/compass-pit2}"
export PIT2_TREE PIT2_WORK PIT2_DATA
export MODEL=${MODEL:-/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4}
export HARNESS_PYTHON=${HARNESS_PYTHON:-$PIT2_WORK/venv-sim/bin/python}
export AIPERF_PYTHON=${AIPERF_PYTHON:-/opt/agentx-venv/bin/python}
export ATOMESH=${ATOMESH:-$PIT2_WORK/mesh-target/release/atomesh}
export AITER_LOG_LEVEL=WARNING
export COMPASS_INTEGRATION_REF=${COMPASS_INTEGRATION_REF:-origin/pit2/m31pd}
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS; do
    [[ -z ${!v-} ]] || echo "env.sh: $v is set; unset it, a law's key holds it" >&2
done
compass() { "$PIT2_TREE/scripts/compass/compass" "$@"; }
# git runs as root on your files in a container: the reuse check reads the head.
git config --global --add safe.directory "$PIT2_TREE" 2>/dev/null || true
# The address on the interface of the default route (eno0 on pit2-p03): a
# calibration's Store and a real run's endpoints are this node's.
pit2_node_ip() {
    python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.19.0.1', 9)); print(s.getsockname()[0])"
}
