#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The two containers of a compass advice search on pit2, from the host.
#
# Usage: containers.sh cpu NAME [CPUSET]   the search's container on this host
#        containers.sh gpu NODE NAME       a calibration's or real run's on pit2-p03-NODE
#
#   cpu   no device, the host's network, CPUSET as its cpuset (default 0-63,
#         which pit2-p03-g27 shares with other sessions' sim-speed timing: run
#         long jobs in it at nice 19), and an empty /sys/class/kfd, so nothing
#         in it sees the host's GPUs. A search simulates on CPU only.
#   gpu   every GPU, /dev/kfd, /dev/dri and the RDMA devices with the host's
#         ionic provider, the host's network and IPC. PIT2_WORK (venv-sim and
#         the atomesh build) is copied to the node first, at the same path. Hold
#         a gpool lease on the node before starting work in it (gpool_steps.sh).
#
# Both mount, each at its own path: PIT2_TREE and its git directory (a
# worktree's .git names the clone's), PIT2_WORK, PIT2_DATA, MODEL read-only,
# and each path in PIT2_MOUNTS (space-separated), such as the shared store,
# traces and network file a space file names. A container of that NAME already
# there is removed first: only ever pass a name of your own.
#
# Reads env.sh's variables (source it first, or set PIT2_DATA) and PIT2_IMAGE
# (default rocm/atom-dev:nightly_202611240650, the image the M3.1 laws were
# fitted under; its aiter keys the platform readings).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
IMAGE=${PIT2_IMAGE:-rocm/atom-dev:nightly_202611240650}
GITDIR=$(cd "$PIT2_TREE" && cd "$(git rev-parse --git-common-dir)" && pwd)

mounts=()
for p in "$PIT2_TREE" "$GITDIR" "$PIT2_WORK" "$PIT2_DATA" ${PIT2_MOUNTS-}; do
    mounts+=(-v "$p:$p")
done
mounts+=(-v "$MODEL:$MODEL:ro")

case ${1-} in
cpu)
    NAME=${2:?containers.sh cpu NAME [CPUSET]}
    mkdir -p "$PIT2_WORK" "$PIT2_DATA" "$PIT2_WORK/empty_kfd"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --init --name "$NAME" --network host --cpuset-cpus "${3:-0-63}" \
        "${mounts[@]}" -v "$PIT2_WORK/empty_kfd:/sys/class/kfd" "$IMAGE" sleep infinity
    ;;
gpu)
    NODE=pit2-p03-${2:?containers.sh gpu NODE NAME}
    NAME=${3:?containers.sh gpu NODE NAME}
    L=/usr/lib/x86_64-linux-gnu
    if [[ $(hostname) != "$NODE" ]]; then
        ssh "$NODE" "mkdir -p '$PIT2_WORK/mesh-target/release'"
        rsync -a "$PIT2_WORK/venv-sim" "$NODE:$PIT2_WORK/"
        rsync -a "$PIT2_WORK/mesh-target/release/atomesh" "$NODE:$PIT2_WORK/mesh-target/release/"
    fi
    # shellcheck disable=SC2029  # expanded here on purpose
    ssh "$NODE" "docker image inspect $IMAGE --format ok >/dev/null 2>&1 || docker pull $IMAGE >/dev/null
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --init --name $NAME \
  --network host --ipc host --shm-size 64g --group-add video --group-add render \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_PTRACE --cap-add SYS_NICE --ulimit memlock=-1 \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v $L/libibverbs/libionic-rdmav34.so:$L/libibverbs/libionic-rdmav34.so:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v $L/libionic.so.1.1.54.0-187:$L/libionic.so.1:ro \
  ${mounts[*]} $IMAGE sleep infinity"
    ;;
*)
    sed -n '4,7p' "$0"
    exit 2
    ;;
esac
