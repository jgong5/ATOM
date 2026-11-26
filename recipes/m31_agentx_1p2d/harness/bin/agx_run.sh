#!/usr/bin/env bash
# agx_run.sh: one MiniMax-M3.1 NVFP4 P/D point (1P1D / 1P2D / 1P3D) on gpool-leased nodes,
# run through main's CI launcher (pd_matrix.py -> pd_submit.sh export block ->
# pd_slurm_job.sh --spur-worker on every node -> pd_server_atom.sh), so its results are
# CI-format. Run it from the dev VM (it ssh-es to the nodes); see ../../README.md.
#
# Required: --tag T --commit REF --topo 1p1d|1p2d|1p3d --nodes gA[,gB] --lease L1[,L2] --conc N
# Common:   --mode agentx|gsm8k  --duration S (3600)  --warmup K (10/lane)  --al 4.04|off
#           --image IMG (default: $AGX_IMAGE_CHOICE's image= line, else rocm/atom-dev:m31-agx-local-c2f06e01)
#           --p-env F --d-env F --common-env F   role env files (bash; AGX_ARGS = extra server flags)
#           --overlay F (repeat)    sourced last in every rank's env (ATOMESH_*_ENV_* overrides)
#           --node-overlay R:F      sourced last in rank R's env only (R=0 node A, 1 node B)
#           --patch F (repeat)      git-apply F onto the run's snapshot (code A/B without a commit)
#           --store-caps P/D/FREE   GiB cap of owners per NUMA node hosting P / a D / no role (1024/768/1152)
#           --store-pools per_nic|shared  --evict HIGH/RATIO  --draft-window 0|1  --pp-inflight N|''
#           --common-args/--p-args/--d-args "..."  --p-seqs N --d-seqs N --mem 0.9 --router-policy P
#           --gsm8k-limit N (50) --gsm8k-conc C (32)  --stall-min M (25)  --cq-max K (3)  --label L
#           --build-mesh (build atomesh from the snapshot instead of the image's)  --dry-run
# AGX_PORT_OFFSET=K|auto (default 0): every launcher port + K (the recipe's combined_port_offset.patch for the combined/
#           benchmark phase, K + 1000 for an eval phase); auto takes the first K of 0, 100, ..., 2900 that every node's
#           probe finds free (the eval phase's ports only when the cell runs one). run.json states port_offset.
#           --keep-model-cache  stop the launcher's model page-cache dropper once every Store pool counts its
#                               owners (it re-evicts the weights every 45 s for 30 min; startup only)
# AGX_LOCAL_RUNS=1: no run dir the nodes share. Run this ON node A (rank 0) with AGX_RUNS and AGX_REPO on its local
#           disk: the harness, snapshot and run dir are copied to each peer at the same paths, its pid files fetched,
#           and the run dir kept in step by a sync loop (peers' files pulled, STOP and rank-rc-0 pushed, every
#           AGX_SYNC_S s, 15), with a last pull after the teardown. --dry-run copies and syncs once, starts nothing.
set -euo pipefail
H=${AGX_HARNESS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${AGX_REPO:-$(git -C "${H}" rev-parse --show-toplevel)}
RUNS=${AGX_RUNS:?set AGX_RUNS to a run-log dir that the nodes mount at the same path}
LOCAL_RUNS=${AGX_LOCAL_RUNS:-0}
LAUNCHER_REF=${AGX_LAUNCHER_REF:-bd78aa2f4bc6fc5d79201b6039306fcc2d52ae1f}
LAUNCHER_BASE=${AGX_LAUNCHER_BASE:-49e3aee9a}
GPOOL_STATE=${AGX_GPOOL_STATE:-}
GPOOL_CLI=${AGX_GPOOL:-$(dirname "${GPOOL_STATE:-/nonexistent/state.json}")/bin/gpool}
GPOOL_USER=${GPOOL_USER:-${USER:-$(id -un)}}; GPOOL_USER=${GPOOL_USER%%@*}
HOST_PREFIX=${AGX_HOST_PREFIX-pit2-p03-}

TAG="" COMMIT="" TOPO="" NODES="" LEASES="" CONC="" MODE=agentx DURATION=3600 WARMUP=10 AL=4.04
IMAGE="" P_ENV="" D_ENV="" C_ENV="" STORE_CAPS="1024/768/1152" STORE_POOLS=per_nic EVICT="" DRAFT_WINDOW=1
PP_INFLIGHT=8 COMMON_ARGS="--index-cache-dtype nvfp4 --spec-decode-kv-cache-dtype fp8" P_ARGS="" D_ARGS=""
P_SEQS=0 D_SEQS=0 MEM=0.9 ROUTER_POLICY="" GSM8K_LIMIT=50 GSM8K_CONC=32 STALL_MIN=25 CQ_MAX=3 LABEL=""
BUILD_MESH=false DRY_RUN=0 KEEP_MODEL_CACHE=0
OVERLAYS=() NODE_OVERLAYS=() PATCHES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --commit) COMMIT=$2; shift 2 ;;
    --topo) TOPO=$2; shift 2 ;;
    --nodes) NODES=$2; shift 2 ;;
    --lease) LEASES=$2; shift 2 ;;
    --conc) CONC=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --duration) DURATION=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --al) AL=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --p-env) P_ENV=$(readlink -f "$2"); shift 2 ;;
    --d-env) D_ENV=$(readlink -f "$2"); shift 2 ;;
    --common-env) C_ENV=$(readlink -f "$2"); shift 2 ;;
    --overlay) OVERLAYS+=("$(readlink -f "$2")"); shift 2 ;;
    --node-overlay) NODE_OVERLAYS+=("${2%%:*}:$(readlink -f "${2#*:}")"); shift 2 ;;
    --patch) PATCHES+=("$(readlink -f "$2")"); shift 2 ;;
    --store-caps) STORE_CAPS=$2; shift 2 ;;
    --store-pools) STORE_POOLS=$2; shift 2 ;;
    --evict) EVICT=$2; shift 2 ;;
    --draft-window) DRAFT_WINDOW=$2; shift 2 ;;
    --pp-inflight) PP_INFLIGHT=$2; shift 2 ;;
    --common-args) COMMON_ARGS=$2; shift 2 ;;
    --p-args) P_ARGS=$2; shift 2 ;;
    --d-args) D_ARGS=$2; shift 2 ;;
    --p-seqs) P_SEQS=$2; shift 2 ;;
    --d-seqs) D_SEQS=$2; shift 2 ;;
    --mem) MEM=$2; shift 2 ;;
    --router-policy) ROUTER_POLICY=$2; shift 2 ;;
    --gsm8k-limit) GSM8K_LIMIT=$2; shift 2 ;;
    --gsm8k-conc) GSM8K_CONC=$2; shift 2 ;;
    --stall-min) STALL_MIN=$2; shift 2 ;;
    --cq-max) CQ_MAX=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --build-mesh) BUILD_MESH=true; shift ;;
    --keep-model-cache) KEEP_MODEL_CACHE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "${TAG}" && -n "${COMMIT}" && -n "${TOPO}" && -n "${NODES}" && -n "${CONC}" ]] || { sed -n '2,21p' "$0"; exit 2; }
case "${TOPO}" in 1p1d) NN=1 ;; 1p2d|1p3d) NN=2 ;; *) echo "bad --topo ${TOPO}" >&2; exit 2 ;; esac
[[ "${MODE}" == gsm8k ]] && AL=real   # accuracy always runs real acceptance (the launcher drops it too)
if [[ -z "${IMAGE}" ]]; then
  IMAGE=$(awk -F= '/^image=/{print $2; exit}' "${AGX_IMAGE_CHOICE:-/dev/null}" 2>/dev/null || true)
  IMAGE=${IMAGE:-rocm/atom-dev:m31-agx-local-c2f06e01}
fi
IFS=, read -r -a NODE_SHORT <<< "${NODES}"
[[ ${#NODE_SHORT[@]} -eq ${NN} ]] || { echo "${TOPO} needs ${NN} node(s), got ${NODES}" >&2; exit 2; }
HOSTS=()
for n in "${NODE_SHORT[@]}"; do
  n=${n#"${HOST_PREFIX}"}; HOSTS+=("${HOST_PREFIX}${n}")
done
NODE_SHORT=("${HOSTS[@]#"${HOST_PREFIX}"}")

STAMP=$(date -u +%Y%m%d%H%M%S)
RUN="${RUNS}/${TAG}-${STAMP}"
SNAP="${RUNS}/src/${TAG}-${STAMP}"
JID="77${STAMP:6:8}"   # local job id: container names atomesh-<cell>-<JID>-<rank>
LOG_ROOT="${RUN}/logs"
mkdir -p "${RUN}" "${LOG_ROOT}" "${RUNS}/src"
exec > >(tee -a "${RUN}/driver.log") 2>&1
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
say "run ${RUN} tag=${TAG} topo=${TOPO} nodes=${HOSTS[*]} conc=${CONC} mode=${MODE} image=${IMAGE}"

# AGX_LOCAL_RUNS=1: node A holds the run root; each peer gets the same paths by copy, and the sync loop keeps the
# run dir in step (the launcher's ranks read each other's rank-rc-<r> there).
PEERS=("${HOSTS[@]:1}")
peer_copy() {  # DIR...: each dir to every peer at the same path
  local d h
  for h in "${PEERS[@]+"${PEERS[@]}"}"; do
    for d in "$@"; do
      ssh -o BatchMode=yes "${h}" "mkdir -p '${d}'" && rsync -a "${d}/" "${h}:${d}/" || return 1
    done
  done
}
sync_once() {  # pull every peer's run dir (its ranks' files), then push node A's control files
  local h
  for h in "${PEERS[@]+"${PEERS[@]}"}"; do
    rsync -a --update "${h}:${RUN}/" "${RUN}/" 2>/dev/null || true
    rsync -a --update --include='STOP' --include='logs/' --include='logs/slurm_job-*/' \
      --include='logs/slurm_job-*/rank-rc-0' --exclude='*' "${RUN}/" "${h}:${RUN}/" 2>/dev/null || true
  done
}
peer_alive() {  # R HOST: rank R's containers running on HOST and its launcher worker alive, as one line
  local cs pid
  cs=$(ssh -o BatchMode=yes "$2" "docker ps --format '{{.Names}}' | grep -E '^atomesh-${CELL_ID//./\\.}-${JID}-$1(-benchmark|-eval)?\$' | paste -sd, -" 2>/dev/null)
  pid=$(ssh -o BatchMode=yes "$2" "p=\$(cat ${RUN}/rank-$1.pid 2>/dev/null); [ -n \"\$p\" ] && [ -d /proc/\$p ] && echo \$p" 2>/dev/null)
  [[ -n "${cs}" && -n "${pid}" ]] && echo "containers ${cs}, worker pid ${pid}"
}
sync_verify() {  # every file a peer wrote since the copy is on node A at its size; RUN/sync_verify.txt
  local h r try bad list
  for try in 1 2 3 4; do
    bad=0; : > "${RUN}/sync_verify.txt"
    for r in "${!PEERS[@]}"; do
      h=${PEERS[r]}
      list=$(ssh -o BatchMode=yes "${h}" "cd ${RUN} && find . -type f -newer .copied_at -printf '%P\t%s\n'" 2>/dev/null) ||
        { echo "MISMATCH ${h}: cannot list its run dir" >> "${RUN}/sync_verify.txt"; bad=1; continue; }
      python3 "${RUN}/.sync_check.py" "${RUN}" "${h}" <<<"${list}" >> "${RUN}/sync_verify.txt" || bad=1
    done
    (( bad == 0 )) && break
    (( try < 4 )) && { sleep 5; sync_once; }
  done
  say "AGX_LOCAL_RUNS: sync check (${try} pass(es)): $(paste -sd'; ' "${RUN}/sync_verify.txt")"
  return ${bad}
}
if [[ "${LOCAL_RUNS}" == 1 ]]; then
  me=$(hostname -s)
  [[ "${me}" == "${HOSTS[0]}" || "${HOST_PREFIX}${me}" == "${HOSTS[0]}" || "${me}" == "${NODE_SHORT[0]}" ]] ||
    { say "AGX_LOCAL_RUNS=1 runs on node A (${HOSTS[0]}), not on ${me}"; exit 2; }
  command -v rsync > /dev/null || { say "AGX_LOCAL_RUNS=1 needs rsync"; exit 2; }
  peer_copy "${H}" || { say "AGX_LOCAL_RUNS: cannot copy the harness ${H} to ${PEERS[*]}"; exit 4; }
  say "AGX_LOCAL_RUNS: run root ${RUNS} on ${HOSTS[0]}; harness copied to ${PEERS[*]:-no peer}"
fi

# 1. The leases must cover every GPU of every node; their est_end bounds the run.
DEADLINE=0
if [[ "${AGX_NO_LEASE_CHECK:-0}" != "1" ]]; then
  [[ -n "${GPOOL_STATE}" ]] || { say "set AGX_GPOOL_STATE (the GPU pool's state.json) or AGX_NO_LEASE_CHECK=1"; exit 3; }
  DEADLINE=$(python3 - "${GPOOL_STATE}" "${GPOOL_USER}" "${LEASES}" "${NODE_SHORT[@]}" <<'PY'
import json, sys, time
state, user = json.load(open(sys.argv[1])), sys.argv[2]
leases, nodes = [x for x in sys.argv[3].split(",") if x], sys.argv[4:]
cover, ends = {}, []
for lid in leases:
    lease = state["leases"].get(lid)
    if lease is None or lease.get("user") != user:
        sys.exit(f"lease {lid} is not a live lease of {user}")
    for a in lease["alloc"]:
        cover.setdefault(a["node"], set()).update(a["gpus"])
    if lease.get("est_end"):
        ends.append(lease["est_end"])
for n in nodes:
    if cover.get(n, set()) != set(range(8)):
        sys.exit(f"node {n}: leases {leases} cover GPUs {sorted(cover.get(n, []))}, need 0-7")
print(int(min(ends)) if ends else 0)
PY
  ) || { say "lease check failed"; exit 3; }
fi
say "leases ${LEASES:-none} ok; deadline $( ((DEADLINE)) && date -u -d @"${DEADLINE}" +%H:%M:%S || echo none)"

# 2. Snapshot of the commit, plus the launcher change when the commit lacks it, plus patches.
SHA=$(git -C "${REPO}" rev-parse "${COMMIT}^{commit}")
git -C "${REPO}" worktree add --detach "${SNAP}" "${SHA}" >/dev/null 2>&1
LAUNCHER_TIP=$(git -C "${REPO}" rev-parse "${LAUNCHER_REF}^{commit}")
LAUNCHER_OVERLAY=false
if ! git -C "${REPO}" merge-base --is-ancestor "${LAUNCHER_TIP}" "${SHA}"; then
  git -C "${REPO}" diff "${LAUNCHER_BASE}" "${LAUNCHER_TIP}" -- .github/scripts/atomesh | git -C "${SNAP}" apply
  LAUNCHER_OVERLAY=true
fi
for p in "${PATCHES[@]+"${PATCHES[@]}"}"; do git -C "${SNAP}" apply "${p}"; cp "${p}" "${RUN}/"; done
git -C "${SNAP}" diff > "${RUN}/snapshot.diff"
say "snapshot ${SNAP} at ${SHA} launcher_overlay=${LAUNCHER_OVERLAY} patches=${#PATCHES[@]}"

# 3. Node probes: the page cache of every model goes first (as the launcher's dropper does),
#    then free DRAM, GPUs, containers, NICs and the image.
IPS=()
for i in "${!HOSTS[@]}"; do
  h=${HOSTS[i]}
  drop=${AGX_PROBE_DROP-/share_nfs/models}
  ssh -o BatchMode=yes "${h}" "python3 ${H}/bin/node_probe.py --image ${IMAGE} ${drop:+--drop ${drop}}" > "${RUN}/probe-${NODE_SHORT[i]}.json"
  IPS+=("$(ssh -o BatchMode=yes "${h}" "ip route get 1.1.1.1 | awk '/src/ {for (i=1;i<NF;i++) if (\$i==\"src\") print \$(i+1); exit}'")")
  python3 - "${RUN}/probe-${NODE_SHORT[i]}.json" "${h}" <<'PY' || { say "preflight failed on ${h}"; exit 4; }
import json, os, sys
p, host = json.load(open(sys.argv[1])), sys.argv[2]
bad = []
if p["containers"] is None:
    bad.append("docker not reachable")
elif p["containers"]:
    # Another user's idle container is left alone; GPUs in use or our ports taken refuse the run.
    print(f"  {host}: WARNING foreign containers present (left untouched): " + ", ".join(c["name"] for c in p["containers"]))
# The launcher's ports at offset 0 (an eval phase's 1000 up): reported here, chosen around after every probe
# (AGX_PORT_OFFSET), which refuses only when no offset is free on every node.
ours = {8000, 8010, 8011, 8020, 8021, 8022, 29100, 29500, 29600, 29700, 29800, 26051, 26052, 26053, 26054, 26055,
        26056, 26057, 26058, 26059, 26080, 26090, 26151, 26180, 26190, 26251, 26280, 26290, 26351, 26380, 26390}
ours |= set(range(6301, 6310))
ours |= {port + 1000 for port in ours}
taken = sorted(ours & set(p.get("listening", [])))
if taken:
    print(f"  {host}: launcher ports listening at offset 0 (AGX_PORT_OFFSET chooses around them): {taken}")
busy = [g["gpu"] for g in p["gpus"] if (g["vram_used_mb"] or 0) > 2048]
if busy:
    bad.append(f"GPUs {busy} hold VRAM")
if len(p["gpus"]) != 8:
    bad.append(f"{len(p['gpus'])} GPUs seen")
down = [d for d, v in p["rdma"].items() if not any(s.startswith("4:") for s in v["ports"])]
if down:
    bad.append(f"RDMA ports not ACTIVE: {down}")
for n, m in p["numa"].items():
    print(f"  {host} numa{n}: total {m['total_gib']} free {m['free_gib']} file {m['file_gib']} big {m['big_free_gib']} GiB")
if bad and os.environ.get("AGX_PREFLIGHT_WARN_ONLY") == "1":
    print(f"  {host}: WARNING (dry run, not refused): " + "; ".join(bad))
elif bad:
    sys.exit(f"{host}: " + "; ".join(bad))
print(f"  {host}: image present={p['image_present']}")
PY
  if [[ "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image_present"])' "${RUN}/probe-${NODE_SHORT[i]}.json")" != "True" ]]; then
    say "pulling ${IMAGE} on ${h}"
    if ! ssh -o BatchMode=yes "${h}" "docker pull -q ${IMAGE}" >/dev/null; then
      say "${IMAGE} is not on ${h} and cannot be pulled (a local tag?): use a node that has it, or --image"
      git -C "${REPO}" worktree remove --force "${SNAP}" || true
      exit 5
    fi
  fi
done

# 4. Store owners sized from free DRAM; 5. the cell.
probe_args=()
for i in "${!HOSTS[@]}"; do probe_args+=(--probe "${HOSTS[i]}=${RUN}/probe-${NODE_SHORT[i]}.json"); done
IFS=/ read -r CAP_P CAP_D CAP_FREE <<< "${STORE_CAPS}"
# AGX_STORE_POOL_GIB: each per_nic pool's exact GiB, spread over both nodes as their huge-page memory allows (in
# place of a fixed owner overlay); a plan the nodes cannot hold refuses here, before anything starts.
python3 "${H}/bin/store_plan.py" --topo "${TOPO}" --pools "${STORE_POOLS}" "${probe_args[@]}" --cap-p "${CAP_P}" --cap-d "${CAP_D}" --cap-free "${CAP_FREE}" \
  ${AGX_STORE_POOL_GIB:+--pool-gib "${AGX_STORE_POOL_GIB}"} > "${RUN}/store_plan.json" || { say "store plan refused (AGX_STORE_POOL_GIB=${AGX_STORE_POOL_GIB:-})"; exit 4; }
say "store plan: $(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["total_gib"],"GiB | A:",d["owners"],"| B:",d["decode_owners"] or "-")' "${RUN}/store_plan.json")"
python3 "${H}/bin/cellgen.py" --snap "${SNAP}" --run "${RUN}" --topo "${TOPO}" --nodes "$(IFS=,; echo "${HOSTS[*]}")" \
  --image "${IMAGE}" --conc "${CONC}" --mode "${MODE}" --duration "${DURATION}" --warmup-per-lane "${WARMUP}" --al "${AL}" \
  --store-plan "${RUN}/store_plan.json" --store-pools "${STORE_POOLS}" --evict "${EVICT}" --draft-window "${DRAFT_WINDOW}" \
  --pp-inflight "${PP_INFLIGHT}" ${P_ENV:+--p-env "${P_ENV}"} ${D_ENV:+--d-env "${D_ENV}"} ${C_ENV:+--common-env "${C_ENV}"} \
  --common-args "${COMMON_ARGS}" --p-args "${P_ARGS}" --d-args "${D_ARGS}" --p-seqs "${P_SEQS}" --d-seqs "${D_SEQS}" \
  --mem "${MEM}" --router-policy "${ROUTER_POLICY}" --gsm8k-limit "${GSM8K_LIMIT}" --gsm8k-conc "${GSM8K_CONC}" --label "${LABEL}"
CELL_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["id"])' "${RUN}/cell.json")
# Files a role env names outside what the containers mount (/share_nfs/models, the snapshot, the job
# dir) are copied into the job dir and the env points at the copy (e.g. recipe-audit's AITER_CONFIG_* CSVs).
mkdir -p "${LOG_ROOT}/slurm_job-${JID}/agx-files"
python3 - "${RUN}/cell.json" "${LOG_ROOT}/slurm_job-${JID}" "${JID}" <<'PY'
import json, os, shutil, sys
path, job, jid = sys.argv[1:4]
cell = json.load(open(path))
moved = []
for role, env in cell["env"].items():
    for k, v in env.items():
        if v.startswith("/") and not v.startswith(("/share_nfs/models/", "/workspace/", "/run_logs/", "/tmp/")) \
                and os.path.isfile(v):
            dst = os.path.join(job, "agx-files", os.path.basename(v))
            shutil.copy2(v, dst)
            env[k] = f"/run_logs/slurm_job-{jid}/agx-files/{os.path.basename(v)}"
            moved.append(f"{role}.{k}")
json.dump(cell, open(path, "w"), indent=2)
if moved:
    print("  copied into the job dir for the containers: " + ", ".join(sorted(set(moved))))
PY

# 5b. The port offset (AGX_PORT_OFFSET): every launcher port + K on every node, the first free K under auto.
PORT_OFFSET=$(python3 - "${RUN}" "${AGX_PORT_OFFSET:-0}" "${NODE_SHORT[@]}" <<'PY'
import json, sys
run, want, nodes = sys.argv[1], sys.argv[2], sys.argv[3:]
cell = json.load(open(f"{run}/cell.json"))
base = {8000, 8010, 8011, 8020, 8021, 8022, 29100, 29500, 29600, 29700, 29800, 26051, 26052, 26053, 26054, 26055,
        26056, 26057, 26058, 26059, 26080, 26090, 26151, 26180, 26190, 26251, 26280, 26290, 26351, 26380, 26390}
base |= set(range(6301, 6310))
phases = [0] + ([1000] if cell.get("run_eval") or cell.get("eval_only") else [])
listen = {n: set(json.load(open(f"{run}/probe-{n}.json")).get("listening", [])) for n in nodes}
need = lambda k: {q + k + ph for q in base for ph in phases}  # noqa: E731
for k in (range(0, 2901, 100) if want == "auto" else [int(want)]):
    if max(need(k)) >= 32768:  # the ephemeral range: an outgoing connection may hold one
        continue
    hit = {n: sorted(need(k) & ports) for n, ports in listen.items() if need(k) & ports}
    if not hit:
        print(k)
        sys.exit(0)
    print(f"  offset {k}: taken {hit}", file=sys.stderr)
sys.exit(f"no port offset ({want}) leaves the launcher's ports free on {nodes}")
PY
) || { say "port choice failed (AGX_PORT_OFFSET=${AGX_PORT_OFFSET:-0})"; exit 4; }
if (( PORT_OFFSET > 0 )); then
  offset_patch="$(cd "${H}/.." && pwd)/patches/combined_port_offset.patch"
  if git -C "${SNAP}" apply --check "${offset_patch}" 2>/dev/null; then
    git -C "${SNAP}" apply "${offset_patch}"; cp "${offset_patch}" "${RUN}/"
    git -C "${SNAP}" diff > "${RUN}/snapshot.diff"
  elif ! git -C "${SNAP}" apply --check -R "${offset_patch}" 2>/dev/null; then
    say "port offset ${PORT_OFFSET}: ${offset_patch} applies neither way to the snapshot"; exit 4
  fi
fi
say "ports: offset ${PORT_OFFSET} (AGX_PORT_OFFSET=${AGX_PORT_OFFSET:-0}; router $((8000 + PORT_OFFSET)), P $((8010 + PORT_OFFSET)), D $((8020 + PORT_OFFSET)))"

# 6. Rank env: the CI export block of pd_submit.sh, the Spur worker's job env, overlays.
cell_json=$(cat "${RUN}/cell.json")
{
  printf 'export CELL_JSON=%q\n' "${cell_json}"
  awk '/^python3 - <<.PY.$/{f=1;next} f&&/^PY$/{exit} f' "${SNAP}/.github/scripts/atomesh/pd_submit.sh" | CELL_JSON="${cell_json}" python3 -
  cat <<EOV
export ATOMESH_BUILD_MESH=${BUILD_MESH} ATOMESH_MESH_BUILD_PROFILE=release ATOMESH_DASHBOARD_HARDWARE=MI355X
export ATOMESH_HOST_CACHE_DROP_TIMEOUT=900
export ATOMESH_COMBINED_PORT_OFFSET=${PORT_OFFSET} ATOMESH_RESTART_PORT_OFFSET=$((PORT_OFFSET + 1000))
export ATOMESH_RUN_TOKEN=$(python3 -c 'import uuid;print(uuid.uuid4().hex)')
export LOG_ROOT=${LOG_ROOT}
export GITHUB_WORKSPACE=${SNAP}
export SPUR_JOB_ID=${JID}
export SPUR_PEER_NODES=$(IFS=,; echo "${IPS[*]}")
export SPUR_NODELIST=$(IFS=,; echo "${HOSTS[*]}")
# pd_slurm_job.sh pulls the image before each container; a local-only tag (the campaign image) has no registry.
export PATH=${H}/bin/docker-wrap:\${PATH}
EOV
} > "${RUN}/rank-env.sh"
for r in "${!HOSTS[@]}"; do
  {
    echo "source ${RUN}/rank-env.sh"
    echo "export SPUR_TASK_OFFSET=${r}"
    # 1P2D: node B's one decode runs on GPUs 4-7 (NUMA1, rdma4-7), as node A's does.
    if [[ "${TOPO}" == 1p2d && "${r}" == 1 ]]; then
      echo "export ATOMESH_DECODE_ENV_ROCR_VISIBLE_DEVICES=4,5,6,7"
    fi
    for o in "${OVERLAYS[@]+"${OVERLAYS[@]}"}"; do echo "source ${o}"; done
    for o in "${NODE_OVERLAYS[@]+"${NODE_OVERLAYS[@]}"}"; do
      if [[ "${o%%:*}" == "${r}" ]]; then echo "source ${o#*:}"; fi
    done
  } > "${RUN}/rank-${r}.env.sh"
  for o in "${OVERLAYS[@]+"${OVERLAYS[@]}"}" "${NODE_OVERLAYS[@]+"${NODE_OVERLAYS[@]#*:}"}"; do cp -n "${o}" "${RUN}/" 2>/dev/null || true; done
done
python3 - "${RUN}" <<PY
import json, sys
json.dump({
  "tag": "${TAG}", "stamp": "${STAMP}", "commit": "${SHA}", "commit_ref": "${COMMIT}", "image": "${IMAGE}",
  "topo": "${TOPO}", "hosts": "$(IFS=,; echo "${HOSTS[*]}")".split(","), "ips": "$(IFS=,; echo "${IPS[*]}")".split(","),
  "leases": "${LEASES}", "deadline": ${DEADLINE}, "jid": "${JID}", "cell_id": "${CELL_ID}", "conc": ${CONC},
  "mode": "${MODE}", "duration": ${DURATION}, "warmup_per_lane": ${WARMUP}, "al": "${AL}",
  "snapshot": "${SNAP}", "launcher_overlay": ${LAUNCHER_OVERLAY^}, "launcher_tip": "${LAUNCHER_TIP}",
  "patches": "${PATCHES[*]+${PATCHES[*]}}", "overlays": "${OVERLAYS[*]+${OVERLAYS[*]}}",
  "node_overlays": "${NODE_OVERLAYS[*]+${NODE_OVERLAYS[*]}}", "p_env": "${P_ENV}", "d_env": "${D_ENV}",
  "common_env": "${C_ENV}", "store_caps": "${STORE_CAPS}", "store_pools": "${STORE_POOLS}",
  "log_root": "${LOG_ROOT}", "job_dir": "${LOG_ROOT}/slurm_job-${JID}", "stall_min": ${STALL_MIN}, "cq_max": ${CQ_MAX},
  "keep_model_cache": ${KEEP_MODEL_CACHE}, "port_offset": ${PORT_OFFSET},
}, open(sys.argv[1] + "/run.json", "w"), indent=1)
PY
if [[ "${LOCAL_RUNS}" == 1 ]]; then
  cat > "${RUN}/.sync_check.py" <<'PY'
import os, sys
run, host = sys.argv[1], sys.argv[2]
want = [l.split("\t") for l in sys.stdin.read().splitlines() if "\t" in l]
missing, size = [], []
for rel, n in want:
    p = os.path.join(run, rel)
    if not os.path.isfile(p):
        missing.append(rel)
    elif os.path.getsize(p) != int(n):
        size.append(f"{rel} {os.path.getsize(p)}!={n}")
total = sum(int(n) for _, n in want)
if missing or size:
    print(f"MISMATCH {host}: {len(missing)} missing {missing[:5]}, {len(size)} sized apart {size[:5]} of {len(want)} files")
    sys.exit(1)
print(f"OK {host}: {len(want)} files, {total} bytes, each on node A at its size")
PY
  peer_copy "${SNAP}" "${RUN}" || { say "AGX_LOCAL_RUNS: cannot copy the snapshot and run dir to ${PEERS[*]}"; exit 4; }
  for h in "${PEERS[@]+"${PEERS[@]}"}"; do ssh -o BatchMode=yes "${h}" "touch ${RUN}/.copied_at"; done
  sync_once
  for h in "${PEERS[@]+"${PEERS[@]}"}"; do
    say "AGX_LOCAL_RUNS: ${h} holds $(ssh -o BatchMode=yes "${h}" "ls ${RUN} | wc -l") run files, snapshot $(ssh -o BatchMode=yes "${h}" "git -C ${SNAP} rev-parse HEAD 2>/dev/null || cat ${SNAP}/.git 2>/dev/null | head -c 0; ls ${SNAP}/.github/scripts/atomesh/pd_slurm_job.sh > /dev/null && echo present")"
  done
fi
if [[ "${DRY_RUN}" == 1 ]]; then
  say "dry run: cell ${CELL_ID}; nothing started"
  git -C "${REPO}" worktree remove --force "${SNAP}" || true
  exit 0
fi

# 7. Samplers, ranks, scraper.
for i in "${!HOSTS[@]}"; do
  ssh -o BatchMode=yes "${HOSTS[i]}" "setsid nohup bash ${H}/bin/nic_sampler.sh ${RUN}/nic-${NODE_SHORT[i]}.csv ${RUN}/STOP > /dev/null 2>&1 < /dev/null & echo \$! > ${RUN}/nic-${NODE_SHORT[i]}.pid"
done
for (( r = ${#HOSTS[@]} - 1; r >= 0; r-- )); do
  # Only the worker goes to the background (a backgrounded '&&' list would keep ssh's channel open).
  ssh -o BatchMode=yes "${HOSTS[r]}" "cd ${SNAP} && { setsid nohup bash -c 'source ${RUN}/rank-${r}.env.sh && exec bash .github/scripts/atomesh/pd_slurm_job.sh --spur-worker' > ${RUN}/rank-${r}.console 2>&1 < /dev/null & echo \$! > ${RUN}/rank-${r}.pid; }"
  # A peer's pid file is on the peer: fetch it (the teardown reads each on its own node).
  if [[ "${LOCAL_RUNS}" == 1 && "${r}" != 0 ]]; then
    ssh -o BatchMode=yes "${HOSTS[r]}" "cat ${RUN}/rank-${r}.pid" > "${RUN}/rank-${r}.pid"
  fi
  say "rank ${r} started on ${HOSTS[r]} (pid $(cat "${RUN}/rank-${r}.pid"))"
done
ssh -o BatchMode=yes "${HOSTS[0]}" "setsid nohup python3 ${H}/bin/scrape.py --run ${RUN} > ${RUN}/scrape.log 2>&1 < /dev/null & echo \$! > ${RUN}/scrape.pid"
SYNC_PID=""
if [[ "${LOCAL_RUNS}" == 1 ]]; then
  ( while [[ ! -e "${RUN}/SYNC_STOP" ]]; do sync_once; sleep "${AGX_SYNC_S:-15}"; done ) > "${RUN}/sync.log" 2>&1 &
  SYNC_PID=$!
  say "AGX_LOCAL_RUNS: run dir sync with ${PEERS[*]:-no peer} every ${AGX_SYNC_S:-15} s (pid ${SYNC_PID})"
fi

# 8. Watch until every rank published its rc, or the watchdog stops the run.
JOB_DIR="${LOG_ROOT}/slurm_job-${JID}"
reason=""
last_yield=0
vram_checked=0
while :; do
  # Every 2 min until it passes: every GPU of every node holds > 10 GB VRAM (/health alone does
  # not prove a loaded model; the container log is block-buffered, so it cannot tell when).
  if (( vram_checked < ${#HOSTS[@]} && $(date +%s) - ${last_vram:-0} > 120 )); then
    last_vram=$(date +%s); vram_checked=0
    for i in "${!HOSTS[@]}"; do
      ssh -o BatchMode=yes "${HOSTS[i]}" "amd-smi monitor -v" > "${RUN}/vram-${NODE_SHORT[i]}.txt" 2>&1 || true
      loaded=$(awk 'NR>1 && $3+0 > 10240' "${RUN}/vram-${NODE_SHORT[i]}.txt" | wc -l)
      (( loaded == 8 )) && vram_checked=$((vram_checked + 1))
    done
    (( vram_checked == ${#HOSTS[@]} )) && say "VRAM: every GPU of ${HOSTS[*]} holds > 10 GB ($(awk 'NR>1 {printf "%s ", $3}' "${RUN}/vram-${NODE_SHORT[0]}.txt")MB on ${NODE_SHORT[0]})"
  fi
  # --keep-model-cache: once every pool counts its planned owners, the dropper's job is done (the owners
  # pinned their memory); stop it in this run's containers so the workers load the weights from page cache.
  if (( KEEP_MODEL_CACHE == 1 )) && python3 - "${RUN}" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
plan = json.loads((run / "store_plan.json").read_text())
want = {}
for spec in (plan.get("owners", ""), plan.get("decode_owners", "")):
    for e in filter(None, spec.split(";")):
        _, gib, nic = e.split(":")
        key = nic if plan.get("pools") == "per_nic" else "shared"
        want[key] = want.get(key, 0) + int(gib) * 2**30
names = sorted(want)
cap = {}
try:
    with open(run / "scrape.jsonl", "rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - (1 << 20)))
        for raw in f.read().splitlines()[1:]:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row["target"].startswith("store"):
                cap[int(row["target"][5:])] = row["m"].get("master_total_capacity_bytes", 0)
except OSError:
    sys.exit(1)
ok = len(cap) == len(names) and all(cap.get(i, 0) >= want[n] for i, n in enumerate(names))
sys.exit(0 if ok else 1)
PY
  then
    KEEP_MODEL_CACHE=2
    for i in "${!HOSTS[@]}"; do
      ssh -o BatchMode=yes "${HOSTS[i]}" "for c in \$(docker ps --format '{{.Names}}' | grep -E '^atomesh-${CELL_ID//./\\.}-${JID}-${i}(-benchmark|-eval)?\$'); do docker exec \$c sh -c 'for p in \$(pgrep -f \"[d]rop_page_cache.py\"); do kill \$p; done'; done" || true
    done
    say "every Store pool counts its owners; stopped the model page-cache droppers (--keep-model-cache)"
  fi
  # Once the servers serve (AgentX warmup or lm_eval started), record VRAM again: the KV pools are allocated.
  if [[ ! -f "${RUN}/vram-serving.txt" ]] && { grep -qs 'Phase warmup (warmup) started' "${JOB_DIR}"/benchmark_results/aiperf-*/aiperf.log \
      || compgen -G "${JOB_DIR}/eval_results/*" >/dev/null; }; then
    for i in "${!HOSTS[@]}"; do
      echo "== ${HOSTS[i]}" >> "${RUN}/vram-serving.txt"
      ssh -o BatchMode=yes "${HOSTS[i]}" "amd-smi monitor -v" >> "${RUN}/vram-serving.txt" 2>&1 || true
    done
    say "VRAM while serving (MB): $(awk '/^==/{printf "%s ", $2} $1 ~ /^[0-9]+$/ {printf "%s ", $3}' "${RUN}/vram-serving.txt")"
  fi
  done_ranks=0
  for r in "${!HOSTS[@]}"; do [[ -f "${JOB_DIR}/rank-rc-${r}" ]] && done_ranks=$((done_ranks + 1)); done
  (( done_ranks == ${#HOSTS[@]} )) && break
  if reason=$(python3 "${H}/bin/watchdog.py" --run "${RUN}"); then
    :
  elif [[ "${LOCAL_RUNS}" == 1 ]] && { sync_once; reason2=$(python3 "${H}/bin/watchdog.py" --run "${RUN}"); }; then
    say "AGX_LOCAL_RUNS: the watchdog's stop (${reason}) cleared after a fresh pull from ${PEERS[*]}"
    echo "$(date -u +%FT%TZ) watchdog stop '${reason}' cleared after a fresh pull (sync lag)" >> "${RUN}/LOCAL_RUNS_NOTES"
  else
    if [[ "${LOCAL_RUNS}" == 1 ]]; then
      reason=${reason2:-${reason}}
      for r in "${!PEERS[@]}"; do
        if alive=$(peer_alive $((r + 1)) "${PEERS[r]}"); then
          say "AGX_LOCAL_RUNS: the watchdog stops the run (${reason}) while rank $((r + 1)) on ${PEERS[r]} is alive (${alive})"
          echo "$(date -u +%FT%TZ) watchdog stop '${reason}' while rank $((r + 1)) on ${PEERS[r]} was alive (${alive}): the local-runs fallback's stop, not a run failure" >> "${RUN}/LOCAL_RUNS_NOTES"
        fi
      done
    fi
    say "watchdog: ${reason}"; echo "${reason}" > "${RUN}/stopped_by_watchdog"
    bash "${H}/bin/agx_teardown.sh" "${RUN}" || true
    break
  fi
  if (( $(date +%s) - last_yield > 300 )); then
    last_yield=$(date +%s)
    "${GPOOL_CLI}" mine 2>/dev/null | grep -E "^(${LEASES//,/|}) " | grep -q YIELD \
      && say "gpool: a lease of this run is flagged YIELD; release it when this point ends" || true
  fi
  sleep 30
done
say "ranks done: $(for r in "${!HOSTS[@]}"; do printf 'rank%s=%s ' "${r}" "$(cat "${JOB_DIR}/rank-rc-${r}" 2>/dev/null || echo '?')"; done)"
for r in "${!HOSTS[@]}"; do
  rc=$(cat "${JOB_DIR}/rank-rc-${r}" 2>/dev/null || echo '?')
  if [[ "${rc}" != 0 ]]; then
    say "rank ${r} rc=${rc}; its failure lines:"
    grep -ahE '\[FAIL\]|ERROR|Error:|Traceback|exited|refus' "${JOB_DIR}/rank-${r}"/container*.log 2>/dev/null | tail -15 || true
    tail -5 "${RUN}/rank-${r}.console" 2>/dev/null || true
  fi
done

# 9. Teardown (samplers, scraper, any container left), summary, cleanup.
[[ "${LOCAL_RUNS}" == 1 ]] && { touch "${RUN}/STOP"; sync_once; }
bash "${H}/bin/agx_teardown.sh" "${RUN}" || true
if [[ "${LOCAL_RUNS}" == 1 ]]; then
  touch "${RUN}/SYNC_STOP"; [[ -n "${SYNC_PID}" ]] && wait "${SYNC_PID}" 2>/dev/null || true
  sync_once; say "AGX_LOCAL_RUNS: last pull from ${PEERS[*]:-no peer} done"
  sync_verify || say "AGX_LOCAL_RUNS: WARNING node A lacks some of a peer's files (RUN/sync_verify.txt)"
fi
python3 "${H}/bin/summarize.py" --run "${RUN}" || say "summary failed"
find "${JOB_DIR}" -name 'server_metrics_export.json' -size +1G -print -delete 2>/dev/null || true
git -C "${REPO}" worktree remove --force "${SNAP}" || true
say "done: ${RUN}/point.txt"
