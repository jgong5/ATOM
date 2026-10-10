#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One cell of cc-traces through agentx-harness on a simulated TP1 or 1P1D run:
# ATOM with --compass-run, driven by `aiperf profile` with the AgentX
# scenario and compass-harness, until the scenario's duration has passed on the
# simulated clock.
#
# Usage: cctraces_sim.sh RUN_FILE OUT_DIR
#   RUN_FILE        a run file; its out_dir and clock_endpoint are replaced
#   MODEL           model directory: config.json and the tokenizer (required)
#   TRACES          a cc-traces traces.jsonl (required)
#   HARNESS_PYTHON  the Python that has agentx-harness and compass-harness
#                   installed (required)
#   SESSIONS        replay the first SESSIONS traces (default 2)
#   CONCURRENCY     aiperf --concurrency (default 1)
#   DURATION_S      aiperf --benchmark-duration, simulated seconds (default 1800)
#   SEED            aiperf --random-seed (default 20260707)
#   SERVER_ARGS     more API server flags (default: --enforce-eager
#                   --max-model-len 262144)
#   DECODE_EXEC, ATOMESH  as pd_sim.sh takes them, for a 1P1D cell
#
# A run file with `router_s` and `kv_write_req_s` makes the cell 1P1D: the
# deployment is pd_sim.sh's, and aiperf drives its atomesh router.
#
# It runs its own tree, as the gates do, compass-harness included. OUT_DIR gets
# run.json (RUN_FILE plus a `workload` entry hashing the traces, the dataset
# and the aiperf arguments), the server and aiperf logs, aiperf's artifacts,
# and the step table and run summary. The last line is the cell's result, with
# the KV transfer count and mean simulated seconds for a 1P1D cell. It exits
# non-zero when aiperf or the deployment did, or when aiperf's benchmark id is
# not the one recorded.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
export PYTHONPATH=$ROOT:$ROOT/compass_harness
compass_require_tree "$ROOT" || exit $?
RUN_FILE=$(realpath "${1:?usage: cctraces_sim.sh RUN_FILE OUT_DIR}")
OUT=${2:?usage: cctraces_sim.sh RUN_FILE OUT_DIR}
MODEL=${MODEL:?MODEL names the model directory}
TRACES=${TRACES:?TRACES names a cc-traces traces.jsonl}
HARNESS_PYTHON=${HARNESS_PYTHON:?HARNESS_PYTHON names the Python with agentx-harness}
SESSIONS=${SESSIONS:-2}
SERVER_ARGS=${SERVER_ARGS:---enforce-eager --max-model-len 262144}
WAIT_S=${WAIT_S:-600}
# compass-harness pins aiperf's benchmark id: uuid4 is UUID(int=0).
BENCHMARK_ID=000000000000

mkdir -p "$OUT/traces"
OUT=$(realpath "$OUT")
harness=$(cd / && "$HARNESS_PYTHON" -c 'import compass_harness; print(compass_harness.__file__)')
[[ $harness == "$ROOT"/* ]] || { echo "cctraces_sim: compass_harness resolved to $harness" >&2; exit 92; }

head -n "$SESSIONS" "$TRACES" |
    awk -v d="$OUT/traces" '{f = sprintf("%s/%05d.json", d, NR - 1); print > f; close(f)}'

# Everything but paths and the URL, so it can be hashed into the run record.
args=(--endpoint-type chat --custom-dataset-type weka_trace
    --concurrency "${CONCURRENCY:-1}" --benchmark-duration "${DURATION_S:-1800}"
    --random-seed "${SEED:-20260707}" --scenario inferencex-agentx-mvp --unsafe-override
    --streaming --use-server-token-count --extra-inputs ignore_eos:true
    --no-server-metrics --no-gpu-telemetry --ui none)

read -r MODE CLOCK_PORT PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT < <(
python3 - "$RUN_FILE" "$OUT" "$TRACES" "$BENCHMARK_ID" "${args[@]}" <<'EOF'
import hashlib, json, socket, sys
from pathlib import Path

run_file, out, traces, benchmark_id, *args = sys.argv[1:]
out = Path(out)
# Held together, so no two are the same port.
socks = [socket.socket() for _ in range(6)]
for s in socks:
    s.bind(("127.0.0.1", 0))
ports = [s.getsockname()[1] for s in socks]
for s in socks:
    s.close()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


run = json.loads(Path(run_file).read_text())
run["out_dir"] = str(out)
run["clock_endpoint"] = f"tcp://127.0.0.1:{ports[0]}"
run["workload"] = {
    "dataset_sha256": sha256(traces),
    "traces_sha256": [sha256(p) for p in sorted((out / "traces").iterdir())],
    "aiperf_args": args,
    "benchmark_id": benchmark_id,
}
(out / "run.json").write_text(json.dumps(run))
print("1p1d" if "kv_write_req_s" in run else "tp1", *ports)
EOF
)

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
if [[ $MODE == 1p1d ]]; then
    # pd_sim.sh takes the authority's port from its own variable; the run file
    # names the same one for the traffic LP.
    export MODEL SERVER_ARGS WAIT_S CLOCK_PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT
    PREFILL_PORT=$PORT bash "$ROOT/scripts/compass/pd_sim.sh" "$OUT/run.json" >"$OUT/pd_sim.log" 2>&1 &
    ready=("pd_sim: ready" "$OUT/pd_sim.log")
    PORT=$ROUTER_PORT
else
    # The keep-alive timeout as in pd_sim.sh, aiperf's pool in the router's place.
    # shellcheck disable=SC2086 # SERVER_ARGS is a word list
    python3 -m atom.entrypoints.openai.api_server --model "$MODEL" --host 127.0.0.1 \
        --server-port "$PORT" --timeout-keep-alive 1000000 $SERVER_ARGS --compass-run "$OUT/run.json" \
        >"$OUT/server.log" 2>&1 &
    # Not /health: before the traffic LP joins the run, the frontend answers nothing.
    ready=("Uvicorn running" "$OUT/server.log")
fi
server=$!
deadline=$((SECONDS + WAIT_S))
until grep -q "${ready[@]}"; do
    kill -0 $server 2>/dev/null || { echo "cctraces_sim: the deployment exited before it started" >&2; exit 1; }
    ((SECONDS < deadline)) || { echo "cctraces_sim: no deployment after $WAIT_S s" >&2; exit 1; }
    sleep 1
done

rc=0
(cd "$OUT" && ATOM_COMPASS_RUN=$OUT/run.json AIPERF_DATASET_MMAP_BASE_PATH=$OUT \
    "$HARNESS_PYTHON" -m aiperf profile --url "127.0.0.1:$PORT" --model "$MODEL" \
    --tokenizer "$MODEL" --input-file "$OUT/traces" --artifact-dir "$OUT/artifacts" \
    "${args[@]}") >"$OUT/aiperf.log" 2>&1 || rc=$?
# The deployment leaves at the finish, once the step table is written.
deadline=$((SECONDS + WAIT_S))
while kill -0 $server 2>/dev/null && ((SECONDS < deadline)); do sleep 1; done
kill -0 $server 2>/dev/null && { echo "cctraces_sim: the deployment is still up $WAIT_S s after aiperf" >&2; exit 1; }
wait $server || rc=$((rc ? rc : $?))

python3 - "$OUT" "$BENCHMARK_ID" "$rc" <<'EOF'
import json, re, sys
from pathlib import Path

out, benchmark_id, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
summary = json.loads((out / "summary.json").read_text())
export = json.loads((out / "artifacts/profile_export_aiperf.json").read_text())
avg = lambda k: export[k]["avg"]
# Prefill logs one line per KV transfer, with its simulated seconds.
prefill = out / "prefill.log"
kv = ""
if prefill.exists():
    t = [float(s) for s in re.findall(r"KV transfer .*, (\S+) simulated s", prefill.read_text())]
    kv = f" kv_transfers={len(t)} kv_transfer_mean_s={sum(t) / len(t) if t else 0:.6g}"
print(
    f"cctraces_sim: rc={rc} requests={avg('request_count'):.0f} "
    f"errors={len(export['error_summary'])} "
    f"simulated_s={summary['cost']['simulated_seconds']:.2f} wall_s={summary['cost']['wall_seconds']:.0f} "
    f"cache_read_pct={avg('overall_usage_prompt_cache_read_pct'):.2f} "
    f"theoretical_hit_pct={avg('theoretical_prefix_cache_hit'):.2f} "
    f"ttft_ms={avg('time_to_first_token'):.2f} itl_ms={avg('inter_token_latency'):.3f} "
    f"request_throughput={avg('request_throughput'):.4f} "
    f"refusals={summary['schedule']['refusals']['count']} "
    f"coverage_report={summary['coverage_report']}"
    + kv
)
if export["benchmark_id"] != benchmark_id:
    sys.exit(f"cctraces_sim: benchmark id {export['benchmark_id']}, recorded {benchmark_id}")
sys.exit(rc)
EOF
