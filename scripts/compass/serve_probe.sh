#!/bin/bash
# Serve a model over HTTP and benchmark it, optionally with Compass predicting.
#
#   scripts/compass/serve_probe.sh real    out/real
#   scripts/compass/serve_probe.sh predict out/predict \
#       --compass --compass-oracle atom.compass.core.cost.calibrated.CalibratedCostOracle \
#       --compass-oracle-option table=out/steps.jsonl
#
# NOTE what this can and cannot show. benchmark_serving times requests with
# perf_counter around an HTTP stream and divides every metric by its own
# wall-clock duration, so under --compass it measures how fast the *simulator*
# ran, not what the simulator predicted. See "An HTTP benchmark cannot measure a
# simulated engine" in atom/compass/DESIGN_NOTES.md. This script is kept as the
# reproducer for that finding, and as the right harness for benchmarking the
# real engine.
#
# RATE=inf is the default deliberately: a paced workload is arrival-bound, and
# 32 requests at 4/s takes 8s however fast the server is, so real and predict
# come out identical for a reason that has nothing to do with accuracy.
set -u

LABEL="$1"; OUT="$2"; shift 2
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
mkdir -p "$OUT"

# The OS picks the port: host networking is shared with about twenty containers,
# so a fixed number collides with whoever holds it, including an earlier run of
# this script.
PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "### $LABEL on port $PORT"

python -m atom.entrypoints.openai_server --model "$MODEL" --server-port "$PORT" "$@" \
  > "$OUT/server.log" 2>&1 &
SRV=$!

for i in $(seq 1 "${STARTUP_TIMEOUT:-180}"); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "### server up after ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "### SERVER DIED during startup"; tail -25 "$OUT/server.log"; exit 1; }
  sleep 1
done

curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || {
  echo "### SERVER NEVER BECAME HEALTHY"; tail -25 "$OUT/server.log"
  kill $SRV 2>/dev/null; exit 1; }

python -m atom.benchmarks.benchmark_serving \
  --backend vllm --model "$MODEL" --base-url "http://localhost:$PORT" \
  --dataset-name random \
  --random-input-len "${INLEN:-256}" --random-output-len "${OUTLEN:-64}" \
  --num-prompts "${NPROMPTS:-64}" --request-rate "${RATE:-inf}" --ignore-eos \
  --save-result --result-filename "$OUT/bench.json" 2>&1 | tail -35

# The engine's own readings, on the engine's clock. Under --compass these are
# the simulated latencies, and they are the only place those exist: the client
# above timed the simulator. Drains the server-side buffer.
curl -s "http://localhost:$PORT/compass/requests" > "$OUT/engine.json" 2>/dev/null \
  && echo "### engine-side timings: $(python -c "import json;print(json.load(open('$OUT/engine.json'))['count'])" 2>/dev/null || echo '?') requests"

echo "### stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "### $LABEL DONE"
