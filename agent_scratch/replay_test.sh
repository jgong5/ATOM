cd /workspace/ATOM
export HF_HUB_OFFLINE=1
# GPU 0 was 88% held by another container when this last ran and the KV
# budget went negative. Pin to a free device rather than race neighbours.
source /workspace/ATOM/agent_scratch/pick_gpu.sh
export HIP_VISIBLE_DEVICES=$(pick_gpus 1)
rm -rf compass_replay && mkdir -p compass_replay
PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "port $PORT"

start() {
  local label=$1; shift
  setsid python -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --server-port "$PORT" \
    --compass-measure-out "compass_replay/steps_$label.jsonl" "$@" \
    > "compass_replay/server_$label.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "  $label up after ${i}s"; return 0; }
    kill -0 $SRV 2>/dev/null || { echo "  $label DIED"; tail -6 "compass_replay/server_$label.log"; return 1; }
    sleep 1
  done
  echo "  $label timed out"; return 1
}
stop() {
  # The wrapper spawns ATOM::EngineCore children that survive a kill on the
  # parent. One survived nine hours holding 169 GB and was misread as another
  # container. setsid above puts them in their own group; kill the group.
  kill -- -"$SRV" 2>/dev/null || kill "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  pkill -f "openai_server.*--server-port $PORT" 2>/dev/null
  sleep 5
}

echo "=== real (declared arrivals ignored on a wall clock) ==="
if start real --compass --compass-mode measure; then
  python scripts/compass/replay.py --port "$PORT" --num-requests 64 --rate 0 \
    --input-tokens 128 --output-tokens 32 --out compass_replay/real.json
  stop
fi

echo "=== simulated (declared arrivals honoured) ==="
if start pred --compass --compass-mode predict \
    --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
    --compass-oracle-option prices=compass_ops/prices_serve.json \
    --compass-oracle-option 'graph=compass_ops/ladder.b*.json' \
    --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json \
    --compass-admission-seconds 0.0132; then
  python scripts/compass/replay.py --port "$PORT" --num-requests 64 --rate 0 \
    --input-tokens 128 --output-tokens 32 --out compass_replay/pred.json
  stop
fi

python3 - <<'PY'
import json, collections
for label in ('real','pred'):
    try: rows=[json.loads(l) for l in open('compass_replay/steps_%s.jsonl'%label)]
    except OSError: print('%-6s no steps'%label); continue
    pre=[r for r in rows if r['num_prefill_tokens']]; dec=[r for r in rows if not r['num_prefill_tokens']]
    print('%-6s %3d prefill %4d decode  prefill tokens %s  buckets %s'
          % (label, len(pre), len(dec),
             sorted(collections.Counter(r['num_prefill_tokens'] for r in pre).items())[:5],
             sorted(collections.Counter(r['capture_bucket'] for r in dec).items())[:6]))
PY
echo "### REPLAY DONE"
