cd /workspace/ATOM
export HF_HUB_OFFLINE=1
M="--model Qwen/Qwen3.8-27B -tp 2 --gpu-memory-utilization 0.40 \
   --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable_prefix_caching"
W="--num-prompts 4 --prompt-tokens 64 --max-tokens 16"
rm -f compass_ops/g27*.json

echo "=== trace (decode + prefill) ==="
timeout 900 python scripts/compass/run.py $M $W --compass --compass-mode trace \
  --compass-graph-out compass_ops/g27.json --compass-trace-prefill 2 \
  --out compass_ops/q27_trace.json > compass_ops/q27_trace.log 2>&1
echo "  rc=$?"; grep -i "traced" compass_ops/q27_trace.log | tail -4

echo "=== price the union ==="
timeout 1500 python scripts/compass/run.py $M $W --compass --compass-mode measure \
  --compass-measure-out compass_ops/q27_price.jsonl \
  --compass-bench-graph 'compass_ops/g27.tp*.json,compass_ops/g27.prefill.tp*.json' \
  --compass-bench-out compass_ops/prices27.json --compass-bench-iters 300 \
  --compass-bench-cache graph --out compass_ops/q27_price.json \
  > compass_ops/q27_price.log 2>&1
echo "  rc=$?"; grep -i "priced" compass_ops/q27_price.log | tail -2
echo "### Q27 TP DONE"
