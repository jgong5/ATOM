cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
rm -f compass_ops/gp.json compass_ops/gp.prefill.json

echo "=== trace one decode step and one prefill step ==="
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode trace \
  --compass-graph-out compass_ops/gp.json --compass-trace-prefill 2 \
  --out compass_ops/p_trace.json 2>&1 | grep -i "traced" 
ls -1 compass_ops/gp*.json

echo "=== price both ==="
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode measure \
  --compass-measure-out compass_ops/p_price.jsonl \
  --compass-bench-graph 'compass_ops/gp.json,compass_ops/gp.prefill.json' \
  --compass-bench-out compass_ops/prices9.json \
  --compass-bench-cache graph --out compass_ops/p_price.json 2>&1 | grep -i priced
echo "### PREFILL DONE"
