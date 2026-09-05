cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
rm -f compass_ops/gm.s*.json

echo "=== trace steps 2,10,20,30 (96 output tokens) ==="
python scripts/compass/run.py $BASE --max-tokens 96 --compass-mode trace \
  --compass-graph-out compass_ops/gm.json --compass-trace-steps 2,10,20,30 \
  --out compass_ops/m_trace.json 2>&1 | grep -c "traced"
ls -1 compass_ops/gm.s*.json

echo "=== price the union ==="
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode measure \
  --compass-measure-out compass_ops/m_price.jsonl \
  --compass-bench-graph 'compass_ops/gm.s*.json' \
  --compass-bench-out compass_ops/prices7.json \
  --compass-bench-cache graph --out compass_ops/m_price.json 2>&1 | grep -i priced
echo "### MULTI DONE"
