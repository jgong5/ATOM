cd /workspace/ATOM
export HF_HUB_OFFLINE=1
COMMON="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --max-tokens 32 --prompt-tokens 64"

echo "=== 1. trace ==="
python scripts/compass/run.py $COMMON --compass-mode trace \
  --compass-graph-out compass_ops/graph5.json --out compass_ops/v_trace.json 2>&1 | tail -3

echo "=== 2. price ==="
python scripts/compass/run.py $COMMON --compass-mode measure \
  --compass-measure-out compass_ops/v_price.jsonl \
  --compass-bench-graph compass_ops/graph5.json \
  --compass-bench-out compass_ops/prices5.json \
  --compass-bench-cache graph --out compass_ops/v_price.json 2>&1 | grep -i "priced" | tail -2

echo "=== 3. ablation with ==="
python scripts/compass/run.py $COMMON --compass-mode measure \
  --compass-measure-out compass_ops/v_with.jsonl --out compass_ops/v_with.json 2>&1 | tail -1
echo "=== 4. ablation without ==="
ATOM_ABLATE_ATTN=1 python scripts/compass/run.py $COMMON --compass-mode measure \
  --compass-measure-out compass_ops/v_without.jsonl --out compass_ops/v_without.json 2>&1 | tail -1
echo "### VERIFY DONE"
