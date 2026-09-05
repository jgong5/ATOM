cd /workspace/ATOM
export HF_HUB_OFFLINE=1
COMMON="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
rm -rf compass_ops/prof8 compass_ops/prof32 && mkdir -p compass_ops/prof8 compass_ops/prof32

echo "=== price (with kernel breakdown) ==="
python scripts/compass/run.py $COMMON --max-tokens 32 --compass-mode measure \
  --compass-measure-out compass_ops/c_price.jsonl \
  --compass-bench-graph compass_ops/graph5.json \
  --compass-bench-out compass_ops/prices6.json \
  --compass-bench-cache graph --out compass_ops/c_price.json 2>&1 | grep -i priced

echo "=== profile 8 decode steps ==="
python scripts/compass/run.py $COMMON --max-tokens 8 --compass-mode measure \
  --compass-measure-out compass_ops/c8.jsonl --torch-profiler-dir compass_ops/prof8 \
  --out compass_ops/c8.json 2>&1 | tail -1
echo "=== profile 32 decode steps ==="
python scripts/compass/run.py $COMMON --max-tokens 32 --compass-mode measure \
  --compass-measure-out compass_ops/c32.jsonl --torch-profiler-dir compass_ops/prof32 \
  --out compass_ops/c32.json 2>&1 | tail -1
echo "### CHECK DONE"
