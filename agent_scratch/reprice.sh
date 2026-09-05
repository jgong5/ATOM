cd /workspace/ATOM
export HF_HUB_OFFLINE=1
python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 --compass-mode measure \
  --compass-measure-out compass_ops/c_price.jsonl \
  --compass-bench-graph compass_ops/graph5.json \
  --compass-bench-out compass_ops/prices6.json \
  --compass-bench-cache graph --out compass_ops/c_price.json 2>&1 | grep -i priced
python scripts/compass/price_check.py compass_ops/prices6.json \
  --trace compass_ops/prof8:8 --trace compass_ops/prof32:32 --step-seconds 0.003115
