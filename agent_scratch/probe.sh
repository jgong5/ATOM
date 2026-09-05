cd /workspace/ATOM
export HF_HUB_OFFLINE=1
export COMPASS_BENCH_PROFILE=unified_attention
python scripts/compass/run.py \
  --model Qwen/Qwen3-0.6B \
  --compass --compass-mode measure \
  --compass-measure-out compass_ops/probe.jsonl \
  --compass-bench-graph compass_ops/attn_only.json \
  --compass-bench-out compass_ops/attn_only_prices.json \
  --compass-bench-cache graph \
  --num-prompts 2 --max-tokens 4 --prompt-tokens 64 \
  --out compass_ops/probe.json
echo "### PROBE RUN DONE rc=$?"
