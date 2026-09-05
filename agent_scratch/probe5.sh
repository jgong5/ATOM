cd /workspace/ATOM
export HF_HUB_OFFLINE=1
export COMPASS_BENCH_PROFILE=unified_attention
export COMPASS_KV_VARIANTS=512
export COMPASS_PROBE_BATCHES=1,16,64,128,256,512
python scripts/compass/run.py --model Qwen/Qwen3-0.6B \
  --compass --compass-mode measure \
  --compass-measure-out compass_ops/p5.jsonl \
  --compass-bench-graph compass_ops/attn_only5.json \
  --compass-bench-out compass_ops/p5_prices.json \
  --compass-bench-cache graph \
  --num-prompts 4 --max-tokens 8 --prompt-tokens 64 \
  --out compass_ops/p5.json
echo "### PROBE5 DONE rc=$?"
