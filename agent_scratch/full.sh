cd /workspace/ATOM
export HF_HUB_OFFLINE=1
python scripts/compass/run.py \
  --model Qwen/Qwen3-0.6B \
  --compass --compass-mode measure \
  --compass-measure-out compass_ops/full.jsonl \
  --compass-bench-graph compass_ops/graph4.json \
  --compass-bench-out compass_ops/prices_reset.json \
  --compass-bench-cache graph \
  --num-prompts 4 --max-tokens 8 --prompt-tokens 64 \
  --out compass_ops/full.json
echo "### FULL RUN DONE rc=$?"
