cd /workspace/ATOM
export HF_HUB_OFFLINE=1
export ATOM_TRACE_ATTN=120
python scripts/compass/run.py \
  --model Qwen/Qwen3-0.6B \
  --compass --compass-mode measure \
  --compass-measure-out compass_ops/prod.jsonl \
  --num-prompts 4 --max-tokens 8 --prompt-tokens 64 \
  --out compass_ops/prod.json
echo "### PROD RUN DONE rc=$?"
