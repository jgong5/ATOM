cd /workspace/ATOM
export HF_HUB_OFFLINE=1
export ATOM_TRACE_ATTN=200
python scripts/compass/run.py \
  --model Qwen/Qwen3-0.6B \
  --compass --compass-mode trace \
  --compass-graph-out compass_ops/graph_probe.json \
  --num-prompts 4 --max-tokens 8 --prompt-tokens 64 \
  --out compass_ops/traceprobe.json
echo "### TRACE RUN DONE rc=$?"
