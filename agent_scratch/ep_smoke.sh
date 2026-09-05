cd /workspace/ATOM
export HF_HUB_OFFLINE=1
# Dummy weights: this measures kernel time, not output quality, and it avoids
# pulling 60 GB. Modest memory share and context, because the GPUs are shared.
timeout 480 python scripts/compass/run.py --model Qwen/Qwen3-30B-A3B \
  -tp 2 --enable-expert-parallel --load_dummy=xavier \
  --gpu-memory-utilization 0.35 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 8 \
  --compass --compass-mode trace --compass-graph-out compass_ops/ep.json \
  --compass-trace-prefill 2 --out compass_ops/ep_trace.json 2>&1 \
  | grep -iE "traced|error|Traceback|expert|assert|not supported|OutOfMemory" | tail -12
echo "### EP SMOKE rc=$?"
