cd /workspace/ATOM
export HF_HUB_OFFLINE=1
timeout 900 python agent_scratch/q27_warm.py --model Qwen/Qwen3.8-27B -tp 2 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --no-enable_prefix_caching --num-prompts 4 --prompt-tokens 64 --max-tokens 16 \
  2>&1 | grep "###"
echo "### WARM DONE"
