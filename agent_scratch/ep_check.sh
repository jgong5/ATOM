cd /workspace/ATOM
export HF_HUB_OFFLINE=1 ATOM_LOG_MOE_PARALLEL=1
DST=/workspace/ATOM/agent_scratch/qwen3moe
COMMON="--model $DST -tp 2 --load_dummy=xavier --gpu-memory-utilization 0.35 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 2 --prompt-tokens 32 --max-tokens 2"
echo "=== WITH --enable-expert-parallel ==="
timeout 400 python scripts/compass/run.py $COMMON --enable-expert-parallel \
  --out compass_ops/epon.json 2>&1 | grep "### MOE" | sort | uniq -c | head
echo "=== WITHOUT ==="
timeout 400 python scripts/compass/run.py $COMMON \
  --out compass_ops/epoff.json 2>&1 | grep "### MOE" | sort | uniq -c | head
echo "### EP CHECK DONE"
