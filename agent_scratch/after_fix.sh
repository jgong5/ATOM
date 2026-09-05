cd /workspace/ATOM
export HF_HUB_OFFLINE=1
DST=/workspace/ATOM/agent_scratch/qwen3moe

echo "=== 1. expert parallel at ep_size=4 (TP=2 x DP=2, dp-attention) ==="
timeout 700 python scripts/compass/run.py --model $DST \
  -tp 2 --data-parallel-size 2 --enable-dp-attention --enable-expert-parallel \
  --load_dummy=xavier --gpu-memory-utilization 0.30 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 32 --max-tokens 4 \
  --compass --compass-mode trace --compass-graph-out compass_ops/epdp.json \
  --compass-trace-prefill 2 --out compass_ops/epdp.json \
  > compass_ops/epdp_full.log 2>&1
echo "  rc=$?"
grep -iE "traced|cmath|CalledProcess|Error" compass_ops/epdp_full.log | grep -v "^\[aiter\]" | tail -5

echo "=== 2. Qwen3.8-27B, the PoC target ==="
timeout 900 python scripts/compass/run.py --model Qwen/Qwen3.8-27B -tp 2 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 2 --prompt-tokens 32 --max-tokens 8 \
  --out compass_ops/q27.json > compass_ops/q27_full.log 2>&1
echo "  rc=$?"
grep -iE "wrote|cmath|CalledProcess|Error|Traceback" compass_ops/q27_full.log | grep -v "^\[aiter\]" | tail -6
echo "### AFTER FIX DONE"
