cd /workspace/ATOM
export HF_HUB_OFFLINE=1 ATOM_LOG_MOE_PARALLEL=1
DST=/workspace/ATOM/agent_scratch/qwen3moe
# ep_size inherits tp_size unless the DP and TP dims are flattened, which is what
# enable_dp_attention turns on. TP=2 x DP=2 flattened should give ep_size=4.
timeout 500 python scripts/compass/run.py --model $DST \
  -tp 2 --data-parallel-size 2 --enable-dp-attention --enable-expert-parallel \
  --load_dummy=xavier --gpu-memory-utilization 0.30 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 32 --max-tokens 2 \
  --out compass_ops/epdp.json > compass_ops/epdp_full.log 2>&1
  ### MOE|Error|Traceback|not support|assert" | sort | uniq -c | head -12
echo "### EP DP DONE"
