# #3636: #975's DPA-TP8 cell, simulated under the op-level backend for a whole-run profile. Sourced by 3636/run scripts.
S=/shared_nfs/xingjche/agent_scratch/3636
R=$S/run
TIP=/shared_nfs/xingjche/ATOM-worktrees/3636     # task-3636 at the integration tip (977f03d4a)
MODEL=/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro
MCOPY=/shared_nfs/xingjche/agent_scratch/1498/model/DeepSeek-V4-Pro   # a gate container's model path (config.json byte-identical)
TRACES=/shared_nfs/huggingface_models/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl
PLATFORM=(--device mi355x --ptl disabled --network crs005-013-ionic)
COMMON=(--model $MODEL --trust-remote-code --kv_cache_dtype fp8 --block-size 16 --no-enable_prefix_caching -tp 8 --enable-dp-attention --max-num-seqs 512)
PREFILL_ARGS=("${COMMON[@]}" --gpu-memory-utilization 0.72 --enable-tbo)
DECODE_ARGS=("${COMMON[@]}" --gpu-memory-utilization 0.75 --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]")
ROUTER_SIM=(--policy dp_sticky --dp-aware --backend atom --log-level info --disable-circuit-breaker --disable-health-check --disable-retries)
ENV_BOTH=(AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING)
ENV_PREFILL=(GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1)
side_args() { # side -> the side's performance environment exported, every other key unset
  unset OMP_NUM_THREADS GPU_MAX_HW_QUEUES ATOM_NUMA_BIND
  export "${ENV_BOTH[@]}"
  [[ $1 == prefill ]] && export "${ENV_PREFILL[@]}"
}
say() { echo "$(basename "$0"): $* $(date -u +%FT%TZ)"; }
