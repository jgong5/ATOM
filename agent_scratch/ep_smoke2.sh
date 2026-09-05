cd /workspace/ATOM
export HF_HUB_OFFLINE=1
SRC=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B', allow_patterns=['*.json','*.txt','*.model'], local_files_only=True))")
DST=/workspace/ATOM/agent_scratch/qwen3moe
rm -rf $DST && mkdir -p $DST && cp -L $SRC/*.json $SRC/*.txt $DST/ 2>/dev/null
# The shard index names 16 safetensors we deliberately did not fetch; without it
# the loader has no checkpoint to look for, which is the point of dummy weights.
rm -f $DST/model.safetensors.index.json
ls $DST
timeout 480 python scripts/compass/run.py --model $DST \
  -tp 2 --enable-expert-parallel --load_dummy=xavier \
  --gpu-memory-utilization 0.35 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 8 \
  --compass --compass-mode trace --compass-graph-out compass_ops/ep.json \
  --compass-trace-prefill 2 --out compass_ops/ep_trace.json 2>&1 \
  | grep -iE "traced|Error|Traceback|expert|assert|not support|OutOfMemory|moe" | tail -12
echo "### EP SMOKE2 DONE"
