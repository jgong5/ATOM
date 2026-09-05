cd /workspace/ATOM
export HF_HUB_OFFLINE=1
M="--model Qwen/Qwen3.8-27B -tp 2 --gpu-memory-utilization 0.40 \
   --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable_prefix_caching"
W="--num-prompts 4 --prompt-tokens 64 --max-tokens 16"
echo "=== 16 tokens, PLAIN (the 16-token measure-mode run gave ttft 339.2) ==="
PASSES=4 timeout 900 python agent_scratch/q27_warm.py $M $W \
  --out compass_ops/adm_plain16.json 2>&1 | grep "###"
python3 -c "
import json
w=json.load(open('compass_ops/adm_plain16.json'))['warm']
t=sum(x['ttft'] for x in w)/len(w)
print()
print('16 tokens plain   : ttft %.2f ms' % (t*1e3))
print('16 tokens measure : ttft 339.22 ms')
print('32 tokens plain   : ttft 326.07 ms')
print()
print('if plain-16 ~= 326, the 13 ms was measure mode, not the workload')
"
echo "### CONFOUND DONE"
