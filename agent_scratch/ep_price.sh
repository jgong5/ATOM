cd /workspace/ATOM
export HF_HUB_OFFLINE=1
DST=/workspace/ATOM/agent_scratch/qwen3moe
timeout 480 python scripts/compass/run.py --model $DST \
  -tp 2 --enable-expert-parallel --load_dummy=xavier \
  --gpu-memory-utilization 0.35 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 8 \
  --compass --compass-mode measure --compass-measure-out compass_ops/ep_price.jsonl \
  --compass-bench-graph 'compass_ops/ep.tp*.json,compass_ops/ep.prefill.tp*.json' \
  --compass-bench-out compass_ops/prices_ep.json --compass-bench-cache graph \
  --out compass_ops/ep_price.json 2>&1 | grep -iE "priced|Traceback" | tail -4
python3 -c "
import json, collections
d=json.load(open('compass_ops/prices_ep.tp0.json'))
print(json.dumps(d['coverage']))
for k,v in d['prices'].items():
    if any(x in k for x in ('moe_forward','fused_allreduce','all_reduce')):
        print('  PRICED   %-32s %9.2f us n=%-4d %s' % (v['name'].split('::')[-1], v['seconds']*1e6, v['occurrences'], k.split('|')[1][:24]))
for k,v in d['unpriced'].items():
    if any(x in k for x in ('moe_forward','fused_allreduce','all_reduce')):
        print('  UNPRICED %-32s %s' % (k.split('|')[0].split('::')[-1], v[:70]))
print('reasons:', collections.Counter(v.split(':')[0] for v in d['unpriced'].values()).most_common())
"
echo "### EP PRICE DONE"
