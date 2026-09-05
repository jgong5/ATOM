cd /workspace/ATOM
export HF_HUB_OFFLINE=1
DST=/workspace/ATOM/agent_scratch/qwen3moe
COMMON="--model $DST -tp 2 --load_dummy=xavier --gpu-memory-utilization 0.35 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 8 --compass"
rm -f compass_ops/noep*.json
timeout 400 python scripts/compass/run.py $COMMON --compass-mode trace \
  --compass-graph-out compass_ops/noep.json --compass-trace-prefill 2 \
  --out compass_ops/noep_trace.json 2>&1 | grep -i "traced" | tail -2
timeout 400 python scripts/compass/run.py $COMMON --compass-mode measure \
  --compass-measure-out compass_ops/noep_price.jsonl \
  --compass-bench-graph 'compass_ops/noep.tp*.json,compass_ops/noep.prefill.tp*.json' \
  --compass-bench-out compass_ops/prices_noep.json --compass-bench-cache graph \
  --out compass_ops/noep_price.json 2>&1 | grep -i "priced" | tail -1
python3 -c "
import json
def moe(p):
    d=json.load(open(p)); out={}
    for k,v in d['prices'].items():
        n=v['name'].split('::')[-1]
        if n in ('moe_forward','fused_allreduce_rmsnorm_'):
            out.setdefault((n,k.split('|')[1][:20]),[]).append(v['seconds']*1e6)
    return {k:(len(v),sum(v)/len(v)) for k,v in out.items()}
on, off = moe('compass_ops/prices_ep.tp0.json'), moe('compass_ops/prices_noep.tp0.json')
print()
print('%-26s %-20s %>0s' % ('operator','shape',''))
for k in sorted(set(on)|set(off)):
    a=on.get(k); b=off.get(k)
    print('  %-24s %-20s  EP on %9s   EP off %9s' % (k[0][:24], k[1],
          ('%.2f us'%a[1]) if a else '-', ('%.2f us'%b[1]) if b else '-'))
"
echo "### EP OFF DONE"
