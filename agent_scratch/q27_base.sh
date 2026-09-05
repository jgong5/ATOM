cd /workspace/ATOM
export HF_HUB_OFFLINE=1
M="--model Qwen/Qwen3.8-27B -tp 2 --gpu-memory-utilization 0.40 \
   --max-model-len 4096 --max-num-batched-tokens 4096"
W="--num-prompts 4 --prompt-tokens 64 --max-tokens 16"
for i in 1 2 3; do
  timeout 600 python scripts/compass/run.py $M $W \
    --out compass_ops/q27_plain$i.json >/dev/null 2>&1
done
timeout 600 python scripts/compass/run.py $M $W --compass --compass-mode measure \
  --compass-measure-out compass_ops/q27_meas.jsonl \
  --out compass_ops/q27_meas.json >/dev/null 2>&1
python3 - <<'PY'
import json, statistics, glob
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
for k in ('ttft','tpot','latency'):
    v=[m('compass_ops/q27_plain%d.json'%i,k) for i in (1,2,3)]
    print('%-8s real %9.3f ms  sd %.3f' % (k, statistics.mean(v)*1e3, statistics.stdev(v)*1e3))
rows=[json.loads(l) for l in open(sorted(glob.glob('compass_ops/q27_meas*.jsonl'))[0])]
pre=[r for r in rows if r['num_prefill_tokens']][0]['seconds']
dec=statistics.median([r['seconds'] for r in rows if not r['num_prefill_tokens']][1:])
adm=m('compass_ops/q27_meas.json','ttft')-pre
print('prefill step %.3f ms  decode step %.3f ms  admission %.3f ms' % (pre*1e3, dec*1e3, adm*1e3))
json.dump({'prefill':pre,'decode':dec,'admission':adm}, open('compass_ops/q27_steps.json','w'))
PY
echo "### Q27 BASE DONE"
