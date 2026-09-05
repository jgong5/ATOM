cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --num-prompts 4 --prompt-tokens 64 --max-tokens 32"
for i in 1 2 3; do
  python scripts/compass/run.py $BASE --out compass_ops/r_plain$i.json >/dev/null 2>&1
  python scripts/compass/run.py $BASE --compass --compass-mode measure \
    --compass-measure-out compass_ops/r_meas$i.jsonl \
    --out compass_ops/r_meas$i.json >/dev/null 2>&1
done
python3 - <<'PY'
import json, statistics
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print('%-9s %-9s %9s %9s %9s   %s' % ('metric','mode','run1','run2','run3','mean +- sd'))
for k in ('ttft','tpot','latency'):
    for mode,pat in (('plain','compass_ops/r_plain%d.json'),('measure','compass_ops/r_meas%d.json')):
        v=[m(pat%i,k)*1e3 for i in (1,2,3)]
        print('%-9s %-9s %8.2f %8.2f %8.2f   %7.2f +- %.2f ms'
              % (k,mode,v[0],v[1],v[2],statistics.mean(v),statistics.stdev(v)))
pre=[[json.loads(l) for l in open('compass_ops/r_meas%d.jsonl'%i)] for i in (1,2,3)]
p=[[r for r in rows if r['num_prefill_tokens']][0]['seconds']*1e3 for rows in pre]
t=[m('compass_ops/r_meas%d.json'%i,'ttft')*1e3 for i in (1,2,3)]
print()
print('prefill step : %s  mean %.2f ms' % ([round(x,2) for x in p], statistics.mean(p)))
print('admission    : %s  mean %.2f ms' % ([round(a-b,2) for a,b in zip(t,p)],
      statistics.mean([a-b for a,b in zip(t,p)])))
PY
echo "### REPEAT DONE"
