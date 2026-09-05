cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B -tp 2 --num-prompts 4 --prompt-tokens 64 --max-tokens 32"
for i in 1 2 3; do
  python scripts/compass/run.py $BASE --out compass_ops/t2_plain$i.json >/dev/null 2>&1
done
python scripts/compass/run.py $BASE --compass --compass-mode measure \
  --compass-measure-out compass_ops/t2_meas.jsonl \
  --out compass_ops/t2_meas.json >/dev/null 2>&1
python3 - <<'PY'
import json, statistics, glob
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
v={k:[m('compass_ops/t2_plain%d.json'%i,k) for i in (1,2,3)] for k in ('ttft','tpot','latency')}
for k in v: print('%-8s real %8.3f ms  sd %.3f' % (k, statistics.mean(v[k])*1e3, statistics.stdev(v[k])*1e3))
tbl=sorted(glob.glob('compass_ops/t2_meas*.jsonl'))
rows=[json.loads(l) for l in open(tbl[0])]
pre=[r for r in rows if r['num_prefill_tokens']][0]['seconds']
dec=statistics.median([r['seconds'] for r in rows if not r['num_prefill_tokens']][1:])
adm=m('compass_ops/t2_meas.json','ttft')-pre
print()
print('prefill step %.3f ms   decode step %.3f ms   admission %.3f ms' % (pre*1e3, dec*1e3, adm*1e3))
open('compass_ops/t2_admission.txt','w').write(str(adm))
PY
echo "### TP2 VAL DONE"
