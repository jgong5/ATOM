cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
ORACLE="--compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices9.json \
  --compass-oracle-option graph=compass_ops/gp.json \
  --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json"
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode predict $ORACLE \
  --compass-admission-seconds 0.0132 --out compass_ops/q32.json 2>&1 | grep -i active | tail -1
python3 - <<'PY'
import json, statistics
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
real={k: statistics.mean([m('compass_ops/r_plain%d.json'%i,k) for i in (1,2,3)])
      for k in ('ttft','tpot','latency')}
sd={k: statistics.stdev([m('compass_ops/r_plain%d.json'%i,k) for i in (1,2,3)])
    for k in ('ttft','tpot','latency')}
print()
print('%-10s %11s %11s %9s   %s' % ('','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    p=m('compass_ops/q32.json',k)
    print('%-10s %9.3f ms %9.3f ms %8.1f%%   +-%.1f%%'
          % (k,p*1e3,real[k]*1e3,100*(p-real[k])/real[k],100*sd[k]/real[k]))
PY
echo "### FINAL2 DONE"
