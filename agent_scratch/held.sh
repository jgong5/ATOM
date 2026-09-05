cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --num-prompts 4 --prompt-tokens 64 --max-tokens 96"
for i in 1 2 3; do
  python scripts/compass/run.py $BASE --out compass_ops/h_plain$i.json >/dev/null 2>&1
done
python scripts/compass/run.py $BASE --compass --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices9.json \
  --compass-oracle-option graph=compass_ops/gp.json \
  --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json \
  --compass-admission-seconds 0.0132 --out compass_ops/q96.json >/dev/null 2>&1
python3 - <<'PY'
import json, statistics
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print('%-10s %11s %11s %9s   %s' % ('96 tokens','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    v=[m('compass_ops/h_plain%d.json'%i,k) for i in (1,2,3)]
    r,s=statistics.mean(v),statistics.stdev(v)
    p=m('compass_ops/q96.json',k)
    print('%-10s %9.3f ms %9.3f ms %8.1f%%   +-%.1f%%'
          % (k,p*1e3,r*1e3,100*(p-r)/r,100*s/r))
PY
echo "### HELD DONE"
