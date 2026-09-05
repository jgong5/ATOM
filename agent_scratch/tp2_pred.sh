cd /workspace/ATOM
export HF_HUB_OFFLINE=1
ADM=$(cat compass_ops/t2_admission.txt)
echo "admission from the real run: $ADM s"
python scripts/compass/run.py --model Qwen/Qwen3-0.6B -tp 2 --compass \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices_tp2.json \
  --compass-oracle-option graph=compass_ops/gt.json \
  --compass-oracle-option prefill_graph=compass_ops/gt.prefill.json \
  --compass-admission-seconds "$ADM" \
  --out compass_ops/t2_pred.json 2>&1 | grep -i "active\|WARNING" | tail -3
python3 - <<'PY'
import json, statistics
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-10s %11s %11s %9s   %s' % ('TP=2','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    v=[m('compass_ops/t2_plain%d.json'%i,k) for i in (1,2,3)]
    r,s=statistics.mean(v),statistics.stdev(v)
    p=m('compass_ops/t2_pred.json',k)
    print('%-10s %9.3f ms %9.3f ms %8.1f%%   +-%.1f%%' % (k,p*1e3,r*1e3,100*(p-r)/r,100*s/r))
PY
echo "### TP2 PRED DONE"
