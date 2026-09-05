cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
ORACLE="--compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices6.json \
  --compass-oracle-option graph=compass_ops/graph5.json \
  --compass-oracle-option fallback=compass_ops/v_with.jsonl"

echo "=== admission modelled, same workload ==="
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode predict $ORACLE \
  --compass-admission-seconds 0.013 --out compass_ops/o_adm.json 2>&1 | tail -1

echo "=== held out: 96 output tokens, real ==="
python scripts/compass/run.py $BASE --max-tokens 96 --compass-mode measure \
  --compass-measure-out compass_ops/h_real.jsonl --out compass_ops/h_real.json 2>&1 | tail -1
echo "=== held out: 96 output tokens, predicted ==="
python scripts/compass/run.py $BASE --max-tokens 96 --compass-mode predict $ORACLE \
  --compass-admission-seconds 0.013 --out compass_ops/h_pred.json 2>&1 | tail -1

python3 - <<'PY'
import json
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-34s %12s %12s %8s' % ('','predicted','real','err'))
for label, pred, real in (
        ('fitted workload, 32 tokens', 'compass_ops/o_adm.json', 'compass_ops/v_with.json'),
        ('held out, 96 tokens', 'compass_ops/h_pred.json', 'compass_ops/h_real.json')):
    for k in ('ttft','tpot','latency'):
        p,r=m(pred,k),m(real,k)
        print('%-34s %10.3f ms %10.3f ms %7.1f%%' % (label+' '+k,p*1e3,r*1e3,100*(p-r)/r))
PY
echo "### HELDOUT DONE"
