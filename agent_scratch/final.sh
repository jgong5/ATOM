cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
ORACLE="--compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices7.json \
  --compass-oracle-option graph=compass_ops/gm.s*.json \
  --compass-oracle-option fallback=compass_ops/v_with.jsonl"
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode predict $ORACLE \
  --compass-admission-seconds 0.013 --out compass_ops/f32.json 2>&1 | grep -i "active" | tail -1
python scripts/compass/run.py $BASE --max-tokens 96 --compass-mode predict $ORACLE \
  --compass-admission-seconds 0.013 --out compass_ops/f96.json 2>&1 | tail -1
python3 - <<'PY'
import json
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-30s %12s %12s %8s' % ('','predicted','real','err'))
for label, pred, real in (
        ('32 tokens (traced range)', 'compass_ops/f32.json', 'compass_ops/v_with.json'),
        ('96 tokens (extrapolated)', 'compass_ops/f96.json', 'compass_ops/h_real.json')):
    for k in ('ttft','tpot','latency'):
        p,r=m(pred,k),m(real,k)
        print('%-30s %10.3f ms %10.3f ms %7.1f%%' % (label+' '+k,p*1e3,r*1e3,100*(p-r)/r))
PY
echo "### FINAL DONE"
