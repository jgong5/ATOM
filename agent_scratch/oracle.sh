cd /workspace/ATOM
export HF_HUB_OFFLINE=1
python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 \
  --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices6.json \
  --compass-oracle-option graph=compass_ops/graph5.json \
  --compass-oracle-option fallback=compass_ops/v_with.jsonl \
  --out compass_ops/o_pred.json 2>&1 | grep -i "ATOMCompass active\|WARNING\|Error" | head -5
python3 -c "
import json
a=json.load(open('compass_ops/o_pred.json')); b=json.load(open('compass_ops/v_with.json'))
def m(d,k): return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-10s %12s %12s %8s' % ('','predicted','real','err'))
for k in ('ttft','tpot','latency'):
    p,r=m(a,k),m(b,k)
    print('%-10s %10.3f ms %10.3f ms %7.1f%%' % (k,p*1e3,r*1e3,100*(p-r)/r))
"
echo "### ORACLE DONE"
