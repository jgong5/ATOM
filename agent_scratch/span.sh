cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --compass --num-prompts 4 --prompt-tokens 64"
rm -f compass_ops/gs.s*.json
python scripts/compass/run.py $BASE --max-tokens 96 --compass-mode trace \
  --compass-graph-out compass_ops/gs.json --compass-trace-steps 2,30,60,90 \
  --out compass_ops/s_trace.json 2>&1 | tail -1
python scripts/compass/run.py $BASE --max-tokens 32 --compass-mode measure \
  --compass-measure-out compass_ops/s_price.jsonl \
  --compass-bench-graph 'compass_ops/gs.s*.json' \
  --compass-bench-out compass_ops/prices8.json \
  --compass-bench-cache graph --out compass_ops/s_price.json 2>&1 | grep -i priced
python scripts/compass/run.py $BASE --max-tokens 96 --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices8.json \
  --compass-oracle-option graph='compass_ops/gs.s*.json' \
  --compass-oracle-option fallback=compass_ops/v_with.jsonl \
  --compass-admission-seconds 0.013 --out compass_ops/s96.json 2>&1 | grep -i active | tail -1
python3 - <<'PY'
import json
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-26s %12s %12s %8s' % ('96 tokens','predicted','real','err'))
for k in ('ttft','tpot','latency'):
    p,r=m('compass_ops/s96.json',k),m('compass_ops/h_real.json',k)
    print('%-26s %10.3f ms %10.3f ms %7.1f%%' % (k,p*1e3,r*1e3,100*(p-r)/r))
PY
echo "### SPAN DONE"
