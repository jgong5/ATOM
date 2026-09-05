cd /workspace/ATOM
export HF_HUB_OFFLINE=1
BASE="--model Qwen/Qwen3-0.6B --num-prompts 4 --prompt-tokens 64 --max-tokens 32"
echo "=== plain (no compass) ==="
python scripts/compass/run.py $BASE --out compass_ops/ob_plain.json 2>&1 | tail -1
echo "=== measure mode ==="
python scripts/compass/run.py $BASE --compass --compass-mode measure \
  --compass-measure-out compass_ops/ob_meas.jsonl \
  --out compass_ops/ob_meas.json 2>&1 | tail -1
python3 - <<'PY'
import json
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-10s %11s %11s %8s' % ('','plain','measured','delta'))
for k in ('ttft','tpot','latency'):
    a,b=m('compass_ops/ob_plain.json',k),m('compass_ops/ob_meas.json',k)
    print('%-10s %9.3f ms %9.3f ms %7.1f%%' % (k,a*1e3,b*1e3,100*(b-a)/a))
rows=[json.loads(l) for l in open('compass_ops/ob_meas.jsonl')]
pre=[r for r in rows if r['num_prefill_tokens']][0]
print()
print('prefill step        : %.3f ms' % (pre['seconds']*1e3))
print('real TTFT (measured): %.3f ms' % (m('compass_ops/ob_meas.json','ttft')*1e3))
print('admission = TTFT - prefill step = %.3f ms'
      % ((m('compass_ops/ob_meas.json','ttft')-pre['seconds'])*1e3))
PY
echo "### OBSERVER DONE"
