cd /workspace/ATOM
export HF_HUB_OFFLINE=1 PASSES=4
M="--model Qwen/Qwen3.8-27B -tp 2 --gpu-memory-utilization 0.40 \
   --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable_prefix_caching"
W="--num-prompts 4 --prompt-tokens 64 --max-tokens 16"

echo "=== real (4 passes; pass 1 is compilation) ==="
timeout 900 python agent_scratch/q27_warm.py $M $W --compass --compass-mode measure \
  --compass-measure-out compass_ops/e2e_real.jsonl \
  --out compass_ops/e2e_real.json 2>&1 | grep "###"

ADM=$(python3 - <<'PY'
import json, glob, statistics
real=json.load(open('compass_ops/e2e_real.json'))['warm']
rows=[json.loads(l) for l in open(sorted(glob.glob('compass_ops/e2e_real*.jsonl'))[0])]
pre=[r['seconds'] for r in rows if r['num_prefill_tokens']]
warm_pre=statistics.median(pre[1:]) if len(pre)>1 else pre[-1]
print(max(0.0, statistics.fmean([r['ttft'] for r in real]) - warm_pre))
PY
)
echo "admission (measured, warmed) = $ADM s"

echo "=== predicted ==="
timeout 900 python agent_scratch/q27_warm.py $M $W --compass --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices27.json \
  --compass-oracle-option graph=compass_ops/g27.json \
  --compass-oracle-option prefill_graph=compass_ops/g27.prefill.json \
  --compass-oracle-option eager_seconds_per_op=34.62e-6 \
  --compass-admission-seconds "$ADM" \
  --out compass_ops/e2e_pred.json 2>&1 | grep -E "###|active"

python3 - <<'PY'
import json, statistics
real=json.load(open('compass_ops/e2e_real.json'))['warm']
pred=json.load(open('compass_ops/e2e_pred.json'))['warm']
print()
print('%-9s %11s %11s %9s   %s' % ('Qwen3.8-27B','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    r=[x[k] for x in real]; p=statistics.fmean([x[k] for x in pred])
    rm, sd = statistics.fmean(r), statistics.stdev(r)
    print('%-9s %9.3f ms %9.3f ms %8.1f%%   +-%.1f%%'
          % (k, p*1e3, rm*1e3, 100*(p-rm)/rm, 100*sd/rm))
PY
echo "### E2E DONE"
