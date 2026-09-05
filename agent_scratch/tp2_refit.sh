cd /workspace/ATOM
export HF_HUB_OFFLINE=1
ADM=$(cat compass_ops/t2_admission.txt)
python3 -c "
K_D, S_D, L = 2.502e-3, 3.369e-3, 441      # decode kernels, real step, launches
K_P, S_P, O = 14.877e-3, 50.114e-3, 381    # prefill kernels, real step, operators
print('refit boundary %.2f us/launch (TP=1 gave 2.25)' % ((S_D-K_D)/L*1e6))
print('refit eager    %.2f us/op     (TP=1 gave 86.35)' % ((S_P-K_P)/O*1e6))
"
python scripts/compass/run.py --model Qwen/Qwen3-0.6B -tp 2 --compass \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices_tp2.json \
  --compass-oracle-option graph=compass_ops/gt.json \
  --compass-oracle-option prefill_graph=compass_ops/gt.prefill.json \
  --compass-oracle-option boundary_seconds=1.966e-6 \
  --compass-oracle-option eager_seconds_per_op=92.48e-6 \
  --compass-admission-seconds "$ADM" \
  --out compass_ops/t2_refit.json 2>&1 | grep -i "active" | tail -1
python3 - <<'PY'
import json, statistics
def m(p,k):
    d=json.load(open(p)); return sum(r[k] for r in d['requests'])/len(d['requests'])
print()
print('%-10s %11s %11s %11s   %s' % ('TP=2','tp1 consts','refitted','real (n=3)','noise'))
for k in ('ttft','tpot','latency'):
    v=[m('compass_ops/t2_plain%d.json'%i,k) for i in (1,2,3)]
    r,s=statistics.mean(v),statistics.stdev(v)
    a,b=m('compass_ops/t2_pred.json',k),m('compass_ops/t2_refit.json',k)
    print('%-10s %+9.1f%% %+10.1f%% %9.3f ms   +-%.1f%%'
          % (k,100*(a-r)/r,100*(b-r)/r,r*1e3,100*s/r))
PY
echo "### TP2 REFIT DONE"
