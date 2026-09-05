cd /workspace/ATOM
export HF_HUB_OFFLINE=1
M="--model Qwen/Qwen3.8-27B -tp 2 --gpu-memory-utilization 0.40 \
   --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable_prefix_caching"
# Admission was measured on the 16-token workload: 18.447 ms. Predict a 32-token
# one with it, unchanged. Same graphs (prefill and decode shapes are identical),
# different run: twice the decode steps and a different latency.
ADM=0.018447

echo "=== real, 32 output tokens (4 passes; pass 1 is compilation) ==="
PASSES=4 timeout 900 python agent_scratch/q27_warm.py $M \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 \
  --out compass_ops/ho_real.json 2>&1 | grep "###"

echo "=== predicted, admission carried over unchanged ==="
PASSES=1 timeout 900 python agent_scratch/q27_warm.py $M \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 \
  --compass --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices27.json \
  --compass-oracle-option graph=compass_ops/g27.json \
  --compass-oracle-option prefill_graph=compass_ops/g27.prefill.json \
  --compass-admission-seconds $ADM \
  --out compass_ops/ho_pred.json 2>&1 | grep "###"

python3 -c "
import json, statistics
real=json.load(open('compass_ops/ho_real.json'))['warm']
pred=json.load(open('compass_ops/ho_pred.json'))['passes'][0]
print()
print('held out: admission from the 16-token run, predicting 32 tokens')
print('%-9s %12s %12s %9s   %s' % ('','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    r=[x[k] for x in real]; rm=sum(r)/len(r); sd=statistics.stdev(r)
    print('%-9s %10.3f ms %10.3f ms %8.2f%%   +-%.1f%%'
          % (k,pred[k]*1e3,rm*1e3,100*(pred[k]-rm)/rm,100*sd/rm))
"
echo "### HELDOUT ADM DONE"
