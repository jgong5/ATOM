cd /workspace/ATOM
export HF_HUB_OFFLINE=1 COMPASS_PRICE_KERNELS=0
M="--model Qwen/Qwen3.8-27B -tp 4 --gpu-memory-utilization 0.40 --max-num-seqs 512 \
   --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable_prefix_caching"
W="--num-prompts 4 --prompt-tokens 64 --max-tokens 16"
echo "=== price, kernel breakdown off ==="
timeout 1800 python scripts/compass/run.py $M $W --compass --compass-mode measure \
  --compass-measure-out compass_ops/s4_p2.jsonl \
  --compass-bench-graph "$(ls compass_ops/s4.tp*.json compass_ops/s4.prefill.tp*.json | paste -sd,)" \
  --compass-bench-out compass_ops/sp4.json --compass-bench-iters 300 \
  --compass-bench-cache graph --out compass_ops/s4_price2.json 2>&1 | grep -iE "priced|WARNING" | tail -2
echo "=== predict ==="
PASSES=1 timeout 900 python agent_scratch/q27_warm.py $M $W \
  --compass --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/sp4.json \
  --compass-oracle-option graph=compass_ops/s4.json \
  --compass-oracle-option prefill_graph=compass_ops/s4.prefill.json \
  --compass-admission-seconds 0.005605389436086017 \
  --out compass_ops/s4_pred.json 2>&1 | grep -E "###|active" | tail -2
python3 -c "
import json, statistics
real=json.load(open('compass_ops/s4_real.json'))['warm']
pred=json.load(open('compass_ops/s4_pred.json'))['passes'][0]
print()
print('Qwen3.8-27B TP=4')
print('%-9s %12s %12s %9s   %s' % ('','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    r=[x[k] for x in real]; rm=sum(r)/len(r); sd=statistics.stdev(r)
    print('%-9s %10.3f ms %10.3f ms %8.2f%%   +-%.1f%%'
          % (k,pred[k]*1e3,rm*1e3,100*(pred[k]-rm)/rm,100*sd/rm))
"
echo "### TP4 FINISH DONE"
