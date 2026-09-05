cd /workspace/ATOM
export HF_HUB_OFFLINE=1 PASSES=1
ADM=0.018447
timeout 900 python agent_scratch/q27_warm.py --model Qwen/Qwen3.8-27B -tp 2 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --no-enable_prefix_caching --num-prompts 4 --prompt-tokens 64 --max-tokens 16 \
  --compass --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices27.json \
  --compass-oracle-option graph=compass_ops/g27.json \
  --compass-oracle-option prefill_graph=compass_ops/g27.prefill.json \
  --compass-admission-seconds $ADM \
  --out compass_ops/e2e_honest.json 2>&1 | grep -E "###|active" | tail -3
python3 -c "
import json, statistics
real=json.load(open('compass_ops/e2e_real.json'))['warm']
fit=json.load(open('compass_ops/e2e_pred.json'))['passes'][0]
hon=json.load(open('compass_ops/e2e_honest.json'))['passes'][0]
print()
print('%-9s %11s %11s %11s   %s' % ('','fitted','computed','real (n=3)','noise'))
for k in ('ttft','tpot','latency'):
    r=[x[k] for x in real]; rm=sum(r)/len(r); sd=statistics.stdev(r)
    print('%-9s %+10.2f%% %+10.2f%% %9.3f ms   +-%.1f%%'
          % (k, 100*(fit[k]-rm)/rm, 100*(hon[k]-rm)/rm, rm*1e3, 100*sd/rm))
"
echo "### HONEST DONE"
