cd /workspace/ATOM
export HF_HUB_OFFLINE=1
TP=$1
M="--model Qwen/Qwen3.8-27B -tp $TP --gpu-memory-utilization ${UTIL:-0.40} --max-num-seqs ${SEQS:-512} \
   --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable_prefix_caching"
W="--num-prompts 4 --prompt-tokens 64 --max-tokens 16"
G=compass_ops/s${TP}.json
P=compass_ops/sp${TP}.json
rm -f compass_ops/s${TP}.*json compass_ops/sp${TP}.*json

echo "=== tp=$TP trace ==="
timeout 900 python scripts/compass/run.py $M $W --compass --compass-mode trace \
  --compass-graph-out $G --compass-trace-prefill 2 \
  --out compass_ops/s${TP}_trace.json > compass_ops/s${TP}_trace.log 2>&1
echo "  rc=$?"; grep -i traced compass_ops/s${TP}_trace.log | tail -2

echo "=== tp=$TP price ==="
timeout 1500 python scripts/compass/run.py $M $W --compass --compass-mode measure \
  --compass-measure-out compass_ops/s${TP}_p.jsonl \
  --compass-bench-graph "$(ls compass_ops/s${TP}.json compass_ops/s${TP}.tp*.json compass_ops/s${TP}.prefill.json compass_ops/s${TP}.prefill.tp*.json 2>/dev/null | paste -sd,)" \
  --compass-bench-out $P --compass-bench-iters 300 --compass-bench-cache graph \
  --out compass_ops/s${TP}_price.json > compass_ops/s${TP}_price.log 2>&1
echo "  rc=$?"; grep -i priced compass_ops/s${TP}_price.log | tail -1

echo "=== tp=$TP real, warmed, measure mode ==="
PASSES=4 timeout 900 python agent_scratch/q27_warm.py $M $W \
  --compass --compass-mode measure --compass-measure-out compass_ops/s${TP}_r.jsonl \
  --out compass_ops/s${TP}_real.json 2>&1 | grep "###"

ADM=$(python3 - <<PY
import json, glob, statistics
real=json.load(open('compass_ops/s${TP}_real.json'))['warm']
rows=[json.loads(l) for l in open(sorted(glob.glob('compass_ops/s${TP}_r*.jsonl'))[0])]
pre=[r['seconds'] for r in rows if r['num_prefill_tokens']]
warm=statistics.median(pre[1:]) if len(pre)>1 else pre[-1]
print(max(0.0, sum(x['ttft'] for x in real)/len(real) - warm))
PY
)
echo "  admission $ADM s (same mode as the baseline)"

echo "=== tp=$TP predicted ==="
PASSES=1 timeout 900 python agent_scratch/q27_warm.py $M $W \
  --compass --compass-mode predict \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=$P \
  --compass-oracle-option graph=$G \
  --compass-oracle-option prefill_graph=compass_ops/s${TP}.prefill.json \
  --compass-admission-seconds "$ADM" \
  --out compass_ops/s${TP}_pred.json 2>&1 | grep -E "###|active" | tail -2

python3 - <<PY
import json, statistics
real=json.load(open('compass_ops/s${TP}_real.json'))['warm']
pred=json.load(open('compass_ops/s${TP}_pred.json'))['passes'][0]
print()
print('Qwen3.8-27B TP=${TP}')
print('%-9s %12s %12s %9s   %s' % ('','predicted','real (n=3)','err','noise'))
for k in ('ttft','tpot','latency'):
    r=[x[k] for x in real]; rm=sum(r)/len(r); sd=statistics.stdev(r)
    print('%-9s %10.3f ms %10.3f ms %8.2f%%   +-%.1f%%'
          % (k,pred[k]*1e3,rm*1e3,100*(pred[k]-rm)/rm,100*sd/rm))
PY
echo "### SWEEP TP=$TP DONE"
