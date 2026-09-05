cd /workspace/ATOM
export HF_HUB_OFFLINE=1
DST=/workspace/ATOM/agent_scratch/qwen3moe
# Same configuration the ep.* graphs and prices_ep were taken under.
PASSES=3 timeout 900 python agent_scratch/q27_warm.py --model $DST \
  -tp 2 --enable-expert-parallel --load_dummy=xavier \
  --gpu-memory-utilization 0.35 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --no-enable_prefix_caching --num-prompts 4 --prompt-tokens 64 --max-tokens 8 \
  --compass --compass-mode measure --compass-measure-out compass_ops/moe_steps.jsonl \
  --out compass_ops/moe_real.json 2>&1 | grep "###"
python3 - <<'PY'
import json, glob, statistics, sys
sys.path.insert(0, '/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of
rows=[json.loads(l) for l in open(sorted(glob.glob('compass_ops/moe_steps*.jsonl'))[0])]
pre=[r['seconds'] for r in rows if r['num_prefill_tokens']]
dec=[r['seconds'] for r in rows if not r['num_prefill_tokens']]
warm_pre = statistics.median(pre[1:]) if len(pre)>1 else pre[-1]
warm_dec = statistics.median(dec[1:])
prices=json.load(open('compass_ops/prices_ep.tp0.json'))['prices']
def priced(g):
    ks=[prices[signature_of(o)]['seconds'] for o in json.load(open(g))['ops']
        if signature_of(o) in prices]
    return ks
kp = priced('compass_ops/ep.prefill.tp0.json')
kd = priced('compass_ops/ep.tp0.json')
def solve(ks, overhead):
    lo, hi = 0.0, overhead + max(ks) + 1e-3
    for _ in range(200):
        mid=(lo+hi)/2
        if sum(max(0.0, mid-k) for k in ks) < overhead: lo=mid
        else: hi=mid
    return (lo+hi)/2
print()
print('prefill steps (ms):', [round(x*1e3,1) for x in pre])
print('warmed prefill %.3f ms   priced %.3f ms over %d ops'
      % (warm_pre*1e3, sum(kp)*1e3, len(kp)))
D = solve(kp, warm_pre - sum(kp))
print('dispatch D = %.2f us   (0.6B gave 132.70, 27B gave 101.22, default 130)' % (D*1e6))
pred = sum(kp) + sum(max(0.0, 130e-6-k) for k in kp)
print('at the shared 130us: prefill %.3f ms vs %.3f real -> %+.1f%%'
      % (pred*1e3, warm_pre*1e3, 100*(pred-warm_pre)/warm_pre))
predd = sum(kd) + len(kd)*2.25e-6
print('decode: priced %.3f + %d launches x 2.25us = %.3f vs %.3f -> %+.1f%%'
      % (sum(kd)*1e3, len(kd), predd*1e3, warm_dec*1e3, 100*(predd-warm_dec)/warm_dec))
PY
echo "### MOE DISP DONE"
