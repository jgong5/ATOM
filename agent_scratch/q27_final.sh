cd /workspace/ATOM
export HF_HUB_OFFLINE=1
timeout 900 python agent_scratch/q27_warm.py --model Qwen/Qwen3.8-27B -tp 2 \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --no-enable_prefix_caching --num-prompts 4 --prompt-tokens 64 --max-tokens 16 \
  --compass --compass-mode measure \
  --compass-measure-out compass_ops/q27w.jsonl 2>&1 | grep "###"
python3 - <<'PY'
import json, glob, statistics
rows=[json.loads(l) for l in open(sorted(glob.glob('compass_ops/q27w*.jsonl'))[0])]
pre=[r['seconds'] for r in rows if r['num_prefill_tokens']]
dec=[r['seconds'] for r in rows if not r['num_prefill_tokens']]
print()
print('prefill steps (ms):', [round(x*1e3,1) for x in pre])
print('decode  steps: %d, median %.3f ms' % (len(dec), statistics.median(dec)*1e3))
warm_pre = statistics.median(pre[1:]) if len(pre) > 1 else pre[-1]
print()
print('warmed prefill step %.3f ms' % (warm_pre*1e3))
KP, OPS = 295.755e-3, 708
KD, LAUNCH = 16.173e-3, 747
D = statistics.median(dec)
print('decode : priced %.3f + %d x 2.25us = %.3f  vs %.3f  -> %+.1f%%'
      % (KD*1e3, LAUNCH, (KD+LAUNCH*2.25e-6)*1e3, D*1e3,
         100*((KD+LAUNCH*2.25e-6)-D)/D))
print('prefill: priced %.3f ms, eager constant needed = %.2f us/op (0.6B gave 86.35)'
      % (KP*1e3, (warm_pre-KP)/OPS*1e6))
PY
echo "### Q27 FINAL DONE"
