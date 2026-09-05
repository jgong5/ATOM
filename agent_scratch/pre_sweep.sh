cd /workspace/ATOM
export HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
rm -f compass_ops/pre.n*.json
# Fixed 200-word prompts, varying how many arrive together, so each run has one
# prefill step of a known size. ~4.9 tokens per word here, so 1..32 prompts
# spans roughly 1k to 16k tokens -- the range chunked prefill actually produces,
# capped by max_num_batched_tokens.
for N in 1 2 4 6 8 12 16 24 32; do
  timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
    --compass-mode trace --compass-graph-out compass_ops/pre.n$N.json \
    --compass-trace-prefill 2 --no-enable_prefix_caching \
    --num-prompts $N --prompt-tokens 200 --max-tokens 3 \
    --out compass_ops/pre_run_n$N.json > compass_ops/pre_n$N.log 2>&1
  echo "  n=$N rc=$?"
done
python3 - <<'PY'
import json, glob
print()
print('%6s %8s %10s' % ('prompts','ops','prefill tokens'))
for p in sorted(glob.glob('compass_ops/pre.n*.prefill.json'),
                key=lambda x:int(x.split('.n')[1].split('.')[0])):
    g=json.load(open(p)); sh=(g.get('provenance') or {}).get('shape') or {}
    print('%6d %8d %10d' % (len(sh.get('num_scheduled_tokens') or []), len(g['ops']),
                            sh.get('num_prefill_tokens', 0)))
PY
echo "### PRE SWEEP DONE"
