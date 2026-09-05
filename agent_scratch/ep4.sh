cd /workspace/ATOM
export HF_HUB_OFFLINE=1 ATOM_LOG_RANK_COORDS=1
DST=/workspace/ATOM/agent_scratch/qwen3moe
rm -f compass_ops/ep4*.json
echo "=== ep_size=4, level 0, no torch.compile ==="
timeout 700 python scripts/compass/run.py --model $DST \
  -tp 2 --data-parallel-size 2 --enable-dp-attention --enable-expert-parallel \
  --level 0 --load_dummy=xavier --gpu-memory-utilization 0.30 \
  --max-model-len 4096 --max-num-batched-tokens 4096 \
  --num-prompts 4 --prompt-tokens 32 --max-tokens 4 \
  --compass --compass-mode trace --compass-graph-out compass_ops/ep4.json \
  --out compass_ops/ep4_run.json > compass_ops/ep4.log 2>&1
echo "  rc=$?"
grep -iE "traced|MOE EP ON|Error|Traceback" compass_ops/ep4.log | grep -v "^\[aiter\]" | tail -4
python3 - <<'PY'
import glob, json, collections
for p in sorted(glob.glob('compass_ops/ep4*.json')):
    try: g=json.load(open(p))
    except Exception: continue
    ops=g.get('ops') or []
    if not ops: continue
    a2a=[o for o in ops if any(k in o['name'].lower() for k in
         ('all_to_all','a2a','dispatch','combine','moe','expert','shuffle'))]
    print('%-32s %4d ops  %s' % (p, len(ops),
          collections.Counter(o['name'] for o in a2a).most_common(6)))
    print('   grouped:', collections.Counter(o['name'] for o in ops if o.get('group')).most_common(5))
PY
echo "### EP4 DONE"
