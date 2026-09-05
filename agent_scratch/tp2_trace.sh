cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -f compass_ops/gt*.json
python scripts/compass/run.py --model Qwen/Qwen3-0.6B -tp 2 --compass \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 \
  --compass-mode trace --compass-graph-out compass_ops/gt.json \
  --compass-trace-prefill 2 --out compass_ops/t2_trace.json 2>&1 | grep -i "traced\|error" | tail -6
ls -1 compass_ops/gt*.json
python3 -c "
import json,glob
for p in sorted(glob.glob('compass_ops/gt*.json')):
    g=json.load(open(p)); ops=g['ops']
    grouped=[o for o in ops if o.get('group')]
    from collections import Counter
    print('%-34s %4d ops, %3d grouped %s' % (p, len(ops), len(grouped),
          Counter(o['name'] for o in grouped).most_common(3)))
"
echo "### TP2 TRACE DONE"
