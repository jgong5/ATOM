cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -f compass_ops/ladder.b*.json
# The capture ladder is [1,2,4,8,16,32,48,64,128,256,512]; a decode batch of
# exactly N replays rung N, so tracing at these N gives the rung shapes without
# any shape rewriting. Trace mode runs eager, so the traced batch is the real
# batch -- which is why N must be a rung and not merely near one.
for N in 1 2 4 8 16 32 64; do
  timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
    --compass-mode trace --compass-graph-out compass_ops/ladder.b$N.json \
    --compass-memory-out compass_ops/mem_ladder_b$N.json \
    --num-prompts $N --prompt-tokens 64 --max-tokens 6 \
    --out compass_ops/ladder_run_b$N.json > compass_ops/ladder_b$N.log 2>&1
  echo "  b$N rc=$? $(grep -c traced compass_ops/ladder_b$N.log) graph(s)"
done
python3 - <<'PY'
import json, glob, collections
print()
print('%-8s %6s %8s  %s' % ('graph', 'ops', 'distinct', 'batch (from provenance)'))
sigs = {}
for p in sorted(glob.glob('compass_ops/ladder.b*.json'),
                key=lambda x: int(x.split('.b')[1].split('.')[0])):
    g = json.load(open(p))
    ops = g['ops']
    shape = (g.get('provenance') or {}).get('shape') or {}
    n = len(shape.get('num_scheduled_tokens') or [])
    names = collections.Counter(o['name'] for o in ops)
    sigs[n] = tuple(sorted(names.items()))
    print('%-8s %6d %8d  %s' % (p.split('/')[-1][:8], len(ops), len(names), n))
# Does the operator sequence depend on the rung, or only the shapes?
distinct = {v for v in sigs.values()}
print()
print('distinct operator multisets across rungs: %d' % len(distinct))
print('structure is rung-independent' if len(distinct) == 1
      else 'STRUCTURE CHANGES WITH RUNG -- worth knowing on its own')
PY
echo "### LADDER DONE"
