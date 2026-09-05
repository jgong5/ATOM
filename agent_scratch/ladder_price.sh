cd /workspace/ATOM
export HF_HUB_OFFLINE=1
timeout 1800 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --compass-mode measure --compass-measure-out compass_ops/lp.jsonl \
  --compass-bench-graph "$(ls compass_ops/ladder.b*.json | paste -sd,)" \
  --compass-bench-out compass_ops/prices_ladder.json --compass-bench-iters 300 \
  --compass-bench-cache graph --compass-memory-out compass_ops/mem_ladder_price.json \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 4 \
  --out compass_ops/lp.json 2>&1 | grep -iE "priced|WARNING" | tail -2
python3 - <<'PY'
import json, glob, sys
sys.path.insert(0,'/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of
pl=json.load(open('compass_ops/prices_ladder.json'))['prices']
print()
print('%6s %8s %10s %10s %12s' % ('rung','ops','priced','plain','kernels us'))
for p in sorted(glob.glob('compass_ops/ladder.b*.json'), key=lambda x:int(x.split('.b')[1].split('.')[0])):
    g=json.load(open(p)); n=int(p.split('.b')[1].split('.')[0])
    tot=0.0; got=0; L=0
    for op in g['ops']:
        e=pl.get(signature_of(op))
        if e is None: continue
        got+=1; tot+=e['seconds']; L+=max(1,len(e.get('kernels') or {}))
    print('%6d %8d %10d %10d %12.3f' % (n, len(g['ops']), got, L, tot*1e6))
PY
echo "### LADDER PRICE DONE"
