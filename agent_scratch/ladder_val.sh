cd /workspace/ATOM
export HF_HUB_OFFLINE=1
for N in 1 2 4 8 16 32 64; do
  timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
    --compass-mode measure --compass-measure-out compass_ops/lv_b$N.jsonl \
    --num-prompts $N --prompt-tokens 64 --max-tokens 12 \
    --out compass_ops/lv_b$N.json > /dev/null 2>&1
done
python3 - <<'PY'
import json, glob, statistics, sys
sys.path.insert(0,'/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of
pl=json.load(open('compass_ops/prices_ladder.json'))['prices']
print('%5s %11s %11s %11s %9s %9s' % ('rung','priced','measured','overhead','per launch','bucket'))
bs=[]
for n in (1,2,4,8,16,32,64):
    g=json.load(open('compass_ops/ladder.b%d.json'%n))
    tot=0.0; L=0
    for op in g['ops']:
        e=pl.get(signature_of(op))
        if e is None: continue
        tot+=e['seconds']; L+=max(1,len(e.get('kernels') or {}))
    rows=[json.loads(x) for x in open(glob.glob('compass_ops/lv_b%d*.jsonl'%n)[0])]
    dec=[r for r in rows if not r['num_prefill_tokens']]
    step=statistics.median([r['seconds'] for r in dec][1:])
    bucket=dec[-1].get('capture_bucket')
    b=(step-tot)/L
    bs.append(b)
    print('%5d %10.3fms %10.3fms %10.3fms %8.2fus %9s' % (n,tot*1e3,step*1e3,(step-tot)*1e3,b*1e6,bucket))
print()
print('per-launch constant across rungs: %.2f to %.2f us, median %.2f'
      % (min(bs)*1e6, max(bs)*1e6, statistics.median(bs)*1e6))
PY
echo "### LADDER VAL DONE"
