cd /workspace/ATOM
export HF_HUB_OFFLINE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
echo "=== price the unchunked prefill family ==="
timeout 1800 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --compass-mode measure --compass-measure-out compass_ops/pp.jsonl \
  --compass-bench-graph "$(ls compass_ops/pre.n1.prefill.json compass_ops/pre.n2.prefill.json compass_ops/pre.n4.prefill.json compass_ops/pre.n6.prefill.json compass_ops/pre.n8.prefill.json compass_ops/pre.n12.prefill.json | paste -sd,)" \
  --compass-bench-out compass_ops/prices_pre.json --compass-bench-iters 200 \
  --compass-bench-cache graph --num-prompts 2 --prompt-tokens 64 --max-tokens 3 \
  --out compass_ops/pp.json 2>&1 | grep -iE "priced|WARNING" | tail -2

echo "=== measure real prefill steps at the same sizes ==="
for N in 1 2 4 6 8 12; do
  timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
    --compass-mode measure --compass-measure-out compass_ops/pm_n$N.jsonl \
    --compass-trace-prefill 2 --no-enable_prefix_caching \
    --num-prompts $N --prompt-tokens 200 --max-tokens 3 \
    --out compass_ops/pm_n$N.json > /dev/null 2>&1
done
python3 - <<'PY'
import json, glob, sys
sys.path.insert(0,'/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of
pl=json.load(open('compass_ops/prices_pre.json'))['prices']
print()
print('%8s %8s %11s %11s %11s %8s' % ('prompts','tokens','priced','measured','pred(D=130)','err'))
pts=[]
for N in (1,2,4,6,8,12):
    g=json.load(open('compass_ops/pre.n%d.prefill.json'%N))
    sh=(g.get('provenance') or {}).get('shape') or {}
    toks=sh.get('num_prefill_tokens',0)
    ks=[pl[signature_of(o)]['seconds'] for o in g['ops'] if signature_of(o) in pl]
    rows=[json.loads(l) for l in open(glob.glob('compass_ops/pm_n%d*.jsonl'%N)[0])]
    pre=[r['seconds'] for r in rows if r['num_prefill_tokens']]
    meas=pre[-1] if pre else None
    pred=sum(ks)+sum(max(0.0,130e-6-k) for k in ks)
    pts.append((toks,sum(ks),meas,pred))
    print('%8d %8d %10.3fms %10s %10.3fms %7s' % (N,toks,sum(ks)*1e3,
          ('%.3fms'%(meas*1e3)) if meas else 'n/a', pred*1e3,
          ('%+.1f%%'%(100*(pred-meas)/meas)) if meas else '-'))
print()
ok=[(t,p,m) for t,k,m,p in pts if m]
if len(ok)>2:
    print('interpolating the measured step between neighbours:')
    for i in range(1,len(ok)-1):
        t,_,m=ok[i]; lo=ok[i-1]; hi=ok[i+1]
        f=(t-lo[0])/(hi[0]-lo[0]); est=lo[2]+f*(hi[2]-lo[2])
        print('   %6d tokens: interp %7.3fms vs %7.3fms  %+.1f%%' % (t,est*1e3,m*1e3,100*(est-m)/m))
PY
echo "### PRE PRICE DONE"
