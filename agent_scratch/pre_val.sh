cd /workspace/ATOM
export HF_HUB_OFFLINE=1
source /workspace/ATOM/agent_scratch/pick_gpu.sh
export HIP_VISIBLE_DEVICES=$(pick_gpus 1)
echo "using GPU $HIP_VISIBLE_DEVICES"
for N in 1 2 4 6 8 12; do
  timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
    --gpu-memory-utilization 0.30 --compass-mode measure \
    --compass-measure-out compass_ops/pm_n$N.jsonl --compass-trace-prefill 2 \
    --no-enable_prefix_caching --num-prompts $N --prompt-tokens 200 --max-tokens 3 \
    --out compass_ops/pm_n$N.json > /dev/null 2>&1
  echo "  n=$N rc=$?"
done
python3 - <<'PY'
import json, glob, sys
sys.path.insert(0,'/workspace/ATOM')
from atom.compass.runtime.microbench import signature_of
pl=json.load(open('compass_ops/prices_pre.json'))['prices']
print()
print('%7s %8s %10s %10s %11s %8s' % ('prompts','tokens','priced','measured','pred D=130','err'))
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
    if meas: pts.append((toks,meas,pred))
    print('%7d %8d %9.3fms %9s %10.3fms %7s' % (N,toks,sum(ks)*1e3,
          ('%.3fms'%(meas*1e3)) if meas else 'n/a', pred*1e3,
          ('%+.1f%%'%(100*(pred-meas)/meas)) if meas else '-'))
if len(pts)>2:
    print()
    print('does a measured step interpolate between its neighbours?')
    errs=[]
    for i in range(1,len(pts)-1):
        t,m,_=pts[i]; lo=pts[i-1]; hi=pts[i+1]
        f=(t-lo[0])/(hi[0]-lo[0]); est=lo[1]+f*(hi[1]-lo[1])
        errs.append(abs(100*(est-m)/m))
        print('   %6d tokens: interp %7.3fms vs measured %7.3fms  %+.1f%%' % (t,est*1e3,m*1e3,100*(est-m)/m))
    errs.sort()
    print('   median %.1f%%, worst %.1f%%' % (errs[len(errs)//2], errs[-1]))
PY
echo "### PRE VAL DONE"
