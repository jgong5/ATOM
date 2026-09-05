cd /workspace/ATOM
export HF_HUB_OFFLINE=1
timeout 900 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --compass-mode measure --compass-measure-out compass_ops/i1.jsonl \
  --compass-bench-graph compass_ops/gemm_sweep_m.json \
  --compass-bench-out compass_ops/prices_m.json \
  --compass-bench-cache graph --out compass_ops/i1.json 2>&1 | grep -i priced | tail -1
python3 - <<'PY'
import json
d = json.load(open('compass_ops/prices_m.json'))['prices']
pts = sorted((int(k.split('|')[1].split(';')[0].split(',')[0]), v['seconds']*1e6)
             for k, v in d.items())
print()
print('%6s %10s   %s' % ('M', 'us', 'us per row'))
for m, us in pts:
    print('%6d %10.2f   %8.3f' % (m, us, us/m))
# Hold out every other point and interpolate linearly between its neighbours.
known = pts[::2]
held = pts[1::2]
print()
print('%6s %10s %10s %8s' % ('M', 'actual', 'interp', 'err'))
errs = []
for m, us in held:
    lo = max((p for p in known if p[0] <= m), default=known[0])
    hi = min((p for p in known if p[0] >= m), default=known[-1])
    if hi[0] == lo[0]:
        pred = lo[1]
    else:
        t = (m - lo[0]) / (hi[0] - lo[0])
        pred = lo[1] + t * (hi[1] - lo[1])
    e = 100*(pred-us)/us
    errs.append(abs(e))
    print('%6d %10.2f %10.2f %+7.1f%%' % (m, us, pred, e))
print()
print('interpolating from half the points: median |err| %.1f%%, worst %.1f%%'
      % (sorted(errs)[len(errs)//2], max(errs)))
PY
echo "### INTERP DONE"
