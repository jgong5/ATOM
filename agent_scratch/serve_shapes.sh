cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -rf compass_serve/meas && mkdir -p compass_serve/meas
bash scripts/compass/serve_probe.sh real compass_serve/meas \
  --compass --compass-mode measure \
  --compass-measure-out compass_serve/meas/steps.jsonl >/dev/null 2>&1
python3 - <<'PY'
import json, collections, statistics, glob
f=sorted(glob.glob('compass_serve/meas/steps*.jsonl'))
rows=[json.loads(l) for l in open(f[0])]
pre=[r for r in rows if r['num_prefill_tokens']]
dec=[r for r in rows if not r['num_prefill_tokens']]
print('steps: %d prefill, %d decode' % (len(pre), len(dec)))
print()
print('prefill token counts :', sorted(collections.Counter(r['num_prefill_tokens'] for r in pre).items())[:8])
print('prefill step seconds : min %.1f  median %.1f  max %.1f ms' % (
    min(r['seconds'] for r in pre)*1e3, statistics.median([r['seconds'] for r in pre])*1e3,
    max(r['seconds'] for r in pre)*1e3))
print()
bs=collections.Counter(len(r['num_scheduled_tokens']) for r in dec)
print('decode batch sizes   :', sorted(bs.items())[:10])
print('decode buckets       :', sorted(collections.Counter(r['capture_bucket'] for r in dec).items())[:10])
print('decode step seconds  : min %.2f  median %.2f  max %.2f ms' % (
    min(r['seconds'] for r in dec)*1e3, statistics.median([r['seconds'] for r in dec])*1e3,
    max(r['seconds'] for r in dec)*1e3))
print()
print('the oracle answers every prefill with ~47ms and every decode with ~3.2ms')
PY
echo "### SHAPES DONE"
