cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -rf compass_serve3 && mkdir -p compass_serve3
bash scripts/compass/serve_probe.sh real compass_serve3/real \
  --compass --compass-mode measure \
  --compass-measure-out compass_serve3/real_steps.jsonl >/dev/null 2>&1
bash scripts/compass/serve_probe.sh predict compass_serve3/predict \
  --compass \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices_serve.json \
  --compass-oracle-option 'graph=compass_ops/ladder.b*.json' \
  --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json \
  --compass-measure-out compass_serve3/pred_steps.jsonl \
  --compass-admission-seconds 0.0132 >/dev/null 2>&1
python scripts/compass/serve_compare.py compass_serve3/real compass_serve3/predict 2>&1 | sed -n '1,14p'
python3 - <<'PY'
import json, collections, statistics
def load(p):
    return [json.loads(l) for l in open(p)]
for label, path in (("real", "compass_serve3/real_steps.jsonl"),
                    ("simulated", "compass_serve3/pred_steps.jsonl")):
    try: rows = load(path)
    except OSError: print(label, "missing"); continue
    pre=[r for r in rows if r['num_prefill_tokens']]
    dec=[r for r in rows if not r['num_prefill_tokens']]
    print()
    print('%s: %d prefill, %d decode' % (label, len(pre), len(dec)))
    print('   prefill tokens:', sorted(collections.Counter(r['num_prefill_tokens'] for r in pre).items())[:6])
    print('   prefill total : %.1f ms' % (sum(r['seconds'] for r in pre)*1e3))
    print('   decode buckets:', sorted(collections.Counter(r['capture_bucket'] for r in dec).items())[:8])
    print('   decode total  : %.1f ms' % (sum(r['seconds'] for r in dec)*1e3))
PY
echo "### SERVE3 DONE"
