cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -rf compass_adm && mkdir -p compass_adm
for A in 0 0.0132 0.0264; do
  bash scripts/compass/serve_probe.sh predict compass_adm/a$A \
    --compass \
    --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
    --compass-oracle-option prices=compass_ops/prices_serve.json \
    --compass-oracle-option 'graph=compass_ops/ladder.b*.json' \
    --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json \
    --compass-measure-out compass_adm/steps_a$A.jsonl \
    --compass-admission-seconds $A >/dev/null 2>&1
  echo "  admission=$A done"
done
python3 - <<'PY'
import json, collections
def summarise(label, path):
    try: rows=[json.loads(l) for l in open(path)]
    except OSError: print('%-22s missing' % label); return
    dec=[r for r in rows if not r['num_prefill_tokens']]
    pre=[r for r in rows if r['num_prefill_tokens']]
    buckets=sorted(collections.Counter(r['capture_bucket'] for r in dec).items())
    print('%-22s %3d prefill %4d decode   buckets %s' % (label, len(pre), len(dec), buckets))
print()
summarise('real', 'compass_serve3/real_steps.jsonl')
for A in ('0','0.0132','0.0264'):
    summarise('simulated adm=%s' % A, 'compass_adm/steps_a%s.jsonl' % A)
PY
echo "### ADM SWEEP DONE"
