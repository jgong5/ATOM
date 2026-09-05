cd /workspace/ATOM
export HF_HUB_OFFLINE=1
echo "=== price the ladder and the prefill graph together ==="
timeout 1800 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --compass-mode measure --compass-measure-out compass_ops/sv.jsonl \
  --compass-bench-graph "$(ls compass_ops/ladder.b*.json compass_ops/gp.prefill.json | paste -sd,)" \
  --compass-bench-out compass_ops/prices_serve.json --compass-bench-iters 300 \
  --compass-bench-cache graph --num-prompts 4 --prompt-tokens 64 --max-tokens 4 \
  --out compass_ops/sv.json 2>&1 | grep -iE "priced|WARNING" | tail -2

rm -rf compass_serve2 && mkdir -p compass_serve2
echo "=== real ==="
bash scripts/compass/serve_probe.sh real compass_serve2/real >/dev/null 2>&1
echo "=== predicted, decode ladder + prefill graph ==="
bash scripts/compass/serve_probe.sh predict compass_serve2/predict \
  --compass \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices_serve.json \
  --compass-oracle-option 'graph=compass_ops/ladder.b*.json' \
  --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json \
  --compass-admission-seconds 0.0132 >/dev/null 2>&1
python scripts/compass/serve_compare.py compass_serve2/real compass_serve2/predict
echo "### SERVE2 DONE"
