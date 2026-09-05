cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -rf compass_serve && mkdir -p compass_serve
# Real, then simulated with the priced-graph oracle. Compared on the engine's own
# clock via /compass/requests, because the HTTP benchmark times the simulator.
bash scripts/compass/serve_probe.sh real compass_serve/real
bash scripts/compass/serve_probe.sh predict compass_serve/predict \
  --compass \
  --compass-oracle atom.compass.core.cost.priced.PricedGraphCostOracle \
  --compass-oracle-option prices=compass_ops/prices9.json \
  --compass-oracle-option graph=compass_ops/gp.json \
  --compass-oracle-option prefill_graph=compass_ops/gp.prefill.json \
  --compass-admission-seconds 0.0132
python scripts/compass/serve_compare.py compass_serve/real compass_serve/predict
echo "### SERVE DONE"
