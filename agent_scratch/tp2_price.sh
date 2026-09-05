cd /workspace/ATOM
export HF_HUB_OFFLINE=1
# The union of both ranks' graphs, so every rank prices the same signature list
# in the same order. A collective needs all ranks calling it in lockstep; lists
# that differ by one entry deadlock rather than fail.
timeout 420 python scripts/compass/run.py --model Qwen/Qwen3-0.6B -tp 2 --compass \
  --num-prompts 4 --prompt-tokens 64 --max-tokens 32 \
  --compass-mode measure --compass-measure-out compass_ops/t2_price.jsonl \
  --compass-bench-graph 'compass_ops/gt.tp*.json,compass_ops/gt.prefill.tp*.json' \
  --compass-bench-out compass_ops/prices_tp2.json \
  --compass-bench-cache graph --out compass_ops/t2_price.json 2>&1 | grep -i "priced\|error\|Traceback" | tail -6
echo "PRICE RC=$?"
ls -1 compass_ops/prices_tp2*.json 2>/dev/null
echo "### TP2 PRICE DONE"
