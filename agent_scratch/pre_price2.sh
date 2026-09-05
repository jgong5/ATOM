cd /workspace/ATOM
export HF_HUB_OFFLINE=1
source /workspace/ATOM/agent_scratch/pick_gpu.sh
export HIP_VISIBLE_DEVICES=$(pick_gpus 1)
echo "using GPU $HIP_VISIBLE_DEVICES"
ls -la compass_ops/pre.n*.prefill.json | head
GRAPHS=$(ls compass_ops/pre.n1.prefill.json compass_ops/pre.n2.prefill.json \
            compass_ops/pre.n4.prefill.json compass_ops/pre.n6.prefill.json \
            compass_ops/pre.n8.prefill.json compass_ops/pre.n12.prefill.json 2>&1 | paste -sd,)
echo "graphs: $GRAPHS"
timeout 1800 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --gpu-memory-utilization 0.30 \
  --compass-mode measure --compass-measure-out compass_ops/pp.jsonl \
  --compass-bench-graph "$GRAPHS" \
  --compass-bench-out compass_ops/prices_pre.json --compass-bench-iters 60 \
  --compass-bench-cache graph --num-prompts 2 --prompt-tokens 64 --max-tokens 3 \
  --out compass_ops/pp.json > compass_ops/pp.log 2>&1
echo "rc=$?"
grep -vE "^\[aiter\]|clang\+\+|Engine kwargs" compass_ops/pp.log | grep -iE "priced|error|Traceback|assert|matched no" | tail -8
echo "### PRE PRICE2 DONE"
