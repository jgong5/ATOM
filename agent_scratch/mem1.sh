cd /workspace/ATOM
export HF_HUB_OFFLINE=1
echo "=== 0.6B TP=1 ==="
timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
  --compass-mode measure --compass-measure-out compass_ops/m1.jsonl \
  --compass-memory-out compass_ops/mem_q06_tp1.json \
  --num-prompts 2 --prompt-tokens 32 --max-tokens 4 \
  --out compass_ops/m1.json 2>&1 | grep -iE "recorded the memory|WARNING: could not" | tail -2
echo "=== 27B TP=2 ==="
timeout 900 python scripts/compass/run.py --model Qwen/Qwen3.8-27B -tp 2 --compass \
  --gpu-memory-utilization 0.40 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --compass-mode measure --compass-measure-out compass_ops/m2.jsonl \
  --compass-memory-out compass_ops/mem_q27_tp2.json \
  --num-prompts 2 --prompt-tokens 32 --max-tokens 4 \
  --out compass_ops/m2.json 2>&1 | grep -iE "recorded the memory|WARNING: could not" | tail -2
python3 -c "
import json, glob
for p in sorted(glob.glob('compass_ops/mem_q*.json')):
    d=json.load(open(p)); r=d['readings']; G=1<<30
    print()
    print(p, d['config']['model'], 'tp=%s' % d['config']['topology'].get('tp',1))
    for k in ('total','free','peak_torch','non_torch','cudagraph_overhead'):
        print('   %-20s %8.2f GB' % (k, r[k]/G))
    print('   %-20s %s blocks' % ('-> num_kvcache_blocks', d['blocks']['num_kvcache_blocks']))
"
echo "### MEM1 DONE"
