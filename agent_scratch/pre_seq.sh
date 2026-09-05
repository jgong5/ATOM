cd /workspace/ATOM
export HF_HUB_OFFLINE=1
source /workspace/ATOM/agent_scratch/pick_gpu.sh
export HIP_VISIBLE_DEVICES=$(pick_gpus 1)
echo "using GPU $HIP_VISIBLE_DEVICES"
# Roughly constant total tokens, varying how many sequences carry them.
# 1x800, 2x400, 4x200, 8x100 words -- the earlier sweep moved tokens and
# sequences together, so nothing in it can tell the two apart.
run() {  # $1 = prompts, $2 = words each
  timeout 600 python scripts/compass/run.py --model Qwen/Qwen3-0.6B --compass \
    --gpu-memory-utilization 0.30 --compass-mode measure \
    --compass-measure-out "compass_ops/sq_${1}x${2}.jsonl" --compass-trace-prefill 2 \
    --no-enable_prefix_caching --num-prompts "$1" --prompt-tokens "$2" --max-tokens 3 \
    --out "compass_ops/sq_${1}x${2}.json" > /dev/null 2>&1
  echo "  ${1}x${2} rc=$?"
}
run 1 800; run 2 400; run 4 200; run 8 100
run 1 1600; run 2 800; run 4 400; run 8 200
python3 - <<'PY'
import json, glob
print()
print('%10s %6s %9s %11s %11s' % ('config','seqs','tokens','step','ms/1k tok'))
rows=[]
for p in sorted(glob.glob('compass_ops/sq_*.jsonl')):
    tag=p.split('sq_')[1].split('.jsonl')[0]
    n=int(tag.split('x')[0])
    recs=[json.loads(l) for l in open(p)]
    pre=[r for r in recs if r['num_prefill_tokens']]
    if not pre: continue
    r=pre[-1]
    rows.append((r['num_prefill_tokens'], n, r['seconds'], tag))
for toks,n,sec,tag in sorted(rows):
    print('%10s %6d %9d %10.3fms %10.3f' % (tag, n, toks, sec*1e3, sec*1e3/(toks/1000)))
print()
print('grouped by comparable token count:')
rows.sort()
import itertools
for lo,hi in ((3000,5000),(6000,10000),(12000,17000)):
    grp=[r for r in rows if lo<=r[0]<hi]
    if len(grp)>1:
        print('  ~%d-%d tokens:' % (lo,hi))
        for toks,n,sec,tag in grp:
            print('     %2d seqs, %5d tok: %8.3f ms' % (n,toks,sec*1e3))
PY
echo "### PRE SEQ DONE"
