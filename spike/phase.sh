#!/bin/bash
# usage: phase.sh CELL -- as root in a gate container, detached: replays of #1498's CELL records (all 16 ranks), P at a
# time. A: graph dumps at b26c26217 (decode graphs only; prefill priced, so its per-step wall at b26c26217 too).
# B: every line of every rank at the tip on those graphs and #1498's price list. C: decode windows (3 x 5,000 lines a
# rank) at b26c26217 and at 3911f83ba. Writes logs/phase-CELL.log, ending PHASE_RC.
CELL=$1 P=${P:-8}
S=/shared_nfs/xingjche/agent_scratch/3636 G=$S/out/graphs/$CELL
PL=/shared_nfs/xingjche/agent_scratch/1498/gt/store/price_list/sha256-5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f5-faecca3f3a7b
exec >> $S/logs/phase-$CELL.log 2>&1
mkdir -p $G $S/out/rep/$CELL
say() { echo "$* $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"; }
par() { local p=${1:-$P}; xargs -P $p -I{} bash -c '{}'; }
pl() { echo $PL/params.dp$1of8.pp0of1.pcp0of1.tp0of1.json; }
say "phase A: dumps at b26c26217"
{ for r in 0 1 2 3 4 5 6 7; do
    [[ -s $G/decode$r.pkl ]] || echo "bash $S/run/rep.sh t4 $CELL decode $r $CELL/dump_d$r --graphs-only --dump-graphs $G/decode$r.pkl"
    [[ -s $G/prefill$r.pkl ]] || echo "bash $S/run/rep.sh t4 $CELL prefill $r $CELL/t4_p$r --params $(pl $r) --dump-graphs $G/prefill$r.pkl --digest --out $S/out/rep/$CELL/t4_p$r.jsonl"
  done; } | par
say "phase B: every line at the tip"
for r in 0 1 2 3 4 5 6 7; do
  echo "bash $S/run/rep.sh tip $CELL decode $r $CELL/tip_d$r --graphs $G/decode$r.pkl --params $(pl $r) --digest --out $S/out/rep/$CELL/tip_d$r.jsonl"
  echo "bash $S/run/rep.sh tip $CELL prefill $r $CELL/tip_p$r --graphs $G/prefill$r.pkl --params $(pl $r) --digest --out $S/out/rep/$CELL/tip_p$r.jsonl"
done | par
say "phase C: decode windows at b26c26217 and 3911f83ba"
for t in t4 t5; do for r in 0 1 2 3 4 5 6 7; do for w in ${WINDOWS:-0 100000 200000}; do
  echo "bash $S/run/rep.sh $t $CELL decode $r $CELL/${t}_d${r}_w$w --skip $w --first 5000 --graphs $G/decode$r.pkl --params $(pl $r) --digest --out $S/out/rep/$CELL/${t}_d${r}_w$w.jsonl"
done; done; done | par
chown -R 50109788:1000 $S/out $S/logs
say "PHASE_RC=0"
