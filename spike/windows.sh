#!/bin/bash
# usage: windows.sh CELL -- in a gate container: the DP group's decode pricing wall over each window phase.sh replayed
# at b26c26217 (t4) and 3911f83ba (t5), and over the same steps of the tip's whole-record replay.
CELL=$1 O=/shared_nfs/xingjche/agent_scratch/3636/out/rep/$CELL C=/shared_nfs/xingjche/agent_scratch/3636/run/critical.py
cd $O
for w in 0 100000 200000; do
  for t in t4 t5; do python3 $C "$CELL decode lines $w+5000 $t" ${t}_d{0..7}_w$w.jsonl; done
  for r in 0 1 2 3 4 5 6 7; do sed -n "$((w + 1)),$((w + 5000))p" tip_d$r.jsonl > /tmp/tipw_$r.jsonl; done
  python3 $C "$CELL decode lines $w+5000 tip" /tmp/tipw_{0..7}.jsonl
done
rm -f /tmp/tipw_*.jsonl
