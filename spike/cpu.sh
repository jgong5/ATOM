#!/bin/bash
# usage: cpu.sh OUT -- inside the run's container, detached: every 60 s, each process's pid, ppid, comm, utime and
# stime (clock ticks) and the time, into OUT.cpu/<epoch>.txt, until OUT.prof/stop or OUT/run.json appears.
OUT=$1 D=$1.cpu; mkdir -p $D
while [[ ! -e $OUT.prof/stop && ! -e $OUT/run.json ]]; do
  t=$(date +%s.%N)
  for s in /proc/[0-9]*/stat; do
    read -r line < $s 2>/dev/null || continue
    pid=${line%% *}; rest=${line##*) }; comm=${line#*(}; comm=${comm%)*}
    set -- $rest; echo "$pid $2 ${comm// /_} ${12} ${13}"
  done > $D/$t.txt
  sleep 60
done
echo "cpu.sh done $(date -u +%FT%TZ)" > $D/done
