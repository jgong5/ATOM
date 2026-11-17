#!/bin/bash
# usage: sampler.sh OUT RATE -- inside the run's container, detached by sim.sh: once OUT's aiperf enters its profiling
# phase, py-spy (nonblocking, idle samples kept, per thread, raw, full file names) every Python process in the
# container at RATE Hz, each until its process exits or OUT.prof/stop appears (then SIGINT, which writes its file).
# OUT.prof/procs.txt names each process (pid, ppid, comm, command line).
OUT=$1 RATE=${2:-50} P=$1.prof PYSPY=/shared_nfs/xingjche/agent_scratch/3636/bin/py-spy
mkdir -p $P/pyspy
until grep -qs "Phase profiling (profiling) started" "$OUT/client/aiperf.log"; do [[ -e $P/stop ]] && exit 0; sleep 2; done
echo "profiling phase seen $(date -u +%FT%TZ) $(date +%s.%N)"
pids=()
for d in /proc/[0-9]*; do
  x=${d#/proc/}
  [[ $(readlink $d/exe 2>/dev/null) == *python* ]] || continue
  echo "$x $(ps -o ppid= -p $x | tr -d ' ') $(cat $d/comm 2>/dev/null) $(tr '\0' ' ' < $d/cmdline 2>/dev/null | cut -c1-400)" >> $P/procs.txt
  $PYSPY record --pid $x --rate $RATE --nonblocking --idle --threads --full-filenames --format raw \
    --output $P/pyspy/$x.txt > $P/pyspy/$x.log 2>&1 &
  pids+=($!)
done
echo "sampling ${#pids[@]} processes from $(date -u +%FT%TZ) $(date +%s.%N)"
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive + 1)); done
  ((alive == 0)) && break
  [[ -e $P/stop ]] && { for p in "${pids[@]}"; do kill -INT $p 2>/dev/null; done; sleep 30; break; }
  sleep 5
done
echo "sampler done $(date -u +%FT%TZ) $(date +%s.%N); files $(ls $P/pyspy/*.txt 2>/dev/null | wc -l)"
