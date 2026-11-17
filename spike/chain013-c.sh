#!/bin/bash
# usage (from the login node or 005's host, detached on 013's host):
#   spur exec 290572 bash -c 'setsid nohup bash /shared_nfs/xingjche/agent_scratch/3636/run/chain013.sh > /dev/null 2>&1 < /dev/null &'
# #3636's profiled c=16 run on 013, no card used. Waits, holding no lock, for the orchestrator's marker GO
# (runpass2/go/3636-013); then takes gpu.lock.013 (mkdir; none taken after START_BY). With the lock and the cards idle:
# in xingjche_2410 the import check at T5, py-spy on a test process, the router's commit; then sim.sh at c=16 for DUR s
# of client, its deploy.sh stopped by PID at STOP_AT (TERM, then KILL 5 min later). The lock goes once the cards read
# 0% VRAM, if its owner line is still ours, by DEADLINE. Writes 3636/logs/chain013.log, ending CHAIN_RC.
. /shared_nfs/xingjche/agent_scratch/3636/run/args.sh
GO=${GO:-/shared_nfs/xingjche/agent_scratch/runpass2/go/3636-013}
LOCK=/shared_nfs/xingjche/agent_scratch/gpu.lock.013
DEADLINE=$(date -ud "${DEADLINE:-2026-11-17T05:00:00Z}" +%s)   # the lock is back by then
STOP_AT=$((DEADLINE - 1200))                                   # stop the run 20 min before the deadline
START_BY=$(date -ud "${START_BY:-@$((DEADLINE - 4200))}" +%s)  # start none past it (default 70 min before the deadline)
C=${C:-16} DUR=${DUR:-1800} OUT=$S/sim/c${C:-16}
OWNER="DEVELOPER #3636 (task-3636): one profiled simulated run of #975's DPA cell at c=$C (T5 3911f83ba on #1498's store, $DUR s of client), in xingjche_2410, no card used, py-spy on every Python process (agent_scratch/3636/run/chain013-c.sh, log 3636/logs/chain013-c$C.log); stopped by $(date -ud @$STOP_AT +%FT%TZ), the lock back by $(date -ud @$DEADLINE +%FT%TZ); from"
L=$S/logs/chain013-c$C.log; exec >> $L 2>&1
dx() { docker exec xingjche_2410 "$@"; }
vram() { rocm-smi --showmemuse | grep -c 'VRAM%): [1-9]'; }
ours() { head -1 $LOCK/owner 2>/dev/null | grep -q '^DEVELOPER #3636 '; }
release() { local i; for i in $(seq 20); do (($(vram) == 0)) && break; sleep 30; done
  if (($(vram) == 0)) && ours; then rm -r $LOCK && say "released $LOCK"
  elif ours; then echo "chain013-c.sh ended $(date -u +%FT%TZ) with cards in use; lock kept" >> $LOCK/owner; say "cards in use; lock kept"
  else say "$LOCK names another owner: not removed"; fi; }
finish() { (($2)) && release; say "CHAIN_RC=$1"; exit $1; }
say "chain pid $$ on $(hostname): c=$C DUR=$DUR OUT=$OUT, stop at $(date -ud @$STOP_AT +%FT%TZ), deadline $(date -ud @$DEADLINE +%FT%TZ); waiting, holding no lock, for $GO"
until [[ -e $GO ]]; do (($(date +%s) > START_BY)) && { say "no $GO by $(date -ud @$START_BY +%FT%TZ)"; finish 8 0; }; sleep 20; done
say "$GO present; waiting for $LOCK"
until mkdir $LOCK 2>/dev/null; do (($(date +%s) > START_BY)) && { say "no lock by $(date -ud @$START_BY +%FT%TZ)"; finish 8 0; }; sleep 20; done
echo "$OWNER $(date -u +%FT%TZ)" > $LOCK/owner
say "took $LOCK; cards in use $(vram); load $(cut -d' ' -f1-3 /proc/loadavg)"
(($(vram) == 0)) || { say "cards in use at the start"; finish 1 1; }
dx bash -c "
  set -e; mkdir -p \$HOME; git config --global --add safe.directory '*'; cd /
  T=/shared_nfs/xingjche/agent_scratch/1498/gt/tree5/ATOM
  [[ \$(cat \$T/.compass-commit) == 3911f83bab97ec0a6951cfd5202de3011a84da1e ]]
  imp=\$(PYTHONPATH=\$T python3 -c 'import atom; print(atom.__file__)' 2>&1 | tail -1); echo \"import \$imp\"; [[ \$imp == \$T/atom/__init__.py ]]
  $S/mesh/atomesh --version-verbose | grep -E 'Git (Commit|Status)'
  $S/mesh/atomesh --version-verbose | grep -q 'Git Commit: 3911f83ba'
  /shared_nfs/xingjche/agent_scratch/1498/gt/hvenv/bin/python3 -c 'import aiperf, compass_harness; print(\"client venv ok\", aiperf.__version__)'
  python3 -c 'import time; time.sleep(4)' & p=\$!; sleep 1
  $S/bin/py-spy dump --pid \$p --nonblocking > /tmp/3636spy.txt 2>&1; wait \$p
  grep -q sleep /tmp/3636spy.txt && echo 'py-spy ok'
  [[ ! -e $OUT ]] || { echo '$OUT exists'; exit 1; }" || { say "checks failed"; finish 1 1; }
began=$(date +%s)
dx bash $R/sim.sh $OUT $C $DUR > $S/logs/sim-c$C.log 2>&1 &
exe=$! stopped=
say "c=$C started (docker exec pid $exe)"
while kill -0 $exe 2>/dev/null; do
  if (($(date +%s) > STOP_AT)) && [[ -z $stopped ]] && [[ -s $OUT.deploy.pid ]]; then
    say "past the bound: TERM to deploy.sh $(cat $OUT.deploy.pid)"; dx kill -TERM "$(cat $OUT.deploy.pid)"; stopped=$(date +%s)
  fi
  if [[ $stopped =~ ^[0-9]+$ ]] && (($(date +%s) > stopped + 300)); then
    g=$(dx ps -o pgid= -p "$(cat $OUT.deploy.pid)" | tr -d ' ')
    say "still running 5 min after TERM: KILL to the run's process group $g"; [[ -n $g ]] && dx kill -KILL -- -$g; stopped=killed
  fi
  sleep 15
done
say "c=$C: $(( $(date +%s) - began )) s; $(grep -E 'ready|price list|CLIENT_RC|DEPLOY_RC|^end|SIM_DONE' $S/logs/sim-c$C.log | tr '\n' ' ')"
grep -q CLIENT_RC=0 $S/logs/sim-c$C.log && grep -q SIM_DONE $S/logs/sim-c$C.log && [[ -z $stopped ]] && finish 0 1
finish 1 1
