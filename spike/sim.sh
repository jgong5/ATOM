#!/bin/bash
# usage: sim.sh OUT C DURATION -- inside a node's xingjche_2410 as root (no card used), alone on its host: one simulated
# run of #975's DPA-TP8 cell under the op-level backend at T5 (3911f83ba: b26c26217 with #3507's caches hand-ported,
# #1498's port) on #1498's store (its memory readings, graphs and price list faecca3f3a7b), as #1498's sim.sh ran it at
# b26c26217: deploy.sh run --op-graph, T5's crs005 host laws and machine spec, both sides on this node; then client.sh at
# concurrency C on traces 0-255 for DURATION s, --stop. Once the client's profiling phase starts, sampler.sh py-spies
# every Python process of the run. A run file that names another price list than #1498's c=16 run's stops the run.
# Writes OUT.deploy.pid (the chain stops the run by it past its bound).
. /shared_nfs/xingjche/agent_scratch/3636/run/args.sh
OUT=$1 C=$2 DUR=$3
T=/shared_nfs/xingjche/agent_scratch/1498/gt/tree5/ATOM
STORE=/shared_nfs/xingjche/agent_scratch/1498/gt/store
REF=/shared_nfs/xingjche/agent_scratch/1498/gt/cell/c16/op_graph/run_file.json
MACHINE=$T/scripts/compass/machines/mi355x-crs005.json
LAW_P=$T/scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005-prefill.json
LAW_D=$T/scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005-decode.json
HVENV=/shared_nfs/xingjche/agent_scratch/1498/gt/hvenv/bin/python3
ulimit -n "$(ulimit -Hn)"
mkdir -p $HOME; git config --global --add safe.directory '*'
cd /
mkdir -p "$(dirname "$OUT")"
side_args prefill
graphs() { ls $STORE/op_graph | wc -l; }
echo "start $(date -u +%FT%TZ) $(date +%s.%N) tree $(cat $T/.compass-commit) c $C duration $DUR load $(cut -d' ' -f1-3 /proc/loadavg) graphs $(graphs)"
ATOMESH=$S/mesh/atomesh WAIT_S=4500 PREFIX_S=300 $T/scripts/compass/deploy.sh "$OUT" run \
  --op-graph --host-laws $LAW_P --host-laws $LAW_D --machine $MACHINE --store $STORE \
  --prefill -- "${PREFILL_ARGS[@]}" \
  --decode --on "env -u GPU_MAX_HW_QUEUES -u ATOM_NUMA_BIND" -- "${DECODE_ARGS[@]}" \
  --router -- "${ROUTER_SIM[@]}" &
dep=$!
echo $dep > $OUT.deploy.pid
until [[ -s $OUT/server.json ]]; do kill -0 $dep 2>/dev/null || break; sleep 5; done
echo "ready $(date -u +%FT%TZ) $(date +%s.%N)"
if [[ -s $OUT/server.json ]]; then
  same=$(python3 -c 'import json, sys; a, b = (json.load(open(f))["price_list"] for f in sys.argv[1:]); print(a == b)' $OUT/run_file.json $REF 2>&1 | tail -1)
  if [[ $same != True ]]; then
    echo "the run file's price list is not #1498's c=16 run's ($same): stopped"; kill -TERM $dep
  else
    echo "price list: #1498's c=16 run's"
    setsid bash $R/sampler.sh $OUT 50 > $OUT.sampler.log 2>&1 &
    MODEL=$MODEL TRACES=$TRACES HARNESS_PYTHON=$HVENV $T/scripts/compass/client.sh "$OUT" --traces 0-255 --concurrency $C --duration $DUR --stop
    echo CLIENT_RC=$?
  fi
fi
wait $dep; echo DEPLOY_RC=$?
echo "end $(date -u +%FT%TZ) $(date +%s.%N) graphs $(graphs)"
touch $OUT.prof/stop 2>/dev/null; sleep 20
chown -R 50109788:1000 "$(dirname "$OUT")"; echo SIM_DONE
