#!/bin/bash
# The 8P16D C1228 recipe sim as m8/sims/c307_canon ran it, with #3996 and #3997, its clock frames stamped (#4029).
#   host:          run.sh prep DIR                         DIR/ATOM: the tree measured
#   CPU container: run.sh sim DIR/ATOM OUT [DURATION] [WARMUP_PER_LANE]
# prep copies c307_canon's tree (m8.conf's a5691dd27: the c307 laws hold only under its `atom_digest`), applies
# #3996 and #3997 as landed (their atom/ and compass_harness/ diffs, which `atom_digest` leaves out) and
# frames.patch. sim is yajizhan's m8/sim_8p16d.sh on pit2-p03-g27 (m8.conf, env.sh, laws_c307.env, the recipe's
# client) with that tree and OUT as the run, plus every frame into OUT.frames/<pid>.trace (COMPASS_TRACE_DIR) and
# the Clock Authority's wall profile into OUT.caprof.jsonl (COMPASS_CA_PROFILE). DURATION 900 s and
# WARMUP_PER_LANE 10 are the recipe's. Writes OUT.log, OUT.deploy.log and OUT.client.log beside OUT.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
case ${1:-} in
prep)
  DIR=${2:?DIR}; BASE=/var/tmp/yajizhan_cellb/tree/a5691dd27/ATOM; REPO=$HERE/../..
  [ -e "$DIR" ] && { echo "refused: $DIR exists"; exit 1; }
  mkdir -p "$DIR/ATOM" && tar -C "$BASE" -cf - --exclude=./atom/mesh/target --exclude=__pycache__ . | tar -C "$DIR/ATOM" -xf - || exit 1
  for c in c15b9a02b 3f41a31fe; do  # #3996, #3997
    git -C "$REPO" diff "$c^1" "$c" -- atom compass_harness | patch -s -d "$DIR/ATOM" -p1 || exit 1
  done
  patch -s -d "$DIR/ATOM" -p1 < "$HERE/frames.patch" || exit 1
  echo "a5691dd27 + #3996 c15b9a02b + #3997 3f41a31fe + frames.patch" > "$DIR/ATOM/MEASURED"
  ;;
sim)
  TREE=${2:?TREE}; DEP=${3:?OUT}; DUR=${4:-900}; WPL=${5:-10}
  . /var/tmp/yajizhan_cellb/g27/m8/m8.conf
  export CELLB_TREE=$TREE
  source /var/tmp/yajizhan_cellb/g27/env.sh
  unset P_EXTRA_ENV COMPASS_HARNESS_SEND_ORDER_SEED
  source /var/tmp/yajizhan_cellb/g27/m8/laws_c307.env
  [ -e "$DEP" ] && { echo "refused: $DEP exists"; exit 1; }
  mkdir -p "$DEP.frames"
  export COMPASS_TRACE_DIR=$DEP.frames COMPASS_CA_PROFILE=$DEP.caprof.jsonl
  export COMPASS_IDLE_DP_GROUP_S=$IDLE WAIT_S=3600
  ulimit -n 1048576 2>/dev/null || ulimit -n 65536
  PN=(rdma0,rdma1 rdma2,rdma3 rdma4,rdma5 rdma6,rdma7)
  log() { echo "$(date -u +%FT%TZ) $DEP: $*" | tee -a "$DEP.log"; }
  log "tree $T ($(cat "$T/MEASURED" 2>/dev/null)), atomesh $ATOMESH, laws P $STEP_P/$HOST_P D $STEP_D/$HOST_D" \
    "in $LAWS, duration $DUR, warmup/lane $WPL, CPUs $(taskset -cp $$ | sed 's/.*: //')"
  prefills=(); for i in 0 1 2 3; do set -f; prefills+=(--prefill $(p_env "${PN[$i]}") -- $(args P)); set +f; done
  decodes=(); for i in 0 1 2 3; do set -f; decodes+=(--decode -- $(args D)); set +f; done
  set -f
  "$T/scripts/compass/deploy.sh" "$DEP" run --kv-over-network --network "$LINK" \
    --step-linear $STEP_P --step-linear $STEP_D --host-laws $HOST_P --host-laws $HOST_D \
    $P --store "$LAWS" --mooncake-store "$ENTRY" \
    "${prefills[@]}" "${decodes[@]}" --router -- $ROUTER_SIM > "$DEP.deploy.log" 2>&1 &
  set +f
  DPID=$!
  until [ -s "$DEP/server.json" ]; do
    kill -0 $DPID 2>/dev/null || { log "EXIT deploy left: $(tail -3 "$DEP.deploy.log" | tr '\n' ' ' | cut -c1-600)"; exit 1; }
    sleep 5
  done
  log "deployment up"
  AIPERF_EXTRA_ARGS="$AIPERF_RECIPE_ARGS" OMP_NUM_THREADS=4 HARNESS_PYTHON="$SIM_PY" \
    "$T/scripts/compass/client.sh" "$DEP" --agentx --concurrency 1228 --duration "$DUR" --warmup-per-lane "$WPL" \
    --stop > "$DEP.client.log" 2>&1
  rc=$?
  log "client rc=$rc: $(grep -h 'client_run: wrote' "$DEP.client.log" | tail -1)"
  wait $DPID; drc=$?
  log "EXIT client=$rc deploy=$drc"
  [ $rc = 0 ] && [ $drc = 0 ]
  ;;
*) sed -n '2,4p' "$0"; exit 2 ;;
esac
