#!/bin/bash
# The 8P16D C1228 recipe sim on #4029's run 2 tree, with the Rust router LP's clock frames stamped (#4117).
#   host:          run.sh prep DIR [BASE]               DIR/ATOM: the tree measured; DIR/mesh: its router's checkout
#   CPU container: run.sh build DIR                     DIR/mesh/atom/mesh/target/release/atomesh
#   CPU container: run.sh sim DIR OUT [DURATION] [WARMUP_PER_LANE]
# BASE is #4029's run 2 tree, made by spike/sim_speed_8p16d/run.sh prep at 4c5e9a65b (PR #4118): a5691dd27 +
# #3996 + #3997 + #4004 + frames.patch. prep copies it to DIR/ATOM, applies router_frames.patch there, and commits the
# patch on a5691dd27 in DIR/mesh, with a fixed author and date, so the commit is the same wherever it is made. The
# router LP refuses a binary whose commit is not its tree's, so DIR/ATOM's .compass-commit names that commit;
# nothing else in a simulated run reads it, and `atom_digest`, which the c307 laws check, reads no Rust. build
# builds the router there, on CPUs 0-63. sim is #4029's sim with that router: every Python LP's clock frame and
# the router's, into OUT.frames/<pid>.trace; the router also stamps each released message it looks for (W), finds
# at its hold point (H) and has handled (D). rank.py OUT [SLICE_S] ranks the run, the router's share split.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BASE_COMMIT=a5691dd273f17e96a6d8f05843683478262d8b9f
case ${1:-} in
prep)
  DIR=${2:?DIR}; BASE=${3:-/var/tmp/xingjche_simspeed/tree/m4029b/ATOM}
  [ -e "$DIR" ] && { echo "refused: $DIR exists"; exit 1; }
  grep -q '+ frames.patch$' "$BASE/MEASURED" || { echo "refused: $BASE is not #4029's prep"; exit 1; }
  mkdir -p "$DIR/ATOM" && tar -C "$BASE" -cf - --exclude=./atom/mesh/target --exclude=__pycache__ . | tar -C "$DIR/ATOM" -xf - || exit 1
  git init -q "$DIR/mesh" && git -C "$DIR/mesh" fetch -q --depth 1 "$HERE/../.." $BASE_COMMIT &&
    git -C "$DIR/mesh" checkout -q FETCH_HEAD && git -C "$DIR/mesh" apply "$HERE/router_frames.patch" || exit 1
  GIT_AUTHOR_DATE=2026-11-25T00:00:00Z GIT_COMMITTER_DATE=2026-11-25T00:00:00Z git -C "$DIR/mesh" \
    -c user.name=spike-4117 -c user.email=spike-4117@invalid commit -q -am "router_frames.patch (#4117)" || exit 1
  cp "$BASE/atom/mesh/Cargo.lock" "$DIR/mesh/atom/mesh/" || exit 1  # the crate versions c307_canon's router ran
  patch -s -d "$DIR/ATOM" -p1 < "$HERE/router_frames.patch" || exit 1
  git -C "$DIR/mesh" rev-parse HEAD > "$DIR/ATOM/.compass-commit"
  sed -i "s/\$/ + router_frames.patch (router commit $(git -C "$DIR/mesh" rev-parse --short HEAD))/" "$DIR/ATOM/MEASURED"
  cat "$DIR/ATOM/MEASURED"
  ;;
build)
  DIR=${2:?DIR}
  cd "$DIR/mesh/atom/mesh" && GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*' \
    taskset -c 0-63 cargo build --release --locked -j 32 || exit 1
  target/release/atomesh --version-verbose | grep -E 'Git (Commit|Status)'
  ;;
sim)
  DIR=${2:?DIR}; DEP=${3:?OUT}; DUR=${4:-900}; WPL=${5:-10}
  ROUTER=$DIR/mesh/atom/mesh/target/release/atomesh
  [ -x "$ROUTER" ] || { echo "refused: no $ROUTER; run.sh build $DIR"; exit 1; }
  . /var/tmp/yajizhan_cellb/g27/m8/m8.conf
  export CELLB_TREE=$DIR/ATOM
  source /var/tmp/yajizhan_cellb/g27/env.sh
  export ATOMESH=$ROUTER
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
