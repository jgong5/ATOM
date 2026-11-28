#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# spike/4161_gap/run.sh: #4267's rounds of eight paired short cells, on pit2-p03-g53's lanes C and D.
#   run.sh stage     on g27: stages under g53's W the r8 laws and inputs, b5/bin's runner with W's paths,
#                    b5/fe-engine 98aed973e exported twice (W/tree/ATOM; W/tree_des/ATOM with #4262's
#                    as_designed.patch and toggles.patch) and an atomesh built at 98aed973e
#   run.sh round K   on g53: round K under both lane locks: eight cells at once in xingjche_simspeed_g53, one
#                    CCD (8 CPUs) each, lane C's four and lane D's four, rotated by 3 CCDs a round: as measured,
#                    as designed twice, as designed with one of #4262's changes reverted (relay, step, idle, timer,
#                    hop); ccdload.py logs outside load per CCD; then ab.py, err.py and walls.py, whose verdict
#                    says whether the round counts, against the two designed cells' mean
#   run.sh clean     on g53: removes W, root-owned files included
# A cell is b5/bin/sim_b5.sh's short cell: C1228, 120 simulated seconds of profiling, warmup 1 a lane, no railwarm.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=/share_nfs/xingjche/x8p16d L=/var/tmp/xingjche_simspeed W=/var/tmp/xingjche_simspeed/s4267
X=$W/x8 R=$W/x8/b5/runs C=xingjche_simspeed_g53 SHA=98aed973e7d8e22ccece7cfa461aa7d09ec67dd8
# Off: #4149's exact options (b5/bin/gcpair_full.sh's EXACT, with W's token ids and atomesh) and the GC tuning.
OFF="COMPASS_WIDE_GRANTS=reactions ATOM_COMPASS_TOKEN_IDS=$W/tok_warm ATOMESH_BIN=$W/atomesh
 COMPASS_B5_STREAM_BYPASS=1 ATOM_COMPASS_STREAM_BYPASS=1 COMPASS_B5_IDS_BY_REF=1 COMPASS_B5_CA_SELECT=1
 COMPASS_B5_MERGE_STEPS=1 COMPASS_B5_P_READ=1 COMPASS_B5_GC=1"
LEAD="COMPASS_B5_FE_ENGINE=1 COMPASS_B5_BUSY_REACTIONS=1 COMPASS_B5_PROMISES=1 COMPASS_B5_DELAY_P_S=0.028"
ERR="BackdatedEvent|Straggler|ClockAbort|UnansweredRequests|malformed compass"
case ${1:-} in
stage)
  G53="ssh pit2-p03-g53" r8=runs/x8_c1228_r8
  $G53 "set -e; mkdir -p $X/fits/x8_c1228_r8 $X/inputs/cc $X/models $X/$r8/railwarm/client $X/$r8/logs $X/b5/bin $R $W/ids
    cp -a $D/venv-sim $X/; cp -a $D/models/dsv41 $X/models/; cp $D/inputs/cc/traces.jsonl $X/inputs/cc/
    cp $D/fits/x8_c1228_r8/{link.json,mooncake_store_entry.json} $X/fits/x8_c1228_r8/
    cp $D/$r8/railwarm/client/aiperf.log $X/$r8/railwarm/client/; cp $D/$r8/logs/driver.log $X/$r8/logs/
    cp $D/b5/bin/{ab.py,err.py,phases.py} $HERE/walls.py $X/b5/bin/
    for f in x8.env fits/x8_c1228_r8/laws.env b5/bin/sim_b5.sh; do sed 's#$D#$X#g' $D/\$f > $X/\$f; done" || exit 1
  # The r8 law directories are root's alone: read them, and the warm token ids, through the b5 container.
  docker exec xingjche_b5_sim tar -C $D -cf - laws_r8 | $G53 "tar -C $X -xf -" || exit 1
  docker exec xingjche_b5_sim tar -C /tmp -cf - tok_warm | $G53 "tar -C $W -xf -" || exit 1
  $G53 "set -e; git init -q $W/src; git -C $W/src fetch -q --depth 1 file://$D/build/ATOM-git $SHA
    git -C $W/src checkout -q FETCH_HEAD
    for t in tree tree_des; do mkdir -p $W/\$t/ATOM; git -C $W/src archive HEAD | tar -C $W/\$t/ATOM -xf -
      echo $SHA > $W/\$t/ATOM/.compass-commit; done
    patch -s -d $W/tree_des/ATOM -p1 < $HERE/as_designed.patch; patch -s -d $W/tree_des/ATOM -p1 < $HERE/toggles.patch
    echo '98aed973e + spike/4161_gap/as_designed.patch + toggles.patch' > $W/tree_des/ATOM/MEASURED
    exec 8>$L/laneC.lock 9>$L/laneD.lock; flock 8; flock 9
    docker exec -w $W/src/atom/mesh -e CARGO_TARGET_DIR=$W/target $C bash -lc 'cp /app/ATOM/atom/mesh/Cargo.lock . &&
      nice -n 10 taskset -c 64-127 cargo build -q --release --bin atomesh' > $W/build.log 2>&1
    cp $W/target/release/atomesh $W/atomesh
    docker exec $C bash -c '$W/atomesh --version-verbose | grep -E \"Git (Commit|Status)\"; source $X/x8.env
      source $X/fits/x8_c1228_r8/laws.env; ls -d \$LAWS/*/*{\$STEP_P,\$HOST_P,\$STEP_D,\$HOST_D} \$ENTRY \$LINK \
        \$MODEL/config.json $X/inputs/cc/traces.jsonl $X/venv-sim/bin/python $W/tok_warm'"
  ;;
round)
  K=${2:?round 1, 2 or 3} P=g$2
  exec 8>$L/laneC.lock 9>$L/laneD.lock; flock 8; flock 9
  echo "$(date -u +%FT%TZ) s4267 round $K: lanes C and D taken" >> $L/out/pairs.log
  cid=$(docker inspect $C --format '{{.Id}}')
  nice -n 19 taskset -c 64-127 python3 -I $HERE/ccdload.py $R/${P}_load.txt ${cid:0:12} & LOAD=$!
  cell() {  # TAG TREE CPUS EXTRA_ENV
    docker exec $C bash -c "export $(echo $OFF) COMPASS_B5_IDS_DIR=$W/ids/$1 $4
      nice -n 10 taskset -c $3 bash $X/b5/bin/sim_b5.sh $1 $2 120 1 0 > $R/$1.out 2>&1" &
  }
  DES="$LEAD COMPASS_B5_DELAY_FD_S=0" c=0
  for v in "meas $W/tree/ATOM $LEAD COMPASS_B5_DELAY_FD_S=0.0018" "des $W/tree_des/ATOM $DES" \
    "relay $W/tree_des/ATOM $LEAD COMPASS_B5_DELAY_FD_S=0.0018" "step $W/tree_des/ATOM $DES COMPASS_4267_REVERT_STEP=1" \
    "idle $W/tree_des/ATOM $DES COMPASS_4267_REVERT_IDLE=1" "timer $W/tree_des/ATOM $DES COMPASS_4267_REVERT_TIMER=1" \
    "hop $W/tree_des/ATOM $DES COMPASS_4267_REVERT_HOP=1" "des2 $W/tree_des/ATOM $DES"; do
    read -r tag tree env <<<"$v"; s=$(((c + 3 * (K - 1)) % 8)) c=$((c + 1))
    cell ${P}_$tag $tree $((64 + 8 * s))-$((71 + 8 * s)) "$env"
  done
  wait $(jobs -p | grep -vx $LOAD); kill $LOAD
  echo "$(date -u +%FT%TZ) s4267 round $K: lanes C and D released" >> $L/out/pairs.log
  T=$(for t in des meas relay step idle timer hop des2; do printf ' %s' ${P}_$t; done)
  E=$(for t in des meas relay step idle timer hop des2; do  # the cells that completed
    [ -s $R/${P}_$t/artifacts/profile_export_aiperf.json ] && printf ' %s' ${P}_$t; done)
  docker exec -w $R $C bash -c "grep -h 'CPUs' ${P}_*.logs/sim.log | sed 's/.*b5 \([^ :]*\):.*CPUs/\1 CPUs/'
    grep -hE 'EXIT|PHASES' ${P}_*.out; python3 $X/b5/bin/ab.py$T
    python3 $X/b5/bin/walls.py ${P}_des,${P}_des2 ${P}_meas ${P}_relay ${P}_step ${P}_idle ${P}_timer ${P}_hop
    python3 $X/b5/bin/err.py$E | grep -E '^==|matched'
    grep -lE '$ERR' ${P}_*/*.log ${P}_*.logs/*.log || echo 'no clock error in the logs'"
  python3 -I $HERE/ccdload.py --mean $R/${P}_load.txt
  ;;
clean) docker exec $C rm -rf $W ;;
*) sed -n '5,14p' "$0"; exit 2 ;;
esac
