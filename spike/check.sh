#!/bin/bash
# usage: check.sh TREE STORE TAG -- as root in a gate container: `compass run --op-graph --check` of each side of
# #975's DPA cell at TREE on STORE, with TREE's crs005 DPA8 host laws and the platform flags, from the gate container's
# model copy. Prints CHECK_RC per side; output in out/check/TAG-<side>.txt.
. /shared_nfs/xingjche/agent_scratch/3636/run/args.sh
T=$1 STORE=$2 TAG=$3
ulimit -n "$(ulimit -Hn)"
git config --global --add safe.directory '*'
cd /
export PYTHONPATH=$T PYTHONDONTWRITEBYTECODE=1
echo "atom $(python3 -c 'import atom; print(atom.__file__)' 2>&1 | tail -1) digest $(python3 -c 'from atom.compass.run import atom_digest; print(atom_digest())' 2>&1 | tail -1)"
LAW_P=$T/scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005-prefill.json
LAW_D=$T/scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005-decode.json
mkdir -p $S/out/check
for side in prefill decode; do
  side_args $side
  if [[ $side == prefill ]]; then A=("${PREFILL_ARGS[@]}"); else A=("${DECODE_ARGS[@]}"); fi
  A=("${A[@]/#$MODEL/$MCOPY}")
  out=$(mktemp -d)
  timeout 900 $T/scripts/compass/compass run --op-graph --host-laws $LAW_P --host-laws $LAW_D "${PLATFORM[@]}" \
    --store $STORE --check $side --out $out -- "${A[@]}" > $S/out/check/$TAG-$side.txt 2>&1
  echo "CHECK_RC=$? $TAG $side"; tail -30 $S/out/check/$TAG-$side.txt | grep -v '^\[aiter\]'
  rm -rf $out
done
chown -R 50109788:1000 $S/out/check
