#!/bin/bash
# usage: rep.sh TREE CELL SIDE RANK TAG [replay.py args...] -- as root in a gate container: one replay of #1498's
# recorded steps (gt/cell/CELL/op_graph/SIDE, rank RANK) through OpGraphBackend at TREE (tip = ATOM-worktrees/3636,
# t4 = b26c26217, t5 = 3911f83ba, or a path), the side's performance variables set. Log into out/rep/TAG.log.
TREE=$1 CELL=$2 SIDE=$3 RANK=$4 TAG=$5; shift 5
case $TREE in
  tip) R=/shared_nfs/xingjche/ATOM-worktrees/3636 ;;
  t4) R=/shared_nfs/xingjche/agent_scratch/1498/gt/tree4/ATOM ;;
  t5) R=/shared_nfs/xingjche/agent_scratch/1498/gt/tree5/ATOM ;;
  *) R=$TREE ;;
esac
cd /
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 HOME=${HOME:-/tmp}
C=/shared_nfs/xingjche/agent_scratch/1498/gt/cell/$CELL/op_graph
M=/shared_nfs/xingjche/agent_scratch/1498/model/DeepSeek-V4-Pro
O=/shared_nfs/xingjche/agent_scratch/3636/out/rep; mkdir -p $O $(dirname $O/$TAG)
E=AITER_BF16_FP8_MOE_BOUND=0,ATOM_MOE_GU_ITLV=1,AITER_LOG_LEVEL=WARNING
[[ $SIDE == prefill ]] && E=$E,GPU_MAX_HW_QUEUES=5,ATOM_NUMA_BIND=1
began=$(date +%s)
python3 /shared_nfs/xingjche/agent_scratch/3636/run/replay.py $C/$SIDE $RANK --model $M --env $E "$@" 2>&1 \
  | grep -v "^\[aiter\]\|Engine kwargs\|transformers\]\|dcp_config" > $O/$TAG.log
rc=${PIPESTATUS[0]}
echo "$TAG rc=$rc $(( $(date +%s) - began )) s load $(cut -d' ' -f1-3 /proc/loadavg) $(grep '^steps' $O/$TAG.log)" | tee -a $O/$TAG.log
