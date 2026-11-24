#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The CPU dry run of a v1.1 calibration design (#739 section 2) on an existing
# law: generate it, plan it and rehearse it, then judge it against the
# config's frozen acceptance criteria.
#
# Usage: dry_v1.sh CONFIG LAW_DIR OUT [run COMPASS_RUN_ARGS... -- SERVE_ARGS...]
#   CONFIG    a design config (scripts/compass/calib/*.json)
#   LAW_DIR   a fitted law's directory (law_eval.sh's laws/NAME): its step
#             linear law, law.json or steps/step_linear_law.json, and its host
#             laws, host/host_laws.json when there
#   OUT       an absent directory: OUT/design (gen_v1.py, plan_v1.py: the
#             traces, design.json, plan.json, plan.md)
#   RECORD    a record whose deployment the design takes in place of the
#             config's own fields (gen_v1.py --record); unset, the config's
#   TARGET    records of target workloads, space-separated, whose GPU-bound
#             steps the report gives each vector's error at (plan_v1.py --target)
#
# A rung the law holds no vector for (a deployment that captures more) is
# rehearsed on its nearest rung's prices (atom/compass/calib/engine.py proxied),
# and plan.md names it.
#
# The rehearsal (atom/compass/calib/engine.py) forms the steps a DP engine
# would from the design, prices them on the law, keeps the GPU-bound ones,
# and reports per vector full rank, VIF, the worst prediction error G over the
# reachable range and the levels each column takes, with the predicted GPU
# minutes: seconds of CPU, no server. Exits 0 when every vector passes, 1 when
# one fails, 2 on bad arguments.
#
# With `run ...`, it then also replays the design on a simulated server
# (cal_v1.sh OUT/design OUT/sim run ...; MODEL and HARNESS_PYTHON as client.sh
# takes them) and checks that record against the plan (check_v1.py). That runs
# ATOM's own scheduler, every DP rank a process: many CPU minutes a design.
set -euo pipefail
. "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/_lib.sh"
ROOT=$(compass_tree_root) || exit $?
USAGE="usage: dry_v1.sh CONFIG LAW_DIR OUT [run COMPASS_RUN_ARGS... -- SERVE_ARGS...]"
(($# >= 3)) || { echo "$USAGE" >&2; exit 2; }
CONFIG=$1 LAW_DIR=$2 OUT=$3
shift 3
[[ -s $CONFIG ]] || { echo "dry_v1: no config $CONFIG" >&2; exit 2; }
[[ ! -e $OUT ]] || { echo "dry_v1: $OUT exists" >&2; exit 2; }
LAW=$LAW_DIR/law.json
[[ -s $LAW ]] || LAW=$LAW_DIR/steps/step_linear_law.json
[[ -s $LAW ]] || { echo "dry_v1: no law.json or steps/step_linear_law.json in $LAW_DIR" >&2; exit 2; }
HOST=()
[[ -s $LAW_DIR/host/host_laws.json ]] && HOST=(--host "$LAW_DIR/host/host_laws.json")
export PYTHONPATH=$ROOT
FROM=()
[[ -z ${RECORD-} ]] || FROM=(--record "$RECORD")
TARGETS=()
for t in ${TARGET-}; do TARGETS+=(--target "$t"); done
python3 -P "$ROOT/scripts/compass/gen_v1.py" "$CONFIG" "$OUT/design" "${FROM[@]}"
python3 -P "$ROOT/scripts/compass/plan_v1.py" "$OUT/design" --law "$LAW" "${HOST[@]}" "${TARGETS[@]}" \
    > "$OUT/plan.log"
cat "$OUT/design/plan.md"
rc=0
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["rehearsal"]["pass"] else 1)' \
    "$OUT/design/plan.json" || rc=1
if (($#)); then
    [[ $1 == run ]] || { echo "$USAGE" >&2; exit 2; }
    "$ROOT/scripts/compass/cal_v1.sh" "$OUT/design" "$OUT/sim" "$@"
    python3 -P "$ROOT/scripts/compass/check_v1.py" "$OUT/design" "$OUT/sim" --law "$LAW" "${HOST[@]}" || rc=1
fi
exit $rc
