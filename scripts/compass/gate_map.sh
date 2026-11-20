#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# gate_map.sh OUT [gate_cpu.sh arguments]: run this tree's CPU gate under
# coverage.py, every Python process it starts included, and write to OUT the
# map gate_cpu.sh's COMPASS_GATE_SELECT reads: per test file, the functions
# whose bodies it ran, keyed by this tree's commit. COMPASS_GATE_SHARDS and the
# arguments go to gate_cpu.sh; its verdict line is the run's, not the map's.
#
# COMPASS_COVERAGE_WHEEL names a coverage.py wheel. It installs offline into a
# directory beside OUT, as the gate installs compass_harness, and never into the
# image. A sitecustomize.py at the tree's root, there for this run only, starts
# coverage in every process with the tree on its path: scripts and tests that
# start Python set PYTHONPATH to the tree, so no other directory reaches them.
# Each process saves its data at exit, at SIGTERM and at os._exit; a process
# killed otherwise records nothing. Run it in a tree of its own, as a snapshot.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

OUT=${1:?usage: gate_map.sh OUT [gate_cpu.sh arguments]}
shift
ROOT=$(compass_tree_root) || exit $?
SHA=$(compass_head_sha "$ROOT") || { echo "gate_map: $ROOT names no commit; stage it with snapshot.sh" >&2; exit 94; }
[ -r "${COMPASS_COVERAGE_WHEEL:-}" ] || { echo "gate_map: COMPASS_COVERAGE_WHEEL names no coverage.py wheel" >&2; exit 96; }
SITE=$ROOT/sitecustomize.py
[ ! -e "$SITE" ] || { echo "gate_map: $SITE exists; this run writes its own there" >&2; exit 94; }
W=$(mktemp -d "$(dirname -- "$OUT")/gate_map.XXXXXX") || exit 96
trap 'rm -rf "$W" "$SITE"' EXIT
python -m pip install -q --no-deps --no-index --target "$W/lib" "$COMPASS_COVERAGE_WHEEL" >&2 || exit 96
cp "$ROOT/scripts/compass/gate_select.py" "$W/lib/" && mkdir "$W/cov" "$W/ctx" "$W/reads" || exit 96
printf '[run]\ncore = ctrace\nparallel = true\ndata_file = %s/cov/c\nsource = %s\n' "$W" "$ROOT" >"$W/rc"
printf 'context = ${COMPASS_MAP_CONTEXT}\npatch = _exit, fork\nsigterm = true\n' >>"$W/rc"
python -c 'import sys; w, root = sys.argv[1:]
print(f"# SPDX-License-Identifier: MIT\nimport sys\nsys.path.append({w + "/lib"!r})\nimport gate_select\ngate_select.startup({w!r}, {root!r})")' \
    "$W" "$ROOT" >"$SITE" || exit 96
env -u COMPASS_GATE_SELECT "$ROOT/scripts/compass/gate_cpu.sh" -p gate_select "$@" | tee "$W/gate.log"
rm -f "$SITE"
grep -q '^pytest: rc=' "$W/gate.log" || { echo "gate_map: pytest did not run; no map written" >&2; exit 1; }
python "$W/lib/gate_select.py" build "$W" "$ROOT" "$SHA" "$OUT"
