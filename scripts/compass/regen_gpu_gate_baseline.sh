#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Re-measure the GPU gate's baseline and rewrite gpu_gate_baseline.env and
# gpu_gate_known_failures.txt beside this script. Run it in the GPU container
# whenever gate_gpu.sh warns that the toolchain differs from the baseline's: a
# new image, a torch, HIP or aiter bump, or another GPU.
#
#   git archive --prefix=ATOM/ -o BASE.tar <commit>     # where the repository is
#   HIP_VISIBLE_DEVICES=<gpu> scripts/compass/regen_gpu_gate_baseline.sh BASE.tar
#
# The commit comes as a git archive tarball because git records the commit in
# it, so the baseline names the commit that ran, and its tree has no .git, as
# the snapshot.sh trees the gate measures have none. Each of the two runs is the
# gate's own suite over a fresh extraction: tests/ outside tests/plugin and
# tests/compass, on the first card HIP_VISIBLE_DEVICES names. tests/compass is
# the gate's separate run, added to its expectation per tree, so the baseline
# leaves it out.
#
# Nothing is written unless both runs complete with no error, pass the same
# count and fail the same node-ids. A baseline its next run would not reproduce
# fails every gate after it by chance.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

refuse() {
    printf 'REFUSED: %s; nothing was written.\n' "$2" >&2
    exit "$1"
}

HERE=$(compass_tree_root) || exit $?
DIR=$HERE/scripts/compass
TAR=$(readlink -f -- "${1:-}")
cd "$HERE" || exit 90
compass_env "$HERE"

SHA=$(python -c 'import sys, tarfile; print(tarfile.open(sys.argv[1]).pax_headers.get("comment", ""))' \
    "$TAR" 2>/dev/null)
[ "${#SHA}" -eq 40 ] ||
    refuse 94 "'${1:-}' is not a git archive tarball naming its commit (git archive --prefix=ATOM/ -o BASE.tar <commit>)"

# The stack, read as gate_gpu.sh reads it, through this tree's atom.
TORCH=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo UNKNOWN)
ROCM=$(python -c 'import torch; print(torch.version.hip)' 2>/dev/null || echo UNKNOWN)
AITER_DIR=$(python -c 'from atom.compass.artifacts.provenance import module_root; print(module_root("aiter"))' 2>/dev/null)
if [ -n "$AITER_DIR" ]; then
    AITER=$(git -C "$AITER_DIR" describe --tags --always --dirty 2>/dev/null || echo UNKNOWN)
else
    AITER=UNKNOWN
fi
printf 'commit: %s\ntorch:  %s\nhip:    %s\naiter:  %s\n' "$SHA" "$TORCH" "$ROCM" "$AITER"
case "$TORCH $ROCM $AITER" in
*UNKNOWN*) refuse 96 "part of the stack reads UNKNOWN, and the gate would match it against any box whose read fails" ;;
esac

"$DIR/preflight.sh" || refuse 91 "pre-flight failed before the runs"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
CARD=${HIP_VISIBLE_DEVICES:-0}
CARD=${CARD%%,*}
P=()
for k in 1 2; do
    ROOT=$WORK/$k/ATOM
    mkdir -p "$WORK/$k" && tar -x -C "$WORK/$k" -f "$TAR" && [ -d "$ROOT" ] ||
        refuse 94 "'$TAR' does not extract to an ATOM/ directory"
    compass_env "$ROOT"
    compass_require_tree "$ROOT" || exit $?
    cd "$ROOT" || exit 90
    printf '===== run %s of 2 at %s on HIP_VISIBLE_DEVICES=%s =====\n' "$k" "${SHA:0:9}" "$CARD"
    HIP_VISIBLE_DEVICES=$CARD python -m pytest tests/ --ignore=tests/plugin --ignore=tests/compass \
        -q --no-header -p no:cacheprovider -rfE 2>&1 | tee "$WORK/$k.log"
    RC=${PIPESTATUS[0]}
    compass_pytest_counts "$WORK/$k.log"
    case "$RC" in 0 | 1) ;; *) refuse 1 "run $k: pytest rc=$RC, so the run did not complete" ;; esac
    [ -n "$SUMMARY" ] || refuse 1 "run $k printed no pytest summary line"
    [ "$ERRORS" -eq 0 ] || refuse 1 "run $k: $ERRORS error(s), and the gate holds its baseline to none"
    # In the order the run failed them, which a re-run keeps; compared sorted.
    grep '^FAILED ' "$WORK/$k.log" | sed -e 's/^FAILED //' -e 's/ - .*$//' | awk '!seen[$0]++' >"$WORK/$k.failed"
    LC_ALL=C sort "$WORK/$k.failed" >"$WORK/$k.sorted"
    NAMED=$(grep -c . "$WORK/$k.failed")
    [ "$NAMED" -eq "$FAILED" ] || refuse 1 "run $k: the summary says $FAILED failed but $NAMED FAILED line(s) name them"
    P[$k]=$PASSED
done

if [ "${P[1]}" -ne "${P[2]}" ] || ! cmp -s "$WORK/1.sorted" "$WORK/2.sorted"; then
    printf 'the two runs disagree: %s passed, then %s\n' "${P[1]}" "${P[2]}" >&2
    LC_ALL=C comm -23 "$WORK/1.sorted" "$WORK/2.sorted" | sed 's/^/  failed in run 1 only: /' >&2
    LC_ALL=C comm -13 "$WORK/1.sorted" "$WORK/2.sorted" | sed 's/^/  failed in run 2 only: /' >&2
    refuse 97 "the two runs disagree, so neither is a baseline"
fi

"$DIR/preflight.sh" || refuse 91 "pre-flight failed after the runs, so the node changed under them"

# Built beside the targets and renamed, so a failure above leaves both intact.
cat >"$DIR/gpu_gate_baseline.env.tmp" <<EOF
# SPDX-License-Identifier: MIT
# The GPU gate's baseline, read by gate_gpu.sh, which says what each value is.
# Written whole by regen_gpu_gate_baseline.sh, with gpu_gate_known_failures.txt.
BASE_COMMIT=${SHA:0:9}
BASE_PASSED=${P[1]}
BASE_FAILED=$FAILED
BASE_COMPASS_TESTS=0
BASE_TORCH=$TORCH
BASE_ROCM=$ROCM
BASE_AITER=$AITER
EOF
{
    cat <<EOF
# SPDX-License-Identifier: MIT
# The failing node-ids at gpu_gate_baseline.env's BASE_COMMIT, verbatim and in
# run order, that both of regen_gpu_gate_baseline.sh's runs failed; it writes
# this file whole. gate_gpu.sh judges failures by NAME and not only by count: a
# same-sized set of *different* failures is a regression the counts cannot see.
# Blank lines and # comments are ignored. The count here and BASE_FAILED are two
# statements of one fact; gate_gpu.sh refuses to run if they disagree.
EOF
    cat "$WORK/1.failed"
} >"$DIR/gpu_gate_known_failures.txt.tmp"
mv -f "$DIR/gpu_gate_baseline.env.tmp" "$DIR/gpu_gate_baseline.env"
mv -f "$DIR/gpu_gate_known_failures.txt.tmp" "$DIR/gpu_gate_known_failures.txt"
printf 'wrote %s and %s: %s passed, %s failed at %s\n' "$DIR/gpu_gate_baseline.env" \
    "$DIR/gpu_gate_known_failures.txt" "${P[1]}" "$FAILED" "${SHA:0:9}"
