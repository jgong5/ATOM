#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Per-task test gate: the 130 of 189 test files that run without a GPU driver,
# ~30 s. Green is the bar: this subset really is green.
#
# Excluded, and why neither is a judgement call:
#   tests/plugin/          30 files, needs sglang + vllm -- in neither image
#   cpu_gate_exclude.txt   29 files that reach the driver: 28 at *collection*
#                          time via rocminfo, plus 1 that collects and then
#                          fails on a driver call. That file names which is
#                          which, and only the first group is generated.
#
# What this tier cannot see is not left to memory. gpu_gate_triggers.txt lists
# the source paths whose only coverage is in the excluded set; a diff touching
# one of them fails this gate with an instruction to run gate_gpu.sh. Before
# that, the rule was a sentence in a design document and nothing read it.
#
# Run in the CPU container. Regenerate the exclusion list with
# regen_cpu_gate_exclude.sh; never hand-edit the GENERATED section.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

# GATE_CPU_RC is printed exactly once, on every path out of this script. It is
# defined here, above the first exit, and not further down beside the pytest
# result, because three drafts got the arithmetic of "exactly once" wrong in
# three different ways:
#   - an early draft printed it after pytest and again after the blind-spot
#     check, so a run that failed the second emitted `GATE_CPU_RC=0` followed by
#     `GATE_CPU_RC=98`, and anything grepping for the first answer got the wrong
#     one;
#   - a second draft exited before pytest on an unanswerable GPU question, so
#     that run printed it zero times -- equally unusable to a caller;
#   - a third draft -- this one, until the review that produced this comment --
#     defined finish() *below* seven FATAL exits. A tree that could not be
#     resolved, an unreadable exclusion list, or a stale --ignore= line all left
#     the script with no verdict line at all, which reads to a caller exactly
#     like a run that has not finished yet. A gate that cannot say it failed has
#     not failed safely.
# The rule this encodes: the verdict is the script's only output contract, so no
# path may skip it, including the ones that give up before measuring anything.
finish() {
    printf 'GATE_CPU_RC=%s\n' "$1"
    exit "$1"
}

# pytest's -r is store-last-wins, so a caller's -rE silently replaces the -rf
# below. This gate's own verdict is pytest's exit code and survives that, but
# its sibling gate_gpu.sh builds its verdict out of those report lines and does
# not -- `gate_gpu.sh -rE` on d78f3bbd3 reported all five baseline failures "no
# longer failing" and exited 0 while they had failed. The two gates are read as
# one contract, so they refuse the same flag rather than differing on which
# arguments are safe; and the -rf list is the only place this script names which
# tests failed, which is the whole of its value to the person reading it.
for arg in "$@"; do
    case "$arg" in
    -r | -r*)
        printf 'REFUSED: %s sets pytest -r, which this gate owns. It selects the\n' "$arg" >&2
        printf '  report lines that name the failing tests, and -r is\n' >&2
        printf '  store-last-wins, so your flag would replace them silently.\n' >&2
        finish 95
        ;;
    esac
done

INTEGRATION=${COMPASS_INTEGRATION_REF:-feature/atomcompass_new}

ROOT=$(compass_tree_root) || finish $?
cd "$ROOT" || finish 90
compass_env "$ROOT"
compass_require_tree "$ROOT" || finish $?
compass_describe "$ROOT"

EXCLUDE=$ROOT/scripts/compass/cpu_gate_exclude.txt
TRIGGERS=$ROOT/scripts/compass/gpu_gate_triggers.txt
[ -r "$EXCLUDE" ] || { printf 'FATAL: cannot read %s\n' "$EXCLUDE" >&2; finish 94; }
[ -r "$TRIGGERS" ] || { printf 'FATAL: cannot read %s\n' "$TRIGGERS" >&2; finish 94; }

# Parse defensively: strip a trailing CR so a list that has been through a
# Windows editor does not turn into --ignore=tests/foo.py^M, which pytest
# accepts and silently matches nothing; drop inline comments; and check the
# file exists, because a renamed test would otherwise be excluded forever by a
# stale line nobody can see failing.
NEX=0
IGN=(--ignore=tests/plugin)
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    [ -n "$line" ] || continue
    [ -e "$ROOT/$line" ] || {
        printf 'FATAL: %s lists %s, which does not exist.\n' "$EXCLUDE" "$line" >&2
        printf '       Re-run regen_cpu_gate_exclude.sh.\n' >&2
        finish 94
    }
    IGN+=(--ignore="$line")
    NEX=$((NEX + 1))
done <"$EXCLUDE"
printf 'gate:   %s files excluded + tests/plugin\n' "$NEX"

# --- does this change need the GPU tier? -------------------------------------
# Answered here, enforced after pytest, so a run yields both answers rather than
# trading one for the other. That includes the case where the question cannot be
# answered at all: it is recorded, not exited on, so the suite still reports.
GPU_NEEDED=
CHANGED=
SRC=
GPU_SRC_UNKNOWN=
if [ -n "${COMPASS_CHANGED_FILES:-}" ]; then
    [ -r "$COMPASS_CHANGED_FILES" ] ||
        { printf 'FATAL: COMPASS_CHANGED_FILES=%s unreadable\n' "$COMPASS_CHANGED_FILES" >&2; finish 94; }
    CHANGED=$(cat "$COMPASS_CHANGED_FILES")
    SRC="COMPASS_CHANGED_FILES"
elif git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 &&
     BASE=$(git -C "$ROOT" merge-base HEAD "$INTEGRATION" 2>/dev/null); then
    CHANGED=$(git -C "$ROOT" diff --name-only "$BASE")
    SRC="diff vs $INTEGRATION (${BASE:0:9})"
elif [ -r "$ROOT/.compass-changed" ]; then
    # A snapshot built by snapshot.sh, which is how this gate normally runs.
    # The stamp was written from the same rev-parse that selected the archived
    # tree, so it describes this tree and not whichever one the archiver
    # happened to be sitting in.
    CHANGED=$(cat "$ROOT/.compass-changed")
    SRC=".compass-changed stamp"
else
    # No git, no supplied list, no stamp: the question cannot be answered. It is
    # recorded as unanswered, and the run below exits 98 rather than passing --
    # reporting "GPU not required" here would be a guess that reads as a clear.
    GPU_SRC_UNKNOWN=1
fi

if [ -z "$GPU_SRC_UNKNOWN" ]; then
    while IFS= read -r pat; do
        pat=${pat%$'\r'}
        case "$pat" in ''|'#'*) continue ;; esac
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            case "$pat" in
            */) case "$f" in "$pat"*) GPU_NEEDED="$GPU_NEEDED$f -> $pat"$'\n' ;; esac ;;
            *)  [ "$f" = "$pat" ] && GPU_NEEDED="$GPU_NEEDED$f -> $pat"$'\n' ;;
            esac
        done <<<"$CHANGED"
    done <"$TRIGGERS"
fi

if [ -n "$GPU_SRC_UNKNOWN" ]; then
    printf 'gpu:    UNKNOWN -- no git, no COMPASS_CHANGED_FILES, no .compass-changed\n'
elif [ -n "$GPU_NEEDED" ]; then
    printf 'gpu:    REQUIRED (%s)\n' "$SRC"
    printf '%s' "$GPU_NEEDED" | sed 's/^/          /'
else
    printf 'gpu:    not required (%s)\n' "$SRC"
fi
printf '\n'

# Identify the tree before running, not after: an attestation that this diff
# already passed the GPU tier is per-tree, so it cannot be honoured on a tree we
# cannot name.
HEAD_SHA=$(compass_head_sha "$ROOT") || HEAD_SHA=UNKNOWN

# pytest's own status, captured directly. Piping it into `tail` discards the
# exit code and a failing suite reports success.
python -m pytest tests/ "${IGN[@]}" -q --no-header -p no:cacheprovider "$@" -rf
RC=$?
printf '\npytest: rc=%s\n' "$RC"

if [ "$RC" -ne 0 ]; then
    printf 'CPU tier of the test gate FAILED. Baseline is 4030 passed, 0 failed at 29\n' >&2
    printf 'exclusions (3956 ATOM + 74 tests/compass) -- see scripts/compass/README.md\n' >&2
    printf 'for how that number moves.\n' >&2
    finish "$RC"
fi

if [ -n "$GPU_SRC_UNKNOWN" ]; then
    printf 'The CPU tier is green, but the test gate is not passed: this run\n' >&2
    printf 'cannot tell whether the diff needs the GPU tier, and an unanswered\n' >&2
    printf 'question is not a "no".\n' >&2
    printf '  Fix by any one of: run in a tree with %s reachable;\n' "$INTEGRATION" >&2
    printf '  set COMPASS_CHANGED_FILES=<file listing the diff>; or rebuild the\n' >&2
    printf '  snapshot with scripts/compass/snapshot.sh, which stamps .compass-changed.\n' >&2
    finish 98
fi

# A green CPU tier is not a passed gate when the diff lands in the blind spot:
# the GPU tier has to have run on this same tree, and to have said so.
if [ -n "$GPU_NEEDED" ]; then
    DONE=${COMPASS_GPU_GATE_DONE:-}
    if [ -z "$DONE" ]; then
        printf 'this diff touches the CPU tier'"'"'s blind spot; the GPU tier has not run.\n' >&2
        printf '  Run gate_gpu.sh, then re-run with COMPASS_GPU_GATE_DONE=<that tree'"'"'s HEAD>\n' >&2
        finish 98
    fi
    # An attestation names a tree. Without a commit we cannot check it names
    # THIS tree, and accepting it unchecked would make the whole rule
    # bypassable by running in a snapshot -- which is where the gate normally
    # runs. Refuse instead: stamp the snapshot with snapshot.sh.
    if [ "$HEAD_SHA" = UNKNOWN ]; then
        printf 'COMPASS_GPU_GATE_DONE=%s cannot be checked: this tree has no commit.\n' "$DONE" >&2
        printf '  No .git and no .compass-commit stamp, so the attestation could name\n' >&2
        printf '  any tree. Rebuild the snapshot with scripts/compass/snapshot.sh.\n' >&2
        finish 98
    fi
    if [ "$DONE" != "$HEAD_SHA" ]; then
        printf 'COMPASS_GPU_GATE_DONE=%s but this tree is %s.\n' "$DONE" "$HEAD_SHA" >&2
        printf '  The GPU tier passed on a different tree. Re-run it here.\n' >&2
        finish 98
    fi
    printf 'gpu:    satisfied by COMPASS_GPU_GATE_DONE=%s\n' "${DONE:0:9}"
fi
finish 0
