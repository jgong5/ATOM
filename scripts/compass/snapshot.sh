#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Build the tarball a gate runs against, with `git archive` and never rsync --
# rsync into a shared container path displaces whoever else is working
# there, and it copies the working tree rather than a named commit, so the
# result cannot say what it is.
#
# The archive carries two stamps the tree cannot otherwise provide once .git is
# gone:
#   .compass-commit   the sha, so gate output says which tree it measured
#                     instead of `commit: UNKNOWN`, and so a
#                     COMPASS_GPU_GATE_DONE attestation can be checked
#   .compass-changed  the diff against the integration ref, so gate_cpu.sh can
#                     answer the blind-spot question without .git
#
# Both are written from the same rev-parse that selects the archived tree, so
# neither can name a different commit than the one inside. The third,
# .compass-gpu-carry, lets gate_cpu.sh carry the GPU gate of COMPASS_GPU_GATE_DONE=G
# here: G, HEAD and the diff from G, or empty, so it replaces a stale extraction's.
# The fourth, .compass-select, lets gate_cpu.sh select with the map
# COMPASS_GATE_SELECT names: the map's commit, HEAD, whether the first is an
# ancestor of the second, and the diff between them; or empty.
#
# Usage: snapshot.sh [<output-dir>]   (default: the current directory)
#
# Run YOUR tree's copy. This archives the tree the script lives in, not the tree
# you are standing in. Until 2026-09-20 it did that silently: run from a
# checkout at 83daf636d, the p0 worktree's copy wrote a tarball stamped
# d78f3bbd3 and exited 0. compass_tree_root now refuses when the two disagree.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

INTEGRATION=${COMPASS_INTEGRATION_REF:-feature/atomcompass_new}
OUTDIR=${1:-$PWD}

ROOT=$(compass_tree_root) || exit $?
git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 || {
    printf 'REFUSED: %s is not a git tree; there is nothing to name.\n' "$ROOT" >&2
    exit 90
}

SHA=$(git -C "$ROOT" rev-parse HEAD)
SHORT=${SHA:0:9}
DIRTY=$(git -C "$ROOT" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || {
    printf 'REFUSED: %s uncommitted change(s). git archive would silently ship\n' "$DIRTY" >&2
    printf '  HEAD, not what you are looking at, and the stamp would name a tree\n' >&2
    printf '  that is not the one you tested. Commit first.\n' >&2
    exit 91
}

# Two different failures used to share one message. `no merge-base with <ref>`
# reads as a claim about the history, and was printed just as often when the ref
# had not resolved at all and nothing had been compared -- sending the reader to
# inspect a tree that was fine. Each step now names itself.
# On success REF is the ref that resolved; on failure it carries git's own
# message, which is what the refusal quotes.
REF=$(compass_resolve_ref "$ROOT" "$INTEGRATION") || {
    printf 'REFUSED: ref resolution failed -- %s names nothing here, with or without\n' "$INTEGRATION" >&2
    printf '  a remote prefix, so no history has been compared. git said:\n    %s\n' "$REF" >&2
    printf '  Set COMPASS_INTEGRATION_REF to a ref this tree resolves.\n' >&2
    exit 92
}
[ "$REF" = "$INTEGRATION" ] ||
    printf 'ref:    %s is not a ref here; resolved it as %s\n' "$INTEGRATION" "$REF"
# A resolved local branch left behind the remote of the same name is named on
# the same line, in the same place. It stays the base; it stops being silent.
compass_ref_drift "$ROOT" "$REF"

BASE=$(git -C "$ROOT" merge-base HEAD "$REF") || {
    printf 'REFUSED: merge-base failed -- %s resolved, but shares no commit with HEAD.\n' "$REF" >&2
    printf '  The ref is fine; these two histories are unrelated.\n' >&2
    exit 92
}

TAR=$OUTDIR/compass-$SHORT.tar
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/ATOM"
printf '%s\n' "$SHA" >"$STAGE/ATOM/.compass-commit"
# Both sides of a rename, unquoted, in both stamps as in gate_cpu.sh: else a
# renamed trigger file or a non-ASCII path under a trigger matches nothing.
git -C "$ROOT" -c core.quotePath=false diff --no-renames --name-only "$BASE" HEAD >"$STAGE/ATOM/.compass-changed"
: >"$STAGE/ATOM/.compass-gpu-carry"
if [ -n "${COMPASS_GPU_GATE_DONE:-}" ]; then
    G=$(git -C "$ROOT" rev-parse --verify --quiet "$COMPASS_GPU_GATE_DONE^{commit}") && [ "$G" = "$COMPASS_GPU_GATE_DONE" ] || {
        printf 'REFUSED: COMPASS_GPU_GATE_DONE=%s is not the full sha of a commit here.\n' "$COMPASS_GPU_GATE_DONE" >&2
        exit 93
    }
    [ "$G" = "$SHA" ] || { printf '%s\n%s\n' "$G" "$SHA"; git -C "$ROOT" -c core.quotePath=false diff --no-renames --name-only "$G" HEAD; } >"$STAGE/ATOM/.compass-gpu-carry"
fi
: >"$STAGE/ATOM/.compass-select"
if [ -n "${COMPASS_GATE_SELECT:-}" ]; then
    python3 "$ROOT/scripts/compass/gate_select.py" stamp "$COMPASS_GATE_SELECT" "$ROOT" >"$STAGE/ATOM/.compass-select" || {
        printf 'REFUSED: COMPASS_GATE_SELECT=%s is not a map gate_map.sh wrote.\n' "$COMPASS_GATE_SELECT" >&2
        exit 93
    }
fi

git -C "$ROOT" archive --format=tar --prefix=ATOM/ HEAD >"$TAR"
tar -rf "$TAR" -C "$STAGE" ATOM/.compass-commit ATOM/.compass-changed
tar -rf "$TAR" -C "$STAGE" ATOM/.compass-gpu-carry
tar -rf "$TAR" -C "$STAGE" ATOM/.compass-select

printf 'wrote %s\n' "$TAR"
printf '  commit:  %s\n' "$SHA"
printf '  base:    %s (%s)\n' "${BASE:0:9}" "$REF"
printf '  changed: %s file(s)\n' "$(grep -c . "$STAGE/ATOM/.compass-changed")"
[ ! -s "$STAGE/ATOM/.compass-gpu-carry" ] || printf '  carry:   %s\n' "$(head -1 "$STAGE/ATOM/.compass-gpu-carry")"
printf '\nExtract with:\n'
printf '  mkdir -p <dir> && tar -x -C <dir> -f %s\n' "$(basename "$TAR")"
printf 'then run <dir>/ATOM/scripts/compass/gate_cpu.sh\n'
