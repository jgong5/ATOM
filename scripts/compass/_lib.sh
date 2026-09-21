# SPDX-License-Identifier: MIT
# Shared helpers for the Compass gate and pre-flight scripts. Source, don't run.

# The checkout that contains $PWD, or nothing. A checkout is recognised by the
# file this comment is in, so the probe cannot disagree with the thing it is
# probing for.
compass_enclosing_tree() {
    local d=$PWD
    while [ -n "$d" ]; do
        [ -f "$d/scripts/compass/_lib.sh" ] && { printf '%s' "$d"; return 0; }
        d=${d%/*}
    done
    return 1
}

# The tree this invocation exercises: the repo root that owns *this copy* of the
# script. Derived from $BASH_SOURCE, so the copy in a worktree points at that
# worktree and cannot be made to point anywhere else.
#
# That settles which tree the script acts on. It leaves open the question it
# looks like it answers -- whether that is the tree the caller meant -- and the
# answer is no whenever the script is invoked by path from inside a different
# checkout.
#
# EXECUTED 2026-09-20: from /workspace/llm_infer_deploy_study/perf_modeling/ATOM
# (HEAD 83daf636d, feature/atomcompass_new), running the p0 worktree's copy
# `bash .../compass-worktrees/p0/scripts/compass/snapshot.sh <outdir>` wrote
# compass-d78f3bbd3.tar stamped `commit: d78f3bbd3...`, exit 0, no warning. The
# stamp was truthful about the tree it archived and silent about the tree the
# caller was standing in -- a wrong answer carrying the full confidence of a
# right one. All five callers of this function have the same exposure; for the
# gates the consequence is worse than for snapshot.sh, since the result of a
# pytest run would be attributed to a commit it did not come from.
#
# The fix is NOT to resolve from $PWD instead: that lets a script act on a tree
# that is not its own, which is the failure this project has already paid for
# twice. It is to refuse when the two disagree -- a result attributed to the
# wrong tree is worse than no result. Being run by path from outside any
# checkout is still allowed -- there the caller has named the tree and nothing
# contradicts it, and that is exactly how snapshot.sh's own extract
# instructions say to run a gate.
compass_tree_root() {
    local here src cwd
    here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 90
    src=$(cd -- "$here/../.." && pwd) || return 90
    cwd=$(compass_enclosing_tree) || { printf '%s' "$src"; return 0; }
    if [ "$(cd -- "$cwd" && pwd -P)" != "$(cd -- "$src" && pwd -P)" ]; then
        printf 'FATAL: this script belongs to %s\n' "$src" >&2
        printf '       but you are standing in    %s\n' "$cwd" >&2
        printf '       Two different checkouts. This script acts on its own tree and\n' >&2
        printf '       not on yours, so it would measure, archive or rewrite a tree\n' >&2
        printf '       you are not looking at and name it only in a path you did\n' >&2
        printf '       not read.\n' >&2
        printf '       Run %s/scripts/compass/%s instead.\n' "$cwd" "${0##*/}" >&2
        return 99
    fi
    printf '%s' "$src"
}

# PYTHONPATH is set to the tree and nothing else. Inherited entries are not
# merged: an inherited entry ahead of ours is exactly how a take2 script silently
# ran another branch's engine, and an entry behind ours can still satisfy an
# import our tree does not have.
compass_env() {
    export PYTHONPATH=$1
    export AITER_LOG_LEVEL=${AITER_LOG_LEVEL:-WARNING}
}

# Assert that `atom` resolves under the tree being exercised, before trusting
# any result -- never after. The failure this catches has cost days: a prior run
# resolved `atom` to a non-git snapshot of a different branch, 72 files
# divergent, and failed silently wherever both trees defined the symbol.
#
# The probe runs from / on purpose. Every caller cd's to the tree first, and
# Python puts the script's directory -- here, the cwd -- at the head of
# sys.path, so a probe run in place resolves `atom` via cwd and passes whatever
# PYTHONPATH says. That is the one thing this function exists to check.
compass_require_tree() {
    local root=$1 resolved
    resolved=$(cd / && python -c 'import atom; print(atom.__file__)' 2>&1) || {
        printf 'FATAL: import atom failed:\n%s\n' "$resolved" >&2
        return 91
    }
    case "$resolved" in
    "$root"/*)
        printf 'tree:   %s\natom:   %s\n' "$root" "$resolved"
        ;;
    *)
        printf 'FATAL: atom resolved to %s\n       expected under %s\n       PYTHONPATH=%s\n' \
            "$resolved" "$root" "${PYTHONPATH:-<unset>}" >&2
        return 92
        ;;
    esac
}

# The commit this tree is, or UNKNOWN. Tried in order: git, then the
# .compass-commit stamp that snapshot.sh writes.
#
# A `git archive` snapshot has no .git, which is the point of using one -- but
# it also meant every gate run in a container reported `commit: UNKNOWN`, so
# results could not be attributed to a tree. The stamp is written by the same
# command that builds the archive, from the same rev-parse, so it cannot name a
# different commit than the one archived.
compass_head_sha() {
    local root=$1 sha
    sha=$(git -C "$root" rev-parse HEAD 2>/dev/null) && { printf '%s' "$sha"; return 0; }
    if [ -r "$root/.compass-commit" ]; then
        read -r sha <"$root/.compass-commit"
        case "$sha" in [0-9a-f]*) printf '%s' "$sha"; return 0 ;; esac
    fi
    printf 'UNKNOWN'
    return 1
}

# Name the commit in every artifact, so a result can never be attributed to a
# tree it did not come from. A dirty tree is reported, not refused -- gates run
# on uncommitted work by definition.
compass_describe() {
    local root=$1 sha src=stamp dirty
    git -C "$root" rev-parse HEAD >/dev/null 2>&1 && src=git
    if ! sha=$(compass_head_sha "$root"); then
        printf 'commit: UNKNOWN -- no .git and no .compass-commit stamp\n'
        return 0
    fi
    dirty=$(git -C "$root" status --porcelain 2>/dev/null | wc -l)
    printf 'commit: %s (%s)%s\n' "${sha:0:9}" "$src" \
        "$([ "$dirty" -gt 0 ] && echo " +${dirty} uncommitted")"
}
