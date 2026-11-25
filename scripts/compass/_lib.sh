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
# right one. Every caller of this function has the same exposure; for the
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

# The integration ref, resolved against this tree: the bare name first, then the
# same name qualified by each configured remote. A linked worktree and a fresh
# clone both carry fork/feature/atomcompass_new and no local branch of that
# name, so the bare default resolves nowhere and the comparison the caller
# wanted never happens. Prints the ref that resolved -- the caller compares that
# against what it asked for to report the fallback. When nothing resolves it
# prints git's own message for the bare form instead, so a refusal can quote git
# rather than paraphrase it into a claim about the history. That message cannot
# travel in a global: the caller reads this through a command substitution,
# which is a subshell, and an assignment made in there never reaches the caller.
compass_resolve_ref() {
    local root=$1 ref=$2 remote err
    err=$(git -C "$root" rev-parse --verify "$ref^{commit}" 2>&1 >/dev/null) &&
        { printf '%s' "$ref"; return 0; }
    # `git remote` prints alphabetically, which is an order and not a
    # preference: with two remotes both carrying the ref at different commits
    # the first by name wins, and that picks a base and hence a diff, not just a
    # name. The chosen remote is printed on the ref: and base: lines, which is
    # what makes the choice checkable rather than silent.
    for remote in $(git -C "$root" remote); do
        git -C "$root" rev-parse --verify --quiet "$remote/$ref^{commit}" >/dev/null 2>&1 &&
            { printf '%s' "$remote/$ref"; return 0; }
    done
    printf '%s' "$err"
    return 1
}

# Whether the ref that resolved is a local branch that has drifted from the
# remote branch of the same name. This reports; it does not redirect. The
# resolution order above is untouched and the base stays the ref that resolved.
#
# Bare-name-first is right when the local branch is what the caller means, and
# nothing told that apart from a local branch that is simply old. MEASURED
# 2026-09-22, in a worktree of /workspace/ATOM checked out at 83ef2a094 --
# which *is* fork/feature/atomcompass_new: with nothing set, the local
# feature/atomcompass_new at 1b473e5af won and .compass-changed listed the 5
# files of the landing in between; with COMPASS_INTEGRATION_REF naming the
# remote, 0 files. Both runs exited 0 and printed no ref: line, so the output
# of the wrong one was indistinguishable from the output of the right one.
#
# The counterpart is found by scanning the remotes for the same name rather
# than by reading branch.<name>.merge: that branch has no upstream configured
# (measured, same tree, `rev-parse feature/atomcompass_new@{upstream}` exits
# 128), and the same-name scan is exactly what the resolver above would have
# chosen had the local branch been absent -- so the comparison is against the
# ref that was passed over, which is the one the caller might have wanted.
#
# What that rule costs. A local branch whose configured upstream is a
# *differently* named remote branch is not compared against what it tracks.
# The class is exactly: branches for which `git config --get
# branch.<name>.merge` names something other than refs/heads/<name>. To
# enumerate it, walk `git for-each-ref --format='%(refname:short)' refs/heads`
# and compare that config value against each branch's own name. A member with
# no refs/remotes/*/<name> gets no counterpart, so this function returns 0 and
# prints nothing; a member that has one gets a line about a ref it does not
# track. Both halves are deliberate: the counterpart has to be the ref
# compass_resolve_ref would have picked, so announcing against
# branch.<name>.merge would name a ref the resolver would never use. The
# silent half is the case a future reader most needs, because it is where the
# tool goes back to being quiet.
#
# How many such branches exist is deliberately not recorded here. Membership
# turns over by ordinary development in minutes: `git push -u` creates the
# same-name remote ref *and* rewrites branch.<name>.merge to it, so publishing
# a branch moves it out of the class and into the level-and-silent early return
# below, while a landing on the integration branch changes the arithmetic of
# the ones that stay. A count in a comment is read without the timestamp that
# would make it checkable, and a reading taken here was already wrong within
# the hour. A dated census belongs in a change record, where it is met as a
# record; run the enumeration above for the reading that is true now.
#
# Announced, not refused, and the choice is not close. The caller named a ref
# that exists here and got that ref; nothing was guessed and no fallback was
# taken, so there is no declined answer to give. Refusing would also change
# what exit 92 means -- today it says "this ref names nothing here", which
# other agents key on and #82 was careful not to move -- and it would refuse
# the legitimate case of a deliberately older base, with no way back except an
# environment variable the caller would have to already suspect.
#
# Behind is the only direction that can corrupt the base: merge-base ignores
# commits the local branch has and HEAD does not, so a local branch that is
# *ahead* leaves the base where it was. It is still named, because when HEAD
# descends from those unlanded commits the base moves forward with them and the
# changed set shrinks -- and a changed set that is too small is how the GPU
# blind-spot question gets answered "no" without being asked.
#
# "diverged from (N ahead, M behind)" is also what *unrelated* histories print:
# rev-list --left-right --count returns the whole of each side when there is no
# merge-base, so an orphan counterpart reads as diverged. The counts are true
# either way and this line only reports them; the refusal that has to tell the
# two apart is snapshot.sh's merge-base one, which says "shares no commit with
# HEAD" on its first line and "these two histories are unrelated" on its
# second, and never says "diverged". That vocabulary is the refusal's own text
# on stderr, not a gloss on it, and tests/compass/test_snapshot_ref.py pins the
# word. So the distinction already exists one file over, in the place that has
# to make it, under a test -- which is the reason not to state it a second time
# here. It is also not split here because the orphan fixture is the only test
# that reaches this branch of the ladder, so a separate wording would leave the
# genuine diverged case with no test at all.
compass_ref_drift() {
    local root=$1 ref=$2 remote up= counts ahead behind how lsha rsha
    git -C "$root" show-ref --verify --quiet "refs/heads/$ref" || return 0
    for remote in $(git -C "$root" remote); do
        git -C "$root" show-ref --verify --quiet "refs/remotes/$remote/$ref" &&
            { up=$remote/$ref; break; }
    done
    [ -n "$up" ] || return 0
    counts=$(git -C "$root" rev-list --left-right --count "$ref...$up") || return 0
    ahead=${counts%%[!0-9]*}
    behind=${counts##*[!0-9]}
    if [ "$ahead" -eq 0 ] && [ "$behind" -eq 0 ]; then return 0; fi
    if [ "$behind" -eq 0 ]; then
        how="$ahead commit(s) ahead of"
    elif [ "$ahead" -eq 0 ]; then
        how="$behind commit(s) behind"
    else
        how="diverged from ($ahead ahead, $behind behind)"
    fi
    lsha=$(git -C "$root" rev-parse "$ref")
    rsha=$(git -C "$root" rev-parse "$up")
    printf 'ref:    %s (%s) is %s %s (%s)\n' \
        "$ref" "${lsha:0:9}" "$how" "$up" "${rsha:0:9}"
    printf '        the base is the local branch; COMPASS_INTEGRATION_REF=%s uses the remote\n' "$up"
}

# compass_changed ROOT INTEGRATION: sets CHANGED to the paths this tree's change
# touches and CHANGED_SRC to where that list came from, from the first source
# that answers: the file COMPASS_CHANGED_FILES names, the diff since HEAD's
# merge-base with INTEGRATION, or snapshot.sh's .compass-changed stamp. When
# none answers, CHANGED_UNKNOWN says why instead. Returns 94 with WHY set when
# COMPASS_CHANGED_FILES names a file it cannot read.
compass_changed() {
    local root=$1 integration=$2 ref base
    CHANGED= CHANGED_SRC= CHANGED_UNKNOWN=
    if [ -n "${COMPASS_CHANGED_FILES:-}" ]; then
        [ -r "$COMPASS_CHANGED_FILES" ] || {
            printf 'FATAL: COMPASS_CHANGED_FILES=%s unreadable\n' "$COMPASS_CHANGED_FILES" >&2
            WHY="COMPASS_CHANGED_FILES unreadable"
            return 94
        }
        CHANGED=$(cat "$COMPASS_CHANGED_FILES")
        CHANGED_SRC="COMPASS_CHANGED_FILES"
    elif git -C "$root" rev-parse --git-dir >/dev/null 2>&1 &&
         ref=$(compass_resolve_ref "$root" "$integration") &&
         base=$(git -C "$root" merge-base HEAD "$ref" 2>/dev/null); then
        # Both sides of a rename, unquoted, here and in gate_cpu.sh's carry:
        # else a renamed trigger file or a non-ASCII path under a trigger
        # matches nothing.
        CHANGED=$(git -C "$root" -c core.quotePath=false diff --no-renames --name-only "$base")
        CHANGED_SRC="diff vs $ref (${base:0:9})"
        # Announce the fallback the way snapshot.sh does. Without this the
        # resolved name appears only inside the source string -- reported, not
        # announced, so a caller who set nothing sees a prefix it never asked for.
        [ "$ref" = "$integration" ] ||
            printf 'ref:    %s is not a ref here; resolved it as %s\n' "$integration" "$ref"
        compass_ref_drift "$root" "$ref"
    elif [ -r "$root/.compass-changed" ]; then
        # A snapshot built by snapshot.sh, which is how the gates normally run.
        # The stamp was written from the same rev-parse that selected the
        # archived tree, so it describes this tree and not whichever one the
        # archiver happened to be sitting in.
        #
        # The drift line above does not reach here, and this is the path the
        # gates normally run on. .compass-changed records the file list and
        # .compass-commit the sha; neither records the *base*, so a stamp built
        # on a stale local branch cannot be told from one built on the remote,
        # and the announcement lives only in the staging log while this is the
        # log that gets pasted. Closing it means recording the base in a stamp,
        # which changes the stamp format and is not #102.
        CHANGED=$(cat "$root/.compass-changed")
        CHANGED_SRC=".compass-changed stamp"
    elif ! git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
        # No supplied list, no usable diff, no stamp: the question cannot be
        # answered, and the reason names which of the three ways in failed.
        # Only a tree with no .git is a staging omission; a git checkout
        # reaches here because the integration ref does not resolve, or shares
        # no commit with HEAD.
        CHANGED_UNKNOWN="this tree was never stamped (no .git, no .compass-changed); a staging omission, see snapshot.sh"
    elif ref=$(compass_resolve_ref "$root" "$integration"); then
        CHANGED_UNKNOWN="a git checkout with no .compass-changed, and $ref shares no commit with HEAD"
    else
        CHANGED_UNKNOWN="a git checkout with no .compass-changed, and $integration resolves to no commit here"
    fi
}

# compass_cpu_ignores ROOT: sets IGN to the --ignore arguments for what the CPU
# gate leaves out, tests/plugin and each file cpu_gate_exclude.txt lists, and
# NEX to how many files the list names. Returns 94 with WHY set when the list
# is unreadable or names a missing file.
#
# Parsed defensively: a trailing CR is stripped so a list that has been through
# a Windows editor does not turn into --ignore=tests/foo.py^M, which pytest
# accepts and silently matches nothing; inline comments are dropped; and each
# file is checked to exist, because a renamed test would otherwise be excluded
# forever by a stale line nobody can see failing.
compass_cpu_ignores() {
    local exclude=$1/scripts/compass/cpu_gate_exclude.txt line
    NEX=0
    IGN=(--ignore=tests/plugin)
    [ -r "$exclude" ] || {
        printf 'FATAL: cannot read %s\n' "$exclude" >&2
        WHY="exclusion list unreadable"
        return 94
    }
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%$'\r'}
        line=${line%%#*}
        line=${line#"${line%%[![:space:]]*}"}
        line=${line%"${line##*[![:space:]]}"}
        [ -n "$line" ] || continue
        [ -e "$1/$line" ] || {
            printf 'FATAL: %s lists %s, which does not exist.\n' "$exclude" "$line" >&2
            printf '       Re-run regen_cpu_gate_exclude.sh.\n' >&2
            WHY="exclusion list names a missing file"
            return 94
        }
        IGN+=(--ignore="$line")
        NEX=$((NEX + 1))
    done <"$exclude"
}

# compass_harness ROOT: when ROOT has a compass_harness, installs it, with the
# aiperf wheel COMPASS_AGENTX_HARNESS names, into the run-local directory
# HARNESS, removed at exit, and appends that to PYTHONPATH. Returns 96 with WHY
# set when the install fails or the compass_harness tests would still skip.
#
# Those tests skip by name when compass-harness, or the aiperf it adapts, does
# not import, and the CPU container has neither. So the gates install the
# tree's own compass_harness every run, and refuse to run while the tests' own
# skip condition still holds.
compass_harness() {
    [ -r "$1/compass_harness/pyproject.toml" ] || return 0
    HARNESS=$(mktemp -d) || { WHY="mktemp failed"; return 96; }
    trap 'rm -rf "$HARNESS"' EXIT
    # A copy, because setuptools writes build/ and *.egg-info into the source.
    cp -r "$1/compass_harness" "$HARNESS/src" || { WHY="cannot copy compass_harness"; return 96; }
    python -m pip install -q --no-deps --no-build-isolation --no-index --target "$HARNESS/site" \
        "$HARNESS/src" ${COMPASS_AGENTX_HARNESS:+"$COMPASS_AGENTX_HARNESS"} >&2 || {
        WHY="pip could not install compass_harness${COMPASS_AGENTX_HARNESS:+ and $COMPASS_AGENTX_HARNESS}"
        return 96
    }
    export PYTHONPATH=$PYTHONPATH:$HARNESS/site
    WHY=$(cd / && python -c 'from importlib import metadata; metadata.version("compass-harness")
import compass_harness, aiperf; print(aiperf.__file__, compass_harness.__file__)' 2>&1) || {
        WHY="the compass_harness tests would skip (${WHY##*$'\n'}); set COMPASS_AGENTX_HARNESS to an agentx-harness 56a0cf70 wheel, see scripts/compass/README.md"
        return 96
    }
    printf 'harness: %s\n' "$WHY"
}

# A SKIPPED line naming one of gate_gpu.sh's driver-run inputs, or the router it
# puts on PATH: a driver-run test that did not run for want of it. The
# alternation is tests/compass/driver_run_skips.py's DRIVER_RUN.
COMPASS_DRIVER_RUN_SKIP='^SKIPPED .*(ATOM_COMPASS_(SLICE|IMAGE|MTP|DSPARK|CCTRACES)|COMPASS_(HARNESS|AIPERF)_PYTHON|ATOMESH|needs atomesh|atomesh on PATH)'

# A SKIPPED line from a test that cargo-builds atomesh, taken where `cargo
# metadata` cannot resolve atomesh's crates: no driver-run input would run it,
# so a gate that runs such a test refuses with 96, as gate_cpu.sh's probe does.
COMPASS_CARGO_SKIP="^SKIPPED .*cargo cannot resolve atomesh's dependencies"

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

# Sets SUMMARY, PASSED, FAILED and ERRORS from the pytest summary line ending a
# log. gate_gpu.sh reads its runs with it and regen_gpu_gate_baseline.sh the
# baseline's, so the two counts are one reading.
compass_pytest_counts() {
    SUMMARY=$(tail -20 "$1" | grep -E '[0-9]+ (passed|failed|error)' | tail -1)
    PASSED=$(grep -oE '[0-9]+ passed' <<<"$SUMMARY" | grep -oE '[0-9]+' || echo 0)
    FAILED=$(grep -oE '[0-9]+ failed' <<<"$SUMMARY" | grep -oE '[0-9]+' || echo 0)
    ERRORS=$(grep -oE '[0-9]+ errors?' <<<"$SUMMARY" | grep -oE '[0-9]+' || echo 0)
}

# compass_gpu_arch CARD: the GPU target of HIP ordinal CARD as torch reads it
# (gfx942, gfx950, gfx1250), or nothing. The GPU gate keeps one baseline per target.
compass_gpu_arch() {
    HIP_VISIBLE_DEVICES=$1 python -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])' 2>/dev/null
}

# This file's directory, resolved as it is sourced: a caller may cd elsewhere
# before a function reads a list beside it, as regen_gpu_gate_baseline.sh cd's
# into the base it measures, where a relative path names nothing.
COMPASS_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# compass_gpu_skip_lines TARGET: what gpu_gate_skip.txt names for GPU target
# TARGET, one per line: a test file it leaves out there, or `driver-run`.
compass_gpu_skip_lines() {
    grep -vE '^[[:space:]]*(#|$)' "$COMPASS_LIB_DIR/gpu_gate_skip.txt" |
        awk -v t="$1" '$1 == t { print $2 }'
}

# compass_gpu_skips TARGET: sets SKIP_ARGS to an --ignore for each file
# gpu_gate_skip.txt leaves out on GPU target TARGET, the first card's, and
# SKIP_N to the tests those files collect, run from the tree being measured. A
# file under tests/compass goes to COMPASS_LEFT instead, uncounted: the gate
# counts what tests/compass passes on the tree. The line `TARGET driver-run`
# sets NO_DRIVER_RUN. A file it cannot count fails, SKIP_WHY naming it.
compass_gpu_skips() {
    local file n
    SKIP_ARGS=() SKIP_N=0 COMPASS_LEFT=() NO_DRIVER_RUN=
    while read -r file; do
        if [ "$file" = driver-run ]; then
            printf 'driver-run inputs: none on %s\n' "$1"
            NO_DRIVER_RUN=1
            continue
        fi
        [ -e "$file" ] || continue
        if [[ $file == tests/compass/* ]]; then
            printf 'left out on %s: %s\n' "$1" "$file"
            COMPASS_LEFT+=("$file")
            continue
        fi
        n=$(python -m pytest "$file" --collect-only -q -p no:cacheprovider 2>&1 | sed -nE 's/^([0-9]+) tests? collected.*/\1/p')
        [ -n "$n" ] || { SKIP_WHY="cannot count the tests of $file, which gpu_gate_skip.txt leaves out on $1"; return 1; }
        printf 'left out on %s: %s (%s tests)\n' "$1" "$file" "$n"
        SKIP_ARGS+=("--ignore=$file") SKIP_N=$((SKIP_N + n))
    done < <(compass_gpu_skip_lines "$1")
}

# compass_cargo_lock MESH: copies into atomesh's directory MESH, unless it has
# one, the lockfile the image built atomesh with (COMPASS_CARGO_LOCK, default
# /app/ATOM/atom/mesh/Cargo.lock). The tree ignores Cargo.lock, and offline
# cargo resolves atomesh's crates only under the lockfile the container's crate
# cache was fetched for. Both gates call it before their cargo runs.
compass_cargo_lock() {
    local lock=${COMPASS_CARGO_LOCK:-/app/ATOM/atom/mesh/Cargo.lock}
    [ -e "$1/Cargo.lock" ] || [ ! -r "$lock" ] || cp "$lock" "$1/Cargo.lock"
}

# compass_held_port [N]: the first of N ports in a row (default 1), each one
# a free TCP port, held until this script exits, so no other process binds it
# before its server does. A block of more than one is drawn below the
# ephemeral range. Its holder binds it on every address, where another
# socket's plain bind fails, then sets SO_REUSEADDR and never listens: a server
# that sets it too binds the port beside the holder, as uvicorn's, libzmq's,
# socket.create_server's, torch's TCPStore's and the atomesh router's listeners
# do.
compass_held_port() {
    python3 -c 'import os, random, select, socket, sys
held, n = [], int(sys.argv[2])
low = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
while len(held) < n:
    first = random.randrange(low // 2, low - n) if n > 1 else 0
    port = held[0].getsockname()[1] + len(held) if held else first
    held.append(socket.socket())
    try:
        held[-1].bind(("", port))
    except (OSError, OverflowError):
        if not port:
            raise
        while held:  # another socket has one: draw the block again
            held.pop().close()
for s in held:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print(held[0].getsockname()[1], flush=True)
if not os.fork():
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    select.select([os.pidfd_open(int(sys.argv[1]))], [], [])' $$ "${1:-1}"
}

# compass_dynamo_header: X-Dynamo-Session-ID, which aiperf then sends, where
# AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID is true as aiperf reads it.
compass_dynamo_header() {
    local v=${AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID-}
    [[ ${v,,} =~ ^(1|t|true|y|yes|on)$ ]] && echo X-Dynamo-Session-ID || :
}

# The router's bound on each request to a worker, its whole response
# included, which `--request-timeout-secs` sets (default 1800 s) and the
# clock cannot reach. A simulated stream lasts as much wall time as the run
# takes to reach its end, past 1800 s at c=256, so a simulated launch sets
# the bound to a year.
COMPASS_SIM_REQUEST_TIMEOUT_S=31536000
# Every wall-clock bound a simulated router gets, each at that year: the one
# above, and the prepare pool's wait for a thread and its preparation of a
# request (`--prepare-queue-timeout-ms`, `--prepare-timeout-ms`; atomesh's
# defaults are 250 ms and 5 s). Those two start once the router LP releases a
# request, so its hold runs neither out (atom/mesh/tests/compass_pd_router_test.rs),
# but a loaded host stretches a preparation's wall time, which the run's clock
# does not see.
COMPASS_SIM_ROUTER_BOUNDS=(--request-timeout-secs "$COMPASS_SIM_REQUEST_TIMEOUT_S"
    --prepare-queue-timeout-ms "${COMPASS_SIM_REQUEST_TIMEOUT_S}000"
    --prepare-timeout-ms "${COMPASS_SIM_REQUEST_TIMEOUT_S}000")

# compass_sim_router_refusal LP WORD...: why a simulated router refuses its
# `atomesh launch` arguments WORD..., or nothing. LP is 1 where the router runs
# as LP `router`, whose retry resends a request under its first seq.
compass_sim_router_refusal() {
    local lp=$1 word why
    shift
    for word; do
        case ${word%%=*} in
        --prefill | --decode)
            echo "the router's arguments carry ${word%%=*}, a worker more for a simulated router to choose"
            return ;;
        --max-concurrent-requests | --rate-limit-tokens-per-second)
            echo "the router's arguments carry ${word%%=*}, a limiter a simulated router leaves out"
            return ;;
        --request-timeout-secs | --prepare-queue-timeout-ms | --prepare-timeout-ms)
            echo "the router's arguments carry ${word%%=*}, a wall-clock bound the script sets for a simulated router"
            return ;;
        esac
    done
    if ((lp)) && [[ " $* " != *" --disable-retries "* ]]; then
        why="carry --dp-aware"
        [[ " $* " == *" --dp-aware "* ]] || why="choose among a side's instances as LP router"
        echo "the router's arguments $why but not --disable-retries: a router LP's retry resends a request under its first seq, which the frontend refuses as arrived twice"
    fi
}

# compass_on_arch RUN_FILE: the words ahead of `-m MODULE` that start a simulated
# server under on_arch on RUN_FILE's machine's arch, as `compass run` starts its
# own (atom/compass/drive/serve.py); none when RUN_FILE names no machine.
compass_on_arch() {
    python3 -c 'import json, sys
from atom.compass.drive.serve import ON_ARCH
from atom.compass.run import _machine
run = json.load(open(sys.argv[1]))
"machine" in run and print(*ON_ARCH, _machine(run).value("device.arch"))' "$1"
}

# compass_wait_up WHAT PID URL: return once URL answers; exit 1 if PID exits
# first or WAIT_S seconds pass.
compass_wait_up() {
    local deadline=$((SECONDS + WAIT_S)) me
    me=$(basename -- "$0" .sh)
    until curl -sf --max-time 10 "$3" >/dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "$me: $1 exited before it answered" >&2; exit 1; }
        ((SECONDS < deadline)) || { echo "$me: no $1 after $WAIT_S s" >&2; exit 1; }
        sleep 1
    done
}

# compass_gpus hip|smi LIST: the comma-separated GPUs LIST names, as HIP's
# ordinals (`hip`, LIST in rocm-smi's indexes) or as rocm-smi's indexes (`smi`,
# LIST in HIP's ordinals), mapped as `atom/compass/drive/gpu.py::hip_visible`
# maps them. A GPU it cannot map fails.
compass_gpus() {
    rocm-smi --showhw | awk '/^[0-9]+ +[0-9]+ /' | sort -k2,2n | awk -v to="$1" -v list="$2" '
        NR > 1 && $2 == last { dup = 1 }
        { last = $2; hip[$1] = NR - 1; smi[NR - 1] = $1 }
        END {
            k = split(list, want, ",")
            for (i = 1; i <= k; i++) {
                if (dup || (to == "hip" ? !(want[i] in hip) : !(want[i] in smi))) exit 1
                out = out (i > 1 ? "," : "") (to == "hip" ? hip[want[i]] : smi[want[i]])
            }
            print out
        }' || { echo "compass_gpus: rocm-smi --showhw does not map every GPU of $2 to its own KFD node" >&2; return 1; }
}

# compass_machine FILE: write to FILE the machine spec `compass run --device
# DEVICE --ptl PTL [--stack STACK]` composes, the platform_readings entry read
# from the store READINGS, for a run file's `machine` block. Exits when DEVICE,
# PTL or READINGS is unset.
compass_machine() {
    python3 -c 'import json, sys
from atom.compass.artifacts import ArtifactStore
from atom.compass.fit.device import compose
device, ptl, readings, stack, out = sys.argv[1:]
spec = compose(device, ArtifactStore(readings), ptl, stack or None)
open(out, "w").write(json.dumps(spec.echo()))' \
        "${DEVICE:?DEVICE names the datasheet of the GPUs, by SKU or path}" \
        "${PTL:?PTL names their PTL state, enabled or disabled}" \
        "${READINGS:?READINGS names a store holding their platform_readings}" \
        "${STACK:-}" "$1"
}

# aiperf as a real server's client runs it, `python -c "$COMPASS_REAL_AIPERF"
# ROOT/compass_harness/compass_harness/session_ids.py profile ...`: as shipped,
# but for what compass-harness also fixes in a simulated run, so both send the
# same requests: the benchmark id, which the prompts carry, and each session's
# id, drawn from its trace and branch (compass_harness/session_ids.py), which
# aiperf's services inherit by fork (session_ids.bind_by_fork).
COMPASS_REAL_AIPERF='import importlib.util as u, sys, uuid, aiperf.cli_runner as r
r.uuid4 = lambda: uuid.UUID(int=0)
spec = u.spec_from_file_location("session_ids", sys.argv.pop(1))
ids = sys.modules["session_ids"] = u.module_from_spec(spec)
spec.loader.exec_module(ids)
ids.bind_by_fork()
from aiperf.cli import app
app()'
