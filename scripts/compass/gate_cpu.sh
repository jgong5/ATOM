#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Per-task test gate: the test files that run without a GPU driver.
# Green is the bar: this subset really is green.
#
# Excluded, and why neither is a judgement call:
#   tests/plugin/          needs sglang + vllm -- in neither image
#   cpu_gate_exclude.txt   files that reach the driver, most at *collection*
#                          time via rocminfo, the rest on a driver call after
#                          collecting. That file names which is which and
#                          which is generated; README.md's row counts both.
#
# What this gate cannot see is not left to memory. gpu_gate_triggers.txt lists
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
#
# The verdict line also carries its reason, because it is the only line every
# pipe keeps. `gate_cpu.sh | tail -6` hands the caller tail's exit status, not
# this script's, and nothing a script does can change that: the status of a
# pipeline belongs to the caller's shell. What survives is the text, and the
# three common pipes keep different parts of it -- `2>/dev/null | tail -6` drops
# every stderr line, so a bare `GATE_CPU_RC=98` there sits under pytest's green
# summary with nothing to say why it is not a pass. Stated on the verdict line,
# the reason survives any pipe that keeps the verdict at all.
#
# Refusing to run when stdout is a pipe was weighed and rejected. It cannot tell
# `| tail` from the callers that keep the status: under `docker exec`, which is
# how the gate runs on the CPU node, stdout is a FIFO too, as it is under any
# harness that captures output. Every normal run would need an override, and an
# override set by habit lets `| tail` through with it.
finish() {
    if [ "$1" -eq 0 ]; then
        printf 'GATE_CPU_RC=0 PASSED\n'
    else
        printf 'GATE_CPU_RC=%s NOT PASSED -- %s\n' "$1" "$2"
    fi
    exit "$1"
}
printf 'verdict: the last line of stdout; if this run is piped, $? is the pipe'"'"'s\n'

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
        finish 95 "refused a -r argument; nothing was run"
        ;;
    esac
done

INTEGRATION=${COMPASS_INTEGRATION_REF:-feature/atomcompass_new}

ROOT=$(compass_tree_root) || finish $? "the tree to measure could not be resolved; nothing was run"
cd "$ROOT" || finish 90 "cannot cd into the tree; nothing was run"
compass_env "$ROOT"
compass_require_tree "$ROOT" || finish $? "atom does not import from this tree; nothing was run"
compass_describe "$ROOT"

compass_cpu_ignores "$ROOT" || finish $? "$WHY; nothing was run"
TRIGGERS=$ROOT/scripts/compass/gpu_gate_triggers.txt
[ -r "$TRIGGERS" ] || { printf 'FATAL: cannot read %s\n' "$TRIGGERS" >&2; finish 94 "trigger list unreadable; nothing was run"; }
printf 'gate:   %s files excluded + tests/plugin\n' "$NEX"

# --- does this change need the GPU gate? -------------------------------------
# Answered here, enforced after pytest, so a run yields both answers rather than
# trading one for the other. That includes the case where the question cannot be
# answered at all: it is recorded, not exited on, so the suite still reports,
# and the run below exits 98 rather than passing -- reporting "GPU not
# required" there would be a guess that reads as a clear.
GPU_NEEDED=
compass_changed "$ROOT" "$INTEGRATION" || finish $? "$WHY; nothing was run"
SRC=$CHANGED_SRC
GPU_SRC_UNKNOWN=$CHANGED_UNKNOWN

# Print `<path> -> <trigger>` for each path in $1 a trigger matches.
trigger_hits() {
    while IFS= read -r pat; do
        pat=${pat%$'\r'}
        case "$pat" in ''|'#'*) continue ;; esac
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            case "$pat" in
            */) case "$f" in "$pat"*) printf '%s -> %s\n' "$f" "$pat" ;; esac ;;
            *)  [ "$f" != "$pat" ] || printf '%s -> %s\n' "$f" "$pat" ;;
            esac
        done <<<"$1"
    done <"$TRIGGERS"
}
[ -n "$GPU_SRC_UNKNOWN" ] || GPU_NEEDED=$(trigger_hits "$CHANGED")

if [ -n "$GPU_SRC_UNKNOWN" ]; then
    printf 'gpu:    UNKNOWN -- %s\n' "$GPU_SRC_UNKNOWN"
elif [ -n "$GPU_NEEDED" ]; then
    printf 'gpu:    REQUIRED (%s)\n' "$SRC"
    printf '%s\n' "$GPU_NEEDED" | sed 's/^/          /'
else
    printf 'gpu:    not required (%s)\n' "$SRC"
fi
printf '\n'

# Identify the tree before running, not after: an attestation that this diff
# already passed the GPU gate is per-tree, so it cannot be honoured on a tree we
# cannot name.
HEAD_SHA=$(compass_head_sha "$ROOT") || HEAD_SHA=UNKNOWN

# The compass_harness tests run, or the gate refuses (_lib.sh, compass_harness).
compass_harness "$ROOT" || finish $? "$WHY; nothing was run"

# atomesh's unit tests and every Compass cargo target (tests/compass_*.rs) run
# here, offline, since the gate has no network: from the container's crate
# cache, under the lockfile that cache was fetched for. test_mesh_clock_client.py
# skips on the same `cargo metadata` probe, so a diff touching atom/mesh is
# refused when the probe fails.
MESH=$ROOT/atom/mesh
MESH_RC=0
CARGO_BIN=
if [ -r "$MESH/Cargo.toml" ]; then
    # pytest's cargo calls in atom/mesh run under flock (below), so a missing
    # flock is refused here, before any cargo runs.
    if CARGO_BIN=$(command -v cargo); then
        FLOCK=$(command -v flock) ||
            finish 96 "flock not found, so cargo in atom/mesh cannot run one process at a time; nothing was run"
    fi
    export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
    compass_cargo_lock "$MESH"
    if WHY=$(cd "$MESH" && cargo metadata --format-version 1 2>&1 >/dev/null); then
        TARGETS=(--lib)
        for f in "$MESH"/tests/compass_*.rs; do
            [ -e "$f" ] && f=${f##*/} && TARGETS+=(--test "${f%.rs}")
        done
        # Tests that fail on a loaded host: a wall-clock throughput floor
        # (#515) and a thread race that records a placeholder tenant (#514).
        SKIP=(--skip core::worker::tests::test_load_counter_performance
              --skip policies::tree::tests::test_tree_structure_integrity_after_stress)
        (cd "$MESH" && cargo test "${TARGETS[@]}" -- --exact "${SKIP[@]}") || MESH_RC=$?
    else
        WHY=${WHY%%$'\n'*}
        grep -q '^atom/mesh/' <<<"$CHANGED" &&
            finish 96 "this diff touches atom/mesh and its cargo tests would skip ($WHY); see scripts/compass/README.md; nothing was run"
        printf 'mesh:   cargo tests skipped, atom/mesh unchanged (%s)\n' "$WHY"
    fi
fi

# pytest's own status, captured directly. Piping it into `tail` discards the
# exit code and a failing suite reports success.
#
# tests/compass holds most of the suite's time, in model traces that each take
# one core, so it runs as COMPASS_GATE_SHARDS pytest processes and the rest of
# tests/ as COMPASS_GATE_REST_SHARDS more, no process running files of both;
# COMPASS_GATE_SHARDS=0 runs tests/ as one process. pack deals each set to its
# processes, never more of them than the set has files. Every process collects
# all of tests/ and deselects what it does not run, because test files collect
# only after earlier files have imported engine modules under an aiter stub;
# alone, they reach rocminfo. The first rest process prints as it runs, the
# others' output follows. A process left with no test to run (exit 5) fails
# the gate only when none ran. A caller's --junitxml or --basetemp path is the
# first rest process's; compass shard k writes <path>.shard<k> and rest process
# k <path>.rest<k>, since a shared one is overwritten or cleared by every
# process.
SHARDS=${COMPASS_GATE_SHARDS:-16}
RS=${COMPASS_GATE_REST_SHARDS:-4}
[[ $SHARDS =~ ^[0-9]+$ ]] || finish 95 "COMPASS_GATE_SHARDS=$SHARDS is not a count; pytest was not run"
[[ $RS =~ ^[1-9][0-9]*$ ]] || finish 95 "COMPASS_GATE_REST_SHARDS=$RS is not a count of 1 or more; pytest was not run"
DURATIONS=$ROOT/scripts/compass/cpu_gate_durations.txt
[ -r "$DURATIONS" ] || finish 94 "cannot read $DURATIONS; nothing was run"
CF=()
[ -d tests/compass ] && mapfile -t CF < <(find tests/compass \( -name 'test_*.py' -o -name '*_test.py' \) | sort)
mapfile -t RF < <(find tests \( -path tests/compass -o -path tests/plugin \) -prune -o \
    \( -name 'test_*.py' -o -name '*_test.py' \) -print | sort | grep -vxF -f <(printf '%s\n' "${IGN[@]#--ignore=}"))
# COMPASS_GATE_SELECT names a map gate_map.sh wrote. gate_select.py prints how
# many test files the diff since the map's commit reaches and then each file
# every process deselects, or why the whole gate runs. Every process still
# collects all of tests/, and a selection of none passes on that collection.
UNSEL=()
SEL_NONE=
if [ -n "${COMPASS_GATE_SELECT:-}" ]; then
    SEL=$(python "$ROOT/scripts/compass/gate_select.py" select "$COMPASS_GATE_SELECT" "$ROOT" "${IGN[@]#--ignore=}") ||
        finish 94 "gate_select.py could not read COMPASS_GATE_SELECT=$COMPASS_GATE_SELECT; nothing was run"
    printf '%s\n' "${SEL%%$'\n'*}"
    [[ $SEL != "select: 0 of "* ]] || SEL_NONE=1
    mapfile -t UNSEL < <(tail -n +2 <<<"$SEL")
    [ "${#UNSEL[@]}" -eq 0 ] || [ "${#CF[@]}" -eq 0 ] ||
        mapfile -t CF < <(printf '%s\n' "${CF[@]}" | grep -vxF -f <(printf '%s\n' "${UNSEL[@]}"))
    [ "${#UNSEL[@]}" -eq 0 ] || [ "${#RF[@]}" -eq 0 ] ||
        mapfile -t RF < <(printf '%s\n' "${RF[@]}" | grep -vxF -f <(printf '%s\n' "${UNSEL[@]}"))
fi
[ "$SHARDS" -le "${#CF[@]}" ] || SHARDS=${#CF[@]}
[ "$SHARDS" -gt 0 ] || RS=1
[ "$RS" -le "${#RF[@]}" ] || RS=$((${#RF[@]} > 0 ? ${#RF[@]} : 1))
LOGS=$(mktemp -d) || finish 96 "mktemp failed; nothing was run"
trap 'rm -rf "${HARNESS:-}" "$LOGS"' EXIT
# More than one of these processes runs cargo in atom/mesh, and on NFS cargo
# takes no lock on atom/mesh/target: two builds there at once delete each
# other's artifacts (#2587). So every process finds cargo through a wrapper
# that runs one started in atom/mesh only under a lock they all share, and any
# other cargo as it is. A cargo started inside a locked one, as atomesh's build
# script runs `cargo --version`, finds the lock's path in
# COMPASS_GATE_CARGO_LOCK and runs without it, instead of waiting on its parent.
if [ -n "$CARGO_BIN" ]; then
    mkdir "$LOGS/bin" &&
        printf '#!/usr/bin/env bash\nL=%q\ncase $(pwd -P)/ in %q/*)\n    [ "${COMPASS_GATE_CARGO_LOCK:-}" = "$L" ] ||\n        { exec 9>>"$L" && %q 9 && export COMPASS_GATE_CARGO_LOCK=$L || exit; } ;;\nesac\nexec %q "$@"\n' \
            "$LOGS/cargo.lock" "$(cd "$MESH" && pwd -P)" "$FLOCK" "$CARGO_BIN" >"$LOGS/bin/cargo" &&
        chmod +x "$LOGS/bin/cargo" || finish 96 "cannot write the cargo wrapper; pytest was not run"
    export PATH=$LOGS/bin:$PATH
fi
PYTEST=(tests/ "${IGN[@]}" -q --no-header -p no:cacheprovider "${UNSEL[@]/#/--deselect=}")
ARGS=("$@")
own() {
    local a prev=
    OWN=()
    for a in "${ARGS[@]}"; do
        case $prev in --junitxml | --junit-xml | --basetemp) a=$a$1 ;; esac
        case $a in --junitxml=* | --junit-xml=* | --basetemp=*) a=$a$1 ;; esac
        prev=$a
        OWN+=("$a")
    done
    OWN+=(-rf)
}
REST=()
for e in tests/*; do
    [ -d "$e" ] && e=$e/
    [ "$e" = tests/compass/ ] || REST+=(--deselect="$e")
done
# pack N FILE...: prints "<process> <file>" for each FILE, dealt to processes
# 0..N-1 longest first, each to the process with the fewest seconds so far by
# cpu_gate_durations.txt, then the fewest files. An unlisted file counts 0.
pack() {
    local n=$1
    shift
    printf '%s\n' "$@" |
        awk -v table="$DURATIONS" 'BEGIN { while ((getline l < table) > 0) if (split(l, a, " ") == 2 && a[1] !~ /^#/) t[a[2]] = a[1] }
            { print ($0 in t ? t[$0] : 0), $0 }' |
        sort -k1,1gr -k2,2 |
        awk -v n="$n" '{ m = 0; for (i = 1; i < n; i++) if (s[i] < s[m] || (s[i] == s[m] && c[i] < c[m])) m = i
            s[m] += $1; c[m]++; print m, $2 }'
}
# others P LINE...: sets OTHERS to a --deselect for each file pack dealt to a
# process other than P.
others() {
    local p=$1 a
    shift
    OTHERS=()
    for a; do [ "${a%% *}" = "$p" ] || OTHERS+=(--deselect="${a#* }"); done
}
CPACK=()
RPACK=()
[ "$SHARDS" -eq 0 ] || mapfile -t CPACK < <(pack "$SHARDS" "${CF[@]}")
[ "$RS" -eq 1 ] || mapfile -t RPACK < <(pack "$RS" "${RF[@]}")
PIDS=()
for ((s = 0; s < SHARDS; s++)); do
    others "$s" "${CPACK[@]}"
    own ".shard$((s + 1))"
    python -m pytest "${PYTEST[@]}" "${OWN[@]}" "${REST[@]}" "${OTHERS[@]}" >"$LOGS/$s" 2>&1 &
    PIDS+=($!)
done
DROP=()
[ "$SHARDS" -eq 0 ] || DROP=(--deselect=tests/compass/)
for ((r = 1; r < RS; r++)); do
    others "$r" "${RPACK[@]}"
    own ".rest$((r + 1))"
    python -m pytest "${PYTEST[@]}" "${OWN[@]}" "${DROP[@]}" "${OTHERS[@]}" >"$LOGS/r$r" 2>&1 &
    PIDS+=($!)
done
others 0 "${RPACK[@]}"
own ""
python -m pytest "${PYTEST[@]}" "${OWN[@]}" "${DROP[@]}" "${OTHERS[@]}"
CODES=($?)
for ((r = 1; r < RS; r++)); do
    wait "${PIDS[SHARDS + r - 1]}"
    CODES+=($?)
    printf '\n== the rest of tests/, process %s of %s: rc=%s\n' $((r + 1)) "$RS" "${CODES[-1]}"
    cat "$LOGS/r$r"
done
for ((s = 0; s < SHARDS; s++)); do
    wait "${PIDS[s]}"
    CODES+=($?)
    printf '\n== tests/compass, shard %s of %s: rc=%s\n' $((s + 1)) "$SHARDS" "${CODES[-1]}"
    cat "$LOGS/$s"
done
RC=0
RAN=
for c in "${CODES[@]}"; do
    case $c in 0) RAN=1 ;; 5) ;; *) [ "$RC" -ne 0 ] || RC=$c ;; esac
done
[ "$RC" -ne 0 ] || [ -n "$RAN$SEL_NONE" ] || RC=5
printf '\npytest: rc=%s\n' "$RC"

if [ "$RC" -ne 0 ]; then
    # Keep these nine lines last in this block, and keep the class name and
    # scripts/compass/README.md inside the final five. A caller who pipes loses
    # $? but not the text: `2>&1 | tail -6` keeps only the last five of them,
    # and `2>/dev/null | tail -6` drops them all but keeps pytest's own FAILED
    # line. Either way the reader still ends up holding the test's identity.
    # Appending to this block -- or inserting after the class name -- breaks
    # that, and nothing here will fail if it does.
    printf 'CPU gate FAILED. The baseline is measured, not read:\n' >&2
    printf 'run this script on the integration head this branch forked from and\n' >&2
    printf 'compare -- see scripts/compass/README.md.\n' >&2
    printf 'Before you read it as your diff, check the FAILED line above. One test in\n' >&2
    printf 'ATOM'"'"'s own suite -- TestTheRegionIsNotCopiedPerChunk in\n' >&2
    printf 'tests/entrypoints/test_stream_marker_properties.py -- asserts a wall-clock\n' >&2
    printf 'timing property and fails intermittently on a loaded box. The\n' >&2
    printf 'scripts/compass/README.md section names it, what it was measured to do, and\n' >&2
    printf 'to run gates one at a time. It is not a Compass defect and is not excluded.\n' >&2
    finish "$RC" "pytest failed; the FAILED lines above name the tests"
fi
[ "$MESH_RC" -eq 0 ] || finish "$MESH_RC" "cargo test failed in atom/mesh; its output is above pytest's"

if [ -n "$GPU_SRC_UNKNOWN" ]; then
    printf 'pytest is green, but the CPU gate is not passed: %s.\n' "$GPU_SRC_UNKNOWN" >&2
    printf 'So this run cannot tell whether the diff needs the GPU gate, and an\n' >&2
    printf 'unanswered question is not a "no". It is not a finding about the tree.\n' >&2
    printf '  Fix by any one of: stage with scripts/compass/snapshot.sh, which stamps\n' >&2
    printf '  .compass-changed; set COMPASS_CHANGED_FILES=<file listing the diff>; or\n' >&2
    printf '  run in a tree where %s resolves.\n' "$INTEGRATION" >&2
    finish 98 "unknown whether this diff needs the GPU gate: $GPU_SRC_UNKNOWN"
fi

# A green pytest run is not a passed gate when the diff lands in the blind spot:
# the GPU gate has to have run on this same tree, or carry to it, and said so.
if [ -n "$GPU_NEEDED" ]; then
    DONE=${COMPASS_GPU_GATE_DONE:-}
    if [ -z "$DONE" ]; then
        printf 'this diff touches the CPU gate'"'"'s blind spot; the GPU gate has not run.\n' >&2
        printf '  Run gate_gpu.sh, then re-run with COMPASS_GPU_GATE_DONE=<that tree'"'"'s HEAD>\n' >&2
        finish 98 "this diff needs the GPU gate, which has not run; run gate_gpu.sh"
    fi
    # An attestation names a tree. Without a commit we cannot check it names
    # THIS tree, and accepting it unchecked would make the whole rule
    # bypassable by running in a snapshot -- which is where the gate normally
    # runs. Refuse instead: stamp the snapshot with snapshot.sh.
    if [ "$HEAD_SHA" = UNKNOWN ]; then
        printf 'COMPASS_GPU_GATE_DONE=%s cannot be checked: this tree has no commit.\n' "$DONE" >&2
        printf '  No .git and no .compass-commit stamp, so the attestation could name\n' >&2
        printf '  any tree. Rebuild the snapshot with scripts/compass/snapshot.sh.\n' >&2
        finish 98 "COMPASS_GPU_GATE_DONE cannot be checked; this tree has no commit"
    fi
    # A GPU gate passed on full sha G carries here when the diff from G, git's or
    # snapshot.sh's .compass-gpu-carry stamp (G, HEAD, the diff), hits no trigger.
    G=$DONE CARRY= WHY= HITS=
    if [ "$DONE" = "$HEAD_SHA" ]; then :
    elif git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        { [ "$(git -C "$ROOT" rev-parse --verify --quiet "$DONE^{commit}")" = "$DONE" ] &&
            CARRY=$(git -C "$ROOT" -c core.quotePath=false diff --no-renames --name-only "$G"); } || WHY="$DONE is not the full sha of a commit here"
    elif [ -s "$ROOT/.compass-gpu-carry" ]; then
        { read -r G; read -r TO; CARRY=$(cat); } <"$ROOT/.compass-gpu-carry"
        [ "$G $TO" = "$DONE $HEAD_SHA" ] || WHY="the .compass-gpu-carry stamp carries $G to $TO"
    else
        WHY="the snapshot has no .compass-gpu-carry stamp; stage it with COMPASS_GPU_GATE_DONE set"
    fi
    [ -n "$WHY" ] || HITS=$(trigger_hits "$CARRY")
    [ -z "$HITS" ] || WHY="the diff from ${G:0:9} touches a trigger: $(printf '%s' "$HITS" | tr '\n' ' ')"
    if [ -n "$WHY" ]; then
        printf 'COMPASS_GPU_GATE_DONE=%s but this tree is %s.\n' "$DONE" "$HEAD_SHA" >&2
        printf '  The GPU gate passed on a different tree. Re-run it here.\n' >&2
        finish 98 "COMPASS_GPU_GATE_DONE names a different tree and does not carry here: $WHY; run gate_gpu.sh here"
    fi
    [ "$G" = "$HEAD_SHA" ] && printf 'gpu:    satisfied by COMPASS_GPU_GATE_DONE=%s\n' "${DONE:0:9}" ||
        printf 'gpu:    satisfied by %s carried to %s\n' "${G:0:9}" "${HEAD_SHA:0:9}"
fi
finish 0
