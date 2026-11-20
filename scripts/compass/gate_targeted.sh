#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The task gate: the test files under tests/ this tree's change adds or edits,
# run as gate_cpu.sh runs them. gate_cpu.sh, the whole CPU suite, is the batch
# gate.
#
# The change is read from gate_cpu.sh's sources (_lib.sh, compass_changed). In
# the CPU container a selected file runs unless it is under tests/plugin/, in
# cpu_gate_exclude.txt, or has a test that skipped for want of gate_gpu.sh's
# driver-run inputs; any of those makes the run exit 98, naming the file. With
# ATOM_COMPASS_SLICE_MODEL set, the input every driver-run test reads, this is
# the GPU container: every selected file outside tests/plugin/ runs, and a
# driver-run skip fails the run, as in gate_gpu.sh.
#
# A change on a GPU-trigger path is the batch gate's to send to gate_gpu.sh,
# not this one's.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

# The verdict line is printed exactly once, on every path, and carries its
# reason, as gate_cpu.sh's does: it is the only line every pipe keeps.
finish() {
    if [ "$1" -eq 0 ]; then
        printf 'GATE_TARGETED_RC=0 PASSED\n'
    else
        printf 'GATE_TARGETED_RC=%s NOT PASSED -- %s\n' "$1" "$2"
    fi
    exit "$1"
}
printf 'verdict: the last line of stdout; if this run is piped, $? is the pipe'"'"'s\n'

for arg in "$@"; do
    case "$arg" in
    -r | -r*)
        printf 'REFUSED: %s sets pytest -r, which this gate owns. Its report lines\n' "$arg" >&2
        printf '  name the failing tests and the driver-run skips, and -r is\n' >&2
        printf '  store-last-wins, so your flag would replace them silently.\n' >&2
        finish 95 "refused a -r argument; nothing was run"
        ;;
    esac
done

ROOT=$(compass_tree_root) || finish $? "the tree to measure could not be resolved; nothing was run"
cd "$ROOT" || finish 90 "cannot cd into the tree; nothing was run"
compass_env "$ROOT"
compass_require_tree "$ROOT" || finish $? "atom does not import from this tree; nothing was run"
compass_describe "$ROOT"

compass_changed "$ROOT" "${COMPASS_INTEGRATION_REF:-feature/atomcompass_new}" ||
    { printf 'FATAL: COMPASS_CHANGED_FILES=%s unreadable\n' "$COMPASS_CHANGED_FILES" >&2; finish 94 "COMPASS_CHANGED_FILES unreadable; nothing was run"; }
[ -z "$CHANGED_UNKNOWN" ] || finish 98 "unknown which files this change touches: $CHANGED_UNKNOWN; nothing was run"

# The test files pytest collects (test_*.py, *_test.py) that the change adds or
# edits; one it deletes has nothing left to run.
SEL=()
while IFS= read -r f; do
    [[ $f =~ ^tests/(.*/)?(test_[^/]*|[^/]*_test)\.py$ ]] && [ -f "$f" ] && SEL+=("$f")
done <<<"$CHANGED"
[ "${#SEL[@]}" -gt 0 ] || { printf 'select: none (%s)\n' "$CHANGED_SRC"; finish 0; }

GPU=${ATOM_COMPASS_SLICE_MODEL:+1}
if [ -n "$GPU" ]; then
    IGN=(--ignore=tests/plugin)
else
    compass_cpu_ignores "$ROOT" || finish $? "$WHY; nothing was run"
fi
# NEED holds `<file>: <why>` for each selected file this container does not run.
RUN=() NEED=()
for f in "${SEL[@]}"; do
    why=
    for i in "${IGN[@]#--ignore=}"; do
        [[ $f == "$i" || $f == "$i"/* ]] || continue
        why="cpu_gate_exclude.txt leaves it out, it needs a driver"
        [ "$i" != tests/plugin ] || why="tests/plugin needs sglang and vllm, in no image"
    done
    [ -z "$why" ] && RUN+=("$f") || NEED+=("$f: $why")
done
printf 'select: %s file(s) (%s), %s run here\n' "${#SEL[@]}" "$CHANGED_SRC" "${#RUN[@]}"
[ "${#RUN[@]}" -eq 0 ] || printf '        %s\n' "${RUN[@]}"

RC=0
if [ "${#RUN[@]}" -gt 0 ]; then
    compass_harness "$ROOT" || finish $? "$WHY; nothing was run"
    export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
    [ ! -r "$ROOT/atom/mesh/Cargo.toml" ] || compass_cargo_lock "$ROOT/atom/mesh"
    # One process that collects all of tests/, as gate_cpu.sh's do, and
    # deselects every test file it does not select: run alone, a file can
    # reach rocminfo at collection that it does not reach in the batch.
    DESEL=()
    mapfile -t DESEL < <(find tests \( -name 'test_*.py' -o -name '*_test.py' \) | sort |
        grep -vxF -f <(printf '%s\n' "${RUN[@]}") | sed 's/^/--deselect=/')
    LOG=$(mktemp) || finish 96 "mktemp failed; nothing was run"
    trap 'rm -rf "${HARNESS:-}" "$LOG"' EXIT
    python -m pytest tests/ "${IGN[@]}" -q --no-header -p no:cacheprovider "${DESEL[@]}" "$@" -rfs 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    while IFS= read -r f; do
        NEED+=("$f: a test skipped for want of gate_gpu.sh's driver-run inputs")
    done < <(grep -E "$COMPASS_DRIVER_RUN_SKIP" "$LOG" | sed -E 's/^SKIPPED \[[0-9]+\] ([^:]+):.*/\1/' | sort -u)
fi
printf '\npytest: rc=%s\n' "$RC"
[ "$RC" -eq 0 ] || finish "$RC" "pytest failed; the FAILED lines above name the tests"

if [ "${#NEED[@]}" -gt 0 ]; then
    printf 'not run here:\n' >&2
    printf '  %s\n' "${NEED[@]}" >&2
    if [ -n "$GPU" ] && printf '%s\n' "${NEED[@]}" | grep -q "driver-run inputs$"; then
        finish 92 "a driver-run test skipped with ATOM_COMPASS_SLICE_MODEL set: ${NEED[*]%%: *}; set every driver-run input gate_gpu.sh names"
    fi
    finish 98 "${#NEED[@]} selected file(s) not run here: ${NEED[*]%%: *}; see 'not run here' above"
fi
finish 0
