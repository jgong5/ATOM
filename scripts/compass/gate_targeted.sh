#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The task gate: the test files under tests/ this tree's change adds or edits,
# run as gate_cpu.sh runs them. gate_cpu.sh, the whole CPU suite, is the batch
# gate.
#
# What it keys on:
#   the change         gate_cpu.sh's sources (_lib.sh, compass_changed):
#                      COMPASS_CHANGED_FILES, the diff since
#                      COMPASS_INTEGRATION_REF's merge-base, or .compass-changed
#   the container      ATOM_COMPASS_SLICE_MODEL, set: the GPU container; unset:
#                      the CPU container. Only the mode reads it; the caller
#                      sets every other driver-run input gate_gpu.sh names,
#                      since not every driver-run test reads this one
#   the GPU target     torch's, of the first card in HIP_VISIBLE_DEVICES (GPU)
#   the smoke's model  ATOM_COMPASS_CCTRACES_MODEL (GPU)
#   the harness wheel  COMPASS_AGENTX_HARNESS, as in gate_cpu.sh
#
# In the CPU container a selected file runs unless it is under tests/plugin/, in
# cpu_gate_exclude.txt, or has a test that skipped for want of gate_gpu.sh's
# driver-run inputs; any of those makes the run exit 98, naming the file. In the
# GPU container every selected file runs but those under tests/plugin/ and those
# gpu_gate_skip.txt leaves out on the target (98 again), the cc-traces smoke
# apart, with its own model as its slice model, as in gate_gpu.sh; a driver-run
# skip fails the run with 92, unless gpu_gate_skip.txt says the target's nodes
# hold no driver-run inputs. A selected file that collects no test passes. A
# selected file that skipped because cargo cannot resolve atomesh's crates
# refuses the run with 96, as gate_cpu.sh refuses an atom/mesh diff.
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
ARGS=("$@")

ROOT=$(compass_tree_root) || finish $? "the tree to measure could not be resolved; nothing was run"
cd "$ROOT" || finish 90 "cannot cd into the tree; nothing was run"
compass_env "$ROOT"
compass_require_tree "$ROOT" || finish $? "atom does not import from this tree; nothing was run"
compass_describe "$ROOT"

compass_changed "$ROOT" "${COMPASS_INTEGRATION_REF:-feature/atomcompass_new}" || finish $? "$WHY; nothing was run"
[ -z "$CHANGED_UNKNOWN" ] || finish 98 "unknown which files this change touches: $CHANGED_UNKNOWN; nothing was run"

# The test files pytest collects (test_*.py, *_test.py) that the change adds or
# edits; one it deletes has nothing left to run.
SEL=()
while IFS= read -r f; do
    [[ $f =~ ^tests/(.*/)?(test_[^/]*|[^/]*_test)\.py$ ]] && [ -f "$f" ] && SEL+=("$f")
done <<<"$CHANGED"
[ "${#SEL[@]}" -gt 0 ] || { printf 'select: none (%s)\n' "$CHANGED_SRC"; finish 0; }

GPU=${ATOM_COMPASS_SLICE_MODEL:+1}
NO_DRIVER_RUN=
if [ -n "$GPU" ]; then
    IGN=(--ignore=tests/plugin)
    # gpu_gate_skip.txt, read for the first card's target as gate_gpu.sh reads it.
    CARD=${HIP_VISIBLE_DEVICES:-0}
    ARCH=$(compass_gpu_arch "${CARD%%,*}")
    [ -n "$ARCH" ] || printf 'gpu:    torch reads no GPU target for card %s, so gpu_gate_skip.txt leaves nothing out\n' "${CARD%%,*}"
    while IFS= read -r f; do
        [ "$f" = driver-run ] && NO_DRIVER_RUN=1 || IGN+=(--ignore="$f")
    done < <([ -z "$ARCH" ] || compass_gpu_skip_lines "$ARCH")
    LEFT="gpu_gate_skip.txt leaves it out on $ARCH"
else
    compass_cpu_ignores "$ROOT" || finish $? "$WHY; nothing was run"
    LEFT="cpu_gate_exclude.txt leaves it out, it needs a driver"
fi
# NEED holds `<file>: <why>` for each selected file this container does not run.
RUN=() NEED=()
for f in "${SEL[@]}"; do
    why=
    for i in "${IGN[@]#--ignore=}"; do
        [[ $f == "$i" || $f == "$i"/* ]] || continue
        why=$LEFT
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
    LOG=$(mktemp) || finish 96 "mktemp failed; nothing was run"
    trap 'rm -rf "${HARNESS:-}" "$LOG"' EXIT
    mapfile -t ALL < <(find tests \( -name 'test_*.py' -o -name '*_test.py' \) | sort)
    # run_tests FILE...: one process that collects all of tests/, as
    # gate_cpu.sh's do, and deselects every test file but FILE...: run alone, a
    # file can reach rocminfo at collection that it does not reach in the batch.
    # RC keeps the first failing status; 5, no test collected, is not one.
    run_tests() {
        local rc desel
        mapfile -t desel < <(printf '%s\n' "${ALL[@]}" | grep -vxF -f <(printf '%s\n' "$@"))
        python -m pytest tests/ "${IGN[@]}" -q --no-header -p no:cacheprovider \
            "${desel[@]/#/--deselect=}" "${ARGS[@]}" -rfs 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
        printf '\npytest: rc=%s\n' "$rc"
        if [ "$rc" -eq 5 ]; then
            printf 'pytest: no test collected from %s\n' "$*"
        elif [ "$RC" -eq 0 ]; then
            RC=$rc
        fi
    }
    # The cc-traces smoke replays turns longer than the slice model's context,
    # so in the GPU container it runs alone, as gate_gpu.sh runs it.
    SMOKE=tests/compass/test_cctraces_sim_smoke.py
    MAIN=() ALONE=
    for f in "${RUN[@]}"; do
        [ -n "$GPU" ] && [ "$f" = "$SMOKE" ] && ALONE=1 || MAIN+=("$f")
    done
    [ "${#MAIN[@]}" -eq 0 ] || run_tests "${MAIN[@]}"
    [ -z "$ALONE" ] ||
        PYTHONPATH=$ROOT/compass_harness:$PYTHONPATH ATOM_COMPASS_SLICE_MODEL=${ATOM_COMPASS_CCTRACES_MODEL:-} \
            run_tests "$SMOKE"
    # skipped PATTERN: the files the SKIPPED lines matching PATTERN name. A skip
    # line names the file and line that raised it. One outside any function in
    # a test file the change did not select is that file's module-level skip,
    # taken because it was collected only to be deselected, and is dropped. One
    # inside a function is a helper a selected test called, and is kept
    # wherever it lives; a fixture's skip names the test that asked for it.
    skipped() {
        grep -E "$1" "$LOG" | python -c '
import ast, re, sys
run = set(sys.argv[1:])
for line in sys.stdin:
    path, n = re.match(r"SKIPPED (?:\[\d+\] )?([^:]*):?(\d+)?", line).groups()
    test = re.fullmatch(r"tests/(.*/)?(test_[^/]*|[^/]*_test)\.py", path)
    if path in run or not test or not n or any(
        isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef))
        and d.lineno <= int(n) <= d.end_lineno
        for d in ast.walk(ast.parse(open(path).read()))
    ):
        print(path)
' "${RUN[@]}" | sort -u
    }
    [ "$RC" -eq 0 ] || finish "$RC" "pytest failed; the FAILED lines above name the tests"
    CARGO=$(skipped "$COMPASS_CARGO_SKIP")
    [ -z "$CARGO" ] || finish 96 "$(tr '\n' ' ' <<<"$CARGO")skipped, cargo cannot resolve atomesh's crates here; see scripts/compass/README.md"
    if [ -n "$NO_DRIVER_RUN" ]; then
        printf 'driver-run: skips accepted on %s, whose nodes hold no inputs\n' "$ARCH"
    else
        while IFS= read -r f; do
            NEED+=("$f: a test skipped for want of gate_gpu.sh's driver-run inputs")
        done < <(skipped "$COMPASS_DRIVER_RUN_SKIP")
    fi
fi

if [ "${#NEED[@]}" -gt 0 ]; then
    printf 'not run here:\n' >&2
    printf '  %s\n' "${NEED[@]}" >&2
    if [ -n "$GPU" ] && printf '%s\n' "${NEED[@]}" | grep -q "driver-run inputs$"; then
        finish 92 "a driver-run test skipped with ATOM_COMPASS_SLICE_MODEL set: ${NEED[*]%%: *}; set every driver-run input gate_gpu.sh names"
    fi
    finish 98 "${#NEED[@]} selected file(s) not run here: ${NEED[*]%%: *}; see 'not run here' above"
fi
finish 0
