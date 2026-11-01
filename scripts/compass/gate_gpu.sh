#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Per-wave test gate: the driver-dependent superset of what gate_cpu.sh runs.
# Judged as a DELTA, not as green -- the baseline carries failures. A report
# that says "green" has not read it.
#
# The delta is judged on the failing node-IDS, not only on their count:
# gpu_gate_known_failures.txt in the card's pair names them verbatim. "5 failed"
# could never be checked against "the SAME 5 failed"; a list can.
#
# Also run per-task by any change gate_cpu.sh flags against
# gpu_gate_triggers.txt. That check is mechanical now; this comment is no longer
# the only place the rule lives.
#
# Run in the GPU container. preflight.sh is invoked here, before and after the
# run -- it used to be named in a comment and called by nobody.
#
# Five ways a green verdict was previously reachable on an unmeasured run, all
# of them measured, all of them now refusals:
#   - a summary carrying `N errors` was parsed for passed/failed only, so a run
#     with 12 collection errors read as PASS (reviewer, replayed against this
#     script);
#   - pytest's own exit status was captured and never consulted, so rc=2
#     (interrupted) or rc=4 (usage error) read as a measurement;
#   - a caller's `-rE` overrode this script's `-rf` -- pytest's `-r` is
#     store-last-wins -- so NO `FAILED ` lines were printed, the by-name check
#     compared against an empty observed set, and the gate reported all five
#     baseline failures as "no longer failing" and exited 0 while they had in
#     fact failed (EXECUTED on d78f3bbd3, run 2, gate_gpu_run2_rE.log);
#   - `GONE_IDS` only printed a note, so "the by-name check found nothing" and
#     "the failures were fixed" were one reading;
#   - the pass count was a floor, so the new tests every task adds loosened it
#     by exactly enough to absorb a silently skipped file.
#
# One way is NOT a refusal, and this header claimed otherwise until 2026-09-20
# ("EVERY question this gate cannot answer is a refusal"). A toolchain differing
# from the baseline's -- including an AITER version that reads UNKNOWN, which the
# toolchain comparison below explicitly calls a mismatch -- warns on stderr and
# can still end
# GATE_GPU_RC=0. Most known failures are AITER kernels, so on an
# AITER bump the delta compares two different things and exits 0 anyway. Making
# it a refusal is a behaviour change with its own cost -- a `git describe` that
# cannot read aiter's checkout would then block every run on a node where the
# stack is fine -- so the gap is stated here rather than closed unilaterally.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

# GATE_GPU_RC is printed exactly once, on stdout, on every path -- including the
# FATAL ones, which is why finish() is defined before the first exit rather than
# after pytest. Previously a compound regression (a count change AND a new
# node-id) printed it twice with no statement of which was the verdict, four
# FATAL paths printed it not at all, and the failure verdicts went to stderr
# while the success verdict went to stdout -- so `gate_gpu.sh 2>/dev/null | grep
# GATE_GPU_RC` was silent on failure and indistinguishable from "never ran".
#
# The verdict line carries its reason, for the reasons given above finish() in
# gate_cpu.sh: it is the only line every pipe keeps.
finish() {
    if [ "$1" -eq 0 ]; then
        printf 'GATE_GPU_RC=0 PASSED\n'
    else
        printf 'GATE_GPU_RC=%s NOT PASSED -- %s\n' "$1" "$2"
    fi
    exit "$1"
}
printf 'verdict: the last line of stdout; if this run is piped, $? is the pipe'"'"'s\n'

ROOT=$(compass_tree_root) || finish $? "the tree to measure could not be resolved; nothing was run"
cd "$ROOT" || finish 90 "cannot cd into the tree; nothing was run"
compass_env "$ROOT"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

PRE=$(dirname -- "${BASH_SOURCE[0]}")/preflight.sh
printf '===== pre-flight (before) =====\n'
"$PRE" || { printf 'ABORT: pre-flight failed before the run.\n' >&2; finish 91 "pre-flight failed before the run; nothing was run"; }

compass_require_tree "$ROOT" || finish $? "atom does not import from this tree; nothing was run"
compass_describe "$ROOT"

# The baseline is a pair AND the tree and toolchain it was measured on. A bare
# pair cannot distinguish a regression from drift: after a torch bump, 4728/7
# has no reading. regen_gpu_gate_baseline.sh measures these seven constants and
# writes them to gpu_gate_baseline.env; changing that file is a reviewed edit.
#
# There is one pair per GPU target, in gpu_gate_baselines/<target>/; the first
# card's is read. BASE_LEFT_OUT, when set, counts the tests gpu_gate_skip.txt
# left out of the baseline's run.
#
# BASE_ROCM is compared against torch.version.hip below, so it carries the HIP
# runtime version, not the ROCm release.
#
# AITER has no __version__; BASE_AITER is `git describe --tags --always` in the
# source checkout aiter imports from. Most known failures are AITER kernels or the
# gfx950-only formats they test, so AITER is the likeliest single cause of a
# change that moves them -- a bump used to pass the drift check in silence
# because only torch and HIP were compared.
#
# BASE_COMPASS_TESTS is how many of BASE_PASSED came from tests/compass/ at
# BASE_COMMIT; the regenerated baseline runs none of them. The pass count
# is checked for EQUALITY against BASE_PASSED adjusted by what tests/compass/
# contributes on this tree, so a surplus is accounted for rather than absorbed.
# Every Compass task adds tests there; a task that adds tests anywhere else makes
# this an explicit mismatch, which is the point.
#
# Both sides are PASS counts. This one was a pass count all along; the tree side
# used to be a COLLECTED count, so the first Compass test to skip or xfail on a
# GPU host would have made a legitimately green tree read `unaccounted -1` and
# blamed tests outside tests/compass/. The tree side is now the passes of the
# tests/compass run below, so the two are the same quantity by construction
# rather than while a condition happens to hold.
ARCH=$(compass_gpu_arch "${HIP_VISIBLE_DEVICES%%,*}")
[ -n "$ARCH" ] || finish 93 "torch reads no GPU target for card ${HIP_VISIBLE_DEVICES%%,*}, so no baseline applies; nothing was run"
PAIR=$(dirname -- "${BASH_SOURCE[0]}")/gpu_gate_baselines/$ARCH
BASELINE=$PAIR/gpu_gate_baseline.env
[ -r "$BASELINE" ] || {
    printf 'FATAL: %s is missing; measure it with regen_gpu_gate_baseline.sh on a %s card.\n' "$BASELINE" "$ARCH" >&2
    finish 93 "no baseline for $ARCH; nothing was run"
}
printf 'baseline pair: %s\n' "$PAIR"
. "$BASELINE"
for key in BASE_PASSED BASE_FAILED BASE_COMMIT BASE_TORCH BASE_ROCM BASE_AITER BASE_COMPASS_TESTS; do
    [ -n "${!key:-}" ] || finish 93 "$(basename "$BASELINE") sets no $key; nothing was run"
done

# The baseline's failures by name. The count above and the list here are two
# statements of one fact, so a disagreement between them is a defect in this
# script rather than a test result -- refuse instead of picking one.
KNOWN_FILE=$PAIR/gpu_gate_known_failures.txt
known_ids() { grep -vE '^[[:space:]]*(#|$)' "$KNOWN_FILE"; }
[ -r "$KNOWN_FILE" ] || {
    printf 'FATAL: %s is missing; the baseline names no failures.\n' "$KNOWN_FILE" >&2
    finish 93 "the known-failures list is missing; nothing was run"
}
KNOWN_N=$(known_ids | grep -c .)
[ "$KNOWN_N" -eq "$BASE_FAILED" ] || {
    printf 'FATAL: BASE_FAILED=%s but %s names %s node-id(s).\n' \
        "$BASE_FAILED" "$KNOWN_FILE" "$KNOWN_N" >&2
    finish 93 "BASE_FAILED disagrees with the known-failures list; nothing was run"
}

# pytest's -r is store-last-wins, so a caller's -rE silently replaces the -rfE
# below and the by-name check loses its input. (EXECUTED: `gate_gpu.sh -rE` on
# d78f3bbd3 reported all five baseline failures "no longer failing" and exited
# GATE_GPU_RC=0 while they had failed.)
#
# What makes the verdict safe is the ORDERING at the pytest call: -rfE is placed
# after "$@", so it wins whatever the caller passed. The refusal below is UX,
# not the fix -- it tells the caller why their flag is being ignored. It is also
# incomplete: argparse accepts combined short options, and `-qrE` matches
# neither arm here. That is tolerable exactly because the ordering, not this
# loop, is what holds.
for arg in "$@"; do
    case "$arg" in
    -r | -r*)
        printf 'REFUSED: %s sets pytest -r, which this gate owns. Its verdict is\n' "$arg" >&2
        printf '  built from the FAILED and ERROR report lines, and -r is\n' >&2
        printf '  store-last-wins, so your flag would silently empty the observed\n' >&2
        printf '  failure set and the by-name check would compare nothing.\n' >&2
        finish 95 "refused a -r argument; nothing was run"
        ;;
    esac
done

TORCH=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo UNKNOWN)
ROCM=$(python -c 'import torch; print(torch.version.hip)' 2>/dev/null || echo UNKNOWN)
# aiter's checkout is located without importing aiter: its import shells out to
# rocminfo, which hangs uninterruptibly on a wedged driver -- the box where the
# version is most worth recording. module_root resolves a top-level name with
# importlib.util.find_spec, which executes nothing, and its import chain loads
# neither aiter nor torch.
AITER_DIR=$(python -c 'from atom.compass.artifacts.provenance import module_root; print(module_root("aiter"))' 2>/dev/null)
if [ -n "$AITER_DIR" ]; then
    AITER=$(git -C "$AITER_DIR" describe --tags --always --dirty 2>/dev/null || echo UNKNOWN)
else
    AITER=UNKNOWN
fi
printf 'torch:  %s\nhip:    %s\naiter:  %s\n' "$TORCH" "$ROCM" "$AITER"
printf 'baseline: %s passed / %s failed / 0 errors at %s\n' \
    "$BASE_PASSED" "$BASE_FAILED" "$BASE_COMMIT"
printf '          torch %s, hip %s, aiter %s\n' "$BASE_TORCH" "$BASE_ROCM" "$BASE_AITER"
# UNKNOWN is a mismatch, not a match: an unreadable version is exactly the case
# where the pair below cannot be trusted.
if [ "$TORCH" != "$BASE_TORCH" ] || [ "$ROCM" != "$BASE_ROCM" ] || [ "$AITER" != "$BASE_AITER" ]; then
    printf 'WARNING: toolchain differs from the baseline'"'"'s. The known failures are\n' >&2
    printf '         mostly AITER kernel results that depend on exactly this\n' >&2
    printf '         stack; the pair below is not comparable until it is\n' >&2
    printf '         re-measured on this stack with regen_gpu_gate_baseline.sh.\n' >&2
fi

# The driver-run tests start ATOM's own servers on the simulated clock. Each
# skips without its inputs, and a skip is not a failure, so a gate that does not
# set them passes without running them. On a tree whose tests read the slice
# model, every input is set here, by default to node 18's, and a missing one
# refuses the run by name.
GPUS=$HIP_VISIBLE_DEVICES
SMOKE=
if grep -rqs ATOM_COMPASS_SLICE_MODEL "$ROOT/tests/compass"; then
    hf_snapshot() { ls -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--$1/snapshots/"*/ 2>/dev/null | head -1; }
    export ATOM_COMPASS_SLICE_MODEL=${ATOM_COMPASS_SLICE_MODEL:-$(hf_snapshot Qwen3-0.6B)}
    export ATOM_COMPASS_IMAGE_MODEL=${ATOM_COMPASS_IMAGE_MODEL:-$(hf_snapshot Qwen3.5-9B)}
    # The speculative runs read a config and a tokenizer, no weights: MTP1 from
    # Qwen3.5-0.8B, DSpark from DeepSeek-V4.1-Flash's config with V4-Flash's tokenizer.
    export ATOM_COMPASS_MTP_MODEL=${ATOM_COMPASS_MTP_MODEL:-$(hf_snapshot Qwen3.5-0.8B)}
    export ATOM_COMPASS_DSPARK_MODEL=${ATOM_COMPASS_DSPARK_MODEL:-/root/cells/deepseek-v41-flash-dspark}
    # The cc-traces smoke replays turns longer than the slice model's context, so
    # it runs alone, with this model as its slice model.
    SMOKE=tests/compass/test_cctraces_sim_smoke.py
    export ATOM_COMPASS_CCTRACES_MODEL=${ATOM_COMPASS_CCTRACES_MODEL:-$(hf_snapshot Qwen3.8-27B)}
    export ATOM_COMPASS_CCTRACES=${ATOM_COMPASS_CCTRACES:-/root/cells/cc-traces-256k/traces.jsonl}
    # agentx-harness with compass-harness installed, and without it.
    export COMPASS_HARNESS_PYTHON=${COMPASS_HARNESS_PYTHON:-/root/cells/venv/bin/python}
    export COMPASS_AIPERF_PYTHON=${COMPASS_AIPERF_PYTHON:-/opt/venv/bin/python}
    # The cards' PTL state is part of the machine's key; node 18's run with it on.
    export ATOM_COMPASS_SLICE_MACHINE=${ATOM_COMPASS_SLICE_MACHINE:-$ROOT/scripts/compass/machines/mi308x-hjbog18.json}
    # The router-LP forms refuse a router not built at this tree's commit, so a
    # snapshot needs one built (cargo build --release in atom/mesh) in a git
    # checkout of it.
    export ATOMESH=${ATOMESH:-$ROOT/atom/mesh/target/release/atomesh}
    export PATH=$(dirname -- "$ATOMESH"):$PATH
    MISSING=
    [ -d "$ATOM_COMPASS_SLICE_MODEL" ] || MISSING+="ATOM_COMPASS_SLICE_MODEL, a Qwen3-0.6B directory; "
    [ -d "$ATOM_COMPASS_IMAGE_MODEL" ] || MISSING+="ATOM_COMPASS_IMAGE_MODEL, a Qwen3.5-9B directory; "
    [ -d "$ATOM_COMPASS_MTP_MODEL" ] || MISSING+="ATOM_COMPASS_MTP_MODEL, a Qwen3.5-0.8B directory; "
    [ -d "$ATOM_COMPASS_DSPARK_MODEL" ] || MISSING+="ATOM_COMPASS_DSPARK_MODEL $ATOM_COMPASS_DSPARK_MODEL; "
    [ -d "$ATOM_COMPASS_CCTRACES_MODEL" ] || MISSING+="ATOM_COMPASS_CCTRACES_MODEL, a Qwen3.8-27B directory; "
    [ -r "$ATOM_COMPASS_CCTRACES" ] || MISSING+="ATOM_COMPASS_CCTRACES $ATOM_COMPASS_CCTRACES; "
    [ -x "$COMPASS_HARNESS_PYTHON" ] || MISSING+="COMPASS_HARNESS_PYTHON $COMPASS_HARNESS_PYTHON; "
    [ -x "$COMPASS_AIPERF_PYTHON" ] || MISSING+="COMPASS_AIPERF_PYTHON $COMPASS_AIPERF_PYTHON; "
    [ -r "$ATOM_COMPASS_SLICE_MACHINE" ] || MISSING+="ATOM_COMPASS_SLICE_MACHINE $ATOM_COMPASS_SLICE_MACHINE; "
    ROUTER=$(python -c 'import sys; from atom.compass.run import router_binary; router_binary(sys.argv[1])' \
        "$ATOMESH" 2>&1) || MISSING+="ATOMESH, ${ROUTER##*$'\n'}; "
    [ "$(tr ',' '\n' <<<"$GPUS" | grep -c .)" -ge 4 ] ||
        MISSING+="four cards in HIP_VISIBLE_DEVICES, which names $GPUS; "
    [ -z "$MISSING" ] || finish 92 "the driver-run tests lack ${MISSING%; }; nothing was run"
    printf 'driver-run: slice %s\n            image %s\n            machine %s\n            router %s\n' \
        "$ATOM_COMPASS_SLICE_MODEL" "$ATOM_COMPASS_IMAGE_MODEL" "$ATOM_COMPASS_SLICE_MACHINE" "$ATOMESH"
    printf 'spec:       mtp %s\n            dspark %s\n' "$ATOM_COMPASS_MTP_MODEL" "$ATOM_COMPASS_DSPARK_MODEL"
    printf 'cc-traces:  model %s\n            traces %s\n            harness %s\n            aiperf %s\n' \
        "$ATOM_COMPASS_CCTRACES_MODEL" "$ATOM_COMPASS_CCTRACES" "$COMPASS_HARNESS_PYTHON" "$COMPASS_AIPERF_PYTHON"
fi

# tests/compass runs once, on every card named, and the rest of the suite on the
# first card alone, as the baseline was measured: an upstream test that needs
# two cards skips there. The surplus over the baseline is what tests/compass
# passes on this tree; an absent tests/compass contributes none.
# The tests that cargo-build atomesh skip unless cargo resolves its crates; the
# gate has no network, so they come from the container's cache.
export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
ARGS=("$@")
COUT=$(mktemp)
COMPASS_N=0 C_FAILED=0 C_ERRORS=0
compass_tests() {
    printf '===== %s on HIP_VISIBLE_DEVICES=%s =====\n' "$1" "$GPUS"
    local log rc
    log=$(mktemp)
    python -m pytest "$@" -q --no-header -p no:cacheprovider "${ARGS[@]}" -rfEs 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    cat "$log" >>"$COUT"
    compass_pytest_counts "$log"
    rm -f "$log"
    # rc 5 is "no tests collected": a directory holding nothing runnable passes none.
    case "$rc" in
    0 | 1 | 5) ;;
    *) rm -f "$COUT"; finish 1 "pytest rc=$rc over $1; the run did not complete" ;;
    esac
    COMPASS_N=$((COMPASS_N + PASSED)) C_FAILED=$((C_FAILED + FAILED)) C_ERRORS=$((C_ERRORS + ERRORS))
}
if [ -f "$SMOKE" ]; then
    compass_tests tests/compass --ignore="$SMOKE"
    # The tree's compass-harness ahead of the one the harness Python has installed.
    PYTHONPATH=$ROOT/compass_harness:$PYTHONPATH ATOM_COMPASS_SLICE_MODEL=$ATOM_COMPASS_CCTRACES_MODEL \
        compass_tests "$SMOKE"
elif [ -d "$ROOT/tests/compass" ]; then
    compass_tests tests/compass
else
    printf 'tests/compass: absent on this tree, so it contributes no tests.\n'
fi
# A file that stalls on one GPU keeps the suite from ever printing a summary.
# gpu_gate_skip.txt names such files per GPU; on a matching first card each is
# left out of the suite and its collected tests come off the expectation.
DEVICE=$(HIP_VISIBLE_DEVICES=${GPUS%%,*} python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)
printf 'device: %s\n' "${DEVICE:-UNKNOWN}"
compass_gpu_skips "$DEVICE" || finish 1 "$SKIP_WHY"
LEFT_OUT=$((SKIP_N - ${BASE_LEFT_OUT:-0}))
EXPECT_PASSED=$((BASE_PASSED + COMPASS_N - BASE_COMPASS_TESTS - LEFT_OUT))
printf 'expected: %s passed = %s baseline + (%s - %s) tests/compass - (%s - %s) left out here and there\n' \
    "$EXPECT_PASSED" "$BASE_PASSED" "$COMPASS_N" "$BASE_COMPASS_TESTS" "$SKIP_N" "${BASE_LEFT_OUT:-0}"
printf '\n'

OUT=$(mktemp)
# -rfE, after "$@" so it wins even if a -r form slips past the refusal above.
HIP_VISIBLE_DEVICES=${GPUS%%,*} python -m pytest tests/ --ignore=tests/plugin --ignore=tests/compass \
    "${SKIP_ARGS[@]}" -q --no-header -p no:cacheprovider "$@" -rfE 2>&1 | tee "$OUT"
PYRC=${PIPESTATUS[0]}
compass_pytest_counts "$OUT"

# A summary line that parses to 0/0/0, or no summary line at all, means the run
# did not get far enough to produce one -- a collection error, an OOM, a killed
# process. That is not a pass count below baseline; it is no measurement at all,
# and it must not be read through the delta rule below.
if [ -z "$SUMMARY" ] || { [ "$PASSED" -eq 0 ] && [ "$FAILED" -eq 0 ] && [ "$ERRORS" -eq 0 ]; }; then
    printf 'NO MEASUREMENT -- no usable pytest summary line found.\n' >&2
    tail -20 "$OUT" >&2
    rm -f "$OUT" "$COUT"
    finish 1 "no usable pytest summary line; the run is not a measurement"
fi

# pytest's own verdict, which the delta rule cannot replace: 0 = all passed,
# 1 = tests failed. Anything else (2 interrupted, 3 internal error, 4 usage
# error, 5 nothing collected, 128+n a signal) means the summary above describes
# a run that did not complete, whatever it says.
case "$PYRC" in
0 | 1) ;;
*)
    printf 'pytest rc=%s -- not "all passed" and not "tests failed", so the run\n' "$PYRC" >&2
    printf '  did not complete and the counts above are not a measurement.\n' >&2
    rm -f "$OUT" "$COUT"
    finish 1 "pytest rc=$PYRC; the run did not complete"
    ;;
esac

# From here on the two runs are judged as one suite.
PASSED=$((PASSED + COMPASS_N)) FAILED=$((FAILED + C_FAILED)) ERRORS=$((ERRORS + C_ERRORS))
printf '\nGATE_GPU passed=%s failed=%s errors=%s pytest_rc=%s (baseline %s/%s/0, expected %s passed)\n' \
    "$PASSED" "$FAILED" "$ERRORS" "$PYRC" "$BASE_PASSED" "$BASE_FAILED" "$EXPECT_PASSED"
# A driver-run test that skipped did not run, whatever the counts say.
SKIPS=$(grep -E '^SKIPPED .*(ATOM_COMPASS_(SLICE|IMAGE|MTP|DSPARK)_MODEL|ATOM_COMPASS_CCTRACES|COMPASS_AIPERF_PYTHON|atomesh)' "$COUT")
cat "$COUT" >>"$OUT"
rm -f "$COUT"

# Each finding goes to stderr in full. The first one's one-line form, and how
# many more there were, travel on the verdict line. The checks run with the
# node-id comparisons ahead of the counts they explain, so a new failure is
# named there by its node-id rather than as a changed count.
# A finding passed without a one-line form is named by its own first line, so
# a call that omits it still ends on a verdict rather than on `set -u`.
VERDICT=0
WHY=
fail() { printf '%s\n' "$1" >&2; VERDICT=$((VERDICT + 1)); [ -n "$WHY" ] || WHY=${2:-${1%%$'\n'*}}; }

# Errors are not failures and were never parsed: a run reading "4800 passed, 5
# failed, 12 errors" produced GATE_GPU_RC=0 without one word about the twelve.
[ "$ERRORS" -eq 0 ] ||
    fail "$(printf '%s collection/teardown error(s). The baseline has none, and an
  error is not a failure -- the tests behind it did not run at all, so the
  pass count below is over a smaller suite than the baseline'"'"'s.' "$ERRORS")" \
    "$ERRORS collection/teardown error(s)"
[ -z "$SKIPS" ] ||
    fail "$(printf 'driver-run test(s) skipped, so not run:\n%s' "$(sed 's/^/  /' <<<"$SKIPS")")" \
        "$(grep -c . <<<"$SKIPS") driver-run skip line(s), first ${SKIPS%%$'\n'*}"

# The names, not only the counts: one baseline failure fixed and one new one
# introduced reads as 5/5.
OBS_IDS=$(mktemp)
ERR_IDS=$(mktemp)
KNOWN_IDS=$(mktemp)
grep '^FAILED ' "$OUT" | sed -e 's/^FAILED //' -e 's/ - .*$//' | sort -u >"$OBS_IDS"
grep '^ERROR ' "$OUT" | sed -e 's/^ERROR //' -e 's/ - .*$//' | sort -u >"$ERR_IDS"
known_ids | sort -u >"$KNOWN_IDS"
OBS_N=$(grep -c . "$OBS_IDS")
ERR_N=$(grep -c . "$ERR_IDS")

# The observed set has to be as big as the summary says, or the by-name check
# below is comparing against an accident. This is the check that makes the
# whole name comparison honest: an empty observed set is otherwise
# indistinguishable from "nothing failed".
[ "$OBS_N" -eq "$FAILED" ] ||
    fail "$(printf 'the summary says %s failed but %s FAILED line(s) were printed.
  The by-name comparison has no set to compare, so it is not evidence either
  way. (pytest -r is store-last-wins: a caller flag, a truncated log or a
  plugin can all empty it.)' "$FAILED" "$OBS_N")"
[ "$ERR_N" -eq "$ERRORS" ] ||
    fail "$(printf 'the summary says %s error(s) but %s ERROR line(s) were printed.' \
        "$ERRORS" "$ERR_N")"

NEW_IDS=$(comm -23 "$OBS_IDS" "$KNOWN_IDS")
GONE_IDS=$(comm -13 "$OBS_IDS" "$KNOWN_IDS")
if [ -n "$NEW_IDS" ]; then
    fail "$(printf 'failure(s) the baseline does not name:\n%s' \
        "$(printf '%s\n' "$NEW_IDS" | sed 's/^/  /')")" \
        "$(printf '%s\n' "$NEW_IDS" | grep -c .) failure(s) the baseline does not name, first ${NEW_IDS%%$'\n'*}"
fi
# Not good news by default, and the gate cannot tell the three causes apart:
# the failure was fixed; the test stopped running; or the observed set could not
# be read. Only the first is a pass, so this refuses and asks for the reviewed
# edit the known-failures file already requires.
if [ -n "$GONE_IDS" ]; then
    fail "$(printf 'baseline failure(s) not observed failing:\n%s\n  Confirm which of the three this is -- fixed, no longer collected, or an
  observed set that could not be read -- then drop them from %s.
  That edit is reviewed; it is not something this gate may assume.' \
        "$(printf '%s\n' "$GONE_IDS" | sed 's/^/  /')" "$(basename "$KNOWN_FILE")")" \
        "$(printf '%s\n' "$GONE_IDS" | grep -c .) baseline failure(s) not observed failing, first ${GONE_IDS%%$'\n'*}"
fi
# grep -c always prints a number, so an emptiness test here would be dead.
if [ "$ERR_N" -gt 0 ]; then
    printf 'erroring node-id(s):\n' >&2
    sed 's/^/  /' "$ERR_IDS" >&2
fi
rm -f "$OBS_IDS" "$ERR_IDS" "$KNOWN_IDS"

[ "$FAILED" -eq "$BASE_FAILED" ] ||
    fail "$(printf '%s failed, baseline %s.' "$FAILED" "$BASE_FAILED")"

# EQUALITY, not a floor. A floor is loosened by exactly the new tests every task
# adds, so a task that adds 30 tests and silently skips a 20-test file still
# clears it, and the only trace was a `note:`. The surplus is derived from this
# tree rather than declared by the caller, and the mismatch text below states it
# as its parts, since a single total cannot show which side moved.
[ "$PASSED" -eq "$EXPECT_PASSED" ] ||
    fail "$(printf '%s passed, expected %s. The difference decomposes as:
    baseline        %s   at %s
    tests/compass   %s now, %s at the baseline (delta %s)
    left out        %s   here, less at the baseline
    expected        %s
    observed        %s
    unaccounted     %s
  A surplus is not slack: it absorbs a file that stopped being collected. Both
  tests/compass figures are pass counts, so a skip or an xfail there is already
  accounted for; if this tree adds tests outside tests/compass/, account for
  them here.' \
        "$PASSED" "$EXPECT_PASSED" "$BASE_PASSED" "$BASE_COMMIT" \
        "$COMPASS_N" "$BASE_COMPASS_TESTS" "$((COMPASS_N - BASE_COMPASS_TESTS))" "$LEFT_OUT" \
        "$EXPECT_PASSED" "$PASSED" "$((PASSED - EXPECT_PASSED))")" \
    "$PASSED passed, expected $EXPECT_PASSED"

rm -f "$OUT"

printf '\n===== pre-flight (after) =====\n'
"$PRE" || printf 'WARNING: pre-flight failed AFTER the run -- the node changed under it,\n         and the numbers above may not be attributable.\n' >&2

[ "$VERDICT" -le 1 ] || WHY="${WHY%.}; $((VERDICT - 1)) more finding(s) on stderr"
[ "$VERDICT" -eq 0 ] || finish 1 "$WHY"

# An attestation names a tree. On a tree with no commit, gate_cpu.sh refuses
# COMPASS_GPU_GATE_DONE=UNKNOWN -- so say that here rather than print an
# instruction that is guaranteed to fail when the caller follows it.
HEAD_SHA=$(compass_head_sha "$ROOT") || HEAD_SHA=UNKNOWN
if [ "$HEAD_SHA" = UNKNOWN ]; then
    printf 'satisfied for this tree, but it cannot be attested: no .git and no\n'
    printf '  .compass-commit stamp, so the attestation could name any tree and\n'
    printf '  gate_cpu.sh will refuse it. Rebuild with scripts/compass/snapshot.sh.\n'
else
    printf 'satisfied for this tree. gate_cpu.sh will accept:\n'
    printf '  export COMPASS_GPU_GATE_DONE=%s\n' "$HEAD_SHA"
fi
finish 0
