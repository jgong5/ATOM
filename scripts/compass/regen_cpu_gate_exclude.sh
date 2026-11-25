#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Regenerate the GENERATED section of cpu_gate_exclude.txt. Run in the CPU
# container when gate_cpu.sh starts erroring at collection -- ATOM added or
# renamed a driver-dependent test.
#
# Why a loop: pytest reports the collection errors it found and then interrupts,
# so ignoring that batch surfaces the next one. At 83daf636d it converged in
# three passes: 37 errors, then 1, then clean. Hand-editing the list gets the
# first batch and looks finished.
#
# What this script can and cannot see. It greps `^ERROR ` from --collect-only,
# so it finds exactly the files that fail at *import*. A file that imports
# cleanly and then calls the driver inside a test body is invisible to it --
# that is what the MANUAL section is for, and why this script preserves that
# section rather than regenerating it. Before the split, four hand-added entries
# sat in a file whose header said "never hand-edit", and the list could not be
# reproduced from the tree it claimed to describe.
#
# tests/plugin/ is excluded by directory in gate_cpu.sh, so its entries are
# dropped here -- listing them twice would imply they are a driver problem.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
cd "$ROOT" || exit 90
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?

OUT=$ROOT/scripts/compass/cpu_gate_exclude.txt
WORK=$(mktemp)
MANUAL=$(mktemp)
trap 'rm -f "$WORK" "$MANUAL"' EXIT

# This script regenerates one section of an existing file; it does not author
# the header or the manual section.
[ -e "$OUT" ] || {
    printf 'REFUSED: %s does not exist. Restore it from git first.\n' "$OUT" >&2
    exit 94
}

# Carry the manual section across verbatim, comments and all. A file predating
# the split has no markers; refuse rather than silently dropping what a human
# put there.
if ! grep -q '^# BEGIN MANUAL$' "$OUT" || ! grep -q '^# END MANUAL$' "$OUT"; then
    printf 'REFUSED: %s has no BEGIN/END MANUAL markers.\n' "$OUT" >&2
    printf '  It predates the generated/manual split. Add the markers by hand\n' >&2
    printf '  first, deciding for each entry which section it belongs in.\n' >&2
    exit 94
fi
sed -n '/^# BEGIN MANUAL$/,/^# END MANUAL$/p' "$OUT" >"$MANUAL"
PREV_GEN=$(sed -n '/^# BEGIN GENERATED$/,/^# END GENERATED$/p' "$OUT" | grep -c '^tests/')

# Seed the ignore set with the manual entries so the loop sees the same tree
# the gate does.
MAN_FILES=$(grep '^tests/' "$MANUAL" || true)

rc=1
for pass in $(seq 1 15); do
    IGN=(--ignore=tests/plugin)
    while read -r f; do [ -n "$f" ] && IGN+=(--ignore="$f"); done <"$WORK"
    while read -r f; do [ -n "$f" ] && IGN+=(--ignore="$f"); done <<<"$MAN_FILES"
    LOG=$(mktemp)
    python -m pytest tests/ "${IGN[@]}" -q --no-header -p no:cacheprovider \
        --collect-only >"$LOG" 2>&1
    rc=$?
    new=$(grep '^ERROR ' "$LOG" | awk '{print $2}' | grep -v '^tests/plugin/' | sort -u)
    printf 'pass %s: rc=%s new=%s total=%s\n' \
        "$pass" "$rc" "$(printf '%s' "$new" | grep -c .)" "$(grep -c . "$WORK")"
    rm -f "$LOG"
    [ "$rc" -eq 0 ] && break
    [ -z "$new" ] && { printf 'STUCK: rc=%s with no new ERROR lines\n' "$rc" >&2; exit 93; }
    printf '%s\n' "$new" >>"$WORK"
    sort -u -o "$WORK" "$WORK"
done
[ "$rc" -eq 0 ] || { printf 'NO CONVERGENCE after 15 passes (rc=%s)\n' "$rc" >&2; exit 95; }

GEN=$(grep -c . "$WORK")

# An empty result is never a legitimate answer: it would mean the whole suite
# collects in a driverless container, which is the thing this file exists
# because it does not. The previous version ended in `grep . "$WORK" >"$OUT"`,
# which truncates the list to zero bytes on that path and reports success.
if [ "$GEN" -eq 0 ]; then
    printf 'REFUSED: converged with an empty generated list.\n' >&2
    printf '  Either pytest is not reaching the tree or --collect-only failed\n' >&2
    printf '  silently. %s left untouched.\n' "$OUT" >&2
    exit 96
fi

# Shrinking is legitimate -- ATOM can make a test driver-free -- but it is never
# something to discover afterwards from a green gate.
if [ "$GEN" -lt "$PREV_GEN" ] && [ -z "${COMPASS_REGEN_ACCEPT_SHRINK:-}" ]; then
    printf 'REFUSED: generated list shrank %s -> %s. Dropped:\n' "$PREV_GEN" "$GEN" >&2
    comm -23 <(sed -n '/^# BEGIN GENERATED$/,/^# END GENERATED$/p' "$OUT" |
                   grep '^tests/' | sort) <(sort "$WORK") | sed 's/^/    /' >&2
    printf '  Those files now collect without a driver. Confirm that is intended,\n' >&2
    printf '  then re-run with COMPASS_REGEN_ACCEPT_SHRINK=1.\n' >&2
    exit 97
fi

# Build beside the target and rename, so a failure above leaves the committed
# list intact.
TMP=$OUT.tmp
{
    sed -n '1,/^# BEGIN GENERATED$/p' "$OUT" | head -n -1
    printf '# BEGIN GENERATED\n'
    sort -u "$WORK"
    printf '# END GENERATED\n'
    cat "$MANUAL"
} >"$TMP"
mv -f "$TMP" "$OUT"

MAN=$(printf '%s' "$MAN_FILES" | grep -c . || true)
printf 'wrote %s: %s generated + %s manual = %s excluded\n' \
    "$OUT" "$GEN" "$MAN" "$((GEN + MAN))"
