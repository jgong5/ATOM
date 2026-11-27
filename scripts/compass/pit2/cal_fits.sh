#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A calibration's fits, once its records are taken (cal_step.sh with
# RECORDS_ONLY=1), inside the CPU container at nice 19: the same command
# cal_step.sh ran, from <OUT>.cmd, which calibrate.sh resumes by skipping each
# record whose output exists.
#
# Usage: cal_fits.sh CAL_OUT
#   CAL_OUT   the calibration's directory, OUT/cal/<candidate>
#
# calibrate.sh refuses a resume given other inputs than its first run's (its
# CAL_OUT/inputs, diffed in the log): another DURATION_S, traces, arguments or
# store, or another tree. Its `tree` line is the commit's SHA, so a resume
# after a commit refuses even when that commit touched atom/compass alone,
# whose files the ATOM digest of a law's key leaves out. Then, and only then,
# move that line to the new commit, keeping the old file:
#     cp CAL_OUT/inputs CAL_OUT/inputs.<old sha>
#     sed -i "1s/.*/tree $(git -C "$PIT2_TREE" rev-parse HEAD)/" CAL_OUT/inputs
# A commit outside atom/compass changes the laws' key: calibrate anew.
# Log: CAL_OUT.fits.log. Ends with `CAL_FITS_EXIT <code>`.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT=${1:?cal_fits.sh CAL_OUT}
OUT=${OUT%/}
[[ -f $OUT.cmd ]] || { echo "cal_fits.sh: no $OUT.cmd; cal_step.sh writes it" >&2; exit 2; }
L=$OUT.fits.log
echo "$(date -u +%FT%TZ) fits of $OUT on $(hostname)" >>"$L"
# Every fit it starts inherits nice 19.
(cd / && nice -n 19 bash -c "eval \"\$(cat '$OUT.cmd')\"") >>"$L" 2>&1
rc=$?
echo "CAL_FITS_EXIT $rc" | tee -a "$L"
