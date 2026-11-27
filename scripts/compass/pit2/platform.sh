#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# This node's platform readings into STORE, inside the GPU container under a
# gpool lease (gpool_steps.sh platform:STORE): QUICKSTART's "The platform" with
# the pipeline carriers a PP prefill prices (--pp). About 45 minutes on two idle
# cards; a store whose readings are at this image's stack already needs none
# (`compass advice` says `missing readings` otherwise).
#
# Usage: platform.sh STORE [GPUS]
#   STORE   the store the readings publish into: the space file's, or a
#           calibration's own (OUT/cal/<candidate>/store)
#   GPUS    two idle cards, default 0,1
# Out: PIT2_DATA/platform/<node>-<UTC time>/{measure.log,fit.log,device,readings}.
# Prints the probe record's token and PLATFORM_EXIT <code>.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
STORE=${1:?platform.sh STORE [GPUS]}
R=$PIT2_DATA/platform/$(hostname)-$(date -u +%Y%m%dT%H%M)
mkdir -p "$R"
rocm-smi --showpids 2>&1 | grep -q "No KFD PIDs" || { echo "GPUs busy"; echo PLATFORM_EXIT 1; exit 1; }
compass measure device --gpu "${2:-0,1}" --pp --store "$STORE" --out "$R/device" 2>&1 | tee "$R/measure.log"
tok=$(awk '/published probe_record/ {t = $NF} END {print t}' "$R/measure.log")
echo "TOKEN=$tok"
[[ -n $tok ]] || { echo PLATFORM_EXIT 1; exit 1; }
compass fit device "$tok" --store "$STORE" --out "$R/readings" 2>&1 | tee "$R/fit.log"
echo "PLATFORM_EXIT ${PIPESTATUS[0]}"
