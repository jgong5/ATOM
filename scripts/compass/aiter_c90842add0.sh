#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# gpu-40's aiter: the image's aiter 41d4d8d7e with upstream aiter c90842add0's
# csrc applied, committed, and the two modules that compile it rebuilt. Run it in
# every mi355-gpu-40 cell container, after the container is created and before
# any server starts in it:
#
#   podman exec CONTAINER bash scripts/compass/aiter_c90842add0.sh
#
# c90842add0 (ROCm/aiter, 2026-10-28) keeps the MX quant + MoE sort and the
# multi-phase MoE sort in bounds under `num_local_tokens`: without it, a MoE pass
# whose `num_local_tokens` is one short of its rows zeroes the row past its
# stage-2 activations, and a count equal to the rows reads past `topk_ids`.
# Its csrc, `aiter_c90842add0.patch` beside this file, applies to 41d4d8d7e
# as is; `module_quant` and `module_moe_sorting_opus` are the modules that
# compile those two files.
#
# The commit takes a fixed author, committer, date and message, so every
# container of the image makes the same one, `PATCHED`, which
# `machines/mi355x-gpu40.json` pins as aiter's commit: a container that has not
# run this reads 41d4d8d7e, and every command that checks the stack against the
# spec refuses there. A checkout already at `PATCHED` is not committed again.
# The modules are rebuilt for gfx950, the arch of gpu-40's cards, unless a build
# stamp names the files a fresh process loads; the check then imports each one
# through aiter's loader in a fresh process and requires that its file is the
# build's, by path and sha256.
#
# Exit 0 when aiter is at `PATCHED` and loads the modules built here; 1, naming
# the step, otherwise.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PATCH=$HERE/aiter_c90842add0.patch
#: The image's aiter commit (rocm/atom-dev:vllm-v0.31.0-nightly_20261026).
BASE=41d4d8d7ea831d35f4a60a6c44228d7a33ed2e27
#: BASE with the patch committed as below.
PATCHED=09d256420152d923359e6752a7920c980f930555
MODULES=(module_quant module_moe_sorting_opus)
ARCH=gfx950
STAMP_NAME=.aiter_c90842add0.built

fail() { echo "aiter_c90842add0: $1" >&2; exit 1; }

# aiter's package directory, found without importing aiter (its import shells
# out to rocminfo).
PKG=$(python3 -c 'import importlib.util, os; s = importlib.util.find_spec("aiter"); print(os.path.dirname(s.origin) if s and s.origin else "")')
[ -n "$PKG" ] || fail "aiter is not importable here"
ROOT=$(git -C "$PKG" rev-parse --show-toplevel) || fail "aiter at $PKG is not a git checkout"
HEAD=$(git -C "$ROOT" rev-parse HEAD)

if [ "$HEAD" != "$PATCHED" ]; then
    [ "$HEAD" = "$BASE" ] ||
        fail "aiter at $ROOT is $HEAD, neither the image's $BASE nor $PATCHED"
    # A fresh container's layer holds the files under other stat data than the
    # image's index records, and `apply --index` refuses a stale entry.
    git -C "$ROOT" update-index -q --refresh || true
    git -C "$ROOT" diff --quiet HEAD -- ||
        fail "aiter at $ROOT has uncommitted changes to tracked files; nothing was applied"
    git -C "$ROOT" apply --index "$PATCH" || fail "the patch does not apply at $BASE"
    GIT_AUTHOR_NAME=honglie GIT_AUTHOR_EMAIL=hyi@amd.com \
        GIT_AUTHOR_DATE=2026-10-28T13:27:38Z \
        GIT_COMMITTER_NAME=honglie GIT_COMMITTER_EMAIL=hyi@amd.com \
        GIT_COMMITTER_DATE=2026-10-28T13:27:38Z \
        git -C "$ROOT" -c commit.gpgsign=false commit -q --no-verify \
        -m "c90842add0's csrc on 41d4d8d7e" \
        -m "Applied by ATOM's scripts/compass/aiter_c90842add0.sh: upstream aiter c90842add0, csrc only." ||
        fail "git commit failed at $ROOT"
    HEAD=$(git -C "$ROOT" rev-parse HEAD)
    [ "$HEAD" = "$PATCHED" ] ||
        fail "the commit is $HEAD, not $PATCHED: this checkout is not the image's"
fi

# The file each module loads from, with its sha256, as a fresh process sees it.
loaded() {
    python3 - "${MODULES[@]}" <<'PY'
import hashlib, sys
from aiter.jit.core import get_module
for name in sys.argv[1:]:
    path = get_module(name).__file__
    with open(path, "rb") as f:
        print(name, path, hashlib.sha256(f.read()).hexdigest())
PY
}

STAMP=$PKG/jit/$STAMP_NAME
if [ -r "$STAMP" ] && [ "$(head -1 "$STAMP")" = "$PATCHED" ] &&
    [ "$(tail -n +2 "$STAMP")" = "$(loaded 2>/dev/null | grep '^module_')" ]; then
    echo "aiter_c90842add0: aiter at ${PATCHED:0:9}, modules already built here"
    exit 0
fi

rm -f "$STAMP"
GPU_ARCHS=$ARCH python3 - "${MODULES[@]}" <<'PY' || fail "the module build failed"
import inspect, sys
from aiter.jit.core import build_module, clear_build, get_args_of_build, rm_module
accepted = set(inspect.signature(build_module).parameters)
for name in sys.argv[1:]:
    rm_module(name)
    clear_build(name)
    args = {**get_args_of_build(name), "md_name": name}
    build_module(**{k: v for k, v in args.items() if k in accepted})
PY

found=$(loaded | grep '^module_') || fail "a rebuilt module does not load"
for name in "${MODULES[@]}"; do
    built=$PKG/jit/$name.so
    line=$(grep "^$name " <<<"$found") || fail "$name did not load"
    read -r _ path sha <<<"$line"
    [ "$(realpath "$path")" = "$(realpath "$built")" ] ||
        fail "$name loads from $path, not the build's $built"
    [ "$sha" = "$(sha256sum "$built" | cut -d' ' -f1)" ] ||
        fail "$name at $path is not the file just built"
done
printf '%s\n%s\n' "$PATCHED" "$found" >"$STAMP"
echo "aiter_c90842add0: aiter at ${PATCHED:0:9}; loads:"
sed 's/^/  /' <<<"$found"
