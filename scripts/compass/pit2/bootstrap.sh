#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Once per PIT2_WORK, inside the CPU container (containers.sh cpu): the two
# things a search needs that the image lacks, both under PIT2_WORK.
#
#   venv-sim   HARNESS_PYTHON: a venv over the image's agentx venv (aiperf) and
#              /opt/venv, with this tree's compass_harness installed, so its
#              aiperf plugin is registered. client.sh still imports the tree's
#              own compass_harness first (PYTHONPATH), so a harness change needs
#              no reinstall; a change to its plugin entry points does.
#   atomesh    ATOMESH: the router deploy.sh runs, built with cargo from the
#              tree's atom/mesh into PIT2_WORK/mesh-target (about 10 minutes on
#              24 jobs). An image with a built atomesh at the tree's commit may
#              set ATOMESH to it instead and skip the build (SKIP_ATOMESH=1).
#
# Prints each tool's check and BOOTSTRAP_DONE. Usage: bootstrap.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$PIT2_WORK"
src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT
cp -r "$PIT2_TREE/compass_harness" "$src/compass_harness"
venv=$PIT2_WORK/venv-sim
/opt/agentx-venv/bin/python -m venv "$venv"
site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo 'import site; site.addsitedir("/opt/agentx-venv/lib/python3.12/site-packages")' >"$site/a_agentx_venv.pth"
echo 'import site; site.addsitedir("/opt/venv/lib/python3.12/site-packages")' >"$site/zz_opt_venv.pth"
"$venv/bin/pip" install -q --no-deps --no-build-isolation "$src/compass_harness"
(cd / && PYTHONPATH=$PIT2_TREE "$venv/bin/python" -c 'import compass_harness, aiperf; print("venv-sim ok")')
if [[ -z ${SKIP_ATOMESH-} ]]; then
    # The image's lock file, where the tree has none: cargo resolves the same crates.
    [[ -f $PIT2_TREE/atom/mesh/Cargo.lock ]] || cp /app/ATOM/atom/mesh/Cargo.lock "$PIT2_TREE/atom/mesh/" 2>/dev/null || true
    (cd "$PIT2_TREE/atom/mesh" && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-24} CARGO_TARGET_DIR=$PIT2_WORK/mesh-target \
        nice -n 19 cargo build -q --release --bin atomesh)
fi
"$ATOMESH" --version-verbose | grep -E "Commit|Status"
echo BOOTSTRAP_DONE
