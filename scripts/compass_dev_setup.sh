#!/usr/bin/env bash
# Point the container's editable install at this working copy.
#
# The container venv lives in the writable layer, so this is lost on
# ./teardown.sh or ./setup.sh --recreate. Re-run it after either.
#
#   ../gpu_docker/shell.sh bash -lc /workspace/ATOM/scripts/compass_dev_setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The checkout is owned by the host user while the container runs as root, so
# git refuses to touch it until told the directory is trusted. Lost on recreate
# along with the rest of /root.
git config --global --add safe.directory "${REPO}" 2>/dev/null || true

echo "repointing editable install -> ${REPO}"
pip install -q -e "${REPO}"
python - <<PY
import os, atom
here = os.path.dirname(atom.__file__)
want = os.path.join("${REPO}", "atom")
print("atom imports from:", here)
assert os.path.realpath(here) == os.path.realpath(want), (
    f"editable install still points elsewhere: {here}"
)
print("ok")
PY
