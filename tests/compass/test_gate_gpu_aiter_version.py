# SPDX-License-Identifier: MIT
"""`gate_gpu.sh` reads aiter's version without executing aiter.

Importing aiter shells out to `rocminfo`, which hangs on a wedged driver. These
run the gate's own resolution lines against stub `aiter` and `torch` packages,
first on the path, that leave a marker and raise when executed. The expected
version is `git describe` on the stub, never a literal: two aiter versions are in
circulation on this project's machines.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE_GPU = REPO / "scripts" / "compass" / "gate_gpu.sh"
GIT = shutil.which("git")
DESCRIBE = ["describe", "--tags", "--always", "--dirty"]


def _resolution_lines():
    """From the `AITER_DIR=` assignment through the `fi` that closes its branch."""
    lines = GATE_GPU.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("AITER_DIR="))
    end = lines.index("fi", start)
    return "\n".join(lines[start : end + 1])


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_aiters_version_is_read_without_executing_aiter_or_torch(tmp_path):
    site, markers = tmp_path / "site", tmp_path / "markers"
    markers.mkdir()
    for name in ("aiter", "torch"):
        (site / name).mkdir(parents=True)
        (site / name / "__init__.py").write_text(
            f"open({str(markers / name)!r}, 'w').close()\nraise RuntimeError\n"
        )
    git = [GIT, "-C", str(site), "-c", "user.name=t", "-c", "user.email=t@t"]
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "s"], ["tag", "v0"]):
        subprocess.run(git + argv, check=True, capture_output=True)

    env = dict(
        os.environ,
        PYTHONPATH=f"{site}{os.pathsep}{REPO}",
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    )
    script = _resolution_lines() + '\nprintf "%s\\n%s\\n" "$AITER_DIR" "$AITER"'
    out = subprocess.check_output(
        ["bash", "-c", script], cwd=tmp_path, env=env, text=True, timeout=60
    )
    aiter_dir, version = out.splitlines()

    assert sorted(p.name for p in markers.iterdir()) == [], "a stub was executed"
    assert aiter_dir == str(site / "aiter")
    described = subprocess.check_output([GIT, "-C", aiter_dir, *DESCRIBE], text=True)
    assert version == described.strip()


def test_the_located_name_is_top_level():
    """`find_spec` imports the parents of a dotted name; `aiter` has none."""
    assert 'module_root("aiter")' in _resolution_lines()
