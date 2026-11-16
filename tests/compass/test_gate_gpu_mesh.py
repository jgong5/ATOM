# SPDX-License-Identifier: MIT
"""`gate_gpu.sh` gives a snapshot's atom/mesh the image's Cargo.lock, as
`gate_cpu.sh` does, so the driver-run tests that cargo-build atomesh resolve its
crates offline instead of skipping.

Each case runs the real script over a throwaway tree, as
`test_gate_gpu_driver_run.py` does; its driver-run test records the lockfile it
finds in atom/mesh.
"""

import pytest
from test_gate_gpu_driver_run import BASH, DRIVER, _run, _tree

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

#: What the driver-run test records: atom/mesh's lockfile as it runs.
LOCK_READ = (
    "open('atom/mesh/Cargo.lock').read() "
    "if os.path.exists('atom/mesh/Cargo.lock') else 'no lockfile'"
)


@pytest.mark.parametrize("own, read", [(None, "image"), ("tree", "tree")])
def test_tests_compass_runs_under_the_image_lockfile_unless_the_tree_has_one(
    tmp_path, own, read
):
    root = _tree(tmp_path, DRIVER.replace("'compass'", LOCK_READ))
    mesh = root / "atom" / "mesh"
    mesh.mkdir(parents=True)
    (mesh / "Cargo.toml").write_text("")
    if own:
        (mesh / "Cargo.lock").write_text(own)
    (tmp_path / "Cargo.lock").write_text("image")
    out, verdict = _run(tmp_path, root, COMPASS_CARGO_LOCK=str(tmp_path / "Cargo.lock"))
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    runs = (root / "runs.txt").read_text().splitlines()
    assert any(line.startswith(f"{read} 4,5,6,7 ") for line in runs), runs
