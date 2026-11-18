# SPDX-License-Identifier: MIT
"""A GPU gate passed on tree G carries to a later tree whose diff from G
touches no GPU-gate trigger, and to no other.

Each case commits G, which adds the trigger file over the integration branch,
then a tree on top of it, in the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py`` made a git repository. The shipped
``snapshot.sh`` stages it and the shipped ``gate_cpu.sh`` runs on the
extraction, as a gate normally runs; the checkout cases run the gate in place.
"""

import subprocess

import pytest
from test_gate_cpu_verdict_through_pipe import BLIND, _run, _tree

SNAPSHOT = "scripts/compass/snapshot.sh"


def _git(root, *args):
    cmd = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    out = subprocess.run(cmd + list(args), check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(root, path, text):
    (root / path).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", path)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    root = _tree(tmp_path / "ATOM", stamped=False)
    _git(root, "init", "-q")
    _commit(root, "atom/__init__.py", "")
    _git(root, "branch", "feature/atomcompass_new")
    return root, _commit(root, BLIND, "g = 1\n")


def _snapshot_gate(root, staged_with, done):
    out = root.parent / "out"
    out.mkdir()
    snap = _run(root, f"COMPASS_GPU_GATE_DONE={staged_with} {SNAPSHOT} {out}")
    assert snap.returncode == 0, snap.stdout + snap.stderr
    subprocess.run(["tar", "-x", "-C", out, "-f", *out.glob("compass-*.tar")], check=True)
    return _run(out / "ATOM", f"COMPASS_GPU_GATE_DONE={done} {{gate}}")


@pytest.mark.parametrize("touch, rc", [("atom/other.py", 0), (BLIND, 98)])
def test_a_snapshot_carries_the_gate_only_over_a_diff_with_no_trigger(repo, touch, rc):
    root, g = repo
    head = _commit(root, touch, "h = 2\n")
    out = _snapshot_gate(root, g, g)
    assert out.returncode == rc, out.stdout + out.stderr
    carried = f"gpu:    satisfied by {g[:9]} carried to {head[:9]}"
    assert (carried in out.stdout) == (rc == 0), out.stdout
    assert rc == 0 or f"touches a trigger: {BLIND} -> {BLIND}" in out.stdout


@pytest.mark.parametrize("staged_with, why", [
    ("", "no .compass-gpu-carry stamp"),
    ("feature/atomcompass_new", "stamp is for "),
])
def test_a_snapshot_without_a_stamp_for_g_does_not_carry(repo, staged_with, why):
    root, g = repo
    _commit(root, "atom/other.py", "h = 2\n")
    out = _snapshot_gate(root, staged_with, g)
    assert out.returncode == 98, out.stdout + out.stderr
    assert why in out.stdout.splitlines()[-1], out.stdout


@pytest.mark.parametrize("named, rc", [(True, 0), (False, 98)])
def test_a_checkout_carries_the_gate_by_its_own_diff(repo, named, rc):
    root, g = repo
    head = _commit(root, "atom/other.py", "h = 2\n")
    done = g if named else "f" * 40
    out = _run(root, f"COMPASS_GPU_GATE_DONE={done} {{gate}}")
    assert out.returncode == rc, out.stdout + out.stderr
    assert named == (f"{g[:9]} carried to {head[:9]}" in out.stdout), out.stdout


def test_snapshot_refuses_a_gated_sha_that_names_no_commit(repo):
    root, _ = repo
    out = _run(root, f"COMPASS_GPU_GATE_DONE={'f' * 40} {SNAPSHOT} {root.parent}")
    assert out.returncode == 93, out.stdout + out.stderr
