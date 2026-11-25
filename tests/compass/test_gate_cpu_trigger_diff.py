# SPDX-License-Identifier: MIT
"""A renamed trigger file and a non-ASCII path under a trigger directory match
a GPU-gate trigger in every diff the match reads.

`git diff --name-only` lists a rename under its new name only and quotes a
non-ASCII path, so neither matched and the gate read `gpu: not required`.
Each case runs the shipped scripts on the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py`` made a git repository, with the
triggers `atom/blind.py` and `atom/sub/`: in place, and staged by
``snapshot.sh`` and extracted; on the diff from the integration branch, and on
the diff a passed GPU gate carries over. A renamed non-trigger file is the
control: it still needs no GPU gate.
"""

import pytest
from test_gate_cpu_gpu_carry import _commit, _git, _snapshot_gate
from test_gate_cpu_verdict_through_pipe import BLIND, _run, _tree

HITS = {
    "renamed": f"{BLIND} -> {BLIND}",
    "non_ascii": "atom/sub/café.py -> atom/sub/",
    "renamed_other": "",
}
cases = pytest.mark.parametrize("staged", [False, True], ids=["checkout", "snapshot"])


@pytest.fixture
def repo(tmp_path):
    root = _tree(tmp_path / "ATOM", stamped=False)
    (root / "scripts/compass/gpu_gate_triggers.txt").write_text(f"{BLIND}\natom/sub/\n")
    (root / "atom/sub").mkdir()
    (root / "atom/other.py").write_text("")
    (root / BLIND).write_text("")
    _git(root, "init", "-q")
    _commit(root, "atom/sub/keep.py", "k = 1\n")
    _git(root, "branch", "feature/atomcompass_new")
    return root


def _change(root, change):
    if change == "non_ascii":
        return _commit(root, "atom/sub/café.py", "c = 1\n")
    _git(root, "mv", BLIND if change == "renamed" else "atom/other.py", "atom/moved.py")
    _git(root, "commit", "-qm", change)
    return _git(root, "rev-parse", "HEAD")


@cases
@pytest.mark.parametrize("change", HITS)
def test_the_diff_from_the_integration_branch_matches_the_trigger(repo, change, staged):
    _change(repo, change)
    out = _snapshot_gate(repo, "", "") if staged else _run(repo, "{gate}")
    hit = HITS[change]
    assert out.returncode == (98 if hit else 0), out.stdout + out.stderr
    want = f"          {hit}\n" if hit else "gpu:    not required"
    assert want in out.stdout, out.stdout


@cases
@pytest.mark.parametrize("change", HITS)
def test_a_carried_gpu_gate_does_not_cross_the_trigger(repo, change, staged):
    g = _commit(repo, "atom/sub/keep.py", "k = 2\n")
    head = _change(repo, change)
    gate = f"COMPASS_GPU_GATE_DONE={g} {{gate}}"
    out = _snapshot_gate(repo, g, g) if staged else _run(repo, gate)
    hit = HITS[change]
    assert out.returncode == (98 if hit else 0), out.stdout + out.stderr
    want = f"touches a trigger: {hit}" if hit else f"carried to {head[:9]}"
    assert want in out.stdout, out.stdout
