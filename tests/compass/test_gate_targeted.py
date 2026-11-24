# SPDX-License-Identifier: MIT
"""`gate_targeted.sh` runs the test files a change adds or edits, as
`gate_cpu.sh` runs them, and says so on one verdict line.

Each case runs the real script over a throwaway tree, built as
`test_gate_cpu_verdict_through_pipe.py` builds its own, with a test file that
passes, one that fails, one the exclusion list names, one under tests/plugin,
and one whose test skips for want of a driver-run input. Nothing needs a driver.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_gate_gpu_verdict_through_pipe import _torch

REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    BASH is None or shutil.which("git") is None, reason="needs bash and git"
)

FILES = {
    "tests/test_ok.py": "def test_ok():\n    pass\n",
    "tests/sub/test_bad.py": "def test_bad():\n    assert False\n",
    "tests/test_kernel.py": "def test_kernel():\n    pass\n",
    "tests/plugin/test_plugin.py": "def test_plugin():\n    pass\n",
    "tests/test_slice.py": "import pytest\n\n\ndef test_run():\n"
    "    pytest.skip('needs a driver and ATOM_COMPASS_SLICE_MODEL')\n",
}


def _tree(root):
    """A minimal checkout the gate accepts; nothing stamped."""
    shutil.copytree(
        REPO / "scripts" / "compass",
        root / "scripts" / "compass",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (root / "scripts" / "compass" / "cpu_gate_exclude.txt").write_text(
        "tests/test_kernel.py\n"
    )
    (root / "atom").mkdir()
    (root / "atom" / "__init__.py").write_text("")
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _run(root, *args, **env):
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COMPASS_", "ATOM_COMPASS_"))
    }
    clean["PATH"] = os.path.dirname(sys.executable) + os.pathsep + clean["PATH"]
    # A checkout above the temp dir must not make the bare tree look like git.
    clean["GIT_CEILING_DIRECTORIES"] = str(root.parent)
    out = subprocess.run(
        [BASH, "scripts/compass/gate_targeted.sh", *args],
        cwd=root,
        env={**clean, **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    keyed = [ln for ln in out.stdout.splitlines() if "GATE_TARGETED_RC=" in ln]
    assert len(keyed) == 1 and out.stdout.splitlines()[-1] == keyed[0], out.stdout
    return out


def _changed(root, *names):
    (root / "changed.txt").write_text("".join(n + "\n" for n in names))
    return str(root / "changed.txt")


@pytest.fixture
def tree(tmp_path):
    return _tree(tmp_path / "ATOM")


def test_no_changed_test_file_passes_without_pytest(tree):
    lst = _changed(tree, "atom/x.py", "tests/conftest.py", "tests/test_gone.py")
    out = _run(tree, COMPASS_CHANGED_FILES=lst)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "select: none (COMPASS_CHANGED_FILES)" in out.stdout
    assert "pytest:" not in out.stdout


@pytest.mark.parametrize(
    "name, rc, verdict",
    [("tests/test_ok.py", 0, "1 passed"), ("tests/sub/test_bad.py", 1, "1 failed")],
)
def test_only_the_changed_test_file_runs(tree, name, rc, verdict):
    out = _run(tree, COMPASS_CHANGED_FILES=_changed(tree, "atom/x.py", name))
    assert out.returncode == rc, out.stdout + out.stderr
    assert "select: 1 file(s) (COMPASS_CHANGED_FILES), 1 run here" in out.stdout
    assert verdict in out.stdout and "deselected" in out.stdout, out.stdout


@pytest.mark.parametrize(
    "name",
    ["tests/test_kernel.py", "tests/plugin/test_plugin.py", "tests/test_slice.py"],
)
def test_a_file_the_cpu_container_cannot_run_exits_98_naming_it(tree, name):
    out = _run(tree, COMPASS_CHANGED_FILES=_changed(tree, name, "tests/test_ok.py"))
    assert out.returncode == 98, out.stdout + out.stderr
    assert "1 passed" in out.stdout, "the runnable file still ran"
    verdict = out.stdout.splitlines()[-1]
    assert verdict.endswith(f"not run here: {name}; see 'not run here' above"), verdict


def test_the_gpu_container_runs_the_excluded_file_and_fails_a_driver_run_skip(tree):
    lst = _changed(tree, "tests/test_kernel.py")
    out = _run(tree, COMPASS_CHANGED_FILES=lst, ATOM_COMPASS_SLICE_MODEL="/m")
    assert out.returncode == 0 and "1 passed" in out.stdout, out.stdout + out.stderr
    lst = _changed(tree, "tests/test_slice.py")
    out = _run(tree, COMPASS_CHANGED_FILES=lst, ATOM_COMPASS_SLICE_MODEL="/m")
    assert out.returncode == 92, out.stdout + out.stderr
    assert "skipped with ATOM_COMPASS_SLICE_MODEL set: tests/test_slice.py" in (
        out.stdout.splitlines()[-1]
    )


def _git(root, *args):
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + list(args), check=True, capture_output=True)


@pytest.mark.parametrize("source", ["git", "stamp", "file", "none"])
def test_the_changed_list_comes_from_gate_cpus_sources(tree, source):
    if source == "git":
        _git(tree, "init", "-q")
        _git(tree, "add", "-A")
        _git(tree, "commit", "-qm", "base")
        _git(tree, "branch", "feature/atomcompass_new")
        (tree / "tests" / "test_new.py").write_text("def test_new():\n    pass\n")
        _git(tree, "add", "-A")
        _git(tree, "commit", "-qm", "head")
    elif source != "none":
        (tree / ".compass-changed").write_text("tests/test_ok.py\n")
    env = {}
    if source == "file":
        # The supplied list wins over the stamp beside it.
        env["COMPASS_CHANGED_FILES"] = _changed(tree, "tests/sub/test_bad.py")
    out = _run(tree, **env)
    if source == "none":
        assert out.returncode == 98, out.stdout + out.stderr
        assert "never stamped" in out.stdout.splitlines()[-1]
        return
    want = {
        "git": "(diff vs feature/atomcompass_new ",
        "stamp": "(.compass-changed stamp)",
        "file": "(COMPASS_CHANGED_FILES)",
    }[source]
    assert f"select: 1 file(s) {want}" in out.stdout, out.stdout
    assert out.returncode == (1 if source == "file" else 0), out.stdout + out.stderr


def test_a_caller_r_flag_is_refused(tree):
    out = _run(tree, "-rE", COMPASS_CHANGED_FILES=_changed(tree, "tests/test_ok.py"))
    assert out.returncode == 95, out.stdout + out.stderr


def test_another_checkouts_copy_is_refused(tree, tmp_path):
    other = _tree(tmp_path / "other")
    lst = _changed(tree, "tests/test_ok.py")
    out = subprocess.run(
        [BASH, str(other / "scripts" / "compass" / "gate_targeted.sh")],
        cwd=tree,
        env={**os.environ, "COMPASS_CHANGED_FILES": lst},
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert out.returncode == 99, out.stdout + out.stderr
    assert out.stdout.splitlines()[-1].startswith("GATE_TARGETED_RC=99 NOT PASSED")


def test_the_gpu_container_follows_gpu_gate_skip_for_the_first_cards_target(tree):
    (tree / "torch").mkdir()
    (tree / "torch" / "__init__.py").write_text(_torch("gfx942"))
    (tree / "scripts" / "compass" / "gpu_gate_skip.txt").write_text(
        "gfx950 tests/test_kernel.py\ngfx942 tests/test_ok.py\ngfx942 driver-run\n"
    )
    lst = _changed(tree, "tests/test_ok.py", "tests/test_kernel.py")
    env = {"ATOM_COMPASS_SLICE_MODEL": "/m"}
    out = _run(tree, COMPASS_CHANGED_FILES=lst, **env)
    assert out.returncode == 98, out.stdout + out.stderr
    assert "tests/test_ok.py: gpu_gate_skip.txt leaves it out on gfx942" in out.stderr
    assert out.stdout.splitlines()[-1].endswith(
        "not run here: tests/test_ok.py; see 'not run here' above"
    )
    # Its nodes hold no driver-run inputs, so a driver-run skip is accepted.
    out = _run(tree, COMPASS_CHANGED_FILES=_changed(tree, "tests/test_slice.py"), **env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "driver-run: skips accepted on gfx942" in out.stdout


SMOKE = "tests/compass/test_cctraces_sim_smoke.py"


def test_the_gpu_container_runs_the_smoke_alone_with_its_own_model(tree):
    (tree / "tests" / "compass").mkdir()
    (tree / SMOKE).write_text(
        "import os\n\n\ndef test_smoke():\n"
        "    assert os.environ['ATOM_COMPASS_SLICE_MODEL'] == '/cc'\n"
    )
    (tree / "tests" / "test_ok.py").write_text(
        "import os\n\n\ndef test_ok():\n"
        "    assert os.environ['ATOM_COMPASS_SLICE_MODEL'] == '/m'\n"
    )
    out = _run(
        tree,
        COMPASS_CHANGED_FILES=_changed(tree, SMOKE, "tests/test_ok.py"),
        ATOM_COMPASS_SLICE_MODEL="/m",
        ATOM_COMPASS_CCTRACES_MODEL="/cc",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("1 passed") == 2, out.stdout


def test_a_selected_file_holding_no_test_passes_saying_so(tree):
    (tree / "tests" / "test_empty.py").write_text("")
    out = _run(tree, COMPASS_CHANGED_FILES=_changed(tree, "tests/test_empty.py"))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no test collected from tests/test_empty.py" in out.stdout


# A module-level skip runs at collection, so a file the change did not select
# still prints its skip line.
@pytest.mark.parametrize(
    "selected, env",
    [
        ("tests/test_empty.py", {}),
        ("tests/test_ok.py", {"ATOM_COMPASS_SLICE_MODEL": "/m"}),
    ],
)
def test_only_a_selected_files_driver_run_skip_counts(tree, selected, env):
    (tree / "tests" / "test_empty.py").write_text("")
    (tree / "tests" / "test_module_skip.py").write_text(
        "import pytest\n\npytest.skip('needs ATOM_COMPASS_SLICE_MODEL', "
        "allow_module_level=True)\n"
    )
    out = _run(tree, COMPASS_CHANGED_FILES=_changed(tree, selected), **env)
    assert "SKIPPED [1] tests/test_module_skip.py" in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.splitlines()[-1] == "GATE_TARGETED_RC=0 PASSED"


@pytest.mark.parametrize("env", [{}, {"ATOM_COMPASS_SLICE_MODEL": "/m"}])
def test_a_selected_files_cargo_skip_refuses_with_96(tree, env):
    (tree / "tests" / "test_mesh.py").write_text(
        "import pytest\n\n\ndef test_router():\n"
        '    pytest.skip("cargo cannot resolve atomesh\'s dependencies here")\n'
    )
    lst = _changed(tree, "tests/test_mesh.py", "tests/test_ok.py")
    out = _run(tree, COMPASS_CHANGED_FILES=lst, **env)
    assert out.returncode == 96, out.stdout + out.stderr
    assert out.stdout.splitlines()[-1].startswith(
        "GATE_TARGETED_RC=96 NOT PASSED -- tests/test_mesh.py skipped, cargo"
    )


MESH = (
    "import pytest\n\n\n{}def build():\n"
    '    pytest.skip("cargo cannot resolve atomesh\'s dependencies here")\n\n\n'
)


# A selected test can reach the skip through a file the change did not select.
# A fixture's skip names the test that asked for it; a helper's names the line
# that raised it, inside the helper. Either way it counts.
@pytest.mark.parametrize(
    "fixture, named",
    [("@pytest.fixture\n", "tests/test_pd.py"), ("", "tests/test_mesh.py")],
)
def test_a_cargo_skip_through_another_file_refuses_with_96(tree, fixture, named):
    (tree / "tests" / "test_mesh.py").write_text(MESH.format(fixture))
    (tree / "tests" / "test_pd.py").write_text(
        "from test_mesh import build\n\n\n"
        + (
            "def test_pd(build):\n    pass\n"
            if fixture
            else "def test_pd():\n    build()\n"
        )
    )
    out = _run(tree, COMPASS_CHANGED_FILES=_changed(tree, "tests/test_pd.py"))
    assert out.returncode == 96, out.stdout + out.stderr
    assert out.stdout.splitlines()[-1].startswith(
        f"GATE_TARGETED_RC=96 NOT PASSED -- {named} skipped, cargo"
    )
