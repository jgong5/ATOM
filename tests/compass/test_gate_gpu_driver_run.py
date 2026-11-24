# SPDX-License-Identifier: MIT
"""`gate_gpu.sh` runs the driver-run tests, once, with their inputs.

The driver-run tests skip without the slice model, the image model and a router
built at the tree's commit, and a skip used to count as
accounted. On a tree whose tests read the slice model the gate now sets each
input, refuses by name when one is missing, runs tests/compass once on every card
named and the rest of the suite on the first card, and fails on a driver-run skip.

Each case runs the real script over a throwaway tree, as
`test_gate_gpu_verdict_through_pipe.py` does, with a stub router check.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_gate_gpu_verdict_through_pipe import (
    BASE,
    BASH,
    EXPECT,
    GATE,
    KNOWN,
    KNOWN_TXT,
    _preflight,
    _suite,
    _torch,
)

from tests.compass import driver_run_skips

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

RECORD = """
import os


def _record(what):
    with open(os.environ["RUNS"], "a") as f:
        f.write(f"{what} {os.environ.get('HIP_VISIBLE_DEVICES')} "
                f"{os.environ.get('ATOM_COMPASS_SLICE_MODEL')}\\n")
"""
DRIVER = (
    "import pytest\n"
    + RECORD
    + "\n\ndef test_slice():\n    # reads ATOM_COMPASS_SLICE_MODEL\n    _record('compass')\n"
)
# Records which compass_harness the harness Python imports.
SMOKE = (
    "import subprocess\n" + RECORD + "\n\ndef test_cctraces():\n"
    "    where = subprocess.run(\n"
    "        [os.environ['COMPASS_HARNESS_PYTHON'], '-c',\n"
    "         'import compass_harness; print(compass_harness.WHERE)'],\n"
    "        cwd='/', capture_output=True, text=True).stdout.strip()\n"
    "    _record(f'cctraces {where}')\n"
)
# Refuses a router whose path does not end in `atomesh`, as router_binary refuses
# one built at another commit, by raising with its reason.
ROUTER = (
    "def router_binary(path):\n"
    "    if not path.endswith('/atomesh'):\n"
    "        raise ValueError(f'router binary {path} names no commit')\n"
)


def _tree(tmp_path, driver=DRIVER, smoke=SMOKE):
    root = tmp_path / "ATOM"
    # Another gate shard may be writing bytecode into __pycache__ as this copies.
    shutil.copytree(
        REPO / "scripts" / "compass",
        root / "scripts" / "compass",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    main = _suite(EXPECT - 1, RECORD + "\n\ndef test_main():\n    _record('main')\n")
    files = {
        "scripts/compass/preflight.sh": _preflight(0),
        KNOWN_TXT: "\n".join(KNOWN) + "\n",
        "atom/__init__.py": "",
        "atom/compass/__init__.py": "",
        "atom/compass/run.py": ROUTER,
        "torch/__init__.py": _torch(),
        "tests/test_suite.py": main,
        "tests/compass/test_d.py": driver,
        "tests/compass/test_cctraces_sim_smoke.py": smoke,
        "compass_harness/compass_harness/__init__.py": "WHERE = 'tree'\n",
    }
    for rel, text in files.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "scripts" / "compass" / "preflight.sh").chmod(0o755)
    hub = tmp_path / "hf" / "hub"
    for name in ("Qwen3-0.6B", "Qwen3.5-9B", "Qwen3.5-0.8B", "Qwen3.8-27B"):
        (hub / f"models--Qwen--{name}" / "snapshots" / "abc").mkdir(parents=True)
    (tmp_path / "dspark").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "atomesh").write_text("")
    (tmp_path / "traces.jsonl").write_text("{}\n")
    return root


def _run(tmp_path, root, **env_over):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("COMPASS_", "ATOM_COMPASS_")) and k != "ATOMESH"
    }
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["ATOMESH"] = str(tmp_path / "bin" / "atomesh")
    env["ATOM_COMPASS_CCTRACES"] = str(tmp_path / "traces.jsonl")
    env["ATOM_COMPASS_DSPARK_MODEL"] = str(tmp_path / "dspark")
    env["COMPASS_HARNESS_PYTHON"] = env["COMPASS_AIPERF_PYTHON"] = sys.executable
    env["HIP_VISIBLE_DEVICES"] = "4,5,6,7"
    env["RUNS"] = str(root / "runs.txt")
    # Every compass file, the superset; the scope tests below pass their own.
    env["COMPASS_GPU_GATE_SCOPE"] = "all"
    env.update(env_over)
    out = subprocess.run(
        [BASH, "-c", f"{root / GATE} 2>&1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    return out, out.stdout.splitlines()[-1]


def test_driver_run_tests_run_once_on_every_card_with_their_inputs(tmp_path):
    root = _tree(tmp_path)
    out, verdict = _run(tmp_path, root)
    assert out.returncode == 0, out.stdout[-3000:]
    assert verdict == "GATE_GPU_RC=0 PASSED"
    hub = f"{tmp_path}/hf/hub/models--Qwen--"
    snap, big = f"{hub}Qwen3-0.6B/snapshots/abc/", f"{hub}Qwen3.8-27B/snapshots/abc/"
    runs = (root / "runs.txt").read_text().splitlines()
    assert sorted(runs) == sorted(
        [
            f"compass 4,5,6,7 {snap}",
            f"cctraces tree 4,5,6,7 {big}",
            f"main 4 {snap}",
        ]
    )


def test_the_speculative_models_reach_the_tests(tmp_path):
    spec = "' '.join(os.environ[f'ATOM_COMPASS_{m}_MODEL'] for m in ('MTP', 'DSPARK'))"
    root = _tree(tmp_path, DRIVER.replace("'compass'", spec))
    out, verdict = _run(tmp_path, root)
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    mtp = f"{tmp_path}/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/abc/"
    assert f"{mtp} {tmp_path}/dspark 4,5,6,7 " in (root / "runs.txt").read_text()


@pytest.mark.parametrize(
    "env, named",
    [
        ({"HF_HOME": "/nonexistent"}, "ATOM_COMPASS_SLICE_MODEL, a Qwen3-0.6B"),
        ({"ATOM_COMPASS_IMAGE_MODEL": "/nonexistent"}, "ATOM_COMPASS_IMAGE_MODEL"),
        (
            {"ATOM_COMPASS_MTP_MODEL": "/nonexistent"},
            "ATOM_COMPASS_MTP_MODEL, a Qwen3.5-0.8B",
        ),
        (
            {"ATOM_COMPASS_DSPARK_MODEL": "/nonexistent"},
            "ATOM_COMPASS_DSPARK_MODEL /nonexistent",
        ),
        (
            {"ATOM_COMPASS_CCTRACES_MODEL": "/nonexistent"},
            "ATOM_COMPASS_CCTRACES_MODEL, a Qwen3.8-27B",
        ),
        (
            {"ATOM_COMPASS_CCTRACES": "/nonexistent"},
            "ATOM_COMPASS_CCTRACES /nonexistent",
        ),
        (
            {"COMPASS_HARNESS_PYTHON": "/nonexistent"},
            "COMPASS_HARNESS_PYTHON /nonexistent",
        ),
        (
            {"COMPASS_AIPERF_PYTHON": "/nonexistent"},
            "COMPASS_AIPERF_PYTHON /nonexistent",
        ),
        (
            {"ATOMESH": "/x/atomesh-old"},
            "ATOMESH, ValueError: router binary /x/atomesh-old",
        ),
        (
            {"HIP_VISIBLE_DEVICES": "4,5"},
            "four cards in HIP_VISIBLE_DEVICES, which names 4,5",
        ),
    ],
)
def test_a_missing_input_refuses_by_name_before_any_test(tmp_path, env, named):
    root = _tree(tmp_path)
    out, verdict = _run(tmp_path, root, **env)
    assert out.returncode == 92, out.stdout[-3000:]
    assert verdict.startswith("GATE_GPU_RC=92 NOT PASSED -- the driver-run tests lack ")
    assert named in verdict, verdict
    assert not (root / "runs.txt").exists()


@pytest.mark.parametrize(
    "which, reason, where",
    [
        ("driver", "needs ATOMESH, atomesh", "test_d.py"),
        ("smoke", "needs COMPASS_AIPERF_PYTHON", "test_cctraces_sim_smoke.py"),
        ("driver", "needs ATOM_COMPASS_MTP_MODEL naming a model", "test_d.py"),
        ("driver", "needs ATOM_COMPASS_DSPARK_MODEL naming a model", "test_d.py"),
    ],
)
def test_a_driver_run_skip_fails_by_name(tmp_path, which, reason, where):
    files = {"driver": DRIVER, "smoke": SMOKE}
    files[which] = files[which].replace(
        "def test_",
        f"import pytest\n\n\n@pytest.mark.skip(reason='{reason}')\ndef test_",
    )
    root = _tree(tmp_path, **files)
    out, verdict = _run(tmp_path, root)
    assert out.returncode == 1, out.stdout[-3000:]
    assert (
        f"-- 1 driver-run skip line(s), first SKIPPED [1] tests/compass/{where}"
        in verdict
    )
    assert reason in verdict


def test_a_cargo_skip_refuses_with_96_before_the_suite(tmp_path):
    reason = "cargo cannot resolve atomesh's dependencies here"
    skip = f"import pytest\n\n\n@pytest.mark.skip(reason={reason!r})\ndef test_"
    root = _tree(tmp_path, DRIVER.replace("def test_", skip))
    out, verdict = _run(tmp_path, root)
    assert out.returncode == 96, out.stdout[-3000:]
    assert verdict.startswith("GATE_GPU_RC=96 NOT PASSED -- a test that cargo-builds")
    assert "tests/compass/test_d.py" in verdict
    assert "main" not in (root / "runs.txt").read_text()


# A tests/compass run that collected nothing (pytest rc 5) completed and passes
# none; one interrupted by a collection error (rc 2) did not complete. rc 1, tests
# failed, is the pipe test's `compass-failure` case.
@pytest.mark.parametrize(
    "driver, rc, verdict_is",
    [
        ("# ATOM_COMPASS_SLICE_MODEL\n", 0, "GATE_GPU_RC=0 PASSED"),
        (
            "# ATOM_COMPASS_SLICE_MODEL\nimport no_such_module\n",
            1,
            "GATE_GPU_RC=1 NOT PASSED -- pytest rc=2 over tests/compass; the run did not complete",
        ),
    ],
)
def test_which_tests_compass_exit_codes_are_a_completed_run(
    tmp_path, driver, rc, verdict_is
):
    root = _tree(tmp_path, driver, smoke=None)
    out, verdict = _run(tmp_path, root)
    assert out.returncode == rc, out.stdout[-3000:]
    assert verdict == verdict_is


#: A compass test no driver-run input gates, which the CPU gate runs.
CPU = RECORD + "\n\ndef test_cpu():\n    _record('cpu')\n"
LIST = "scripts/compass/gpu_gate_driver_run.txt"


def _scoped(tmp_path, *listed):
    root = _tree(tmp_path)
    (root / "tests/compass/test_cpu.py").write_text(CPU)
    (root / LIST).write_text("# a comment\n" + "".join(f"{f}\n" for f in listed))
    return root


# An empty COMPASS_GPU_GATE_SCOPE reads as unset. The list names a file this tree
# lacks, which batch drops, and the smoke, which runs apart and only there.
@pytest.mark.parametrize(
    "scope, runs",
    [
        ("", ["cctraces tree", "compass", "main"]),
        ("batch", ["cctraces tree", "compass", "main"]),
        ("all", ["cctraces tree", "compass", "cpu", "main"]),
    ],
)
def test_batch_runs_only_the_listed_compass_files_and_all_runs_every_one(
    tmp_path, scope, runs
):
    root = _scoped(
        tmp_path,
        "tests/compass/test_d.py",
        "tests/compass/test_gone.py",
        "tests/compass/test_cctraces_sim_smoke.py",
    )
    out, verdict = _run(tmp_path, root, COMPASS_GPU_GATE_SCOPE=scope)
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    ran = (root / "runs.txt").read_text().splitlines()
    assert sorted(line.rsplit(" ", 2)[0] for line in ran) == runs
    # The expectation counts what the scope ran.
    base = BASE["BASE_COMPASS_TESTS"]
    counted = f"+ ({len(runs) - 1} - {base}) tests/compass, {scope or 'batch'} -"
    assert counted in out.stdout


def test_batch_drops_a_listed_file_left_out_on_the_target(tmp_path):
    root = _scoped(tmp_path, "tests/compass/test_d.py", "tests/compass/test_cpu.py")
    (root / "scripts/compass/gpu_gate_skip.txt").write_text(
        "gfx950 tests/compass/test_cpu.py\n"
    )
    out, verdict = _run(tmp_path, root, COMPASS_GPU_GATE_SCOPE="batch")
    assert verdict == "GATE_GPU_RC=0 PASSED", out.stdout[-3000:]
    assert "cpu" not in (root / "runs.txt").read_text().split()


@pytest.mark.parametrize(
    "scope, drop, rc, why",
    [
        ("some", False, 95, "COMPASS_GPU_GATE_SCOPE=some is neither batch nor all"),
        ("batch", True, 94, f"{LIST}; nothing was run"),
    ],
)
def test_an_unknown_scope_or_an_unreadable_list_refuses_before_any_test(
    tmp_path, scope, drop, rc, why
):
    root = _scoped(tmp_path, "tests/compass/test_d.py")
    if drop:
        (root / LIST).unlink()
    out, verdict = _run(tmp_path, root, COMPASS_GPU_GATE_SCOPE=scope)
    assert out.returncode == rc, out.stdout[-3000:]
    assert why in verdict, verdict
    assert not (root / "runs.txt").exists()


def test_every_file_the_repositorys_list_names_is_a_compass_test_file():
    lines = (REPO / LIST).read_text().splitlines()
    listed = [line for line in lines if line.strip() and not line.startswith("#")]
    assert listed
    assert [f for f in listed if not f.startswith("tests/compass/test_")] == []
    assert [f for f in listed if not (REPO / f).is_file()] == []


def test_the_suites_conftest_loads_the_driver_run_hooks(request):
    hook = request.config.pluginmanager.hook
    for name in ("pytest_runtest_makereport", "pytest_make_collect_report"):
        impls = getattr(hook, name).get_hookimpls()
        assert getattr(driver_run_skips, name) in [i.function for i in impls]


def test_the_gates_and_the_hooks_take_a_skip_for_driver_run_by_one_pattern():
    shell = subprocess.run(
        [BASH, "-c", '. scripts/compass/_lib.sh; printf %s "$COMPASS_DRIVER_RUN_SKIP"'],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert shell == f"^SKIPPED .*({driver_run_skips.DRIVER_RUN.pattern})"


# The hooks, loaded as tests/conftest.py loads them, over a throwaway tree.
def _pytest(tmp_path, listed, text):
    root = tmp_path / "ATOM"
    (root / "tests/compass").mkdir(parents=True)
    shutil.copy(REPO / "tests/compass/driver_run_skips.py", root / "tests/compass")
    (root / "tests/conftest.py").write_text(
        "from tests.compass.driver_run_skips import (  # noqa: F401\n"
        "    pytest_make_collect_report,\n    pytest_runtest_makereport,\n)\n"
    )
    (root / "tests/compass/test_a.py").write_text(text)
    (root / "scripts/compass").mkdir(parents=True)
    (root / LIST).write_text("tests/compass/test_a.py\n" if listed else "")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/compass",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    ).stdout


SKIP = "import pytest\n\n\n@pytest.mark.skip(reason={!r})\ndef test_a():\n    pass\n"


# A skip turned into a failure at setup or collection reads as an error, one in
# the test's body as a failure.
@pytest.mark.parametrize(
    "listed, text, summary",
    [
        (False, SKIP.format("needs ATOM_COMPASS_SLICE_MODEL"), "1 error"),
        (True, SKIP.format("needs ATOM_COMPASS_SLICE_MODEL"), "1 skipped"),
        (False, SKIP.format("a run needs atomesh on PATH"), "1 error"),
        (False, SKIP.format("needs COMPASS_HARNESS_PYTHON"), "1 error"),
        (
            False,
            (
                "import pytest\n\n\ndef test_a():\n"
                "    pytest.skip('needs ATOM_COMPASS_MTP_MODEL')\n"
            ),
            "1 failed",
        ),
        (False, SKIP.format("needs bash"), "1 skipped"),
        # The cargo probe's skip names no input: the CPU gate may lack the crates.
        (
            False,
            SKIP.format("cargo cannot resolve atomesh's dependencies"),
            "1 skipped",
        ),
        (
            False,
            (
                "import pytest\n\npytest.skip('a 1P1D cell needs atomesh', "
                "allow_module_level=True)\n"
            ),
            "1 error",
        ),
        (
            False,
            (
                "import pytest\n\n\n@pytest.mark.xfail(reason='ATOM_COMPASS_SLICE_MODEL')\n"
                "def test_a():\n    assert False\n"
            ),
            "1 xfailed",
        ),
    ],
)
def test_a_driver_run_skip_outside_the_list_fails_the_cpu_gate(
    tmp_path, listed, text, summary
):
    out = _pytest(tmp_path, listed, text)
    assert f"{summary} in" in out, out[-3000:]
    if summary in ("1 error", "1 failed"):
        assert (
            f"tests/compass/test_a.py skipped for want of a driver-run input, but {LIST} does not list it"
            in out
        )
