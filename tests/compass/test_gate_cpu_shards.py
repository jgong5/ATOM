# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` runs tests/compass as COMPASS_GATE_SHARDS pytest processes
beside the rest of tests/: every file once, its failures still fail the gate.

Each case runs the real script over the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py`` plus a tests/compass whose tests
record the process that ran them. The stamped diff touches a GPU trigger, so a
green run exits 98 after pytest. Nothing needs a driver.
"""

import pytest
from test_gate_cpu_verdict_through_pipe import _run, _tree

TEST = """import os, pathlib
def test_{name}():
    (pathlib.Path({ran!r}) / ("{name}." + str(os.getpid()))).touch()
    assert {ok}
"""
COMPASS = ["test_c0", "test_c1", "test_c2", "test_c3", "deep/test_c4"]


def _gate(tmp_path, shards, failing=(), excluded=()):
    root = _tree(tmp_path / "ATOM", stamped=True)
    ran = root / "ran"
    ran.mkdir()
    (root / "tests" / "compass" / "deep").mkdir(parents=True)
    tests = {"test_ok.py": "ok"}
    tests |= {f"compass/{name}.py": name for name in COMPASS + list(excluded)}
    for path, name in tests.items():
        name = name.split("/")[-1].removeprefix("test_")
        test = TEST.format(name=name, ok=name not in failing, ran=str(ran))
        (root / "tests" / path).write_text(test)
    (root / "scripts" / "compass" / "cpu_gate_exclude.txt").write_text(
        "".join(f"tests/compass/{name}.py\n" for name in excluded)
    )
    out = _run(root, f"COMPASS_GATE_SHARDS={shards} {{gate}}")
    pids = {}
    for marker in ran.iterdir():
        name, pid = marker.name.split(".")
        pids.setdefault(name, []).append(pid)
    return out, pids


@pytest.mark.parametrize("shards", [0, 1, 2, 8])
def test_every_file_runs_once_with_compass_in_its_own_processes(tmp_path, shards):
    out, pids = _gate(tmp_path, shards, excluded=["test_x"])
    assert out.returncode == 98, out.stdout + out.stderr
    assert "pytest: rc=0" in out.stdout, out.stdout
    assert sorted(pids) == ["c0", "c1", "c2", "c3", "c4", "ok"], pids
    assert all(len(p) == 1 for p in pids.values()), pids
    compass = {p[0] for name, p in pids.items() if name != "ok"}
    if shards:
        assert len(compass) == min(shards, len(COMPASS)), pids
        assert pids["ok"][0] not in compass, pids
    else:
        assert compass == set(pids["ok"]), pids


def test_a_failing_compass_test_fails_the_gate_by_name(tmp_path):
    out, pids = _gate(tmp_path, 2, failing=["c3"])
    assert out.returncode == 1, out.stdout + out.stderr
    assert "pytest: rc=1" in out.stdout, out.stdout
    assert "FAILED tests/compass/test_c3.py::test_c3" in out.stdout, out.stdout
    assert out.stdout.splitlines()[-1].startswith("GATE_CPU_RC=1 NOT PASSED")
    assert len(pids) == 6, pids


def test_a_shard_count_that_is_not_a_number_is_refused(tmp_path):
    out, pids = _gate(tmp_path, "four")
    assert out.returncode == 95, out.stdout + out.stderr
    assert "COMPASS_GATE_SHARDS=four is not a count" in out.stdout.splitlines()[-1]
    assert pids == {}
