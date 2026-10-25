# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` runs tests/compass as COMPASS_GATE_SHARDS pytest processes
beside the rest of tests/: every file once, its failures still fail the gate.

Each case runs the real script over the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py`` plus a tests/compass whose tests
record the process that ran them and its basetemp. `test_c0.py` plants a
module that `test_c3.py` and the root `test_ok.py` import, as tests/compass's
aiter stub leaves engine modules imported for the files collected after it.
The stamped diff touches a GPU trigger, so a green run exits 98 after pytest.
Nothing needs a driver.
"""

import xml.etree.ElementTree as ET

import pytest
from test_gate_cpu_verdict_through_pipe import _run, _tree

TEST = """import os, pathlib, sys
{head}
def test_{name}(tmp_path):
    p = pathlib.Path({ran!r}) / ("{name}." + str(os.getpid()))
    p.write_text(str(tmp_path.parent))
    assert {ok}
"""
COMPASS = ["test_c0", "test_c1", "test_c2", "test_c3", "deep/test_c4"]
HEADS = {
    "c0": "sys.modules['planted'] = sys",
    "c3": "import planted",
    "ok": "import planted",
}


def _gate(tmp_path, shards, failing=(), excluded=(), args=""):
    root = _tree(tmp_path / "ATOM", stamped=True)
    ran = root / "ran"
    ran.mkdir()
    (root / "tests" / "compass" / "deep").mkdir(parents=True)
    tests = {"test_ok.py": "ok"}
    tests |= {f"compass/{name}.py": name for name in COMPASS + list(excluded)}
    for path, name in tests.items():
        name = name.split("/")[-1].removeprefix("test_")
        head = HEADS.get(name, "")
        test = TEST.format(name=name, head=head, ok=name not in failing, ran=str(ran))
        (root / "tests" / path).write_text(test)
    (root / "scripts" / "compass" / "cpu_gate_exclude.txt").write_text(
        "".join(f"tests/compass/{name}.py\n" for name in excluded)
    )
    out = _run(root, f"COMPASS_GATE_SHARDS={shards} {{gate}} {args}")
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


@pytest.mark.parametrize("form", ["--junitxml={}", "--junit-xml {}"])
def test_each_process_writes_its_own_junit_and_basetemp(tmp_path, form):
    junit = tmp_path / "junit.xml"
    args = form.format(junit) + f" --basetemp={tmp_path / 'bt'}"
    out, pids = _gate(tmp_path, 2, args=args)
    assert out.returncode == 98, out.stdout + out.stderr
    files = sorted(tmp_path.glob("junit.xml*"))
    assert [f.name for f in files] == [
        "junit.xml",
        "junit.xml.shard1",
        "junit.xml.shard2",
    ]
    cases = [c.get("name") for f in files for c in ET.parse(f).iter("testcase")]
    assert sorted(cases) == ["test_" + name for name in sorted(pids)], cases
    ran = tmp_path / "ATOM" / "ran"
    temps = {m.name.split(".")[1]: m.read_text() for m in ran.iterdir()}
    assert len(set(temps.values())) == len(temps) == 3, temps


def test_a_gate_where_no_process_ran_a_test_fails(tmp_path):
    out, pids = _gate(tmp_path, 2, args="-k no_such_test")
    assert out.returncode == 5, out.stdout + out.stderr
    assert "pytest: rc=5" in out.stdout, out.stdout
    assert out.stdout.splitlines()[-1].startswith("GATE_CPU_RC=5 NOT PASSED")
    assert pids == {}
