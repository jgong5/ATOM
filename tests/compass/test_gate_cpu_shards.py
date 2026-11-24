# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` runs tests/compass as COMPASS_GATE_SHARDS pytest processes
and the rest of tests/ as COMPASS_GATE_REST_SHARDS more, each set packed by
cpu_gate_durations.txt: every file once, its failures still fail the gate.

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


def _gate(
    tmp_path,
    shards,
    failing=(),
    excluded=(),
    args="",
    rest=(),
    rest_shards=None,
    table=None,
):
    root = _tree(tmp_path / "ATOM", stamped=True)
    ran = root / "ran"
    ran.mkdir()
    (root / "tests" / "compass" / "deep").mkdir(parents=True)
    tests = {"test_ok.py": "ok"}
    tests |= {f"compass/{name}.py": name for name in COMPASS + list(excluded)}
    tests |= {f"{name}.py": name for name in rest}
    for path, name in tests.items():
        (root / "tests" / path).parent.mkdir(parents=True, exist_ok=True)
        name = name.split("/")[-1].removeprefix("test_")
        head = HEADS.get(name, "")
        test = TEST.format(name=name, head=head, ok=name not in failing, ran=str(ran))
        (root / "tests" / path).write_text(test)
    (root / "scripts" / "compass" / "cpu_gate_exclude.txt").write_text(
        "".join(f"tests/compass/{name}.py\n" for name in excluded)
    )
    if table is not None:
        (root / "scripts" / "compass" / "cpu_gate_durations.txt").write_text(table)
    env = f"COMPASS_GATE_SHARDS={shards}"
    if rest_shards is not None:
        env += f" COMPASS_GATE_REST_SHARDS={rest_shards}"
    out = _run(root, f"{env} {{gate}} {args}")
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


REST = ["test_r1", "test_r2", "sub/test_r3"]


@pytest.mark.parametrize("rest_shards", [None, 1, 2, 4])
def test_every_rest_file_runs_once_apart_from_every_compass_file(tmp_path, rest_shards):
    out, pids = _gate(tmp_path, 2, rest=REST, rest_shards=rest_shards)
    assert out.returncode == 98, out.stdout + out.stderr
    assert "pytest: rc=0" in out.stdout, out.stdout
    assert sorted(pids) == ["c0", "c1", "c2", "c3", "c4", "ok", "r1", "r2", "r3"]
    assert all(len(p) == 1 for p in pids.values()), pids
    rest = {pids[name][0] for name in ("ok", "r1", "r2", "r3")}
    compass = {pids[f"c{k}"][0] for k in range(5)}
    assert not rest & compass, pids
    assert len(rest) == min(rest_shards or 4, 4), pids


def test_processes_are_packed_by_the_duration_table(tmp_path):
    # Round-robin would put c0 beside c2 and c4, and ok beside r2.
    table = "# seconds\n100.0 tests/compass/test_c0.py\n50.0 tests/test_ok.py\n"
    table += "".join(f"1.0 tests/{name}.py\n" for name in ["compass/test_c1"] + REST)
    out, pids = _gate(tmp_path, 2, rest=REST, rest_shards=2, table=table)
    assert out.returncode == 98, out.stdout + out.stderr
    by_pid = {}
    for name, (pid,) in pids.items():
        by_pid.setdefault(pid, set()).add(name)
    groups = sorted(map(sorted, by_pid.values()))
    assert groups == [["c0"], ["c1", "c2", "c3", "c4"], ["ok"], ["r1", "r2", "r3"]]


@pytest.mark.parametrize("count", ["0", "four"])
def test_a_rest_shard_count_that_is_not_a_count_is_refused(tmp_path, count):
    out, pids = _gate(tmp_path, 2, rest_shards=count)
    assert out.returncode == 95, out.stdout + out.stderr
    last = out.stdout.splitlines()[-1]
    assert f"COMPASS_GATE_REST_SHARDS={count} is not a count" in last
    assert pids == {}


def test_regen_writes_each_test_file_s_seconds_from_a_green_run(tmp_path):
    root = _tree(tmp_path / "ATOM", stamped=True)
    slow = "import time\ndef test_slow():\n    time.sleep(0.3)\n"
    (root / "tests" / "compass").mkdir()
    (root / "tests" / "compass" / "test_slow.py").write_text(slow)
    out = _run(root, "scripts/compass/regen_cpu_gate_durations.sh")
    assert out.returncode == 0, out.stdout + out.stderr
    table = (root / "scripts" / "compass" / "cpu_gate_durations.txt").read_text()
    rows = [line.split() for line in table.splitlines() if not line.startswith("#")]
    assert [path for _, path in rows] == ["tests/compass/test_slow.py"], table
    assert float(rows[0][0]) >= 0.3, table
