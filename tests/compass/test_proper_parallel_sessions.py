"""Owned fresh interpreters retain repeat identity under concurrent completion."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compass"))
spec = importlib.util.spec_from_file_location("proper_parallel", ROOT / "scripts/compass/cc_traces_proper.py")
paired = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paired
spec.loader.exec_module(paired)

CHILD = r'''
import json, os, pathlib, sys, time
from atom.utils.clock import VirtualClock, get_clock, set_clock
root, repeat, fail = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
directory = root / f'modelled.r{repeat}.raw'
directory.mkdir()
set_clock(VirtualClock(epoch=100. + repeat))
(directory/'startup_ready.json').write_text(json.dumps({'at':time.time(), 'server':{'pid':os.getpid()}}))
signal = directory/'start_profile.json'
while not signal.exists(): time.sleep(.01)
ack = json.loads(signal.read_text())
assert ack['pid'] == os.getpid()
if fail:
    if repeat == 2: raise SystemExit(7)
    while True: time.sleep(.01)
previous = {2:None, 3:2, 1:3}[repeat]
if previous is not None:
    while not (root/f'observed{previous}').exists(): time.sleep(.01)
(root/f'modelled.r{repeat}.json').write_text(json.dumps({
    'repeat':repeat, 'pid':os.getpid(), 'marker':os.environ['REPEAT_MARKER'],
    'tmpdir':os.environ['TMPDIR'], 'clock':get_clock().time(),
    'execution_id':ack['execution_id']}))
'''


def runner(tmp_path, monkeypatch, *, fail=False):
    run = paired.ProperSideRun.__new__(paired.ProperSideRun)
    run.cell, run.running, run.failures = tmp_path, {}, []
    run.wall, run.now, run.sleep = time.time, time.monotonic, time.sleep
    run.processes = paired.lifecycle.Processes()
    run.diagnostic_case = {"modelled_concurrency": 3, "plan": {}, "workload_sha256": "fixture-plan"}
    monkeypatch.setattr(paired.core, "read_pinned", lambda _: {"session_wall_timeout_seconds": 10})
    run._check_diagnostic_case = lambda _: True
    run._serve_env = lambda step: dict(os.environ, REPEAT_MARKER=str(step["repeat"]),
        TMPDIR=str(tmp_path/f"tmp{step['repeat']}"), PYTHONPATH=str(ROOT))
    run._log = lambda step: tmp_path/f"{step['id']}.log"
    executions, journal, completed = {}, [], []
    def mint(step, proc, started):
        value = {"repeat": step["repeat"], "execution_id": f"repeat-{step['repeat']}-{proc.pid}",
                 "process": {"pid": proc.pid, "launched_at": started, "ended_at":None, "exit":None}, "config": {}}
        executions[step["repeat"]] = value
        return value
    run._mint = mint
    run._check_server_process = lambda step, said, execution, proc: said["pid"] == proc.pid
    def record(step, **values):
        result = dict(step, **values)
        journal.append(result)
        return result
    run._record = record
    run._stamp_diagnostic_failure = lambda step, execution: None
    def check(step, entry, execution):
        row = json.loads((tmp_path/f"modelled.r{step['repeat']}.json").read_text())
        assert row["pid"] == execution["process"]["pid"]
        assert row["execution_id"] == execution["execution_id"]
        completed.append(row)
        (tmp_path/f"observed{step['repeat']}").touch()
        return True
    run._check_artifact = check
    run._write_execution = lambda value: (tmp_path/f"execution.modelled.r{value['repeat']}.json").write_text(json.dumps(value))
    sessions = [{"id": f"proper-modelled-{i}", "role": "proper_session", "side": "modelled", "repeat": i,
                 "command": [sys.executable, "-c", CHILD, str(tmp_path), str(i), str(int(fail))]}
                for i in range(1, 4)]
    group = {"id": "proper-modelled-group-1", "role": "proper_sessions", "side": "modelled", "sessions": sessions}
    return run, group, executions, journal, completed


def test_fresh_processes_finish_out_of_order_and_keep_per_repeat_costs(tmp_path, monkeypatch):
    run, group, executions, journal, completed = runner(tmp_path, monkeypatch)
    assert run._command(group)
    assert [row["repeat"] for row in completed] == [2, 3, 1]
    assert len({row["pid"] for row in completed}) == 3
    assert len({row["tmpdir"] for row in completed}) == 3
    for row in completed:
        repeat = row["repeat"]
        assert row["marker"] == str(repeat) and row["clock"] == 100. + repeat
        execution = executions[repeat]
        replay = execution["replay"]
        assert replay["seconds"] == replay["ended_at"] - replay["started_at"]
        assert replay["started_at"] == execution["process"]["launched_at"]
        entry = next(item for item in journal if item["role"] == "replay" and item["repeat"] == repeat)
        assert entry["seconds"] == replay["seconds"] > 0
    assert not run.running and not run.failures
    assert len(list(tmp_path.glob("execution.modelled.r*.json"))) == 3


def test_child_failure_fails_group_and_stops_owned_siblings(tmp_path, monkeypatch):
    run, group, executions, _, completed = runner(tmp_path, monkeypatch, fail=True)
    assert not run._command(group)
    assert run.failures and not run.running and not completed
    assert executions[2]["process"]["exit"] == 7
    for execution in executions.values():
        assert execution["process"]["ended_at"] is not None and execution["process"]["exit"] is not None
        with pytest.raises(ProcessLookupError):
            os.kill(execution["process"]["pid"], 0)


def test_launch_metadata_failure_also_stops_the_just_started_child(tmp_path, monkeypatch):
    run, group, executions, _, _ = runner(tmp_path, monkeypatch)
    processes = []
    start = run.processes.start
    def tracked_start(*args, **kwargs):
        process = start(*args, **kwargs)
        processes.append(process)
        return process
    run.processes.start = tracked_start
    mint = run._mint
    def failed_mint(step, proc, started):
        if step["repeat"] == 2:
            raise ValueError("source identity unavailable")
        return mint(step, proc, started)
    run._mint = failed_mint
    assert not run._command(group)
    assert len(processes) == 2 and not run.running
    assert any("source identity unavailable" in failure for failure in run.failures)
    assert all(process.poll() is not None for process in processes)
    assert executions[1]["process"]["ended_at"] is not None


@pytest.mark.parametrize("concurrency", [0, 4, True, 1.5])
def test_concurrency_is_bounded_before_any_process_launch(tmp_path, concurrency):
    path = tmp_path/"plan.json"
    path.write_text(json.dumps({"schema":"compass.aiperf_proper_pair/1", "purpose":"acceptance",
                               "repeats":3, "modelled_concurrency":concurrency}))
    with pytest.raises(ValueError, match="modelled_concurrency"):
        paired.load_case(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "aiperf_proper_fixture")


@pytest.mark.parametrize("name", ["TMPDIR", "AIPERF_DATASET_MMAP_BASE_PATH"])
def test_concurrent_repeats_refuse_shared_mutable_paths(tmp_path, name):
    environment = {key:str(tmp_path/"r{repeat}"/key) for key in ("TMPDIR", "AIPERF_DATASET_MMAP_BASE_PATH")}
    environment[name] = str(tmp_path/"shared")
    path = tmp_path/"plan.json"
    path.write_text(json.dumps({"schema":"compass.aiperf_proper_pair/1", "purpose":"acceptance",
        "repeats":3, "modelled_concurrency":3, "modelled_environment":environment}))
    with pytest.raises(ValueError, match="distinct absolute"):
        paired.load_case(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "aiperf_proper_fixture")


def test_grouping_preserves_native_lifecycles_and_distinct_repeat_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paired.core, "read_pinned", lambda _: {"native_engine_args": []})
    case = {"plan": {"path":"/frozen/plan.json", "sha256":"frozen"}, "case_id":"aiperf_proper_fixture",
            "repeats":3, "modelled_concurrency":3, "purpose":"acceptance",
            "workload":"/frozen/plan.json", "cache_policy":{}}
    built = paired.build_steps(case, tmp_path, port=8850, engine_port=8860, advisory=False)
    native = [step for step in built["steps"] if step["side"] == "real"]
    assert [step["repeat"] for step in native if step["role"] == "serve"] == [1, 2, 3]
    servers = [step["command"] for step in native if step["role"] == "serve"]
    assert [argv[argv.index("--compass-memory-out")+1] for argv in servers] == [
        str(tmp_path/f"real.r{repeat}_memory.json") for repeat in range(1, 4)]
    groups = [step for step in built["steps"] if step["role"] == "proper_sessions"]
    assert len(groups) == 1 and [step["repeat"] for step in groups[0]["sessions"]] == [1, 2, 3]
    outputs = [step["command"][step["command"].index("--out")+1] for step in groups[0]["sessions"]]
    assert outputs == [str(tmp_path/f"modelled.r{repeat}.json") for repeat in range(1, 4)]
