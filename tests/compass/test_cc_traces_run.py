"""What the run harness must refuse, checked without a device or a server.

The processes, the health probe, the provenance endpoint and the clock are all
injected, so these tests are about sequencing and refusal rather than about
`subprocess`. Each one stands for a way a run has looked finished and meant
nothing: a second repeat against a warmed server, an artifact with no drained
preparation, a server that died mid-replay, a costs file assembled out of terms
nobody measured.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_mod = _load("cc_traces_run")
plan_mod = _load("cc_traces_plan")


# --------------------------------------------------------------------------
# the fakes


class FakeProc:
    def __init__(self, pid, command):
        self.pid = pid
        self.command = list(command)
        self.returncode = None
        self.signalled = []

    def poll(self):
        return self.returncode


#: What `/compass/provenance` actually returns, field for field, from
#: `atom/entrypoints/openai/api_server.py::compass_provenance`. A fake thinner
#: than the producer lets the harness pass a check the real server would fail.
SERVER_CODE = "a" * 64


def fake_provenance(mode, *, tp=2, code=SERVER_CODE, **over):
    said = {
        "server_revision": "deadbeef",
        "server_code_sha256": code,
        "model": plan_mod.MODEL,
        "model_revision": "m-rev",
        "tensor_parallel_size": tp,
        "max_model_len": 262144,
        "enable_prefix_caching": False,
        "compass": {
            "enabled": True,
            "mode": mode,
            "oracle": "atom.compass.oracles.transfer.TransferOracle",
            "oracle_options": {},
            "oracle_option_sha256": {},
            "virtual_clock": mode == "predict",
            "admission_seconds": None,
        },
        "calibration_sha256": None,
        "visible_devices": None if mode == "predict" else "0",
    }
    said.update(over)
    return said


HOST = "test-host"
BOOT = "11111111-2222-3333-4444-555555555555"


def fake_process_identity(pid, *, host=HOST, boot=BOOT, ticks=None, ppid=1):
    """What `atom.compass.core.process_identity.identity()` returns.

    Start ticks default to a function of the pid so that no two processes in a
    test share one -- which is what the real reading gives, and exactly what
    pid reuse violates.
    """
    return {
        "pid": pid,
        "ppid": ppid,
        "host": host,
        "boot_id": boot,
        "start_ticks": 900_000 + pid if ticks is None else ticks,
        "ticks_per_second": 100,
    }


class FakeProbe:
    """A `/proc` that knows only the processes this test started.

    The harness reads through this rather than calling the module directly, so
    a test can describe a machine -- a pid that no longer exists, a pid reused
    by something else, a forked descendant -- without starting anything.
    """

    def __init__(self, processes, *, boot=BOOT, ticks=None, parents=None, missing=()):
        self.processes = processes
        self.boot = boot
        self.ticks = dict(ticks or {})
        self.parents = dict(parents or {})
        self.missing = set(missing)

    def _known(self, pid):
        return pid in {p.pid for p in self.processes.started} | set(self.ticks)

    def start_ticks(self, pid):
        if pid in self.missing:
            return None
        if pid in self.ticks:
            return self.ticks[pid]
        return 900_000 + pid if self._known(pid) else None

    def parent_of(self, pid):
        parent = self.parents.get(pid)
        # A callable lets a test name a parent that does not exist until the
        # harness has actually launched something.
        return parent(self.processes) if callable(parent) else parent

    def boot_id(self):
        return self.boot


def _serving(processes):
    """The pid of the most recent server -- not the sampler, which outlives it."""
    for proc in reversed(processes.started):
        text = " ".join(proc.command)
        if "api_server" in text or "replay_server.py" in text:
            return proc.pid
    return None


def _with_process(said, processes):
    """Say who answered, unless the test is making a point about that.

    A test that wants to describe a server which will not identify itself
    passes `server_process` explicitly; anything else gets the process the
    harness actually launched, which is the honest default.
    """
    if isinstance(said, dict) and "server_process" not in said:
        return dict(said, server_process=fake_process_identity(_serving(processes)))
    return said


class FakeProcesses:
    """Every start, run and stop, in order, and nothing that looks anything up.

    `stop` takes the handle it was given, so a test can assert that the harness
    only ever signalled processes it started -- by the pid it recorded, never
    by a name. What it writes for a replay is `replay.py`'s manifest shape and
    what it writes for the sampler is `gpu_sampler.py`'s, so a check that
    passes here is a check that could pass there.
    """

    def __init__(
        self,
        *,
        exits=None,
        dies_after=None,
        cell=None,
        artifacts=True,
        wall=None,
        served=SERVER_CODE,
        sampler=True,
    ):
        self.started = []
        self.ran = []
        self.stopped = []
        self.exits = dict(exits or {})
        self.dies_after = dies_after
        self.cell = cell
        self.artifacts = artifacts
        self.served = served
        self.sampler = sampler
        self.wall = wall or (lambda: 1_700_000_000.0)
        self._pid = 4000

    def start(self, command, *, log, cwd=None, env=None):
        self._pid += 1
        proc = FakeProc(self._pid, command)
        self.started.append(proc)
        return proc

    def run(self, command, *, log, cwd=None, env=None):
        self.ran.append(list(command))
        key = self._key(command)
        text = " ".join(command)
        if self.cell is not None and "replay.py" in text:
            self._write_artifact(command)
        if self.cell is not None and "gpu_sampler.py" in text:
            self._gpu_sample(command, "baseline")
        if self.dies_after == key and self.started:
            self.started[-1].returncode = 1
        return self.exits.get(key, 0)

    def alive(self, proc):
        return proc.returncode is None

    def stop(self, proc, grace=None):
        self.stopped.append(proc.pid)
        if self.cell is not None and "gpu_sampler.py" in " ".join(proc.command):
            self._gpu_sample(proc.command, "window")
        if proc.returncode is None:
            proc.returncode = 0
        return proc.returncode

    @staticmethod
    def _key(command):
        text = " ".join(command)
        if "--out" in command:
            return Path(command[command.index("--out") + 1]).name
        return text

    def _gpu_sample(self, command, phase):
        if not self.sampler:
            return
        # The path the sampler was told to write, the way the real one does.
        path = Path(next(a for a in command if a.endswith("gpu.jsonl")))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "t": self.wall(),
                        "phase": phase,
                        "visible": "0",
                        "own_pids": [],
                        "smi": {"card0": {"VRAM Total Used Memory (B)": "0"}},
                        "pids": {},
                    }
                )
                + "\n"
            )

    def _write_artifact(self, command):
        out = Path(command[command.index("--out") + 1])
        if not self.artifacts:
            return
        side = "modelled" if "modelled" in out.name else "real"
        trace = Path(command[command.index("--trace") + 1])
        full = ROOT / trace
        manifest = {
            "paced": side == "real",
            "trace": str(trace),
            "trace_sha256": (run_mod.file_digest(full) or {}).get("sha256"),
            "requests": 0,
            "server": fake_provenance(
                "measure" if side == "real" else "predict", code=self.served
            ),
            "server_code_sha256": self.served,
            "prepare": None,
        }
        if side == "real":
            manifest["prepare"] = {
                "requested": 3,
                "returned": 3,
                "drained": True,
                "drained_records": 3,
                "store_empty_after_drain": True,
                "declared_workload_size": False,
            }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"run": manifest, "workload": [], "results": []}))


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def wall(self):
        return 1_700_000_000.0 + self.t

    def sleep(self, seconds):
        self.t += seconds


def _plan(tmp_path, tp=2, klass="long"):
    built = plan_mod.cell_steps(
        tp,
        klass,
        root=str(tmp_path),
        oracle="transfer",
        options=(),
        port=8000,
        repeats=3,
        target="/w/target.json",
    )
    return built


def _runner(
    tmp_path, side, *, processes=None, health=None, provenance=None, probe=None, **kw
):
    clock = Clock()
    mode = "predict" if side == "modelled" else "measure"
    procs = processes or FakeProcesses(cell=tmp_path, wall=clock.wall)
    said = provenance or (lambda url: fake_provenance(mode))
    return run_mod.SideRun(
        _plan(tmp_path),
        side,
        processes=procs,
        health=health or (lambda url: {}),
        provenance=lambda url: _with_process(said(url), procs),
        probe=probe or FakeProbe(procs),
        host=HOST,
        now=clock.now,
        wall=clock.wall,
        sleep=clock.sleep,
        **kw,
    )


def _journal(runner):
    return json.loads((runner.cell / f"run.{runner.side}.json").read_text())


# --------------------------------------------------------------------------


class TestARepeatIsItsOwnProcess:
    def test_three_servers_are_started_and_each_is_stopped(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0
        procs = runner.processes
        assert len(procs.started) == 3
        assert procs.stopped == [p.pid for p in procs.started]

    def test_a_replay_runs_against_the_server_its_own_repeat_started(self, tmp_path):
        """Not the previous repeat's: a predicting server's virtual epoch is
        fixed when it starts, so the second replay would be stamped from an
        origin the first one already moved."""
        runner = _runner(tmp_path, "modelled")
        runner.run()
        order = [e["id"] for e in _journal(runner)["steps"]]
        assert order == [
            "serve-modelled-1",
            "replay-modelled-1",
            "stop-modelled-1",
            "serve-modelled-2",
            "replay-modelled-2",
            "stop-modelled-2",
            "serve-modelled-3",
            "replay-modelled-3",
            "stop-modelled-3",
            "gpu-free",
        ]

    def test_it_refuses_to_start_a_repeat_while_one_is_still_up(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        steps = [s for s in runner.plan["steps"] if s.get("side") == "modelled"]
        runner.plan["steps"] = [s for s in steps if s["id"] != "stop-modelled-1"]
        assert runner.run() == 1
        assert any("fresh process" in f for f in runner.failures)

    def test_only_processes_it_started_are_ever_signalled(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        started = {p.pid for p in runner.processes.started}
        assert set(runner.processes.stopped) <= started
        assert all(
            e.get("pid") in started
            for e in _journal(runner)["steps"]
            if e["role"] == "stop" and e.get("pid") is not None
        )


class TestTheRealSideIsWatchedAndPrepared:
    def test_the_baseline_sample_precedes_the_first_server(self, tmp_path):
        runner = _runner(tmp_path, "real")
        assert runner.run() == 0
        ids = [e["id"] for e in _journal(runner)["steps"]]
        assert ids.index("sample-baseline") < ids.index("serve-real-1")
        assert ids.index("sample") < ids.index("serve-real-1")

    def test_the_sampler_is_stopped_only_after_the_last_repeat(self, tmp_path):
        runner = _runner(tmp_path, "real")
        runner.run()
        ids = [e["id"] for e in _journal(runner)["steps"]]
        assert ids.index("stop-real-3") < ids.index("stop-sample")

    def test_an_undrained_real_artifact_fails_the_repeat(self, tmp_path):
        """The drain is where the measurement starts; without it the first
        requests are answered by an engine that is still waking up."""
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def undrained(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"]["prepare"] = {"requests": 3, "drained": False}
            out.write_text(json.dumps(blob))

        procs._write_artifact = undrained
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("drained preparation" in f for f in runner.failures)

    def test_an_unpaced_real_artifact_fails_the_repeat(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def unpaced(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"]["paced"] = False
            out.write_text(json.dumps(blob))

        procs._write_artifact = unpaced
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("not paced" in f for f in runner.failures)

    def test_a_prepared_modelled_artifact_fails_the_repeat(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def prepared(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"]["prepare"] = {"requests": 3, "drained": True}
            out.write_text(json.dumps(blob))

        procs._write_artifact = prepared
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("prepared" in f for f in runner.failures)


class TestWhatItDoesWhenSomethingFails:
    def test_a_server_that_never_answers_is_not_waited_on_forever(self, tmp_path):
        runner = _runner(
            tmp_path, "modelled", health=lambda url: None, startup_timeout=30.0
        )
        assert runner.run() == 1
        assert any("no /health" in f for f in runner.failures)

    def test_a_server_that_exits_before_health_is_reported_as_that(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path)
        original = procs.start

        def start(command, **kw):
            proc = original(command, **kw)
            proc.returncode = 2
            return proc

        procs.start = start
        runner = _runner(tmp_path, "modelled", processes=procs, health=lambda u: None)
        assert runner.run() == 1
        assert any("exited 2 before it was healthy" in f for f in runner.failures)

    def test_a_refusal_is_kept_as_a_refusal(self, tmp_path):
        """replay.py exits 3 when it will not measure what it was asked to.
        Retrying that into a pass is the whole failure mode."""
        procs = FakeProcesses(cell=tmp_path, exits={"modelled.r1.json": 3})
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == run_mod.REFUSAL_EXIT
        assert _journal(runner)["refused"] is True
        assert len(procs.ran) == 1

    def test_a_failing_replay_stops_the_side(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, exits={"modelled.r2.json": 1})
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert len(procs.ran) == 2
        assert len(procs.started) == 2

    def test_a_server_that_died_during_the_replay_invalidates_it(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, dies_after="modelled.r1.json")
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("died during the replay" in f for f in runner.failures)

    def test_a_missing_artifact_is_not_a_silent_pass(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, artifacts=False)
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("was not written" in f for f in runner.failures)

    def test_whatever_is_still_running_is_stopped_on_the_way_out(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, exits={"real.r2.json": 1})
        runner = _runner(tmp_path, "real", processes=procs)
        runner.run()
        assert runner.running == {}
        assert set(procs.stopped) == {p.pid for p in procs.started}

    def test_the_journal_is_written_even_when_the_side_failed(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, exits={"modelled.r1.json": 1})
        runner = _runner(tmp_path, "modelled", processes=procs)
        runner.run()
        journal = _journal(runner)
        assert journal["ok"] is False and journal["failures"]


class TestWhatAnsweredIsWhatWeThinkAnswered:
    def test_each_repeat_s_provenance_is_kept(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for n in (1, 2, 3):
            assert (tmp_path / "tp2_long" / f"provenance.modelled.r{n}.json").exists()

    def test_a_server_in_the_other_side_s_mode_fails_the_repeat(self, tmp_path):
        """A modelled side served by a measuring engine is a real run wearing
        the modelled side's filename."""
        runner = _runner(
            tmp_path,
            "modelled",
            provenance=lambda url: {"compass": {"mode": "measure"}},
        )
        assert runner.run() == 1
        assert any("mode='measure'" in f for f in runner.failures)

    def test_a_server_that_will_not_say_what_it_is_fails_the_repeat(self, tmp_path):
        runner = _runner(tmp_path, "modelled", provenance=lambda url: None)
        assert runner.run() == 1
        assert any("/compass/provenance" in f for f in runner.failures)


class TestTheCostsItCanMeasure:
    def test_a_side_records_its_own_startup_and_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            run_mod.compare,
            "load_run",
            lambda path, label: types.SimpleNamespace(joined={}),
        )
        monkeypatch.setattr(
            run_mod.compare,
            "metrics",
            lambda run, indices: {"window_s": 12.0},
        )
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0
        costs = json.loads((runner.cell / "costs.modelled.json").read_text())
        assert costs["execution_modelled"] == 12.0
        assert len(costs["execution_s"]) == 3
        assert costs["convention"] == run_mod.compare.QUANTILE_CONVENTION

    def test_a_failed_side_writes_no_cost_partial(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, exits={"modelled.r1.json": 1})
        runner = _runner(tmp_path, "modelled", processes=procs)
        runner.run()
        assert not (runner.cell / "costs.modelled.json").exists()

    def _partials(self, cell):
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "costs.real.json").write_text(
            json.dumps({"startup_real": 100.0, "execution_real": 300.0})
        )
        (cell / "costs.modelled.json").write_text(
            json.dumps({"startup_modelled": 9.0, "execution_modelled": 30.0})
        )

    def test_the_merge_needs_the_terms_nothing_here_measures(self, tmp_path, capsys):
        cell = tmp_path / "tp2_long"
        self._partials(cell)
        assert run_mod.main(["costs", str(cell)]) == 2
        assert not (cell / "costs.json").exists()
        message = capsys.readouterr().err
        for term in run_mod.SUPPLIED_TERMS:
            assert f"--{term}" in message

    def test_the_merge_needs_both_sides(self, tmp_path, capsys):
        cell = tmp_path / "tp2_long"
        cell.mkdir(parents=True)
        (cell / "costs.real.json").write_text(
            json.dumps({"startup_real": 1.0, "execution_real": 2.0})
        )
        argv = [
            "costs",
            str(cell),
            "--capture",
            "1",
            "--calibration",
            "1",
            "--derivation",
            "1",
            "--load",
            "1",
        ]
        assert run_mod.main(argv) == 2
        assert "costs.modelled.json" in capsys.readouterr().err

    def test_a_complete_merge_carries_every_term_the_validator_reads(self, tmp_path):
        cell = tmp_path / "tp2_long"
        self._partials(cell)
        argv = [
            "costs",
            str(cell),
            "--capture",
            "412",
            "--calibration",
            "1980",
            "--derivation",
            "31.5",
            "--load",
            "96",
        ]
        assert run_mod.main(argv) == 0
        costs = json.loads((cell / "costs.json").read_text())
        validate = _load("cc_traces_validate")
        for term in validate.COST_TERMS:
            assert isinstance(costs[term], float)
        assert costs["supplied"] == list(run_mod.SUPPLIED_TERMS)


class TestTheSidesCannotBeRunWrong:
    def test_the_modelled_side_needs_a_captured_target(self, tmp_path, capsys):
        argv = [
            "side",
            "--cell",
            str(tmp_path / "tp2_long"),
            "--side",
            "modelled",
            "--tp",
            "2",
            "--class",
            "long",
        ]
        assert run_mod.main(argv) == 2
        assert "--replay-target" in capsys.readouterr().err

    def test_the_real_side_refuses_one(self, tmp_path, capsys):
        argv = [
            "side",
            "--cell",
            str(tmp_path / "tp2_long"),
            "--side",
            "real",
            "--tp",
            "2",
            "--class",
            "long",
            "--replay-target",
            "/w/t.json",
        ]
        assert run_mod.main(argv) == 2
        assert "modelled side's input" in capsys.readouterr().err

    def test_the_cell_directory_has_to_be_the_one_the_validator_expects(self, tmp_path):
        argv = [
            "side",
            "--cell",
            str(tmp_path / "somewhere"),
            "--side",
            "real",
            "--tp",
            "2",
            "--class",
            "long",
        ]
        with pytest.raises(SystemExit):
            run_mod.main(argv)

    def test_a_side_runs_only_its_own_steps(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        assert all(s["side"] == "modelled" for s in runner.steps())
        assert not any(
            "gpu_sampler" in " ".join(s["command"] or []) for s in runner.steps()
        )


class TestEveryRepeatIsAnExecutionWithAName:
    """A digest says what is in a file, not which run produced it.

    Two repeats of one cell can produce byte-identical artifacts and still be
    independent executions -- which is exactly the distinction a reader has to
    make between a source residual and an independent repeat. So the identity
    is minted where the process is launched, from facts about the launch, and
    is carried by the artifact rather than inferred from it.
    """

    def _executions(self, runner):
        return _journal(runner)["executions"]

    def test_one_identity_per_fresh_server_repeat(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0
        executions = self._executions(runner)
        assert len(executions) == 3
        assert [e["repeat"] for e in executions] == [1, 2, 3]
        assert len({e["execution_id"] for e in executions}) == 3

    def test_an_id_is_re_derivable_from_its_own_recorded_inputs(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for execution in self._executions(runner):
            assert run_mod.verify_execution_id(execution)
            assert execution["schema"] == run_mod.EXECUTION_SCHEMA
            assert execution["execution_id"].startswith("cx-")

    def test_a_tampered_record_does_not_verify(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        execution = dict(self._executions(runner)[0])
        execution["id_inputs"] = dict(execution["id_inputs"], server_pid=1)
        assert not run_mod.verify_execution_id(execution)

    def test_the_id_does_not_come_from_the_payload(self, tmp_path):
        """Identical artifacts, different executions -- the case a content
        hash cannot tell apart."""
        runner = _runner(tmp_path, "modelled")
        runner.run()
        executions = self._executions(runner)
        stamped = [
            json.loads((runner.cell / f"modelled.r{n}.json").read_text())
            for n in (1, 2)
        ]
        for blob in stamped:
            blob.pop("execution")
        assert stamped[0] == stamped[1]
        assert executions[0]["execution_id"] != executions[1]["execution_id"]

    def test_the_artifact_carries_the_id_so_a_copy_still_has_it(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for execution in self._executions(runner):
            path = runner.cell / f"modelled.r{execution['repeat']}.json"
            carried = json.loads(path.read_text())["execution"]
            assert carried["execution_id"] == execution["execution_id"]
            assert run_mod.verify_execution_id(carried)
            copied = json.loads(json.dumps(carried))
            assert run_mod.verify_execution_id(copied)

    def test_an_artifact_from_another_execution_is_refused(self, tmp_path):
        """A stale file left in the cell, or one copied in from elsewhere."""
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def foreign(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["execution"] = {"execution_id": "cx-000000000000dead"}
            out.write_text(json.dumps(blob))

        procs._write_artifact = foreign
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("cx-000000000000dead" in f for f in runner.failures)

    def test_an_execution_records_the_process_that_ran_it(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        started = {p.pid for p in runner.processes.started}
        for execution in self._executions(runner):
            process = execution["process"]
            assert process["pid"] in started
            assert process["launched_at"] <= process["healthy_at"]
            assert process["ended_at"] is not None
            assert process["exit"] == 0
            assert process["command"][0] == "python"
            assert execution["replay"]["exit"] == 0
            assert execution["replay"]["started_at"] <= execution["replay"]["ended_at"]

    def test_an_execution_records_its_source_and_configuration(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for execution in self._executions(runner):
            source, config = execution["source"], execution["config"]
            assert source["workload"].endswith("cc_traces_long.jsonl")
            assert len(source["workload_sha256"]) == 64
            assert source["replay_target"] == "/w/target.json"
            assert source["oracle"] == "transfer"
            assert config["mode"] == "predict"
            assert config["tp"] == "2"
            assert config["provenance"]["compass"]["mode"] == "predict"
            assert config["engine_args"] == list(plan_mod.ENGINE_ARGS)

    def test_an_execution_records_what_it_produced(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for execution in self._executions(runner):
            artifacts = execution["artifacts"]
            name = f"modelled.r{execution['repeat']}.json"
            assert name in artifacts
            on_disk = run_mod.file_digest(runner.cell / name)
            assert artifacts[name] == on_disk, "digest taken after stamping"
            assert f"provenance.modelled.r{execution['repeat']}.json" in artifacts

    def test_the_record_sits_beside_the_artifacts_too(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for n in (1, 2, 3):
            path = runner.cell / f"execution.modelled.r{n}.json"
            assert run_mod.verify_execution_id(json.loads(path.read_text()))

    def test_a_failed_repeat_still_has_an_identity(self, tmp_path):
        """The run that has to be explained later is the one that failed."""
        procs = FakeProcesses(cell=tmp_path, exits={"modelled.r2.json": 1})
        runner = _runner(tmp_path, "modelled", processes=procs)
        runner.run()
        executions = self._executions(runner)
        assert len(executions) == 2
        assert executions[-1]["replay"]["exit"] == 1
        assert run_mod.verify_execution_id(executions[-1])

    def test_the_ids_of_the_two_sides_do_not_collide(self, tmp_path):
        real = _runner(tmp_path, "real")
        real.run()
        modelled = _runner(tmp_path, "modelled")
        modelled.run()
        ids = {e["execution_id"] for e in self._executions(real)}
        ids |= {e["execution_id"] for e in self._executions(modelled)}
        assert len(ids) == 6

    def test_the_cost_terms_name_the_executions_that_spent_them(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        costs = json.loads((runner.cell / "costs.modelled.json").read_text())
        spent = [row["execution_id"] for row in costs["per_execution"]]
        assert spent == [e["execution_id"] for e in self._executions(runner)]

    def test_the_journal_names_the_execution_each_step_belonged_to(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        known = {e["execution_id"] for e in self._executions(runner)}
        for entry in _journal(runner)["steps"]:
            if entry.get("execution_id") is not None:
                assert entry["execution_id"] in known


class TestTheProducersAreCheckedAtTheirOwnFieldNames:
    """Each check below reads a field some other component writes.

    The names come from `api_server.compass_provenance` and from the manifest
    `replay.py` writes, so what is asserted here is agreement between the two
    sides of a boundary rather than agreement between the harness and itself.
    """

    def test_a_server_with_compass_off_is_not_a_compass_result(self, tmp_path):
        said = fake_provenance("predict")
        said["compass"]["enabled"] = False
        runner = _runner(tmp_path, "modelled", provenance=lambda url: said)
        assert runner.run() == 1
        assert any("compass disabled" in f for f in runner.failures)

    def test_a_predictor_on_no_virtual_clock_is_refused(self, tmp_path):
        """`replay.py::_clock_of` reads this exact field to decide whether to
        refuse a warmed predictor. False here means that refusal never fires."""
        said = fake_provenance("predict")
        said["compass"]["virtual_clock"] = False
        runner = _runner(tmp_path, "modelled", provenance=lambda url: said)
        assert runner.run() == 1
        assert any("virtual clock" in f for f in runner.failures)

    def test_a_measuring_server_on_a_virtual_clock_is_refused(self, tmp_path):
        said = fake_provenance("measure")
        said["compass"]["virtual_clock"] = True
        runner = _runner(tmp_path, "real", provenance=lambda url: said)
        assert runner.run() == 1
        assert any("modelled ones and not measurements" in f for f in runner.failures)

    def test_a_server_at_the_wrong_width_is_refused(self, tmp_path):
        runner = _runner(
            tmp_path,
            "modelled",
            provenance=lambda url: fake_provenance("predict", tp=4),
        )
        assert runner.run() == 1
        assert any("tensor_parallel_size=4" in f for f in runner.failures)

    def test_a_replay_of_another_trace_is_refused(self, tmp_path):
        """The frozen corpus, by its bytes rather than by its path."""
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def other_trace(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"]["trace_sha256"] = "b" * 64
            out.write_text(json.dumps(blob))

        procs._write_artifact = other_trace
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("frozen workload" in f for f in runner.failures)

    def test_an_artifact_answered_by_another_server_is_refused(self, tmp_path):
        """Something else listening on the port is the failure that looks
        most like a success."""
        procs = FakeProcesses(cell=tmp_path, served="c" * 64)
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("listening on that port" in f for f in runner.failures)

    def test_an_artifact_with_no_server_provenance_is_refused(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def anonymous(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"]["server"] = None
            out.write_text(json.dumps(blob))

        procs._write_artifact = anonymous
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("no server provenance" in f for f in runner.failures)

    @pytest.mark.parametrize(
        ("field", "value", "says"),
        [
            ("store_empty_after_drain", False, "not empty after the drain"),
            ("drained_records", 0, "drained no engine records"),
        ],
    )
    def test_a_preparation_that_did_not_really_drain_is_refused(
        self, tmp_path, field, value, says
    ):
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def undrained(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            if blob["run"].get("prepare"):
                blob["run"]["prepare"][field] = value
                out.write_text(json.dumps(blob))

        procs._write_artifact = undrained
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any(says in f for f in runner.failures)


class TestTheWatchHasToCoverTheWindow:
    """A clean audit over four seconds of a forty-minute window is not
    evidence that the node was quiet."""

    def test_a_sampler_that_wrote_nothing_fails_the_side(self, tmp_path):
        runner = _runner(
            tmp_path, "real", processes=FakeProcesses(cell=tmp_path, sampler=False)
        )
        assert runner.run() == 1
        assert any("went unwatched" in f for f in runner.failures)

    def test_samples_without_a_baseline_fail_the_side(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path)
        original = procs._gpu_sample
        procs._gpu_sample = lambda command, phase: original(command, "window")
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("no baseline sample" in f for f in runner.failures)

    def test_a_watch_that_started_late_fails_the_side(self, tmp_path):
        clock = Clock()
        procs = FakeProcesses(cell=tmp_path, wall=lambda: clock.wall() + 600.0)
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("after the first server did" in f for f in runner.failures)

    def test_a_watch_that_stopped_early_fails_the_side(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, wall=lambda: 1_600_000_000.0)
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("before the last server did" in f for f in runner.failures)


class TestOnlyTheServerThisRepeatStartedCounts:
    """A code digest is a fact about bytes on disk, not about who replied.

    The case these cover is ordinary rather than exotic: a server from an
    earlier repeat is still holding the port, so it is already warm and
    answers `/health` at once, while the process this repeat launched is still
    loading weights. Every configuration field matches, because it is the same
    tree and the same flags. Without an identity for the process itself, the
    repeat gets measured against a server nobody meant to start.
    """

    def _stale(self, tmp_path, side="modelled", **over):
        procs = FakeProcesses(cell=tmp_path)
        mode = "predict" if side == "modelled" else "measure"
        stale_pid = 999_111
        runner = _runner(
            tmp_path,
            side,
            processes=procs,
            # Same digest, same config, same mode, and a process that really
            # does exist on this machine -- just not one of ours.
            provenance=lambda url: fake_provenance(
                mode, server_process=fake_process_identity(stale_pid, **over)
            ),
            probe=FakeProbe(procs, ticks={stale_pid: 900_000 + stale_pid}),
        )
        return runner, stale_pid

    def test_a_stale_server_with_identical_code_is_refused(self, tmp_path):
        runner, stale_pid = self._stale(tmp_path)
        assert runner.run() == 1
        assert any(
            "not the process this repeat launched" in f for f in runner.failures
        ), runner.failures
        # The thing that would have waved it through, had we asked only that.
        said = runner.provenance("http://x/compass/provenance")
        assert said["server_code_sha256"] == SERVER_CODE
        assert said["server_process"]["pid"] == stale_pid

    def test_the_refusal_says_why_a_matching_digest_was_not_enough(self, tmp_path):
        runner, _ = self._stale(tmp_path)
        runner.run()
        why = " ".join(runner.failures)
        assert "built from the same tree" in why, why

    def test_a_pid_reused_by_another_process_is_refused(self, tmp_path):
        """Our pid, someone else's process. Start time is what tells them apart."""
        procs = FakeProcesses(cell=tmp_path)
        runner = _runner(
            tmp_path,
            "modelled",
            processes=procs,
            provenance=lambda url: fake_provenance(
                "predict",
                server_process=fake_process_identity(_serving(procs), ticks=12_345),
            ),
        )
        assert runner.run() == 1
        assert any("reused" in f for f in runner.failures), runner.failures

    def test_a_descendant_of_the_launched_process_is_accepted(self, tmp_path):
        """A launcher may fork; the socket is then held by a child of ours."""
        procs = FakeProcesses(cell=tmp_path)
        forked = 777_333
        runner = _runner(
            tmp_path,
            "modelled",
            processes=procs,
            provenance=lambda url: fake_provenance(
                "predict", server_process=fake_process_identity(forked)
            ),
            probe=FakeProbe(
                procs,
                ticks={forked: 900_000 + forked},
                parents={forked: _serving},
            ),
        )
        assert runner.run() == 0, runner.failures

    def test_a_server_that_will_not_say_which_process_it_is_is_refused(self, tmp_path):
        runner = _runner(
            tmp_path,
            "modelled",
            provenance=lambda url: fake_provenance("predict", server_process=None),
        )
        assert runner.run() == 1
        assert any(
            "does not report which process it is" in f for f in runner.failures
        ), runner.failures

    def test_a_reply_from_another_machine_is_refused(self, tmp_path):
        """`/proc` here cannot vouch for a process somewhere else."""
        procs = FakeProcesses(cell=tmp_path)
        runner = _runner(
            tmp_path,
            "modelled",
            processes=procs,
            provenance=lambda url: fake_provenance(
                "predict",
                server_process=fake_process_identity(_serving(procs), host="elsewhere"),
            ),
        )
        assert runner.run() == 1
        assert any("another machine" in f for f in runner.failures), runner.failures

    def test_a_reply_from_before_this_boot_is_refused(self, tmp_path):
        """Ticks are counted from boot, so they only compare within one."""
        procs = FakeProcesses(cell=tmp_path)
        runner = _runner(
            tmp_path,
            "modelled",
            processes=procs,
            provenance=lambda url: fake_provenance(
                "predict",
                server_process=fake_process_identity(
                    _serving(procs), boot="99999999-9999-9999-9999-999999999999"
                ),
            ),
        )
        assert runner.run() == 1
        assert any("different boot" in f for f in runner.failures), runner.failures

    def test_a_child_that_died_before_answering_is_refused(self, tmp_path):
        """It lost the bind race and exited; the warm one replied instead."""
        procs = FakeProcesses(cell=tmp_path)

        def answers_after_ours_dies(url):
            procs.started[-1].returncode = 1
            return fake_provenance(
                "predict", server_process=fake_process_identity(999_111)
            )

        runner = _runner(
            tmp_path, "modelled", processes=procs, provenance=answers_after_ours_dies
        )
        assert runner.run() == 1
        assert any("had already exited" in f for f in runner.failures), runner.failures

    def test_the_two_accounts_are_kept_apart_in_the_record(self, tmp_path):
        """The server's word and ours are recorded separately.

        A harness stamp on its own proves nothing about who served the
        requests -- it is the harness talking about itself. What makes it
        evidence is that an independent reading agrees with it, and that is
        only visible if the two are not written from each other.
        """
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0, runner.failures
        record = runner.executions[1]
        seen = record["server_process"]
        assert seen["verified"] is True
        assert seen["said"]["pid"] == seen["observed"]["launched_pid"]
        assert seen["observed"]["start_ticks"] == seen["said"]["start_ticks"]
        # The launch fact stays the harness's own and is not restated from
        # what the server said.
        assert record["id_inputs"]["server_pid"] == seen["observed"]["launched_pid"]
        assert "start_ticks" not in record["id_inputs"]

    def test_a_refused_repeat_still_keeps_what_the_server_said(self, tmp_path):
        """Evidence of the wrong server is evidence, and worth keeping."""
        runner, stale_pid = self._stale(tmp_path)
        runner.run()
        seen = runner.executions[1]["server_process"]
        assert seen["verified"] is False
        assert seen["said"]["pid"] == stale_pid
        assert seen["observed"]["launched_pid"] != stale_pid

    def test_the_check_runs_on_the_real_side_too(self, tmp_path):
        runner, _ = self._stale(tmp_path, side="real")
        assert runner.run() == 1
        assert any(
            "not the process this repeat launched" in f for f in runner.failures
        ), runner.failures
