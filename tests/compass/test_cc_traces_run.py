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
            # The plan names this on the modelled side; the server echoes what
            # it is actually serving under.
            "rank_aggregation": "slowest" if mode == "predict" else "rank0",
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
        proc.env = env
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
        # The root the runner itself digests, which a test may have moved: a
        # fake that reads the real checkout would hand the runner a digest of
        # bytes the runner never looked at.
        full = Path(run_mod.ROOT) / trace
        manifest = {
            "paced": side == "real",
            "trace": str(trace),
            "trace_sha256": (run_mod.file_digest(full) or {}).get("sha256"),
            # What `replay.py` writes when every request completed. Carried
            # here because the harness reads the tally rather than the exit
            # code, and a fixture that omits it would stand for an artifact
            # no run of this client produces.
            "failed": 0,
            "missing": 0,
            "truncated": 0,
            "completed": 0,
            "complete": True,
            "incomplete_reasons": None,
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


def _a_workload_on_disk(tmp_path, monkeypatch, klass="long"):
    """Bytes where the runner looks for the workload, for the digest checks.

    The runner digests whatever is at `ROOT/atom/compass/
    cc_traces_<class>.jsonl` and refuses a replay whose trace digest differs.
    Those registered files are reproduced from the corpus rather than
    committed, so a test about that plumbing supplies its own workload instead
    of depending on a checkout having the acceptance inputs in it.
    """
    root = tmp_path / "repo"
    path = root / "atom" / "compass" / f"cc_traces_{klass}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "arrival_s": 0.0, "input_tokens": 640,
        "input_blocks": 10, "output_tokens": 23}) + "\n")
    # The runner reads two defaults out of the tree it is pointed at -- the
    # server's and the engine's `--port` -- so a moved root has to carry those
    # files as well, or a test about the workload digest quietly becomes a
    # test about a missing entry point.
    for relative in ("atom/entrypoints/openai/api_server.py",
                     "atom/model_engine/arg_utils.py"):
        source = ROOT / relative
        if not source.exists():
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(run_mod, "ROOT", root)
    return path


@pytest.fixture(autouse=True)
def _the_workload_is_on_disk(tmp_path, monkeypatch):
    """Every side run digests the frozen corpus, so every test needs one.

    The `.jsonl` corpora are reproduced by the plan's own workload step rather
    than committed, so a checkout does not have them and a side run in a test
    would find nothing to digest. That used to be invisible -- the digest
    check passed whatever it could not answer -- and is now a refusal, which
    is the point of the check.
    """
    _a_workload_on_disk(tmp_path, monkeypatch)


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
    tmp_path,
    side,
    *,
    processes=None,
    health=None,
    provenance=None,
    probe=None,
    held_ports=(),
    **kw,
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
        # Nothing is listening unless the test says so: the real reading
        # would connect to whatever happens to be up on this machine.
        ports_in_use=lambda ports: {p for p in map(int, ports) if p in held_ports},
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

    def test_an_incomplete_replay_fails_the_repeat(self, tmp_path):
        """The exit code is one bit and it is the client's own opinion. A
        repeat whose replay answered a fraction of the workload and returned
        zero would otherwise be accepted, which is how a TP1 development run
        that served 3 of 62 requests reached a result file."""
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def short(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"].update(
                {
                    "complete": False,
                    "failed": 59,
                    "completed": 3,
                    "requests": 62,
                    "incomplete_reasons": {
                        "failed": [
                            {"reason": "TimeoutError: timed out", "requests": 59}
                        ]
                    },
                }
            )
            out.write_text(json.dumps(blob))

        procs._write_artifact = short
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("did not complete its workload" in f for f in runner.failures)
        assert any("TimeoutError" in f for f in runner.failures)

    def test_an_artifact_with_no_tally_fails_the_repeat(self, tmp_path):
        """Unknown is not complete. An artifact from a client too old to count
        says nothing about whether its workload finished, and reading it as a
        result is the assumption this whole check exists to refuse."""
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def untallied(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            for name in ("complete", "completed", "failed", "missing",
                         "truncated", "incomplete_reasons"):
                blob["run"].pop(name, None)
            out.write_text(json.dumps(blob))

        procs._write_artifact = untallied
        runner = _runner(tmp_path, "real", processes=procs)
        assert runner.run() == 1
        assert any("no completeness tally" in f for f in runner.failures)

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
            lambda path, label: types.SimpleNamespace(joined={}, clock="virtual"),
        )
        monkeypatch.setattr(
            run_mod.compare,
            "metrics",
            lambda run, indices: {"window_s": 12.0},
        )
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0
        costs = json.loads((runner.cell / "costs.modelled.json").read_text())
        wall = [
            e["seconds"]
            for e in _journal(runner)["steps"]
            if e["role"] == "replay" and e["ok"]
        ]
        # The cost term is the client's own stopwatch, not the window the
        # engine reported serving: on a predicting server that window is a
        # prediction about duration.
        assert costs["execution_s"] == wall
        assert costs["execution_modelled"] == run_mod._median(wall)
        assert costs["served_window_s"] == [12.0, 12.0, 12.0]
        assert costs["served_window_modelled"] == 12.0
        assert len(costs["execution_s"]) == 3
        assert costs["convention"] == run_mod.compare.QUANTILE_CONVENTION

    def test_the_two_clocks_are_named_and_the_cost_one_is_the_wall(
        self, tmp_path, monkeypatch
    ):
        """A virtual window and a wall window are both seconds and are not the
        same quantity, so the record says which clock each was taken on.

        The numbers are the TP1 plumbing diagnostic's: a 98.87 s served window
        against a replay that took about three and a half seconds. Dividing a
        real side by the first would have reported a speedup nobody measured.
        """
        monkeypatch.setattr(
            run_mod.compare,
            "load_run",
            lambda path, label: types.SimpleNamespace(joined={}, clock="virtual"),
        )
        monkeypatch.setattr(
            run_mod.compare,
            "metrics",
            lambda run, indices: {"window_s": 98.87},
        )
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0
        costs = json.loads((runner.cell / "costs.modelled.json").read_text())
        assert costs["cost_schema"] == run_mod.COSTS_SCHEMA
        assert costs["clocks"]["startup"] == run_mod.WALL_CLOCK
        assert costs["clocks"]["execution"] == run_mod.WALL_CLOCK
        assert costs["clocks"]["served_window"] == "virtual"
        assert costs["execution_modelled"] != 98.87
        assert costs["served_window_modelled"] == 98.87
        assert costs["per_execution"][0]["served_window_s"] == 98.87
        assert costs["per_execution"][0]["execution_s"] == costs["execution_s"][0]

    def test_a_failed_side_writes_no_cost_partial(self, tmp_path):
        procs = FakeProcesses(cell=tmp_path, exits={"modelled.r1.json": 1})
        runner = _runner(tmp_path, "modelled", processes=procs)
        runner.run()
        assert not (runner.cell / "costs.modelled.json").exists()

    def _partials(self, cell, **over):
        cell.mkdir(parents=True, exist_ok=True)
        wall = {"startup": run_mod.WALL_CLOCK, "execution": run_mod.WALL_CLOCK}
        real = {
            "cost_schema": run_mod.COSTS_SCHEMA,
            "clocks": dict(wall, served_window="wall"),
            "startup_real": 100.0,
            "execution_real": 300.0,
            "served_window_real": 299.0,
        }
        modelled = {
            "cost_schema": run_mod.COSTS_SCHEMA,
            "clocks": dict(wall, served_window="virtual"),
            "startup_modelled": 9.0,
            "execution_modelled": 30.0,
            "served_window_modelled": 300.0,
        }
        modelled.update(over)
        (cell / "costs.real.json").write_text(json.dumps(real))
        (cell / "costs.modelled.json").write_text(json.dumps(modelled))

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
        assert run_mod.main(self._argv(cell)) == 0
        costs = json.loads((cell / "costs.json").read_text())
        validate = _load("cc_traces_validate")
        for term in validate.MEASURED_COST_TERMS:
            assert isinstance(costs[term], float)
        # A supplied second carries where it was read from, what contains it
        # and which repeat spent it, so a total can tell it from a measured
        # one, not double it, and keep it beside one repeat of execution.
        for term in validate.SUPPLIED_COST_TERMS:
            assert set(costs[term]) == {"seconds", "source", "within", "repeat"}
            assert costs[term]["source"]
        assert costs["load"]["within"] == "startup_real"
        assert costs["capture"]["within"] is None
        assert costs["supplied"] == list(run_mod.SUPPLIED_TERMS)
        # The gate reads this to check it is dividing wall seconds by wall
        # seconds, and the served windows travel beside it, not inside it.
        assert costs["execution_clocks"] == {
            "real": run_mod.WALL_CLOCK,
            "modelled": run_mod.WALL_CLOCK,
        }
        assert costs["cost_schema"] == run_mod.COSTS_SCHEMA
        assert costs["served_window_modelled"] == 300.0
        assert costs["execution_modelled"] == 30.0

    def test_a_journal_places_each_derivation_by_its_own_interval(self, tmp_path):
        """The split is measured on both sides: the journal stamps each
        derivation's wall interval and the side record stamps each window's,
        so which contains which is an intersection rather than a claim."""
        cell = tmp_path / "tp2_long"
        self._partials(
            cell,
            per_execution=[
                {
                    "repeat": 0,
                    "startup_window": [1000.0, 1040.0],
                    "execution_window": [1050.0, 1100.0],
                }
            ],
        )
        journal = tmp_path / "derivations.jsonl"
        journal.write_text(
            # One while the server came up, one mid-schedule, one in neither.
            json.dumps({"t0": 1005.0, "t1": 1035.0})
            + "\n"
            + json.dumps({"t0": 1060.0, "t1": 1080.0})
            + "\n"
            + json.dumps({"t0": 1041.0, "t1": 1044.0})
            + "\n"
        )
        argv = [a for a in self._argv(cell) if a not in ("--derivation-within", "none")]
        argv += ["--derivation-journal", str(journal)]
        assert run_mod.main(argv) == 0
        parts = json.loads((cell / "costs.json").read_text())["derivation"]
        by_window = {p["within"]: p["seconds"] for p in parts}
        assert by_window["startup_modelled"] == pytest.approx(30.0)
        assert by_window["execution_modelled"] == pytest.approx(20.0)
        assert by_window[None] == pytest.approx(3.0)
        # The manual --derivation number was not read; the journal decided.
        assert sum(by_window.values()) == pytest.approx(53.0)

    def test_a_journal_without_the_windows_it_needs_is_not_guessed_at(
        self, tmp_path, capsys
    ):
        """An interval means nothing without the windows it would fall inside,
        and inventing them is exactly the assertion this replaces."""
        cell = tmp_path / "tp2_long"
        self._partials(cell, per_execution=[{"repeat": 0}])
        journal = tmp_path / "derivations.jsonl"
        journal.write_text(json.dumps({"t0": 1.0, "t1": 2.0}) + "\n")
        argv = self._argv(cell) + ["--derivation-journal", str(journal)]
        assert run_mod.main(argv) == 2
        assert "startup_window" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_journal_the_run_wrote_into_the_cell_needs_no_flag(self, tmp_path):
        """The modelled server writes its journal here, so the merge finds it.

        Without this, the measured split depends on the operator remembering a
        flag, and forgetting it silently falls back to a declared number.
        """
        cell = tmp_path / "tp2_long"
        self._partials(
            cell,
            per_execution=[
                {
                    "repeat": 0,
                    "startup_window": [1000.0, 1040.0],
                    "execution_window": [1050.0, 1100.0],
                }
            ],
        )
        # The runtime's own naming: one file per server process.
        (cell / "derivation.modelled.r0.4001.jsonl").write_text(
            json.dumps({"t0": 1005.0, "t1": 1035.0}) + "\n"
        )
        (cell / "derivation.modelled.r1.4002.jsonl").write_text(
            json.dumps({"t0": 1060.0, "t1": 1080.0}) + "\n"
        )
        argv = [a for a in self._argv(cell) if a not in ("--derivation-within", "none")]
        assert run_mod.main(argv) == 0
        parts = json.loads((cell / "costs.json").read_text())["derivation"]
        by_window = {p["within"]: p["seconds"] for p in parts}
        assert by_window["startup_modelled"] == pytest.approx(30.0)
        assert by_window["execution_modelled"] == pytest.approx(20.0)

    def test_no_journal_in_the_cell_is_still_a_refusal_not_a_zero(
        self, tmp_path, capsys
    ):
        """Discovery finding nothing must not read as 'derivation was free'."""
        cell = tmp_path / "tp2_long"
        self._partials(cell)
        full = self._argv(cell)
        drop = {"--derivation", "--derivation-source", "--derivation-within"}
        argv, skip = [], 0
        for item in full:
            if skip:
                skip = 0
                continue
            if item in drop:
                skip = 1
                continue
            argv.append(item)
        assert run_mod.main(argv) == 2
        assert "--derivation" in capsys.readouterr().err

    def test_a_derivation_with_no_stated_container_is_not_merged(
        self, tmp_path, capsys
    ):
        """`load` has a container the protocol states, so it defaults. Nobody
        has measured whether the oracle build is inside `startup_modelled`, so
        `derivation` has no default and the operator has to say which it is."""
        cell = tmp_path / "tp2_long"
        self._partials(cell)
        argv = [a for a in self._argv(cell) if a not in ("--derivation-within", "none")]
        assert run_mod.main(argv) == 2
        assert "--derivation-within" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_supplied_second_with_no_artifact_is_not_merged(self, tmp_path, capsys):
        cell = tmp_path / "tp2_long"
        self._partials(cell)
        argv = [a for a in self._argv(cell) if a not in ("--load-source", "server.log")]
        assert run_mod.main(argv) == 2
        assert "--load-source" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_partial_from_the_old_schema_is_not_merged(self, tmp_path, capsys):
        """The old record called a virtual window `execution_modelled`, and
        nothing in it says so. It cannot be read as if it were the new one."""
        cell = tmp_path / "tp2_long"
        self._partials(cell)
        (cell / "costs.modelled.json").write_text(
            json.dumps({"startup_modelled": 9.0, "execution_modelled": 98.87})
        )
        assert run_mod.main(self._argv(cell)) == 2
        assert run_mod.COSTS_SCHEMA in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_partial_whose_execution_is_not_wall_is_not_merged(
        self, tmp_path, capsys
    ):
        cell = tmp_path / "tp2_long"
        self._partials(cell, clocks={"startup": "wall", "execution": "virtual"})
        assert run_mod.main(self._argv(cell)) == 2
        assert "wall-clock" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def _argv(self, cell):
        return [
            "costs",
            str(cell),
            "--capture",
            "412",
            "--capture-source",
            "capture/manifest.json",
            "--calibration",
            "1980",
            "--calibration-source",
            "registry/calibration.json",
            "--derivation",
            "31.5",
            "--derivation-source",
            "startup.json",
            # Nobody has measured whether the oracle build is inside the
            # modelled startup, so this run says it is not and the record
            # carries that claim.
            "--derivation-within",
            "none",
            "--load",
            "96",
            "--load-source",
            "server.log",
        ]


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

    def test_the_modelled_side_needs_the_profile_it_is_sized_from(
        self, tmp_path, capsys
    ):
        # A target record carries block counts, so a run without the profile
        # starts and serves -- sized from the captured numbers, publishing
        # `captured`, and answering a question nobody asked.
        argv = [
            "side",
            "--cell",
            str(tmp_path / "tp1_long"),
            "--side",
            "modelled",
            "--tp",
            "1",
            "--class",
            "long",
            "--replay-target",
            "/w/t.json",
        ]
        assert run_mod.main(argv) == 2
        assert "--memory-model" in capsys.readouterr().err

    def test_the_real_side_refuses_a_profile(self, tmp_path, capsys):
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
            "--memory-model",
            "/w/p.json",
        ]
        assert run_mod.main(argv) == 2
        assert "modelled side's input" in capsys.readouterr().err

    def test_the_profile_reaches_the_modelled_server(self, tmp_path):
        args = types.SimpleNamespace(
            cell=str(tmp_path / "tp2_long"),
            tp=2,
            klass="long",
            oracle=None,
            oracle_option=[],
            port=8000,
            engine_port=8006,
            repeats=3,
            replay_target="/w/t.json",
            memory_model="/w/p.json",
            corpus=None,
        )
        built = run_mod._cell_plan(args)
        serves = [s for s in built["steps"]
                  if s["role"] == "serve" and s["side"] == "modelled"]
        assert serves
        for step in serves:
            command = step["command"]
            assert command[command.index("--compass-memory-model") + 1] == "/w/p.json"

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

    def test_an_execution_records_its_source_and_configuration(self, tmp_path, monkeypatch):
        _a_workload_on_disk(tmp_path, monkeypatch)
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

    def test_an_execution_records_the_port_the_server_listened_on(self, tmp_path):
        """Not the engine's internal port. The stale-server refusal names this
        port, and a server is only reachable on the one it bound."""
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for execution in self._executions(runner):
            command = execution["process"]["command"]
            listener = command[command.index("--server-port") + 1]
            rendezvous = command[command.index("--port") + 1]
            assert execution["config"]["port"] == listener
            assert execution["config"]["port"] != rendezvous

    def test_the_internal_port_is_not_read_as_the_http_port(self, tmp_path):
        """`--port` on the server parser is the engine's internal port. A
        manifest that fell back to it would name a port nothing bound, and the
        stale-server refusal would name it too."""
        runner = _runner(tmp_path, "modelled")
        step = {"role": "serve", "id": "serve-modelled-1"}
        command = ["python", "-m", "atom.entrypoints.openai.api_server", "--port", "1"]
        assert runner._config(step, command)["port"] == run_mod.server_default_port()
        assert runner._config(step, command)["port"] != "1"

    def test_the_default_listener_port_is_read_from_the_entry_point(self):
        """Not restated here: a change there has to show up here."""
        source = (
            run_mod.ROOT / "atom" / "entrypoints" / "openai" / "api_server.py"
        ).read_text()
        declared = [
            line.partition("=")[2].strip()
            for line in source.splitlines()
            if line.partition("=")[0].strip() == "DEFAULT_PORT"
        ]
        assert declared == [run_mod.server_default_port()]

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

    def test_a_predictor_pricing_only_its_own_rank_is_refused(self, tmp_path):
        """One process stands in for the group. On `rank0` it reports the rank
        it calls itself, which no step row and no gate below contradicts, so
        the TP4 rank-1 outlier would be dropped without a trace."""
        said = fake_provenance("predict")
        said["compass"]["rank_aggregation"] = "rank0"
        runner = _runner(tmp_path, "modelled", provenance=lambda url: said)
        assert runner.run() == 1
        assert any("rank_aggregation='rank0'" in f for f in runner.failures)

    def test_a_server_too_old_to_declare_its_aggregation_is_refused(self, tmp_path):
        # An absent field is not a passing one: a server built before the flag
        # existed prices rank 0 and cannot say so.
        said = fake_provenance("predict")
        said["compass"].pop("rank_aggregation")
        runner = _runner(tmp_path, "modelled", provenance=lambda url: said)
        assert runner.run() == 1
        assert any("rank_aggregation=None" in f for f in runner.failures)

    def test_a_server_at_the_wrong_width_is_refused(self, tmp_path):
        runner = _runner(
            tmp_path,
            "modelled",
            provenance=lambda url: fake_provenance("predict", tp=4),
        )
        assert runner.run() == 1
        assert any("tensor_parallel_size=4" in f for f in runner.failures)

    def test_a_replay_of_another_trace_is_refused(self, tmp_path, monkeypatch):
        """The frozen corpus, by its bytes rather than by its path."""
        _a_workload_on_disk(tmp_path, monkeypatch)
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

    def test_an_artifact_that_names_no_trace_at_all_is_refused(
        self, tmp_path, monkeypatch
    ):
        """An unanswerable question is not a pass.

        The digest check used to read `want and got and want != got`, so an
        artifact carrying no trace digest -- precisely the one that cannot
        show which corpus it answered -- went through silently. A replay of
        the wrong trace was caught; a replay of an unnameable one was not.
        """
        _a_workload_on_disk(tmp_path, monkeypatch)
        procs = FakeProcesses(cell=tmp_path)
        original = procs._write_artifact

        def no_trace(command):
            original(command)
            out = Path(command[command.index("--out") + 1])
            blob = json.loads(out.read_text())
            blob["run"].pop("trace_sha256")
            out.write_text(json.dumps(blob))

        procs._write_artifact = no_trace
        runner = _runner(tmp_path, "modelled", processes=procs)
        assert runner.run() == 1
        assert any("no trace digest" in f for f in runner.failures)

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


class TestARunSaysWhatItWasFor:
    """So that a diagnostic cannot be laundered into a cell by copying.

    The harness is the same one either way -- that is the point of running a
    plumbing diagnostic through it. What separates the two is the question
    being asked, and the only place that can live is in what the run writes.
    """

    def test_acceptance_is_the_default(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        assert runner.run() == 0
        assert _journal(runner)["purpose"] == "acceptance"

    def test_a_diagnostic_says_so_in_its_journal(self, tmp_path):
        runner = _runner(tmp_path, "modelled", purpose="diagnostic")
        assert runner.run() == 0
        assert _journal(runner)["purpose"] == "diagnostic"

    def test_every_execution_record_carries_it(self, tmp_path):
        runner = _runner(tmp_path, "modelled", purpose="diagnostic")
        runner.run()
        executions = _journal(runner)["executions"]
        assert executions
        assert all(e["purpose"] == "diagnostic" for e in executions)

    def test_the_record_beside_the_artifacts_carries_it(self, tmp_path):
        runner = _runner(tmp_path, "modelled", purpose="diagnostic")
        runner.run()
        written = sorted(runner.cell.glob("execution.modelled.r*.json"))
        assert written
        for path in written:
            assert json.loads(path.read_text())["purpose"] == "diagnostic"

    def test_the_stamp_inside_each_artifact_carries_it(self, tmp_path):
        """The one that survives a copy into somebody else's directory."""
        runner = _runner(tmp_path, "modelled", purpose="diagnostic")
        runner.run()
        stamped = sorted(runner.cell.glob("modelled.r*.json"))
        assert stamped
        for path in stamped:
            stamp = json.loads(path.read_text())["execution"]
            assert stamp["purpose"] == "diagnostic"

    def test_the_purpose_does_not_reach_the_id(self, tmp_path):
        """A diagnostic and an acceptance run are not different schemes.

        The id identifies a process launch. Folding the purpose into it would
        make two ids for one execution and break every record already minted.
        """
        runner = _runner(tmp_path, "modelled", purpose="diagnostic")
        runner.run()
        for execution in _journal(runner)["executions"]:
            assert "purpose" not in execution["id_inputs"]
            assert run_mod.verify_execution_id(execution) is True

    def test_the_cli_offers_only_the_two_purposes(self):
        with pytest.raises(SystemExit):
            run_mod.main(
                [
                    "side",
                    "--cell",
                    "/tmp/x/tp2_long",
                    "--side",
                    "modelled",
                    "--tp",
                    "2",
                    "--class",
                    "long",
                    "--purpose",
                    "acceptance-ish",
                ]
            )


class TestBothPortsAreCheckedBeforeAnythingIsLaunched:
    """A held listener is loud; a held rendezvous port is not.

    At TP>1 the ranks of a server whose rendezvous port is already held join
    the group that is holding it, and nothing about the run says so: the cell
    then reports numbers from a configuration it never had. So both scopes are
    checked, and the check happens before the process is started rather than
    after a health probe times out.
    """

    def _held(self, tmp_path, port):
        runner = _runner(tmp_path, "modelled", held_ports={port})
        assert runner.run() == 1
        return runner

    def test_a_held_rendezvous_port_stops_the_repeat(self, tmp_path):
        runner = self._held(tmp_path, plan_mod.ENGINE_PORT)
        assert any("engine_rendezvous" in f for f in runner.failures)
        assert not runner.processes.started

    def test_a_held_listener_stops_the_repeat(self, tmp_path):
        runner = self._held(tmp_path, plan_mod.PORT)
        assert any("http_listener" in f for f in runner.failures)
        assert not runner.processes.started

    def test_a_free_pair_launches(self, tmp_path):
        runner = _runner(tmp_path, "modelled", held_ports={plan_mod.PORT + 1000})
        assert runner.run() == 0

    def test_two_scopes_on_one_socket_stop_the_repeat(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        step = {
            "role": "serve",
            "id": "serve-modelled-1",
            "command": [
                "python",
                "-m",
                "atom.entrypoints.openai.api_server",
                "--server-port",
                "9",
                "--port",
                "9",
            ],
        }
        reason = runner._port_conflicts(step)
        assert "same socket" in reason

    def test_a_port_nobody_declares_cannot_be_checked(self, tmp_path, monkeypatch):
        """With the flag absent and the producer's default unreadable, the
        port is unknown, and an unknown port is not a free one."""
        monkeypatch.setattr(run_mod, "engine_default_port", lambda: None)
        runner = _runner(tmp_path, "modelled")
        step = {
            "role": "serve",
            "id": "serve-modelled-1",
            "command": ["python", "-m", "atom.entrypoints.openai.api_server"],
        }
        reason = runner._port_conflicts(step)
        assert "engine_rendezvous" in reason and "cannot be checked" in reason

    def test_the_manifest_names_each_port_by_scope_and_producer(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        runner.run()
        for execution in json.loads((runner.cell / "run.modelled.json").read_text())[
            "executions"
        ]:
            ports = execution["config"]["ports"]
            assert ports["http_listener"] == {
                "port": plan_mod.PORT,
                "producer": "--server-port",
            }
            assert ports["engine_rendezvous"] == {
                "port": plan_mod.ENGINE_PORT,
                "producer": "--port",
            }

    def test_an_unflagged_port_is_recorded_against_the_default_it_came_from(
        self, tmp_path
    ):
        runner = _runner(tmp_path, "modelled")
        step = {
            "role": "serve",
            "id": "serve-modelled-1",
            "command": ["python", "-m", "atom.entrypoints.openai.api_server"],
        }
        ports = runner._ports(step, lambda flag: None)
        assert ports["http_listener"]["port"] == int(run_mod.server_default_port())
        assert "api_server.py" in ports["http_listener"]["producer"]
        assert ports["engine_rendezvous"]["port"] == int(run_mod.engine_default_port())
        assert "arg_utils.py" in ports["engine_rendezvous"]["producer"]

    def test_the_engine_default_is_read_from_the_engine(self):
        assert run_mod.engine_default_port() == str(plan_mod.ENGINE_PORT)

    def test_a_client_step_has_only_the_address_it_dials(self, tmp_path):
        runner = _runner(tmp_path, "modelled")
        step = {
            "role": "replay",
            "id": "replay-modelled-1",
            "command": ["python", "scripts/compass/replay.py", "--port", "8000"],
        }
        ports = runner._ports(step, lambda flag: run_mod._after(step["command"], flag))
        assert set(ports) == {"http_dial"}
        assert ports["http_dial"]["port"] == 8000


class TestDerivationIsAttributedToTheRepeatThatSpentIt:
    """A journal covers the whole cell; the cost terms are per repeat.

    `execution_modelled` is the median of the cell's repeats -- what one
    replay cost. So every second placed beside it has to be one repeat's
    worth too, or the ratio built from the two is wrong by the repeat count.
    The journal already carries each derivation's own interval and the side
    record already carries each repeat's windows, so the attribution is an
    intersection, not an allocation.
    """

    def _cell(self, tmp_path, repeats=3):
        cell = tmp_path / "tp2_long"
        cell.mkdir(parents=True, exist_ok=True)
        wall = {"startup": run_mod.WALL_CLOCK, "execution": run_mod.WALL_CLOCK}
        base = 1000.0
        rows = []
        for index in range(1, repeats + 1):
            start = base + (index - 1) * 1000.0
            rows.append(
                {
                    "repeat": index,
                    "execution_s": 10.0,
                    "startup_s": 20.0,
                    "startup_window": [start, start + 20.0],
                    "execution_window": [start + 20.0, start + 30.0],
                }
            )
        (cell / "costs.modelled.json").write_text(
            json.dumps(
                {
                    "cost_schema": run_mod.COSTS_SCHEMA,
                    "clocks": dict(wall, served_window="virtual"),
                    "repeats": repeats,
                    "startup_modelled": 20.0,
                    "execution_modelled": 10.0,
                    "served_window_modelled": 300.0,
                    "per_execution": rows,
                }
            )
        )
        (cell / "costs.real.json").write_text(
            json.dumps(
                {
                    "cost_schema": run_mod.COSTS_SCHEMA,
                    "clocks": dict(wall, served_window="wall"),
                    "repeats": repeats,
                    "startup_real": 100.0,
                    "execution_real": 120.0,
                    "served_window_real": 119.0,
                    "per_execution": [
                        {
                            "repeat": index,
                            "execution_s": 120.0,
                            "startup_s": 100.0,
                            "startup_window": [0.0, 100.0],
                            "execution_window": [100.0, 220.0],
                        }
                        for index in range(1, repeats + 1)
                    ],
                }
            )
        )
        return cell

    def _journal(self, tmp_path, cell, repeats=3):
        """One 10-second derivation inside each repeat's startup window."""
        path = tmp_path / "derivations.jsonl"
        lines = []
        for index in range(1, repeats + 1):
            start = 1000.0 + (index - 1) * 1000.0 + 5.0
            lines.append(json.dumps({"t0": start, "t1": start + 10.0}))
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_each_derivation_is_charged_to_its_own_repeat(self, tmp_path):
        """The defect: three repeats' derivations were merged into one 30 s
        part, which then sat beside a 10 s median execution."""
        cell = self._cell(tmp_path)
        journal = self._journal(tmp_path, cell)
        assert run_mod.main(self._argv(cell, journal)) == 0
        parts = json.loads((cell / "costs.json").read_text())["derivation"]
        by_repeat = {p.get("repeat"): p["seconds"] for p in parts}
        assert by_repeat == {1: 10.0, 2: 10.0, 3: 10.0}
        assert {p["within"] for p in parts} == {"startup_modelled"}

    def test_the_merged_record_says_what_each_repeat_cost(self, tmp_path):
        cell = self._cell(tmp_path)
        assert run_mod.main(self._argv(cell, self._journal(tmp_path, cell))) == 0
        costs = json.loads((cell / "costs.json").read_text())
        assert costs["repeats"] == {"real": 3, "modelled": 3}
        assert costs["execution_by_repeat"]["modelled"] == {
            "1": 10.0,
            "2": 10.0,
            "3": 10.0,
        }
        assert costs["startup_by_repeat"]["modelled"]["2"] == 20.0

    def test_the_gate_built_from_it_is_per_repeat(self, tmp_path):
        """End to end: 120 s of real serving against one repeat's 10 s replay
        plus that repeat's own 10 s of derivation is 6×, not 3×."""
        cell = self._cell(tmp_path)
        assert run_mod.main(self._argv(cell, self._journal(tmp_path, cell))) == 0
        validate = _load("cc_traces_validate")
        got = validate._speedup(
            json.loads((cell / "costs.json").read_text()), reuse_cells=1
        )
        assert got["replay_ratio"] == pytest.approx(6.0)

    def test_a_derivation_in_no_repeats_window_carries_no_repeat(self, tmp_path):
        cell = self._cell(tmp_path)
        path = tmp_path / "derivations.jsonl"
        path.write_text(json.dumps({"t0": 900.0, "t1": 903.0}) + "\n")
        assert run_mod.main(self._argv(cell, path)) == 0
        parts = json.loads((cell / "costs.json").read_text())["derivation"]
        parts = parts if isinstance(parts, list) else [parts]
        assert [(p["within"], p.get("repeat"), p["seconds"]) for p in parts] == [
            (None, None, 3.0)
        ]

    def _mangle(self, cell, rows):
        blob = json.loads((cell / "costs.modelled.json").read_text())
        blob["per_execution"] = rows
        (cell / "costs.modelled.json").write_text(json.dumps(blob))

    def test_a_repeat_listed_twice_is_not_merged(self, tmp_path, capsys):
        """Two rows for repeat 2 and none for repeat 3 is a three-repeat side
        covering two, and the second row silently replaced the first."""
        cell = self._cell(tmp_path)
        rows = json.loads((cell / "costs.modelled.json").read_text())["per_execution"]
        rows[2]["repeat"] = 2
        self._mangle(cell, rows)
        assert run_mod.main(self._argv(cell, self._journal(tmp_path, cell))) == 2
        assert "twice" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_repeat_with_no_duration_is_not_merged(self, tmp_path, capsys):
        cell = self._cell(tmp_path)
        rows = json.loads((cell / "costs.modelled.json").read_text())["per_execution"]
        rows[1]["execution_s"] = None
        self._mangle(cell, rows)
        assert run_mod.main(self._argv(cell, self._journal(tmp_path, cell))) == 2
        assert "execution_s" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_side_short_of_a_repeat_is_not_merged(self, tmp_path, capsys):
        """It says three and describes two; the missing one is not zero."""
        cell = self._cell(tmp_path)
        rows = json.loads((cell / "costs.modelled.json").read_text())["per_execution"]
        self._mangle(cell, rows[:2])
        assert run_mod.main(self._argv(cell, self._journal(tmp_path, cell))) == 2
        assert "3" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def test_a_side_that_kept_no_per_repeat_record_is_not_merged(
        self, tmp_path, capsys
    ):
        """Three repeats and one number is a median; there is nothing to
        match a per-repeat cost against."""
        cell = self._cell(tmp_path)
        blob = json.loads((cell / "costs.modelled.json").read_text())
        del blob["per_execution"]
        (cell / "costs.modelled.json").write_text(json.dumps(blob))
        assert run_mod.main(self._argv(cell, self._journal(tmp_path, cell))) == 2
        assert "per_execution" in capsys.readouterr().err
        assert not (cell / "costs.json").exists()

    def _argv(self, cell, journal):
        return [
            "costs",
            str(cell),
            "--capture",
            "0",
            "--capture-source",
            "capture/manifest.json",
            "--calibration",
            "0",
            "--calibration-source",
            "registry/calibration.json",
            "--derivation-journal",
            str(journal),
            "--load",
            "0",
            "--load-source",
            "server.log",
        ]
