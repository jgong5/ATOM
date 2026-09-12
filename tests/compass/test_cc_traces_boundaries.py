"""Do the plan's commands fit the programs that will receive them?

Everything here is checked against the producer itself rather than against a
stand-in: `replay.py`'s own argument parser, `replay.py`'s own clock rule, and
the engine's own barrier state, read out through the engine's own handler. A
mocked process will accept any flag you invent for it, which is exactly how a
command plan stays plausible and wrong until the night it is run.

`replay.py` imports nothing outside the standard library, so its real functions
can be driven here against a local HTTP server, and the engine modules import
without a device, so the barrier round trip is driven through the real ones.
Nothing in this file starts an engine or touches a device: the serving side is
a stub, and what is being tested is the agreement at the seam.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _core(name: str):
    """The runtime package module, by path: importing `atom` starts an engine."""
    path = ROOT / "atom" / "compass" / "core" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"atom_compass_core_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = _load("cc_traces_plan")
replay_mod = _load("replay")


def _plan_steps(**kw):
    built = plan_mod.cell_steps(
        2,
        "long",
        root="/runs",
        oracle="transfer",
        options=(),
        port=8000,
        repeats=3,
        target="/w/target.json",
        **kw,
    )
    return built["steps"]


def _commands(role):
    return [s["command"] for s in _plan_steps() if s["role"] == role and s["command"]]


@pytest.fixture(scope="module")
def replay_help():
    """`replay.py --help`, from the program itself."""
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts/compass/replay.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


class Stub(BaseHTTPRequestHandler):
    """A server that answers the two endpoints `replay.py` reads."""

    provenance: ClassVar[dict] = {}
    #: Every body `replay.py` posted, in the order this server received them.
    #: The arrival protocol lives in those bodies and nowhere else, so this is
    #: what lets it be tested at the seam without an engine behind it.
    posted: ClassVar[list] = []
    #: What the drain endpoint answers. `replay.py` reads the engine's barrier
    #: state off this same response, so a test sets it here to stand for what
    #: the engine observed.
    requests_reply: ClassVar[dict] = {"count": 0, "requests": []}

    def do_POST(self):  # BaseHTTPRequestHandler names it this way
        length = int(self.headers.get("Content-Length") or 0)
        sent = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/compass/requests"):
            # `replay.py` drains the engine's record store with a POST at the
            # end of a run. Recording it here would count it as a request.
            body = json.dumps(type(self).requests_reply).encode()
        else:
            type(self).posted.append(sent)
            # A completion shaped like the one a server really returns: the
            # requested number of output tokens, counted in `usage`, and the
            # finish reason that goes with having produced them. `replay.py`
            # reads all three to decide whether the workload completed, and a
            # stub that answered without them would make every run here look
            # like one that came back short.
            produced = int(sent.get("max_tokens") or 0)
            body = json.dumps(
                {
                    "choices": [{"text": "x " * produced,
                                 "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": produced},
                }
            ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # BaseHTTPRequestHandler names it this way
        if self.path.startswith("/compass/provenance"):
            body = json.dumps(type(self).provenance).encode()
        elif self.path.startswith("/compass/requests"):
            body = json.dumps(type(self).requests_reply).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def served():
    server = HTTPServer(("127.0.0.1", 0), Stub)
    Stub.provenance = {}
    Stub.posted = []
    Stub.requests_reply = {"count": 0, "requests": []}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestThePlanSpeaksReplaysLanguage:
    """Every flag the plan emits, checked against the parser that will read
    it. An invented flag is an argparse error 40 minutes into a GPU lease."""

    def test_every_replay_flag_exists(self, replay_help):
        flags = {
            part
            for command in _commands("replay")
            for part in command
            if part.startswith("--")
        }
        assert flags
        missing = sorted(f for f in flags if f not in replay_help)
        assert not missing, f"replay.py does not accept {missing}"

    def test_the_flags_the_protocol_turns_on_are_real(self, replay_help):
        for flag in (
            "--pace",
            "--prepare",
            "--prepare-out",
            "--check-lengths",
            "--trace",
            "--out",
            "--time-scale",
        ):
            assert flag in replay_help

    def test_the_real_side_paces_and_prepares_and_the_modelled_side_does_not(self):
        real = [c for c in _commands("replay") if "real.r1.json" in " ".join(c)]
        modelled = [c for c in _commands("replay") if "modelled.r1.json" in " ".join(c)]
        assert real and modelled
        assert "--pace" in real[0] and "--prepare" in real[0]
        assert "--pace" not in modelled[0] and "--prepare" not in modelled[0]

    def test_the_trace_the_plan_names_is_a_registered_workload(self):
        """Not that the file is there -- it is reproduced, not committed --
        but that the plan names something the repository actually registers.
        """
        for command in _commands("replay"):
            trace = ROOT / command[command.index("--trace") + 1]
            manifest = trace.with_name(
                trace.name.replace(".jsonl", ".manifest.json"))
            assert manifest.exists(), manifest
            assert not trace.exists() or trace.is_file()

    def test_the_refusal_exit_the_harness_expects_is_the_one_replay_uses(self):
        run_mod = _load("cc_traces_run")
        source = (ROOT / "scripts/compass/replay.py").read_text()
        assert "return 3" in source
        assert run_mod.REFUSAL_EXIT == 3


class TestTheClockRuleIsReadFromTheServer:
    """`_clock_of` is what makes replay.py refuse a warmed predictor. The
    harness gates on the same field, so the two must read it the same way."""

    def _clock(self, base, provenance, stub):
        stub.provenance = provenance
        return replay_mod._clock_of(base, 10.0)

    def test_a_predicting_server_on_a_virtual_clock_reads_as_virtual(self, served):
        base, stub = served
        said = {"compass": {"enabled": True, "mode": "predict", "virtual_clock": True}}
        assert self._clock(base, said, stub) == "virtual"

    def test_a_measuring_server_reads_as_wall(self, served):
        base, stub = served
        said = {"compass": {"enabled": True, "mode": "measure", "virtual_clock": True}}
        assert self._clock(base, said, stub) == "wall"

    def test_a_predictor_reporting_no_virtual_clock_reads_as_wall(self, served):
        """Which is why the harness refuses it: at 'wall' the preparation
        refusal never fires, and a warmed predictor would be measured."""
        base, stub = served
        said = {"compass": {"enabled": True, "mode": "predict", "virtual_clock": False}}
        assert self._clock(base, said, stub) == "wall"

    def test_a_server_with_compass_off_reads_as_wall(self, served):
        base, stub = served
        said = {"compass": {"enabled": False, "mode": "predict", "virtual_clock": True}}
        assert self._clock(base, said, stub) == "wall"

    def test_a_server_that_cannot_be_reached_is_not_a_virtual_clock(self):
        assert replay_mod._clock_of("http://127.0.0.1:1", 0.5) is None

    def test_the_harness_gates_on_the_same_field(self, served):
        """Not a paraphrase of it: the name is the contract."""
        run_source = (ROOT / "scripts/compass/cc_traces_run.py").read_text()
        replay_source = (ROOT / "scripts/compass/replay.py").read_text()
        assert 'compass.get("virtual_clock")' in replay_source
        assert '"virtual_clock"' in run_source


class TestTheFieldNamesTheHarnessReadsAreTheServers:
    """The server cannot be imported without a device, so its source is read.

    A weaker check than calling it, and named so nobody mistakes it for the
    stronger one -- but it still fails the day a field is renamed, which is
    the failure this is here to catch.
    """

    ENDPOINT = ROOT / "atom/entrypoints/openai/api_server.py"

    def _endpoint_source(self):
        source = self.ENDPOINT.read_text()
        start = source.index("async def compass_provenance")
        return source[start : source.index("\n@app.", start)]

    @pytest.mark.parametrize(
        "field",
        [
            "server_code_sha256",
            "tensor_parallel_size",
            "model",
            "visible_devices",
            "calibration_sha256",
        ],
    )
    def test_the_top_level_fields_are_there(self, field):
        assert f'"{field}"' in self._endpoint_source()

    @pytest.mark.parametrize(
        "field", ["enabled", "mode", "virtual_clock", "oracle", "oracle_options"]
    )
    def test_the_compass_fields_are_there(self, field):
        assert f'"{field}"' in self._endpoint_source()

    def test_the_manifest_fields_the_harness_reads_are_replays(self):
        source = (ROOT / "scripts/compass/replay.py").read_text()
        for field in (
            "paced",
            "prepare",
            "trace_sha256",
            "server",
            "drained",
            "store_empty_after_drain",
            "drained_records",
        ):
            assert f'"{field}"' in source, field


class TestTheModelledEntryPointIsTheOneThatExists:
    def test_the_plan_starts_the_replay_server_with_the_flag_it_requires(self):
        serves = [
            s["command"]
            for s in _plan_steps()
            if s["role"] == "serve" and s.get("side") == "modelled"
        ]
        assert serves
        for command in serves:
            assert command[1].endswith("scripts/compass/replay_server.py")
            assert "--compass-replay-target" in command

    def test_that_flag_is_the_one_the_entry_point_looks_for(self):
        source = (ROOT / "scripts/compass/replay_server.py").read_text()
        assert '"--compass-replay-target"' in source

    def test_the_real_side_starts_atoms_own_server(self):
        serves = [
            s["command"]
            for s in _plan_steps()
            if s["role"] == "serve" and s.get("side") == "real"
        ]
        assert serves
        for command in serves:
            assert command[1:3] == ["-m", "atom.entrypoints.openai.api_server"]


class TestWhoAnsweredIsSomethingTheServerReports:
    """The harness refuses a reply it cannot attribute to a process it started.

    That refusal is only meaningful if the server actually reports which
    process it is, and if the numbers it reports are the ones a caller can
    read back out of `/proc`. Both are checked here against the real module
    rather than against a fake, because a fake would report whatever shape the
    harness happens to want.
    """

    ENDPOINT = ROOT / "atom/entrypoints/openai/api_server.py"
    SHARED = ROOT / "atom/compass/core/process_identity.py"

    def _endpoint_source(self):
        source = self.ENDPOINT.read_text()
        start = source.index("async def compass_provenance")
        return source[start : source.index("\n@app.", start)]

    def test_the_endpoint_says_which_process_answered(self):
        assert '"server_process"' in self._endpoint_source()

    def test_the_server_reads_it_from_the_shared_module(self):
        """Not a second copy of the `/proc` parsing, for the same reason the
        identity rule is not copied: two readings can disagree."""
        source = self.ENDPOINT.read_text()
        assert "from atom.compass.core import process_identity" in source
        assert "process_identity.identity()" in source
        assert "/proc/" not in source, "the endpoint should not parse /proc itself"

    def test_the_shared_reading_gives_what_the_harness_compares(self):
        shared = _core("process_identity")
        said = shared.identity()
        for field in (
            "pid",
            "ppid",
            "host",
            "boot_id",
            "start_ticks",
            "ticks_per_second",
        ):
            assert field in said, field
        import os

        assert said["pid"] == os.getpid()
        assert said["host"] == os.uname().nodename
        assert said["start_ticks"] == shared.start_ticks("self")
        assert isinstance(said["start_ticks"], int)

    def test_the_harness_reads_the_very_same_file(self):
        run_mod = _load("cc_traces_run")
        assert Path(run_mod.process_identity.__file__).resolve() == self.SHARED

    def test_two_processes_do_not_share_an_identity(self):
        """The property the whole check rests on."""
        shared = _core("process_identity")
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            import os

            assert other.pid != os.getpid()
            theirs = shared.start_ticks(other.pid)
            assert isinstance(theirs, int)
            assert shared.parent_of(other.pid) == os.getpid()
        finally:
            other.kill()
            other.wait()

    def test_a_pid_nobody_is_using_reads_as_unknown(self):
        """Which is what makes a stale pid refusable rather than an exception."""
        shared = _core("process_identity")
        assert shared.start_ticks(2**22) is None
        assert shared.parent_of(2**22) is None

    def test_asking_who_answered_does_not_import_the_engine(self):
        """A machine with no device still has to be able to check this."""
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util, sys\n"
                    "spec = importlib.util.spec_from_file_location('p', sys.argv[1])\n"
                    "m = importlib.util.module_from_spec(spec)\n"
                    "spec.loader.exec_module(m)\n"
                    "assert m.identity()['pid'] > 0\n"
                    "print([n for n in sys.modules if n.split('.')[0] == 'atom'])\n"
                ),
                str(self.SHARED),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]", done.stdout


class TestTheArrivalBarrierIsOnlyArmedByTheSideThatCanFillIt:
    """`compass_workload_size` is a promise the *declared* path keeps.

    The scheduler holds the virtual clock until that many requests are waiting,
    so that no arrival can turn up late and be stamped retroactively. An
    unpaced client keeps the promise trivially: every request is posted as fast
    as the socket allows, before any step is decided. A paced client cannot --
    it delivers requests across the trace's own span on purpose, so the count
    arms a barrier that submission will not fill within
    `ARRIVAL_BARRIER_TIMEOUT_S`. The barrier then opens anyway and says so, and
    every latency from that point on is invalid.

    That is not hypothetical: a 62-request cc_pilot development run was sent
    with `--pace` against a predictor, 11 had arrived when the 120s ran out,
    and the first answered batch landed after the barrier had already given up.
    These drive `replay.py::main` against a stub, so what is checked is the
    bodies the program really posts rather than a description of them.
    """

    VIRTUAL: ClassVar[dict] = {
        "compass": {"enabled": True, "mode": "predict", "virtual_clock": True}
    }
    WALL: ClassVar[dict] = {
        "compass": {"enabled": True, "mode": "measure", "virtual_clock": False}
    }

    def _trace(self, tmp_path, count=3):
        path = tmp_path / "trace.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {"arrival_s": i * 0.01, "input_tokens": 8, "output_tokens": 1}
                )
                + "\n"
                for i in range(count)
            )
        )
        return path

    def _run(self, base, tmp_path, *extra, count=3):
        out = tmp_path / "out.json"
        return (
            replay_mod.main(
                [
                    "--port",
                    base.rsplit(":", 1)[1],
                    "--model",
                    "m",
                    "--trace",
                    str(self._trace(tmp_path, count)),
                    "--out",
                    str(out),
                    "--timeout",
                    "20",
                    *extra,
                ]
            ),
            out,
        )

    def test_a_declared_run_still_arms_the_barrier_with_its_own_size(
        self, served, tmp_path
    ):
        """The protocol the predictive side depends on, unchanged."""
        base, stub = served
        stub.provenance = self.VIRTUAL
        code, _ = self._run(base, tmp_path, count=5)
        assert code == 0
        assert len(stub.posted) == 5
        assert all(b["compass_workload_size"] == 5 for b in stub.posted)
        assert sorted(b["compass_arrival"] for b in stub.posted) == [
            0.0,
            0.01,
            0.02,
            0.03,
            0.04,
        ]

    def test_a_paced_run_declares_no_count_it_cannot_honour(self, served, tmp_path):
        """Paced against a *real* clock is the supported combination, and it
        must not leave a barrier armed behind it."""
        base, stub = served
        stub.provenance = self.WALL
        code, _ = self._run(base, tmp_path, "--pace")
        assert code == 0
        assert len(stub.posted) == 3
        assert not any("compass_workload_size" in b for b in stub.posted)
        assert not any("compass_arrival" in b for b in stub.posted)

    def test_pacing_a_predictor_is_refused_before_anything_is_sent(
        self, served, tmp_path
    ):
        """The combination that produced the invalid run. Refused at the same
        exit the harness already treats as a refusal, and refused early: a run
        that has posted half a workload before noticing has already moved the
        engine's clock."""
        base, stub = served
        stub.provenance = self.VIRTUAL
        code, out = self._run(base, tmp_path, "--pace")
        assert code == 3
        assert stub.posted == []
        assert not out.exists(), "a refused run must not leave a result behind"

    def test_the_refusal_exit_is_the_one_the_harness_reads_as_a_failure(self):
        run_mod = _load("cc_traces_run")
        assert run_mod.REFUSAL_EXIT == 3

    def test_a_server_that_says_nothing_is_not_treated_as_a_predictor(
        self, served, tmp_path
    ):
        """`_clock_of` returns None for a server with no provenance endpoint,
        and a real engine older than this protocol is exactly that. Refusing
        it would make the real side of every pair unrunnable."""
        base, stub = served
        stub.provenance = {}
        code, _ = self._run(base, tmp_path, "--pace")
        assert code == 0

    def test_the_declared_count_is_the_whole_workload(self, served, tmp_path):
        """Not the number that happened to be posted by then. A barrier armed
        with fewer than the workload opens early, which is the same invalid
        run reached from the other side."""
        base, stub = served
        stub.provenance = self.VIRTUAL
        code, _ = self._run(base, tmp_path, count=7)
        assert code == 0
        assert {b["compass_workload_size"] for b in stub.posted} == {7}

    def test_the_plans_modelled_side_is_the_declared_one(self):
        """The wiring, checked against the plan that emits it: this whole
        failure came from a hand-typed command, not from the plan."""
        modelled = [c for c in _commands("replay") if "modelled.r1.json" in " ".join(c)]
        real = [c for c in _commands("replay") if "real.r1.json" in " ".join(c)]
        assert modelled and real
        assert "--pace" not in modelled[0]
        assert "--pace" in real[0]


class TestABarrierThatActuallyTimedOutFailsTheRun:
    """The state the engine really observed, carried to the harness.

    `compare.py` has refused a manifest whose `arrival_barrier_timed_out` is
    true for some time, but nothing ever wrote that key: the scheduler sets the
    flag on itself in the EngineCore process and no endpoint exported it, so
    the check could not fire. These check the producer end -- that
    `/compass/requests` is read for it, that it reaches the manifest, and that
    a run which timed out fails instead of reporting "0 failed".

    Three states throughout. A barrier that could not be read is not a barrier
    that held: unknown is reported and never silently promoted to either
    answer.
    """

    def _trace(self, tmp_path, count=3):
        path = tmp_path / "trace.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {"arrival_s": i * 0.01, "input_tokens": 8, "output_tokens": 1}
                )
                + "\n"
                for i in range(count)
            )
        )
        return path

    def _run(self, base, tmp_path, barrier):
        """One unpaced run against a stub whose drain reports `barrier`."""
        out = tmp_path / "out.json"
        Stub.requests_reply = {"count": 0, "requests": [], **barrier}
        code = replay_mod.main(
            [
                "--port",
                base.rsplit(":", 1)[1],
                "--model",
                "m",
                "--trace",
                str(self._trace(tmp_path)),
                "--out",
                str(out),
                "--timeout",
                "20",
            ]
        )
        return code, json.loads(out.read_text())["run"]

    def test_a_barrier_that_held_is_a_passing_run(self, served, tmp_path):
        base, _ = served
        code, run = self._run(base, tmp_path, {"arrival_barrier": {"timed_out": False}})
        assert code == 0
        assert run["arrival_barrier_timed_out"] is False

    def test_a_barrier_that_timed_out_fails_the_run(self, served, tmp_path):
        base, _ = served
        code, run = self._run(
            base,
            tmp_path,
            {
                "arrival_barrier": {
                    "timed_out": True,
                    "ranks": [{"detail": {"arrived": 11, "expected": 62}}],
                }
            },
        )
        assert code != 0, "a run whose barrier timed out must not exit 0"
        assert run["arrival_barrier_timed_out"] is True

    def test_the_failed_run_still_leaves_its_evidence(self, served, tmp_path):
        """Failing by deleting the artifact would leave only a log line, which
        is the situation this whole field exists to end."""
        base, _ = served
        code, run = self._run(base, tmp_path, {"arrival_barrier": {"timed_out": True}})
        assert code != 0
        assert run["arrival_barrier"] == {"timed_out": True}

    def test_the_field_compare_refuses_on_is_the_field_replay_writes(
        self, served, tmp_path
    ):
        """Named once. A manifest key nothing writes is how this check spent
        its whole life so far being unable to fire."""
        base, _ = served
        _, run = self._run(base, tmp_path, {"arrival_barrier": {"timed_out": True}})
        compare_source = (ROOT / "scripts/compass/compare.py").read_text()
        assert 'm.get("arrival_barrier_timed_out")' in compare_source
        assert "arrival_barrier_timed_out" in run

    def test_an_unreadable_barrier_is_unknown_and_neither_answer(
        self, served, tmp_path
    ):
        """A server too old to report one, or a round trip that failed. The
        run is not refused -- nothing says it was bad -- but nothing may read
        it as verified either."""
        base, _ = served
        code, run = self._run(base, tmp_path, {})
        assert code == 0
        assert run["arrival_barrier_timed_out"] is None

    def test_an_unknown_barrier_is_not_truthy_to_the_check(self, served, tmp_path):
        """`compare.py` refuses on truthiness, so unknown must not be a dict
        or a non-empty string that happens to be true."""
        base, _ = served
        _, run = self._run(base, tmp_path, {"arrival_barrier": {"timed_out": None}})
        assert not run["arrival_barrier_timed_out"]
        assert run["arrival_barrier_timed_out"] is None


class TestTheServerSideOfTheBarrierReading:
    """The whole round trip, driven through the real code on both ends.

    The scheduler's barrier is a piece of its own state and needs no device to
    exercise, so nothing here is a stand-in for the producer: the state is put
    there by `Scheduler._arrival_barrier_unmet` itself, read out by the real
    `EngineUtilityHandler` through its real command table, aggregated by the
    real `LLMEngine` method, and served by the real endpoint helper. A test
    that handed a fake scheduler the attribute name the handler expects would
    agree with the implementation by construction and would keep agreeing
    after the scheduler stopped writing it.

    An engine is still never started and no device is touched. What is
    constructed is one `Scheduler.__new__` with the four fields the barrier
    reads -- building a real one needs a model, a device and a KV cache, none
    of which the barrier consults.
    """

    COMMAND = "get_compass_arrival_barrier"

    @pytest.fixture
    def virtual(self):
        """A real virtual clock installed for the duration of a test.

        `_arrival_barrier_unmet` decides whether there is anything to wait for
        by asking the installed clock for an epoch, so the clock has to be the
        real one: a wall clock opens the barrier immediately.
        """
        from atom.utils.clock import VirtualClock, get_clock, set_clock

        previous = get_clock()
        set_clock(VirtualClock(epoch=1_700_000_000.0))
        yield
        set_clock(previous)

    @staticmethod
    def _scheduler(arrived=0, declared=None, waited=0.0):
        """A scheduler with `arrived` of `declared` requests waiting.

        `waited` is how long submission has been going on, in real seconds,
        which is what the barrier's timeout is measured against.
        """
        import time
        from types import SimpleNamespace

        from atom.model_engine.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._arrival_barrier_open = False
        scheduler._arrival_barrier_since = time.monotonic() - waited if waited else None
        scheduler.waiting = [
            SimpleNamespace(compass_workload_size=declared) for _ in range(arrived)
        ]
        return scheduler

    @classmethod
    def _consulted(cls, **kwargs):
        """A scheduler that has been through its barrier check once.

        The state the handler reports only comes into existence when the real
        method runs: a scheduler nobody consulted has no reading, which is
        itself one of the cases below.
        """
        scheduler = cls._scheduler(**kwargs)
        scheduler._arrival_barrier_unmet()
        return scheduler

    @classmethod
    def _ask(cls, scheduler):
        """What the utility handler answers for this scheduler.

        Dispatched by command name through the real `_UTILITY_HANDLERS` table,
        because the name is half of what can go wrong: a handler nobody can
        reach answers nothing, and the caller reads that as unknown forever.
        """
        import queue

        from atom.model_engine.engine_utility import EngineUtilityHandler

        output = queue.Queue()
        handler = EngineUtilityHandler(
            runner_mgr=None, output_queue=output, scheduler=scheduler
        )
        handler._execute_utility_command(cls.COMMAND, {})
        kind, payload = output.get_nowait()
        assert kind == "UTILITY_RESPONSE"
        assert payload["cmd"] == cls.COMMAND
        return payload

    @classmethod
    def _aggregate(cls, responses):
        """What the engine makes of one answer per rank."""
        from types import SimpleNamespace

        from atom.model_engine.llm_engine import LLMEngine

        sent = {}

        def broadcast(command, timeout=None):
            sent["command"] = command
            return responses

        engine = SimpleNamespace(
            core_mgr=SimpleNamespace(broadcast_utility_command_sync=broadcast)
        )
        reading = LLMEngine.get_compass_arrival_barrier(engine)
        # The command the engine broadcasts has to be one the handler table
        # answers, or every rank stays silent and every run reads unknown.
        assert sent["command"] == cls.COMMAND
        return reading

    def test_a_barrier_that_actually_timed_out_reaches_the_caller(self, virtual):
        """The failing case end to end: two of five arrived, submission ran
        past the timeout, the scheduler gave up and ran anyway."""
        from atom.model_engine.scheduler import Scheduler

        scheduler = self._scheduler(
            arrived=2, declared=5, waited=Scheduler.ARRIVAL_BARRIER_TIMEOUT_S + 1.0
        )
        assert scheduler._arrival_barrier_unmet() is False  # gave up, not met
        assert scheduler._arrival_barrier_open is True

        answer = self._ask(scheduler)["result"]
        assert answer["timed_out"] is True
        assert answer["detail"]["arrived"] == 2
        assert answer["detail"]["expected"] == 5
        assert answer["detail"]["timeout_s"] == Scheduler.ARRIVAL_BARRIER_TIMEOUT_S

        reading = self._aggregate([self._ask(scheduler)])
        assert reading["timed_out"] is True

    def test_a_barrier_still_waiting_has_not_timed_out(self, virtual):
        """Held, not failed. The engine is holding the virtual clock exactly
        as intended, and a reading taken mid-wait must not say otherwise."""
        scheduler = self._scheduler(arrived=2, declared=5)
        assert scheduler._arrival_barrier_unmet() is True
        assert self._ask(scheduler)["result"]["timed_out"] is False

    def test_a_barrier_the_workload_filled_reports_false(self, virtual):
        """The passing case: everything declared turned up, the barrier
        opened on its own terms, and the run is verifiable."""
        scheduler = self._scheduler(arrived=5, declared=5)
        assert scheduler._arrival_barrier_unmet() is False
        assert scheduler._arrival_barrier_open is True
        assert self._aggregate([self._ask(scheduler)])["timed_out"] is False

    def test_a_scheduler_that_never_reached_the_barrier_is_unknown(self):
        """Never consulted, so there is nothing to report -- and unknown is
        what that is. Reporting False here would claim a barrier held that was
        never reached."""
        answer = self._ask(self._scheduler())["result"]
        assert answer["timed_out"] is None
        assert answer["why"]
        assert self._aggregate([answer])["timed_out"] is None

    def test_a_rank_with_no_scheduler_is_unknown_rather_than_good(self):
        answer = self._ask(None)["result"]
        assert answer["timed_out"] is None
        assert answer["why"]

    def test_a_rank_that_cannot_answer_does_not_vote_for_a_good_run(self, virtual):
        """True beats unknown beats False. One silent rank leaves the reading
        unknown however many ranks answered, and one rank that timed out
        decides it however many did not."""
        good = self._ask(self._consulted(arrived=5, declared=5))
        silent = self._ask(self._scheduler())
        from atom.model_engine.scheduler import Scheduler

        bad = self._ask(
            self._consulted(
                arrived=1,
                declared=4,
                waited=Scheduler.ARRIVAL_BARRIER_TIMEOUT_S + 1.0,
            )
        )
        assert self._aggregate([good, good])["timed_out"] is False
        assert self._aggregate([good, silent])["timed_out"] is None
        assert self._aggregate([good, silent, bad])["timed_out"] is True
        assert self._aggregate([])["timed_out"] is None

    def test_the_ranks_that_answered_are_reported_alongside_the_verdict(self, virtual):
        """The verdict is a summary; the run has to be able to show which rank
        said what, or an unknown reading cannot be chased down."""
        reading = self._aggregate(
            [self._ask(self._consulted(arrived=5, declared=5)), self._ask(None)]
        )
        assert [rank["timed_out"] for rank in reading["ranks"]] == [False, None]

    def test_the_endpoint_answers_with_the_engine_s_reading(self, monkeypatch):
        """The server helper the harness drains through, against a stand-in
        engine -- the one seam where a real object cannot be built here."""
        from types import SimpleNamespace

        import atom.entrypoints.openai.api_server as server
        from atom.entrypoints.openai.api_server import _compass_arrival_barrier

        reading = {"timed_out": False, "ranks": [{"timed_out": False}]}
        monkeypatch.setattr(
            server,
            "engine",
            SimpleNamespace(get_compass_arrival_barrier=lambda timeout: reading),
        )
        assert _compass_arrival_barrier() == reading

    def test_an_endpoint_that_cannot_reach_the_engine_says_unknown(self, monkeypatch):
        """A reading that failed must not fail the run it describes, and must
        not pass it either."""
        from types import SimpleNamespace

        import atom.entrypoints.openai.api_server as server

        def boom(timeout):
            raise TimeoutError("no response from rank 0")

        monkeypatch.setattr(server, "engine", None)
        assert server._compass_arrival_barrier()["timed_out"] is None
        monkeypatch.setattr(
            server, "engine", SimpleNamespace(get_compass_arrival_barrier=boom)
        )
        unreadable = server._compass_arrival_barrier()
        assert unreadable["timed_out"] is None
        assert "no response from rank 0" in unreadable["why"]
