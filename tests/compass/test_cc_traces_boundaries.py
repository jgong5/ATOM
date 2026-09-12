"""Do the plan's commands fit the programs that will receive them?

Everything here is checked against the producer itself rather than against a
stand-in: `replay.py`'s own argument parser, `replay.py`'s own clock rule, and
the source of the server endpoint whose field names the harness reads. A mocked
process will accept any flag you invent for it, which is exactly how a command
plan stays plausible and wrong until the night it is run.

`replay.py` imports nothing outside the standard library, so its real functions
can be driven here against a local HTTP server. Nothing in this file starts an
engine or touches a device; the server the engine would be is a stub, and what
is being tested is the agreement at the seam, not the engine behind it.
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

    def do_POST(self):  # BaseHTTPRequestHandler names it this way
        length = int(self.headers.get("Content-Length") or 0)
        sent = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/compass/requests"):
            # `replay.py` drains the engine's record store with a POST at the
            # end of a run. Recording it here would count it as a request.
            body = json.dumps({"count": 0, "requests": []}).encode()
        else:
            type(self).posted.append(sent)
            body = json.dumps(
                {"choices": [{"text": "x"}], "usage": {"prompt_tokens": 0}}
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
            body = json.dumps({"count": 0, "requests": []}).encode()
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
