"""What the orchestrator has to get right before it costs a GPU hour.

The phases themselves need two servers and a GPU, so they are not what these
cover. What is covered is everything that decides whether a five-phase run is
worth reading afterwards: that the two sides are served the same engine, that
the real side is served on a clock that can be paced, that a server is never
left holding the GPU, and that a gate which did not run is not reported as a
gate that passed.
"""

import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "compass" / "replay_validate.py")


def _load():
    spec = importlib.util.spec_from_file_location("replay_validate", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rv = _load()


def _args(**over):
    argv = ["--model", "M", "--trace", "t.jsonl", "--out-dir", "o"]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return rv.build_parser().parse_args(argv)


class TestBothSidesGetTheSameEngine:
    """A setting that differs between the two runs comes out as model error,
    and nothing downstream can tell the difference."""

    def test_shared_flags_are_built_once(self):
        flags = rv._engine_flags(_args(tp=2, max_model_len=4096,
                                      max_num_seqs=8))
        assert flags == ["-tp", "2", "--block-size", "16",
                         "--max-model-len", "4096", "--max-num-seqs", "8",
                         "--max-num-batched-tokens", "16384"]

    def test_an_unset_engine_flag_is_left_to_the_engine(self):
        # Not passed as "None", which the engine would reject, and not
        # substituted with a default this script invented.
        assert "--max-num-seqs" not in rv._engine_flags(_args())

    def test_the_native_block_size_stays_at_the_calibrated_sixteen(self):
        # The corpus hashes at 64. Following it here would invalidate every
        # calibration this project has measured.
        assert _args().block_size == 16


class TestTheRealSideCanActuallyBePaced:
    """--pace against a virtual clock is a deadlock, not a slow run: the
    arrival barrier holds every request for 120s while the client sleeps out
    the trace's own span."""

    def test_measure_mode_leaves_the_clock_real(self):
        from atom.compass.config import CompassConfig
        config = CompassConfig(enabled=True, mode="measure",
                               measure_out="steps.jsonl")
        assert config.virtual_clock is False

    def test_predict_mode_keeps_it_virtual(self):
        from atom.compass.config import CompassConfig
        assert CompassConfig(enabled=True, mode="predict").virtual_clock is True


class TestWarningsSurvive:
    """The oracle's extrapolation warning is the one safeguard against a
    confident number from outside the table. Captured output that is never
    scanned leaves it inert."""

    def test_a_warning_in_captured_output_is_echoed(self, capsys):
        rv._echo_warnings("INFO starting\n"
                          "x ATOMCompass WARNING: extrapolating 3.1x\n")
        assert "extrapolating 3.1x" in capsys.readouterr().out

    def test_the_same_warning_ten_thousand_times_is_printed_once(self, capsys):
        # It fires per step. Echoing each one buries the warning that is new.
        rv._echo_warnings("\n".join(
            ["ATOMCompass WARNING: outside the hull"] * 10000))
        assert capsys.readouterr().out.count("outside the hull") == 1

    def test_two_different_warnings_both_appear(self, capsys):
        rv._echo_warnings("ATOMCompass WARNING: a\nATOMCompass WARNING: b\n")
        out = capsys.readouterr().out
        assert "WARNING: a" in out and "WARNING: b" in out

    def test_an_ordinary_line_is_not_echoed(self, capsys):
        rv._echo_warnings("WARNING this is some other component\n")
        assert capsys.readouterr().out == ""


class TestAPhaseThatFailedIsNotAPhaseThatRan:
    def test_a_nonzero_phase_stops_the_run(self):
        with pytest.raises(SystemExit, match="calibration failed"):
            rv._run([sys.executable, "-c", "raise SystemExit(3)"], "calibration")

    def test_a_failing_phase_shows_its_output(self, capsys):
        with pytest.raises(SystemExit):
            rv._run([sys.executable, "-c",
                     "import sys; sys.stderr.write('allocator refused\\n');"
                     "raise SystemExit(1)"], "real replay")
        assert "allocator refused" in capsys.readouterr().err


_FAKE_SERVER = '''
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--server-port") + 1])


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


print("engine started")
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

_DIES = '''
import sys
sys.stderr.write("ATOMCompass WARNING: no KV cache fits\\n")
raise SystemExit(1)
'''

_NEVER_HEALTHY = '''
import time
time.sleep(600)
'''


def _fake(tmp_path, body, name="fake"):
    """A stand-in for `python -m atom.entrypoints.openai_server`.

    Served spawns argv[0] with the server's own arguments, so anything that
    ignores the ones it does not know and binds the port it is told to will do.
    """
    script = tmp_path / f"{name}.py"
    script.write_text(body)
    launcher = tmp_path / f"{name}_python"
    launcher.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" '
                        f'"{script}" "$@"\n')
    launcher.chmod(0o755)
    return str(launcher)


def _served(tmp_path, body, name="fake", timeout=30.0):
    return rv.Served(_fake(tmp_path, body, name), "M", rv._free_port(),
                     tmp_path / f"{name}.log", [], timeout)


class TestTheServerIsNeverLeftHoldingTheGpu:
    """The failure that matters is the one where the client raises. A leaked
    engine keeps the GPU, and the next phase fails to allocate for a reason
    that looks nothing like the real one."""

    def test_it_stops_when_the_block_ends(self, tmp_path):
        served = _served(tmp_path, _FAKE_SERVER)
        with served:
            assert served._healthy()
        assert served.proc.poll() is not None

    def test_it_stops_when_the_body_raises(self, tmp_path):
        served = _served(tmp_path, _FAKE_SERVER)
        with pytest.raises(RuntimeError):
            with served:
                raise RuntimeError("replay blew up")
        assert served.proc.poll() is not None

    def test_a_server_that_dies_is_reported_with_its_log(self, tmp_path):
        with pytest.raises(SystemExit, match="died during startup"):
            with _served(tmp_path, _DIES, name="dies"):
                pass

    def test_a_server_that_never_answers_is_not_waited_on_forever(self, tmp_path):
        served = _served(tmp_path, _NEVER_HEALTHY, name="hangs", timeout=2.0)
        start = time.monotonic()
        with pytest.raises(SystemExit, match="never became healthy"):
            with served:
                pass
        assert time.monotonic() - start < 30
        assert served.proc.poll() is not None

    def test_warnings_in_the_server_log_reach_the_reader(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            with _served(tmp_path, _DIES, name="dies2"):
                pass
        # The engine logs to its own file, so without this pass they would be
        # written down and never read.
        assert "no KV cache fits" in capsys.readouterr().out


class TestPortsAreNotGuessed:
    def test_the_os_picks_and_the_port_is_free(self):
        port = rv._free_port()
        with socket.socket() as s:
            s.bind(("", port))  # would raise if something already held it

    def test_two_calls_do_not_collide(self):
        # Host networking is shared with about twenty containers, including an
        # earlier run of this script.
        assert rv._free_port() != rv._free_port()


class TestAGateThatDidNotRunIsNotAGateThatPassed:
    """A skipped real phase leaves no step table. Reporting that as coverage
    would turn the one check against extrapolation into a no-op."""

    def _plumbed(self, tmp_path, monkeypatch, *, blocking=(), steps=None,
                 coverage_rc=0):
        work = tmp_path / "out"
        work.mkdir()
        (work / "real.json").write_text("{}")
        (work / "compare.json").write_text(
            json.dumps({"blocking": list(blocking)}))
        if steps is not None:
            (work / "real_steps.jsonl").write_text(steps)

        calls = []

        def fake_run(cmd, *a, **kw):
            calls.append(cmd)
            name = Path(cmd[1]).name if len(cmd) > 1 else ""
            rc = coverage_rc if name == "coverage.py" else 0
            return subprocess.CompletedProcess(cmd, rc, "", "")

        monkeypatch.setattr(rv.subprocess, "run", fake_run)
        monkeypatch.setattr(rv, "_run", lambda cmd, label: calls.append(cmd))
        monkeypatch.setattr(rv, "Served",
                            lambda *a, **kw: _NullServer())
        code = rv.main(["--model", "M", "--trace", str(tmp_path / "t.jsonl"),
                        "--out-dir", str(work), "--table", str(tmp_path / "s"),
                        "--skip-real"])
        return code, calls

    def test_no_step_table_means_the_verdict_is_withheld(self, tmp_path,
                                                         monkeypatch):
        code, _ = self._plumbed(tmp_path, monkeypatch, steps=None)
        assert code == 1

    def test_a_failing_coverage_run_withholds_it_too(self, tmp_path, monkeypatch):
        code, _ = self._plumbed(tmp_path, monkeypatch, steps="{}\n",
                                coverage_rc=1)
        assert code == 1

    def test_the_comparison_still_runs_so_the_report_exists(self, tmp_path,
                                                            monkeypatch):
        _, calls = self._plumbed(tmp_path, monkeypatch, steps=None)
        names = [Path(c[1]).name for c in calls if len(c) > 1]
        # Withheld, not skipped: finding out why must not cost two more servers.
        assert "replay_compare.py" in names

    def test_everything_clean_is_reportable(self, tmp_path, monkeypatch):
        code, _ = self._plumbed(tmp_path, monkeypatch, steps="{}\n")
        assert code == 0

    def test_a_blocked_comparison_is_not_overridden_by_a_clean_gate(
            self, tmp_path, monkeypatch):
        code, _ = self._plumbed(tmp_path, monkeypatch, steps="{}\n",
                                blocking=["arrival barrier timed out"])
        assert code == 1


class _NullServer:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class TestTheTwoSidesAreHandedTheirWorkDifferently:
    """A duration-bounded rung discovers its own graph. Lanes recycle until the
    clock says stop, so how many session instances each got through is not
    knowable before the run. The modelled side cannot discover the same one --
    the engine's arrival barrier holds every declared request until all
    `compass_workload_size` of them have arrived, so it has to be handed a
    graph that is already complete. Hence the asymmetry: the real side
    recycles and records what it executed, and the modelled side replays that.
    Both still carry `--clients`, because that is the rung's name and
    `saturation.py` reads it off each manifest.
    """

    def _commands(self, tmp_path, monkeypatch, *extra):
        work = tmp_path / "out"
        work.mkdir()
        (work / "compare.json").write_text(json.dumps({"blocking": []}))
        (work / "real_steps.jsonl").write_text("{}\n")
        calls = []
        monkeypatch.setattr(
            rv.subprocess, "run",
            lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
        monkeypatch.setattr(rv, "_run",
                            lambda cmd, label: calls.append((label, cmd)))
        monkeypatch.setattr(rv, "Served", lambda *a, **kw: _NullServer())
        monkeypatch.setattr(rv, "_free_port", lambda: 8000)
        rv.main(["--model", "M", "--trace", str(tmp_path / "t.jsonl"),
                 "--out-dir", str(work), "--table", str(tmp_path / "s"),
                 *extra])
        return work, {label: cmd for label, cmd in calls}

    def test_the_real_side_recycles_against_a_clock(self, tmp_path, monkeypatch):
        _, cmds = self._commands(tmp_path, monkeypatch,
                                 "--clients", "4",
                                 "--benchmark-duration", "1800")
        real = cmds["real replay"]
        assert "--pace" in real
        assert real[real.index("--benchmark-duration") + 1] == "1800.0"
        assert "--schedule" not in real

    def test_the_modelled_side_replays_the_schedule_the_real_one_executed(
            self, tmp_path, monkeypatch):
        work, cmds = self._commands(tmp_path, monkeypatch,
                                    "--clients", "4",
                                    "--benchmark-duration", "1800")
        modelled = cmds["modelled replay"]
        assert modelled[modelled.index("--schedule") + 1] == str(
            work / "real.json")
        # A declared run on a virtual clock: pacing it would time the
        # simulator, and a duration would ask it to recycle, which it cannot.
        assert "--pace" not in modelled
        assert "--benchmark-duration" not in modelled

    def test_both_sides_carry_the_rungs_client_count(self, tmp_path, monkeypatch):
        _, cmds = self._commands(tmp_path, monkeypatch, "--clients", "8")
        for cmd in cmds.values():
            if Path(cmd[1]).name == "replay.py":
                assert cmd[cmd.index("--clients") + 1] == "8"

    def test_an_open_loop_run_has_no_schedule_to_hand_over(self, tmp_path,
                                                           monkeypatch):
        # Without --clients both sides replay the trace's own arrivals, which
        # both already have. There is nothing for one side to discover.
        _, cmds = self._commands(tmp_path, monkeypatch)
        for cmd in cmds.values():
            assert "--schedule" not in cmd
            assert "--benchmark-duration" not in cmd
