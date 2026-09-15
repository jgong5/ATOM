"""The seams, exercised for real: real processes, real signals, real sockets.

Everything else about the harness is tested with the processes injected, which
is how the sequencing tests stay deterministic. That leaves the parts where the
fakes could be wrong about the world: whether a sampler started as a subprocess
actually finishes its sample when it is signalled, whether what it appends is
still what the audit parses, and whether the health and provenance probes read
an ordinary HTTP server. None of it needs a device, so all of it runs here.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
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


run_mod = _load("cc_traces_run")
isolation = _load("isolation")

CARD = {
    "card0": {
        "GPU use (%)": "0",
        "GPU Memory Allocated (VRAM%)": "0",
        "VRAM Total Used Memory (B)": "0",
        "GUID": "4123",
        "Device ID": "0x74a1",
    }
}


@pytest.fixture
def fake_smi(tmp_path):
    """A `rocm-smi` that is a shell script, so the sampler really forks."""
    path = tmp_path / "rocm-smi"
    path.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = "--showpids" ]; then echo \'{"system": {}}\'; exit 0; fi\n'
        "done\n"
        f"cat <<'JSON'\n{json.dumps(CARD)}\nJSON\n"
    )
    path.chmod(0o755)
    return str(path)


def _wait_for(predicate, timeout=30.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestTheSamplerAsARealProcess:
    def test_it_samples_reacts_to_the_phase_file_and_stops_when_signalled(
        self, tmp_path, fake_smi
    ):
        """The whole lifecycle a cell puts it through, in one process.

        Started before the first server with a baseline phase, retagged while
        it runs, and ended by the harness signalling the pid it recorded --
        which is the only way the harness ever ends anything.
        """
        out = tmp_path / "gpu.jsonl"
        phase_file = tmp_path / "phase.json"
        phase_file.write_text(json.dumps({"phase": "baseline", "own_pids": []}))
        processes = run_mod.Processes()
        proc = processes.start(
            [
                sys.executable,
                str(ROOT / "scripts" / "compass" / "gpu_sampler.py"),
                str(out),
                "--smi",
                fake_smi,
                "--interval",
                "0.2",
                "--phase-file",
                str(phase_file),
            ],
            log=tmp_path / "sampler.log",
        )
        try:
            assert _wait_for(lambda: len(_rows(out)) >= 2), out.read_text()
            phase_file.write_text(
                json.dumps({"phase": "serving", "own_pids": [str(proc.pid)]})
            )
            assert _wait_for(lambda: any(r["phase"] == "serving" for r in _rows(out)))
        finally:
            code = processes.stop(proc)
        assert code == 0, "a signalled sampler should leave cleanly, not be killed"

        rows = _rows(out)
        assert [r["phase"] for r in rows[:2]] == ["baseline", "baseline"]
        assert rows[-1]["phase"] == "serving"
        # The last line is whole: a half-written sample is one the audit skips,
        # which is a silently shorter window.
        assert rows[-1]["smi"]["card0"]["GUID"] == "4123"

    def test_the_audit_reads_the_file_that_process_wrote(self, tmp_path, fake_smi):
        out = tmp_path / "gpu.jsonl"
        processes = run_mod.Processes()
        proc = processes.start(
            [
                sys.executable,
                str(ROOT / "scripts" / "compass" / "gpu_sampler.py"),
                str(out),
                "--smi",
                fake_smi,
                "--phase",
                "baseline",
                "--samples",
                "3",
                "--interval",
                "0.05",
            ],
            log=tmp_path / "sampler.log",
        )
        assert _wait_for(lambda: proc.poll() is not None)
        assert processes.stop(proc) == 0
        verdict = isolation.audit(isolation.read(str(out)))
        assert verdict["samples"] == 3
        assert verdict["baseline_provenance"] == "phase-stamped"
        assert verdict["verdict"] == "clean"
        assert verdict["cards_seen"] == [0]

    def test_a_rocm_smi_that_is_not_there_is_an_unwatched_run_not_a_quiet_one(
        self, tmp_path
    ):
        out = tmp_path / "gpu.jsonl"
        processes = run_mod.Processes()
        proc = processes.start(
            [
                sys.executable,
                str(ROOT / "scripts" / "compass" / "gpu_sampler.py"),
                str(out),
                "--smi",
                str(tmp_path / "no-such-rocm-smi"),
                "--once",
            ],
            log=tmp_path / "sampler.log",
        )
        assert _wait_for(lambda: proc.poll() is not None)
        assert processes.stop(proc) == 1
        assert isolation.audit(isolation.read(str(out)))["verdict"] == "unwatched"


class Endpoint(BaseHTTPRequestHandler):
    payloads: ClassVar[dict] = {}

    def do_GET(self):  # the base class spells it this way
        body = self.payloads.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        blob = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *_args):
        pass


@pytest.fixture
def served():
    """An ordinary HTTP server, so the probes are tested against a socket."""
    Endpoint.payloads = {
        "/health": {},
        "/compass/provenance": {"compass": {"mode": "predict"}, "model": "Q"},
    }
    httpd = HTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


class TestTheProbes:
    def test_health_and_provenance_are_read_over_a_real_socket(self, served):
        assert run_mod.http_get(f"{served}/health") == {}
        said = run_mod.http_get(f"{served}/compass/provenance")
        assert said["compass"]["mode"] == "predict"

    def test_a_missing_endpoint_is_not_a_healthy_server(self, served):
        assert run_mod.http_get(f"{served}/nothing-here") is None

    def test_a_closed_port_is_not_a_healthy_server(self):
        """What the harness polls on while a server is still loading weights."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert run_mod.http_get(f"http://127.0.0.1:{port}/health", timeout=1.0) is None
