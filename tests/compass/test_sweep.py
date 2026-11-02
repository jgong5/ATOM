# SPDX-License-Identifier: MIT
"""A JIT sweep sent round by round, and its coverage kept in a library.

The rounds send their requests to ATOM's real `Scheduler` and the simulated
runner's `forward` on each DP rank, as `test_sweep_plan.py` drives them, and
read the record those write. ``client.sh --sweep --stop`` runs against a
stand-in compass command whose server answers each completion at once and
records no step. Nothing needs a driver.
"""

import json
import os
import queue
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from test_parity_record import _config, _Runner

from atom.compass.drive import sweep, sweep_plan
from atom.compass.drive.sweep import cover, send
from atom.compass.parity import ENV
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

pytestmark = pytest.mark.usefixtures("unpriced")

TREE = Path(__file__).resolve().parents[2]
WORKLOAD = {"traces": [{"requests": [{"in": 300}]}], "sampling": {"ignore_eos": True}}


class Ranks:
    """Each DP rank's scheduler and simulated runner, recording in
    `directory`. A sent phase's requests all queue on their ranks, which then
    step until every request has ended; with `apart`, each reaches an idle
    engine, which steps once before the next."""

    def __init__(self, directory, monkeypatch, size=2, apart=False):
        monkeypatch.setenv(ENV, str(directory))
        self.ranks = []
        for rank in range(size):
            config = _config(dp_rank=rank, budget=64, enable_prefix_caching=True)
            config.max_num_seqs = 32
            config.parallel_config.data_parallel_size = size
            self.ranks.append((Scheduler(config), _Runner(config)))
        self.apart, self.sent = apart, []

    def send(self, bodies):
        for body in bodies:
            assert body["max_tokens"] == 2 and body["stream"]
            self.sent.append(body)
            scheduler, runner = self.ranks[body["data_parallel_rank"]]
            params = SamplingParams(max_tokens=2, ignore_eos=body["ignore_eos"])
            scheduler.add(Sequence(body["prompt_token_ids"], 16, sampling_params=params))
            if self.apart:
                self._step(scheduler, runner)
        while any(s.has_unfinished_requests() for s, _ in self.ranks):
            for scheduler, runner in self.ranks:
                if scheduler.has_unfinished_requests():
                    self._step(scheduler, runner)

    @staticmethod
    def _step(scheduler, runner):
        batch, seqs = scheduler.schedule()
        reply = runner.forward(batch)
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )

    def close(self):
        pass


def _round(tmp_path, monkeypatch, k, library, apart=False):
    out = tmp_path / f"round{k}"
    out.mkdir()
    (out / "server.json").write_text(json.dumps({"url": "", "library": str(library)}))
    ranks = Ranks(out / "steps", monkeypatch, apart=apart)
    sent = send(out, WORKLOAD, ranks)
    return sent, cover(out), ranks


def test_rounds_send_only_the_points_still_missing_until_the_cell_is_swept(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sweep_plan, "SPACING", 8)
    library = tmp_path / "library"
    # Round 1's requests each reach an idle engine, so a point of two rows
    # runs as two steps and is missed.
    planned, missed, ranks = _round(tmp_path, monkeypatch, 1, library, apart=True)
    assert sorted({p.rank for p in planned}) == [0, 1]
    assert {how for p in planned for how in p.cached} == {"", "chunk", "prefix", "turn"}
    assert missed == [p for p in planned if len(p.rows) > 1] != []
    # Round 2 sends those alone; queued behind its blockers, each runs as one batch.
    sent, missing, ranks = _round(tmp_path, monkeypatch, 2, library)
    assert (sent, missing) == (missed, [])
    rows = sorted(len(b["prompt_token_ids"]) for b in ranks.sent)
    assert rows.count(64) == len(sent)  # one budget blocker a point
    out = capsys.readouterr().out.splitlines()
    assert out[-1].endswith(f"{len(planned)} in all, 0 missing; swept")
    (cell,) = (library / sweep.COVERAGE).iterdir()
    assert json.loads(cell.read_text())["swept"] is True
    # A swept cell's round primes each rank's engine file and sends nothing more.
    sent, missing, ranks = _round(tmp_path, monkeypatch, 3, library)
    assert (sent, missing) == ([], [])
    assert [len(b["prompt_token_ids"]) for b in ranks.sent] == [1, 1, 1]


def test_a_points_requests_reuse_its_context_and_queue_behind_its_blockers():
    engine = {"max_num_batched_tokens": 64, "max_model_len": 40}
    point = sweep_plan.Point(1, ((5, 37),), ("prefix",), 3, {"ignore_eos": True})
    tokens = iter(range(10**6))
    first, main = sweep._phases(point, engine, lambda n: [next(tokens) for _ in range(n)])
    (shared,) = [b["prompt_token_ids"] for b in first]
    assert len(shared) == 32 and {b["data_parallel_rank"] for b in first + main} == {1}
    lengths = [len(b["prompt_token_ids"]) for b in main]
    # Blockers no longer than the model takes fill the budget, then the held
    # requests and a budget's rest beside them, then the row on the prefix.
    assert lengths == [38, 26, 1, 1, 1, 38, 23, 37]
    assert main[-1]["prompt_token_ids"][:32] == shared


STANDIN = """
import json, signal, subprocess, sys
from pathlib import Path
out, engine, library = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
(out / "steps").mkdir()
(out / "steps/dp0.engine.json").write_text(engine)
(out / "steps/dp0.jsonl").write_text("")
server = subprocess.Popen([sys.executable, "-c", SERVER, str(out), library])
def stop(*_):
    server.kill()
    (out / "run.json").write_text("{}")
    sys.exit()
signal.signal(signal.SIGTERM, stop)
signal.pause()
"""
SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
out, library = Path(sys.argv[1]), sys.argv[2]
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        with open(out / "sent.jsonl", "ab") as f:
            f.write(body + b"\\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"data: [DONE]\\n\\n")
server = HTTPServer(("127.0.0.1", 0), Handler)
ready = {"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()}
if library:
    ready["library"] = library
(out / "s").write_text(json.dumps(ready))
(out / "s").rename(out / "server.json")
server.serve_forever()
"""
ENGINE = {
    "max_num_seqs": 4,
    "max_num_batched_tokens": 8,
    "max_model_len": 64,
    "kv_cache_block_size": 16,
    "num_kvcache_blocks": 100,
    "data_parallel_size": 1,
    "enable_dp_attention": False,
    "enable_tbo": False,
    "enable_prefix_caching": False,
}


def _client(tmp_path, name, *flags, library="lib"):
    out, bin_ = tmp_path / name, tmp_path / "bin"
    out.mkdir()
    bin_.mkdir(exist_ok=True)
    (bin_ / "curl").write_text("#!/bin/sh\n")
    (bin_ / "curl").chmod(0o755)
    (tmp_path / "traces.jsonl").write_text('{"requests": [{"in": 5}]}\n' * 2)
    lib = str(tmp_path / library) if library else ""
    (tmp_path / "compass.py").write_text(STANDIN.replace("SERVER", repr(SERVER)))
    command = subprocess.Popen(
        [sys.executable, tmp_path / "compass.py", out, json.dumps(ENGINE), lib]
    )
    try:
        while not (out / "server.json").exists():
            assert command.poll() is None
        env = os.environ | {
            "PATH": f"{bin_}:{os.environ['PATH']}",
            "TRACES": "traces.jsonl",
            "PYTHONPATH": str(TREE),
        }
        done = subprocess.run(
            ["bash", TREE / "scripts/compass/client.sh", out, *flags],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        if command.poll() is None:
            command.send_signal(signal.SIGTERM)
        command.wait(30)
    sent = out / "sent.jsonl"
    bodies = [json.loads(b) for b in sent.read_text().splitlines()] if sent.exists() else []
    return done, bodies, out


def test_client_sh_sweeps_a_cell_and_after_the_stop_keeps_its_coverage(tmp_path):
    done, bodies, out = _client(tmp_path, "r1", "--sweep", "--traces", "0-1", "--stop")
    assert done.returncode == 0, done.stdout + done.stderr
    state = json.loads((out / sweep.STATE).read_text())
    planned = len(state["points"])
    # A record with no steps covers nothing, and every point is missing.
    assert len(state["missing"]) == planned == 8
    assert done.stdout.splitlines()[-1].endswith(
        "0 of 8 points covered by this record, 0 in all, 8 missing; not swept"
    )
    assert {b["data_parallel_rank"] for b in bodies} == {0}
    assert {b["ignore_eos"] for b in bodies} == {True}
    assert (out / "run.json").exists()
    # Marked covered, the cell is swept, and its next sweep sends nothing.
    coverage = tmp_path / "lib" / sweep.COVERAGE / f"{state['cell']}.json"
    keys = [json.dumps([r, rows]) for r, rows, _, _ in state["points"]]
    coverage.write_text(json.dumps({"covered": keys}))
    done, bodies, _ = _client(tmp_path, "r2", "--sweep", "--traces", "0-1", "--stop")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "8 points planned, 8 covered before, 0 to send" in done.stdout
    assert done.stdout.splitlines()[-1].endswith("0 missing; swept")
    assert bodies == []


@pytest.mark.parametrize(
    "flags, library, refusal, rc",
    [
        (["--sweep", "--traces", "0-1"], "lib", "--sweep takes --traces and --stop", 2),
        (["--sweep", "--traces", "0-1", "--stop"], None, "sweep: atom.compass", 1),
    ],
)
def test_client_sh_refuses_a_sweep_it_cannot_cover(tmp_path, flags, library, refusal, rc):
    done, bodies, out = _client(tmp_path, "r", *flags, library=library)
    assert done.returncode == rc
    assert done.stderr.splitlines()[-1].startswith(f"client: {refusal}")
    assert bodies == [] and "stopped the server" not in done.stdout
    if library is None:
        assert "names no JIT cache library" in done.stderr
