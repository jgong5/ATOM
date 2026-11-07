# SPDX-License-Identifier: MIT
"""A JIT sweep sent round by round, and its coverage kept in a library.

The rounds send their requests to ATOM's real `Scheduler`, paced by its
`PrefillDelayer` as `compass record --sweep` paces a server of one DP rank, and
the simulated runner's `forward`, as `test_parity_record.py` drives them, and
read the record those write. ``client.sh --sweep --stop`` runs against a
stand-in compass command whose server, of two DP ranks, answers each
completion at once and records no step. Nothing needs a driver.
"""

import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from test_drive_identity import GIT
from test_drive_record import planted_gpu  # noqa: F401
from test_parity_record import _config, _Runner

from atom.compass import cli
from atom.compass.drive import sweep, sweep_plan
from atom.compass.drive.sweep import cover, send
from atom.compass.drive.sweep_plan import Point
from atom.compass.parity import ENV
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

pytestmark = pytest.mark.usefixtures("unpriced")

TREE = Path(__file__).resolve().parents[2]
WORKLOAD = {"traces": [{"requests": [{"in": 300}]}], "sampling": {"ignore_eos": True}}
BUDGET = 256


@pytest.fixture
def paced():
    """A process whose environment is the one `compass record --sweep` gives
    the server of ATOM's default arguments."""
    env = os.environ | sweep.pacing([])
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
    with subprocess.Popen(sleeper, env=env) as process:
        yield process.pid
        process.kill()


class Engine:
    """One DP rank's scheduler and simulated runner, recording in `directory`,
    with the PrefillDelayer `compass record --sweep` sets unless not `delayer`.
    Each group of rows reaches it a step after the one before."""

    def __init__(self, directory, monkeypatch, delayer=True):
        monkeypatch.setenv(ENV, str(directory))
        config = _config(budget=BUDGET, enable_prefix_caching=True)
        config.max_num_seqs = 32
        self.scheduler, self.runner = Scheduler(config), _Runner(config)
        if delayer:
            self.scheduler.set_prefill_delayer(
                PrefillDelayer(
                    1, None, BUDGET, 1.0, stall_ticks=16, prefill_decode_interval=1
                )
            )
        self.sent = []

    def send(self, first, parked, groups):
        self._until(self._add(first), lambda s: s.is_finished)
        self._until(self._add(parked), lambda s: s.num_completion_tokens)
        for group in groups:
            self._add(group)
            self._step()
        while self.scheduler.has_unfinished_requests():
            self._step()

    def _add(self, bodies):
        added = []
        for body in bodies:
            assert body["stream"] and body["data_parallel_rank"] == 0
            self.sent.append(body)
            params = SamplingParams(max_tokens=body["max_tokens"], ignore_eos=True)
            added.append(Sequence(body["prompt_token_ids"], 16, sampling_params=params))
            self.scheduler.add(added[-1])
        return added

    def _until(self, sequences, done):
        while not all(map(done, sequences)):
            self._step()

    def _step(self):
        scheduled = self.scheduler.schedule()
        if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
            return
        batch, seqs = scheduled
        reply = self.runner.forward(batch)
        if self.scheduler.prefill_delayer and batch.total_seqs_num_prefill:
            self.scheduler.prefill_delayer.notify_prefill_executed()
        self.scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )

    def close(self):
        pass


def _round(tmp_path, monkeypatch, k, library, pid, delayer=True):
    out = tmp_path / f"round{k}"
    out.mkdir()
    server = {"url": "", "library": str(library), "pid": pid}
    (out / "server.json").write_text(json.dumps(server))
    engine = Engine(out / "steps", monkeypatch, delayer)
    sent = send(out, WORKLOAD, engine)
    return sent, cover(out), engine


def test_rounds_send_the_steps_still_missing_until_the_cell_is_swept(
    tmp_path, monkeypatch, capsys, paced
):
    monkeypatch.setattr(sweep_plan, "SLOTS", 4)
    monkeypatch.setattr(sweep_plan, "PARKED_TOKENS", 32)
    library = tmp_path / "library"
    # Round 1's server has no PrefillDelayer, so each row runs as it arrives:
    # every point of two rows, and every chunk behind a first row, is missed.
    planned, missed, _ = _round(tmp_path, monkeypatch, 1, library, paced, False)
    kinds = {("",), ("", ""), ("chunk",), ("prefix",)}
    assert {p.cached for p in planned} == kinds

    def behind(p):
        return p.cached == ("chunk",) and (p.rows[0][1] - p.rows[0][0]) % BUDGET

    assert missed == [p for p in planned if len(p.rows) == 2 or behind(p)]
    assert any(map(behind, missed))
    (cell,) = (library / sweep.COVERAGE).iterdir()
    assert json.loads(cell.read_text())["swept"] is False
    # Round 2's delayer holds each step's rows behind its parked decodes until
    # all have arrived, and runs them as one batch.
    sent, missing, engine = _round(tmp_path, monkeypatch, 2, library, paced)
    assert (sent, missing) == (missed, [])
    parked = [b for b in engine.sent if b["max_tokens"] == 32]
    assert len(parked) == sum(p.held for p in missed)
    out = capsys.readouterr().out.splitlines()
    assert out[-1].endswith(f"{len(planned)} in all, 0 missing; swept")
    assert json.loads(cell.read_text())["swept"] is True
    # A swept cell's round primes the engine file and sends nothing more.
    sent, missing, engine = _round(tmp_path, monkeypatch, 3, library, paced)
    assert (sent, missing) == ([], [])
    assert [len(b["prompt_token_ids"]) for b in engine.sent] == [1]


def test_a_steps_requests_park_decodes_reuse_prefixes_and_fill_before_a_cut():
    engine = {"max_num_batched_tokens": 64, "enable_dp_attention": True}
    body = {"ignore_eos": True, "max_tokens": 2}
    step = [
        Point(0, ((5, 37),), ("prefix",), 2, False, 7, body),
        Point(1, ((3, 3), (9, 9)), ("", ""), 1, False, 7, body),
        Point(2, ((5, 45),), ("chunk",), 1, False, 7, body),
    ]
    tokens = iter(range(10**6))
    first, parked, groups = sweep._requests(
        step, engine, lambda n: list(itertools.islice(tokens, n))
    )
    (prefix,) = first
    assert len(prefix["prompt_token_ids"]) == 32 and prefix["data_parallel_rank"] == 0
    held = [(b["data_parallel_rank"], len(b["prompt_token_ids"])) for b in parked]
    assert held == [(0, 1), (0, 1), (1, 1), (2, 1)]
    assert {b["max_tokens"] for b in parked} == {sweep_plan.PARKED_TOKENS}
    # Every rank's first row, then its second: the prefixed row, the costly
    # row last, and a first row of the budget's rest before the chunked prompt
    # whose first chunk ends at 40.
    lengths = [
        [(b["data_parallel_rank"], len(b["prompt_token_ids"])) for b in group]
        for group in groups
    ]
    assert lengths == [[(0, 37), (1, 3), (2, 24)], [(1, 9), (2, 45)]]
    assert groups[0][0]["prompt_token_ids"][:32] == prefix["prompt_token_ids"]


def test_a_failed_request_is_printed_and_the_sweep_goes_on(capsys):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            ok = body["prompt_token_ids"] != [0]
            self.send_response(200 if ok else 500)
            self.end_headers()
            self.wfile.write(b"data: {}\n\ndata: [DONE]\n\n" if ok else b"no")

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rows = [{"prompt_token_ids": [0]}, {"prompt_token_ids": [2]}]
    try:
        sweep._Wall(server.server_port).send([], [{"prompt_token_ids": [1]}], [rows])
    finally:
        server.shutdown()
    err = capsys.readouterr().err
    assert err.count("sweep: a request failed") == 1 and "HTTP 500" in err


@pytest.mark.skipif(GIT is None, reason="needs git")
def test_compass_record_sweep_starts_its_server_with_the_pacing_variables(
    tmp_path, planted_gpu  # noqa: F811
):
    out = tmp_path / "out"
    atom = f"--model {planted_gpu} --max-num-batched-tokens 64"
    cli.main(f"record --gpu 7 --sweep --out {out} -- {atom}".split())
    env = json.loads((out / "run.json").read_text())["env"]
    assert env.items() >= sweep.pacing(atom.split()).items()
    assert env["ATOM_DP_PREFILL_BALANCE_MIN_TOKENS"] == "64"
    assert env["ATOM_PREFILL_DECODE_INTERVAL"] == "1"


STANDIN = """
import json, signal, subprocess, sys
from pathlib import Path
out, engine, library = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
(out / "steps").mkdir()
for rank in range(json.loads(engine)["data_parallel_size"]):
    (out / f"steps/dp{rank}.engine.json").write_text(engine)
    (out / f"steps/dp{rank}.jsonl").write_text("")
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
        self.wfile.write(b"data: {}\\n\\ndata: [DONE]\\n\\n")
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
    "data_parallel_size": 2,
    "enable_dp_attention": True,
    "enable_tbo": False,
    "enable_prefix_caching": False,
}


def _client(
    tmp_path, name, *flags, library="lib", paced=True, engine=ENGINE, server_env=None
):
    out, bin_ = tmp_path / name, tmp_path / "bin"
    out.mkdir()
    bin_.mkdir(exist_ok=True)
    (bin_ / "curl").write_text("#!/bin/sh\n")
    (bin_ / "curl").chmod(0o755)
    (tmp_path / "traces.jsonl").write_text('{"requests": [{"in": 5}]}\n' * 2)
    lib = str(tmp_path / library) if library else ""
    (tmp_path / "compass.py").write_text(STANDIN.replace("SERVER", repr(SERVER)))
    command = subprocess.Popen(
        [sys.executable, tmp_path / "compass.py", out, json.dumps(engine), lib],
        env=os.environ | sweep.pacing([]) | (server_env or {}) if paced else None,
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
    bodies = (
        [json.loads(b) for b in sent.read_text().splitlines()] if sent.exists() else []
    )
    return done, bodies, out


def test_client_sh_sweeps_a_cell_and_after_the_stop_keeps_its_coverage(tmp_path):
    done, bodies, out = _client(tmp_path, "r1", "--sweep", "--traces", "0-1", "--stop")
    assert done.returncode == 0, done.stdout + done.stderr
    state = json.loads((out / sweep.STATE).read_text())
    planned = len(state["points"])
    # A record with no steps covers nothing, and every point is missing: A's
    # counts 1 to 8, those past the 5-token prompt as two rows.
    assert len(state["missing"]) == planned == 8
    assert done.stdout.splitlines()[-1].endswith(
        "0 of 8 points covered by this record, 0 in all, 8 missing; not swept"
    )
    # The ranks are sent in rounds, a count a rank: 1 and 2 first.
    sent = [(b["data_parallel_rank"], len(b["prompt_token_ids"])) for b in bodies]
    assert sorted(sent[:2]) == [(0, 1), (1, 2)]
    # Counts 6 to 8 each park a decode before their two rows.
    assert len(bodies) == 5 + 3 * 3
    parked = [b for b in bodies if b["max_tokens"] == sweep_plan.PARKED_TOKENS]
    assert sorted(b["data_parallel_rank"] for b in parked) == [0, 1, 1]
    assert {b["ignore_eos"] for b in bodies} == {True}
    assert (out / "run.json").exists()
    # Marked covered, the cell is swept, and its next sweep sends nothing.
    coverage = tmp_path / "lib" / sweep.COVERAGE / f"{state['cell']}.json"
    keys = [sweep._key(Point(*p[:6], {})) for p in state["points"]]
    coverage.write_text(json.dumps({"covered": keys}))
    done, bodies, _ = _client(tmp_path, "r2", "--sweep", "--traces", "0-1", "--stop")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "8 points planned, 8 covered before, 0 to send" in done.stdout
    assert done.stdout.splitlines()[-1].endswith("0 missing; swept")
    assert bodies == []


def test_client_sh_plans_two_batch_overlap_by_the_servers_variables(tmp_path):
    # The server splits a step at 4 tokens, this process at its default, past
    # the budget: from the round of 3 and 4 on, rank 1 has no row.
    done, _, out = _client(
        tmp_path,
        "r",
        "--sweep",
        "--traces",
        "0-1",
        "--stop",
        engine=ENGINE | {"enable_tbo": True},
        server_env={"ATOM_TBO_PREFILL_MIN_TOKENS": "4"},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    points = json.loads((out / sweep.STATE).read_text())["points"]
    assert [(p[0], p[5]) for p in points] == [(0, 0), (1, 0)] + [
        (0, k) for k in range(1, 7)
    ]


@pytest.mark.parametrize(
    "flags, library, paced, refusal, rc",
    [
        (["--traces", "0-1"], "lib", True, "--sweep takes --traces and --stop", 2),
        (["--traces", "0-1", "--stop"], None, True, "names no JIT cache library", 1),
        (["--traces", "0-1", "--stop"], "lib", False, "compass record --sweep", 1),
    ],
)
def test_client_sh_refuses_a_sweep_it_cannot_cover_or_whose_server_is_not_paced(
    tmp_path, flags, library, paced, refusal, rc
):
    done, bodies, _ = _client(
        tmp_path, "r", "--sweep", *flags, library=library, paced=paced
    )
    assert done.returncode == rc
    assert refusal in done.stderr
    assert bodies == [] and "stopped the server" not in done.stdout
