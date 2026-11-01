# SPDX-License-Identifier: MIT
"""The HTTP intake probe: what it stamps on a server it runs, what its fit
recovers, and the machine specs that state its terms."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[2]
PROBE = TREE / "scripts/compass/http_intake.py"
_loaded = importlib.util.spec_from_file_location("http_intake", PROBE)
probe = importlib.util.module_from_spec(_loaded)
_loaded.loader.exec_module(probe)

#: A stand-in for ATOM's API server: reads one request's body twice, as the
#: real handler does, renders its template, and prints its arguments.
SERVER = {
    "__init__.py": "",
    "entrypoints/__init__.py": "",
    "entrypoints/openai/__init__.py": "",
    "entrypoints/openai/chat_encoders.py": (
        "def apply_chat_template(*a, **k):\n    return 'prompt'\n"
    ),
    "entrypoints/openai/api_server.py": """
import asyncio, sys
from starlette.requests import Request
from .chat_encoders import apply_chat_template

async def main():
    async def receive():
        return {"type": "http.request", "body": b"x" * 1234}

    request = Request({"type": "http"}, receive)
    assert await request.body() == await request.body()
    apply_chat_template(None, None, [])

if __name__ == "__main__":
    print(sys.argv[1:])
    asyncio.run(main())
""",
}


def test_serve_stamps_a_body_read_off_the_socket_once_and_each_template(tmp_path):
    for name, text in SERVER.items():
        (tmp_path / "atom" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "atom" / name).write_text(text)
    stamps = tmp_path / "stamps"
    done = subprocess.run(
        [sys.executable, PROBE, "serve", stamps, "--model", "m"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "['--model', 'm']\n"
    rows = [line.split() for line in stamps.read_text().splitlines()]
    assert [(r[2], r[4]) for r in rows] == [("body", "1234"), ("tmpl", "6")]
    assert all(int(r[0]) >= int(r[3]) for r in rows)


def test_a_cell_with_http_intake_runs_its_server_under_the_probe(tmp_path):
    # The script is executed as a program, as a user runs it, with a python3
    # stand-in that records the server's arguments and exits before it is ready.
    bin_, args = tmp_path / "bin", tmp_path / "args"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        f'case "$1" in *http_intake.py) printf "%s\\n" "$@" >{args}; exit 1;; esac\n'
        f'exec {sys.executable} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        '#!/bin/sh\nexec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces, run_file = tmp_path / "traces.jsonl", tmp_path / "run.json"
    traces.write_text('{"id": 0}\n')
    run_file.write_text("{}")
    stamps = tmp_path / "stamps"
    done = subprocess.run(
        [TREE / "scripts/compass/cctraces_sim.sh", run_file, tmp_path / "cell"],
        cwd="/",
        env=dict(
            os.environ,
            PATH=f"{bin_}:{os.environ['PATH']}",
            MODEL=str(tmp_path),
            TRACES=str(traces),
            HARNESS_PYTHON=str(bin_ / "harness-python"),
            HTTP_INTAKE=str(stamps),
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "the deployment exited before it started" in done.stderr, done.stderr
    probe_path, *rest = args.read_text().splitlines()
    assert Path(probe_path).resolve() == PROBE
    assert rest[:4] == ["serve", str(stamps), "--model", str(tmp_path)]


def _burst(fixed, per_byte, sizes, stall=0.0, t0=10**12):
    """Stamp lines for a loop that takes `sizes` in from `t0` one request at a
    time, each template starting once its own request is taken in, the largest
    request's intake `stall` seconds longer, with a template before the burst and
    a later turn after it."""
    lines = [f"{t0 - 10**9} 1 tmpl {t0 - 10**9} 9"]
    done = 0
    for i, size in enumerate(sizes, 1):
        done += size + (size == max(sizes)) * stall / per_byte
        start = t0 + round((i * fixed + done * per_byte) * 1e9)
        lines += [f"{start - 1000} 1 body {start - 2000} {size}"]
        lines += [f"{start + 500} 1 tmpl {start} {size}"]
    end = t0 + 10**11
    lines += [f"{end} 1 body {end - 10} 7", f"{end + 20} 1 tmpl {end + 10} 7"]
    return "\n".join(lines) + "\n"


def test_the_fit_recovers_the_terms_of_a_loop_that_takes_a_burst_one_at_a_time(
    tmp_path,
):
    sizes = np.random.default_rng(0).integers(100, 200_000, 64).tolist()
    stamps = tmp_path / "stamps"
    stamps.write_text(_burst(2.1e-3, 1.8e-8, sizes))
    got = probe.fit(str(stamps), 64)
    assert got["fixed_s"] == pytest.approx(2.1e-3, rel=1e-3)
    assert got["per_byte_s"] == pytest.approx(1.8e-8, rel=1e-3)
    assert got["rms_s"] < 1e-8
    assert (got["requests"], got["bytes"]) == (64, sum(sizes))
    with pytest.raises(SystemExit, match="65 bodies and 65 templates, not 66"):
        probe.fit(str(stamps), 66)


def test_a_stall_on_the_loop_moves_the_fixed_term_and_not_the_per_byte_one(
    tmp_path,
):
    # 50 ms of other work on the loop while it takes in the largest request:
    # spread over the 63 gaps, so the burst still spans what it measured.
    sizes = np.random.default_rng(0).integers(100, 200_000, 64).tolist()
    stamps = tmp_path / "stamps"
    stamps.write_text(_burst(2.1e-3, 1.8e-8, sizes, stall=0.05))
    got = probe.fit(str(stamps), 64)
    print(f"\n  {got}")
    assert got["per_byte_s"] == pytest.approx(1.8e-8, rel=0.05)
    span = 63 * 2.1e-3 + 0.05 + sum(sizes[1:]) * 1.8e-8
    assert 63 * got["fixed_s"] + sum(sizes[1:]) * got["per_byte_s"] == pytest.approx(
        span
    )


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39", "mi355x-gpu45"])
def test_every_committed_spec_states_a_probed_intake_from_a_named_record(name):
    from atom.compass.spec import MachineSpec

    document = json.loads((TREE / f"scripts/compass/machines/{name}.json").read_text())
    spec = MachineSpec.from_mapping(document)
    for term in ("fixed_s", "per_byte_s"):
        origin = spec.origin(f"host.http_intake.{term}")
        assert origin["origin"] == "probed"
        assert origin["record"].startswith("sha256:")


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39", "mi355x-gpu45"])
def test_every_committed_intake_band_spans_more_than_one_burst(name):
    from atom.compass.spec import MachineSpec

    document = json.loads((TREE / f"scripts/compass/machines/{name}.json").read_text())
    spec = MachineSpec.from_mapping(document)
    for term in ("fixed_s", "per_byte_s"):
        origin = spec.origin(f"host.http_intake.{term}")
        assert origin["p10"] < origin["p90"]


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_mi308x_specs_rebuild_byte_for_byte_from_their_own_sweeps(
    monkeypatch, tmp_path, name
):
    monkeypatch.setenv("COMPASS_MACHINE", name)
    path = TREE / "scripts/compass/mi308x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi308x_spec", path)
    script = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(script)
    document = TREE / f"scripts/compass/machines/{name}.json"
    out = tmp_path / "rebuilt.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    assert out.read_bytes() == document.read_bytes()


#: The stand-in server: a Starlette app's lifespan, as uvicorn starts it, then
#: one GET; prints when the app sent the GET's response.
APP_SERVER = """
import asyncio, time
from starlette.applications import Starlette

async def main():
    app, sends = Starlette(), []
    lifespan = iter(["lifespan.startup", "lifespan.shutdown"])

    async def started():
        return {"type": next(lifespan)}

    async def ignore(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sends.append(time.monotonic_ns())

    await app({"type": "lifespan"}, started, ignore)
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    await app(scope | {"query_string": b"", "root_path": ""}, receive, send)
    print(sends[0])

if __name__ == "__main__":
    asyncio.run(main())
"""


def test_serve_stamps_each_http_request_handed_to_the_asgi_app(tmp_path):
    server = SERVER | {"entrypoints/openai/api_server.py": APP_SERVER}
    for name, text in server.items():
        (tmp_path / "atom" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "atom" / name).write_text(text)
    stamps = tmp_path / "stamps"
    done = subprocess.run(
        [sys.executable, PROBE, "serve", stamps],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.split() for line in stamps.read_text().splitlines()]
    # The lifespan call is no request: `floor` would pair it with a send.
    assert [(r[2], r[4]) for r in rows] == [("app", "0")]
    # Stamped as the app is called, before it answers.
    assert int(rows[0][3]) < int(done.stdout)


def test_the_floor_is_each_send_to_its_app_call_one_request_in_flight(tmp_path):
    sent = [10**12 + k * 10**9 for k in range(10)]
    floors = [400_000 + 10_000 * k for k in range(10)]
    stamps, arrivals = tmp_path / "stamps", tmp_path / "arrivals.json"
    stamps.write_text(
        "".join(f"{s + f} 1 app {s + f} 0\n" for s, f in zip(sent, floors))
        + f"{sent[0] + 9} 1 body {sent[0] + 8} 7\n"
    )
    arrivals.write_text(json.dumps({"sent_ns": sent[::-1], "admitted_ns": sent}))
    got = probe.floor(str(stamps), str(arrivals))
    assert got["median_s"] == pytest.approx(445e-6)
    assert (got["p10_s"], got["p90_s"]) == pytest.approx((409e-6, 481e-6))
    assert got["requests"] == 10
    # A second request sent before the first reached the app pairs nothing.
    arrivals.write_text(json.dumps({"sent_ns": [sent[0], sent[0] + 1] + sent[2:]}))
    with pytest.raises(SystemExit, match="10 app calls, not one after each of 10"):
        probe.floor(str(stamps), str(arrivals))


def test_serve_stamps_each_tokenizer_pool_job_and_engine_send(tmp_path):
    server = SERVER | {
        "model_engine/__init__.py": "",
        "model_engine/llm_engine.py": (
            "class InputOutputProcessor:\n"
            "    def preprocess(self, prompt):\n"
            "        return self.preprocess_fanout(prompt)[0]\n"
            "    def preprocess_fanout(self, prompt):\n"
            "        return [prompt]\n"
        ),
        "model_engine/engine_core_mgr.py": (
            "class CoreManager:\n"
            "    def add_request(self, seqs):\n"
            "        print(seqs)\n"
            "    def _mark_seq_prefill_complete(self, seq_id): pass\n"
            "    def _deliver_terminal_output(self, seq): return False\n"
        ),
        "entrypoints/openai/api_server.py": (
            "from atom.model_engine.engine_core_mgr import CoreManager\n"
            "from atom.model_engine.llm_engine import InputOutputProcessor\n"
            "if __name__ == '__main__':\n"
            "    CoreManager().add_request([InputOutputProcessor().preprocess('p')])\n"
        ),
    }
    for name, text in server.items():
        (tmp_path / "atom" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "atom" / name).write_text(text)
    stamps = tmp_path / "stamps"
    done = subprocess.run(
        [sys.executable, PROBE, "serve", stamps],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "['p']\n"
    rows = [line.split() for line in stamps.read_text().splitlines()]
    assert [(r[2], r[4]) for r in rows] == [("prep", "0"), ("send", "0")]
    assert int(rows[0][3]) <= int(rows[0][0]) <= int(rows[1][3]) <= int(rows[1][0])


def test_the_idle_intake_is_the_loops_time_on_a_lone_request_less_the_bytes(
    tmp_path,
):
    # Each request: app call, body, 300 us + 5 ns/B to its pool job, a 2 ms job,
    # then 100 us + 10 us k to its engine send.
    calls = [10**12 + k * 10**9 for k in range(10)]
    sizes = [1000 * (k + 1) for k in range(10)]
    lines = []
    for k, (c, b) in enumerate(zip(calls, sizes)):
        j = c + 300_000 + 5 * b
        e = j + 2_000_000
        s = e + 100_000 + 10_000 * k
        lines.append(
            f"{c} 1 app {c} 0\n{c + 2} 1 body {c + 1} {b}\n"
            f"{e} 2 prep {j} 0\n{s} 1 send {s - 50} 0\n"
        )
    stamps = tmp_path / "stamps"
    stamps.write_text("".join(lines[::-1]))
    got = probe.idle(str(stamps), 5e-9)
    assert got["idle_s"] == pytest.approx(445e-6)
    assert (got["p10_s"], got["p90_s"]) == pytest.approx((409e-6, 481e-6))
    assert (got["requests"], got["per_byte_s"]) == (10, 5e-9)
    # A second app call before the first request's engine send pairs nothing.
    stamps.write_text("".join(lines) + f"{calls[0] + 1} 1 app {calls[0] + 1} 0\n")
    with pytest.raises(
        SystemExit, match="10 engine sends, not one each after each of 11"
    ):
        probe.idle(str(stamps))


def test_serve_stamps_each_sse_head_and_each_engine_output_taken_in(tmp_path):
    server = SERVER | {
        "model_engine/__init__.py": "",
        "model_engine/engine_core_mgr.py": (
            "class CoreManager:\n"
            "    def add_request(self, seqs): pass\n"
            "    def _mark_seq_prefill_complete(self, seq_id): pass\n"
            "    def _deliver_terminal_output(self, seq): return False\n"
        ),
        "entrypoints/openai/api_server.py": """
import asyncio
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route
from atom.model_engine.engine_core_mgr import CoreManager

async def sse(request):
    async def chunks():
        yield "data: 1\\n\\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")

async def text(request):
    return PlainTextResponse("x")

async def main():
    app = Starlette(routes=[Route("/sse", sse), Route("/text", text)])

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        pass

    for path in ("/sse", "/text"):
        scope = {"type": "http", "method": "GET", "path": path, "headers": []}
        await app(scope | {"query_string": b"", "root_path": ""}, receive, send)
    CoreManager()._mark_seq_prefill_complete(1)
    CoreManager()._deliver_terminal_output(None)

if __name__ == "__main__":
    asyncio.run(main())
""",
    }
    for name, text in server.items():
        (tmp_path / "atom" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "atom" / name).write_text(text)
    stamps = tmp_path / "stamps"
    subprocess.run(
        [sys.executable, PROBE, "serve", stamps],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        check=True,
    )
    rows = [line.split() for line in stamps.read_text().splitlines()]
    # The plain-text response sends a head too, and is no stream.
    assert [r[2] for r in rows] == ["app", "head", "app", "out", "out"]


def test_hops_pair_each_send_with_its_arrival_one_request_in_flight(tmp_path):
    sent = [10**12 + k * 10**9 for k in range(10)]
    hop = [10_000 * k for k in range(10)]
    stamps, handoff = [], []
    for s, h in zip(sent, hop):
        # Two streamed outputs in one message, then the finished sequence, whose
        # send stamp trails its receipt.
        put = s + 5 * 10**6
        stamps += [f"{s} 1 send {s - 50} 0", f"{s + 7} 1 head {s + 5} 0"]
        got = [put + 100_000 + h + p for p in (0, 5, 10)]
        stamps += [f"{g} 2 out {g} 0" for g in got]
        handoff += [
            {"t_out_put_ns": put, "t_out_sent_ns": put + 50, "rows": 2},
            {"t_out_put_ns": put + 10, "t_out_sent_ns": got[2] + 20, "rows": 1},
        ]
        # Request 0 finds the engine idle, and its receipt waits.
        idle = 5 * 10**6 if s == sent[0] else 0
        handoff.append({"t_recv_ns": s + 200_000 + h + idle, "t_in_put_ns": 0})
    # An output before the first request.
    handoff.append({"t_out_put_ns": 0, "t_out_sent_ns": sent[0] - 10**9, "rows": 1})
    admitted = [s + 7 + 400_000 + h for s, h in zip(sent, hop)]
    paths = [tmp_path / name for name in ("stamps", "handoff", "arrivals")]
    paths[0].write_text("\n".join(stamps[::-1]) + "\n")
    paths[1].write_text("".join(json.dumps(line) + "\n" for line in handoff))
    paths[2].write_text(json.dumps({"sent_ns": sent, "admitted_ns": admitted}))
    got = probe.hops(*map(str, paths))
    for name, median in (("request", 250e-6), ("output", 145e-6), ("stream", 445e-6)):
        assert got[name]["median_s"] == pytest.approx(median), name
    assert (got["request"]["p10_s"], got["request"]["p90_s"]) == pytest.approx(
        (218e-6, 282e-6)
    )
    assert [got[k]["messages"] for k in ("request", "output", "stream")] == [9, 20, 10]
    # A request the engine never received pairs nothing.
    paths[1].write_text(
        "".join(json.dumps(line) + "\n" for line in handoff[:2] + handoff[3:])
    )
    with pytest.raises(
        SystemExit, match="request: 8 arrivals, not one after each of 9"
    ):
        probe.hops(*map(str, paths))
    # Nor does a message with a row the server never took in.
    paths[1].write_text("".join(json.dumps(line) + "\n" for line in handoff))
    paths[0].write_text("\n".join(r for r in stamps if " out " not in r) + "\n")
    with pytest.raises(SystemExit, match="output: 0 rows taken in, not the 30 sent"):
        probe.hops(*map(str, paths))


def test_serve_stamps_each_kv_write_request_sent_and_received_in_a_spawned_process(
    tmp_path,
):
    server = SERVER | {
        "kv_transfer/__init__.py": "",
        "kv_transfer/disaggregation/__init__.py": "",
        "kv_transfer/disaggregation/mooncake/__init__.py": "",
        "kv_transfer/disaggregation/mooncake/mooncake_connector.py": """
import contextlib
MSG_WRITE_REQUEST, MSG_WRITE_DONE = b"write_request", b"write_done"

class Socket:
    def __init__(self):
        self.parts = [[b"id", MSG_WRITE_DONE, b""], [b"id", MSG_WRITE_REQUEST, b""]]

    def recv_multipart(self):
        return self.parts.pop()

@contextlib.contextmanager
def zmq_socket_ctx(path, kind, bind=None):
    yield Socket()

class MooncakeConnector:
    def _send_on_socket(self, addr, parts, repeat=1):
        pass
""",
        "worker.py": """
from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

def work():
    mc.MooncakeConnector()._send_on_socket("tcp://a", [mc.MSG_WRITE_REQUEST, b""])
    mc.MooncakeConnector()._send_on_socket("tcp://a", [mc.MSG_WRITE_DONE, b""])
    with mc.zmq_socket_ctx("tcp://*:1", "ROUTER", bind=True) as sock:
        assert [sock.recv_multipart()[1] for _ in range(2)] == [
            mc.MSG_WRITE_REQUEST, mc.MSG_WRITE_DONE
        ]
""",
        "entrypoints/openai/api_server.py": """
import multiprocessing
from atom.worker import work

if __name__ == "__main__":
    p = multiprocessing.get_context("spawn").Process(target=work)
    p.start()
    p.join()
    assert p.exitcode == 0
""",
    }
    for name, text in server.items():
        (tmp_path / "atom" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "atom" / name).write_text(text)
    stamps = tmp_path / "stamps"
    subprocess.run(
        [sys.executable, PROBE, "serve", stamps],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        check=True,
    )
    rows = [line.split() for line in stamps.read_text().splitlines()]
    # Only the write requests; the server process itself sent none.
    assert [r[2] for r in rows] == ["kvsend", "kvrecv"]
    assert int(rows[0][3]) <= int(rows[0][0]) <= int(rows[1][0])


def test_pd_pairs_each_kv_write_request_with_its_receipt_in_order(tmp_path):
    sent = [10**12 + k * 10**7 for k in range(10)]
    # Sent back to back, each received 100 us + 10 us k later, so receipts
    # overlap later sends.
    got = [s + 100_000 + 10_000 * k for k, s in enumerate(sent)]
    rows = [f"{s} 1 kvsend {s - 50} 0" for s in sent]
    rows += [f"{g} 2 kvrecv {g} 0" for g in got] + [f"{sent[0]} 3 send 0 0"]
    stamps = tmp_path / "stamps"
    stamps.write_text("\n".join(rows[::-1]) + "\n")
    hop = probe.pd(str(stamps))["kv_write_req"]
    assert hop["median_s"] == pytest.approx(145e-6)
    assert (hop["p10_s"], hop["p90_s"]) == pytest.approx((109e-6, 181e-6))
    assert hop["messages"] == 10
    # A write request prefill never received pairs nothing.
    stamps.write_text("\n".join(r for r in rows if str(got[3]) not in r) + "\n")
    with pytest.raises(
        SystemExit, match="kv_write_req: 9 arrivals, not one after each of 10"
    ):
        probe.pd(str(stamps))
