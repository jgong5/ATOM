"""A fresh model refusal stops an owned replay with an actual held HTTP reply."""

import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "corpus_refusal_harness_fixture", Path(__file__).with_name("test_cc_traces_run.py"))
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


run = harness.run_mod
RefusalWatch = run.refusal_module.RefusalWatch


def marker(path, why="outside the measured source domain"):
    path.write_text(json.dumps({"refused_by": "region model", "why": why,
                                "shape": {"total_tokens": 128}}))


@contextmanager
def http_peer(held):
    arrived, release = threading.Event(), threading.Event()
    if not held:
        release.set()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            arrived.set()
            release.wait(4)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1/completions", arrived
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


class ObservedProcesses(run.Processes):
    def __init__(self):
        self.stops = []

    def stop(self, proc, **kwargs):
        self.stops.append((proc.pid, proc.poll()))
        return super().stop(proc, **kwargs)


def client(url):
    source = ("import urllib.request; "
              f"r=urllib.request.Request({url!r},data=b'{{}}'); "
              "print(urllib.request.urlopen(r,timeout=5).read())")
    return [sys.executable, "-c", source]


def test_fresh_owned_marker_cancels_only_the_replay_and_preserves_exception(tmp_path):
    watch = RefusalWatch.create(tmp_path, "modelled", 1)
    path = watch.directory / "region_refusal_b1_t128_1.json"
    processes = ObservedProcesses()
    with http_peer(held=True) as (url, arrived):
        def refuse():
            assert arrived.wait(2)
            marker(path)
        writer = threading.Thread(target=refuse, daemon=True)
        writer.start()
        start = time.monotonic()
        observed = processes.run_observed(client(url), log=tmp_path / "replay.log", observe=watch.read)
        elapsed = time.monotonic() - start
        writer.join()
    assert elapsed < 2, "the held response must not wait for its HTTP timeout"
    refusal = observed["model_refusal"]
    assert refusal["exception_text"] == "outside the measured source domain"
    assert refusal["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed["exit"] != 0
    assert {pid for pid, _ in processes.stops} == {observed["pid"]}
    assert sum(status is None for _, status in processes.stops) == 1


def test_stale_and_foreign_markers_do_not_affect_successful_http(tmp_path):
    watch = RefusalWatch.create(tmp_path, "modelled", 1)
    stale = watch.directory / "region_refusal_b1_t128_1.json"
    marker(stale)
    os.utime(stale, ns=(0, 0))
    foreign = tmp_path / "region_refusal_b1_t128_2.json"
    marker(foreign)
    before = stale.read_bytes()
    processes = ObservedProcesses()
    with http_peer(held=False) as (url, _):
        observed = processes.run_observed(client(url), log=tmp_path / "success.log", observe=watch.read)
    assert observed["exit"] == 0 and observed["model_refusal"] is None
    assert all(status is not None for _, status in processes.stops)
    assert stale.read_bytes() == before


def test_preexisting_and_partial_markers_are_not_new_refusals(tmp_path):
    old = tmp_path / "region_refusal_b1_t128_1.json"
    marker(old)
    watch = RefusalWatch(tmp_path)
    assert watch.read() is None
    current = tmp_path / "region_refusal_b1_t128_2.json"
    current.write_text('{"refused_by":')
    assert watch.read() is None
    marker(current)
    assert watch.read()["path"] == str(current)


def test_graph_refusal_keeps_evidence_without_inventing_exception_text(tmp_path):
    watch = RefusalWatch.create(tmp_path, "modelled", 1)
    path = watch.directory / "refusal_b1_t128_1.json"
    path.write_text(json.dumps({"shape": {"total_tokens": 128}, "coverage": {"missing": 1},
                                "body_graph": [], "head_graph": []}))
    result = watch.read()
    assert result["kind"] == "library_coverage"
    assert result["exception_text"] is None
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
