"""A run that did not complete as asked must fail, at the boundary that reads it.

The checks these tests cover all existed once as printed warnings above numbers
that read as results, and each was believed for a while:

  * a client reported "0 failed" while the server had abandoned 236 requests;
  * a comparison with two requests declared and one engine record printed
    metrics and exited zero;
  * every saved run carried the *client's* git revision, which names the machine
    that sent the requests rather than the one that served them.

So these drive the real client against a real socket and the real comparison
over what it wrote. Nothing here inspects source text or calls a helper written
for the test: a stub server answers HTTP, `replay.main` posts to it, and
`compare.main` reads the file that came out. A check that is not wired to the
consumer is a check that does not run.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Import a `scripts/compass` program as a module, as an operator runs it."""
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, so a module that is not there yet raises
    # while its own body is still running.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load("replay")
compare = _load("compare")


# --------------------------------------------------------------------------
# a server that answers the endpoints the client actually uses


class _Stub:
    """Minimal OpenAI-shaped server with the two Compass endpoints.

    Timings are stamped by the *server*, which is the contract under test: the
    engine's clock is the only one whose numbers mean anything when the run is
    simulated.
    """

    def __init__(self, *, calibration: Path | None = None,
                 provenance: bool = True, drop_records: int = 0,
                 ttft: float = 1.0, per_token: float = 0.1,
                 break_ordering: bool = False, git: bool = True,
                 oracle: str | None = None, virtual: bool = False,
                 arrivals: str = "counter", abandon: int = 0,
                 short: int = 0, stop_early: int = 0,
                 overlong: int = 0, no_usage: bool = False):
        #: How many of the requests this server abandons: it answers 503 and
        #: never produces a completion, which is what a real engine that has
        #: run out of room does and what the client saw on the TP1 run where
        #: 59 of 62 timed out.
        self.abandon = abandon
        #: How many come back with fewer output tokens than were asked for,
        #: still claiming `finish_reason="length"` -- a generation the engine
        #: cut off rather than one the model ended.
        self.short = short
        #: How many stop on the model's own EOS one token short. The client
        #: asks for `ignore_eos`, so this cannot happen to a request it sent;
        #: it is here because a reply that claims it did happen still did not
        #: replay the registered length, and must not be waved through on the
        #: strength of its finish reason.
        self.stop_early = stop_early
        #: How many come back with *more* output tokens than were asked for.
        #: A mismatch in the other direction, and just as much a different
        #: workload from the registered one.
        self.overlong = overlong
        #: Answer without `usage.completion_tokens`, so nothing can be said
        #: about what was produced.
        self.no_usage = no_usage
        #: Requests answered so far, which is how the counts above are spent.
        self.served = 0
        self.calibration = calibration
        self.provenance = provenance
        self.git = git
        self.oracle = oracle
        #: Report as a predictor on a virtual clock, as `--compass-mode
        #: predict` does.
        self.virtual = virtual
        #: How arrivals are stamped. "counter" is the original: half a second
        #: apart whatever the server was doing. "serial" is what a real server
        #: does -- "now", which is after everything it has already served, so a
        #: request sent after a warmup arrives after the warmup finished.
        #: "epoch" is the defect this file exists to catch: every arrival at
        #: the start of the run however much the engine did first, which is
        #: what a virtual-clock engine does with a declared arrival.
        self.arrivals = arrivals
        self.drop_records = drop_records
        self.ttft = ttft
        self.per_token = per_token
        self.break_ordering = break_ordering
        #: Every completion body this server was sent, in arrival order, so a
        #: test can ask what the client actually requested rather than what
        #: it says it requests.
        self.bodies: list[dict] = []
        self.records: list[dict] = []
        self._next_arrival = 0.0
        self._served_until = 0.0
        self._lock = threading.Lock()

    def completion(self, body: dict):
        """The reply, or None for a request this server abandons.

        Abandonment is answered as a 503 by the handler rather than by never
        replying: a test that waited out a real timeout would take the
        client's `--timeout` to run, and what is under test is what the client
        does with a request that did not produce a completion, not how long it
        waits for one.
        """
        with self._lock:
            self.bodies.append(body)
        prompt_tokens = (len(body["prompt"].split()) if isinstance(body["prompt"], str)
                         else len(body["prompt"]))
        n = int(body["max_tokens"])
        with self._lock:
            self.served += 1
            mine = self.served
            if mine <= self.abandon:
                return None
            produced, reason = n, "length"
            if mine <= self.abandon + self.short:
                # Cut off: fewer tokens than asked for, and the engine still
                # says it stopped because it reached the length.
                produced, reason = max(0, n - 1), "length"
            elif mine <= self.abandon + self.short + self.stop_early:
                # Ended by the model rather than by the engine. Still short.
                produced, reason = max(1, n - 1), "stop"
            elif (mine <= self.abandon + self.short + self.stop_early
                    + self.overlong):
                # More than was asked for, reported as reaching the length.
                produced, reason = n + 1, "length"
            arrive = {"epoch": 0.0, "serial": self._served_until}.get(
                self.arrivals, self._next_arrival)
            self._next_arrival += 0.5
            rid = f"cmpl-{len(self.records)}"
            first = arrive + self.ttft
            finish = first + self.per_token * max(0, produced - 1)
            self._served_until = max(self._served_until, finish)
            if self.break_ordering and not self.records:
                first, arrive = arrive, first  # first token before arrival
            self.records.append({
                "request_id": rid, "seq_id": str(len(self.records)),
                "arrive_time": arrive, "first_token_time": first,
                "finish_time": finish,
                "ttft": first - arrive, "latency": finish - arrive,
            })
        usage = {"prompt_tokens": prompt_tokens,
                 "total_tokens": prompt_tokens + produced}
        if not self.no_usage:
            usage["completion_tokens"] = produced
        return {"id": rid, "object": "text_completion", "model": body["model"],
                "choices": [{"index": 0, "text": "x " * produced,
                             "finish_reason": reason}],
                "usage": usage}

    def requests_blob(self) -> dict:
        kept = list(self.records[self.drop_records:] if self.drop_records
                    else self.records)
        # The real endpoint drains as it reads, and the drain assertion depends
        # on exactly that: a store that answered a second time with the same
        # rows would make a correctly drained boundary look like a leaking one.
        self.records = []
        return {"count": len(kept), "clock": "wall", "requests": kept}

    def provenance_blob(self) -> dict:
        if not self.provenance:
            return {}
        import hashlib
        digest = None
        if self.calibration:
            digest = hashlib.sha256(self.calibration.read_bytes()).hexdigest()
        return {"server_revision": "deadbeef" if self.git else None,
                "server_code_sha256": "c0de" * 16,
                "model": "stub-model",
                "model_revision": "modelrev", "calibration_sha256": digest,
                "compass": None if not (self.oracle or self.virtual) else {
                    "oracle": self.oracle,
                    "oracle_options": {"table": "/no/such/sweep.jsonl"},
                    "oracle_option_sha256": {},
                    "enabled": True,
                    "virtual_clock": self.virtual,
                    "mode": "predict" if self.virtual else "measure"},
                "tensor_parallel_size": 1}


def _serve(stub: _Stub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep pytest output readable
            pass

        def _json(self, blob, code=200):
            body = json.dumps(blob).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._json({"data": [{"id": "stub-model"}]})
            elif self.path.startswith("/compass/requests"):
                self._json(stub.requests_blob())
            elif self.path.startswith("/compass/provenance"):
                blob = stub.provenance_blob()
                if not blob:
                    self._json({"detail": "not found"}, code=404)
                else:
                    self._json(blob)
            else:
                self._json({"detail": "no"}, code=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path.startswith("/compass/requests"):
                self._json(stub.requests_blob())
            elif self.path.startswith("/v1/completions"):
                reply = stub.completion(body)
                if reply is None:
                    self._json({"detail": "engine overloaded"}, code=503)
                else:
                    self._json(reply)
            else:
                self._json({"detail": "no"}, code=404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def trace(tmp_path: Path) -> Path:
    path = tmp_path / "trace.jsonl"
    rows = [{"arrival_s": 0.0, "input_tokens": 8, "output_tokens": 4},
            {"arrival_s": 0.0, "input_tokens": 16, "output_tokens": 4},
            {"arrival_s": 0.0, "input_tokens": 32, "output_tokens": 4}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _run(stub: _Stub, trace: Path, out: Path, extra=()) -> int:
    server = _serve(stub)
    try:
        return replay.main(["--port", str(server.server_address[1]),
                            "--trace", str(trace), "--out", str(out),
                            "--check-lengths", *extra])
    finally:
        server.shutdown()
        server.server_close()


def test_client_wall_window_excludes_warmup_drain_and_reporting(tmp_path, trace, monkeypatch):
    class Clock:
        value = 0.0

        def monotonic(self):
            return self.value

        def time(self):
            return 1_700_000_000.0 + self.value

        def sleep(self, seconds):
            self.value += seconds

    clock = Clock()

    class DelayedStub(_Stub):
        def completion(self, body):
            response = super().completion(body)
            clock.sleep(30.0 if self.served == 1 else 2.0)
            return response

        def requests_blob(self):
            if self.records:
                clock.sleep(5.0 if self.served == 1 else 7.0)
            return super().requests_blob()

        def provenance_blob(self):
            if self.served == 4:
                clock.sleep(20.0)
            return super().provenance_blob()

    monkeypatch.setattr(replay, "_time", clock)
    out = tmp_path / "timed.json"
    assert _run(DelayedStub(arrivals="serial"), trace, out,
                ("--prepare", "1", "--pace")) == 0
    run = json.loads(out.read_text())["run"]
    window = replay.read_wall_window(run["wall_execution"])
    assert window["seconds"] == 6.0
    assert window["started_at"] == 1_700_000_035.0
    assert window["ended_at"] == 1_700_000_041.0
    assert run["prepare"]["wall_seconds"] == 30.0
    assert clock.value == 68.0  # preparation + drain + requests + reporting


def test_pretokenization_precedes_pacing_but_remains_in_execution(tmp_path, monkeypatch):
    class Clock:
        value = 0.0

        def monotonic(self):
            return self.value

        def time(self):
            return 1_700_000_000.0 + self.value

        def sleep(self, seconds):
            self.value += seconds

    clock = Clock()

    class Tokenizer:
        init_kwargs = {"_commit_hash": "test-tokenizer-revision"}

        def encode(self, prompt, *, add_special_tokens):
            assert add_special_tokens is False
            words = prompt.split()
            clock.sleep(0.01 if len(words) == 2 else 0.10)
            return [{"the": 101, "of": 102}[word] for word in words]

    monkeypatch.setattr(replay, "_time", clock)
    monkeypatch.setattr(replay, "_load_prompt_tokenizer", lambda model: Tokenizer())
    sent = []
    original_send = replay._send

    def send(url, body, timeout):
        sent.append((len(body["prompt"]), clock.monotonic(), body["prompt"]))
        return original_send(url, body, timeout)

    monkeypatch.setattr(replay, "_send", send)
    path, out = tmp_path / "paced.jsonl", tmp_path / "paced.json"
    path.write_text('\n'.join(json.dumps(row) for row in (
        {"arrival_s": 0.0, "input_tokens": 2, "output_tokens": 1},
        {"arrival_s": 0.25, "input_tokens": 10, "output_tokens": 1})))
    assert _run(_Stub(), path, out, ("--pace", "--pretokenize")) == 0
    by_size = {size: (at, ids) for size, at, ids in sent}
    assert by_size[2][0] == pytest.approx(0.11)
    assert by_size[10][0] - by_size[2][0] == pytest.approx(0.25)
    assert by_size[2][1] == [101, 101]
    assert by_size[10][1] == [102] + [101] * 9
    run = json.loads(out.read_text())["run"]
    assert run["prompt_encoding"]["conversion_seconds"] == pytest.approx(0.11)
    assert run["prompt_encoding"]["conversion_in_execution"] is True
    assert run["wall_execution"]["seconds"] == pytest.approx(0.36)


def test_pretokenization_refuses_a_tokenizer_that_changes_workload_length(tmp_path, trace,
                                                                        monkeypatch):
    class WrongTokenizer:
        def encode(self, prompt, **kwargs):
            return [1]

    monkeypatch.setattr(replay, "_load_prompt_tokenizer", lambda model: WrongTokenizer())
    with pytest.raises(ValueError, match="expected 8"):
        _run(_Stub(), trace, tmp_path / "wrong.json", ("--pretokenize",))


class TestARunThatDidNotCompleteExitsNonZero:
    """Reporting "0 failed" was the client's opinion of its own sending.

    A TP1 development run answered 3 of 62 requests -- the other 59 timed out
    -- and `replay.py` exited 0, wrote a manifest saying `failed: 0`, and the
    artifact was read as a measurement. The client counted only the requests
    whose *send* raised, and a request that was accepted and then abandoned
    raised nothing on the way out.

    These drive the real client against a socket that abandons, cuts short and
    under-reports, and assert on the exit code an operator or the acceptance
    harness actually reads.
    """

    def _manifest(self, out: Path) -> dict:
        return json.loads(out.read_text())["run"]

    def test_a_workload_that_completed_still_exits_zero(self, tmp_path, trace):
        """The control. Everything below has to fail *against* this."""
        out = tmp_path / "run.json"
        assert _run(_Stub(), trace, out) == 0
        manifest = self._manifest(out)
        assert manifest["complete"] is True
        assert (manifest["completed"], manifest["requests"]) == (3, 3)
        assert (manifest["failed"], manifest["missing"],
                manifest["truncated"]) == (0, 0, 0)
        assert manifest["incomplete_reasons"] is None

    def test_requests_the_engine_abandoned_fail_the_run(self, tmp_path, trace):
        out = tmp_path / "run.json"
        assert _run(_Stub(abandon=2), trace, out) == replay.INCOMPLETE_EXIT
        manifest = self._manifest(out)
        assert manifest["complete"] is False
        assert manifest["failed"] == 2
        assert manifest["completed"] == 1

    def test_the_artifact_survives_the_failure(self, tmp_path, trace):
        """Written first, then failed. Deleting it would leave only a log
        line, and the file is what says which requests did not complete."""
        out = tmp_path / "run.json"
        assert _run(_Stub(abandon=3), trace, out) == replay.INCOMPLETE_EXIT
        blob = json.loads(out.read_text())
        assert len(blob["results"]) == 3
        assert not any(r["ok"] for r in blob["results"])

    def test_the_reasons_are_counted_rather_than_sampled(self, tmp_path, trace,
                                                         capsys):
        """Fifty-nine timeouts and fifty-eight timeouts plus one 400 are
        different runs, and only the second is worth getting out of bed for.
        The old client printed the first failure and nothing else."""
        out = tmp_path / "run.json"
        assert _run(_Stub(abandon=2), trace, out) == replay.INCOMPLETE_EXIT
        reasons = self._manifest(out)["incomplete_reasons"]["failed"]
        assert [r["requests"] for r in reasons] == [2]
        assert "503" in reasons[0]["reason"]
        assert "2 failed: " in capsys.readouterr().err

    def test_a_completion_cut_short_is_not_a_completion(self, tmp_path, trace):
        """Three of four tokens, and the engine still calling it "length".
        The step sequence such a run performed is not the trace's, so the
        artifact's metrics are over a workload nobody asked for."""
        out = tmp_path / "run.json"
        assert _run(_Stub(short=1), trace, out) == replay.INCOMPLETE_EXIT
        manifest = self._manifest(out)
        assert (manifest["truncated"], manifest["failed"]) == (1, 0)
        assert manifest["completed"] == 2
        reason = manifest["incomplete_reasons"]["truncated"][0]["reason"]
        assert "produced 3 output tokens where 4 were registered" in reason

    def test_a_sequence_the_model_ended_itself_is_still_short(self, tmp_path,
                                                              trace):
        """`finish_reason="stop"` names the cause, not a dispensation. The
        client asks for `ignore_eos`, so a reply that stopped early did not
        run the registered decode lengths however it reports itself, and a
        comparison against it would be over a different workload."""
        out = tmp_path / "run.json"
        assert _run(_Stub(stop_early=3), trace, out) == replay.INCOMPLETE_EXIT
        manifest = self._manifest(out)
        assert (manifest["truncated"], manifest["completed"]) == (3, 0)
        assert manifest["complete"] is False
        assert "'stop'" in (
            manifest["incomplete_reasons"]["truncated"][0]["reason"])

    def test_a_reply_longer_than_asked_for_is_a_mismatch_too(self, tmp_path,
                                                             trace):
        """The other direction. An extra token is an extra decode step, so an
        over-produced reply is no more the registered workload than a short
        one -- and a `got >= want` check would have taken it."""
        out = tmp_path / "run.json"
        assert _run(_Stub(overlong=1), trace, out) == replay.INCOMPLETE_EXIT
        manifest = self._manifest(out)
        assert (manifest["truncated"], manifest["completed"]) == (1, 2)
        reason = manifest["incomplete_reasons"]["truncated"][0]["reason"]
        assert "produced 5 output tokens where 4 were registered" in reason

    def test_fixed_length_generation_is_what_was_asked_for(self, tmp_path,
                                                           trace):
        """Enforcing the count is only honest if the client asked for it.
        `CompletionRequest.ignore_eos` is what says "run the whole length"
        (atom/entrypoints/openai/protocol.py:246 -> api_server.py:1697); this
        client never sent it, so every request it made was free to stop early
        and the count check below could be argued with."""
        stub = _Stub()
        out = tmp_path / "run.json"
        assert _run(stub, trace, out) == 0
        assert len(stub.bodies) == 3
        assert all(b.get("ignore_eos") is True for b in stub.bodies)

    def test_a_reply_that_counts_nothing_cannot_be_read_as_complete(
            self, tmp_path, trace):
        """`compare.py` already refuses a run whose replies carry no
        `usage.completion_tokens`, so a client that accepted one would write
        an artifact nothing downstream will take."""
        out = tmp_path / "run.json"
        assert _run(_Stub(no_usage=True), trace, out) == replay.INCOMPLETE_EXIT
        manifest = self._manifest(out)
        assert manifest["truncated"] == 3
        assert "usage.completion_tokens" in (
            manifest["incomplete_reasons"]["truncated"][0]["reason"])

    def test_the_incomplete_exit_is_its_own_code(self):
        """Not the refusal's, which the harness reads as "do not retry", and
        not the barrier's: three boundaries, three codes, so a log line says
        which one rejected the run."""
        assert replay.INCOMPLETE_EXIT not in (0, 1, 3)

    def test_the_shortfall_is_named_without_the_length_check(self, tmp_path,
                                                             trace, capsys):
        """`--check-lengths` is optional and the tally is not. Run without it,
        as the acceptance plan's modelled side is, and the warning still says
        how much of the workload ran."""
        stub = _Stub(abandon=2)
        server = _serve(stub)
        out = tmp_path / "run.json"
        try:
            code = replay.main(["--port", str(server.server_address[1]),
                                "--trace", str(trace), "--out", str(out)])
        finally:
            server.shutdown()
            server.server_close()
        assert code == replay.INCOMPLETE_EXIT
        assert self._manifest(out)["complete"] is False
        assert "1 of 3 requests completed" in capsys.readouterr().err


class TestTheClientRecordsWhoServedIt:
    def test_the_manifest_carries_the_server_not_the_client(self, tmp_path, trace):
        calibration = tmp_path / "sweep.jsonl"
        calibration.write_text('{"seconds": 1.0}\n')
        out = tmp_path / "real.json"
        assert _run(_Stub(calibration=calibration), trace, out) == 0
        manifest = json.loads(out.read_text())["run"]
        assert manifest["server_revision"] == "deadbeef"
        assert manifest["model_revision"] == "modelrev"
        import hashlib
        assert manifest["calibration_sha256"] == hashlib.sha256(
            calibration.read_bytes()).hexdigest()

    def test_a_tree_without_git_is_still_attributable(self, tmp_path, trace):
        """The GPU nodes hold an rsync copy, so `rev-parse` fails there.

        A digest of the source says what a revision was wanted for -- which
        code ran -- and says it for a working tree with uncommitted edits too.
        """
        calibration = tmp_path / "sweep.jsonl"
        calibration.write_text('{"seconds": 1.0}\n')
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        _run(_Stub(calibration=calibration, git=False), trace, real)
        _run(_Stub(calibration=calibration, git=False, ttft=1.1), trace, modelled)
        assert json.loads(real.read_text())["run"]["server_revision"] is None
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 0

    def test_a_server_without_the_endpoint_leaves_it_unattributed(
            self, tmp_path, trace):
        """Not an error at the client. It becomes one at the consumer."""
        out = tmp_path / "real.json"
        assert _run(_Stub(provenance=False), trace, out) == 0
        assert json.loads(out.read_text())["run"]["server_revision"] is None


def _pair(tmp_path, trace, *, real_kwargs=None, modelled_kwargs=None):
    calibration = tmp_path / "sweep.jsonl"
    calibration.write_text('{"seconds": 1.0}\n')
    real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
    _run(_Stub(calibration=calibration, **(real_kwargs or {})), trace, real)
    _run(_Stub(calibration=calibration, ttft=1.1, per_token=0.11,
               **(modelled_kwargs or {})), trace, modelled)
    return real, modelled


class TestTheComparisonRefusesAnInvalidRun:
    def test_a_complete_pair_is_reported(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace)
        summary = tmp_path / "summary.json"
        rc = compare.main(["--real", str(real), "--modelled", str(modelled),
                           "--repeats", "3", "--summary-out", str(summary)])
        assert rc == 0
        report = json.loads(summary.read_text())
        assert report["requests"] == 3
        # ttft 1.0 -> 1.1 is exactly +10% on every request, which is the point:
        # a comparison that cannot reproduce a known ratio cannot be trusted
        # with an unknown one.
        assert report["metrics"]["ttft"]["error_pct"]["median"] == pytest.approx(10.0)
        assert report["metrics"]["tpot"]["real"]["n"] == 3
        assert report["metrics"]["throughput_tok_s"]["real"] > 0

    def test_a_missing_engine_record_fails(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace,
                               real_kwargs={"drop_records": 1})
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "no engine record" in capsys.readouterr().out

    def test_a_first_token_before_arrival_fails(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace,
                               real_kwargs={"break_ordering": True})
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "out of order" in capsys.readouterr().out

    def test_a_short_run_fails_against_the_expected_count(self, tmp_path, trace,
                                                          capsys):
        real, modelled = _pair(tmp_path, trace)
        assert compare.main(["--real", str(real), "--modelled", str(modelled),
                             "--repeats", "4"]) == 1
        assert "expected 4 requests" in capsys.readouterr().out

    def test_two_workloads_are_two_experiments(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(modelled.read_text())
        blob["run"]["trace_sha256"] = "0" * 64
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "different workload" in capsys.readouterr().out

    def test_different_lengths_on_the_two_sides_fail(self, tmp_path, trace,
                                                     capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(modelled.read_text())
        blob["workload"][1]["input_tokens"] = 999
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "different lengths" in capsys.readouterr().out

    def test_missing_provenance_fails_unless_waived(self, tmp_path, trace,
                                                    capsys):
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        _run(_Stub(provenance=False), trace, real)
        _run(_Stub(provenance=False, ttft=1.1), trace, modelled)
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "server_revision" in capsys.readouterr().out
        assert compare.main(["--real", str(real), "--modelled", str(modelled),
                             "--allow-missing-provenance"]) == 0

    def test_a_measured_run_needs_no_calibration_digest(self, tmp_path, trace):
        """There is no table to name when nothing was fitted.

        The real half of a comparison runs in measure mode. Demanding a
        calibration digest from it refused every ground-truth run for a reason
        that was not a defect.
        """
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        _run(_Stub(), trace, real)
        _run(_Stub(ttft=1.1), trace, modelled)
        assert json.loads(real.read_text())["run"]["calibration_sha256"] is None
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 0

    def test_a_predicted_run_must_name_what_it_was_fitted_to(
            self, tmp_path, trace, capsys):
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        _run(_Stub(), trace, real)
        _run(_Stub(oracle="atom.compass.core.cost.calibrated.CalibratedCostOracle",
                   ttft=1.1), trace, modelled)
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "read no such file" in capsys.readouterr().out

    def test_a_setting_that_is_not_a_file_needs_no_digest(
            self, tmp_path, trace, capsys):
        """The source factory is configured with flags as well as tables.

        `require_complete=true` is not a path and has no digest to report.
        Demanding one refused every run of the integrated factory for a reason
        that was not a defect -- the same mistake this check made once for the
        measured side.
        """
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        _run(_Stub(), trace, real)
        _run(_Stub(oracle="atom.compass.runtime.source_oracle.source_cost_oracle",
                   ttft=1.1), trace, modelled)
        blob = json.loads(modelled.read_text())
        blob["run"]["server"]["compass"]["oracle_options"] = {
            "require_complete": "true", "head": "true",
            "regions": "source-27b-tp2", "model": "Qwen/Qwen3.8-27B"}
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 0
        assert "read no such file" not in capsys.readouterr().out

    @pytest.mark.parametrize("value", ["/m/prices.json", "./prices.json",
                                       "~/prices.json", "prices.jsonl",
                                       "prices.json:graph.json"])
    def test_anything_written_as_a_file_still_needs_one(
            self, tmp_path, trace, capsys, value):
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        _run(_Stub(), trace, real)
        _run(_Stub(oracle="atom.compass.runtime.source_oracle.source_cost_oracle",
                   ttft=1.1), trace, modelled)
        blob = json.loads(modelled.read_text())
        blob["run"]["server"]["compass"]["oracle_options"] = {"price": value}
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "read no such file" in capsys.readouterr().out

    def test_an_unverified_length_check_fails(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(real.read_text())
        blob["run"]["prompt_lengths"] = "not requested"
        real.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "prompt lengths were not verified" in capsys.readouterr().out

    def test_a_timeout_flag_in_the_manifest_fails(self, tmp_path, trace, capsys):
        """The arrival barrier's timeout has to reach the result to mean anything."""
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(real.read_text())
        blob["run"]["arrival_barrier_timed_out"] = True
        real.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
        assert "arrival barrier timed out" in capsys.readouterr().out

    def test_report_anyway_prints_but_still_fails(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace,
                               real_kwargs={"drop_records": 1})
        assert compare.main(["--real", str(real), "--modelled", str(modelled),
                             "--report-anyway"]) == 1
        out = capsys.readouterr().out
        assert "REPORTING ANYWAY" in out and "throughput" in out


class TestMetricsMatchTheirDefinitions:
    """POC_STATUS.md section 0 is the contract; this is where it is enforced."""

    def test_tpot_divides_by_one_fewer_than_the_tokens(self, tmp_path, trace):
        real, modelled = _pair(tmp_path, trace)
        report = compare.compare(compare.load_run(str(real), "real"),
                                 compare.load_run(str(modelled), "modelled"))
        # The stub produced 4 tokens at 0.1s each after the first, so decode
        # spans 0.3s over 3 intervals.
        assert report["metrics"]["tpot"]["real"]["median"] == pytest.approx(0.1)

    def test_throughput_is_tokens_over_the_engine_clock_window(self, tmp_path,
                                                              trace):
        real, modelled = _pair(tmp_path, trace)
        run = compare.load_run(str(real), "real")
        got = compare.metrics(run, sorted(run.joined))
        assert got["output_tokens"] == 12
        assert got["throughput_tok_s"] == pytest.approx(
            12 / got["window_s"])

    def test_a_request_with_one_token_has_no_tpot(self, tmp_path):
        path = tmp_path / "single.jsonl"
        path.write_text(json.dumps({"arrival_s": 0.0, "input_tokens": 8,
                                    "output_tokens": 1}))
        out = tmp_path / "one.json"
        _run(_Stub(), path, out)
        run = compare.load_run(str(out), "real")
        got = compare.metrics(run, sorted(run.joined))
        assert got["tpot"] == {}


class TestTokenCountsAreTheServersNotTheWorkloads:
    """What was *produced*, never what was *asked for*.

    Substituting the workload's requested length for a missing `usage` block
    makes every run agree with itself by construction: TPOT divides by the
    number that was requested, and two sides that produced different numbers of
    tokens compare as if they had produced the same ones.
    """

    def test_a_response_without_usage_is_refused(self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(modelled.read_text())
        blob["results"][1]["response"].pop("usage")
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real),
                             "--modelled", str(modelled)]) == 1
        assert "no usage.completion_tokens" in capsys.readouterr().out

    def test_a_non_integer_token_count_is_refused(self, tmp_path, trace,
                                                  capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(modelled.read_text())
        blob["results"][0]["response"]["usage"]["completion_tokens"] = None
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real),
                             "--modelled", str(modelled)]) == 1
        assert "no usage.completion_tokens" in capsys.readouterr().out

    def test_different_produced_tokens_fail_rather_than_warn(
            self, tmp_path, trace, capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(modelled.read_text())
        blob["results"][2]["response"]["usage"]["completion_tokens"] = 3
        modelled.write_text(json.dumps(blob))
        assert compare.main(["--real", str(real),
                             "--modelled", str(modelled)]) == 1
        out = capsys.readouterr().out
        assert "different numbers of tokens" in out
        assert "2: 4 against 3" in out

    def test_report_anyway_still_carries_the_caveat(self, tmp_path, trace,
                                                    capsys):
        real, modelled = _pair(tmp_path, trace)
        blob = json.loads(modelled.read_text())
        blob["results"][2]["response"]["usage"]["completion_tokens"] = 3
        modelled.write_text(json.dumps(blob))
        summary = tmp_path / "summary.json"
        assert compare.main(["--real", str(real), "--modelled", str(modelled),
                             "--report-anyway",
                             "--summary-out", str(summary)]) == 1
        report = json.loads(summary.read_text())
        assert "not like for like" in report["throughput_warning"]
        assert report["problems"]["pair"]


class TestTheQuantileConventionIsFrozen:
    def test_the_summary_says_which_quantile_it_means(self, tmp_path, trace):
        real, modelled = _pair(tmp_path, trace)
        summary = tmp_path / "summary.json"
        compare.main(["--real", str(real), "--modelled", str(modelled),
                      "--summary-out", str(summary)])
        report = json.loads(summary.read_text())
        assert report["quantile_convention"] == compare.QUANTILE_CONVENTION
        assert "no interpolation" in report["quantile_convention"]

    def test_the_median_is_the_documented_order_statistic(self):
        """Not `statistics.median`: no averaging of the middle two."""
        got = compare._quantiles([1.0, 2.0, 3.0, 4.0])
        assert got["median"] == 3.0 and got["p90"] == 4.0
        assert got["mean"] == 2.5


class TestTheExpectedCountIsRequestsNotRepeats:
    def test_both_spellings_reach_the_same_check(self, tmp_path, trace,
                                                 capsys):
        real, modelled = _pair(tmp_path, trace)
        for flag in ("--expect-requests", "--repeats"):
            assert compare.main(["--real", str(real), "--modelled",
                                 str(modelled), flag, "4"]) == 1
            assert "expected 4 requests" in capsys.readouterr().out


class TestPreparationMustPrecedeTheMeasurement:
    """Warming a server is only warming if the measurement comes after it.

    The first warmed 27B cell drained its preparation correctly -- no
    preparation row reached the result -- and was still wrong by +71% TTFT.
    A declared arrival is an offset from the engine's *epoch*, and the process
    that stamps arrivals holds a virtual clock frozen there, so 14.4s of
    preparation moved the engine but not the origin its workload was measured
    against. Every measured request was stamped as having arrived before the
    warmup that preceded it, and the warmup landed inside its TTFT.

    Two checks, because either alone leaves the hole open: the client refuses
    to warm a predictor at all, and the comparison refuses any saved run whose
    arrivals precede its own preparation, whoever produced it.
    """

    def test_a_predictor_is_not_warmed_by_executing_a_warmup(
            self, tmp_path, trace, capsys):
        """It has no kernels to compile; it would only model the warmup."""
        out = tmp_path / "modelled.json"
        assert _run(_Stub(virtual=True), trace, out, ("--prepare", "2")) == 3
        err = capsys.readouterr().err
        assert "refusing to warm a predictor" in err
        assert "fresh empty run" in err
        assert not out.exists(), "a refused run must not leave a result behind"

    def test_a_real_server_is_still_warmed(self, tmp_path, trace):
        out = tmp_path / "real.json"
        assert _run(_Stub(arrivals="serial"), trace, out,
                    ("--prepare", "2")) == 0
        prepare = json.loads(out.read_text())["run"]["prepare"]
        assert prepare["requested"] == 2 and prepare["drained"] is True

    def test_arrivals_stamped_before_the_warmup_finished_are_refused(
            self, tmp_path, trace, capsys):
        """The defect itself, end to end: drained, and still not measurable."""
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        assert _run(_Stub(arrivals="epoch"), trace, real,
                    ("--prepare", "2")) == 0
        assert _run(_Stub(arrivals="serial", ttft=1.1), trace, modelled) == 0
        assert compare.main(["--real", str(real), "--modelled",
                             str(modelled)]) == 1
        out = capsys.readouterr().out
        assert "preparation ran until engine time" in out
        assert "inside every measured TTFT and latency" in out

    def test_a_warmed_run_whose_arrivals_follow_it_is_reportable(
            self, tmp_path, trace):
        real, modelled = tmp_path / "real.json", tmp_path / "modelled.json"
        assert _run(_Stub(arrivals="serial"), trace, real,
                    ("--prepare", "2")) == 0
        assert _run(_Stub(arrivals="serial", ttft=1.1), trace, modelled) == 0
        assert compare.main(["--real", str(real), "--modelled",
                             str(modelled)]) == 0

    def test_preparation_does_not_send_the_workloads_own_prompts(
            self, tmp_path, trace):
        """Distinct text, so prefix caching cannot pre-fill what is measured."""
        stub = _Stub(arrivals="serial")
        server = _serve(stub)
        try:
            replay.main(["--port", str(server.server_address[1]),
                         "--trace", str(trace), "--check-lengths",
                         "--prepare", "3", "--out", str(tmp_path / "r.json")])
        finally:
            server.shutdown()
            server.server_close()
        prepared = {replay._prompt(8, replay._PREPARE_PROMPT_BASE + i)
                    for i in range(3)}
        measured = {replay._prompt(8, i) for i in range(3)}
        assert not (prepared & measured)
