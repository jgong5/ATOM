# SPDX-License-Identifier: MIT
"""A real server recording its steps records its served tokenizer and HTTP intake.

`compass_run.tokenizer` is what the API server's ``main`` calls on the tokenizer
it serves. On a real run with the step record's variable set it writes each
tokenizer call beside the record, and stamps each HTTP request's intake there
as ``scripts/compass/http_intake.py serve`` does. The requests here go through
Starlette, the API server's binding of ``apply_chat_template``, ATOM's
`InputOutputProcessor` on a pool thread and ATOM's `CoreManager`, with no GPU.
"""

import asyncio
import hashlib
import importlib.util
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from atom.compass import parity
from atom.compass import run as compass_run
from atom.entrypoints.openai import api_server, chat_encoders
from atom.model_engine import segment_tokenizer
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.llm_engine import InputOutputProcessor
from atom.sampling_params import SamplingParams

TREE = pathlib.Path(__file__).resolve().parents[2]
_loaded = importlib.util.spec_from_file_location(
    "http_intake", TREE / "scripts/compass/http_intake.py"
)
probe = importlib.util.module_from_spec(_loaded)
_loaded.loader.exec_module(probe)
WORDS = ["hi", "there", "user", "assistant", "how", "are", "you"]


@pytest.fixture
def tok(tmp_path):
    """A fast tokenizer loaded from a model directory, as a server loads one."""
    vocab = {"[UNK]": 0} | {w: i + 1 for i, w in enumerate(WORDS)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    saved = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    saved.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    saved.save_pretrained(tmp_path / "model")
    return AutoTokenizer.from_pretrained(tmp_path / "model")


@pytest.fixture
def patched(monkeypatch):
    """Everything a recording server patches, put back after the test."""
    for owner, name in (
        (Request, "body"),
        (Starlette, "__call__"),
        (InputOutputProcessor, "preprocess_fanout"),
        (CoreManager, "add_request"),
        (DisaggCoreManager, "add_request"),
        (CoreManager, "_mark_seq_prefill_complete"),
        (CoreManager, "_deliver_terminal_output"),
        (ThreadPoolExecutor, "submit"),
        (segment_tokenizer.SegmentTokenizer, "_encode_cached"),
        (segment_tokenizer, "_shared_chars"),
    ):
        monkeypatch.setattr(owner, name, vars(owner)[name])
    render = chat_encoders.apply_chat_template
    for module in list(sys.modules.values()):
        if getattr(module, "__dict__", {}).get("apply_chat_template") is render:
            monkeypatch.setattr(module, "apply_chat_template", render)
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.delenv(parity.ENV, raising=False)


@pytest.fixture
def record(patched, monkeypatch, tmp_path):
    """A real run recording its steps; the record's directory."""
    monkeypatch.setenv(parity.ENV, str(tmp_path / "steps"))
    return tmp_path / "steps"


def _calls(record) -> tuple[dict, list[dict]]:
    compass_run._move_line_files()  # as the server exits
    path = record / compass_run.TOKENIZER_RECORD
    head, *calls = (json.loads(line) for line in path.read_text().splitlines())
    return head, calls


def test_a_recorded_server_writes_each_tokenizer_call_it_serves(record, tok):
    compass_run.tokenizer(tok, None)
    ids = tok.encode("hi there user", add_special_tokens=False)
    assert tok.decode(ids) == "hi there user"
    tok._tokenizer.encode_batch(["hi", "how are you"])
    head, calls = _calls(record)
    loaded = (pathlib.Path(tok.name_or_path) / "tokenizer.json").read_bytes()
    assert head == {
        "backend": "fast",
        "fingerprint": "sha256:" + hashlib.sha256(loaded).hexdigest(),
    }
    # A backend call the encode makes is the encode's: one line per served call.
    got = [(c["call"], c["tokens"], c["items"]) for c in calls]
    assert got == [("encode", 3, 1), ("decode", 3, 1), ("encode_batch", 4, 2)]
    assert all(c["t_start_ns"] <= c["t_end_ns"] for c in calls)
    assert calls[0]["t_end_ns"] <= calls[1]["t_start_ns"]


def test_a_recorded_server_writes_each_segment_tokenizer_call_around_its_own(
    record, monkeypatch
):
    # Every prompt takes the cache, and none is checked with a plain encode.
    monkeypatch.setattr(segment_tokenizer, "_MIN_CACHED_CHARS", 0)
    monkeypatch.setattr(segment_tokenizer, "_VERIFY_FIRST", 0)
    vocab = {"[UNK]": 0} | {w: i + 1 for i, w in enumerate(WORDS)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.add_special_tokens([AddedToken("<sep>", special=True)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    seg = segment_tokenizer.SegmentTokenizer.maybe_create(tok, 1 << 20)
    compass_run.tokenizer(tok, None)  # the engine is built before the wrap
    ids = [seg.encode("hi there<sep>how are you") for _ in range(2)]
    _, calls = _calls(record)
    # Each call's tokens are the prompt's ids; the second encodes nothing.
    got = [(c["call"], c["tokens"], c["items"]) for c in calls]
    assert got == [("encode_batch", 5, 2)] + [("segment", len(i), 1) for i in ids]
    batch, first = calls[:2]
    assert batch["thread"] == first["thread"]
    assert first["t_start_ns"] < batch["t_start_ns"] < batch["t_end_ns"]
    assert batch["t_end_ns"] < first["t_end_ns"]


def test_a_recorded_segment_tokenizer_call_writes_what_its_stored_prefix_search_compared(
    record, monkeypatch
):
    # Every prompt takes the cache unchecked, and its first two characters bucket it.
    monkeypatch.setattr(segment_tokenizer, "_MIN_CACHED_CHARS", 0)
    monkeypatch.setattr(segment_tokenizer, "_VERIFY_FIRST", 0)
    monkeypatch.setattr(segment_tokenizer, "_PREFIX_KEY_CHARS", 2)
    vocab = {"[UNK]": 0} | {w: i + 1 for i, w in enumerate(WORDS)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.add_special_tokens([AddedToken("<sep>", special=True)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    seg = segment_tokenizer.SegmentTokenizer.maybe_create(tok, 1 << 20)
    compass_run.tokenizer(tok, None)
    # Two conversations that share "hi there<sep>", 13 characters, each stored,
    # then a third prompt that the search compares with both.
    for prompt in (
        "hi there<sep>how<sep>",
        "hi there<sep>you<sep>",
        "hi there<sep>are",
    ):
        seg.encode(prompt)
    _, calls = _calls(record)
    searched = [c["compared_chars"] for c in calls if c["call"] == "segment"]
    assert searched == [0, 13, 26]


def _server(tok):
    """A chat handler as the API server's: the body, the template through the
    server's binding, the tokenizer pool job, the engine send."""
    processor = InputOutputProcessor(
        SimpleNamespace(hf_config=SimpleNamespace(model_type="llama")),
        tokenizer=tok,
        block_size=16,
    )
    manager = CoreManager.__new__(CoreManager)
    manager.label, manager.pp_size, manager.local_engine_count = "t", 1, 1
    manager._seq_id_to_callback, manager.sent = {}, []
    manager._send_request = lambda _rank, payload: manager.sent.append(payload)

    async def chat(request):
        messages = (await request.json())["messages"]
        prompt = api_server.apply_chat_template(tok, None, messages)

        def do_preprocess():
            return processor.preprocess(prompt, SamplingParams())

        seq = await asyncio.get_running_loop().run_in_executor(None, do_preprocess)
        manager.add_request([seq])
        return JSONResponse({"prompt_tokens": seq.num_prompt_tokens})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


async def _post(app, body: bytes) -> dict:
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "root_path": "",
    }
    await app(scope, receive, send)
    return json.loads(sent[-1]["body"])


def test_a_recorded_server_stamps_each_request_s_intake_as_the_probe_does(record, tok):
    app = _server(tok)  # the engine is built before the server wraps
    compass_run.tokenizer(tok, None)
    bodies = [
        json.dumps({"messages": [{"role": "user", "content": text}]}).encode()
        for text in ("hi there", "how are you user")
    ]

    async def lone():
        return [await _post(app, body) for body in bodies]

    answers = asyncio.run(lone())
    compass_run._move_line_files()  # as the server exits
    rows = [
        line.split() for line in (record / "intake.stamps").read_text().splitlines()
    ]
    kinds = [(r[2], int(r[4])) for r in rows]
    assert kinds == [
        k
        for body, text in zip(bodies, ("hi there ", "how are you user "))
        for k in [("app", 0), ("recv", 0), ("body", len(body)), ("tmpl", len(text))]
        + [("prep", 0), ("wait", 0), ("send", 0)]
    ]
    # The loop waits on the body's receive inside its read, and on the pool job
    # from its submit to its resume.
    for k in range(len(bodies)):
        app, receive, body, _, job, waited, _ = (
            (int(r[3]), int(r[0])) for r in rows[7 * k : 7 * k + 7]
        )
        assert body[0] <= receive[0] <= receive[1] <= body[1]
        assert waited[0] <= job[0] <= job[1] <= waited[1]
    assert all(int(r[3]) <= int(r[0]) and int(r[6]) <= int(r[5]) for r in rows)
    # The pool job runs off the event loop's thread; the rest on it.
    threads = {r[2]: r[1] for r in rows}
    assert threads["prep"] != threads["app"] == threads["tmpl"] == threads["send"]
    assert threads["recv"] == threads["wait"] == threads["app"]
    # The pool job's encode is recorded too, its tokens the prompt's.
    _, calls = _calls(record)
    assert [(c["call"], c["tokens"]) for c in calls] == [
        ("encode", a["prompt_tokens"]) for a in answers
    ]
    # The lone-request reader of the probe's stamps reads the record as it is.
    assert probe.idle(str(record / "intake.stamps"))["requests"] == len(bodies)


def test_a_recorded_server_stamps_each_output_row_taken_in_and_each_stream_head(
    record, tok
):
    compass_run.tokenizer(tok, None)
    send_s = 0.001  # each send of a response message takes this long
    manager = CoreManager.__new__(CoreManager)
    manager._seq_load, manager._seq_id_to_callback = {}, {}

    async def streamed(request):
        await request.body()
        # The output thread takes in a streamed row and a finished sequence's.
        manager._mark_seq_prefill_complete(1)
        manager._deliver_terminal_output(SimpleNamespace(id=2))
        return StreamingResponse(iter([b"data: x\n\n"]), media_type="text/event-stream")

    async def answered(request):
        return JSONResponse({})

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", streamed, methods=["POST"]),
            Route("/v1/completions", answered, methods=["POST"]),
        ]
    )

    async def post(path):
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]
        scope = {"type": "http", "method": "POST", "path": path, "headers": []}
        scope |= {"query_string": b"", "root_path": ""}

        async def receive():
            if messages:
                return messages.pop(0)
            await asyncio.Event().wait()  # the client stays connected

        async def send(message):
            await asyncio.sleep(send_s)

        await app(scope, receive, send)

    async def both():
        await post("/v1/chat/completions")
        await post("/v1/completions")

    asyncio.run(both())
    compass_run._move_line_files()  # as the server exits
    rows = [
        line.split() for line in (record / "intake.stamps").read_text().splitlines()
    ]
    # Two rows taken in, and one head: the stream's, not the JSON answer's.
    assert [r[2] for r in rows if r[2] in ("out", "head")] == ["out", "out", "head"]
    assert all(int(r[3]) <= int(r[0]) for r in rows)
    # The head is stamped from its send call's start, not its return.
    head = next(r for r in rows if r[2] == "head")
    assert int(head[0]) - int(head[3]) >= send_s * 1e9


def test_a_recorded_server_s_engine_send_keeps_its_caller_s_response_id(record, tok):
    parity.carry_response_ids()  # as the engine's construction does, before the wrap
    compass_run.tokenizer(tok, None)
    manager = CoreManager.__new__(CoreManager)
    manager.label, manager.pp_size, manager.local_engine_count = "t", 1, 1
    manager._seq_id_to_callback = {}
    manager._send_request = lambda _rank, payload: None
    processor = InputOutputProcessor(
        SimpleNamespace(hf_config=SimpleNamespace(model_type="llama")), tok, 16
    )
    seq = processor.preprocess("hi there", SamplingParams())
    request_id = "chatcmpl-1"  # noqa: F841 -- the API server's handler holds it
    manager.add_request([seq])
    assert seq.external_request_id == "chatcmpl-1"


def test_with_compass_off_nothing_is_wrapped_or_written(patched, tok, tmp_path):
    body = Request.body
    compass_run.tokenizer(tok, None)
    assert "encode" not in vars(tok) and "decode" not in vars(tok)
    assert Request.body is body
    assert not list(tmp_path.rglob(compass_run.TOKENIZER_RECORD))


def test_a_simulated_run_recording_its_steps_records_no_host_calls(
    record, tok, monkeypatch, tmp_path
):
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"bound_s": 600.0, "host": {}}))
    monkeypatch.setenv(compass_run.ENV, str(run))
    body = Request.body
    compass_run.tokenizer(tok, None)
    assert "encode" not in vars(tok)
    assert Request.body is body
    assert not record.exists()


def test_a_recorded_server_writes_no_line_under_its_record_until_it_exits(
    record, tok, monkeypatch, tmp_path
):
    """On NFS a line written right after another waits for the first's round
    trip; served requests write theirs to the host's own file system."""
    import builtins
    import io
    import os
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "local"))
    (tmp_path / "local").mkdir()
    app = _server(tok)
    compass_run.tokenizer(tok, None)
    opened = []

    def spied(real):
        def spy(file, *a, **k):
            opened.append(os.fspath(file) if not isinstance(file, int) else "")
            return real(file, *a, **k)

        return spy

    for owner in (builtins, io, os):
        monkeypatch.setattr(owner, "open", spied(owner.open))
    body = json.dumps({"messages": [{"role": "user", "content": "hi there"}]})

    async def lone():
        return [await _post(app, body.encode()) for _ in range(2)]

    asyncio.run(lone())
    monkeypatch.undo()
    # No file under the record was opened, and none of its lines is there.
    assert not [p for p in opened if pathlib.Path(p).is_relative_to(record)]
    names = (compass_run.INTAKE_RECORD, compass_run.TOKENIZER_RECORD)
    assert not [n for n in names if (record / n).exists()]
    local = {
        n: pathlib.Path((record / (n + compass_run.LOCAL)).read_text().strip())
        for n in names
    }
    assert {p.parent for p in local.values()} == {tmp_path / "local"}
    lines = {n: p.read_text() for n, p in local.items()}
    assert len(lines[compass_run.INTAKE_RECORD].splitlines()) == 2 * 7
    # The server's stop, once uvicorn has returned, moves them into the record.
    monkeypatch.setattr(compass_run.clock, "installed", lambda: None)  # a real run
    assert not compass_run.frontend_done(None)
    assert {n: (record / n).read_text() for n in names} == lines
    assert not [p for n, p in local.items() if p.exists()]
    assert not list(record.glob("*" + compass_run.LOCAL))


#: A recording server with no engine: its tokenizer and one request stamped,
#: then served by uvicorn until a stop, as the API server's ``main`` does.
SIGTERM_SERVER = """
import asyncio, contextlib, pathlib, sys
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import AutoTokenizer
from atom.compass import run as compass_run

model, ready = sys.argv[1], pathlib.Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model)
compass_run.tokenizer(tok, None)
tok.encode("hi there")


async def read(request):
    return JSONResponse({"bytes": len(await request.body())})


@contextlib.asynccontextmanager
async def lifespan(app):
    ready.touch()
    yield


app = Starlette(routes=[Route("/", read, methods=["POST"])], lifespan=lifespan)


async def post():
    messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/", "headers": []}
    await app(scope | {"query_string": b"", "root_path": ""}, receive, send)


asyncio.run(post())
uvicorn.run(app, host="127.0.0.1", port=0, log_level="warning")
compass_run.frontend_done(None)
"""


def test_a_recording_server_stopped_by_sigterm_moves_its_lines_into_its_record(
    tok, tmp_path
):
    """A stop SIGTERMs the server's process group. uvicorn takes the signal while
    it serves, then raises it again, which by default ends the process before
    any exit handler or `frontend_done` runs."""
    import os
    import signal
    import subprocess
    import time

    record, local, ready = tmp_path / "steps", tmp_path / "local", tmp_path / "ready"
    local.mkdir()
    env = dict(os.environ, TMPDIR=str(local)) | {parity.ENV: str(record)}
    env.pop(compass_run.ENV, None)
    server = subprocess.Popen(
        [sys.executable, "-c", SIGTERM_SERVER, tok.name_or_path, str(ready)],
        env=env,
        start_new_session=True,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        end = time.monotonic() + 300
        while not ready.exists():
            assert server.poll() is None, server.stderr.read()
            assert time.monotonic() < end, "the server never served"
            time.sleep(0.1)
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=120)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)
    # It ends of the signal, as it did, and its lines are in the record.
    assert server.returncode == -signal.SIGTERM
    stamps = (record / compass_run.INTAKE_RECORD).read_text().splitlines()
    assert [r.split()[2] for r in stamps] == ["app", "recv", "body"]
    _, calls = _calls(record)
    assert [(c["call"], c["tokens"]) for c in calls] == [("encode", 2)]
    assert not list(record.glob("*" + compass_run.LOCAL))
    assert not list(local.iterdir())


@pytest.mark.parametrize("held", ["nfs", "overlay"])
def test_a_temp_dir_on_a_network_file_system_refuses_the_record(
    record, tok, monkeypatch, tmp_path, held
):
    """A line written there right after another would wait for the first's
    round trip, the wait the local file keeps out of the request's stages."""
    import re
    import tempfile

    shared, temp = tmp_path / "shared", tmp_path / "tmp"
    if held == "nfs":
        temp = shared / "tmp"
    temp.mkdir(parents=True)
    table = tmp_path / "mountinfo"
    root = f"upperdir={shared / 'upper'},workdir={shared / 'work'}"
    table.write_text(
        f"1 0 0:40 / / rw - {'overlay overlay' if held == 'overlay' else 'ext4 /dev/sdb2'}"
        f" rw,{root if held == 'overlay' else 'relatime'}\n"
        f"2 1 0:48 / {shared} rw,relatime - nfs 10.0.0.1:/home rw,vers=3\n"
    )
    monkeypatch.setattr(compass_run, "MOUNTINFO", str(table), raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    refused = f"the stamp lines' temp directory {temp} is on nfs (10.0.0.1:/home)"
    with pytest.raises(ValueError, match=re.escape(refused) + ".*set TMPDIR"):
        compass_run.tokenizer(tok, None)
    assert not list(record.iterdir())
