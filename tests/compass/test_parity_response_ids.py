# SPDX-License-Identifier: MIT
"""Each request's response id, from the API server to the step record and its split.

Requests go through ATOM's API server handlers over a planted engine: its
`preprocess` builds the sequence and answers it at once, and its core manager
is ATOM's own, whose `add_request` sends each sequence the way it reaches the
engine core, pickled. The record is then written by ATOM's scheduler and the
simulated runner from the sequences the engine core received.
"""

import asyncio
import json
import pickle
import queue
from types import SimpleNamespace

import numpy as np
import pytest
from test_parity_record import BLOCK, _config, _key, _prompt, _Runner

from atom.compass import run
from atom.compass.parity import ENV, TRANSFERS, compare, read, transfers
from atom.compass.parity.served import served
from atom.compass.validate.runs import _requests
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import CompletionRequest
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType

SAMPLING = {"temperature": 0.0, "top_p": 1.0}


@pytest.fixture
def engine(monkeypatch):
    """The planted engine; `sent` holds what each `add_request` sent, unpickled."""
    for cls in (CoreManager, DisaggCoreManager):  # ATOM's back after the test
        monkeypatch.setattr(cls, "add_request", cls.add_request)

    def preprocess(tokens, sampling_params, stream_callback=None, **kwargs):
        params = kwargs.get("kv_transfer_params")
        seq = Sequence(list(tokens), BLOCK, sampling_params, kv_transfer_params=params)
        stream_callback(
            SimpleNamespace(output_tokens=[5], finished=True, finish_reason="length")
        )
        return seq

    sent = []
    planted = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
        core_mgr=None,
        sent=sent,
    )

    def core(cls=CoreManager):
        manager = cls.__new__(cls)
        manager.label, manager.pp_size, manager.local_engine_count = "t", 1, 1
        manager._seq_id_to_callback = {}
        manager._send_request = lambda _rank, payload: sent.append(
            pickle.loads(payload)[1]
        )
        planted.core_mgr = manager

    planted.core = core
    core()
    monkeypatch.setattr(api_server, "engine", planted)
    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(
        api_server, "tokenizer", SimpleNamespace(decode=lambda *_a, **_k: "x")
    )
    return planted


def _frontend(monkeypatch, record=None):
    """`run.frontend` as `LLMEngine` enters it on a real run."""
    if record is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, str(record))
    monkeypatch.delenv(run.ENV, raising=False)
    config = SimpleNamespace(runner_qualname=run.ATOM_RUNNER)
    with run.frontend(config):
        pass


def _complete(tokens) -> str:
    """Send one completion request; its response `id`."""
    request = CompletionRequest(
        model="m", prompt_token_ids=tokens, max_tokens=3, ignore_eos=True, **SAMPLING
    )
    response = asyncio.run(api_server.completions(request, None))
    return json.loads(response.body)["id"]


def test_a_server_not_recording_sends_its_sequences_unchanged(engine, monkeypatch):
    add = CoreManager.add_request
    _frontend(monkeypatch)
    _complete(_prompt(0))

    assert CoreManager.add_request is add
    assert [s.external_request_id for s in engine.sent[0]] == [None]


@pytest.mark.parametrize("cls", [CoreManager, DisaggCoreManager])
def test_a_recording_server_s_sequences_carry_their_response_ids(
    engine, monkeypatch, tmp_path, cls
):
    _frontend(monkeypatch, tmp_path)
    engine.core(cls)
    ids = [_complete(_prompt(0))]
    # The streaming and multimodal paths hand their sequences over in the same way.
    dispatcher = SimpleNamespace(new_state=lambda: None, enqueue=lambda **_: None)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", dispatcher)
    for name in ("_stream_loops", "_request_start_times", "_seq_id_to_request_id"):
        monkeypatch.setattr(api_server, name, {})
    ids.append("cmpl-stream")
    asyncio.run(api_server.setup_streaming_request(_prompt(1), None, ids[-1]))

    async def multimodal():
        async for _ in api_server.generate_async_multimodal(
            _prompt(2), {}, None, ids[-1]
        ):
            pass

    ids.append("chatcmpl-multimodal")
    asyncio.run(multimodal())

    assert [s.external_request_id for seqs in engine.sent for s in seqs] == [
        i for i in ids for _ in range(1 if cls is CoreManager else 2)
    ]


def test_a_record_splits_per_client_run_by_the_response_ids_its_clients_saw(
    engine, monkeypatch, tmp_path
):
    _frontend(monkeypatch, tmp_path)
    groups = [[0, 1], [2, 3]]
    clients = [{_complete(_prompt(i)): i for i in group} for group in groups]
    received = iter(seq for seqs in engine.sent for seq in seqs)
    config = _config()
    scheduler, runner = Scheduler(config), _Runner(config)
    for group in groups:  # each client run served to its end before the next
        for _ in group:
            scheduler.add(next(received))
        while scheduler.has_unfinished_requests():
            scheduled = scheduler.schedule()
            if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
                continue
            batch, seqs = scheduled
            scheduler.postprocess(
                list(seqs.values()),
                runner.forward(batch),
                stream_output_queue=queue.Queue(),
                batch=batch,
            )
    exports = [
        {rid: {"trace": f"t{i}", "sampling": SAMPLING} for rid, i in c.items()}
        for c in clients
    ]
    split = served(tmp_path, exports)[0]

    keys = {_key(_prompt(i)): k for k, group in enumerate(groups) for i in group}
    assert {s["client_run"] for s in split} == {0, 1}
    for step in split:
        assert {keys[row[0]] for row in step["batch"]} == {step["client_run"]}
    names = {k: v for s in split for k, v in s.get("response_ids", {}).items()}
    assert names == {_key(_prompt(i)): rid for c in clients for rid, i in c.items()}


def test_a_record_written_before_response_ids_reads_as_it_did(tmp_path):
    for side, named in (("old", [[1, "k1"]]), ("new", [[1, "k1", "cmpl-a"]])):
        (tmp_path / side).mkdir()
        lines = [
            {"step": 0, "batch": [[1, 8, 8]], "rows": [[8, 0, False]], "named": named},
            {"step": 1, "batch": [[1, 1, 9]], "rows": [[1, 9, True]], "named": []},
        ]
        text = "".join(json.dumps(x) + "\n" for x in lines)
        (tmp_path / side / "dp0.jsonl").write_text(text)
    old, new = read(tmp_path / "old")[0], read(tmp_path / "new")[0]

    assert [s["batch"] for s in old] == [[["k1", 8, 8]], [["k1", 1, 9]]]
    assert "response_ids" not in old[0] and "response_ids" not in old[1]
    assert new[0].pop("response_ids") == {"k1": "cmpl-a"}
    assert new[0].pop("request_ids") == {"k1": 1}
    assert new == old
    assert compare(tmp_path / "old", tmp_path / "new")["first_divergence"] == {0: None}
    steps = [json.loads(x) for x in (tmp_path / "new" / "dp0.jsonl").open()]
    assert _requests(steps) == {(8, 0): 8}


def _complete_decode(tokens, transfer_id, remote_dp_rank) -> str:
    """Send one completion request as the router sends decode its half; its response `id`."""
    params = {"transfer_id": transfer_id, "remote_dp_rank": remote_dp_rank}
    request = CompletionRequest(
        model="m",
        prompt_token_ids=tokens,
        max_tokens=3,
        ignore_eos=True,
        kv_transfer_params=params,
        **SAMPLING,
    )
    response = asyncio.run(api_server.completions(request, None))
    return json.loads(response.body)["id"]


def _two_sides(engine, monkeypatch, tmp_path, n=4):
    """A planted 1P1D deployment's records: prefill's driven by ATOM's scheduler,
    decode's by decode batches of the sequences its API server received, each
    request handed `transfer_id`, its prefill sequence's id, as ATOM's connector
    hands it. Returns prefill's and decode's response ids, by prompt."""
    added = {cls: cls.add_request for cls in (CoreManager, DisaggCoreManager)}
    _frontend(monkeypatch, tmp_path / "prefill")
    prefill_ids = [_complete(_prompt(i)) for i in range(n)]
    received = [seq for seqs in engine.sent for seq in seqs]
    config = _config()
    scheduler, runner = Scheduler(config), _Runner(config)
    for seq in received:
        scheduler.add(seq)
    while scheduler.has_unfinished_requests():
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None or not scheduled[0].req_ids:
            continue
        batch, seqs = scheduled
        scheduler.postprocess(
            list(seqs.values()),
            runner.forward(batch),
            stream_output_queue=queue.Queue(),
            batch=batch,
        )

    for cls, add in added.items():  # each side is its own server process
        monkeypatch.setattr(cls, "add_request", add)
    (tmp_path / "decode").mkdir()
    (tmp_path / "decode" / TRANSFERS).write_text('{"stale": 1}\n')
    engine.sent.clear()
    _frontend(monkeypatch, tmp_path / "decode")
    # The prefill rank is 1 here, so the join is keyed by the rank as well.
    decode_ids = [_complete_decode(_prompt(i), s.id, 1) for i, s in enumerate(received)]
    seqs = {s.id: s for sent in engine.sent for s in sent}
    for seq in seqs.values():
        seq.type = SequenceType.DECODE
    runner = _Runner(_config())
    for _ in range(2):
        runner.forward(
            ScheduledBatch(
                seqs=seqs,
                num_scheduled_tokens=np.ones(n, dtype=np.int32),
                total_tokens_num=n,
                total_tokens_num_decode=n,
                total_seqs_num=n,
                total_seqs_num_decode=n,
            )
        )
    return received, prefill_ids, list(seqs), decode_ids


def test_a_decode_instance_names_each_request_at_its_first_step_by_its_response_id(
    engine, monkeypatch, tmp_path
):
    _, _, decode_seqs, decode_ids = _two_sides(engine, monkeypatch, tmp_path)
    first, second = read(tmp_path / "decode")[0]

    assert first["response_ids"] == {rid: rid for rid in decode_ids}
    assert first["request_ids"] == dict(zip(decode_ids, decode_seqs))
    assert [row[0] for row in second["batch"]] == decode_ids
    assert "response_ids" not in second and "request_ids" not in second
    export = {rid: {"trace": "t", "sampling": SAMPLING} for rid in decode_ids}
    assert [s["client_run"] for s in served(tmp_path / "decode", [export])[0]] == [0, 0]


def test_a_decode_instance_writes_each_request_s_transfer_and_prefill_none(
    engine, monkeypatch, tmp_path
):
    received, _, _, decode_ids = _two_sides(engine, monkeypatch, tmp_path)

    joined = {(1, s.id): rid for s, rid in zip(received, decode_ids)}
    assert transfers(tmp_path / "decode") == joined
    assert transfers(tmp_path / "prefill") == {}
    assert not (tmp_path / "prefill" / TRANSFERS).exists()


def test_every_prefill_request_joins_the_decode_response_id_that_names_it(
    engine, monkeypatch, tmp_path
):
    received, prefill_ids, _, decode_ids = _two_sides(engine, monkeypatch, tmp_path)
    prefill = read(tmp_path / "prefill")[0]
    joined = transfers(tmp_path / "decode")

    by_key = {k: v for s in prefill for k, v in s.get("request_ids", {}).items()}
    assert by_key == {_key(_prompt(i)): s.id for i, s in enumerate(received)}
    named = {k: v for s in prefill for k, v in s.get("response_ids", {}).items()}
    assert named == {_key(_prompt(i)): rid for i, rid in enumerate(prefill_ids)}
    decode_names = {r[0] for s in read(tmp_path / "decode")[0] for r in s["batch"]}
    assert {k: joined[(1, i)] for k, i in by_key.items()} == {
        _key(_prompt(i)): rid for i, rid in enumerate(decode_ids)
    }
    assert decode_names == set(decode_ids)
