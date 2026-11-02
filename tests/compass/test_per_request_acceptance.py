# SPDX-License-Identifier: MIT
"""Each request's own accepted drafts, from a real record to the simulated runner.

`compass run --acceptance-record` reads them off the record's rows
(`drive.run.per_request`) into the run file; the harness sends each request
its digits in ``tracestate``; `HttpChannel` serves the request with them set;
the frontend's `add_request` puts them on the request's sequence, and the
engine's first batch of it hands them to the runner, whose stream replays
them in place of its draws.

The round trip is served by ATOM's scheduler and the simulated runner's
`forward` with the step record on, as `test_parity_record` drives them, and
read back by `per_request` from that record.
"""

import gzip
import http.client
import json
import queue
import threading
from collections import Counter
from types import SimpleNamespace

import pytest
import uvicorn
from test_compass_event_loop import HTTP, _traffic, _until, run  # noqa: F401
from test_parity_record import BLOCK, _config, _prompt, _Runner
from test_parity_response_ids import _complete, engine  # noqa: F401
from test_run_measured_acceptance import mtp, recorded, started
from test_run_refusal_summary import STAGE1
from test_run_speculative_summary import _run, _stated, prefill, runner
from test_runner_control_refusals import Worker

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends import StepStubBackend
from atom.compass.carriers import (
    tracestate_acceptance,
    tracestate_stamp,
    tracestate_with,
)
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve
from atom.compass.runner.step_output import DeferredTokenStream
from atom.entrypoints.openai import api_server
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.compass_loop import ACCEPTANCE, HttpChannel

pytestmark = pytest.mark.usefixtures("unpriced")

K = 3
VENDORS = "rojo=00f067aa0ba902b7"


class TestTheEntry:
    def test_it_rides_beside_the_stamp_and_reads_back(self):
        header = tracestate_with(VENDORS, 1.5, 3, "05z")
        assert header == f"compass=a:1.5;s:3,compass-acceptance=05z,{VENDORS}"
        assert tracestate_stamp(header) == (1.5, 3)
        assert tracestate_acceptance(header) == "05z"
        assert tracestate_acceptance(tracestate_with(VENDORS, 1.5, 3)) is None

    @pytest.mark.parametrize("digits", ["", "5A", "5,1", "5 1"])
    def test_a_writer_refuses_what_is_not_base_36_digits(self, digits):
        with pytest.raises(ValueError, match="is not base-36 digits"):
            tracestate_with(None, 1.5, 3, digits)

    @pytest.mark.parametrize("digits", ["", "5A", "5 1", "5.1"])
    def test_a_reader_refuses_a_malformed_entry_by_name(self, digits):
        with pytest.raises(ValueError, match="malformed compass-acceptance entry"):
            tracestate_acceptance(f"compass=a:1.5;s:3,compass-acceptance={digits}")

    def test_two_entries_are_refused(self):
        with pytest.raises(ValueError, match="more than one compass-acceptance"):
            tracestate_acceptance("compass-acceptance=1,compass-acceptance=2")


def test_a_stamped_request_is_served_with_its_digits_through_uvicorn(run):  # noqa: F811
    """Two stamped requests reach the app over a real socket: the first with a
    9000-row entry, past W3C's 256 characters, the second with none."""
    stamps, seen = queue.Queue(), []
    digits = "0123" * 2250

    async def app(scope, receive, send):
        seen.append(ACCEPTANCE.get())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    server = uvicorn.Server(
        uvicorn.Config(
            HttpChannel(app, api_server._tracestate_stamp),
            host="127.0.0.1",
            port=0,
            loop="atom.utils.compass_loop:CompassEventLoop",
            lifespan="off",
            log_level="warning",
        )
    )
    traffic = _traffic(run, stamps, before_ner=lambda: None, n=2)
    sent = [stamps.get(timeout=10) for _ in range(2)]
    got = []

    def client():
        # Both are sent before either is read: the channel hands a released
        # request on only once every released request has been read.
        _until(lambda: server.started and all(_released(run, s) for _, s in sent))
        port = server.servers[0].sockets[0].getsockname()[1]
        conns = []
        for (arrival, seq), entry in zip(sent, (digits, None)):
            conns.append(http.client.HTTPConnection("127.0.0.1", port, timeout=10))
            header = tracestate_with(None, arrival, seq, entry)
            conns[-1].request("POST", "/v1/completions", b"{}", {"tracestate": header})
        for conn in conns:
            got.append(conn.getresponse().status)
            conn.close()

    thread = threading.Thread(target=client, name="client", daemon=True)
    thread.start()
    try:
        server.run()  # the finish ends it once both are served
    finally:
        thread.join(10)
        traffic.join(10)
    assert got == [200, 200] and seen == [digits, None]
    assert ACCEPTANCE.get() is None


def _released(run, seq):  # noqa: F811
    with run.rt.lock:
        return run.rt.is_released(HTTP, seq)


def test_a_request_s_digits_reach_the_runner_once_in_its_first_batch(
    engine, monkeypatch  # noqa: F811
):
    """Through ATOM's API server handlers: the request served with digits set
    sends its sequence with them, pickled as the engine core gets it; ATOM's
    scheduler then builds a batch carrying them, taken off the sequence, and
    the batches after it carry none."""
    monkeypatch.setattr(ScheduledBatch, "__init__", ScheduledBatch.__init__)
    monkeypatch.setattr(compass_run, "_replayed_requests", 0)
    compass_run._attach_acceptance()
    compass_run._carry_acceptance()
    token = ACCEPTANCE.set("2033")
    try:
        _complete(_prompt(0))
    finally:
        ACCEPTANCE.reset(token)
    _complete(_prompt(1))
    sent = [s for seqs in engine.sent for s in seqs]
    assert [vars(s).get(compass_run.ACCEPTANCE_ATTR) for s in sent] == ["2033", None]

    scheduler = Scheduler(_config(budget=1024))
    for s in sent:
        scheduler.add(s)
    batch, seqs = scheduler.schedule()
    assert sorted(seqs) == [s.id for s in sent]
    assert batch.compass_acceptance == {sent[0].id: "2033"}
    assert compass_run.ACCEPTANCE_ATTR not in vars(sent[0])
    again = ScheduledBatch(seqs, [1] * len(seqs), len(seqs))
    assert not hasattr(again, "compass_acceptance")
    assert compass_run._replayed_requests == 1


def test_a_fan_out_s_sequences_carry_none(engine, monkeypatch):  # noqa: F811
    compass_run._attach_acceptance()
    seqs = [Sequence([5, 6], BLOCK), Sequence([5, 6], BLOCK)]
    token = ACCEPTANCE.set("1")
    try:
        engine.core_mgr.add_request(seqs)
    finally:
        ACCEPTANCE.reset(token)
    assert [vars(s).get(compass_run.ACCEPTANCE_ATTR) for s in engine.sent[0]] == [
        None,
        None,
    ]


def _chunk(doc) -> dict:
    return {"packets": [{"name": "data", "value": json.dumps(doc)}]}


def _raw(record, requests) -> None:
    """The client run's raw records: per request its conversation, turn,
    response id and completion tokens, as aiperf writes them."""
    lines = []
    for conversation, turn, response, length, meta in requests:
        responses = [_chunk({"id": response, "choices": []})]
        if length is not None:
            responses.append(
                _chunk({"id": response, "usage": {"completion_tokens": length}})
            )
        responses.append({"packets": [{"name": "data", "value": "[DONE]"}]})
        metadata = {"conversation_id": conversation, "turn_index": turn, **meta}
        lines.append(json.dumps({"metadata": metadata, "responses": responses}))
    (record / "artifacts").mkdir(parents=True, exist_ok=True)
    with gzip.open(record / "artifacts" / drive_run.RAW_RECORDS[0], "wt") as f:
        f.write("\n".join(lines) + "\n")


def _length(digits: str, short: int) -> int:
    """The output length a request whose rows accept `digits` ends at, `short`
    tokens before its last row's last."""
    return 1 + sum(int(d, 36) + 1 for d in digits) - short


def _serve(tmp_path, monkeypatch, lists, lengths):
    """One request per entry of `lists`, its digits on its sequence (None for
    none), served by ATOM's scheduler and the simulated runner to `lengths`
    with the step record on; the record directory, with its client run's raw
    records, and the scheduler."""
    record = tmp_path / "record"
    monkeypatch.setenv(parity.ENV, str(record / serve.STEPS))
    monkeypatch.setattr(ScheduledBatch, "__init__", ScheduledBatch.__init__)
    monkeypatch.setattr(compass_run, "_replayed_requests", 0)
    compass_run._carry_acceptance()
    spec = SimpleNamespace(
        method="mtp", num_speculative_tokens=K, use_dspark=lambda: False
    )
    config = _config(speculative_config=spec)
    scheduler, runner = Scheduler(config), _Runner(config)
    runner._token_stream = DeferredTokenStream([7], rates=[0.5] * K, seed=0)
    for i, (digits, length) in enumerate(zip(lists, lengths)):
        params = SamplingParams(max_tokens=length, ignore_eos=True)
        seq = Sequence(_prompt(i), BLOCK, sampling_params=params)
        seq.external_request_id = f"r{i}"
        if digits is not None:
            setattr(seq, compass_run.ACCEPTANCE_ATTR, digits)
        scheduler.add(seq)
    steps = 0
    while scheduler.has_unfinished_requests():
        steps += 1
        assert steps < 1000, "the served requests do not finish"
        batch, seqs = scheduler.schedule()
        if batch is None or not batch.req_ids:
            continue
        reply = runner.forward(batch)
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=queue.Queue(), batch=batch
        )
    _raw(
        record,
        [(f"c{i}", 0, f"r{i}", n, {}) for i, n in enumerate(lengths)],
    )
    return record, scheduler


#: Each ends on two K, as `per_request` states the last one or two rows read.
LISTS = ["2033", "0133", "3", "10233"]


def test_a_record_of_replayed_requests_reads_back_their_digits(tmp_path, monkeypatch):
    """Four requests replay their digits and a fifth draws; the record's rows
    give each listed one's digits back, and the drawn one's are its draws."""
    lengths = [_length(d, s) for d, s in zip(LISTS, (0, 2, 3, 1))] + [40]
    record, scheduler = _serve(tmp_path, monkeypatch, [*LISTS, None], lengths)

    lists, left = drive_run.per_request(record, K)

    print("read back:", lists)
    assert {c: t["0"] for c, t in lists.items() if c != "c4"} == {
        f"c{i}": d for i, d in enumerate(LISTS)
    }
    drawn = lists["c4"]["0"]
    assert _length(drawn, 0) >= 40 > _length(drawn[:-1], 0) and left == Counter()
    assert compass_run._replayed_requests == len(LISTS)
    # The scheduler counted every row read, the replayed ones and the drawn.
    counted = Counter(int(d) for t in lists.values() for d in t["0"])
    assert sum(scheduler.engine_stats.distribution.values()) == counted.total()


def test_digits_past_the_draft_width_are_refused_by_name(tmp_path, monkeypatch):
    with pytest.raises(
        ValueError, match="names 4 accepted drafts at a row that verified 3"
    ):
        _serve(tmp_path, monkeypatch, ["14"], [20])


@pytest.mark.parametrize(
    "contexts, length, digits",
    [
        # The last verify row was scheduled after the end was read: one row
        # read past the growths, whatever its tokens.
        ([100, 104, 107, 109, 113], 1 + 3 + 2 + 4 + 1, "2133"),
        ([100, 104, 107, 109, 113], 1 + 3 + 2 + 4 + 4, "2133"),
        # No row after the end: a prefill step read it, and two rows were read.
        ([100, 104, 107, 109, 113], 1 + 3 + 2 + 4 + 5, "21333"),
        ([100, 104, 107, 109, 113], 1 + 3 + 2 + 4 + 8, "21333"),
        # One verify row, read or not.
        ([100], 1, ""),
        ([100], 3, "3"),
        ([], 1, ""),
        # Growths and length disagree.
        ([100, 104, 107, 109, 113], 1 + 3 + 2 + 4 + 9, None),
        ([100, 104, 107, 109, 113], 1 + 3 + 2 + 4, None),
        ([100, 104, 107, 109, 113], 1 + 3 + 2, None),
        ([100], 6, None),
        ([], 2, None),
        ([100, 104, 109, 110], 12, None),
        # A first growth other than the prefill's token's K + 1.
        ([100, 103, 106], 1 + 4 + 3, None),
    ],
)
def test_a_request_s_digits_are_its_growths_and_its_length_s_rows(
    contexts, length, digits
):
    assert drive_run._digits(contexts, length, K) == digits


def _line(rows, named=()):
    """A step record line holding `rows`, (request id, tokens, context, decode)."""
    return {
        "is_dummy_run": False,
        "named": [list(n) for n in named],
        "batch": [[r, n, c] for r, n, c, _ in rows],
        "rows": [[n, c, d] for _, n, c, d in rows],
    }


def test_requests_the_record_cannot_give_are_left_out_by_why(tmp_path):
    """Request 0 reads back; 1 was preempted (a prefill row after a verify
    row); 2 has no usage, 3 was cancelled, 4 is not in the steps; 5's growths
    and length disagree; 6's prefill row is its last."""
    record = tmp_path / "record"
    lines = [
        _line(
            [(r, 64, 64, False) for r in range(7)],
            [(r, "k", f"r{r}") for r in range(7)],
        ),
        _line([(r, K + 1, 68 + 0, True) for r in (0, 1, 5)]),
        _line([(0, K + 1, 72, True), (1, 16, 80, False), (5, K + 1, 72, True)]),
        _line([(0, K + 1, 75, True), (5, K + 1, 74, True)]),
    ]
    (record / serve.STEPS).mkdir(parents=True)
    (record / serve.STEPS / "dp0.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    cancelled = {"was_cancelled": True}
    _raw(
        record,
        [
            ("c", 0, "r0", 1 + 3 + 2, {}),
            ("c", 1, "r1", 9, {}),
            ("c", 2, "r2", None, {}),
            ("c", 3, "r3", 1, cancelled),
            ("c", 4, "absent", 1, {}),
            ("c", 5, "r5", 30, {}),
            ("c", 6, "r6", 1, {}),
        ],
    )

    lists, left = drive_run.per_request(record, K)

    assert lists == {"c": {"0": "23"}}
    assert left == {
        "preempted": 1,
        "unfinished": 2,
        "not in the steps": 1,
        "growths disagree with the length": 1,
    }


def test_two_requests_of_one_turn_are_refused(tmp_path):
    record = tmp_path / "record"
    (record / serve.STEPS).mkdir(parents=True)
    line = _line(
        [(0, 64, 64, False), (1, 64, 64, False)], [(0, "a", "r0"), (1, "b", "r1")]
    )
    (record / serve.STEPS / "dp0.jsonl").write_text(json.dumps(line) + "\n")
    _raw(record, [("c", 0, "r0", 1, {}), ("c", 0, "r1", 1, {})])
    # Neither decoded, so neither has digits to replay and neither is refused.
    assert drive_run.per_request(record, K) == ({}, Counter())
    line["rows"] = [[K + 1, 64, True], [K + 1, 64, True]]
    (record / serve.STEPS / "dp0.jsonl").write_text(json.dumps(line) + "\n")
    _raw(record, [("c", 0, "r0", 2, {}), ("c", 0, "r1", 2, {})])
    with pytest.raises(ValueError, match="two requests of conversation c turn 0"):
        drive_run.per_request(record, K)


def test_a_record_with_no_client_run_gives_no_lists(tmp_path):
    assert drive_run.per_request(tmp_path, K) == ({}, Counter())


def test_compass_run_hands_the_run_file_its_record_s_lists(monkeypatch, tmp_path):
    """A record with its client run's raw records: one request, two verify
    rows read and one after its end."""
    record, *_ = recorded(monkeypatch, tmp_path, [0.7])
    lines = [_line([(0, 64, 64, False)], [(0, "k", "r0")])]
    lines += [_line([(0, 2, c, True)]) for c in (66, 68, 69)]
    (record / serve.STEPS / "dp0.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    _raw(record, [("c", 0, "r0", 1 + 1 + 1, {})])

    _, run_file = started(
        monkeypatch, tmp_path, "--acceptance-record", str(record), atom=mtp(1)
    )

    assert run_file[compass_run.ACCEPTANCE_LISTS] == {"c": {"0": "01"}}


def test_the_summary_names_the_requests_listed_and_each_client_run_s_replayed(
    monkeypatch, tmp_path
):
    """Three requests listed; the engine replayed 2 by client run 0's end and 5
    by client run 1's, so client run 1 replayed 3."""
    file = _run(monkeypatch, tmp_path)
    file[compass_run.ACCEPTANCE_LISTS] = {"a": {"0": "1", "1": "0"}, "b": {"0": "1"}}
    a = compass_run._RecordingAuthority(file)
    spec = _stated(monkeypatch, 1, synthetic_acceptance_rates=[0.5])
    scheduler = SimpleNamespace(
        config=SimpleNamespace(speculative_config=spec),
        engine_stats=EngineStats(use_spec=True, mtp_k=1),
    )
    r = runner(StepStubBackend(STAGE1))
    r.forward(prefill(8))
    rt = LPRuntime(compass_run.ENGINE, compass_run.channel_table(file), None)
    monkeypatch.setattr(clock, "_installed", rt)
    core = SimpleNamespace(runner_mgr=Worker(r), scheduler=scheduler)
    for k, replayed in enumerate((2, 5)):
        monkeypatch.setattr(compass_run, "_replayed_requests", replayed)
        compass_run._engine_hand_in(file, core, {}, client_run=k)

    def counted(k):
        return compass_run._read_hand_ins(
            file, tmp_path / compass_run.CLIENT_DIR.format(k), False
        )

    said = compass_run._summary(file, a, a, 0.0, counted(1), counted(0))
    said = said["speculative"]["engine"]
    assert said["input"]["requests"] == 3 and said["replayed_requests"] == 3
