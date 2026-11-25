# SPDX-License-Identifier: MIT
"""Mooncake's chunked P->D transfer on the simulated connector.

Under ``enable_chunked_transfer`` the router sends a request to prefill and
decode at once. Decode admits it and sends its write request; prefill holds
the request until that write request is in; each prefill forward publishes the
prompt's blocks computed by its device end, and the send writes them as they
come, coalescing what one wait finds; after the handoff, prefill tells decode
the write is done. `ChunkedSend` and `chunk_writes` are the pricing, which a
record's replay calls too; the engines are real `Scheduler`s on scripted
clocks, as in `test_kv_remote_prefill`.
"""

import itertools
import math
from types import SimpleNamespace

import pytest
from conftest import atom_config_double
from test_kv_remote_prefill import (
    FIRST_TOKEN_ID,
    L,
    PROMPT,
    Script,
    geometry,  # noqa: F401  a fixture
    scheduler_with,
)
from test_kv_simulated_connector import PEAKS, model_for

from atom.compass.clock import prefill_decode_table
from atom.compass.kv import TRANSFER_KEY, WRITE_REQ_KEY
from atom.compass.kv import connector as connector_module
from atom.compass.kv.connector import WRITE_DONE_KEY
from atom.compass.kv.transfer import ChunkedSend, Link, TransferModel, chunk_writes
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.scheduler import ScheduledBatchOutput
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams
from atom.utils import clock
from atom.utils.clock import LPRuntime

#: One block a second, half a second of latency a write.
UNIT = TransferModel(latency_s=0.5, bandwidth_bytes_per_s=1.0, bytes_per_block=1)


def _send(start=0.0, cursor=0, total=4, model=UNIT):
    return ChunkedSend(Link(model), start, cursor, total)


# -- the pricing -----------------------------------------------------------


def test_a_send_writes_each_forward_s_blocks_as_they_come():
    """Blocks 0-1 at 2 s, 2-3 at the last forward, 5 s: a write of two from
    2 s, done at 4.5 s; the reader finds nothing new then and waits for 5 s."""
    s = _send()
    s.publish(2.0, 8, 16, 4)
    s.publish(5.0, 16, 16, 4)
    s.finish(5.0)
    assert chunk_writes([s], math.inf) == [s]
    assert s.writes == [(2.0, 4.5, 2), (5.0, 7.5, 2)]
    assert s.done == 7.5


def test_a_reader_back_from_a_write_takes_every_block_published_meanwhile():
    """One block at 1, 2 and 3 s, the last at 3.5 s: back at 2.5 s the reader
    takes the 2 s block; back at 4 s, the two published since, in one write."""
    s = _send()
    for t, tokens in ((1.0, 4), (2.0, 8), (3.0, 12), (3.5, 16)):
        s.publish(t, tokens, 16, 4)
    s.finish(3.5)
    chunk_writes([s], math.inf)
    assert s.writes == [(1.0, 2.5, 1), (2.5, 4.0, 1), (4.0, 6.5, 2)]


def test_only_the_prompt_s_last_block_may_be_partial():
    s = _send(total=3)
    s.publish(1.0, 10, 10, 4)  # 2 whole blocks and a partial: the prompt is done
    assert s.pubs == [(1.0, 3)]
    t = _send(total=3)
    t.publish(1.0, 6, 10, 4)  # mid-prompt: the whole block alone
    assert t.pubs == [(1.0, 1)]
    with pytest.raises(ValueError, match="forward in time and in blocks"):
        t.publish(0.5, 8, 10, 4)


def test_the_send_is_done_at_the_later_of_its_last_write_and_the_handoff():
    s = _send(cursor=3, total=4)  # decode holds three blocks
    s.publish(1.0, 16, 16, 4)
    s.finish(9.0)
    chunk_writes([s], math.inf)
    assert s.writes == [(1.0, 2.5, 1)] and s.done == 9.0


def test_a_decision_waits_for_the_time_it_is_told_of():
    """Up to 1.5 s the send has written nothing it was not told to; a
    publication it has not been told of yet is not used."""
    s = _send()
    s.publish(1.0, 4, 16, 4)
    assert chunk_writes([s], 0.5) == [] and s.writes == []
    assert chunk_writes([s], 1.5) == [] and s.writes == [(1.0, 2.5, 1)]
    assert s.next_at() is None  # nothing past block 1 published yet
    s.publish(3.0, 16, 16, 4)
    s.finish(3.0)
    assert chunk_writes([s], math.inf) == [s] and s.done == 6.5


def test_two_sends_on_one_link_take_it_in_decision_order():
    link = Link(UNIT)
    a, b = ChunkedSend(link, 0.0, 0, 2), ChunkedSend(link, 0.0, 0, 2)
    a.publish(1.0, 8, 8, 4)
    b.publish(1.5, 8, 8, 4)
    a.finish(1.0)
    b.finish(1.5)
    assert chunk_writes([b, a], math.inf) == [a, b]
    assert a.writes == [(1.0, 3.5, 2)] and b.writes == [(1.5, 5.5, 2)]


def test_a_middle_write_moves_blocks_alone_and_the_last_the_slot():
    model = TransferModel(0.5, 1.0, 1, slot_bytes=10)
    s = _send(total=2, model=model)
    s.publish(1.0, 4, 8, 4)
    s.publish(2.0, 8, 8, 4)
    s.finish(2.0)
    chunk_writes([s], math.inf)
    assert s.writes == [(1.0, 2.5, 1), (2.5, 14.0, 1)]


def test_a_link_s_whole_writes_price_as_before():
    """The off path: a write that is not `bare` moves the slot, as ever."""
    model = TransferModel(0.5, 1.0, 1, slot_bytes=10)
    assert Link(model).start(0.0, 2) == model.duration_s(2)
    assert Link(model).start(0.0, 2, bare=True) == 0.5 + 2.0


# -- the engines -----------------------------------------------------------

REQ = "engine-D->engine-P:kv_write_req"
DONE = "engine-P->engine-D:kv_write_done"
CHUNKED = prefill_decode_table(
    admission_s=0.54e-3, ipc_s=L, stream_s=L, router_s=L, kv_write_req_s=L, chunked=True
)
_PAIRS = itertools.count()
#: Mooncake's handoff fields the router sends decode under chunked transfer.
TEMPLATE = {"remote_pp_size": 1, "block_size": 4, "dcp_size": 1}


def _rt(lp):
    rt = LPRuntime(clock.LpId(lp), CHUNKED, Script())
    rt.start_run()
    return rt


def _at(rt, t, **released):
    rt.conn.grants.append((t, {k: list(v) for k, v in released.items()}))
    rt.advance_to(t)


def _half(rt, monkeypatch, model, *, kv_role, k, **extra):
    monkeypatch.setattr(clock, "_installed", rt)
    kv = {
        "kv_connector": "compass",
        "kv_role": kv_role,
        TRANSFER_KEY: model,
        WRITE_REQ_KEY: f"inproc://chunked-req-{k}",
        WRITE_DONE_KEY: f"inproc://chunked-done-{k}",
        **extra,
    }
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    half = KVConnectorFactory.create_connector(
        atom_config_double(kv_transfer_config=kv, kv_cache_block_size=4),
        role="scheduler",
    )
    monkeypatch.setattr(clock, "_installed", None)
    return half


@pytest.fixture
def chunked(geometry, monkeypatch):  # noqa: F811
    model = model_for(geometry, PEAKS[0])
    k = next(_PAIRS)
    rt_p, rt_d = _rt("engine-P"), _rt("engine-D")
    # Decode binds the notices, so it is built first.
    d = _half(rt_d, monkeypatch, model, kv_role="kv_consumer", k=k)
    p = _half(
        rt_p, monkeypatch, model, kv_role="kv_producer", k=k,
        enable_chunked_transfer=True, chunked_admission_timeout_s=50.0,
    )  # fmt: skip
    return SimpleNamespace(
        model=model,
        rt_p=rt_p,
        rt_d=rt_d,
        p=scheduler_with(p, max_num_batched_tokens=16),
        d=scheduler_with(d),
    )


def _forward(engine, rt, t):
    """`EngineCore`'s step: schedule, the forward ending at `t` on the device,
    whose batch the producer is told of, the KV poll, postprocess."""
    result = engine.schedule()
    batch = None if result is None else result[0]
    _at(rt, t)
    if batch is not None and batch.req_ids:
        connector_module.forward_returned(
            SimpleNamespace(device_end=t, predicted_s=0.0), batch
        )
    engine._update_from_kv_xfer_finished(KVConnectorOutput())
    if batch is not None and batch.req_ids:
        engine.postprocess(
            list(result[1].values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(FIRST_TOKEN_ID,)] * len(batch.req_ids),
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
    return batch


def test_prefill_holds_a_request_until_decode_s_write_request_then_writes_as_it_goes(
    chunked, seq_factory
):
    pd, tid = chunked, "xfer-1"
    x = Sequence(
        PROMPT,
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "do_remote_decode": True, "chunked_transfer": True, "transfer_id": tid
        },
    )  # fmt: skip
    pd.p.add(x)
    held = _forward(pd.p, pd.rt_p, 0.5)
    assert held is None or not held.req_ids, "held for decode's write request"
    assert x.status is SequenceStatus.WAITING

    y = seq_factory(
        PROMPT,
        kv_transfer_params={
            **TEMPLATE, "do_remote_prefill": True, "do_remote_decode": False,
            "chunked_transfer": True, "transfer_id": tid, "remote_dp_rank": 0,
        },
    )  # fmt: skip
    _at(pd.rt_d, 1.0)
    pd.d.add(y)
    pd.d.schedule()
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())  # posts it, width one
    a = 1.0 + L
    assert pd.rt_d.send_log == [(REQ, 0, a)]
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS

    _at(pd.rt_p, a, **{REQ: [(0, a)]})
    ends = [2.0, 3.0, 4.0]  # three forwards of 16, 16 and 8 tokens
    batches = [_forward(pd.p, pd.rt_p, t) for t in ends]
    assert [list(b.context_lens) for b in batches] == [[16], [32], [40]]
    send, *_ = pd.p.kv_connector._sends[x.id]
    assert send.pubs == [(2.0, 4), (3.0, 8), (4.0, 10)]
    assert send.handoff == 4.0, "the last forward's postprocess hands off"
    T = pd.model.duration_s  # a whole write; the middle ones move blocks alone
    bare = (
        lambda n: pd.model.latency_s
        + n * pd.model.bytes_per_block / pd.model.bandwidth_bytes_per_s
    )  # noqa: E731
    w1 = (2.0, 2.0 + bare(4), 4)
    assert send.writes[0] == w1

    t = 4.0
    while x.id in pd.p.kv_connector._sends or pd.p.kv_connector._notices:
        t += 0.25
        _forward(pd.p, pd.rt_p, t)
    c = pd.p.kv_connector._due.get(x.id) or max(e for _, e, _ in send.writes)
    assert (
        send.writes[-1][2] + send.writes[0][2] + sum(w[2] for w in send.writes[1:-1])
        == 10
    )
    assert send.writes[-1][1] == send.done == c
    assert send.writes[-1][1] == send.writes[-1][0] + T(send.writes[-1][2])
    [(ch, _, arrival)] = [e for e in pd.rt_p.send_log if e[0] == DONE]
    assert arrival == max(c, arrival - L) + L

    _at(pd.rt_d, arrival, **{DONE: [(0, arrival)]})
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    _at(pd.rt_d, arrival + 1.0)
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    pd.d.schedule()
    assert y.status is SequenceStatus.RUNNING
    assert y.kv_transfer_params["first_token_id"] == PROMPT[-1]


def test_prefill_aborts_a_request_decode_never_admits_at_its_timeout(chunked):
    pd = chunked
    x = Sequence(
        PROMPT,
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "do_remote_decode": True, "chunked_transfer": True, "transfer_id": "late"
        },
    )  # fmt: skip
    pd.p.add(x)
    _forward(pd.p, pd.rt_p, 1.0)
    assert x.status is SequenceStatus.WAITING
    _at(pd.rt_p, 51.0 + 1e-6)
    _forward(pd.p, pd.rt_p, 52.0)
    assert x.status is SequenceStatus.ABORTED
    assert x.kv_transfer_params_output is None, "nothing is relayed"
