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
    PROMPT,
    L,
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


def test_a_write_decided_after_a_later_one_queues_behind_it():
    """A later PP stage decides a send that handed off to its end at its last
    forward (it takes no poll): `a`'s second write arrives at 2.5 s. `b`'s
    publication at 2 s, told afterwards, writes once the link's latest has
    arrived, and says how long it queued; told first, nothing queues."""
    link = Link(UNIT)
    a, b = ChunkedSend(link, 0.0, 0, 2), ChunkedSend(link, 0.0, 0, 2)
    a.publish(1.0, 4, 8, 4)
    a.publish(1.5, 8, 8, 4)
    a.finish(1.5)
    assert chunk_writes([a, b], math.inf) == [a]
    assert a.writes == [(1.0, 2.5, 1), (2.5, 4.0, 1)] and a.clamped == 0.0
    b.publish(2.0, 8, 8, 4)
    b.finish(2.0)
    assert chunk_writes([a, b], math.inf) == [b]
    assert b.writes == [(2.5, 6.0, 2)] and b.clamped == 0.5
    link = Link(UNIT)
    a, b = ChunkedSend(link, 0.0, 0, 2), ChunkedSend(link, 0.0, 0, 2)
    for s, t in ((a, 1.0), (b, 2.0)):
        s.publish(t, 8, 8, 4)
        s.finish(t)
    assert chunk_writes([a, b], math.inf) == [a, b]
    assert b.writes == [(2.0, 5.5, 2)] and b.clamped == 0.0


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


def _free_ports(n):
    """The first of `n` consecutive free local ports: decode binds a notice
    socket a stage, at the configured port plus the stage's index, and a socket
    an earlier test bound stays bound."""
    import socket

    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        try:
            for k in range(1, n):
                with socket.socket() as other:
                    other.bind(("127.0.0.1", port + k))
        except OSError:
            continue
        return port
    raise RuntimeError(f"no {n} consecutive free ports")


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
    )
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
    [(_, _, arrival)] = [e for e in pd.rt_p.send_log if e[0] == DONE]
    assert arrival >= c + L  # the notice leaves once the write is done

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


def test_a_write_request_after_prefill_s_timeout_fails_decode_s_receive(
    chunked, seq_factory, monkeypatch
):
    """Decode, which deferred the request past prefill's admission timeout,
    admits it later; prefill fails the write request, as Mooncake fails a
    retired transfer's reader, and decode aborts the request once its failed
    write hold elapses. Before, decode kept it parked for good, its
    reservation charged to pd_admission, and both engines polled forever."""
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "5")
    pd, tid = chunked, "late"
    x = Sequence(
        PROMPT,
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "do_remote_decode": True, "chunked_transfer": True, "transfer_id": tid
        },
    )  # fmt: skip
    pd.p.add(x)
    _forward(pd.p, pd.rt_p, 1.0)
    _at(pd.rt_p, 51.0 + 1e-6)
    _forward(pd.p, pd.rt_p, 52.0)
    assert x.status is SequenceStatus.ABORTED, "prefill's admission timeout"

    y = seq_factory(
        PROMPT,
        kv_transfer_params={
            **TEMPLATE, "do_remote_prefill": True, "do_remote_decode": False,
            "chunked_transfer": True, "transfer_id": tid, "remote_dp_rank": 0,
        },
    )  # fmt: skip
    _at(pd.rt_d, 60.0)
    pd.d.add(y)
    pd.d.schedule()
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    a = 60.0 + L
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS

    _at(pd.rt_p, a, **{REQ: [(0, a)]})
    _forward(pd.p, pd.rt_p, a + 0.5)
    assert not pd.p.kv_connector._reads, "no read waits for a retired transfer"
    assert not pd.p.kv_connector.has_pending_work()
    [(_, _, arrival)] = [e for e in pd.rt_p.send_log if e[0] == DONE]
    assert arrival == pytest.approx(a + 0.5 + L), "read at the step's KV poll"

    _at(pd.rt_d, arrival, **{DONE: [(0, arrival)]})
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    assert y.status is SequenceStatus.WAITING_FOR_REMOTE_KVS, "the hold"
    assert pd.d.kv_connector.has_pending_work()
    for t in (arrival + 5.0, arrival + 5.5):  # the poll after the one that saw it
        _at(pd.rt_d, t)
        pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    # `_abort_failed_remote_receive` ends it as a rejected waiter.
    assert y.status is SequenceStatus.FINISHED
    assert y not in pd.d.waiting and y not in pd.d.running
    assert not pd.d.kv_connector.has_pending_work()


def test_an_aborted_send_tells_decode_its_writer_failed(chunked, seq_factory):
    """A chunked request aborted after allocating stops its reads, and decode,
    which would otherwise wait for a write-done that never comes, hears of it."""
    pd, tid = chunked, "cut"
    x = Sequence(
        PROMPT,
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "do_remote_decode": True, "chunked_transfer": True, "transfer_id": tid
        },
    )  # fmt: skip
    pd.p.add(x)
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
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    a = 1.0 + L
    _at(pd.rt_p, a, **{REQ: [(0, a)]})
    _forward(pd.p, pd.rt_p, 2.0)
    assert x.id in pd.p.kv_connector._sends, "allocated and sending"

    x.status, x.leave_reason = SequenceStatus.ABORTED, "aborted"
    pd.p.kv_connector.request_finished(x)
    assert x.id not in pd.p.kv_connector._sends
    [(_, _, arrival)] = [e for e in pd.rt_p.send_log if e[0] == DONE]
    _at(pd.rt_d, arrival, **{DONE: [(0, arrival)]})
    pd.d._update_from_kv_xfer_finished(KVConnectorOutput())
    assert tid not in pd.d.kv_connector._listening
    assert y.id in pd.d.kv_connector._failed


def test_a_later_stage_writes_its_layers_from_the_relay_and_frees_nothing(
    geometry, monkeypatch  # noqa: F811
):
    """Stage 1 of a two-stage prefill: the head's relay opens its send, each
    forward publishes its layers' blocks, and its last forward hands off; the
    stage tells decode on its own notice channel and holds nothing after, as
    the head's scheduler frees the blocks on decode's release."""

    table = prefill_decode_table(
        admission_s=0.54e-3, ipc_s=L, stream_s=L, router_s=L, kv_write_req_s=L,
        prefill_pp=2, chunked=True,
    )  # fmt: skip
    rts = {}
    for lp in ("engine-D", "engine-P-pp1"):
        rts[lp] = LPRuntime(clock.LpId(lp), table, Script())
        rts[lp].start_run()
    port = _free_ports(2)
    model = model_for(geometry, PEAKS[0])
    kv = {
        "kv_connector": "compass",
        TRANSFER_KEY: (model, model),
        WRITE_REQ_KEY: f"inproc://chunked-stage-req-{port}",
        WRITE_DONE_KEY: f"tcp://127.0.0.1:{port}",
    }
    parallel = SimpleNamespace(
        data_parallel_rank=0, data_parallel_size=1, pipeline_parallel_rank=1
    )
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    halves = {}
    # Decode binds a notice socket per stage, so it is built first.
    for lp, role, extra in (
        ("engine-D", "kv_consumer", {}),
        ("engine-P-pp1", "kv_producer", {"enable_chunked_transfer": True}),
    ):
        monkeypatch.setattr(clock, "_installed", rts[lp])
        halves[lp] = KVConnectorFactory.create_connector(
            atom_config_double(
                kv_transfer_config={**kv, "kv_role": role, **extra},
                kv_cache_block_size=4,
                pipeline_parallel_size=2,
                parallel_config=parallel if role == "kv_producer" else
                SimpleNamespace(data_parallel_rank=0, data_parallel_size=1),
            ),
            role="scheduler",
        )  # fmt: skip
    monkeypatch.setattr(clock, "_installed", None)
    stage = halves["engine-P-pp1"]
    pair = ("engine-D", 0)

    def forward(end, tokens, relay=None):
        meta = SimpleNamespace(compass_chunked=relay) if relay else None
        batch = SimpleNamespace(
            connector_meta_output=meta,
            req_ids=["r"],
            context_lens=[tokens],
            total_seqs_num_prefill=1,
        )
        stage.prefill_done(end, batch)

    # A 10-token prompt, three blocks, all of which decode asked for.
    forward(1.0, 8, relay={"r": ("t", pair, 0.5, 3, 10)})
    send = stage._sends["r"][0]
    assert send.writes and send.writes[0][0] == 1.0 and send.writes[0][2] == 2
    # Its forwards move it and wake the stage: an idle poll would not, and
    # one aborted midway would keep the stage polling for good.
    assert send.done is None and not stage.has_pending_work()
    forward(2.0, 10)
    assert "r" not in stage._sends and send.done is not None
    assert [w[2] for w in send.writes] == [2, 1]
    assert send.done == max(send.writes[-1][1], 2.0)
    (notice,) = [e for e in rts["engine-P-pp1"].send_log if ":kv_write_done" in e[0]]
    assert notice[0] == "engine-P-pp1->engine-D:kv_write_done"
    assert notice[2] == send.done + L
    assert not stage._released and not stage.has_pending_work()
    # The head repeats its relay in every metadata until its own send is
    # done: a relay seen again opens no second send.
    forward(3.0, 10, relay={"r": ("t", pair, 0.5, 3, 10)})
    assert "r" not in stage._sends and not stage.has_pending_work()
    assert (
        len([e for e in rts["engine-P-pp1"].send_log if ":kv_write_done" in e[0]]) == 1
    )


def test_the_head_relays_a_send_in_every_metadata_until_its_own_is_done(
    geometry, monkeypatch  # noqa: F811
):
    """A chunked send the head admitted rides every connector metadata it
    builds, not the first alone: a request parked for a tier load is admitted
    under connector-only batches, which run no forward on the stages and so
    reach no stage's `relayed` (M3.1 B c=64: request 195's three later stages
    never opened their sends, decode never heard them, and its reservation
    held pd_admission for good). An abort ends the relay."""

    table = prefill_decode_table(
        admission_s=0.54e-3, ipc_s=L, stream_s=L, router_s=L, kv_write_req_s=L,
        prefill_pp=2, chunked=True,
    )  # fmt: skip
    rts = {}
    for lp in ("engine-D", "engine-P-pp0"):
        rts[lp] = LPRuntime(clock.LpId(lp), table, Script())
        rts[lp].start_run()
    port = _free_ports(2)
    model = model_for(geometry, PEAKS[0])
    kv = {
        "kv_connector": "compass",
        TRANSFER_KEY: (model, model),
        WRITE_REQ_KEY: f"inproc://chunked-head-req-{port}",
        WRITE_DONE_KEY: f"tcp://127.0.0.1:{port}",
    }
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    halves = {}
    for lp, role, extra, parallel in (
        ("engine-D", "kv_consumer", {},
         SimpleNamespace(data_parallel_rank=0, data_parallel_size=1)),
        ("engine-P-pp0", "kv_producer", {"enable_chunked_transfer": True},
         SimpleNamespace(data_parallel_rank=0, data_parallel_size=1,
                         pipeline_parallel_rank=0)),
    ):  # fmt: skip
        monkeypatch.setattr(clock, "_installed", rts[lp])
        halves[lp] = KVConnectorFactory.create_connector(
            atom_config_double(
                kv_transfer_config={**kv, "kv_role": role, **extra},
                kv_cache_block_size=4,
                pipeline_parallel_size=2,
                parallel_config=parallel,
            ),
            role="scheduler",
        )
    monkeypatch.setattr(clock, "_installed", None)
    head = halves["engine-P-pp0"]
    x = Sequence(
        PROMPT,
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "do_remote_decode": True, "chunked_transfer": True, "transfer_id": "t"
        },
    )  # fmt: skip
    head._reads["t"] = (("engine-D", 0), 3, 0.5)  # decode's write request is in
    assert head.admission_ready(x)
    head.update_state_after_alloc(x)
    for _ in range(3):  # connector-only batches, then the first forward's
        assert set(head.build_connector_meta().compass_chunked) == {x.id}
    x.status, x.leave_reason = SequenceStatus.ABORTED, "aborted"
    head.request_finished(x)
    assert getattr(head.build_connector_meta(), "compass_chunked", None) is None
