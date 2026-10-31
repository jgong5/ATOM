# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The P/D producer's claim on a finished request's source blocks.

`Scheduler` has no producer branch: it parks a finished request while any
connector claims its HBM (`should_defer_free`) and retires the send's claim
through `send_finished`. A claim never taken frees the source under a live
RDMA read; one never dropped parks the blocks forever.

Timing matters as much as existence: `Scheduler._is_preemptable` negates the
same predicate, so claiming at alloc would pin every running request.
"""

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

with stubbed_aiter():
    # `importorskip`, not a plain import: `moriio_connector` reaches
    # `disaggregation.utils`, which imports triton, and a CPU-only runner has
    # none of it. A bare import there is a *collection* error, which takes the
    # whole suite down rather than this module. Every test below is
    # parametrized over both backends, so the module is the right granularity.
    # Same guard as `test_transfer_engine.py` and `test_pd_pp.py`.
    _SKIP = "P/D backend deps (triton, mooncake) are absent on a CPU-only runner"
    MooncakeConnectorScheduler = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
        reason=_SKIP,
    ).MooncakeConnectorScheduler
    MoRIIOConnectorScheduler = pytest.importorskip(
        "atom.kv_transfer.disaggregation.moriio.moriio_connector",
        reason=_SKIP,
    ).MoRIIOConnectorScheduler
    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        MooncakeConnector,
    )


def _build(cls, is_producer, **extra):
    """Only what `update_state_after_alloc` / `request_finished` read -- the
    rest of either scheduler needs a live backend."""
    sched = cls.__new__(cls)
    sched.is_producer = is_producer
    sched._awaiting_send = set()
    for name in ("_reqs_need_save", "_reqs_need_recv", "request_id_to_transfer_id"):
        setattr(sched, name, {})
    sched.transfer_id_to_request_id = {}
    for name, value in dict(
        engine_id="e0",
        host_ip="10.0.0.1",
        handshake_port=6300,
        base_handshake_port=6301,
        tp_size=1,
        dp_rank=0,
        **extra,
    ).items():
        setattr(sched, name, value)
    return sched


def _mooncake(is_producer=True, pp_size=1):
    return _build(
        MooncakeConnectorScheduler,
        is_producer,
        block_size=16,
        dcp_size=1,
        hash_block_size=16,
        pp_size=pp_size,
    )


def _moriio(is_producer=True):
    return _build(MoRIIOConnectorScheduler, is_producer)


def _seq(seq_id=7, **params):
    return SimpleNamespace(
        id=seq_id,
        block_table=[1, 2, 3],
        kv_transfer_params=dict(params),
        output_tokens=[42],
        spec_token_ids=None,
        prefix_cache_hit_tokens=0,
    )


BACKENDS = [pytest.param(_mooncake, id="mooncake"), pytest.param(_moriio, id="moriio")]


@pytest.mark.parametrize("build", BACKENDS)
def test_alloc_takes_no_claim_so_a_running_request_stays_preemptable(build):
    sched = build()
    seq = _seq(do_remote_decode=True)

    sched.update_state_after_alloc(seq)

    assert (
        sched.should_defer_free(seq) is False
    ), "claiming here would leave `_preempt_one_running` no candidates"


# The aggregator may report a request id as str or int; the claim keys by str.
@pytest.mark.parametrize("as_str", [False, True])
@pytest.mark.parametrize("build", BACKENDS)
def test_claim_is_taken_at_publication_and_dropped_on_send(build, as_str):
    sched = build()
    seq = _seq(do_remote_decode=True)
    sched.update_state_after_alloc(seq)
    seq.leave_reason = "stop_sequence"

    sched.request_finished(seq)
    assert (
        sched.should_defer_free(seq) is True
    ), "the peer now has these addresses and the read outlives the call"
    assert seq.kv_transfer_params_output["do_remote_prefill"] is True
    assert seq.kv_transfer_params_output["remote_block_ids"] == seq.block_table

    sched.send_finished(str(seq.id) if as_str else seq.id)
    assert sched.should_defer_free(seq) is False


@pytest.mark.parametrize("build", BACKENDS)
def test_aborted_producer_neither_claims_nor_advertises_blocks(build):
    sched = build()
    seq = _seq(do_remote_decode=True)
    sched.update_state_after_alloc(seq)
    seq.leave_reason = "aborted"

    sched.request_finished(seq)

    assert (
        sched.should_defer_free(seq) is False
    ), "an abort never sends, so nothing would retire the claim"
    assert seq.kv_transfer_params_output is None
    assert not sched.build_connector_meta().reqs_to_save


@pytest.mark.parametrize("build", BACKENDS)
def test_nothing_is_claimed_when_nothing_will_be_sent(build):
    """A consumer instance, and (mooncake only) a request the proxy never
    marked `do_remote_decode` -- the gate that fills `_reqs_need_save`.

    The old unconditional producer branch parked both anyway, waiting for a
    send that was never issued. moriio has no such gate: it never reads the
    field and sends everything it prefills.
    """
    consumer = build(is_producer=False)
    seq = _seq()
    consumer.update_state_after_alloc(seq)
    consumer.request_finished(seq)
    assert consumer.should_defer_free(seq) is False

    if build is _mooncake:
        local_only = build()
        seq = _seq()
        seq.leave_reason = "stop_sequence"
        local_only.request_finished(seq)
        assert local_only.should_defer_free(seq) is False


# -- An abort between the prefill's finish and the decode's pull ------------


def _producer_rank():
    """A Mooncake producer rank holding only the state its send path reads."""
    rank = object.__new__(MooncakeConnector)
    prefills_lock = threading.Lock()
    rank.__dict__.update(
        is_producer=True,
        request_id_to_transfer_id={},
        _completion_lock=threading.Lock(),
        _completed_prefills={},
        _completed_prefills_lock=prefills_lock,
        _completed_prefills_cv=threading.Condition(prefills_lock),
        _kv_cache_ready_events={},
        _kv_cache_ready_batches=deque(),
        _kv_ready_expired={},
        _kv_ready_done_event=None,
        _transfer_refcount={},
        _transfer_refcount_lock=threading.Lock(),
        _writes_in_flight={},
        _aborted_transfers={},
        _pending_recv_stall_watch={},
        _pending_recv_acks={},
        _held_failures={},
        done_sending=set(),
        done_recving=set(),
        failed_recving=set(),
    )
    return rank


def _parked(half, rank):
    """A request the router sends on, prefilled and finished on a real
    scheduler with Mooncake's scheduler half; its save metadata reached `rank`."""
    sched = Scheduler(MockConfig())
    sched.kv_connector = half
    seq = Sequence(
        [1, 2, 3, 4],
        4,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={"do_remote_decode": True},
    )
    sched.add(seq)
    batch, seqs = sched.schedule()
    rank.start_load_kv(batch.connector_meta_output)
    sched.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(7,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    assert seq.id in sched.deferred_free_blocks
    return sched, seq


def _poll(sched, rank):
    """One engine step's metadata to the rank, and the rank's report back."""
    rank.start_load_kv(sched.kv_connector.build_connector_meta())
    sched._update_from_kv_xfer_finished(rank.get_finished())


def _write_request(seq):
    return {"request_id": "decode-1", "transfer_id": seq.id}


def test_mooncake_reports_an_abort_before_any_write_request_at_once():
    rank = _producer_rank()
    sched, seq = _parked(_mooncake(), rank)

    assert sched.abort_request(seq.id) is True
    _poll(sched, rank)

    assert sched.deferred_free_blocks == {}
    assert sched.is_finished()
    assert not seq.block_table
    assert rank._take_write(_write_request(seq)) is False, "its blocks are free"


@pytest.mark.parametrize("wrote", [True, False], ids=["written", "failed"])
def test_mooncake_reports_an_abort_mid_write_once_the_write_ends(wrote):
    rank = _producer_rank()
    sched, seq = _parked(_mooncake(), rank)
    request = _write_request(seq)
    assert rank._take_write(request), "the listener took it before the abort"

    assert sched.abort_request(seq.id) is True
    _poll(sched, rank)
    assert seq.id in sched.deferred_free_blocks, "freed under a running write"
    assert rank._take_write(dict(request, request_id="decode-2")) is False

    def write(_request):
        if wrote:  # what a served write does on its way out
            rank.done_sending.add(seq.id)
            rank._completed_prefills.pop(seq.id)

    rank._execute_transfer = write
    rank._serve_write(request)
    output = rank.get_finished()
    assert output.finished_sending == {seq.id}, "once, written or not"
    sched._update_from_kv_xfer_finished(output)

    assert sched.deferred_free_blocks == {}
    assert sched.is_finished()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _mooncake(pp_size=2), id="mooncake-pp"),
        pytest.param(_moriio, id="moriio"),
    ],
)
def test_a_backend_that_cannot_see_every_transfer_keeps_an_aborted_claim(build):
    """Under PP prefill a stage sees only its own writes; MoRIIO's producer
    cannot see a consumer's read at all."""
    sched = build()
    seq = _seq(do_remote_decode=True)
    seq.leave_reason = "length"
    sched.request_finished(seq)

    sched.abort_send(seq.id)

    assert sched.should_defer_free(seq) is True
    assert not sched.build_connector_meta().reqs_not_processed


def test_multi_hands_an_abort_to_its_producer_sub_as_work():
    """The engine drops metadata that reports no work, so an abort alone must
    count as some."""
    producer = _mooncake()
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [producer, SimpleNamespace(build_connector_meta=lambda: None)]
    seq = _seq(do_remote_decode=True)
    seq.leave_reason = "length"
    producer.request_finished(seq)

    multi.abort_send(seq.id)
    meta = multi.build_connector_meta()

    assert meta.metas[0].reqs_not_processed == {seq.id}
    assert connector_metadata_has_work(meta)
