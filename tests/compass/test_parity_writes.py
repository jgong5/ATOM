# SPDX-License-Identifier: MIT
"""A recorded P/D run stamps each Mooncake write on the calls ATOM makes to serve it.

Each side's connector is a real `MooncakeConnector` holding only the state its
send or receive path reads, and the recording runner is `StepRecording` over a
base whose `allocate_kv_cache` builds that connector, as ATOM's does. The
producer's listener takes write requests off a scripted socket, and its send
workers serve them through `_execute_transfer` into fake transfer engines that
log each write they are handed; the two writes that move bytes are in flight
at once. Its write-dones, which ATOM resends, reach the consumer's listener the
same way. A producer is recorded alike when a multi connector holds it beside
an offload tier, as a prefill instance with the Mooncake Store runs.
"""

import importlib
import json
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest

from atom.compass.parity import ENV, StepRecording
from atom.compass.parity.writes import REQUEST_STAMPS

PRODUCER = {"pp_rank": 0, "dp_rank": 2, "tp_rank": 0}
CONSUMER = {"pp_rank": 0, "dp_rank": 6, "tp_rank": 0}
#: The producer's KV regions and their bytes per block; each prefill holds three blocks.
REGIONS, BLOCK_BYTES, BLOCKS = [0x10000, 0x20000], [64, 32], [1, 2, 3]
#: Six descriptors, two per RDMA call.
CALL_BYTES = [64 + 64, 64 + 32, 32 + 32]
PRODUCER_FIELDS = {
    "request_id",
    "transfer_id",
    "write_nonce",
    "t_receipt_ns",
    "t_serve_ns",
    "rdma",
    "hca",
    "t_notify_ns",
    "success",
}
CONSUMER_FIELDS = {
    "request_id",
    "pp_rank",
    "tp_rank",
    "write_nonce",
    "success",
    "t_write_done_ns",
}


@pytest.fixture
def mc(monkeypatch):
    """ATOM's Mooncake connector module, its process groups stubbed: on a CPU
    runner the real ones probe a GPU arch on import."""
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", MagicMock())
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")
    return importlib.import_module(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )


class _Engine:
    """A transfer engine on one HCA, logging each write it is handed as
    ``(HCA, target, bytes, start, end)`` once its partner write has reached one."""

    def __init__(self, log, meet, device=None):
        self.log, self.meet, self.device = log, meet, device

    def initialize(self, local_ip, metadata, protocol, device):
        self.device = device  # the rail `RailEnginePool` starts it on
        return 0

    def register_memory(self, ptr, size):
        return 0

    def get_rpc_port(self):
        return 0

    def batch_transfer_sync_write(self, target, src_addrs, dst_addrs, sizes):
        start = time.monotonic_ns()
        self.meet.wait()
        self.log.append((self.device, target, sum(sizes), start, time.monotonic_ns()))
        return 0


def _producer(mc, log, meet, matched):
    """A producer rank on ionic_2 with two send workers, through matched rails
    on ionic_2 and ionic_6 if `matched`. Its write-dones go in `sent`."""
    p = object.__new__(mc.MooncakeConnector)
    lock = threading.Lock()
    own = _Engine(log, meet, "ionic_2")
    p.__dict__.update(
        PRODUCER,
        is_producer=True,
        pp_size=1,
        tp_size=1,
        dcp_size=1,
        ib_devices=["ionic_2"],
        transfer_engine=own,
        kv_caches_base_addr=REGIONS,
        _per_block_bytes_list=BLOCK_BYTES,
        _block_region_roles=[None, None],
        _block_region_consumer_indices=None,
        _block_head_major_runs=(None, None),
        _block_unit_specs=None,  # no backend here names its units
        _unit_plans={},
        _num_local_layers=0,
        _start_layer=0,
        _MAX_RDMA_ENTRIES_PER_BATCH=2,
        _completed_prefills={
            t: {"block_ids": BLOCKS, "slot_index": -1} for t in ("a", "b", "c")
        },
        _completed_prefills_lock=lock,
        _completed_prefills_cv=threading.Condition(lock),
        _kv_cache_ready_events={},
        _transfer_refcount={},
        _transfer_refcount_lock=threading.Lock(),
        _completion_lock=threading.Lock(),
        _writes_in_flight={},
        _aborted_transfers={},
        # Chunked transfer state, which the listener dispatches from each pass.
        _chunked_prefills={},
        _chunked_local_ids={},
        _retired_chunked_transfers=OrderedDict(),
        _pending_chunked_requests=[],
        _chunked_admissions={},
        _admission_results=set(),
        done_sending=set(),
        _send_executor=ThreadPoolExecutor(2),
        _side_channel_port=0,
        _publish_listener_state=lambda exc=None: None,
        sent=[],
    )
    p._send_on_socket = lambda _addr, parts, repeat=1: p.sent.extend([parts] * repeat)
    if matched:
        p._rail_pool = mc.RailEnginePool(
            lambda: _Engine(log, meet),
            own,
            "ionic_2",
            ["ionic_2", "ionic_6"],
            lambda _device: "127.0.0.1",
        )
        p._rail_pool.set_regions(REGIONS, [4096, 4096])
    return p


def _request(transfer_id, nonce, port, device, dst_block_ids):
    return {
        "request_id": f"d-{transfer_id}",
        "transfer_id": transfer_id,
        "write_nonce": nonce,
        "consumer_host": "10.0.0.9",
        "consumer_rpc_port": port,
        "consumer_ib_device": device,
        "consumer_base_addrs": [0x90000, 0xA0000],
        "dst_block_ids": dst_block_ids,
        "notify_host": "10.0.0.9",
        "notify_port": 5555,
    }


def _requests():
    """Two decode ranks' write requests, on ionic_2 and ionic_6, and a third
    whose destination is a block short, which the producer fails unwritten."""
    return [
        _request("a", 11, 7002, "ionic_2", [5, 6, 7]),
        _request("b", 12, 7006, "ionic_6", [8, 9, 10]),
        _request("c", 13, 7010, "ionic_6", [11, 12]),
    ]


def _consumer(mc, requests):
    """A consumer rank awaiting each request's write-done from one producer rank."""
    c = object.__new__(mc.MooncakeConnector)
    c.__dict__.update(
        CONSUMER,
        is_producer=False,
        _pending_recv_expected={r["request_id"]: 1 for r in requests},
        _pending_recv_nonce={r["request_id"]: r["write_nonce"] for r in requests},
        _pending_recv_stages={},
        _pending_recv_failed=set(),
        _pending_recv_acks={},
        _pending_recv_stall_watch={},
        _pending_recv={r["request_id"] for r in requests},
        _pending_recv_blocks={},
        _pending_recv_slots={},
        _pending_handoffs={},  # chunked transfers' handoffs: none here
        _held_failures={},
        _blocks_pending_fence=[],
        _completion_lock=threading.Lock(),
        _fence_lock=threading.Lock(),
        done_recving=set(),
        failed_recving=set(),
        _scatter_slot=None,
        _release_targets={},
        _received_handoffs={},
        _publish_listener_state=lambda exc=None: None,
    )
    return c


def _multi(producer):
    """A multi connector holding an offload tier, then `producer`, as a prefill
    instance beside the Mooncake Store builds it."""
    from atom.kv_transfer.disaggregation.multi.multi_connector import MultiConnector

    multi = object.__new__(MultiConnector)
    offload = SimpleNamespace(is_offload=True, is_producer=False)
    multi.__dict__.update(
        _connectors=[offload, producer], is_producer=True, _state_tier=None
    )
    return multi


def _runner(conn, monkeypatch):
    """The recording runner over a base whose `allocate_kv_cache` builds `conn`."""
    forward_context = importlib.import_module("atom.utils.forward_context")

    class _Base:
        config = SimpleNamespace(enable_rapidserve=False)

        def allocate_kv_cache(self, num_kvcache_blocks):
            monkeypatch.setattr(forward_context, "_global_kvconnector", conn)
            return True

        def exit(self):
            return True

    class _Recorded(StepRecording, _Base):
        # A real runner's step clock, on which ATOM's listener stamps its receipt.
        step_clock_ns = staticmethod(time.monotonic_ns)

    runner = _Recorded()
    assert runner.allocate_kv_cache(16)
    return runner


class _Ended(Exception):
    """The scripted socket has handed over every frame."""


def _listen(mc, monkeypatch, listener, frames):
    """Run `listener` over `frames`, as its ROUTER socket receives them."""
    frames = list(frames)

    def recv_multipart():
        if not frames:
            raise _Ended
        return frames.pop(0)

    sock = SimpleNamespace(
        recv_multipart=recv_multipart,
        poll=lambda timeout=None: True,  # a frame is always waiting
        getsockopt=lambda _option: b"tcp://127.0.0.1:5555",
    )
    monkeypatch.setattr(mc, "zmq_socket_ctx", lambda *_a, **_k: nullcontext(sock))
    with pytest.raises(_Ended):
        listener()


def _serve(mc, monkeypatch, producer, requests):
    """`requests` through the producer's listener and send workers; returns
    the monotonic clock before and after the listener took them in."""
    frames = [[b"d", mc.MSG_WRITE_REQUEST, msgpack.dumps(r)] for r in requests]
    before = time.monotonic_ns()
    _listen(mc, monkeypatch, producer._write_listener, frames)
    after = time.monotonic_ns()
    producer._send_executor.shutdown(wait=True)
    return before, after


def _lines(path):
    """`path`'s lines by request id, one each."""
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    by_request = {line["request_id"]: line for line in lines}
    assert len(by_request) == len(lines), "a write on more than one line"
    return by_request


@pytest.mark.parametrize("multi", [False, True], ids=["mooncake", "multi"])
@pytest.mark.parametrize("matched", [False, True], ids=["own-hca", "matched-rails"])
def test_each_write_is_stamped_on_the_calls_that_serve_it(
    mc, tmp_path, monkeypatch, matched, multi
):
    log, requests = [], _requests()
    producer = _producer(mc, log, threading.Barrier(2, timeout=10), matched)
    monkeypatch.setenv(ENV, str(tmp_path / "prefill"))
    prefill = _runner(_multi(producer) if multi else producer, monkeypatch)
    before, after = _serve(mc, monkeypatch, producer, requests)
    prefill.exit()

    consumer = _consumer(mc, requests)
    monkeypatch.setenv(ENV, str(tmp_path / "decode"))
    decode = _runner(consumer, monkeypatch)
    # More write-dones than writes: the consumer's record must drop the repeats.
    assert len(producer.sent) > len(requests)
    frames = [[b"p", *parts] for parts in producer.sent]
    _listen(mc, monkeypatch, consumer._notification_listener, frames)
    decode.exit()
    assert producer.done_sending == {"a", "b"}
    assert (consumer.done_recving, consumer.failed_recving) == ({"d-a", "d-b"}, {"d-c"})

    writes = _lines(tmp_path / "prefill" / "dp2.tp0.writes.jsonl")
    dones = _lines(tmp_path / "decode" / "dp6.tp0.writes.jsonl")
    assert writes.keys() == dones.keys() == {r["request_id"] for r in requests}
    a, b, c = (writes[r["request_id"]] for r in requests)
    assert set(a) == set(b) == PRODUCER_FIELDS
    assert set(c) == PRODUCER_FIELDS - {"hca"}
    assert (c["rdma"], c["success"]) == ([], False)
    hcas = ["ionic_2", "ionic_6"] if matched else ["ionic_2", "ionic_2"]
    assert [a["hca"], b["hca"]] == hcas
    for line, request in zip((a, b), requests):
        target = f"10.0.0.9:{request['consumer_rpc_port']}"
        calls = [call for call in log if call[1] == target]
        assert {call[0] for call in calls} == {line["hca"]}
        assert [n for _, _, n, _, _ in calls] == [n for *_, n in line["rdma"]]
        assert [n for *_, n in line["rdma"]] == CALL_BYTES
        for (start, end, _), (*_, logged_start, logged_end) in zip(line["rdma"], calls):
            assert start <= logged_start <= logged_end <= end
        assert line["success"] is True
    # Both writes were inside their RDMA calls at once.
    assert a["rdma"][0][0] < b["rdma"][0][1] and b["rdma"][0][0] < a["rdma"][0][1]
    for line in writes.values():
        assert before <= line["t_receipt_ns"] <= after
        rdma = [t for start, end, _ in line["rdma"] for t in (start, end)]
        serve, notify = line["t_serve_ns"], line["t_notify_ns"]
        stamps = [line["t_receipt_ns"], serve[0], *rdma, *notify, serve[1]]
        assert stamps == sorted(stamps)

    for request_id, done in dones.items():
        write = writes[request_id]
        assert set(done) == CONSUMER_FIELDS
        assert done["write_nonce"] == write["write_nonce"]
        assert (done["pp_rank"], done["tp_rank"]) == (0, 0)
        assert done["success"] is write["success"]
        assert done["t_write_done_ns"] >= write["t_notify_ns"][1]


def _receiving(mc, sent):
    """A consumer rank whose `start_load_kv` builds its write requests, its
    write-request thread driven by hand and its sends kept in `sent`."""
    c = _consumer(mc, [])
    c.__dict__.update(
        tp_size=1,
        pp_size=1,
        dcp_size=1,
        dcp_rank=0,
        dcp_interleave_size=1,
        local_ip="10.0.0.9",
        rpc_port=7002,
        ib_device="ionic_2",
        _notification_port=5555,
        _num_local_layers=0,
        _block_region_roles=[None, None],
        _block_regions=list(zip([0x90000, 0xA0000], BLOCK_BYTES)),
        _unit_layout=None,
        _has_slot_regions=False,
        _swa_block_regions=[],
        kv_caches_base_addr=[0x90000, 0xA0000],
        _mla_landing=None,
        _cuda_device=0,
        _write_queue=[],
        _write_cv=threading.Condition(),
        _write_closed=False,
        _write_thread=object(),  # no thread: `_post_queued_write_requests` by hand
        _write_request_error=None,
        _verified_producers={"tcp://10.0.0.1:6301"},
        _staging_slots=mc.SlotPool(),
    )
    c._send_on_socket = lambda addr, parts, repeat=1: sent.append((addr, parts))
    return c


def _receive(transfer_id, blocks, chunked=False):
    """A receive's metadata: P at TP 1, or under a chunked transfer TP 2, which a
    TP 1 consumer refuses before it posts anything."""
    from atom.kv_transfer.disaggregation.types import ConnectorMetadata

    return ConnectorMetadata._build_req_meta(
        req_id=f"d-{transfer_id}",
        local_block_ids=blocks,
        kv_transfer_params={
            "remote_block_ids": BLOCKS,
            "remote_host": "10.0.0.1",
            "remote_handshake_port": 6301,
            "tp_size": 2 if chunked else 1,
            "transfer_id": transfer_id,
            "chunked_transfer": chunked,
        },
    )


def test_a_write_request_is_stamped_from_its_load_to_its_send(
    mc, tmp_path, monkeypatch
):
    """D's stamps say when each write request was asked for, queued, taken by
    the write-request thread for its device synchronize, and sent, on the
    line of the write-done that answered it, or on its own line."""
    sent, synced = [], []
    consumer = _receiving(mc, sent)
    monkeypatch.setenv(ENV, str(tmp_path / "decode"))
    decode = _runner(consumer, monkeypatch)
    # The synchronize runs once the thread has taken the queue.
    monkeypatch.setattr(
        mc.torch.cuda, "synchronize", lambda *a: synced.append(time.monotonic_ns())
    )
    receives = {
        "d-a": _receive("a", [5, 6, 7]),
        # Refused before it is queued: it must not take the next one's stamps.
        "d-r": _receive("r", [8, 9, 10], chunked=True),
        "d-b": _receive("b", [11, 12, 13]),
    }
    meta = SimpleNamespace(request_id_to_transfer_id={}, reqs_to_recv=receives)
    before = time.monotonic_ns()
    consumer.start_load_kv(meta)
    assert not sent and len(consumer._write_queue) == 2
    assert consumer.failed_recving == {"d-r"}
    assert consumer._post_queued_write_requests()
    after = time.monotonic_ns()
    assert len(synced) == 1 and len(sent) == 2
    nonces = {}
    for addr, (kind, payload) in sent:
        request = msgpack.loads(payload)
        assert (addr, kind) == ("tcp://10.0.0.1:6301", mc.MSG_WRITE_REQUEST)
        nonces[request["request_id"]] = request["write_nonce"]
    assert nonces.keys() == {"d-a", "d-b"}
    # Only d-a is answered, twice, as the producer resends.
    done = {
        "request_id": "d-a",
        "pp_rank": 0,
        "tp_rank": 0,
        "write_nonce": nonces["d-a"],
        "success": True,
    }
    frame = [b"p", mc.MSG_WRITE_DONE, msgpack.dumps(done)]
    _listen(mc, monkeypatch, consumer._notification_listener, [frame, frame])
    decode.exit()
    assert consumer.done_recving == {"d-a"}

    lines = _lines(tmp_path / "decode" / "dp6.tp0.writes.jsonl")
    assert lines.keys() == {"d-a", "d-b"}
    a, b = lines["d-a"], lines["d-b"]
    assert set(a) == CONSUMER_FIELDS | set(REQUEST_STAMPS)
    assert set(b) == {"request_id", "write_nonce", *REQUEST_STAMPS}
    assert (a["write_nonce"], b["write_nonce"]) == (nonces["d-a"], nonces["d-b"])
    assert a["t_load_ns"] == b["t_load_ns"]  # one metadata asked for both
    # One take of the queue, then one synchronize, covered both.
    assert a["t_sync_ns"] == b["t_sync_ns"] <= synced[0]
    for line in (a, b):
        (send,) = line["t_send_ns"]
        posted = [line["t_load_ns"], line["t_post_ns"], line["t_sync_ns"]]
        stamps = [before, *posted, *send, after]
        assert stamps == sorted(stamps)
    assert a["t_post_ns"] < b["t_post_ns"]
    assert a["t_send_ns"][0][1] <= b["t_send_ns"][0][0]
    assert a["t_write_done_ns"] >= a["t_send_ns"][0][1]


def test_an_unrecorded_run_wraps_nothing(mc, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    log, requests = [], _requests()[:2]
    producer = _producer(mc, log, threading.Barrier(2, timeout=10), matched=True)
    consumer = _consumer(mc, requests)
    runners = [_runner(conn, monkeypatch) for conn in (producer, consumer)]

    assert [runner._writes for runner in runners] == [None, None]
    wrapped = {
        "_serve_write",
        "_rdma_write",
        "_notify_transfer_result",
        "_record_write_done",
        "start_load_kv",
        "_enqueue_write_requests",
    }
    assert not wrapped & (vars(producer).keys() | vars(consumer).keys())
    # The producer double's own `_send_on_socket` is an instance attribute.
    assert "_send_on_socket" not in vars(consumer)
    _serve(mc, monkeypatch, producer, requests)
    assert producer.done_sending == {"a", "b"}
    assert sorted(call[0] for call in log) == ["ionic_2"] * 3 + ["ionic_6"] * 3
    for runner in runners:
        runner.exit()
