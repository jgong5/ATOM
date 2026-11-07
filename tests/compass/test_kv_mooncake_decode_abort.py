# SPDX-License-Identifier: MIT
"""Decode aborts a request before it asks for its write, through Mooncake.

Prefill's leg of the request is over, so no abort reaches it, and decode never
sent the write request whose completion would have retired prefill's send
claim. Without a word from decode, prefill keeps the claim for ever: its blocks
never return, and under DP attention its group runs dummy forwards for ever.

Both engines are real schedulers over Mooncake's scheduler halves, as
`test_pd_source_claim.py` builds them. Decode's notice goes out on a real
socket to one bound in place of the producer rank's listener, and the rank's
own listener then reads those frames from a scripted socket.
"""

from __future__ import annotations

import contextlib
import signal
import socket
from types import SimpleNamespace

import msgpack
import pytest
import zmq
from test_pd_source_claim import (
    MooncakeConnector,
    _engine,
    _mooncake,
    _parked,
    _producer_rank,
    _refused,
    _step,
    _write_request,
)

from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

#: The namespace of the Mooncake module the helpers' classes were built from.
#: `stubbed_aiter` drops that module from `sys.modules` once it is built, so an
#: import here would be a second copy, which those classes never read.
MC = MooncakeConnector._write_listener.__globals__


class _Ended(Exception):
    """The scripted listener socket has handed over every message."""


def _listen(rank, monkeypatch, messages) -> None:
    """Run producer `rank`'s listener over `messages`, as its ROUTER reads them."""
    messages = list(messages)

    def recv_multipart():
        if not messages:
            raise _Ended
        return messages.pop(0)

    sock = SimpleNamespace(recv_multipart=recv_multipart)
    monkeypatch.setitem(
        MC, "zmq_socket_ctx", lambda *_a, **_k: contextlib.nullcontext(sock)
    )
    rank._side_channel_port = 0
    rank._publish_listener_state = lambda exc=None: None
    with pytest.raises(_Ended):
        rank._write_listener()


def _consumer():
    consumer = _mooncake(is_producer=False)
    consumer._abort_sockets = {}
    return consumer


def _unwritten(blob):
    """A request a consumer finishes before asking for its write."""
    return SimpleNamespace(
        id=5,
        kv_transfer_params={**blob, "do_remote_prefill": True},
        block_table=[],
        output_tokens=[],
        spec_token_ids=None,
        prefix_cache_hit_tokens=0,
    )


def test_a_decode_abort_before_the_write_request_frees_the_producer(monkeypatch):
    rank = _producer_rank()
    prefill = _engine(rank)
    x = _parked(prefill)
    listener = zmq.Context.instance().socket(zmq.ROUTER)
    listener.setsockopt(zmq.LINGER, 0)
    port = listener.bind_to_random_port("tcp://127.0.0.1")
    decode = _engine(None)
    decode.scheduler.kv_connector = consumer = _consumer()
    blob = dict(x.kv_transfer_params_output)
    blob.update(remote_host="127.0.0.1", remote_handshake_port=port)
    y = Sequence(
        [1, 2, 3, 4],
        4,
        sampling_params=SamplingParams(max_tokens=8),
        kv_transfer_params=blob,
    )
    decode.scheduler.add(y)

    try:
        assert decode.scheduler.abort_request(y.id) is True
        assert listener.poll(5000), "the producer's listener heard nothing"
        message = listener.recv_multipart()
        consumer.request_finished(y)
        assert not listener.poll(100), "told once"
    finally:
        listener.close()
        for sock in consumer._abort_sockets.values():
            sock.close()
    assert message[1:] == [MC["MSG_ABORT"], msgpack.dumps({"transfer_id": x.id})]

    _listen(rank, monkeypatch, [message])
    _step(prefill)

    assert prefill.scheduler.deferred_free_blocks == {}
    assert prefill.scheduler.is_finished()
    assert _refused(rank, _write_request(x)), "its blocks may be another's now"


def test_every_rank_of_every_producer_stage_is_told(monkeypatch):
    """Each producer rank reports its own send, and the producer's KV
    aggregation waits for all of them, so each one is told: at the port its
    listener binds."""
    sent = []

    class Socket:
        def setsockopt(self, *_):
            pass

        def connect(self, addr):
            self.addr = addr

        def send_multipart(self, frames):
            sent.append(self.addr)

    context = SimpleNamespace(socket=lambda _kind: Socket())
    fake = SimpleNamespace(
        Context=SimpleNamespace(instance=lambda: context),
        DEALER=0,
        LINGER=0,
        SNDHWM=0,
    )
    monkeypatch.setitem(MC, "zmq", fake)
    blob = {
        "transfer_id": 9,
        "remote_host": "producer",
        "remote_handshake_port": 7000,
        "tp_size": 2,
        "remote_pp_size": 2,
        "remote_dp_rank": 1,
        "remote_dp_size": 2,
    }

    _consumer().request_finished(_unwritten(blob))

    port = MC["_port_offset"]
    listeners = [
        f"tcp://producer:{7000 + port(1, tp_rank, 2, pp_rank, 2, 2)}"
        for pp_rank in (0, 1)
        for tp_rank in (0, 1)
    ]
    assert sent == listeners


def test_notices_to_an_unreachable_producer_rank_never_block_the_step_loop():
    """A producer rank that never answers keeps every notice sent to it queued,
    and decode's step loop, which sends them, goes on. At a DEALER's default
    high-water mark, the send after the thousandth queued one would block."""
    with socket.socket() as unbound:
        unbound.bind(("127.0.0.1", 0))
        port = unbound.getsockname()[1]
    blob = {
        "transfer_id": 9,
        "remote_host": "127.0.0.1",
        "remote_handshake_port": port,
        "tp_size": 1,
    }
    consumer = _consumer()

    def blocked(*_):
        raise TimeoutError("a notice blocked the step loop")

    previous = signal.signal(signal.SIGALRM, blocked)
    signal.alarm(20)
    try:
        for _ in range(1001):
            consumer.request_finished(_unwritten(blob))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        for sock in consumer._abort_sockets.values():
            sock.close()
