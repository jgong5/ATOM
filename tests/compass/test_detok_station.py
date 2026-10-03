# SPDX-License-Identifier: MIT
"""The detokenization station: one width-1 station per frontend output thread.

ATOM's own output thread (`CoreManager._create_output_thread`) reads engine
output through the zmq shim on a frontend whose loop is a `CompassEventLoop`,
against the co-hosted clock authority. Stream updates go through ATOM's
`StreamBatchDispatcher`, whose tokenizer's decode `wrap_decode` charges.
"""

import math
import pickle
import threading
import uuid
from types import SimpleNamespace

import pytest
import zmq
from aiter_stub import stubbed_aiter

from atom.compass import clock_transport
from atom.compass.clock import NER, ClockAuthority, LpId, single_engine_table
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.streaming_dispatch import StreamBatchDispatcher
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.request import RequestOutput
from atom.utils import clock, get_open_zmq_ipc_path, make_zmq_socket, zmq_shim
from atom.utils.clock import LPRuntime, WrappedSocket
from atom.utils.compass_loop import CompassEventLoop, wrap_decode
from tests.compass.test_tokenizer_station import _entry

with stubbed_aiter():
    from atom.model_engine.engine_core_mgr import CoreManager

IPC = 0.001
TABLE = single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=0.002)
OUT = "engine->frontend:output#dp0"


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def _run(monkeypatch, callback, token_counts) -> None:
    """Send one STREAM message per count at engine time 0, all arriving at IPC,
    to an output thread whose callback is `callback(loop)`; run to +inf."""
    endpoint = f"inproc:test-detok-{uuid.uuid4().hex}"
    server = clock_transport.serve(ClockAuthority(TABLE), endpoint)
    traffic = clock_transport.connect(LpId("traffic"), endpoint)
    traffic.send((NER, math.inf, [], math.inf))
    frontend = LPRuntime(
        LpId("frontend"), TABLE, clock_transport.connect(LpId("frontend"), endpoint)
    )
    monkeypatch.setattr(clock, "_installed", frontend)
    address = get_open_zmq_ipc_path()
    clock.name_endpoints(0, "", "", address)
    ctx, raw = zmq_shim._Context(), zmq.Context()
    loop = CompassEventLoop()
    mgr = CoreManager.__new__(CoreManager)
    mgr.ctx, mgr.label, mgr._lb_lock, mgr._seq_load = ctx, "fe", threading.Lock(), {}
    mgr._seq_id_to_callback = {7: callback(loop)}
    mgr._flush_stream_batch_fn = api_server.flush_stream_batch
    thread = mgr._create_output_thread(
        0, make_zmq_socket(ctx, address, zmq.PULL, bind=True), "inproc://detok-stop"
    )
    thread.start()
    push = raw.socket(zmq.PUSH)
    push.connect(address)
    engine = LPRuntime(LpId("engine"), TABLE, None)
    engine.start_run()
    for n in token_counts:
        out = RequestOutput(7, list(range(n)), False)
        WrappedSocket(engine, push, OUT).send(
            pickle.dumps((EngineCoreRequestType.STREAM, [(7, out)]))
        )
    engine_conn = clock_transport.connect(LpId("engine"), endpoint)
    engine_conn.send((NER, math.inf, engine.send_log, math.inf))
    guard = threading.Timer(20, loop.call_soon_threadsafe, (loop.stop,))
    guard.start()
    frontend.start_run()
    try:
        loop.run_forever()
    finally:
        guard.cancel()
        stop = ctx.socket(zmq.PAIR)
        stop.connect("inproc://detok-stop")
        stop.send(b"")
        thread.join(5)
        loop.close()
        ctx.destroy(linger=0)
        raw.destroy(linger=0)
        server.close()
    assert frontend.now == math.inf


def test_three_messages_on_one_thread_predicted_beside_observed(monkeypatch):
    tokenizer, entry = _Tokenizer(), _entry()
    tokenizer.decode = wrap_decode(tokenizer.decode, entry)
    dispatcher = StreamBatchDispatcher(tokenizer)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", dispatcher)
    state, observed = dispatcher.new_state(), []

    def callback(loop):
        # Each update's in-job clock read, and the loop time it is delivered at.
        collector = SimpleNamespace(
            put_nowait=lambda c: observed.append((c["finished_at"], loop.time()))
        )
        return lambda out: api_server._send_stream_chunk_direct(
            out, "r", collector, loop, state
        )

    _run(monkeypatch, callback, [100, 50, 25])
    # Predicted by hand: one stream; each update decodes the window before it
    # and the window with it, ``fixed + tokens / (rate x derate)`` per call.
    # The three arrive together at IPC and run back to back.
    rate = entry.decode_tokens_per_s * entry.derate
    windows = [(0, 100), (100, 150), (50, 75)]
    start, predicted = IPC, []
    for prefix, full in windows:
        end = start + 2 * entry.decode_fixed_s + (prefix + full) / rate
        predicted.append((start, end))
        start = end
    # (job start, completion): (0.001, 0.203), (0.203, 0.705), (0.705, 0.957)
    assert observed == [pytest.approx(row) for row in predicted]


def test_callbacks_of_jobs_completing_together_run_in_job_order(monkeypatch):
    def callback(loop):
        return lambda out: loop.call_soon_threadsafe(
            seen.append, len(out.output_tokens)
        )

    seen = []
    _run(monkeypatch, callback, [1, 2, 3, 4, 5])
    assert seen == [1, 2, 3, 4, 5]
