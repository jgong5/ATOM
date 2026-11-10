# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A model runner killed with SIGKILL turns the server's ``/health`` to 503
within seconds, though its engine core's cleanup never returns.

Three tiers of real processes stand where a served engine's do. This process is
the frontend: ATOM's ``/health`` handler over a ``CoreManager`` whose output
thread reads the engine core's socket. A spawned child is the engine core:
``EngineCore``'s output thread, and an ``AsyncIOProcManager`` that monitors its
runners and reports a death through ``EngineCore._send_engine_dead``, as the
engine core wires it. Two processes the child spawns stand in for its model
runners. The child's cleanup blocks for good, as it did when one DP rank's
runner died and the other ranks waited in a gloo all-gather, so only a report
sent before cleanup can reach ``/health``.
"""

import multiprocessing
import os
import queue
import signal
import threading
import time
import uuid
from types import SimpleNamespace

import zmq
from aiter_stub import stubbed_aiter
from fastapi.testclient import TestClient

from atom.entrypoints.openai import api_server
from atom.model_engine.engine_core_mgr import CoreManager
from atom.utils import get_open_zmq_ipc_path, make_zmq_socket

with stubbed_aiter():
    from atom.model_engine.async_proc import AsyncIOProcManager
    from atom.model_engine.engine_core import EngineCore

#: The bound from a runner's death to a failing ``/health``.
REPORT_S = 30.0


def _engine_core(output_address: str, runner_pids) -> None:
    """An engine core with two live runners, whose cleanup never returns."""
    ctx = multiprocessing.get_context("spawn")
    runners = [
        ctx.Process(target=time.sleep, args=(600,), name=f"ModelRunner{i}")
        for i in range(2)
    ]
    for runner in runners:
        runner.start()
    core = object.__new__(EngineCore)
    core.label = "EngineCore"
    core.output_queue = queue.Queue()
    core.output_thread = threading.Thread(
        target=core.process_output_sockets, args=(output_address,), daemon=True
    )
    core.output_thread.start()
    mgr = object.__new__(AsyncIOProcManager)
    mgr.procs, mgr._dead_ranks = runners, {}
    mgr.label = mgr.runner_label = "EngineCore"
    mgr.on_worker_failure = core._send_engine_dead
    mgr.exit = threading.Event().wait
    mgr.monitor_procs()
    runner_pids.put([runner.pid for runner in runners])
    threading.Event().wait()


def time_to_failing_health(deadline_s: float = REPORT_S):
    """Kill one runner; return the seconds until ``/health`` first answers other
    than 200 (None if it still answers 200 after `deadline_s`), and whether the
    engine core was still alive then."""
    ctx = multiprocessing.get_context("spawn")
    output_address = get_open_zmq_ipc_path()
    shutdown_path = f"inproc://shutdown-{uuid.uuid4().hex}"
    mgr = CoreManager.__new__(CoreManager)
    mgr._init_shared_state(
        SimpleNamespace(dp_load_balance="round_robin"),
        label="CoreManager",
        local_engine_count=1,
    )
    receiver = make_zmq_socket(mgr.ctx, output_address, zmq.PULL)
    output_thread = mgr._create_output_thread(0, receiver, shutdown_path)
    output_thread.start()
    runner_pids = ctx.Queue()
    engine = ctx.Process(
        target=_engine_core, args=(output_address, runner_pids), name="EngineCore"
    )
    engine.start()
    mgr.engine_core_processes = [engine]
    served, api_server.engine = api_server.engine, SimpleNamespace(core_mgr=mgr)
    pids = []
    try:
        pids = runner_pids.get(timeout=120)
        client = TestClient(api_server.app)
        assert client.get("/health").status_code == 200
        killed_at = time.monotonic()
        os.kill(pids[1], signal.SIGKILL)
        while time.monotonic() - killed_at < deadline_s:
            if client.get("/health").status_code != 200:
                return time.monotonic() - killed_at, engine.is_alive()
            time.sleep(0.005)
        return None, engine.is_alive()
    finally:
        api_server.engine = served
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        engine.kill()
        engine.join(timeout=10)
        if output_thread.is_alive():
            stop = mgr.ctx.socket(zmq.PAIR)
            stop.connect(shutdown_path)
            stop.send(b"")
            stop.close(linger=0)
        output_thread.join(timeout=10)
        mgr.ctx.term()


def test_a_sigkilled_runner_fails_health_within_seconds_while_cleanup_blocks():
    seconds, engine_alive = time_to_failing_health()
    assert seconds is not None, f"/health still answers 200 {REPORT_S:g} s on"
    assert engine_alive, "the engine core's exit, not the report, failed /health"
