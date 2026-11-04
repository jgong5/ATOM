# SPDX-License-Identifier: MIT
"""A simulated run's workers sleep between their engine's calls, and take each.

The ring is aiter's own, written by `RingingQueue` and read by the reader that
`MessageQueue.create_from_handle` opens, as a worker's is; the loop is ATOM's
`AsyncIOProc.busy_loop`, run on a worker given only what it reads. The manager
is ATOM's `AsyncIOProcManager`, built inside the run's engine bootstrap with no
worker process started.
"""

import json
import queue
import signal
import socket
import threading
import time
from types import SimpleNamespace

import torch.distributed as dist
from aiter.dist.shm_broadcast import MessageQueue

from atom.compass import run as compass_run
from atom.compass.runner.doorbell import RingingQueue, SleepingWorker
from atom.model_engine import async_proc
from atom.model_engine.block_table_codec import BlockTableDeltaDecoder
from atom.utils import clock
from atom.utils.distributed import utils

IDLE_S = 0.5


class Runner:
    """Takes the calls, noting the worker thread's CPU seconds at each."""

    def __init__(self):
        self.calls = []

    def record(self, n):
        self.calls.append((n, time.thread_time()))

    def exit(self):
        pass


def _worker(writer):
    """A worker reading `writer`'s ring as its reader 0, given only what it reads."""
    worker = object.__new__(SleepingWorker)
    worker.label, worker.runners, worker.io_threads = "worker", [Runner()], []
    worker.rpc_broadcast_mq = MessageQueue.create_from_handle(writer.export_handle(), 0)
    worker._block_table_decoder = BlockTableDeltaDecoder()
    worker.io_addrs, worker.io_queues = (None, None), (queue.Queue(), queue.Queue())
    worker.kv_queue = worker.rpc_queue = worker.all_ranks_barrier = None
    return worker


def _loop(worker):
    try:
        worker.busy_loop()
    except TypeError:  # the ring `exit` closed, as it ends a spin
        pass


def _close(writer, worker):
    worker.doorbell.socket.close()
    worker.rpc_broadcast_mq.close()
    writer.close()


def test_a_worker_sleeps_between_calls_and_takes_each_in_order():
    writer = RingingQueue(1, 1, max_chunk_bytes=1 << 16)
    worker = _worker(writer)
    thread = threading.Thread(target=_loop, args=(worker,), daemon=True)
    thread.start()
    try:
        writer.enqueue(("record", 0))
        time.sleep(IDLE_S)  # the worker waits for its next call
        writer.enqueue(("record", 1))
        writer.enqueue(("exit",))
        thread.join(10)
        calls, alive = list(worker.runners[0].calls), thread.is_alive()
    finally:
        # A worker still asleep holds its ring's buffer, which then cannot
        # close: the doorbell shut wakes it to the calls left, and it leaves.
        worker.doorbell.socket.shutdown(socket.SHUT_RD)
        thread.join(10)
        _close(writer, worker)
    assert ([n for n, _ in calls], alive) == ([0, 1], False)
    (_, before), (_, after) = calls
    assert after - before < IDLE_S / 10, f"waiting took {after - before:.3f} CPU s"


def test_a_call_enqueued_before_the_worker_binds_its_doorbell_is_taken():
    """An engine enqueues its first call while its workers still build their
    runners: the ring finds no doorbell bound, and the worker, which binds
    it before its first look at the ring, still takes the call."""
    writer = RingingQueue(1, 1, max_chunk_bytes=1 << 16)
    worker = _worker(writer)
    thread = threading.Thread(target=_loop, args=(worker,), daemon=True)
    try:
        writer.enqueue(("record", 0))
        thread.start()
        writer.enqueue(("exit",))
        thread.join(10)
        calls, alive = list(worker.runners[0].calls), thread.is_alive()
    finally:
        if thread.is_alive():  # asleep: the doorbell shut wakes it to the exit
            worker.doorbell.socket.shutdown(socket.SHUT_RD)
            thread.join(10)
        if thread.ident is None:  # the first ring raised: no doorbell is bound
            worker.rpc_broadcast_mq.close()
            writer.close()
        else:
            _close(writer, worker)
    assert ([n for n, _ in calls], alive) == ([0], False)


def test_a_worker_signalled_while_it_sleeps_leaves_its_loop():
    """A signal whose handler runs `exit`, as the one `init_exit_handler`
    installs does, ends the loop of a worker asleep on its doorbell."""
    writer = RingingQueue(1, 1, max_chunk_bytes=1 << 16)
    worker = _worker(writer)
    main = threading.main_thread().ident
    before = signal.signal(signal.SIGUSR1, lambda *_: worker.exit())
    signalled = threading.Timer(IDLE_S, signal.pthread_kill, (main, signal.SIGUSR1))
    # A worker the signal leaves asleep wakes to the ring `exit` closed.
    rung = threading.Timer(10 * IDLE_S, writer.enqueue, (("exit",),))
    signalled.start()
    rung.start()
    started = time.monotonic()
    try:
        _loop(worker)
    finally:
        left = time.monotonic() - started
        rung.cancel()
        signalled.join()
        rung.join()
        signal.signal(signal.SIGUSR1, before)
        _close(writer, worker)
    assert left < 5 * IDLE_S, f"the worker left its loop {left:.1f} s after it began"


def test_an_engine_built_in_a_simulated_run_rings_workers_that_sleep(
    monkeypatch, tmp_path
):
    run = {
        "admission_path": "serving",
        "ipc_s": 2.0**-10,
        "stream_s": 2.0**-9,
        "bound_s": 100.0,
        "clock_endpoint": f"inproc:{tmp_path}",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    for module, name in [(clock, "_installed"), (utils, "LP_OF_RANK")]:
        monkeypatch.setattr(module, name, None, raising=False)
    for name in clock.COLLECTIVES:  # the engine's bootstrap wraps each
        monkeypatch.setattr(dist, name, getattr(dist, name))
    monkeypatch.setattr(compass_run, "_member", None)
    monkeypatch.setattr(compass_run, "connect", lambda lp, endpoint, member: None)
    # ATOM's manager constructor, with no worker process, socket thread or signal.
    targets = []
    process = SimpleNamespace(start=lambda: None)
    ctx = SimpleNamespace(
        Barrier=lambda n: None,
        Process=lambda target, **kw: targets.append(target) or process,
    )
    monkeypatch.setattr(async_proc, "get_mp_context", lambda: ctx)
    monkeypatch.setattr(async_proc, "init_exit_handler", lambda self: None)
    for name in (
        "process_output_sockets",
        "process_kv_output_sockets",
        "process_rpc_output_sockets",
        "monitor_procs",
    ):
        monkeypatch.setattr(async_proc.AsyncIOProcManager, name, lambda *args: None)
    rank = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
    config = SimpleNamespace(parallel_config=rank, kv_transfer_config={})
    with compass_run.engine(config):
        manager = async_proc.AsyncIOProcManager(None, 2, "workers.Runner")
    ring = manager.rpc_broadcast_mq
    try:
        assert type(ring) is RingingQueue
        assert (len(ring.ringers), targets) == (2, [SleepingWorker, SleepingWorker])
        # Outside the engine's construction, a manager built takes ATOM's own.
        assert async_proc.MessageQueue is MessageQueue
        assert async_proc.AsyncIOProc is SleepingWorker.__base__
    finally:
        ring.close()
        manager.zmq_ctx.term()
