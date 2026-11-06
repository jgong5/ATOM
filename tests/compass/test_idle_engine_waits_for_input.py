# SPDX-License-Identifier: MIT
"""An idle engine parks on its input queue instead of spinning, and each kind of
work its busy loop serves wakes it: a request, a utility command (an abort is
one), a shutdown, the metrics push and pending KV work. A request or utility
command that lands just before the wait, or while it checks for one, is not
waited out. A simulated run never parks there.

ATOM's `EngineCore.busy_loop` and input thread (`process_input_sockets`) run on
the real clock over a fake wire, with ATOM's utility handler and a scheduler
that has nothing to run. The metrics interval is long, so a wake that is
missing shows as work taken a whole interval late.
"""

import contextlib
import pickle
import queue
import statistics
import threading
import time
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.sequence import Sequence
from atom.utils import clock, zmq_shim

with stubbed_aiter():
    from atom.model_engine import engine_core
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.engine_utility import EngineUtilityHandler

QUIET_S = 30.0  # the metrics interval: no timeout ends a wait inside a test
PROMPT_S = 1.0  # a woken loop takes its work well inside this
SERVED = ("served input", "served loop")  # `_serving`'s thread names


@pytest.fixture(autouse=True)
def leaves_nothing_behind():
    """After each test, monkeypatch undone: no served thread is alive, and
    `zmq_shim` still picks `Context`, `Poller` and `Socket` on each read. A
    later test in this process installs an LP runtime and needs the shim's
    `Poller`; a pyzmq one fixed on the shim never reports its requests."""
    yield
    threads = [t.name for t in threading.enumerate() if t.name in SERVED]
    fixed = [n for n in ("Context", "Poller", "Socket") if n in vars(zmq_shim)]
    assert not threads and not fixed, (threads, fixed)


class Wire:
    """The input thread's two sockets and their poller: `poll` parks until the
    test sends a frame, as zmq's does, and `recv` returns it."""

    def __init__(self):
        self.frames, self.frame = queue.Queue(), None

    def deliver(self, frame) -> float:
        """Hand the input thread `frame`; returns when it was sent."""
        sent = time.perf_counter()
        self.frames.put(pickle.dumps(frame))
        return sent

    def send(self, _):  # the input thread's greeting on each socket
        pass

    def register(self, *_):
        pass

    def poll(self):
        self.frame = self.frames.get()
        return [(self, engine_core.zmq.POLLIN)]

    def recv(self, copy=True):
        return self.frame


class Zmq:
    """`engine_core`'s `zmq` with the wire as its `Poller`. Set on
    `engine_core`, not on the shim, whose `Poller` monkeypatch would undo by
    fixing there the one it read."""

    def __init__(self, wire):
        self.Poller = lambda: wire

    def __getattr__(self, name):
        return getattr(zmq_shim, name)


class Scheduler:
    """Nothing to run. Notes each request and abort the loop hands it, and its
    KV work stays pending until `kv_done` is set."""

    prefill_delayer = None

    def __init__(self):
        self.got, self.passes = queue.Queue(), 0
        self.deferred_free_blocks, self.kv_done = [], threading.Event()

    def is_finished(self):
        self.passes += 1
        return True

    def extend(self, seqs):
        for seq in seqs:
            self.got.put(("request", seq))

    def abort_request(self, req_id):
        self.got.put(("abort", req_id))
        return False

    def _update_from_kv_xfer_finished(self, output):
        if self.kv_done.is_set() and self.deferred_free_blocks:
            self.deferred_free_blocks.clear()
            self.got.put(("kv", output))

    def heartbeat_throughput(self, now):
        pass

    def publish_kv_events(self):
        pass

    def shutdown_kv_events(self):
        pass


def _engine(kv: bool = False, cls=EngineCore) -> EngineCore:
    engine = cls.__new__(cls)
    engine.label, engine.scheduler = "engine", Scheduler()
    engine.input_queue, engine.utility_queue = queue.Queue(), queue.Queue()
    engine.output_queue = queue.Queue()
    engine._has_pending_utility = engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced, engine._next_idle_kv_drain = False, 0.0
    engine.kv_transfer_enabled = kv
    engine.runner_mgr = SimpleNamespace(
        call_func_with_aggregation=lambda name: SimpleNamespace(name=name)
    )
    engine.utility_handler = EngineUtilityHandler(
        engine.runner_mgr, engine.output_queue, scheduler=engine.scheduler
    )
    engine.pushes = []
    engine.utility_handler.push_metrics = lambda: engine.pushes.append(time.monotonic())
    return engine


@contextlib.contextmanager
def _serving(engine, monkeypatch, interval=QUIET_S):
    """Run the input thread and the busy loop; yield the wire and the loop's
    thread. On the way out a shutdown frame ends both."""
    monkeypatch.setenv("ATOM_METRICS_UPDATE_INTERVAL_S", str(interval))
    wire = Wire()
    monkeypatch.setattr(
        engine_core, "make_zmq_socket", lambda *a, **k: contextlib.nullcontext(wire)
    )
    monkeypatch.setattr(engine_core, "zmq", Zmq(wire))
    threads = [
        threading.Thread(
            target=engine.process_input_sockets, args=("in", "ctl"), name=SERVED[0]
        ),
        threading.Thread(target=EngineCore.busy_loop, args=(engine,), name=SERVED[1]),
    ]
    for t in threads:
        t.daemon = True
        t.start()
    try:
        yield wire, threads[1]
    finally:
        wire.deliver((EngineCoreRequestType.SHUTDOWN, None))
        for t in threads:
            t.join(QUIET_S + PROMPT_S)


def _taken(engine):
    try:
        return engine.scheduler.got.get(timeout=PROMPT_S)
    except queue.Empty:
        pytest.fail(f"the idle loop took nothing within {PROMPT_S} s")


def test_an_idle_engine_parks_instead_of_spinning(monkeypatch):
    engine = _engine()
    with _serving(engine, monkeypatch):
        time.sleep(0.3)
        passes = engine.scheduler.passes
    assert passes <= 2, f"{passes} passes over an idle 0.3 s"


def test_a_request_reaching_an_idle_engine_is_received_at_once(monkeypatch):
    engine = _engine()
    late = []
    with _serving(engine, monkeypatch) as (wire, _):
        for _ in range(10):
            time.sleep(0.03)
            seq = Sequence([1, 2, 3], block_size=4)
            sent = wire.deliver((EngineCoreRequestType.ADD, [seq]))
            kind, got = _taken(engine)
            assert (kind, got.id) == ("request", seq.id)
            late.append(got.queue_timing.received_at - sent)
    assert statistics.median(late) < 0.001, late


def test_a_utility_command_wakes_an_idle_engine(monkeypatch):
    engine = _engine()
    abort = {"cmd": "abort_request", "req_id": 7}
    with _serving(engine, monkeypatch) as (wire, _):
        time.sleep(0.1)
        wire.deliver((EngineCoreRequestType.UTILITY, abort))
        assert _taken(engine) == ("abort", 7)


def test_a_shutdown_wakes_an_idle_engine(monkeypatch):
    engine = _engine()
    with _serving(engine, monkeypatch) as (wire, loop):
        time.sleep(0.1)
        wire.deliver((EngineCoreRequestType.SHUTDOWN, None))
        loop.join(PROMPT_S)
        assert not loop.is_alive()


def test_an_idle_engine_pushes_its_metrics_on_time(monkeypatch):
    engine = _engine()
    with _serving(engine, monkeypatch, interval=0.1):
        time.sleep(0.55)
    gaps = [b - a for a, b in zip(engine.pushes, engine.pushes[1:])]
    assert len(engine.pushes) >= 4 and max(gaps) < 0.2, engine.pushes


def test_pending_kv_work_is_polled_while_the_engine_is_idle(monkeypatch):
    engine = _engine(kv=True)
    engine.scheduler.deferred_free_blocks.append(1)
    with _serving(engine, monkeypatch):
        time.sleep(0.1)
        engine.scheduler.kv_done.set()
        assert _taken(engine)[0] == "kv"


def _parks(engine) -> bool:
    """Whether `_wait_for_input`, handed a deadline no test reaches, is still
    waiting after `PROMPT_S`; one that is gets woken."""
    waiter = threading.Thread(
        target=engine._wait_for_input, args=(time.monotonic() + QUIET_S,), daemon=True
    )
    waiter.start()
    waiter.join(PROMPT_S)
    parked = waiter.is_alive()
    with engine.input_queue.not_empty:
        engine.input_queue.not_empty.notify_all()
    return parked


def test_a_simulated_engine_never_parks(monkeypatch):
    monkeypatch.setattr(clock, "_installed", SimpleNamespace(read_clock=time.monotonic))
    assert not _parks(_engine())


def test_a_request_queued_before_the_wait_is_not_waited_out():
    engine = _engine()
    engine.input_queue.put_nowait([])
    assert not _parks(engine)


def test_a_utility_command_queued_before_the_wait_is_not_waited_out():
    engine = _engine()
    engine.utility_queue.put_nowait(("abort_request", {"req_id": 7}))
    engine._has_pending_utility = True  # as the input thread leaves it
    assert not _parks(engine)


def _arriving(arrive) -> None:
    """Run `arrive`, the input thread's side, on another thread, waiting up to
    0.2 s for it: while the wait's caller holds the input queue's lock, it
    cannot finish until the wait releases the lock."""
    thread = threading.Thread(target=arrive, daemon=True)
    thread.start()
    thread.join(0.2)


class RacingQueue(queue.Queue):
    """An input queue whose first size read sees a request put as it reads."""

    raced = False

    def _qsize(self):
        n = super()._qsize()
        if not self.raced:
            self.raced = True
            _arriving(lambda: self.put_nowait([]))
        return n


class RacingEngine(EngineCore):
    """An engine whose first read of its utility flag sees a command put,
    flagged and notified as it reads, as the input thread does."""

    raced = False

    @property
    def _has_pending_utility(self):
        pending = self.__dict__.get("_pending", False)
        if not self.raced:
            self.raced = True
            _arriving(self._arrive)
        return pending

    @_has_pending_utility.setter
    def _has_pending_utility(self, value):
        self.__dict__["_pending"] = value

    def _arrive(self):
        self.utility_queue.put_nowait(("abort_request", {"req_id": 7}))
        self._has_pending_utility = True
        self._notify_input()


def test_a_request_racing_the_check_is_not_lost():
    engine = _engine()
    engine.input_queue = RacingQueue()
    assert not _parks(engine)


def test_a_utility_command_racing_the_check_is_not_lost():
    assert not _parks(_engine(cls=RacingEngine))
