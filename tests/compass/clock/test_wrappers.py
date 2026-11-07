# SPDX-License-Identifier: MIT
"""`WrappedSocket`, `WrappedPoller` and `RelayQueue` over real zmq ``inproc`` pairs.

Each test builds the socket pair ATOM uses on that channel: ROUTER to DEALER for
requests, PUSH to PULL for the engine's output. The clock authority is the
scripted fake connection of the LP runtime tests, and each LP's clock owner is
a real thread, so a release that never completes fails the test instead of
hanging it.
"""

import math
import re
import threading
import time
from types import SimpleNamespace

import pytest
import zmq

from atom.compass.clock import LpId, prefill_decode_table, single_engine_table
from atom.utils.clock import (
    LPRuntime,
    RelayQueue,
    UnsentRelayItem,
    WrappedPoller,
    WrappedSocket,
    _Wakeup,
)

from .test_lp_runtime import ENGINE, FRONTEND, IPC, OUT, REQ, FakeConn, _owner, _table


@pytest.fixture
def ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)


def _pair(ctx, name, send_type, recv_type):
    tx, rx = ctx.socket(send_type), ctx.socket(recv_type)
    rx.bind(f"inproc://{name}")
    tx.connect(f"inproc://{name}")
    return tx, rx


def _rt(lp, *replies, table=None):
    thread, call = _owner()
    conn = FakeConn(*replies)
    rt = LPRuntime(lp, table or _table(), conn, owner=thread)
    rt.start_run()
    return rt, call


def _requests(rt):
    """Each clock request's kind, time and send log; later fields are not checked."""
    return [m[:3] for m in rt.conn.sent]


def _relayed(rt, raw, ch, *stamps):
    """A sending wrapper whose stamps are scripted, as a relay would hand them over."""
    ws = WrappedSocket(rt, raw, ch)
    ws.relay = SimpleNamespace(take_stamp=iter(stamps).__next__)
    return ws


def test_out_of_order_frames_are_handed_over_in_arrival_order_and_counted_by_seq(ctx):
    """One channel, seq 1 physically delivered before seq 0."""
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(ENGINE)[0], push, OUT, (10.1, 1), (10.05, 0))
    rt, call = _rt(FRONTEND, (10.2, {OUT: [(0, 10.05), (1, 10.1)]}))
    rx = WrappedSocket(rt, pull, OUT)
    stop_tx, stop_rx = _pair(ctx, "stop", zmq.PAIR, zmq.PAIR)
    seen = []

    def output_thread():
        poller = WrappedPoller(rt)
        poller.register(stop_rx, zmq.POLLIN)
        poller.register(rx, zmq.POLLIN)
        while True:
            socks = poller.poll()
            if socks[0][0] is stop_rx:
                return
            seen.append((rx.recv(copy=False).bytes, rt.read_clock()))

    thread = threading.Thread(target=output_thread, daemon=True)
    thread.start()
    tx.send(b"p1")
    tx.send(b"p0")
    call(rt.advance_to, 10.2)
    assert seen == [(b"p0", 10.05), (b"p1", 10.1)]
    assert rt.handled[OUT] == {0, 1}
    assert rt.now == 10.2
    stop_tx.send(b"")
    thread.join(5)
    assert not thread.is_alive()


def test_a_release_wakes_only_the_pollers_waiting_on_its_channel(ctx):
    """A frontend's output threads, one per DP rank: a release on one rank's
    channel sets no other rank's wakeup, and one no poller registered, as
    lmcache's polling thread adds, is set by every release."""
    table = single_engine_table(
        admission_path="serving", ipc_s=IPC, stream_s=0.002, dp=2
    )
    outs = ("engine->frontend:output#dp0", "engine->frontend:output#dp1")
    grants = (0.2, {outs[0]: [(0, 0.1)]}), (0.4, {outs[1]: [(0, 0.3)]})
    rt, call = _rt(FRONTEND, *grants, table=table)
    engine = _rt(ENGINE, table=table)[0]
    txs, stops, threads, seen, woken = [], [], [], [], []
    for r, (ch, arrival) in enumerate(zip(outs, (0.1, 0.3))):
        push, pull = _pair(ctx, f"out{r}", zmq.PUSH, zmq.PULL)
        txs.append(_relayed(engine, push, ch, (arrival, 0)))
        stop_tx, stop_rx = _pair(ctx, f"stop{r}", zmq.PAIR, zmq.PAIR)
        stops.append(stop_tx)
        poller = WrappedPoller(rt, [stop_rx, WrappedSocket(rt, pull, ch)])

        def output_thread(poller=poller):
            stop_rx, rx = poller.socks
            while poller.poll()[0][0] is not stop_rx:
                seen.append((rx.ch, rx.recv(), rt.read_clock()))

        threads.append(threading.Thread(target=output_thread, daemon=True))
        threads[-1].start()
    deadline = time.monotonic() + 5
    while not all(t in rt.wakes for t in threads):  # until both pollers wait
        assert time.monotonic() < deadline
        time.sleep(0.01)

    def counted(r, set_):
        return lambda: (woken.append(r), set_())

    for r, t in enumerate(threads):
        rt.wakes[t].set = counted(r, rt.wakes[t].set)
    plain = _Wakeup()
    rt.wakes[plain] = plain

    txs[0].send(b"d0")
    call(rt.advance_to, 0.2)
    assert woken == [0] and plain.is_set()
    plain.clear()
    txs[1].send(b"d1")
    call(rt.advance_to, 0.4)
    assert woken == [0, 1] and plain.is_set()
    assert seen == [(outs[0], b"d0", 0.1), (outs[1], b"d1", 0.3)]
    for stop, t in zip(stops, threads):
        stop.send(b"")
        t.join(5)
        assert not t.is_alive()


def test_the_relay_sends_every_put_stamped_at_its_put(ctx):
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    rt, call = _rt(ENGINE, (0.5, {}), (0.7, {}))
    tx = WrappedSocket(rt, push, OUT)
    relay = RelayQueue(rt, tx)
    rt.end_run()
    relay.put(b"ready")  # outside the run: any thread, nothing stamped
    tx.send(relay.get())
    rt.start_run()
    call(relay.put, b"a")
    call(rt.advance_to, 0.5)
    call(relay.put_nowait, b"b")
    call(rt.advance_to, 0.7)
    call(relay.put, b"c")
    for _ in range(3):
        tx.send(relay.get())
    with pytest.raises(RuntimeError, match=f"^{OUT}: one relay item sent twice"):
        tx.send(b"c")

    rx_rt, _ = _rt(FRONTEND)
    rx = WrappedSocket(rx_rt, pull, OUT)
    while len(rx.buf) < 4:
        assert pull.poll(5000)
        rx.pull()
    assert [f[3].bytes for f in rx.buf] == [b"ready", b"a", b"b", b"c"]
    stamps = {(OUT, 0): IPC, (OUT, 1): 0.5 + IPC, (OUT, 2): 0.7 + IPC}
    assert rx_rt.unreleased == stamps
    assert _requests(rt) == [
        ("TAR", 0.5, [(OUT, 0, IPC)]),
        ("TAR", 0.7, [(OUT, 1, 0.5 + IPC)]),
    ]

    call(relay.put, b"d")
    call(relay.put, b"e")
    relay.get()  # taken, never sent
    unsent = f"{OUT}: the item stamped (arrival {0.7 + IPC}, seq 3) was taken"
    with pytest.raises(UnsentRelayItem, match="^" + re.escape(unsent)):
        relay.get()
    with pytest.raises(RuntimeError, match="^stamp_send from thread 'MainThread'"):
        relay.put(b"f")

    refused = []
    second = threading.Thread(target=lambda: refused.append(_raised(relay.get)))
    second.start()
    second.join(5)
    assert "the relay's one consumer is 'MainThread'" in refused[0]


def _raised(fn):
    try:
        fn()
    except RuntimeError as e:
        return str(e)


def test_a_router_puts_the_header_after_the_identity_frame(ctx):
    router, dealer = _pair(ctx, "req", zmq.ROUTER, zmq.DEALER)
    dealer.send(b"")
    identity, _ = router.recv_multipart()
    tx_rt, call = _rt(FRONTEND)
    tx = WrappedSocket(tx_rt, router, REQ)
    rt, _ = _rt(ENGINE)
    rx = WrappedSocket(rt, dealer, REQ)

    tx_rt.end_run()
    call(lambda: tx.send_multipart([identity, b"ready"], copy=False))
    assert dealer.poll(5000)
    rx.pull()
    assert rx.ready() and rt.arrived[REQ] == set()
    assert rx.recv() == b"ready"

    tx_rt.start_run()
    call(lambda: tx.send_multipart([identity, b"req"], copy=False))
    assert dealer.poll(5000)
    rx.pull()
    assert rt.unreleased == {(REQ, 0): IPC}
    assert [f[3].bytes for f in rx.buf] == [b"req"]


def test_a_duplicate_seq_is_refused_on_read(ctx):
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(ENGINE)[0], push, OUT, (1.0, 3), (1.0, 3))
    rx = WrappedSocket(_rt(FRONTEND)[0], pull, OUT)
    tx.send(b"x")
    tx.send(b"x")
    with pytest.raises(RuntimeError, match=f"^{OUT} seq 3 arrived twice"):
        while len(rx.buf) < 2:
            assert pull.poll(5000)
            rx.pull()


def test_holds_unstamped_is_asked_only_while_an_unstamped_frame_is_buffered(ctx):
    """`holds_unstamped` reads every frame the LP's thread channels brought in
    the run. An output thread taking stamped frames, which their release alone
    decides, never asks it. The sender's unstamped stop, buffered beside the
    unreleased frame 2, is asked about and held until frame 2 is handled."""
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    tx_rt = _rt(ENGINE)[0]
    tx = _relayed(tx_rt, push, OUT, (10.1, 1), (10.05, 0), (10.3, 2))
    releases = (10.2, {OUT: [(0, 10.05), (1, 10.1)]}), (10.4, {OUT: [(2, 10.3)]})
    rt, call = _rt(FRONTEND, *releases)
    asked = []

    def holds_unstamped(real=rt.holds_unstamped):
        asked.append(real())
        return asked[-1]

    rt.holds_unstamped = holds_unstamped
    rx = WrappedSocket(rt, pull, OUT)
    seen = []

    def output_thread():
        poller = WrappedPoller(rt, [rx])
        while b"stop" not in seen:
            poller.poll()
            seen.append(rx.recv())

    thread = threading.Thread(target=output_thread, daemon=True)
    thread.start()
    tx.send(b"p1")
    tx.send(b"p0")
    call(rt.advance_to, 10.2)
    assert seen == [b"p0", b"p1"] and asked == []
    tx.send(b"p2")
    tx_rt.end_run()
    tx.send(b"stop")
    deadline = time.monotonic() + 5
    while True not in asked:
        assert time.monotonic() < deadline, "the stop was never held"
        time.sleep(0.01)
    call(rt.advance_to, 10.4)
    thread.join(5)
    assert seen == [b"p0", b"p1", b"p2", b"stop"] and not thread.is_alive()


def _output_thread(rt, rx, stop, seen, handle=lambda frame: None):
    """ATOM's output thread: wait on its shutdown socket and `rx`, take and
    handle each frame, until the shutdown is among the sockets ready."""

    def run():
        poller = WrappedPoller(rt)
        poller.register(stop, zmq.POLLIN)
        poller.register(rx, zmq.POLLIN)
        while True:
            socks = poller.poll()
            if len(socks) == 2 or socks[0][0] is stop:
                return
            seen.append(rx.recv())
            handle(seen[-1])

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _shut(stop_tx, thread):
    stop_tx.send(b"")
    thread.join(5)
    assert not thread.is_alive()


def test_a_frame_counts_as_handled_once_its_thread_has_read_what_came_meanwhile(ctx):
    """Frame 1 reaches the socket while the output thread handles frame 0. The
    thread is back at its wait point, and frame 0 handled, only once it has
    read frame 1: the owner it lets go finds nothing left on the socket."""
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    late = ctx.socket(zmq.PUSH)  # frame 1's sender, used on the output thread only
    late.connect("inproc://out")
    engine = _rt(ENGINE)[0]
    tx = _relayed(engine, push, OUT, (10.05, 0))
    tx_late = _relayed(engine, late, OUT, (10.1, 1))
    rt, call = _rt(FRONTEND, (10.2, {OUT: [(0, 10.05), (1, 10.1)]}))
    read_when_handled = []

    def count_done_locked(ch, seq, real=rt.count_done_locked):
        read_when_handled.append((seq, sorted(rt.arrived[ch])))
        real(ch, seq)

    def handle(frame):
        if frame == b"p0":
            tx_late.send(b"p1")
            assert pull.poll(5000)  # on the socket, not yet read

    rt.count_done_locked = count_done_locked
    stop_tx, stop_rx = _pair(ctx, "stop", zmq.PAIR, zmq.PAIR)
    thread = _output_thread(rt, WrappedSocket(rt, pull, OUT), stop_rx, [], handle)
    tx.send(b"p0")
    call(rt.advance_to, 10.2)
    assert read_when_handled == [(0, [0, 1]), (1, [0, 1])]
    _shut(stop_tx, thread)


class _Reads:
    """A socket that counts its reads that found nothing (`zmq.Again`)."""

    def __init__(self, raw) -> None:
        self.raw, self.empty = raw, 0

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def recv_multipart(self, *args, **kw):
        try:
            return self.raw.recv_multipart(*args, **kw)
        except zmq.Again:
            self.empty += 1
            raise


def test_reading_a_socket_stops_at_its_last_frame_without_a_failed_read(ctx):
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(ENGINE)[0], push, OUT, (0.1, 0), (0.2, 1))
    reads = _Reads(pull)
    rx = WrappedSocket(_rt(FRONTEND)[0], reads, OUT)
    rx.pull()
    tx.send(b"p0")
    tx.send(b"p1")
    assert pull.poll(5000)
    rx.pull()
    rx.pull()
    assert [f[3].bytes for f in rx.buf] == [b"p0", b"p1"]
    assert reads.empty == 0


class _Shutdown:
    """ATOM's shutdown socket as its output thread registers it: no channel's,
    the shim's wrapper around a pyzmq socket, here counting its own polls."""

    def __init__(self, raw) -> None:
        self.raw, self.polls = raw, 0

    def poll(self, *args):
        self.polls += 1
        return self.raw.poll(*args)


def test_a_shutdown_socket_counts_once_the_wait_reports_it_and_is_never_polled(ctx):
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    stamps = [(0.1 * (k + 1), k) for k in range(3)]
    tx = _relayed(_rt(ENGINE)[0], push, OUT, *stamps)
    rt, call = _rt(FRONTEND, (1.0, {OUT: [(k, a) for a, k in stamps]}))
    stop_tx, stop_raw = _pair(ctx, "stop", zmq.PAIR, zmq.PAIR)
    stop, seen = _Shutdown(stop_raw), []
    thread = _output_thread(rt, WrappedSocket(rt, pull, OUT), stop, seen)
    for k in range(3):
        tx.send(b"p%d" % k)
    call(rt.advance_to, 1.0)
    assert seen == [b"p0", b"p1", b"p2"]
    _shut(stop_tx, thread)
    assert stop.polls == 0


def test_a_release_sets_no_wakeup_of_the_clock_owner(ctx):
    """Before the run the frontend's owner waits for an engine's first frame
    through the poller, as `CoreManager` does. In the run only the output
    thread's wakeup is set by a release: the owner makes every release itself."""
    push, pull = _pair(ctx, "out", zmq.PUSH, zmq.PULL)
    engine = _rt(ENGINE)[0]
    engine.end_run()
    WrappedSocket(engine, push, OUT).send(b"ready")
    engine.start_run()
    rt, call = _rt(FRONTEND, (0.2, {OUT: [(0, 0.1)]}))
    rt.end_run()
    rx = WrappedSocket(rt, pull, OUT)

    def first_frame():
        WrappedPoller(rt, [rx]).poll()
        return rx.recv()

    assert call(first_frame) == b"ready"
    rt.start_run()
    stop_tx, stop_rx = _pair(ctx, "stop", zmq.PAIR, zmq.PAIR)
    seen = []
    thread = _output_thread(rt, rx, stop_rx, seen)
    _relayed(engine, push, OUT, (0.1, 0)).send(b"d0")
    call(rt.advance_to, 0.2)
    assert seen == [b"d0"] and not rt.wakes[rt.owner].is_set()
    _shut(stop_tx, thread)


KV = "engine-D->engine-P:kv_write_req"


def _pd_table():
    return prefill_decode_table(
        admission_path="serving",
        ipc_s=IPC,
        stream_s=0.002,
        router_s=0.001,
        kv_write_req_s=0.5,
    )


def test_inline_receive_idles_until_the_frame_is_released(ctx):
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(LpId("engine-D"), table=table)[0], push, KV, (0.6, 1), (0.5, 0))
    tx.send(b"p1")
    tx.send(b"p0")
    grant = (0.6, {KV: [(0, 0.5), (1, 0.6)]})
    rt, call = _rt(LpId("engine-P"), (0.1, {}), (0.3, {}), grant, table=table)
    rx = WrappedSocket(rt, pull, KV)

    while len(rx.buf) < 2:
        assert pull.poll(5000)
        rx.pull()
    assert call(rx.poll, 0) is False
    assert rt.conn.sent == []
    assert call(rx.poll, 100) is False
    assert rt.conn.sent == [("NER", math.inf, [], 0.1)]
    assert call(rx.recv) == b"p0"
    assert _requests(rt)[1:] == [("NER", math.inf, [])] * 2
    assert call(rx.recv) == b"p1"
    assert len(rt.conn.sent) == 3
    assert rt.handled[KV] == {0, 1} and rt.now == 0.6


def test_settle_waits_for_a_released_frame_still_in_flight(ctx):
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    tx = _relayed(_rt(LpId("engine-D"), table=table)[0], push, KV, (0.5, 0))
    rt, call = _rt(LpId("engine-P"), (0.5, {KV: [(0, 0.5)]}), table=table)
    rx = WrappedSocket(rt, pull, KV)
    call(rt.next_event, math.inf)  # released before it is sent
    settling = threading.Thread(target=rx.settle, daemon=True)
    settling.start()
    settling.join(0.2)
    assert settling.is_alive()
    tx.send(b"p0")
    settling.join(5)
    assert not settling.is_alive() and rt.arrived[KV] == {0}


def test_inline_receive_takes_a_frame_released_while_still_in_flight(ctx):
    """Released by the receive's own grant, or by an earlier `advance_to`, then sent."""
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    stamps = (0.5, 0), (0.55, 1), (0.7, 2)
    tx = _relayed(_rt(LpId("engine-D"), table=table)[0], push, KV, *stamps)
    grants = [(a, {KV: [(seq, a)]}) for a, seq in stamps]
    rt, call = _rt(LpId("engine-P"), *grants, table=table)
    rx = WrappedSocket(rt, pull, KV)

    def send_later(payload):
        threading.Timer(0.2, lambda: tx.send(payload)).start()

    send_later(b"p0")
    assert call(rx.recv) == b"p0"
    send_later(b"p1")
    assert call(rx.poll, 100) is True
    assert call(rx.recv) == b"p1"
    call(rt.advance_to, 0.7)
    send_later(b"p2")
    assert call(rx.recv) == b"p2"
    assert _requests(rt) == [("NER", math.inf, [])] * 2 + [("TAR", 0.7, [])]
    assert rt.conn.sent[1][3] == 0.6


def test_outside_the_run_an_inline_receive_waits_on_the_socket(ctx):
    """Before the run and after the +inf grant, as ATOM's loops poll until shutdown."""
    table = _pd_table()
    push, pull = _pair(ctx, "kv", zmq.PUSH, zmq.PULL)
    tx_rt, _ = _rt(LpId("engine-D"), table=table)
    tx_rt.end_run()
    tx = WrappedSocket(tx_rt, push, KV)
    rt, call = _rt(LpId("engine-P"), (math.inf, {}), table=table)
    rt.end_run()
    rx = WrappedSocket(rt, pull, KV)

    def outside_the_run(asked):
        start = time.monotonic()
        assert call(rx.poll, 100) is False
        assert _requests(rt) == asked
        assert time.monotonic() - start >= 0.05
        threading.Timer(0.2, lambda: tx.send(b"x")).start()
        assert call(rx.recv) == b"x"
        assert _requests(rt) == asked

    outside_the_run([])
    rt.start_run()
    call(rt.next_event, math.inf)
    rt.end_run()  # the window closes at the +inf grant
    outside_the_run([("NER", math.inf, [])])
