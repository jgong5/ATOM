# SPDX-License-Identifier: MIT
"""`atom.utils.clock.LPRuntime`: release order, the clock a handler sees, and the detectors.

The clock authority is a scripted fake connection that records each request and
answers with the next scripted grant. Receiving threads are real threads doing
what the socket and poller wrappers do: read a frame, wait for its release, take
it, handle it, return to the wait point. The clock owner is a real thread too,
so a call that should return but blocks fails the test instead of hanging it.
"""

import logging
import queue
import threading
import time

import pytest

from atom.compass.clock import ChannelTable, LpId, LpRegistry, single_engine_table
from atom.utils.clock import LPRuntime, Straggler

ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
REQ = "frontend->engine:request#dp0"
CTL = "frontend->engine:control#dp0"
HTTP = "traffic->frontend:http"
OUT = "engine->frontend:output#dp0"
IPC = 0.001
INF = float("inf")


def _table():
    return single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=0.002)


class FakeConn:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    def recv(self):
        return self.replies.pop(0)


def _owner():
    """A daemon clock-owner thread and `call(fn, *args)`, which runs fn on it."""
    jobs = queue.Queue()

    def loop():
        while True:
            fn, box = jobs.get()
            try:
                box.put((True, fn()))
            except Exception as e:  # noqa: BLE001 - re-raised on the calling thread
                box.put((False, e))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    def call(fn, *args):
        box = queue.Queue()
        jobs.put((lambda: fn(*args), box))
        ok, value = box.get(timeout=10)
        if not ok:
            raise value
        return value

    return thread, call


def _runtime(lp, *replies, **kw):
    thread, call = _owner()
    return LPRuntime(lp, _table(), FakeConn(*replies), owner=thread, **kw), call


def _receiver(rt, ch, n, seen, hold=lambda: None):
    """A thread that takes `n` released messages off `ch`, recording the clock it sees."""

    def run():
        wake = rt.my_wakeup()
        for _ in range(n):
            rt.back_at_wait_point()
            while True:
                wake.clear()
                with rt.lock:
                    pending = rt.released[ch] - rt.handled[ch]
                if pending:
                    break
                wake.wait()
            seq = min(pending)
            rt.handed_over(ch, seq)
            seen.append((ch, seq, rt.read_clock()))
            hold()
        rt.back_at_wait_point()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_handler_runs_at_its_arrival_and_the_owner_waits_for_it():
    rt, call = _runtime(ENGINE, (10.2, {REQ: [(0, 10.1)]}))
    rt.check_arrival(REQ, 10.1, 0)
    seen, now_while_handling, inline_while_handling = [], [], []

    def hold():
        time.sleep(0.2)
        now_while_handling.append(rt.now)
        inline_while_handling.append(rt.inline_pending())

    handler = _receiver(rt, REQ, 1, seen, hold)
    call(rt.advance_to, 10.2)
    assert seen == [(REQ, 0, 10.1)]
    assert now_while_handling == [10.1]
    assert inline_while_handling == [False]
    assert rt.now == 10.2
    handler.join(5)
    assert not handler.is_alive()


def test_release_is_in_arrival_channel_seq_order_not_seq_order():
    grant = {REQ: [(0, 10.05)], CTL: [(0, 10.01), (1, 10.05)]}
    rt, call = _runtime(ENGINE, (10.2, grant))
    seen = []
    threads = [_receiver(rt, REQ, 1, seen), _receiver(rt, CTL, 2, seen)]
    call(rt.advance_to, 10.2)
    assert seen == [(CTL, 0, 10.01), (CTL, 1, 10.05), (REQ, 0, 10.05)]
    for t in threads:
        t.join(5)


def test_an_inline_release_does_not_wait_and_stays_pending_until_taken():
    rt, call = _runtime(FRONTEND, (10.2, {HTTP: [(0, 10.1)]}))
    assert not rt.inline_pending()
    call(rt.advance_to, 10.2)
    assert rt.now == 10.2
    assert rt.inline_pending()
    with rt.lock:
        rt.count_done_locked(HTTP, 0)
    assert not rt.inline_pending()


def test_sends_are_stamped_now_plus_lookahead_and_ride_the_next_request():
    rt, call = _runtime(ENGINE, (0.5, {}), (0.7, {}))
    assert call(rt.stamp_send, OUT) == (IPC, 0)
    call(rt.advance_to, 0.5)
    assert call(rt.stamp_send, OUT) == (0.5 + IPC, 1)
    assert call(rt.next_event, 0.7) == 0.7
    assert rt.conn.sent == [
        ("TAR", 0.5, [(OUT, 0, IPC)], INF),
        ("NER", 0.7, [(OUT, 1, 0.5 + IPC)], INF),
    ]
    with pytest.raises(ValueError, match="cannot advance to 0.1, it is at 0.7"):
        call(rt.advance_to, 0.1)


def test_the_inf_grant_closes_the_clock_and_leaves_sends_stamped_inf():
    rt, call = _runtime(ENGINE, (INF, {}))
    with pytest.raises(RuntimeError, match=r"left its loop at 0.0, before the \+inf"):
        rt.close()
    call(rt.stamp_send, OUT)
    assert call(rt.next_event, INF, 5.0) == INF
    assert rt.conn.sent == [("NER", INF, [(OUT, 0, IPC)], 5.0)]
    for fn, args in [(rt.advance_to, (1.0,)), (rt.next_event, (INF,))]:
        with pytest.raises(RuntimeError, match=rf"^{fn.__name__} from engine after"):
            call(fn, *args)
    assert call(rt.stamp_send, OUT) == (INF, 1)
    rt.close()
    assert len(rt.conn.sent) == 1


def test_a_send_or_clock_call_off_the_owner_thread_is_refused():
    rt, _ = _runtime(ENGINE)
    for fn, args in [
        (rt.stamp_send, (OUT,)),
        (rt.advance_to, (1.0,)),
        (rt.next_event, (1.0,)),
    ]:
        with pytest.raises(RuntimeError, match=f"^{fn.__name__} from thread"):
            fn(*args)
    assert rt.send_log == [] and rt.conn.sent == []


def test_a_duplicate_seq_is_refused():
    rt, _ = _runtime(ENGINE)
    rt.check_arrival(REQ, 1.0, 3)
    with pytest.raises(RuntimeError, match="seq 3 arrived twice"):
        rt.check_arrival(REQ, 1.0, 3)


def test_a_frame_read_after_its_arrival_was_passed_is_a_straggler():
    rt, call = _runtime(ENGINE, (10.2, {}))
    # Buffered at the grant time: that instant's next round, not late.
    rt.check_arrival(CTL, 10.2, 0)
    call(rt.advance_to, 10.2)
    rt.check_arrival(REQ, 10.2, 0)  # the same, read at the grant time
    with pytest.raises(
        Straggler, match=r"seq 1 arrives at 10.1 but engine has released up to 10.2"
    ):
        rt.check_arrival(REQ, 10.1, 1)


def test_a_released_frame_read_after_its_arrival_is_not_a_straggler():
    rt, call = _runtime(FRONTEND, (10.2, {HTTP: [(0, 10.1)]}), (10.3, {}))
    call(rt.advance_to, 10.2)
    rt.check_arrival(HTTP, 10.1, 0)
    call(rt.advance_to, 10.3)
    assert rt.now == 10.3


def test_a_buffered_frame_no_grant_released_is_a_straggler_at_the_drain():
    rt, call = _runtime(ENGINE, (10.2, {}))
    rt.check_arrival(REQ, 10.1, 0)
    with pytest.raises(
        Straggler, match=r"seq 0 arrives at 10.1 but engine has released up to 10.2"
    ):
        call(rt.advance_to, 10.2)


def test_an_unhandled_release_prints_one_diagnostic_and_keeps_waiting(caplog):
    rt, call = _runtime(ENGINE, (10.2, {REQ: [(0, 10.1)]}), diag_s=0.05)
    seen = []
    _receiver(rt, REQ, 1, seen, hold=lambda: time.sleep(0.4))
    with caplog.at_level(logging.WARNING, logger="atom"):
        call(rt.advance_to, 10.2)
    diags = [
        r.getMessage() for r in caplog.records if "still waiting" in r.getMessage()
    ]
    assert len(diags) == 1
    assert f"released {REQ} seq 0" in diags[0]
    assert rt.now == 10.2


def test_a_channel_of_another_data_parallel_rank_is_refused_by_name():
    registry = LpRegistry()
    for lp in (FRONTEND, ENGINE):
        registry.register(lp)
    table = ChannelTable(registry)
    table.declare("frontend->engine:request#dp1", FRONTEND, ENGINE, IPC, "thread")
    with pytest.raises(NotImplementedError, match="'frontend->engine:request#dp1'"):
        LPRuntime(ENGINE, table, FakeConn())
