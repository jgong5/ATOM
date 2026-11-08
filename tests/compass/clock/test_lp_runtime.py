# SPDX-License-Identifier: MIT
"""`atom.utils.clock.LPRuntime`: release order, the clock a handler sees, and the detectors.

The clock authority is a scripted fake connection that records each request and
answers with the next scripted grant. Receiving threads are real threads doing
what the socket and poller wrappers do: read a frame, wait for its release, take
it, handle it, return to the wait point. The clock owner is a real thread too,
so a call that should return but blocks fails the test instead of hanging it.
"""

import logging
import math
import queue
import random
import threading
import time

import pytest

from atom.compass.clock import LpId, Released, single_engine_table
from atom.utils.clock import LPRuntime, Straggler

ENGINE, FRONTEND = LpId("engine"), LpId("frontend")
REQ = "frontend->engine:request#dp0"
CTL = "frontend->engine:control#dp0"
HTTP = "traffic->frontend:http"
OUT = "engine->frontend:output#dp0"
IPC = 0.001
INF = float("inf")


def _table():
    return single_engine_table(admission_s=0.54e-3, ipc_s=IPC, stream_s=0.002)


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
    rt.start_run()
    with pytest.raises(RuntimeError, match=r"left its loop at 0.0, before the \+inf"):
        rt.close()
    call(rt.stamp_send, OUT)
    assert call(rt.next_event, INF, 5.0) == INF
    assert not rt.in_run
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


# --- below a ceiling ----------------------------------------------------------


def _under(ceiling, released=None):
    """A grant's releases with `ceiling`, as the authority sends them."""
    out = Released(released or {})
    out.ceiling = ceiling
    return out


def test_below_a_ceiling_the_owner_advances_on_its_own_and_its_next_call_says_so():
    rt, call = _runtime(FRONTEND, (0.5, _under(0.6)), (0.7, {}), (0.75, {}))
    assert call(rt.next_event, INF) == 0.5
    call(rt.stamp_send, REQ)
    call(rt.advance_to, 0.55)
    call(rt.stamp_send, REQ)
    call(rt.advance_to, 0.7)
    call(rt.advance_to, 0.75)  # the grant to 0.7 carried no ceiling
    assert rt.conn.sent == [
        ("NER", INF, [], INF),
        (
            "TAR",
            0.7,
            [(REQ, 0, 0.5 + IPC), (REQ, 1, 0.55 + IPC)],
            INF,
            [("TAR", 0.55, 0.55, 1)],
            False,
        ),
        ("TAR", 0.75, [], INF),
    ]
    assert rt.calls == 4 and rt.now == 0.75


def test_messages_sent_ahead_are_released_one_step_at_a_time():
    ahead = {HTTP: [(0, 1.0), (1, 1.2), (2, 1.2), (3, 1.5)]}
    rt, call = _runtime(
        FRONTEND, (0.5, _under(0.75)), (1.0, _under(1.6, ahead)), (INF, {})
    )
    seen = []
    for t, t_daemon in [(INF, INF)] * 3 + [(1.3, INF), (INF, 1.4)] + [(INF, INF)] * 2:
        seen.append((call(rt.next_event, t, t_daemon), sorted(rt.released[HTTP])))
    assert seen == [
        (0.5, []),
        (1.0, [0]),
        (1.2, [0, 1, 2]),
        (1.3, [0, 1, 2]),
        (1.4, [0, 1, 2]),
        (1.5, [0, 1, 2, 3]),
        (INF, [0, 1, 2, 3]),
    ]
    steps = [
        ("NER", INF, 1.2, 0),
        ("NER", 1.3, 1.3, 0),
        ("NER", INF, 1.4, 0),
        ("NER", INF, 1.5, 0),
    ]
    assert rt.conn.sent == [
        ("NER", INF, [], INF),
        ("NER", INF, [], INF, [], True),
        ("NER", INF, [], INF, steps, True),
    ]


def test_a_handler_runs_at_the_arrival_of_a_message_sent_ahead():
    ahead = {REQ: [(0, 10.1), (1, 10.3)]}
    rt, call = _runtime(ENGINE, (10.0, _under(10.05)), (10.1, _under(10.5, ahead)))
    seen = []
    handler = _receiver(rt, REQ, 2, seen)
    call(rt.next_event, INF)
    call(rt.next_event, INF)
    with pytest.raises(RuntimeError, match="messages sent ahead unreached"):
        call(rt.ask_next_event, INF)
    call(rt.advance_to, 10.4)
    assert seen == [(REQ, 0, 10.1), (REQ, 1, 10.3)]
    assert rt.now == 10.4 and len(rt.conn.sent) == 2
    handler.join(5)
    assert not handler.is_alive()


def _frontend_after(n):
    """A frontend that has taken `n` requests on its inline channel, each read,
    released and handled, as a run's frontend has by its `n`-th."""
    rt, call = _runtime(FRONTEND, (n + 1.0, {HTTP: [(k, k + 0.5) for k in range(n)]}))
    for k in range(n):
        rt.check_arrival(HTTP, k + 0.5, k)
    call(rt.advance_to, n + 1.0)
    with rt.lock:
        for k in range(n):
            rt.count_done_locked(HTTP, k)
    return rt


def _us_a_call(fn):
    """`fn()`'s cost: the best of three passes of 200 calls, in microseconds a call."""
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(200):
            fn()
        best = min(best, time.perf_counter() - start)
    return best / 200 * 1e6


def test_the_idle_check_costs_the_same_however_many_requests_the_run_has_taken():
    """`inline_pending` runs at every idle point of a frontend's event loop."""
    early, late = _frontend_after(10), _frontend_after(50_000)
    assert not early.inline_pending() and not late.inline_pending()
    a, b = _us_a_call(early.inline_pending), _us_a_call(late.inline_pending)
    assert b < 5 * a + 5, f"{a:.1f} us after 10 requests, {b:.1f} us after 50,000"


def test_pending_and_unread_follow_each_request_read_released_and_handled():
    """Requests read before or after their release, and handled in any order
    once both: after each step, `inline_pending` and the unread set say what
    the message sets say."""
    rng, n = random.Random(0), 300
    rt, call = _runtime(
        FRONTEND, *((k + 1.0, {HTTP: [(k, k + 0.5)]}) for k in range(n))
    )

    def check():
        with rt.lock:
            pending = bool(rt.released[HTTP] - rt.handled[HTTP])
            unread = rt.released[HTTP] - rt.arrived[HTTP]
        assert rt.inline_pending() == pending
        assert rt.unread[HTTP] == unread

    open_ = []
    for k in range(n + 40):
        if k < n:
            if rng.random() < 0.5:  # read before its release
                rt.check_arrival(HTTP, k + 0.5, k)
                check()
            call(rt.advance_to, k + 1.0)
            open_.append(k)
            check()
        for seq in list(open_):
            if seq not in rt.arrived[HTTP] and rng.random() < 0.5:
                rt.check_arrival(HTTP, seq + 0.5, seq)
                check()
            if seq in rt.arrived[HTTP] and rng.random() < 0.3:
                with rt.lock:
                    rt.count_done_locked(HTTP, seq)
                open_.remove(seq)
                check()
    for seq in open_:
        if seq not in rt.arrived[HTTP]:
            rt.check_arrival(HTTP, seq + 0.5, seq)
        with rt.lock:
            rt.count_done_locked(HTTP, seq)
    check()
    assert not rt.inline_pending() and not rt.unread[HTTP]
