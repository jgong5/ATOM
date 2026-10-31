# SPDX-License-Identifier: MIT
"""Which thread serves a request, and which writes its reply.

A simulated step makes several clock calls, and every thread a call crosses on
the serve side is a wake and a GIL hand-off on the run's critical path. So a
request is served on the thread that brings it -- the sender's own for the
in-process carrier, the connection's reader for ``tcp`` -- and the reply is
written from there: no serve loop and no writer thread in between.
"""

import math
import threading

from atom.compass.clock import NER, TAR, ClockAuthority
from atom.compass.clock_transport import connect, serve

from .test_clock_transport import AB, A, B, _later, _result, _table

INF = math.inf
LOCAL = "tcp://127.0.0.1:0"


class _Threads(ClockAuthority):
    """Records, per LP, the thread each of its requests was served on."""

    def __init__(self, channels):
        super().__init__(channels)
        self.served_on = {}

    def on_request(self, lp, kind, t, log, t_daemon=INF, member=None):
        self.served_on.setdefault(str(lp), set()).add(threading.get_ident())
        return super().on_request(lp, kind, t, log, t_daemon, member)


def test_an_in_process_request_is_served_on_its_senders_thread_before_send_returns():
    ca = _Threads(_table())
    server = serve(ca, "inproc:threads")
    try:
        a, b = connect(A, "inproc:threads"), connect(B, "inproc:threads")
        b.send((NER, INF, [], INF))
        a.send((TAR, 1.0, [(AB, 0, 0.7)], INF))
        # Served, and both replies in their slots, by the time `send` returned.
        replies = [a._slot.get_nowait(), b._slot.get_nowait()]
    finally:
        server.close()
    me = threading.get_ident()
    assert ca.served_on == {"a": {me}, "b": {me}}
    assert len(replies) == 2


def test_each_tcp_connection_is_served_on_its_own_reader_and_needs_no_other_thread():
    ca = _Threads(_table())
    server = serve(ca, LOCAL)
    try:
        before = set(threading.enumerate())
        a = _result(_later(connect, A, server.endpoint))
        b = _result(_later(connect, B, server.endpoint))
        added = set(threading.enumerate()) - before
        _result(_later(b.send, (NER, INF, [], INF)))
        held = _later(b.recv)
        _result(_later(a.send, (TAR, 1.0, [(AB, 0, 0.7)], INF)))
        assert _result(_later(a.recv)) == (1.0, {})
        # b's held reply, released by a's request, is written by a's reader.
        assert _result(held) == (0.7, {AB: [(0, 0.7)]})
        for conn in (a, b):
            conn.close()
    finally:
        server.close()
    readers = {t.ident for t in added}
    assert len(added) == 2  # one reader per connection, nothing else
    assert len(ca.served_on) == 2
    assert ca.served_on["a"] != ca.served_on["b"]
    assert ca.served_on["a"] | ca.served_on["b"] <= readers


def test_an_lp_in_the_serving_process_reaches_a_tcp_clock_with_no_socket_and_can_poll():
    import select

    from atom.compass.clock_transport import connect_here

    ca = _Threads(_table())
    server = serve(ca, LOCAL)
    try:
        before = set(threading.enumerate())
        b = connect_here(B, server.endpoint)
        assert set(threading.enumerate()) == before  # no reader for it
        a = _result(_later(connect, A, server.endpoint))
        b.send((NER, INF, [], INF))
        waiting = select.poll()
        waiting.register(b, select.POLLIN)
        assert waiting.poll(0) == []  # held: nothing to read yet
        _result(_later(a.send, (TAR, 1.0, [(AB, 0, 0.7)], INF)))
        assert waiting.poll(10_000) == [(b.fileno(), select.POLLIN)]
        assert b.recv() == (0.7, {AB: [(0, 0.7)]})
        assert waiting.poll(0) == []
        assert _result(_later(a.recv)) == (1.0, {})
        a.close()
        b.close()
    finally:
        server.close()
    assert ca.served_on["b"] == {threading.get_ident()}
