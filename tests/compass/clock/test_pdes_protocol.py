# SPDX-License-Identifier: MIT
"""Two LPs over the real clock authority: a message still in flight when it is
released, and delivery in timestamp order.

`prefill` sends on one PUSH/PULL channel to `decode`. A handler thread in
`decode` receives through `WrappedSocket` and queues what it takes, with the
clock it read, for the stand-in of `schedule()` that `decode`'s clock owner
runs after each grant. Everything between the two LPs is real: `LPRuntime` on
each side, the wrappers, the in-process carrier and `ClockAuthority`. A
forwarder thread between the two sockets is the transport, and adds real delay.
Each run ends by the finish, both LPs idle and granted ``+inf``.
"""

import math
import queue
import threading
import time

import zmq

from atom.compass.clock import ChannelTable, ClockAuthority, LpId, LpRegistry
from atom.compass.clock_transport import connect, serve
from atom.utils.clock import LPRuntime, WrappedSocket

from .test_clock_transport import WAIT_S, _later, _result

PREFILL, DECODE = LpId("prefill"), LpId("decode")
DONE = "prefill->decode:prefill_done"
L = 0.5
RUNS = 20


def _table():
    registry = LpRegistry()
    registry.register(PREFILL)
    registry.register(DECODE)
    table = ChannelTable(registry)
    table.declare(DONE, PREFILL, DECODE, L, "thread")
    return table


def _forward(pull, push, delay_s):
    """The transport: each frame leaves `delay_s` wall seconds after it came in."""
    while True:
        frames = pull.recv_multipart()
        time.sleep(delay_s)
        push.send_multipart(frames)
        if frames[-1] == b"stop":
            return


def _run(sends, steps, delay_s=0.0, present=0):
    """One run: `prefill` sends one message after each grant in `sends`, and
    `decode` drains its queue after each grant in `steps`, first waiting until
    `present` frames have physically reached it.

    Returns each drain as ``(time, [(payload, clock its handler read)])``.
    """
    table = _table()
    server = serve(ClockAuthority(table), "inproc:pdes")
    ctx = zmq.Context()
    try:
        tx, fwd_in, fwd_out, rx = (
            ctx.socket(t) for t in (zmq.PUSH, zmq.PULL, zmq.PUSH, zmq.PULL)
        )
        fwd_in.bind("inproc://sent")
        tx.connect("inproc://sent")
        rx.bind("inproc://delivered")
        fwd_out.connect("inproc://delivered")
        forwarder = threading.Thread(
            target=_forward, args=(fwd_in, fwd_out, delay_s), daemon=True
        )
        forwarder.start()

        def prefill():
            rt = LPRuntime(PREFILL, table, connect(PREFILL, server.endpoint))
            ws = WrappedSocket(rt, tx, DONE)
            rt.start_run()
            for k, t in enumerate(sends):
                rt.advance_to(t)
                ws.send(b"m%d" % k)
            rt.next_event(math.inf)
            rt.close()
            ws.send(b"stop")  # after the +inf grant: unstamped and not counted

        def decode():
            rt = LPRuntime(DECODE, table, connect(DECODE, server.endpoint))
            ws = WrappedSocket(rt, rx, DONE)
            taken = queue.Queue()

            def handler():
                while (m := ws.recv()) != b"stop":
                    taken.put((m, rt.read_clock()))

            rt.start_run()
            thread = threading.Thread(target=handler, daemon=True)
            thread.start()
            while len(rt.arrived[DONE]) < present:
                time.sleep(0.001)
            batches = []
            for t in steps:
                rt.advance_to(t)
                batches.append((t, [taken.get() for _ in range(taken.qsize())]))
            rt.next_event(math.inf)
            rt.close()
            return batches, thread

        sender, receiver = _later(prefill), _later(decode)
        batches, handler = _result(receiver)
        _result(sender)
        handler.join(WAIT_S)
        forwarder.join(WAIT_S)
        assert not handler.is_alive() and not forwarder.is_alive()
        return batches
    finally:
        ctx.destroy(linger=0)
        server.close()


def _every_run(expected, **run):
    differ = [b for b in (_run(**run) for _ in range(RUNS)) if b != expected]
    assert not differ, f"{len(differ)} of {RUNS} runs differ, first {differ[0]}"


def test_a_drain_sees_a_message_released_while_still_in_flight():
    """m0 is registered at 9.6 and arrives at 10.1, and the transport holds it for
    5 ms of wall time. `decode`'s grant to 10.2 releases it before it reaches
    `decode`, and its drain at 10.2 has it anyway."""
    _every_run(
        [(10.0, []), (10.2, [(b"m0", 9.6 + L)])],
        sends=(9.6,),
        steps=(10.0, 10.2),
        delay_s=0.005,
    )


def test_a_handler_runs_at_each_arrival_and_a_drain_sees_no_later_one():
    """Three messages, all physically in `decode` before its drain at 10.0 and
    all released by its one grant from 10.0 to 10.2. The drain at 10.0 sees none
    of them, and the handler reads each one's arrival as the clock."""
    _every_run(
        [(10.0, []), (10.2, [(b"m0", 9.6 + L), (b"m1", 9.62 + L), (b"m2", 9.65 + L)])],
        sends=(9.6, 9.62, 9.65),
        steps=(10.0, 10.2),
        present=3,
    )
