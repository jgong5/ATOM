# SPDX-License-Identifier: MIT
"""A simulated run's workers sleep between their engine's calls instead of spinning.

A worker's `AsyncIOProc.busy_loop` takes each call off aiter's shared-memory
ring with `MessageQueue.dequeue`, which calls its spin timer, a ``sched_yield``,
between looks at the ring: a core per worker. A simulated worker's wake-up costs
no simulated time, so the spin only takes cores the run's other processes, or a
second run on the host, could use. Inside `sleeping_workers` an engine's
manager takes a `RingingQueue`, which rings each reader's doorbell, an abstract
Unix datagram socket, after every enqueue, and starts `SleepingWorker`s, which
sleep on it in place of the spin. The ring, and each call on it, stays aiter's.

No ring is lost to a sleeping reader: the writer rings after it marks the block
written, and the reader looks at the ring after each wake and before it sleeps.
A ring is dropped only when its reader is not bound yet, and it binds before its
first look; has shut its doorbell to leave; or has rings queued, its queue or
its own sender's buffer full.
"""

from __future__ import annotations

import contextlib
import socket

from aiter.dist.shm_broadcast import MessageQueue, SpinTimer

import atom.model_engine.async_proc as async_proc


def doorbell_address(handle, reader: int) -> str:
    """The abstract socket local reader `reader` of the ring `handle` sleeps on."""
    return f"\0atom-compass-doorbell-{handle.buffer_handle[3]}-{reader}"


class RingingQueue(MessageQueue):
    """aiter's ring, which rings each local reader's doorbell after each enqueue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kind = socket.SOCK_DGRAM | socket.SOCK_NONBLOCK
        self.ringers = [
            (socket.socket(socket.AF_UNIX, kind), doorbell_address(self.handle, reader))
            for reader in range(self.n_local_reader)
        ]

    def enqueue(self, obj, timeout=None):
        super().enqueue(obj, timeout)
        for ringer, address in self.ringers:
            with contextlib.suppress(BlockingIOError, ConnectionError):
                ringer.sendto(b"\1", address)

    def close(self):
        for ringer, _ in getattr(self, "ringers", ()):
            ringer.close()
        super().close()


class Doorbell(SpinTimer):
    """A reader's spin timer that sleeps until the writer rings."""

    def __init__(self, address: str):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(address)

    def spin(self):
        self.socket.recv(1)


class SleepingWorker(async_proc.AsyncIOProc):
    """ATOM's worker, whose ring reader sleeps on its doorbell between calls."""

    def busy_loop(self):
        ring = self.rpc_broadcast_mq
        self.doorbell = Doorbell(doorbell_address(ring.handle, ring.local_reader_rank))
        ring._read_spin_timer = self.doorbell
        super().busy_loop()

    def exit(self):
        super().exit()
        # A SIGTERM runs this inside the wait: shutting the doorbell ends the
        # wait, and the ring `AsyncIOProc.exit` closed then ends the loop, as
        # it ends a spin.
        if (doorbell := getattr(self, "doorbell", None)) is not None:
            doorbell.socket.shutdown(socket.SHUT_RD)


@contextlib.contextmanager
def sleeping_workers(inner):
    """`inner`, inside which an `AsyncIOProcManager` built takes a ringing queue
    and starts sleeping workers; outside it, ATOM's own classes are back."""
    built = async_proc.MessageQueue, async_proc.AsyncIOProc
    async_proc.MessageQueue, async_proc.AsyncIOProc = RingingQueue, SleepingWorker
    try:
        with inner:
            yield
    finally:
        async_proc.MessageQueue, async_proc.AsyncIOProc = built
