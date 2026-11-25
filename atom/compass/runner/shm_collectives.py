# SPDX-License-Identifier: MIT
"""A simulated run's node-local DP groups run their small host collectives over shared memory.

Each collective a simulated DP group runs, Compass's and ATOM's alike, moves a
few bytes, and gloo takes half a millisecond or more a call to move them.
`attach` maps one POSIX shared memory segment into every rank of a gloo group,
once, right after the group is built; `install` routes
`torch.distributed.all_reduce` and `all_gather` on an attached group to it. A
call writes this rank's bytes to its row, publishes the call's round on its own
cache line and waits for every rank's, then reads the rows in rank order. It
relies on x86's store order: a rank that sees a peer's round sees the row the
peer wrote before it.

A group stays on gloo unless every rank maps the segment and binds its
doorbell, from one PID and network namespace, on x86; a group spanning hosts is
one. A call stays on gloo when it is async, its tensor is not a CPU tensor of
one of the integer or float types both gloo and numpy hold, it moves over `ROW`
bytes a rank, or it is a float SUM, whose bits hang on the order of the adds;
the ranks of a group make the same call, so each makes the same choice. What
comes back is gloo's bit for bit: an integer SUM wraps as gloo's does, MAX and
MIN are exact in any order, and a float MAX or MIN that meets a NaN or a zero,
whose sign gloo's order picks, is redone on gloo.

A rank waiting for its peers yields its core between looks for `SPIN_S`, about
what waking a sleeping core costs, then sleeps on its doorbell, an abstract Unix
datagram socket. A rank that publishes its round rings each peer that published
it first, so a wait on the group's last rank, such as a DP state sync waiting on
a peer still handling its step's output, holds no core. No ring is lost: a rank
sleeps only `SPIN_S` after it published, and a peer that publishes later reads
its round and rings it. Asleep, it checks its late peers' processes about every
second, and raises RuntimeError, as gloo does, once one has exited.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import secrets
import select
import socket
import time
from multiprocessing import shared_memory
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

#: Bytes a rank's row holds; a call moving more stays on gloo.
ROW = 4096
#: A rank's round sits alone on a cache line.
LINE = 64
#: A waiting rank yields its core between looks at its peers' rounds for
#: `SPIN_S` seconds, then sleeps on its doorbell, checking its late peers'
#: processes every `CHECK_S`.
SPIN_S, CHECK_S = 1e-4, 1.0
#: A doorbell's socket: a ring never blocks its sender, and its reader takes
#: the rings queued without blocking.
_DATAGRAM = socket.SOCK_DGRAM | socket.SOCK_NONBLOCK
#: The types gloo reduces and numpy holds.
_INTS = (torch.uint8, torch.int8, torch.int32, torch.int64)
_FLOATS = (torch.float16, torch.float32, torch.float64)
#: Each attached group's segment, by its gloo group.
_GROUPS: dict = {}


class _Segment:
    """A group's segment: each rank's round on a line of its own, then two
    slots of every rank's row, which the rounds take in turn; and this rank's
    doorbell, with its peers' addresses."""

    def __init__(self, memory, rank: int, world: int, pids: list[int], bell, name):
        self.memory, self.rank, self.world, self.pids = memory, rank, world, pids
        self.round = 0
        self.bell, self.asleep = bell, select.poll()
        self.asleep.register(bell, select.POLLIN)
        self.ringer = socket.socket(socket.AF_UNIX, _DATAGRAM)
        self.peers = [(p, _address(name, p)) for p in range(world) if p != rank]

    def exchange(self, a: np.ndarray) -> np.ndarray:
        """Every rank's `a`, stacked in rank order."""
        self.round += 1
        r, buf, world = self.round, self.memory.buf, self.world
        rounds = buf[: LINE * world].cast("q")[:: LINE // 8]
        slot = LINE * world + (r & 1) * world * ROW
        rows = np.ndarray((world, ROW), np.uint8, buf, slot)
        raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        rows[self.rank, : raw.size] = raw
        rounds[self.rank] = r
        for p, address in self.peers:
            if rounds[p] >= r:  # it came first and may be asleep
                with contextlib.suppress(BlockingIOError, ConnectionError):
                    self.ringer.sendto(b"\1", address)
        if min(rounds) < r:
            self._wait(rounds, r)
        out = rows[:, : raw.size].copy().view(a.dtype)
        return out.reshape((world, *a.shape))

    def _wait(self, rounds, r: int) -> None:
        """Until every rank has published round `r`; then takes the rings its
        peers sent it on the way."""
        spun = time.monotonic() + SPIN_S
        while min(rounds) < r:
            if time.monotonic() < spun:
                os.sched_yield()
            elif self.asleep.poll(CHECK_S * 1e3):
                self._take_rings()
            else:
                late = [p for p in range(self.world) if rounds[p] < r]
                if gone := [p for p in late if not _alive(self.pids[p])]:
                    raise RuntimeError(f"DP ranks {gone} of the group have exited")
        self._take_rings()

    def _take_rings(self) -> None:
        with contextlib.suppress(BlockingIOError):
            while True:
                self.bell.recv(1)


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in "ZX"
    except (OSError, IndexError):
        return False


def _segment(name: str, size: int = 0):
    """The segment `name`, created at `size` bytes or, with none, opened; None
    if that fails."""
    try:
        if size:
            return shared_memory.SharedMemory(name, create=True, size=size)
        # Opening registers the segment with this process's resource tracker,
        # which would unlink it when this process exits; aiter's ring opens
        # its own the same way.
        with mock.patch("multiprocessing.resource_tracker.register", lambda *a: None):
            return shared_memory.SharedMemory(name)
    except OSError:
        return None


def _address(name: str, rank: int) -> str:
    """The abstract socket rank `rank` of the group on segment `name` sleeps on."""
    return f"\0{name}-{rank}"


def _doorbell(name: str, rank: int):
    """Rank `rank`'s doorbell for the segment `name`, bound; None if that fails."""
    bell = socket.socket(socket.AF_UNIX, _DATAGRAM)
    try:
        bell.bind(_address(name, rank))
    except OSError:
        bell.close()
        return None
    return bell


def _network() -> int:
    """This process's network namespace, which an abstract socket lives in."""
    return os.stat("/proc/self/ns/net").st_ino


def _here() -> list[int]:
    """Whether this host is x86, this process's PID namespace and its pid."""
    x86 = platform.machine() in ("x86_64", "AMD64")
    return [x86, os.stat("/proc/self/ns/pid").st_ino, os.getpid()]


def attach(group) -> bool:
    """Put the gloo group `group` on shared memory if every rank maps rank 0's
    segment and binds its doorbell, from one PID and network namespace, on
    x86; whether it did. A collective over `group`, which stays on gloo
    otherwise."""
    world, rank = group.size(), group.rank()
    if world < 2:
        return False
    size = LINE * world + 2 * world * ROW
    token = torch.tensor([secrets.randbits(63) if rank == 0 else 0])
    memory = _segment(f"atom-compass-dp-{token.item():x}", size) if rank == 0 else None
    dist.all_reduce(token, group=group)  # rank 0's token reaches every rank
    name = f"atom-compass-dp-{token.item():x}"
    if rank:
        memory = _segment(name)
    bell = _doorbell(name, rank)
    joins = memory is not None and bell is not None
    me = torch.tensor([joins, *_here(), _network()], dtype=torch.int64)
    rows = [torch.empty_like(me) for _ in range(world)]
    dist.all_gather(rows, me, group=group)
    rows = [r.tolist() for r in rows]
    if rank == 0 and memory is not None:
        memory.unlink()  # every rank has opened it or failed to
    same = rows[0][2], rows[0][4]
    if all(j and x86 and (ns, net) == same for j, x86, ns, _, net in rows):
        pids = [row[3] for row in rows]
        _GROUPS[group] = _Segment(memory, rank, world, pids, bell, name)
        said = "on shared memory"
    else:
        for held in (memory, bell):
            if held is not None:
                held.close()
        said = (
            f"kept on gloo: {rows} (mapped and bound, x86, PID namespace, pid, "
            "network namespace by rank)"
        )
    logger.info("DP group, rank %d of %d, %s", rank, world, said)
    return group in _GROUPS


def _fits(t) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.device.type == "cpu"
        and (t.dtype in _INTS or t.dtype in _FLOATS)
        and t.numel() * t.element_size() <= ROW
    )


def install() -> None:
    """Route `torch.distributed.all_reduce` and `all_gather` on an attached
    group to its segment, around whatever the two are now."""
    reduce, gather = dist.all_reduce, dist.all_gather
    SUM, MAX, MIN = dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN

    def all_reduce(*args, **kwargs):
        call = dict(zip(("tensor", "op", "group", "async_op"), args), **kwargs)
        tensor, op = call.get("tensor"), call.get("op", SUM)
        segment = None if call.get("async_op") else _GROUPS.get(call.get("group"))
        fn = None
        if segment is not None and _fits(tensor):
            fn = np.max if op == MAX else np.min if op == MIN else None
            fn = np.sum if op == SUM and tensor.dtype in _INTS else fn
        if fn is None:
            return reduce(*args, **kwargs)
        a = tensor.detach().numpy()
        rows = segment.exchange(a)
        if fn is not np.sum and rows.dtype.kind == "f" and not (abs(rows) > 0).all():
            return reduce(*args, **kwargs)
        np.copyto(a, fn(rows, axis=0), casting="unsafe")

    def all_gather(*args, **kwargs):
        call = dict(zip(("tensor_list", "tensor", "group", "async_op"), args), **kwargs)
        outs, tensor = call.get("tensor_list"), call.get("tensor")
        segment = None if call.get("async_op") else _GROUPS.get(call.get("group"))
        if segment is None or not _fits(tensor) or len(outs) != segment.world:
            return gather(*args, **kwargs)
        for out, row in zip(outs, segment.exchange(tensor.detach().numpy())):
            np.copyto(out.detach().numpy(), row.reshape(out.shape), casting="unsafe")

    dist.all_reduce, dist.all_gather = all_reduce, all_gather


@contextlib.contextmanager
def attaching_engine_group(inner):
    """`inner`, inside which the DP group ATOM's engine core builds, from
    `ParallelConfig.stateless_init_dp_group`, is attached; outside it ATOM's
    own method is back."""
    from atom.config import ParallelConfig

    build = ParallelConfig.stateless_init_dp_group

    def built(self):
        group = build(self)
        attach(group)
        return group

    ParallelConfig.stateless_init_dp_group = built
    try:
        with inner:
            yield
    finally:
        ParallelConfig.stateless_init_dp_group = build
