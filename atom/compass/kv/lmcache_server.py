# SPDX-License-Identifier: MIT
"""A stand-in for the lmcache MP server: its wire protocol and its chunk index.

It answers the request types ATOM's lmcache adapters send, in lmcache's own MQ
frames, with the reply payload the real server returns, and keeps the state
that decides later hits: which chunks are resident, their LRU order, their read
locks and the lock TTL, the byte budget and its eviction, and the per-request
sessions. It holds no bytes and binds no socket; the caller owns the loop, the
clock and the socket, and passes the time of each event in.

`handle` takes the frames a ROUTER socket receives (identity, request id,
operation, payloads) at a job's start and returns the reply frames, a
callable to run at the job's end (a store's chunks become visible, and a
retrieve's read locks are released, when its copy completes) and the chunks
the job works on: a lookup's prompt chunks, the chunks a store or retrieve
copies, none otherwise. `evict` is one tick of the server's eviction loop,
which wakes every `EVICTION_INTERVAL_S`.
`expire_sessions` is one tick of its session cleanup, which runs at the
server's start and every `SESSION_SWEEP_INTERVAL_S` after it, and drops each
session created more than `SESSION_TTL_S` earlier.

What it does not reproduce. Under `--separate-object-groups`, where a chunk's
recurrent state (`state_bytes` a rank) is an object of its own, sliding-window
groups price and hit as full attention, and all-null attention chunks are
stored. The byte budget is
`--l1-size-gb` taken whole, so a lazy pool's growth and allocator alignment
are not modelled. Stores and retrieves need no prior REGISTER_KV_CACHE,
because a simulated worker registers no device memory. The session sweep's
phase is its caller's start: a real server starts before its engine loads, so
the same session can drop at another sweep, up to an interval apart.

A flag or a request type outside these is refused by name.
"""

from __future__ import annotations

import argparse
import heapq
import math
import pickle
import sys
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

EVICTION_INTERVAL_S = 1.0
SESSION_TTL_S = 600.0  # lmcache's SessionManager.DEFAULT_SESSION_TTL
SESSION_SWEEP_INTERVAL_S = 60.0
#: The request loop's laws as a run file's ``lmcache`` entry states them,
#: declared, not measured: a LOOKUP 2^-13 s plus 2^-17 s a prompt chunk, any
#: other request 2^-14 s.
DECLARED_LAWS = {"lookup_s": [2.0**-13, 2.0**-17], "request_s": 2.0**-14}


class LMCacheRefusal(Exception):
    """A server flag or a request type the stand-in does not model."""


class _Flags(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise LMCacheRefusal(f"lmcache server flags: {message}")


def _flag_parser() -> argparse.ArgumentParser:
    p = _Flags(prog="lmcache server", add_help=False, allow_abbrev=False)
    a = p.add_argument
    a("--host", default="localhost")
    a("--port", type=int, default=5555)
    a("--chunk-size", type=int, default=256)
    a("--hash-algorithm", default="blake3")
    a("--l1-size-gb", type=float, required=True)
    a("--l1-read-ttl-seconds", type=int, default=300)
    a("--eviction-policy", choices=["LRU"], required=True)
    a("--eviction-trigger-watermark", type=float, default=0.8)
    a("--eviction-ratio", type=float, default=0.2)
    a("--supported-transfer-mode", choices=["lmcache_driven"], default="lmcache_driven")
    a("--null-block-id", type=int, default=0)
    a("--separate-object-groups", action="store_true")
    # Accepted, and read by nothing here: the budget is the configured size
    # (lazy pool), and the rest set workers and endpoints no reply depends on.
    a("--l1-use-lazy", action=argparse.BooleanOptionalAction, default=True)
    a("--max-workers", type=int, default=1)
    a("--max-gpu-workers", type=int)
    a("--max-cpu-workers", type=int)
    a("--http-host")
    a("--http-port", type=int)
    return p


def parse_server_flags(argv: list[str]) -> argparse.Namespace:
    """Parse `lmcache server` flags, refusing any this module does not model."""
    args, unknown = _flag_parser().parse_known_args(argv)
    if unknown:
        raise LMCacheRefusal(
            f"lmcache stand-in does not model server flag(s) {unknown}"
        )
    return args


@dataclass
class _Obj:
    readers: int = 0
    lock_until: float = float("-inf")

    def locked(self, now: float) -> bool:
        return self.readers > 0 and now < self.lock_until


@dataclass
class _Session:
    created: float
    tokens: tuple[int, ...] = ()
    key: Any = None
    hit: int = -1


def _noop(now: float) -> None:
    pass


class LMCacheServer:
    """The state and replies of one `lmcache server` process."""

    def __init__(self, argv: list[str], *, chunk_bytes: int, state_bytes=0) -> None:
        # lmcache brings torch with it, so importing this module stays free of both.
        from lmcache.v1.multiprocess.rpc import get_rpc_spec
        from lmcache.v1.multiprocess.token_hasher import TokenHasher
        from lmcache.v1.multiprocess.transport.zmq_impl import mq, wire

        self._spec, self._mq, self._wire = get_rpc_spec, mq, wire
        self.args = parse_server_flags(argv)
        self.chunk_size = self.args.chunk_size
        self.chunk_bytes, self.state_bytes = chunk_bytes, state_bytes
        self._recurrent = False  # a store has carried recurrent state
        self.capacity = int(self.args.l1_size_gb * (1 << 30))
        self._ttl = self.args.l1_read_ttl_seconds
        self._hasher = TokenHasher(self.chunk_size, self.args.hash_algorithm)
        self.objects: OrderedDict[tuple, _Obj] = OrderedDict()  # LRU first
        self._staged: set[tuple] = set()
        self._prefetch: dict[str, int] = {}
        self._sessions: dict[str, _Session] = {}
        self._handlers: dict[str, Callable[..., tuple[Any, Callable]]] = {
            "get_chunk_size": self._get_chunk_size,
            "lookup": self._lookup,
            "query_prefetch_status": self._query_prefetch_status,
            "free_lookup_locks": self._free_lookup_locks,
            "end_session": self._end_session,
            "store": self._store,
            "retrieve": self._retrieve,
        }

    @property
    def url(self) -> str:
        return f"tcp://{self.args.host}:{self.args.port}"

    def handle(
        self, frames: list[bytes], now: float
    ) -> tuple[list[bytes], Callable[[float], None], int]:
        """Serve one request at its job's start; return its reply frames, the
        effect to apply at its job's end and the chunks the job works on."""
        identity, b_uid, b_op, *payloads = frames
        op = self._wire.decode_operation(b_op)
        handler = self._handlers.get(op)
        if handler is None:
            raise LMCacheRefusal(f"lmcache stand-in does not model request type {op!r}")
        spec = self._spec(op)
        args = self._mq.unwrap_request_payloads(payloads, list(spec.payload_types))
        # A handler that works on no chunks returns two values.
        response, at_end, *chunks = handler(*args, now=now)
        reply = [identity, b_uid, b_op]
        if response is not None:
            reply.append(self._mq.msgspec_encode(response, cls=spec.response_type))
        return reply, at_end, sum(chunks)

    def evict(self, now: float) -> list[tuple]:
        """One eviction-loop tick: above the watermark, drop the configured
        share of tracked objects, resident and staged, least recently used
        first, skipping read-locked and staged ones, and say so on stderr as
        lmcache logs it."""
        tracked = len(self.objects) + len(self._staged)
        mark = self.args.eviction_trigger_watermark
        used = self._size(self.objects) + self._size(self._staged)
        if not self.capacity or (usage := used / self.capacity) < mark:
            return []
        ratio = min(1.0, max(0.0, self.args.eviction_ratio))
        target = int(tracked * ratio) or int(ratio > 0)
        victims = [k for k, o in self.objects.items() if not o.locked(now)][:target]
        for k in victims:
            del self.objects[k]
        print(
            f"lmcache stand-in at {now:.6f} s: L1 memory usage {usage:.2f} above "
            f"watermark {mark:.2f}; triggering eviction of {len(victims)} objects",
            file=sys.stderr,
            flush=True,
        )
        return victims

    def expire_sessions(self, now: float) -> None:
        """One session-cleanup tick: drop every session created more than
        `SESSION_TTL_S` before `now`, and say so on stderr as lmcache logs it."""
        expired = [
            r for r, s in self._sessions.items() if now - s.created > SESSION_TTL_S
        ]
        for rid in expired:
            del self._sessions[rid]
        if expired:
            print(
                f"lmcache stand-in at {now:.6f} s: Cleaned up {len(expired)} "
                f"expired sessions {expired}",
                file=sys.stderr,
                flush=True,
            )

    # Keys and LRU order, as lmcache's ObjectKey and its LRU policy.

    def _hashes(self, key: Any, start: int, end: int) -> list[bytes]:
        return self._hasher.compute_chunk_hashes(
            list(key.token_ids), start=start, end=end
        )

    @staticmethod
    def _okey(key: Any, rank: int, h: bytes) -> tuple:
        return (key.model_name, key.world_size, rank, h, key.cache_salt)

    def _size(self, keys) -> int:  # a recurrent state's key is the longer
        s = self.state_bytes
        c = self.chunk_bytes + (0 if self.args.separate_object_groups else s)
        return sum(s if len(k) > 5 else c for k in keys)

    def _ranks(self, key: Any) -> range:
        if key.worker_id is None:
            return range(key.world_size)
        return range(key.worker_id, key.worker_id + 1)

    def _touch(self, keys: list[tuple]) -> None:
        for k in reversed(keys):
            if k in self.objects:
                self.objects.move_to_end(k)

    def _lock(self, keys: list[tuple], n: int, now: float) -> None:
        for k in keys:
            o = self.objects[k]
            if not o.locked(now):
                o.readers = 0
            o.readers += n
            o.lock_until = now + self._ttl

    def _unlock(self, keys: list[tuple], n: int) -> None:
        for k in keys:
            if k in self.objects:
                o = self.objects[k]
                o.readers = max(0, o.readers - n)

    def _session(self, request_id: str, now: float) -> _Session:
        return self._sessions.setdefault(request_id, _Session(now))

    # Request types, as lmcache's LookupModule, ManagementModule and
    # LMCacheDrivenTransferModule answer them.

    def _get_chunk_size(self, *, now: float) -> tuple[Any, Callable]:
        return self.chunk_size, _noop

    def _lookup(self, key: Any, tp_size: int, *, now: float) -> tuple[Any, Callable]:
        readers = key.require_num_kv_readers()
        hashes = self._hashes(key, 0, key.end)
        ranks = self._ranks(key)
        hit = 0
        while hit < len(hashes) and all(
            self._okey(key, r, hashes[hit]) in self.objects for r in ranks
        ):
            hit += 1
        while (
            self._recurrent
            and hit
            and not all(
                (*self._okey(key, r, hashes[hit - 1]), 1) in self.objects for r in ranks
            )
        ):
            hit -= 1
        held = [self._okey(key, r, h) for r in ranks for h in hashes[:hit]]
        if self._recurrent and hit:  # and the recurrent state at its last chunk
            held += [(*self._okey(key, r, hashes[hit - 1]), 1) for r in ranks]
        self._lock(held, readers, now)
        self._touch(held)
        s = self._session(key.request_id, now)
        s.tokens, s.key, s.hit = key.token_ids, key, -1
        self._prefetch[key.request_id] = hit
        return None, _noop, len(hashes)

    def _query_prefetch_status(
        self, request_id: str, *, now: float
    ) -> tuple[Any, Callable]:
        hit = self._prefetch.pop(request_id, None)
        if hit is None:
            return 0, _noop
        self._session(request_id, now).hit = hit
        return hit, _noop

    def _free_lookup_locks(
        self, key: Any, tp_size: int, *, now: float
    ) -> tuple[Any, Callable]:
        if key.start >= key.end:
            return None, _noop
        hit = self._session(key.request_id, now).hit
        first = key.start // self.chunk_size
        hashes = self._hashes(key, key.start, key.end)
        # A hit's recurrent state, when its last chunk is in this range.
        state = hashes[hit - 1 - first : hit - first] if hit > first else []
        if hit >= 0:
            hashes = hashes[: max(0, hit - first)]
        ranks = self._ranks(key)
        held = [self._okey(key, r, h) for h in hashes for r in ranks]
        held += [(*self._okey(key, r, h), 1) for h in state for r in ranks]
        self._unlock(held, key.require_num_kv_readers())
        return None, _noop

    def _end_session(self, request_id: str, *, now: float) -> tuple[Any, Callable]:
        s = self._session(request_id, now)
        del self._sessions[request_id]
        if s.key is not None:
            hashes = self._hasher.compute_chunk_hashes(list(s.tokens))
            ranks = range(s.key.world_size)
            keys = [self._okey(s.key, r, h) for h in hashes for r in ranks]
            self._touch(keys + [(*k, 1) for k in keys])
        return None, _noop

    def _worker_keys(self, key: Any, now: float) -> list[tuple]:
        s = self._session(key.request_id, now)
        s.tokens = key.token_ids
        if s.key is None:
            s.key = key
        return [
            self._okey(key, key.worker_id, h)
            for h in self._hashes(key, key.start, key.end)
        ]

    def _store(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event: bytes,
        *,
        now: float,
    ) -> tuple[Any, Callable]:
        keys = self._worker_keys(key, now)
        # Under --separate-object-groups the block-id groups after the first
        # hold recurrent state, one block a chunk, as ATOM's native-state layout
        # sends it: lmcache keeps it where a block is not the null block.
        state = block_ids[1:] if self.args.separate_object_groups else []
        self._recurrent |= bool(state)
        null = self.args.null_block_id
        states = [(*k, 1) for k, *ids in zip(keys, *state) if set(ids) - {null}]
        used, wanted = self._size(self.objects) + self._size(self._staged), []
        for group in (keys, states):  # one allocation a group: all or nothing
            new = [k for k in group if k not in self.objects and k not in self._staged]
            if used + self._size(new) <= self.capacity:
                used, wanted = used + self._size(new), wanted + new
        self._staged.update(wanted)

        def admit(now: float) -> None:
            self._staged.difference_update(wanted)
            fresh = [k for k in wanted if k not in self.objects]
            for k in fresh:
                self.objects[k] = _Obj()
            for k in reversed(fresh):
                self.objects.move_to_end(k)

        return (b"", True), admit, self._size(wanted) / self.chunk_bytes

    def _retrieve(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event: bytes,
        skip_first_n_tokens: int,
        *,
        now: float,
    ) -> tuple[Any, Callable]:
        keys = self._worker_keys(key, now)
        keys += [(*keys[-1], 1)] if self._recurrent and keys else []
        readable = [
            k for k in keys if k in self.objects and self.objects[k].locked(now)
        ]
        if len(readable) < len(keys):
            self._unlock(readable, 1)
            return (b"", False), _noop
        size = self._size(keys) / self.chunk_bytes
        return (b"", True), lambda now: self._unlock(readable, 1), size


#: The request types the copy engine serves; the request loop serves the rest.
COPIES = ("store", "retrieve")
#: The server's periodic ticks and their intervals.
TICKS = {"evict": EVICTION_INTERVAL_S, "expire_sessions": SESSION_SWEEP_INTERVAL_S}


def reply_channel(ch: str) -> str:
    """The channel a request on `ch` is answered on: its two ends swapped."""
    src, _, rest = ch.partition("->")
    dst, _, kind = rest.partition(":")
    return f"{dst}->{src}:{kind}"


class StationLP:
    """The loop of an LP that plays a server its engines share, the clock owner
    of its process.

    Each stamped request released to it is taken (`_take`) in ``(arrival,
    channel, seq)`` order. A job runs on a station of a stated width: it starts
    (`_start`) when one of the station's servers is free and the server picks it
    (`_next_job`, its queue's head by default), and at its end it finishes and
    sends its reply, if it has one, on its request's channel with the ends
    swapped. The server's own timed events run between job ends and starts
    (`_tick`). The loop asks only NER, `_deadlines`: `t` its next job end, and
    its daemon deadline the server's next tick. While it waits it answers at
    once the frames sent outside the run, which carry no stamp (`_answer`).

    A frame is ``[identity, header, payloads...]``, the header ``(channel,
    arrival, seq)`` as every wrapped socket sends it.
    """

    def __init__(self, rt, sock, width: dict[str, int]):
        import zmq

        self.zmq, self.rt, self.sock = zmq, rt, sock
        self.width = width
        self.queued = {station: deque() for station in width}
        self.busy = dict.fromkeys(width, 0)
        # A heap of (end, start order, station, channel, finish).
        self.jobs: list = []
        self.started = 0
        self.inbox: dict = {}  # (channel, seq) -> (arrival, frames), read, not taken

    def serve(self) -> None:
        """Serve until the ``+inf`` grant."""
        rt, poller = self.rt, self.zmq.Poller()
        poller.register(self.sock, self.zmq.POLLIN)
        poller.register(rt.conn.fileno(), self.zmq.POLLIN)
        while True:
            self._settle()
            self._at(rt.now)
            rt.ask_next_event(*self._deadlines())
            while rt.conn.fileno() not in dict(poller.poll()):
                self._pull()
            if rt.take_grant() == math.inf:
                return

    def _deadlines(self) -> tuple[float, float]:
        """NER's `t` and daemon deadline."""
        return (self.jobs[0][0] if self.jobs else math.inf), math.inf

    def _at(self, now: float) -> None:
        """Everything due at `now`: job ends, the server's ticks, the requests
        released, and the job starts they allow, until no job ends at `now`."""
        released = sorted(
            (a, ch, seq)
            for (ch, seq), (a, _) in self.inbox.items()
            if seq in self.rt.released[ch]
        )
        for arrival, ch, seq in released:
            self._take(ch, arrival, self.inbox.pop((ch, seq))[1])
        while True:
            while self.jobs and self.jobs[0][0] <= now:
                _, _, station, ch, finish = heapq.heappop(self.jobs)
                self.busy[station] -= 1
                if (reply := finish(now)) is not None:
                    back = reply_channel(ch)
                    self._send(back, reply, self.rt.stamp_send(back))
            self._tick(now)
            for station, width in self.width.items():
                while self.busy[station] < width:
                    if (job := self._next_job(station, now)) is None:
                        break
                    ch, end, finish = self._start(station, job, now)
                    heapq.heappush(self.jobs, (end, self.started, station, ch, finish))
                    self.started += 1
                    self.busy[station] += 1
            if not (self.jobs and self.jobs[0][0] <= now):
                return

    def _take(self, ch: str, arrival: float, frames: list[bytes]) -> None:
        """Queue the request `frames` released on `ch` at `arrival`."""
        raise NotImplementedError

    def _next_job(self, station: str, now: float):
        """The next job `station` starts at `now`, or None."""
        queue = self.queued[station]
        return queue.popleft() if queue else None

    def _start(self, station: str, job, now: float) -> tuple:
        """Start `job` at `now`: its request's channel, its end, and a callable
        that finishes it at its end and returns its reply frames or None."""
        raise NotImplementedError

    def _tick(self, now: float) -> None:
        """The server's own events due at `now`."""

    def _answer(self, identity: bytes, ch: str, frames: list[bytes]) -> None:
        """Answer a request sent on `ch` outside the run."""
        raise NotImplementedError

    def _pull(self) -> None:
        """Read every frame already here without blocking: answer one sent
        outside the run, check and hold a stamped one."""
        while True:
            try:
                identity, hdr, *rest = self.sock.recv_multipart(self.zmq.NOBLOCK)
            except self.zmq.Again:
                return
            ch, arrival, seq = pickle.loads(hdr)
            if arrival is None:
                self._answer(identity, ch, rest)
            else:
                self.rt.check_arrival(ch, arrival, seq)
                self.inbox[ch, seq] = (arrival, [identity, *rest])

    def _settle(self) -> None:
        """Wait until every frame released to this LP has physically arrived."""
        while any(self.rt.unread.values()):
            self.sock.poll()
            self._pull()

    def _send(self, ch: str, reply: list[bytes], stamp: tuple) -> None:
        identity, *rest = reply
        self.sock.send_multipart([identity, pickle.dumps((ch, *stamp)), *rest])


class LMCacheLP(StationLP):
    """LP ``lmcache``: the stand-in's one loop, the clock owner of its process.

    Each request released to it joins a station's queue in ``(arrival,
    channel, seq)`` order: the copy engine, `copy_width` servers pricing a
    STORE or RETRIEVE over the chunks it copies as the `TransferModel` that
    `size_copy()` returns at the first stamped request released, after it has
    sized the server's chunk and state image from what the engines it serves
    planned, or the request
    loop, one server pricing a LOOKUP at ``lookup_s[0] + lookup_s[1]`` per
    prompt chunk and any other request at `request_s`. A job is served
    (`LMCacheServer.handle`) at its start and, at its end, applies its effect
    and sends its reply. The loop asks only NER: `t` its next job end, its
    daemon deadline the server's next tick: the eviction loop's
    (`LMCacheServer.evict`), first a second after the server starts, or the
    session sweep's (`LMCacheServer.expire_sessions`), first at its start.
    While it waits it answers at once the frames sent outside the run, which
    carry no stamp: lmcache's startup and teardown requests.

    A frame is ``[identity, header, request id, operation, payloads...]``.
    """

    def __init__(
        self, rt, sock, server, *, lookup_s, request_s, size_copy, copy_width=1
    ):
        from lmcache.v1.multiprocess.transport.zmq_impl import wire

        # ponytail: each copy server moves at the link's whole rate; share it
        # between them if a measured copy engine says they contend.
        super().__init__(rt, sock, {"copy": copy_width, "request": 1})
        self.server, self._wire = server, wire
        self.lookup_s, self.request_s, self._size_copy = lookup_s, request_s, size_copy
        self.copy = None  # the copy engine's model, from the first stamped request
        self.due = {"evict": rt.now + EVICTION_INTERVAL_S, "expire_sessions": rt.now}

    def _deadlines(self) -> tuple[float, float]:
        return super()._deadlines()[0], min(self.due.values())

    def _take(self, ch: str, arrival: float, frames: list[bytes]) -> None:
        if self.copy is None:
            self.copy = self._size_copy()
        op = self._wire.decode_operation(frames[2])
        self.queued["copy" if op in COPIES else "request"].append((ch, op, frames))

    def _tick(self, now: float) -> None:
        for tick, due in self.due.items():
            if now >= due:
                getattr(self.server, tick)(now)
                self.due[tick] += TICKS[tick]

    def _start(self, station: str, job, now: float) -> tuple:
        ch, op, frames = job
        reply, effect, chunks = self.server.handle(frames, now)

        def finish(end: float) -> list[bytes]:
            effect(end)
            return reply

        return ch, now + self._price(op, chunks), finish

    def _price(self, op: str, chunks: int) -> float:
        if op == "lookup":
            return self.lookup_s[0] + self.lookup_s[1] * chunks
        if op in COPIES:
            return self.copy.duration_s(chunks)
        return self.request_s

    def _answer(self, identity: bytes, ch: str, frames: list[bytes]) -> None:
        reply, effect, _ = self.server.handle([identity, *frames], self.rt.now)
        effect(self.rt.now)
        self._send(reply_channel(ch), reply, (None, None))
