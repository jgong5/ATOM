# SPDX-License-Identifier: MIT
"""ATOM's `mooncake_store` connector on a simulated run: its scheduler half as
ATOM runs it, its lookups and renewals sent to LP ``mooncake_store`` (the
master, `atom.compass.kv.mooncake_store_server`), and each rank's puts and
gets priced on that rank's lanes on the engine's LP clock.

`substitute` makes the connector factory build, for `mooncake_store`, the
scheduler half `StoreShell` and the worker half of `compass`, which moves
nothing; a `multi` that lists it builds `MultiShell`
(`atom.compass.kv.lmcache_mp`) as its scheduler half.

`StoreShell` wraps ATOM's `MooncakeStoreOffloadScheduler`, built under
`simulated_arch` so the Store's namespace names the run's device. Its
`AsyncStoreLookup` runs no thread: a query is sent on the engine's
``lookup#dp<N>`` channel when `submit` is called, and the step loop's next poll
(`pump`, `is_pending`, `poll`, `answered`, `lookup`) takes each reply released
by then, runs `_StoreLookup.lookup` there over a client whose ``exists``
answers as the reply did, and sets the query's future; the query's
``asked_at`` is the reply's start less a hop, when the real client sent it. A
renewal waits for its reply at an idle point bounded by the scheduler's
timeout, a daemon deadline. Every send is the clock owner's
(`clock.send_on_owner`), an offer the input thread makes included.

Each load and save `build_connector_meta` hands a TP rank waits on the rank's
`Lanes` and runs at its priced end through ATOM's own `_do_load_req` or
`_do_save_req`, over stand-ins for the slot pool, the GPU copy and the Store
client. A save sends its put on ``transfer#dp<N>`` as it is queued, carrying
its keys, their groups and the lane's start and end; a load's get finds each
key as the request's last lookup or renewal reply did, all or nothing. A load
whose lane starts past that reply's lease deadline is refused by name: it
joins the engine's refusals, and ends a run whose ``on_refusal`` is
``abort``. `offload_statistics` counts the lookups, renewals, loads, loads of
another engine's saves, saves and past-lease loads.

Importing this module loads no third-party package; using it loads zmq and
torch.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import pickle
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

from atom.compass.kv import mooncake_store_server as server
from atom.compass.kv.lmcache_offload import Lanes
from atom.compass.kv.lmcache_server import reply_channel

logger = logging.getLogger(__name__)

ENTRY = server.ENTRY
#: What `offload_statistics` counts, each from zero.
COUNTS = (
    "lookups",
    "renewals",
    "loads",
    "foreign_loads",
    "saves",
    "past_lease_loads",
)


def substitute() -> None:
    """Build this module's scheduler half for every `mooncake_store` connector
    from now on, and `compass`'s worker half."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    registry = KVConnectorFactory._registry
    registry[ENTRY] = registry[ENTRY] | {
        "worker_module": "atom.compass.kv.connector",
        "worker_class": "SimulatedKVConnector",
        "scheduler_module": __name__,
        "scheduler_class": StoreShell.__name__,
    }
    registry["multi"] = registry["multi"] | {
        "scheduler_module": "atom.compass.kv.lmcache_mp",
        "scheduler_class": "MultiShell",
    }


class StoreRefusal(RuntimeError):
    """A Store call the shell refuses by name in a run that aborts on one."""


class _Executor:
    """`AsyncStoreLookup`'s query or renewal worker, which starts no thread: a
    task is a call to LP mooncake_store (`StoreShell._ask`)."""

    def __init__(self, shell: StoreShell, name: str, limit=None) -> None:
        kinds = {"store-query": "query", "store-renew": "renew"}
        if name not in kinds:
            raise ValueError(f"the Store's shell runs no {name!r} worker")
        self.shell, self.kind = shell, kinds[name]

    def submit(self, fn, seq, *rest):
        """`fn` is ``_query(seq, sid)`` or ``_renew(seq, start, end)``."""
        return self.shell._ask(self.kind, seq, *(rest if self.kind == "renew" else ()))

    def shutdown(self) -> None:
        """Nothing runs to stop."""


class _Renewal(Future):
    """A renewal's future: `result` is an idle point on the LP clock, until
    its reply is taken or its timeout, a daemon deadline, passes. It counts as
    running from its send, so a cancel leaves it to its reply, as a call in
    flight is."""

    def __init__(self, shell: StoreShell) -> None:
        super().__init__()
        self.shell = shell
        self.set_running_or_notify_cancel()

    def result(self, timeout=None):
        rt = self.shell._rt
        if not self.done() and rt.in_run and threading.current_thread() is rt.owner:
            end = rt.now + (math.inf if timeout is None else timeout)
            self.shell._take()
            while not self.done() and rt.now < end:
                rt.next_event(math.inf, end)
                self.shell._take()
            if not self.done():
                raise FutureTimeout
        return super().result(timeout=0)


def _polled():
    """`AsyncStoreLookup`, taking the replies released before each poll."""
    from atom.kv_transfer.offload.mooncake_store.async_lookup import (
        AsyncStoreLookup,
    )

    class Polled(AsyncStoreLookup):
        shell = None  # the `StoreShell`, set once the scheduler half is built

        def pump(self):
            self.shell._take()
            return super().pump()

        def is_pending(self, sid):
            self.shell._take()
            return super().is_pending(sid)

        def answered(self, sid):
            self.shell._take()
            return super().answered(sid)

        def poll(self, sid):
            self.shell._take()
            return super().poll(sid)

        def lookup(self, seq, lookup_id=None):
            self.shell._take()
            return super().lookup(seq, lookup_id)

    return Polled


class _Client:
    """A Store client: `exists` answers as `present`, the reply being taken,
    and `get` as the request's last reply (each key's chunk bytes, or not
    found); `put` takes every key."""

    master_server_addr = "LP mooncake_store"

    def __init__(self, chunk_bytes: int = 0) -> None:
        self.chunk_bytes, self.present = chunk_bytes, {}

    def exists(self, keys: list[str]) -> list[int]:
        return [int(key in self.present) for key in keys]

    def put(self, keys, ptrs, sizes, *, group_ids=None) -> list[int]:
        return [0] * len(keys)

    def get(self, keys, ptrs, sizes) -> list[int]:
        from atom.kv_transfer.offload.mooncake_store.client import OBJECT_NOT_FOUND

        return [
            self.chunk_bytes if key in self.present else OBJECT_NOT_FOUND
            for key in keys
        ]

    def stats(self) -> dict:
        return {"counts": {}, "failures": {}}


class _Pool:
    """A rank's transfer slot pool, its regions as many slots as the entry
    states, handing out slots at once."""

    device = "cpu"

    def __init__(self, slots: dict[str, int]) -> None:
        self.slots = slots

    def window(self, region: str, n_threads: int) -> int:
        from atom.kv_transfer.offload.mooncake_store.pool import (
            _MIN_WINDOWS_PER_REGION,
        )

        return max(1, self.slots[region] // max(_MIN_WINDOWS_PER_REGION, n_threads))

    def acquire(self, region: str, n: int) -> list:
        return [SimpleNamespace(ptr=0) for _ in range(n)]

    def release(self, slots) -> None:
        """Nothing is held."""

    def quarantine(self, slots, reason: str = "") -> None:
        raise RuntimeError(f"a simulated slot is never quarantined: {reason}")

    def quarantined(self, region: str) -> int:
        return 0

    def contiguous_view(self, slots):
        return None

    def strided_view(self, slots):
        return None


class _Copies:
    """The block GPU connector, which copies nothing: a pack reports each
    chunk source-safe, last chunk first, as the real one does once a group's
    copy has fenced."""

    def __init__(self, source_safe) -> None:
        self.source_safe, self.operation = source_safe, None

    @contextlib.contextmanager
    def track_save_source(self, operation):
        self.operation, previous = operation, self.operation
        try:
            yield
        finally:
            self.operation = previous

    def batched_from_gpu(self, slots, starts, ends, *, block_ids, producer_event=None):
        from atom.kv_transfer.disaggregation.types import (
            SaveOperationId,
            SaveSourceGroupId,
        )

        if self.source_safe and isinstance(self.operation, SaveOperationId):
            for start, end in sorted(zip(starts, ends, strict=True), reverse=True):
                self.source_safe(SaveSourceGroupId(self.operation, ((start, end),)))

    def batched_to_gpu(self, slots, starts, ends, *, block_ids) -> None:
        """Nothing to unpack."""

    def close(self) -> None:
        """Nothing to release."""


class StoreShell:
    """ATOM's `mooncake_store` scheduler half, its master LP mooncake_store
    and its ranks' copies on the step loop."""

    def __init__(self, config) -> None:
        import zmq

        from atom.compass import run as compass_run
        from atom.compass.runner.overrides import simulated_arch
        from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
        from atom.kv_transfer.offload import config as offcfg
        from atom.kv_transfer.offload.mooncake_store import async_lookup
        from atom.kv_transfer.offload.mooncake_store import scheduler as store
        from atom.kv_transfer.offload.mooncake_store.worker import (
            MooncakeStoreOffloadConnector,
        )
        from atom.utils import clock

        self._rt = rt = clock.installed()
        if rt is None:
            raise RuntimeError("no LP runtime is installed to run the Store on")
        run = compass_run.spec()
        entry = run[ENTRY]
        server.check_entry(entry)
        self._hop = server.store_rpc_s(entry)
        self._abort = run.get("on_refusal") == "abort"
        executor, lookup_class = async_lookup.DaemonExecutor, store.AsyncStoreLookup
        async_lookup.DaemonExecutor = functools.partial(_Executor, self)
        store.AsyncStoreLookup = _polled()
        try:
            with simulated_arch():
                self._inner = store.MooncakeStoreOffloadScheduler(config)
        finally:
            async_lookup.DaemonExecutor = executor
            store.AsyncStoreLookup = lookup_class
        self._lookups = self._inner._lookup_client
        self._lookups.shell = self
        self._store = self._lookups.backend
        self._answers = _Client()  # the scheduler's pools all ask this one
        for pool in self._store._pools:
            pool.client = self._answers
        self._namespace, self._chunk = self._inner._namespace, self._inner.chunk_size
        self._world = offcfg.lmcache_replica_world_size(config)
        self._me = rt.me.name
        dp = config.parallel_config.data_parallel_rank
        self._ask_ch = f"{self._me}->{server.ENTRY}:lookup#dp{dp}"
        self._back = reply_channel(self._ask_ch)
        self._put_ch = f"{self._me}->{server.ENTRY}:transfer#dp{dp}"
        where = compass_run.mooncake_store_endpoint(run)
        self._socks = {}
        for ch in (self._ask_ch, self._put_ch):
            self._socks[ch] = zmq.Context.instance().socket(zmq.DEALER)
            self._socks[ch].connect(where)
        self._ranks = []
        slots = {"save": entry["save_slots"], "load": entry["load_slots"]}
        for k in range(self._world):
            rank = MooncakeStoreOffloadConnector(config)
            rank._rank, rank._world = k, self._world
            rank._namespace, rank._chunk_bytes = self._namespace, entry["chunk_bytes"]
            rank._pool, rank._client = _Pool(slots), _Client(entry["chunk_bytes"])
            safe = rank._source_group_safe if rank._early_release else None
            rank._gpu_connector = _Copies(safe)
            self._ranks.append(rank)
        self._lanes = Lanes(
            ENTRY,
            [(r.load_workers, r.save_workers) for r in self._ranks],
            {"load_s": entry["load_s"], "store_s": entry["store_s"]},
            rt.read_clock,
        )
        self._calls: dict = {}  # request id -> (kind, seq, span, future), in flight
        self._ids = itertools.count()
        self._read: dict = {}  # reply seq -> (arrival, payload), read, not taken
        # sid -> (present keys and who put each, lease deadline): its last reply.
        self._replies: dict = {}
        self._counts = collections.Counter(dict.fromkeys(COUNTS, 0))
        self._tp = KVOutputAggregator(world_size=self._world)
        self._held = None  # what the ranks reported at the previous call

    def __getattr__(self, name: str):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # -- lookups and renewals -------------------------------------------------

    def _keys(self, seq, *span) -> list[str]:
        """Every rank's keys of `seq`'s prompt chunks, or of tokens ``span``,
        as `_StoreLookup` asks them."""
        from atom.kv_transfer.offload.mooncake_store.keys import (
            DIGEST_BYTES,
            chunk_keys,
        )
        from atom.kv_transfer.offload.mooncake_store.scheduler import (
            prompt_chunk_hashes,
        )

        hashes = prompt_chunk_hashes(seq, self._chunk)
        first, last = 0, len(hashes) // DIGEST_BYTES
        if span:
            first, last = int(span[0]) // self._chunk, min(
                int(span[1]) // self._chunk, last
            )
        if first >= last:
            return []
        ns, world = self._namespace, self._world
        return [
            key
            for rank in range(world)
            for key in chunk_keys(ns, rank, world, hashes, first, last)
        ]

    def _ask(self, kind: str, seq, *span) -> Future:
        """Send a query, or a renewal of tokens ``span``, for `seq`'s chunks;
        its future. One asking for no chunk is answered at once, as the real
        call sends nothing."""
        from atom.kv_transfer.offload.mooncake_store.keys import STORE_TAINT_KEY
        from atom.utils import clock

        future = _Renewal(self) if kind == "renew" else Future()
        keys = self._keys(seq, *span)
        if not keys:
            future.set_result(self._answer(kind, seq, span, {}))
            return future
        rid = f"{kind}-{next(self._ids)}"
        self._calls[rid] = (kind, seq, span, future)
        call = (kind, rid, [STORE_TAINT_KEY, *keys])
        clock.send_on_owner(lambda: self._send(self._ask_ch, call))
        return future

    def _answer(self, kind: str, seq, span, present: dict):
        """ATOM's lookup or renewal of `seq`, its pools answering `present`."""
        self._answers.present = present
        try:
            if kind == "query":
                return self._store.lookup(seq, lookup_id=str(seq.id))
            return self._store.renew(seq, *span)
        finally:
            self._answers.present = {}

    def _send(self, ch: str, payload) -> None:
        rt = self._rt
        stamp = rt.stamp_send(ch) if rt.in_run else (None, None)
        self._socks[ch].send_multipart(
            [pickle.dumps((ch, *stamp)), pickle.dumps(payload)]
        )

    def _take(self) -> None:
        """On the step loop: take every reply released so far, in arrival
        order, first waiting for any released one still on the wire."""
        import zmq

        rt, back = self._rt, self._back
        if threading.current_thread() is not rt.owner or not rt.in_run:
            return
        sock = self._socks[self._ask_ch]
        while True:
            while True:
                try:
                    hdr, payload = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                ch, arrival, seq = pickle.loads(hdr)
                rt.check_arrival(ch, arrival, seq)
                self._read[seq] = (arrival, payload)
            with rt.lock:
                if rt.released[back] <= rt.arrived[back]:
                    break
            sock.poll()
        with rt.lock:
            due = sorted(
                (a, s) for s, (a, _) in self._read.items() if s in rt.released[back]
            )
            for _, s in due:
                rt.count_done_locked(back, s)
        for _, s in due:
            self._taken(*pickle.loads(self._read.pop(s)[1]))

    def _taken(self, rid: str, start: float, lease: float, present: dict) -> None:
        """A reply: ATOM's lookup or renewal over its answer sets the call's
        future, a query's ``asked_at`` its start less a hop."""
        kind, seq, span, future = self._calls.pop(rid)
        sid = str(seq.id)
        self._replies[sid] = (present, lease)
        self._counts["lookups" if kind == "query" else "renewals"] += 1
        keys = self._keys(seq, *span)
        logger.info(
            "compass: mooncake_store %s of %s from %r: %d of its %d keys present, "
            "%d put by another engine, lease to %r; first key %s",
            kind,
            sid,
            start,
            sum(k in present for k in keys),
            len(keys),
            sum(present.get(k, self._me) != self._me for k in keys),
            lease,
            keys[0],
        )
        if future.cancelled():
            return
        try:
            result = self._answer(kind, seq, span, present)
        except Exception as e:  # noqa: BLE001  # the poll's, as a real worker's
            future.set_exception(e)
            return
        if kind == "query":
            lookups = self._lookups
            with lookups.lock:
                lookups.asked_at[sid] = start - self._hop
                lookups.asked_at.move_to_end(sid)
                while len(lookups.asked_at) > 2 * lookups.capacity:
                    lookups.asked_at.popitem(last=False)
        future.set_result(result)

    # -- copies ---------------------------------------------------------------

    def build_connector_meta(self):
        """ATOM's metadata, each rank's share queued on its lanes: a save's
        put sent as it is queued, a load checked against its lease."""
        meta = self._inner.build_connector_meta()
        self._lanes.run_due()
        for req in meta.requests:
            if req.load_spec is not None:
                self._load(req)
            if req.save_spec is not None:
                self._save(req)
        return meta

    def _save(self, req) -> None:
        from atom.kv_transfer.offload.mooncake_store.keys import (
            DIGEST_BYTES,
            chunk_group_ids,
            chunk_keys,
        )
        from atom.utils import clock

        hashes = req.chunk_hashes or b""
        first = int(req.save_spec.skip_leading_tokens) // self._chunk
        end = len(hashes) // DIGEST_BYTES
        chunks = max(0, end - first)
        for k, rank in enumerate(self._ranks):
            if not rank._do_save:
                continue
            run = functools.partial(rank._guard, "save", rank._do_save_req, req)
            start, stop = self._lanes.queue(k, "save", chunks, req.req_id, run)
            if not chunks:
                continue
            keys = chunk_keys(self._namespace, k, self._world, hashes, first, end)
            groups = chunk_group_ids(self._namespace, hashes, first, end)
            put = ("put", keys, groups, start, stop)
            clock.send_on_owner(lambda put=put: self._send(self._put_ch, put))
            if k == 0:
                self._counts["saves"] += 1

    def _load(self, req) -> None:
        from atom.kv_transfer.offload.mooncake_store.keys import chunk_keys

        ls, chunk = req.load_spec, self._chunk
        hbm, lmc = int(ls.hbm_cached_tokens), int(ls.lmcache_cached_tokens)
        end = lmc if ls.transfer_end_tokens is None else int(ls.transfer_end_tokens)
        chunks = (end - hbm) // chunk if lmc > hbm and not hbm % chunk else 0
        present, lease = self._replies.pop(str(req.req_id), ({}, -math.inf))
        for k, rank in enumerate(self._ranks):
            if not rank._do_load:
                continue

            def run(rank=rank):
                rank._client.present = present
                try:
                    rank._guard("load", rank._do_load_req, req)
                finally:
                    rank._client.present = {}

            start, _ = self._lanes.queue(k, "load", chunks, req.req_id, run)
            if k or not chunks:
                continue
            self._counts["loads"] += 1
            keys = chunk_keys(
                self._namespace,
                0,
                self._world,
                req.chunk_hashes,
                hbm // chunk,
                end // chunk,
            )
            if any(present.get(key, self._me) != self._me for key in keys):
                self._counts["foreign_loads"] += 1
            if start > lease:
                self._past_lease(req, start, lease)

    def _past_lease(self, req, start: float, lease: float) -> None:
        """Refuse a load whose lane starts past its reply's lease: whether the
        master still held its objects then is what the get would have asked."""
        self._counts["past_lease_loads"] += 1
        why = (
            f"store:a load of {req.req_id} starts at {start!r} s, past the lease "
            f"its last Store reply gave it, to {lease!r} s"
        )
        self._rt.refusals.append(why)
        logger.warning("compass: mooncake_store refused %s", why)
        if self._abort:
            raise StoreRefusal(why)

    def process_completions(self, output):
        """Run the copies due; report what the ranks gave at the previous
        call, then let ATOM's scheduler half read it."""
        self._lanes.run_due()
        held = self._tp.aggregate([rank.get_finished() for rank in self._ranks])
        held, self._held = self._held, held
        for field in dataclasses.fields(held) if held is not None else ():
            value = getattr(held, field.name)
            if isinstance(value, set):
                getattr(output, field.name).update(value)
        return self._inner.process_completions(output)

    def has_pending_work(self) -> bool:
        held = self._held is not None and not self._held.is_empty()
        return (
            bool(self._lanes.due or self._calls)
            or held
            or self._inner.has_pending_work()
        )

    def offload_statistics(self) -> dict:
        """ATOM's counters, rank 0's priced copies, and the Store calls."""
        return {
            "connector": self._inner.get_statistics(),
            "copies": self._lanes.copies(),
            "store": dict(self._counts),
        }

    def close(self) -> None:
        self._inner.close()
        for sock in self._socks.values():
            sock.close(linger=0)
