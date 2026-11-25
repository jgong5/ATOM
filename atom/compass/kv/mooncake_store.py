# SPDX-License-Identifier: MIT
"""ATOM's `mooncake_store` offload on a simulated run: its scheduler half as
ATOM runs it, its Store calls sent to LP ``mooncake_store`` and its copies
priced on each writer rank's lanes.

`substitute` makes the connector factory build, for `mooncake_store`, the
scheduler half `StoreShell` and a worker half that moves nothing,
`SimulatedStoreWorker`, whose HBM reserve is the real worker's
(`MooncakeStoreOffloadConnector.kv_budget_reserve_bytes`), so a runner sizes
its KV pool as a real one does whenever it substitutes. A `multi` that lists
it builds `MultiShell` (`atom.compass.kv.lmcache_mp`) as its scheduler half.

`StoreShell` wraps ATOM's `MooncakeStoreOffloadScheduler`, built under
`runner.overrides.simulated_arch`, so the Store's namespace names the run's
device. Its lookup client is ATOM's `AsyncStoreLookup` as `ClockedLookup`:
its query and renewal workers are replaced, and its pool's Store client is a
stand-in, `StandInClient`, whose ``exists`` answers as the master's reply did.

- A query is sent on ``<engine>->mooncake_store:lookup#dp<N>`` when
  ``submit`` is called, carrying every rank's keys of the prompt's chunks, as
  `_StoreLookup.lookup` asks them. Its reply is taken at the step loop's next
  poll of the lookup client (``pump``, ``is_pending``, ``poll``, ``answered``,
  ``lookup``): `_StoreLookup.lookup` runs there over the reply, and the
  query's ``asked_at`` is the reply's start at the master less one hop.
- A renewal (``renew``) is a request and reply on the same channel. Its wait,
  ``future.result(timeout=RENEW_TIMEOUT_S)``, is an idle point of the step
  loop whose bound is a daemon deadline; ATOM's ``_renew`` runs over the reply.
  In a DP group the members park before their forwards (`run.engine`).
- An offer from the engine's input thread (`offer_lookups`) cannot send: only
  the clock owner stamps a send in a run. It is handed to the owner, through
  #3767's `LPRuntime.offer` where the runtime has it, and made there
  (`StoreShell._send_lookup`). The engine's request channels are offering
  (`add_mooncake_store`), so under #3767 (b) an arrival inside a forward is
  granted first and its lookup stamped at that arrival.

Each writer rank (``world`` of them, ATOM's replica world) keeps an ATOM
`MooncakeStoreOffloadConnector` with no device: a stand-in slot pool
(`StandInPool`), GPU copy (`StandInCopies`) and the shared client. A save or
load that `build_connector_meta` hands a rank waits on that rank's save or
load lanes (`lmcache_offload.Lanes`), priced ``store_s`` or ``load_s`` over
its chunks, and ATOM's ``_do_save_req`` or ``_do_load_req`` runs at its priced
end. As a save is queued its put goes to LP ``mooncake_store`` on
``transfer#dp<N>`` with the rank's keys, each chunk's group and the lane's
start and end, and gets no reply. A load reads what the request's last lookup
or renewal reply found, all or nothing; one whose lane starts past that
reply's lease is refused by name (`PAST_LEASE`) and counted, and stops only a
run whose ``on_refusal`` is ``abort``.

`process_completions` reports what the ranks' `get_finished`, aggregated over
the TP group, gave at the previous call, as the engine's aggregation answers
one poll behind.

Under a pipelined prefill the writer ranks are every stage's (``world`` =
stages x TP, ATOM's `pp_aware_rank_and_world`), all kept by the head stage's
shell, the one that schedules: a request's lookup asks every rank's keys, a
load is all or nothing over them, and each rank's copies run on its own lanes
at its own chunk bytes, slots and laws where the entry states them a rank
(`mooncake_store_server.PER_RANK`). A later stage's shell is inert: its
scheduler never runs, as ATOM's does not. Importing this module loads no third-party package; using it
loads zmq and torch, and ATOM's Store modules.
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
from pathlib import Path
from types import SimpleNamespace

from atom.compass.kv.connector import SimulatedKVConnector

logger = logging.getLogger(__name__)

#: The run-file key of the entry (`mooncake_store_server.ENTRY`).
ENTRY = "mooncake_store"
#: What the shell counts: queries and renewals sent, rank 0's loads, those that
#: read another instance's saves, its saves of one chunk or more, loads (of any
#: rank) whose lane starts past their lease, and arrival-time offers, with the
#: offers the owner stamped later than their message's arrival.
COUNTS = (
    "lookups",
    "renewals",
    "loads",
    "foreign_loads",
    "saves",
    "past_lease_loads",
    "offers",
    "offers_late",
)
#: The refusal a load whose lane starts past its lease hands in.
PAST_LEASE = "mooncake_store:a load starts past its lease"
#: Wall seconds a shell waits for LP mooncake_store to name its endpoint, and
#: how often it looks.
ENDPOINT_WAIT_S, ENDPOINT_LOOK_S = 300.0, 0.05


def substitute() -> None:
    """Build this module's halves for every `mooncake_store` connector from now
    on, and `MultiShell` for a `multi`."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    registry = KVConnectorFactory._registry
    registry["mooncake_store"] = registry["mooncake_store"] | {
        "worker_module": __name__,
        "worker_class": SimulatedStoreWorker.__name__,
        "scheduler_module": __name__,
        "scheduler_class": StoreShell.__name__,
    }
    registry["multi"] = registry["multi"] | {
        "scheduler_module": "atom.compass.kv.lmcache_mp",
        "scheduler_class": "MultiShell",
    }


class SimulatedStoreWorker(SimulatedKVConnector):
    """`compass`'s worker half, which moves nothing, holding back the HBM the
    real Store worker allocates once the KV cache is sized."""

    @classmethod
    def kv_budget_reserve_bytes(cls, config) -> int:
        from atom.kv_transfer.offload.mooncake_store.worker import (
            MooncakeStoreOffloadConnector,
        )

        return MooncakeStoreOffloadConnector.kv_budget_reserve_bytes(config)


class StandInClient:
    """Each pool's Store client, the lookups' and every rank's: it answers as
    the master's reply the shell is taking said (`answer`, key -> the engine
    LP that published it). A put stores, a get reads all of what the answer
    holds, and the taint key is never present."""

    def __init__(self, master: str, chunk_bytes: int, size=None) -> None:
        from atom.kv_transfer.offload.mooncake_store import client

        self.master_server_addr, self.chunk_bytes = master, chunk_bytes
        # A key's bytes, where they differ by rank (`Master.size`).
        self.size = size or (lambda key: chunk_bytes)
        self.missing = client.OBJECT_NOT_FOUND
        self.answer: dict[str, str] = {}

    def exists(self, keys):
        return [int(k in self.answer) for k in keys]

    def put(self, keys, ptrs, sizes, group_ids=None):
        return [0] * len(keys)

    def get(self, keys, ptrs, sizes):
        return [self.size(k) if k in self.answer else self.missing for k in keys]

    def stats(self) -> dict:
        return {"counts": {}, "failures": {}}

    def register(self, ptr, nbytes) -> None:
        """Nothing to register: no pool memory exists."""

    def close(self) -> None:
        """Nothing to close."""


class StandInPool:
    """A worker's transfer slot pool, its regions `save_slots` and `load_slots`
    as the entry states them: windows as ATOM cuts them, slots that hold no
    bytes, none ever busy or quarantined."""

    device = "cpu"

    def __init__(self, entry: dict, rank: int = 0) -> None:
        from atom.compass.kv.mooncake_store_server import of_rank

        self._capacity = {
            "save": of_rank(entry, "save_slots", rank),
            "load": of_rank(entry, "load_slots", rank),
        }
        self.chunk_bytes = of_rank(entry, "chunk_bytes", rank)
        self.slot_stride = self.chunk_bytes

    def capacity(self, region: str) -> int:
        return self._capacity[region]

    def usable(self, region: str) -> int:
        return self._capacity[region]

    def quarantined(self, region: str) -> int:
        return 0

    def window(self, region: str, n_threads: int) -> int:
        from atom.kv_transfer.offload.kv_store.pool import _MIN_WINDOWS_PER_REGION

        return max(1, self.usable(region) // max(_MIN_WINDOWS_PER_REGION, n_threads))

    def acquire(self, region: str, n: int) -> list:
        return [SimpleNamespace(ptr=0, region=region) for _ in range(n)]

    def release(self, slots) -> None:
        pass

    def quarantine(self, slots, *, reason: str = "") -> None:
        pass

    def contiguous_view(self, slots):
        return None

    def strided_view(self, slots):
        return None

    def close(self) -> None:
        pass


class StandInCopies:
    """A worker's block GPU connector, which copies nothing: a save's pack
    reports each chunk of its window source-safe, highest first, as the real
    one does once its copy is done."""

    def __init__(self, source_safe) -> None:
        self.source_safe, self.operation = source_safe, None

    @contextlib.contextmanager
    def track_save_source(self, operation):
        self.operation, previous = operation, self.operation
        try:
            yield
        finally:
            self.operation = previous

    def batched_from_gpu(self, slots, starts, ends, **kwargs) -> None:
        from atom.kv_transfer.disaggregation.types import (
            SaveOperationId,
            SaveSourceGroupId,
        )

        if self.source_safe and isinstance(self.operation, SaveOperationId):
            for s, e in sorted(zip(starts, ends), reverse=True):
                self.source_safe(SaveSourceGroupId(self.operation, ((s, e),)))

    def batched_to_gpu(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _Executor:
    """`AsyncStoreLookup`'s query or renewal worker: a submitted call is sent
    to LP mooncake_store, and its future set when the step loop takes the
    reply."""

    def __init__(self, shell: StoreShell, kind: str) -> None:
        self.shell, self.kind = shell, kind

    def submit(self, fn, *args):
        return self.shell._call(self.kind, fn, args)

    def shutdown(self) -> None:
        """No thread to stop."""


class _Reply(Future):
    """A Store call's future. On the step loop in a run a wait on it is an
    idle point until its reply is taken or the wait's bound, a daemon
    deadline that keeps no run alive."""

    def __init__(self, shell: StoreShell) -> None:
        super().__init__()
        self.shell = shell

    def result(self, timeout=None):
        rt = self.shell._rt
        if not self.done() and rt.in_run and threading.current_thread() is rt.owner:
            end = rt.now + (math.inf if timeout is None else timeout)
            while True:
                self.shell._take()
                if self.done() or rt.now >= end:
                    break
                rt.next_event(math.inf, end)
            if not self.done():
                raise FutureTimeout
        return super().result(timeout)


def _clocked_lookup():
    """`ClockedLookup`, defined once ATOM's `AsyncStoreLookup` may be imported."""
    from atom.kv_transfer.offload.mooncake_store import async_lookup

    class ClockedLookup(async_lookup.AsyncStoreLookup):
        """ATOM's `AsyncStoreLookup`, its workers the shell's sends, each poll
        taking the replies released so far first."""

        def __init__(self, shell: StoreShell, backend, capacity=4096):
            made = async_lookup.DaemonExecutor
            kinds = iter(("query", "renew"))
            async_lookup.DaemonExecutor = lambda *a, **k: _Executor(shell, next(kinds))
            try:
                super().__init__(backend, capacity)
            finally:
                async_lookup.DaemonExecutor = made
            self.shell = shell

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

    return ClockedLookup


class StoreShell:
    """ATOM's `mooncake_store` scheduler half, its Store calls on LP
    ``mooncake_store`` and its ranks' copies on the step loop."""

    def __init__(self, config) -> None:
        from atom.compass import run as compass_run
        from atom.compass.kv import mooncake_store_server as store
        from atom.compass.kv.lmcache_offload import Lanes
        from atom.compass.kv.lmcache_server import reply_channel
        from atom.compass.runner.overrides import simulated_arch
        from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
        from atom.kv_transfer.offload.mooncake_store import scheduler
        from atom.kv_transfer.offload.mooncake_store.worker import (
            MooncakeStoreOffloadConnector,
        )
        from atom.utils import clock

        self._rt = rt = clock.installed()
        if rt is None:
            raise RuntimeError("no LP runtime is installed to run the Store's calls on")
        if getattr(config.parallel_config, "pipeline_parallel_rank", 0):
            # A later pipeline stage: the head's shell keeps its ranks. ATOM's
            # half is built, for what a composite asks of it (`is_offload`,
            # ...), and never asked to schedule or look up, as ATOM's is not.
            self._inert = True
            with simulated_arch():
                self._inner = scheduler.MooncakeStoreOffloadScheduler(config)
            return
        run = compass_run.spec()
        entry = run[ENTRY]
        flags = store.check_entry(entry)
        self._ttl, self._hop = flags.default_kv_lease_ttl, store.store_rpc_s(entry)
        self._abort = run.get("on_refusal", "mark") == "abort"
        self._endpoint = Path(run["out_dir"]) / store.ENDPOINT_FILE
        me, dp = rt.me.name, config.parallel_config.data_parallel_rank
        self._me = me
        self._look = f"{me}->{ENTRY}:lookup#dp{dp}"
        self._xfer = f"{me}->{ENTRY}:transfer#dp{dp}"
        self._back = reply_channel(self._look)
        sent = {c.name for c in rt.table.channels_from(rt.me)}
        if missing := sorted({self._look, self._xfer} - sent):
            raise ValueError(
                f"{me} offloads KV to a Mooncake Store and the run's channel table "
                f"has no {', '.join(missing)}: LP mooncake_store serves the "
                "prefill side's engines, or the run's one engine"
            )
        made, lookups = scheduler.AsyncStoreLookup, _clocked_lookup()
        scheduler.AsyncStoreLookup = lambda backend, capacity=4096: lookups(
            self, backend, capacity
        )
        try:
            with contextlib.suppress(ImportError):
                # Loaded before the swap, so the namespace reads the run's arch.
                import aiter.jit.utils.chip_info  # noqa: F401
            with simulated_arch():
                self._inner = scheduler.MooncakeStoreOffloadScheduler(config)
        finally:
            scheduler.AsyncStoreLookup = made
        self._lookup = self._inner._lookup_client
        backend = self._lookup.backend
        # One master, or per-NIC pools LP mooncake_store plays as one Store a
        # rank a pool (`Pools`): a lookup asks it once for the keys each pool
        # would be asked, at once, for.
        pool = backend._pools[0]
        backend._pools = [pool]
        pool.executor = None
        self._client = StandInClient(
            pool.pool.master, entry["chunk_bytes"], store.master_of(entry).size
        )
        pool.client = self._client
        self.chunk, self._world = self._inner.chunk_size, backend._world
        self._namespace = self._inner._namespace
        self._ranks = []
        for k in range(self._world):
            rank = MooncakeStoreOffloadConnector(config)
            rank._rank, rank._world, rank._namespace = k, self._world, self._namespace
            rank._chunk_bytes = store.of_rank(entry, "chunk_bytes", k)
            rank._in_place = False
            rank._client, rank._pool = self._client, StandInPool(entry, k)
            safe = rank._source_group_safe if rank._early_release else None
            rank._gpu_connector = StandInCopies(safe)
            self._ranks.append(rank)
        terms = {name: entry[name] for name in store.TERMS}
        self._lanes = Lanes(rt, self._ranks, terms, ENTRY)
        self._tp = KVOutputAggregator(world_size=self._world)
        self._held = None  # what the ranks reported at the previous call
        self._socks: dict = {}  # channel -> its DEALER
        self._read: dict = {}  # reply seq -> (arrival, payload), untaken
        self._calls: dict = {}  # call id -> (kind, fn, args, future, keys)
        self._ids = itertools.count()
        # Per lookup id, its last reply's lease deadline and present keys.
        self._leases: collections.OrderedDict = collections.OrderedDict()
        # #3767's owner-stamped send, when the runtime has it; else the
        # offers the shell queues itself for the owner's next clock call.
        self._offer = getattr(rt, "offer", None)
        self._offered: list = []
        self._lock = threading.Lock()
        self.counts = dict.fromkeys(COUNTS, 0)
        self.offered_late_s = 0.0  # the latest offer's lateness, at most
        if self._offer is None:
            self._install_flush()

    def __getattr__(self, name: str):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    # -- the Store's calls ---------------------------------------------------

    def _keys(self, seq, first: int, last: int | None) -> list[str]:
        """Every rank's keys of `seq`'s prompt chunks ``[first, last)``, as
        `_StoreLookup` asks them; to the last whole chunk with `last` None."""
        from atom.kv_transfer.offload.mooncake_store.keys import (
            DIGEST_BYTES,
            chunk_keys,
        )
        from atom.kv_transfer.offload.mooncake_store.scheduler import (
            prompt_chunk_hashes,
        )

        hashes = prompt_chunk_hashes(seq, self.chunk)
        chunks = len(hashes) // DIGEST_BYTES
        last = chunks if last is None else min(last, chunks)
        keys = []
        for rank in range(self._world):
            keys += chunk_keys(self._namespace, rank, self._world, hashes, first, last)
        return keys

    def _call(self, kind: str, fn, args) -> Future:
        """A query (`fn` ``_query``, `args` the sequence and its lookup id) or a
        renewal (``_renew``, the sequence and its tokens' range) the lookup
        client submits: sent, its future set when the reply is taken."""
        future, rt, seq = _Reply(self), self._rt, args[0]
        if kind == "query":
            keys = self._keys(seq, 0, None)
        else:
            keys = self._keys(seq, args[1] // self.chunk, args[2] // self.chunk)
        if not keys or not rt.in_run:
            # Nothing a master would be asked, which ATOM answers itself (0),
            # or no clock to ask it on, outside the run: no answer, and a
            # renewal that renews nothing.
            future.set_result(0 if not keys or kind == "renew" else None)
            return future
        with self._lock:
            rid = next(self._ids)
            self._calls[rid] = (kind, fn, args, future, keys)
            self.counts["lookups" if kind == "query" else "renewals"] += 1
        self._send_lookup((kind, rid, keys))
        return future

    def _send_lookup(self, payload) -> None:
        """Send a query or renewal on this engine's lookup channel.

        The #3767 seam, the one place a Store call is sent off the step loop.
        The step loop, the clock owner, sends at once. An offer from the
        engine's input thread (`EngineCore._offer_tier_lookups`) may not stamp
        a send in a run, so it is handed to the clock owner as a callable
        (`_offered_send`), with the arrival of the message the handler was
        given, the owner's clock while it waits for the handler:

        - with #3767's `LPRuntime.offer`, the owner makes it once the handler
          is back at its wait point, stamped at that arrival, before it
          releases the next message; as the engine's request channels are
          offering (`add_mooncake_store`), (b) grants an arrival inside a
          forward's TAR first, so it is stamped there too;
        - on a runtime without it (a tree before #3767's base), the shell
          queues it and the owner makes it at its next clock
          call (`_install_flush`), at the clock the releasing grant left it: the
          arrival when the engine was idle, the forward's end inside one,
          counted in ``offers_late``.
        """
        rt = self._rt
        if threading.current_thread() is rt.owner:
            self._send(self._look, payload)
            return
        send = functools.partial(self._offered_send, rt.now, payload)
        with self._lock:
            self.counts["offers"] += 1
            if self._offer is None:
                self._offered.append(send)
                return
        self._offer(send)

    def _offered_send(self, arrival: float, payload) -> None:
        """Make an offered send on the owner, at its clock now; count it late
        when that is past the arrival of the message that offered it."""
        late = self._rt.now - arrival
        if late > 0:
            self.counts["offers_late"] += 1
            self.offered_late_s = max(self.offered_late_s, late)
        self._send(self._look, payload)

    def _flush(self) -> None:
        """On the owner, make the offers the shell queued itself."""
        if not self._offered or threading.current_thread() is not self._rt.owner:
            return
        with self._lock:
            offered, self._offered = self._offered, []
        for send in offered:
            send()

    def _install_flush(self) -> None:
        """On a runtime with no `offer`, have the owner make the queued offers
        before each clock call it makes or answers itself, as `offer` makes
        those of a thread no release handed a message."""
        rt = self._rt
        for name in ("_ca_send", "_own_step"):
            call = getattr(rt, name)

            def flushing(*args, _call=call, **kwargs):
                self._flush()
                return _call(*args, **kwargs)

            setattr(rt, name, flushing)

    def _socket(self, ch: str):
        """`ch`'s DEALER to LP mooncake_store, opened at its first send once
        the LP names its endpoint."""
        import zmq

        if ch not in self._socks:
            # Named before the LP's first clock call, so before any grant lets
            # a request reach this engine: the wait reads no clock, and its
            # bound only turns a missing LP into an error naming the file.
            looks = int(ENDPOINT_WAIT_S / ENDPOINT_LOOK_S)
            while not self._endpoint.exists():
                if not looks:
                    raise RuntimeError(
                        f"LP mooncake_store named no endpoint in {self._endpoint} "
                        f"within {ENDPOINT_WAIT_S:.0f} s"
                    )
                looks -= 1
                threading.Event().wait(ENDPOINT_LOOK_S)
            sock = zmq.Context.instance().socket(zmq.DEALER)
            sock.connect(self._endpoint.read_text().strip())
            self._socks[ch] = sock
        return self._socks[ch]

    def _send(self, ch: str, payload) -> None:
        rt = self._rt
        header = pickle.dumps((ch, *rt.stamp_send(ch)))
        self._socket(ch).send_multipart([header, pickle.dumps(payload)])

    def _take(self) -> None:
        """On the step loop in a run: flush the queued offers, then take every
        lookup reply released so far, in arrival order, first waiting for any
        released one still on the wire."""
        import zmq

        rt = self._rt
        if not rt.in_run or threading.current_thread() is not rt.owner:
            return
        self._flush()
        sock, back = self._socks.get(self._look), self._back
        if sock is None:
            return
        while True:
            while True:
                try:
                    header, payload = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                ch, arrival, seq = pickle.loads(header)
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
            for _, seq in due:
                rt.count_done_locked(back, seq)
        for _, seq in due:
            self._answer(pickle.loads(self._read.pop(seq)[1]))

    def _answer(self, reply) -> None:
        """Run the call a reply answers over it: `_StoreLookup.lookup`, its
        ``asked_at`` the reply's start less a hop, or ATOM's ``_renew``; keep
        the reply's lease for the request's load."""
        rid, start, deadline, present = reply
        with self._lock:
            kind, fn, args, future, keys = self._calls.pop(rid)
        seq = args[0]
        sid = args[1] if kind == "query" else str(seq.id)
        logger.info(
            "compass: mooncake_store %s of %s at %r: chunk 0 %s, %d of %d keys "
            "present, %d another instance's",
            kind,
            sid,
            self._rt.now,
            keys[0].rpartition("/")[2][:12],
            len(present),
            len(keys),
            sum(p != self._me for p in present.values()),
        )
        self._leases[sid] = (deadline, present)
        self._leases.move_to_end(sid)
        lookup = self._lookup
        while len(self._leases) > 2 * lookup.capacity:
            self._leases.popitem(last=False)
        if future.cancelled():
            return
        self._client.answer = present
        try:
            if kind == "query":
                with lookup.lock:
                    lookup.asked_at[sid] = start - self._hop
                    lookup.asked_at.move_to_end(sid)
                    while len(lookup.asked_at) > 2 * lookup.capacity:
                        lookup.asked_at.popitem(last=False)
                value = lookup.backend.lookup(seq, lookup_id=sid)
            else:
                value = fn(*args)
        except Exception as exc:  # noqa: BLE001  # handed to the waiter
            future.set_exception(exc)
        else:
            future.set_result(value)
        finally:
            self._client.answer = {}

    # -- the ranks' copies ---------------------------------------------------

    def _chunks(self, kind: str, req) -> tuple[int, int]:
        """The chunks ``[first, end)`` ATOM's worker copies for `req`."""
        from atom.kv_transfer.offload.mooncake_store.keys import DIGEST_BYTES

        if kind == "save":
            end = len(req.chunk_hashes or b"") // DIGEST_BYTES
            return min(req.save_spec.skip_leading_tokens // self.chunk, end), end
        spec = req.load_spec
        hbm, lmc = spec.hbm_cached_tokens, spec.lmcache_cached_tokens
        if lmc <= hbm or hbm % self.chunk:
            return 0, 0
        end = lmc if spec.transfer_end_tokens is None else spec.transfer_end_tokens
        return hbm // self.chunk, end // self.chunk

    def _queue(self, k: int, kind: str, req) -> None:
        """Queue rank `k`'s copy of `req` on its lanes; send a save's put."""
        from atom.kv_transfer.offload.mooncake_store.keys import (
            chunk_group_ids,
            chunk_keys,
        )

        first, end = self._chunks(kind, req)
        lease = None
        if kind == "load":
            # The request's last reply, as the load's get reads it.
            deadline, present = self._leases.get(str(req.req_id), (-math.inf, {}))
            lease = {"deadline": deadline, "present": present}
        start, stop = self._lanes.queue(k, kind, req, end - first, lease)
        if lease is not None:
            lease["start"] = start
        if kind == "save" and end > first:
            hashes = req.chunk_hashes
            keys = chunk_keys(self._namespace, k, self._world, hashes, first, end)
            groups = chunk_group_ids(self._namespace, hashes, first, end)
            self._send(self._xfer, ("put", keys, groups, start, stop))
            self.counts["saves"] += k == 0

    def _run_due(self) -> None:
        """Run each copy whose priced end the clock has reached, as its rank's
        worker runs it, in end order: a load over its last reply's answer."""
        from atom.kv_transfer.offload.mooncake_store.keys import chunk_keys

        for k, kind, req, lease in self._lanes.due():
            rank = self._ranks[k]
            if kind == "save":
                rank._guard(kind, rank._do_save_req, req)
                continue
            present = lease["present"]
            first, end = self._chunks(kind, req)
            if lease["start"] > lease["deadline"]:
                self._past_lease(req, k, lease["start"], lease["deadline"])
            if k == 0 and end > first:
                keys = chunk_keys(
                    self._namespace, k, self._world, req.chunk_hashes, first, end
                )
                self.counts["loads"] += 1
                self.counts["foreign_loads"] += any(
                    present.get(key, self._me) != self._me for key in keys
                )
            self._client.answer = present
            try:
                rank._guard(kind, rank._do_load_req, req)
            finally:
                self._client.answer = {}

    def _past_lease(self, req, k: int, start: float, deadline: float) -> None:
        """Refuse, by name, a load whose lane starts after its lease ran out:
        whether the master still held the objects is what its get would ask."""
        from atom.compass.kv.mooncake_store_server import MooncakeStoreRefusal

        self.counts["past_lease_loads"] += 1
        why = (
            f"{self._me}: rank {k}'s load of {req.req_id} starts at {start!r}, "
            f"past its lease to {deadline!r}"
        )
        logger.warning("compass: %s: %s", PAST_LEASE, why)
        self._rt.refusals.append(PAST_LEASE)
        if self._abort:
            raise MooncakeStoreRefusal(f"{PAST_LEASE}: {why}")

    # -- the connector contract ----------------------------------------------

    def build_connector_meta(self):
        """ATOM's metadata, each rank's share of it queued as its worker takes
        it, each load and save on its lanes."""
        if self.__dict__.get("_inert"):
            from atom.kv_transfer.disaggregation.types import ConnectorMetadata

            return ConnectorMetadata()
        meta = self._inner.build_connector_meta()
        self._run_due()
        for k, rank in enumerate(self._ranks):
            for req in meta.requests:
                if req.load_spec is not None and rank._do_load:
                    self._queue(k, "load", req)
                if req.save_spec is not None and rank._do_save:
                    self._queue(k, "save", req)
        return meta

    def process_completions(self, output):
        """Run the copies due; report what the ranks gave at the previous
        call, then let ATOM's scheduler half read it."""
        if self.__dict__.get("_inert"):
            return output
        self._flush()
        self._run_due()
        held = self._tp.aggregate([rank.get_finished() for rank in self._ranks])
        held, self._held = self._held, held
        for field in dataclasses.fields(held) if held is not None else ():
            value = getattr(held, field.name)
            if isinstance(value, set):
                getattr(output, field.name).update(value)
        return self._inner.process_completions(output)

    def has_pending_work(self) -> bool:
        if self.__dict__.get("_inert"):
            return False
        self._flush()
        held = self._held is not None and not self._held.is_empty()
        pending = bool(self._lanes) or held or bool(self._offered)
        return pending or self._inner.has_pending_work()

    def offload_statistics(self) -> dict:
        """ATOM's counters, rank 0's priced copies, and the shell's counts."""
        if self.__dict__.get("_inert"):
            return {}
        return {
            "connector": self._inner.get_statistics(),
            "copies": self._lanes.copies(),
            "store": dict(self.counts) | {"offers_late_max_s": self.offered_late_s},
        }

    def close(self) -> None:
        if self.__dict__.get("_inert"):
            return
        self._inner.close()
        for rank in self._ranks:
            rank.close()
        for sock in self._socks.values():
            sock.close(linger=0)
