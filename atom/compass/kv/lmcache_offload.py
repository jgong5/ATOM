# SPDX-License-Identifier: MIT
"""ATOM's `lmcache_offload` connector on a simulated run: its scheduler half as
ATOM runs it, each TP rank's CPU tier an index of chunks without bytes, and the
tier's lookups and copies priced on the engine's LP clock.

`substitute` makes the connector factory build, for `lmcache_offload`, the
scheduler half `OffloadShell` and the worker half of `compass`, which moves
nothing; a `multi` that lists it builds `MultiShell`
(`atom.compass.kv.lmcache_mp`) as its scheduler half.

`OffloadShell` wraps ATOM's `LMCacheOffloadConnectorScheduler`, with the layout
its config selects, built with `Lookup` in place of LMCache's lookup client;
a layout whose worker keeps a state tier beside its chunks is refused, so it
is dense's or MiniMax-M3's `M3OffloadScheduler`. It keeps one ATOM worker
connector of that layout per TP rank, with no device, whose LMCache engine is
the rank's `Engine` over the rank's `Tier`. Each load and save that
`build_connector_meta` hands a rank waits on the rank's load or save queue, as
wide as the worker's `load_workers` or `save_workers`, and the worker runs it
at its priced end: the entry's ``load_s`` or ``store_s``, ``[fixed, per
chunk]`` seconds, over the chunks it copies, and nothing for a copy of none.
`process_completions` reports what the ranks' `get_finished`, aggregated over
the TP group, gave at the previous call, since the engine's aggregation
answers one poll behind.

`Lookup` asks each rank that LMCache's ``lookup_server_worker_ids`` names, each
pinning and touching the prefix its tier holds as LMCache's lookup server does,
takes the minimum hit, as LMCache's lookup client does, and charges the step
loop the entry's ``lookup_s`` over the chunks asked for.

A `Tier` holds a rank's chunk keys, chained hashes of whole chunks as LMCache
keys them, in as many slots as the rank's CPU size holds chunks of
``chunk_bytes`` aligned as LMCache's allocator aligns one, under the LMCache
cache policy the deployment names (ATOM's `SLRUCachePolicy` under
``ATOM_SLRU``). A store takes a slot per chunk, evicting a candidate of the
policy while none is free, and stops at the first chunk it cannot place; a
chunk already held gives its slot back, as LMCache's put drops a duplicate. A
lookup pins what it hits under its id, a later lookup by the same id leaving
the first one's pins, until ATOM's worker unpins the id; LMCache's pin monitor
unpins a chunk pinned longer than ``pin_timeout_sec``, checking every
``pin_check_interval_sec``.

The run file's `ENTRY` holds ``chunk_size``, ``tensor_parallel_size`` and
``chunk_bytes``, as the deployment it was fitted on logged them, and the
three priced terms (`atom.compass.backends.step_linear_fit.fit_offload`). A
deployment of another chunk size or TP width is refused, and so is an LMCache
config with a tier or a lookup this does not model (`UNMODELLED`). Importing
this module loads no third-party package; using it loads lmcache, numpy and
torch.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import heapq
import itertools
import logging
import math

logger = logging.getLogger(__name__)

#: The run-file key of the connector's entry.
ENTRY = "lmcache_offload"
#: The entry's priced terms, each ``[fixed, per chunk]`` seconds.
TERMS = ("lookup_s", "store_s", "load_s")
#: The entry's fields the deployment must match.
MATCHED = ("chunk_size", "tensor_parallel_size")
#: The layouts whose worker stores chunks alone, `DenseOffloadConnector`'s.
LAYOUTS = ("dense", "m3")
#: LMCache config fields that add a tier, a transport or a lookup this does not
#: model; any one set, not None or False, is refused by name.
UNMODELLED = (
    "local_disk",
    "remote_url",
    "enable_p2p",
    "enable_pd",
    "use_layerwise",
    "enable_blending",
    "save_unfull_chunk",
    "external_lookup_client",
    "hit_miss_ratio",
    "enable_scheduler_bypass_lookup",
    "enable_chunk_statistics",
    "extra_config",
)


def substitute() -> None:
    """Build this module's scheduler half for every `lmcache_offload` connector
    from now on, and `compass`'s worker half."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    registry = KVConnectorFactory._registry
    registry["lmcache_offload"] = registry["lmcache_offload"] | {
        "worker_module": "atom.compass.kv.connector",
        "worker_class": "SimulatedKVConnector",
        "scheduler_module": __name__,
        "scheduler_class": OffloadShell.__name__,
    }
    registry["multi"] = registry["multi"] | {
        "scheduler_module": "atom.compass.kv.lmcache_mp",
        "scheduler_class": "MultiShell",
    }


def check_entry(entry) -> None:
    """Refuse an `ENTRY` that is not the fields and terms `OffloadShell` reads."""
    if not isinstance(entry, dict):
        raise TypeError(f"the run file's {ENTRY} entry is {entry!r}, not a mapping")
    need = {*MATCHED, "chunk_bytes", *TERMS}
    if missing := sorted(need - entry.keys()):
        raise ValueError(f"the run file's {ENTRY} entry lacks {', '.join(missing)}")
    if extra := sorted(entry.keys() - need - {"fit"}):
        raise ValueError(f"the run file's {ENTRY} entry has unknown {', '.join(extra)}")
    for name in (*MATCHED, "chunk_bytes"):
        if not (isinstance(entry[name], int) and entry[name] > 0):
            raise ValueError(f"{ENTRY}.{name} is {entry[name]!r}, not a positive int")
    for name in TERMS:
        term = entry[name]
        if not (
            isinstance(term, list)
            and len(term) == 2
            and all(
                isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                for v in term
            )
        ):
            raise ValueError(
                f"{ENTRY}.{name} is {term!r}, not [fixed, per chunk] seconds, "
                "finite and non-negative"
            )


def chunk_keys(tokens, chunk: int) -> list[bytes]:
    """The key of each whole chunk of `tokens`: a hash of its tokens chained on
    the key of the chunk before, as LMCache's token database keys a prefix."""
    import numpy as np

    data = np.asarray(tokens, dtype=np.int64).tobytes()
    step, key, keys = 8 * chunk, b"", []
    for end in range(step, len(data) + 1, step):
        key = hashlib.blake2b(key + data[end - step : end], digest_size=16).digest()
        keys.append(key)
    return keys


class _Chunk:
    """A held chunk: its pins, and when it was last pinned."""

    __slots__ = ("pinned_at", "pins")

    def __init__(self) -> None:
        self.pins, self.pinned_at = 0, 0.0

    @property
    def can_evict(self) -> bool:
        return self.pins == 0


class Tier:
    """One rank's CPU tier: its chunk keys in `slots` slots under `policy`."""

    def __init__(self, slots: int, policy, pin_timeout_s: float, check_s: float):
        self.policy, self.free = policy, slots
        self.chunks = policy.init_mutable_mapping()
        self.lookups: dict[str, list[bytes]] = {}
        self.pinned: dict[bytes, _Chunk] = {}
        self.pin_timeout_s, self.check_s, self.next_check = pin_timeout_s, check_s, 0.0
        self.counts = collections.Counter()

    def lookup(self, keys: list[bytes], lookup_id: str, now: float) -> int:
        """Pin the held prefix of `keys` under `lookup_id`, touch it last chunk
        first, and say how many chunks it holds."""
        self._expire(now)
        hit = []
        for key in keys:
            if (chunk := self.chunks.get(key)) is None:
                break
            chunk.pins, chunk.pinned_at = chunk.pins + 1, now
            self.pinned[key] = chunk
            hit.append(key)
        if hit:
            self.lookups[lookup_id] = hit
        for key in reversed(hit):
            self.policy.update_on_hit(key, self.chunks)
        self.counts.update(lookups=1, lookup_chunks=len(keys), hit_chunks=len(hit))
        return len(hit)

    def unpin_lookup(self, lookup_id: str) -> None:
        for key in self.lookups.pop(lookup_id, ()):
            if (chunk := self.chunks.get(key)) is not None and chunk.pins:
                chunk.pins -= 1
                if not chunk.pins:
                    del self.pinned[key]

    def held(self, keys: list[bytes]) -> int:
        """How many of `keys`, from the first, the tier holds."""
        return next((i for i, k in enumerate(keys) if k not in self.chunks), len(keys))

    def store(self, keys: list[bytes], now: float) -> int:
        """Take a slot per key, evicting while none is free, then put the keys
        placed; how many were placed."""
        self._expire(now)
        placed = 0
        for _ in keys:
            if not self.free:
                victims = self.policy.get_evict_candidates(self.chunks, 1)
                if not victims:
                    self.counts["store_cut"] += 1
                    break
                for key in victims:
                    del self.chunks[key]
                self.free += len(victims)
                self.counts["evicted"] += len(victims)
            self.free, placed = self.free - 1, placed + 1
        for key in keys[:placed]:
            if key in self.chunks:
                self.free += 1
                self.counts["duplicates"] += 1
            else:
                self.chunks[key] = _Chunk()
                self.policy.update_on_put(key)
                self.counts["stored"] += 1
        return placed

    def _expire(self, now: float) -> None:
        """LMCache's pin monitor: unpin each chunk pinned past the timeout."""
        if now < self.next_check:
            return
        self.next_check = (math.floor(now / self.check_s) + 1) * self.check_s
        for key, chunk in list(self.pinned.items()):
            if now - chunk.pinned_at > self.pin_timeout_s:
                chunk.pins = 0
                del self.pinned[key]
                self.counts["forced_unpins"] += 1


class _Copies:
    """The engine's GPU connector, which copies nothing: it tells ATOM's worker
    each chunk a store placed has been read, as source-safe."""

    def __init__(self, source_safe) -> None:
        self.source_safe, self.operation = source_safe, None

    @contextlib.contextmanager
    def track_save_source(self, operation):
        self.operation, previous = operation, self.operation
        try:
            yield
        finally:
            self.operation = previous


class Engine:
    """LMCache's engine as ATOM's worker connector calls it, over one rank's
    tier, at the clock `now` reads."""

    def __init__(self, tier: Tier, chunk: int, now, source_safe) -> None:
        self.tier, self.chunk, self.now = tier, chunk, now
        self.gpu_connector = _Copies(source_safe)

    def _from(self, tokens, mask) -> tuple[int, list[bytes]]:
        """The first token `mask` keeps, and the keys of the whole chunks from it."""
        start = 0 if mask is None else len(mask) - int(mask.sum())
        return start, chunk_keys(tokens, self.chunk)[start // self.chunk :]

    def lookup_unpin(self, lookup_id: str) -> None:
        self.tier.unpin_lookup(lookup_id)

    def store(self, tokens, mask=None, **kwargs) -> None:
        from atom.kv_transfer.disaggregation.types import (
            SaveOperationId,
            SaveSourceGroupId,
        )

        start, keys = self._from(tokens, mask)
        placed = self.tier.store(keys, self.now())
        copies = self.gpu_connector
        if copies.source_safe and isinstance(copies.operation, SaveOperationId):
            for s in range(start, start + placed * self.chunk, self.chunk):
                copies.source_safe(
                    SaveSourceGroupId(copies.operation, ((s, s + self.chunk),))
                )

    def retrieve(self, tokens, mask=None, **kwargs):
        import torch

        start, keys = self._from(tokens, mask)
        got = torch.zeros(len(tokens), dtype=torch.bool)
        got[start : start + self.tier.held(keys) * self.chunk] = True
        return got


class Lookup:
    """LMCache's lookup client over the tiers of `ranks`."""

    def __init__(self, shell: OffloadShell, ranks) -> None:
        self.shell, self.ranks = shell, list(ranks)

    def lookup(self, token_ids, lookup_id, request_configs=None) -> int:
        shell = self.shell
        keys = chunk_keys(token_ids, shell.chunk)
        if not keys:
            return 0
        rt = shell._rt
        shell._run_due()
        hits = [
            shell._tiers[r].lookup(keys, str(lookup_id), rt.now) for r in self.ranks
        ]
        fixed, per_chunk = shell._terms["lookup_s"]
        if rt.in_run:
            rt.advance_to(rt.now + fixed + per_chunk * len(keys))
        return min(hits) * shell.chunk

    def clear_lookup_status(self, lookup_id) -> None:
        """The client keeps no status a later call reads."""


class OffloadShell:
    """ATOM's `lmcache_offload` scheduler half, its ranks' tiers and copies
    simulated on the step loop."""

    def __init__(self, config) -> None:
        from lmcache.v1.lookup_client.factory import LookupClientFactory
        from lmcache.v1.memory_management import AddressManager
        from lmcache.v1.storage_backend.cache_policy import get_cache_policy

        from atom.compass.run import spec
        from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
        from atom.kv_transfer.offload import config as offcfg
        from atom.kv_transfer.offload.connector import (
            LMCacheOffloadConnectorScheduler,
            _build_worker,
        )
        from atom.utils import clock

        self._rt = clock.installed()
        if self._rt is None:
            raise RuntimeError("no LP runtime is installed to price the tier on")
        entry = spec()[ENTRY]
        check_entry(entry)
        if (layout := offcfg.select_offload_layout(config)) not in LAYOUTS:
            raise ValueError(
                f"lmcache_offload's {layout!r} layout keeps a state tier beside "
                f"its chunks; only {', '.join(LAYOUTS)} are simulated"
            )
        # As each rank's worker builds its engine's config.
        cfg = offcfg.build_lmcache_config(config.kv_transfer_config)
        offcfg.scale_cpu_size_for_pp(cfg, config)
        if not cfg.local_cpu:
            raise ValueError("lmcache_offload is simulated with its CPU tier on")
        if unmodelled := [
            f for f in UNMODELLED if getattr(cfg, f) not in (None, False)
        ]:
            raise ValueError(
                f"lmcache_offload is simulated with a CPU tier alone; this "
                f"deployment sets LMCache's {', '.join(unmodelled)}"
            )
        world = offcfg.lmcache_replica_world_size(config)
        for name, value in zip(MATCHED, (cfg.chunk_size, world)):
            if entry[name] != value:
                raise ValueError(
                    f"the run file's {ENTRY} was fitted at {name} {entry[name]}, "
                    f"and this deployment runs {value}"
                )
        self.chunk = cfg.chunk_size
        align = AddressManager.ALIGN_BYTES
        slots = int(cfg.max_local_cpu_size * 1024**3) // (
            -(-entry["chunk_bytes"] // align) * align
        )
        self._terms = {name: tuple(entry[name]) for name in TERMS}
        self._tiers = [
            Tier(
                slots,
                get_cache_policy(cfg.cache_policy),
                cfg.pin_timeout_sec,
                cfg.pin_check_interval_sec,
            )
            for _ in range(world)
        ]
        factory = vars(LookupClientFactory)["create_lookup_client"]

        def lookup(cfg, meta, *args):
            ids = cfg.get_lookup_server_worker_ids(meta.use_mla, meta.world_size)
            return Lookup(self, ids or range(meta.world_size))

        # The scheduler half builds its lookup client once, here.
        LookupClientFactory.create_lookup_client = staticmethod(lookup)
        try:
            self._inner = LMCacheOffloadConnectorScheduler(config)
        finally:
            LookupClientFactory.create_lookup_client = factory
        self._ranks = []
        for k, tier in enumerate(self._tiers):
            rank = _build_worker(config)
            safe = rank._source_group_safe if rank._early_release else None
            rank._engine = Engine(tier, self.chunk, self._rt.read_clock, safe)
            rank.chunk_size, rank._rank = self.chunk, k
            self._ranks.append(rank)
        self._lanes = [
            {"load": [0.0] * r.load_workers, "save": [0.0] * r.save_workers}
            for r in self._ranks
        ]
        self._due: list = []  # (end, order, rank, kind, request)
        self._order = itertools.count()
        self._copied = {kind: collections.Counter() for kind in ("load", "save")}
        self._tp = KVOutputAggregator(world_size=world)
        self._held = None  # what the ranks reported at the previous call

    def __getattr__(self, name: str):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def _chunks(self, kind: str, req) -> int:
        """The chunks ATOM's worker copies for `req`'s load or save."""
        if kind == "load":
            hbm = req.load_spec.hbm_cached_tokens
            lmc = req.load_spec.lmcache_cached_tokens
            return (
                (lmc - hbm) // self.chunk if lmc > hbm and not hbm % self.chunk else 0
            )
        whole = len(req.token_ids) // self.chunk
        return max(0, whole - req.save_spec.skip_leading_tokens // self.chunk)

    def _queue(self, k: int, kind: str, req) -> None:
        now, lanes = self._rt.read_clock(), self._lanes[k][kind]
        chunks = self._chunks(kind, req)
        fixed, per_chunk = self._terms["load_s" if kind == "load" else "store_s"]
        start = max(now, heapq.heappop(lanes))
        end = start + (fixed + per_chunk * chunks if chunks else 0.0)
        heapq.heappush(lanes, end)
        heapq.heappush(self._due, (end, next(self._order), k, kind, req))
        if k == 0:
            self._copied[kind].update(copies=1, chunks=chunks)
            self._copied[kind]["seconds"] += end - start
            self._copied[kind]["waited"] += start - now
            logger.info(
                "compass: lmcache_offload %s of %s, %d chunks: queued at %r, from "
                "%r to %r",
                kind,
                req.req_id,
                chunks,
                now,
                start,
                end,
            )

    def _run_due(self) -> None:
        """Run each copy whose priced end the clock has reached, as its rank's
        worker runs it, in end order."""
        now = self._rt.read_clock()
        while self._due and self._due[0][0] <= now:
            _, _, k, kind, req = heapq.heappop(self._due)
            rank = self._ranks[k]
            run = rank._do_load_req if kind == "load" else rank._do_save_req
            rank._guard(kind, run, req)

    def build_connector_meta(self):
        """ATOM's metadata, each rank's share of it queued as its worker takes
        it: a lookup of the step that is no load unpinned, each load and save
        on its queue."""
        meta = self._inner.build_connector_meta()
        self._run_due()
        for k, rank in enumerate(self._ranks):
            loads = [r for r in meta.requests if r.load_spec is not None]
            loading = {str(r.req_id) for r in loads if rank._do_load}
            for lookup_id in meta.lookup_requests_in_step:
                if str(lookup_id) not in loading:
                    rank._lookup_unpin(lookup_id)
            for req in meta.requests:
                if req.load_spec is not None and rank._do_load:
                    self._queue(k, "load", req)
                if req.save_spec is not None and rank._do_save:
                    self._queue(k, "save", req)
        return meta

    def process_completions(self, output):
        """Run the copies due; report what the ranks gave at the previous
        call, then let ATOM's scheduler half read it."""
        self._run_due()
        held = self._tp.aggregate([rank.get_finished() for rank in self._ranks])
        held, self._held = self._held, held
        for field in dataclasses.fields(held) if held is not None else ():
            value = getattr(held, field.name)
            if isinstance(value, set):
                getattr(output, field.name).update(value)
        return self._inner.process_completions(output)

    def has_pending_work(self) -> bool:
        held = self._held is not None and not self._held.is_empty()
        return bool(self._due) or held or self._inner.has_pending_work()

    def offload_statistics(self) -> dict:
        """ATOM's counters, rank 0's priced copies, and the tiers' counts
        summed over the ranks, with what rank 0's tier holds."""
        tiers = sum((t.counts for t in self._tiers), collections.Counter())
        return {
            "connector": self._inner.get_statistics(),
            "copies": {kind: dict(c) for kind, c in self._copied.items()},
            "tier": dict(tiers) | {"held_chunks": len(self._tiers[0].chunks)},
        }
