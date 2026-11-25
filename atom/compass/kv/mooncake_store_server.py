# SPDX-License-Identifier: MIT
"""LP ``mooncake_store``: the master of a Mooncake Store that prefill engines
share, as Mooncake's master answers them at commit 206d634.

`Master` holds the index: objects keyed as ATOM keys them, each in its chunk's
group, the bytes they take against the owners' mounted capacity, and the leases
and eviction that decide what later lookups find. Every object takes the
entry's ``chunk_bytes``, or under a pipelined prefill its rank's (`PER_RANK`).

- A group's members share one lease, a deadline that only grows. A query or
  renewal leases each present key's group to its end plus
  ``--default_kv_lease_ttl``, and a put of a key the Store has published leases
  it the same way; a published object's lease reaches at least its
  publication.
- A put takes its new objects' bytes at its start and publishes them at its
  end. A put the capacity left cannot hold is refused.
- Eviction runs while used bytes over capacity exceed
  ``--eviction_high_watermark_ratio``: a round evicts ``ceil(filed x
  max(eviction_ratio, used_ratio - mark + eviction_ratio))`` objects, `filed`
  the published objects when it starts. It walks the groups by the deadline
  each was filed at when a member was last published, in 100 ms buckets and in
  filing order within one, up to the bucket holding now. A group a lookup
  extended since is filed again at its deadline when the walk reaches it, one
  whose lease has not run out is skipped, and the rest leave whole.
- The eviction thread wakes every 10 ms; a round runs at the first such
  boundary at or after the earliest lease deadline among filed groups, and none
  runs that would evict nothing.

`MooncakeStoreLP` is the LP's loop (`StationLP`). It takes Compass's frames,
since ATOM reaches the master over Mooncake's C++ client, which no wrapped
socket carries; each payload is pickled:

- on ``lookup#dp<r>``: ``(kind, request id, keys)``, `kind` ``query`` or
  ``renew``. One station `--rpc_thread_num` wide, the master's handler
  threads, serves each for ``lookup_s[1]`` seconds a key. A channel's calls
  run one at a time, as ATOM's scheduler makes them: one reaches the master
  no earlier than its arrival and twice the hop after the previous one's
  end, the hop half of ``lookup_s[0]``, and a renewal goes ahead of the
  channel's queued queries. Its reply, ``(request id, start, lease deadline,
  {present key: publishing engine LP})``, leaves at its job's end, its start
  when it reached the master;
- on ``transfer#dp<r>``: ``("put", keys, groups, start, end)``, a save's keys,
  each one's group and the save lane's start and end. Its bytes are taken at
  the start and its objects published at the end, neither before its
  arrival; it gets no reply.

The LP hands in its lookups, the keys they found present, those another
engine LP published, its renewals and the objects it evicted (`COUNTS`).

With ``clear_between_client_runs`` the Store is emptied at the end of each
client run, and mounted again at ``capacity_by_client_run``'s next size
(`MooncakeStoreLP.clear`), as a record's launcher empties it between segments.

What it does not model: contention at the owners, a get's own master call,
failures, soft pins, tenants, the owners' allocator and its fragmentation, and
the eviction thread's phase. The master flags it reads are
``--default_kv_lease_ttl``, ``--eviction_high_watermark_ratio``,
``--eviction_ratio`` and ``--rpc_thread_num``; it accepts `ACCEPTED` and
refuses any other, and flags that set no positive ``--rpc_thread_num``, since
the master then runs as many handler threads as its host has cores, up to 16.

Importing this module loads lmcache's server module, which imports only the
standard library.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import pickle
import re
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

from atom.compass.kv.lmcache_server import StationLP

#: The run-file key of the entry.
ENTRY = "mooncake_store"
#: The entry's laws, each ``[fixed, per key]`` (lookup) or ``[fixed, per chunk]``
#: seconds; the fixed lookup term is both hops.
TERMS = ("lookup_s", "store_s", "load_s")
#: Optional fields: the Store a run with several client runs starts each one
#: with, as a record's launcher empties it (``store_check --clear``) and
#: mounts it again at each segment's size. ``clear_between_client_runs``: the
#: Store is emptied at the end of each client run (`MooncakeStoreLP.clear`);
#: ``capacity_by_client_run``: the owners' bytes for client run ``k``, the
#: last repeating, its first ``capacity_bytes``'s.
CLIENT_RUN_FIELDS = ("clear_between_client_runs", "capacity_by_client_run")
#: The entry's fields a pipelined prefill's writer ranks may each state apart:
#: a list of one value per rank, ``world`` of them in ATOM's rank order
#: (``pp_rank x tp + tp_rank``, `pp_aware_rank_and_world`). Its stages hold
#: different layers, so each stage's chunk takes other bytes, its pools other
#: slots and its copies other seconds. An entry fitted on one stage's ranks
#: states each as one value, which every rank takes.
PER_RANK = ("chunk_bytes", "save_slots", "load_slots", "store_s", "load_s")
#: A chunk key's rank and world (`kv_store.keys.rank_key_prefix`).
KEY_RANK = re.compile(r"/w(\d+)of(\d+)/")
#: The entry's fields the deployment must match.
MATCHED = (
    "protocol",
    "chunk_tokens",
    "chunk_bytes",
    "world",
    "save_slots",
    "load_slots",
    "save_workers",
    "load_workers",
)
#: Master flags that change nothing a requester is told: endpoints, a client's
#: death, which no simulated client dies of, and the allocator, which is one
#: pool of the owners' summed bytes here.
ACCEPTED = (
    "--rpc_port",
    "--enable_http_metadata_server",
    "--http_metadata_server_host",
    "--http_metadata_server_port",
    "--metrics_port",
    "--client_ttl",
    "--allocation_strategy",
    "--memory_allocator",
)
#: What the LP hands in, each counted from its start.
COUNTS = ("lookups", "hit_keys", "foreign_hit_keys", "renewals", "evicted")
#: The file in the run's out_dir that names the endpoint the LP serves at.
ENDPOINT_FILE = "mooncake_store_endpoint.txt"
#: The eviction thread's wakes and the eviction index's buckets, a second.
TICKS_PER_S, BUCKETS_PER_S = 100, 10


class MooncakeStoreRefusal(ValueError):
    """A master flag, an entry or a put LP mooncake_store does not model."""


class _Flags(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise MooncakeStoreRefusal(f"mooncake_master flags: {message}")


def _duration_s(text: str) -> float:
    """A duration flag as Mooncake reads one: whole milliseconds, or a whole
    number with an ``ms``, ``s``, ``m`` or ``h`` suffix."""
    found = re.fullmatch(r"\s*(\d+)\s*(ms|s|m|h)?\s*", text, re.IGNORECASE)
    if found is None:
        raise ValueError(text)
    unit = (found[2] or "ms").lower()
    return int(found[1]) * {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]


def _ratio(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise ValueError(text)
    return value


def parse_master_flags(argv: list[str]) -> argparse.Namespace:
    """The ``mooncake_master`` flags `argv`, refusing one this LP neither
    models nor accepts, and a missing or non-positive ``--rpc_thread_num``."""
    p = _Flags(prog="mooncake_master", add_help=False, allow_abbrev=False)
    p.add_argument("--default_kv_lease_ttl", type=_duration_s, default=10.0)
    p.add_argument("--eviction_high_watermark_ratio", type=_ratio, default=0.9)
    p.add_argument("--eviction_ratio", type=_ratio, default=0.05)
    p.add_argument("--rpc_thread_num", type=int, default=0)
    for flag in ACCEPTED:
        p.add_argument(flag, nargs="?")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        raise MooncakeStoreRefusal(
            f"LP mooncake_store does not model master flag(s) {unknown}"
        )
    if args.rpc_thread_num <= 0:
        raise MooncakeStoreRefusal(
            "the master flags set no positive --rpc_thread_num, so the master "
            "would run as many handler threads as its host has cores, up to 16"
        )
    return args


def check_entry(entry) -> argparse.Namespace:
    """Refuse an `ENTRY` that is not the laws, matched fields, master flags and
    capacity this LP and the engines read; return its master flags."""
    if not isinstance(entry, dict):
        raise MooncakeStoreRefusal(f"the run file's {ENTRY} entry is {entry!r}")
    need = {*TERMS, *MATCHED, "master", "capacity_bytes"}
    if missing := sorted(need - entry.keys()):
        raise MooncakeStoreRefusal(
            f"the run file's {ENTRY} entry lacks {', '.join(missing)}"
        )
    if extra := sorted(entry.keys() - need - {"fit", *CLIENT_RUN_FIELDS}):
        raise MooncakeStoreRefusal(
            f"the run file's {ENTRY} entry has unknown {', '.join(extra)}"
        )

    def law(term) -> bool:
        return (
            isinstance(term, list)
            and len(term) == 2
            and all(isinstance(v, (int, float)) and 0 <= v < math.inf for v in term)
        )

    def positive(value) -> bool:
        return type(value) is int and value > 0

    world = entry["world"]
    for name in TERMS:
        term = entry[name]
        if not (law(term) or _ranked(entry, name, law)):
            raise MooncakeStoreRefusal(
                f"{ENTRY}.{name} is {term!r}, not [fixed, per item] seconds, or "
                f"one such a rank of world {world!r} (a pipelined prefill's)"
            )
    if not entry["lookup_s"][0] > 0:
        raise MooncakeStoreRefusal(
            f"{ENTRY}.lookup_s's fixed term is {entry['lookup_s'][0]!r}: each hop "
            "to and from the master is half of it, and a query and its reply are "
            "a cycle that needs a positive lookahead"
        )
    if entry["protocol"] not in ("rdma", "tcp"):
        raise MooncakeStoreRefusal(f"{ENTRY}.protocol is {entry['protocol']!r}")
    for name in (*MATCHED[1:], "capacity_bytes"):
        if not (positive(entry[name]) or _ranked(entry, name, positive)):
            raise MooncakeStoreRefusal(
                f"{ENTRY}.{name} is {entry[name]!r}, not a positive int"
                + (" or one a rank" if name in PER_RANK else "")
            )
    clear = entry.get("clear_between_client_runs", False)
    if type(clear) is not bool:
        raise MooncakeStoreRefusal(
            f"{ENTRY}.clear_between_client_runs is {clear!r}, not a bool"
        )
    sizes = entry.get("capacity_by_client_run")
    if sizes is not None:
        if not (
            isinstance(sizes, list)
            and sizes
            and all(type(b) is int and b > 0 for b in sizes)
        ):
            raise MooncakeStoreRefusal(
                f"{ENTRY}.capacity_by_client_run is {sizes!r}, not a list of "
                "positive ints"
            )
        if sizes[0] != entry["capacity_bytes"]:
            raise MooncakeStoreRefusal(
                f"{ENTRY}.capacity_by_client_run starts at {sizes[0]}, and "
                f"capacity_bytes, client run 0's, is {entry['capacity_bytes']}"
            )
        if not clear:
            raise MooncakeStoreRefusal(
                f"{ENTRY}.capacity_by_client_run sizes a Store that is never "
                "emptied: a mounted Store changes size only when it is mounted "
                "again, so it needs clear_between_client_runs"
            )
    argv = entry["master"]
    if not (isinstance(argv, list) and all(isinstance(a, str) for a in argv)):
        raise MooncakeStoreRefusal(f"{ENTRY}.master is {argv!r}, not argv")
    return parse_master_flags(argv)


def _ranked(entry: dict, name: str, each) -> bool:
    """Whether `entry`'s `name` is one value a rank, each of which `each`
    accepts: a list of ``world`` values, for a field `PER_RANK` names."""
    value, world = entry[name], entry.get("world")
    return (
        name in PER_RANK
        and type(world) is int
        and isinstance(value, list)
        and len(value) == world
        and all(each(v) for v in value)
    )


def of_rank(entry: dict, name: str, rank: int):
    """Writer rank `rank`'s `name` in `entry`: the entry's one value, or the
    rank's own where it states one a rank (`PER_RANK`)."""
    value = entry[name]
    ranked = isinstance(value, list) and (
        name not in TERMS or isinstance(value[0], list)
    )
    return value[rank] if ranked else value


def client_run_capacity(entry: dict, k: int) -> int:
    """The owners' bytes client run `k` starts with: ``capacity_by_client_run``'s
    ``k``-th, its last past its end, else ``capacity_bytes``."""
    sizes = entry.get("capacity_by_client_run") or [entry["capacity_bytes"]]
    return sizes[min(k, len(sizes) - 1)]


def check_plans(entry: dict, plans: dict, block_tokens: int) -> None:
    """Refuse a member of the engines LP mooncake_store serves whose published
    plan (`atom.compass.run.publish_transfer`) gives a chunk, its KV block's
    bytes over the entry's ``chunk_tokens`` of `block_tokens`, or a TP world
    other than the entry's, at which its laws were fitted. Under a pipelined
    prefill each stage's members (``-pp<k>``) plan their own layers' chunk,
    which the entry states at the stage's ranks, and the world is the stages'
    TP worlds together."""
    blocks = entry["chunk_tokens"] // block_tokens
    stages = len({_stage(name) for name in plans}) or 1
    for name, plan in plans.items():
        chunk, tp = plan["block_bytes"] * blocks, plan["tp"]
        want = of_rank(entry, "chunk_bytes", _stage(name) * tp)
        if (chunk, tp * stages) != (want, entry["world"]):
            raise MooncakeStoreRefusal(
                f"{name} plans a {chunk} B chunk at TP {tp} of {stages} stage(s), "
                f"and the {ENTRY} entry states {want} B at world {entry['world']}"
            )


def _stage(name: str) -> int:
    """The pipeline stage of an engine LP's member `name` (``engine-P-pp2.dp0``
    is stage 2), 0 outside a pipeline."""
    m = re.search(r"-pp(\d+)$", name.partition(".")[0])
    return int(m[1]) if m else 0


def store_rpc_s(entry: dict) -> float:
    """One hop between an engine and the master: half of ``lookup_s``'s fixed
    term."""
    return entry["lookup_s"][0] / 2


def _bucket(t: float) -> int:
    return math.floor(t * BUCKETS_PER_S)


def _wake(t: float) -> int:
    """The eviction thread's first wake at or after `t`, by its index: the
    least `k` with ``k / TICKS_PER_S >= t`` as floats compare."""
    k = math.ceil(t * TICKS_PER_S)
    if (k - 1) / TICKS_PER_S >= t:
        return k - 1
    return k + 1 if k / TICKS_PER_S < t else k


@dataclass
class _Object:
    group: str
    publisher: str
    published: bool = False


@dataclass
class _Group:
    deadline: float = -math.inf  # the lease its members share
    published: list = field(default_factory=list)  # filed members, by key
    pending: int = 0  # members put and not yet published
    rank: int | None = None  # the bucket it is filed in, None while unfiled
    order: int = 0  # its place among its bucket's groups


class Round(NamedTuple):
    """One eviction round: its target ratio and object count, the objects and
    groups it evicted, and the groups it skipped, each with why: ``lease``
    while its lease runs, ``moved`` when it was filed again at the deadline a
    lookup gave it."""

    target: float
    goal: int
    objects: int
    evicted: list
    skipped: list


class Master:
    """A Mooncake Store master's objects, groups, leases and eviction, holding
    no bytes; every call passes the LP clock in."""

    def __init__(self, entry: dict) -> None:
        flags = check_entry(entry)
        self.ttl = flags.default_kv_lease_ttl
        self.mark, self.ratio = (
            flags.eviction_high_watermark_ratio,
            flags.eviction_ratio,
        )
        self.threads = flags.rpc_thread_num
        self.chunk_bytes = entry["chunk_bytes"]
        self._entry = entry
        self.reset(entry["capacity_bytes"])

    def size(self, key: str) -> int:
        """The bytes object `key` takes: the entry's ``chunk_bytes``, or the
        one of the rank its key names where it states one a rank."""
        if not isinstance(self.chunk_bytes, list):
            return self.chunk_bytes
        m = KEY_RANK.search(key)
        if m is None:
            raise MooncakeStoreRefusal(
                f"key {key!r} names no rank, and the {ENTRY} entry states its "
                "chunk bytes a rank"
            )
        return of_rank(self._entry, "chunk_bytes", int(m[1]))

    def reset(self, capacity: int) -> None:
        """An empty Store of `capacity` bytes: every object, group, lease and
        eviction deadline gone, as a cleared and remounted Store has none."""
        self.capacity = capacity
        self.objects: dict[str, _Object] = {}
        self.groups: dict[str, _Group] = {}
        self.used = 0
        self.filed = 0  # published objects
        self._filing: list = []  # heap of (bucket, order, group id), lazily stale
        self._deadlines: list = []  # heap of (deadline, group id), lazily stale
        self._order = itertools.count()
        self._last_wake = -math.inf  # the wake of the latest round

    def put(self, keys: list[str], groups: list[str], publisher: str, now: float):
        """A put's start at `now`: take the bytes of the keys the Store does not
        hold, a read of those it has published; return the keys it took."""
        new = []
        for key, gid in zip(keys, groups, strict=True):
            if (obj := self.objects.get(key)) is None:
                new.append((key, gid))
            elif obj.published:
                self._lease(obj.group, now + self.ttl)
        need = sum(self.size(key) for key, _ in new)
        if self.used + need > self.capacity:
            raise MooncakeStoreRefusal(
                f"a put of {len(new)} objects ({need} B) at {now:.6f} s, with "
                f"{self.used} of the Store's {self.capacity} B in use: Mooncake "
                "fails it at once, which ATOM answers with its failed-save path"
            )
        for key, gid in new:
            self.objects[key] = _Object(gid, publisher)
            self.groups.setdefault(gid, _Group()).pending += 1
        self.used += need
        return [key for key, _ in new]

    def publish(self, keys: list[str], now: float) -> None:
        """A put's end at `now`: its objects become visible, and each group is
        filed at its lease deadline, which reaches at least `now`."""
        for key in keys:
            obj = self.objects[key]
            obj.published = True
            g = self.groups[obj.group]
            g.pending -= 1
            g.published.append(key)
            self.filed += 1
            self._lease(obj.group, now)
            self._file(obj.group, _bucket(g.deadline))

    def exists(self, keys: list[str], now: float) -> dict[str, str]:
        """Each of `keys` the Store has published, with the engine LP that put
        it; each one's group leased until `now` plus the lease TTL."""
        present = {}
        for key in keys:
            obj = self.objects.get(key)
            if obj is not None and obj.published:
                present[key] = obj.publisher
                self._lease(obj.group, now + self.ttl)
        return present

    def next_tick(self, now: float) -> float:
        """When the eviction thread next runs a round: the first 10 ms boundary
        at or after both `now` and the earliest lease deadline among filed
        groups, while used bytes are above the mark; else never."""
        if not self.used / self.capacity > self.mark:
            return math.inf
        heap = self._deadlines
        while heap and not self._live(*heap[0]):
            heapq.heappop(heap)
        if not heap:
            return math.inf
        return max(_wake(max(now, heap[0][0])), self._last_wake + 1) / TICKS_PER_S

    def evict(self, now: float) -> Round:
        """One eviction round at `now`."""
        target = max(self.ratio, self.used / self.capacity - self.mark + self.ratio)
        goal = math.ceil(self.filed * min(target, 1.0))
        self._last_wake = _wake(now)
        last, evicted, gone, skipped, kept = _bucket(now), 0, [], [], []
        while evicted < goal and self._filing and self._filing[0][0] <= last:
            rank, order, gid = heapq.heappop(self._filing)
            g = self.groups.get(gid)
            if g is None or (g.rank, g.order) != (rank, order):
                continue
            if (real := _bucket(g.deadline)) > rank:
                self._file(gid, real)
                skipped.append((gid, "moved"))
            elif now < g.deadline:
                kept.append((rank, order, gid))
                skipped.append((gid, "lease"))
            else:
                for key in g.published:
                    del self.objects[key]
                evicted += len(g.published)
                self.used -= sum(self.size(key) for key in g.published)
                self.filed -= len(g.published)
                g.published, g.rank = [], None
                if not g.pending:
                    del self.groups[gid]
                gone.append(gid)
        for entry in kept:
            heapq.heappush(self._filing, entry)
        return Round(target, goal, evicted, gone, skipped)

    def _lease(self, gid: str, deadline: float) -> None:
        """Extend group `gid`'s lease to `deadline`; it never shortens."""
        g = self.groups[gid]
        if deadline > g.deadline:
            g.deadline = deadline
            if g.rank is not None:
                self._push_deadline(gid, g)

    def _file(self, gid: str, rank: int) -> None:
        """File group `gid` last in bucket `rank`, unless it is filed there."""
        g = self.groups[gid]
        if g.rank != rank:
            g.rank, g.order = rank, next(self._order)
            heapq.heappush(self._filing, (rank, g.order, gid))
            self._push_deadline(gid, g)

    def _push_deadline(self, gid: str, g: _Group) -> None:
        heapq.heappush(self._deadlines, (g.deadline, gid))
        if len(self._deadlines) > 4 * len(self.groups) + 64:  # drop the stale
            self._deadlines = [e for e in self._deadlines if self._live(*e)]
            heapq.heapify(self._deadlines)

    def _live(self, deadline: float, gid: str) -> bool:
        g = self.groups.get(gid)
        return g is not None and g.rank is not None and g.deadline == deadline


class MooncakeStoreLP(StationLP):
    """LP ``mooncake_store``'s loop over `master`: its lookups and renewals on
    the master's handler threads, its puts' starts and ends and the eviction
    thread's rounds as daemon deadlines. `check_plans()` runs at the first
    stamped request, when every engine it serves has planned its pool."""

    def __init__(self, rt, sock, master: Master, *, lookup_s, check_plans):
        super().__init__(rt, sock, {"master": master.threads})
        self.master, self.per_key, self.hop = master, lookup_s[1], lookup_s[0] / 2
        self._check_plans = check_plans
        self.calls: dict[str, deque] = {}  # channel -> calls waiting, in turn
        self.free_at: dict[str, float] = {}  # channel -> when its next call can arrive
        self.calling: set[str] = set()  # channels with a call on the station
        self.events: list = []  # heap of (time, order, callable): puts' starts and ends
        self._order = itertools.count()
        self.counts = dict.fromkeys(COUNTS, 0)

    def _take(self, ch: str, arrival: float, frames: list[bytes]) -> None:
        if self._check_plans is not None:
            self._check_plans()
            self._check_plans = None
        identity, payload = frames
        kind, *body = pickle.loads(payload)
        if kind == "put":
            keys, groups, start, end = body
            engine = ch.partition("->")[0]

            def take(now: float) -> None:
                new = self.master.put(keys, groups, engine, now)
                self._at_time(max(end, arrival), lambda t: self.master.publish(new, t))

            self._at_time(max(start, arrival), take)
        elif kind in ("query", "renew"):
            calls = self.calls.setdefault(ch, deque())
            call = (kind, identity, *body, arrival)
            if kind == "renew":  # behind the renewals ahead, before the queries
                calls.insert(sum(1 for c in calls if c[0] == "renew"), call)
            else:
                calls.append(call)
        else:
            raise MooncakeStoreRefusal(f"LP mooncake_store takes no {kind!r} on {ch}")

    def clear(self, capacity: int) -> None:
        """Empty the Store and mount it again at `capacity` bytes, at the end
        of a client run (``clear_between_client_runs``), as a record's
        launcher runs ``store_check --clear`` between segments: every object
        and the put starts and publications still pending go. The calls
        queued on the master stay, answered from the empty Store."""
        self.events = []
        self.master.reset(capacity)

    def _at_time(self, t: float, action) -> None:
        heapq.heappush(self.events, (t, next(self._order), action))

    def _ready(self, ch: str) -> float:
        """When the call at the head of `ch`'s queue reaches the master."""
        return max(self.calls[ch][0][-1], self.free_at.get(ch, -math.inf))

    def _waiting(self) -> list[str]:
        return [c for c, q in self.calls.items() if q and c not in self.calling]

    def _next_job(self, station: str, now: float):
        ready = [(self._ready(c), c) for c in self._waiting()]
        ready = [r for r in ready if r[0] <= now]
        if not ready:
            return None
        start, ch = min(ready)
        return ch, start, self.calls[ch].popleft()

    def _start(self, station: str, job, now: float) -> tuple:
        ch, start, (kind, identity, rid, keys, _) = job
        self.calling.add(ch)
        engine = ch.partition("->")[0]

        def finish(end: float) -> list[bytes]:
            self.calling.discard(ch)
            self.free_at[ch] = end + 2 * self.hop
            present = self.master.exists(keys, end)
            if kind == "renew":
                self.counts["renewals"] += 1
            else:
                self.counts["lookups"] += 1
                self.counts["hit_keys"] += len(present)
                self.counts["foreign_hit_keys"] += sum(
                    p != engine for p in present.values()
                )
            reply = (rid, start, end + self.master.ttl, present)
            return [identity, pickle.dumps(reply)]

        return ch, now + self.per_key * len(keys), finish

    def _tick(self, now: float) -> None:
        while self.events and self.events[0][0] <= now:
            at, _, action = heapq.heappop(self.events)
            action(at)
        if self.master.next_tick(now) <= now:
            self.counts["evicted"] += self.master.evict(now).objects

    def _deadlines(self) -> tuple[float, float]:
        t, now = super()._deadlines()[0], self.rt.now
        later = [r for c in self._waiting() if (r := self._ready(c)) > now]
        put = self.events[0][0] if self.events else math.inf
        return min([t, *later]), min(put, self.master.next_tick(now))

    def _answer(self, identity: bytes, ch: str, frames: list[bytes]) -> None:
        raise MooncakeStoreRefusal(
            f"a frame on {ch} sent outside the run: ATOM's Store calls are in it"
        )
