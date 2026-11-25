# SPDX-License-Identifier: MIT
"""A simulated run's Prometheus series, each LP's on its own clock.

Each engine LP, or DP member, and each frontend LP samples the instruments its
own processes hold at ATOM's metrics timer: an engine just before its loop
calls ``utility_handler.push_metrics``, from its own process's and its
workers' multiprocess files (`sample_engine`); a frontend just after its
refresh calls `AtomMetricsExporter.update`, from the families the exporter's
snapshot collector renders and its own process's files (`sample_frontend`).
A sample reads no other LP's process, and makes no clock call: one that moved
its LP's clock is refused. Each point carries the LP clock it was taken at,
seconds from the run's start.

Only `FAMILIES` is written, each family by the LP that observes it, each
series once per sample but those of `ONCE`, with the labels CI's scrape gives
the side's series (`Sampler`). The LP writes
its samples when it hands in for the last time, before its hand-in, as
OpenMetrics 1.0 under gzip, the file whole or not at all (`Sampler.write`),
having sampled once more with its timer's read at its last finite clock,
which replaces a point at that instant (`Sampler.final`).
"""

import functools
import glob
import gzip
import os
from array import array
from pathlib import Path
from types import SimpleNamespace

from prometheus_client.metrics_core import Metric
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client.openmetrics.exposition import ALLOWUTF8, generate_latest
from prometheus_client.samples import Sample

from atom.utils import clock

#: Each family a sampler writes, by the kind of LP whose sampler writes it, with
#: the class of its value: ``S`` state read when sampled, ``E`` a tally of
#: events, ``D`` a duration whose observation takes the simulated clock. A
#: family not here is written by no sampler, and one here by no other kind's:
#: every engine and worker process also holds the API server's latency
#: instruments, unobserved, made as the spawned process imports the server's
#: module.
FAMILIES = {
    "frontend": {
        # Its snapshot of its engines.
        "atom:scheduler_requests": "S",
        "atom:scheduler_kv_cache_blocks": "S",
        "atom:prefix_cache_cached_tokens": "E",
        "atom:prefix_cache_full_tokens": "E",
        "atom:prefix_cache_offload_tokens": "E",
        # Its DP router's `share_cache` decisions (`ShareCacheRouter.DECISIONS`),
        # the requests that named their rank, and each rank's requests.
        "atom:dp_share_cache_hit": "E",
        "atom:dp_share_cache_source_rank": "E",
        "atom:dp_share_cache_cold": "E",
        "atom:dp_share_cache_stale": "E",
        "atom:dp_route_explicit": "E",
        "atom:dp_requests_routed": "E",
        # Its own.
        "atom:time_to_first_token_seconds": "D",
        "atom:inter_token_latency_seconds": "D",
    },
    "engine": {
        # Its scheduler's.
        "atom:request_queue_time_seconds": "D",
        "atom:pd_kv_transfer_seconds": "D",
        "atom:decode_batch_size": "E",
        "atom:prefill_request_tokens": "E",
        "atom:prefill_batch_tokens": "E",
        "atom:prefill_context_tokens": "E",
        "atom:decode_context_tokens": "E",
        "atom:prefill_request_context_tokens": "S",
        "atom:decode_request_context_tokens": "S",
        # Its workers'.
        "atom:gpu_forward_seconds": "D",
        "atom:prefill_request_gpu_forward_seconds": "D",
    },
}
#: Families whose each series is written once, at the first sample that holds
#: it: a request's context length, set once per request.
ONCE = frozenset(
    {"atom:prefill_request_context_tokens", "atom:decode_request_context_tokens"}
)
#: The role CI's scrape gives a side's series, by its prefill-decode side.
ROLES = {"P": "prefill", "D": "decode", None: "standalone"}
#: The file an LP writes its series to, in the out_dir, by its hand-in name.
FILE = "metrics-{}.om.gz"
_EOF = b"# EOF\n"
#: A histogram's samples at one instant, in the order a point lists them.
_ORDER = {"_bucket": 0, "_count": 1, "_sum": 2}


class Sampler:
    """The series of the `FAMILIES` of `kind`, ``frontend`` or ``engine``, that
    LP runtime `rt`'s processes hold, labelled as CI's scrape labels its
    side's: ``job="atom"``, ``observer="api"``, the side's `role` and as
    ``instance`` the side's frontend LP."""

    def __init__(self, rt, kind: str, role: str, instance: str) -> None:
        self.rt, self.written = rt, FAMILIES[kind]
        self.labels = {"job": "atom", "observer": "api"}
        self.labels |= {"role": role, "instance": instance}
        # By family: its header, and by series its samples' names and labels
        # and its points, each a time then the samples' values in that order.
        self.families: dict[str, tuple[tuple, dict]] = {}
        # The read its LP's metrics timer last sampled with.
        self.read = None

    def sample(self, collect) -> None:
        """Take the families `collect()` returns at the LP's clock, while its
        run is on and its runtime is the one installed, and keep `collect` as
        the timer's read. Refused, as a reason of the LP's hand-in, when
        collecting moved the clock or called it."""
        rt = self.rt
        if not rt.in_run or clock.installed() is not rt:
            return
        self.read = collect
        self._take(collect, rt.now)

    def final(self, t: float) -> None:
        """Sample once more with the timer's read, at the LP's last finite
        clock `t`: a series' point at `t` is replaced, not followed by a
        second. Nothing before the timer has sampled."""
        if self.read is not None:
            self._take(self.read, t, replace=True)

    def _take(self, collect, t: float, replace: bool = False) -> None:
        rt = self.rt
        now, calls = rt.now, rt.calls
        metrics = list(collect())
        if (rt.now, rt.calls) != (now, calls):
            why = f"sampling its Prometheus series moved or called its clock at {t}"
            rt.refusals.append(f"clock:{why}")
            raise RuntimeError(f"{rt.me}: {why}")
        for m in metrics:
            if m.name not in self.written:
                continue
            head = m.name, m.documentation, m.type, m.unit
            _, series = self.families.setdefault(m.name, (head, {}))
            for key, samples in _series(m):
                if key not in series:
                    labels = [(s.name, s.labels | self.labels) for s in samples]
                    series[key] = labels, array("d")
                elif m.name in ONCE:
                    continue
                labels, points = series[key]
                if replace and points and points[-len(labels) - 1] == t:
                    del points[-len(labels) - 1 :]
                points.append(t)
                points.extend(s.value for s in samples)

    def write(self, out: Path, lp: str) -> None:
        """Write the series to `FILE` for hand-in name `lp` in `out`, whole or
        not at all: OpenMetrics 1.0 under gzip, each family's series together,
        each series' points in time order."""
        path = out / FILE.format(lp)
        part = path.with_name(path.name + ".part")
        with gzip.open(part, "wb") as f:
            for head, series in self.families.values():
                family = Metric(*head)
                family.samples = [
                    s
                    for labels, points in series.values()
                    for s in _points(labels, points)
                ]
                f.write(_text(family))
            f.write(_EOF)
        part.replace(path)


def _text(family: Metric) -> bytes:
    """`family` in OpenMetrics text, less the end-of-file line."""
    one = SimpleNamespace(collect=lambda: [family])
    return generate_latest(one, escaping=ALLOWUTF8).removesuffix(_EOF)


def _series(m: Metric):
    """`m`'s samples by series, its labels less ``le``: a histogram's buckets
    in ascending order, then its count, then its sum."""
    by_series = {}
    for s in m.samples:
        key = tuple(sorted((k, v) for k, v in s.labels.items() if k != "le"))
        by_series.setdefault(key, []).append(s)
    cut = len(m.name)
    for key, samples in by_series.items():
        samples.sort(
            key=lambda s: (_ORDER.get(s.name[cut:], 0), float(s.labels.get("le", 0)))
        )
        yield key, samples


def _points(labels: list, points: array):
    """The samples of one series' `points`, each `labels` at its time."""
    step = len(labels) + 1
    for i in range(0, len(points), step):
        for (name, label), value in zip(labels, points[i + 1 : i + step]):
            yield Sample(name, label, value, points[i])


def _merged(pids) -> list:
    """The instruments in the multiprocess files of processes `pids`."""
    directory = os.environ["PROMETHEUS_MULTIPROC_DIR"]
    files = [f for p in pids for f in glob.glob(os.path.join(directory, f"*_{p}.db"))]
    return list(MultiProcessCollector.merge(files))


def sample_engine(sampler: Sampler, engine_core) -> None:
    """Have `engine_core` sample its own process's and its workers' files just
    before each ``push_metrics`` its loop calls, which sets off its workers'
    polls without waiting for them."""
    handler = engine_core.utility_handler
    push = handler.push_metrics
    pids = [os.getpid(), *(p.pid for p in engine_core.runner_mgr.procs)]

    @functools.wraps(push)
    def pushed(*args, **kwargs):
        sampler.sample(lambda: _merged(pids))
        return push(*args, **kwargs)

    handler.push_metrics = pushed


def sample_frontend(sampler: Sampler) -> None:
    """Have this process's `AtomMetricsExporter` sample just after each
    ``update``: the snapshot families it renders, and this process's files."""
    from atom.metrics.exporter import AtomMetricsExporter

    update = AtomMetricsExporter.update
    names = {f + suffix for f in FAMILIES["frontend"] for suffix in ("", "_total")}
    pid = os.getpid()

    @functools.wraps(update)
    def updated(self, snapshot) -> None:
        update(self, snapshot)
        snapshot_families = self.registry.restricted_registry(names)
        sampler.sample(lambda: [*snapshot_families.collect(), *_merged([pid])])

    AtomMetricsExporter.update = updated
