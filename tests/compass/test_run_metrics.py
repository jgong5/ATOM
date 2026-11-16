# SPDX-License-Identifier: MIT
"""A simulated run's Prometheus series: each engine and frontend LP samples
its own processes' instruments on its own clock, and writes them as it hands
in for the last time.

An engine and a frontend, each started by the run bootstrap on a standalone
authority's run file and connected to none, their clocks moved by hand, drive
ATOM's own instruments in multiprocess files: an engine its `SchedulerMetrics`
and its worker's `GPUForwardMetrics`, a frontend the API server's
`RequestMetrics`, `StreamMetrics` and `AtomMetricsExporter`. Each file, read
back with prometheus_client's OpenMetrics parser one instant at a time, holds
at each instant what ATOM's ``/metrics`` held then of the sampler's families,
from the LP's own processes, with CI's scrape labels.
"""

import contextlib
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from prometheus_client import CollectorRegistry, Gauge, generate_latest, values
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client.openmetrics import parser as openmetrics
from prometheus_client.parser import text_string_to_metric_families

from atom.compass import run as compass_run
from atom.compass.run import metrics
from atom.metrics.exporter import AtomMetricsExporter
from atom.metrics.gpu import GPUForwardMetrics
from atom.metrics.scheduler import SchedulerMetrics
from atom.utils import clock
from atom.utils.distributed import utils

INF = math.inf
#: Process ids above any the kernel hands out: the engine's worker's, and
#: another engine's.
WORKER, OTHER = 9_000_001, 9_000_002
#: The labels CI's scrape adds.
SCRAPE = ("job", "observer", "role", "instance")
KV_ROLES = {"P": "kv_producer", "D": "kv_consumer"}
#: Each side's LP names, and its series' role and instance.
SIDES = {
    None: ("engine", "frontend", "standalone"),
    "P": ("engine-P", "frontend-P", "prefill"),
    "D": ("engine-D", "frontend-D", "decode"),
}
#: A prefill of request 1, 100 prompt tokens of which 36 cached, then its
#: first decode.
PREFILL = SimpleNamespace(
    req_ids=[1],
    is_dummy_run=False,
    total_seqs_num_decode=0,
    total_seqs_num_prefill=1,
    total_tokens_num_prefill=64,
    num_cached_tokens=[36],
    context_lens=[100],
    prefill_gpu_requests=[(1, 1, True)],
)
DECODE = SimpleNamespace(
    req_ids=[1],
    is_dummy_run=False,
    total_seqs_num_decode=1,
    total_seqs_num_prefill=0,
    context_lens=[101],
    prefill_gpu_requests=(),
)
#: What an engine's workers say as it hands in.
SAID = {"refused_commands": [], "priced_steps": {}, "predicted_memory": {}}


class _Event:
    """A device event recorded at its engine's clock, done at once."""

    def record(self) -> None:
        self.t = clock.installed().now

    def query(self) -> bool:
        return True

    def elapsed_time(self, end) -> float:
        return (end.t - self.t) * 1000


def _config(side):
    kv = {} if side is None else {"kv_connector": "compass", "kv_role": KV_ROLES[side]}
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        tensor_parallel_size=1,
        enable_dp_attention=False,
        pipeline_parallel_size=1,
        enable_rapidserve=False,
        runner_qualname=compass_run.ATOM_RUNNER,
        kv_transfer_config=kv,
    )


@pytest.fixture
def files(monkeypatch, tmp_path) -> Path:
    """The multiprocess directory this process's instruments write to."""
    directory = tmp_path / "prometheus"
    directory.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(directory))
    monkeypatch.setattr(values, "ValueClass", values.MultiProcessValue())
    return directory


@pytest.fixture
def started(monkeypatch, tmp_path):
    """`start(kind, side)` starts an ``engine`` or ``frontend`` LP of `side`
    as the run bootstrap does, on a standalone authority's run file, and
    returns its runtime; the classes it wraps are ATOM's own again after."""
    from atom.model_engine.engine_core_mgr import CoreManager
    from atom.multimodal.media import MediaLoader

    for module, name in [
        (compass_run, "_authority"),
        (compass_run, "_member"),
        (clock, "_installed"),
        (utils, "LP_OF_RANK"),
    ]:
        monkeypatch.setattr(module, name, None, raising=False)
    for name in clock.COLLECTIVES:  # the engine's bootstrap wraps each
        monkeypatch.setattr(torch.distributed, name, getattr(torch.distributed, name))
    for cls, name in [
        (AtomMetricsExporter, "update"),
        (CoreManager, "broadcast_utility_command_sync"),
        (MediaLoader, "load_image"),
    ]:
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    monkeypatch.setenv(compass_run.CLOCK_ENV, "tcp://127.0.0.1:1")

    def start(kind: str, side=None):
        run = {"bound_s": 100.0, "out_dir": str(tmp_path)}
        if side is not None:
            run["router_s"] = 2.0**-10
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        with getattr(compass_run, kind)(_config(side)):
            pass
        rt = clock.installed()
        rt.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]), stopped=None)
        return rt

    return start


@pytest.fixture
def renames(monkeypatch) -> list:
    """Each rename into place, in order: the name it gives, and whether a file
    had it before."""
    done, replace = [], Path.replace

    def spied(self, target):
        done.append((Path(target).name, Path(target).exists()))
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", spied)
    return done


def _key(labels: dict) -> tuple:
    return tuple(sorted(labels.items()))


def _held(exposition: str) -> dict:
    """What ATOM's ``/metrics`` text `exposition` holds: by (family, sample,
    labels), each sample's value."""
    return {
        (f.name, s.name, _key(s.labels)): s.value
        for f in text_string_to_metric_families(exposition)
        for s in f.samples
    }


def _of_table(held: dict) -> dict:
    """`held`, by instant, of the sampler's families alone."""
    return {
        t: {k: v for k, v in at.items() if k[0] in metrics.FAMILIES}
        for t, at in held.items()
    }


def _instants(path: Path, role: str, instance: str) -> dict:
    """The file at `path` read back one instant at a time, each instant's
    samples on their own under every family's header, by the OpenMetrics
    parser, which keeps a histogram series' first point alone: by time, what
    it holds then by (family, sample, labels less the scrape's). Each sample
    carries CI's scrape labels for `role` and `instance`."""
    scrape = {"job": "atom", "observer": "api", "role": role, "instance": instance}
    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    assert lines.pop() == "# EOF"
    stamp = lambda line: line.rsplit(" ", 1)[1]
    by_time = {}
    for t in dict.fromkeys(stamp(line) for line in lines if line[0] != "#"):
        alone = [line for line in lines if line[0] == "#" or stamp(line) == t]
        at = by_time[float(t)] = {}
        text = "\n".join([*alone, "# EOF", ""])
        for f in openmetrics.text_string_to_metric_families(text):
            for s in f.samples:
                assert {k: s.labels[k] for k in SCRAPE} == scrape, s
                labels = {k: v for k, v in s.labels.items() if k not in SCRAPE}
                at[(f.name, s.name, _key(labels))] = s.value
    return by_time


def _first_holding(held: dict) -> dict:
    """`held`, by instant, less each gauge of a family written once at every
    instant after the first that holds it."""
    seen, kept = set(), {}
    for t, at in held.items():
        kept[t] = {
            k: v for k, v in at.items() if k[0] not in metrics.ONCE or k not in seen
        }
        seen |= at.keys()
    return kept


def _whole_before_hand_in(renames: list, out: Path, lp: str) -> None:
    """LP `lp`'s file was renamed into place, onto no file of its name, before
    its hand-in was, and nothing part-written is left in `out`."""
    order = [name for name, _ in renames]
    assert (metrics.FILE.format(lp), False) in renames
    assert order.index(metrics.FILE.format(lp)) < order.index(f"lp-{lp}.json")
    assert not list(out.glob("*.part"))


@contextlib.contextmanager
def _process(value_class):
    """Inside, a value prometheus_client makes is in `value_class`'s process's
    files: a labelled child is made at its first use."""
    own, values.ValueClass = values.ValueClass, value_class
    try:
        yield
    finally:
        values.ValueClass = own


def _engine(files, started, side, loop):
    """An engine LP of `side` whose scheduler and worker observe a prefill and
    a decode between metrics pushes at 0, 1 and 2 s, the worker's device times
    landing at the push after them, as ATOM's push polls its workers. Another
    engine's scheduler and an unclassified gauge in the engine's own process
    observe too. `loop(push, steps)` runs inside `engine_loop`, `steps()` the
    pushes at 0 and 1 s and the steps around them; returns the engine's runtime
    and what ``/metrics`` held just before each push."""
    worker, another = (values.MultiProcessValue(lambda p=p: p) for p in (WORKER, OTHER))
    with _process(worker):
        gpu = GPUForwardMetrics(_Event)
    with _process(another):
        other = SchedulerMetrics(7)
    scheduler = SchedulerMetrics(0)
    unclassified = Gauge("atom:unclassified", "No class.", registry=None)
    rt = started("engine", side)
    pushed, held = [], {}
    core = SimpleNamespace(
        utility_handler=SimpleNamespace(
            push_metrics=lambda: (pushed.append(rt.now), gpu.poll())
        ),
        runner_mgr=SimpleNamespace(
            procs=[SimpleNamespace(pid=WORKER)],
            call_func=lambda name, wait_out: SAID[name],
        ),
    )
    registry = CollectorRegistry()
    MultiProcessCollector(registry, path=str(files))

    def push(t: float) -> None:
        rt.now, calls = t, rt.calls
        held[t] = _held(generate_latest(registry).decode())
        core.utility_handler.push_metrics()
        assert (rt.now, rt.calls) == (t, calls)

    def steps() -> None:
        push(0.0)
        rt.now = 0.25
        seq = SimpleNamespace(external_request_id="request-1", num_prompt_tokens=100)
        theirs = SimpleNamespace(external_request_id="request-7", num_prompt_tokens=50)
        scheduler.enqueue(seq)
        other.enqueue(theirs)
        rt.now = 0.5
        scheduler.record_forward(PREFILL, {1: seq})
        with _process(another):
            other.record_forward(PREFILL, {1: theirs})
        with gpu.measure(PREFILL):
            rt.now = 0.75
        unclassified.set(1)
        push(1.0)
        rt.now = 1.25
        scheduler.record_forward(DECODE, {1: seq})
        with gpu.measure(DECODE):
            rt.now = 1.5

    with compass_run.engine_loop(core):
        loop(push, steps)
    assert pushed == list(held)
    return rt, held


@pytest.mark.parametrize("side", SIDES)
def test_an_engine_writes_what_its_processes_held_at_each_push(
    files, started, renames, tmp_path, side
):
    """Each instant of the file is a push's, at the engine's clock, and holds
    what ``/metrics`` held of the families just before the push, from the
    engine's process and its worker's alone, each request's context once;
    the file is renamed into place whole before the engine's hand-in."""

    def loop(push, steps):
        steps()
        push(2.0)
        clock.installed().now = INF  # the finish

    rt, held = _engine(files, started, side, loop)
    lp, frontend, role = SIDES[side]
    written = _instants(tmp_path / metrics.FILE.format(lp), role, frontend)
    print(f"\n{lp}: {sum(map(len, written.values()))} samples at {list(written)}")
    assert list(written) == [0.0, 1.0, 2.0]
    # Another engine's series, which ``/metrics`` holds too, are not this one's.
    ours = {
        t: {k: v for k, v in at.items() if ("dp_rank", "7") not in k[2]}
        for t, at in _of_table(held).items()
    }
    assert written == _first_holding(ours)
    assert any(("dp_rank", "7") in k[2] for k in _of_table(held)[2.0])
    assert any(k[0] == "atom:unclassified" for k in held[2.0])
    # The worker's device time of the prefill landed after the push at 1 s,
    # and the decode's after the one at 2 s.
    count = "atom:gpu_forward_seconds", "atom:gpu_forward_seconds_count"
    assert [v for k, v in written[2.0].items() if k[:2] == count] == [1.0]
    # Each request's context, at the first push after its dispatch alone.
    for family, at in (("prefill", [1.0]), ("decode", [2.0])):
        name = f"atom:{family}_request_context_tokens"
        assert [t for t in written for k in written[t] if k[0] == name] == at
    _whole_before_hand_in(renames, tmp_path, lp)
    assert rt.refusals == []


def test_an_engine_that_ends_early_keeps_what_it_sampled(files, started, tmp_path):
    """An engine whose loop raises after its push at 1 s hands in the
    exception, its file holding the pushes at 0 and 1 s."""

    def loop(push, steps):
        steps()
        raise RuntimeError("the engine's loop died")

    with pytest.raises(RuntimeError, match="died"):
        _engine(files, started, None, loop)
    written = _instants(tmp_path / "metrics-engine.om.gz", "standalone", "frontend")
    assert list(written) == [0.0, 1.0]
    said = json.loads((tmp_path / "lp-engine.json").read_text())
    assert any("died" in r for r in said["refusals"])


def _snapshot(running: int, used: int, cached: int, full: int) -> dict:
    """The engines' snapshot a frontend refresh takes: one DP rank's requests
    and KV blocks, and the prefix cache's counters."""
    return {
        "enabled": True,
        "requests_running": running,
        "scheduler_metrics": [
            {
                "dp_rank": 0,
                "engine_role": "default",
                "running": running,
                "waiting": 1 - running,
                "waiting_kv": 0,
                "kv_blocks": {"used": used, "evictable": 0, "vacant": 64 - used},
            }
        ],
        "cache": {"cached_tokens": cached, "full_tokens": full, "offload_tokens": 0},
    }


def _no_gc(exposition: str) -> list:
    """`exposition` less this process's collector's lines, which a collection
    between two renders moves."""
    return [line for line in exposition.splitlines() if "atom:gc_" not in line]


def _frontend(started, side, stop=None):
    """A frontend LP of `side` that refreshes at 0, 1 and 2 s, with a
    streamed and an unstreamed request's first output and an output interval
    between them, then finishes at 2.5 s, or with `stop` is stopped by it then.
    Returns its runtime and what ``/metrics`` held just after each refresh,
    checking that it renders what an exporter ATOM's own update alone updates
    does."""
    from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
    from atom.entrypoints.openai.streaming_dispatch import longest_silence_seconds

    own = AtomMetricsExporter.update
    rt = started("frontend", side)
    exporter, requests, streams = create_metrics_exporter()
    plain = AtomMetricsExporter(stream_silence=longest_silence_seconds)
    MultiProcessCollector(plain.registry)
    held = {}

    def refresh(t: float, snapshot: dict) -> None:
        rt.now, calls = t, rt.calls
        exporter.update(snapshot)
        assert (rt.now, rt.calls) == (t, calls)
        own(plain, snapshot)
        rendered = exporter.render().decode()
        assert _no_gc(rendered) == _no_gc(plain.render().decode())
        held[t] = _held(rendered)

    refresh(0.0, _snapshot(0, 0, 0, 0))
    rt.now = 0.5
    requests.observe_time_to_first_token(0.5, True)
    streams.observe_inter_token_latency(0.25, 2)
    refresh(1.0, _snapshot(1, 2, 64, 100))
    rt.now = 1.75
    requests.observe_time_to_first_token(1.25, False)
    refresh(2.0, _snapshot(0, 2, 64, 200))
    rt.now = 2.5
    if stop is None:
        rt.now = INF  # the finish
    rt.loop.stopped = stop
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    return rt, held


@pytest.mark.parametrize("side", SIDES)
def test_a_frontend_writes_what_it_held_at_each_refresh(
    started, files, renames, tmp_path, side
):
    """Each instant of the file is a refresh's, at the frontend's clock, and
    holds what ``/metrics`` held of the families just after it: the snapshot
    families and the frontend's own latencies; ``/metrics`` itself is what
    ATOM's own update gives."""
    rt, held = _frontend(started, side)
    _, lp, role = SIDES[side]
    written = _instants(tmp_path / metrics.FILE.format(lp), role, lp)
    print(f"\n{lp}: {sum(map(len, written.values()))} samples at {list(written)}")
    assert written == _of_table(held)
    assert any(k[0] == "atom:requests_running" for k in held[2.0])
    assert {k[0] for k in written[2.0]} == {
        "atom:scheduler_requests",
        "atom:scheduler_kv_cache_blocks",
        "atom:prefix_cache_cached_tokens",
        "atom:prefix_cache_full_tokens",
        "atom:prefix_cache_offload_tokens",
        "atom:time_to_first_token_seconds",
        "atom:inter_token_latency_seconds",
    }
    _whole_before_hand_in(renames, tmp_path, lp)
    assert rt.refusals == []


def test_a_stopped_frontend_keeps_what_it_sampled(started, files, tmp_path):
    """A frontend a signal stops at 2.5 s hands in the signal, its file holding
    every refresh it took."""
    _, held = _frontend(started, None, stop="SIGTERM")
    written = _instants(tmp_path / "metrics-frontend.om.gz", "standalone", "frontend")
    assert written == _of_table(held)
    said = json.loads((tmp_path / "lp-frontend.json").read_text())
    assert said["stopped"] == "SIGTERM"


def test_a_sample_that_moves_its_lp_s_clock_is_refused(monkeypatch, files, started):
    """A sample whose collection moved the LP's clock raises and leaves the
    LP's hand-in a reason."""
    rt = started("engine")
    core = SimpleNamespace(
        utility_handler=SimpleNamespace(push_metrics=lambda: None),
        runner_mgr=SimpleNamespace(procs=[]),
    )
    metrics.sample_engine(rt.sampler, core)

    def moving(pids):
        rt.now += 0.5
        return []

    monkeypatch.setattr(metrics, "_merged", moving)
    with pytest.raises(RuntimeError, match="moved or called its clock"):
        core.utility_handler.push_metrics()
    assert rt.refusals == [
        "clock:sampling its Prometheus series moved or called its clock at 0.0"
    ]


def test_with_compass_off_atom_s_metrics_are_its_own(monkeypatch):
    """With no run file, Compass off or a real run, the frontend's refresh and
    the engine's push are ATOM's own, and no sampler runs to write a file."""
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setattr(clock, "_installed", None)
    update = AtomMetricsExporter.update
    assert isinstance(compass_run.frontend(_config(None)), contextlib.nullcontext)
    assert AtomMetricsExporter.update is update
    push = lambda: None
    core = SimpleNamespace(utility_handler=SimpleNamespace(push_metrics=push))
    with compass_run.engine_loop(core):
        assert core.utility_handler.push_metrics is push
