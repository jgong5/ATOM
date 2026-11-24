# SPDX-License-Identifier: MIT
"""`lmcache_mp` held to the scenario checks: one deployment, and two DP ranks
sharing one server, each run twice under a constant step stub and constant
station laws.

The workload is the parity workload's with a shared prefix
(`atom.compass.parity.traffic.bodies`): request 0 stores the prefix, its
blocks leave HBM when it ends since prefix caching is off, and a burst of
three requests per DP rank later loads it back. With
``OFFLOAD_ASYNC_LOOKUP_DEPTH=1`` two requests of each burst reach admission
unprefetched and look up blocking, and with ``OFFLOAD_LOOKUP_DEFER_S=0`` the
prefetched one is deferred once, then looks up blocking too: a lookup job is
priced above a step, so its reply is not in by the next pass.

Two more runs serve DeepSeek-V4.1-Flash as the prefill instance of ROCm/ATOM's
6P16D C1228 share-cache recipe (`P_INSTANCE`), routed by ``--dp-load-balance
share_cache`` over this tier in place of the recipe's ``mooncake_store``, with
prefix caching on: request 0's blocks stay in its rank's HBM, and the burst
goes to both ranks, the other one loading the prefix from the server. They
need `ATOM_COMPASS_DSPARK_MODEL`, as `test_spec_scenario` does, read readings
published for them, as its DSpark run does, and no driver.

The runs need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`, and as many GPUs in ``HIP_VISIBLE_DEVICES`` as DP ranks.
"""

import gzip
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

import pytest
from test_lmcache_lp import lmcache_port  # noqa: F401  the lmcache_port fixture
from test_lmcache_mp_engine import on_cards
from test_parity_record import _config as _parity_config
from test_parity_record import _key
from test_parity_record import _Runner as _SimRunner
from test_vertical_slice import (
    CONSTANT,
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    Traffic,
    _run_file,
    publish_entry_on_arch,
)

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.carriers import tracestate_with
from atom.compass.clock import TAR
from atom.compass.detect.scenario import check_run, expected_timeline
from atom.compass.drive.serve import ON_ARCH, SERVER
from atom.compass.kv.lmcache_server import reply_channel
from atom.compass.parity import ENV as RECORD
from atom.compass.parity import StepRecording, read
from atom.compass.parity.traffic import bodies
from atom.compass.run import metrics
from atom.compass.runner import overrides
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.model_engine.dp_routing.share_cache import ShareCacheRouter
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from tests.compass.held_ports import held_ports

PREFIX_WORDS = 600  # two whole 256-token chunks
BURST_S, BURST = 0.5, 3  # the burst's time, and its requests per DP rank
STATION_S = 2.0**-6  # every request-loop job: above a step under `CONSTANT`
COPY_S = 2.0**-7  # every copy: the host link's latency, its rate adding nothing
OFFLOAD_ENV = {
    "OFFLOAD_ASYNC_LOOKUP_DEPTH": "1",
    "OFFLOAD_LOOKUP_DEFER_S": "0",
    "OFFLOAD_MIN_LOAD_TOKENS": "256",
    "OFFLOAD_MIN_SAVE_TOKENS": "256",
}
ATOM_ARGS = ["--model", MODEL, "--enforce-eager", "--max-model-len", "2048"]
DP2 = ["-tp", "2", "--enable-dp-attention", "--enable-expert-parallel"]
CHUNK = 256  # the server's chunk, which `share_cache` needs ATOM to state too
#: The machine the recipe's prefill instance runs on: one MI355X node.
CRS005 = TREE / "tests/compass/machines/mi355x-crs005.json"
#: DeepSeek-V4.1-Flash's model directory, a config and a tokenizer, as
#: `test_spec_scenario` takes it; no weight is read.
DSV41 = os.environ.get("ATOM_COMPASS_DSPARK_MODEL")
#: The prefill instance of ROCm/ATOM's DeepSeek-V4.1 6P16D C1228 share-cache
#: recipe: two attention DP ranks over TP2, its MoE path, KV and index dtypes,
#: HBM share, scheduler bounds and block size, prefix caching, routed by
#: `share_cache`; eager, with no draft, under a short context.
P_INSTANCE = ["--model", DSV41, "--enforce-eager", "--max-model-len", "2048"]
P_INSTANCE += ["--trust-remote-code", "-tp", "2", "--enable-dp-attention"]
P_INSTANCE += ["--moe-backend", "standard", "--all2all-backend", "none"]
P_INSTANCE += ["--kv_cache_dtype", "bf16", "--index-cache-dtype", "fp8"]
P_INSTANCE += ["--gpu-memory-utilization", "0.85", "--max-num-seqs", "64"]
P_INSTANCE += ["--max-num-batched-tokens", "32768"]
P_INSTANCE += ["--attn-prefill-chunk-size", "65536"]
P_INSTANCE += ["--block-size", "128", "--enable_prefix_caching"]
P_INSTANCE += ["--dp-load-balance", "share_cache"]
#: A `share_cache` run's frontend metrics refresh: often enough that its
#: metrics file holds a sample taken after the burst is placed.
METRICS_S = 2.0**-5


class Workload(Traffic):
    """The traffic LP: request 0 at 0, the rest at `BURST_S`."""

    def __init__(self, requests, *args) -> None:
        super().__init__(*args)
        self.requests = requests

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            at = BURST_S if k else 0.0
            while rt.now < at:
                rt.next_event(at)
                self._take()
            self.sent, seq = rt.stamp_send(self.http)
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()


def _server(port: int) -> list[str]:
    """The lmcache server's flags, which LP lmcache reads from the run file."""
    flags = ["--host", "127.0.0.1", "--port", str(port), "--chunk-size", str(CHUNK)]
    return flags + ["--l1-size-gb", "1", "--eviction-policy", "LRU"]


def _offload(port: int) -> dict[str, str]:
    """The offload environment of a run and of the readings it reads."""
    extra = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": port}
    extra["lmcache.chunk_size"] = CHUNK
    return OFFLOAD_ENV | {
        "ATOM_KV_OFFLOAD": "lmcache_mp",
        "ATOM_KV_OFFLOAD_EXTRA_CONFIG": json.dumps(extra),
    }


def _measure(out: Path, machine: Path, width: list, port: int) -> None:
    """The readings, taken with a real lmcache server beside the engine."""
    from atom.compass import cli

    gpus = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    beside = shlex.join(["lmcache", "server", *_server(port)])
    cli.main(
        ["measure", "memory", "--gpu", gpus]
        + ["--beside", beside]
        + ["--store", str(out / "store"), "--out", str(out / "measure")]
        + ["--", *ATOM_ARGS, *width]
    )


def run_scenario(
    root: Path,
    seed: str,
    dp: int,
    server_port: int,
    ports: list[int],
    argv: list,
    named: tuple = (),
) -> dict:
    """One run in its own process tree of a server of flags `argv`, reading
    the readings in ``root/store``, its clock and server on `ports`; with
    `named`, request k names DP rank ``named[k]``."""
    out = root / f"seed{seed}"
    out.mkdir()
    (clock_port, port), flags = ports, _server(server_port)
    machine = json.loads((root / "machine.json").read_text())
    run_file = _run_file(
        out,
        clock_endpoint=f"tcp://127.0.0.1:{clock_port}",
        step_stub=CONSTANT,
        machine=machine,
        store=str(root / "store"),
        data_parallel_size=dp,
        bound_s=60.0,
        lmcache={"server": flags, "lookup_s": [STATION_S, 0.0], "request_s": STATION_S},
    )
    env = dict(
        os.environ,
        **_offload(server_port),
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
    )
    model = argv[argv.index("--model") + 1]
    requests = bodies(model, 1 + BURST * dp, PREFIX_WORDS)
    for body, rank in zip(requests, named, strict=bool(named)):
        body["data_parallel_rank"] = rank
    # As `serve` starts one: on its machine's arch, so it needs no driver.
    arch = machine["device"]["arch"]
    with open(out / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, *ON_ARCH, arch, *SERVER[1:], *argv]
            + ["--host", "127.0.0.1", "--server-port", str(port)]
            + ["--compass-run", str(run_file)],
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        watchdog = threading.Timer(WALL_S, server.kill)
        watchdog.start()
        try:
            traffic = Workload(requests, json.loads(run_file.read_text()), port, server)
            traffic.run()
            code = server.wait(timeout=120)
        finally:
            watchdog.cancel()
            if server.poll() is None:
                server.kill()
    text = (out / "server.log").read_text()
    assert code == 0, f"the server exited {code} after the finish:\n{text[-4000:]}"
    return {
        "out": out,
        "run": json.loads(run_file.read_text()),
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "log": text,
        "sent": [r["max_tokens"] for r in requests],
        "taken": [line for _, line in traffic.events],
    }


def _priced_machine(root: Path, machine: dict) -> dict:
    """`machine` with a copy priced at `COPY_S`, as ``root/machine.json``."""
    machine["interconnect"]["host_link"] = {
        "bandwidth_bytes_per_s": 1e30,
        "latency_s": COPY_S,
    }
    origins = machine["provenance"]["origins"]
    for field in ("bandwidth_bytes_per_s", "latency_s"):
        origins[f"interconnect.host_link.{field}"] = {"origin": "assumed"}
    (root / "machine.json").write_text(json.dumps(machine))
    return machine


@pytest.fixture(scope="module", params=[1, 2], ids=["dp1", "dp2"])
def runs(request, tmp_path_factory, lmcache_port):  # noqa: F811
    dp = request.param
    root = tmp_path_factory.mktemp(f"lmcache_dp{dp}")
    _priced_machine(root, on_cards(os.environ.get("HIP_VISIBLE_DEVICES", "0")))
    width = DP2 if dp > 1 else []
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AITER_LOG_LEVEL", "WARNING")  # the readings are keyed by it
        for name, value in _offload(lmcache_port).items():
            mp.setenv(name, value)
        _measure(root, root / "machine.json", width, lmcache_port)
    argv = [*ATOM_ARGS, *width, "--no-enable_prefix_caching"]
    # The run file names the lmcache port, so both runs share it.
    with held_ports(4) as ports:
        return dp, [
            run_scenario(root, seed, dp, lmcache_port, ports[2 * i : 2 * i + 2], argv)
            for i, seed in enumerate(("1", "2"))
        ]


def _p_instance_runs(root: Path, port: int, named: tuple = ()) -> list[dict]:
    """Two seeds' runs of the recipe's prefill instance, `P_INSTANCE`, on
    `CRS005`, reading readings published for it, as `test_spec_scenario`'s
    DSpark run does. They need no driver."""
    if not DSV41:
        pytest.skip("a run needs ATOM_COMPASS_DSPARK_MODEL")
    spec = _priced_machine(root, json.loads(CRS005.read_text()))
    env = dict(os.environ, **_offload(port), AITER_LOG_LEVEL="WARNING")
    publish_entry_on_arch(root / "store", spec, P_INSTANCE, env)
    with held_ports(4) as ports, pytest.MonkeyPatch.context() as mp:
        mp.setenv("ATOM_METRICS_UPDATE_INTERVAL_S", str(METRICS_S))
        return [
            run_scenario(
                root, seed, 2, port, ports[2 * i : 2 * i + 2], P_INSTANCE, named
            )
            for i, seed in enumerate(("1", "2"))
        ]


@pytest.fixture(scope="module")
def share_cache_runs(tmp_path_factory, lmcache_port):  # noqa: F811
    """`share_cache` choosing each request's rank."""
    return _p_instance_runs(tmp_path_factory.mktemp("share_cache"), lmcache_port)


#: Each request's DP rank as the recipe's ``--dp-aware`` router names one:
#: request 0 and the burst's last on rank 0, the rest of the burst on rank 1,
#: which holds no prefix of its own.
NAMED = (0, *[1] * (2 * BURST - 1), 0)


@pytest.fixture(scope="module")
def named_runs(tmp_path_factory, lmcache_port):  # noqa: F811
    """Each request naming its rank (`NAMED`), as the recipe's router does."""
    root = tmp_path_factory.mktemp("share_cache_named")
    return _p_instance_runs(root, lmcache_port, NAMED)


def _releases(table: str, lp=None) -> list[tuple[float, str, int]]:
    """``(arrival, channel, seq)`` of each message released, to `lp` alone if named."""
    return sorted(
        (float(f[2]), f[4], int(f[5]))
        for f in map(str.split, table.splitlines()[1:])
        if f[3] == "release" and lp in (None, f[0])
    )


def station_timeline(table: str, ipc: float) -> dict[str, list[float]]:
    """Each reply channel's arrivals, worked out from the requests LP lmcache
    was released alone: each station starts a job at the request's arrival or
    once its previous job ends, whichever is later, in ``(arrival, channel,
    seq)`` order; a request-loop job takes `STATION_S` and a copy `COPY_S`;
    the reply arrives `ipc` after the job ends."""
    free, want = {}, defaultdict(list)
    for arrival, ch, _ in _releases(table, "lmcache"):
        station = ":transfer#" in ch
        free[station] = max(arrival, free.get(station, 0.0))
        free[station] += COPY_S if station else STATION_S
        want[reply_channel(ch)].append(free[station] + ipc)
    return want


def _sends(run, rank) -> dict[str, list[float]]:
    """Each of DP rank `rank`'s channels to LP lmcache: its sends' times, in seq
    order, each its arrival less the channel's lookahead."""
    ipc = run["run"].get("lmcache_ipc_s", run["run"]["ipc_s"])
    sent = defaultdict(list)
    for arrival, ch, _ in sorted(
        _releases(run["table"], "lmcache"), key=lambda r: r[1:]
    ):
        if ch.endswith(f"#dp{rank}"):
            sent[ch].append(arrival - ipc)
    return sent


def step_starts(table: str, record) -> list[float]:
    """Each step's start, its first TAR row's clock, as `expected_timeline`
    lays the record's steps over the engine's TAR rows."""
    tars = [
        (float(f[1]), float(f[2]))
        for f in map(str.split, table.splitlines()[1:])
        if f[0] == "engine" and f[3] == TAR
    ]
    stub = Coefficients(**CONSTANT)
    rows = (stub.prefill_step, stub.decode_step, tars)
    n = len(record[0])
    return [
        tars[len(expected_timeline({r: s[:k] for r, s in record.items()}, *rows))][0]
        for k in range(n)
    ]


# --- (a) ----------------------------------------------------------------------


def _meet_every_check_and_load_the_prefix_back(seeds: list, dp: int) -> None:
    """Both seeds' runs `seeds` meet every scenario check, and a request loads
    the stored prefix back, at DP width above one on a rank that did not store
    it."""
    run = seeds[0]
    failed = check_run(
        [r["out"] for r in seeds],
        run["sent"],
        run["taken"],
        {"engine": run["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)
    transfers = re.findall(
        r"compass: lmcache_mp (\w+) of (\S+):\S+ (\w+): arrived", run["log"]
    )
    print("", *(" ".join(t) for t in transfers), sep="\n")
    stored = [e for op, e, ok in transfers if op == "store" and ok == "succeeded"]
    loaded = {e for op, e, ok in transfers if op == "retrieve" and ok == "succeeded"}
    assert loaded, "no request loaded the stored prefix back"
    if dp > 1:
        assert loaded - {stored[0]}, "no rank loaded a prefix another rank stored"


@NEEDS_A_RUN
def test_each_run_meets_every_scenario_check_and_loads_the_prefix_back(runs):
    _meet_every_check_and_load_the_prefix_back(runs[1], runs[0])


def _last(out: Path, family: str) -> dict[str, float]:
    """Each series of counter `family` in the frontend's metrics file at its
    last sample, by its ``rank`` label, ``""`` for none."""
    text = gzip.decompress((out / metrics.FILE.format("frontend")).read_bytes())
    last = {}
    for line in text.decode().splitlines():
        if line.startswith(family + "_total{"):
            rank = re.search(r'rank="(\d+)"', line)
            last[rank[1] if rank else ""] = float(line.split()[-2])
    return last


def _decisions(run: dict) -> dict[str, float]:
    """The frontend's `share_cache` decision counts at the run's last sample."""
    families = {
        key: "atom:dp_" + key.removesuffix("_total")
        for key in ShareCacheRouter.DECISIONS
    }
    return {key: _last(run["out"], family)[""] for key, family in families.items()}


def test_share_cache_runs_meet_every_scenario_check_and_place_a_hit(
    share_cache_runs,
):
    """The recipe's prefill instance under `share_cache`: both seeds' runs
    meet every scenario check, a rank loads the prefix the other stored, and
    the frontend counts a placement on a prefix the shared tier holds."""
    _meet_every_check_and_load_the_prefix_back(share_cache_runs, 2)
    for run in share_cache_runs:
        decisions = _decisions(run)
        print(f"\n{run['out'].name}: {decisions}")
        assert decisions["share_cache_hit_total"] > 0, decisions


def test_named_runs_meet_every_scenario_check_and_keep_each_request_on_its_rank(
    named_runs,
):
    """The recipe's prefill instance, each request naming its rank as the
    recipe's ``--dp-aware`` router does: both seeds' runs meet every scenario
    check, the router counts every request explicit and places each on the
    rank it names, choosing none, and rank 1 loads the prefix rank 0 stored."""
    _meet_every_check_and_load_the_prefix_back(named_runs, 2)
    want = {str(r): float(NAMED.count(r)) for r in set(NAMED)}
    for run in named_runs:
        explicit = _last(run["out"], "atom:dp_route_explicit")[""]
        routed = _last(run["out"], "atom:dp_requests_routed")
        print(f"\n{run['out'].name}: explicit {explicit}, per rank {routed}")
        print(f"{run['out'].name}: {_decisions(run)}")
        assert explicit == len(NAMED)
        assert routed == want
        assert not any(_decisions(run).values())


# --- (b) ----------------------------------------------------------------------


@NEEDS_A_RUN
def test_each_server_job_and_reply_follows_the_constant_station_laws(runs):
    _, (run, _) = runs
    got = defaultdict(list)
    for arrival, ch, _ in sorted(_releases(run["table"]), key=lambda r: r[1:]):
        if ch.startswith("lmcache->"):
            got[ch].append(arrival)
    want = station_timeline(run["table"], run["run"]["ipc_s"])
    assert sorted(got) == sorted(want)
    for ch in want:
        assert got[ch] == pytest.approx(want[ch], rel=0, abs=1e-12), ch


@NEEDS_A_RUN
def test_each_offload_decision_is_sent_once_from_each_writer_before_its_forward(runs):
    """A step line's load or save goes out once from the one writer rank of
    each DP rank (TP 1), on that rank's transfer channel, after the previous
    step starts and by its own step's start."""
    _, (run, _) = runs
    record = read(run["out"] / "parity")
    starts = step_starts(run["table"], record)
    for rank, steps in record.items():
        decided = [k for k, step in enumerate(steps) for _ in step.get("kv", ())]
        ch = f"engine->lmcache:transfer#dp{rank}"
        sent = _sends(run, rank)[ch]
        assert len(sent) == len(decided), ch
        for k, at in zip(decided, sent):
            assert (starts[k - 1] if k else 0.0) <= at <= starts[k], (ch, k, at)


TAKEN = re.compile(
    r"lmcache_mp (store|retrieve) of atom-offload-(?:\w+-)?dp(\d+):(\d+) \w+: "
    r"arrived at (\S+), taken at (\S+)"
)
REPORTED = re.compile(r"lmcache_mp reports at (\S+): (.*)")


@NEEDS_A_RUN
def test_a_load_parked_request_is_released_no_earlier_than_its_load_is_reported(runs):
    """The shell takes each reply no earlier than it arrives and reports it
    once, no earlier than it took it; the request a load parked prefills on a
    step that starts no earlier than that report."""
    _, (run, _) = runs
    takes, reports, retrieved = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in run["log"].splitlines():
        if m := TAKEN.search(line):
            op, rank, rid, arrived, at = m.groups()
            assert float(at) >= float(arrived), line
            takes[op == "retrieve", rid].append(float(at))
            if op == "retrieve":
                retrieved[int(rank)].append(rid)
        elif m := REPORTED.search(line):
            for kind, rid in re.findall(r"(Save|Load)OperationId\(req_id=(\d+)", m[2]):
                reports[kind == "Load", rid].append(float(m[1]))
    assert takes.keys() == reports.keys()
    for op, at in takes.items():
        assert len(reports[op]) == len(at), op
        assert all(r >= t for t, r in zip(at, reports[op])), op
    record = read(run["out"] / "parity")
    starts = step_starts(run["table"], record)
    for rank, steps in record.items():
        loads = [
            (k, key)
            for k, step in enumerate(steps)
            for key, kind, *_ in step.get("kv", ())
            if kind == "load"
        ]
        for (k, key), rid in zip(loads, retrieved[rank], strict=True):
            p = next(
                j
                for j in range(k, len(steps))
                if key in {r[0] for r in steps[j]["batch"]}
            )
            assert starts[p] >= reports[True, rid][0], (rank, key, rid, p)


@NEEDS_A_RUN
def test_each_dp_member_parks_once_a_step_and_leaves_at_the_latest_park(runs):
    """Each member parks once before each forward and sends LP lmcache nothing
    while parked; every member leaves when the last one parks, as the step
    starts. Each member's park CPU is reported."""
    dp, (run, _) = runs
    if dp == 1:
        pytest.skip("a member parks at DP width above one")
    record = read(run["out"] / "parity")
    starts, parks = step_starts(run["table"], record), {}
    for rank in record:
        lp = compass_run.LP_FILE.format(f"engine.dp{rank}")
        parks[rank] = json.loads((run["out"] / lp).read_text())["parks"]
        cpu_us = 1e6 * parks[rank]["cpu_s"] / parks[rank]["count"]
        print(f"\ndp{rank}: {parks[rank]['count']} parks, {cpu_us:.1f} us CPU each")
        assert parks[rank]["count"] == len(parks[rank]["at"]) == len(starts)
        sent = sorted(t for times in _sends(run, rank).values() for t in times)
        for parked, left in parks[rank]["at"]:
            assert not [t for t in sent if parked < t < left], (rank, parked, left)
    for k, start in enumerate(starts):
        at = [parks[rank]["at"][k] for rank in record]
        assert {left for _, left in at} == {max(p for p, _ in at)} == {start}, k


@NEEDS_A_RUN
def test_each_rank_looks_up_blocking_by_both_triggers(runs):
    """Each rank looks a request up with no prefetch to consume, and stops
    waiting for a deferred prefetch at least once."""
    dp, (run, _) = runs
    for rank in read(run["out"] / "parity"):
        lp = compass_run.LP_FILE.format("engine" if dp == 1 else f"engine.dp{rank}")
        n = json.loads((run["out"] / lp).read_text())["lookups"]
        print(f"\ndp{rank}: {n}")
        assert n["lookup_n"] > n["lookup_prefetched"], (rank, n)
        assert n["lookup_defer_expired"] > 0, (rank, n)


# --- (c): the parity record's offload decisions, without a driver -------------


class _Dispatched:
    """The base the runners below dispatch offload metadata to."""

    def process_kvconnector_output(self, meta):
        self.dispatched = meta


def test_a_step_line_carries_the_offload_decisions_dispatched_before_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(RECORD, str(tmp_path))
    monkeypatch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
    config = _parity_config()
    scheduler = Scheduler(config)
    runner = type("Sim", (_SimRunner, _Dispatched), {})(config)
    seq = Sequence(list(range(5, 25)), 16)  # one 20-token prompt, one step
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    meta = LMCacheOffloadMetadata()
    meta.add_request(
        LMCacheReqMeta(seq.id, list(range(20)), [0, 1], LoadSpec(0, 16), SaveSpec(16))
    )
    runner.process_kvconnector_output(meta)
    assert runner.dispatched is meta
    runner.forward(batch)
    runner.forward(batch)  # nothing dispatched before it
    key = _key(seq.prompt_token_ids)
    steps = read(tmp_path)[0]
    assert steps[0]["kv"] == [[key, "load", 0, 16], [key, "save", 16, 20]]
    assert "kv" not in steps[1]


def test_a_real_runner_keeps_what_it_dispatches_once_recording(monkeypatch):
    kept = []
    monkeypatch.setattr(parity, "record_kv", lambda runner, meta: kept.append(meta))
    runner = object.__new__(type("Real", (StepRecording, _Dispatched), {}))
    runner.process_kvconnector_output("warmup")
    runner._recording = True
    runner.process_kvconnector_output("meta")
    assert kept == ["meta"] and runner.dispatched == "meta"
