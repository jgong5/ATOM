# SPDX-License-Identifier: MIT
"""A simulated run's node-local DP group runs its small collectives over shared memory.

Each rank is a thread on its own gloo group, built as ATOM builds the engine
core's. An attached group returns what gloo returns, bit for bit; a call the
module leaves out, and a group a rank cannot join, stay on gloo; a rank whose
peer has exited raises, which ATOM's DP state sync reads as a shutdown.
"""

import contextlib
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
from test_dp_slice import _deployment
from test_vertical_slice import _run_file

import atom.utils.distributed.utils as dist_utils
from atom.compass import run as compass_run
from atom.compass.runner import shm_collectives
from atom.config import ParallelConfig
from atom.model_engine.engine_core import DPEngineCoreProc
from atom.utils import clock
from tests.compass.held_ports import held_ports

#: op, dtype, width, whether NaNs and zeros of either sign come up: ATOM's DP
#: state sync and prefill delayer; Compass's forward mode, dummy ranks, device
#: end and step targets; then a MIN, a wrapping sum, and a float MAX.
CASES = (
    ("max", torch.int64, 3, False),
    ("sum", torch.int64, 21, False),
    ("gather", torch.int32, 9, False),
    ("sum", torch.int32, 1, False),
    ("max", torch.float64, 2, False),
    ("gather", torch.float64, 3, True),
    ("min", torch.int32, 5, False),
    ("sum", torch.int8, 6, False),
    ("max", torch.float32, 7, True),
)
CALLS = 40
OPS = {"max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN, "sum": dist.ReduceOp.SUM}
local = threading.local()


@pytest.fixture
def on_ranks(monkeypatch):
    """`run(world, fn)`: `fn(rank, port)` on a thread per rank, their results by rank."""
    monkeypatch.setattr(
        dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=10)
    )
    monkeypatch.setattr(dist_utils, "LP_OF_RANK", None)
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    monkeypatch.setattr(dist, "all_reduce", dist.all_reduce)
    monkeypatch.setattr(dist, "all_gather", dist.all_gather)
    monkeypatch.setattr(shm_collectives, "_GROUPS", {})

    def run(world, fn):
        def on(rank):
            local.rank = rank
            return fn(rank, port)

        with held_ports(1) as [port], ThreadPoolExecutor(world) as pool:
            return list(pool.map(on, range(world)))

    return run


@contextlib.contextmanager
def gloo_group(rank, world, port):
    group = dist_utils.stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world, backend="gloo"
    )
    try:
        yield group
    finally:
        dist_utils.stateless_destroy_torch_distributed_process_group(group)


def values(rng, dtype, width, special=False):
    if dtype.is_floating_point:
        v = rng.normal(0, 1e3, width)
        if special:
            v[rng.random(width) < 0.2] = rng.choice([np.nan, 0.0, -0.0])
        return torch.tensor(v, dtype=dtype)
    info = torch.iinfo(dtype)
    return torch.from_numpy(rng.integers(info.min, info.max, width, endpoint=True)).to(
        dtype
    )


def call(fns, op, t, group, async_op=False):
    """One collective through `fns`; its result as bytes."""
    reduce, gather = fns
    if op == "gather":
        rows = [torch.empty_like(t) for _ in range(group.size())]
        work = gather(rows, t, group=group, async_op=async_op)
        if work is not None:
            work.wait()
        return b"".join(r.numpy().tobytes() for r in rows)
    work = reduce(t, OPS[op], group, async_op)
    if work is not None:
        work.wait()
    return t.numpy().tobytes()


def test_an_attached_group_returns_what_gloo_returns_bit_for_bit(on_ranks):
    gloo = dist.all_reduce, dist.all_gather
    shm_collectives.install()
    routed = dist.all_reduce, dist.all_gather

    def rank(r, port):
        rng = np.random.default_rng(r)
        with gloo_group(r, 4, port) as group:
            assert shm_collectives.attach(group)
            differ = 0
            for op, dtype, width, special in CASES:
                for _ in range(CALLS):
                    v = values(rng, dtype, width, special)
                    a = call(routed, op, v.clone(), group)
                    differ += a != call(gloo, op, v.clone(), group)
            return differ, shm_collectives._GROUPS[group].round

    assert on_ranks(4, rank) == [(0, len(CASES) * CALLS)] * 4


@pytest.mark.parametrize(
    "op, dtype, width, async_op",
    [("sum", torch.float64, 4, False), ("max", torch.int64, 513, False)]
    + [("max", torch.int64, 3, True), ("gather", torch.int32, 9, True)],
    ids=["float SUM", "over a row", "async all_reduce", "async all_gather"],
)
def test_a_call_the_module_leaves_out_stays_on_gloo(
    on_ranks, op, dtype, width, async_op
):
    gloo = dist.all_reduce, dist.all_gather
    shm_collectives.install()

    def rank(r, port):
        v = values(np.random.default_rng(r), dtype, width)
        with gloo_group(r, 2, port) as group:
            shm_collectives.attach(group)
            a = call((dist.all_reduce, dist.all_gather), op, v.clone(), group, async_op)
            return (
                a == call(gloo, op, v.clone(), group),
                shm_collectives._GROUPS[group].round,
            )

    assert on_ranks(2, rank) == [(True, 0)] * 2


@pytest.mark.parametrize(
    "cannot", ["map the segment", "share the PID namespace", "run on x86"]
)
def test_a_group_one_rank_cannot_join_stays_on_gloo_on_every_rank(
    on_ranks, monkeypatch, cannot
):
    segment, here = shm_collectives._segment, shm_collectives._here

    def opened(name, size=0):
        refused = cannot == "map the segment" and local.rank == 1
        return None if refused else segment(name, size)

    def where():
        x86, ns, pid = here()
        if local.rank == 1:
            ns += cannot == "share the PID namespace"
            x86 = x86 and cannot != "run on x86"
        return [x86, ns, pid]

    monkeypatch.setattr(shm_collectives, "_segment", opened)
    monkeypatch.setattr(shm_collectives, "_here", where)
    shm_collectives.install()

    def rank(r, port):
        t = torch.tensor([r, -r])
        with gloo_group(r, 2, port) as group:
            attached = shm_collectives.attach(group)
            dist.all_reduce(t, dist.ReduceOp.MAX, group)
            return attached, t.tolist()

    assert on_ranks(2, rank) == [(False, [1, 0])] * 2
    assert shm_collectives._GROUPS == {}


def _state_sync(rank, group):
    """ATOM's DP state sync on `group`, of an engine core with nothing else."""
    core = object.__new__(DPEngineCoreProc)
    core.dp_group, core.label, core._shutting_down = group, f"dp{rank}", False
    return core, core._sync_dp_state(rank == 1, False, rank == 0)


def test_a_peer_that_exits_makes_atoms_state_sync_shut_down(on_ranks):
    shm_collectives.install()
    reaped = subprocess.Popen(["true"])
    reaped.wait()

    def rank(r, port):
        with gloo_group(r, 2, port) as group:
            assert shm_collectives.attach(group)
            if r == 1:
                return None
            shm_collectives._GROUPS[group].pids[1] = reaped.pid
            t0 = time.monotonic()
            core, synced = _state_sync(r, group)
            return synced, core._shutting_down, time.monotonic() - t0

    (synced, shutting_down, waited), _ = on_ranks(2, rank)
    assert synced == (False, True, True) and shutting_down
    assert 1.0 < waited < 5.0


def test_a_simulated_dp_engine_core_syncs_its_state_over_shared_memory(
    on_ranks, monkeypatch, tmp_path
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, data_parallel_size=2)))
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(compass_run, "_member", None)
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    build = ParallelConfig.stateless_init_dp_group

    def rank(r, port):
        config = ParallelConfig(
            data_parallel_size=2,
            data_parallel_rank=r,
            data_parallel_size_local=1,
            data_parallel_master_port=port,
        )
        group = config.stateless_init_dp_group()  # as the engine core builds it
        try:
            _, synced = _state_sync(r, group)
            return synced, shm_collectives._GROUPS[group].round
        finally:
            dist_utils.stateless_destroy_torch_distributed_process_group(group)

    with compass_run.engine(_deployment(dp=2, rank=0)):
        assert on_ranks(2, rank) == [((True, False, True), 1)] * 2
    assert ParallelConfig.stateless_init_dp_group is build
