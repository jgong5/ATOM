# SPDX-License-Identifier: MIT
"""The bench's window, measured on a planted device.

`BenchWindow` and `StepRecording` are composed over a base that calls what
`ModelRunner.forward` calls, in its order: input preparation queues device
work, then `run_model` queues the model's. The device is one stream in host
time: a piece of work starts once it is queued and once the work before it has
ended, an event completes when the stream reaches it, and a poll that finds an
event pending returns once it is not. The host takes a planted time to issue
each forward, half before `run_model` and half inside it. Two ranks run as two
threads over a gloo group.
"""

import itertools
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import pytest

import atom.utils.distributed.utils as dist_utils
from atom.compass.bench import window
from atom.compass.bench.window import BenchWindow
from atom.compass.parity import StepRecording
from tests.compass.held_ports import held_ports

MS = 1_000_000
PREPARE, BUSY = 200_000, 5 * MS  # ns of device work, input preparation and model


class _Event:
    def __init__(self, device, t):
        self.device, self.t = device, t

    def query(self):
        done = self.t <= self.device.host
        self.device.host = max(self.device.host, self.t)
        return done

    def elapsed_time(self, other):
        assert self.t <= self.device.host and other.t <= self.device.host
        return (other.t - self.t) / 1e6  # ms, as torch reports it


class Planted:
    """`ModelRunner.forward`'s device work, issued in `issue` ns of host time
    per forward; the model's work lasts `busy` ns; with `sync`, the forward
    waits for the device before it returns. A forward appends to its batch's
    `rows`, as a real one may rewrite batch fields in place."""

    def __init__(self, issue, busy, sync=False):
        self.issue, self.busy, self.sync = issue, busy, sync
        self.host = self.free = 0  # the host clock, and when queued work ends
        self.forwards, self.preloads, self.seen = 0, [], []  # seen: ids, rows

    def _queue(self, ns):
        self.free = max(self.free, self.host) + ns

    def _gate_staging_reuse(self):
        pass

    def forward(self, batch):
        self.forwards += 1
        self.seen.append((batch.req_ids, len(batch.rows)))
        batch.rows.append(self.forwards)
        self.issuing = next(self.issue)
        self.host += self.issuing // 2
        self._queue(PREPARE)
        self.run_model(batch)
        if self.sync:
            self.host = max(self.host, self.free)
        return batch

    def run_model(self, batch):
        self.host += self.issuing - self.issuing // 2
        self._queue(next(self.busy))

    def step_clock_ns(self):
        return self.host

    def device_event(self):
        self._queue(0)
        return _Event(self, self.free)

    def preload(self, ns):
        self.preloads.append(ns)
        self._queue(ns)

    def bench_group(self):
        return None

    def exit(self):
        pass


class Bench(BenchWindow, StepRecording, Planted):
    pass


def batch():
    return SimpleNamespace(req_ids=[7, 8], rows=[])


def test_the_preload_doubles_until_it_outlasts_the_host_and_each_kept_repeat_is_planted():
    """The named result: the host takes 1 ms to issue each warm-up and 3 ms to
    issue each repeat, so the first preload is 1 ms and doubles to 4 ms."""
    issue = itertools.chain([MS] * window.WARMUPS, itertools.repeat(3 * MS))
    busy = [BUSY + 1_000 * n * n for n in range(window.WARMUPS + 2 + window.REPEATS)]
    bench = Bench(issue, iter(busy))

    handed = batch()
    result = bench.bench_step(handed)

    assert bench.preloads == [MS, 2 * MS] + [4 * MS] * window.REPEATS
    # Every forward ran the batch as handed in, its rows as new requests.
    ids = [i for req_ids, _ in bench.seen for i in req_ids]
    assert len(set(ids)) == len(ids) == 2 * bench.forwards and 7 not in ids
    assert [rows for _, rows in bench.seen] == [0] * bench.forwards
    assert vars(handed) == vars(batch())
    assert bench.forwards == window.WARMUPS + 2 + window.REPEATS
    kept = busy[window.WARMUPS + 2 :]
    assert result["repeats"] == [[PREPARE + b, PREPARE, b] for b in kept]
    assert result["median_ns"] == statistics.median(PREPARE + b for b in kept)
    assert result["discarded"] == 2 and result["preload_ns"] == 4 * MS
    assert result["protocol"] == window.PROTOCOL
    assert bench._parity_record is None


def test_a_forward_that_waits_for_the_device_never_counts_and_its_point_refuses():
    bench = Bench(itertools.repeat(MS), itertools.repeat(BUSY), sync=True)

    result = bench.bench_step(batch())

    # The warm-ups' host waited for the device, so the first preload is that long.
    first = MS + BUSY
    assert bench.preloads == [first * 2**n for n in range(window.DOUBLINGS + 1)]
    assert set(result) == {"protocol", "refused"}
    assert "host synchronisation" in result["refused"]


def test_a_host_that_stalls_every_other_forward_is_measured_between_the_stalls():
    """Every other repeat's host stalls for 10 s, longer than any preload: each
    stall is discarded and none refuses, since no two miss in a row, and the
    preload stops doubling after `DOUBLINGS` doublings."""
    repeats = itertools.cycle([MS // 2, 10_000 * MS])
    bench = Bench(
        itertools.chain([MS] * window.WARMUPS, repeats), itertools.repeat(BUSY)
    )

    result = bench.bench_step(batch())

    assert result["discarded"] == window.REPEATS - 1
    assert result["preload_ns"] == MS * 2**window.DOUBLINGS
    assert result["repeats"] == [[PREPARE + BUSY, PREPARE, BUSY]] * window.REPEATS


class Linked(Planted):
    """Rank `rank` of two: its model's work starts once both ranks have queued
    theirs, as at a forward's first collective, and its spin clock runs `spin`
    times as long as a preload asks."""

    def __init__(self, issue, busy, *, rank, group, spin, link):
        super().__init__(issue, busy)
        self.rank, self.group, self.spin, self.link = rank, group, spin, link

    def bench_group(self):
        return self.group

    def preload(self, ns):
        self.preloads.append(ns)
        self._queue(round(ns * self.spin))

    def run_model(self, batch):
        self.host += self.issuing - self.issuing // 2
        self.link.starts[self.rank] = max(self.free, self.host)
        self.link.barrier.wait()
        begin = max(self.link.starts)
        self.link.barrier.wait()
        self.free = begin + next(self.busy)


class Rank(BenchWindow, StepRecording, Linked):
    pass


@pytest.fixture
def pair(monkeypatch):
    """`pair(step)`: `step(rank, group)` on each rank of a two-rank gloo group."""
    monkeypatch.setattr(
        dist_utils, "_get_default_timeout", lambda _: timedelta(seconds=5)
    )
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")

    def on_rank(rank, step):
        group = dist_utils.stateless_init_torch_distributed_process_group(
            "127.0.0.1", port, rank, 2, backend="gloo"
        )
        try:
            return step(rank, group)
        finally:
            dist_utils.stateless_destroy_torch_distributed_process_group(group)

    def run(step):
        with ThreadPoolExecutor(2) as pool:
            return [f.result() for f in [pool.submit(on_rank, r, step) for r in (0, 1)]]

    with held_ports(1) as [port]:  # until rank 0's store binds it
        yield run


@pytest.mark.parametrize("spins", [(1.01, 1.04), (1.04, 1.01)])
def test_above_one_rank_each_kept_repeat_is_the_last_opened_ranks_window(pair, spins):
    """Rank r's host takes (2 + r) / 2 ms to issue each forward, so the preload
    is the slower host's 1.5 ms on both. The rank whose clock spins less opens
    first and waits for the other at the collective, 45 us longer than the
    step; each kept repeat is the other's window, the planted step."""
    link = SimpleNamespace(barrier=threading.Barrier(2, timeout=10), starts=[0, 0])
    busy = [BUSY + 1_000 * n * n for n in range(window.WARMUPS + window.REPEATS)]

    def step(rank, group):
        issue = itertools.repeat(MS + rank * MS // 2)
        spin = spins[rank]
        bench = Rank(issue, iter(busy), rank=rank, group=group, spin=spin, link=link)
        return bench.bench_step(batch()), bench.preloads

    (result, preloads), other = pair(step)

    assert other == (result, preloads)
    assert preloads == [3 * MS // 2] * window.REPEATS
    kept = busy[window.WARMUPS :]
    assert result["repeats"] == [[PREPARE + b, PREPARE, b] for b in kept]
    assert result["median_ns"] == statistics.median(PREPARE + b for b in kept)
    assert result["discarded"] == 0
