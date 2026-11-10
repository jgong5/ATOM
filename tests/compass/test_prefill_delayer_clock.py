# SPDX-License-Identifier: MIT
"""The prefill delayer's cost gate times steps and arrivals on the run's clock.

`EngineCore` builds the delayer with no clock of its own, so the gate reads
`time.perf_counter` through `atom.utils.time_shim` at each decision: the LP
clock while a runtime is installed, the machine's otherwise. What a rank puts
into the group's reduce is what every rank's gate decides on.
"""

import time
from types import SimpleNamespace
from unittest import mock

import pytest

from atom.model_engine.engine_core import EngineCore
from atom.model_engine.prefill_delayer import _Gate
from atom.utils import clock

BUDGET = 16384
#: A decode step, a prefill step and the cost of the chunk it runs: binary
#: fractions of a second, so the LP clock's differences are exact.
DECODE_S, PREFILL_S, COST = 1 / 64, 1 / 32, 2.5e8
#: The other rank of the group: one decode, a full budget of prefill waiting.
SIBLING = {0: 1, 1: BUDGET, 2: 1}


class _Rank0:
    def rank(self):
        return 0


class _Runtime:
    def __init__(self, t):
        self.t = t

    def read_clock(self):
        return self.t


@pytest.fixture
def lp_clock(monkeypatch):
    runtime = _Runtime(100.0)
    monkeypatch.setattr(clock, "_installed", runtime)
    return runtime


@pytest.fixture
def delayer(monkeypatch):
    """Rank 0's delayer in a DP group of two, as `EngineCore` builds it."""
    monkeypatch.setenv("ATOM_ENABLE_PREFILL_DELAYER", "1")
    monkeypatch.setenv("ATOM_PREFILL_DELAYER_COST_GATE", "1")
    built = []
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(cost_balance=True, set_prefill_delayer=built.append)
    )
    config = SimpleNamespace(
        enable_rapidserve=False,
        max_num_batched_tokens=BUDGET,
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )
    EngineCore._init_prefill_delayer(engine, config, _Rank0())
    return built[0]


def _decide(delayer, reports, prefillable=False, pending_tokens=0, **local):
    """One decision of rank 0, with one decode running; `reports` gains its
    gate slots as it sent them."""

    def all_reduce(tensor, op=None, group=None):
        reports.append(_Gate(*tensor.tolist()[-len(_Gate._fields) :]))
        for slot, value in SIBLING.items():
            tensor[slot] += value

    with mock.patch("torch.distributed.all_reduce", all_reduce):
        decision = delayer.should_allow_prefill(
            prefillable, pending_tokens, running_decode_batch=1, **local
        )
    return decision.allow


def _timed(delayer, advance):
    """What rank 0 reports after two decode steps and a prefill step, each
    taking the time `advance` lets pass: its decode step in µs and its prefill
    time a cost unit in ps."""
    reports = []
    _decide(delayer, reports)  # the first decision fires and times nothing
    advance(DECODE_S)
    _decide(delayer, reports)
    advance(DECODE_S)
    # both ranks hold a full budget: the chunk fires
    assert _decide(delayer, reports, True, BUDGET, prefill_cost=COST) is True
    delayer.notify_prefill_executed()
    advance(PREFILL_S)
    _decide(delayer, reports)
    advance(DECODE_S)
    _decide(delayer, reports)
    return reports[-1].decode_tick_us, reports[-1].prefill_ps_per_cost


def test_under_a_runtime_a_step_is_timed_in_simulated_seconds(delayer, lp_clock):
    def advance(seconds):
        lp_clock.t += seconds

    assert _timed(delayer, advance) == (15625, 125)


def test_under_a_runtime_arrival_gaps_are_simulated_seconds(delayer, lp_clock):
    """Requests reach rank 0 half a second apart on the LP clock: it reports
    2000 arrivals a 1000 s, then 500 once two seconds pass with none."""
    reports = []
    _decide(delayer, reports, arrived_requests=1)
    lp_clock.t += 0.5
    _decide(delayer, reports, arrived_requests=1)
    lp_clock.t += 2.0
    _decide(delayer, reports)
    assert [r.arrivals_per_ks for r in reports] == [0, 2000, 500]


def test_without_a_runtime_a_step_is_timed_on_the_machine_clock(delayer):
    """Each step sleeps its time: the times reported are at least that, and at
    most what the whole run took on `time.perf_counter`."""
    start = time.perf_counter()
    decode_us, prefill_ps = _timed(delayer, time.sleep)
    took = time.perf_counter() - start
    assert DECODE_S * 1e6 <= decode_us <= took * 1e6
    assert PREFILL_S * 1e12 / COST <= prefill_ps <= took * 1e12 / COST
