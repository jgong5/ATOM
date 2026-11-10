# SPDX-License-Identifier: MIT
"""A DP group leaves a step's exchange when its slowest rank to price arrives:
in an op-graph run the group's first rank keeps that rank's lookup wall and the
rest of its price wall, and the run summary states them for each group beside
the wall outside both.

Two ranks run `forward` as `test_dp_step_max` runs them, each priced by the
backend `run.op_graph` builds and times, its `OpGraphBackend` a stub that sleeps
through a lookup, counted as `_op_graphs` counts a lookup's wall, and then
through the rest of its price.
"""

import json
import time
from types import SimpleNamespace

import torch
from test_dp_step_max import Recording, dp2, prefill, runner  # noqa: F401

from atom.compass import run as compass_run
from atom.compass.backends import op_graph
from atom.compass.runner.overrides import PricedSteps


class Sleeping(Recording):
    """The stub, its price a lookup of `lookup_s` then `pricing_s` besides."""

    def __init__(self, lazy, lookup_s=0.0, pricing_s=0.0):
        super().__init__()
        self.lazy, self.lookup_s, self.pricing_s = lazy, lookup_s, pricing_s

    def price(self, batch_view):
        if self.lookup_s:
            began = time.monotonic()
            time.sleep(self.lookup_s)
            self.lazy["lookup_wall_seconds"] += time.monotonic() - began
        time.sleep(self.pricing_s)
        return super().price(batch_view)


def ranks(monkeypatch, *sleeps):
    """A runner per `(lookup_s, pricing_s)` in `sleeps`, one DP group, each
    priced by `run.op_graph` over a `Sleeping` stub, and the stubs."""
    monkeypatch.setattr(compass_run, "_op_graphs", lambda *a, **k: None)
    config = SimpleNamespace(
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=len(sleeps)),
        kv_cache_dtype="bf16",
        enable_expert_parallel=False,
    )
    machine = SimpleNamespace(value=lambda key: "gfx950")
    runners, stubs = [], []
    for rank, sleep in enumerate(sleeps):
        lazy = {"lookup_wall_seconds": 0.0}
        stubs.append(Sleeping(lazy, *sleep))
        monkeypatch.setattr(op_graph, "OpGraphBackend", lambda *a, **k: stubs[-1])
        backend = compass_run.op_graph({"store": "-"}, config, machine, lazy=lazy)
        runners.append(runner(backend, dp_size=len(sleeps), dp_rank=rank))
        runners[-1].compass_lazy = lazy
    return runners, stubs


def test_the_group_keeps_its_slowest_ranks_lookups_and_pricing_once(
    dp2, monkeypatch  # noqa: F811
):
    # Step 1 waits on rank 1's lookup, step 2 on rank 0's pricing.
    (first, second), stubs = ranks(monkeypatch, (0.0, 0.3), (0.6, 0.0))
    dp2(lambda: first.forward(prefill(8)), lambda: second.forward(prefill(8)))

    lookups, pricing = first._waited
    assert lookups >= 0.6 and pricing < 0.3
    assert not hasattr(second, "_waited")

    stubs[0].pricing_s, stubs[1].lookup_s = 0.9, 0.3
    dp2(lambda: first.forward(prefill(8)), lambda: second.forward(prefill(8)))

    assert first._waited[0] == lookups
    assert first._waited[1] >= pricing + 0.9
    assert "group_waited_seconds" not in PricedSteps.priced_steps(second)
    assert PricedSteps.priced_steps(first)["group_waited_seconds"] == first._waited


def test_a_rank_of_no_group_keeps_its_own(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_gather", None)
    (alone,), _ = ranks(monkeypatch, (0.2, 0.1))

    alone.forward(prefill(8))

    lookups, pricing = alone._waited
    assert lookups >= 0.2 and pricing >= 0.1


def test_a_run_with_no_op_graph_keeps_none(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_gather", None)
    alone = runner(Recording())

    alone.forward(prefill(8))

    assert "group_waited_seconds" not in PricedSteps.priced_steps(alone)


def test_the_run_summary_states_each_groups_waits_and_the_wall_outside_them(
    monkeypatch, tmp_path
):
    run = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    authority.starts, authority.finished = [0.0], 10.0
    group = SimpleNamespace(_waited=[2.5, 1.5])
    priced = PricedSteps.priced_steps(group)
    compass_run._hand_in(run, "engine", [], priced=priced)
    compass_run._hand_in(run, "frontend", [])

    compass_run._write_summary(run, authority, ended=True)

    cost = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())["cost"]
    waited = {"lookups": 2.5, "pricing": 1.5, "rest": 6.0}
    assert cost["group_waited_seconds"] == {"engine": waited}
