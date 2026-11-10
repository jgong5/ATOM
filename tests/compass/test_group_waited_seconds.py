# SPDX-License-Identifier: MIT
"""A DP group leaves a step's exchange when its slowest rank to price arrives:
the group's first rank keeps that rank's lookup wall and the rest of its
pricing wall, and the run summary states them for each group beside the wall
outside both.

Two ranks run `forward` as `test_dp_step_max` runs them, each priced by a stub
that first sleeps through a lookup, counted in the runner's `compass_lazy` as
an op-graph lookup counts its wall, then sleeps through the rest of its
pricing.
"""

import json
import time
from types import SimpleNamespace

import torch
from test_dp_step_max import Recording, dp2, prefill, runner  # noqa: F401

from atom.compass import run as compass_run
from atom.compass.runner.overrides import PricedSteps


class Sleeping(Recording):
    """The stub, its price a lookup of `lookup_s` then `pricing_s` besides."""

    def __init__(self, lookup_s=0.0, pricing_s=0.0):
        super().__init__()
        self.lookup_s, self.pricing_s, self.lazy = lookup_s, pricing_s, None

    def price(self, batch_view):
        if self.lookup_s:
            began = time.monotonic()
            time.sleep(self.lookup_s)
            self.lazy["lookup_wall_seconds"] += time.monotonic() - began
        time.sleep(self.pricing_s)
        return super().price(batch_view)


def ranks(*backends):
    out = [runner(b, dp_size=len(backends), dp_rank=r) for r, b in enumerate(backends)]
    for r, b in zip(out, backends):
        r.compass_lazy = b.lazy = {"lookup_wall_seconds": 0.0}
    return out


def test_the_group_keeps_its_slowest_ranks_lookups_and_pricing_once(dp2):  # noqa: F811
    # Step 1 waits on rank 1's lookup, step 2 on rank 0's pricing.
    backends = Sleeping(pricing_s=0.3), Sleeping(lookup_s=0.6)
    first, second = ranks(*backends)
    dp2(lambda: first.forward(prefill(8)), lambda: second.forward(prefill(8)))

    lookups, pricing = first._waited
    assert lookups >= 0.6 and pricing < 0.3
    assert not hasattr(second, "_waited")

    backends[0].pricing_s, backends[1].lookup_s = 0.9, 0.3
    dp2(lambda: first.forward(prefill(8)), lambda: second.forward(prefill(8)))

    assert first._waited[0] == lookups
    assert first._waited[1] >= pricing + 0.9
    assert "group_waited_seconds" not in PricedSteps.priced_steps(second)
    assert PricedSteps.priced_steps(first)["group_waited_seconds"] == first._waited


def test_a_rank_of_no_group_keeps_its_own(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_gather", None)
    (alone,) = ranks(Sleeping(lookup_s=0.2, pricing_s=0.1))

    alone.forward(prefill(8))

    lookups, pricing = alone._waited
    assert lookups >= 0.2 and pricing >= 0.1


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
