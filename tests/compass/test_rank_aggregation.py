# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Pricing a TP group's step from one physical executor.

A GPU-free replay runs one process and calls itself rank 0, so `rank_coords`
arrived at the oracle as `{"tp": 0}` on every step of every width. Whatever the
oracle then knew about the other ranks was never asked for, and a deployment
whose step ends when its slowest rank ends was priced from its first one. The
TP4 head measurements are not interchangeable -- 13.510 / 16.158 / 13.459 /
13.452 ms -- so this is a real number, not a tidiness point.

`slowest` prices every logical rank and keeps the maximum. That is the group's
step exactly when one rank is slowest in every phase. When the bottleneck
alternates it is *not* an upper bound -- max_r sum_p t[r,p] <= sum_p max_r
t[r,p], so a maximum of whole-rank totals sits at or below the serial-phase
reading of the same prices. `test_alternating_phases_are_under_not_over` is the
counterexample. Against measured engine time neither reading is a bound in
either direction, because the component prices carry their own error. The row
says which reading it is.
"""

import types

import pytest

from atom.compass.config import CompassConfig
from atom.compass.core.cost.base import StepCost, StepShape

# The documented TP4 head spread, in seconds.
HEAD_MS = {0: 13.510, 1: 16.158, 2: 13.459, 3: 13.452}


class _PerRankOracle:
    """Prices a step by which rank is asking. Refuses a rank it has no price for."""

    def __init__(self, seconds_by_rank, refuse=()):
        self.seconds_by_rank = seconds_by_rank
        self.refuse = set(refuse)
        self.asked = []

    def estimate(self, shape: StepShape) -> StepCost:
        rank = int((shape.rank_coords or {}).get("tp", 0))
        self.asked.append(rank)
        if rank in self.refuse:
            raise LookupError(f"no artifact for rank {rank}")
        return StepCost(seconds=self.seconds_by_rank[rank])

    def describe(self) -> str:
        return "per-rank stub"


def _runner(policy, oracle):
    # The mixin, not `CompassModelRunner`: the runner drags in `aiter`, which
    # resolves the chip by shelling out to `rocminfo` at import. Nothing below
    # this line is device-dependent, and the mixin is where the code under test
    # lives.
    from atom.compass.runtime.predict import CompassPredictMixin

    stub = CompassPredictMixin.__new__(CompassPredictMixin)
    stub.__dict__["_compass_config_cache"] = CompassConfig(
        enabled=True, mode="predict", rank_aggregation=policy)
    stub.config = types.SimpleNamespace()
    stub._oracle = oracle
    return stub


def _shape(tp):
    return StepShape(num_scheduled_tokens=(1, 1), context_lens=(128, 130),
                     topology={"tp": tp}, rank_coords={"tp": 0})


def test_rank0_prices_one_rank_and_says_so():
    oracle = _PerRankOracle({r: ms / 1000 for r, ms in HEAD_MS.items()})
    cost, ranks = _runner("rank0", oracle)._estimate_over_ranks(_shape(4))
    assert cost.seconds == pytest.approx(0.013510)
    assert oracle.asked == [0]
    assert ranks["policy"] == "rank0"
    assert ranks["priced_ranks"] == [0]
    # The label is the point: at TP4 this number is a rank's, not a step's.
    assert "not the group's step" in ranks["exactness"]


def test_slowest_prices_every_rank_and_keeps_the_outlier():
    oracle = _PerRankOracle({r: ms / 1000 for r, ms in HEAD_MS.items()})
    cost, ranks = _runner("slowest", oracle)._estimate_over_ranks(_shape(4))
    assert oracle.asked == [0, 1, 2, 3]
    assert cost.seconds == pytest.approx(0.016158)
    assert ranks["slowest_rank"] == 1
    assert ranks["seconds_by_rank"] == {
        "0": pytest.approx(0.013510), "1": pytest.approx(0.016158),
        "2": pytest.approx(0.013459), "3": pytest.approx(0.013452)}
    # Not averaged away: the spread survives into the record.
    assert ranks["spread_seconds"] == pytest.approx(0.016158 - 0.013452)
    assert ranks["exactness"].startswith("approximation")


def test_the_slow_rank_is_not_averaged_away():
    # The mean of the four is 14.145 ms. A step priced at the mean is a step no
    # rank ever ran, and it is 12.5% under the one that decides when the step
    # ends.
    oracle = _PerRankOracle({r: ms / 1000 for r, ms in HEAD_MS.items()})
    cost, _ = _runner("slowest", oracle)._estimate_over_ranks(_shape(4))
    mean = sum(HEAD_MS.values()) / 4 / 1000
    assert cost.seconds > mean


def test_alternating_phases_are_under_not_over():
    # Two ranks, two collective-delimited phases, in milliseconds:
    #
    #             attn   mlp   whole-rank total
    #   rank 0    10.0   1.0   11.0
    #   rank 1     2.0   8.0   10.0
    #
    # A step that synchronises between the phases takes max(10, 2) +
    # max(1, 8) = 18.0 ms: rank 0 holds everyone up in the first phase, rank 1
    # in the second. `slowest` answers 11.0 -- the largest whole-rank total --
    # which is 39% *under* that, not over it. The arithmetic is general:
    # max_r sum_p t[r,p] <= sum_p max_r t[r,p] for any table, with equality
    # only when one rank attains the maximum in every phase.
    phases = {0: (0.010, 0.001), 1: (0.002, 0.008)}
    oracle = _PerRankOracle({r: sum(p) for r, p in phases.items()})
    cost, ranks = _runner("slowest", oracle)._estimate_over_ranks(_shape(2))

    serial_phase = sum(max(phases[r][p] for r in phases) for p in (0, 1))
    assert serial_phase == pytest.approx(0.018)
    assert cost.seconds == pytest.approx(0.011)
    assert cost.seconds < serial_phase

    # And the record has to say so in that direction. A reader who took
    # "approximation" for "conservative" would size a deployment from a number
    # that is short by seven milliseconds a step.
    assert "upper" not in ranks["exactness"]
    assert "at or below" in ranks["bound_direction"]
    assert "not a bound on measured engine time" in ranks["bound_direction"]


def test_one_rank_slowest_in_every_phase_is_the_exact_case():
    # Same shape of table, but rank 0 leads both phases, so the whole-rank
    # maximum and the serial-phase reading agree. This is the symmetry-like
    # condition under which `slowest` is not an approximation at all.
    phases = {0: (0.010, 0.008), 1: (0.002, 0.001)}
    oracle = _PerRankOracle({r: sum(p) for r, p in phases.items()})
    cost, _ = _runner("slowest", oracle)._estimate_over_ranks(_shape(2))
    serial_phase = sum(max(phases[r][p] for r in phases) for p in (0, 1))
    assert cost.seconds == pytest.approx(serial_phase)


def test_tp1_has_no_rank_question_to_answer():
    oracle = _PerRankOracle({0: 0.004})
    cost, ranks = _runner("slowest", oracle)._estimate_over_ranks(_shape(1))
    assert cost.seconds == pytest.approx(0.004)
    assert ranks is None
    assert oracle.asked == [0]


def test_a_rank_the_oracle_cannot_price_refuses_rather_than_falls_back():
    # Falling back to rank 0's price is the silent substitution this policy
    # exists to stop: the group would be priced from the rank that happened to
    # have an artifact.
    oracle = _PerRankOracle({r: ms / 1000 for r, ms in HEAD_MS.items()},
                            refuse=(2,))
    with pytest.raises(LookupError, match="rank 2"):
        _runner("slowest", oracle)._estimate_over_ranks(_shape(4))


def test_the_policy_has_to_be_one_of_the_two():
    with pytest.raises(ValueError, match="rank_aggregation"):
        CompassConfig(enabled=True, mode="predict", rank_aggregation="mean")


def test_the_default_is_the_old_behaviour():
    # Every frozen diagnostic was produced under rank0; changing the default
    # would relabel them by moving the code under them.
    assert CompassConfig().rank_aggregation == "rank0"


def test_the_recorded_row_carries_the_aggregation(tmp_path):
    import json

    oracle = _PerRankOracle({r: ms / 1000 for r, ms in HEAD_MS.items()})
    runner = _runner("slowest", oracle)
    out = tmp_path / "steps.jsonl"
    runner._compass_config.measure_out = str(out)
    runner._measure_fh = None
    runner._topology = lambda: {"tp": 4}
    runner._rank_coords = lambda: {"tp": 0}
    shape = _shape(4)
    cost, ranks = runner._estimate_over_ranks(shape)
    runner._record_measurement(shape, cost.seconds, None, ranks=ranks)
    runner._measure_fh.close()
    # The file is still named for the executor's own coordinates -- one process
    # wrote it -- while the row inside it is the group's. Both facts are
    # recorded rather than reconciled by renaming.
    written = tmp_path / "steps.tp0.jsonl"
    row = json.loads(written.read_text().splitlines()[0])
    assert row["seconds"] == pytest.approx(0.016158)
    assert row["rank_aggregation"]["slowest_rank"] == 1
    assert row["rank_aggregation"]["policy"] == "slowest"
