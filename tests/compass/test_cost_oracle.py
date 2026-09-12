"""Cost oracle interface and the constant oracle.

These exercise atom.compass.core, which imports nothing from ATOM and needs no
GPU — the point of keeping that boundary strict.
"""

import pytest

from atom.compass.core.cost.base import CostOracle, StepCost, StepShape
from atom.compass.core.cost.constant import ConstantCostOracle


def _decode_shape(context_lens=(10, 2000)):
    return StepShape(
        num_scheduled_tokens=tuple(1 for _ in context_lens),
        context_lens=tuple(context_lens),
        num_prefill_tokens=0,
    )


def test_shape_derives_batch_and_token_counts():
    shape = StepShape(
        num_scheduled_tokens=(512, 1, 1),
        context_lens=(0, 100, 5000),
        num_prefill_tokens=512,
    )
    assert shape.batch_size == 3
    assert shape.total_tokens == 514
    assert shape.num_decode_tokens == 2
    assert shape.is_prefill


def test_shape_keeps_per_request_lengths_unreduced():
    # The whole point: a skewed batch must remain distinguishable from a
    # uniform one with the same mean, because it does not cost the same.
    skewed = _decode_shape((10, 2000))
    uniform = _decode_shape((1005, 1005))
    assert skewed.total_tokens == uniform.total_tokens
    assert skewed != uniform


def test_constant_oracle_separates_prefill_from_decode():
    oracle = ConstantCostOracle(prefill_seconds=0.02, decode_seconds=0.002)
    prefill = StepShape(num_scheduled_tokens=(256,), context_lens=(0,), num_prefill_tokens=256)
    assert oracle.estimate(prefill).seconds == pytest.approx(0.02)
    assert oracle.estimate(_decode_shape()).seconds == pytest.approx(0.002)


def test_constant_oracle_attributes_its_cost():
    cost = ConstantCostOracle().estimate(_decode_shape())
    assert cost.breakdown == {"decode": cost.seconds}


def test_constant_oracle_ignores_batch_content_by_design():
    oracle = ConstantCostOracle()
    assert oracle.estimate(_decode_shape((1,))).seconds == pytest.approx(
        oracle.estimate(_decode_shape((1, 2, 3, 100_000))).seconds
    )


def test_constant_oracle_satisfies_the_protocol():
    assert isinstance(ConstantCostOracle(), CostOracle)


def test_negative_costs_are_rejected():
    with pytest.raises(ValueError):
        StepCost(seconds=-1.0)
    with pytest.raises(ValueError):
        ConstantCostOracle(decode_seconds=-1.0)


def test_describe_is_informative():
    assert "ConstantCostOracle" in ConstantCostOracle().describe()
