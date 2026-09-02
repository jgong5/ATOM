"""Costing a step from its op graph rather than from its shape.

The sum of priced kernels is about three quarters of a step and no coverage
closes the rest. The missing quarter is a fixed cost per kernel launch, not a
factor: against the same kernels inside a real step the ratio runs 0.43 to 0.99,
which no single multiplier corrects, while as an additive term it is ~2us
whatever the kernel. These pin that arithmetic down, and pin down what the
oracle does when asked about a step its graph does not describe -- which is the
failure that matters, because answering a prefill step with a decode cost is not
an approximation but a different question.
"""

import json

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.priced import PricedGraphCostOracle
from atom.compass.runtime.microbench import signature_of


def _op(name, kernels=("k",), context=()):
    return {"name": name, "input_shapes": [], "output_shapes": [], "dtypes": [],
            "group": None, "scalars": [], "int_values": [],
            "context": [list(kv) for kv in context]}


def _artifacts(tmp_path, ops, seconds=1e-5, kernels=("k",)):
    graph = {"version": 2, "key": None, "provenance": {}, "ops": ops}
    prices = {"prices": {}}
    for op in ops:
        prices["prices"][signature_of(op)] = {
            "name": op["name"], "seconds": seconds, "occurrences": 1,
            "kernels": {k: seconds / len(kernels) for k in kernels},
        }
    graph_path, prices_path = tmp_path / "g.json", tmp_path / "p.json"
    graph_path.write_text(json.dumps(graph))
    prices_path.write_text(json.dumps(prices))
    return str(prices_path), str(graph_path)


DECODE = StepShape(num_scheduled_tokens=(1, 1), context_lens=(8, 8))
PREFILL = StepShape(num_scheduled_tokens=(8,), context_lens=(0,),
                    num_prefill_tokens=8)


class TestTheArithmetic:
    def test_a_step_is_its_operators_plus_its_launches(self, tmp_path):
        prices, graph = _artifacts(tmp_path, [_op("a"), _op("b")], seconds=1e-5)
        oracle = PricedGraphCostOracle(prices, graph, boundary_seconds=2e-6)
        # Two operators at 10us, one kernel each, 2us a launch.
        assert oracle.estimate(DECODE).seconds == pytest.approx(2 * 1e-5 + 2 * 2e-6)

    def test_an_operator_is_not_one_kernel(self, tmp_path):
        """Attention launches three, and a boundary is paid at each."""
        prices, graph = _artifacts(tmp_path, [_op("attn")], seconds=1e-5,
                                   kernels=("pa", "rope", "cache"))
        oracle = PricedGraphCostOracle(prices, graph, boundary_seconds=2e-6)
        assert oracle.launches == 3
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-5 + 3 * 2e-6)

    def test_zero_boundary_reproduces_the_naive_sum(self, tmp_path):
        """Which is the 26%-low number, kept reachable so it can be compared."""
        prices, graph = _artifacts(tmp_path, [_op("a")], seconds=1e-5)
        oracle = PricedGraphCostOracle(prices, graph, boundary_seconds=0.0)
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-5)

    def test_an_unpriced_operator_contributes_nothing_and_is_counted(self, tmp_path):
        prices, graph = _artifacts(tmp_path, [_op("a")], seconds=1e-5)
        blob = json.loads(open(graph).read())
        blob["ops"].append(_op("never_priced"))
        open(graph, "w").write(json.dumps(blob))
        oracle = PricedGraphCostOracle(prices, graph, boundary_seconds=0.0)
        assert oracle.unpriced == 1
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-5)

    def test_the_floor_holds(self, tmp_path):
        """An empty graph must not run a virtual clock backwards or stop it."""
        prices, graph = _artifacts(tmp_path, [])
        oracle = PricedGraphCostOracle(prices, graph, floor_seconds=1e-6)
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-6)


class TestShapesTheGraphDoesNotDescribe:
    def test_a_prefill_step_against_a_decode_graph_warns(self, tmp_path, caplog):
        prices, graph = _artifacts(tmp_path, [_op("a")])
        oracle = PricedGraphCostOracle(prices, graph)
        with caplog.at_level("WARNING"):
            oracle.estimate(PREFILL)
        assert "prefill" in caplog.text
        assert "ATOMCompass WARNING" in caplog.text

    def test_and_warns_once(self, tmp_path, caplog):
        prices, graph = _artifacts(tmp_path, [_op("a")])
        oracle = PricedGraphCostOracle(prices, graph)
        with caplog.at_level("WARNING"):
            for _ in range(5):
                oracle.estimate(PREFILL)
        assert caplog.text.count("this one is prefill") == 1

    def test_a_matching_step_does_not_warn(self, tmp_path, caplog):
        prices, graph = _artifacts(tmp_path, [_op("a")])
        oracle = PricedGraphCostOracle(prices, graph)
        with caplog.at_level("WARNING"):
            oracle.estimate(DECODE)
        assert "this one is" not in caplog.text
