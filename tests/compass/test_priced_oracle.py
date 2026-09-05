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


#: A replayed step: `capture_bucket` is the rung it padded up to. A step with
#: none ran eager, and pays for dispatching every operator rather than for the
#: boundaries between kernels of one submission -- thirtyfold apart, so which of
#: the two applies is not a detail.
DECODE = StepShape(num_scheduled_tokens=(1, 1), context_lens=(8, 8),
                   capture_bucket=2)
EAGER_DECODE = StepShape(num_scheduled_tokens=(1, 1), context_lens=(8, 8))
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
        assert oracle.decode.launches == 3
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-5 + 3 * 2e-6)

    def test_an_eager_step_pays_per_operator_instead(self, tmp_path):
        """No `capture_bucket` means nothing was replayed and the host
        dispatched each operator. Using the replayed constant for a prefill step
        priced it at 0.42 of the step."""
        prices, graph = _artifacts(tmp_path, [_op("a"), _op("b")], seconds=1e-5)
        oracle = PricedGraphCostOracle(prices, graph, boundary_seconds=2e-6,
                                       eager_seconds_per_op=7e-5)
        assert oracle.estimate(EAGER_DECODE).seconds == pytest.approx(
            2 * 1e-5 + 2 * 7e-5)

    def test_dispatch_is_hidden_by_the_kernels_that_outrun_it(self, tmp_path):
        """The default: compute the eager cost rather than take a flat figure.

        The host dispatches while the device works, so an operator pays only
        `max(0, dispatch - kernel)`. That is why a flat per-operator figure came
        out 86us on a model with small kernels and 35us on one with large ones
        while the dispatch behind both was nearly the same number.
        """
        # Two operators at 10us each, dispatch 25us: each pays the 15us the
        # kernel cannot hide.
        prices, graph = _artifacts(tmp_path, [_op("a"), _op("b")], seconds=1e-5)
        oracle = PricedGraphCostOracle(prices, graph, dispatch_seconds=2.5e-5)
        assert oracle.estimate(EAGER_DECODE).seconds == pytest.approx(
            2 * 1e-5 + 2 * 1.5e-5)

    def test_a_kernel_longer_than_the_dispatch_pays_nothing(self, tmp_path):
        prices, graph = _artifacts(tmp_path, [_op("a")], seconds=1e-4)
        oracle = PricedGraphCostOracle(prices, graph, dispatch_seconds=2.5e-5)
        assert oracle.estimate(EAGER_DECODE).seconds == pytest.approx(1e-4)

    def test_a_flat_figure_still_overrides(self, tmp_path):
        """Kept so a deployment that has measured its own can say so."""
        prices, graph = _artifacts(tmp_path, [_op("a")], seconds=1e-5)
        oracle = PricedGraphCostOracle(prices, graph, eager_seconds_per_op=7e-5,
                                       dispatch_seconds=2.5e-5)
        assert oracle.estimate(EAGER_DECODE).seconds == pytest.approx(1e-5 + 7e-5)

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
        assert caplog.text.count("no prefill graph") == 1

    def test_a_matching_step_does_not_warn(self, tmp_path, caplog):
        prices, graph = _artifacts(tmp_path, [_op("a")])
        oracle = PricedGraphCostOracle(prices, graph)
        with caplog.at_level("WARNING"):
            oracle.estimate(DECODE)
        assert "no prefill graph" not in caplog.text


class TestPrefill:
    """A decode graph cannot answer for a prefill step.

    Different operators at different shapes -- answering with the decode cost is
    not an approximation but a different question. Interpolating between decode
    graphs at different context lengths was tried and dropped as not worth its
    complexity; two graphs of two *kinds* is a different proposition, and is the
    difference between predicting TTFT and not predicting it at all.
    """

    def test_the_prefill_graph_answers_prefill_steps(self, tmp_path):
        prices, decode = _artifacts(tmp_path, [_op("d")], seconds=1e-5)
        sub = tmp_path / "p"
        sub.mkdir()
        prefill_prices, prefill = _artifacts(sub, [_op("p")], seconds=4e-5)
        # One price list has to cover both, as pricing a glob produces.
        merged = json.loads(open(prices).read())
        merged["prices"].update(json.loads(open(prefill_prices).read())["prices"])
        (tmp_path / "both.json").write_text(json.dumps(merged))

        oracle = PricedGraphCostOracle(str(tmp_path / "both.json"), decode,
                                       prefill_graph=prefill,
                                       boundary_seconds=0.0,
                                       eager_seconds_per_op=0.0)
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-5)
        assert oracle.estimate(PREFILL).seconds == pytest.approx(4e-5)

    def test_without_one_a_prefill_step_warns(self, tmp_path, caplog):
        prices, graph = _artifacts(tmp_path, [_op("a")])
        oracle = PricedGraphCostOracle(prices, graph)
        with caplog.at_level("WARNING"):
            oracle.estimate(PREFILL)
        assert "no prefill graph" in caplog.text


class TestDecodeRungs:
    """Decode shapes are enumerable, so they are measured rather than guessed.

    They are the rungs of the CUDA-graph ladder, known from config before
    anything runs. Interpolating a price across shapes was probed and rejected:
    the library re-tunes its kernel at nearly every shape and a retune can cost
    2.4x, with nothing in the price list able to say in advance which neighbours
    are safe to interpolate between.
    """

    def _ladder(self, tmp_path, rungs):
        import json as _json

        prices = {"prices": {}}
        for n in rungs:
            op = _op(f"a{n}")
            graph = {"version": 2, "key": None, "ops": [op],
                     "provenance": {"shape": {
                         "num_scheduled_tokens": [1] * n,
                         "context_lens": [8] * n,
                         "num_prefill_tokens": 0, "capture_bucket": n}}}
            (tmp_path / f"g.b{n}.json").write_text(_json.dumps(graph))
            prices["prices"][signature_of(op)] = {
                "name": op["name"], "seconds": 1e-5 * n, "occurrences": 1,
                "kernels": {"k": 1e-5 * n}}
        (tmp_path / "p.json").write_text(_json.dumps(prices))
        return str(tmp_path / "p.json"), str(tmp_path / "g.b*.json")

    def _shape(self, rung):
        return StepShape(num_scheduled_tokens=(1,) * rung,
                         context_lens=(8,) * rung, capture_bucket=rung)

    def test_each_rung_gets_its_own_graph(self, tmp_path):
        prices, graphs = self._ladder(tmp_path, [1, 2, 4, 8])
        oracle = PricedGraphCostOracle(prices, graphs, boundary_seconds=0.0)
        assert sorted(oracle.by_rung) == [1, 2, 4, 8]
        for rung in (1, 2, 4, 8):
            assert oracle.estimate(self._shape(rung)).seconds == pytest.approx(
                1e-5 * rung)

    def test_an_unmeasured_rung_warns_rather_than_interpolating(self, tmp_path,
                                                                caplog):
        prices, graphs = self._ladder(tmp_path, [1, 2, 8])
        oracle = PricedGraphCostOracle(prices, graphs, boundary_seconds=0.0)
        with caplog.at_level("WARNING"):
            got = oracle.estimate(self._shape(4)).seconds
        # The largest measured rung, not something interpolated between 2 and 8.
        assert got == pytest.approx(1e-5 * 8)
        assert "no decode graph for rung 4" in caplog.text

    def test_and_warns_only_once(self, tmp_path, caplog):
        prices, graphs = self._ladder(tmp_path, [1, 8])
        oracle = PricedGraphCostOracle(prices, graphs, boundary_seconds=0.0)
        with caplog.at_level("WARNING"):
            for _ in range(4):
                oracle.estimate(self._shape(4))
        assert caplog.text.count("no decode graph for rung") == 1

    def test_one_graph_behaves_as_before(self, tmp_path):
        prices, graph = _artifacts(tmp_path, [_op("a")], seconds=1e-5)
        oracle = PricedGraphCostOracle(prices, graph, boundary_seconds=0.0)
        assert oracle.estimate(DECODE).seconds == pytest.approx(1e-5)


class TestHowTheStepWasRun:
    """Replayed, compiled, and eager are three ways to run a step, not two.

    A compiled step submits its kernels from generated code: it pays neither
    eager dispatch nor a replay's per-launch boundary. Charging it the eager
    term overstated one chunked prefill's overhead four times over, 21.5ms
    against a measured 5.7ms.
    """

    def _oracle(self, tmp_path, monkeypatch):
        import json

        from atom.compass.core.cost.priced import PricedGraphCostOracle

        op = {"name": "aten::mm", "input_shapes": [[4, 8]],
              "output_shapes": [], "dtypes": ["bfloat16"]}
        graph = tmp_path / "g.json"
        graph.write_text(json.dumps({"version": 1, "ops": [op] * 10}))
        from atom.compass.runtime.microbench import signature_of
        prices = tmp_path / "p.json"
        prices.write_text(json.dumps({"prices": {
            signature_of(op): {"seconds": 1e-6, "occurrences": 1,
                               "kernels": {"k": 1e-6}}}}))
        return PricedGraphCostOracle(prices=str(prices), graph=str(graph))

    def test_a_compiled_step_pays_neither_other_term(self, tmp_path, monkeypatch):
        from atom.compass.core.cost.base import StepShape

        oracle = self._oracle(tmp_path, monkeypatch)
        shape = dict(num_scheduled_tokens=(1,) * 4, context_lens=(8,) * 4)
        compiled = oracle.estimate(StepShape(**shape, compiled=True)).seconds
        eager = oracle.estimate(StepShape(**shape, compiled=False)).seconds
        assert compiled < eager, (compiled, eager)
        # ten operators, one kernel each, at the compiled per-launch figure
        assert compiled == pytest.approx(
            10 * 1e-6 + 10 * oracle.compiled_seconds_per_launch, rel=1e-6)

    def test_a_shape_that_does_not_say_is_treated_as_eager(self, tmp_path,
                                                           monkeypatch):
        """Every graph traced before this existed says nothing."""
        from atom.compass.core.cost.base import StepShape

        oracle = self._oracle(tmp_path, monkeypatch)
        shape = dict(num_scheduled_tokens=(1,) * 4, context_lens=(8,) * 4)
        assert (oracle.estimate(StepShape(**shape)).seconds
                == oracle.estimate(StepShape(**shape, compiled=False)).seconds)
