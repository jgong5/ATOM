"""The library oracle: a step summed from measured operator prices.

What these check is not the arithmetic -- a sum is a sum -- but the four ways
the sum could be wrong while looking right: an operator answered from a price
measured against different memory, a collective answered from another parallel
width, a partial run read as a complete library, and a missing operator
silently contributing zero.
"""

import json

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.library import (
    LibraryCostOracle, PriceLibrary, StaticGraphs)


def _op(name, shapes, dtypes=("bfloat16",), layouts=None, context=None):
    op = {"name": name,
          "input_shapes": [list(s) for s in shapes],
          "dtypes": list(dtypes)}
    if layouts is not None:
        op["layouts"] = layouts
    if context is not None:
        op["context"] = context
    return op


def _graph(ops, topology=None):
    blob = {"ops": ops}
    if topology is not None:
        blob["key"] = {"topology": [list(x) for x in topology.items()]}
    return blob


def _price_list(tmp_path, name, ops, seconds, kernels=1, topology=None,
                only=None, unpriced=None):
    """A price list in the shape `price_graph` writes, for the given ops."""
    from atom.compass.runtime.microbench import signature_of

    prices = {}
    for op in ops:
        prices[signature_of(op)] = {
            "name": op["name"], "seconds": seconds, "occurrences": 1,
            "kernels": {f"k{i}": seconds / kernels for i in range(kernels)},
        }
    blob = {"prices": prices, "unpriced": unpriced or {},
            "coverage": {}, "provenance": {"topology": topology, "only": only}}
    path = tmp_path / name
    path.write_text(json.dumps(blob))
    return str(path)


class TestAPriceIsAPriceOfAParticularArrangementOfMemory:
    """A signature does not carry layout, so matching one proves less than it
    looks like it does."""

    def test_a_dense_price_does_not_answer_a_strided_call(self, tmp_path):
        dense = _op("triton::norm", [[4, 24, 256]])
        strided = _op("triton::norm", [[4, 24, 256]],
                      layouts=[[0, [[14336, 256, 1], 6144, 57344, 0]]])
        # Same key: this is the whole hazard.
        from atom.compass.runtime.microbench import signature_of
        assert signature_of(dense) == signature_of(strided)

        prices = _price_list(tmp_path, "v1.json", [dense], 1e-3)
        lib = PriceLibrary.load([(prices, None)])
        # With no graph to say what was measured, the match is taken.
        assert lib.lookup(strided)[0] is not None

        graph_path = tmp_path / "v1graph.json"
        graph_path.write_text(json.dumps(_graph([dense])))
        told = PriceLibrary.load([(prices, str(graph_path))])
        record, why = told.lookup(strided)
        assert record is None
        assert "different operand layout" in why

    def test_the_same_layout_matches(self, tmp_path):
        strided = _op("triton::norm", [[4, 24, 256]],
                      layouts=[[0, [[14336, 256, 1], 6144, 57344, 0]]])
        prices = _price_list(tmp_path, "v2.json", [strided], 1e-3)
        graph_path = tmp_path / "v2graph.json"
        graph_path.write_text(json.dumps(_graph([strided])))
        lib = PriceLibrary.load([(prices, str(graph_path))])
        assert lib.lookup(strided)[0]["seconds"] == pytest.approx(1e-3)


class TestACollectiveIsPricedAtItsOwnWidth:
    """`all_reduce_` over 2 ranks and over 4 sign identically."""

    def test_a_four_way_price_does_not_answer_a_two_way_call(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[16, 4096]])
        prices = _price_list(tmp_path, "tp4.json", [coll], 2e-4,
                             topology={"tp": 4})
        lib = PriceLibrary.load([(prices, None)])
        _seconds, coverage, _launches = lib.body(
            _graph([coll], topology={"tp": 2}))
        assert coverage.priced == 0
        assert "group width" in coverage.reasons["aiter::all_reduce_"]

    def test_its_own_width_is_paid(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[16, 4096]])
        prices = _price_list(tmp_path, "tp2.json", [coll], 2e-4,
                             topology={"tp": 2})
        lib = PriceLibrary.load([(prices, None)])
        _seconds, coverage, _launches = lib.body(
            _graph([coll], topology={"tp": 2}))
        assert coverage.priced == 1


class TestWhatWasNotPricedIsSaidRatherThanZeroed:

    def test_a_missing_operator_is_named_not_absorbed(self, tmp_path):
        known = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        unknown = _op("aiter::linear_attention", [[16, 128, 256]])
        prices = _price_list(tmp_path, "p.json", [known], 1e-3)
        lib = PriceLibrary.load([(prices, None)])
        seconds, coverage, _l = lib.body(_graph([known, known, unknown]))
        assert seconds == pytest.approx(2e-3)
        assert (coverage.operators, coverage.priced) == (3, 2)
        assert coverage.refused == {"aiter::linear_attention": 1}
        assert not coverage.complete
        assert "UNPRICED 1" in coverage.describe()

    def test_a_pricing_run_s_own_refusal_is_carried_through(self, tmp_path):
        op = _op("triton::norm", [[4, 24, 256]])
        from atom.compass.runtime.microbench import signature_of
        prices = _price_list(
            tmp_path, "p.json", [], 0.0,
            unpriced={signature_of(op): "#9=14336 exceeds every argument's "
                                        "row extent (6144)"})
        lib = PriceLibrary.load([(prices, None)])
        _seconds, coverage, _l = lib.body(_graph([op]))
        assert "row extent" in coverage.reasons["triton::norm"]

    def test_require_complete_refuses_a_partial_sum(self, tmp_path):
        known = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        unknown = _op("aiter::linear_attention", [[16, 128, 256]])
        prices = _price_list(tmp_path, "p.json", [known], 1e-3)
        shape = StepShape(num_scheduled_tokens=(1,), context_lens=(16,))
        graphs = StaticGraphs({StaticGraphs.key(shape):
                               _graph([known, unknown])})
        oracle = LibraryCostOracle(PriceLibrary.load([(prices, None)]), graphs,
                                   require_complete=True)
        with pytest.raises(ValueError, match="incomplete"):
            oracle.estimate(shape)


class TestALibraryKnowsWhatItIsMadeOf:

    def test_a_narrowed_run_marks_the_library_partial(self, tmp_path):
        op = _op("aiter::linear_attention", [[16, 128, 256]])
        prices = _price_list(tmp_path, "only.json", [op], 1e-3,
                             only="linear_attention")
        lib = PriceLibrary.load([(prices, None)])
        assert lib.partial and "linear_attention" in lib.partial[0]
        assert "PARTIAL" in lib.describe()

    def test_two_runs_that_disagree_are_recorded_not_averaged(self, tmp_path):
        op = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        a = _price_list(tmp_path, "a.json", [op], 1e-3)
        b = _price_list(tmp_path, "b.json", [op], 4e-3)
        lib = PriceLibrary.load([(a, None), (b, None)])
        assert len(lib.conflicts) == 1
        # The first is kept; nothing is blended into a number neither run saw.
        assert lib.lookup(op)[0]["seconds"] == pytest.approx(1e-3)
        assert "disagree" in lib.describe()

    def test_coverage_says_which_run_answered(self, tmp_path):
        op = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        a = _price_list(tmp_path, "a.json", [op], 1e-3)
        lib = PriceLibrary.load([(a, None)])
        _s, coverage, _l = lib.body(_graph([op, op]))
        assert coverage.sources == {a: 2}


class TestTheStepIsTheBodyPlusWhatTheBodyIsNot:

    def test_launches_come_from_the_priced_kernels_not_the_operators(
            self, tmp_path):
        # Attention launches three kernels and pays the execution term at each.
        op = _op("aiter::attention", [[16, 24, 256]])
        prices = _price_list(tmp_path, "p.json", [op], 1e-3, kernels=3)
        lib = PriceLibrary.load([(prices, None)])
        _s, _c, launches = lib.body(_graph([op, op]))
        assert launches == 6

    def test_the_runner_term_is_additive_and_visible(self, tmp_path):
        op = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        prices = _price_list(tmp_path, "p.json", [op], 1e-3)
        shape = StepShape(num_scheduled_tokens=(1,), context_lens=(16,))
        graphs = StaticGraphs({StaticGraphs.key(shape): _graph([op])})
        oracle = LibraryCostOracle(PriceLibrary.load([(prices, None)]), graphs,
                                   seconds_per_launch=2e-6,
                                   extra_seconds=5e-4)
        cost = oracle.estimate(shape)
        assert cost.seconds == pytest.approx(1e-3 + 2e-6 + 5e-4)
        assert cost.breakdown["<runner>"] == pytest.approx(5e-4)
        assert cost.breakdown["<body>"] == pytest.approx(1e-3)

    def test_a_shape_with_no_graph_is_refused_not_approximated(self, tmp_path):
        op = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        prices = _price_list(tmp_path, "p.json", [op], 1e-3)
        have = StepShape(num_scheduled_tokens=(1,), context_lens=(16,))
        want = StepShape(num_scheduled_tokens=(1, 1), context_lens=(16, 16))
        graphs = StaticGraphs({StaticGraphs.key(have): _graph([op])})
        oracle = LibraryCostOracle(PriceLibrary.load([(prices, None)]), graphs)
        with pytest.raises(KeyError):
            oracle.estimate(want)

    def test_a_reordered_batch_is_the_same_shape(self, tmp_path):
        """Equivalence up to permutation of interchangeable requests, which is
        what makes a derivation cache worth having. Whole rows move together:
        request 2 is still (4 tokens, context 4096) after the swap."""
        op = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        prices = _price_list(tmp_path, "p.json", [op], 1e-3)
        one = StepShape(num_scheduled_tokens=(1, 4), context_lens=(16, 4096))
        other = StepShape(num_scheduled_tokens=(4, 1), context_lens=(4096, 16))
        graphs = StaticGraphs({StaticGraphs.key(one): _graph([op])})
        oracle = LibraryCostOracle(PriceLibrary.load([(prices, None)]), graphs)
        assert oracle.estimate(other).seconds == pytest.approx(1e-3)

    def test_repairing_the_rows_is_not_a_permutation(self):
        """The hazard a sorted-lists key hides: same lengths, different
        pairing, and an attention cost that is nothing like the same."""
        short_reads_little = StepShape(num_scheduled_tokens=(1, 64),
                                       context_lens=(1, 1088))
        short_reads_a_lot = StepShape(num_scheduled_tokens=(64, 1),
                                      context_lens=(64, 1025))
        # Sorted independently these agree on both lists -- (1, 64) queries and
        # (0, 1024) cached -- which is exactly why they must not key alike.
        assert (sorted(short_reads_little.num_scheduled_tokens)
                == sorted(short_reads_a_lot.num_scheduled_tokens))
        assert (StaticGraphs.key(short_reads_little)
                != StaticGraphs.key(short_reads_a_lot))

    def test_a_different_group_width_is_a_different_graph(self):
        one = StepShape(num_scheduled_tokens=(1,), context_lens=(16,),
                        topology={"tp": 2})
        other = StepShape(num_scheduled_tokens=(1,), context_lens=(16,),
                          topology={"tp": 4})
        assert StaticGraphs.key(one) != StaticGraphs.key(other)

    def test_a_prefill_row_and_a_decode_row_of_the_same_lengths_differ(self):
        """Same rows, different branch: `num_prefill_tokens` selects it."""
        prefill = StepShape(num_scheduled_tokens=(4,), context_lens=(4,),
                            num_prefill_tokens=4)
        decode = StepShape(num_scheduled_tokens=(4,), context_lens=(4,))
        assert StaticGraphs.key(prefill) != StaticGraphs.key(decode)

    def test_a_padded_replay_is_not_the_eager_step(self):
        eager = StepShape(num_scheduled_tokens=(12,), context_lens=(64,))
        replayed = StepShape(num_scheduled_tokens=(12,), context_lens=(64,),
                             capture_bucket=16)
        assert StaticGraphs.key(eager) != StaticGraphs.key(replayed)
