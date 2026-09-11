"""The library oracle: a step summed from measured operator prices.

What these check is not the arithmetic -- a sum is a sum -- but the four ways
the sum could be wrong while looking right: an operator answered from a price
measured against different memory, a collective answered from another parallel
width, a partial run read as a complete library, and a missing operator
silently contributing zero.
"""

import json
import pathlib

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.library import LibraryCostOracle, PriceLibrary, StaticGraphs


def _op(name, shapes, dtypes=("bfloat16",), layouts=None, context=None):
    op = {"name": name,
          "input_shapes": [list(s) for s in shapes],
          "dtypes": list(dtypes)}
    if layouts is not None:
        op["layouts"] = layouts
    if context is not None:
        op["context"] = context
    return op


def _graph(ops, topology=None, head_in_graph=None, padded_rows=False,
           body_rows=None, in_replay=None, registration=None):
    """A graph blob in the shape `graph_diff.py` writes.

    `head_in_graph` is the recorded fact -- does this graph's operator list
    contain `compute_logits` -- and `in_replay` is the separate claim about the
    production step, which needs the step kind and the cudagraph mode and is
    `None` when the deriver could not determine it. They are not the same field
    and TP alone decides neither. Both are left off by default so the tests
    above keep exercising the graphs that predate them.
    """
    blob = {"ops": ops}
    if topology is not None:
        blob["key"] = {"topology": [list(x) for x in topology.items()]}
    if head_in_graph is not None:
        blob["provenance"] = {"head_placement": {
            "in_this_graph": head_in_graph,
            "in_replayed_body_graph": in_replay,
            "rows_padded_to_capture_bucket": padded_rows,
        }}
    if body_rows is not None:
        blob.setdefault("provenance", {})["execution"] = {
            "body_rows_traced": body_rows,
        }
    if registration is not None:
        # A graph states a requirement. What a benchmark measured is a
        # different field with a different name; see the test class below.
        blob.setdefault("provenance", {})[
            "collective_registration_required"] = registration
    return blob


def _price_list(tmp_path, name, ops, seconds, kernels=1, topology=None,
                only=None, unpriced=None, registration=None):
    """A price list in the shape `price_graph` writes, for the given ops."""
    from atom.compass.runtime.microbench import signature_of

    prices = {}
    for op in ops:
        prices[signature_of(op)] = {
            "name": op["name"], "seconds": seconds, "occurrences": 1,
            "kernels": {f"k{i}": seconds / kernels for i in range(kernels)},
        }
    blob = {"prices": prices, "unpriced": unpriced or {},
            "coverage": {}, "provenance": {"topology": topology, "only": only,
                                           "collective_registration_measured":
                                               registration}}
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
                             topology={"tp": 4}, registration="registered")
        lib = PriceLibrary.load([(prices, None)])
        _seconds, coverage, _launches = lib.body(
            _graph([coll], topology={"tp": 2}, registration="registered"))
        assert coverage.priced == 0
        assert "group width" in coverage.reasons["aiter::all_reduce_"]

    def test_its_own_width_is_paid(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[16, 4096]])
        prices = _price_list(tmp_path, "tp2.json", [coll], 2e-4,
                             topology={"tp": 2}, registration="registered")
        lib = PriceLibrary.load([(prices, None)])
        _seconds, coverage, _launches = lib.body(
            _graph([coll], topology={"tp": 2}, registration="registered"))
        assert coverage.priced == 1


class TestACollectiveIsPricedOnTheDataPathItActuallyRuns:
    """One operator, two paths, one signature.

    `CustomAllreduce` either copies the input into a registered IPC buffer or
    lets the peers read it where it lies, and which one runs depends on whether
    the communicator was armed at capture -- not on anything in the recorded
    signature. The measured gap is 9.06 us against 6.29 us for the same 4x5120
    reduction, so a library that cannot tell them apart is a library that can
    be 44% wrong on every all-reduce in the step while reporting full coverage.
    """

    def _two_lists(self, tmp_path, coll):
        return (_price_list(tmp_path, "copy.json", [coll], 9.06e-6,
                            topology={"tp": 2}, registration="unregistered"),
                _price_list(tmp_path, "reg.json", [coll], 6.29e-6,
                            topology={"tp": 2}, registration="registered"))

    def test_the_needed_path_is_selected_whichever_order_they_load_in(
            self, tmp_path):
        # The point of the scoped store: correctness must not depend on which
        # file the caller happened to list first.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, reg = self._two_lists(tmp_path, coll)
        graph = _graph([coll], topology={"tp": 2}, registration="registered")
        for order in ((copy, reg), (reg, copy)):
            lib = PriceLibrary.load([(p, None) for p in order])
            seconds, coverage, _l = lib.body(graph)
            assert coverage.priced == 1
            assert seconds == pytest.approx(6.29e-6)
            assert coverage.sources == {reg: 1}

    def test_the_other_path_is_still_there_under_its_own_scope(self, tmp_path):
        # The copy-path measurement is a real measurement of a real path. It is
        # kept, and it answers the region that runs that path -- the eager head.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, reg = self._two_lists(tmp_path, coll)
        lib = PriceLibrary.load([(reg, None), (copy, None)])
        seconds, coverage, _l = lib.body(
            _graph([coll], topology={"tp": 2}, registration="unregistered"))
        assert coverage.priced == 1
        assert seconds == pytest.approx(9.06e-6)

    def test_two_paths_are_not_a_disagreement(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, reg = self._two_lists(tmp_path, coll)
        lib = PriceLibrary.load([(copy, None), (reg, None)])
        # 9.06 and 6.29 differ by far more than 5%, and they are not in
        # conflict: they are prices of two different things.
        assert lib.conflicts == {}
        assert "more than one scope" in lib.describe()

    def test_one_path_measured_twice_is_still_a_disagreement(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        a = _price_list(tmp_path, "a.json", [coll], 9.06e-6,
                        topology={"tp": 2}, registration="unregistered")
        b = _price_list(tmp_path, "b.json", [coll], 14.0e-6,
                        topology={"tp": 2}, registration="unregistered")
        lib = PriceLibrary.load([(a, None), (b, None)])
        assert len(lib.conflicts) == 1

    def test_an_undeclared_requirement_is_refused_not_guessed(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, reg = self._two_lists(tmp_path, coll)
        lib = PriceLibrary.load([(copy, None), (reg, None)])
        # A graph that does not say which path its collectives take gets no
        # price, even though two prices match its signature exactly.
        _s, coverage, _l = lib.body(_graph([coll], topology={"tp": 2}))
        assert coverage.priced == 0
        assert "not declared" in coverage.reasons["aiter::all_reduce_"]

    def test_an_unmeasured_path_is_refused_not_substituted(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, _reg = self._two_lists(tmp_path, coll)
        lib = PriceLibrary.load([(copy, None)])
        _s, coverage, _l = lib.body(
            _graph([coll], topology={"tp": 2}, registration="registered"))
        assert coverage.priced == 0
        why = coverage.reasons["aiter::all_reduce_"]
        assert "registered path" in why and "unregistered" in why

    def test_a_price_list_that_does_not_say_cannot_pay_for_a_collective(
            self, tmp_path):
        # Everything written before the field existed. Its non-collective
        # prices are unaffected; only the collectives it cannot scope are.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        gemm = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        old = _price_list(tmp_path, "old.json", [coll, gemm], 9.06e-6,
                          topology={"tp": 2})
        lib = PriceLibrary.load([(old, None)])
        _s, coverage, _l = lib.body(
            _graph([coll, gemm], topology={"tp": 2}, registration="registered"))
        assert coverage.priced == 1
        assert coverage.refused == {"aiter::all_reduce_": 1}

    def test_a_caller_may_name_the_scope_an_old_file_leaves_unstated(
            self, tmp_path):
        # `microbench` captures into a bare `torch.cuda.graph`, which never arms
        # the communicator, and the probe confirmed the resulting price at
        # 0.7%. That is what the third element of a load pair records.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        old = _price_list(tmp_path, "old.json", [coll], 9.06e-6,
                          topology={"tp": 2})
        lib = PriceLibrary.load([(old, None, "unregistered")])
        _s, coverage, _l = lib.body(
            _graph([coll], topology={"tp": 2}, registration="unregistered"))
        assert coverage.priced == 1

    def test_a_caller_may_not_overrule_what_a_file_states(self, tmp_path):
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        stated = _price_list(tmp_path, "reg.json", [coll], 6.29e-6,
                             topology={"tp": 2}, registration="registered")
        with pytest.raises(ValueError, match="not overrule"):
            PriceLibrary.load([(stated, None, "unregistered")])

    def test_a_requirement_is_not_readable_as_a_measurement(self, tmp_path):
        # The two sides carry different key names on purpose. A pricing run
        # aimed at a graph that *requires* the registered path still measures
        # whichever path it actually took -- `microbench` captures into a bare
        # `torch.cuda.graph`, which never arms the communicator -- so a blob
        # that carries the graph's key has copied a requirement, not observed
        # anything, and is refused rather than believed.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        path = _price_list(tmp_path, "copied.json", [coll], 9.06e-6,
                           topology={"tp": 2})
        blob = json.loads((tmp_path / "copied.json").read_text())
        blob["provenance"]["collective_registration_required"] = "registered"
        (tmp_path / "copied.json").write_text(json.dumps(blob))
        with pytest.raises(ValueError, match="not a measurement"):
            PriceLibrary.load([(path, None)])

    def test_a_graph_s_requirement_is_not_read_from_the_measured_key(
            self, tmp_path):
        # The mirror of the above: a graph states what it needs, so the reader
        # of the graph side accepts only the required key. A graph carrying the
        # benchmark's key says nothing about what it needs, and its collective
        # is refused.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, reg = self._two_lists(tmp_path, coll)
        lib = PriceLibrary.load([(copy, None), (reg, None)])
        graph = _graph([coll], topology={"tp": 2})
        graph.setdefault("provenance", {})[
            "collective_registration_measured"] = "registered"
        _s, coverage, _l = lib.body(graph)
        assert coverage.priced == 0
        assert "not declared" in coverage.reasons["aiter::all_reduce_"]

    def test_same_shape_tp2_and_tp4_records_coexist(self, tmp_path):
        # One signature, two widths, both needed: the four-rank record must not
        # be blocked by the two-rank one having been loaded first, and neither
        # may answer for the other.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        tp2 = _price_list(tmp_path, "tp2.json", [coll], 6.29e-6,
                          topology={"tp": 2}, registration="registered")
        tp4 = _price_list(tmp_path, "tp4.json", [coll], 6.78e-6,
                          topology={"tp": 4}, registration="registered")
        for order in ((tp2, tp4), (tp4, tp2)):
            lib = PriceLibrary.load([(p, None) for p in order])
            for width, expected in ((2, 6.29e-6), (4, 6.78e-6)):
                seconds, coverage, _l = lib.body(
                    _graph([coll], topology={"tp": width},
                           registration="registered"))
                assert coverage.priced == 1
                assert seconds == pytest.approx(expected)

    def test_the_caller_may_state_the_regime_the_graph_does_not(self, tmp_path):
        # Regions differ within one step: the replayed body reduces on the
        # registered path and the eager head on the copy path, so the oracle
        # names each rather than the library assuming one for both.
        coll = _op("aiter::all_reduce_", [[4, 5120]])
        copy, reg = self._two_lists(tmp_path, coll)
        lib = PriceLibrary.load([(copy, None), (reg, None)])
        bare = _graph([coll], topology={"tp": 2})
        assert lib.body(bare, "registered")[0] == pytest.approx(6.29e-6)
        assert lib.body(bare, "unregistered")[0] == pytest.approx(9.06e-6)


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


class TestTheHeadIsChargedWhereItActuallyRuns:
    """Where the LM head runs is a property of the step, not of the width.

    `ModelRunner.logits_in_graph = self.world_size == 1 and not is_tbo`
    (model_runner.py:4104) is consulted by exactly one of the three branches
    that produce logits: the manual whole-forward replay (:3238-3241). A
    prefill (:3182) and a piecewise-compiled decode (:3230) call
    `compute_logits` eagerly at every width and never read the flag. So "TP1"
    does not mean "head in the graph" -- a TP1 prefill's head is outside it --
    and the same head term is required in one case and a double count in the
    other. The graph has to say which; the oracle never infers it.
    """

    #: The head's own operators: the vocab projection, and at TP>1 the gather
    #: that reassembles the shards. The gather's price cannot come from a
    #: rebuild -- `OpSpec.abi == "live-state"` -- so it enters the library from
    #: a real two-rank measurement, keyed on this same signature.
    PROJ = _op("aiter::gemm_a16w16", [[1, 5120], [5120, 124160]])
    GATHER = _op("aiter::all_gather_unreg", [[1, 124160]])

    def _oracle(self, tmp_path, body, head=None, shape=None, prices=None):
        op = _op("aiter::gemm", [[16, 4096], [4096, 4096]])
        priced = _price_list(tmp_path, "p.json",
                             [op] + (prices if prices is not None
                                     else [self.PROJ]), 1e-3)
        shape = shape or StepShape(num_scheduled_tokens=(1,),
                                   context_lens=(16,))
        heads = None
        if head is not None:
            heads = StaticGraphs({StaticGraphs.key(shape): head})
        return LibraryCostOracle(PriceLibrary.load([(priced, None)]),
                                 StaticGraphs({StaticGraphs.key(shape): body}),
                                 head_graphs=heads), shape

    def test_a_body_without_the_head_is_charged_for_one(self, tmp_path):
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=False),
            head=_graph([self.PROJ], head_in_graph=True))
        cost = oracle.estimate(shape)
        assert cost.breakdown["<head>"] == pytest.approx(1e-3)
        assert cost.seconds == pytest.approx(2e-3)

    def test_a_graph_that_already_has_the_head_refuses_a_second_one(
            self, tmp_path):
        # The TP1 level-3 case. Silently summing here inflates every sampling
        # step by a whole projection.
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=True),
            head=_graph([self.PROJ], head_in_graph=True))
        with pytest.raises(ValueError, match="already contains the LM head"):
            oracle.estimate(shape)

    def test_a_graph_that_does_not_say_is_refused_too(self, tmp_path):
        # Not read as "outside": an unstated placement is the case where both
        # the double count and the missing head are invisible.
        oracle, shape = self._oracle(
            tmp_path, _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])]),
            head=_graph([self.PROJ], head_in_graph=True))
        with pytest.raises(ValueError, match="does not say"):
            oracle.estimate(shape)

    def test_a_step_that_samples_nothing_pays_no_head(self, tmp_path):
        # `is_pure_middle_chunk(batch)` -> `logits = None`
        # (model_runner.py:3174): a middle chunk of a long prompt projects no
        # row at all. Charging it one is the error the prefhead/prefdeep
        # fixtures exist to catch.
        middle = StepShape(num_scheduled_tokens=(512,), context_lens=(1024,),
                           num_prefill_tokens=512, produces_output=False)
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=False),
            head=_graph([self.PROJ], head_in_graph=True), shape=middle)
        cost = oracle.estimate(shape)
        assert "<head>" not in cost.breakdown
        assert cost.seconds == pytest.approx(1e-3)

    def test_an_unstated_placement_is_fine_when_no_head_region_is_supplied(
            self, tmp_path):
        # The refusal is about composition, not about provenance for its own
        # sake: every graph derived before the field existed must still price.
        oracle, shape = self._oracle(
            tmp_path, _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])]))
        assert oracle.estimate(shape).seconds == pytest.approx(1e-3)

    def test_an_unpriced_gather_makes_the_whole_step_incomplete(self, tmp_path):
        # The head's all-gather has no price until a real two-rank measurement
        # supplies one. The failure to avoid is a step that reports complete
        # coverage because the gap is in a region whose record nobody merged.
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=False),
            head=_graph([self.PROJ, self.GATHER], head_in_graph=True))
        oracle.estimate(shape)
        coverage = oracle.last_coverage
        assert not coverage.complete
        assert coverage.refused == {"aiter::all_gather_unreg": 1}

    def test_a_padded_head_gemm_is_not_charged_to_an_eager_step(self, tmp_path):
        # A capture projects `outputs[:bs * max_q_len]` and slices afterwards
        # (model_runner.py:4297, 3239); the eager head at TP>1 is handed hidden
        # states already cut to the real count (model_runner.py:3237-3241). The
        # same twenty requests are not the same head GEMM at both widths.
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=False),
            head=_graph([self.PROJ], head_in_graph=True, padded_rows=True))
        with pytest.raises(ValueError, match="padded rows"):
            oracle.estimate(shape)

    def test_a_body_graph_handed_in_as_the_head_is_refused(self, tmp_path):
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=False),
            head=_graph([self.PROJ], head_in_graph=False))
        with pytest.raises(ValueError, match="it is a body graph"):
            oracle.estimate(shape)

    def test_a_prefill_at_tp1_still_gets_a_head(self, tmp_path):
        # The case the width rule gets wrong. A final prefill chunk at TP1
        # produces one output position and its head runs eagerly
        # (model_runner.py:3182), so `in_this_graph` is false even though the
        # width is one, and the head term is required.
        tail = StepShape(num_scheduled_tokens=(256,), context_lens=(768,),
                         num_prefill_tokens=256, topology={"tp": 1})
        oracle, shape = self._oracle(
            tmp_path,
            _graph([_op("aiter::gemm", [[16, 4096], [4096, 4096]])],
                   head_in_graph=False, in_replay=False),
            head=_graph([self.PROJ], head_in_graph=True), shape=tail)
        assert oracle.estimate(shape).breakdown["<head>"] == pytest.approx(1e-3)


class TestABodyIsChargedForTheRowsItActuallyRuns:
    """A replay runs its bucket. A graph traced at the batch size does not.

    `num_tokens_pad = running_bs * max_q_len` (model_runner.py:3841-3843) is
    what the captured body executes; the real count only slices the result
    afterwards (:3189, :3227-3229). So a twenty-row trace and a thirty-two-row
    replay are different amounts of work under the same name, and the gap is
    the padding -- invisible, because both graphs price without complaint.
    """

    GEMM = _op("aiter::gemm", [[16, 4096], [4096, 4096]])

    def _oracle(self, tmp_path, graph, shape):
        priced = _price_list(tmp_path, "p.json", [self.GEMM], 1e-3)
        return LibraryCostOracle(
            PriceLibrary.load([(priced, None)]),
            StaticGraphs({StaticGraphs.key(shape): graph}))

    def _decode(self, requests, bucket=None):
        return StepShape(num_scheduled_tokens=(1,) * requests,
                         context_lens=(128,) * requests,
                         capture_bucket=bucket)

    def test_a_twenty_row_trace_is_refused_for_a_thirty_two_row_replay(
            self, tmp_path):
        shape = self._decode(20, bucket=32)
        oracle = self._oracle(tmp_path, _graph([self.GEMM], body_rows=20),
                              shape)
        with pytest.raises(ValueError, match="traced over 20 rows"):
            oracle.estimate(shape)

    def test_a_bucket_sized_trace_is_what_the_replay_pays_for(self, tmp_path):
        shape = self._decode(20, bucket=32)
        oracle = self._oracle(tmp_path, _graph([self.GEMM], body_rows=32),
                              shape)
        assert oracle.estimate(shape).seconds == pytest.approx(1e-3)

    def test_an_eager_step_runs_its_real_tokens(self, tmp_path):
        # No bucket, so no padding: the rows are the tokens.
        shape = StepShape(num_scheduled_tokens=(256,), context_lens=(0,),
                          num_prefill_tokens=256)
        oracle = self._oracle(tmp_path, _graph([self.GEMM], body_rows=256),
                              shape)
        assert oracle.estimate(shape).seconds == pytest.approx(1e-3)

    def test_a_bucket_sized_trace_is_refused_for_an_eager_step(self, tmp_path):
        # The mirror of the first case, and the more tempting one: a graph
        # derived once for the captured decode, reused for the eager one.
        shape = self._decode(20)
        oracle = self._oracle(tmp_path, _graph([self.GEMM], body_rows=32),
                              shape)
        with pytest.raises(ValueError, match="eager, unpadded"):
            oracle.estimate(shape)

    def test_a_graph_that_does_not_state_its_rows_is_priced_as_before(
            self, tmp_path):
        # The guard catches a mismatch it can see. It is not a second
        # completeness rule, and it does not invalidate graphs derived before
        # the field existed.
        shape = self._decode(20, bucket=32)
        oracle = self._oracle(tmp_path, _graph([self.GEMM]), shape)
        assert oracle.estimate(shape).seconds == pytest.approx(1e-3)


class TestADomainRefusalIsAskedBeforeTheWorkItWouldDiscard:
    """Whether a shape is inside the region model is a fact about the shape.

    It needs no graph, so asking it after binding -- and, on a miss, after a
    trace -- buys nothing and spends the most expensive thing the oracle does.
    The refusal itself is unchanged: same predicate, same message. What this
    pins is the order, because the order is invisible in the answer and only
    shows up in a profile.
    """

    GEMM = _op("aiter::gemm", [[16, 4096], [4096, 4096]])

    class CountingGraphs:
        """A graph source that records being asked."""

        def __init__(self, graph):
            self.graph, self.asked = graph, 0

        def graph_for(self, shape):
            self.asked += 1
            return self.graph

        def describe(self):
            return "CountingGraphs"

    class OnlyAtThirtyTwo:
        """The shape of a region model, narrowed to the one rule under test."""

        def refusal(self, shape):
            if len(shape.num_scheduled_tokens) != 32:
                return (f"decode over {len(shape.num_scheduled_tokens)} "
                        "sequences, measured only at [32]")
            return None

        def breakdown(self, shape):
            return {"<prepare>": 1e-4}

    def _oracle(self, tmp_path, graphs):
        priced = _price_list(tmp_path, "p.json", [self.GEMM], 1e-3)
        return LibraryCostOracle(PriceLibrary.load([(priced, None)]), graphs,
                                 regions=self.OnlyAtThirtyTwo())

    def _decode(self, requests):
        return StepShape(num_scheduled_tokens=(1,) * requests,
                         context_lens=(128,) * requests, capture_bucket=32)

    def test_a_shape_outside_the_domain_never_reaches_the_graph_source(
            self, tmp_path):
        graphs = self.CountingGraphs(_graph([self.GEMM]))
        oracle = self._oracle(tmp_path, graphs)
        with pytest.raises(ValueError, match="measured only at"):
            oracle.estimate(self._decode(20))
        assert graphs.asked == 0

    def test_a_shape_inside_the_domain_still_gets_its_graph(self, tmp_path):
        graphs = self.CountingGraphs(_graph([self.GEMM]))
        oracle = self._oracle(tmp_path, graphs)
        assert oracle.estimate(self._decode(32)).seconds > 0
        assert graphs.asked == 1

    def test_without_a_region_model_nothing_is_hoisted(self, tmp_path):
        """No regions means no domain, not an empty domain that refuses all."""
        graphs = self.CountingGraphs(_graph([self.GEMM]))
        priced = _price_list(tmp_path, "p.json", [self.GEMM], 1e-3)
        oracle = LibraryCostOracle(PriceLibrary.load([(priced, None)]), graphs)
        assert oracle.estimate(self._decode(20)).seconds > 0
        assert graphs.asked == 1


def _interpolate(path, *, seconds=None):
    """Mark every entry in a written price list as derived, not measured.

    The seam between the two modules is exactly this field: the module that
    fits family curves sets `"interpolated": true` on the record it returns,
    and this one reads it. Written here as JSON rather than imported so the
    test fails if the flag is renamed on either side.
    """
    blob = json.loads(pathlib.Path(path).read_text())
    for record in blob["prices"].values():
        record["interpolated"] = True
        if seconds is not None:
            record["seconds"] = seconds
            record["kernels"] = {"k0": seconds}
    pathlib.Path(path).write_text(json.dumps(blob))
    return path


def _declare_zero_work(path):
    """Mark every entry as an operator the configuration does not run."""
    blob = json.loads(pathlib.Path(path).read_text())
    for record in blob["prices"].values():
        record["zero_work"] = True
    pathlib.Path(path).write_text(json.dumps(blob))
    return path


class TestAFittedPriceIsNotAMeasurement:
    """Complete coverage may contain interpolation; it may not be *called*
    measurement.

    The failure this prevents is silent and terminal for the claim: a step
    summed entirely from fitted values reports `1/1 operators` exactly as a
    step summed from timings does, and every report downstream inherits the
    confusion. The counts are kept apart rather than recovered by subtraction,
    because a reader who has to subtract has already been told the wrong thing.
    """

    GEMM = _op("aiter::gemm_a16w16", [[4, 5120], [5120, 17408]])
    NORM = _op("triton::norm", [[4, 5120]])

    def test_an_interpolated_price_is_counted_apart_from_a_measured_one(
            self, tmp_path):
        measured = _price_list(tmp_path, "m.json", [self.GEMM], 1e-3)
        fitted = _interpolate(_price_list(tmp_path, "f.json", [self.NORM], 2e-4))
        library = PriceLibrary.load([(measured, None), (fitted, None)])
        _, coverage, _ = library.body(_graph([self.GEMM, self.NORM]))
        assert (coverage.measured, coverage.interpolated) == (1, 1)
        assert coverage.priced == 2

    def test_a_step_priced_entirely_by_fitting_is_complete_but_not_measured(
            self, tmp_path):
        """The distinction the PoC needs both halves of.

        It is complete -- there is a predicted cost for every operator, which
        is what a predictive claim requires. It is not direct measurement, and
        `complete_measured` is the question an acceptance gate asks.
        """
        fitted = _interpolate(_price_list(tmp_path, "f.json", [self.GEMM], 1e-3))
        library = PriceLibrary.load([(fitted, None)])
        seconds, coverage, _ = library.body(_graph([self.GEMM]))
        assert seconds > 0.0
        assert coverage.complete
        assert not coverage.complete_measured
        assert (coverage.interpolated, coverage.measured) == (1, 0)

    def test_an_operator_declared_not_to_run_is_counted_apart(self, tmp_path):
        """Accounted for, and not a timing.

        A collective at group width one, or a head on a chunk producing no
        token, is known not to run. Counting it as measured inflates the
        measured share with operators nothing was timed for; counting it as
        refused makes a fully-accounted step look incomplete. It is neither.
        """
        free = _declare_zero_work(
            _price_list(tmp_path, "z.json", [self.GEMM], 0.0))
        library = PriceLibrary.load([(free, None)])
        _, coverage, _ = library.body(_graph([self.GEMM]))
        assert (coverage.zero_work, coverage.measured) == (1, 0)
        assert coverage.complete and coverage.complete_measured

    def test_a_zero_time_alone_is_not_a_declaration_that_it_does_not_run(
            self, tmp_path):
        """Both markers are read, neither is inferred.

        A measurement can round to zero at the timer's floor. Promoting that to
        "the engine is known not to run this" turns a resolution limit into a
        structural claim about the configuration, which is the kind of error
        that reads as a stronger result than was obtained.
        """
        free = _price_list(tmp_path, "z0.json", [self.GEMM], 0.0)
        library = PriceLibrary.load([(free, None)])
        _, coverage, _ = library.body(_graph([self.GEMM]))
        assert (coverage.zero_work, coverage.measured) == (0, 1)

    def test_a_fit_that_lands_on_zero_is_still_a_fit(self, tmp_path):
        fitted = _interpolate(
            _price_list(tmp_path, "fz.json", [self.GEMM], 1e-3), seconds=0.0)
        library = PriceLibrary.load([(fitted, None)])
        _, coverage, _ = library.body(_graph([self.GEMM]))
        assert (coverage.interpolated, coverage.zero_work) == (1, 0)
        assert not coverage.complete_measured

    def test_the_split_survives_merging_two_regions_and_is_printed(
            self, tmp_path):
        """A body and a head are one step, and one coverage line."""
        measured = _price_list(tmp_path, "m.json", [self.GEMM], 1e-3)
        fitted = _interpolate(_price_list(tmp_path, "f.json", [self.NORM], 2e-4))
        library = PriceLibrary.load([(measured, None), (fitted, None)])
        _, body, _ = library.body(_graph([self.GEMM]))
        _, head, _ = library.body(_graph([self.NORM]))
        both = body.merged(head)
        assert (both.measured, both.interpolated, both.operators) == (1, 1, 2)
        # And a reader of the line is told what the 2/2 is made of.
        assert "1 measured, 1 interpolated" in both.describe()
