"""The attention family model: what it prices, and what it refuses.

**These are software unit tests.** The structures are synthetic and the
seconds are invented, so nothing here is evidence that any law holds of the
hardware. What they establish is that the module does what it says: pairs
queries with histories per request, partitions the native branches, refuses a
rank-deficient or unscoped or underdetermined fit, and never turns an absent
fact into a zero.

Empirical validation needs source primitive measurements at points nobody has
measured yet, and is tracked separately.
"""

import pytest

from atom.compass.core.cost.families import attention
from atom.compass.core.cost.families.attention import (CHUNK_SIZE, GDN,
                                                       REQUIRED_SCOPE, UNIFIED,
                                                       Model, Refusal,
                                                       Structure, features_for,
                                                       fit_regime, regime_of,
                                                       structure_of)

SCOPE = {"kv_cache_dtype": "fp8", "kv_cache_layout": "NHD",
         "sliding_window": None, "attention_backend": "aiter-mha"}


def _op(name=UNIFIED, **context):
    return {"name": name, "input_shapes": [[1, 1]],
            "context": [[k, v] for k, v in context.items()]}


def _unified(queries, contexts, *, is_prefill, has_cached, bucket=None):
    cu, total = [0], 0
    for q in queries:
        total += q
        cu.append(total)
    ctx = {"cu_seqlens_q": cu, "context_lens": list(contexts),
           "is_prefill": is_prefill, "has_cached": has_cached}
    if bucket is not None:
        ctx["capture_bucket"] = bucket
    return _op(**ctx)


class TestTheStructureIsPairedPerRequest:

    def test_queries_come_from_the_offsets(self):
        op = _unified([3, 5], [10, 20], is_prefill=True, has_cached=True)
        assert structure_of(op).queries == (3, 5)

    def test_history_is_context_minus_this_requests_own_query(self):
        op = _unified([3, 5], [10, 20], is_prefill=True, has_cached=True)
        assert structure_of(op).histories == (7, 15)

    def test_paired_work_is_the_sum_of_products_not_the_product_of_sums(self):
        """The distinction the whole model rests on. One long query over a
        short history and one short query over a long history do not cost what
        their totals suggest."""
        # The same queries and the same histories, paired the other way round:
        # q=(4,1) against h=(100,10), versus q=(1,4) against the same two
        # histories. Totals identical, work very different.
        paired = structure_of(
            _unified([4, 1], [104, 11], is_prefill=True, has_cached=True))
        swapped = structure_of(
            _unified([1, 4], [101, 14], is_prefill=True, has_cached=True))

        assert paired.query_total == swapped.query_total
        assert sorted(paired.histories) == sorted(swapped.histories)
        assert paired.history_total == swapped.history_total
        assert paired.paired_work() == 4 * 100 + 10 + 1 * 10 + 1
        assert swapped.paired_work() == 1 * 100 + 1 + 4 * 10 + 10
        assert paired.paired_work() != swapped.paired_work()

    def test_the_causal_triangle_is_counted_among_the_new_rows(self):
        one = structure_of(_unified([4], [4], is_prefill=True,
                                    has_cached=False))
        assert one.paired_work() == 4 * 5 // 2

    def test_chunks_are_per_sequence_not_of_the_total(self):
        """Two sequences of 65 are four chunks, not three: the scan cannot
        carry a tail from one sequence into the next."""
        two = structure_of(_unified([65, 65], [65, 65], is_prefill=True,
                                    has_cached=False))
        assert two.chunks() == 4
        assert -(-130 // CHUNK_SIZE) == 3


class TestTheRegimesAreNativeBranches:

    def test_prefill_and_decode_are_different_regimes(self):
        cold = regime_of(_unified([8], [8], is_prefill=True, has_cached=False))
        decode = regime_of(_unified([1], [99], is_prefill=False,
                                    has_cached=False))
        assert cold.name == "unified.prefill.cold"
        assert decode.name == "unified.decode"

    def test_cold_and_cached_prefill_are_different_regimes(self):
        """They read different KV, so they are different functions of the
        same structure rather than one function with a parameter."""
        cached = regime_of(_unified([8], [108], is_prefill=True,
                                    has_cached=True))
        assert cached.name == "unified.prefill.cached"

    def test_an_unstated_prefill_flag_is_refused(self):
        out = regime_of(_unified([8], [8], is_prefill=None, has_cached=False))
        assert isinstance(out, Refusal) and "prefill" in out.reason

    def test_an_unstated_cache_flag_is_refused(self):
        out = regime_of(_unified([8], [8], is_prefill=True, has_cached=None))
        assert isinstance(out, Refusal) and "cached prefix" in out.reason

    def test_gdn_splits_on_the_prefill_decode_counts(self):
        fresh = regime_of(_op(GDN, num_prefills=1, num_decodes=0,
                              num_actual_tokens=128))
        continued = regime_of(_op(GDN, num_prefills=0, num_decodes=32,
                                  num_actual_tokens=32))
        assert fresh.name == "gdn.prefill"
        assert continued.name == "gdn.decode"

    def test_a_mixed_gdn_call_is_refused_rather_than_split(self):
        out = regime_of(_op(GDN, num_prefills=1, num_decodes=8,
                            num_actual_tokens=40))
        assert isinstance(out, Refusal) and "mixes" in out.reason

    def test_gdn_decode_has_no_history_term(self):
        """The conv update and recurrence consume a fixed state. A full
        history term here would be a coefficient fitted to noise."""
        features = attention.REGIMES["gdn.decode"].features
        assert "context_rows" not in features
        assert "history_rows" not in features


class TestAbsentFactsAreNotZeros:

    def test_an_unrecorded_bucket_refuses_rather_than_padding_zero(self):
        regime = attention.REGIMES["unified.decode"]
        structure = structure_of(_unified([1, 1], [50, 50], is_prefill=False,
                                          has_cached=False))
        out = features_for(regime, structure)
        assert isinstance(out, Refusal)
        assert "capture_bucket" in out.missing

    def test_a_recorded_bucket_gives_the_padded_rows(self):
        regime = attention.REGIMES["unified.decode"]
        structure = structure_of(_unified([1, 1], [50, 50], is_prefill=False,
                                          has_cached=False, bucket=8))
        assert features_for(regime, structure)[2] == 6.0


class TestAFitRefusesWhatItCannotIdentify:

    def _points(self, pairs, scope=SCOPE):
        return [(structure_of(_unified([q], [q + h], is_prefill=True,
                                       has_cached=False)), seconds, "src",
                 dict(scope) if scope else {})
                for q, h, seconds in pairs]

    def test_too_few_points_for_the_terms_is_refused(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points([(64, 0, 1e-4),
                                               (128, 0, 2e-4)]))
        assert isinstance(out, Refusal) and "left over to check" in out.reason

    def test_a_rank_deficient_design_is_refused_by_name(self):
        """Perfectly collinear features cannot be told apart, and the refusal
        says to measure a point that separates them."""
        regime = attention.REGIMES["gdn.prefill"]
        points = []
        for q, seconds in ((64, 1e-4), (128, 2e-4), (256, 4e-4), (512, 8e-4)):
            points.append((structure_of(
                _unified([q], [q], is_prefill=True, has_cached=False)),
                seconds, "src", dict(SCOPE)))
        out = fit_regime(regime, points)
        assert isinstance(out, Refusal)
        assert "rank deficient" in out.reason and "separates them" in out.reason

    def test_a_negative_coefficient_is_refused_as_not_a_cost(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [(64, 0, 1e-3), (128, 0, 5e-4), (256, 0, 1e-4)]))
        assert isinstance(out, Refusal) and "not a cost" in out.reason

    def test_undeclared_scope_is_refused_under_strict(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [(64, 0, 1e-4), (128, 0, 2e-4), (256, 0, 4e-4)], scope=None))
        assert isinstance(out, Refusal)
        assert set(out.missing) == set(REQUIRED_SCOPE)

    def test_disagreeing_scope_is_refused_even_without_strict(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        points = self._points([(64, 0, 1e-4), (128, 0, 2e-4), (256, 0, 4e-4)])
        points[1][3]["kv_cache_dtype"] = "bf16"
        out = fit_regime(regime, points, strict=False)
        assert isinstance(out, Refusal) and "different kernels" in out.reason

    def test_undeclared_scope_is_allowed_for_diagnosis_and_stamped(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        # Seconds generated from a positive law, so the fit is refused for the
        # scope and for nothing else.
        k = 2.25e-10
        fit = fit_regime(regime, self._points(
            [(q, 0, k * (q * (q + 1) // 2)) for q in (64, 128, 256)],
            scope=None), strict=False)
        assert not isinstance(fit, Refusal)
        assert set(fit.scope_undeclared) == set(REQUIRED_SCOPE)
        assert "scope undeclared" in fit.describe()


class TestPricingAnUnseenStructure:

    def _model(self, strict=True):
        """A cold-prefill law over three points, on a known coefficient."""
        k = 2.25e-10
        points = []
        for q in (64, 4672, 16384):
            structure = structure_of(_unified([q], [q], is_prefill=True,
                                              has_cached=False))
            points.append((structure, k * structure.paired_work(), "src",
                           dict(SCOPE)))
        return Model.from_observations({"unified.prefill.cold": points},
                                       strict=strict)

    def test_an_unmeasured_query_length_is_priced(self):
        """The point of the exercise: 641 was never measured and is priced
        from the law rather than from a table."""
        model = self._model()
        seconds = model.price(_unified([641], [641], is_prefill=True,
                                       has_cached=False))
        assert not isinstance(seconds, Refusal)
        assert seconds == pytest.approx(2.25e-10 * (641 * 642 // 2), rel=0.05)

    def test_a_structure_outside_the_measured_range_is_refused(self):
        model = self._model()
        out = model.price(_unified([65536], [65536], is_prefill=True,
                                   has_cached=False))
        assert isinstance(out, Refusal) and "outside the measured range" in out.reason

    def test_a_regime_with_no_fit_is_refused_by_name(self):
        model = self._model()
        out = model.price(_unified([1], [99], is_prefill=False,
                                   has_cached=False, bucket=8))
        assert isinstance(out, Refusal) and "unified.decode" in out.reason

    def test_coverage_states_both_halves(self):
        model = self._model()
        cover = model.coverage()
        assert "unified.prefill.cold" in cover["fitted"]
        assert cover["strict"] is True

    def test_two_structures_with_equal_totals_price_differently(self):
        """Not a table lookup: the pairing changes the answer even though the
        summed queries and summed histories do not."""
        k = 2.25e-10
        points = []
        # Queries and histories vary independently: holding h/q fixed across
        # the design makes query_rows and history_rows proportional, and the
        # fit refuses that as rank deficient -- correctly.
        for queries, contexts in (([2, 2], [130, 130]),
                                  ([8, 8], [136, 136]),
                                  ([2, 2], [514, 514]),
                                  ([16, 4], [272, 1028])):
            structure = structure_of(_unified(queries, contexts,
                                              is_prefill=True,
                                              has_cached=True))
            points.append((structure, k * structure.paired_work(), "src",
                           dict(SCOPE)))
        model = Model.from_observations({"unified.prefill.cached": points})
        balanced = model.price(_unified([6, 6], [390, 390], is_prefill=True,
                                        has_cached=True))
        skewed = model.price(_unified([2, 10], [130, 650], is_prefill=True,
                                      has_cached=True))
        assert not isinstance(balanced, Refusal)
        assert not isinstance(skewed, Refusal)
        assert balanced != skewed


class TestObservationsPoolOnlyWhenEverythingDeclaredAgrees:
    """The required scope names the kernel; the rest names the conditions.

    Two captures can agree on dtype, layout, window and backend and still be
    measurements of different things -- a different tensor-parallel geometry, a
    superseded collection, a capture taken before the rotation was corrected.
    Pooling those into one training observation is the failure these close.
    """

    def _points(self, extra):
        """Three cold-prefill points on one positive law, scoped as given."""
        k = 2.25e-10
        points = []
        for q, more in zip((64, 128, 256), extra):
            structure = structure_of(_unified([q], [q], is_prefill=True,
                                              has_cached=False))
            scope = dict(SCOPE)
            scope.update(more)
            points.append((structure, k * structure.paired_work(), "src", scope))
        return points

    def test_one_declared_condition_throughout_pools(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        fit = fit_regime(regime, self._points([{"tensor_parallel_size": 1}] * 3))
        assert not isinstance(fit, Refusal)
        assert fit.scope["tensor_parallel_size"] == 1

    def test_two_tp_geometries_do_not_pool(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [{"tensor_parallel_size": 1}, {"tensor_parallel_size": 4},
             {"tensor_parallel_size": 1}]))
        assert isinstance(out, Refusal)
        assert "tensor_parallel_size" in out.reason
        assert "different deployments" in out.reason

    def test_a_superseded_capture_does_not_pool_with_a_current_one(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [{"superseded": False}, {"superseded": True},
             {"superseded": False}]))
        assert isinstance(out, Refusal) and "superseded" in out.reason

    def test_the_old_single_visited_capture_does_not_pool_with_the_rotation(self):
        """`visited=1` throughout is the uncorrected capture; the corrected one
        rotates. They measure different cache residency for the same shape."""
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [{"visited": "single"}, {"visited": "rotated"},
             {"visited": "rotated"}]))
        assert isinstance(out, Refusal) and "visited" in out.reason

    def test_silence_is_not_agreement(self):
        """One observation carries the key and another does not. The second has
        not said it matches, so they are not shown to be the same condition."""
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [{"tensor_parallel_size": 1}, {}, {"tensor_parallel_size": 1}]))
        assert isinstance(out, Refusal)
        assert "tensor_parallel_size" in out.reason

    def test_a_declared_none_is_a_statement_not_an_absence(self):
        """`sliding_window=None` says there is no window. Every observation
        declares it, so the required scope is satisfied and nothing is missing."""
        regime = attention.REGIMES["unified.prefill.cold"]
        fit = fit_regime(regime, self._points([{}] * 3))
        assert not isinstance(fit, Refusal)
        assert fit.scope_undeclared == ()
        assert fit.scope["sliding_window"] is None

    def test_a_list_valued_condition_compares_by_value(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        fit = fit_regime(regime, self._points([{"pool": [2, 16, 131072]}] * 3))
        assert not isinstance(fit, Refusal)
        out = fit_regime(regime, self._points(
            [{"pool": [2, 16, 131072]}, {"pool": [2, 16, 65536]},
             {"pool": [2, 16, 131072]}]))
        assert isinstance(out, Refusal) and "pool" in out.reason


class TestTheKVByteScaleIsStatedAtTheRightOrder:
    """Not a feature -- per-layer bytes are a constant times `history_rows`,
    so a byte column would be exactly collinear with the row column and the fit
    would refuse it. It is here because the wrong value has been written down
    before, from dividing the pool across all 64 bound modules."""

    def test_one_history_row_costs_four_kib_of_kv_per_mha_layer(self):
        # heads 4 x head_dim 256 x 2 bytes (BF16) x K and V.
        assert attention.KV_BYTES_PER_HISTORY_ROW_PER_MHA_LAYER == 4096

    def test_the_pool_records_a_block_of_sixteen_rows(self):
        assert 16 * attention.KV_BYTES_PER_HISTORY_ROW_PER_MHA_LAYER == 64 * 1024

    def test_only_the_mha_layers_hold_kv(self):
        """64 bound modules, 16 of them MHA. Dividing the pool across all 64
        gives 16 KiB a block, which is the error this guards."""
        assert attention.MHA_LAYERS == 16
        assert attention.MHA_LAYERS != 64
