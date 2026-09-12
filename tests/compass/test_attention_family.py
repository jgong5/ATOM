"""The attention family model: what it prices, what it refuses, and the seam.

**These are software unit tests.** The structures are synthetic and the
seconds are invented, so nothing here is evidence that any law holds of the
hardware -- and the scope dicts the fixtures declare are not evidence that any
artifact declares them. What they establish is that the module and the adapter
do what they say: pair queries with histories per request, partition the native
branches, refuse a rank-deficient or unscoped or underdetermined fit, keep an
absent fact from becoming a zero, and carry a modelled price through the
library marked as a prediction with its launch composition intact.

Empirical validation needs source primitive measurements at points nobody has
measured yet, and real integration needs a resolved scope written by the
collector rather than by a fixture. Both are tracked separately.
"""

import json
import os

import pytest

from atom.compass.core.cost.families import attention
from atom.compass.core.cost.families.adapter import ParametricPriceLibrary
from atom.compass.core.cost.families.attention import (CHUNK_SIZE, GDN,
                                                       REQUIRED_SCOPE, UNIFIED,
                                                       Model, Refusal,
                                                       Structure, features_for,
                                                       fit_regime, regime_of,
                                                       structure_of)

#: A resolved unified-attention scope. `attention_backend` names the kernel the
#: dispatcher actually took, which is what the decode split is identified by.
SCOPE = {"kv_cache_dtype": "fp8", "kv_cache_layout": "NHD",
         "kv_cache_block_size": 16, "sliding_window": None,
         "attention_backend": "unified_attention"}

#: A resolved GDN scope. No KV dtype or layout: linear attention reads the conv
#: and recurrent state pool and never the paged cache.
GDN_SCOPE = {"gdn_decode_lossy_fast": False,
             "gdn_state_geometry": [1, 3, 10240, 128]}


def _op(name=UNIFIED, shapes=None, **context):
    return {"name": name, "input_shapes": shapes or [[1, 1]],
            "dtypes": ["bf16"],
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


def _gdn(queries=(), *, decodes=0, tokens=None, allocated=None,
         initial=None, serialized=True, **extra):
    """A linear-attention call as `forward_ctx` serializes one.

    Its captured tensors are written ``[values, dtype_name]`` because they are
    a mix of index tensors and boolean masks and a mask rebuilt as int32
    selects nothing. `serialized=False` writes bare lists, to show the reader
    accepts both.
    """
    def wrap(values, dtype):
        return [list(values), dtype] if serialized else list(values)

    ctx = {}
    total = 0
    if queries:
        starts = [0]
        for q in queries:
            total += q
            starts.append(total)
        ctx["non_spec_query_start_loc"] = wrap(starts, "torch.int32")
    tokens = total if tokens is None else tokens
    ctx.update(num_prefills=len(queries), num_decodes=decodes,
               num_actual_tokens=tokens)
    if initial is not None:
        ctx["has_initial_state"] = wrap(initial, "torch.bool")
    ctx.update(extra)
    rows = tokens if allocated is None else allocated
    shapes = [[rows, 4, 128], [rows, 4], [rows, 4], [rows, 4, 128]]
    return _op(GDN, shapes=shapes, **ctx)


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

    def test_a_mixed_cold_and_cached_batch_pairs_each_member_as_it_is(self):
        """The run4 shape: one request over a cached prefix beside one with
        none. Nothing special-cases it -- the cold member simply has h=0."""
        structure = structure_of(_unified([14592, 1792], [63744, 1792],
                                          is_prefill=True, has_cached=True))
        assert structure.queries == (14592, 1792)
        assert structure.histories == (49152, 0)
        assert structure.paired_work() == (
            14592 * 49152 + 14592 * 14593 // 2 + 1792 * 1793 // 2)


class TestGDNMetadataIsReadAsTheCollectorWritesIt:

    def test_the_serialized_pair_is_parsed(self):
        structure = structure_of(_gdn([64, 128], initial=[False, False]))
        assert structure.queries == (64, 128)
        assert structure.num_actual_tokens == 192

    def test_a_bare_list_is_accepted_too(self):
        structure = structure_of(_gdn([64], initial=[False],
                                      serialized=False))
        assert structure.queries == (64,)

    def test_the_allocated_width_comes_from_the_output_operand(self):
        structure = structure_of(_gdn([64], tokens=64, allocated=128,
                                      initial=[False]))
        assert structure.executed_rows == 128
        assert structure.num_actual_tokens == 64

    def test_the_initial_state_mask_is_counted_not_discarded(self):
        structure = structure_of(_gdn([64, 64], initial=[True, False]))
        assert structure.continued() == 1

    def test_no_mask_is_not_all_fresh(self):
        assert structure_of(_gdn([64])).continued() is None


class TestTheRegimesAreNativeBranches:

    def test_prefill_and_decode_are_different_regimes(self):
        cold = regime_of(_unified([8], [8], is_prefill=True, has_cached=False))
        decode = regime_of(_unified([1], [99], is_prefill=False,
                                    has_cached=False), None, SCOPE)
        assert cold.name == "unified.prefill.cold"
        assert decode.name == "unified.decode.unified_attn"

    def test_the_two_decode_kernels_are_two_regimes(self):
        """`paged_attention_triton` forks: with the flash layout or
        ATOM_USE_UNIFIED_ATTN it calls aiter unified attention, otherwise the
        gluon paged decode. They tile the context differently."""
        op = _unified([1], [99], is_prefill=False, has_cached=False)
        gluon = dict(SCOPE, attention_backend="paged_gluon")
        assert regime_of(op, None, gluon).name == "unified.decode.paged_gluon"
        assert regime_of(op, None, SCOPE).name == "unified.decode.unified_attn"

    def test_an_undeclared_decode_kernel_is_refused(self):
        op = _unified([1], [99], is_prefill=False, has_cached=False)
        out = regime_of(op, None, dict(SCOPE, attention_backend=None))
        assert isinstance(out, Refusal)
        assert "attention_backend" in out.missing

    def test_an_unproven_decode_kernel_refuses_by_name_not_by_alias(self):
        """`paged_attention_persistent_asm` is a separate implementation and
        nothing measured shows it follows the gluon tile law."""
        op = _unified([1], [99], is_prefill=False, has_cached=False)
        out = regime_of(op, None,
                        dict(SCOPE,
                             attention_backend="paged_attention_persistent_asm"))
        assert isinstance(out, Refusal)
        assert "no law for" in out.reason

    def test_cold_and_cached_prefill_are_different_regimes(self):
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
        fresh = regime_of(_gdn([128], initial=[False]))
        continued = regime_of(_op(GDN, num_prefills=0, num_decodes=32,
                                  num_actual_tokens=32))
        assert fresh.name == "gdn.prefill"
        assert continued.name == "gdn.decode"

    def test_a_mixed_gdn_call_is_refused_rather_than_split(self):
        out = regime_of(_op(GDN, num_prefills=1, num_decodes=8,
                            num_actual_tokens=40))
        assert isinstance(out, Refusal) and "mixes" in out.reason

    def test_a_gdn_prefill_without_the_initial_state_mask_is_refused(self):
        """Fresh and continued scans are different work per sequence. Reading
        the mask's absence as all-fresh would be assuming the branch."""
        out = regime_of(_gdn([128]))
        assert isinstance(out, Refusal)
        assert "has_initial_state" in out.missing

    def test_a_replayssm_call_is_refused(self):
        out = regime_of(_gdn([128], initial=[False], replayssm=True))
        assert isinstance(out, Refusal) and "ReplaySSM" in out.reason

    def test_a_speculative_call_is_refused(self):
        out = regime_of(_gdn([128], initial=[False], num_spec_decodes=2,
                             spec_query_start_loc=[[0, 4, 8], "torch.int32"]))
        assert isinstance(out, Refusal) and "speculative" in out.reason

    def test_gdn_decode_has_no_history_term(self):
        """The conv update and recurrence consume a fixed state. A full
        history term here would be a coefficient fitted to noise."""
        features = attention.REGIMES["gdn.decode"].features
        assert "context_rows" not in features
        assert "history_rows" not in features

    def test_gdn_carries_no_kv_cache_scope(self):
        """Linear attention reads the state pool, never the paged KV cache, so
        a KV dtype here would be a scope key that means nothing."""
        required = attention.REGIMES["gdn.decode"].required_scope
        assert "kv_cache_dtype" not in required
        assert "gdn_decode_lossy_fast" in required


class TestAbsentFactsAreNotZeros:

    def test_an_unrecorded_bucket_refuses_rather_than_padding_zero(self):
        regime = attention.REGIMES["unified.decode.unified_attn"]
        structure = structure_of(_unified([1, 1], [50, 50], is_prefill=False,
                                          has_cached=False))
        out = features_for(regime, structure, SCOPE)
        assert isinstance(out, Refusal)
        assert "capture_bucket" in out.missing

    def test_a_recorded_bucket_gives_the_padded_rows(self):
        regime = attention.REGIMES["unified.decode.unified_attn"]
        structure = structure_of(_unified([1, 1], [50, 50], is_prefill=False,
                                          has_cached=False, bucket=8))
        values = features_for(regime, structure, SCOPE)
        assert values[regime.features.index("bucket_pad")] == 6.0

    def test_an_unrecorded_output_width_refuses_rather_than_free_underfill(self):
        """The wrapper slices to num_actual_tokens, and then zeros the output
        rows past it. That tail is work, and it cannot be counted without the
        allocated width."""
        regime = attention.REGIMES["gdn.decode"]
        structure = Structure(num_prefills=0, num_decodes=4,
                              num_actual_tokens=4)
        out = features_for(regime, structure, GDN_SCOPE)
        assert isinstance(out, Refusal) and "output_rows" in out.missing

    def test_the_zeroed_tail_is_counted_when_both_widths_are_recorded(self):
        regime = attention.REGIMES["gdn.decode"]
        structure = structure_of(
            _op(GDN, shapes=[[8, 4], [8, 4], [8, 4], [8, 4, 128]],
                num_prefills=0, num_decodes=5, num_actual_tokens=5))
        values = features_for(regime, structure, GDN_SCOPE)
        assert values[regime.features.index("tail_pad_rows")] == 3.0

    def test_grid_pad_rows_is_what_the_longest_sequence_forces(self):
        regime = attention.REGIMES["unified.decode.unified_attn"]
        structure = structure_of(_unified([1, 1], [16, 1000],
                                          is_prefill=False, has_cached=False,
                                          bucket=2))
        values = features_for(regime, structure, SCOPE)
        assert values[regime.features.index("grid_pad_rows")] == 1000 - 16


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
            structure = structure_of(_gdn([q], initial=[False]))
            points.append((structure, seconds, "src", dict(GDN_SCOPE)))
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
        k = 2.25e-10
        fit = fit_regime(regime, self._points(
            [(q, 0, k * (q * (q + 1) // 2)) for q in (64, 128, 256)],
            scope=None), strict=False)
        assert not isinstance(fit, Refusal)
        assert set(fit.scope_undeclared) == set(REQUIRED_SCOPE)
        assert "scope undeclared" in fit.describe()

    def test_a_column_zero_everywhere_is_pinned_not_fitted(self):
        """Every GDN decode measured so far ran a full bucket, so the padding
        term is zero at every point. Requiring it would make every fit rank
        deficient; pinning it states the subdomain the law covers."""
        regime = attention.REGIMES["gdn.decode"]
        points = []
        for decodes, seconds in ((8, 1e-5), (16, 1.2e-5), (32, 1.6e-5),
                                 (64, 2.4e-5)):
            structure = structure_of(
                _op(GDN,
                    shapes=[[decodes, 4], [decodes, 4], [decodes, 4],
                            [decodes, 4, 128]],
                    num_prefills=0, num_decodes=decodes,
                    num_actual_tokens=decodes))
            points.append((structure, seconds, "src", dict(GDN_SCOPE)))
        fit = fit_regime(regime, points)
        assert not isinstance(fit, Refusal)
        assert "tail_pad_rows" in fit.pinned
        assert "measured only where tail_pad_rows is zero" in fit.describe()


class TestPricingAnUnseenStructure:

    def _model(self, strict=True):
        """A cold-prefill law over three points, on a known coefficient."""
        k = 2.25e-10
        points = []
        for q in (64, 4672, 16384):
            op = _unified([q], [q], is_prefill=True, has_cached=False)
            points.append((op, k * structure_of(op).paired_work(), "src",
                           dict(SCOPE)))
        # `from_priced`, which is the adapter's own path: it folds each call's
        # static operand geometry into the scope, so the law is identified by
        # the heads it was measured on as well as by the deployment.
        return Model.from_priced(points, strict=strict)

    def test_an_unmeasured_query_length_is_priced(self):
        """The point of the exercise: 641 was never measured and is priced
        from the law rather than from a table."""
        model = self._model()
        seconds = model.price(_unified([641], [641], is_prefill=True,
                                       has_cached=False), SCOPE)
        assert not isinstance(seconds, Refusal)
        assert seconds == pytest.approx(2.25e-10 * (641 * 642 // 2), rel=0.05)

    def test_a_structure_outside_the_measured_range_is_refused(self):
        model = self._model()
        out = model.price(_unified([65536], [65536], is_prefill=True,
                                   has_cached=False), SCOPE)
        assert isinstance(out, Refusal)
        assert "outside the measured range" in out.reason

    def test_a_law_does_not_answer_for_another_deployment(self):
        model = self._model()
        out = model.price(_unified([641], [641], is_prefill=True,
                                   has_cached=False),
                          dict(SCOPE, kv_cache_dtype="bf16"))
        assert isinstance(out, Refusal) and "kv_cache_dtype" in out.reason

    def test_an_undeclared_request_does_not_match_a_scoped_law(self):
        model = self._model()
        out = model.price(_unified([641], [641], is_prefill=True,
                                   has_cached=False), None)
        assert isinstance(out, Refusal)
        assert "not for this deployment" in out.reason

    def test_a_regime_with_no_fit_is_refused_by_name(self):
        model = self._model()
        out = model.price(_unified([1], [99], is_prefill=False,
                                   has_cached=False, bucket=8), SCOPE)
        assert isinstance(out, Refusal)
        assert "unified.decode.unified_attn" in out.reason

    def test_a_pinned_column_refuses_rather_than_pricing_it_free(self):
        """The law was measured where the padding tail was always zero. A call
        that has one is unsupported, not free."""
        points = []
        for decodes, seconds in ((8, 1e-5), (16, 1.2e-5), (32, 1.6e-5),
                                 (64, 2.4e-5)):
            op = _op(GDN,
                     shapes=[[decodes, 4], [decodes, 4], [decodes, 4],
                             [decodes, 4, 128]],
                     num_prefills=0, num_decodes=decodes,
                     num_actual_tokens=decodes)
            points.append((op, seconds, "src", dict(GDN_SCOPE)))
        # `from_priced`, the adapter's own path: it folds each call's static
        # operand geometry into the scope, so the allocated width travels with
        # the law rather than being rediscovered at prediction.
        model = Model.from_priced(points)
        underfilled = _op(GDN,
                          shapes=[[32, 4], [32, 4], [32, 4], [32, 4, 128]],
                          num_prefills=0, num_decodes=20,
                          num_actual_tokens=20)
        out = model.price(underfilled, GDN_SCOPE)
        assert isinstance(out, Refusal)
        assert "unsupported rather than free" in out.reason
        assert "tail_pad_rows" in out.missing

    def test_a_continued_gdn_prefill_is_unsupported_on_all_fresh_evidence(self):
        """Current GDN sources are all-fresh single sequences. A batch that
        resumes a held state is a named gap, not a price."""
        k = 1e-9
        points = []
        # Not all multiples of CHUNK_SIZE: if they were, `chunks` would be
        # exactly `query_rows / 64` and the law would be rank deficient by
        # construction rather than by anything about the evidence.
        for q in (96, 100, 300, 1000, 4680):
            op = _gdn([q], initial=[False])
            points.append((op, k * q, "src", dict(GDN_SCOPE)))
        model = Model.from_priced(points)
        asked = _gdn([256], initial=[True])
        # Asked with the op, so the question carries the same operand geometry
        # `price` selects by. Without it an answerable law reads as an
        # unidentifiable one, and a real scope mismatch would be
        # indistinguishable from a rank-deficient design.
        fit, why = model.fit_for("gdn.prefill", GDN_SCOPE, op=asked)
        assert fit is not None, why.reason
        assert "continued_sequences" in fit.pinned
        out = model.price(asked, GDN_SCOPE)
        assert isinstance(out, Refusal)
        assert "continued_sequences" in out.missing

    def test_coverage_states_both_halves(self):
        model = self._model()
        cover = model.coverage()
        assert any(name.startswith("unified.prefill.cold")
                   for name in cover["fitted"])
        assert cover["strict"] is True

    def test_two_structures_with_equal_totals_price_differently(self):
        """Not a table lookup: the pairing changes the answer even though the
        summed queries and summed histories do not."""
        k = 2.25e-10
        points = []
        for queries, contexts in (([2, 2], [130, 130]),
                                  ([8, 8], [136, 136]),
                                  ([2, 2], [514, 514]),
                                  ([16, 4], [272, 1028])):
            op = _unified(queries, contexts, is_prefill=True,
                          has_cached=True)
            points.append((op, k * structure_of(op).paired_work(), "src",
                           dict(SCOPE)))
        model = Model.from_priced(points)
        balanced = model.price(_unified([6, 6], [390, 390], is_prefill=True,
                                        has_cached=True), SCOPE)
        skewed = model.price(_unified([2, 10], [130, 650], is_prefill=True,
                                      has_cached=True), SCOPE)
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
        k = 2.25e-10
        points = []
        for q, more in zip((64, 128, 256), extra):
            structure = structure_of(_unified([q], [q], is_prefill=True,
                                              has_cached=False))
            scope = dict(SCOPE)
            scope.update(more)
            points.append((structure, k * structure.paired_work(), "src",
                           scope))
        return points

    def test_one_declared_condition_throughout_pools(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        fit = fit_regime(regime,
                         self._points([{"tensor_parallel_size": 1}] * 3))
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
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [{"visited": "single"}, {"visited": "rotated"},
             {"visited": "rotated"}]))
        assert isinstance(out, Refusal) and "visited" in out.reason

    def test_silence_is_not_agreement(self):
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [{"tensor_parallel_size": 1}, {}, {"tensor_parallel_size": 1}]))
        assert isinstance(out, Refusal)
        assert "tensor_parallel_size" in out.reason

    def test_a_declared_none_is_a_statement_not_an_absence(self):
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


# -- the adapter seam -----------------------------------------------------


def _layer_copy(op, layer):
    """The op as one bound layer records it.

    A call in a real graph carries its layer identity, so a price is keyed to
    the layer that made it. Tests that want the exact record back have to ask
    with the same identity the library was given.
    """
    copy = dict(op)
    copy["context"] = list(op["context"]) + [["layer_name", layer]]
    return copy


def _library(tmp_path, designs, *, scope=None, kernels=("k0", "k1"),
             layers=16, graph_extra=None, name="p"):
    """A library holding one attention price per design, replicated by layer.

    The price files are synthetic, so this establishes what the adapter does
    with artifacts -- not that any real artifact declares a resolved scope.
    """
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary()
    for index, (op, seconds) in enumerate(designs):
        ops, prices = [], {}
        for layer in range(layers):
            copy = _layer_copy(op, layer)
            ops.append(copy)
            prices[signature_of(copy)] = {
                "seconds": seconds, "kernels": {k: seconds / len(kernels)
                                                for k in kernels},
                "occurrences": 1, "name": op["name"],
                "signature": signature_of(copy),
            }
        graph = {"ops": ops, "provenance": dict(graph_extra or {})}
        blob = {"prices": prices,
                "provenance": {"attention_scope": dict(scope or SCOPE)}}
        gpath = tmp_path / f"g{name}{index}.json"
        ppath = tmp_path / f"{name}{index}.json"
        gpath.write_text(json.dumps(graph))
        ppath.write_text(json.dumps(blob))
        library.add(str(ppath), str(gpath))
    return library


def _cold_designs():
    k = 2.25e-10
    designs = []
    for q in (64, 4672, 16384):
        op = _unified([q], [q], is_prefill=True, has_cached=False)
        designs.append((op, k * structure_of(op).paired_work()))
    return designs


def test_a_standalone_attention_graph_is_collected_though_it_has_no_body(
        tmp_path):
    """A primitive graph has no embedding and no `body_rows_traced`, so the
    row-family check refuses it -- correctly, and irrelevantly. The attention
    observations are taken before that refusal, not after it."""
    library = _library(tmp_path, _cold_designs())
    assert library.unbuildable  # the row curve cannot use these files
    assert len(library._attention_obs) == 3 * 16
    assert len(library.attention_design_points()) == 3


def test_layer_copies_collapse_to_one_design_point_each(tmp_path):
    """16 layers of one step are 16 measurements of one point. Counting them
    as points would report a law as checked that nothing checked."""
    library = _library(tmp_path, _cold_designs(), layers=48)
    points = library.attention_design_points()
    assert len(points) == 3
    assert all("47 replicates" in note or "replicates" in note
               for _op, _s, note, _scope in points)


def test_an_unseen_structure_is_priced_through_the_library(tmp_path):
    """The working seam: a query length nobody measured, answered by the law
    and returned as a prediction rather than as a measurement."""
    library = _library(tmp_path, _cold_designs())
    library.request_attention_scope = dict(SCOPE)
    op = _unified([641], [641], is_prefill=True, has_cached=False)

    record, source = library.lookup(op)

    assert record is not None
    assert record["seconds"] == pytest.approx(2.25e-10 * (641 * 642 // 2),
                                              rel=0.05)
    assert record["interpolated"] is True
    assert source.startswith("interpolated://")
    assert "unified.prefill.cold" in source


def test_a_modelled_price_carries_the_evidenced_launch_composition(tmp_path):
    """`body` counts `max(1, len(record["kernels"]))` launches. An empty map
    would silently make a two-kernel wrapper one launch."""
    library = _library(tmp_path, _cold_designs(), kernels=("k0", "k1"))
    library.request_attention_scope = dict(SCOPE)
    record, _source = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert sorted(record["kernels"]) == ["k0", "k1"]
    # Names, and no fabricated attribution.
    assert all(value is None for value in record["kernels"].values())
    assert "unknown" in record["kernel_attribution"]


def test_a_price_whose_launch_composition_is_unevidenced_is_refused(tmp_path):
    library = _library(tmp_path, _cold_designs(), kernels=())
    library.request_attention_scope = dict(SCOPE)
    record, detail = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert record is None
    assert "launch" in detail


def test_an_exact_measurement_is_never_displaced_by_a_law(tmp_path):
    designs = _cold_designs()
    library = _library(tmp_path, designs)
    library.request_attention_scope = dict(SCOPE)
    measured, source = library.lookup(_layer_copy(designs[0][0], 0))
    assert measured is not None
    assert not source.startswith("interpolated://")
    assert measured.get("interpolated") is not True


def test_a_request_that_declares_no_scope_gets_no_modelled_price(tmp_path):
    """A law measured under one deployment does not answer for a request that
    has not said which deployment it is."""
    library = _library(tmp_path, _cold_designs())
    record, detail = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert record is None
    assert "deployment" in detail


def test_a_config_request_is_not_read_as_a_resolved_fact(tmp_path):
    """`kv_cache_dtype: auto` in a config is what was asked for. Reading it as
    the dtype the kernel ran on would attribute a law to a kernel that may
    never have run."""
    library = ParametricPriceLibrary()
    from atom.compass.core.cost.families.adapter import _attention_scope

    scope = _attention_scope(
        {"provenance": {"config": {"kv_cache_dtype": "auto",
                                   "attention_backend": "auto"}}}, None)
    assert "kv_cache_dtype" not in scope
    assert scope["requested.kv_cache_dtype"] == "auto"
    assert library is not None


def test_conditions_of_the_measurement_are_carried_not_dropped(tmp_path):
    """Two files that differ in how they were measured do not pool, and they
    can only refuse if the condition was carried in the first place."""
    from atom.compass.core.cost.families.adapter import _attention_scope

    scope = _attention_scope(
        {"provenance": {"attention_scope": dict(SCOPE), "repeats": 3,
                        "warmup": 1, "visited": "rotated",
                        "tensor_parallel_size": 1}}, None)
    assert scope["repeats"] == 3
    assert scope["visited"] == "rotated"
    assert scope["tensor_parallel_size"] == 1


def test_two_measurement_conditions_are_two_design_points(tmp_path):
    """A record measured cold and one measured warm are not replicates, even
    at the same structure and the same layer count."""
    library = ParametricPriceLibrary()
    from atom.compass.runtime.microbench import signature_of

    op = _unified([64], [64], is_prefill=True, has_cached=False)
    for index, cache in enumerate(("cold", "warm")):
        prices = {signature_of(op): {
            "seconds": 1e-5 * (index + 1), "kernels": {"k0": 1e-5},
            "occurrences": 1, "name": op["name"], "cache": cache,
            "signature": signature_of(op)}}
        gpath = tmp_path / f"gc{index}.json"
        ppath = tmp_path / f"pc{index}.json"
        gpath.write_text(json.dumps({"ops": [op], "provenance": {}}))
        ppath.write_text(json.dumps(
            {"prices": prices,
             "provenance": {"attention_scope": dict(SCOPE)}}))
        library.add(str(ppath), str(gpath))
    assert len(library.attention_design_points()) == 2




def _file(tmp_path, name, ops, prices, scope=None):
    """One (graph, price) pair on disk, as the collector lays them out."""
    gpath = tmp_path / f"g{name}.json"
    ppath = tmp_path / f"p{name}.json"
    gpath.write_text(json.dumps({"ops": ops, "provenance": {}}))
    ppath.write_text(json.dumps(
        {"prices": prices,
         "provenance": {"attention_scope": dict(scope or SCOPE)}}))
    return str(ppath), str(gpath)


def test_two_treatments_do_not_rejoin_as_points_of_one_law(tmp_path):
    """Separating cold from warm only at collapse time is not enough.

    Kept apart as design points but pooled into one fit, they become two
    independent points of the same law -- the same averaging, one step later.
    The treatment has to travel into the fit's scope.
    """
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary()
    for cache, factor in (("cold", 2.0), ("warm", 1.0)):
        for op, seconds in _cold_designs():
            prices = {signature_of(op): {
                "seconds": seconds * factor, "kernels": {"k0": seconds},
                "occurrences": 1, "name": op["name"], "cache": cache,
                "signature": signature_of(op)}}
            library.add(*_file(tmp_path, f"{cache}{op['input_shapes'][0][0]}",
                               [op], prices))
    treatments = {scope["measurement_treatment"]
                  for _op, _s, _note, scope in library.attention_design_points()}
    assert len(treatments) == 2
    model = library.attention_model()
    cold = [label for label in model.fits if "unified.prefill.cold" in label]
    assert len(cold) == 2, model.refusals

    # And a request that does not say which treatment it wants is refused,
    # rather than answered by whichever law was fitted first.
    library.request_attention_scope = dict(SCOPE)
    record, detail = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert record is None
    assert "measurement_treatment" in detail


def test_host_time_noise_does_not_split_a_design_point(tmp_path):
    """Host seconds is an outcome, not a treatment. Keying on it would make
    every repeat of one design its own point and report the design as checked
    many times over."""
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary()
    op, seconds = _cold_designs()[0]
    ops, prices = [], {}
    for layer in range(8):
        copy = _layer_copy(op, layer)
        ops.append(copy)
        record = {"seconds": seconds, "kernels": {"k0": seconds},
                  "occurrences": 1, "name": op["name"],
                  "signature": signature_of(copy), "cache": "warm"}
        if layer % 3:  # some records do not say, which is not a difference
            record["host_seconds"] = 0.012 + layer * 1e-4
        prices[signature_of(copy)] = record
    library.add(*_file(tmp_path, "host", ops, prices))
    points = library.attention_design_points()
    assert len(points) == 1
    note = points[0][2]
    assert "8 replicates" in note
    assert "host time" in note and "absent on 3 of 8" in note


def _real_schema_op(layer, module, name, shapes, context):
    """An operator as a real captured graph records one.

    The layer is a SCALAR under a positional key -- `["#5", "...layers.3..."]`
    -- and not a named `layer_name` context entry. That is what the actual
    run5 payload carries, and it is why the layer cannot be recognised by key.
    """
    return {
        "name": name,
        "input_shapes": shapes,
        "dtypes": ["bfloat16"] * len(shapes),
        "layouts": [],
        "int_values": [],
        "scalars": [["#5", f"language_model.model.layers.{layer}.{module}"]],
        "context": context,
    }


def test_real_graph_layer_copies_collapse_by_module_path(tmp_path):
    """64 bound modules of one step are not 64 design points.

    The schema here is the actual one: shapes, dtypes and the module path as a
    positional scalar, taken from the run5 refusal payload. Without the layer
    index being normalised out, this reads as 64 independent measurements of
    64 different laws.
    """
    from atom.compass.runtime.microbench import signature_of

    gdn_context = [["num_prefills", 2], ["num_decodes", 0],
                   ["num_actual_tokens", 16384],
                   ["non_spec_query_start_loc", [[0, 14592, 16384], "int32"]],
                   ["has_initial_state", [[1, 0], "bool"]]]
    mha_context = [["cu_seqlens_q", [0, 14592, 16384]],
                   ["context_lens", [63744, 1792]],
                   ["is_prefill", True], ["has_cached", True],
                   ["state", "prefill_prefix"]]
    ops, prices = [], {}
    for layer in range(64):
        if layer % 4 == 3:
            op = _real_schema_op(layer, "self_attn", UNIFIED,
                                 [[16384, 6144], [16384, 1024], [16384, 1024]],
                                 mha_context)
        else:
            op = _real_schema_op(layer, "linear_attn", GDN,
                                 [[16384, 10240], [16384, 48], [16384, 48],
                                  [16384, 48, 128]], gdn_context)
        ops.append(op)
        prices[signature_of(op)] = {
            "seconds": 1e-4, "kernels": {"k0": 1e-4}, "occurrences": 1,
            "name": op["name"], "signature": signature_of(op)}
    library = ParametricPriceLibrary()
    library.add(*_file(tmp_path, "real", ops, prices))
    assert len(library._attention_obs) == 64
    points = library.attention_design_points()
    # Two: one GDN design and one unified design. The module KIND still
    # separates them -- only the index is normalised away.
    assert len(points) == 2
    names = sorted(point[0]["name"] for point in points)
    assert names == [GDN, UNIFIED]


REAL_REFUSAL = ("/workspace/ATOM/agent_scratch/wt_r5/agent_scratch/devreplay"
                "/refusals/refusal_b2_t16384_1.json")


@pytest.mark.skipif(not os.path.exists(REAL_REFUSAL),
                    reason="the actual run5 refusal payload is not here")
def test_the_actual_payload_reads_as_the_regimes_it_is():
    """Read against the real artifact, not a fixture shaped like one.

    Development payload: its timing is held out and nothing here prices it.
    What is checked is the reading -- the initial-state bits, the paired
    structure, and which native branch each of the 64 bound modules is in.
    """
    with open(REAL_REFUSAL) as handle:
        payload = json.load(handle)
    ops = [op for op in payload["body_graph"]["ops"]
           if op.get("name") in (GDN, UNIFIED)]
    assert len(ops) == 64

    gdn = [op for op in ops if op["name"] == GDN]
    mha = [op for op in ops if op["name"] == UNIFIED]
    assert len(gdn) == 48 and len(mha) == 16

    gdn_structure = structure_of(gdn[0])
    assert list(gdn_structure.queries) == [14592, 1792]
    # The actual bits are [True, False]: the long sequence resumes a held
    # state and the short one is fresh. Reading them as all-fresh would be
    # assuming the branch.
    assert gdn_structure.continued() == 1
    assert regime_of(gdn[0], gdn_structure).name == "gdn.prefill"

    mha_structure = structure_of(mha[0])
    assert list(mha_structure.queries) == [14592, 1792]
    assert list(mha_structure.histories) == [49152, 0]
    assert regime_of(mha[0], mha_structure).name == "unified.prefill.cached"


class TestTheKVByteScaleIsStatedAtTheRightOrder:
    """Not a feature -- per-layer bytes are a constant times `history_rows`,
    so a byte column would be exactly collinear with the row column and the fit
    would refuse it. It is here because the wrong value has been written down
    before, from dividing the pool across all 64 bound modules."""

    def test_one_history_row_costs_four_kib_of_kv_per_mha_layer(self):
        assert attention.KV_BYTES_PER_HISTORY_ROW_PER_MHA_LAYER == 4096

    def test_the_pool_records_a_block_of_sixteen_rows(self):
        assert 16 * attention.KV_BYTES_PER_HISTORY_ROW_PER_MHA_LAYER == 64 * 1024

    def test_only_the_mha_layers_hold_kv(self):
        assert attention.MHA_LAYERS == 16
        assert attention.MHA_LAYERS != 64


class TestTheFactoryCarriesTheDeclaredDeployment:
    """The scope reaches the library through the factory, not through a test.

    Setting `request_attention_scope` by hand establishes what the library
    does with a scope; it establishes nothing about how a run gets one. These
    go through `_price_library`, which is the call `build_source_oracle`
    makes, so what is checked is the path a run actually takes.
    """

    def _files(self, tmp_path):
        from atom.compass.runtime.microbench import signature_of

        entries = []
        for index, (op, seconds) in enumerate(_cold_designs()):
            prices = {signature_of(op): {
                "seconds": seconds, "kernels": {"k0": seconds / 2,
                                                "k1": seconds / 2},
                "occurrences": 1, "name": op["name"],
                "signature": signature_of(op)}}
            # Named by index: these fixtures share their operand shapes, so a
            # name taken from the shape would write all three designs to one
            # file and leave one design point behind.
            entries.append(_file(tmp_path, "f%d" % index, [op], prices))
        return [(price, graph) for price, graph in entries]

    def test_a_scope_file_reaches_the_library_and_prices_an_unseen_call(
            self, tmp_path):
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        scope_path = tmp_path / "attn_scope.json"
        scope_path.write_text(json.dumps({"attention_scope": dict(SCOPE)}))
        library = _price_library(self._files(tmp_path), gap_ratio(True), None,
                                 str(scope_path))
        assert library.request_attention_scope["kv_cache_dtype"] == "fp8"

        record, source = library.lookup(
            _unified([641], [641], is_prefill=True, has_cached=False))
        assert record is not None
        assert record["interpolated"] is True
        assert source.startswith("interpolated://")
        assert sorted(record["kernels"]) == ["k0", "k1"]

    def test_a_mapping_is_accepted_as_well_as_a_file(self, tmp_path):
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        library = _price_library(self._files(tmp_path), gap_ratio(True), None,
                                 dict(SCOPE))
        assert library.request_attention_scope["attention_backend"] == \
            "unified_attention"

    def test_a_scope_with_modelling_off_is_an_error_not_a_silent_drop(
            self, tmp_path):
        from atom.compass.runtime.source_oracle import _price_library

        with pytest.raises(ValueError, match="modelling is off"):
            _price_library(self._files(tmp_path), None, None, dict(SCOPE))

    def test_a_missing_scope_file_is_refused(self, tmp_path):
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        with pytest.raises(ValueError, match="neither a mapping nor a file"):
            _price_library(self._files(tmp_path), gap_ratio(True), None,
                           str(tmp_path / "nothing.json"))

    def test_a_resolution_dump_is_not_a_scope(self, tmp_path):
        """A file can record the whole environment and still not say which of
        it the kernel turns on. Reading one as a scope is a judgement, and it
        has to be written down as one."""
        from atom.compass.runtime.source_oracle import _attention_request_scope

        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps({"atom_envs": {"ATOM_V4_BACKEND": "legacy"},
                                    "caches_summary": {"kv_heads": 4}}))
        with pytest.raises(ValueError, match="resolution dump"):
            _attention_request_scope(str(dump))


PRICING_SCOPE = ("/workspace/ATOM/agent_scratch/pricing_coverage/p_caseb"
                 "/ATTN_SCOPE.json")


@pytest.mark.skipif(not os.path.exists(PRICING_SCOPE),
                    reason="the resolved scope file is not here")
def test_the_resolved_scope_file_is_still_a_dump_not_a_scope():
    """Stated as a test so the gap is visible rather than assumed closed.

    What was relayed resolves the deployment -- 48 GDN and 16 MHA layers, BF16,
    heads 24/4, head_dim 256, block 16, window disabled -- but records it as an
    environment dump under its own keys. Until the declared keys are written
    into it (or a mapping standing behind that reading is passed), a run
    pointed at this file is refused, and that refusal is the honest answer.
    """
    from atom.compass.runtime.source_oracle import _attention_request_scope

    with pytest.raises(ValueError, match="resolution dump"):
        _attention_request_scope(PRICING_SCOPE)
