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
        # Every width a multiple of CHUNK_SIZE, so `chunks` is exactly
        # `query_rows / 64`, and every batch one sequence, so `sequences` is
        # exactly `calls`. Enough of them to clear the point count: the
        # refusal under test is that the design cannot separate the terms, not
        # that it is short of rows.
        for q, seconds in ((64, 1e-4), (128, 2e-4), (256, 4e-4), (512, 8e-4),
                           (1024, 1.6e-3), (2048, 3.2e-3), (4096, 6.4e-3)):
            structure = structure_of(_gdn([q], initial=[False]))
            points.append((structure, seconds, "src", dict(GDN_SCOPE)))
        out = fit_regime(regime, points)
        assert isinstance(out, Refusal)
        assert "rank deficient" in out.reason and "separates them" in out.reason

    def test_a_negative_free_coefficient_bounds_the_term_and_says_so(self):
        """One term grows the cost and another reduces it. A cost cannot be
        negative, so the constrained law charges zero for the second and
        reports that the bound is active there. The sign of the free solve is
        not itself a verdict -- whether this law is usable is decided by its
        error -- but which terms the bound holds at zero is published. This is
        the shape the measured GDN prefill has: at equal query rows, spreading
        them over more sequences measures faster."""
        regime = attention.REGIMES["unified.prefill.cold"]
        points = [(q, 0, 1.0e-9 * (q * (q + 1) // 2) - 1.0e-8 * q)
                  for q in (64, 128, 256, 512, 1024, 2048)]
        fit = fit_regime(regime, self._points(points))
        assert not isinstance(fit, Refusal), getattr(fit, "reason", "")
        assert "query_rows" in fit.bounded
        assert fit.coefficients[fit.features.index("query_rows")] == 0.0
        described = fit.describe()
        assert "query_rows" in described and "held at zero" in described
        # The free solve wanting a negative is the reason the bound is
        # active, and the report says that rather than guessing at a cause.
        assert "wanted a negative cost" in described

    def test_a_law_no_better_than_a_flat_charge_is_refused(self):
        """The same falling seconds at three points: too few and too scattered
        for the design to determine the sign, so the nonnegative fit returns
        something. It explains nothing, and saying so is the gate -- not the
        sign of a coefficient."""
        regime = attention.REGIMES["unified.prefill.cold"]
        out = fit_regime(regime, self._points(
            [(64, 0, 1e-3), (128, 0, 5e-4), (256, 0, 1e-4)]))
        assert isinstance(out, Refusal)
        assert "the same number" in out.reason

    def test_an_undetermined_negative_is_bounded_rather_than_refused(self):
        """Two columns that move together leave the split between them open,
        and the noise puts one side of it below zero. The law inside the
        region a cost can live in is the honest one -- and it says which term
        it is holding at zero."""
        # Two columns that move together, and a right-hand side whose free
        # solution puts the second below zero. Solved directly, so what is
        # under test is the constraint and not the noise of some fixture.
        matrix = [[1.0, 1.02], [2.0, 2.01], [3.0, 3.03], [4.0, 3.98],
                  [5.0, 5.01], [6.0, 6.02]]
        rhs = [1.0, 2.05, 2.9, 4.1, 4.95, 6.0]
        free = attention._solve(matrix, rhs)
        assert min(free) < 0  # the design does not determine the split
        solved, at_zero = attention._solve_nonnegative(matrix, rhs)
        assert all(c >= 0 for c in solved)
        assert at_zero and all(solved[i] == 0.0 for i in at_zero)
        # What the free column carries is the work both columns describe, not
        # half of it: the total charged at each point still tracks the data.
        for row, y in zip(matrix, rhs):
            assert abs(sum(c * v for c, v in zip(solved, row)) - y) < 0.2

    def test_the_constrained_solve_finds_the_feasible_optimum(self):
        """A greedy drop-the-negative-column solver is not NNLS: it never
        reconsiders a column it dropped, and a column that is negative next to
        its correlated partner can be positive once the partner is held at
        zero. This design is the counterexample -- greedy removal returns
        (0, 0, 6) at SSE 10, and (1, 0, 1) is feasible at SSE 8."""
        matrix = [[6.0, 2.0, 1.0], [4.0, 8.0, 1.0],
                  [5.0, 7.0, 1.0], [5.0, 3.0, 1.0]]
        rhs = [7.0, 5.0, 4.0, 8.0]
        free = attention._solve(matrix, rhs)
        assert [round(c, 6) for c in free] == [-2.0, -1.0, 21.0]

        solved, at_zero = attention._solve_nonnegative(matrix, rhs)
        assert all(c >= 0.0 for c in solved)
        sse = sum((sum(c * v for c, v in zip(solved, row)) - y) ** 2
                  for row, y in zip(matrix, rhs))
        assert sse < 10.0 - 1e-9  # strictly better than the greedy answer
        assert sse == pytest.approx(8.0, abs=1e-9)
        assert [round(c, 6) for c in solved] == [1.0, 0.0, 1.0]
        assert at_zero == (1,)

        # KKT: the residual must not be correlated with a free column, and a
        # held column's gradient must point out of the feasible region.
        residual = [y - sum(c * v for c, v in zip(solved, row))
                    for row, y in zip(matrix, rhs)]
        for index in range(len(matrix[0])):
            gradient = -sum(row[index] * r for row, r in zip(matrix, residual))
            if index in at_zero:
                assert gradient >= -1e-9
            else:
                assert abs(gradient) < 1e-9

    def test_a_bounded_term_is_named_where_a_reader_of_the_price_sees_it(self):
        """A zero coefficient and a term held at zero are different claims."""
        regime = attention.REGIMES["gdn.prefill"]
        fit = attention.Fit(regime, ("query_rows", "chunks"), (),
                            (1.0, 0.0), (1.0, 1.0), 6, 4, 0.01,
                            [[1.0, 1.0]], dict(GDN_SCOPE), (), ("chunks",))
        described = fit.describe()
        assert "held at zero by the nonnegativity bound" in described
        assert "does not claim they are free" in described

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
        points = [(op, seconds, "src", dict(GDN_SCOPE))
                  for op, seconds in _gdn_designs()]
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


#: A stand-in for the content hash of `atom.compass.runtime.microbench`, which
#: is what the collector records and what the adapter reads as the policy.
POLICY_A = "a" * 64
POLICY_B = "b" * 64


def _library(tmp_path, designs, *, scope=None, kernels=("k0", "k1"),
             layers=16, graph_extra=None, name="p", policy=POLICY_A,
             arg_sets=None, kernels_per_design=None, acquisition=None):
    """A library holding one attention price per design, replicated by layer.

    The price files are synthetic, so this establishes what the adapter does
    with artifacts -- not that any real artifact declares a resolved scope.

    `policy` is the module hash the collector records; `arg_sets` is the
    rotation that policy produced at each design's footprint, which is a
    number per design rather than a setting. `acquisition` is the metadata a
    run carries about itself, such as the iteration count.
    """
    from atom.compass.runtime.microbench import signature_of

    library = ParametricPriceLibrary()
    for index, (op, seconds) in enumerate(designs):
        ops, prices = [], {}
        served = (kernels_per_design[index] if kernels_per_design is not None
                  else kernels)
        for layer in range(layers):
            copy = _layer_copy(op, layer)
            ops.append(copy)
            record = {
                "seconds": seconds, "kernels": {k: seconds / len(served)
                                                for k in served},
                "occurrences": 1, "name": op["name"],
                "signature": signature_of(copy),
            }
            if arg_sets is not None:
                record["arg_sets"] = (arg_sets[index]
                                      if isinstance(arg_sets, (list, tuple))
                                      else arg_sets)
            prices[signature_of(copy)] = record
        graph = {"ops": ops, "provenance": dict(graph_extra or {})}
        provenance = {"attention_scope": dict(scope or SCOPE)}
        if policy is not None:
            digest = (policy[index] if isinstance(policy, (list, tuple))
                      else policy)
            provenance["collector"] = {"source_identity": {"modules": {
                "atom.compass.runtime.microbench": {
                    "file": "/snap/atom/compass/runtime/microbench.py",
                    "sha256": digest}}}}
        for key, value in (acquisition or {}).items():
            provenance[key] = (value[index]
                               if isinstance(value, (list, tuple))
                               else value)
        blob = {"prices": prices, "provenance": provenance}
        gpath = tmp_path / f"g{name}{index}.json"
        ppath = tmp_path / f"{name}{index}.json"
        gpath.write_text(json.dumps(graph))
        ppath.write_text(json.dumps(blob))
        library.add(str(ppath), str(gpath))
    return library


def _gdn_designs():
    """GDN prefill designs that can identify the regime's terms.

    Three things have to vary independently or the law is unidentifiable by
    construction: the query rows, the chunk count (so widths that are not all
    multiples of CHUNK_SIZE), and the sequence count (so batches that are not
    all single-sequence, which would make `sequences` a copy of `calls`).

    The synthetic truth charges every term the regime carries, at a positive
    rate. A truth that left one out would make its coefficient whatever the
    collinearity between the columns happened to produce -- a small negative
    number, refused for the right reason by the wrong fixture -- and a truth
    with no per-call term would assert the opposite of what the measured
    family shows.
    """
    rates = {"calls": 4.0e-8, "query_rows": 1.0e-9, "chunks": 3.0e-9,
             "sequences": 7.0e-9, "continued_sequences": 0.0,
             "tail_pad_rows": 0.0}
    regime = attention.REGIMES["gdn.prefill"]
    designs = []
    for queries in ([96], [100], [300], [1000], [4680],
                    [128, 96], [64, 64, 96], [512, 300]):
        op = _gdn(queries, initial=[False] * len(queries))
        values = attention.features_for(regime, structure_of(op))
        assert not isinstance(values, Refusal), values
        seconds = sum(rates[name] * value
                      for name, value in zip(regime.features, values))
        designs.append((op, seconds))
    return designs


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
    """A library that charges per launch needs the count. Nothing records it
    here, so the charge cannot be stated and no count is invented for it."""
    library = _library(tmp_path, _cold_designs(), kernels=())
    library.launch_charge_seconds = 95.8e-6
    library.request_attention_scope = dict(SCOPE)
    record, detail = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert record is None
    assert "launch" in detail
    # The refusal says which of the two absences this is: nobody wrote the
    # composition down, and the charge is what makes that matter.
    assert "records which kernels" in detail and "9.58e-05s" in detail


def test_an_unrecorded_composition_passes_where_a_launch_costs_nothing(
        tmp_path):
    """The real collector can be run without kernel-id collection, and the
    source factory charges nothing per launch. Refusing there would withhold a
    price over a number that could not have changed it -- so it is allowed,
    and says in the record that the composition is unknown and uncharged."""
    library = _library(tmp_path, _cold_designs(), kernels=())
    # As `build_source_oracle` publishes it: the source factory charges
    # nothing per launch, and says so rather than leaving it unstated.
    library.launch_charge_seconds = 0.0
    library.request_attention_scope = dict(SCOPE)
    record, source = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert record is not None and source.startswith("interpolated://")
    assert record["kernels"] == {}
    assert "unknown and uncharged" in record["kernel_attribution"]


def test_a_nonzero_launch_charge_is_never_zeroed_to_let_a_price_through(
        tmp_path):
    """The allowance is conditional on the charge the library is configured
    with, and reading it is all that happens: the charge is not touched."""
    library = _library(tmp_path, _cold_designs(), kernels=())
    library.launch_charge_seconds = 95.8e-6
    library.request_attention_scope = dict(SCOPE)
    library.lookup(_unified([641], [641], is_prefill=True, has_cached=False))
    assert library.launch_charge_seconds == 95.8e-6


def test_two_kernel_sets_are_two_treatments_even_where_launches_are_free(
        tmp_path):
    """Not the same absence as an unrecorded composition, and not reached the
    same way. The kernels a record was served by are part of its measurement
    identity, so records served by different kernels are separated before any
    law exists -- they never arrive at one fit to disagree inside it. The
    allowance for an unrecorded composition does not touch that: what it
    passes is silence, and this is evidence."""
    library = _library(tmp_path, _cold_designs(),
                       kernels_per_design=[("k0",), ("k0", "k1"), ("k0",)])
    library.launch_charge_seconds = 0.0
    treatments = {scope["measurement_treatment"] for _op, _s, _note, scope
                  in library.attention_design_points()}
    assert len(treatments) == 2
    library.request_attention_scope = dict(SCOPE)
    record, _detail = library.lookup(
        _unified([641], [641], is_prefill=True, has_cached=False))
    assert record is None


class TestTheRotationIsAPolicyNotAScope:
    """`arg_sets` is what one residency policy produced at a given footprint.

    It is `max(2, min(64, COLD_WORKING_SET_BYTES // per_set))`, so it falls
    out of the operand size. Treating it as part of the treatment makes the
    treatment a function of the design point: every size becomes its own law
    with one point in it, and nothing is ever fitted.
    """

    def _ask(self, library):
        library.request_attention_scope = dict(SCOPE)
        return library.lookup(
            _unified([641], [641], is_prefill=True, has_cached=False))

    @staticmethod
    def _treatments(library) -> int:
        """How many distinct treatments the collected points fall into.

        The direct observation of pooling: a refusal further downstream says
        the law is short of points, which is a consequence rather than the
        fact under test.
        """
        return len({scope["measurement_treatment"]
                    for _op, _s, _note, scope
                    in library.attention_design_points()})

    def test_the_clamp_this_reads_is_the_clamp_the_collector_applies(self):
        """Pinned against the source, so a change there is caught here."""
        import inspect

        from atom.compass.runtime import microbench
        body = inspect.getsource(microbench._build_arg_sets)
        assert "max(2, min(64," in body.replace(" ", "").replace(
            "max(2,min(64,", "max(2, min(64,")

    def test_one_policy_at_three_footprints_is_one_law(self, tmp_path):
        """Three sizes measured by one collector rotated over three different
        counts. That is one policy applied three times, so they pool."""
        library = _library(tmp_path, _cold_designs(), policy=POLICY_A,
                           arg_sets=[64, 12, 2])
        assert len(library.attention_design_points()) == 3
        assert self._treatments(library) == 1
        record, _source = self._ask(library)
        assert record is not None

    def test_two_policies_do_not_pool(self, tmp_path):
        """A different microbench is a different byte budget and a different
        clamp. Those numbers were not taken the same way."""
        library = _library(tmp_path, _cold_designs(),
                           policy=[POLICY_A, POLICY_B, POLICY_A],
                           arg_sets=[64, 64, 64])
        assert self._treatments(library) == 2
        record, _detail = self._ask(library)
        assert record is None

    def test_an_unrecorded_policy_pools_with_nothing_that_has_one(
            self, tmp_path):
        library = _library(tmp_path, _cold_designs(),
                           policy=[POLICY_A, None, POLICY_A])
        assert self._treatments(library) == 2
        record, _detail = self._ask(library)
        assert record is None

    def test_the_realized_rotation_is_kept_as_evidence_on_the_point(
            self, tmp_path):
        """Out of the identity, not out of the record: a reader still sees how
        cold each point's operands were held."""
        library = _library(tmp_path, _cold_designs(), arg_sets=[64, 12, 2])
        notes = " ".join(note for _op, _s, note, _scope
                         in library.attention_design_points())
        assert "arg_sets 64" in notes and "arg_sets 2" in notes

    def test_the_policy_is_reported_in_coverage(self, tmp_path):
        library = _library(tmp_path, _cold_designs(), arg_sets=[64, 12, 2])
        library.launch_charge_seconds = 0.0
        coverage = library.attention_coverage()
        assert coverage["acquisition_policies"] == [
            ("atom.compass.runtime.microbench", POLICY_A)]
        assert coverage["launch_charge_seconds"] == 0.0


class TestAcquisitionMetadataIsNotADeployment:
    """`iters` and `only` are true of the collection, not of a request.

    A served step does not choose an iteration count or a family filter, so
    requiring a request to state one before a law applies would make every law
    unmatchable for a reason nobody could act on.
    """

    def _ask(self, library):
        library.request_attention_scope = dict(SCOPE)
        return library.lookup(
            _unified([641], [641], is_prefill=True, has_cached=False))

    def test_files_that_differ_only_in_iteration_count_still_pool(
            self, tmp_path):
        library = _library(tmp_path, _cold_designs(),
                           acquisition={"iters": [50, 20, 50]})
        record, _detail = self._ask(library)
        assert record is not None

    def test_files_that_differ_only_in_the_family_filter_still_pool(
            self, tmp_path):
        library = _library(tmp_path, _cold_designs(),
                           acquisition={"only": ["gdn", "attention", "gdn"]})
        record, _detail = self._ask(library)
        assert record is not None

    def test_it_is_kept_and_reported_rather_than_dropped(self, tmp_path):
        library = _library(tmp_path, _cold_designs(),
                           acquisition={"iters": 50, "only": "attention"})
        context = library.attention_coverage()["acquisition_context"]
        assert context
        assert all(entry == {"iters": 50, "only": "attention"}
                   for entry in context.values())

    def test_a_real_deployment_difference_still_refuses(self, tmp_path):
        """The relaxation is confined to acquisition metadata: a difference in
        what was actually deployed is still a difference."""
        library = _library(tmp_path, _cold_designs(),
                           acquisition={"model": ["m1", "m2", "m1"]})
        scopes = {attention.scope_key(scope) for _op, _s, _note, scope
                  in library.attention_design_points()}
        assert len(scopes) == 2  # m1 and m2 are two deployments, not one law
        record, _detail = self._ask(library)
        assert record is None


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


def _resolved_record(*, gdn_layers=(0, 1), mha_layers=(3,), dtype="bf16",
                     asked_dtype="bf16", envs=None, block=16, window=-1,
                     state_slots=32, views=True):
    """A stood-up deployment's record, in the schema the collector writes.

    Values follow the real one: BF16 KV in a packed paged view, GDN state in
    its own conv/recurrent pool, the dispatch environment as strings. The
    keyword arguments exist so a test can withhold exactly one fact and see it
    named rather than defaulted.
    """
    atom_envs = {"ATOM_ENABLE_GDN_DECODE_LOSSY_FAST": "False",
                 "ATOM_USE_UNIFIED_ATTN": "False",
                 "ATOM_FORCE_ATTN_TRITON": "False",
                 "ATOM_V4_BACKEND": "legacy", "ATOM_V4_BACKEND_LAYERS": ""}
    atom_envs.update(envs or {})
    layers, kv_views = {}, {}
    for number in gdn_layers:
        layers["language_model.model.layers.%d.linear_attn" % number] = {
            "attn_backend": "atom.model_ops.attentions.gdn_attn."
                            "GDNAttentionBackend",
            "class": "atom.model_ops.base_attention.LinearAttention",
            "impl": "atom.model_ops.attention_gdn.GatedDeltaNet",
            "impl_attrs": {"activation": "silu", "head_k_dim": 128,
                           "head_v_dim": 128, "hidden_size": 5120,
                           "key_dim": 2048, "layer_num": number,
                           "num_k_heads": 16, "num_v_heads": 48,
                           "value_dim": 6144}}
        kv_views["holder0:layer_%d" % number] = _view([32, 3, 10240],
                                                      [30720, 10240, 1],
                                                      [32, 48, 128, 128],
                                                      [786432, 16384, 128, 1])
    for number in mha_layers:
        layers["language_model.model.layers.%d.self_attn" % number] = {
            "attn_backend": "atom.model_ops.attentions.aiter_attention."
                            "AiterBackend",
            "class": "atom.model_ops.paged_attention.Attention",
            "impl": "atom.model_ops.attention_mha.PagedAttentionImpl",
            "impl_attrs": {"head_dim": 256, "kv_cache_dtype": dtype,
                           "layer_num": number, "num_heads": 24,
                           "num_kv_heads": 4, "scale": 0.0625,
                           "sliding_window": window}}
        kv_views["holder0:layer_%d" % number] = _view(
            [131072, 4, 32, 16, 8], [16384, 4096, 128, 8, 1],
            [131072, 4, 2, 256, 8], [16384, 4096, 2048, 8, 1])
    record = {
        "atom_envs": atom_envs,
        "caches_summary": {
            "blocks": 131072, "kv_heads": 4, "state_slots": state_slots,
            "pool_shapes": {"kv_cache": [[2, 16, 131072, block, 4, 256],
                                         "torch.bfloat16"]}},
        "config": {"kv_cache_block_size": block,
                   "kv_cache_dtype": asked_dtype,
                   "max_model_len": 262144},
        "layers": layers,
    }
    if views:
        record["kv_views"] = kv_views
    return record


def _view(k_shape, k_stride, v_shape, v_stride):
    return {"entry_class": "KVCacheTensor",
            "k": {"shape": k_shape, "stride": k_stride,
                  "dtype": "torch.bfloat16", "element_size": 2,
                  "is_contiguous": True, "data_ptr": "0x7ecf1e000000",
                  "storage_offset": 0, "bytes": 1966080},
            "v": {"shape": v_shape, "stride": v_stride,
                  "dtype": "torch.bfloat16", "element_size": 2,
                  "is_contiguous": True, "data_ptr": "0x7ece8de00000",
                  "storage_offset": 34359738368, "bytes": 50331648}}


class TestAResolvedDeploymentIsReadAsAScope:
    """The judgement, written down once and checkable against the record.

    A dump is not a scope, but refusing every dump is not the answer either:
    the process that measured these prices wrote down what it stood up, and
    reading that as the five facts the kernel turns on is exactly the reading
    somebody has to make. It is made here, with each value carrying the path
    it came from.
    """

    def _read(self, **kwargs):
        from atom.compass.core.cost.families import attention_scope

        return attention_scope.read_resolved(_resolved_record(**kwargs),
                                             where="rec")

    def test_both_families_get_their_own_scope(self):
        declared = self._read()
        assert set(declared.scopes) == {"unified", "gdn"}
        assert declared.for_family("unified")["kv_cache_dtype"] == "bf16"
        assert declared.for_family("unified")["sliding_window"] == -1
        assert declared.for_family("gdn")["gdn_decode_lossy_fast"] is False
        # The KV facts are not handed to the family that never reads the KV
        # cache, and the state geometry is not handed to the one that has no
        # state.
        assert "kv_cache_layout" not in declared.for_family("gdn")
        assert "gdn_state_geometry" not in declared.for_family("unified")

    def test_every_declared_key_is_covered(self):
        declared = self._read()
        assert set(attention.UNIFIED_SCOPE) <= set(
            declared.for_family("unified"))
        assert set(attention.GDN_SCOPE) <= set(declared.for_family("gdn"))

    def test_the_layout_is_the_view_not_the_pool(self):
        """A pool allocation proves storage; the view proves the arrangement
        the kernel indexes it through."""
        layout = dict(self._read().for_family("unified"))["kv_cache_layout"]
        fields = dict(dict(layout)["k"])
        assert fields["shape"] == (131072, 4, 32, 16, 8)
        assert fields["stride"] == (16384, 4096, 128, 8, 1)
        assert fields["dtype"] == "torch.bfloat16"

    def test_where_a_tensor_sits_is_not_part_of_its_layout(self):
        """Otherwise two layers holding identical views would be two
        deployments, and nothing would ever pool."""
        one = self._read().for_family("unified")["kv_cache_layout"]
        other = _resolved_record(mha_layers=(3, 7))
        other["kv_views"]["holder0:layer_7"]["k"]["data_ptr"] = "0xdeadbeef"
        other["kv_views"]["holder0:layer_7"]["k"]["storage_offset"] = 4096
        from atom.compass.core.cost.families import attention_scope

        assert attention_scope.read_resolved(
            other, where="rec").for_family("unified")["kv_cache_layout"] == one

    def test_the_backend_carries_the_environment_it_dispatches_on(self):
        """`AiterBackend` does not say which branch ran. Two runs that differ
        on the dispatch environment took different kernels through the same
        backend class."""
        mine = self._read().for_family("unified")["attention_backend"]
        theirs = self._read(
            envs={"ATOM_USE_UNIFIED_ATTN": "True"}
        ).for_family("unified")["attention_backend"]
        assert mine != theirs
        assert dict(mine)["impl"].endswith("PagedAttentionImpl")

    def test_requested_and_resolved_are_both_recorded_and_not_confused(self):
        declared = self._read(asked_dtype="auto")
        assert declared.for_family("unified")["kv_cache_dtype"] == "bf16"
        statuses = {(f.key, f.status): f.value for f in declared.facts}
        assert statuses[("kv_cache_dtype", "resolved")] == "bf16"
        assert statuses[("kv_cache_dtype", "requested")] == "auto"

    def test_a_config_that_contradicts_the_layers_is_refused(self):
        with pytest.raises(ValueError, match="one of the two is wrong"):
            self._read(asked_dtype="fp8")

    def test_a_block_size_the_pool_does_not_carry_is_not_resolved(self):
        """The config value is what was asked for. It becomes a resolved fact
        only where the pool that was allocated carries it."""
        from atom.compass.core.cost.families import attention_scope

        record = _resolved_record()
        record["config"]["kv_cache_block_size"] = 64
        with pytest.raises(ValueError, match="kv_cache_block_size"):
            attention_scope.read_resolved(record, where="rec")

    def test_a_missing_fact_is_named_rather_than_defaulted(self):
        with pytest.raises(ValueError, match="kv_views"):
            self._read(views=False)

    def test_a_short_view_table_still_proves_the_layout(self):
        """The collector writes the holders it was given, not one entry per
        bound module, and every layer of a family indexes the same pool. A
        layout two holders show and none contradicts is resolved; requiring
        all sixty-four would refuse every real record."""
        from atom.compass.core.cost.families import attention_scope

        record = _resolved_record(gdn_layers=(0, 1, 2), mha_layers=(3, 7))
        del record["kv_views"]["holder0:layer_7"]
        del record["kv_views"]["holder0:layer_2"]
        declared = attention_scope.read_resolved(record, where="rec")
        assert declared.for_family("unified")["kv_cache_layout"] == \
            self._read().for_family("unified")["kv_cache_layout"]
        # And it says which modules showed it, so a one-witness reading is
        # not mistaken for a unanimous one.
        source = [f.source for f in declared.facts
                  if f.key == "kv_cache_layout"][0]
        assert "layers.3.self_attn" in source
        assert "layers.7.self_attn" not in source

    def test_two_recorded_views_that_disagree_have_no_one_layout(self):
        from atom.compass.core.cost.families import attention_scope

        record = _resolved_record(mha_layers=(3, 7))
        record["kv_views"]["holder0:layer_7"]["k"]["stride"] = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="disagree on it"):
            attention_scope.read_resolved(record, where="rec")

    def test_a_family_no_recorded_view_covers_is_refused(self):
        with pytest.raises(ValueError, match="for none of the"):
            self._read(views=False)

    def test_a_dispatch_environment_nobody_recorded_is_named(self):
        from atom.compass.core.cost.families import attention_scope

        record = _resolved_record()
        del record["atom_envs"]["ATOM_V4_BACKEND"]
        with pytest.raises(ValueError, match="ATOM_V4_BACKEND"):
            attention_scope.read_resolved(record, where="rec")

    def test_layers_that_disagree_have_no_one_value(self):
        from atom.compass.core.cost.families import attention_scope

        record = _resolved_record(mha_layers=(3, 7))
        record["layers"]["language_model.model.layers.7.self_attn"][
            "impl_attrs"]["sliding_window"] = 4096
        with pytest.raises(ValueError, match="disagree on"):
            attention_scope.read_resolved(record, where="rec")

    def test_every_scope_member_says_where_it_came_from(self):
        declared = self._read()
        for fact in declared.facts:
            assert fact.source.startswith("rec")
            assert fact.status in ("resolved", "requested")


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
        assert library.request_attention_scope.for_family(
            "unified")["kv_cache_dtype"] == "fp8"

        record, source = library.lookup(
            _unified([641], [641], is_prefill=True, has_cached=False))
        assert record is not None
        assert record["interpolated"] is True
        assert source.startswith("interpolated://")
        assert sorted(record["kernels"]) == ["k0", "k1"]

    def test_the_scope_file_is_recorded_as_the_bytes_that_were_parsed(
            self, tmp_path):
        """Cost-critical input: which laws a run may price from depends on it,
        so what it WAS is part of what the run was. Read once, digested at the
        read, and recorded beside the prices -- not re-derived from the path
        afterwards, which would describe bytes the run never used."""
        import hashlib

        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        scope_path = tmp_path / "attn_scope.json"
        scope_path.write_text(json.dumps({"attention_scope": dict(SCOPE)}))
        raw = scope_path.read_bytes()
        library = _price_library(self._files(tmp_path), gap_ratio(True), None,
                                 str(scope_path))

        rows = [row for row in library.loaded_inputs
                if row.role == "oracle.attention_scope"]
        assert len(rows) == 1
        assert rows[0].sha256 == hashlib.sha256(raw).hexdigest()
        assert rows[0].size == len(raw)
        assert rows[0].requested == str(scope_path)

        # Replaced after the load: the record still describes what was read.
        scope_path.write_text(json.dumps({"attention_scope": {}}))
        assert rows[0].sha256 == hashlib.sha256(raw).hexdigest()

    def test_a_rank_is_served_its_own_scope_file(self, tmp_path):
        """Every rank writes `name.tp0.json`, `name.tp1.json`; the option can
        only name the stem. Which file this rank got is the thing worth
        recording, and the record holds both ends of it."""
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        stem = tmp_path / "attn_scope.json"
        stem.write_text(json.dumps({"attention_scope": dict(SCOPE)}))
        mine = tmp_path / "attn_scope.tp1.json"
        mine.write_text(json.dumps(
            {"attention_scope": dict(SCOPE, kv_cache_dtype="bf16")}))

        library = _price_library(self._files(tmp_path), gap_ratio(True),
                                 {"tp": 1}, str(stem))
        row = [r for r in library.loaded_inputs
               if r.role == "oracle.attention_scope"][0]
        assert row.path == str(mine)
        assert row.requested == str(stem)
        assert row.rank_own is True
        assert library.request_attention_scope.for_family(
            "unified")["kv_cache_dtype"] == "bf16"

    def test_a_mapping_is_accepted_as_well_as_a_file(self, tmp_path):
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        library = _price_library(self._files(tmp_path), gap_ratio(True), None,
                                 dict(SCOPE))
        assert library.request_attention_scope.for_family(
            "unified")["attention_backend"] == "unified_attention"

    def test_a_scope_with_modelling_off_is_an_error_not_a_silent_drop(
            self, tmp_path):
        from atom.compass.runtime.source_oracle import _price_library

        with pytest.raises(ValueError, match="modelling is off"):
            _price_library(self._files(tmp_path), None, None, dict(SCOPE))

    def test_a_missing_scope_file_is_refused(self, tmp_path):
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        with pytest.raises(ValueError, match="not a file that exists"):
            _price_library(self._files(tmp_path), gap_ratio(True), None,
                           str(tmp_path / "nothing.json"))

    def test_a_file_that_is_neither_a_scope_nor_a_deployment_is_refused(
            self, tmp_path):
        """A scope has to be written down by whoever resolved it, or be
        readable from a record of what was stood up. A file that is neither
        cannot become one by being passed here."""
        from atom.compass.runtime.source_oracle import _attention_request_scope

        other = tmp_path / "other.json"
        other.write_text(json.dumps({"model": "x", "tensor_parallel_size": 1}))
        with pytest.raises(ValueError, match="none of the facts"):
            _attention_request_scope(str(other))

    def test_a_deployment_record_short_of_a_fact_is_refused_by_name(
            self, tmp_path):
        """Read, not rejected out of hand -- and refused with the missing fact
        named, which is what somebody needs to close the gap."""
        from atom.compass.runtime.source_oracle import _attention_request_scope

        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps({"atom_envs": {"ATOM_V4_BACKEND": "legacy"},
                                    "caches_summary": {"kv_heads": 4}}))
        with pytest.raises(ValueError, match="names no module of either"):
            _attention_request_scope(str(dump))


class TestOneLibraryPricesBothFamilies:
    """The ordinary case, and the one a mixed deployment actually is.

    64 bound modules, 48 linear and 16 paged, measured in the same process
    under treatments that differ naturally between them. A library that can
    only answer when every observation in it shares one treatment cannot
    answer for this model at all.
    """

    def _library(self, tmp_path):
        from atom.compass.runtime.microbench import signature_of
        from atom.compass.runtime.source_oracle import (_price_library,
                                                        gap_ratio)

        entries = []
        for index, (op, seconds) in enumerate(_cold_designs()):
            prices = {signature_of(op): {
                "seconds": seconds, "kernels": {"k0": seconds},
                # Two regions rebuilt for the paged family and one for the
                # linear family: a real difference in what was measured, and
                # the reason a library-wide treatment question has no answer.
                "kv_regions": 2, "cache": "graph",
                "occurrences": 1, "name": op["name"],
                "signature": signature_of(op)}}
            entries.append(_file(tmp_path, "u%d" % index, [op], prices))
        for index, (op, seconds) in enumerate(_gdn_designs()):
            prices = {signature_of(op): {
                "seconds": seconds, "kernels": {"gdn0": seconds},
                "kv_regions": 1, "cache": "graph",
                "occurrences": 1, "name": op["name"],
                "signature": signature_of(op)}}
            entries.append(_file(tmp_path, "g%d" % index, [op], prices,
                                 scope=GDN_SCOPE))
        return _price_library(entries, gap_ratio(True), None,
                              {"unified": dict(SCOPE), "gdn": dict(GDN_SCOPE)})

    def test_an_unseen_call_of_each_family_is_priced_from_one_body(
            self, tmp_path):
        library = self._library(tmp_path)

        mha, mha_source = library.lookup(
            _unified([641], [641], is_prefill=True, has_cached=False))
        gdn, gdn_source = library.lookup(_gdn([256], initial=[False]))

        assert mha is not None and gdn is not None
        assert mha["interpolated"] is True and gdn["interpolated"] is True
        assert mha_source.startswith("interpolated://")
        assert gdn_source.startswith("interpolated://")
        # Each priced from its own family's law, with its own evidenced
        # launch composition -- not one family's kernels charged to the other.
        assert sorted(mha["kernels"]) == ["k0"]
        assert sorted(gdn["kernels"]) == ["gdn0"]

    def test_the_treatment_is_resolved_inside_the_asking_family(self, tmp_path):
        """The two families were measured under different region rebuilds,
        which is ordinary. Asking whether the whole library shares one
        treatment answers no, and would refuse both prices."""
        library = self._library(tmp_path)
        treatments = {obs[4] for obs in library._attention_obs}
        assert len(treatments) > 1

        asked = _gdn([256], initial=[False])
        resolved = library._treatment_for(asked, dict(GDN_SCOPE))
        assert resolved is not None
        assert dict(resolved[1:])["kv_regions"] == 1

    def test_a_second_treatment_inside_one_family_is_still_refused(
            self, tmp_path):
        """Ambiguity inside the domain is not resolved by picking one: a
        cold-cache and a warm-cache measurement of the same law are two laws.
        """
        from atom.compass.runtime.microbench import signature_of

        library = self._library(tmp_path)
        op = _gdn([2048], initial=[False])
        prices = {signature_of(op): {
            "seconds": 2e-6, "kernels": {"gdn0": 2e-6},
            "kv_regions": 9, "cache": "cold", "occurrences": 1,
            "name": op["name"], "signature": signature_of(op)}}
        price, graph = _file(tmp_path, "gx", [op], prices, scope=GDN_SCOPE)
        library.add(price, graph)

        assert library._treatment_for(_gdn([256], initial=[False]),
                                      dict(GDN_SCOPE)) is None


#: The scope Pricing resolved on the same stand-up path, and a price record
#: from the CORRECTED acquisition, which carries its own resolution inline.
#: Batch 1 is withdrawn from every calibrated input -- it was taken against a
#: stale module -- so nothing here reads it.
PRICING_SCOPE = ("/workspace/ATOM/agent_scratch/pricing_coverage/p_caseb"
                 "/ATTN_SCOPE.json")
B2_PRICE = "/workspace/ATOM/agent_scratch/b2acq/training/PRICE_G5.rep2.json"


@pytest.mark.skipif(not os.path.exists(PRICING_SCOPE),
                    reason="the resolved scope file is not here")
def test_the_resolved_scope_file_becomes_a_usable_declaration():
    """Against the real file, not a fixture shaped like one.

    It records the deployment as an environment dump under its own keys, which
    is not a scope; reading it as one is the judgement `attention_scope` makes,
    and this is where that reading is checked against what Pricing actually
    wrote: 48 linear and 16 paged modules, BF16, window disabled, block 16.
    """
    from atom.compass.runtime.source_oracle import _attention_request_scope

    declared, loaded = _attention_request_scope(PRICING_SCOPE)
    assert loaded.role == "oracle.attention_scope"
    assert loaded.sha256 and loaded.size

    unified = declared.for_family("unified")
    assert set(attention.UNIFIED_SCOPE) <= set(unified)
    assert unified["kv_cache_dtype"] == "bf16"
    assert unified["sliding_window"] == -1
    assert unified["kv_cache_block_size"] == 16
    gdn = declared.for_family("gdn")
    assert set(attention.GDN_SCOPE) <= set(gdn)
    assert gdn["gdn_decode_lossy_fast"] is False
    assert all(fact.source.startswith(loaded.path) for fact in declared.facts)


@pytest.mark.skipif(not os.path.exists(B2_PRICE),
                    reason="the corrected price record is not here")
def test_a_price_record_carries_the_deployment_its_own_process_resolved():
    """The strongest evidence a scope can have: not a later reading of a
    config, but what the process that took these measurements stood up,
    written down before it priced anything. The adapter takes it from the
    record rather than requiring the request to declare it again.

    Nothing here is fitted. What is checked is that the record proves its own
    deployment: the seconds are not read at all.
    """
    from atom.compass.core.cost.families.adapter import _resolved_declaration

    with open(B2_PRICE) as handle:
        blob = json.load(handle)
    declared, unreadable = _resolved_declaration(blob, B2_PRICE)
    assert unreadable is None, unreadable
    assert declared is not None
    assert set(declared.scopes) == {"unified", "gdn"}
    assert declared.for_family("unified")["kv_cache_dtype"] == "bf16"
    assert declared.for_family("gdn")["gdn_decode_lossy_fast"] is False
    assert declared.for_op({"name": GDN}) == declared.for_family("gdn")
    assert declared.for_op({"name": UNIFIED}) == declared.for_family("unified")
