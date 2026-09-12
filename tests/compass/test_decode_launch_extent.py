"""What a decode step actually launches, and what a derivation says it does.

Two builders write ``max_seqlen_k`` to different values. ``prepare_decode``
takes ``context_lens.max()`` -- the batch's own longest history
(aiter_attention.py:1100, :1139) -- and ``build_for_cudagraph_capture`` pins it
to the engine's ``max_model_len`` (:1367, :1331). A FULL capture replays the
whole forward from the buffers the capture holds, so a replayed step runs the
captured value however short the batch is; a PIECEWISE capture leaves attention
eager (model_runner.py:4019-4031) and rebuilds it from the batch every step.
The derivation used the eager rule in both places, so every FULL decode graph
it produced carried an extent the native run never had -- and ``max_seqlen_k``
is part of the operator identity key (core/cost/identity.py:38), so those
graphs could not match a captured measurement at all.

The launch extent proper is not that field. On the branch this deployment
dispatches -- see ``atom/compass/DECODE_CALIBRATION_SCOPE.md`` -- the kernel
reads ``batch_size = query.shape[0] // query_length`` and launches ``grid =
(batch_size, num_kv_heads, max_context_partition_num)``, and never looks at
``max_seqlen_k``. So the executed rows come off the operand shape, which is why
the attention family derives them instead of requiring a ``capture_bucket``
field that no operator context carries.

3 -> 4 and 31 -> 32 under FULL, PIECEWISE and eager, with the warm binder
asked the same question a second time.
"""

import argparse

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.families import attention as A
from atom.compass.runtime.batch_spec import BatchSpec
from atom.compass.runtime.templates import (BindRefusal, CarriedAllocation,
                                            _bind, bind_cohort, template_key)
from atom.compass.runtime.tracer import ShapeDeriver

MAX_MODEL_LEN = 262144
DECLARED = {"block_size": 16, "max_model_len": MAX_MODEL_LEN,
            "position_rows": 3}
CONTEXT = 1151

#: The gluon decode branch's declared scope. `compute_units` is the part's CU
#: count -- 304 for MI300X -- and the launcher assumes two workgroups per CU,
#: so the split count is min(8, ceil(608 / (sequences * num_kv_heads)))
#: (`attention_mha.py`:552, `pa_decode_gluon.py`:111-118). Both are here
#: because the tile geometry follows from them and neither is in the key.
GLUON_SCOPE = {"attention_backend": "paged_gluon", "sliding_window": -1,
               "num_kv_heads": 4, "compute_units": 304}


def shape(n, context=CONTEXT, *, bucket=None):
    return StepShape(
        num_scheduled_tokens=tuple([1] * n),
        context_lens=tuple([context] * n),
        num_prefill_tokens=0,
        topology={"tp": 1}, rank_coords={"tp": 0},
        capture_bucket=bucket, compiled=None, produces_output=True)


def spec_of(n, *, bucket=None, mode=None, context=CONTEXT, kind="decode"):
    if kind == "decode":
        return ShapeDeriver(None, cudagraph_mode=mode, **DECLARED).spec_for(
            shape(n, context, bucket=bucket))
    return BatchSpec(kind="prefill", query_lens=tuple([context] * n),
                     context_lens=tuple([context] * n),
                     block_size=16, max_model_len=MAX_MODEL_LEN,
                     capture_bucket=bucket, cudagraph_mode=mode)


def recorded(spec) -> dict:
    return dict(spec.attention_context())


class TestWhichRuleTheSpecFollows:

    @pytest.mark.parametrize("batch,bucket", [(3, 4), (31, 32)])
    def test_a_full_replay_carries_the_captures_extent(self, batch, bucket):
        spec = spec_of(batch, bucket=bucket, mode="full")
        assert spec.replays_captured_metadata is True
        assert spec.launch_extent_scope == "captured"
        assert spec.launch_max_seqlen_k == MAX_MODEL_LEN
        assert recorded(spec)["max_seqlen_k"] == MAX_MODEL_LEN

    @pytest.mark.parametrize("batch,bucket", [(3, 4), (31, 32)])
    def test_a_piecewise_replay_carries_the_batchs(self, batch, bucket):
        # Padded rows, eager attention metadata: the two are independent, and
        # this is the case that shows it.
        spec = spec_of(batch, bucket=bucket, mode="piecewise")
        assert spec.replays_captured_metadata is False
        assert spec.launch_extent_scope == "batch"
        assert spec.launch_max_seqlen_k == CONTEXT
        assert recorded(spec)["max_seqlen_k"] == CONTEXT
        # ... and it is still padded. The rows are the bucket's either way.
        assert spec.executed_rows == bucket

    @pytest.mark.parametrize("batch", [3, 31])
    def test_an_eager_step_carries_the_batchs(self, batch):
        spec = spec_of(batch, bucket=None, mode="full")
        # No bucket, so nothing was replayed whatever the mode declares.
        assert spec.replays_captured_metadata is False
        assert spec.launch_max_seqlen_k == CONTEXT
        assert spec.executed_rows == batch

    def test_a_bucket_without_a_declared_mode_is_unanswered(self):
        spec = spec_of(3, bucket=4, mode=None)
        assert spec.replays_captured_metadata is None
        assert spec.launch_extent_scope == "undeclared"
        # The batch value is still what gets recorded -- the point is that the
        # scope says it is not evidence, not that the field goes missing.
        assert spec.launch_max_seqlen_k == CONTEXT

    def test_a_prefill_is_eager_at_any_bucket(self):
        spec = spec_of(2, bucket=32, mode="full", context=64, kind="prefill")
        assert spec.replays_captured_metadata is False
        assert spec.launch_max_seqlen_k == 64

    def test_a_ragged_batch_takes_its_own_longest_history(self):
        spec = BatchSpec(kind="decode", query_lens=(1, 1, 1),
                         context_lens=(512, 56320, 1024),
                         block_size=16, max_model_len=MAX_MODEL_LEN,
                         cudagraph_mode="piecewise")
        assert spec.launch_max_seqlen_k == 56320
        spec_full = BatchSpec(kind="decode", query_lens=(1, 1, 1),
                              context_lens=(512, 56320, 1024),
                              block_size=16, max_model_len=MAX_MODEL_LEN,
                              capture_bucket=4, cudagraph_mode="full")
        assert spec_full.launch_max_seqlen_k == MAX_MODEL_LEN


class TestTheWarmBinderKeepsIt:
    """A template is derived once and bound per cohort. The binding is where
    a correctly derived extent came back as the cohort's, one cohort later."""

    def test_a_captured_extent_is_a_constant_of_the_graph(self):
        assert _bind("max_seqlen_k", MAX_MODEL_LEN, [(1, 1151)] * 3,
                     pad_rows=1, extent_scope="captured") == MAX_MODEL_LEN

    def test_an_eager_extent_follows_the_cohort(self):
        assert _bind("max_seqlen_k", 1151, [(1, 2048), (1, 999)],
                     extent_scope="batch") == 2048

    def test_the_padded_rows_do_not_raise_the_extent(self):
        # A padded row's context is zero; it must not pull `max` down either.
        assert _bind("max_seqlen_k", 1151, [(1, 1151)] * 3, pad_rows=1,
                     extent_scope="batch") == 1151

    def test_an_undeclared_mode_refuses_rather_than_picking_one(self):
        with pytest.raises(BindRefusal):
            _bind("max_seqlen_k", MAX_MODEL_LEN, [(1, 1151)] * 3,
                  pad_rows=1, extent_scope="undeclared")

    def test_bind_cohort_rejects_a_scope_it_has_no_rule_for(self):
        with pytest.raises(BindRefusal):
            bind_cohort({"ops": []}, shape(3, bucket=4),
                        extent_scope="FULL")

    def test_bind_cohort_defaults_to_the_eager_rule(self):
        template = {"ops": [{"name": A.UNIFIED,
                             "context": [["max_seqlen_k", 1151],
                                         ["max_seqlen_q", 1]]}]}
        bound = bind_cohort(template, shape(2, context=4096))
        assert dict(bound["ops"][0]["context"])["max_seqlen_k"] == 4096

    def test_bind_cohort_keeps_a_captured_extent(self):
        spec = spec_of(2, bucket=4, mode="full", context=4096)
        template = {"ops": [{"name": A.UNIFIED,
                             "context": [["max_seqlen_k", MAX_MODEL_LEN],
                                         ["max_seqlen_q", 1]]}],
                    "provenance": {"batch_spec": spec.to_dict()}}
        bound = bind_cohort(template, shape(2, context=4096, bucket=4),
                            extent_scope="captured")
        assert (dict(bound["ops"][0]["context"])["max_seqlen_k"]
                == MAX_MODEL_LEN)


def unified_op(rows, active, *, max_seqlen_q=1, context=CONTEXT, bucket=None):
    """A decode call as the key records it: bucket-wide rows, `active` of them
    carrying a request. The padded offsets repeat the last real one
    (model_runner.py:2649-2652) and the padded contexts are zero."""
    starts = list(range(active + 1)) + [active] * (rows - active)
    ctx = [["context_lens", [context] * active + [0] * (rows - active)],
           ["cu_seqlens_q", starts],
           ["cu_seqlens_k", None],
           ["max_seqlen_q", max_seqlen_q],
           ["max_seqlen_k", context],
           ["is_prefill", False]]
    if bucket is not None:
        ctx.append(["capture_bucket", bucket])
    return {"name": A.UNIFIED,
            # q, q_scale, k, v -- q is operand 0, [rows, heads, head_dim].
            "input_shapes": [[rows * max_seqlen_q, 24, 256], None,
                             [rows * max_seqlen_q, 4, 256],
                             [rows * max_seqlen_q, 4, 256]],
            "context": ctx}


class TestTheModelReadsTheExecutedExtent:
    """`capture_bucket` is a field of `StepShape` and `BatchSpec` that no
    operator context carries, so requiring it refused every decode vector this
    family can build. The rows are on the operand instead."""

    @pytest.mark.parametrize("rows,active", [(4, 3), (32, 31), (4, 4),
                                             (32, 32)])
    def test_executed_rows_come_off_the_query_operand(self, rows, active):
        st = A.structure_of(unified_op(rows, active))
        assert st.executed_rows == rows
        assert st.active_sequences == active
        assert st.sequences == rows

    def test_a_speculative_decode_divides_by_its_query_length(self):
        st = A.structure_of(unified_op(4, 4, max_seqlen_q=2))
        assert st.executed_rows == 4

    def test_a_row_count_that_is_not_whole_query_lengths_is_unknown(self):
        op = unified_op(4, 3)
        op["input_shapes"][0] = [7, 24, 256]
        op["context"] = [["max_seqlen_q", 2] if k == "max_seqlen_q" else [k, v]
                         for k, v in (tuple(e) for e in op["context"])]
        assert A.structure_of(op).executed_rows is None

    @pytest.mark.parametrize("rows,active", [(4, 3), (32, 31), (4, 4),
                                             (32, 32)])
    def test_a_padded_launch_is_charged_at_its_launched_width(self, rows,
                                                             active):
        """Padding is still charged -- through the launch, not beside it.

        The gluon decode regime no longer carries `bucket_pad`. It does not
        need to: `crit_waves` counts the CTAs the grid actually launches, and
        a padded row launches its CTAs like any other. So an 8-of-16 bucket
        occupies a 16-row launch and is charged for one, which is what the
        padded training point (8 active in bucket 16) and the padded frozen
        holdout point (4 in bucket 16) measured.

        What a padded row does *not* do is lengthen anyone's walk: its context
        is zero, so it adds nothing to `max_cta_tiles`.
        """
        st = A.structure_of(unified_op(rows, active))
        scope = dict(GLUON_SCOPE)
        regime = A.regime_of(unified_op(rows, active), st, scope)
        assert not isinstance(regime, A.Refusal), regime
        vec = A.features_for(regime, st, scope)
        assert not isinstance(vec, A.Refusal), vec
        splits = A._decode_splits(scope, st.sequences)
        ctas = rows * scope["num_kv_heads"] * splits
        longest = vec[regime.features.index("max_cta_tiles")]
        assert vec[regime.features.index("crit_waves")] == pytest.approx(
            longest * ctas
            / float(scope["compute_units"] * A.DECODE_OCCUPANCY))

    def test_a_declared_bucket_that_contradicts_the_rows_refuses(self):
        op = unified_op(4, 3, bucket=8)
        st = A.structure_of(op)
        scope = dict(GLUON_SCOPE)
        regime = A.regime_of(op, st, scope)
        vec = A.features_for(regime, st, scope)
        assert isinstance(vec, A.Refusal)

    def test_a_call_with_no_offsets_still_refuses(self):
        """Still a refusal, and the reason moved rather than weakened.

        It used to be `bucket_pad`: the offsets were the only record of which
        rows were padding, and reading their absence as "no padding" was the
        failure it closed. The gluon decode law no longer separates launched
        rows from active ones -- it charges the launch -- so that particular
        reason is gone. The refusal is not: the split count the launcher asks
        for is computed from the sequence count, the offsets are where that
        count is, and a grid whose split axis is unknown is not a tile
        geometry. The other decode regime refuses on its own grounds.
        """
        op = unified_op(4, 3)
        op["context"] = [e for e in op["context"]
                         if e[0] != "cu_seqlens_q"]
        st = A.structure_of(op)
        scope = dict(GLUON_SCOPE)
        out = A.features_for(A.regime_of(op, st, scope), st, scope)
        assert isinstance(out, A.Refusal)
        assert "split count" in out.reason
        flash = A.REGIMES["unified.decode.unified_attn"]
        assert isinstance(A.features_for(flash, st, {}), A.Refusal)


def gluon_op(contexts):
    """A gluon paged decode over the given per-sequence histories."""
    rows = len(contexts)
    return {"name": A.UNIFIED,
            "input_shapes": [[rows, 24, 256], None, [rows, 4, 256],
                             [rows, 4, 256]],
            "context": [["context_lens", list(contexts)],
                        ["cu_seqlens_q", list(range(rows + 1))],
                        ["cu_seqlens_k", None],
                        ["max_seqlen_q", 1],
                        ["max_seqlen_k", max(contexts)],
                        ["is_prefill", False]]}


class TestTheGluonSplitGeometry:
    """`sum(ceil(C / 256))` is not what this kernel iterates.

    The launcher hands split ``j`` the page ``[ceil(C/S)*j, ceil(C/S)*(j+1))``
    and that page is covered by whole 256-token partitions of its own
    (`pa_decode_gluon.py`:1481-1491), so every page boundary that lands inside
    a partition is loaded twice. The split count itself is per call:
    ``min(8, ceil(compute_units * 2 / (sequences * num_kv_heads)))``
    (`attention_mha.py`:552).
    """

    def _tiles(self, contexts):
        """The summed tile count, taken off `Structure` rather than the regime.

        `split_tiles` is no longer one of the gluon decode regime's features --
        the summed form collides, see `TestTheLongestCTAIsItsOwnFact` -- but it
        is still the arithmetic the launcher does, and it is still the quantity
        `max_cta_tiles` decomposes, so its rounding stays tested here.
        """
        op = gluon_op(contexts)
        st = A.structure_of(op)
        splits = A._decode_splits(GLUON_SCOPE, st.sequences)
        assert not isinstance(splits, A.Refusal), splits
        return st.split_tiles(A.DECODE_PARTITION_SIZE, splits)

    def test_the_split_count_is_the_launchers_own(self):
        # 304 CUs at two workgroups each, over two sequences of four KV heads:
        # ceil(608 / 8) = 76, capped at 8.
        assert A._decode_splits(GLUON_SCOPE, 2) == 8
        # A wide batch falls below the cap: ceil(608 / (256 * 4)) = 1.
        assert A._decode_splits(GLUON_SCOPE, 256) == 1
        # The sliding-window branch is pinned to one by the caller.
        assert A._decode_splits({**GLUON_SCOPE, "sliding_window": 4096}, 2) == 1

    def test_two_batches_that_sum_alike_do_not_tile_alike(self):
        """The lead's case. Both sum to 18 partitions of 256; at eight splits
        one runs 25 tile iterations per KV head and the other 32."""
        assert self._tiles([2048, 2305]) == 25
        assert self._tiles([2176, 2177]) == 32

    def test_the_summed_partition_count_would_have_tied_them(self):
        """Stated explicitly, because it is the defect being fixed: the old
        feature gave both batches the same number and no fit over them could
        have told the two apart."""
        def summed(contexts):
            return sum(-(-c // 256) for c in contexts)

        assert summed([2048, 2305]) == summed([2176, 2177]) == 18
        assert self._tiles([2048, 2305]) != self._tiles([2176, 2177])

    def test_a_context_that_divides_evenly_pays_no_boundary(self):
        """Eight splits of 2048 are 256 each, so every page is one whole
        partition and the split form and the summed form agree."""
        assert self._tiles([2048]) == 8

    def test_an_undeclared_part_refuses_rather_than_assuming_a_grid(self):
        for absent in ("num_kv_heads", "compute_units"):
            scope = {k: v for k, v in GLUON_SCOPE.items() if k != absent}
            op = gluon_op([2048, 2305])
            st = A.structure_of(op)
            out = A.features_for(A.regime_of(op, st, scope), st, scope)
            assert isinstance(out, A.Refusal)
            assert absent in out.missing

    def test_the_declared_part_is_part_of_the_regimes_scope(self):
        regime = A.REGIMES["unified.decode.paged_gluon"]
        assert "num_kv_heads" in regime.required_scope
        assert "compute_units" in regime.required_scope


class TestTheLongestCTAIsItsOwnFact:
    """The summed basis collided, and these are the measured pairs it tied.

    Priced on node18 GPU2 at rotation 8 under snapshot 40ebae73, source tree
    `g4/src2p/tree`. Both pairs carry the same context rows, the same summed
    split tiles, the same launched rows and the same padding, and price 2.76x
    and 3.17x apart:

        r08geom / x08c04080   218.67 vs  79.34 us
        r16geom / x16c04080   497.26 vs 156.72 us

    `x08c04080` is eight uniform 4080-token rows; `r08geom` is the geometric
    ragged batch that sums to the same 32640. The sixteen-row pair is the same
    construction doubled.
    """

    RAGGED_8 = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    UNIFORM_8 = (4080,) * 8

    def _vector(self, contexts):
        op = gluon_op(list(contexts))
        st = A.structure_of(op)
        scope = dict(GLUON_SCOPE, compute_units=80)  # the measured part
        regime = A.regime_of(op, st, scope)
        assert not isinstance(regime, A.Refusal), regime
        vec = A.features_for(regime, st, scope)
        assert not isinstance(vec, A.Refusal), vec
        return regime, scope, st, vec

    def test_the_old_basis_gave_the_measured_pair_one_vector(self):
        old = A.Regime("old.summed", ("context_rows", "split_tiles", "active",
                                      "bucket_pad"), A.PAGED_GLUON_SCOPE)
        vectors = []
        for contexts in (self.RAGGED_8, self.UNIFORM_8):
            op = gluon_op(list(contexts))
            st = A.structure_of(op)
            scope = dict(GLUON_SCOPE, compute_units=80)
            vec = A.features_for(old, st, scope)
            assert not isinstance(vec, A.Refusal), vec
            vectors.append(vec)
        assert vectors[0] == vectors[1]
        assert vectors[0][:2] == [32640.0, 160.0]

    def test_the_new_basis_separates_them(self):
        longest = []
        for contexts in (self.RAGGED_8, self.UNIFORM_8):
            regime, _scope, _st, vec = self._vector(contexts)
            longest.append(vec[regime.features.index("max_cta_tiles")])
        assert longest == [14.0, 4.0]

    def test_the_per_cta_counts_sum_to_the_summed_feature(self):
        """The new term is a decomposition of the old one, not a new estimate
        of it: the same rounding, taken at its maximum instead of its sum."""
        for contexts in (self.RAGGED_8, self.UNIFORM_8, (2048, 2305),
                         (131072, 1024), (176128,)):
            st = A.structure_of(gluon_op(list(contexts)))
            scope = dict(GLUON_SCOPE, compute_units=80)
            splits = A._decode_splits(scope, st.sequences)
            per_cta = []
            for context in st.contexts():
                page = -(-int(context) // splits)
                for index in range(splits):
                    low = page * index
                    if low >= context:
                        break
                    high = min(int(context), low + page)
                    per_cta.append(-(-high // A.DECODE_PARTITION_SIZE)
                                   - low // A.DECODE_PARTITION_SIZE)
            assert sum(per_cta) == st.split_tiles(A.DECODE_PARTITION_SIZE,
                                                 splits)
            assert max(per_cta) == st.max_cta_tiles(A.DECODE_PARTITION_SIZE,
                                                    splits)

    def test_crit_waves_is_the_walk_times_the_machine_it_fills(self):
        regime, scope, st, vec = self._vector(self.RAGGED_8)
        splits = A._decode_splits(scope, st.sequences)
        ctas = st.executed_rows * scope["num_kv_heads"] * splits
        slots = scope["compute_units"] * A.DECODE_OCCUPANCY
        assert vec[regime.features.index("crit_waves")] == pytest.approx(
            14.0 * ctas / float(slots))
        assert vec[regime.features.index("calls")] == 1.0


class TestTheMakespanLawIsFittedAndBounded:
    """The law is `c0 + max(c_lat * max_cta_tiles, c_bw * crit_waves)`.

    Not a sum: the launch is concurrent, so the call ends when its last CTA
    ends, and the two terms are two bounds on that. Which one binds is decided
    by the fitted coefficients -- the crossover is `c_lat / c_bw` waves -- and
    not by a threshold anybody chose.
    """

    #: The measured part: MI308X, 80 CUs, 4 KV heads, bfloat16 NHD, block 16.
    SCOPE = dict(GLUON_SCOPE, compute_units=80, kv_cache_dtype="bfloat16",
                 kv_cache_layout="NHD", kv_cache_block_size=16)

    #: Seven of the twelve measured grid-2 training points, in microseconds.
    #: Source-oracle prices on node18 GPU2 at KV rotation 8 under snapshot
    #: 40ebae73; medians over three repeats. Widths deliberately mixed -- at a
    #: single launch width the two bounds are proportional and the fit refuses.
    POINTS = [((256,) * 8, 29.1777), ((4096,) * 8, 79.9579),
              ((4080,) * 8, 79.3375), ((16384,) * 8, 236.4947),
              ((128, 256, 512, 1024, 2048, 4096, 8192, 16384), 218.6679),
              ((32768,) * 2, 138.7911), ((8192,) * 32, 508.1840)]

    def _observation(self, contexts, microseconds):
        """One training observation, scoped the way `price` scopes a request.

        `scoped` folds the call's static operand geometry into the scope, and
        a fit whose scope was not folded the same way is a law `price` will
        not select -- so the fixture folds it here rather than passing the
        bare declared scope.
        """
        op = gluon_op(list(contexts))
        structure = A.structure_of(op)
        scope = A.scoped(op, dict(self.SCOPE), structure)
        return (structure, microseconds * 1e-6, "test", scope)

    def _fit(self):
        regime = A.REGIMES["unified.decode.paged_gluon"]
        return regime, A.fit_regime(regime, [self._observation(c, us)
                                             for c, us in self.POINTS])

    def _vector(self, regime, contexts):
        structure, _us, _src, scope = self._observation(contexts, 1.0)
        values = A.features_for(regime, structure, scope)
        assert not isinstance(values, A.Refusal), values
        return values

    def test_it_fits_and_recovers_both_bounds(self):
        regime, fit = self._fit()
        assert not isinstance(fit, A.Refusal), fit
        assert fit.regime.law == A.MAKESPAN
        assert fit.features == ("calls", "max_cta_tiles", "crit_waves")
        assert all(c > 0 for c in fit.coefficients)
        # The crossover is the ratio of the two slopes, and it lands where
        # the launch is around half occupancy -- not at a chosen threshold.
        crossover = fit.coefficients[1] / fit.coefficients[2]
        assert 0.2 < crossover < 1.0
        # A law over seven points, judged against its own residuals.
        assert fit.relative_error < 0.20
        # The collided pair is no longer one price.
        ragged = fit.predict(self._vector(
            regime, (128, 256, 512, 1024, 2048, 4096, 8192, 16384)))
        uniform = fit.predict(self._vector(regime, (4080,) * 8))
        assert ragged > 2 * uniform

    def test_one_launch_width_alone_cannot_identify_the_two_bounds(self):
        """At a fixed width `crit_waves` is `max_cta_tiles` times a constant,
        so which bound binds never changes and the split between them is
        arbitrary. That is a refusal, not a law with wide error bars."""
        regime = A.REGIMES["unified.decode.paged_gluon"]
        same_width = [((256,) * 8, 29.1777), ((4096,) * 8, 79.9579),
                      ((16384,) * 8, 236.4947), ((8192,) * 8, 134.4206)]
        out = A.fit_regime(regime, [self._observation(c, us)
                                    for c, us in same_width])
        assert isinstance(out, A.Refusal)
        assert "proportional" in out.reason

    def test_the_description_names_the_crossover(self):
        _regime, fit = self._fit()
        assert not isinstance(fit, A.Refusal), fit
        text = fit.describe()
        assert "max(" in text and "crossover" in text
        assert "max_cta_tiles" in text and "crit_waves" in text

    def test_a_regime_declared_makespan_needs_exactly_three_terms(self):
        with pytest.raises(ValueError):
            A.Regime("bad", ("calls", "max_cta_tiles"), law=A.MAKESPAN)
        with pytest.raises(ValueError):
            A.Regime("bad", ("calls",), law="guesswork")

    def test_a_price_outside_the_measured_walk_refuses(self):
        """The law is a fit over what was measured. A walk far longer than
        any behind it is unsupported, not extrapolated -- which is what keeps
        a long-context request from being answered by a short-context law."""
        regime, fit = self._fit()
        assert not isinstance(fit, A.Refusal), fit
        model = A.Model()
        model.fits[A._label(regime.name, A.scope_key(fit.scope))] = fit
        out = model.price(gluon_op([262144] * 8), dict(self.SCOPE))
        assert isinstance(out, A.Refusal)
        assert "outside the measured range" in out.reason

    def test_a_structure_inside_the_hull_prices(self):
        regime, fit = self._fit()
        assert not isinstance(fit, A.Refusal), fit
        model = A.Model()
        model.fits[A._label(regime.name, A.scope_key(fit.scope))] = fit
        out = model.price(gluon_op([8192] * 8), dict(self.SCOPE))
        assert not isinstance(out, A.Refusal), out
        # 134.42us measured at this point; the law is within a quarter of it.
        assert abs(out * 1e6 - 134.4206) / 134.4206 < 0.25


def gdn_op(rows, active, *, context=CONTEXT, mode="full"):
    """One GDN decode call, recorded the way the named mode records it.

    The two modes differ in every count, and the previous fixture here was
    neither: it wrote ``num_decodes = num_decode_tokens = rows`` with
    ``num_actual_tokens = active``, a B/B/A combination the engine does not
    produce.

    FULL (`gdn_attn.py`:1264-1281). The capture freezes ``num_decodes =
    num_decode_tokens = num_actual_tokens = bs`` into the graph, and the
    replay refills the buffers around those frozen scalars (:1189-1235): the
    query offsets repeat the last real one and the state-index tail is
    ``PAD_SLOT_ID``. So a bucket of four holding three requests records
    ``[0, 1, 2, 3, 3]`` and ``[0, 1, 2, -1]`` with all three counts at four.
    There is no tail to zero -- ``core_attn_out[num_actual_tokens:]`` is an
    empty slice when ``num_actual_tokens`` is the whole width.

    PIECEWISE. Attention runs eagerly (`model_runner.py`:4019-4031) and the
    counts are the active ones, with an offset view of length ``active + 1``
    and a state-index tensor of length ``active``. The output tensor is still
    allocated at the bucket width, so here the zeroed tail is real.
    """
    if mode == "full":
        starts = list(range(active + 1)) + [active] * (rows - active)
        slots = list(range(active)) + [-1] * (rows - active)
        counts = (rows, rows, rows)
    elif mode == "piecewise":
        starts = list(range(active + 1))
        slots = list(range(active))
        counts = (active, active, active)
    else:
        raise AssertionError(f"no native metadata recorded for {mode!r}")
    return {"name": A.GDN,
            "input_shapes": [[rows, 10240], [rows, 48], [rows, 48],
                             [rows, 48, 128]],
            "context": [["num_prefills", 0],
                        ["num_prefill_tokens", 0],
                        ["non_spec_query_start_loc", starts],
                        ["non_spec_state_indices_tensor", slots],
                        ["num_decodes", counts[0]],
                        ["num_decode_tokens", counts[1]],
                        ["num_actual_tokens", counts[2]],
                        ["num_spec_decodes", 0],
                        ["num_spec_decode_tokens", 0],
                        ["replayssm", False],
                        ["context_lens", [context] * active
                         + [0] * (rows - active)],
                        ["is_prefill", False]]}


class TestTheRecurrentTailIsUnchanged:
    """What a GDN decode call costs is three widths, not one.

    The convolution skips ``PAD_SLOT_ID`` lanes and the recurrence skips
    zero-length ones, so the state work follows the *active* lanes. The gating
    and the output copy run over ``num_actual_tokens`` -- `attention_gdn.py`
    slices ``a`` and ``b`` to it and copies ``output[:num_actual_tokens]`` --
    which is the bucket under FULL and the active rows under PIECEWISE. The
    rest of the allocation is only zeroed, and that is the third term. A
    single `active` term made 3-of-4 and 4-of-4 the same point.
    """

    @pytest.mark.parametrize("rows,active", [(4, 3), (32, 31), (4, 4)])
    def test_a_full_replay_prices_active_lanes_against_processed_rows(
            self, rows, active):
        op = gdn_op(rows, active)
        st = A.structure_of(op)
        assert st.executed_rows == rows
        assert st.state_lanes == active
        vec = A.features_for(A.regime_of(op, st, {}), st, {})
        assert not isinstance(vec, A.Refusal), vec
        # No zeroed tail under FULL: `num_actual_tokens` is the bucket.
        assert vec == [1.0, float(active), float(rows), 0.0]

    def test_three_of_four_and_four_of_four_are_different_points(self):
        """The collision the single `active` term produced.

        Both launch four rows; one runs three recurrences and the other four,
        and a fit cannot separate them if they key the same.
        """
        def vec(active):
            op = gdn_op(4, active)
            st = A.structure_of(op)
            return A.features_for(A.regime_of(op, st, {}), st, {})

        assert vec(3) == [1.0, 3.0, 4.0, 0.0]
        assert vec(4) == [1.0, 4.0, 4.0, 0.0]
        # Both process the bucket under FULL; only the lane count separates
        # them, which is the point of the second term.
        assert vec(3) != vec(4)

    def test_piecewise_records_the_active_counts_and_a_real_zeroed_tail(self):
        """Same batch, same bucket, different mode -- and it must key apart.

        PIECEWISE runs the same three lanes, gates and copies three rows, and
        zeroes the fourth. `tail_pad_rows` is what says so -- and the processed
        term is three, not four, or a PIECEWISE step would be priced above a
        FULL one that gates strictly more rows.
        """
        op = gdn_op(4, 3, mode="piecewise")
        st = A.structure_of(op)
        assert st.executed_rows == 4 and st.num_actual_tokens == 3
        vec = A.features_for(A.regime_of(op, st, {}), st, {})
        assert vec == [1.0, 3.0, 3.0, 1.0]
        full = gdn_op(4, 3)
        stf = A.structure_of(full)
        full_vec = A.features_for(A.regime_of(full, stf, {}), stf, {})
        assert vec != full_vec
        # The ordering the allocated width got backwards: FULL processes four
        # rows here and PIECEWISE three, so the processed term must not tie.
        processed = A.REGIMES["gdn.decode"].features.index("actual_rows")
        assert vec[processed] < full_vec[processed]

    def test_the_allocated_width_is_still_recoverable(self):
        """Splitting the width did not lose it: gated plus zeroed is the
        allocation, under either mode."""
        regime = A.REGIMES["gdn.decode"]
        processed = regime.features.index("actual_rows")
        tail = regime.features.index("tail_pad_rows")
        for mode in ("full", "piecewise"):
            op = gdn_op(4, 3, mode=mode)
            st = A.structure_of(op)
            vec = A.features_for(regime, st, {})
            assert vec[processed] + vec[tail] == float(st.executed_rows)

    def test_a_padded_lane_is_skipped_rather_than_counted(self):
        """`gdn_attn.py`:1224-1226. The tail indexes no state, so it is not a
        fourth recurrence -- and it is not slot zero's either."""
        op = gdn_op(4, 3)
        ctx = dict(map(tuple, op["context"]))
        assert ctx["non_spec_state_indices_tensor"] == [0, 1, 2, -1]
        assert A.structure_of(op).state_lanes == 3

    def test_the_lane_count_and_the_offsets_must_agree(self):
        """Two recordings of the same step that disagree are not one step."""
        op = gdn_op(4, 3)
        for entry in op["context"]:
            if entry[0] == "non_spec_state_indices_tensor":
                entry[1] = [0, 1, 2, 3]
        st = A.structure_of(op)
        out = A.features_for(A.regime_of(op, st, {}), st, {})
        assert isinstance(out, A.Refusal)
        assert "different steps" in out.reason

    def test_no_full_history_term_appears(self):
        regime = A.REGIMES["gdn.decode"]
        assert "history_rows" not in regime.features
        assert "context_rows" not in regime.features
        assert "split_tiles" not in regime.features


class TestTheProviderPathCarriesTheMode:
    """The binder is not the entry point. `TemplateGraphs.graph_for` is what a
    served step calls, and it binds -- on the cold return as well as every warm
    one after it. A fixture that passes ``extent_scope`` by hand proves the
    binder's rule and nothing about whether production reaches it.
    """

    def _template_from(self, spec):
        """The two operators a decode derivation writes, not just the MHA one.

        The GDN operator is here because its metadata is padded under a
        different rule from attention's: `non_spec_query_start_loc` is
        `A + 1 + pad` under FULL and `A + 1` under PIECEWISE. A template
        carrying only `aiter::unified_attention_with_output_base` exercises the
        extent rule and says nothing about that one, and a bind that required
        the padded length in both modes refused every correctly derived
        PIECEWISE decode on the cold return.
        """
        ctx = dict(spec.attention_context())
        gdn = dict(spec.gdn_context())
        return {"ops": [{"name": A.UNIFIED,
                         "input_shapes": [[spec.padded_rows, 24, 256]],
                         "context": [[k, ctx[k]] for k in
                                     ("context_lens", "cu_seqlens_q",
                                      "max_seqlen_q", "max_seqlen_k")]},
                        {"name": A.GDN,
                         "input_shapes": [[spec.padded_rows, 10240]],
                         "context": [[k, gdn[k]] for k in
                                     ("num_decodes", "num_decode_tokens",
                                      "num_actual_tokens",
                                      "non_spec_query_start_loc",
                                      "non_spec_state_indices_tensor")]}],
                # What `ModelTracer.provenance` writes: the spec the graph
                # was derived from, mode included.
                "provenance": {"region": "body",
                               "batch_spec": spec.to_dict()}}

    def _graphs(self, mode, *, derived_mode=None, seeded=None):
        """A cache whose deriver builds the template the declared mode implies.

        ``derived_mode`` defaults to ``mode``: the deriver and the cache take
        the same declared mode in production, and the defect this guards is
        what happens when the bind forgets it.
        """
        from atom.compass.runtime.templates import TemplateGraphs

        built = derived_mode if derived_mode is not None else mode
        self.derived = []

        def derive(shape):
            spec = spec_of(len(shape.num_scheduled_tokens),
                           bucket=shape.capture_bucket, mode=built)
            self.derived.append(spec)
            return self._template_from(spec)

        return TemplateGraphs(seeded, derive=derive,
                              allocation=CarriedAllocation("structural"),
                              cudagraph_mode=mode)

    def _extent(self, bound):
        return dict(bound["ops"][0]["context"])["max_seqlen_k"]

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_a_cold_full_derivation_is_not_rebound_to_the_batch(self, n,
                                                                bucket):
        graphs = self._graphs("full")
        bound = graphs.graph_for(shape(n, bucket=bucket))
        assert graphs.derivations == 1 and not graphs.refusals
        # The derivation wrote the capture's extent; the bind that follows it
        # in the same call must not overwrite it with the cohort's.
        assert self._extent(bound) == MAX_MODEL_LEN

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_the_warm_return_answers_the_same(self, n, bucket):
        graphs = self._graphs("full")
        first = graphs.graph_for(shape(n, bucket=bucket))
        # A different cohort of the same structure: a hit, and the bind is the
        # only thing that runs.
        second = graphs.graph_for(shape(n, context=4096, bucket=bucket))
        assert graphs.derivations == 1 and graphs.hits == 1
        assert self._extent(first) == self._extent(second) == MAX_MODEL_LEN

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_a_piecewise_step_follows_the_cohort_cold_and_warm(self, n,
                                                               bucket):
        graphs = self._graphs("piecewise")
        cold = graphs.graph_for(shape(n, bucket=bucket))
        warm = graphs.graph_for(shape(n, context=4096, bucket=bucket))
        assert graphs.derivations == 1 and graphs.hits == 1
        assert self._extent(cold) == CONTEXT
        assert self._extent(warm) == 4096

    def test_an_eager_step_follows_the_cohort(self):
        graphs = self._graphs("full")
        bound = graphs.graph_for(shape(3, context=4096))
        assert self._extent(bound) == 4096

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_an_undeclared_mode_refuses_at_the_provider(self, n, bucket):
        graphs = self._graphs(None)
        s = shape(n, bucket=bucket)
        assert graphs.graph_for(s) is None
        key = template_key(s)
        # Now that the template carries a GDN operator the refusal comes from
        # the derivation rather than the bind -- `gdn_context` has no shape to
        # write for a bucketed decode whose mode is undeclared, and neither
        # candidate is right under both. Either way the provider answers None
        # with a recorded reason; what it must not do is raise past the caller
        # for one operator family and refuse for another.
        assert "was not declared" in graphs.refusals[key]
        # And it refuses on the warm path too, not only the cold one.
        graphs.derivations = 0
        assert graphs.graph_for(s) is None

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_the_attention_only_refusal_is_still_the_binds(self, n, bucket):
        """With no GDN operator in the template there is nothing for the
        derivation to refuse, and the bind is what names the undeclared mode.
        Both stages have to answer the same way."""
        from atom.compass.runtime.templates import TemplateGraphs

        def derive(_shape):
            spec = spec_of(n, bucket=bucket, mode="piecewise")
            ctx = dict(spec.attention_context())
            return {"ops": [{"name": A.UNIFIED,
                             "context": [[k, ctx[k]] for k in
                                         ("context_lens", "cu_seqlens_q",
                                          "max_seqlen_q", "max_seqlen_k")]}],
                    "provenance": {"batch_spec": spec.to_dict()}}

        graphs = TemplateGraphs(None, derive=derive,
                                allocation=CarriedAllocation("structural"),
                                cudagraph_mode=None)
        s = shape(n, bucket=bucket)
        assert graphs.graph_for(s) is None
        assert "--cudagraph-mode was not declared" in \
            graphs.refusals[template_key(s)]

    def test_a_template_traced_under_another_mode_is_refused(self):
        """A seeded template was traced by some other run, and the key it is
        stored under says nothing about that run's mode. Under FULL the binder
        keeps the template's own extent, so a PIECEWISE trace would supply its
        longest history as a captured one."""
        s = shape(3, bucket=4)
        seeded = {template_key(s): self._template_from(
            spec_of(3, bucket=4, mode="piecewise"))}
        graphs = self._graphs("full", seeded=seeded)
        assert graphs.graph_for(s) is None
        assert "traced under cudagraph_mode" in graphs.refusals[template_key(s)]

    def test_a_full_traced_template_is_served(self):
        s = shape(3, bucket=4)
        seeded = {template_key(s): self._template_from(
            spec_of(3, bucket=4, mode="full"))}
        graphs = self._graphs("full", seeded=seeded)
        bound = graphs.graph_for(s)
        assert graphs.derivations == 0 and graphs.hits == 1
        assert self._extent(bound) == MAX_MODEL_LEN

    def test_the_cache_says_which_mode_it_binds_under(self):
        assert "cudagraph_mode full" in self._graphs("full").describe()
        assert "no cudagraph_mode declared" in self._graphs(None).describe()


class TestTheFactoryHandsTheModeToTheCache:
    """`seeded_graphs` is the only place a served run's `TemplateGraphs` is
    built. If the mode stops here the provider path is undeclared however the
    command line was written."""

    def test_seeded_graphs_passes_the_declared_mode_through(self):
        from atom.compass.runtime.source_oracle import seeded_graphs

        cache = seeded_graphs([], None, CarriedAllocation("structural"),
                              cudagraph_mode="full")
        assert cache._cudagraph_mode == "full"
        assert cache.graph_for(shape(3, bucket=4)) is None   # no deriver
        assert "no template and no deriver" in "".join(cache.refusals.values())

    def test_the_factory_signature_accepts_it_from_a_command_line(self):
        import inspect

        from atom.compass.runtime.source_oracle import build_source_oracle

        params = inspect.signature(build_source_oracle).parameters
        assert "cudagraph_mode" in params


class TestAnActiveSeedHasToQualifyForFULL:
    """A seed is served verbatim under FULL. Its label is not enough.

    ``agent_scratch/g4/src1/b27dec32.tp1.r0.json``, the template the registered
    TP1 composition passes as ``--compass-oracle-option template=``, records
    ``cudagraph_mode: "full"``, ``capture_bucket: 32``, ``max_model_len:
    262144`` -- and ``max_seqlen_k: 1151``, the longest history of the 32
    requests it holds. It was derived before the extent rule was fixed, so it
    carries the eager value under a FULL label. Checking the label alone lets
    that through, and under ``"captured"`` the binder keeps it: a wrong
    ``max_seqlen_k`` in the operator identity key, with correct-looking
    provenance over it.

    The check is the seed against itself. Its provenance holds the whole
    `BatchSpec`, and `BatchSpec.launch_max_seqlen_k` is the one rule that says
    what that spec's extent is. Nothing is relabelled here -- a seed that
    disagrees with its own declaration is refused, and the remedy is to
    re-derive it.
    """

    #: The active seed's shape, as recorded: a full bucket, so no padding is
    #: involved and the extent is the only thing in question.
    ACTIVE_N = 32
    ACTIVE_BUCKET = 32

    def _seeded(self, *, mode, extent=None, drop_mode=False):
        spec = spec_of(self.ACTIVE_N, bucket=self.ACTIVE_BUCKET,
                       mode=mode, context=CONTEXT)
        ctx = dict(spec.attention_context())
        if extent is not None:
            ctx["max_seqlen_k"] = extent
        declared = spec.to_dict()
        if drop_mode:
            declared.pop("cudagraph_mode", None)
        return {"ops": [{"name": A.UNIFIED,
                         "input_shapes": [[spec.padded_rows, 24, 256]],
                         "context": [[k, ctx[k]] for k in
                                     ("context_lens", "cu_seqlens_q",
                                      "max_seqlen_q", "max_seqlen_k")]}],
                "provenance": {"region": "body", "batch_spec": declared}}

    def _graphs(self, mode, template):
        from atom.compass.runtime.templates import TemplateGraphs

        s = shape(self.ACTIVE_N, CONTEXT,
                  bucket=self.ACTIVE_BUCKET)
        graphs = TemplateGraphs({template_key(s): template}, derive=None,
                                allocation=CarriedAllocation("structural"),
                                cudagraph_mode=mode)
        return graphs, s

    def test_the_active_seeds_recorded_extent_is_the_eager_one(self):
        """What the file holds, restated as the derivation would produce it
        under the old rule: the batch's longest history, not max_model_len."""
        spec = spec_of(32, bucket=32, mode="piecewise")
        assert dict(spec.attention_context())["max_seqlen_k"] == CONTEXT
        assert spec.max_model_len == MAX_MODEL_LEN

    def test_a_full_labelled_seed_carrying_the_eager_extent_is_refused(self):
        graphs, s = self._graphs(
            "full", self._seeded(mode="full", extent=CONTEXT))
        assert graphs.graph_for(s) is None
        reason = graphs.refusals[template_key(s)]
        assert "Re-derive it" in reason and str(CONTEXT) in reason

    def test_a_seed_with_no_declared_mode_is_refused_not_assumed(self):
        graphs, s = self._graphs(
            "full", self._seeded(mode="full", drop_mode=True))
        assert graphs.graph_for(s) is None
        assert "does not say which cudagraph_mode" in \
            graphs.refusals[template_key(s)]

    def test_a_seed_derived_under_the_fixed_rule_is_served(self):
        """The remedy, asserted rather than assumed: re-derived through the
        source derivation path, the same seed binds."""
        graphs, s = self._graphs("full", self._seeded(mode="full"))
        bound = graphs.graph_for(s)
        assert bound is not None and not graphs.refusals
        assert dict(bound["ops"][0]["context"])["max_seqlen_k"] == MAX_MODEL_LEN

    def test_an_unqualified_seed_still_serves_a_piecewise_deployment(self):
        """The refusal is scoped to where the value is kept verbatim. Under
        PIECEWISE the extent is recomputed from the cohort, so an old seed is
        not evidence of anything the bind relies on."""
        graphs, s = self._graphs(
            "piecewise", self._seeded(mode="piecewise", drop_mode=True))
        bound = graphs.graph_for(s)
        assert bound is not None and not graphs.refusals
        assert dict(bound["ops"][0]["context"])["max_seqlen_k"] == CONTEXT


class TestTheDeclaredModeReachesTheInstalledBatch:
    """The two deployment inputs arrive on the command line and are read off
    the batch. `graph_diff.py trace` installs the spec for the trace and
    records the request in the provenance, so a spec silent on both was traced
    under the eager rule while its provenance said ``full`` -- which is
    exactly the pair of facts `_check_traced_as_captured` refuses, and exactly
    how the active TP1 seeds were produced.
    """

    @staticmethod
    def _module():
        import importlib.util
        import sys
        from pathlib import Path

        path = (Path(__file__).resolve().parents[2] / "scripts" / "compass"
                / "graph_diff.py")
        spec = importlib.util.spec_from_file_location("compass_graph_diff",
                                                      path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _args(mode=None, bucket=None):
        return argparse.Namespace(cudagraph_mode=mode, capture_bucket=bucket)

    @staticmethod
    def _spec(mode=None, bucket=None):
        return BatchSpec(kind="decode", query_lens=(1,) * 4,
                         context_lens=(CONTEXT,) * 4, capture_bucket=bucket,
                         cudagraph_mode=mode, **DECLARED)

    def test_a_flag_lands_on_a_silent_spec(self):
        """The defect itself: before this the flag reached the provenance and
        the spec kept the eager rule, so the recorded extent was the batch's
        longest context under a FULL label."""
        module = self._module()
        spec = self._spec()
        assert spec.launch_max_seqlen_k == CONTEXT
        reconciled, why = module.reconcile_declared(
            spec, self._args(mode="full", bucket=4))
        assert why is None
        assert reconciled.cudagraph_mode == "full"
        assert reconciled.capture_bucket == 4
        assert reconciled.launch_max_seqlen_k == MAX_MODEL_LEN

    def test_a_spec_that_declares_them_is_left_alone(self):
        module = self._module()
        reconciled, why = module.reconcile_declared(
            self._spec(mode="full", bucket=4), self._args(mode="full",
                                                          bucket=4))
        assert why is None
        assert reconciled.launch_max_seqlen_k == MAX_MODEL_LEN

    def test_a_silent_flag_takes_the_spec_s_value_onto_the_request(self):
        """Or the provenance records null for something the batch declared."""
        module = self._module()
        args = self._args()
        reconciled, why = module.reconcile_declared(
            self._spec(mode="full", bucket=4), args)
        assert why is None
        assert args.cudagraph_mode == "full" and args.capture_bucket == 4
        assert reconciled.launch_max_seqlen_k == MAX_MODEL_LEN

    @pytest.mark.parametrize("flag,spec_kwargs,args_kwargs", [
        ("cudagraph-mode", {"mode": "piecewise"}, {"mode": "full"}),
        ("capture-bucket", {"bucket": 8}, {"bucket": 4}),
    ])
    def test_a_contradiction_is_refused_rather_than_resolved(
            self, flag, spec_kwargs, args_kwargs):
        """Neither one wins. The trace would install the spec and the
        provenance would record the flag, and a reader cannot tell which
        produced the extent."""
        module = self._module()
        _, why = module.reconcile_declared(self._spec(**spec_kwargs),
                                           self._args(**args_kwargs))
        assert why is not None and flag in why

    def test_the_reconciled_spec_is_the_one_the_active_seeds_needed(self):
        """The regenerated TP1 seeds, in miniature: 32 decode rows at the
        bucket they replay, declared FULL, priced at the engine's extent."""
        module = self._module()
        spec = BatchSpec(kind="decode", query_lens=(1,) * 32,
                         context_lens=(CONTEXT,) * 32, **DECLARED)
        reconciled, why = module.reconcile_declared(
            spec, self._args(mode="full", bucket=32))
        assert why is None
        assert dict(reconciled.attention_context())["max_seqlen_k"] == \
            MAX_MODEL_LEN
        assert reconciled.launch_extent_scope == "captured"


#: The reduce kernel the paged decode launches, as the compiler names it: one
#: instantiation per split count. Abbreviated in the argument list only --
#: `_canonical_kernel` reads the TEMPLATE list, which is verbatim.
def reduce_kernel(splits):
    return ("void aiter::pa_decode_ps_reduce_hip_kernel<__hip_bfloat16, "
            "__hip_bfloat16, __hip_bfloat16, false, 256, 6, %d>"
            "(__hip_bfloat16*, float const*, int)" % splits)


class TestTheSplitSpecializationIsNotATreatment:
    """Four instantiations of one kernel are one kernel.

    The launcher picks the split count from the launch geometry --
    `min(8, ceil(compute_units * 2 / (rows * num_kv_heads)))` -- and the
    compiler emits a reduce kernel per split count. Kept in the measurement
    identity, that specialization files every row width under its own law, and
    inside one row width `crit_waves` is exactly proportional to
    `max_cta_tiles`: the makespan law is then unidentifiable by construction
    and no further measurement can fix it. So the integer template arguments
    come out, and what they were is reported rather than dropped.
    """

    def test_the_split_count_comes_out_of_the_symbol(self):
        from atom.compass.core.cost.families import adapter

        canonical = {adapter._canonical_kernel(reduce_kernel(splits))
                     for splits in (2, 3, 5, 8)}
        assert len(canonical) == 1
        assert "256" not in canonical.pop().split(">")[0]

    def test_a_kernel_nobody_declared_specialized_is_untouched(self):
        """Only the named symbol. Anything else keeps every argument it has,
        including the integers -- a cache dtype enum is not a launch width."""
        from atom.compass.core.cost.families import adapter

        other = ("void aiter::reshape_and_cache_kernel<std::bfloat16_t, "
                 "(vllm::Fp8KVCacheDataType)0, true>(int, int, 256)")
        assert adapter._canonical_kernel(other) == other

    def test_the_type_arguments_stay(self):
        """A bfloat16 instantiation and an fp8 one are different work, and
        pooling them would be the mistake this is preventing elsewhere."""
        from atom.compass.core.cost.families import adapter

        canonical = adapter._canonical_kernel(reduce_kernel(8))
        assert "__hip_bfloat16" in canonical
        assert adapter._canonical_kernel(
            reduce_kernel(8).replace("__hip_bfloat16", "__hip_fp8")) \
            != canonical


class TestADeclarationFillsASilenceOnly:
    """A price file that does not record its deployment, and a caller who does.

    `scripts/compass/primitives.py` records the pools it stood up and the
    kernel its dispatch probe saw, in provenance sections the adapter does not
    read as a scope. Without a declaration every ragged decode observation in
    such a file is refused for want of a declared backend and the family fits
    over nothing. With one, the silences are filled -- and only the silences.
    """

    DECLARED = {"unified": dict(GLUON_SCOPE, kv_cache_dtype="bfloat16",
                                kv_cache_layout="NHD", kv_cache_block_size=16)}

    def _library(self, declared=None):
        from atom.compass.core.cost.families import ParametricPriceLibrary
        from atom.compass.core.cost.families.attention_scope import \
            declaration_of

        library = ParametricPriceLibrary()
        if declared is not None:
            library.declared_attention_scope = declaration_of(
                declared, where="the test declaration")
        return library

    def test_a_silence_is_filled(self):
        library = self._library(self.DECLARED)
        filled = library._with_declared_scope({}, gluon_op([256]), "p.json")
        assert filled["attention_backend"] == "paged_gluon"
        assert filled["num_kv_heads"] == 4
        # The two facts the paged decode law is identified by beyond its
        # family's own scope. Dropping either leaves the law refused for want
        # of a fact the declaration stated.
        assert filled["compute_units"] == 304

    def test_a_stated_fact_is_not_overruled(self):
        library = self._library(dict(self.DECLARED))
        with pytest.raises(ValueError) as raised:
            library._with_declared_scope(
                {"attention_backend": "aiter_mha"}, gluon_op([256]), "p.json")
        assert "not overrule" in str(raised.value)

    def test_agreement_is_not_an_error(self):
        library = self._library(self.DECLARED)
        filled = library._with_declared_scope(
            {"attention_backend": "paged_gluon"}, gluon_op([256]), "p.json")
        assert filled["attention_backend"] == "paged_gluon"

    def test_no_declaration_changes_nothing(self):
        library = self._library()
        assert library._with_declared_scope({}, gluon_op([256]), "p.json") == {}

    def test_the_declaration_is_reported(self):
        """A law fitted over declared facts is worth exactly what the
        declaration is, so coverage says there was one."""
        library = self._library(self.DECLARED)
        coverage = library.attention_coverage()
        scopes = coverage["declared_measured_scope"]["scopes"]
        assert scopes["unified"]["attention_backend"] == "paged_gluon"
        assert self._library().attention_coverage()[
            "declared_measured_scope"] is None


class TestTheOracleCarriesBothEndsOfTheDeployment:
    """`attention_scope` is what a price is ASKED for, `measured_attention_
    scope` what a silent price list was TAKEN in. Different questions, one
    shape, and neither is read as the other."""

    DECLARED = {"unified": dict(GLUON_SCOPE, kv_cache_dtype="bfloat16",
                                kv_cache_layout="NHD", kv_cache_block_size=16)}

    def test_a_measured_scope_reaches_the_library(self):
        from atom.compass.runtime.source_oracle import (_DEFAULT_GAP_RATIO,
                                                        _price_library)

        library = _price_library([], _DEFAULT_GAP_RATIO,
                                 measured_attention_scope=self.DECLARED)
        assert library.declared_attention_scope is not None
        assert not library.request_attention_scope

    def test_a_measured_scope_without_modelling_is_refused(self):
        """Only a fitted family reads it. Accepting it with modelling off
        would take a declaration that changes nothing and report success."""
        from atom.compass.runtime.source_oracle import _price_library

        with pytest.raises(ValueError) as raised:
            _price_library([], None, measured_attention_scope=self.DECLARED)
        assert "Turn the family provider on" in str(raised.value)

    def test_the_two_ends_are_independent(self):
        from atom.compass.runtime.source_oracle import (_DEFAULT_GAP_RATIO,
                                                        _price_library)

        library = _price_library([], _DEFAULT_GAP_RATIO,
                                 attention_scope=self.DECLARED)
        assert library.request_attention_scope is not None
        assert library.declared_attention_scope is None
