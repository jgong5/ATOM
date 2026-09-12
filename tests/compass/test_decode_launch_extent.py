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
        template = {"ops": [{"name": A.UNIFIED,
                             "context": [["max_seqlen_k", MAX_MODEL_LEN],
                                         ["max_seqlen_q", 1]]}]}
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

    @pytest.mark.parametrize("rows,active,pad", [(4, 3, 1), (32, 31, 1),
                                                 (4, 4, 0), (32, 32, 0)])
    def test_the_padded_rows_are_priced_as_a_term(self, rows, active, pad):
        op = unified_op(rows, active)
        st = A.structure_of(op)
        scope = dict(GLUON_SCOPE)
        regime = A.regime_of(op, st, scope)
        assert not isinstance(regime, A.Refusal), regime
        vec = A.features_for(regime, st, scope)
        assert not isinstance(vec, A.Refusal), vec
        assert vec[regime.features.index("bucket_pad")] == float(pad)
        assert vec[regime.features.index("active")] == float(rows)

    def test_a_declared_bucket_that_contradicts_the_rows_refuses(self):
        op = unified_op(4, 3, bucket=8)
        st = A.structure_of(op)
        scope = dict(GLUON_SCOPE)
        regime = A.regime_of(op, st, scope)
        vec = A.features_for(regime, st, scope)
        assert isinstance(vec, A.Refusal)

    def test_a_call_with_no_offsets_still_refuses(self):
        op = unified_op(4, 3)
        op["context"] = [e for e in op["context"]
                         if e[0] != "cu_seqlens_q"]
        st = A.structure_of(op)
        scope = dict(GLUON_SCOPE)
        vec = A.features_for(A.regime_of(op, st, scope), st, scope)
        assert isinstance(vec, A.Refusal)


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
        op = gluon_op(contexts)
        st = A.structure_of(op)
        regime = A.regime_of(op, st, GLUON_SCOPE)
        assert not isinstance(regime, A.Refusal), regime
        vec = A.features_for(regime, st, GLUON_SCOPE)
        assert not isinstance(vec, A.Refusal), vec
        return vec[regime.features.index("split_tiles")]

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
    and the output copy run over the allocated width, so that work follows the
    *bucket*. Under PIECEWISE the kernel additionally zeroes the rows between
    the two. A single `active` term made 3-of-4 and 4-of-4 the same point.
    """

    @pytest.mark.parametrize("rows,active", [(4, 3), (32, 31), (4, 4)])
    def test_a_full_replay_prices_active_lanes_against_bucket_rows(
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
        assert vec(3) != vec(4)

    def test_piecewise_records_the_active_counts_and_a_real_zeroed_tail(self):
        """Same batch, same bucket, different mode -- and it must key apart.

        PIECEWISE runs the same three lanes but allocates four rows and zeroes
        the fourth. `tail_pad_rows` is what says so.
        """
        op = gdn_op(4, 3, mode="piecewise")
        st = A.structure_of(op)
        assert st.executed_rows == 4 and st.num_actual_tokens == 3
        vec = A.features_for(A.regime_of(op, st, {}), st, {})
        assert vec == [1.0, 3.0, 4.0, 1.0]
        full = gdn_op(4, 3)
        stf = A.structure_of(full)
        assert vec != A.features_for(A.regime_of(full, stf, {}), stf, {})

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
        """A one-op template carrying exactly what a derivation would write."""
        ctx = dict(spec.attention_context())
        return {"ops": [{"name": A.UNIFIED,
                         "input_shapes": [[spec.padded_rows, 24, 256]],
                         "context": [[k, ctx[k]] for k in
                                     ("context_lens", "cu_seqlens_q",
                                      "max_seqlen_q", "max_seqlen_k")]}],
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
        assert "cudagraph-mode" in graphs.refusals[key]
        # And it refuses on the warm path too, not only the cold one.
        graphs.derivations = 0
        assert graphs.graph_for(s) is None

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
