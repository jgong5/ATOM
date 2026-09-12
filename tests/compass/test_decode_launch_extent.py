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
from atom.compass.runtime.templates import BindRefusal, _bind, bind_cohort
from atom.compass.runtime.tracer import ShapeDeriver

MAX_MODEL_LEN = 262144
DECLARED = {"block_size": 16, "max_model_len": MAX_MODEL_LEN,
            "position_rows": 3}
CONTEXT = 1151


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
        scope = {"attention_backend": "paged_gluon", "sliding_window": -1}
        regime = A.regime_of(op, st, scope)
        assert not isinstance(regime, A.Refusal), regime
        vec = A.features_for(regime, st, scope)
        assert not isinstance(vec, A.Refusal), vec
        assert vec[regime.features.index("bucket_pad")] == float(pad)
        assert vec[regime.features.index("active")] == float(rows)

    def test_a_declared_bucket_that_contradicts_the_rows_refuses(self):
        op = unified_op(4, 3, bucket=8)
        st = A.structure_of(op)
        scope = {"attention_backend": "paged_gluon", "sliding_window": -1}
        regime = A.regime_of(op, st, scope)
        vec = A.features_for(regime, st, scope)
        assert isinstance(vec, A.Refusal)

    def test_a_call_with_no_offsets_still_refuses(self):
        op = unified_op(4, 3)
        op["context"] = [e for e in op["context"]
                         if e[0] != "cu_seqlens_q"]
        st = A.structure_of(op)
        scope = {"attention_backend": "paged_gluon", "sliding_window": -1}
        vec = A.features_for(A.regime_of(op, st, scope), st, scope)
        assert isinstance(vec, A.Refusal)


def gdn_op(rows, active, *, context=CONTEXT):
    starts = list(range(active + 1)) + [active] * (rows - active)
    return {"name": A.GDN,
            "input_shapes": [[rows, 10240], [rows, 48], [rows, 48],
                             [rows, 48, 128]],
            "context": [["num_prefills", 0],
                        ["num_prefill_tokens", 0],
                        ["non_spec_query_start_loc", starts],
                        ["non_spec_state_indices_tensor",
                         list(range(active)) + [-1] * (rows - active)],
                        ["num_decodes", rows],
                        ["num_decode_tokens", rows],
                        ["num_actual_tokens", active],
                        ["num_spec_decodes", 0],
                        ["num_spec_decode_tokens", 0],
                        ["replayssm", False],
                        ["context_lens", [context] * active
                         + [0] * (rows - active)],
                        ["is_prefill", False]]}


class TestTheRecurrentTailIsUnchanged:
    """GDN prices off its own output operand and `num_actual_tokens`, and
    nothing here touches either. The regression to avoid is a decode-side
    change moving a number the recurrence already got right."""

    @pytest.mark.parametrize("rows,active,tail", [(4, 3, 1), (32, 31, 1),
                                                  (4, 4, 0)])
    def test_the_zeroed_tail_is_still_the_allocated_width_minus_the_slice(
            self, rows, active, tail):
        op = gdn_op(rows, active)
        st = A.structure_of(op)
        assert st.executed_rows == rows
        regime = A.regime_of(op, st, {})
        vec = A.features_for(regime, st, {})
        assert not isinstance(vec, A.Refusal), vec
        assert vec == [1.0, float(rows), float(tail)]

    def test_no_full_history_term_appears(self):
        regime = A.REGIMES["gdn.decode"]
        assert "history_rows" not in regime.features
        assert "context_rows" not in regime.features
        assert "context_tiles" not in regime.features
