"""An ordinary padded decode, derived on demand and then reused.

Three requests replaying a bucket of four is the commonest step a PoC run
takes, and it could not be priced. Every layer was individually defensible:
``ShapeDeriver`` built a spec of the three requests the scheduler had,
``model_inputs`` allocated a row per token of it, ``execution_record`` reported
``body_rows_traced=3`` against ``body_rows_executed=4``, and
``LibraryCostOracle`` refused the pair -- correctly, because a graph of three
rows is not a graph of what the step runs. The refusal was the only part that
was right.

What the older tests could not catch is *why*: they hand a template in already
padded, so nothing in them ever asks a deriver for one. These go through the
real ``ShapeDeriver`` and the real ``TemplateGraphs``, cold (a miss that
derives) and warm (a hit that binds), at 3 -> 4 and 31 -> 32, with an unpadded
control at each stage so that the padding is shown to be a consequence of the
bucket and not of these tests.

The model is stood in for -- there is no 27B here -- but nothing about row
counts is. ``model_inputs`` really allocates the tensors, ``_head_input``
really narrows the hidden states, ``head_placement`` and ``execution_record``
really write the metadata, and the row count the oracle checks against is
``library.executed_body_rows`` itself rather than a number repeated here.
"""

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.library import executed_body_rows
from atom.compass.runtime import tracer as tracer_mod
from atom.compass.runtime.batch_spec import PAD_SLOT_ID, BatchSpec, model_inputs
from atom.compass.runtime.templates import (CarriedAllocation, TemplateGraphs,
                                            bind_cohort, template_key)
from atom.compass.runtime.tracer import (BuildRefusal, ShapeDeriver,
                                         TraceRequest, execution_record,
                                         head_placement, head_rows_for)

#: The 27B deployment's declared rules, as a caller supplies them.
DECLARED = {"block_size": 16, "max_model_len": 262144,
            "position_rows": 3}


def shape(n, context=1151, *, bucket=None, tp=1, rank=0):
    """``n`` decoding requests, optionally replaying ``bucket``."""
    return StepShape(
        num_scheduled_tokens=tuple([1] * n),
        context_lens=tuple([context] * n),
        num_prefill_tokens=0,
        topology={"tp": tp}, rank_coords={"tp": rank},
        capture_bucket=bucket, compiled=None, produces_output=True)


def spec_of(n, context=1151, *, bucket=None, cudagraph_mode="full", **kw):
    """A derived spec. FULL by default: a bucketed decode has no metadata
    shape that is right under both capture modes, so the mode is declared
    here rather than left for `gdn_context` to refuse."""
    return ShapeDeriver(None, cudagraph_mode=cudagraph_mode,
                        **{**DECLARED, **kw}).spec_for(
        shape(n, context, bucket=bucket))


class TestTheSpecKnowsItsPadding:
    """``BatchSpec`` is where the bucket becomes a row count."""

    def test_three_requests_in_a_bucket_of_four_run_four_rows(self):
        spec = spec_of(3, bucket=4)
        assert spec.batch_size == 3            # logical, unchanged
        assert spec.num_tokens == 3            # logical, unchanged
        assert spec.running_bs == 4
        assert spec.padded_rows == 4
        assert spec.is_padded

    def test_thirty_one_in_a_bucket_of_thirty_two(self):
        spec = spec_of(31, bucket=32)
        assert (spec.batch_size, spec.num_tokens) == (31, 31)
        assert spec.padded_rows == 32

    def test_an_eager_step_pads_nothing(self):
        spec = spec_of(3)
        assert spec.running_bs == 3 and spec.padded_rows == 3
        assert not spec.is_padded

    def test_a_batch_that_lands_on_a_rung_pads_nothing(self):
        """Having a bucket is not the same as having padding."""
        spec = spec_of(4, bucket=4)
        assert spec.padded_rows == 4 and not spec.is_padded

    def test_the_row_count_is_the_oracle_s_own(self):
        """Producer and consumer must not compute this two ways."""
        for n, bucket in ((3, 4), (31, 32), (4, 4), (3, None)):
            assert (spec_of(n, bucket=bucket).padded_rows
                    == executed_body_rows(shape(n, bucket=bucket)))


class TestWhatThePaddedRowsCarry:
    """Each buffer's tail, against the value the native side writes.

    Not one padding value: a row that reads no history, a token that maps to no
    KV slot and a request that owns no recurrent state are three different
    statements, and the engine writes three different numbers.
    """

    def test_the_attention_tail_is_the_native_contract(self):
        ctx = dict(spec_of(3, context=66, bucket=4).attention_context())
        # model_runner.py:2649-2652 -- padded rows repeat the last offset, so
        # each is an empty sequence rather than a fourth request.
        assert ctx["cu_seqlens_q"] == [0, 1, 2, 3, 3]
        # aiter_attention.py:1115 -- a padded row holds no history.
        assert ctx["context_lens"] == [66, 66, 66, 0]
        # aiter_attention.py:1106 -- the tail maps to no slot, and the real
        # rows keep the allocation's own values. Computed from spec.tables()
        # rather than written out, so a block-policy change fails a block
        # policy test and not this one.
        spec = spec_of(3, context=66, bucket=4)
        block, tables = spec.block_size, spec.tables()
        real = [tables[i][65 // block] * block + 65 % block for i in range(3)]
        assert ctx["slot_mapping"][:3] == real
        assert ctx["slot_mapping"][3:] == [PAD_SLOT_ID]
        # `max_seqlen_q` is the scheduled batch's, one token per decode row.
        assert ctx["max_seqlen_q"] == 1
        # `max_seqlen_k` is not. This bucket is replayed from a FULL capture,
        # and `aiter_attention.py:1367` froze the declared `max_model_len`
        # into that graph's metadata; the replay does not rewrite it.
        assert ctx["max_seqlen_k"] == 262144
        # Under PIECEWISE the same batch runs attention eagerly
        # (`model_runner.py:4019-4031`), so `prepare_decode` recomputes it from
        # the live histories (`aiter_attention.py:1100`) and the real rows'
        # longest is what the kernel sees.
        eager = dict(spec_of(3, context=66, bucket=4,
                             cudagraph_mode="piecewise").attention_context())
        assert eager["max_seqlen_k"] == 66
        assert ctx["block_tables_shape"][0] == 4

    def test_the_state_index_tail_is_pad_slot_id_and_never_zero(self):
        """gdn_attn.py:1224-1226. Zero is request zero's state entry, so a row

        padded with it reads and writes a live sequence's recurrent state.
        """
        gdn = dict(spec_of(3, bucket=4).gdn_context())
        for field in ("non_spec_state_indices_tensor",
                      "non_spec_state_indices_in_tensor"):
            values, dtype = gdn[field]
            assert values == [0, 1, 2, PAD_SLOT_ID], field
            assert dtype == "int32"
        # gdn_attn.py:1231-1233, and the capture's own counts at :1264-1281.
        assert gdn["non_spec_query_start_loc"][0] == [0, 1, 2, 3, 3]
        assert gdn["num_decodes"] == 4
        assert gdn["num_decode_tokens"] == 4
        assert gdn["num_actual_tokens"] == 4

    def test_an_unpadded_decode_is_untouched(self):
        ctx = dict(spec_of(3, context=66).attention_context())
        gdn = dict(spec_of(3, context=66).gdn_context())
        assert ctx["cu_seqlens_q"] == [0, 1, 2, 3]
        assert ctx["context_lens"] == [66, 66, 66]
        assert PAD_SLOT_ID not in ctx["slot_mapping"]
        assert gdn["non_spec_state_indices_tensor"][0] == [0, 1, 2]
        assert gdn["num_decodes"] == 3

    def test_thirty_one_in_thirty_two(self):
        ctx = dict(spec_of(31, context=66, bucket=32).attention_context())
        gdn = dict(spec_of(31, context=66, bucket=32).gdn_context())
        assert len(ctx["context_lens"]) == 32 and ctx["context_lens"][-1] == 0
        assert len(ctx["cu_seqlens_q"]) == 33
        assert ctx["cu_seqlens_q"][-2:] == [31, 31]
        assert len(ctx["slot_mapping"]) == 32
        assert ctx["slot_mapping"][-1] == PAD_SLOT_ID
        assert gdn["non_spec_state_indices_tensor"][0][-1] == PAD_SLOT_ID


class TestWhatTheModelIsHanded:
    """The tensors, which is where the traced row count comes from."""

    def test_the_forward_is_the_bucket_wide(self):
        torch = pytest.importorskip("torch")
        ids, pos = model_inputs(spec_of(3, context=66, bucket=4,
                                        position_rows=1), device="cpu")
        assert ids.shape == (4,) and ids.dtype == torch.int32
        # Zeros: a legal vocab id and a legal position, which is what the
        # runner writes into its own tail (model_runner.py:583-600, 3199-3203).
        assert ids.tolist() == [0, 0, 0, 0]
        assert pos.tolist() == [65, 65, 65, 0]

    def test_mrope_pads_inside_each_section(self):
        """The buffer is [3, num_tokens_pad], so the pad is in every row --

        not three real sections followed by a fourth of padding.
        """
        pytest.importorskip("torch")
        _, pos = model_inputs(spec_of(3, context=66, bucket=4), device="cpu")
        assert pos.shape == (3, 4)
        assert pos.tolist() == [[65, 65, 65, 0]] * 3

    def test_the_flattened_rows_are_what_the_context_records(self):
        pytest.importorskip("torch")
        spec = spec_of(3, context=66, bucket=4)
        _, pos = model_inputs(spec, device="cpu")
        assert pos.flatten().tolist() == dict(spec.attention_context())[
            "positions"]

    def test_an_unpadded_decode_gets_its_own_rows(self):
        pytest.importorskip("torch")
        ids, _ = model_inputs(spec_of(3, context=66), device="cpu")
        assert ids.shape == (3,)


class _Graph:
    """The little a derived graph has to be for a cache to store it."""

    def __init__(self, ops):
        self.ops = ops
        self.key = None
        self.provenance = None

    def to_dict(self):
        return {"ops": self.ops, "key": self.key,
                "provenance": self.provenance}


class _StandInTracer:
    """A ``ModelTracer`` without the 27B, and without faking what is tested.

    ``trace`` is the real one: it calls the real ``model_inputs``, the real
    ``head_rows_for``, and assembles the real provenance. Only the forward is
    replaced, by a ``trace_regions`` that takes its row counts from the tensors
    it is handed exactly as the real one does (``input_ids.shape[0]``) and runs
    the real ``_head_input`` over stand-in hidden states. Operators are not
    what these tests are about; row counts are, and every row count here is
    computed by the code under test.
    """

    def __init__(self, monkeypatch, tp=1):
        import torch

        from atom.compass.runtime.tracer import ModelTracer

        self.tp = tp
        self.rank = 0
        self.arch = "Qwen3NextForCausalLM"
        self.model_path = "/models/Qwen3.8-27B"
        self.model = object()
        self.device = torch.device("meta")
        self.config = type("C", (), {"torch_dtype": torch.bfloat16})()
        self.traces = 0
        self.trace_seconds = 0.0
        self.rank_rebinds = 0
        self.rank_modules_rebound = 0
        self.specs = []
        self.trace = ModelTracer.trace.__get__(self)
        self.set_rank = lambda rank: 0
        self.provenance = ModelTracer.provenance.__get__(self)
        monkeypatch.setattr(tracer_mod, "trace_regions", self._regions)

    def _regions(self, model, input_ids, positions, topology=None,
                 on_meta=False, spec=None, region="body", head_rows=None):
        import torch

        self.specs.append(spec)
        rows = int(input_ids.shape[0])
        notes = {} if region == "head" else {"body_rows": rows}
        if region in ("head", "both"):
            hidden = torch.zeros(rows, 8, device="meta")
            notes["hidden_rows"] = int(
                tracer_mod._head_input(hidden, head_rows).shape[0])
        notes["deaths_stamped"] = 0
        return _Graph([{"name": "aten::detach", "input_shapes": "1"}]), 0.0, 0, notes


def _deriver(monkeypatch, *, tp=1, region="body", cudagraph_mode="full"):
    tracer = _StandInTracer(monkeypatch, tp=tp)
    return tracer, ShapeDeriver(tracer, region=region,
                                cudagraph_mode=cudagraph_mode, **DECLARED)


class TestDerivingOnDemand:
    """A cold ``TemplateGraphs`` miss: the thing the PoC run actually does."""

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_the_derived_body_is_the_rows_the_step_executes(self, monkeypatch,
                                                            n, bucket):
        pytest.importorskip("torch")
        _, derive = _deriver(monkeypatch)
        graphs = TemplateGraphs(derive=derive,
                                allocation=CarriedAllocation("structural"))
        s = shape(n, bucket=bucket)
        assert template_key(s) not in graphs._templates      # cold
        bound = graphs.graph_for(s)
        assert bound is not None and graphs.derivations == 1

        record = bound["provenance"]["execution"]
        assert record["rows_real"] == n                       # logical
        assert record["capture_bucket"] == bucket
        assert record["body_rows_traced"] == bucket
        assert record["body_rows_executed"] == bucket
        # The guard the oracle applies, applied here against the same function.
        assert record["body_rows_traced"] == executed_body_rows(s)

    def test_the_logical_batch_survives_the_padding(self, monkeypatch):
        pytest.importorskip("torch")
        tracer, derive = _deriver(monkeypatch)
        TemplateGraphs(derive=derive,
                       allocation=CarriedAllocation("structural")).graph_for(
            shape(3, bucket=4))
        spec = tracer.specs[-1]
        assert spec.query_lens == (1, 1, 1)
        assert spec.context_lens == (1151, 1151, 1151)
        assert len(spec.tables()) == 3

    def test_an_unpadded_decode_derives_its_own_rows(self, monkeypatch):
        pytest.importorskip("torch")
        _, derive = _deriver(monkeypatch)
        s = shape(3)
        bound = TemplateGraphs(
            derive=derive,
            allocation=CarriedAllocation("structural")).graph_for(s)
        record = bound["provenance"]["execution"]
        assert record["body_rows_traced"] == 3
        assert record["body_rows_executed"] == 3
        assert record["body_rows_traced"] == executed_body_rows(s)


class TestBindingKeepsThePadding:
    """A warm hit. The template is padded; binding must not narrow it again.

    ``template_key`` carries the bucket and the query lengths, so the template
    and the cohort agree on the padding by construction -- what differs is the
    contexts, which is exactly what binding recomputes.
    """

    def _template(self, n, context, bucket):
        ctx = dict(spec_of(n, context, bucket=bucket).attention_context())
        gdn = dict(spec_of(n, context, bucket=bucket).gdn_context())
        op = {"name": "aiter::unified_attention_with_output_base",
              "input_shapes": "1,2", "dtypes": "bfloat16",
              "context": [[k, ctx[k]] for k in
                          ("context_lens", "positions", "cu_seqlens_q",
                           "max_seqlen_q", "max_seqlen_k", "cu_seqlens_k",
                           "slot_mapping")]}
        linear = {"name": "aiter::fused_recurrent", "input_shapes": "1",
                  "dtypes": "bfloat16",
                  "context": [[k, gdn[k]] for k in
                              ("non_spec_query_start_loc",
                               "non_spec_state_indices_tensor",
                               "num_decodes")]}
        return {"ops": [op, linear],
                # The spec these two were derived from, mode included. Under
                # the captured rule the binder keeps the template's own extent,
                # so it qualifies the seed against this before it does.
                "provenance": {"region": "body",
                               "batch_spec": spec_of(n, context,
                                                     bucket=bucket).to_dict()}}

    @pytest.mark.parametrize("n,bucket", [(3, 4), (31, 32)])
    def test_a_bound_cohort_keeps_the_bucket_s_width(self, n, bucket):
        template = self._template(n, 66, bucket)
        # Same structure, different histories: what binding is for.
        cohort = shape(n, 4096, bucket=bucket)
        # The template is a FULL derivation -- `spec_of` declares it -- so it
        # binds under the rule it was written for. Under the batch rule its
        # padded GDN offsets are one row per request and this refuses.
        bound = bind_cohort(template, cohort, CarriedAllocation("structural"),
                            extent_scope="captured")
        ctx = {k: v for k, v in bound["ops"][0]["context"]}
        gdn = {k: v for k, v in bound["ops"][1]["context"]}

        assert len(ctx["context_lens"]) == bucket
        assert ctx["context_lens"] == [4096] * n + [0] * (bucket - n)
        assert len(ctx["cu_seqlens_q"]) == bucket + 1
        assert ctx["cu_seqlens_q"][-1] == ctx["cu_seqlens_q"][n] == n
        # M-RoPE: three sections of the padded width, not four of the real one.
        assert len(ctx["positions"]) == bucket * 3
        assert ctx["positions"][:bucket] == [4095] * n + [0] * (bucket - n)
        assert gdn["non_spec_query_start_loc"][0][-1] == n
        assert len(gdn["non_spec_query_start_loc"][0]) == bucket + 1
        # Carried, not rebound -- and the tail is still the pad.
        assert ctx["slot_mapping"][-1] == PAD_SLOT_ID
        assert gdn["non_spec_state_indices_tensor"][0][-1] == PAD_SLOT_ID
        assert bound["provenance"]["binding"]["replay_pad_rows"] == bucket - n

    def test_an_unpadded_cohort_binds_unpadded(self):
        template = self._template(3, 66, None)
        bound = bind_cohort(template, shape(3, 4096),
                            CarriedAllocation("structural"))
        ctx = {k: v for k, v in bound["ops"][0]["context"]}
        assert ctx["context_lens"] == [4096, 4096, 4096]
        assert ctx["cu_seqlens_q"] == [0, 1, 2, 3]
        assert len(ctx["positions"]) == 9
        assert bound["provenance"]["binding"]["replay_pad_rows"] == 0

    def test_the_warm_hit_matches_the_cold_derivation(self, monkeypatch):
        """Two cohorts of one structure: derived once, bound once, same width."""
        pytest.importorskip("torch")
        _, derive = _deriver(monkeypatch)
        graphs = TemplateGraphs(derive=derive,
                                allocation=CarriedAllocation("structural"))
        first = graphs.graph_for(shape(3, 1151, bucket=4))
        second = graphs.graph_for(shape(3, 2048, bucket=4))
        assert graphs.derivations == 1 and graphs.hits == 1
        for bound in (first, second):
            assert bound["provenance"]["execution"]["body_rows_traced"] == 4


class TestTheHeadHasItsOwnWidth:
    """The body's rows are not the head's, and which is which turns on TP."""

    def _request(self, tp, bucket, mode="full", region="head"):
        return TraceRequest(tp=tp, rank=0, region=region, cudagraph_mode=mode,
                            capture_bucket=bucket, tokens=3,
                            model="/models/Qwen3.8-27B")

    def test_tp1_full_projects_the_padded_bucket(self):
        """model_runner.py:4104, 4293-4300 -- the capture holds the head and

        projects all four rows; :3238-3239 slices the logits afterwards.
        """
        spec = spec_of(3, bucket=4)
        args = self._request(1, 4)
        assert head_rows_for(args, spec) == 4
        placement = head_placement(args, spec, {"hidden_rows": 4})
        assert placement["in_replayed_body_graph"] is True
        assert placement["rows_padded_to_capture_bucket"] is True

    @pytest.mark.parametrize("tp", [2, 4])
    def test_a_wider_deployment_projects_the_scheduled_rows(self, tp):
        """model_runner.py:3235, 3241 -- the hidden states are sliced to the

        real count first and an eager head projects those.
        """
        spec = spec_of(3, bucket=4)
        args = self._request(tp, 4)
        assert head_rows_for(args, spec) == 3
        placement = head_placement(args, spec, {"hidden_rows": 3})
        assert placement["in_replayed_body_graph"] is False
        assert placement["rows_padded_to_capture_bucket"] is False

    def test_piecewise_at_tp1_projects_the_scheduled_rows(self):
        """The dense pieces self-capture; the head is not among them

        (model_runner.py:3228-3230), so it is the batch's width at every TP.
        """
        spec = spec_of(3, bucket=4)
        assert head_rows_for(self._request(1, 4, mode="piecewise"), spec) == 3

    def test_an_eager_step_pads_nothing_at_any_width(self):
        spec = spec_of(3)
        for tp in (1, 2, 4):
            args = self._request(tp, None)
            assert head_rows_for(args, spec) == 3
            assert head_placement(args, spec, {})[
                "rows_padded_to_capture_bucket"] is False

    def test_an_undeclared_cudagraph_mode_refuses(self):
        """Both answers are wrong and both look plausible, so neither is taken."""
        spec = spec_of(3, bucket=4)
        args = self._request(1, 4, mode=None)
        assert head_placement(args, spec, {})[
            "rows_padded_to_capture_bucket"] is None
        with pytest.raises(BuildRefusal, match="cudagraph mode"):
            head_rows_for(args, spec)

    def test_the_head_region_is_traced_at_that_width(self, monkeypatch):
        pytest.importorskip("torch")
        for tp, expected in ((1, 4), (2, 3)):
            _, derive = _deriver(monkeypatch, tp=tp, region="head")
            graphs = TemplateGraphs(derive=derive,
                                    allocation=CarriedAllocation("structural"))
            bound = graphs.graph_for(shape(3, bucket=4, tp=tp))
            record = bound["provenance"]["execution"]
            assert record["head_rows_traced"] == expected, tp
            assert bound["provenance"]["head_placement"][
                "rows_into_compute_logits"] == expected

    def test_the_head_cannot_be_wider_than_the_body(self, monkeypatch):
        torch = pytest.importorskip("torch")
        with pytest.raises(BuildRefusal, match="did not compute"):
            tracer_mod._head_input(torch.zeros(3, 8), 4)


class TestTheExecutionRecordCountsRows:
    """``body_rows_executed`` is rows, not requests."""

    def test_a_speculative_decode_multiplies_by_the_query_length(self):
        """Three requests verifying four tokens each in a bucket of four is a

        sixteen-row forward, and the bucket alone says four.
        """
        spec = BatchSpec(kind="decode", query_lens=(4, 4, 4),
                         context_lens=(66, 66, 66), block_size=16,
                         max_model_len=4096, capture_bucket=4, num_spec_step=3)
        args = TraceRequest(tp=1, rank=0, region="body", cudagraph_mode="full",
                            capture_bucket=4, tokens=12, model="m")
        record = execution_record(args, spec, {"body_rows": 16})
        assert record["rows_real"] == 12
        assert record["capture_bucket"] == 4
        assert record["body_rows_executed"] == 16
        assert record["body_rows_executed"] == executed_body_rows(
            StepShape(num_scheduled_tokens=(4, 4, 4),
                      context_lens=(66, 66, 66), capture_bucket=4))
