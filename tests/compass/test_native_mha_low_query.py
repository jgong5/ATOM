from copy import deepcopy

import pytest

from atom.compass.core.cost.native_mha_low_query import NativeMhaLowQueryModel, coordinates, sharing_pattern
from atom.compass.runtime.batch_spec import BatchSpec


def operator(queries=(2, 1), history=(512, 1024)):
    total = sum(queries)
    spec = BatchSpec(kind="prefill", query_lens=queries,
                     context_lens=tuple(q + h for q, h in zip(queries, history)),
                     block_size=16, max_model_len=262144, block_policy="packed")
    return dict(name="aiter::unified_attention_with_output_base", group=None,
        input_shapes=[[total, 6144], [total, 1024], [total, 1024]], dtypes=["bfloat16"] * 3,
        output_shapes=[[total, 6144]], output_dtypes=["bfloat16"], output_aliases=[None],
        layouts=[[2, [[14336, 1], 13312, total * 14336, 2]]],
        scalars=[["#1", None], ["#4", None], ["#5", "language_model.model.layers.3.self_attn"],
                 ["#6", False], ["#7", None]], context=[list(entry) for entry in spec.attention_context()])


def model_for(op):
    row = coordinates(op)
    group = f'N{row["n"]}_causal'
    model = NativeMhaLowQueryModel.__new__(NativeMhaLowQueryModel)
    model.models = {group: dict(coefficients_seconds=[.001, .01, .02],
        measured_feature_intervals=dict(sum_context=[1, 200000], max_context=[1, 200000], max_query=[2, 16]))}
    model.patterns = {group: {sharing_pattern(op, row)}}
    model.allowed_rows = set(zip(row["q"], row["history"]))
    model.handoff_sha256 = "a" * 64
    model.source_model = {"path": "frozen.json", "sha256": "b" * 64}
    return model


def test_multirow_law_counts_both_total_and_maximum_context():
    op = operator()
    quote = model_for(op).quote(op)
    assert quote["seconds"] == pytest.approx(.001 + .01 * 1539 / 100000 + .02 * 1025 / 100000)
    assert quote["model_provenance"]["whole_forward_validation_required"]


def test_cumulative_starts_are_refused_even_when_the_table_access_can_stay_in_bounds():
    op = operator()
    model = model_for(op)
    next(entry for entry in op["context"] if entry[0] == "seq_starts")[1] = [0, 512]
    assert coordinates(op) is None
    assert model.quote(op) is None


def test_unmeasured_feature_range_is_not_extrapolated():
    op = operator()
    model = model_for(op)
    model.models["N2_causal"]["measured_feature_intervals"]["sum_context"] = [2000, 3000]
    assert model.quote(op) is None


def test_other_prompt_rows_are_not_implicitly_added_to_the_domain():
    model = model_for(operator())
    assert model.quote(operator((2, 1), (528, 1024))) is None


def test_q1_range_and_dense_v_do_not_receive_native_causal_quotes():
    model = model_for(operator())
    assert model.quote(operator((1,), (1024,))) is None
    op = operator()
    op["layouts"] = []
    assert model.quote(op) is None


def test_unmeasured_aliases_are_not_treated_as_equivalent_allocations():
    op = operator()
    model = model_for(op)
    changed = deepcopy(op)
    context = dict(changed["context"])
    width = len(context["block_tables"]) // 2
    context["block_tables"][width] = 6  # Alias an interior block of the first row.
    assert sharing_pattern(changed, coordinates(changed)) is None
    assert model.quote(changed) is None


def test_native_v_backing_capacity_and_void_writes_remain_guarded():
    op = operator()
    model = model_for(op)
    changed = deepcopy(op)
    changed["layouts"][0][1][2] += 1
    assert model.quote(changed) is None
    next(entry for entry in op["context"] if entry[0] == "slot_mapping")[1][0] = -1
    assert model.quote(op) is None


@pytest.mark.parametrize("change", ["duplicate_tail", "prefix_overlap", "wrong_slots"])
def test_query_writes_must_match_distinct_non_prefix_tail_blocks(change):
    op = operator()
    model = model_for(op)
    context = dict(op["context"])
    width = len(context["block_tables"]) // 2
    if change == "wrong_slots":
        context["slot_mapping"][0] -= 16
    else:
        block = context["block_tables"][32] if change == "duplicate_tail" else 6
        context["block_tables"][width + 64] = block
        context["slot_mapping"][-1] = block * 16
    assert model.quote(op) is None
