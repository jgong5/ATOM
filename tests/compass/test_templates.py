"""Binding a template to a cohort, and everything it refuses to guess.

The claim these tests defend is narrow and worth stating: a graph derived for
one cohort can be re-pointed at another of the same *structure* by rewriting
per-request metadata, and anything outside that rule raises rather than
producing a graph that looks right. The whole-cohort evidence is elsewhere --
five cohorts compared field by field against independently derived ground truth
-- and what is here is the rules that evidence rests on, checked without a
model.
"""

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.runtime.templates import (ALLOCATOR_FIELDS, BindRefusal,
                                            CarriedAllocation, TemplateGraphs,
                                            bind_cohort, template_key)

#: Three M-RoPE sections, as the 27B lays positions out. Kept as a name because
#: a bare 3 in a slice length is the kind of constant that goes unexplained.
POSITION_ROWS = 3


def shape(queries, contexts, *, bucket=32, tp=1, rank=0, prefill=0):
    return StepShape(
        num_scheduled_tokens=tuple(queries),
        context_lens=tuple(contexts),
        num_prefill_tokens=prefill,
        topology={"tp": tp}, rank_coords={"tp": rank},
        capture_bucket=bucket, compiled=None, produces_output=True)


def attention_op(rows, *, extra=(), allocator=True):
    """One operator carrying the context an attention call records."""
    positions = []
    for q, c in rows:
        positions.extend(range(c - q, c))
    context = [
        ["context_lens", [c for _, c in rows]],
        ["positions", positions * POSITION_ROWS],
        ["max_seqlen_k", max(c for _, c in rows)],
        ["max_seqlen_q", max(q for q, _ in rows)],
        ["min_seqlen_q", 0],
        ["cu_seqlens_k", None],
    ]
    if allocator:
        context.append(["slot_mapping", list(range(len(rows)))])
        context.append(["block_tables", [[0]] * len(rows)])
    context.extend(list(e) for e in extra)
    return {"name": "aiter::unified_attention_with_output_base",
            "input_shapes": "1,2", "dtypes": "bfloat16", "context": context}


def template_for(rows, **kw):
    return {"ops": [{"name": "aten::detach", "input_shapes": "1"},
                    attention_op(rows, **kw)],
            "provenance": {"region": "body"}}


def carried():
    return CarriedAllocation("test; allocation is not the subject")


# -- the key ----------------------------------------------------------------

def test_template_key_drops_context_and_keeps_everything_else():
    a = shape([1] * 4, [1151] * 4)
    b = shape([1] * 4, [131072] * 4)
    assert template_key(a) == template_key(b)

    assert template_key(a) != template_key(shape([1] * 8, [1151] * 8))
    assert template_key(a) != template_key(shape([1] * 4, [1151] * 4,
                                                 bucket=16))
    assert template_key(a) != template_key(shape([1] * 4, [1151] * 4, rank=1))


def test_a_heterogeneous_cohort_shares_a_uniform_cohorts_template():
    """The case the whole method turns on: contexts differ, structure does not."""
    assert (template_key(shape([1] * 4, [1151] * 4))
            == template_key(shape([1] * 4, [811, 4096, 1151, 65536])))


# -- binding ----------------------------------------------------------------

def test_binding_rewrites_the_cohorts_metadata():
    rows = [(1, 1151)] * 4
    bound = bind_cohort(template_for(rows),
                        shape([1] * 4, [811, 4096, 1151, 65536]), carried())
    ctx = dict(map(tuple, bound["ops"][1]["context"]))
    assert ctx["context_lens"] == [811, 4096, 1151, 65536]
    assert ctx["max_seqlen_k"] == 65536
    assert ctx["max_seqlen_q"] == 1
    # One position per token, the three M-RoPE sections laid end to end.
    assert ctx["positions"] == [810, 4095, 1150, 65535] * POSITION_ROWS


def test_binding_leaves_operators_without_context_untouched():
    bound = bind_cohort(template_for([(1, 1151)] * 4),
                        shape([1] * 4, [4096] * 4), carried())
    assert bound["ops"][0] == {"name": "aten::detach", "input_shapes": "1"}


def test_binding_does_not_mutate_the_template():
    template = template_for([(1, 1151)] * 4)
    before = dict(map(tuple, template["ops"][1]["context"]))
    bind_cohort(template, shape([1] * 4, [4096] * 4), carried())
    assert dict(map(tuple, template["ops"][1]["context"])) == before


def test_min_seqlen_q_is_carried_not_recomputed():
    """It is a literal 0 in the producer, not ``min(query_lens)``.

    ``aiter_attention.py:1071`` and ``backends.py:532`` both write 0. A rule
    that recomputed it from the cohort would write 1 for every decode and be
    wrong on every one of them, which is how this was found.
    """
    bound = bind_cohort(template_for([(1, 1151)] * 4),
                        shape([1] * 4, [4096] * 4), carried())
    assert dict(map(tuple, bound["ops"][1]["context"]))["min_seqlen_q"] == 0


def test_multi_token_queries_get_a_position_run_each():
    rows = [(4, 100), (4, 200)]
    bound = bind_cohort(template_for(rows), shape([4, 4], [64, 512]),
                        carried())
    ctx = dict(map(tuple, bound["ops"][1]["context"]))
    assert ctx["positions"] == ([60, 61, 62, 63, 508, 509, 510, 511]
                                * POSITION_ROWS)


# -- what it refuses --------------------------------------------------------

def test_binding_refuses_a_template_with_allocation_and_no_source():
    with pytest.raises(BindRefusal) as exc:
        bind_cohort(template_for([(1, 1151)] * 4), shape([1] * 4, [4096] * 4))
    message = str(exc.value)
    assert "AllocationSource" in message
    for field in ALLOCATOR_FIELDS:
        assert field in message


def test_a_template_without_allocation_needs_no_source():
    bound = bind_cohort(template_for([(1, 1151)] * 4, allocator=False),
                        shape([1] * 4, [4096] * 4))
    assert bound["provenance"]["binding"]["allocation"] == "none needed"


def test_binding_refuses_a_context_field_it_has_no_rule_for():
    template = template_for([(1, 1151)] * 4,
                            extra=[["seqlen_agnostic_thing", [1, 2, 3, 4]]])
    with pytest.raises(BindRefusal) as exc:
        bind_cohort(template, shape([1] * 4, [4096] * 4), carried())
    assert "seqlen_agnostic_thing" in str(exc.value)


def test_binding_refuses_a_positions_layout_it_does_not_recognise():
    template = template_for([(1, 1151)] * 4)
    context = dict(map(tuple, template["ops"][1]["context"]))
    # Five entries for four tokens: not a whole number of sections.
    context["positions"] = [1, 2, 3, 4, 5]
    template["ops"][1]["context"] = [[k, v] for k, v in context.items()]
    with pytest.raises(BindRefusal) as exc:
        bind_cohort(template, shape([1] * 4, [4096] * 4), carried())
    assert "section layout" in str(exc.value)


def test_binding_refuses_a_non_spec_start_loc_that_is_not_the_whole_batch():
    """Speculative decoding changes the structure, not the cohort."""
    template = template_for(
        [(1, 1151)] * 4,
        extra=[["non_spec_query_start_loc", [[0, 1, 2], "int32"]]])
    with pytest.raises(BindRefusal) as exc:
        bind_cohort(template, shape([1] * 4, [4096] * 4), carried())
    assert "whole batch" in str(exc.value)


def test_non_spec_start_loc_over_the_whole_batch_binds():
    template = template_for(
        [(1, 1151)] * 4,
        extra=[["non_spec_query_start_loc", [[0, 1, 2, 3, 4], "int32"]]])
    bound = bind_cohort(template, shape([1] * 4, [4096] * 4), carried())
    ctx = dict(map(tuple, bound["ops"][1]["context"]))
    assert ctx["non_spec_query_start_loc"] == [[0, 1, 2, 3, 4], "int32"]


# -- provenance -------------------------------------------------------------

def test_carried_allocation_is_recorded_as_unmeasured():
    bound = bind_cohort(template_for([(1, 1151)] * 4),
                        shape([1] * 4, [4096] * 4), carried())
    binding = bound["provenance"]["binding"]
    assert binding["allocation_measured"] is False
    assert "unmeasured" in binding["allocation"]
    assert binding["rows"] == [(1, 4096)] * 4
    assert binding["operators_rebound"] == 1
    assert "min_seqlen_q" in binding["carried_constants"]


def test_binding_keeps_the_templates_own_provenance():
    bound = bind_cohort(template_for([(1, 1151)] * 4),
                        shape([1] * 4, [4096] * 4), carried())
    assert bound["provenance"]["region"] == "body"


# -- the cache --------------------------------------------------------------

def test_template_graphs_binds_a_hit_and_counts_it():
    cache = TemplateGraphs(allocation=carried())
    cache.add(shape([1] * 4, [1151] * 4), template_for([(1, 1151)] * 4))

    graph = cache.graph_for(shape([1] * 4, [65536] * 4))
    assert graph is not None
    assert dict(map(tuple, graph["ops"][1]["context"]))["max_seqlen_k"] == 65536
    assert (cache.hits, cache.binds, cache.derivations) == (1, 1, 0)


def test_template_graphs_refuses_a_miss_with_no_deriver():
    cache = TemplateGraphs(allocation=carried())
    cache.add(shape([1] * 4, [1151] * 4), template_for([(1, 1151)] * 4))

    assert cache.graph_for(shape([1] * 8, [1151] * 8)) is None
    assert cache.hits == 0
    assert list(cache.refusals.values()) == ["no template and no deriver"]


def test_template_graphs_derives_a_miss_when_it_can():
    derived = {}

    def derive(want):
        derived["rows"] = list(zip(want.num_scheduled_tokens,
                                   want.context_lens))
        return template_for([(1, 1151)] * 8)

    cache = TemplateGraphs(derive=derive, allocation=carried())
    graph = cache.graph_for(shape([1] * 8, [4096] * 8))
    assert graph is not None
    assert cache.derivations == 1 and cache.binds == 1 and cache.hits == 0
    assert derived["rows"] == [(1, 4096)] * 8
    # And the derived template is kept, so the next cohort of that structure
    # is a hit rather than a second derivation.
    cache.graph_for(shape([1] * 8, [16384] * 8))
    assert cache.derivations == 1 and cache.hits == 1


def test_template_graphs_records_a_refusal_rather_than_raising():
    cache = TemplateGraphs(allocation=carried())
    cache.add(shape([1] * 4, [1151] * 4),
              template_for([(1, 1151)] * 4, extra=[["mystery", [0]]]))
    assert cache.graph_for(shape([1] * 4, [4096] * 4)) is None
    assert "mystery" in str(list(cache.refusals.values())[0])


def test_template_graphs_without_an_allocation_source_refuses():
    """The default is not "carry it quietly"; the default is no graph."""
    cache = TemplateGraphs()
    cache.add(shape([1] * 4, [1151] * 4), template_for([(1, 1151)] * 4))
    assert cache.graph_for(shape([1] * 4, [4096] * 4)) is None
    assert "AllocationSource" in str(list(cache.refusals.values())[0])
