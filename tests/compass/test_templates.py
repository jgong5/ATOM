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


def test_the_collectives_group_is_carried_and_not_refused():
    """A TP>1 head graph carries the group its all-gather runs in.

    `derive.py` writes the group's name and width into the synthesized
    operator's context, so a head template at TP>1 has two context fields no
    attention call ever has. Before there was a rule for them, binding refused
    the whole head graph -- so the served composition priced the body at TP2
    and TP4 and answered nothing at all for the head.
    """
    template = template_for([(1, 1151)] * 4,
                            extra=[["group", "tp:0-1"],
                                   ["group_world_size", 2]])
    bound = bind_cohort(template, shape([1] * 4, [4096] * 4, tp=2), carried())
    context = dict(map(tuple, bound["ops"][1]["context"]))
    assert context["group"] == "tp:0-1"
    assert context["group_world_size"] == 2


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


class TestTheRepresentativeRank:
    """A rank-1 shape and a rank-0 template, which is every frozen template.

    Derivation builds a one-rank gloo group, tells it to report the wider
    width, and leaves ``rank_in_group`` at 0 -- so every graph it produces is
    rank 0's shard whatever rank asked, and every template on disk is keyed at
    rank 0. `template_key` carries the rank, so before this fallback a served
    rank 1 matched none of them and either re-derived the same graph or, with
    derivation off, was refused. Both were silent.
    """

    def test_a_rank_is_served_by_the_representative_and_the_borrow_is_counted(self):
        cache = TemplateGraphs(allocation=carried())
        cache.add(shape([1] * 4, [1151] * 4, tp=2, rank=0),
                  template_for([(1, 1151)] * 4))
        assert cache.graph_for(shape([1] * 4, [4096] * 4, tp=2, rank=1))
        assert cache.hits == 1
        assert cache.representative_hits == 1
        assert "rank 0's graph" in cache.describe()

    def test_a_rank_with_its_own_template_does_not_borrow(self):
        """Precedence, for the day a derivation really is per-rank."""
        cache = TemplateGraphs(allocation=carried())
        cache.add(shape([1] * 4, [1151] * 4, tp=2, rank=0),
                  template_for([(1, 1151)] * 4))
        cache.add(shape([1] * 4, [1151] * 4, tp=2, rank=1),
                  template_for([(1, 1151)] * 4))
        assert cache.graph_for(shape([1] * 4, [4096] * 4, tp=2, rank=1))
        assert cache.hits == 1 and cache.representative_hits == 0

    def test_the_representative_is_not_a_wildcard_over_structure(self):
        """Only the rank moves. A different batch structure is still a miss."""
        cache = TemplateGraphs(allocation=carried())
        cache.add(shape([1] * 4, [1151] * 4, tp=2, rank=0),
                  template_for([(1, 1151)] * 4))
        assert cache.graph_for(shape([1] * 8, [1151] * 8, tp=2, rank=1)) is None
        assert cache.hits == 0 and cache.representative_hits == 0

    def test_rank_zero_is_unchanged_so_frozen_results_are_unchanged(self):
        """The fallback cannot fire where the key already matches."""
        cache = TemplateGraphs(allocation=carried())
        cache.add(shape([1] * 4, [1151] * 4, tp=2, rank=0),
                  template_for([(1, 1151)] * 4))
        assert cache.graph_for(shape([1] * 4, [4096] * 4, tp=2, rank=0))
        assert cache.hits == 1 and cache.representative_hits == 0


# -- the native allocation ---------------------------------------------------

def native(**kw):
    """A source over a 16-token block and this deployment's 256k context."""
    from atom.compass.runtime.templates import NativeAllocation

    kw.setdefault("block_size", 16)
    kw.setdefault("max_model_len", 262144)
    return NativeAllocation(**kw)


def offered(source, rows, tables, slots=(0, 1), prefill_seqs=0,
            state_rows=None):
    from atom.compass.runtime.templates import NativeStepAllocation

    source.offer(NativeStepAllocation(
        rows=rows, block_tables=tables,
        state_slots=(None if slots is None else list(slots)),
        state_rows=state_rows,
        num_prefill_seqs=prefill_seqs,
        source="test", rank_coords={"tp": 0}))
    return source


def test_the_scheduler_s_own_blocks_reach_the_bound_graph():
    """The whole point of the bridge: these slots are not the template's.

    Two requests at context 32 hold two 16-token blocks each. The scheduler put
    them at blocks 7,8 and 3,4, so the decode token of each is the last slot of
    its second block -- 8*16+15 and 4*16+15. Nothing here is a model of an
    allocator: the block ids come from the record and the arithmetic is
    `BatchSpec`'s, the same code that describes a captured batch.
    """
    rows = [(1, 32), (1, 32)]
    source = offered(native(), rows, [[7, 8], [3, 4]])
    bound = bind_cohort(template_for(rows), shape([1, 1], [32, 32], bucket=2),
                        source)
    context = dict(map(tuple, bound["ops"][1]["context"]))
    assert context["slot_mapping"] == [8 * 16 + 15, 4 * 16 + 15]
    binding = bound["provenance"]["binding"]
    assert binding["allocation_measured"] is True
    assert binding["allocation_padding"] == {}
    assert set(binding["allocation_fields"]) == set(ALLOCATOR_FIELDS)


def test_a_step_nobody_offered_an_allocation_for_is_refused():
    """The offline case, and the one that keeps an unattended run honest.

    A `NativeAllocation` with no record does not fall back to the template's
    blocks. It says so, and every shape whose template carries allocator fields
    is refused -- which is what `allocation=native` does in a CLI, where no
    runner is offering anything.
    """
    rows = [(1, 32), (1, 32)]
    with pytest.raises(BindRefusal, match="no native allocation was offered"):
        bind_cohort(template_for(rows), shape([1, 1], [32, 32], bucket=2),
                    native())


def test_the_previous_step_s_allocation_is_not_this_step_s():
    """A record is checked against the shape rather than trusted.

    The failure this prevents is the quiet one: a runner that offers on some
    steps and not others would otherwise price the second step with the first
    step's blocks and report the result as measured.
    """
    source = offered(native(), [(1, 32), (1, 32)], [[7, 8], [3, 4]])
    with pytest.raises(BindRefusal, match="another step"):
        bind_cohort(template_for([(1, 48), (1, 48)]),
                    shape([1, 1], [48, 48], bucket=2), source)


def test_a_batch_with_no_state_slots_is_refused_not_defaulted():
    """`gdn_context` defaults the slots to the batch order. That default is
    what a fresh pool hands out and nothing else, so a bridge that took it
    would price a fragmented pool as a fresh one and call it measured."""
    rows = [(1, 32), (1, 32)]
    source = offered(native(), rows, [[7, 8], [3, 4]], slots=None)
    with pytest.raises(BindRefusal, match="no state slots"):
        bind_cohort(template_for(rows), shape([1, 1], [32, 32], bucket=2),
                    source)


def test_the_capture_s_padding_is_kept_and_counted():
    """A capture at a wider rung records the buffer, not the active rows.

    The active entries are the scheduler's; the tail is whatever the runner
    left in the padded buffer, and it stays -- overwriting it would invent an
    assignment for rows that are not running, and dropping it would resize a
    buffer the graph's own shapes still describe.
    """
    rows = [(1, 32), (1, 32)]
    template = template_for(rows)
    context = template["ops"][1]["context"]
    for entry in context:
        if entry[0] == "slot_mapping":
            entry[1] = [-1, -1, -1, -1]
    source = offered(native(), rows, [[7, 8], [3, 4]])
    bound = bind_cohort(template, shape([1, 1], [32, 32], bucket=4), source)
    slots = dict(map(tuple, bound["ops"][1]["context"]))["slot_mapping"]
    assert slots == [8 * 16 + 15, 4 * 16 + 15, -1, -1]
    assert bound["provenance"]["binding"]["allocation_padding"] == {
        "slot_mapping": 2}


def test_a_mixed_prefill_decode_batch_is_refused_by_name():
    """The representation gap cc-traces will hit, surfaced rather than papered
    over.

    `BatchSpec` carries one `kind` for the whole batch, and so does the engine:
    `backends.py` sends any batch holding a prefill token down
    `prepare_prefill`, which prepares metadata for
    `batch.total_seqs_num_prefill` leading rows only. So a batch of one
    prefilling request and one decoding one has no encoding here -- and the
    tempting shortcut, reading the batch's prefill *token* count and calling
    the whole thing a prefill, is exactly the inference that would misplace the
    decode row's slot. Closing this needs a per-request kind in `BatchSpec` and
    a deriver that can trace such a batch.
    """
    rows = [(8, 40), (1, 32)]
    source = offered(native(), rows, [[7, 8, 9], [3, 4, 5]], prefill_seqs=1)
    with pytest.raises(BindRefusal, match="1 prefill rows and 1 decode rows"):
        bind_cohort(template_for(rows),
                    shape([8, 1], [40, 32], bucket=2, prefill=8), source)


def test_an_all_prefill_batch_takes_the_prefill_encoding():
    """The other side of the same rule: the kind comes from the scheduler's
    request count, not from the token count. Both requests are prefilling, so
    every scheduled token gets a slot in its request's own blocks -- request
    one's four tokens land in block 7 and request two's two in block 3."""
    rows = [(4, 4), (2, 2)]
    source = offered(native(), rows, [[7], [3]], prefill_seqs=2)
    template = template_for(rows)
    for entry in template["ops"][1]["context"]:
        if entry[0] == "slot_mapping":
            # A prefill's buffer is one entry per scheduled token, not per
            # request; the helper's default is a decode-shaped one.
            entry[1] = [0] * 6
    bound = bind_cohort(template,
                        shape([4, 2], [4, 2], bucket=2, prefill=6), source)
    context = dict(map(tuple, bound["ops"][1]["context"]))
    assert context["slot_mapping"] == [7 * 16 + i for i in range(4)] + [
        3 * 16 + i for i in range(2)]


def test_state_slots_are_placed_by_row_not_by_list_position():
    """`state_slots_committed` is a filtered list, so its index is not the
    batch index. Here the state-bearing requests are rows 1 and 2 of three, and
    their slots are 5 and 9; reading the list positionally would give row 0 the
    slot 5 that belongs to row 1."""
    rows = [(1, 32), (1, 32), (1, 32)]
    source = offered(native(), rows, [[7, 8], [3, 4], [1, 2]],
                     slots=(5, 9), state_rows=[1, 2])
    with pytest.raises(BindRefusal, match=r"rows \[0\] of 3 hold no state"):
        bind_cohort(template_for(rows), shape([1] * 3, [32] * 3, bucket=4),
                    source)

    source = offered(native(), rows, [[7, 8], [3, 4], [1, 2]],
                     slots=(9, 5, 2), state_rows=[2, 0, 1])
    bound = bind_cohort(template_for(rows, extra=[
        ["non_spec_state_indices_tensor", [[0, 0, 0], "int32"]]]),
        shape([1] * 3, [32] * 3, bucket=4), source)
    context = dict(map(tuple, bound["ops"][1]["context"]))
    assert context["non_spec_state_indices_tensor"] == [[5, 2, 9], "int32"]


def test_an_active_count_below_the_capture_bucket_pads_every_field():
    """Two running requests bound to a graph captured at four.

    The template's buffers are four-wide, and they stay four-wide: the two
    active entries are the scheduler's and the tail is the capture's, counted
    per field so a reader can see how much of the bound metadata was not this
    step's. Overwriting the tail would invent an assignment for rows that are
    not running.
    """
    rows = [(1, 32), (1, 32)]
    template = template_for(rows, extra=[
        ["non_spec_state_indices_tensor", [[0, 0, 0, 0], "int32"]]])
    for entry in template["ops"][1]["context"]:
        if entry[0] == "slot_mapping":
            entry[1] = [-1, -1, -1, -1]
    source = offered(native(), rows, [[7, 8], [3, 4]], slots=(5, 9),
                     state_rows=[0, 1])
    bound = bind_cohort(template, shape([1, 1], [32, 32], bucket=4), source)
    context = dict(map(tuple, bound["ops"][1]["context"]))
    assert context["slot_mapping"] == [8 * 16 + 15, 4 * 16 + 15, -1, -1]
    assert context["non_spec_state_indices_tensor"] == [[5, 9, 0, 0], "int32"]
    assert bound["provenance"]["binding"]["allocation_padding"] == {
        "slot_mapping": 2, "non_spec_state_indices_tensor": 2}
