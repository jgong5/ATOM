"""The activation peak derived at a width nobody has run.

Two things are under test and they are different in kind. The first is
arithmetic: `WarmupPeak` has to divide the terms that shard and leave alone the
ones that do not, and it has to keep the unexplained residue out of the
mechanism. The second is an artifact -- `frozen_activation_candidates.json`
was written before anything was compared against a TP=2 or TP=4 measurement,
and its value comes entirely from the fact that it has not moved since. These
tests pin it in place. A change here is a change to a frozen prediction and
has to be argued for, not merged.
"""

import json
import math
from pathlib import Path

import pytest

from atom.compass.core.memory_activation import (
    ACTIVATION_SCHEMA, CHUNK_SIZE, PeakTerm, gdn_prefill_terms,
    warmup_prefill_peak)

RECORDS = Path(__file__).parent / "memory_records"
WARMUP_TOKENS = 16384
HIDDEN = 5120

#: The measured TP=1 activation term, class S27: the source configuration's own
#: full-engine start-up, `peak_torch - current_torch` across `warmup_model`.
SOURCE_TP1_ACTIVATION = 2956984320

#: Class X27 -- the TP=2 and TP=4 measured peaks. Evaluation only. They appear
#: here to be checked *against*, never as an input to anything derived.
X27_PEAKS = {2: 1730150400, 4: 1191969280}


@pytest.fixture(scope="module")
def config():
    with open(RECORDS / "qwen3_5_27b.config.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def frozen():
    with open(RECORDS / "frozen_activation_candidates.json",
              encoding="utf-8") as fh:
        return json.load(fh)


def test_sharded_terms_divide_and_replicated_terms_do_not():
    shard = PeakTerm("q", 4096, True, "where", "why")
    stream = PeakTerm("residual", 4096, False, "where", "why")
    assert [shard.at(tp) for tp in (1, 2, 4)] == [4096, 2048, 1024]
    assert [stream.at(tp) for tp in (1, 2, 4)] == [4096, 4096, 4096]


def test_residue_is_carried_replicated_by_default(config):
    """The conservative direction, and it must be the default.

    A residue whose mechanism is unknown cannot be told to shard. Carrying it
    whole over-states the peak at width, which under-sizes the KV budget --
    fewer blocks than the card could hold, rather than more.
    """
    peak = warmup_prefill_peak(config, tokens=WARMUP_TOKENS,
                               tensor_parallel=4, residue=1024)
    assert peak.residue_at == 1024
    assert peak.total == peak.derived + 1024

    shared = warmup_prefill_peak(config, tokens=WARMUP_TOKENS,
                                 tensor_parallel=4, residue=1024,
                                 residue_shards=True)
    assert shared.residue_at == 256
    assert shared.total < peak.total


def test_peak_carries_its_schema_and_shape(config):
    peak = warmup_prefill_peak(config, tokens=WARMUP_TOKENS)
    assert peak.schema == ACTIVATION_SCHEMA
    assert peak.tokens == WARMUP_TOKENS
    assert peak.tensor_parallel == 1
    assert peak.replicated + peak.sharded_at_one == peak.derived


def test_every_gdn_term_says_where_it_comes_from_and_whether_it_shards(config):
    terms = gdn_prefill_terms(config, tokens=WARMUP_TOKENS)
    assert terms, "the derivation enumerates nothing"
    for term in terms:
        assert term.bytes_at_one > 0, term.name
        assert term.where and term.why, term.name
        assert isinstance(term.shards, bool), term.name


def test_gdn_region_does_not_set_the_high_water_mark(config, frozen):
    """The GDN peak is real and it is not the peak.

    This derivation was built on the assumption that the gated-delta-rule
    workspaces were the largest thing alive. The TP=1 lifetime walk put the
    peak in the MLP instead, above the whole GDN region at every width. The
    numbers are pinned here because the frozen candidate's zero for the opaque
    term rests on them.
    """
    expected = {1: 2684878848, 2: 1510211584, 4: 922877952}
    for tp, want in expected.items():
        peak = warmup_prefill_peak(config, tokens=WARMUP_TOKENS,
                                   tensor_parallel=tp)
        assert peak.derived == want, tp
    visible = frozen["terms"]["visible"]["bytes"]
    for tp, want in expected.items():
        assert want < visible[str(tp)], tp


def test_chunk_size_is_geometry_not_a_knob():
    assert CHUNK_SIZE == 64


def test_frozen_candidate_totals_are_the_sum_of_its_stated_terms(frozen):
    for width, total in frozen["candidate_bytes"].items():
        stated = sum(term["bytes"][width] for term in frozen["terms"].values())
        assert stated == total, width


def test_frozen_candidate_is_anchored_at_the_source_measurement(frozen):
    """TP=1 closes by construction, and the artifact has to say so.

    The residue is the source measurement minus the derived walk. That makes
    the TP=1 candidate exact and worth nothing as evidence of transfer; it is
    the two wider widths that are a prediction.
    """
    assert frozen["candidate_bytes"]["1"] == SOURCE_TP1_ACTIVATION
    residue = frozen["terms"]["unexplained_residue"]
    assert residue["bytes_at_tp1"] == (
        SOURCE_TP1_ACTIVATION - frozen["terms"]["visible"]["bytes"]["1"])
    assert "UNKNOWN" in residue["tp_mechanism"]


def test_frozen_candidate_was_frozen_before_the_comparison(frozen):
    named = " ".join(frozen["frozen_before_any_comparison_with"])
    assert "TP=2" in named and "TP=4" in named
    assert frozen["schema"] == "compass.activation.candidate/1"
    assert "Nothing here is fitted" in frozen["scope"]


def test_frozen_candidate_still_over_reads_at_width(frozen):
    """The error the candidate actually has, written down as a number.

    It is +21.5% at TP=2 and +40.4% at TP=4 against the class-X27 peaks. This
    test exists so that the day someone improves the mechanism, the frozen
    artifact cannot be quietly edited to agree with the measurement instead.
    Both directions of change fail here and have to be explained.
    """
    errors = {}
    for tp, measured in X27_PEAKS.items():
        candidate = frozen["candidate_bytes"][str(tp)]
        assert candidate > measured, tp
        errors[tp] = (candidate - measured) / measured
    assert math.isclose(errors[2], 0.2146, abs_tol=5e-4)
    assert math.isclose(errors[4], 0.4040, abs_tol=5e-4)


def test_live_set_at_the_peak_adds_up_to_the_walk(frozen):
    """And the one place the two numbers differ is the named bias.

    `walk_bytes` is what the walk counted, `visible_peak_bytes` is what the
    candidate uses, and the gap is `embedding_dtype_correction`: at TP>1 the
    walk sizes `aiter::masked_embedding` by its int32 first input, 335 544 320
    B where the output is a bfloat16 hidden-width activation at 167 772 160.
    The listed tensors carry the corrected size, so they sum to the peak the
    candidate uses, and the difference has to be exactly one such buffer --
    not a fitted amount.
    """
    for width, entry in frozen["widths"].items():
        total = 0
        for tensor in entry["live_at_peak"]:
            size = 1
            for dim in tensor["shape"]:
                size *= dim
            itemsize = {"bfloat16": 2, "float32": 4, "int32": 4,
                        "int64": 8}[tensor["dtype"]]
            assert tensor["bytes"] == size * itemsize, (width, tensor["name"])
            total += tensor["bytes"]
        assert total == entry["visible_peak_bytes"], width
        assert (entry["walk_bytes"] - entry["embedding_dtype_correction"]
                == entry["visible_peak_bytes"]), width
        correction = entry["embedding_dtype_correction"]
        assert correction in (0, WARMUP_TOKENS * HIDDEN * 2), width


def test_the_frozen_artifacts_per_tensor_width_rule_is_withdrawn(frozen):
    """What the `shards` flag in the frozen artifact meant, and why it is wrong.

    The rule was: trailing dimension equal to the hidden size means the
    residual stream, therefore replicated. It cannot tell the residual stream
    from an attention projection's input, because at TP=1 `heads * head_dim`
    *is* the hidden size and that tensor shards. The flag is consistent with
    the rule that produced it -- this pins that it was applied uniformly, so
    nothing was hand-set -- and the rule itself is replaced by `width_classes`,
    which reads each tensor's behaviour from the graphs derived at the other
    widths.

    The frozen candidate's *bytes* do not rest on it: `visible_peak_bytes` at
    each width is the walk over that width's own derived graph, so the flag was
    an annotation. It is still withdrawn, and a reader comparing the artifact
    against a later one has to know which field changed meaning.
    """
    for width, entry in frozen["widths"].items():
        for tensor in entry["live_at_peak"]:
            assert tensor["shards"] is (tensor["shape"][-1] != HIDDEN), (
                width, tensor["name"])


def test_derived_widths_narrow_the_projections_and_leave_the_stream_alone(
        frozen):
    def widths_of(tp, name_startswith):
        return sorted(t["shape"][-1]
                      for t in frozen["widths"][str(tp)]["live_at_peak"]
                      if t["shape"][-1] != HIDDEN)

    assert widths_of(1, "") == [17408, 34816]
    assert widths_of(2, "") == [8704, 17408]
    assert widths_of(4, "") == [4352, 8704]
    # The projections narrow with the width in the derived graphs themselves,
    # which is ATOM's own sharding arithmetic and is what the candidate's bytes
    # rest on. Hidden-width tensors stay 5120 at every width -- but that is a
    # shape, not a class: `width_classes` decides which of them shard, because
    # at TP=1 an attention projection's input is `heads * head_dim` = 5120 too.
    for tp in (1, 2, 4):
        stream = [t for t in frozen["widths"][str(tp)]["live_at_peak"]
                  if t["shape"][-1] == HIDDEN]
        assert stream, tp


def _graph(ops):
    return {"ops": ops, "key": {"topology": {"tp": 1}}}


def test_a_recorded_output_dtype_is_used_over_any_rule():
    from atom.compass.core.memory_model import (dtype_ambiguities,
                                                peak_activation_bytes)

    op = {"name": "aiter::masked_embedding", "output_shapes": [[16384, 5120]],
          "dtypes": ["int32", "bfloat16"], "output_dtypes": ["bfloat16"],
          "dies_at": [-1]}
    assert peak_activation_bytes(_graph([op])) == 16384 * 5120 * 2
    assert dtype_ambiguities(_graph([op])) == []
    # and a recorded dtype is the only thing a strict walk accepts
    assert peak_activation_bytes(_graph([op]), strict_dtypes=True) \
        == 16384 * 5120 * 2


def test_agreeing_arguments_settle_the_dtype_without_a_record():
    from atom.compass.core.memory_model import (dtype_ambiguities,
                                                peak_activation_bytes)

    op = {"name": "aiter::gemm_a16w16", "output_shapes": [[16384, 5120]],
          "dtypes": ["bfloat16", "bfloat16"], "dies_at": [-1]}
    assert peak_activation_bytes(_graph([op])) == 16384 * 5120 * 2
    assert dtype_ambiguities(_graph([op])) == []


def test_promotion_is_a_rule_not_a_per_operator_correction():
    """int32 ids and a bfloat16 weight make a bfloat16 activation.

    The rule is PyTorch's own: a float argument beats every integer one. It
    replaces both the argument-0 approximation and the ad-hoc "if this is
    masked_embedding, subtract 167 772 160" correction that the frozen
    candidate had to carry.
    """
    from atom.compass.core.memory_model import _promote

    assert _promote(["int32", "bfloat16"]) == "bfloat16"
    assert _promote(["int64", "float32", "bfloat16"]) == "float32"
    assert _promote(["int32", "int64"]) == "int64"
    assert _promote(["float16", "bfloat16"]) == "float32"
    assert _promote(["bfloat16", "bfloat16"]) == "bfloat16"
    assert _promote([]) is None


def test_a_promoted_output_is_sized_by_the_rule_and_still_reported():
    from atom.compass.core.memory_model import (dtype_ambiguities,
                                                peak_activation_bytes)

    op = {"name": "aiter::masked_embedding", "output_shapes": [[16384, 5120]],
          "dtypes": ["int32", "bfloat16"], "dies_at": [-1]}
    graph = _graph([op])
    assert peak_activation_bytes(graph) == 16384 * 5120 * 2
    ambiguous = dtype_ambiguities(graph)
    assert len(ambiguous) == 1
    entry = ambiguous[0]
    assert entry["name"] == "aiter::masked_embedding"
    assert entry["basis"] == "promoted"
    assert entry["chosen_dtype"] == "bfloat16"
    assert entry["bytes_chosen"] == 167772160
    assert entry["bytes_if_argument_0"] == 335544320


def test_a_strict_walk_takes_the_graphs_word_and_nothing_else():
    from atom.compass.core.memory_model import (UnfoundedActivation,
                                                peak_activation_bytes)

    for dtypes in (["int32", "bfloat16"], ["bfloat16", "bfloat16"]):
        op = {"name": "aiter::masked_embedding",
              "output_shapes": [[16384, 5120]], "dtypes": dtypes,
              "dies_at": [-1]}
        with pytest.raises(UnfoundedActivation) as raised:
            peak_activation_bytes(_graph([op]), strict_dtypes=True)
        assert "output_dtypes" in str(raised.value)


def test_an_in_place_output_raises_no_dtype_question_at_all():
    """No dtype is needed for a tensor the operator did not allocate."""
    from atom.compass.core.memory_model import dtype_ambiguities

    ops = [{"name": "aiter::gemm_a16w16", "output_shapes": [[16384, 5120]],
            "dtypes": ["bfloat16"], "dies_at": [1]},
           {"name": "aiter::rmsnorm2d_fwd_", "output_shapes": [[16384, 5120]],
            "dtypes": ["bfloat16", "float32"], "output_aliases": [0],
            "dies_at": [-1]}]
    assert dtype_ambiguities(_graph(ops)) == []


# --- width behaviour is read across widths, never guessed from one ----------

HEADS_TIMES_HEAD_DIM = 5120  # at TP=1 this *is* the hidden size


def _three_widths():
    """The same model at three widths, with the two tensors that look alike.

    `attn_out` is the attention projection's input: `heads * head_dim`, which
    at TP=1 is 5120 and at TP=2 is 2560 -- it shards. `residual` is the
    residual stream: 5120 at every width -- it does not. At TP=1 they have the
    same shape, which is exactly why one width cannot classify them.
    """
    def model(tp, collective):
        ops = [
            {"name": "aten::embedding", "output_shapes": [[16384, 5120]],
             "dtypes": ["bfloat16"], "inputs_from": [-1, -1]},
            {"name": "aiter::gemm_a16w16",
             "output_shapes": [[16384, HEADS_TIMES_HEAD_DIM // tp]],
             "dtypes": ["bfloat16"], "inputs_from": [0, -1]},
        ]
        if collective:
            ops.append({"name": "aiter::all_reduce_", "group": "tp",
                        "output_shapes": [[16384, 5120]],
                        "dtypes": ["bfloat16"], "inputs_from": [1]})
        ops.append({"name": "aiter::add_rmsnorm",
                    "output_shapes": [[16384, 5120]], "dtypes": ["bfloat16"],
                    "inputs_from": [len(ops) - 1, 0]})
        return {"ops": ops, "key": {"topology": {"tp": tp}}}

    return {1: model(1, False), 2: model(2, True), 4: model(4, True)}


def test_lineage_alignment_survives_the_collectives_width_adds():
    """Index 2 is a different operator at each width; ancestry is not."""
    from atom.compass.core.memory_model import lineage_keys

    graphs = _three_widths()
    at_1 = lineage_keys(graphs[1])
    at_2 = lineage_keys(graphs[2])
    assert len(at_1) == 3 and len(at_2) == 4
    # the all-reduce passes its input's identity through, so the norm after it
    # aligns with the norm that follows the matmul directly at TP=1
    assert at_2[2] == at_2[1]
    assert at_1[-1] == at_2[-1]
    assert graphs[1]["ops"][2]["name"] != graphs[2]["ops"][2]["name"]


def test_a_hidden_width_tensor_can_still_be_sharded():
    """The failure the trailing-dimension rule could not see.

    Both tensors are [16384, 5120] at TP=1. The rule said "trailing dimension
    is the hidden size, therefore replicated" and got one of them wrong, which
    is an over-read at every width above 1.
    """
    from atom.compass.core.memory_model import width_classes

    classes = width_classes(_three_widths())
    by_name = {entry["name"]: entry for entry in classes.values()}
    assert by_name["aiter::gemm_a16w16"]["class"] == "sharded"
    assert by_name["aiter::gemm_a16w16"]["axis"] == 1
    assert by_name["aiter::gemm_a16w16"]["shapes"][1] == [16384, 5120]
    assert by_name["aiter::add_rmsnorm"]["class"] == "replicated"
    assert by_name["aten::embedding"]["class"] == "replicated"
    # what the withdrawn rule would have said about the same two tensors
    for entry in (by_name["aiter::gemm_a16w16"], by_name["aiter::add_rmsnorm"]):
        assert entry["shapes"][1][-1] == 5120


def test_a_ratio_the_width_does_not_explain_is_not_classified():
    from atom.compass.core.memory_model import width_classes

    graphs = _three_widths()
    graphs[4]["ops"][1]["output_shapes"] = [[16384, 999]]
    classes = width_classes(graphs)
    by_name = {entry["name"]: entry for entry in classes.values()}
    assert by_name["aiter::gemm_a16w16"]["class"] == "unresolved"
    assert by_name["aiter::gemm_a16w16"]["axis"] is None


def test_coverage_is_reported_beside_the_classification():
    """A classification that drops half the graph must say so."""
    from atom.compass.core.memory_model import width_coverage

    coverage = width_coverage(_three_widths())
    assert coverage["aligned"] == 3
    assert coverage["unaligned_at_base"] == 0
    assert coverage["outputs_per_width"] == {1: 3, 2: 3, 4: 3}
    assert coverage["by_class"] == {"replicated": 2, "sharded": 1,
                                    "unresolved": 0}


def test_one_width_cannot_be_classified_at_all():
    from atom.compass.core.memory_model import width_classes

    with pytest.raises(ValueError) as raised:
        width_classes({1: _three_widths()[1]})
    assert "two or more" in str(raised.value)


def test_a_width_conditional_branch_defeats_a_name_keyed_lineage():
    """The limitation the real graphs showed, pinned so it cannot be forgotten.

    `VocabParallelEmbedding.forward` takes `masked_embedding` plus an
    all-reduce at TP>1 and `F.embedding` at TP=1 (`embed_head.py:168-178`), so
    the *same module* emits two operator names. An ancestry key that interns
    the name diverges at operator 0 and stays diverged, which is why alignment
    over the real TP=1/2/4 graphs reaches 12 of 3014 outputs. The name split
    is real but it is not the whole cause -- 725 of 4378 source edges are
    unknown at TP>1 -- so the replacement is `module_path_keys`, and a
    name-keyed lineage stays unreadable at width.
    """
    from atom.compass.core.memory_model import width_coverage

    graphs = _three_widths()
    for tp in (2, 4):
        graphs[tp]["ops"][0]["name"] = "aiter::masked_embedding"
    coverage = width_coverage(graphs)

    assert coverage["aligned"] == 0
    assert coverage["unaligned_at_base"] == len(graphs[1]["ops"])
    assert coverage["by_class"] == {"replicated": 0, "sharded": 0,
                                    "unresolved": 0}


def _three_widths_with_paths():
    """The same three graphs, plus the module each operator ran in.

    `VocabParallelEmbedding` is the case that defeated the name-keyed
    ancestry: one module, `aten::embedding` at TP=1 and
    `aiter::masked_embedding` above it. The module path is the same string at
    every width, because the module tree is the same tree.
    """
    graphs = _three_widths()
    for tp in (2, 4):
        graphs[tp]["ops"][0]["name"] = "aiter::masked_embedding"
    paths = {
        1: ["model.embed_tokens", "model.layers.0.self_attn.o_proj",
            "model.layers.0.post_attention_layernorm"],
        2: ["model.embed_tokens", "model.layers.0.self_attn.o_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.post_attention_layernorm"],
    }
    paths[4] = list(paths[2])
    return graphs, paths


def test_a_module_path_aligns_what_an_operator_name_could_not():
    """The O23 failure, fixed by the key rather than by naming the exception.

    The name-keyed ancestry aligns nothing here -- operator 0 diverges and
    every descendant inherits it. The module path aligns all three outputs,
    and says out loud that one of them joins two differently-named operators.
    """
    from atom.compass.core.memory_model import width_coverage

    graphs, paths = _three_widths_with_paths()
    assert width_coverage(graphs)["aligned"] == 0

    coverage = width_coverage(graphs, paths)
    assert coverage["aligned"] == 3
    assert coverage["unaligned_at_base"] == 0
    assert coverage["renamed"] == 1
    assert coverage["by_class"] == {"replicated": 2, "sharded": 1,
                                    "unresolved": 0}


def test_the_renamed_join_is_recorded_on_the_entry_itself():
    from atom.compass.core.memory_model import width_classes

    graphs, paths = _three_widths_with_paths()
    classes = width_classes(graphs, paths)
    renamed = [entry for entry in classes.values() if "names" in entry]
    assert len(renamed) == 1
    assert renamed[0]["names"] == {1: "aten::embedding",
                                   2: "aiter::masked_embedding",
                                   4: "aiter::masked_embedding"}
    assert renamed[0]["class"] == "replicated"


def test_a_collective_consumes_no_ordinal():
    """Otherwise every operator after the first all-reduce is renumbered.

    The norm is ordinal 0 of its own module at TP=1 and must stay ordinal 0
    at TP=2, where an all-reduce sits between it and the matmul.
    """
    from atom.compass.core.memory_model import module_path_keys

    graphs, paths = _three_widths_with_paths()
    at_1 = module_path_keys(graphs[1], paths[1])
    at_2 = module_path_keys(graphs[2], paths[2])
    assert at_1[-1] == at_2[-1]
    # and the collective itself passes its input's identity through
    assert at_2[2] == at_2[1]


def test_a_module_that_does_different_work_at_width_is_refused():
    """The ordinal only means something while the module is the same module."""
    from atom.compass.core.memory_model import (alignment_integrity,
                                                width_classes)

    graphs, paths = _three_widths_with_paths()
    # TP=4 runs an extra operator inside the attention module
    graphs[4]["ops"].insert(2, {"name": "aten::mul",
                                "output_shapes": [[16384, 1280]],
                                "dtypes": ["bfloat16"], "inputs_from": [1]})
    paths[4].insert(2, "model.layers.0.self_attn.o_proj")

    integrity = alignment_integrity(graphs, paths)
    assert integrity["safe"] is False
    assert integrity["paths_disagreeing"] == ["model.layers.0.self_attn.o_proj"]

    with pytest.raises(ValueError) as raised:
        width_classes(graphs, paths)
    assert "not safe to read" in str(raised.value)


def test_surviving_ancestry_that_contradicts_the_ordinal_ends_it():
    """Ancestry is weak evidence at width, but it is not ignorable evidence."""
    from atom.compass.core.memory_model import (alignment_integrity,
                                                width_classes)

    graphs, paths = _three_widths_with_paths()
    clean = alignment_integrity(graphs, paths)
    assert clean["safe"] is True
    assert clean["ancestry_contradicts"] == 0

    # the last operator at TP=4 claims a different producer than at TP=1
    graphs[4]["ops"][-1]["inputs_from"] = [0, 0]
    broken = alignment_integrity(graphs, paths)
    assert broken["ancestry_contradicts"] == 1
    assert broken["safe"] is False
    with pytest.raises(ValueError):
        width_classes(graphs, paths)


def test_an_unknown_producer_is_counted_apart_from_agreement():
    """A check that could not run is not a check that passed."""
    from atom.compass.core.memory_model import alignment_integrity

    graphs, paths = _three_widths_with_paths()
    integrity = alignment_integrity(graphs, paths)
    # operator 0 has two unknown sources at every width
    assert integrity["ancestry_unknown"] >= 1
    assert (integrity["ancestry_agrees"] + integrity["ancestry_contradicts"]
            + integrity["ancestry_unknown"]) == 3


def test_a_module_path_is_needed_for_every_operator():
    from atom.compass.core.memory_model import module_path_keys

    graphs, paths = _three_widths_with_paths()
    with pytest.raises(ValueError) as raised:
        module_path_keys(graphs[1], paths[1][:-1])
    assert "every operator" in str(raised.value)
