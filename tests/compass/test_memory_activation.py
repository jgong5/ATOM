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


def test_width_mechanism_is_read_off_each_tensors_own_shape(frozen):
    """Trailing dimension == hidden size means the residual stream.

    That is the whole rule, and it is a statement about the tensor rather than
    about the term: a buffer as wide as the model is replicated on every rank,
    anything wider is a column-parallel projection and the graph derived at
    that width already carries the narrower shape.
    """
    for width, entry in frozen["widths"].items():
        for tensor in entry["live_at_peak"]:
            replicated = tensor["shape"][-1] == HIDDEN
            assert tensor["shards"] is not replicated, (width, tensor["name"])


def test_derived_widths_narrow_the_projections_and_leave_the_stream_alone(
        frozen):
    def widths_of(tp, name_startswith):
        return sorted(t["shape"][-1]
                      for t in frozen["widths"][str(tp)]["live_at_peak"]
                      if t["shape"][-1] != HIDDEN)

    assert widths_of(1, "") == [17408, 34816]
    assert widths_of(2, "") == [8704, 17408]
    assert widths_of(4, "") == [4352, 8704]
    for tp in (1, 2, 4):
        stream = [t for t in frozen["widths"][str(tp)]["live_at_peak"]
                  if t["shape"][-1] == HIDDEN]
        assert stream, tp
        assert all(not t["shards"] for t in stream), tp
