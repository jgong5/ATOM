"""Disjoint source intervals cannot borrow anchors across an unmeasured gap."""
from dataclasses import replace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    BucketedRunnerRegions, Measured, POOLED_SEQS, PrefillIntervalRegions,
)
from atom.compass.runtime.source_oracle import region_snapshot


def measured(seconds):
    return Measured(seconds, seconds / 2, seconds * 2, 3, "synthetic source")


def shape(tokens, output=False, seqs=1):
    per = tokens // seqs
    scheduled = (tokens - per * (seqs - 1),) + (per,) * (seqs - 1)
    return StepShape(scheduled, scheduled, num_prefill_tokens=tokens,
                     topology={"tp": 1}, produces_output=output)


@pytest.fixture
def source():
    return BucketedRunnerRegions(
        postprocess_decode=measured(1), prepare_decode_cells=(),
        postprocess_prefill=measured(1), prepare_prefill=measured(1),
        tp_broadcast=measured(1), decode_context=(1, 1024),
        prefill_sequences=(1, 2), prefill_tokens=(64, 16384), topologies=(1,),
        prepare_prefill_cells=(((1, 16384, False), measured(30)),),
        prepare_prefill_anchors=(
            ((1, 64, False), measured(2)), ((1, 640, False), measured(8)),
            ((2, 2048, True), measured(20)), ((2, 16384, True), measured(40)),
            ((POOLED_SEQS, 1536, True), measured(20)),
            ((POOLED_SEQS, 16384, True), measured(40))),
        postprocess_prefill_anchors=(
            ((2, 2048, True), measured(1)), ((2, 16384, True), measured(2)),
            ((POOLED_SEQS, 1536, True), measured(1)),
            ((POOLED_SEQS, 16384, True), measured(2))),
        prefill_pooled_sequences=(3, 32),
    )


def bounded(source, intervals=((64, 640),)):
    return PrefillIntervalRegions.from_model(source, (((1, False), intervals),))


def test_small_segment_interpolates_and_keeps_union_band(source):
    model = bounded(source)
    query = shape(352)
    assert model.refusal(query) is None
    assert model.breakdown(query) == {"<prepare>": 5, "<postprocess>": 0}
    assert model.band(query) == (1, 16)


@pytest.mark.parametrize("tokens", [641, 8192, 16383])
def test_gap_is_refused_despite_anchors_on_both_sides(source, tokens):
    model = bounded(source)
    assert "outside declared interpolation intervals" in model.refusal(shape(tokens))
    with pytest.raises(ValueError, match="no measured region"):
        model.seconds(shape(tokens))


def test_exact_measured_point_outside_intervals_is_preserved(source):
    model = bounded(source)
    assert model.breakdown(shape(16384)) == source.breakdown(shape(16384))
    assert model.band(shape(16384)) == source.band(shape(16384))
    assert model._prefill_at(model._prefill_table(model.prepare_prefill_cells,
        model.prepare_prefill_anchors), (1, 16384, False)) is source.prepare_prefill_cells[0][1]


def test_each_disjoint_interval_uses_its_own_neighbours(source):
    source = replace(source, prepare_prefill_anchors=(
        source.prepare_prefill_anchors + (((1, 8192, False), measured(15)),)))
    model = bounded(source, ((64, 640), (8192, 16384)))
    assert model.breakdown(shape(12288))["<prepare>"] == 22.5
    assert model.band(shape(12288)) == (7.5, 60)
    assert model.refusal(shape(4096)) is not None


def test_declared_interval_cannot_borrow_missing_endpoint_from_far_anchor(source):
    missing_upper = replace(source, prepare_prefill_anchors=(
        ((1, 64, False), measured(2)),))
    model = bounded(missing_upper)
    # 352 is allowed by the declaration, but only 64 lies inside its interval.
    assert model.refusal(shape(352)) is not None
    with pytest.raises(ValueError, match="no measured region"):
        model.seconds(shape(352))


def test_postprocess_must_have_its_own_brackets_inside_same_interval(source):
    output_source = replace(source,
        prepare_prefill_cells=(((1, 16384, True), measured(30)),),
        prepare_prefill_anchors=(((1, 64, True), measured(2)),
                                 ((1, 640, True), measured(8))),
        postprocess_prefill_anchors=(((1, 64, True), measured(1)),
                                     ((1, 16384, True), measured(4))))
    model = PrefillIntervalRegions.from_model(output_source,
        (((1, True), ((64, 640),)),))
    assert "postprocess" in model.refusal(shape(352, output=True))


def test_n1_intervals_do_not_widen_other_sequence_groups(source):
    model = bounded(source)
    for seqs in [2, 3, 8, 32]:
        assert model.refusal(shape(512, output=True, seqs=seqs)) == source.refusal(
            shape(512, output=True, seqs=seqs))
        query = shape(8192, output=True, seqs=seqs)
        assert model.breakdown(query) == source.breakdown(query)
        assert model.band(query) == source.band(query)


def test_interval_declaration_is_part_of_calibration_identity(source):
    before = region_snapshot("synthetic", source)
    model = bounded(source)
    changed = region_snapshot("synthetic", model)
    assert before == region_snapshot("synthetic", source)
    assert "prefill_interpolation_intervals" not in before["parameters"]
    assert changed["parameters"]["prefill_interpolation_intervals"]
    assert changed["sha256"] != before["sha256"]


@pytest.mark.parametrize("intervals", [((0, 640),), ((640, 64),),
                                      ((64, 640), (512, 1024)),
                                      ((1024, 16384), (64, 640))])
def test_ambiguous_or_invalid_interval_declaration_is_refused(source, intervals):
    with pytest.raises(ValueError, match="ordered and disjoint"):
        bounded(source, intervals)
