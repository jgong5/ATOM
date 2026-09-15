"""Composing a TP=1 engine calibration with a standalone topology delta.

The delta was measured with no model on the engine's own `init_dist_env` path.
What these tests hold onto is the arithmetic that makes it admissible: the
width-1 control is zero, so the composition is an identity there; a width with
no measurement is refused instead of being answered from a narrower one; and
the environment the readings were taken under is part of the reading.
"""

import pytest

from atom.compass.core.memory_model import MIB, UnfoundedPrediction, derived_readings
from atom.compass.core.memory_topology import (
    TOPOLOGY_DELTAS, UnmeasuredWidth, compose_calibration, topology_delta)

#: A TP=1 full-engine calibration, class S27, shaped the way the profile's
#: calibration file is.
BASE = {
    "persistent": 252339712,
    "non_torch": {1: 1157627904},
    "load_residue": {1: 14924832},
    "provenance": {"persistent": "S27", "non_torch": "S27",
                   "load_residue": "S27"},
}


class TestTheControlMakesItADelta:
    def test_width_one_measured_zero_for_both_terms(self):
        entry = TOPOLOGY_DELTAS[1]
        assert entry["non_torch_by_rank"] == {0: 0}
        assert entry["load_residue"] == 0
        assert entry["ca_instances"] == 0

    def test_composing_at_width_one_returns_the_engine_calibration(self):
        """Not approximately: the control is zero, so this is an identity."""
        composed = compose_calibration(BASE, 1)
        assert composed["non_torch"] == {1: BASE["non_torch"][1]}
        assert composed["load_residue"] == {1: BASE["load_residue"][1]}

    def test_the_delta_is_added_to_the_tp1_term_not_used_instead_of_it(self):
        composed = compose_calibration(BASE, 2)
        assert composed["non_torch"][2] == 1157627904 + 5676990464
        assert composed["load_residue"][2] == 14924832 + 2164260864


class TestTwoInstancesWereMeasuredNotAssumed:
    def test_the_allocator_delta_is_two_instances_worth(self):
        for width in (2, 4):
            entry = TOPOLOGY_DELTAS[width]
            assert entry["ca_instances"] == 2
            assert entry["load_residue"] == 2 * (8 * MIB + 1024 * MIB)

    def test_the_allocator_delta_is_flat_in_width(self):
        assert TOPOLOGY_DELTAS[2]["load_residue"] == \
            TOPOLOGY_DELTAS[4]["load_residue"]


class TestPointAndRangeAreDifferentClaims:
    def test_the_point_is_the_rank_maximum(self):
        """Under-reserving non_torch over-allocates KV and dies at steady
        state; over-reserving costs blocks. The asymmetry picks the maximum."""
        by_rank = TOPOLOGY_DELTAS[4]["non_torch_by_rank"]
        assert topology_delta(4)["non_torch"] == max(by_rank.values())

    def test_the_spread_travels_with_the_point(self):
        reading = topology_delta(4)
        assert reading["non_torch_range"] == [5827985408, 5962203136]
        assert reading["non_torch_range"][1] - reading["non_torch_range"][0] \
            == 128 * MIB

    def test_a_named_rank_gets_its_own_reading(self):
        assert topology_delta(4, rank=3)["non_torch"] == 5827985408
        assert topology_delta(4, rank=3)["side"] == "rank"

    def test_an_unmeasured_rank_is_refused(self):
        with pytest.raises(UnmeasuredWidth):
            topology_delta(4, rank=7)

    def test_the_two_ranks_at_width_two_agreed(self):
        reading = topology_delta(2)
        assert reading["non_torch_range"][0] == reading["non_torch_range"][1]


class TestAnUnmeasuredWidthIsRefused:
    def test_width_eight_is_not_answered_from_width_four(self):
        with pytest.raises(UnmeasuredWidth) as excinfo:
            topology_delta(8)
        assert "carrying a value forward" in str(excinfo.value)

    def test_the_refusal_names_what_was_measured(self):
        with pytest.raises(UnmeasuredWidth) as excinfo:
            compose_calibration(BASE, 3)
        assert "1, 2, 4" in str(excinfo.value)


class TestTheEnvironmentIsPartOfTheReading:
    def test_expandable_segments_moves_two_gib_and_is_refused(self):
        with pytest.raises(UnmeasuredWidth) as excinfo:
            topology_delta(4, env={"PYTORCH_HIP_ALLOC_CONF":
                                   "expandable_segments:True"})
        assert "load_residue to non_torch" in str(excinfo.value)

    def test_a_raw_input_pool_is_refused(self):
        with pytest.raises(UnmeasuredWidth):
            compose_calibration(BASE, 2,
                                env={"AITER_CUSTOM_AR_RAW_INPUT_POOL": "1"})

    def test_the_measured_environment_passes(self):
        env = {"PYTORCH_HIP_ALLOC_CONF": None, "HIP_VISIBLE_DEVICES": "0,1"}
        assert topology_delta(2, env=env)["non_torch"] == 5676990464


class TestTheComposedCalibrationIsUsable:
    def test_a_base_without_a_tp1_entry_cannot_be_composed(self):
        with pytest.raises(UnmeasuredWidth):
            compose_calibration({"non_torch": {2: 1}, "load_residue": {1: 1},
                                 "persistent": 1}, 2)

    def test_the_provenance_names_both_halves(self):
        provenance = compose_calibration(BASE, 4)["provenance"]
        for term in ("non_torch", "load_residue"):
            assert "S27 at TP=1" in provenance[term]
            assert "standalone topology" in provenance[term]

    def test_the_provenance_is_not_a_target_class(self):
        """`derived_readings` refuses any term whose class starts with X, and
        a composed term must not accidentally read as one."""
        provenance = compose_calibration(BASE, 4)["provenance"]
        assert not provenance["non_torch"].strip().upper().startswith("X")
        assert not provenance["load_residue"].strip().upper().startswith("X")

    def test_derived_readings_accepts_it_and_predicts_at_width(self):
        config = {"text_config": {
            "hidden_size": 5120, "intermediate_size": 17408,
            "linear_num_key_heads": 16, "linear_key_head_dim": 128,
            "linear_num_value_heads": 48, "linear_value_head_dim": 128,
            "vocab_size": 248320, "num_hidden_layers": 62}}
        composed = compose_calibration(BASE, 4)
        files = {"config.json": config, "cal.json": composed}
        profile = {"total": 206141652992, "world_size": 4,
                   "parameters": 13892386272, "buffers": 33554432,
                   "model_config": "config.json", "compile_mode": "inductor",
                   "calibration": "cal.json", "dtype_bytes": 2}
        readings, activation = derived_readings(
            profile, warmup_tokens=16384, load=files.__getitem__)
        assert activation > 0
        assert readings["non_torch"] == composed["non_torch"][4]

    def test_a_calibration_still_has_to_carry_its_provenance(self):
        composed = compose_calibration(BASE, 2)
        composed.pop("provenance")
        files = {"cal.json": composed,
                 "g.json": {"key": {"topology": {"tp": 2}}, "operators": []}}
        profile = {"total": 1 << 40, "world_size": 2, "parameters": 1,
                   "graph": "g.json", "calibration": "cal.json"}
        with pytest.raises(UnfoundedPrediction):
            derived_readings(profile, warmup_tokens=1024,
                             load=files.__getitem__)
