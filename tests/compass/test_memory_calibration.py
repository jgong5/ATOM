"""The source-calibration boundary: what may be fitted, and what that proves."""

import hashlib
import json
from pathlib import Path

import pytest

from atom.compass.core.feasibility import blocks_from_readings
from atom.compass.core.memory import MemoryReadings
from atom.compass.core.memory_calibration import for_model

RECORDS = Path(__file__).parent / "memory_records"
MODEL = "Qwen/Qwen3.8-27B"


def _sha(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _record(name: str) -> dict:
    with open(RECORDS / name, encoding="utf-8") as fh:
        return json.load(fh)


def _persistent(blob: dict) -> int:
    r = blob["readings"]
    return r["current_torch"] - r["weights_torch"]


def test_a_fit_checked_against_its_own_run_is_a_residual():
    """The distinction the module exists for.

    `persistent` reproduces the TP=1 source record exactly, because that record
    is where the number came from. Calling that agreement would be quoting the
    fit back to itself, so `classify` refuses to.
    """
    calib = for_model(MODEL)
    source = _record("27b.tp1.memory.json")
    assert calib.terms["persistent"].value == _persistent(source)
    assert calib.classify("persistent", source["config"]) == "residual"


@pytest.mark.parametrize(
    "name,expected,tolerance_bytes",
    [
        ("27b.tp2.rank0.memory.json", 252332544, 8192),
        ("27b.tp4.rank0.memory.json", 252328960, 12288),
        ("27b.tp4.rank1.memory.json", 252328960, 12288),
    ],
)
def test_persistent_transfers_to_widths_the_fit_never_saw(
    name, expected, tolerance_bytes
):
    """TP=2 and TP=4 are targets, so these are validations, not residuals.

    The term is flat in width to within 11 KiB on a 240 MiB term -- which is
    what "flat" should mean when it is asserted, rather than a claim resting on
    a rounded percentage.
    """
    calib = for_model(MODEL)
    blob = _record(name)
    assert _persistent(blob) == expected
    assert calib.classify("persistent", blob["config"]) == "validation"
    assert abs(calib.terms["persistent"].value - expected) <= tolerance_bytes


def test_load_residue_is_not_offered_at_widths_it_was_not_fitted_at():
    """14 MiB at TP=1, 2.1 GB at TP=2: the collective pools dominate it.

    So a TP=1 calibration must not be exported to TP=2 or TP=4. It is not, and
    the C06 table keeps those widths.
    """
    calib = for_model(MODEL)
    assert calib.mapping(1)["load_residue"] == {1: 14924832}
    assert "load_residue" not in calib.mapping(2)
    assert "load_residue" not in calib.mapping(4)
    assert calib.mapping(4)["persistent"] == 252339712


def test_non_torch_comes_from_the_exclusive_run_and_not_the_busy_one():
    """It is a device-wide reading, so the record it is taken from matters.

    The original TP=1 source run recorded 1 191 182 336 B; the same
    configuration on an exclusively-held device recorded 1 157 627 904 B,
    32 MiB lower. Calibrating from the first would freeze a neighbour's
    allocation into the model, so the constant is the exclusive-device value
    and its run points at the exclusive record.
    """
    calib = for_model(MODEL)
    term = calib.terms["non_torch"]
    exclusive = _record("27b.tp1.exclusive.memory.json")
    assert term.value == exclusive["readings"]["non_torch"] == 1157627904
    assert term.run.record.endswith("27b.tp1.exclusive.memory.json")
    # And it is not the busy reading, which is what makes this a choice.
    assert term.value != _record("27b.tp1.memory.json")["readings"]["non_torch"]


def test_non_torch_is_offered_only_at_the_width_it_was_measured_at():
    """Device-wide at TP=1 is one rank's whole card; at TP=4 it is not."""
    calib = for_model(MODEL)
    assert calib.mapping(1)["non_torch"] == {1: 1157627904}
    assert "non_torch" not in calib.mapping(2)
    assert "non_torch" not in calib.mapping(4)


def test_the_calibrated_non_torch_carries_its_own_counterexample():
    """The +42% run is in the record, not filed away as an outlier.

    Substituting the constant for the reading predicts the frozen ladder
    exactly at three rungs and misses the fourth by +45.1%, because that run
    read 486 MiB more than every other exclusive-device run of the same
    configuration. Nobody knows why yet. A constant that cannot bound that has
    to say so, and the test is here so that removing the warning breaks.
    """
    term = for_model(MODEL).terms["non_torch"]
    assert any("KNOWN RISK" in note for note in term.validated_against)
    assert any("1 677 721 600" in note for note in term.validated_against)


def test_the_source_run_is_a_configuration_not_a_model():
    """Same model, different utilization or concurrency, is an independent run."""
    calib = for_model(MODEL)
    run = calib.terms["persistent"].run
    base = {
        "model": MODEL,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 32,
        "topology": {"tp": 1},
    }
    assert run.matches(base)
    assert not run.matches(base | {"gpu_memory_utilization": 0.4})
    assert not run.matches(base | {"max_num_seqs": 1400})
    assert not run.matches(base | {"topology": {"tp": 4}})
    assert not run.matches(base | {"model": "Qwen/Qwen3-0.6B"})


def test_an_unmeasured_model_gets_nothing_rather_than_a_neighbour():
    assert for_model("Qwen/Qwen3-0.6B") is None
    assert for_model("some/model-nobody-ran") is None


# ── what the calibrated constant predicts ─────────────────────────────────


def _model_config() -> dict:
    with open(RECORDS / "qwen3_5_27b.config.json", encoding="utf-8") as fh:
        return json.load(fh)


def _readings_of(blob: dict, *, non_torch: int) -> MemoryReadings:
    r = blob["readings"]
    return MemoryReadings(
        total=r["total"],
        free=r["free"],
        peak_torch=r["peak_torch"],
        non_torch=non_torch,
        cudagraph_overhead=r["cudagraph_overhead"],
    )


def _blocks(blob: dict, *, utilization: float, non_torch: int) -> int:
    plan = blocks_from_readings(
        _model_config(),
        _readings_of(blob, non_torch=non_torch),
        utilization=utilization,
        max_num_seqs=32,
        tensor_parallel=1,
        block_size=16,
    )
    return plan.paged_entries


@pytest.mark.parametrize(
    "tag,utilization,engine",
    [("settled_u0.33", 0.33, 1583), ("settled_u0.40", 0.40, 15238)],
)
def test_the_calibrated_constant_predicts_the_ladder_without_reading_it(
    tag, utilization, engine
):
    """The point of a constant: predict a run's block count without measuring it.

    These rungs were run at utilizations the source configuration never used,
    and the calibrated `non_torch` puts the block count on the nose at each of
    them. `0.41 -> 17188` is the same result on the phase A ladder.
    """
    blob = _record("g3_util_phasebc.json")["phase_b"]["runs"][tag]
    got = _blocks(
        {"readings": blob["readings"]},
        utilization=utilization,
        non_torch=for_model(MODEL).terms["non_torch"].value,
    )
    assert blob["blocks"] == engine
    assert got == engine


def test_the_one_run_the_constant_does_not_predict():
    """The +42.2% failure, kept where it cannot be quietly dropped.

    Phase A's first 0.33 run read 1 677 721 600 B and the engine gave it 1091
    blocks. The constant predicts 1583 for that run -- +45.1% -- and three
    controls have failed to reproduce the excursion, so the miss is recorded
    rather than explained. If this test ever starts failing because the
    constant changed, the excursion still happened.
    """
    phase_a = _record("g3_util_phasea.json")
    rung = phase_a["rungs"]["0.33"]
    assert rung["num_kvcache_blocks"] == 1091
    assert rung["readings"]["non_torch"] == 1677721600
    got = _blocks(
        {"readings": rung["readings"]},
        utilization=0.33,
        non_torch=for_model(MODEL).terms["non_torch"].value,
    )
    assert got == 1583
    assert (
        round(
            100.0 * (got - rung["num_kvcache_blocks"]) / rung["num_kvcache_blocks"], 1
        )
        == 45.1
    )


def test_a_rerun_of_the_source_configuration_is_a_repeat_not_a_residual():
    """Same configuration, different run: reproducibility, not transfer.

    `persistent` was fitted on the original TP=1 record and the exclusive-device
    record is a second run of that same configuration. Calling the agreement a
    validation would overclaim -- nothing about the configuration changed -- and
    calling it a residual would hide that two independent runs agree. It needs
    the record's hash to tell them apart, so without one the stricter answer
    stands.
    """
    calib = for_model(MODEL)
    original = _record("27b.tp1.memory.json")
    exclusive = _record("27b.tp1.exclusive.memory.json")
    fitted_on = calib.terms["persistent"].run.record_sha256

    assert (
        calib.classify("persistent", original["config"], record_sha256=fitted_on)
        == "residual"
    )
    assert (
        calib.classify(
            "persistent",
            exclusive["config"],
            record_sha256=_sha(RECORDS / "27b.tp1.exclusive.memory.json"),
        )
        == "repeat"
    )
    assert calib.classify("persistent", exclusive["config"]) == "residual"
    assert (
        calib.classify(
            "persistent",
            _record("27b.tp4.rank0.memory.json")["config"],
            record_sha256="whatever",
        )
        == "validation"
    )


def test_the_two_tp1_records_are_different_runs_of_one_configuration():
    """Which is what makes the labelling above a real distinction.

    They agree on everything the engine computes and differ by 32 MiB on the
    one device-wide reading, which is the neighbour the exclusive run did not
    have.
    """
    original = _record("27b.tp1.memory.json")["readings"]
    exclusive = _record("27b.tp1.exclusive.memory.json")["readings"]
    for term in (
        "peak_torch",
        "weights_torch",
        "parameter_bytes",
        "current_torch",
        "cudagraph_overhead",
    ):
        assert original[term] == exclusive[term], term
    assert original["non_torch"] - exclusive["non_torch"] == 33554432
