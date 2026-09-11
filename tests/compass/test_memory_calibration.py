"""The source-calibration boundary: what may be fitted, and what that proves."""

import json
from pathlib import Path

import pytest

from atom.compass.core.memory_calibration import for_model

RECORDS = Path(__file__).parent / "memory_records"
MODEL = "Qwen/Qwen3.8-27B"


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


def test_non_torch_is_deliberately_absent():
    """It is a device-wide reading, so the source record charges it for others.

    The TP=1 source run recorded 1 191 182 336 B; the same configuration on an
    exclusively-held device recorded 1 157 627 904 B, 32 MiB lower. Calibrating
    from the source record would freeze a neighbour's allocation into the model.
    """
    calib = for_model(MODEL)
    assert "non_torch" not in calib.terms
    assert calib.classify("non_torch", _record("27b.tp1.memory.json")["config"]) == (
        "underived"
    )


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
