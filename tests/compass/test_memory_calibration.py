"""The source-calibration boundary: what may be fitted, and what that proves."""

import hashlib
import json
from pathlib import Path

import pytest

from atom.compass.core.feasibility import blocks_from_readings
from atom.compass.core.memory import MemoryReadings
from atom.compass.core.execution_id import (EXECUTION_SCHEMA, ID_INPUTS,
                                            derive_execution_id)
from atom.compass.core.memory_calibration import (CalibratedTerm,
                                                  SourceCalibration, SourceRun,
                                                  for_model, producer_key)

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


#: One run of the source configuration, named the way the campaign harness
#: names runs: `compare.py` reads a record's `run` block, `merge_sweep.py`
#: labels fresh-process shards by `started_at`, `residual.py` attributes steps
#: by `pid`.
def _run_block(pid: int, launched_at_ns: int) -> dict:
    """A record's `run` block carrying a well-formed `compass.execution/1` id.

    Built through the shared derivation rather than written out, so that a
    change to the rule breaks these tests instead of silently making the
    fixtures the only thing still following the old one.
    """
    inputs = {"host": "hjbog-srdc-18", "cell": "RESULTS/tp1_source",
              "side": "real", "repeat": 1, "server_pid": pid,
              "launched_at_ns": launched_at_ns}
    return {"execution": {
        "schema": EXECUTION_SCHEMA,
        "execution_id": derive_execution_id(*[inputs[n] for n in ID_INPUTS]),
        "id_inputs": inputs}}


def _calibration_with_producers(*producers: str) -> SourceCalibration:
    """A calibration whose source run names the executions it was fitted on.

    The shipped one cannot: its records carry no producer block, so its
    `producers` is empty by construction and every classification falls to the
    conservative answer. The distinction still has to be tested.
    """
    run = SourceRun(
        model=MODEL, tensor_parallel=1, gpu_memory_utilization=0.9,
        max_num_seqs=32, max_model_len=262144,
        capture_sizes=(1, 2, 4, 8, 16, 32), enable_prefix_caching=False,
        record="tests/compass/memory_records/27b.tp1.exclusive.memory.json",
        record_sha256=_sha(RECORDS / "27b.tp1.exclusive.memory.json"),
        producers=producers,
        producer_basis="synthetic, for the test",
    )
    return SourceCalibration(
        model=MODEL,
        terms={"persistent": CalibratedTerm(
            term="persistent", value=252339712, run=run, basis="test")},
    )


def test_a_rerun_of_the_source_configuration_is_a_repeat_not_a_residual():
    """Same configuration, different run: reproducibility, not transfer.

    Calling the agreement a validation would overclaim -- nothing about the
    configuration changed -- and calling it a residual would hide that two
    independent runs agree. The separator is the producer, not the bytes.
    """
    first = _run_block(695009, 1757580940000000000)
    second = _run_block(751439, 1757581240000000000)
    calib = _calibration_with_producers(producer_key(first))
    config = _record("27b.tp1.exclusive.memory.json")["config"]

    assert calib.classify("persistent", config, producer=first) == "residual"
    assert calib.classify("persistent", config, producer=second) == "repeat"
    assert (
        calib.classify("persistent",
                       _record("27b.tp4.rank0.memory.json")["config"],
                       producer=second)
        == "validation"
    )


def test_identical_bytes_are_not_one_run_and_the_hash_cannot_say_otherwise():
    """The phase C case, which is why identity stopped being a hash.

    Three executions of the source configuration -- three processes, three
    device allocations, three teardowns, witnessed by the ownership sampler --
    wrote records that are equal byte for byte. A classifier keyed on the hash
    reads that as one run and reports two repeats as residuals, which is the
    direction that overstates nothing and understates the only reproducibility
    evidence the calibration has.
    """
    first, second, third = (_run_block(695009, 1757580940000000000),
                            _run_block(751439, 1757581240000000000),
                            _run_block(775417, 1757581350000000000))
    assert len({producer_key(r) for r in (first, second, third)}) == 3

    calib = _calibration_with_producers(producer_key(first))
    config = _record("27b.tp1.exclusive.memory.json")["config"]
    sha = _sha(RECORDS / "27b.tp1.exclusive.memory.json")

    # One artifact, three producers: the bytes are the same in all three.
    assert calib.integrity("persistent", sha) == "intact"
    assert [calib.classify("persistent", config, producer=run)
            for run in (first, second, third)] == ["residual", "repeat",
                                                   "repeat"]


def test_a_reserialised_record_is_the_same_run_with_a_different_hash():
    """The other direction: bytes change, the execution does not.

    Re-indenting a record produces a file no hash-keyed classifier recognises,
    and a classifier that treated an unrecognised hash as a different run would
    manufacture a repeat out of a text edit. The producer is unchanged, so the
    classification is unchanged; the integrity answer is the one that moves.
    """
    calib = _calibration_with_producers(producer_key(_run_block(695009, 1757580940000000000)))
    blob = _record("27b.tp1.exclusive.memory.json")
    reserialised = hashlib.sha256(
        json.dumps(blob, indent=4).encode("utf-8")).hexdigest()

    assert reserialised != calib.terms["persistent"].run.record_sha256
    assert calib.integrity("persistent", reserialised) == "altered"
    assert (
        calib.classify("persistent", blob["config"],
                       producer=_run_block(695009, 1757580940000000000))
        == "residual"
    )


def test_an_unidentified_run_is_classified_as_the_weaker_claim():
    """Which is every record the campaign has written so far.

    An id is never inferred. A legacy block naming a host and a pid is not an
    execution identity and is not promoted into one; an id that does not follow
    from its own recorded inputs is damaged or transplanted, which is worse
    than unidentified rather than better; and an id under a schema this module
    does not know is one it cannot check. All three read as "cannot tell",
    because the alternative -- upgrading a residual to a repeat on a guess --
    is the error this classification exists to prevent.
    """
    assert producer_key(None) is None
    assert producer_key({"host": "h", "pid": 695009,
                         "started_at": "2026-09-11T08:55:40Z"}) is None

    identified = _run_block(695009, 1757580940000000000)["execution"]
    assert producer_key(identified) == identified["execution_id"]

    transplanted = dict(identified, id_inputs=dict(
        identified["id_inputs"], server_pid=695010))
    assert producer_key(transplanted) is None

    later = dict(identified, schema="compass.execution/2")
    assert producer_key(later) is None

    calib = for_model(MODEL)
    exclusive = _record("27b.tp1.exclusive.memory.json")
    # The shipped calibration names no producers, so even a well-formed one
    # cannot promote the answer.
    assert calib.terms["non_torch"].run.producers == ()
    assert calib.classify("non_torch", exclusive["config"]) == "residual"
    assert (
        calib.classify("non_torch", exclusive["config"],
                       producer=_run_block(999999, "2026-09-11T09:00:00Z"))
        == "residual"
    )
    assert calib.classify("persistent",
                          _record("27b.tp4.rank0.memory.json")["config"]) \
        == "validation"


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


def test_only_the_fitted_run_can_have_its_bytes_called_altered():
    """`integrity` answers a question nobody asked of an unrelated record.

    A hash mismatch means "not the fitted artifact", and for a record that was
    never the fitted artifact -- a different capture, another TP -- that is
    already what `classify` says. Reporting it as tampering there fired on
    every row of every record but one. `identifies` is the guard: the bytes
    are only worth challenging once the producer claims to be that run.
    """
    fitted = _run_block(695009, 1757580940000000000)
    other = _run_block(751439, 1757581240000000000)
    calib = _calibration_with_producers(producer_key(fitted))
    foreign_sha = _sha(RECORDS / "27b.tp1.memory.json")

    assert calib.integrity("persistent", foreign_sha) == "altered"
    assert calib.identifies("persistent", fitted) is True
    assert calib.identifies("persistent", other) is False
    assert calib.identifies("persistent", None) is False
    assert calib.identifies("no_such_term", fitted) is False


def test_the_shipped_calibration_identifies_nobody():
    """Not a limitation of the guard -- of the records.

    Neither source record carries a run block, so no producer can match, and
    every shipped term answers the conservative way: a residual that names no
    execution, and bytes that are never challenged. This test is what will
    fail, loudly and correctly, once the runner starts writing a run block
    and the constants are refitted against an identified execution.
    """
    calib = for_model(MODEL)
    producer = _run_block(695009, 1757580940000000000)
    for term in calib.terms:
        assert calib.terms[term].run.producers == ()
        assert calib.identifies(term, producer) is False


class TestTheActivationTermIsAPeakAtAShape:
    """The warmup peak is measured, and carries its batch geometry with it.

    Unlike `persistent` and `non_torch`, which are flat across the six
    utilizations and three widths they were checked at, this term is the peak
    of one dummy prefill of one shape. Stretching it to another
    `max_num_batched_tokens` would be an extrapolation nobody measured, so the
    match holds the shape and the calibration withholds the number rather than
    scaling it.
    """

    def test_it_is_the_records_own_warmup_peak_in_both_tp1_runs(self):
        """`peak_torch - current_torch`, to the byte, twice.

        Two independent engine starts -- the historical record and the
        exclusive phase C capture -- agree exactly. That is the whole basis for
        the constant: it could not be derived, because the only TP=1 prefill
        graphs at this shape are meta derivations with no recorded liveness.
        """
        term = for_model(MODEL).terms["activations"]
        for name in ("27b.tp1.memory.json", "27b.tp1.exclusive.memory.json"):
            readings = _record(name)["readings"]
            assert readings["peak_torch"] - readings["current_torch"] == term.value
        assert term.value == 2956984320

    def test_another_token_budget_is_another_warmup_shape(self):
        """`warmup_model` divides the budget, so changing it changes the peak.

        And only this term notices. `persistent` is the engine's own forward
        buffers and does not move with the batch geometry; answering
        `validation` for it here would claim a transfer where nothing changed.
        """
        calib = for_model(MODEL)
        config = dict(_record("27b.tp1.exclusive.memory.json")["config"])
        assert calib.classify("activations", config) == "residual"

        config["max_num_batched_tokens"] = 8192
        assert calib.classify("activations", config) == "validation"
        assert calib.classify("persistent", config) == "residual"

    def test_it_is_not_offered_as_a_model_input_at_any_width(self):
        """`mapping` feeds `modelled_readings`, which takes the peak elsewhere.

        The activation argument there comes from a graph at the *target's* own
        shape. Handing it this constant would substitute the source's warmup
        peak for the target's and let the result be called derived. And above
        width one there is nothing to offer in any case: the TP=2 and TP=4
        readings are class X27, measurements of the configuration under
        prediction, so activations at those widths stay underived.
        """
        calib = for_model(MODEL)
        assert "activations" not in calib.mapping(1)
        assert "activations" not in calib.mapping(2)
        assert "activations" not in calib.mapping(4)
        assert calib.terms["activations"].run.tensor_parallel == 1
        assert calib.classify(
            "activations", _record("27b.tp4.rank0.memory.json")["config"]
        ) == "validation"


def test_the_source_only_ledger_closes_and_says_what_that_is_worth():
    """Every input classed S/C06/S27, and the budget the source run recorded.

    The number is not the finding. Four of the five terms in `peak_torch` were
    read off this record, so reproducing it demonstrates the arithmetic and not
    the model -- it is a residual, and the test is named for what it checks:
    that the ledger is *complete*, that no class X27 reading is needed to close
    it, and that nothing has quietly changed one of the constants.

    `free` is the one term that legitimately differs: the model assumes a clean
    box and the record has the neighbours in it. It binds nothing here, because
    the utilization budget is well below both.
    """
    import json

    from atom.compass.core.kv_geometry import blocks_from_readings
    from atom.compass.core.memory_model import modelled_readings

    calib = for_model(MODEL)
    readings = modelled_readings(
        total_bytes=206141652992, world_size=1, parameters=54713457120,
        buffers=33554432,
        activation_bytes=calib.terms["activations"].value,
        calibration=calib.mapping(1), enforce_eager=False)

    record = _record("27b.tp1.exclusive.memory.json")
    recorded = record["readings"]
    for term in ("peak_torch", "non_torch", "cudagraph_overhead"):
        assert readings[term] == recorded[term], term
    assert readings["free"] < recorded["free"]

    with open(RECORDS / "qwen3_5_27b.config.json", encoding="utf-8") as fh:
        config = json.load(fh)
    plan = blocks_from_readings(
        config, type("R", (), readings)(), utilization=0.90, max_num_seqs=32,
        tensor_parallel=1)
    assert plan.entries == record["blocks"]["pool_entries"]
    assert plan.entries["kv"] == 112772
