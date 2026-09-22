# SPDX-License-Identifier: MIT
"""Two instruments on one breakdown: the per-term gate, and the sum that hid it.

The memory model records an incident rather than a rule and then draws the
rule from it. A summed non-KV memory check read **+13.8%** and was three
errors, two of which cancelled: weights over by **+0.280 GB**, activations
compared at the wrong shape (**-0.015 GB**), and **-0.084 GB** of a resident
term nobody had noticed existed. **The largest single error was 25% of its own term.**

`HISTORICAL` below is that breakdown, and the named result of this task is the
two instruments run on it side by side -- the per-term comparator naming three
failures, and the summed check reading +13.8% and passing.

**How the fixture's totals were reconstructed, since the record carries deltas
and ratios rather than totals.** Three deltas are given (+0.280, -0.015,
-0.084 GB), and two ratios: the largest error is 25% of its term, and the sum
is +13.8%. The largest error by bytes is the weights one, so the recorded
weights follow exactly: `0.280 / 0.25 = 1.120 GB`. The three deltas sum to
+0.181 GB, so the recorded terms sum to `0.181 / 0.138`, and the rounding of
13.8% pins that sum to `(1.30686, 1.31636]` GB -- leaving the recorded
activations in `(0.10286, 0.11236]` GB. The fixture takes **0.110 GB**, the
only figure the record does not determine, and the resulting sum is +13.77%,
which is the +13.8% the record states. Nothing else in the fixture is chosen.

Every test in this file runs with both device readings patched to raise, for the
same reason the readings tests do: a comparator that needed a card would be
useless for sizing a card nobody has. The import-graph check that settles it
for every branch at once lives in `test_memory_readings.py` and globs the
package, so `compare.py` is inside it already.
"""

import copy
import json
import pathlib
import re

import pytest
import torch
from transformers import PretrainedConfig

import atom.compass.memory as memory_package
from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.memory import (
    NON_KV_TERM_GATE,
    Basis,
    Comparison,
    MemoryRefusal,
    ModelTerms,
    PiecewiseCapture,
    Predicted,
    Recorded,
    Shape,
    SummedCheck,
    Term,
    TermComparison,
    Verdict,
    capture_token_shapes,
    compare,
    compare_graph_pool,
    device_readings,
    footprint_terms,
    piecewise_per_token_bytes,
    predicts,
    reserves,
    tied_lm_head_bytes,
)
from atom.compass.spec import MachineSpec
from tests.compass.test_memory_readings import (
    CAPTURE_SIZES,
    CONFIG_JSON,
    DOCUMENT,
    GPU_MEMORY_UTILIZATION,
    MAX_NUM_BATCHED_TOKENS,
    PARAMETERS,
    WARMUP_TOKENS,
)

MIB = 1 << 20


@pytest.fixture(autouse=True)
def no_device_readings(monkeypatch):
    """Both device readings raise, for every test in this file."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a device reading was taken; a comparator that needs a card cannot "
            "be run against the card it is sizing for"
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse)
    monkeypatch.setattr(torch.cuda, "memory_stats", refuse)


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))


@pytest.fixture(scope="module")
def qwen():
    raw = json.loads(CONFIG_JSON.read_text())
    return PretrainedConfig.from_dict(raw["text_config"])


# --- the historical breakdown, as data ---------------------------------------

#: The shape the incident's activations belonged to. Both sides of the fixture
#: state it, which is the whole of what the -0.015 GB needed and did not have.
HISTORICAL_SHAPE = Shape(tokens=4096, phase="prefill")

#: The terms taken at that shape. Stated, never inferred from a name.
AT_SHAPE = frozenset({"activations"})

#: The record does not name the third term. It is named here for the two things
#: it does say about it: it is resident, and the model that missed it
#: attributed nothing to it.
UNATTRIBUTED = "resident (unattributed)"

HISTORICAL_RECORDED = Recorded(
    run="the summed non-KV check that read +13.8%",
    shape=HISTORICAL_SHAPE,
    high_water_reset=True,
    terms=footprint_terms(
        {
            "weights": 1_120_000_000,
            "activations": 110_000_000,
            UNATTRIBUTED: 84_000_000,
        },
        source="the breakdown the run printed",
    ),
    at_shape=AT_SHAPE,
)

HISTORICAL_PREDICTED = Predicted(
    label="the memory model of the incident",
    shape=HISTORICAL_SHAPE,
    terms=(
        Term(
            "weights",
            1_400_000_000,
            Basis.OBTAINED,
            "a meta build, deduped by storage, that had not been through the loader",
        ),
        Term(
            "activations",
            95_000_000,
            Basis.OBTAINED,
            "a liveness walk of the traced graph",
        ),
    ),
    at_shape=AT_SHAPE,
)


def test_the_named_result_two_instruments_on_one_breakdown(capsys):
    """Three named term errors, beside a sum that reads +13.8% and passes.

    The per-term table and the summed check are printed here rather than only
    asserted, because the argument this task makes is that the two disagree on
    one breakdown and the disagreement is the finding.
    """
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    # This project states no band for a sum. Every band it does state -- the
    # 10% a non-KV term carries, and the 25% the unmeasured-device tier allows
    # one -- is per term, and taking one of them for a sum is the substitution
    # the incident is made of. 25% is the loosest, so it flatters the sum most.
    summed = comparison.summed(band=0.25)
    with capsys.disabled():
        print()
        print(summed.table())

    errors = {t.name: t for t in comparison.compared}
    assert set(errors) == {"weights", "activations", UNATTRIBUTED}
    assert errors["weights"].delta_bytes == 280_000_000
    assert errors["activations"].delta_bytes == -15_000_000
    assert errors[UNATTRIBUTED].delta_bytes == -84_000_000
    assert all(t.verdict is Verdict.FAIL for t in comparison.compared)
    assert len(comparison.failures()) == 3

    largest = comparison.worst()
    assert largest.name == "weights"
    assert largest.relative == pytest.approx(0.25, abs=5e-5)
    # The one the sum cannot see at all: nothing predicted it, so it is absent
    # from the prediction and from any sum over the prediction.
    assert errors[UNATTRIBUTED].predicted is None
    assert errors[UNATTRIBUTED].relative == -1.0

    assert summed.delta_bytes == 181_000_000
    assert summed.relative == pytest.approx(0.13775, abs=5e-5)
    assert f"{summed.relative:+.1%}" == "+13.8%"
    assert summed.passed


def test_the_sum_at_the_per_term_band_still_names_nothing():
    """Even given the per-term 10%, the sum fails without saying which term did.

    This is the half of the argument a looser band hides: the objection to a
    summed check is not that its band was wrong, it is that its answer has no
    decomposition. At 10% this sum is red and a reader is no closer to the
    +0.280 GB than they were at 25%.
    """
    summed = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED).summed(
        band=NON_KV_TERM_GATE
    )
    assert not summed.passed
    aggregate = summed.table().splitlines()[0]
    for name in ("weights", "activations", UNATTRIBUTED):
        assert name not in aggregate


def test_the_two_compensating_errors_are_what_the_sum_folds():
    """+0.280 and -0.099 GB of error make a sum of +0.181 GB, which is the point."""
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    over = sum(t.delta_bytes for t in comparison.compared if t.delta_bytes > 0)
    under = sum(t.delta_bytes for t in comparison.compared if t.delta_bytes < 0)
    assert over == 280_000_000
    assert under == -99_000_000
    assert over + under == comparison.summed(band=0.25).delta_bytes


# --- an aggregate never without its decomposition ---------------------------


def test_a_comparison_has_no_total_and_a_sum_cannot_be_built_without_one():
    assert not hasattr(Comparison, "total")
    assert not hasattr(Comparison, "__int__")
    assert not hasattr(TermComparison, "__int__")
    # The only state a SummedCheck holds is the comparison and the band: there
    # is no constructor that takes two totals, so the figure cannot exist
    # without the table that produced it.
    assert set(SummedCheck.__dataclass_fields__) == {"comparison", "band"}


def test_printing_a_sum_prints_the_terms_it_folded():
    summed = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED).summed(band=0.25)
    rendered = str(summed)
    for name in ("weights", "activations", UNATTRIBUTED):
        assert name in rendered
    assert "+13.8%" in rendered


def test_the_largest_error_comes_back_as_its_term_not_as_a_number():
    largest = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED).worst()
    assert isinstance(largest, TermComparison)
    assert largest.recorded.nbytes == 1_120_000_000
    assert largest.predicted is not None


def test_a_comparison_with_no_comparable_term_refuses_a_largest_error():
    # Every recorded term was taken at a shape and the run did not reset the
    # peak, so every one of them refuses and nothing is left to be largest.
    recorded = Recorded(
        run="a tracing run that did not reset the peak",
        shape=HISTORICAL_SHAPE,
        high_water_reset=False,
        terms=footprint_terms({"activations": 110_000_000}, source="the printout"),
        at_shape=AT_SHAPE,
    )
    predicted = Predicted(
        label="a prediction of that one term",
        shape=HISTORICAL_SHAPE,
        terms=(Term("activations", 95_000_000, Basis.OBTAINED, "a liveness walk"),),
        at_shape=AT_SHAPE,
    )
    comparison = compare(predicted, recorded)
    assert not comparison.compared
    with pytest.raises(MemoryRefusal) as refusal:
        comparison.worst()
    assert "no largest error" in str(refusal.value)


# --- trap 1: a meta build has not been through the loader --------------------


def qwen_0_6b():
    """The published Qwen3-0.6B geometry this correction is recorded against."""
    return PretrainedConfig.from_dict(
        {
            "hidden_size": 1024,
            "vocab_size": 151936,
            "tie_word_embeddings": True,
            "dtype": "bfloat16",
        }
    )


def test_the_tied_head_is_one_embedding_and_the_design_records_its_size():
    config = qwen_0_6b()
    nbytes = tied_lm_head_bytes(config, dtype_bytes=element_bytes(config.dtype))
    assert nbytes == 151_936 * 1_024 * 2
    assert nbytes == 311_164_928
    # The record: "worth one embedding, 0.290 GiB on the 0.6B".
    assert round(nbytes / (1 << 30), 3) == 0.290


def test_an_untied_model_owes_no_correction(qwen):
    assert qwen.tie_word_embeddings is False
    assert tied_lm_head_bytes(qwen, dtype_bytes=2) == 0


def test_a_config_that_does_not_state_the_tie_refuses_rather_than_assuming():
    # The absence is observable, which is checked here rather than assumed:
    # this tree's transformers does not fill the field in, so a config that was
    # never given it raises on the attribute instead of answering. Assuming
    # untied is exactly the 0.290 GiB gap, and it is the same species as the
    # absent partial_rotary_factor that made the buffers term 4x high without
    # saying it had assumed anything.
    config = PretrainedConfig.from_dict({"hidden_size": 1024, "vocab_size": 151936})
    assert not hasattr(config, "tie_word_embeddings")
    with pytest.raises(MemoryRefusal) as refusal:
        tied_lm_head_bytes(config, dtype_bytes=2)
    assert "tie_word_embeddings" in str(refusal.value)


#: The 0.6B's resident weight total is not in the design record. This fixture
#: states one so a percentage can be shown; the only figure the test below pins
#: to the record is the tie itself.
RECORDED_0_6B_WEIGHTS = 1_192_000_000


def weights_comparison(predicted_bytes):
    predicted = Predicted(
        label="a meta build of the 0.6B",
        shape=Shape(tokens=4096, phase="prefill"),
        terms=(
            Term(
                "weights",
                predicted_bytes,
                Basis.OBTAINED,
                "a meta build, deduped by storage",
            ),
        ),
    )
    recorded = Recorded(
        run="fixture: the 0.6B after the loader tied its head",
        shape=Shape(tokens=4096, phase="prefill"),
        high_water_reset=True,
        terms=footprint_terms(
            {"weights": RECORDED_0_6B_WEIGHTS}, source="parameter_bytes"
        ),
    )
    return compare(predicted, recorded).compared[0]


def test_the_invisible_tied_head_fails_the_gate_and_the_correction_closes_it():
    tie = tied_lm_head_bytes(qwen_0_6b(), dtype_bytes=2)
    before = weights_comparison(RECORDED_0_6B_WEIGHTS + tie)
    assert before.verdict is Verdict.FAIL
    assert before.delta_bytes == tie
    assert before.relative == pytest.approx(0.2610, abs=5e-5)

    after = weights_comparison(RECORDED_0_6B_WEIGHTS + tie - tie)
    assert after.verdict is Verdict.PASS
    assert after.delta_bytes == 0


# --- trap 2: a term the run does not record ----------------------------------


def ladder():
    return capture_token_shapes(
        CAPTURE_SIZES, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS
    )


def reserved(qwen, total_bytes):
    return reserves(
        piecewise=PiecewiseCapture(
            per_token_bytes=piecewise_per_token_bytes(
                hidden_size=qwen.hidden_size,
                layers=qwen.num_hidden_layers,
                dtype_bytes=element_bytes(qwen.dtype),
            ),
            token_shapes=ladder(),
            budget_bytes=int(total_bytes * GPU_MEMORY_UTILIZATION),
        )
    )


def live_readings(spec, qwen, tp_width=1):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    return device_readings(
        spec,
        tp_width=tp_width,
        model=ModelTerms.declared_for_m1(
            qwen,
            parameter_count=PARAMETERS,
            tp_size=tp_width,
            warmup_tokens=WARMUP_TOKENS,
        ),
        cudagraph_overhead=reserved(qwen, total_bytes),
    )


def live_prediction(spec, qwen, tp_width=1):
    return Predicted.from_readings(
        live_readings(spec, qwen, tp_width),
        shape=Shape(tokens=WARMUP_TOKENS, phase="prefill"),
        at_shape=AT_SHAPE,
    )


def agreeing_recording(prediction, *, drop=("buffers",)):
    """A run whose every recorded term is the predicted byte count exactly.

    The strongest form of the discharge question: with the two sides equal to
    the byte, whether a term's gate is discharged turns on nothing but where
    the predicted number came from.

    `buffers` is dropped because the split at source is three
    non-subtractive readings -- `parameter_bytes`, `weights_torch`,
    `current_torch` -- and none of them is buffers. Buffers are not parameters,
    so they fall inside `weights_torch - parameter_bytes` together with every
    other resident non-parameter allocation; isolating them needs the recording
    the memory model asks for, which is a separate instrument.
    """
    return Recorded(
        run="fixture: every recorded term set to the predicted byte count",
        shape=prediction.shape,
        high_water_reset=True,
        terms=footprint_terms(
            {t.name: t.nbytes for t in prediction.terms if t.name not in drop},
            source="the breakdown the run printed",
        ),
        at_shape=prediction.at_shape - set(drop),
    )


def test_a_term_the_run_does_not_record_refuses_and_carries_its_own_reason(
    spec, qwen, capsys
):
    """`buffers` refuses by name, and the refusal quotes the reason.

    The rule for this term is *recorded, not formula'd*: the formula that
    matched the 0.6B exactly was 4x wrong on the 27B, was tested on a second
    model, failed and did not ship. The replacement that shipped was itself 4x
    high and now says on the term that it is derived from ATOM'"'"'s rotary source
    and validated against no card. A comparator cannot discharge a 10% gate on
    a term with no recording, and the refusal is the result.
    """
    prediction = live_prediction(spec, qwen)
    comparison = compare(prediction, agreeing_recording(prediction))
    with capsys.disabled():
        print()
        print(comparison.table())

    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"buffers"}
    message = str(refused["buffers"])
    assert "records no 'buffers'" in message
    assert "not a term of zero bytes" in message
    assert "validated against no card" in message
    assert "buffers" not in {t.name for t in comparison.compared}


def test_an_absent_term_is_not_a_term_of_zero_bytes(spec, qwen):
    # The failure mode this refusal exists to stop: a comparator that filled in
    # a zero would report buffers as -100% and pass a reader a number.
    prediction = live_prediction(spec, qwen)
    comparison = compare(prediction, agreeing_recording(prediction))
    assert all(t.name != "buffers" for t in comparison.compared)
    assert comparison.summed(band=0.25).predicted_total == sum(
        t.predicted_bytes for t in comparison.compared
    )


def test_the_two_terms_that_cannot_discharge_their_gate_while_agreeing_exactly(
    spec, qwen
):
    """Every recorded term equals its prediction, and two gates still do not close.

    `weights` is a declared coefficient over a **round 27e9**, and the note it
    carries names a second reason that a better parameter count would not
    touch: it shards every parameter, where a real stack replicates its norms.
    `activations` is a declared formula over one live layer, and the memory
    open issue says a graph without the invisible-scratch table does not
    discharge the 10% gate on it.
    """
    prediction = live_prediction(spec, qwen)
    comparison = compare(prediction, agreeing_recording(prediction))

    by_name = {t.name: t for t in comparison.compared}
    assert all(t.delta_bytes == 0 for t in comparison.compared)
    assert {t.name for t in comparison.not_discharged()} == {"weights", "activations"}
    assert not comparison.failures()
    assert {t.name for t in comparison.compared if t.verdict is Verdict.PASS} == {
        "load residue",
        "persistent",
        "driver and collective reserve",
    }

    weights = by_name["weights"].why
    assert "27000000000 stated parameters" in weights
    assert "shards every parameter, where a real stack replicates its norms" in weights
    activations = by_name["activations"].why
    assert "liveness walk" in activations and "invisible-scratch" in activations


def test_a_declared_term_can_fail_its_gate_even_though_it_cannot_pass_one():
    # The asymmetry is deliberate: a recording that contradicts a coefficient
    # is evidence, and agreement with one run is not a measurement.
    declared = Term(
        "activations",
        200_000_000,
        Basis.DECLARED,
        "a coefficient over one live layer",
        "a liveness walk over a traced op graph",
    )
    predicted = Predicted(
        label="a declared activation term",
        shape=HISTORICAL_SHAPE,
        terms=(declared,),
        at_shape=AT_SHAPE,
    )
    recorded = Recorded(
        run="a run that contradicts it",
        shape=HISTORICAL_SHAPE,
        high_water_reset=True,
        terms=footprint_terms({"activations": 100_000_000}, source="the printout"),
        at_shape=AT_SHAPE,
    )
    assert compare(predicted, recorded).compared[0].verdict is Verdict.FAIL


# --- trap 3: compared at the wrong shape -------------------------------------


def test_a_pair_taken_at_two_shapes_refuses_and_names_both():
    # 3,494 tokens is the trace the analytic memory law scales to an
    # 4,096-token peak. Comparing the one against the other without saying so
    # is the -0.015 GB of the incident.
    at_3494 = Recorded(
        run="the 3,494-token trace",
        shape=Shape(tokens=3494, phase="prefill"),
        high_water_reset=True,
        terms=HISTORICAL_RECORDED.terms,
        at_shape=AT_SHAPE,
    )
    comparison = compare(HISTORICAL_PREDICTED, at_3494)
    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"activations"}
    message = str(refused["activations"])
    assert "3494 tokens, prefill" in message
    assert "4096 tokens, prefill" in message
    # The terms that do not move with the shape are still compared, so a shape
    # disagreement costs the activation row and nothing else.
    assert {t.name for t in comparison.compared} == {"weights", UNATTRIBUTED}


def test_the_table_states_the_shape_of_each_side():
    rendered = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED).table()
    assert "predicted at 4096 tokens, prefill" in rendered
    assert "recorded at 4096 tokens, prefill" in rendered


def test_two_shapes_and_no_side_naming_a_shaped_term_refuses_outright():
    # A comparison that cannot say which of its terms move with the shape
    # cannot say whether the two sides measured the same thing at all.
    predicted = Predicted(
        label="a prediction that names no shaped term",
        shape=Shape(tokens=8192, phase="prefill"),
        terms=HISTORICAL_PREDICTED.terms,
    )
    recorded = Recorded(
        run="a run at another shape",
        shape=Shape(tokens=4096, phase="prefill"),
        high_water_reset=True,
        terms=HISTORICAL_RECORDED.terms,
    )
    with pytest.raises(MemoryRefusal) as refusal:
        compare(predicted, recorded)
    assert "neither names a term it took at one" in str(refusal.value)


def test_a_side_that_claims_a_shaped_term_it_does_not_carry_is_rejected():
    with pytest.raises(ValueError) as bad:
        Recorded(
            run="a run",
            shape=HISTORICAL_SHAPE,
            high_water_reset=True,
            terms=footprint_terms({"weights": 1}, source="the printout"),
            at_shape=frozenset({"activations"}),
        )
    assert "records no such term" in str(bad.value)


# --- trap 4: the allocator's high-water mark ---------------------------------


def test_a_run_that_did_not_reset_the_peak_refuses_every_shaped_term():
    not_reset = Recorded(
        run="a tracing run that did not reset the peak",
        shape=HISTORICAL_SHAPE,
        high_water_reset=False,
        terms=HISTORICAL_RECORDED.terms,
        at_shape=AT_SHAPE,
    )
    comparison = compare(HISTORICAL_PREDICTED, not_reset)
    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"activations"}
    message = str(refused["activations"])
    assert "warmup prefill" in message
    assert "-14.7%" in message
    assert {t.name for t in comparison.compared} == {"weights", UNATTRIBUTED}


def test_whether_the_peak_was_reset_has_no_default():
    with pytest.raises(TypeError):
        Recorded(
            run="a run that does not say",
            shape=HISTORICAL_SHAPE,
            terms=HISTORICAL_RECORDED.terms,
        )


def test_a_recorded_term_that_was_not_read_off_a_card_is_rejected():
    with pytest.raises(ValueError) as bad:
        Recorded(
            run="a run whose terms are formulas",
            shape=HISTORICAL_SHAPE,
            high_water_reset=True,
            terms=(Term("weights", 1, Basis.SPEC, "a spec field"),),
        )
    assert "Basis.OBTAINED" in str(bad.value)


# --- the graph pool: two numbers, both reported ------------------------------

#: The measured pool at this ladder: 91.1 MiB + 0.3033 MiB per
#: captured token at width 1, and flat 104 MiB above it. This is the recording
#: the two functions are compared against; the spec's own fields are the
#: authored, rounded form of the same line, which is why `predicts()` lands on
#: top of it and `reserves()` does not.
RECORDED_POOL_W1 = int(91.1 * MIB + 0.3033 * MIB * 1023)
RECORDED_POOL_W_GT1 = 104 * MIB


def recorded_pool(nbytes, where):
    return Term("recorded pool", int(nbytes), Basis.OBTAINED, where)


@pytest.mark.parametrize(
    "tp_width, pool, where",
    [
        (1, RECORDED_POOL_W1, "the measured line: 91.1 MiB + 0.3033 MiB x 1023 tokens"),
        (2, RECORDED_POOL_W_GT1, "the measured line: flat 104 MiB above width 1"),
    ],
)
def test_both_graph_pool_numbers_are_reported_against_the_recorded_pool(
    spec, qwen, tp_width, pool, where, capsys
):
    """Neither function is picked, and the one that reserves is labelled.

    The memory model keeps the two apart because they disagree by 4-19x. A
    that reported one of them would have reconciled what the design says to
    keep apart, so this one has a row for each and no accessor for *the* error.
    """
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    comparison = compare_graph_pool(
        recorded_pool(pool, where),
        reserves=reserved(qwen, total_bytes),
        predicts=predicts(spec, tp_width=tp_width, captured_tokens=sum(ladder())),
    )
    with capsys.disabled():
        print()
        print(comparison.table())

    assert comparison.reserves.total == 1_877_213_184
    # The measured predictor lands on the recording; the one that reserves does
    # not, and it is the one ATOM subtracts from the budget.
    assert abs(comparison.predicts_relative) < 0.001
    assert comparison.reserves_relative > 3.0
    rendered = comparison.table()
    assert "reserves the memory" in rendered
    assert "reserves nothing" in rendered
    assert f"{comparison.reserves_relative:+.2%}" in rendered
    assert f"{comparison.predicts_relative:+.2%}" in rendered


def test_the_recorded_band_is_the_four_to_nineteen_times_the_design_records(spec, qwen):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    captured = sum(ladder())
    ratios = {}
    for tp_width in (1, 2):
        comparison = compare_graph_pool(
            recorded_pool(
                RECORDED_POOL_W1 if tp_width == 1 else RECORDED_POOL_W_GT1,
                "the measured line",
            ),
            reserves=reserved(qwen, total_bytes),
            predicts=predicts(spec, tp_width=tp_width, captured_tokens=captured),
        )
        ratios[tp_width] = comparison.disagreement
    assert ratios[1] == pytest.approx(4.4609, abs=5e-5)
    assert ratios[2] == pytest.approx(17.2221, abs=5e-5)


def test_the_two_graph_pool_readings_cannot_be_handed_in_the_wrong_way_round(
    spec, qwen
):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    reserving = reserved(qwen, total_bytes)
    predicting = predicts(spec, tp_width=1, captured_tokens=sum(ladder()))
    with pytest.raises(MemoryRefusal) as refusal:
        compare_graph_pool(
            recorded_pool(RECORDED_POOL_W1, "the measured line"),
            reserves=predicting,
            predicts=reserving,
        )
    assert "cudagraph_pool" in str(refusal.value)
    assert "it is the one that reserves" in str(refusal.value)


def test_the_graph_pool_cannot_be_folded_into_the_per_term_table(spec, qwen):
    # One row for the pool would pick one of the two numbers, and picking one
    # is what throws the finding away.
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    prediction = live_prediction(spec, qwen)
    pool = reserved(qwen, total_bytes)
    folded = Predicted(
        label="a prediction that folded the pool in",
        shape=prediction.shape,
        terms=prediction.terms
        + (
            Term(
                pool.name,
                pool.total,
                Basis.DECLARED,
                "ATOM's own estimator",
                "the measured pool, which disagrees by 4-19x",
            ),
        ),
        at_shape=prediction.at_shape,
    )
    recording = agreeing_recording(prediction)
    with pytest.raises(MemoryRefusal) as refusal:
        compare(folded, recording)
    assert "compare_graph_pool" in str(refusal.value)


# --- what `from_readings` takes, and what it deliberately leaves -------------


def test_the_footprint_is_peak_torch_and_non_torch_and_nothing_else(spec, qwen):
    readings = live_readings(spec, qwen)
    prediction = Predicted.from_readings(
        readings, shape=Shape(tokens=WARMUP_TOKENS, phase="prefill"), at_shape=AT_SHAPE
    )
    assert [t.name for t in prediction.terms] == [
        "weights",
        "buffers",
        "load residue",
        "persistent",
        "activations",
        "driver and collective reserve",
    ]
    # `total` is the card and `free` is what is left of it, so neither is a
    # footprint term; `cudagraph_overhead` is one of two numbers.
    assert "capacity" not in {t.name for t in prediction.terms}
    assert "per-token x captured tokens" not in {t.name for t in prediction.terms}
    assert readings.spec_digest in prediction.label


# --- the package says what the code does, and cites nothing ------------------

#: Built from parts so that this pattern does not match its own source, which
#: lets the guard below read the file it is written in if it is ever widened.
_TAGS = re.compile(
    r"\b[DTW]\d+(\.\d+)?\b|principles? \d+|Gate \d+|`\d{2}`|\bMEM-\d+\b|#\d+"
)

#: The package as the suite imported it, never a walk up from this file: if
#: `atom` resolves from another root, a path-derived location would scan one
#: tree while every other test here imports another, and pass.
PACKAGE = pathlib.Path(memory_package.__file__).parent


@pytest.mark.parametrize(
    "module", sorted(p.name for p in pathlib.Path(PACKAGE).glob("*.py"))
)
def test_no_module_in_the_package_carries_a_design_reference(module):
    """Say what the code does. Nothing here points at a document by number.

    The rule is mechanical and it reaches runtime data, which is the half that
    matters most here: two refusal strings and every `why` this module renders
    are emitted output, and a tag in one of them is a citation in whatever
    record that output lands in.

    This is parametrised over a glob rather than a list, so it covers a module
    added after it was written -- the same shape as the device-free guard in
    `test_memory_readings.py`, and for the same reason.
    """
    found = _TAGS.findall((PACKAGE / module).read_text())
    assert not found, f"{module} carries {len(found)} design references"


def test_the_guard_catches_the_forms_that_were_actually_removed():
    """A guard nobody drove is a guard nobody knows the reach of.

    Each string below was in this package before this cut swept it, and each
    is a form the rule names. Driving them is how the guard is shown to hold
    in the direction that matters -- it would be worth nothing if it only ever
    saw text that was already clean.
    """
    removed = [
        "03 D16 records buffers rather than computing them",
        "a meta build deduped by storage (02 D10.1) replaces this",
        "the liveness walk of 04 D22 plus the invisible-scratch constants of 04 T4",
        "above width 1 (03 D15)",
        "a number without one is a defect (principle 8)",
        "an open owner ruling (**#87**)",
        "proving it is MEM-2's",
        "`16` row W2.2",
    ]
    for text in removed:
        assert _TAGS.search(text), text
    assert not _TAGS.search("buffers are recorded rather than computed")
