# SPDX-License-Identifier: MIT
"""A step's total, and the parts it is folded from.

Two properties are asserted structurally rather than by inspection. First, the
total is not stored: it is folded from the terms on every read, so no sequence
of calls leaves a whole that disagrees with its parts. Second, the fold order
is fixed and observable -- the same terms in a different order are a different
number, bit for bit, which is why the order is part of the value and not an
implementation detail of whoever writes the next summing loop.

The empty-breakdown refusal has a specific failure behind it: the mean of an
empty sample is 0.0, which is not an error and not obviously wrong, and it
reports a step that took no time rather than a model that was missing.
"""

import math

import pytest

from atom.compass.backends import (
    CostTerm,
    Provenance,
    ProvenanceMix,
    Refusal,
    Species,
    StepCost,
    fold_seconds,
)

MEASURED = Provenance(Species.MEASURED, "step-level")
ANALYTIC = Provenance(Species.ANALYTICAL, "roofline")


def term(name, seconds, provenance=MEASURED):
    return CostTerm(name, seconds, provenance)


def test_the_total_is_the_fold_of_the_parts():
    step = StepCost(
        [term("attention", 0.004), term("mlp", 0.006), term("collective", 0.001)]
    )
    assert step.seconds == fold_seconds([0.004, 0.006, 0.001])
    assert step.seconds == pytest.approx(0.011)


def test_a_reader_can_check_the_total_against_the_rows():
    """The artifact's rows re-fold to the artifact's total, bit for bit."""
    step = StepCost([term("a", 0.1), term("b", 0.2), term("c", 0.3)])
    rows = step.rows()
    assert fold_seconds(seconds for _, seconds, _ in rows) == step.seconds


def test_the_fold_order_is_part_of_the_answer():
    """0.1 + 0.2 + 0.3 is not 0.3 + 0.2 + 0.1 in binary floating point."""
    forward = StepCost([term("a", 0.1), term("b", 0.2), term("c", 0.3)])
    reverse = StepCost([term("c", 0.3), term("b", 0.2), term("a", 0.1)])
    assert forward.seconds == 0.6000000000000001
    assert reverse.seconds == 0.6
    assert forward.seconds != reverse.seconds


def test_an_aggregate_with_no_decomposition_cannot_be_built():
    with pytest.raises(ValueError, match="no decomposition"):
        StepCost([])


def test_terms_are_read_by_name_so_names_are_unique():
    with pytest.raises(ValueError, match="duplicate term"):
        StepCost([term("attention", 0.001), term("attention", 0.002)])


def test_a_term_must_carry_a_provenance():
    with pytest.raises(TypeError):
        CostTerm("attention", 0.001)


@pytest.mark.parametrize("seconds", [-1e-9, float("nan"), math.inf])
def test_a_term_refuses_a_duration_that_is_not_one(seconds):
    with pytest.raises(ValueError):
        term("attention", seconds)


def test_a_refused_term_cannot_be_priced_at_zero():
    """A free stand-in drops the step out of the schedule while looking priced."""
    stood_in = ANALYTIC.after(Refusal("price_list", "unpriced leaf"))
    with pytest.raises(ValueError, match="priced at"):
        CostTerm("attention", 0.0, stood_in)


def test_a_zero_term_is_fine_when_nothing_refused():
    """No collective on this step is a fact about the step, not a missing model."""
    assert StepCost([term("mlp", 0.006), term("collective", 0.0)]).seconds == 0.006


def test_a_step_reports_its_mixture_and_its_refused_share():
    stood_in = ANALYTIC.after(Refusal("price_list", "unpriced leaf"))
    step = StepCost([term("mlp", 0.006), CostTerm("attention", 0.004, stood_in)])
    assert step.is_refused
    assert step.refused_seconds == pytest.approx(0.004)
    assert step.seconds_by_species() == {
        Species.MEASURED: pytest.approx(0.006),
        Species.ANALYTICAL: pytest.approx(0.004),
    }
    assert step.refusals == (Refusal("price_list", "unpriced leaf"),)


def test_refusals_are_counted_three_ways_over_a_run():
    """Few steps can be most of the time, so both fractions are reported."""
    stood_in = ANALYTIC.after(Refusal("price_list", "unpriced leaf"))
    clean = StepCost([term("step", 0.001)])
    refused = StepCost([CostTerm("step", 0.096, stood_in)])
    mix = ProvenanceMix()
    for step in (clean, clean, clean, refused):
        mix.record(step)

    assert mix.steps == 4
    assert mix.refused_steps == 1
    assert mix.refused_step_fraction == pytest.approx(0.25)
    assert mix.seconds == pytest.approx(0.099)
    assert mix.refused_seconds == pytest.approx(0.096)
    assert mix.refused_second_fraction == pytest.approx(0.096 / 0.099)
    assert mix.reasons() == {("price_list", "unpriced leaf"): 1}


def test_an_empty_run_reports_zero_rather_than_dividing_by_zero():
    mix = ProvenanceMix()
    assert mix.refused_step_fraction == 0.0
    assert mix.refused_second_fraction == 0.0
