"""Per-cell prefill regions: a lookup, not a domain.

`prepare_prefill` is one scalar per source, and the measured cells disagree by
an order of magnitude -- 0.39 ms over one sequence of 640 tokens, 3.9 ms over
two full prompts, 1.19 ms over fifteen and sixteen. A scalar cannot represent
that and a widened range would claim every token count in between on the
strength of a term measured at neither end.
"""

from __future__ import annotations

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    Measured,
    REGION_MODELS,
    SOURCE_27B_TP1_CONC_V2,
    region_model,
)

CELLS = region_model("source-27b-tp1-prefill-cells")


def prefill(seqs, tokens, *, final=True, tp=1):
    per = tokens // seqs
    sched = tuple([per] * (seqs - 1) + [tokens - per * (seqs - 1)])
    return StepShape(num_scheduled_tokens=sched, context_lens=sched,
                     num_prefill_tokens=tokens, topology={"tp": tp},
                     capture_bucket=None, produces_output=final)


@pytest.mark.parametrize("seqs,tokens,prepare", [
    (1, 640, 3.916815e-4),
    (1, 15232, 7.823394e-4),
    (2, 15360, 3.860857e-3),
    (2, 16384, 3.931546e-3),
])
def test_each_measured_cell_is_priced_from_its_own_measurement(
        seqs, tokens, prepare):
    shape = prefill(seqs, tokens)
    assert CELLS.refusal(shape) is None, CELLS.refusal(shape)
    assert CELLS.breakdown(shape)["<prepare>"] == pytest.approx(prepare)


def test_the_cells_are_not_one_number_in_disguise(tmp_path=None):
    """The whole point: one sequence and two differ by an order of magnitude."""
    one = CELLS.breakdown(prefill(1, 640))["<prepare>"]
    two = CELLS.breakdown(prefill(2, 16384))["<prepare>"]
    assert two / one > 9, (one, two)


@pytest.mark.parametrize("seqs,tokens", [
    (1, 8192),      # between 640 and 15232, measured at neither
    (2, 12288),     # between the two 2-sequence cells
    (3, 16384),     # a sequence count nothing measured
    (1, 8192),
])
def test_an_unmeasured_cell_is_refused_and_not_bracketed(seqs, tokens):
    why = CELLS.refusal(prefill(seqs, tokens))
    assert why is not None, f"priced an unmeasured cell ({seqs}, {tokens})"
    assert "was not measured" in why


def test_a_wider_topology_is_still_refused():
    """Nothing here is fitted to a TP2 or TP4 engine."""
    assert CELLS.refusal(prefill(1, 640, tp=2)) is not None


def test_decode_is_carried_through_unchanged(tmp_path=None):
    """The steps after the first are priced by the evidence they always were."""
    dec = StepShape(num_scheduled_tokens=(1,) * 32, context_lens=(1151,) * 32,
                    num_prefill_tokens=0, topology={"tp": 1}, capture_bucket=32)
    assert CELLS.refusal(dec) is None
    assert (CELLS.breakdown(dec)
            == SOURCE_27B_TP1_CONC_V2.breakdown(dec))


def test_a_source_without_cells_behaves_exactly_as_before():
    """The field is additive: every existing source keeps its scalar domain."""
    old = SOURCE_27B_TP1_CONC_V2
    assert old.prepare_prefill_cells == ()
    shape = prefill(16, 16384)
    assert old.refusal(shape) is None
    assert (old.breakdown(shape)["<prepare>"]
            == pytest.approx(old.prepare_prefill.seconds))


def test_the_registry_offers_it():
    assert "source-27b-tp1-prefill-cells" in REGION_MODELS


def test_a_middle_chunk_pays_no_postprocess():
    """It samples no token, so the runner skips postprocess entirely.

    Zero work declared, not a small measurement and not a gap.
    """
    middle = StepShape(num_scheduled_tokens=(16384,), context_lens=(32128,),
                       num_prefill_tokens=16384, topology={"tp": 1},
                       capture_bucket=None, produces_output=False)
    assert CELLS.refusal(middle) is None, CELLS.refusal(middle)
    parts = CELLS.breakdown(middle)
    assert parts["<postprocess>"] == 0.0
    assert parts["<prepare>"] == pytest.approx(1.086426e-3)


def test_a_pooled_cell_spans_the_cohorts_it_pooled():
    """The range must cover both, so pooling is visible rather than a choice."""
    ragged = CELLS._prefill_cells(CELLS.prepare_prefill_cells)[(2, 16384, True)]
    assert ragged.samples == 6
    assert ragged.low <= 3.8410e-3 and ragged.high >= 4.2090e-3
    middle = CELLS._prefill_cells(CELLS.prepare_prefill_cells)[(1, 16384, False)]
    assert middle.samples == 6
    assert middle.low <= 7.3780e-4 and middle.high >= 1.2807e-3


class TestPartialCellTables:
    """A table can be missing a postprocess cell, and it matters which kind."""

    def _with(self, prepare_cells, post_cells):
        from dataclasses import replace
        return replace(CELLS, prepare_prefill_cells=prepare_cells,
                       postprocess_prefill_cells=post_cells)

    def test_a_sampling_cell_without_a_postprocess_measurement_refuses(self):
        """Missing is a gap, not zero. Charging zero drops a region it pays."""
        m = Measured(seconds=1e-3, low=1e-3, high=1e-3, samples=3, how="x")
        lib = self._with((((1, 777, True), m),), ())
        shape = prefill(1, 777, final=True)
        why = lib.refusal(shape)
        assert why is not None, "priced a sampling step with no postprocess"
        assert "missing rather than zero" in why, why

    def test_an_explicitly_output_less_cell_may_zero_its_postprocess(self):
        m = Measured(seconds=1e-3, low=1e-3, high=1e-3, samples=3, how="x")
        lib = self._with((((1, 777, False), m),), ())
        shape = prefill(1, 777, final=False)
        assert lib.refusal(shape) is None
        assert lib.breakdown(shape)["<postprocess>"] == 0.0

    def test_the_two_tables_may_be_populated_consistently(self):
        m = Measured(seconds=1e-3, low=1e-3, high=1e-3, samples=3, how="x")
        p = Measured(seconds=2e-4, low=2e-4, high=2e-4, samples=3, how="y")
        lib = self._with((((1, 777, True), m),), (((1, 777, True), p),))
        shape = prefill(1, 777, final=True)
        assert lib.refusal(shape) is None
        assert lib.breakdown(shape)["<postprocess>"] == pytest.approx(2e-4)


def test_a_middle_chunk_pays_no_tp_broadcast():
    """The broadcast carries the sampled token, and there is not one.

    Charging it would bill a collective the step never ran.
    """
    from dataclasses import replace
    at_tp2 = replace(CELLS, topologies=(1, 2))
    middle = prefill(1, 16384, final=False, tp=2)
    final_ = prefill(2, 16384, final=True, tp=2)
    assert "<tp-broadcast>" not in at_tp2.breakdown(middle)
    assert "<tp-broadcast>" in at_tp2.breakdown(final_)


def test_the_pooled_value_is_labelled_an_approximation():
    """Not a bound: it generalises over the cohorts it pooled."""
    cell = CELLS._prefill_cells(CELLS.prepare_prefill_cells)[(1, 16384, False)]
    assert "POOLED" in cell.how
    assert "32128" in cell.how and "48512" in cell.how
