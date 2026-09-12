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
    REGION_MODELS,
    SOURCE_27B_TP1_CONC_V2,
    region_model,
)

CELLS = region_model("source-27b-tp1-prefill-cells")


def prefill(seqs, tokens, *, final=True, tp=1):
    per = tokens // seqs
    sched = tuple([per] * (seqs - 1) + [tokens - per * (seqs - 1)])
    ctx = sched if final else ()
    return StepShape(num_scheduled_tokens=sched, context_lens=ctx,
                     num_prefill_tokens=tokens, topology={"tp": tp},
                     capture_bucket=None)


@pytest.mark.parametrize("seqs,tokens,prepare", [
    (1, 640, 3.916815e-4),
    (1, 15232, 7.823394e-4),
    (2, 15360, 3.860857e-3),
    (2, 16384, 3.875352e-3),
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
    (1, 16384),     # measured only as a MIDDLE chunk, and only per context
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
