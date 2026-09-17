"""Native output predicates survive step-table repricing."""

import pytest

from atom.compass.core.cost.regions import SOURCE_27B_TP1_PREFILL_SEQS
from scripts.compass.price_steps import shape_of


def test_repricing_middle_chunk_skips_postprocess_and_tp_broadcast():
    shape = shape_of({
        "num_scheduled_tokens": [16384],
        "context_lens": [16384],
        "num_prefill_tokens": 16384,
        "topology": {"tp": 2},
        "compiled": True,
        "produces_output": False,
    })
    parts = SOURCE_27B_TP1_PREFILL_SEQS.breakdown(shape)
    assert parts["<postprocess>"] == 0.0
    assert "<tp-broadcast>" not in parts
    assert parts["<prepare>"] > 0.0


@pytest.mark.parametrize("recorded", [True, False])
def test_shape_preserves_the_native_output_predicate(recorded):
    assert shape_of({"produces_output": recorded}).produces_output is recorded


def test_legacy_rows_keep_the_previous_output_default():
    assert shape_of({}).produces_output is True
