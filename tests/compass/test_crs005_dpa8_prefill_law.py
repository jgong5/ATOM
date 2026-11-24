# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 prefill law for crsuse2-m2m-v3-005 prices the first
and the later 16384-token chunks one rank of #975's c=1 cell ran beside seven
dummy forwards, a run it was not fitted on, inside its prefill vector's hull and
held-out hull at its data-parallel width. The later chunk adds the vector's
`cached_seqs` cost and its `query_cached` slope over the history, and its
`moe_rows` term reads the group's MoE rows.
"""

import pytest
from test_crs005_dpa8_laws import backend

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.step_linear import SOURCE

#: The group's MoE rows of a chunk one rank prefills beside seven dummy forwards.
MOE_ROWS = 16391
#: The median history of the later chunks in that run.
HISTORY = 49152


def chunk(history: int) -> BatchView:
    """A 16384-token middle chunk on `history` cached tokens, not split in two."""
    rows = (RequestShape(16384, history + 16384, False),)
    return BatchView(rows, moe_rows=MOE_ROWS, produces_output=False)


def test_the_crs005_prefill_law_prices_a_later_chunk_by_its_cached_seqs_term():
    law = backend("crs005", "prefill")
    first, later = law.price(chunk(0)), law.price(chunk(HISTORY))
    c = dict(zip(law.law.prefill.features, law.law.prefill.coefficients))

    for cost in (first, later):
        assert {t.provenance.source for t in cost.terms} == {SOURCE}
        assert all(t.provenance.validated for t in cost.terms)
    terms = {t.name: t.seconds for t in later.terms}
    assert terms["prefill.cached_seqs"] == c["cached_seqs"] > 0
    assert terms["prefill.moe_rows"] == c["moe_rows"] * MOE_ROWS
    assert later.seconds - first.seconds == pytest.approx(
        c["cached_seqs"] + c["query_cached"] * 16384 * HISTORY
    )
