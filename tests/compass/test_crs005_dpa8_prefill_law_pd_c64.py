# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 prefill law for crsuse2-m2m-v3-005 prices a step a
P/D prefill side runs at c=64 (#975's cell) inside its prefill vector's hull and
held-out hull: one rank's 16384-token later chunk on 92416 cached tokens while
five peers prefill as many and two run dummy forwards, so the group's MoE
gather carries six full chunks and two dummy rows.
"""

from test_crs005_dpa8_laws import backend

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.step_linear import SOURCE

#: The group's MoE rows: six ranks' 16384-token chunks and two dummy rows.
MOE_ROWS = 6 * 16384 + 2


def test_the_crs005_prefill_law_prices_a_c64_pd_prefill_step_inside_its_hulls():
    chunk = (RequestShape(16384, 92416 + 16384, False),)
    view = BatchView(chunk, moe_rows=MOE_ROWS, produces_output=False)

    cost = backend("crs005", "prefill").price(view)

    assert {t.provenance.source for t in cost.terms} == {SOURCE}
    assert all(t.provenance.validated for t in cost.terms)
