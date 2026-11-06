# SPDX-License-Identifier: MIT
"""A draft law whose `near_propose` is bounded at a token count prices a longer
output prefill's `propose` pass at its far price whatever its lag."""

import pytest
from test_backend_step_linear_fit import _fit
from test_step_linear_draft import PROPOSE
from test_step_linear_near_propose import (
    LATE,
    WAIT,
    _output_prefills,
    _priced,
    with_near_propose,
)

from atom.compass.backends import CostRefused

#: The prefill tokens from which the law's `near_propose` no longer prices the pass.
BELOW = 12288


@pytest.fixture(scope="module")
def law(tmp_path_factory):
    record = _output_prefills(tmp_path_factory.mktemp("below") / "dp0.jsonl", 1600, 0)
    return with_near_propose(_fit([record]), below=BELOW)


def test_a_bounded_near_propose_prices_a_longer_prefill_at_its_far_price(law):
    assert _priced(law, -LATE, BELOW - 1) == pytest.approx(
        sum(PROPOSE) + WAIT, rel=1e-4
    )
    assert _priced(law, -LATE, BELOW) == pytest.approx(sum(PROPOSE), rel=1e-4)
    assert _priced(law, 0.0, BELOW - 1) == pytest.approx(sum(PROPOSE), rel=1e-4)
    # Its lag does not move its price, so a view stating none is priced.
    assert _priced(law, None, BELOW) == pytest.approx(sum(PROPOSE), rel=1e-4)
    with pytest.raises(CostRefused, match="how far behind its host"):
        _priced(law, None, BELOW - 1)
