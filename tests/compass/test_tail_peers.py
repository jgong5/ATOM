# SPDX-License-Identifier: MIT
"""The opt-in prefill column `tail_peers` (#752 cell A v2): one function the
fit and the backend both call counts it, 0 off a tail step, on one rank and on
a dummy forward, else the group's real-row peers on a tail step; a vector
without the group's width is refused; COMPASS_FIT_TAIL_PEERS=1 switches it on,
off by default. No device."""

import pytest

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import (
    GATHER_COLUMNS,
    PREFILL_EXTRA,
    gather_column,
    hull_axes,
    tail_peers,
)
from atom.compass.backends.step_linear_fit import _columns, _gathered, env_options

DP = 4
BUCKETS = (256, 512, 1024, 2048, 4096)


def _prefill(tokens, moe=None, peak=None, **fields):
    rows = (RequestShape(tokens, tokens, False),)
    return BatchView(
        rows, moe_rows=moe, moe_tokens=moe, group_max_tokens=peak, **fields
    )


def _law(**spec) -> dict:
    coefficients = {
        "step": 0.02,
        "tokens": 6e-5,
        "query_square": 2e-10,
        "query_cached": 3e-10,
        "moe_rows": 2e-5,
        "uneven_rows": 3e-6,
        "tail_peers": 8e-3,
    }
    free = sum(1 for f in coefficients if f not in ("step", "tail_peers"))
    hull = {"equal": {}, "facets": [[0.0] * free + [-1.0]]}
    vector = {
        "coefficients": coefficients,
        "hull": hull,
        "ranks": DP,
        "min_spread": 0.2,
        **spec,
    }
    return {"provenance": "the test's", "prefill": vector, "decode": {}}


def test_tail_peers_count_the_real_peers_of_a_tail_step():
    tail = {"group_tail": True, "produces_output": False}
    assert (
        tail_peers(
            _prefill(32768, moe=4 * 32768, peak=32768, group_dummies=0, **tail), DP
        )
        == 3
    )
    assert (
        tail_peers(
            _prefill(32768, moe=2 * 32768, peak=32768, group_dummies=2, **tail), DP
        )
        == 1
    )
    assert (
        tail_peers(
            _prefill(
                500,
                moe=500,
                peak=500,
                group_dummies=3,
                group_tail=True,
                produces_output=True,
            ),
            DP,
        )
        == 0
    )
    # no tail at this step, one rank, a dummy forward
    assert (
        tail_peers(
            _prefill(
                32768, moe=4 * 32768, peak=32768, group_dummies=0, group_tail=False
            ),
            DP,
        )
        == 0
    )
    assert tail_peers(_prefill(32768, group_tail=False), DP) == 0
    dummy = _prefill(
        1,
        moe=3 * 32768 + 1,
        peak=32768,
        group_dummies=1,
        group_tail=True,
        dp_dummy=True,
    )
    assert tail_peers(dummy, DP) == 0
    view = _prefill(32768, moe=4 * 32768, peak=32768, group_dummies=0, **tail)
    assert gather_column("tail_peers", view, BUCKETS, DP) == tail_peers(view, DP)


def test_the_column_follows_uneven_rows_and_is_no_hull_axis():
    assert GATHER_COLUMNS[-1] == "tail_peers"
    assert PREFILL_EXTRA.index("tail_peers") == PREFILL_EXTRA.index("uneven_rows") + 1
    features = ("step", "tokens", "uneven_rows", "tail_peers")
    assert hull_axes(features) == (1, 2)


def test_the_backend_prices_tail_peers_by_the_function_the_fit_counts_it_with():
    backend = StepLinearBackend(StepLinearLaw(_law()), DP)
    for dummies, tail in ((0, True), (2, True), (0, False)):
        view = _prefill(
            32768,
            moe=(DP - dummies) * 32768,
            peak=32768,
            group_dummies=dummies,
            group_tail=tail,
            produces_output=False,
        )
        terms = {t.name: t.seconds for t in backend.price(view).terms}
        assert terms["prefill.tail_peers"] == pytest.approx(8e-3 * tail_peers(view, DP))
    with pytest.raises(ValueError, match="tail_peers over a data-parallel group"):
        StepLinearLaw(_law(ranks=None))


def test_the_switch_turns_the_column_on_and_is_off_by_default():
    assert env_options({}) == {}
    assert env_options({"COMPASS_FIT_TAIL_PEERS": "1"}) == {"tail_peers": True}
    with pytest.raises(ValueError, match="is not 0 or 1"):
        env_options({"COMPASS_FIT_TAIL_PEERS": "2"})
    on = _columns({"form": "history", "dp": DP, "tail_peers": True})
    assert "tail_peers" in _gathered(on)
    assert "tail_peers" not in _gathered(_columns({"form": "history", "dp": DP}))
    assert "tail_peers" not in _gathered(
        _columns({"form": "history", "dp": DP, "tail_peers": True}, draft=True)
    )
