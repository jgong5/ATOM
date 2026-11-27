# SPDX-License-Identifier: MIT
"""The opt-in prefill hinge on `seqs` (#734 Cell B): with a vector's
`seqs_hinge` H, `seqs` counts a rank's prompts up to H and `seqs_over` those
past it, each its own coefficient, in the backend as in the fit
(`step_linear_fit._prefill_counts`); a vector carrying `seqs_over` states H,
1 or more, and `seqs` beside it; COMPASS_FIT_PREFILL_SEQS_HINGE=H switches it
on, off by default, and a draft law takes none. No device."""

import pytest

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import PREFILL_EXTRA, hull_axes
from atom.compass.backends.step_linear_fit import (
    _columns,
    _prefill_counts,
    env_options,
)

SEQS, OVER, OUT = 3.2e-3, 1.1e-3, 5e-3


def _prefill(n, tokens=300, output=True):
    rows = tuple(RequestShape(tokens, tokens, False) for _ in range(n))
    return BatchView(rows, produces_output=output)


def _law(hinge=12, over=True, **spec) -> dict:
    coefficients = {
        "step": 0.02,
        "tokens": 1.1e-5,
        "query_square": 0.0,
        "query_cached": 0.0,
        "seqs": SEQS,
        **({"seqs_over": OVER} if over else {}),
        "out": OUT,
    }
    free = len(coefficients) - 1
    hull = {"equal": {}, "facets": [[0.0] * free + [-1.0]]}
    vector = {"coefficients": coefficients, "hull": hull, **spec}
    if hinge is not None:
        vector["seqs_hinge"] = hinge
    return {"provenance": "the test's", "prefill": vector, "decode": {}}


def test_the_column_follows_seqs_and_is_a_hull_axis():
    assert PREFILL_EXTRA.index("seqs_over") == PREFILL_EXTRA.index("seqs") + 1
    assert hull_axes(("step", "seqs", "seqs_over", "out")) == (1, 2, 3)


@pytest.mark.parametrize("n", [1, 12, 13, 48])
def test_the_backend_counts_seqs_up_to_the_hinge_and_seqs_over_past_it(n):
    backend = StepLinearBackend(StepLinearLaw(_law()))
    terms = {t.name: t.seconds for t in backend.price(_prefill(n)).terms}
    assert terms["prefill.seqs"] == pytest.approx(SEQS * min(n, 12))
    assert terms["prefill.seqs_over"] == pytest.approx(OVER * max(n - 12, 0))
    # A step producing no output counts neither, as `seqs` alone counts none.
    quiet = backend.price(_prefill(n, output=False)).terms
    assert {t.name: t.seconds for t in quiet}["prefill.seqs_over"] == 0.0


def test_without_the_hinge_seqs_counts_every_prompt():
    backend = StepLinearBackend(StepLinearLaw(_law(hinge=None, over=False)))
    terms = {t.name: t.seconds for t in backend.price(_prefill(48)).terms}
    assert terms["prefill.seqs"] == pytest.approx(SEQS * 48)
    assert "prefill.seqs_over" not in terms


def test_a_vector_with_seqs_over_states_its_hinge_and_seqs():
    with pytest.raises(ValueError, match="seqs_over over a seqs_hinge of None"):
        StepLinearLaw(_law(hinge=None))
    with pytest.raises(ValueError, match="seqs_over over a seqs_hinge of 0"):
        StepLinearLaw(_law(hinge=0))
    law = _law()
    del law["prefill"]["coefficients"]["seqs"]
    law["prefill"]["hull"]["facets"] = [[0.0] * 5 + [-1.0]]
    with pytest.raises(ValueError, match="seqs_over and no seqs"):
        StepLinearLaw(law)


def test_the_fit_counts_them_as_the_backend_does():
    cols = _columns({"form": "base", "seqs_hinge": 12})
    view = _prefill(48)
    args = ("here", "base", 1, False, True, ())
    assert _prefill_counts(view, *args, False, cols=cols)[-3:] == (12, 36, 1)
    plain = _columns({"form": "base"})
    assert _prefill_counts(view, *args, False, cols=plain)[-2:] == (48, 1)


def test_the_switch_sets_the_hinge_and_is_off_by_default():
    assert env_options({}) == {}
    assert env_options({"COMPASS_FIT_PREFILL_SEQS_HINGE": "0"}) == {}
    assert env_options({"COMPASS_FIT_PREFILL_SEQS_HINGE": "12"}) == {"seqs_hinge": 12}
    with pytest.raises(ValueError, match="is not a positive integer"):
        env_options({"COMPASS_FIT_PREFILL_SEQS_HINGE": "1.5"})
    assert _columns({"form": "base", "seqs_hinge": 12})["seqs_hinge"] == 12
    assert _columns({"form": "base", "seqs_hinge": 12}, draft=True)["seqs_hinge"] is None
