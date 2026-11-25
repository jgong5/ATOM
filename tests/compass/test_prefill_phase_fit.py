# SPDX-License-Identifier: MIT
"""`COMPASS_FIT_PREFILL_PHASE_MAX`: the target prefill vector fitted under the
runner's phase-wise group rule (`runner.overrides.history_phase_wait`), a
step's cost the max over its ranks of each rank's price plus `k` x (the
group's largest history-term seconds less the rank's own). Off, the fit is the
max-affine one it was."""

import numpy as np
import pytest

from atom.compass.backends.step_linear_fit import (
    PHASE_COLUMNS,
    PREFILL_PHASE_MAX_ENV,
    _fit_form,
    _phase,
    env_options,
)

NAMES = ["step", "tokens", "history"]
TRUE = np.array([0.010, 2.0e-5, 4.0e-9])


def _phased_steps(n=80, k=1.0, seed=0):
    """`n` two-rank steps whose labels follow the phase-wise rule at blend `k`:
    one rank holds the longer history, the other the more new tokens, so the
    max over ranks misses the history wait."""
    rng = np.random.default_rng(seed)
    x = np.ones((n, 2, 3))
    tokens = rng.uniform(1e3, 3e4, (n, 2))
    history = rng.uniform(1e5, 4e6, (n, 2))
    # rank 0 the longer history, rank 1 the more tokens, on most steps
    flip = rng.uniform(size=n) < 0.8
    tokens[flip] = np.sort(tokens[flip], axis=1)
    history[flip] = np.sort(history[flip], axis=1)[:, ::-1]
    x[..., 1], x[..., 2] = tokens, history
    own = x @ TRUE
    a = x[..., 2] * TRUE[2]
    y = (own + k * (a.max(axis=1, keepdims=True) - a)).max(axis=1)
    return x, np.ones((n, 2), dtype=bool), y


def test_the_switch_reads_a_positive_blend_and_off_states_nothing():
    assert env_options({}) == {}
    assert env_options({PREFILL_PHASE_MAX_ENV: "0"}) == {}
    assert env_options({PREFILL_PHASE_MAX_ENV: ""}) == {}
    assert env_options({PREFILL_PHASE_MAX_ENV: "1"}) == {"phase_max": 1.0}
    assert env_options({PREFILL_PHASE_MAX_ENV: "0.44"}) == {"phase_max": 0.44}
    for bad in ("-1", "x", "inf", "nan"):
        with pytest.raises(ValueError, match=PREFILL_PHASE_MAX_ENV):
            env_options({PREFILL_PHASE_MAX_ENV: bad})


def test_phase_masks_the_history_columns_and_is_none_when_off_or_absent():
    assert _phase(None, NAMES) is None
    assert _phase(0, NAMES) is None
    assert _phase(1.0, ["step", "tokens"]) is None
    blend, mask = _phase(1.0, NAMES + ["query_cached", "query_square", "out"])
    assert blend == 1.0
    assert mask == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert set(PHASE_COLUMNS) == {"query_cached", "history", "query_square"}


def test_the_phase_fit_recovers_the_coefficients_the_max_rule_misses():
    x, real, y = _phased_steps()
    _, mask = _phase(1.0, NAMES)

    phased = _fit_form(x, real, y, NAMES, phase=(1.0, mask))
    plain = _fit_form(x, real, y, NAMES)

    assert phased["coef"] == pytest.approx(TRUE, rel=1e-6)
    assert np.abs(phased["residual"]).max() < 1e-9
    # the max rule cannot price the history wait: it misses by more
    assert np.abs(plain["residual"]).max() > 1e-3
    # the priced rank is the one the phase-wise price picks
    own = x @ phased["coef"]
    a = x[..., 2] * phased["coef"][2]
    assert (phased["pick"] == (own + a.max(axis=1, keepdims=True) - a).argmax(axis=1)).all()


def test_without_phase_the_fit_is_the_max_affine_one():
    x, real, y = _phased_steps(seed=1)

    a = _fit_form(x, real, y, NAMES)
    b = _fit_form(x, real, y, NAMES, phase=None)

    assert np.array_equal(a["coef"], b["coef"])
    assert np.array_equal(a["keep"], b["keep"])
    assert np.array_equal(np.asarray(a["pick"]), np.asarray(b["pick"]))
    assert a["passes"] == b["passes"]


def test_a_partial_blend_is_fitted_at_its_own_rule():
    x, real, y = _phased_steps(k=0.5, seed=2)
    _, mask = _phase(0.5, NAMES)

    phased = _fit_form(x, real, y, NAMES, phase=(0.5, mask))

    assert phased["coef"] == pytest.approx(TRUE, rel=1e-6)
