# SPDX-License-Identifier: MIT
"""A prefill forward pays a cost per row that has cached context, whatever its
history's length: the step linear law's `cached_seqs` column.

Records are written from a known law that charges it; the `cached_seqs` form
finds it where `base` cannot, and the backend prices it on each row with
cached context and on no other.
"""

import numpy as np
import pytest
from test_backend_step_linear_fit import PREFILL, _fit, _price, _record, _steps, _view

from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.backends.step_linear_fit import held_out, label_records

#: Seconds a prefill row with cached context costs past the base form's price.
CACHED = 0.008


def _cached_steps(seed, n=240):
    """`_steps`' steps, each prefill row on cached context or not at random,
    a prefill priced with `CACHED` per row on cached context."""
    rng = np.random.default_rng(seed)
    out = []
    for rows, rung, seconds in _steps(seed, n):
        if not rows[0][2]:
            rows = [(q, c if rng.random() < 0.5 else q, False) for q, c, _ in rows]
            seconds = _price(rows, None) + CACHED * sum(c > q for q, c, _ in rows)
        out.append((rows, rung, seconds))
    return out


def test_the_cached_seqs_form_finds_a_cost_per_row_on_cached_context(tmp_path):
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _cached_steps(s)) for s in (3, 4, 5)
    ]
    samples = label_records(records)[0]
    errors = {}
    for form in ("base", "cached_seqs"):
        bands = held_out(samples, {"form": form})
        errors[form] = np.abs(
            np.concatenate([e for b, e in bands.items() if "prefill" in b])
        )
    assert errors["cached_seqs"].max() < 1e-6 < np.median(errors["base"])

    law = _fit(records, "cached_seqs")
    assert law["prefill"]["coefficients"] == pytest.approx(
        {**PREFILL, "cached_seqs": CACHED}, rel=1e-6
    )
    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, _, seconds in _cached_steps(6):
        if rows[0][2]:
            continue
        cost = backend.price(_view(rows))
        assert cost.seconds == pytest.approx(seconds, rel=1e-6)
        terms = {t.name: t.seconds for t in cost.terms}
        cached = sum(c > q for q, c, _ in rows)
        assert terms["prefill.cached_seqs"] == pytest.approx(CACHED * cached)
