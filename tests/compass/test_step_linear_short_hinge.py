# SPDX-License-Identifier: MIT
"""The step linear law's opt-in prefill column `short_hinge` (#3720):
max(0, K - Sum N_Q), the tokens a chunk falls short of the knee K the vector
states as `short_knee` (`COMPASS_FIT_SHORT_HINGE=K`).

A MoE prefill's grouped GEMMs read every routed expert's weights until each
expert's rows fill a tile, so a short chunk pays a floor that one `tokens`
slope misses: MiniMax-M3.1's P stage 0 (128 experts, top-4) ran 128-1024-token
chunks at a 21-23 ms median that its law priced 17-26% low. The knee is set
from the roofline (64-row tile x 128 / 4 = 2048 tokens), not fitted. Off, a fit
writes the law it wrote before the column. Records are written from a known
law. No device.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _engine_file, _record

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import (
    SHORT_HINGE_ENV,
    env_options,
    fit,
    label_records,
)

KNEE = 2048
TRUTH = {
    "step": 0.012,
    "tokens": 6e-6,
    "query_square": 2.5e-11,
    "query_cached": 1.4e-11,
    "short_hinge": 4.5e-6,
}


def _view(rows):
    return BatchView(tuple(RequestShape(q, n, False) for q, n in rows))


def _anywhere(features: int) -> dict:
    """A hull every point is inside: one facet, `0 . x - 1 <= 0`."""
    return {"equal": {}, "facets": [[0.0] * features + [-1.0]]}


def _law(prefill=TRUTH, knee=KNEE) -> dict:
    vector = {"coefficients": prefill, "hull": _anywhere(len(prefill) - 1)}
    if knee is not None:
        vector["short_knee"] = knee
    decode = {"step": 0.01, "context": 1e-8, "graph_padding": 1e-9}
    return {
        "provenance": "the test's",
        "prefill": vector,
        "decode": {"1": {"coefficients": decode, "hull": _anywhere(2)}},
    }


def _price(rows) -> float:
    q = sum(r[0] for r in rows)
    features = {
        "step": 1,
        "tokens": q,
        "query_square": sum(a * a for a, _ in rows),
        "query_cached": sum(a * (n - a) for a, n in rows),
        "short_hinge": max(0, KNEE - q),
    }
    return float(sum(c * features[f] for f, c in TRUTH.items()))


def test_the_backend_prices_the_tokens_a_chunk_falls_short_of_the_knee():
    backend = StepLinearBackend(StepLinearLaw(_law()))
    for rows, short in (([(300, 300)], 1748), ([(500, 40_500), (700, 700)], 848),
                        ([(2048, 2048)], 0), ([(8192, 100_000)], 0)):
        terms = {t.name: t.seconds for t in backend.price(_view(rows)).terms}
        assert terms["prefill.short_hinge"] == pytest.approx(4.5e-6 * short)
        assert backend.price(_view(rows)).seconds == pytest.approx(_price(rows))


def test_a_law_carrying_the_column_states_its_knee():
    for knee in (None, 0, 2048.0, "2048"):
        with pytest.raises(ValueError, match="carries short_hinge and states its short_knee"):
            StepLinearLaw(_law(knee=knee))
    # A law without the column ignores a stray knee as before.
    plain = {k: v for k, v in TRUTH.items() if k != "short_hinge"}
    StepLinearLaw(_law(plain, knee=None))


def _records(path, n=300, seed=0):
    """`n` one-request prefill steps, a third under the knee, each on a history
    of 0 to 200k tokens, labelled with `TRUTH`'s price."""
    rng = np.random.default_rng(seed)
    steps = []
    for i in range(n):
        q = int(rng.integers(64, KNEE) if i % 3 == 0 else rng.integers(KNEE, 16_384))
        n_kv = q + int(rng.integers(0, 200_000)) * int(rng.integers(0, 2))
        steps.append(([(q, n_kv, False)], None, _price([(q, n_kv)])))
    path.mkdir()
    record = _record(path / "dp0.jsonl", steps)
    _engine_file(record, {"data_parallel_size": 1})
    return [record]


def _fit(paths, **options):
    samples = label_records(paths, 1)[0]
    return fit(samples, {"form": "base", "dp": 1, **options})[0]


def test_the_column_is_fitted_when_asked_and_off_by_default(tmp_path):
    paths = _records(tmp_path / "run")
    law = _fit(paths, short_hinge=KNEE)
    vector = law["prefill"]
    assert list(vector["coefficients"]) == list(TRUTH)
    assert vector["coefficients"] == pytest.approx(TRUTH, rel=1e-5)
    assert vector["short_knee"] == KNEE
    assert set(vector["fit"]["vif"]) == {"short_hinge"}
    backend = StepLinearBackend(StepLinearLaw(law))
    for rows in ([(300, 300)], [(1500, 101_500)], [(9000, 9000)]):
        assert backend.price(_view(rows)).seconds == pytest.approx(_price(rows), rel=1e-5)
    # Off, the law is the one written before the column: byte for byte.
    off = _fit(paths)
    assert "short_hinge" not in off["prefill"]["coefficients"]
    assert "short_knee" not in off["prefill"] and "vif" not in off["prefill"]["fit"]
    assert json.dumps(_fit(paths, short_hinge=None)) == json.dumps(off)


def test_a_draft_law_takes_no_column(tmp_path):
    samples = label_records(_records(tmp_path / "run", n=60), 1)[0]
    draft = fit(samples, {"form": "base", "dp": 1, "short_hinge": KNEE}, draft=True)[0]
    assert "short_hinge" not in draft["prefill"]["coefficients"]


def test_a_knee_that_is_not_a_positive_token_count_is_refused(tmp_path):
    samples = label_records(_records(tmp_path / "run", n=40), 1)[0]
    for knee in (0, -5, 2048.5, True):
        with pytest.raises(ValueError, match="short_hinge's knee is"):
            fit(samples, {"form": "base", "dp": 1, "short_hinge": knee})


def test_the_switch_reads_from_the_environment():
    assert SHORT_HINGE_ENV == "COMPASS_FIT_SHORT_HINGE"
    assert env_options({SHORT_HINGE_ENV: "2048"}) == {"short_hinge": 2048}
    assert env_options({SHORT_HINGE_ENV: "0"}) == env_options({SHORT_HINGE_ENV: ""}) == {}
    for word in ("2k", "-1", "1.5"):
        with pytest.raises(ValueError, match=f"{SHORT_HINGE_ENV}={word} is not a whole"):
            env_options({SHORT_HINGE_ENV: word})
