# SPDX-License-Identifier: MIT
"""A step feature that is an affine function of another over a fit, as c=1
records give: each one-request decode's first EAGLE3 draft pass runs 4 tokens
and 16 query squares, each later one 1 and 1, and a one-request prefill's
`seqs` equals its `out`. The step linear fit holds such a feature to the other
and hulls the rest, where qhull refuses the flat set.
"""

import numpy as np
import pytest
from test_backend_step_linear_fit import _price
from test_eagle3_m3 import K, _eagle_record

from atom.compass.backends.hull import Hull
from atom.compass.backends.step_linear_fit import (
    _hull,
    fit,
    label_records,
    mark_held_out,
)

NAMES = ("tokens", "query_square", "query_cached")


@pytest.mark.parametrize(
    "square", [lambda t: 5 * t - 4, lambda t: 20 - 4 * t], ids=["rising", "falling"]
)
def test_a_feature_affine_in_another_is_held_to_it_and_the_rest_hulled(square):
    points = np.array(
        [(t, square(t), t * c) for c in range(200, 400) for t in (4, 1)], dtype=float
    )

    hull = Hull("the forward hull", NAMES, _hull(points, NAMES))

    assert all(hull.contains(tuple(p)) for p in points)
    # Between the two token counts, on the line and off it.
    assert hull.contains((2, square(2), 600))
    assert not hull.contains((2, square(2) + 1, 600))


def _c1_records(tmp_path):
    """Two EAGLE3 records of one-request steps, a prefill chunk then a decode
    at rung 1; the first's every fourth decode is two requests at rung 2."""
    records = []
    for seed in (1, 2):
        rng, steps = np.random.default_rng(seed), []
        for i in range(120):
            if i % 2 == 0:
                q = int(rng.integers(16, 8192))
                rows = [(q, q + int(rng.integers(0, 60000)), False)]
                steps.append((rows, None, _price(rows, None)))
                continue
            width = 2 if seed == 1 and i % 8 == 1 else 1
            rows = [(K + 1, int(rng.integers(K + 1, 60000)), True)] * width
            steps.append((rows, width, 0.02 + 1e-7 * sum(r[1] for r in rows)))
        records.append(_eagle_record(tmp_path / f"dp0-{seed}.jsonl", steps))
    return label_records(records)[0]


def test_a_c1_law_fits_its_prefill_vector_whose_seqs_is_its_out(tmp_path):
    law = fit(_c1_records(tmp_path))[0]

    assert "prefill" not in law["unfitted"]
    assert {"seqs", "out"} <= set(law["prefill"]["coefficients"])


def test_a_record_whose_decodes_are_all_one_request_is_held_out(tmp_path):
    samples = _c1_records(tmp_path)
    law, report = fit(samples)

    mark_held_out(law, report, samples)

    bands = law["draft"]["forward"]["held_out"]["bands"]
    assert set(bands) == {"pass 0", "pass 1", "pass 2"}
    assert len(report["held_out"]) == 2
    for run in report["held_out"].values():
        assert {"draft pass 0", "draft pass 2"} <= set(run)
