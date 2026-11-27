# SPDX-License-Identifier: MIT
"""COMPASS_FIT_HOST_MEDIAN=kind.segment,... (#3720): a host segment's law scaled so the median label over price is 1."""

import numpy as np
import pytest

from atom.compass.backends.step_linear_fit import _median_centred


def test_a_tailed_segment_is_priced_at_its_median_not_its_mean():
    labels = np.array([0.001, 0.0012, 0.0015, 0.0018, 0.050])  # one rare long forward
    mean_law = [labels.mean(), 0.0, 0.0]  # what the solver prices a tailed class at
    centre = float(np.median(1.0 - labels / mean_law[0]))
    coef, fitted = _median_centred(mean_law, {"centre": centre, "mad": 0.1})
    assert coef[0] == pytest.approx(np.median(labels))
    assert fitted["centre"] == 0.0 and fitted["mad"] == 0.1
    assert fitted["median_centred"]["solver_centre"] == centre
    # A launch column's coefficient scales with the law.
    assert _median_centred([0.01, 0.0, 0.0, 0.004], {"centre": 0.5})[0] == pytest.approx([0.005, 0, 0, 0.002])
    with pytest.raises(ValueError, match="no positive scale"):
        _median_centred([0.01, 0.0, 0.0], {"centre": 1.0})
