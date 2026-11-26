# SPDX-License-Identifier: MIT
"""COMPASS_FIT_ADMIT=median (#3720): `admit` as the median of the admitting forwards' E past their kind's law."""

import pytest

from atom.compass.backends.step_linear_fit import _admit_median


def test_admit_is_the_median_extra_e_per_admitting_forward():
    admit, report = _admit_median([0.001] * 5, [0.002, 0.004, 0.005, 0.030, 0.200])
    assert admit == pytest.approx([0.004, 0.0, 0.0])
    assert report["form"] == "median" and report["samples"] == 5
    # E below its law's price counts as no extra, not as a negative one.
    assert _admit_median([0.003, 0.003, 0.003], [0.001, 0.002, 0.010])[0][0] == 0.0
    with pytest.raises(ValueError, match="no forward that admits"):
        _admit_median([], [])
