# SPDX-License-Identifier: MIT
"""qhull refuses some sets of step features in the order they come in and
hulls them in another. The step linear fit's hull of such a set is the hull of
its distinct points in sorted order, which holds every point. The points are
held-out prefill steps of the crs005 prefill law's records, as `cached_seqs`
prices them.
"""

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from atom.compass.backends.hull import Hull
from atom.compass.backends.step_linear_fit import _hull

NAMES = (
    "tokens",
    "query_square",
    "query_cached",
    "cached_seqs",
    "moe_rows",
    "serial_moe_rows",
)
POINTS = np.array(
    [
        (16384, 268435456, 536870912, 1, 16391, 16391),
        (2392, 5721664, 352714752, 1, 2399, 2399),
        (16384, 268435456, 805306368, 1, 130959, 0),
        (14848, 212992000, 2648571904, 2, 129175, 0),
        (14336, 205520896, 2132279296, 1, 112641, 112641),
        (14080, 191168512, 2116419584, 2, 111164, 0),
        (15872, 251920384, 2531393536, 1, 48645, 48645),
        (14848, 220463104, 269877248, 1, 27673, 27673),
        (16254, 156862980, 1580129280, 1, 130690, 0),
        (14848, 220463104, 1972764672, 1, 129260, 0),
        (16384, 268435456, 1992294400, 1, 131072, 0),
        (14592, 212926464, 2756837376, 1, 129280, 0),
        (15844, 136004368, 2360053760, 2, 128961, 0),
        (15104, 220528640, 3246325760, 2, 129219, 0),
        (15616, 243859456, 1299251200, 1, 106979, 0),
        (16218, 149380004, 1288323584, 2, 130906, 0),
        (2392, 5721664, 352714752, 1, 2399, 2399),
        (9218, 84971524, 2114387968, 1, 9225, 9225),
        (197, 38809, 47153920, 1, 204, 204),
        (13, 169, 3168256, 1, 20, 20),
        (175, 30625, 18278400, 1, 182, 182),
        (16384, 268435456, 1073741824, 1, 32774, 32774),
        (16384, 268435456, 0, 0, 16391, 16391),
        (16384, 268435456, 268435456, 1, 16391, 16391),
        (16384, 268435456, 2147483648, 1, 16391, 16391),
    ],
    dtype=np.float64,
)


def test_a_set_qhull_refuses_in_its_order_is_hulled_whole():
    lo, hi = POINTS.min(axis=0), POINTS.max(axis=0)
    with pytest.raises(QhullError):
        ConvexHull((POINTS - lo) / (hi - lo))

    hull = Hull("the fit's hull", NAMES, _hull(POINTS, NAMES))

    assert all(hull.contains(tuple(p)) for p in POINTS)
