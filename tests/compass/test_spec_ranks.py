# SPDX-License-Identifier: MIT
"""Comparing the ranks' readings of one machine.

The rank numbers are the measured ones, because these checks are judgements
about size: the cross-rank spread accepted is the one the hardware really
shows, and the spread refused is the one a neighbour really produced. Round
numbers would make both tests vacuous.
"""

import pytest

from atom.compass.spec import SpecRefusal, across_ranks
from atom.compass.spec.rules import Rule

MIB = 1024**2


@pytest.mark.parametrize(
    "tp_width,reading,spread",
    [
        (1, 926 * MIB, 0),
        (2, 6906 * MIB, 0),
        (4, 7266 * MIB, 192 * MIB),
        (8, 10704 * MIB, 640 * MIB),
    ],
)
def test_the_spread_this_hardware_really_shows_is_not_an_error(
    tp_width, reading, spread
):
    across_ranks("non_torch", [reading] * (tp_width - 1) + [reading + spread])


def test_a_neighbour_holding_fifty_times_the_rank_is_refused():
    # The case this exists for: a rank had reserved 2.94 GB and the card read
    # 152.01 GB, and six runs died on a negative memory budget before anything
    # compared the ranks.
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [2.94e9, 2.94e9, 2.94e9, 152.01e9])
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "2940000000.0" in refused.value.what
    assert "152010000000.0" in refused.value.what


def test_a_reading_no_spec_could_hold_is_refused_where_the_ranks_are():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [0, 970.0e6])
    assert refused.value.rule is Rule.SHAPE
