# SPDX-License-Identifier: MIT
"""`cost.idle_group_dummy` prices a producer's idle-group dummy at the seconds COMPASS_IDLE_DP_GROUP_S states, 0 where unset
(unchanged): at 0 s its group stepped without the clock moving and a 1P1D simulation never advanced.
"""

from atom.compass.backends import cost
from atom.compass.backends.cost import IDLE_DP_GROUP_S, idle_group_dummy


class _View:
    prefill = ()
    dp_dummy = True
    group_dummies = 2


def test_unset_prices_zero_as_before(monkeypatch):
    monkeypatch.delenv(IDLE_DP_GROUP_S, raising=False)
    got = idle_group_dummy(_View(), 2, "step_linear")
    assert [t.seconds for t in got.terms] == [0.0]


def test_a_stated_time_prices_that_many_seconds(monkeypatch):
    monkeypatch.setenv(IDLE_DP_GROUP_S, "0.0075")
    got = idle_group_dummy(_View(), 2, "step_linear")
    assert [t.seconds for t in got.terms] == [0.0075]
    assert IDLE_DP_GROUP_S in got.terms[0].provenance.note


def test_other_steps_are_still_not_idle_group_dummies(monkeypatch):
    monkeypatch.setenv(IDLE_DP_GROUP_S, "0.0075")
    one_rank = type("V", (_View,), {"group_dummies": 1})()
    assert idle_group_dummy(one_rank, 2, "step_linear") is None
    assert cost  # the module under test imported
