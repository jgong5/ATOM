# SPDX-License-Identifier: MIT
"""COMPASS_FIT_GRADED_SECONDS: a record's graded window cut to its first seconds."""

import pytest

from atom.compass.fit.samples import graded_to_step


def _record(enter_s_by_rank):
    return {
        rank: [{"step": n, "t_enter_ns": int(t * 1e9)} for n, t in enumerate(times)]
        for rank, times in enter_s_by_rank.items()
    }


def test_the_cut_is_the_first_step_entering_that_many_seconds_after_the_window_starts():
    # Steps enter at 0, 10, 20, ... s; the window starts at step 2 (20 s).
    record = _record(
        {0: [10.0 * n for n in range(10)], 1: [10.0 * n + 0.5 for n in range(10)]}
    )
    assert graded_to_step(record, start=2, seconds=30) == 5  # 20 + 30 = 50 s
    assert (
        graded_to_step(record, start=2, seconds=25) == 5
    )  # the first at or after 45 s


def test_a_step_s_entry_is_its_earliest_rank_s():
    # Rank 1 enters step 3 first, at 29 s: 9 s after step 2 entered on rank 0.
    record = _record({0: [0.0, 10.0, 20.0, 31.0], 1: [0.0, 10.0, 20.5, 29.0]})
    assert graded_to_step(record, start=2, seconds=9) == 3


def test_a_record_ending_before_the_cut_keeps_every_step():
    record = _record({0: [0.0, 10.0, 20.0]})
    assert graded_to_step(record, start=0, seconds=600) is None


def test_a_window_start_with_no_entry_stamp_refuses():
    record = {0: [{"step": 0}, {"step": 1, "t_enter_ns": 1}]}
    with pytest.raises(ValueError, match="no entry stamp"):
        graded_to_step(record, start=0, seconds=1)
