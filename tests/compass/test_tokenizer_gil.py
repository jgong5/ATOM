# SPDX-License-Identifier: MIT
"""``COMPASS_TOKENIZER_GIL``: the tokenizer station keeps one interpreter lock.

A job's work charged as holding it (the segment tokenizer's own work) follows
the job's other work, the event loop's own charges hold it from the loop's
clock, and each such section takes the first gap in the lock that holds it.
Off, a station places its jobs as it always has. No device, no run.
"""

import pytest

from atom.utils import compass_loop
from atom.utils.compass_loop import GIL_ENV, SimExecutor, Station


def _place(station, jobs):
    """Submit `jobs`, each (submitted, [(seconds, gil), ...]), then finish them
    in order; their completions."""
    ks = []
    for t, charges in jobs:
        k = station.enqueue(t)
        for d, gil in charges:
            station.charge(k, d, gil)
        ks.append(k)
    for k in ks:
        station.finish(k)
    return [station.completion(k) for k in ks]


JOBS = [
    (0.0, [(1.0, False), (0.5, True)]),
    (0.0, [(1.0, False), (0.5, True)]),
    (0.0, [(0.25, False), (0.5, True)]),
]


def test_off_a_station_places_every_charge_on_the_job_s_own_server():
    # Two servers, no lock: each job ends its whole charge after its start.
    assert _place(Station(2), JOBS) == [1.5, 1.5, 2.25]


def test_on_each_job_s_locked_part_waits_for_the_one_lock_in_turn():
    # Job 0: its own 1 s, then the lock 1.0-1.5; job 1: own 1 s, lock 1.5-2.0;
    # job 2 starts on the first server free (1.5), own 0.25 s, lock 2.0-2.5.
    station = Station(2, gil=True)
    assert _place(station, JOBS) == [1.5, 2.0, 2.5]
    assert station.gil_busy == [(1.0, 2.5)]  # touching sections merge


def test_the_event_loop_s_work_takes_the_first_gap_in_the_lock_that_holds_it():
    station = Station(2, gil=True)
    _place(station, JOBS)
    # Before the jobs' locked sections (from 1.0) there is room: at once.
    assert station.hold(0.25, 0.125) == 0.375
    # 0.75 s from 0.5 does not fit before 1.0: after the last section, 2.5.
    assert station.hold(0.5, 0.75) == 3.25
    assert station.hold(4.0, 0.125) == 4.125
    assert station.gil_busy == [(0.25, 0.375), (1.0, 3.25), (4.0, 4.125)]


def test_the_lock_forgets_sections_long_before_the_one_it_takes(monkeypatch):
    monkeypatch.setattr(compass_loop, "GIL_PRUNE_EVERY", 2)
    station = Station(1, gil=True)
    station.hold(0.0, 1.0)
    station.hold(2.0, 1.0)  # the second section: prune, nothing old yet
    assert station.gil_busy == [(0.0, 1.0), (2.0, 3.0)]
    station.hold(100.0, 1.0)
    station.hold(200.0, 1.0)  # the fourth: drops those ended 60 s before 200
    assert station.gil_busy == [(100.0, 101.0), (200.0, 201.0)]


def test_a_job_with_no_locked_work_ignores_the_lock():
    station = Station(1, gil=True)
    assert _place(station, [(0.0, [(1.0, False)]), (0.0, [(1.0, False)])]) == [1.0, 2.0]
    assert station.gil_busy == []


@pytest.mark.parametrize(
    "value, on", [(None, False), ("", False), ("0", False), ("1", True)]
)
def test_the_switch_sets_the_station_s_lock(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(GIL_ENV, raising=False)
    else:
        monkeypatch.setenv(GIL_ENV, value)
    executor = SimExecutor(loop=None, max_workers=4)
    try:
        executor.start_run()
        assert executor.station.gil is on
    finally:
        executor.shutdown(wait=False)


def test_the_segment_tokenizer_s_own_work_is_charged_as_holding_the_lock(monkeypatch):
    seen = []
    monkeypatch.setattr(
        compass_loop, "charge", lambda d, gil=False: seen.append((d, gil))
    )
    compass_loop._encoded.ids = 0

    def encode_cached(*a, **k):
        compass_loop._encoded.ids = 2000
        return "ids"

    wrapped = compass_loop.wrap_segment(
        encode_cached, {"fixed_s": 0.001, "tokens_per_s": 1000.0}
    )
    assert wrapped() == "ids"
    assert seen == [(pytest.approx(2.001), True)]
