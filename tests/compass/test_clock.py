"""The injectable clock."""

import time

import pytest

from atom.utils.clock import (
    VirtualClock,
    WallClock,
    get_clock,
    now,
    perf_counter,
    reset_clock,
    set_clock,
)


@pytest.fixture(autouse=True)
def _restore_default_clock():
    yield
    reset_clock()


def test_default_is_wall_clock_and_tracks_real_time():
    assert isinstance(get_clock(), WallClock)
    before = time.time()
    reading = now()
    after = time.time()
    assert before <= reading <= after


def test_virtual_clock_does_not_move_on_its_own():
    set_clock(VirtualClock())
    first = now()
    time.sleep(0.01)
    assert now() == first, "virtual time advanced without being told to"


def test_advance_moves_both_readings_by_the_same_amount():
    clock = VirtualClock()
    set_clock(clock)
    t0, p0 = now(), perf_counter()
    clock.advance(1.5)
    assert now() == pytest.approx(t0 + 1.5)
    assert perf_counter() == pytest.approx(p0 + 1.5)
    assert clock.elapsed == pytest.approx(1.5)


def test_perf_counter_starts_at_zero_but_time_looks_like_an_epoch():
    clock = VirtualClock()
    set_clock(clock)
    assert perf_counter() == 0.0
    # Plausible as a wall-clock timestamp, so anything that formats it still works.
    assert now() > 1_600_000_000.0


def test_advance_rejects_going_backwards():
    clock = VirtualClock()
    with pytest.raises(ValueError):
        clock.advance(-0.001)


def test_set_clock_returns_the_previous_one():
    original = get_clock()
    replaced = set_clock(VirtualClock())
    assert replaced is original
