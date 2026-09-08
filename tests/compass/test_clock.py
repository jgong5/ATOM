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


class TestPacedArrivalsOnARealRun:
    """A recorded trace declares arrivals; a real run used to discard them.

    `WallClock.epoch` is None because serving has no start-of-run, and
    `_stamp_arrival` therefore stamped "now" and warned. That is right for
    serving and wrong for a replay: the simulated side honours arrivals through
    its virtual clock, so a real side that does not is answering a burst while
    the simulated one answers the trace, and the two schedule nothing alike.

    `PacedWallClock` is real time with a declared origin -- enough for an
    arrival to be placed, and deliberately without `advance`, so the scheduler's
    skip-the-idle-gap shortcut stays off where idle is real.
    """

    def test_epoch_is_unset_until_the_workload_starts(self):
        from atom.utils.clock import PacedWallClock

        # The engine is built minutes before a workload arrives; an origin from
        # construction would leave every declared arrival already in the past.
        assert PacedWallClock().epoch is None

    def test_the_first_declared_arrival_sets_the_origin(self):
        import time

        from atom.utils.clock import PacedWallClock

        clock = PacedWallClock()
        clock.start(5.0)
        assert time.time() - clock.epoch == pytest.approx(5.0, abs=0.5)

    def test_the_origin_is_set_once(self):
        """Requests are posted concurrently, so the first one stamped is not
        the one with the smallest offset."""
        from atom.utils.clock import PacedWallClock

        clock = PacedWallClock()
        first = clock.start(5.0)
        assert clock.start(900.0) == first

    def test_it_cannot_skip_idle(self):
        """`_advance_to_next_arrival` requires `advance`; real idle is real."""
        from atom.utils.clock import PacedWallClock

        assert not hasattr(PacedWallClock(), "advance")

    def test_a_wall_clock_still_refuses_declared_arrivals(self):
        from atom.utils.clock import WallClock

        assert WallClock().epoch is None

    def test_a_declared_arrival_is_placed_on_the_origin(self):
        from atom.model_engine.llm_engine import _stamp_arrival
        from atom.utils.clock import PacedWallClock, set_clock, reset_clock

        from atom.utils.clock import get_clock

        try:
            set_clock(PacedWallClock())  # returns the clock it replaced
            first = _stamp_arrival(0.0)
            later = _stamp_arrival(30.0)
            assert later - first == pytest.approx(30.0, abs=0.5)
            assert get_clock().epoch is not None
        finally:
            reset_clock()

    def test_an_undeclared_arrival_is_still_now(self):
        """Ordinary serving alongside a paced replay is unaffected."""
        import time

        from atom.model_engine.llm_engine import _stamp_arrival
        from atom.utils.clock import PacedWallClock, set_clock, reset_clock

        try:
            set_clock(PacedWallClock())
            assert _stamp_arrival(None) == pytest.approx(time.time(), abs=1.0)
        finally:
            reset_clock()


class TestPacingIsNotAppliedToASimulation:
    def test_a_predicting_run_leaves_pacing_off(self):
        """It already honours arrivals, and advances straight to the next one
        when idle; pacing it would make it wait out the trace for no gain."""
        from atom.compass.config import CompassConfig

        config = CompassConfig(enabled=True, mode="predict", virtual_clock=True,
                               paced_arrivals=True)
        assert not config.paced_arrivals

    def test_a_measuring_run_keeps_it(self):
        from atom.compass.config import CompassConfig

        config = CompassConfig(enabled=True, mode="measure",
                               measure_out="/tmp/x.jsonl", paced_arrivals=True)
        assert config.paced_arrivals
        assert not config.virtual_clock


class TestTheSchedulerHoldsAPacedArrival:
    """Stamping an arrival is half of it; the scheduler has to hold the request.

    Without the gate a declared arrival only changes what TTFT is measured
    *from* while the work still happens immediately -- which is how 62 of 64
    requests once finished before they arrived. The gate already existed and
    was switched off by `epoch is None`, which is exactly what a paced clock
    supplies.
    """

    def _gate(self, seq):
        from atom.model_engine.scheduler import Scheduler

        class _Stub:
            _admission_seconds = 0.0

            def _schedulable_at(self, s):
                return s.arrive_time + self._admission_seconds

        return Scheduler._declared_arrival_pending(_Stub(), seq)

    def _seq(self, arrive_time):
        return type("Seq", (), {"arrive_time": arrive_time})()

    def test_a_future_arrival_is_held(self):
        import time

        from atom.utils.clock import PacedWallClock, set_clock, reset_clock

        try:
            clock = set_clock(PacedWallClock())
            from atom.utils.clock import get_clock

            get_clock().start(0.0)
            assert self._gate(self._seq(time.time() + 30.0))
        finally:
            reset_clock()

    def test_an_arrival_already_past_is_not(self):
        import time

        from atom.utils.clock import PacedWallClock, get_clock, set_clock, reset_clock

        try:
            set_clock(PacedWallClock())
            get_clock().start(0.0)
            assert not self._gate(self._seq(time.time() - 1.0))
        finally:
            reset_clock()

    def test_a_plain_wall_clock_holds_nothing(self):
        """Ordinary serving must not start gating on a stamped arrival."""
        import time

        from atom.utils.clock import reset_clock

        reset_clock()
        assert not self._gate(self._seq(time.time() + 30.0))
