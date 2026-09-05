"""The engine-side half of simulated timing.

ATOM runs its workers in separate processes even at world size one, so a
simulated runner cannot advance time itself — it reports a predicted duration
and the scheduling process moves its own clock. These tests cover that handoff
and, importantly, that a normal run is unaffected by any of it.
"""

import pytest

from atom.model_engine.engine_core import _advance_clock_for
from atom.model_engine.scheduler import ScheduledBatchOutput
from atom.utils.clock import (
    VirtualClock,
    WallClock,
    get_clock,
    now,
    reset_clock,
    set_clock,
)


def _output(seconds=None):
    return ScheduledBatchOutput(
        req_ids=[1],
        token_ids=[(5,)],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
        compass_step_seconds=seconds,
    )


@pytest.fixture(autouse=True)
def _restore_default_clock():
    yield
    reset_clock()


def test_a_real_step_carries_no_duration():
    assert _output().compass_step_seconds is None


def test_advancing_is_a_no_op_on_the_wall_clock():
    reset_clock()
    _advance_clock_for(_output(0.25))
    assert isinstance(get_clock(), WallClock)


def test_a_real_step_does_not_move_a_virtual_clock():
    set_clock(VirtualClock())
    before = now()
    _advance_clock_for(_output(None))
    assert now() == before


def test_a_simulated_step_moves_the_clock_by_its_prediction():
    set_clock(VirtualClock())
    before = now()
    _advance_clock_for(_output(0.25))
    assert now() - before == pytest.approx(0.25)


def test_durations_accumulate_without_spending_wall_time():
    clock = VirtualClock()
    set_clock(clock)
    for _ in range(4):
        _advance_clock_for(_output(0.25))
    # A second of modelled serving, in no time at all. This is the whole
    # reason a simulated run is faster than the run it stands in for.
    assert clock.elapsed == pytest.approx(1.0)
