"""An agentic replay cannot state when its requests arrive, so it states why.

A declared arrival is an offset from a shared epoch, which works for an open
loop: the recording says request 41 arrived at t=812.4, and both sides can be
told so. A closed loop has nothing it can declare. Turn two goes out some think
time after turn one comes back, and *when turn one comes back is what the run
is measuring* -- stating it would replay the recorded engine's speed instead of
the modelled one's, and the sweep would return the trace whatever the model
said.

The client cannot resolve it either. Its wall clock and the engine's virtual
clock race: a millisecond of round trip can be seconds of simulated time, so a
request sent "right after" a response lands after work it was meant to precede.

So the whole run is declared up front as a graph -- each request naming the
requests it waits on and the recorded gap -- and the engine stamps the arrival
as those predecessors finish. These tests cover that stamping.
"""
import pytest


class _Seq:
    """Enough of a Sequence for the arrival logic: the fields it touches."""

    def __init__(self, sid, after=(), think_s=0.0):
        self.id = sid
        self.compass_id = str(sid)
        self.compass_after = tuple(str(a) for a in after)
        self.compass_think_s = float(think_s)
        self.compass_arrival_resolved = False
        self.compass_arrival_declared = True
        self.arrive_time = 0.0
        self.finish_time = None


class _Plain:
    """A request that declared nothing -- an ordinary open-loop arrival."""

    def __init__(self, arrive=0.0):
        self.id = 99
        self.compass_id = None
        self.compass_after = ()
        self.compass_think_s = None
        self.compass_arrival_resolved = False
        self.arrive_time = arrive


def _sched(epoch=0.0):
    """The arrival logic on its own, without building an engine.

    Borrowed as unbound methods rather than constructed, because a real
    Scheduler needs a model, a KV cache and a device, and none of those take
    any part in deciding when a request arrived.
    """
    from atom.model_engine.scheduler import Scheduler

    class Stub:
        _admission_seconds = Scheduler._admission_seconds
        _compass_note_finished = Scheduler._compass_note_finished
        _resolve_relative_arrivals = Scheduler._resolve_relative_arrivals
        _compass_track_unresolved = Scheduler._compass_track_unresolved
        _schedulable_at = Scheduler._schedulable_at

        def __init__(self):
            self.config = None          # -> admission_seconds 0.0
            self._compass_finished = {}
            self._compass_unresolved = {}

    return Stub()


@pytest.fixture(autouse=True)
def _virtual_clock():
    from atom.utils.clock import VirtualClock, reset_clock, set_clock

    set_clock(VirtualClock(epoch=1000.0))
    yield
    reset_clock()


class TestAChainResolvesOneLinkAtATime:
    def test_a_request_that_waits_for_nothing_arrives_at_the_epoch(self):
        sched = _sched()
        first = _Seq(0, after=(), think_s=0.0)
        sched._compass_track_unresolved(first)
        sched._resolve_relative_arrivals()
        assert first.compass_arrival_resolved
        assert first.arrive_time == 1000.0

    def test_its_own_offset_is_added_to_the_epoch(self):
        sched = _sched()
        # A sub-agent branch that was already running when the session opened.
        branch = _Seq(1, after=(), think_s=2.5)
        sched._compass_track_unresolved(branch)
        sched._resolve_relative_arrivals()
        assert branch.arrive_time == 1002.5

    def test_the_next_turn_arrives_a_think_time_after_the_previous_finished(self):
        sched = _sched()
        second = _Seq(1, after=(0,), think_s=4.0)
        sched._compass_track_unresolved(second)

        # Nothing has finished, so there is nothing to measure from.
        sched._resolve_relative_arrivals()
        assert not second.compass_arrival_resolved

        first = _Seq(0)
        first.finish_time = 1007.0
        sched._compass_note_finished(first)
        sched._resolve_relative_arrivals()
        assert second.arrive_time == 1011.0

    def test_it_waits_for_the_last_predecessor_not_the_first(self):
        sched = _sched()
        # A turn and a sub-agent; the user could not have read the answer
        # before the slower branch returned.
        third = _Seq(2, after=(0, 1), think_s=1.0)
        sched._compass_track_unresolved(third)

        done = _Seq(0)
        done.finish_time = 1005.0
        sched._compass_note_finished(done)
        sched._resolve_relative_arrivals()
        assert not third.compass_arrival_resolved, (
            "one of two predecessors is not all of them")

        slow = _Seq(1)
        slow.finish_time = 1020.0
        sched._compass_note_finished(slow)
        sched._resolve_relative_arrivals()
        assert third.arrive_time == 1021.0


class TestAnUnresolvedArrivalIsNotAnArrivalAtZero:
    def test_it_has_no_schedulable_instant_yet(self):
        sched = _sched()
        waiting = _Seq(1, after=(0,), think_s=4.0)
        sched._compass_track_unresolved(waiting)
        # The placeholder in arrive_time is 0.0. Reading it as an arrival would
        # make the request schedulable immediately -- which is the whole bug
        # this mechanism exists to avoid, and it looks like a fast engine.
        assert sched._schedulable_at(waiting) == float("inf")

    def test_it_becomes_schedulable_once_its_predecessor_finishes(self):
        sched = _sched()
        waiting = _Seq(1, after=(0,), think_s=4.0)
        sched._compass_track_unresolved(waiting)
        first = _Seq(0)
        first.finish_time = 1007.0
        sched._compass_note_finished(first)
        sched._resolve_relative_arrivals()
        assert sched._schedulable_at(waiting) == 1011.0

    def test_a_request_that_declared_nothing_is_answered_from_its_stamp(self):
        sched = _sched()
        plain = _Plain(arrive=1003.0)
        sched._compass_track_unresolved(plain)
        assert sched._compass_unresolved == {}, (
            "an open-loop request must not enter the relative-arrival index")
        assert sched._schedulable_at(plain) == 1003.0


class TestTheIndexEmpties:
    def test_a_resolved_request_leaves_the_index(self):
        sched = _sched()
        seq = _Seq(1, after=(0,), think_s=1.0)
        sched._compass_track_unresolved(seq)
        assert len(sched._compass_unresolved) == 1
        first = _Seq(0)
        first.finish_time = 1002.0
        sched._compass_note_finished(first)
        sched._resolve_relative_arrivals()
        assert sched._compass_unresolved == {}

    def test_a_resolved_arrival_is_never_restamped(self):
        sched = _sched()
        seq = _Seq(1, after=(0,), think_s=1.0)
        sched._compass_track_unresolved(seq)
        first = _Seq(0)
        first.finish_time = 1002.0
        sched._compass_note_finished(first)
        sched._resolve_relative_arrivals()
        assert seq.arrive_time == 1003.0

        # The clock moves on and the request has still not been scheduled.
        # Re-deriving the arrival now would let it drift forward under a
        # request that was merely queued, and TTFT would come out near zero.
        from atom.utils.clock import get_clock

        get_clock().advance(50.0)
        sched._resolve_relative_arrivals()
        assert seq.arrive_time == 1003.0

    def test_a_workload_that_declares_nothing_costs_nothing_to_resolve(self):
        sched = _sched()
        # The guard that keeps this off the hot path: with an empty index the
        # call returns without reading the waiting queue at all, which at 3,000
        # waiting requests and millions of ticks is the difference between a
        # constant and the run not finishing.
        sched._resolve_relative_arrivals()
        assert sched._compass_unresolved == {}


class TestAFailedRequestDoesNotDeadlockItsSuccessors:
    def test_a_rejected_predecessor_still_releases_what_waited_on_it(self):
        sched = _sched()
        waiting = _Seq(1, after=(0,), think_s=2.0)
        sched._compass_track_unresolved(waiting)

        # Rejected before it ran -- unschedulable, or aborted while waiting.
        # It still stamped a finish time, and the successor must see it: a
        # session whose turn three waits forever on a failed turn two takes the
        # whole run down with it, and the artifact reports it as a hang.
        rejected = _Seq(0)
        rejected.finish_time = 1004.0
        sched._compass_note_finished(rejected)
        sched._resolve_relative_arrivals()
        assert waiting.arrive_time == 1006.0
