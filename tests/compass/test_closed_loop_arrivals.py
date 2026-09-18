"""A closed-loop client cannot declare its arrivals, so the engine must stamp them.

Declared arrivals were the only path a simulated run had, and they work because
a declaration is an *offset* from an epoch both processes agree on -- no clock
is read at all. Everything else about arrival stamping was therefore never
exercised, and what it does is wrong: the API process stamps `clock.time()` on
a virtual clock that never advances, because the engine core advances its own
copy in another process. Every undeclared request then arrives at the start of
the run.

That is not a latent edge. A closed loop has nothing it *can* declare -- it
learns simulated time only when the previous response returns -- and declaring
a workload size instead would deadlock the arrival barrier, which waits for
requests a closed loop will not send until it is released. So the closed loop
takes this path on every single request.

Measured at the one-client smoke rung before the fix: real arrivals at
0.000/0.120/0.322/0.322s, modelled at 0.000/0.000/0.000/0.000. TTFT then
measures from the start of the run rather than from arrival (+203% p50, an
artifact of the clock and not of any cost model), and the run's span collapses
onto its busy time, overstating tokens/s per GPU by 68%.
"""
import pytest


def _restamp(*args):
    from atom.model_engine.engine_core import _restamp_undeclared_arrivals

    return _restamp_undeclared_arrivals(*args)


class _Seq:
    """Enough of a Sequence for the stamp: the two fields it touches."""

    def __init__(self, declared, arrive=0.0):
        self.compass_arrival_declared = declared
        self.arrive_time = arrive


class TestTheTwoClocksAreNotTheSameClock:
    def test_the_api_processes_clock_stands_still_while_the_cores_advances(self):
        """The fact the bug rests on, pinned so it cannot be assumed away.

        Both processes construct a VirtualClock from the same epoch and only
        the engine core ever calls advance. A unit test that advances one clock
        and reads it back cannot see this; it takes two instances.
        """
        from atom.utils.clock import VirtualClock

        api, core = VirtualClock(epoch=1000.0), VirtualClock(epoch=1000.0)
        core.advance(7.0)
        assert core.time() == 1007.0
        assert api.time() == 1000.0, (
            "if this ever advances, the restamp below is redundant")


class TestAnUndeclaredArrivalIsStampedByTheCore:
    def test_it_takes_the_cores_current_virtual_time(self):
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        set_clock(clock)
        try:
            clock.advance(7.0)
            seq = _Seq(declared=False, arrive=1000.0)  # the frozen-epoch stamp
            _restamp([seq])
            assert seq.arrive_time == 1007.0
        finally:
            reset_clock()

    def test_requests_admitted_at_different_times_do_not_share_an_arrival(self):
        """The whole symptom, in one assertion.

        Four sequential requests of one closed-loop client landed on the same
        instant before this, which is what made their TTFTs accumulate.
        """
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        set_clock(clock)
        try:
            stamped = []
            for step in (0.1, 0.2, 0.3):
                clock.advance(step)
                seq = _Seq(declared=False)
                _restamp([seq])
                stamped.append(seq.arrive_time)
            assert stamped == pytest.approx([1000.1, 1000.3, 1000.6])
            assert len(set(stamped)) == 3
        finally:
            reset_clock()


class TestADeclaredArrivalIsLeftAlone:
    def test_the_declaration_survives_admission(self):
        """Restamping a declared arrival would break the arrival barrier.

        The barrier holds the run until every declared request is in, precisely
        so virtual time is never advanced past an arrival nobody announced.
        Overwriting the declaration with admission time would make the schedule
        the client asked for unobservable -- and it would do so silently, since
        the run still completes.
        """
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        set_clock(clock)
        try:
            clock.advance(7.0)
            seq = _Seq(declared=True, arrive=1002.5)
            _restamp([seq])
            assert seq.arrive_time == 1002.5
        finally:
            reset_clock()


class TestARealRunIsUntouched:
    def test_a_wall_clock_keeps_the_stamp_taken_nearer_the_wire(self):
        """On a real clock the API process's "now" is the better reading.

        It is taken at the entrypoint, before the sequence crosses a socket, so
        restamping here would move arrival later by the crossing and quietly
        shrink every TTFT.
        """
        from atom.utils.clock import reset_clock

        reset_clock()  # the default WallClock, whose epoch is None
        seq = _Seq(declared=False, arrive=12345.0)
        _restamp([seq])
        assert seq.arrive_time == 12345.0
