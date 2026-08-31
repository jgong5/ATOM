"""A simulated run's latency has to leave the engine, or it does not exist.

`benchmark_serving` times an HTTP stream and divides by its own wall clock, so
against a Compass server it measures the simulator. The engine's own readings
are the simulated latency, and before this they were computed only in
`postprocess`, which the serving path never calls.
"""

import ast
from pathlib import Path

import pytest

from atom.model_engine.request import RequestOutput

ROOT = Path(__file__).resolve().parents[2]


class TestRequestOutputCarriesTheEngineClock:
    def test_the_fields_exist_and_default_to_unstamped(self):
        out = RequestOutput(request_id=1, output_tokens=[], finished=False)
        assert out.arrive_time == 0.0
        assert out.first_token_time == 0.0
        assert out.finish_time == 0.0

    def test_they_survive_the_process_boundary(self):
        """Pickled: the streaming process is not the one that owns the seq."""
        import pickle

        out = RequestOutput(request_id=1, output_tokens=[7], finished=True,
                            arrive_time=100.0, first_token_time=100.5,
                            finish_time=102.0)
        back = pickle.loads(pickle.dumps(out))
        assert (back.arrive_time, back.first_token_time, back.finish_time) == (
            100.0, 100.5, 102.0)


class TestEveryStampGoesThroughTheClock:
    """One of the three sites used `time.time()` and produced a mixed-clock TTFT.

    A wall-clock instant minus a virtual one is not a duration. Guarded by
    inspection rather than by a run, because the site only fires on the
    speculative-decode path.
    """

    def test_no_first_token_time_is_stamped_from_wall_time(self):
        source = (ROOT / "atom/model_engine/scheduler.py").read_text()
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t for t in node.targets
                       if isinstance(t, ast.Attribute)
                       and t.attr in ("first_token_time", "finish_time",
                                      "arrive_time")]
            if not targets:
                continue
            call = node.value
            if isinstance(call, ast.Call):
                func = call.func
                # time.time() / time.monotonic() rather than get_clock().time()
                if isinstance(func, ast.Attribute) and isinstance(
                        func.value, ast.Name) and func.value.id == "time":
                    offenders.append((targets[0].attr, node.lineno))
        assert not offenders, (
            f"stamped off the wall clock, bypassing get_clock(): {offenders}")


class TestTheRecorder:
    @pytest.fixture(autouse=True)
    def _server(self):
        import atom.entrypoints.openai.api_server as server
        server._compass_records.clear()
        yield server
        server._compass_records.clear()

    def _out(self, **kw):
        base = dict(request_id=1, output_tokens=[], finished=True,
                    arrive_time=10.0, first_token_time=10.25, finish_time=11.0)
        base.update(kw)
        return RequestOutput(**base)

    def test_it_records_what_the_engine_measured(self, _server):
        _server._record_engine_timings("req-a", self._out())
        row = _server._compass_records["req-a"]
        assert row["ttft"] == pytest.approx(0.25)
        assert row["latency"] == pytest.approx(1.0)

    def test_a_request_with_no_token_has_no_ttft(self, _server):
        """0.0 is not a time-to-first-token; averaging it in would be a lie."""
        _server._record_engine_timings("req-b", self._out(first_token_time=0.0))
        assert _server._compass_records["req-b"]["ttft"] is None
        assert _server._compass_records["req-b"]["latency"] == pytest.approx(1.0)

    def test_an_unstamped_request_is_not_recorded(self, _server):
        _server._record_engine_timings("req-c", self._out(finish_time=0.0))
        _server._record_engine_timings("req-d", self._out(arrive_time=0.0))
        assert _server._compass_records == {}

    def test_it_is_bounded_and_drops_the_oldest(self, _server, monkeypatch):
        """A long-lived server must not grow a row per request forever."""
        monkeypatch.setattr(_server, "COMPASS_MAX_RECORDS", 3)
        for i in range(5):
            _server._record_engine_timings(f"r{i}", self._out())
        assert list(_server._compass_records) == ["r2", "r3", "r4"]


class TestDeclaredArrival:
    """An arrival offset is only meaningful on a clock that knows where it began.

    A simulated engine advances time only when it takes a step, so it does not
    track the wall clock a client sends on: every request lands at the same
    instant and its TTFT is measured from the start of the run instead of from
    when it turned up. Measured, that was 293ms of a 293ms error. So a client
    declares the schedule -- as an offset into the run, which is the part that
    has to be got right.
    """

    @staticmethod
    def _stamp(*args):
        from atom.model_engine.llm_engine import _stamp_arrival

        return _stamp_arrival(*args)

    def test_a_real_clock_has_no_start_of_run(self):
        from atom.utils.clock import WallClock

        assert WallClock().epoch is None, (
            "0.0 would offset from 1970 and yield a duration in the billions")

    def test_a_virtual_clock_reports_where_it_started(self):
        from atom.utils.clock import VirtualClock

        clock = VirtualClock(epoch=1000.0)
        clock.advance(2.5)
        assert clock.epoch == 1000.0
        assert clock.time() == 1002.5

    def test_no_declaration_stamps_now(self):
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        clock.advance(7.0)
        set_clock(clock)
        try:
            assert self._stamp(None) == 1007.0
        finally:
            reset_clock()

    def test_a_declaration_is_an_offset_into_the_run(self):
        """Not a timestamp. This is the bug the first run produced."""
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1_788_000_000.0)
        clock.advance(30.0)
        set_clock(clock)
        try:
            # Half a second into the run, even though the engine is 30s in.
            assert self._stamp(0.5) == pytest.approx(1_788_000_000.5)
        finally:
            reset_clock()

    def test_a_real_clock_refuses_a_declaration_and_says_so(self, caplog):
        from atom.utils.clock import WallClock, reset_clock, set_clock

        set_clock(WallClock())
        try:
            with caplog.at_level("WARNING"):
                stamped = self._stamp(0.5)
        finally:
            reset_clock()
        assert stamped > 1_000_000_000, "must fall back to now, not to 1970"
        assert any("ATOMCompass WARNING:" in r.getMessage()
                   for r in caplog.records)


class TestTheEngineHonoursDeclaredArrivals:
    """Declaring an arrival is only half of it.

    Fixing what TTFT is measured from, while the engine still starts work the
    moment a request is received, produced 62 of 64 requests finishing before
    they had arrived. The scheduler has to hold a request until its time, and
    -- since virtual time only moves when a step runs -- jump the clock forward
    when there is nothing else to do.
    """

    class _Sched:
        """Enough of a Scheduler for the two methods under test."""

        def __init__(self, running=(), waiting=()):
            self.running = list(running)
            self.waiting = list(waiting)

    class _Seq:
        def __init__(self, arrive_time):
            self.arrive_time = arrive_time

    def _pending(self, sched, seq):
        from atom.model_engine.scheduler import Scheduler

        return Scheduler._declared_arrival_pending(sched, seq)

    def _advance(self, sched):
        from atom.model_engine.scheduler import Scheduler

        return Scheduler._advance_to_next_arrival(sched)

    def test_a_real_clock_never_defers(self):
        """Off a simulated run this must be inert, whatever arrive_time says."""
        from atom.utils.clock import WallClock, reset_clock, set_clock

        set_clock(WallClock())
        try:
            future = self._Seq(arrive_time=2**40)
            assert self._pending(self._Sched(), future) is False
        finally:
            reset_clock()

    def test_a_future_arrival_is_deferred(self):
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        set_clock(VirtualClock(epoch=1000.0))
        try:
            assert self._pending(self._Sched(), self._Seq(1005.0)) is True
            assert self._pending(self._Sched(), self._Seq(1000.0)) is False
        finally:
            reset_clock()

    def test_time_jumps_to_the_next_arrival_when_idle(self):
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        set_clock(clock)
        try:
            sched = self._Sched(waiting=[self._Seq(1007.0), self._Seq(1003.0)])
            self._advance(sched)
            assert clock.time() == 1003.0, "goes to the earliest, not the first"
        finally:
            reset_clock()

    def test_time_does_not_move_while_work_is_runnable(self):
        """It must never skip past a request that was ready to go."""
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        set_clock(clock)
        try:
            # One arrival already due: that one runs, time stays put.
            self._advance(self._Sched(waiting=[self._Seq(1005.0),
                                               self._Seq(999.0)]))
            assert clock.time() == 1000.0
            # Something running: the step advances time, not this.
            self._advance(self._Sched(running=[object()],
                                      waiting=[self._Seq(1005.0)]))
            assert clock.time() == 1000.0
        finally:
            reset_clock()

    def test_an_empty_queue_is_left_alone(self):
        from atom.utils.clock import VirtualClock, reset_clock, set_clock

        clock = VirtualClock(epoch=1000.0)
        set_clock(clock)
        try:
            self._advance(self._Sched())
            assert clock.time() == 1000.0
        finally:
            reset_clock()
