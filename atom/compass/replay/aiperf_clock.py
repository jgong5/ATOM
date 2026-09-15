"""Shared clock and asyncio pump for an internal controlled AIPerf replay.

The CPython 3.12 seam below only selects the next time boundary. Asyncio still
runs every coroutine, timer and cancellation; AIPerf owns all branch policy.
"""
from collections import deque
from contextlib import contextmanager
import heapq
import math
import sys
import asyncio


class ReplayClock:
    """Use the engine's absolute timeline for all semantic clock readings.

    Perf/monotonic readings share its epoch. Their differences retain the
    usual meaning, and an initial request never receives an invalid zero
    performance timestamp. Conversion to integer nanoseconds is a unit
    projection; the engine's floating-point timestamps remain unchanged.
    """

    def __init__(self, clock):
        self.clock = clock

    def time(self):
        return self.clock.time()

    def time_ns(self):
        return round(self.time() * 1_000_000_000)

    perf_counter = time
    monotonic = time
    perf_counter_ns = time_ns
    monotonic_ns = time_ns


class Acknowledgements:
    """Outstanding deliveries that must settle before virtual time can move."""

    def __init__(self):
        self._pending = set()

    def expect(self, key):
        if key in self._pending:
            raise ValueError(f"duplicate pending acknowledgement: {key!r}")
        self._pending.add(key)

    def acknowledge(self, key):
        if key not in self._pending:
            raise ValueError(f"unknown acknowledgement: {key!r}")
        self._pending.remove(key)

    @contextmanager
    def delivering(self, key):
        self.expect(key)
        try:
            yield
        finally:
            self.acknowledge(key)

    @property
    def pending(self):
        return frozenset(self._pending)


class ControlledEventLoop(asyncio.SelectorEventLoop):
    """Run real asyncio callbacks between explicitly committed core boundaries."""

    def __init__(self, engine, acknowledgements, on_engine_yield):
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("controlled replay supports the audited CPython 3.12 asyncio loop")
        self.engine = engine
        self.acknowledgements = acknowledgements
        self.on_engine_yield = on_engine_yield
        self.teardown_only = False
        super().__init__()
        required = {"_ready", "_scheduled", "_selector", "_clock_resolution", "_process_events"}
        if (not required.issubset(set(asyncio.BaseEventLoop._run_once.__code__.co_names))
                or not isinstance(self._ready, deque) or not isinstance(self._scheduled, list)):
            self.close()
            raise RuntimeError("unsupported CPython asyncio ready/timer contract")
        # BaseEventLoop promotes timers slightly early by its clock resolution.
        # We promote only timestamps <= the committed frontier before calling
        # its unchanged callback runner.
        self._clock_resolution = 0.0

    def time(self):
        return self.engine.clock.time()

    def _promote_due_timers(self):
        while self._scheduled:
            handle = self._scheduled[0]
            if not handle._cancelled and handle._when > self.time():
                break
            heapq.heappop(self._scheduled)
            handle._scheduled = False
            if handle._cancelled:
                self._timer_cancelled_count -= 1
            else:
                self._ready.append(handle)

    def _observe_io(self):
        self._process_events(self._selector.select(0))

    def _advance_core(self):
        while not self._ready and not self.acknowledgements.pending:
            result = self.engine.advance_until(self.time(), include_horizon=True)
            self.on_engine_yield(result)
            self._observe_io()
            self._promote_due_timers()
            if self._ready or self.acknowledgements.pending:
                return
            next_core = math.inf if result.reason == "blocked" else result.next_boundary_at
            next_timer = self._scheduled[0]._when if self._scheduled else math.inf
            target = min(next_core, next_timer)
            if not math.isfinite(target):
                if result.idle:
                    return
                raise RuntimeError("controlled replay is blocked without a timed event")
            if target < self.time():
                raise RuntimeError(
                    f"controlled replay boundary is behind frontier: now={self.time()!r}, "
                    f"next_core={next_core!r}, next_timer={next_timer!r}, reason={result.reason}")
            if target == self.time():
                continue
            # Client timers at the horizon run before the core may include it.
            result = self.engine.advance_until(target, include_horizon=False)
            self.on_engine_yield(result)
            self._observe_io()
            self._promote_due_timers()

    def _run_once(self):
        self._observe_io()
        self._promote_due_timers()
        if (not self.teardown_only and not self._ready and not self._stopping
                and not self.acknowledgements.pending):
            self._advance_core()
        self._promote_due_timers()
        super()._run_once()
