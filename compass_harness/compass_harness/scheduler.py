# SPDX-License-Identifier: MIT
"""A ``LoopScheduler`` whose delayed work fires on the Compass clock.

Imported only after ``aiperf.timing.phase.runner`` has executed (by the
package's import hook, or by the strategy plugin aiperf loads lazily), never
from the package ``__init__``.
"""

import asyncio

from aiperf.common.loop_scheduler import LoopScheduler


class _Timer:
    """A pending entry in ``LoopScheduler._handles`` that fires at clock time ``at``."""

    __slots__ = ("at",)

    def __init__(self, at: float) -> None:
        self.at = at

    def when(self) -> float:
        return self.at

    def cancel(self) -> None:
        pass


class ClockPacedLoopScheduler(LoopScheduler):
    """``LoopScheduler`` whose pacing calls wait on ``clock``, never on the event loop's timers.

    ``clock`` is bound before the first phase runner is built. It has
    ``now()``, the current time in seconds, and ``async advance_to(t)``, which
    returns once the clock may run to ``t``; it may return earlier, at a time
    something else is due, and the scheduler then asks again. Each grant wakes
    every pending wait. A timer armed or capped earlier than the time the driver
    waits for is asked for in a wait of its own, and the driver's wait stays
    pending: cancelling it would cancel a future the clock holds.
    """

    clock = None

    def __init__(self, *args, **kwargs) -> None:
        if self.clock is None:
            raise RuntimeError(
                "Compass clock not bound: set ClockPacedLoopScheduler.clock "
                "before aiperf builds a phase runner."
            )
        super().__init__(*args, **kwargs)
        self._clock = self.clock
        self._driver: asyncio.Task | None = None
        # While the driver waits on the clock: the earliest time asked since.
        self._waiting_for: float | None = None
        self._asks: set[asyncio.Task] = set()

    def schedule_later(self, delay_sec, coro, *, group_id=None):
        if delay_sec <= 0:
            return self.execute_async(coro)
        timer = _Timer(self._clock.now() + delay_sec)
        handle_id = self._track_handle_and_return_id(timer, coro, group_id=group_id)
        if self._driver is None or self._driver.done():
            self._driver = self._loop.create_task(self._drive())
        self._ask_earlier(timer.at)
        return handle_id

    def _ask_earlier(self, at: float) -> None:
        """Ask the clock for ``at`` beside the driver's wait for a later time."""
        if self._waiting_for is None or at >= self._waiting_for:
            return
        self._waiting_for = at
        # Not execute_async: cancel_all would cancel the wait the clock holds.
        ask = self._loop.create_task(self._clock.advance_to(at))
        self._asks.add(ask)
        ask.add_done_callback(self._asks.discard)

    def _wall_clock(self, *args, **kwargs):
        raise NotImplementedError(
            "ClockPacedLoopScheduler refuses absolute-time scheduling: "
            "schedule_at and schedule_at_perf_* read the wall clock."
        )

    schedule_at = schedule_at_perf_sec = schedule_at_perf_ns = _wall_clock

    def cap_pending_delay(self, max_delay_sec):
        return self._advance(max_delay_sec, lambda handle_id: True)

    def cap_pending_delay_for_group(self, group_id, max_delay_sec):
        return self._advance(
            max_delay_sec,
            lambda handle_id: self._handle_groups.get(handle_id) == group_id,
        )

    def _advance(self, max_delay_sec, selected) -> float:
        """Move the selected timers earlier together, so the first is due ``max_delay_sec`` from now."""
        if max_delay_sec < 0:
            raise ValueError("max_delay_sec must be non-negative")
        timers = [
            t for handle_id, (t, _) in self._handles.items() if selected(handle_id)
        ]
        if not timers:
            return 0.0
        shift = min(t.at for t in timers) - self._clock.now() - max_delay_sec
        if shift <= 0:
            return 0.0
        for t in timers:
            t.at -= shift
        self._ask_earlier(self._clock.now() + max_delay_sec)
        return shift

    async def _drive(self) -> None:
        # A cancel never interrupts the clock wait: the clock holds that wait and
        # wakes it at its next grant, and the loop then exits if no timer is left.
        # ponytail: linear scan for the earliest timer; a heap if pending timers reach thousands.
        while self._handles:
            timer, coro = min(self._handles.values(), key=lambda entry: entry[0].at)
            if timer.at > self._clock.now():
                self._waiting_for = timer.at
                await self._clock.advance_to(timer.at)
                self._waiting_for = None
            else:
                self._safe_callback([timer], coro)
            await asyncio.sleep(0)
