# SPDX-License-Identifier: MIT
"""A ``LoopScheduler`` whose delayed work fires on the Compass clock.

Imported only after ``aiperf.timing.phase.runner`` has executed (by the
package's import hook, or by the strategy plugin aiperf loads lazily), never
from the package ``__init__``.
"""

import asyncio
import heapq
import itertools
import math

from aiperf.common.loop_scheduler import LoopScheduler


class _Timer:
    """A pending entry in ``LoopScheduler._handles`` that fires at clock time ``at``.

    ``LoopScheduler`` calls ``cancel`` as it drops the timer.
    """

    __slots__ = ("at", "cancel")

    def __init__(self, at: float, cancel) -> None:
        self.at, self.cancel = at, cancel

    def when(self) -> float:
        return self.at


class ClockPacedLoopScheduler(LoopScheduler):
    """``LoopScheduler`` whose pacing calls wait on ``clock``, never on the event loop's timers.

    ``clock`` is bound before the first phase runner is built. It has
    ``now()``, the current time in seconds, ``wait(t)``, a future resolved once
    the clock reaches ``t``, and ``withdraw(fut)``, which drops a wait, so it
    asks the clock for nothing, and resolves it now. The driver waits for the
    earliest pending timer; a timer armed or capped earlier than that wait, or
    any timer's cancel, withdraws it, and the driver reads its timers again.

    The driver takes the earliest pending timer, the first armed among equals,
    from a heap of the timers armed, so a wake does not read every pending
    timer; a timer that fired or was cancelled keeps its entry until it reaches
    the top, and the heap empties once no timer is pending.
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
        # While the driver waits on the clock: the time it waits for, and the wait.
        self._wait: tuple[float, asyncio.Future] | None = None
        # (at, arming order, timer) of every timer armed, a heap.
        self._queue: list[tuple[float, int, _Timer]] = []
        self._armed = itertools.count()

    def schedule_later(self, delay_sec, coro, *, group_id=None):
        if delay_sec <= 0:
            return self.execute_async(coro)
        timer = _Timer(self._clock.now() + delay_sec, self._wake)
        handle_id = self._track_handle_and_return_id(timer, coro, group_id=group_id)
        heapq.heappush(self._queue, (timer.at, next(self._armed), timer))
        if self._driver is None or self._driver.done():
            self._driver = self._loop.create_task(self._drive())
        self._wake(timer.at)
        return handle_id

    def _wake(self, before: float = -math.inf) -> None:
        """Withdraw the driver's clock wait if it waits for a time past ``before``
        (any wait, by default), so the driver reads its timers again."""
        if self._wait is not None and self._wait[0] > before:
            self._clock.withdraw(self._wait[1])
            self._wait = None

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
        self._queue = [(t.at, armed, t) for _, armed, t in self._queue]
        heapq.heapify(self._queue)
        self._wake(self._clock.now() + max_delay_sec)
        return shift

    def _earliest(self) -> tuple:
        """The pending ``(timer, coro)`` due first, and among equals the first armed."""
        while True:
            _, _, timer = self._queue[0]
            # The heap keeps `timer` alive, so no other object holds its id.
            entry = self._handles.get(id(timer))
            if entry is not None:
                return entry
            heapq.heappop(self._queue)

    async def _drive(self) -> None:
        # Its clock wait is withdrawn, never cancelled: the clock holds that wait.
        while self._handles:
            timer, coro = self._earliest()
            if timer.at > self._clock.now():
                self._wait = (timer.at, self._clock.wait(timer.at))
                await self._wait[1]
                self._wait = None
            else:
                self._safe_callback([timer], coro)
            await asyncio.sleep(0)
        self._queue.clear()
