# SPDX-License-Identifier: MIT
"""The ``agentic_replay`` timing strategy this package registers, and the phase
runner the import hook puts in the orchestrator."""

import asyncio

from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy

from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.timesource import ClockTimer


class CompassAgenticReplay(AgenticReplayStrategy):
    """Upstream ``agentic_replay``, built only on a clock-paced scheduler.

    The pacing itself is the rebound scheduler's, and the system idle cap
    measures idle on the rebound ``time``; this class only moves the idle
    watchdog onto the clock. It refuses, by name, the options whose timing the
    Compass clock does not reach: a second live phase runner, the prefill slot
    released by a first token the traffic LP does not hold, cancellation on a
    wall-clock timer, and the collectors that poll on the wall clock. It also
    refuses the accelerated warmup bounded by a duration, which no client run
    here has checked against a real one; the one bounded by a request count
    per lane runs, its drain on the clock (``ClockPhaseRunner``).
    """

    def __init__(self, *, config, scheduler, run=None, **kwargs) -> None:
        if not isinstance(scheduler, ClockPacedLoopScheduler):
            raise TypeError(
                f"Compass clock not installed: scheduler is {type(scheduler).__name__}. "
                "The bootstrap did not run before PhaseRunner was constructed."
            )
        if run is None:
            raise RuntimeError(
                "CompassAgenticReplay needs the benchmark run to check its options."
            )
        super().__init__(config=config, scheduler=scheduler, run=run, **kwargs)
        cfg = run.cfg
        refused = [
            name
            for name, on in (
                ("seamless", config.seamless),
                (
                    "prefill concurrency (--prefill-concurrency)",
                    config.prefill_concurrency is not None,
                ),
                (
                    "request cancellation (--request-cancellation-rate)",
                    bool(config.request_cancellation.rate),
                ),
                (
                    "accelerated warmup (--agentic-cache-warmup-duration)",
                    self._cache_warmup_duration is not None,
                ),
                (
                    "server metrics (--no-server-metrics)",
                    not cfg.server_metrics_disabled,
                ),
                ("GPU telemetry (--no-gpu-telemetry)", not cfg.gpu_telemetry_disabled),
            )
            if on
        ]
        if refused:
            raise ValueError(
                f"Compass does not pace these yet; turn them off: {', '.join(refused)}"
            )

    def _arm_system_idle_watchdog(self, delay_seconds: float) -> None:
        """Upstream's idle recheck, on the Compass clock instead of ``loop.call_later``."""
        if self._system_idle_watchdog is None:
            self._system_idle_watchdog = ClockTimer(
                delay_seconds, self._run_system_idle_watchdog
            )


async def _clock_sleep(delay: float) -> None:
    """``asyncio.sleep`` on the Compass clock."""
    woken = asyncio.get_running_loop().create_future()
    timer = ClockTimer(delay, lambda: woken.done() or woken.set_result(None))
    try:
        await woken
    finally:
        timer.cancel()


class ClockPhaseRunner(PhaseRunner):
    """``PhaseRunner`` whose sending and grace deadlines fall on the Compass clock.

    So does the drain after the cancel, through ``ClockAsyncio``: it ends early
    only once no branch work is pending, so a run whose subagent children are
    still tracked drains for the full timeout, real or simulated. So does the
    accelerated warmup's drain, which polls for its last request's return.
    """

    async def _wait_for_accelerated_warmup_wire_drain(self) -> None:
        """Upstream's poll for the warmup's requests to return, each 0.1 s
        sleep on the Compass clock: on the wall clock, the clock would reach
        the drain's grace deadline while the poll sleeps."""
        while self._progress.in_flight > 0:
            await _clock_sleep(0.1)

    async def _wait_for_event_with_timeout(
        self, *, name, event, timeout, task_to_cancel, set_event_on_timeout=False
    ) -> bool:
        if timeout is None or timeout <= 0:
            return await super()._wait_for_event_with_timeout(
                name=name,
                event=event,
                timeout=timeout,
                task_to_cancel=task_to_cancel,
                set_event_on_timeout=set_event_on_timeout,
            )
        if event.is_set():
            return False
        self.info(f"Waiting for event '{name}' with timeout of {timeout}s on the clock")
        # One future both sides resolve in the task that sets the event or fires
        # the deadline, so the traffic LP's hold sees this task ready at once and
        # never grants a deadline the event has already made moot.
        wake = asyncio.get_running_loop().create_future()
        # ponytail: Event._waiters is private; Event.set resolves each of them.
        event._waiters.append(wake)
        deadline = ClockTimer(
            timeout, lambda: wake.done() or wake.set_result(None), phase_deadline=True
        )
        try:
            await wake
        finally:
            deadline.cancel()
            if wake in event._waiters:
                event._waiters.remove(wake)
        if event.is_set():
            return False
        self.info(f"Timeout of {timeout}s elapsed for event '{name}'")
        if set_event_on_timeout:
            event.set()
        if task_to_cancel:
            task_to_cancel.cancel()
        return True
