# SPDX-License-Identifier: MIT
"""The ``agentic_replay`` timing strategy this package registers."""

from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy

from compass_harness.scheduler import ClockPacedLoopScheduler


class CompassAgenticReplay(AgenticReplayStrategy):
    """Upstream ``agentic_replay``, built only on a clock-paced scheduler.

    The pacing itself is the rebound scheduler's; this class wraps nothing. It
    refuses, by name, the options whose timing the Compass clock does not reach:
    a second live phase runner, the two idle caps (real-clock timers outside
    the scheduler), and the collectors that poll on the wall clock.
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
                    "trace_idle_gap_cap_seconds",
                    getattr(
                        cfg.get_default_dataset(), "trace_idle_gap_cap_seconds", None
                    )
                    is not None,
                ),
                (
                    "system_idle_gap_cap_seconds",
                    self._system_idle_gap_cap_seconds is not None,
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
