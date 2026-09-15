"""Host-return and device-queue times for the native PP1 deferred runner.

ModelRunner gates pinned staging reuse on the previous preparation event,
then queues the current forward. Sampling drains the previous sampled output;
middle prefill chunks do not. Internal synchronizations can keep the host in
the current forward until a priced stream prefix completes.

The preparation offset is the source-measured GPU preparation/idle remainder,
used as an approximation to the staging-H2D boundary. CPU enqueue/IPC time is
not separately calibrated here and is assumed overlapped with device work.
These are modelling assumptions, not measured host-return coefficients.
"""

from dataclasses import dataclass
import math


@dataclass
class ForwardTimeline:
    device_end: float | None = None
    staging_ready: float | None = None
    sample_ready: float | None = None

    def submit(self, host_now: float, seconds: float, preparation: float,
               blocking_prefix: float, produces_output: bool) -> dict:
        if (not all(math.isfinite(x) for x in
                    (host_now, seconds, preparation, blocking_prefix))
                or not 0.0 <= preparation <= seconds
                or not 0.0 <= blocking_prefix <= seconds):
            raise ValueError("forward timing boundaries must be finite and within the step")

        # Scheduling already happened. Arrivals during this wait may influence
        # the NEXT schedule, never the batch just selected.
        after_staging_wait = max(host_now, self.staging_ready
                                if self.staging_ready is not None else host_now)
        device_start = max(after_staging_wait, self.device_end
                           if self.device_end is not None else host_now)
        device_end = device_start + seconds
        host_return = after_staging_wait
        if blocking_prefix > 0.0:
            host_return = max(host_return, device_start + blocking_prefix)
        previous_sample_ready = self.sample_ready
        if produces_output:
            if previous_sample_ready is not None:
                host_return = max(host_return, previous_sample_ready)
            self.sample_ready = device_end
        self.staging_ready = device_start + preparation
        self.device_end = device_end
        return {
            "host_started_at": host_now,
            "host_after_staging_wait": after_staging_wait,
            "host_returned_at": host_return,
            "device_started_at": device_start,
            "device_ended_at": device_end,
            "staging_ready_at": self.staging_ready,
            "previous_sample_ready_at": previous_sample_ready,
            "produces_output": produces_output,
            "blocking_prefix_seconds": blocking_prefix,
            "preparation_seconds": preparation,
        }
