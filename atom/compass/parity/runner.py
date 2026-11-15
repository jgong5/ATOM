# SPDX-License-Identifier: MIT
"""ATOM's own runner, recording each step: what the frontend names on a real run
with the parity record's variable set.

Kept apart from the package because importing `ModelRunner` needs a driver.
"""

import time

import torch

from atom.compass.parity import (
    StepRecording,
    blocked,
    key_memory_formats_by_name,
    record_step,
)
from atom.compass.run import host_key
from atom.model_engine.model_runner import ModelRunner, RapidServeModelRunner

key_memory_formats_by_name()

#: Spin cycles timed once, to turn a preload's nanoseconds into cycles.
CALIBRATION_CYCLES = 10_000_000


class _Recording(StepRecording):
    """`StepRecording` on this worker's clocks, with the host key read at
    startup, the blocking-call tool where HIP's initialization loaded it, and
    the spin a gated record and the bench queue before a forward."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Read once at startup, so no forward pays for it.
        self.host_key = host_key(self.config)
        self._blocked = blocked.loaded()

    @staticmethod
    def step_clock_ns() -> int:
        """The worker's monotonic clock: a real step's host timestamps."""
        return time.monotonic_ns()

    @staticmethod
    def device_event() -> torch.cuda.Event:
        """A timing event recorded on the current stream: a real step's device stamps."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def calibrate_preload(self) -> None:
        """Time `CALIBRATION_CYCLES` of the spin `preload` queues, once: its
        cycles per nanosecond on this worker's device."""
        # The first launch also loads the kernel, which takes longer than it spins.
        torch.cuda._sleep(1)
        start = self.device_event()
        torch.cuda._sleep(CALIBRATION_CYCLES)
        end = self.device_event()
        end.synchronize()
        self.cycles_per_ns = CALIBRATION_CYCLES / (start.elapsed_time(end) * 1e6)

    def preload(self, ns: int) -> None:
        """Queue about `ns` of device work on the current stream: a spin kernel."""
        torch.cuda._sleep(round(ns * self.cycles_per_ns))


class RecordingModelRunner(_Recording, ModelRunner):
    """`ModelRunner`, unchanged, with each forward call recorded around it."""


class RecordingRapidServeModelRunner(_Recording, RapidServeModelRunner):
    """`RapidServeModelRunner`, unchanged, with each forward call recorded
    around it, the prefill engine's `prefill_forward` among them."""

    def prefill_forward(self, batch):
        if self._recording:
            return record_step(self, batch, super().prefill_forward)
        return super().prefill_forward(batch)
