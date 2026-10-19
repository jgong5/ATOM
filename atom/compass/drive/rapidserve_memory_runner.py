# SPDX-License-Identifier: MIT
"""The runner `compass measure memory` starts a RapidServe engine with.

Apart from `memory_runner` because importing `RapidServeModelRunner` beside
`ModelRunner` is what a RapidServe start needs and a plain one does not.
"""

from pathlib import Path

import torch

from atom.compass.drive.measure_memory import DECODE
from atom.compass.drive.memory_runner import MemoryReadingModelRunner
from atom.model_engine.model_runner import RapidServeModelRunner


class MemoryReadingRapidServeModelRunner(
    MemoryReadingModelRunner, RapidServeModelRunner
):
    """`RapidServeModelRunner`, unchanged, read as `MemoryReadingModelRunner`
    reads a plain one. Prefill writes its readings, sized with the decode
    process already on the card. Decode writes its own in `DECODE`, graph pool
    included: it sizes no pool and warms up nothing, so it reads the device
    once it holds prefill's weights, while prefill waits for it with no pool
    yet, and reads no activations."""

    def import_model_weight_ipc_handles(self, paths) -> bool:
        reply = super().import_model_weight_ipc_handles(paths)
        torch.cuda.reset_peak_memory_stats()
        self._read_device_memory()
        return reply

    def _out(self) -> Path:
        out = super()._out()
        if self.config.disagg_is_decode:
            out /= DECODE
            out.mkdir(exist_ok=True)
        return out
