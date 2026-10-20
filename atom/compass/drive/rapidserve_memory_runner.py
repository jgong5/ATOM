# SPDX-License-Identifier: MIT
"""The runner `compass measure memory` starts a RapidServe engine with.

Apart from `memory_runner` because importing `RapidServeModelRunner` beside
`ModelRunner` is what a RapidServe start needs and a plain one does not.
"""

from atom.compass.drive.memory_runner import MemoryReadingModelRunner
from atom.model_engine.model_runner import RapidServeModelRunner


class MemoryReadingRapidServeModelRunner(
    MemoryReadingModelRunner, RapidServeModelRunner
):
    """`RapidServeModelRunner`, unchanged, read as `MemoryReadingModelRunner`
    reads a plain one. Prefill writes its readings, sized with the decode
    process already on the card; decode owns no pool and writes none."""

    def _write(self, pool) -> None:
        if not self.config.disagg_is_decode:
            super()._write(pool)
