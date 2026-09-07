"""**empirical/measured** -- a configuration's memory, from an artifact.

Sizing a configuration needs five numbers off a device. That is why a
configuration has to exist before it can be sized, and why the 27B at TP=1 could
not be evaluated at all: it failed to fit before any timing could be taken.

The five are recorded by `--compass-memory-out`. This reads them back, so a
configuration can be sized on a box that could not hold it. Still
`empirical/measured` -- the same measurement from a different source, not a
different kind of number. `analytical`, which derives the terms instead of
reading them, is the next species and is what removes the need for the
configuration to have run anywhere.

**What must not be reused.** The budget is

    available_for_kv = min(total * utilization - overheads, free)

and `free` is a property of *the box at that moment* -- what the neighbours left
-- where every other term is a property of the configuration. A record whose
`free` was the binding term describes an accident of scheduling, so it is
refused rather than replayed. Both records taken so far were budget-bound, which
is why they are reusable, and the check exists so that stops being luck.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = ["MemoryReadings", "RecordedMemory", "SAFETY_FRACTION"]

#: The engine holds this back before anything else (`model_runner.py`).
SAFETY_FRACTION = 0.02


@dataclass(frozen=True)
class MemoryReadings:
    """The five device readings a KV budget is computed from."""

    total: int
    free: int
    peak_torch: int
    non_torch: int
    cudagraph_overhead: int
    #: Optional, and absent from records written before the terms were split.
    #: `peak_torch` is weights + persistent buffers + peak activations; these
    #: two are where one ends and the next begins. Nothing in the budget
    #: arithmetic reads them -- they exist so each term can be checked alone.
    weights_torch: Optional[int] = None
    parameter_bytes: Optional[int] = None
    current_torch: Optional[int] = None

    def free_was_binding(self, utilization: float) -> bool:
        """Whether the box, rather than the configuration, set the budget.

        Computed without the engine's extra reserve, which is not recorded and
        only ever shrinks the budget. Leaving it out overstates the budget and
        so overstates how often `free` binds -- erring towards refusing a record
        that might have been reusable, rather than reusing one that was not.
        """
        overheads = (self.peak_torch + self.non_torch
                     + self.cudagraph_overhead + int(self.total * SAFETY_FRACTION))
        return self.free < int(self.total * utilization) - overheads


def _optional_int(value: Any) -> Optional[int]:
    """An int, or None -- a reading a record predating the split has not got."""
    return None if value is None else int(value)


def _key(config: Mapping[str, Any]) -> tuple:
    """What makes two configurations the same for memory.

    Everything the readings depend on, and nothing else. The topology and this
    rank's place in it are part of it: at TP=2 the weights are halved but the
    collective buffers are not, so rank 0 of a pair is not rank 0 of a four.
    """
    return (
        str(config.get("model")),
        float(config.get("gpu_memory_utilization", 0.0)),
        int(config.get("max_num_seqs", 0)),
        int(config.get("max_model_len", 0)),
        str(config.get("kv_cache_dtype")),
        int(config.get("block_size", 0)),
        tuple(sorted((config.get("topology") or {}).items())),
        tuple(sorted((config.get("rank_coords") or {}).items())),
    )


class RecordedMemory:
    """Readings for a configuration, from artifacts an earlier run wrote."""

    def __init__(self, paths: list[str]) -> None:
        self.records: dict[tuple, dict] = {}
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            self.records[_key(blob.get("config") or {})] = blob

    def refusal(self, config: Mapping[str, Any]) -> Optional[str]:
        """Why this configuration cannot be sized from a record, or None."""
        blob = self.records.get(_key(config))
        if blob is None:
            return "no record for this configuration"
        readings = self.readings_for(config)
        if readings is None:
            return "record is missing readings"
        utilization = float(config.get("gpu_memory_utilization", 0.0))
        if readings.free_was_binding(utilization):
            return ("recorded `free` was the binding term, so the record "
                    "describes what the neighbours left rather than what the "
                    "configuration needs")
        return None

    def readings_for(self, config: Mapping[str, Any]) -> Optional[MemoryReadings]:
        blob = self.records.get(_key(config))
        if blob is None:
            return None
        got = blob.get("readings") or {}
        try:
            return MemoryReadings(
                total=int(got["total"]), free=int(got["free"]),
                peak_torch=int(got["peak_torch"]),
                non_torch=int(got["non_torch"]),
                cudagraph_overhead=int(got["cudagraph_overhead"]),
                weights_torch=_optional_int(got.get("weights_torch")),
                parameter_bytes=_optional_int(got.get("parameter_bytes")),
                current_torch=_optional_int(got.get("current_torch")))
        except (KeyError, TypeError, ValueError):
            return None
