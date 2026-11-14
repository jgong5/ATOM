# SPDX-License-Identifier: MIT
"""The machine specification: what Compass is told about the machine it simulates.

Compute, memory, bandwidth and interconnect are configured, not read from a
device runtime. The spec describes the machine; ATOM's config describes the
deployment, and the closed schema refuses any key it does not declare. An
engine start's memory readings are not the spec's: `compass measure memory`
publishes them per configuration as a `memory_readings` entry.

`validate` reports every refusal of a document. A run reads one with
`MachineSpec`, which states each field's origin. `across_ranks` is
`compass measure memory`'s spread check.

Nothing here imports a tensor library, a device runtime or the engine.
"""

from .machine import MachineSpec
from .ranks import across_ranks
from .rules import SpecRefusal
from .validate import validate

__all__ = [
    "MachineSpec",
    "SpecRefusal",
    "across_ranks",
    "validate",
]
