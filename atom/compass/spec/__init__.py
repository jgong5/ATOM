# SPDX-License-Identifier: MIT
"""The machine specification: what Compass is told about the machine it simulates.

Compute, memory, bandwidth and interconnect are configured, not read from a
device runtime. The spec describes the machine; ATOM's config describes the
deployment, and the closed schema refuses any key it does not declare. An
engine start's memory readings are not the spec's: `compass measure memory`
publishes them per configuration as a `memory_readings` entry.

The verbs: `merge` combines fragments and refuses ones that describe two
machines, `validate` reports every refusal of the merged document, and
`explain` takes a field or block back to its values and their origins. A run
reads the result with `MachineSpec`. `across_ranks` is `compass measure
memory`'s spread check.

Nothing here imports a tensor library, a device runtime or the engine.
"""

from .explain import explain
from .machine import MachineSpec
from .merge import Fragment, merge
from .ranks import across_ranks
from .rules import SpecRefusal
from .validate import validate

__all__ = [
    "Fragment",
    "MachineSpec",
    "SpecRefusal",
    "across_ranks",
    "explain",
    "merge",
    "validate",
]
