# SPDX-License-Identifier: MIT
"""Simulated KV transfer: the engine's connector contract, on a handed-in clock.

Two pieces, split by what each one may depend on.

`transfer` is arithmetic -- a latency, a derated bandwidth and an exact byte
count -- and imports nothing from the engine, so a transfer can be priced
anywhere Python runs.

`connector` implements the engine's connector interface over that price and is
what the connector factory builds under the name `compass`. It reads the clock
the harness binds and never any other.
"""

from atom.compass.kv.connector import (
    CLOCK_KEY,
    TRANSFER_KEY,
    SimulatedKVConnector,
    SimulatedKVConnectorScheduler,
    UnboundSeam,
)
from atom.compass.kv.transfer import Scope, TransferModel

__all__ = [
    "CLOCK_KEY",
    "TRANSFER_KEY",
    "Scope",
    "SimulatedKVConnector",
    "SimulatedKVConnectorScheduler",
    "TransferModel",
    "UnboundSeam",
]
