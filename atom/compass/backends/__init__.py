# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Cost backends: what a simulated step costs, and where that number came from.

The pieces, and the rule each exists to make structural rather than customary:

- `CostBackend` -- `estimate(batch_view) -> StepCost` plus `describe()`. Takes a
  projection of the batch, so nothing here imports the engine.
- `StepCost` / `CostTerm` -- a total that is folded from its parts on every
  read, so it cannot drift from them, and a breakdown that cannot be empty.
- `Provenance` / `Species` / `Refusal` -- every number says how it was obtained,
  with no default that would let one avoid saying.
- `Resolver` / `CostSource` -- sources consulted in a fixed order, with each
  fall-through recorded on the answer rather than inferred from it.
- `ProvenanceMix` -- the run-level mixture, and refusals counted by number, by
  fraction of steps and by fraction of predicted seconds.
"""

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.cost import CostTerm, ProvenanceMix, StepCost, fold_seconds
from atom.compass.backends.ladder import (
    CostRefused,
    CostSource,
    Resolution,
    Resolver,
)
from atom.compass.backends.provenance import Provenance, Refusal, Species

__all__ = [
    "CostBackend",
    "CostRefused",
    "CostSource",
    "CostTerm",
    "Provenance",
    "ProvenanceMix",
    "Refusal",
    "Resolution",
    "Resolver",
    "Species",
    "StepCost",
    "Tier",
    "fold_seconds",
]
