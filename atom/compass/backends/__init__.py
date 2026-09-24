# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Cost backends: what a simulated step costs, and where that number came from.

The pieces, and the rule each exists to make structural rather than customary:

- `CostBackend` -- `estimate(batch_view) -> StepCost` plus `describe()`. Takes a
  projection of the batch, so nothing here imports the engine.
- `StepCost` / `CostTerm` -- a total folded from its parts on every read, with
  the parts re-checked on every read, so it cannot drift from them; a breakdown
  that cannot be empty; and no subclass that can shadow either.
- `Provenance` / `Species` / `Refusal` -- every number says how it was obtained
  and what produced it, with no default that would let one avoid saying.
- `Resolver` / `CostSource` -- sources consulted in a fixed order, with every
  refusal on the way down recorded on the answer rather than inferred from it.
- `KvGeometry` / `Parallelism` -- what a KV block costs and how each parallel
  width divides it, so ATOM's own pool sizing and block manager run for real
  against a stand-in model.
- `ProvenanceMix` -- the run-level mixture, and refusals counted by number, by
  fraction of steps and by fraction of predicted seconds. A run with nothing in
  it has no fractions and says so, rather than reporting a reassuring zero.
"""

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.cost import (
    CostTerm,
    ProvenanceMix,
    StepCost,
    fold_seconds,
    fold_step,
)
from atom.compass.backends.geometry import KvGeometry, Parallelism
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
    "KvGeometry",
    "Parallelism",
    "Provenance",
    "ProvenanceMix",
    "Refusal",
    "Resolution",
    "Resolver",
    "Species",
    "StepCost",
    "Tier",
    "fold_seconds",
    "fold_step",
]
