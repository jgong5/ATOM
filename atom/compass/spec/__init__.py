# SPDX-License-Identifier: MIT
"""The machine specification: what Compass is told about the machine it simulates.

Compute capability, memory size, bandwidth and interconnect are configured, not
read from a device runtime, so this artifact is load-bearing rather than
convenient -- a simulated run is only as true as the document that describes the
hardware it is pretending to be.

One rule decides what may appear in it. The spec describes the **machine**;
ATOM's config describes the **deployment**; anything ATOM could configure
belongs to ATOM, and Compass reads it from there. A spec that carried a
tensor-parallel width or a block size would silently contradict the engine it
claims to describe, because the engine was launched with its own.

The rule is kept by the schema being closed rather than by a list of exclusions,
so it holds for knobs nobody has thought of yet. Everything else here is a
refusal: a missing runtime constant names itself and the width nobody measured,
a spec-peak number without its derate is declined, a stack pin that no longer
matches what is loaded warns and names both versions, and an unmeasured
tokenizer is refused rather than guessed.

Nothing here imports a tensor library, a device runtime or the engine, so a spec
can be authored, read and checked anywhere Python runs. The document is a plain
mapping: turning a file into one, combining two of them, and reporting on one
belong to the tools built over this.
"""

from .fields import DECLARED, SCHEMA, SCHEMA_VERSION, Field, Kind
from .machine import MachineSpec
from .rules import (
    DEPLOYMENT_OWNED,
    FingerprintMismatch,
    Rule,
    SpecRefusal,
    StackMismatch,
)
from .tokenizers import Backend, TokenizerEntry, TokenizerKey, TokenizerTable

__all__ = [
    "DECLARED",
    "DEPLOYMENT_OWNED",
    "SCHEMA",
    "SCHEMA_VERSION",
    "Backend",
    "Field",
    "FingerprintMismatch",
    "Kind",
    "MachineSpec",
    "Rule",
    "SpecRefusal",
    "StackMismatch",
    "TokenizerEntry",
    "TokenizerKey",
    "TokenizerTable",
]
