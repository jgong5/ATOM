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
mapping, and turning a file into one belongs to the tools built over this, which
can declare the parser they need.

Three verbs work over that mapping. `merge` combines the fragments separate
probes produce and refuses the ones that name different machines, which
is the hazard the schema cannot see: two tokenizers measured on two machines
contradict nothing in their shape. `validate` reports everything a document is
missing or inconsistent about, rather than completing it. `explain` takes a
predicted quantity back to the spec fields under it, which is what makes echoing
the whole spec into every artifact worth anything. `across_ranks` reduces the
one reading that is taken per rank, keeping the smallest and reporting how far
the ranks disagreed.
"""

from .explain import QUANTITIES, Basis, Contribution, explain
from .fields import DECLARED, SCHEMA, SCHEMA_VERSION, Field, Kind
from .machine import MachineSpec
from .merge import Fragment, Merge, merge
from .ranks import SPREAD_LIMIT, RankSpread, across_ranks
from .rules import (
    DEPLOYMENT_OWNED,
    FingerprintMismatch,
    Rule,
    SpecRefusal,
    StackMismatch,
)
from .tokenizers import Backend, TokenizerEntry, TokenizerKey, TokenizerTable
from .validate import Validation, validate

__all__ = [
    "DECLARED",
    "DEPLOYMENT_OWNED",
    "QUANTITIES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SPREAD_LIMIT",
    "Backend",
    "Basis",
    "Contribution",
    "Field",
    "FingerprintMismatch",
    "Fragment",
    "Kind",
    "MachineSpec",
    "Merge",
    "RankSpread",
    "Rule",
    "SpecRefusal",
    "StackMismatch",
    "TokenizerEntry",
    "TokenizerKey",
    "TokenizerTable",
    "Validation",
    "across_ranks",
    "explain",
    "merge",
    "validate",
]
