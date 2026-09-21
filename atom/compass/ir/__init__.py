# SPDX-License-Identifier: MIT
"""The cost graph's data model: the shape of a recorded forward pass.

Four regions compose into a tree -- an operator, a sequence, a repeat, an
overlap -- and a graph pairs one of them with a statement of where the record is
valid. The detector that finds the repetition in a flat block sequence and names
it is here too. Nothing here prices a tree, builds one from a trace, or decides
whether a repeat was safe to take; those are separate, and each one reads this.

Nothing here imports a tensor library, a symbolic-algebra library or a device
runtime, so a graph can be built and checked anywhere.
"""

from .graph import Applicability, Graph
from .nodes import (
    AMBIENT_READINGS,
    ContextRef,
    EqualPrice,
    GroupingEvidence,
    IdenticalStructure,
    IndexBinding,
    JoinPolicy,
    NodeKind,
    Op,
    Par,
    Region,
    Repeat,
    Seq,
)
from .repeats import detect_repeats, signature_of
from .shapes import Shape, SymExpr, as_dim, as_shape, as_shapes, is_symbolic

__all__ = [
    "AMBIENT_READINGS",
    "Applicability",
    "ContextRef",
    "EqualPrice",
    "Graph",
    "GroupingEvidence",
    "IdenticalStructure",
    "IndexBinding",
    "JoinPolicy",
    "NodeKind",
    "Op",
    "Par",
    "Region",
    "Repeat",
    "Seq",
    "Shape",
    "SymExpr",
    "as_dim",
    "as_shape",
    "as_shapes",
    "detect_repeats",
    "is_symbolic",
    "signature_of",
]
