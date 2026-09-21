# SPDX-License-Identifier: MIT
"""The cost graph's data model: the shape of a recorded forward pass.

Four regions compose into a tree -- an operator, a sequence, a repeat, an
overlap -- and a graph pairs one of them with a statement of where the record is
valid. Nothing here walks a tree, prices one, builds one from a trace, or
decides whether a repeat was safe to take; those are separate, and each one
reads this.

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
    "is_symbolic",
]
