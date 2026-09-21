# SPDX-License-Identifier: MIT
"""Shapes, and the canonical form a size that is not known yet is held in.

A traced operator records the shapes it was called with. Recording them as
plain integers would tie the record to the one batch it was traced at, so a
dimension here is either a concrete `int` or a `SymDim` standing for a size that
is still open -- a token count, a batch size, a context length, a parallelism
degree.

**A live symbolic size is canonicalised on the way in and never stored.** The
object a tracer holds for an open size is not a value: on torch 2.10 it cannot
be hashed at all -- `hash()` on one raises `TypeError: unhashable type:
non-nested SymInt`, for a backed symbol and an unbacked one alike -- and
comparing it or calling `int()` on it is how a guard gets installed, which would
mean that recording or reading a shape changes what the trace says about where
it is valid. A node has to be hashable and comparable, because deciding that two
blocks are interchangeable is exactly how repetition is found. Those two
requirements are irreconcilable for the live object, so `as_dim` renders it once
into a `SymDim` and the node holds that.

Rendering is the one operation that is safe: measured on torch
2.10.0+rocm7.2.4, `str()` and `repr()` of a symbol left the shape environment's
guard list untouched, for `s26`, for `2*s26 + 1` and for the unbacked `u0`.
Nothing in this module compares a dimension it has not already proved to be a
concrete `int`, and nothing calls `int()` on one; the bound below is reached only
after `isinstance(dim, int)` has settled that the dimension is a number.

**What a symbolic size is, is still not decided here.** This module imports
neither a tensor library nor a symbolic-algebra library, so the data model stays
constructible and testable on a machine with no device runtime. It needs only
that an open size can render itself, and the text it renders to is its identity
from then on. Rendering has to be the tracer's one convention: two spellings of
one expression are two dimensions here.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: One dimension: a concrete `int`, or a `SymDim` standing for an open size.
SymExpr = Any

#: The shape of one tensor: its dimensions, in order.
Shape = tuple

_REFUSED_DIM_TYPES = (bool, float, complex, str, bytes, bytearray)


@dataclass(frozen=True, slots=True)
class SymDim:
    """A size that is not known yet, held as the text its expression renders to.

    The text is the whole of the identity. Two dimensions are the same size when
    they render the same, which makes a node hashable and comparable without any
    operation ever reaching the live symbolic object.
    """

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(
                f"a symbolic dimension renders to a str, got {type(self.text).__name__}"
            )
        if not self.text.strip() or "\n" in self.text:
            raise ValueError(
                f"a symbolic dimension needs a rendering to be identified by, "
                f"on one line; got {self.text!r}"
            )

    @classmethod
    def of(cls, expr: object) -> "SymDim":
        """Canonicalise a live symbolic size by rendering it, and nothing else."""
        return cls(str(expr))

    def __str__(self) -> str:
        return self.text


def is_symbolic(dim: SymExpr) -> bool:
    """True when `dim` stands for a size that is not known yet."""
    return not isinstance(dim, int)


def as_dim(dim: SymExpr) -> SymExpr:
    """Return `dim` as a dimension, canonicalising a live symbolic size."""
    if dim is None or isinstance(dim, _REFUSED_DIM_TYPES):
        raise TypeError(
            "a dimension is a concrete int or a symbolic size, got "
            f"{type(dim).__name__}: {dim!r}"
        )
    if isinstance(dim, int):
        if dim < 0:
            raise ValueError(f"a concrete dimension cannot be negative, got {dim}")
        return dim
    if isinstance(dim, SymDim):
        return dim
    return SymDim.of(dim)


def as_shape(dims: Iterable[SymExpr]) -> Shape:
    """Return one tensor's shape as a tuple of dimensions."""
    if isinstance(dims, (str, bytes)) or not isinstance(dims, Iterable):
        raise TypeError(
            f"a shape is an iterable of dimensions, got {type(dims).__name__}"
        )
    return tuple(as_dim(dim) for dim in dims)


def as_shapes(shapes: Iterable[Iterable[SymExpr]]) -> tuple[Shape, ...]:
    """Return one shape per operand, in operand order.

    The grammar writes an operator's inputs as a list of expressions, which can
    be read as a single shape. It is one shape per operand: an operator with two
    tensor inputs has two of them, and flattening the two into one list would
    lose which dimension belonged to which operand.
    """
    if isinstance(shapes, (str, bytes)) or not isinstance(shapes, Iterable):
        raise TypeError(f"expected one shape per operand, got {type(shapes).__name__}")
    return tuple(as_shape(shape) for shape in shapes)
