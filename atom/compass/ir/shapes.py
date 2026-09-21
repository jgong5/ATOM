# SPDX-License-Identifier: MIT
"""Shapes, and what counts as a size that is not known yet.

A traced operator records the shapes it was called with. Recording them as
plain integers would tie the record to the one batch it was traced at, so a
dimension here is either a concrete `int` or an object standing for a size that
is still open -- a token count, a batch size, a context length, a parallelism
degree.

**Nothing in this module compares a dimension it has not already proved to be
concrete, and nothing calls `int()` on one.** Both of those install a guard on a
symbolic size, which would mean the act of recording or checking a shape changes
what the trace says about where it is valid. The bound on a concrete dimension
below is therefore reached only after `isinstance(dim, int)` has settled that
the dimension is a number.

**What a symbolic dimension is, is deliberately not decided here.** This module
imports neither a tensor library nor a symbolic-algebra library, so the data
model stays constructible and testable on a machine with no device runtime, and
so the class that carries a symbol stays the capture side's choice rather than a
commitment made here. A dimension is accepted when it is hashable -- nodes are
values and have to be comparable and keyable -- and when it is not one of the
types that only ever arrives by mistake: a `bool` (which is an `int`, so `True`
would otherwise pass as the size 1), a `float`, a `str`, or `None`.
"""

from collections.abc import Iterable
from typing import Any

#: One dimension: a concrete `int`, or an object standing for an open size.
#: Left as `Any` on purpose -- the module docstring says why the symbolic half
#: of that union is not named here.
SymExpr = Any

#: The shape of one tensor: its dimensions, in order.
Shape = tuple

_REFUSED_DIM_TYPES = (bool, float, complex, str, bytes, bytearray)


def is_symbolic(dim: SymExpr) -> bool:
    """True when `dim` stands for a size that is not known yet."""
    return not isinstance(dim, int)


def as_dim(dim: SymExpr) -> SymExpr:
    """Return `dim` as a dimension, refusing what cannot be one."""
    if dim is None or isinstance(dim, _REFUSED_DIM_TYPES):
        raise TypeError(
            "a dimension is a concrete int or an object standing for an open "
            f"size, got {type(dim).__name__}: {dim!r}"
        )
    if isinstance(dim, int):
        if dim < 0:
            raise ValueError(f"a concrete dimension cannot be negative, got {dim}")
        return dim
    try:
        hash(dim)
    except TypeError:
        raise TypeError(
            "a symbolic dimension must be hashable, because the node holding it "
            f"is a value that is compared and keyed; {type(dim).__name__} is not"
        ) from None
    return dim


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
