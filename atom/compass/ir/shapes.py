# SPDX-License-Identifier: MIT
"""Shapes, and the canonical form a size that is not known yet is held in.

A traced operator records the shapes it was called with. Recording them as
plain integers would tie the record to the one batch it was traced at, so a
dimension here is either a concrete `int` or a `SymDim` standing for a size that
is still open -- a token count, a batch size, a context length, a parallelism
degree.

**Nothing is accepted because of what it can do; only because of what it is.**
Every check below is an `isinstance`, and no value is hashed, compared,
converted or rendered to decide whether it is acceptable. Asking a live
symbolic object any question is how a guard gets installed, and the question
that looks safest is the one that leaks: a hashability test refuses a `SymInt`
loudly, accepts a `SymBool` and a `SymFloat` silently, and raises
`GuardOnDataDependentSymNode` on an unbacked one. Measured on torch
2.10.0+rocm7.2.4, `isinstance` against `bool`, `int`, `float`, `str`, `bytes`
and `tuple` answers False for all three symbolic types and leaves the guard
list untouched.

**A live symbolic size is canonicalised by the caller and never stored.** It
cannot be held as itself: on torch 2.10 `hash()` on one raises `TypeError:
unhashable type: non-nested SymInt`, backed and unbacked alike, while a node has
to be hashable and comparable because deciding that two blocks are
interchangeable is how repetition gets found. So the capture side calls
`SymDim.of(expr, scope)`, which renders the expression once -- the one safe
operation, measured: `str()` and `repr()` of `s26`, of `2*s26 + 1` and of `u0`
left the guard list untouched -- and the node holds the rendering. `as_dim`
takes an `int` or a `SymDim` and nothing else, so the canonicalisation is an
act the caller performs rather than one that happens to whatever is passed.

**A rendering is read against one capture's symbol table, which is why a
`SymDim` carries its scope.** A shape environment numbers symbols per capture,
so two unrelated traces both produce `s26` and both produce `u0`. Comparing a
body recorded in one against a body recorded in another is exactly what finding
repetition and validating a grouping do, and on the text alone those two
comparisons would silently agree. The scope names the capture; two sizes from
different captures are different sizes even when they render the same.

**What a symbolic size is, is still not decided here.** This module imports
neither a tensor library nor a symbolic-algebra library, so the data model stays
constructible and testable on a machine with no device runtime. It needs only
that an open size can render itself, and the text it renders to is its identity
within its scope from then on.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: One dimension: a concrete `int`, or a `SymDim` standing for an open size.
SymExpr = Any

#: The shape of one tensor: its dimensions, in order.
Shape = tuple


@dataclass(frozen=True, slots=True)
class SymDim:
    """A size that is not known yet: the text it renders to, and whose text it is.

    The pair is the whole of the identity. Two dimensions are the same size when
    they come from the same capture and render the same, which makes a node
    hashable and comparable without any operation ever reaching the live
    symbolic object.
    """

    text: str
    scope: str

    def __post_init__(self) -> None:
        for name in ("text", "scope"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} is a str, got {type(value).__name__}; render a live "
                    "symbolic size with SymDim.of(expr, scope)"
                )
        if not self.text.strip() or "\n" in self.text:
            raise ValueError(
                "a symbolic size needs a rendering to be identified by, on one "
                f"line; got {self.text!r}"
            )
        if not self.scope.strip():
            raise ValueError(
                "a symbolic size names the capture its rendering is read "
                "against; symbols are numbered per capture, so a rendering with "
                "no capture would match an unrelated symbol that spells the same"
            )
        if _is_a_plain_number(self.text):
            raise ValueError(
                f"{self.text!r} is a number, so it is a size that is known and "
                "belongs in the shape as an int. Held here it would compare "
                "unequal to that int and the one shape would become two nodes. "
                "A width from another library is converted, not rendered."
            )
        if self.text.startswith("<") and self.text.endswith(">"):
            raise ValueError(
                f"{self.text!r} is how an object renders when it has no "
                "rendering of its own, and usually carries a memory address, so "
                "it names nothing that could be compared across two runs"
            )

    @classmethod
    def of(cls, expr: object, scope: str) -> "SymDim":
        """Canonicalise a live symbolic size by rendering it, and nothing else."""
        return cls(str(expr), scope)

    def __str__(self) -> str:
        return self.text


def _is_a_plain_number(text: str) -> bool:
    """True when the text is a literal number. Asked of a str, never of a value."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def is_symbolic(dim: SymExpr) -> bool:
    """True when `dim` stands for a size that is not known yet."""
    return not isinstance(dim, int)


def as_dim(dim: SymExpr) -> SymExpr:
    """Return `dim` as a dimension. A concrete `int` or an already-rendered size."""
    if isinstance(dim, SymDim):
        return dim
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(
            "a dimension is a concrete int, or a size that is not known yet "
            "rendered with SymDim.of(expr, scope); got "
            f"{type(dim).__name__}. Nothing is rendered on your behalf: a value "
            "that is not a size renders to something that looks like one."
        )
    if dim < 0:
        raise ValueError(f"a concrete dimension cannot be negative, got {dim}")
    return dim


def as_shape(dims: Sequence[SymExpr]) -> Shape:
    """Return one tensor's shape as a tuple of dimensions."""
    _refuse_unless_ordered(dims, "a shape is a sequence of dimensions")
    return tuple(as_dim(dim) for dim in dims)


def as_shapes(shapes: Sequence[Sequence[SymExpr]]) -> tuple[Shape, ...]:
    """Return one shape per operand, in operand order.

    The grammar writes an operator's inputs as a list of expressions, which can
    be read as a single shape. It is one shape per operand: an operator with two
    tensor inputs has two of them, and flattening the two into one list would
    lose which dimension belonged to which operand.
    """
    _refuse_unless_ordered(shapes, "expected one shape per operand")
    return tuple(as_shape(shape) for shape in shapes)


def _refuse_unless_ordered(value: Any, what: str) -> None:
    """Refuse anything but an ordered sequence, a mapping most of all.

    Iterating a mapping yields its keys, so a shape handed in as a dict would be
    read as its keys and accepted as a shape nobody wrote.
    """
    if isinstance(value, Mapping):
        raise TypeError(
            f"{what}; a mapping iterates as its keys, so what was read would not "
            "be what was written"
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{what}, got {type(value).__name__}")
