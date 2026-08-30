"""Compare two op graphs.

A graph is meant to be a faithful record of what a rank did, so a graph derived
on meta and one captured on real hardware for the same batch should be the same
graph. Checking that is what turns meta-derivation from an assumption into a
validated mechanism: if they agree, a graph can be derived for a configuration
nobody has run and trusted.

Comparison is on structure — operator order, shapes, dtypes — never on values,
which meta does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from atom.compass.core.graph import OpGraph, OpSpec

__all__ = ["OpMismatch", "GraphDiff", "diff_graphs"]


@dataclass(frozen=True)
class OpMismatch:
    """One position where two graphs disagree."""

    index: int
    field: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"op[{self.index}] {self.field}: {self.left!r} != {self.right!r}"


@dataclass
class GraphDiff:
    """The outcome of comparing two graphs."""

    left_len: int = 0
    right_len: int = 0
    mismatches: list[OpMismatch] = field(default_factory=list)
    left_only: list[str] = field(default_factory=list)
    right_only: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return (
            self.left_len == self.right_len
            and not self.mismatches
            and not self.left_only
            and not self.right_only
        )

    def report(self, left_name: str = "meta", right_name: str = "captured",
               limit: int = 12) -> str:
        lines = [
            f"  {left_name:<10} : {self.left_len} operators",
            f"  {right_name:<10} : {self.right_len} operators",
        ]
        if self.identical:
            lines.append("  result     : IDENTICAL — meta reproduces the captured graph")
            return "\n".join(lines)

        lines.append(f"  result     : {len(self.mismatches)} mismatch(es)")
        if self.left_only:
            lines.append(f"  only in {left_name}: {', '.join(self.left_only[:8])}")
        if self.right_only:
            lines.append(f"  only in {right_name}: {', '.join(self.right_only[:8])}")
        for m in self.mismatches[:limit]:
            lines.append(f"    {m}")
        if len(self.mismatches) > limit:
            lines.append(f"    ... and {len(self.mismatches) - limit} more")
        return "\n".join(lines)


def _compare(index: int, a: OpSpec, b: OpSpec) -> list[OpMismatch]:
    out = []
    for attr in ("name", "input_shapes", "output_shapes", "dtypes", "group"):
        left, right = getattr(a, attr), getattr(b, attr)
        if left != right:
            out.append(OpMismatch(index=index, field=attr, left=left, right=right))
    return out


def diff_graphs(left: OpGraph, right: OpGraph,
                compare_dtypes: bool = True) -> GraphDiff:
    """Compare two graphs position by position.

    Stops describing a position after its first disagreement, since one
    divergence usually shifts everything after it and listing the shift adds
    nothing.
    """
    result = GraphDiff(left_len=len(left), right_len=len(right))

    left_names = set(left.op_names())
    right_names = set(right.op_names())
    result.left_only = sorted(left_names - right_names)
    result.right_only = sorted(right_names - left_names)

    for i, (a, b) in enumerate(zip(left.ops, right.ops)):
        found = _compare(i, a, b)
        if not compare_dtypes:
            found = [m for m in found if m.field != "dtypes"]
        if found:
            result.mismatches.append(found[0])
    return result
