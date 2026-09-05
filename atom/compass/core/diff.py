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

__all__ = ["OpMismatch", "GraphDiff", "diff_graphs", "Alignment", "align_graphs"]


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


# -- meta-vs-capture alignment ----------------------------------------------
#
# A positional diff is the right check between two graphs of the same kind. It
# is the wrong check between a derived graph and a captured one, because they
# are not graphs of the same kind: a derived graph is the model body alone,
# while a capture also holds what the runner does around it — batch-metadata
# preparation, the LM head, sampling, device transfers. Compared position by
# position, the two disagree from the first operator onwards while in fact
# agreeing about everything Compass will cost.
#
# So the question to ask of a capture is containment, not equality: does every
# operator the model performs appear in the capture, in the same order, with
# the same shapes? That is what makes derivation trustworthy. The operators the
# capture has in addition are reported rather than ignored — an operator that
# quietly appears only on hardware is exactly the kind of thing this check
# exists to surface.
#
# Both graphs must describe the same batch. Derive at the token count the
# capture used, or the shapes will differ for reasons that mean nothing.


@dataclass
class Alignment:
    """The outcome of matching a derived graph into a captured one."""

    derived_len: int = 0
    captured_len: int = 0
    matched: int = 0
    unmatched: list[tuple[int, OpSpec]] = field(default_factory=list)
    extra: list[OpSpec] = field(default_factory=list)

    @property
    def contained(self) -> bool:
        """Every derived operator was found, in order, with matching shapes."""
        return self.derived_len > 0 and not self.unmatched

    def extra_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for op in self.extra:
            out[op.name] = out.get(op.name, 0) + 1
        return out

    def report(self, limit: int = 8) -> str:
        lines = [
            f"  derived    : {self.derived_len} operators",
            f"  captured   : {self.captured_len} operators",
            f"  matched    : {self.matched} in order, shapes and dtypes equal",
        ]
        if self.unmatched:
            lines.append(f"  UNMATCHED  : {len(self.unmatched)} derived operator(s) "
                         "absent from the capture")
            for index, op in self.unmatched[:limit]:
                lines.append(f"    derived[{index}] {op.name} {list(op.input_shapes)}")
            if len(self.unmatched) > limit:
                lines.append(f"    ... and {len(self.unmatched) - limit} more")
        extra = self.extra_counts()
        if extra:
            lines.append(f"  capture-only: {len(self.extra)} operator(s) the model "
                         "body does not contain")
            for name, n in sorted(extra.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"    {n:5d}  {name}")
        return "\n".join(lines)


def _signature(op: OpSpec, compare_dtypes: bool) -> tuple:
    base = (op.name, op.input_shapes, op.output_shapes, op.group)
    return base + (op.dtypes,) if compare_dtypes else base


def align_graphs(derived: OpGraph, captured: OpGraph,
                 compare_dtypes: bool = True) -> Alignment:
    """Match a derived graph into a captured one as an ordered subsequence.

    Matching is greedy, which is sound as long as the capture runs the body in
    order and does not interleave an operator whose whole signature — name,
    shapes, dtypes, group — coincides with the next one expected. Should that
    ever happen the match fails closed, reporting derived operators as absent
    rather than silently pairing the wrong two.
    """
    result = Alignment(derived_len=len(derived), captured_len=len(captured))

    want = [_signature(op, compare_dtypes) for op in derived.ops]
    i = 0
    for op in captured.ops:
        if i < len(want) and want[i] == _signature(op, compare_dtypes):
            i += 1
        else:
            result.extra.append(op)
    result.matched = i
    result.unmatched = [(j, derived.ops[j]) for j in range(i, len(derived))]
    return result
