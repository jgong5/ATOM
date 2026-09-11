"""The region the measurements cover, and the rule for answering inside it.

Support here is the measured points themselves, not a box drawn around them.
The distinction matters: knowing a price at 32 rows and at 16384 rows does not
make 512 rows supported, and a coordinate-wise bound would say it does. What
makes an unmeasured row count answerable is having measurements on *both* sides
of it that are close enough together for an interpolant between them to mean
something.

"Close enough" is a declared rule, not a discovered one. ``max_gap_ratio`` is
the largest ratio between two adjacent measured row counts that may be
interpolated across; the default of 2.0 says a price may be read off a curve
only where the curve was sampled at least once per doubling. On today's
evidence -- rows measured at 4, 20, 32, 4672 and 16384 -- that refuses almost
everything, which is the correct reading of five points spanning three orders
of magnitude, and it states exactly what an acquisition has to fill.

Every answer carries where it came from. An interpolated price names both
bracketing measurements and their files; an exact one names its own. The
uncertainty is assembled from what the measurements show -- the spread across
repeats of the same point, and the declared spread of any nuisance the family
collapsed -- and never from a residual against anything being predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["MeasuredCurve", "MeasuredPoint", "Refusal", "RowSupport"]


@dataclass(frozen=True)
class Refusal:
    """Why a price could not be given. Carried, not raised."""

    reason: str
    component: str = ""

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class MeasuredPoint:
    """One family/template priced at one row count, across however many files."""

    rows: int
    #: every reading of this point, one per price file
    seconds: tuple[float, ...]
    #: the file each reading came from, aligned with ``seconds``
    sources: tuple[str, ...]
    #: kernel names the benchmark saw, for the launch count the caller needs
    kernels: tuple[str, ...] = ()

    @property
    def value(self) -> float:
        """The reading. The median, so one outlying repeat cannot carry it."""
        ordered = sorted(self.seconds)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @property
    def spread(self) -> float:
        """Relative spread across repeats, 0.0 when there is only one reading."""
        if len(self.seconds) < 2:
            return 0.0
        lo, hi = min(self.seconds), max(self.seconds)
        return (hi - lo) / lo if lo > 0 else 0.0


@dataclass
class MeasuredCurve:
    """Every measurement of one template, indexed by row count."""

    template_key: str
    family: str
    points: dict[int, MeasuredPoint] = field(default_factory=dict)
    #: spread contributed by nuisances this family collapsed, as a fraction
    nuisance_spread: float = 0.0

    def add(self, rows: int, seconds: float, source: str,
            kernels: tuple[str, ...] = ()) -> None:
        existing = self.points.get(rows)
        if existing is None:
            self.points[rows] = MeasuredPoint(rows, (seconds,), (source,),
                                              kernels)
            return
        self.points[rows] = MeasuredPoint(
            rows,
            existing.seconds + (seconds,),
            existing.sources + (source,),
            existing.kernels or kernels,
        )

    @property
    def measured_rows(self) -> list[int]:
        return sorted(self.points)


@dataclass(frozen=True)
class Price:
    """A price and everything needed to judge it."""

    seconds: float
    kernels: tuple[str, ...]
    #: fractional uncertainty, assembled from measured spreads only
    uncertainty: float
    #: how this was arrived at: ``"measured"`` or ``"interpolated"``
    basis: str
    #: the files the measurements came from
    sources: tuple[str, ...]
    detail: str = ""


class RowSupport:
    """Answers a row count against one template's measured curve, or refuses."""

    def __init__(self, curve: MeasuredCurve, max_gap_ratio: float = 2.0,
                 template_verified: bool = True) -> None:
        self.curve = curve
        self.max_gap_ratio = max_gap_ratio
        self.template_verified = template_verified

    def describe(self) -> str:
        rows = self.curve.measured_rows
        if not rows:
            return f"{self.curve.family}: no measured points"
        gaps = [f"{a}->{b} (x{b / a:.1f})"
                for a, b in zip(rows, rows[1:]) if b / a > self.max_gap_ratio]
        note = (f", gaps too wide to interpolate: {'; '.join(gaps)}"
                if gaps else "")
        return (f"{self.curve.family}: measured at {rows}"
                f", max gap ratio {self.max_gap_ratio}{note}")

    def price(self, rows: int) -> Price | Refusal:
        curve = self.curve
        if not curve.points:
            return Refusal("no measurement for this operator's template")
        if not self.template_verified:
            return Refusal(
                "this template abstracts an integer as a multiple of the row "
                "count and no second measured row count confirms it, so the "
                "abstraction is a claim rather than a reading",
                component="row_template")

        exact = curve.points.get(rows)
        if exact is not None:
            return Price(
                seconds=exact.value,
                kernels=exact.kernels,
                uncertainty=_combine(exact.spread, curve.nuisance_spread),
                basis="measured",
                sources=exact.sources,
                detail=f"measured at {rows} rows across "
                       f"{len(exact.seconds)} file(s)",
            )

        measured = curve.measured_rows
        if rows < measured[0] or rows > measured[-1]:
            return Refusal(
                f"{rows} rows is outside the measured range "
                f"[{measured[0]}, {measured[-1]}]; a price there would be an "
                "extrapolation, which no measurement supports",
                component="rows")

        lo = max(r for r in measured if r < rows)
        hi = min(r for r in measured if r > rows)
        ratio = hi / lo
        if ratio > self.max_gap_ratio:
            return Refusal(
                f"{rows} rows falls in the unsampled gap {lo}..{hi} "
                f"(x{ratio:.1f}), wider than the declared max gap ratio "
                f"{self.max_gap_ratio}: the curve was never sampled close "
                "enough here for an interpolant to stand for a measurement",
                component="rows")

        left, right = curve.points[lo], curve.points[hi]
        seconds = _log_interpolate(rows, lo, left.value, hi, right.value)
        return Price(
            seconds=seconds,
            kernels=left.kernels or right.kernels,
            # The bracketing spreads bound the interpolant no better than they
            # bound their own points, and the gap itself is uncertainty the
            # measurements do not resolve; carry the gap as a term.
            uncertainty=_combine(max(left.spread, right.spread),
                                 curve.nuisance_spread,
                                 _gap_uncertainty(ratio)),
            basis="interpolated",
            sources=left.sources + right.sources,
            detail=(f"interpolated between {lo} and {hi} rows "
                    f"(x{ratio:.2f} apart)"),
        )


def _log_interpolate(x: float, x0: float, y0: float,
                     x1: float, y1: float) -> float:
    """Linear in log-log, which keeps a cost that scales with work monotone.

    Falls back to linear where a reading is non-positive, which a time should
    never be but a degenerate file could contain.
    """
    if min(x0, x1, y0, y1) <= 0:
        weight = (x - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)
    weight = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + weight * (math.log(y1) - math.log(y0)))


def _gap_uncertainty(ratio: float) -> float:
    """How much an interpolation across a gap of this ratio is worth doubting.

    Not a measurement and not presented as one: it is a stated penalty that
    grows with the gap, so a price read across a x1.9 gap is not reported with
    the same confidence as one read across a x1.05 gap. It is deliberately
    crude because refusing wide gaps, not pricing them cheaply, is the
    mechanism this module relies on.
    """
    return max(0.0, ratio - 1.0) * 0.05


def _combine(*fractions: float) -> float:
    """Independent fractional spreads, added in quadrature."""
    return math.sqrt(sum(f * f for f in fractions))
