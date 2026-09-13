"""Where the library changes kernel between two measured points.

`support.py` refuses an interpolant whose two bracketing measurements ran
under different kernels. That test reads the two endpoints and nothing
between them, and on this source that is not enough: the probe in
`agent_scratch/stage/DISPATCH_BANDS.md` found the Tensile dispatch is NOT
monotone in the row count, so a tile NAME REAPPEARS IN DISJOINT BANDS. For the
two widest body geometries `MT256x192x64_MI32x32x1` serves 8256..9216, again
11328..12288, and again 14400..15360. So 9216 and 14400 name the same kernel
while four other tiles sit between them, and an endpoint test alone would read
that as one curve and interpolate the whole 9216..14400 gap.

What rules that out is the probe itself, used as evidence rather than as a
planning note: it dispatched every row on the 64-token grid it covers and
recorded which kernel served it. Consecutive probed rows whose kernels differ
bracket a switch, and the upper row of such a pair is a BAND START. A price
interpolated across a band start crosses a switch the probe has already seen,
whatever its endpoints happen to be called.

Two limits are deliberate, because the probe is a finite reading:

* Its span. Outside `[first probed row, last probed row]` this module says
  nothing and the endpoint test stands alone. Silence here is silence, not
  permission.
* Its grid. A switch is located to a 64-row pair, not to a row. The band start
  is reported as the upper row of the pair, which is the first row KNOWN to
  run the new kernel; the switch itself sits somewhere in (start-step, start].
  A bracket is refused when a band start falls in it, which is conservative in
  exactly the direction a missed switch is not.

No timing in a probe file is a price and none is read here. Only the rows and
the kernel names are taken; `seconds` is ignored on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
__all__ = ["BandEvidence", "BandMap", "geometry_key", "load_band_map",
           "static_shapes"]

#: Stands in for a row two files disagree about. Unique per row, so it differs
#: from its neighbours on both sides and the rows either side of it are read
#: as switches. A disagreement is not a reading, and it is not silence either.
_CONFLICT = "<probe files disagree at this row>"


def geometry_key(family: str, dtypes: Iterable[str],
                 static: Iterable[Iterable[int]]) -> tuple:
    """The key a curve and a probed geometry are matched on.

    The family, the operand dtypes, and the shapes that do NOT carry the row
    count -- for a projection, its weight. Deliberately not `grouping_key`:
    that one drops the shapes themselves, and two projections of different
    widths share it while dispatching to entirely different tiles.
    """
    return (
        family,
        tuple(str(d) for d in dtypes or ()),
        tuple(tuple(int(d) for d in shape) for shape in static or ()),
    )


def static_shapes(op: dict, rows: int) -> tuple:
    """The static operand geometry used to select a dispatch survey.

    GEMM operand 0 is A[M,K]; its remaining tensor operands are static in M.
    In particular, B[N,K] stays the weight when M happens to equal N or K.
    Numeric equality cannot identify the operand's role.
    """
    out = []
    gemm = op.get("name") == "aiter::gemm_a16w16"
    for position, shape in enumerate(op.get("input_shapes") or ()):
        if not isinstance(shape, (list, tuple)):
            continue
        dims = tuple(int(d) for d in shape)
        is_row_operand = (position == 0) if gemm else (rows in dims)
        if is_row_operand:
            continue
        out.append(dims)
    return tuple(out)


def _starts_from(readings: tuple) -> tuple:
    """The first row KNOWN to run a new kernel, for each change in `readings`.

    Rows with no recorded kernel are dropped rather than treated as a change:
    an unrecorded kernel says nothing, which is the rule `support._switches`
    already applies to an endpoint.
    """
    known = sorted((rows, kernels) for rows, kernels in readings if kernels)
    return tuple(hi for (_, k_lo), (hi, k_hi) in zip(known, known[1:])
                 if k_lo != k_hi)


@dataclass(frozen=True)
class BandEvidence:
    """One probed geometry: where its kernel changes, and over what span."""

    #: rows that are the first KNOWN row of a new kernel, ascending
    starts: tuple[int, ...]
    #: the probed span, inclusive; nothing is claimed outside it
    low: int
    high: int
    #: the probe's row step, so a caller can say how precisely a switch is
    #: located; 0 where fewer than two rows were probed
    step: int
    #: the files this came from
    sources: tuple[str, ...] = ()
    #: the readings `starts` was computed from, where they are known: (rows,
    #: kernels) pairs. Carried because two shards must have their starts
    #: recomputed over the UNION of their rows -- a switch that straddles the
    #: shard boundary is witnessed by neither shard alone. Empty where a
    #: caller declared the bands directly.
    readings: tuple = ()

    def crossings(self, lo: int, hi: int) -> tuple[int, ...]:
        """Band starts strictly inside the bracket ``lo..hi``.

        A start exactly at `hi` counts: the row below it ran another kernel,
        so an interpolant reaching down from `hi` crosses the switch. A start
        exactly at `lo` does not: the switch is at or below `lo`, and the
        bracket begins on the new kernel's side of it.
        """
        return tuple(b for b in self.starts if lo < b <= hi)

    def covers(self, lo: int, hi: int) -> bool:
        """Whether the probe spans the whole bracket."""
        return self.low <= lo and hi <= self.high


def _merge(old: BandEvidence, new: BandEvidence) -> BandEvidence:
    """Two shards of one geometry's evidence, as if probed in one pass.

    The starts are RECOMPUTED over the union of the readings, not unioned:
    the probe ran in shards (8192..9280, 9280..16384, a smoke pair), and a
    switch that falls across a shard boundary is witnessed by neither shard
    on its own. Where a side declared bands without readings its starts are
    kept as they stand, because there is nothing to recompute them from.
    """
    readings: dict[int, tuple] = {}
    for side in (old, new):
        for rows, kernels in side.readings:
            existing = readings.get(rows)
            if existing is None or existing == kernels:
                readings[rows] = kernels
            elif existing[:1] == (_CONFLICT,):
                continue
            else:
                readings[rows] = (_CONFLICT, str(rows))
    merged = tuple(sorted(readings.items()))
    starts = set(_starts_from(merged)) if merged else set()
    for side in (old, new):
        if not side.readings:
            starts |= set(side.starts)
    return BandEvidence(
        starts=tuple(sorted(starts)),
        low=min(old.low, new.low),
        high=max(old.high, new.high),
        step=max(old.step, new.step),
        sources=tuple(dict.fromkeys(old.sources + new.sources)),
        readings=merged,
    )


class BandMap:
    """Band evidence for every probed geometry, keyed by :func:`geometry_key`."""

    def __init__(self) -> None:
        self._by_key: dict[tuple, BandEvidence] = {}

    def __len__(self) -> int:
        return len(self._by_key)

    def add(self, key: tuple, evidence: BandEvidence) -> None:
        """Merge one geometry's evidence, which may arrive in several files."""
        old = self._by_key.get(key)
        self._by_key[key] = (evidence if old is None
                             else _merge(old, evidence))

    def get(self, key: tuple) -> Optional[BandEvidence]:
        return self._by_key.get(key)

    def keys(self) -> Iterable[tuple]:
        return self._by_key.keys()

    def describe(self) -> str:
        if not self._by_key:
            return "no dispatch-band evidence"
        parts = []
        for key, ev in sorted(self._by_key.items(), key=lambda kv: repr(kv[0])):
            parts.append(f"{key[0]}{list(key[2])}: {len(ev.starts)} band "
                         f"start(s) over {ev.low}..{ev.high} step {ev.step}")
        return "; ".join(parts)


def load_band_map(paths: Iterable[Any], family: str) -> BandMap:
    """Read probe files into a :class:`BandMap` for one family.

    `family` is declared by the caller because a probe file records the
    geometry it dispatched and not the operator name the price library knows
    it by; guessing that name from a file path is how evidence gets attached
    to the wrong family.
    """
    band_map = BandMap()
    for path in paths:
        blob = json.loads(Path(path).read_text())
        for geometry in blob.get("geometries") or ():
            readings = tuple(sorted(
                (int(point["rows"]), tuple(point.get("kernels") or ()))
                for point in geometry.get("points") or []))
            known = [rows for rows, kernels in readings if kernels]
            if len(known) < 2:
                continue
            gaps = [b - a for a, b in zip(known, known[1:])]
            weight = geometry.get("weight")
            key = geometry_key(family, geometry.get("dtypes") or (),
                               [weight] if weight else [])
            band_map.add(key, BandEvidence(
                starts=_starts_from(readings),
                low=known[0], high=known[-1],
                step=min(gaps) if gaps else 0,
                sources=(str(path),), readings=readings))
    return band_map
