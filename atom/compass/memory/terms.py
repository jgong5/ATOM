# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A memory reading and the terms it was composed of, which it cannot shed.

`03` D16 is a rule about arithmetic that has already gone wrong once: a summed
non-KV memory check read +13.8% while holding three errors, two of which
cancelled, and the largest of them was 25% of its own term. The conclusion
recorded there is that every term is validated individually and never as a sum.
A reading object that can be printed only as a total reproduces that failure by
design, so this one cannot be:

- `Reading` holds its terms and derives the total on every read, so the two
  cannot drift apart and there is no constructor that takes a total.
- It defines no `__int__` and no `__index__`. A reading cannot be spent as a
  number by accident; a caller that wants the number asks for `.total` and the
  call site says so.
- `__str__` is the per-term table. Printing a reading prints its decomposition,
  which is principle 7 made structural rather than remembered.

**`Basis` is not `Species`.** `backends/provenance.py` answers *how a cost was
obtained* -- analytical, measured, fitted, interpolated, extrapolated. This enum
answers a different question about a different subject: *where this memory
term's bytes came from* -- a named field of the machine spec, the model's own
geometry, arithmetic over other readings, or a coefficient somebody wrote down.
The two vocabularies are kept apart deliberately. A declared coefficient has no
word in `Species` and adding one is an open owner ruling (**#87**), so nothing
here touches that enum; `Basis.DECLARED` carries the distinction on this side of
the boundary, and it is the one place a ruling would land.

A declared term must say what replaces it. That is the whole difference between
a declared number and a guess: `Term` refuses to hold `DECLARED` without a note,
so the record of what is still owed is written where the number is, not in a
document beside it.
"""

from __future__ import annotations

import enum
import textwrap
from dataclasses import dataclass


class Basis(enum.Enum):
    """Where a term's bytes came from."""

    SPEC = "spec"
    GEOMETRY = "geometry"
    DECLARED = "declared"
    DERIVED = "derived"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Term:
    """One named quantity of bytes, with the place it came from.

    `source` is the dotted path of the spec field, the config fields that were
    read, or the readings that were composed -- specific enough that a reader
    who doubts the number knows where to go and look. `nbytes` may be negative:
    a clean box subtracts, and naming the subtraction as a term is what keeps
    the derivation visible in the table rather than folded into one figure.
    """

    name: str
    nbytes: int
    basis: Basis
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a term is named, or the table it sits in is unreadable")
        if not isinstance(self.basis, Basis):
            raise TypeError(f"{self.name}: basis must be a Basis, got {self.basis!r}")
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int):
            raise TypeError(
                f"{self.name}: bytes are whole, got {self.nbytes!r}; round at the "
                "call site so the rounding is visible where it happens"
            )
        if not self.source.strip():
            raise ValueError(
                f"{self.name} states no source; a number without one is a defect "
                "(principle 8), and this is the field that carries it"
            )
        if self.basis is Basis.DECLARED and not self.note.strip():
            raise ValueError(
                f"{self.name} is declared and does not say what replaces it; a "
                "declared coefficient that records no successor is a guess"
            )

    @property
    def mib(self) -> float:
        return self.nbytes / (1 << 20)


@dataclass(frozen=True, slots=True)
class Reading:
    """One of the readings `get_num_blocks` takes off a card, and its terms."""

    name: str
    terms: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a reading is named after the reading it substitutes")
        if not isinstance(self.terms, tuple) or not self.terms:
            raise ValueError(
                f"{self.name} has no terms; a reading is its decomposition, so "
                "one with nothing in it is a total wearing a name"
            )
        seen = set()
        for term in self.terms:
            if not isinstance(term, Term):
                raise TypeError(f"{self.name}: not a term: {term!r}")
            if term.name in seen:
                raise ValueError(
                    f"{self.name} carries two terms called {term.name!r}; the "
                    "table would show one number twice and name it once"
                )
            seen.add(term.name)

    @property
    def total(self) -> int:
        """The sum of the terms, folded on every read so it cannot drift."""
        return sum(term.nbytes for term in self.terms)

    @property
    def declared(self) -> tuple[str, ...]:
        """The terms still standing on a declared coefficient."""
        return tuple(t.name for t in self.terms if t.basis is Basis.DECLARED)

    def table(self) -> str:
        """The per-term table: what this reading is, never only what it totals."""
        rows = [(t.name, f"{t.mib:,.1f}", str(t.basis), t.source) for t in self.terms]
        rows.append((f"= {self.name}", f"{self.total / (1 << 20):,.1f}", "", ""))
        widths = [max(len(row[col]) for row in rows) for col in range(4)]
        lines = [f"{self.name}  (MiB)"]
        for name, mib, basis, source in rows:
            lines.append(
                f"  {name:<{widths[0]}}  {mib:>{widths[1]}}  "
                f"{basis:<{widths[2]}}  {source}".rstrip()
            )
        for term in self.terms:
            if term.note:
                wrapped = textwrap.wrap(term.note, 72)
                lines.append(f"  * {term.name}: {wrapped[0]}")
                lines.extend(f"      {line}" for line in wrapped[1:])
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()
