# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""How a cost was obtained, and what a declined answer carries.

Every number a cost backend produces says where it came from. The vocabulary
is closed and small, and an unlabelled cost cannot be built: `Provenance`
takes a `Species` with no default, and `CostTerm` takes a `Provenance` with no
default, so there is no constructor path that produces a number whose origin
is unknown.

The five species answer "how was this answer obtained", which is a different
question from "which cost model was asked". A step-level timing read back out
of a file is still `MEASURED`; what a lookup changes is whether the key matched
exactly (`MEASURED`), matched the nearest neighbour (`INTERPOLATED`) or sat
outside the range that was measured at all (`EXTRAPOLATED`). "Priced" is not a
species either -- a price is a measurement of a smaller unit, and the unit goes
in `detail`, as `measured (op-level)` against `measured (step-level)`.

`ANALYTICAL` is the one word that is reserved rather than aspirational: it
means the subject was computed without ever being measured, and a backend that
fits coefficients to timings is `FITTED`, not analytical, however tidy the
closed form looks.

A `Refusal` is a declined answer with a named reason, and it is a value rather
than an error. It is carried on the `Provenance` of whatever cost was produced
in its place, so a cost that came from further down a resolver ladder cannot
be mistaken for one that came from the top of it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Species(enum.Enum):
    """How one answer was obtained."""

    ANALYTICAL = "analytical"
    MEASURED = "measured"
    FITTED = "fitted"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Refusal:
    """A source declining to answer, with the reason it declined.

    `source` names the thing that declined so a reader can go and fix it, and
    `reason` says what it could not do. Both are required: a refusal nobody can
    act on is as useless as a guess, and the counting a run reports groups by
    the pair.
    """

    source: str
    reason: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("a refusal must name the source that declined")
        if not self.reason.strip():
            raise ValueError(f"{self.source} declined without naming a reason")

    def __str__(self) -> str:
        return f"refused({self.source}: {self.reason})"


@dataclass(frozen=True)
class Provenance:
    """The origin of one cost.

    `detail` is the unit or the key that was matched, free text, and is how
    `measured (op-level)` is distinguished from `measured (step-level)` without
    inventing a species for each. `refusal` is set when this cost was produced
    only because something above it declined; it is what makes a fall-through
    visible in the record instead of an inference from a changed number.
    """

    species: Species
    detail: str = ""
    refusal: Refusal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.species, Species):
            raise TypeError(f"species must be a Species, got {self.species!r}")

    @property
    def is_refused(self) -> bool:
        return self.refusal is not None

    def after(self, refusal: Refusal) -> Provenance:
        """This same origin, marked as the answer that stood in for a refusal.

        The first refusal wins: it names the source that should have answered,
        which is the one an operator has to go and measure. Later declines are
        kept on the resolution, not here.
        """
        if self.refusal is not None:
            return self
        return Provenance(self.species, self.detail, refusal)

    def __str__(self) -> str:
        named = f"{self.species} ({self.detail})" if self.detail else str(self.species)
        if self.refusal is None:
            return named
        return f"{self.refusal} -> {named}"
