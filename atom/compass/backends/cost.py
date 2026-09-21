# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The cost of one step, and the parts it was built from.

A `StepCost` does not store a total. `seconds` is folded from `terms` on every
read, so there is no state in which the whole disagrees with the parts, and an
aggregate with no decomposition is not a thing this module can represent: the
constructor refuses an empty breakdown.

That refusal is the interesting one. The failure it prevents is the mean of an
empty sample, which is not an error, is not zero-ish, and is exactly 0.0 -- a
precise and entirely fictional answer that reads as a fast step rather than as
a missing model. An empty breakdown is the same defect one layer up.

The fold order is part of the answer. Float addition is not associative, so two
readers who sum the same terms in different orders get different bits, and a
run that has to be reproducible cannot leave that to whoever writes the next
summing loop. `fold_seconds` is the one order this package uses: a left fold
from 0.0 over the terms in the order they were given. Anyone checking a
breakdown against its total -- a reader with the artifact, a test, a later
audit -- folds it the same way and gets the same bits.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from atom.compass.backends.provenance import Provenance, Refusal, Species


def fold_seconds(values: Iterable[float]) -> float:
    """Sum in the given order, left fold from 0.0.

    The only summation this package performs, so that a total and a check of
    that total agree bit for bit.
    """
    total = 0.0
    for value in values:
        total += value
    return total


@dataclass(frozen=True)
class CostTerm:
    """One labelled part of a step's cost.

    Terms are flat. A backend that wants a hierarchy spells it in the name --
    `attention.decode` under `attention` -- rather than nesting, because the
    consumers of a breakdown want to group and sum, not to walk a tree.
    """

    name: str
    seconds: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a cost term must be named")
        if not isinstance(self.provenance, Provenance):
            raise TypeError(
                f"{self.name}: provenance is required, got {self.provenance!r}"
            )
        if not math.isfinite(self.seconds):
            raise ValueError(f"{self.name}: {self.seconds} is not a finite duration")
        if self.seconds < 0.0:
            raise ValueError(f"{self.name}: {self.seconds} s is negative")
        if self.provenance.is_refused and self.seconds <= 0.0:
            raise ValueError(
                f"{self.name}: a refusal was priced at {self.seconds} s. Something "
                "declined and the stand-in charged nothing, which removes the step "
                "from the schedule while still reporting it as priced."
            )

    @property
    def refusal(self) -> Refusal | None:
        return self.provenance.refusal


@dataclass(frozen=True)
class StepCost:
    """What one step costs, and what that cost is made of.

    Construct it from its parts. There is no constructor that takes a total,
    because a total that can be set independently of the parts is a total that
    will eventually disagree with them.
    """

    terms: tuple[CostTerm, ...]

    def __init__(self, terms: Sequence[CostTerm]) -> None:
        terms = tuple(terms)
        if not terms:
            raise ValueError(
                "a step cost with no terms is an aggregate with no decomposition; "
                "name at least one part, even if the model has only one"
            )
        seen: set[str] = set()
        for term in terms:
            if not isinstance(term, CostTerm):
                raise TypeError(f"not a cost term: {term!r}")
            if term.name in seen:
                raise ValueError(
                    f"duplicate term {term.name!r}: a breakdown is read by name"
                )
            seen.add(term.name)
        object.__setattr__(self, "terms", terms)

    @property
    def seconds(self) -> float:
        """The total, folded from the terms in their stored order."""
        return fold_seconds(term.seconds for term in self.terms)

    @property
    def refusals(self) -> tuple[Refusal, ...]:
        return tuple(t.refusal for t in self.terms if t.refusal is not None)

    @property
    def is_refused(self) -> bool:
        return any(t.provenance.is_refused for t in self.terms)

    @property
    def refused_seconds(self) -> float:
        """Seconds priced by a stand-in after something declined.

        Reported alongside the count because they answer different questions:
        a run can refuse 2% of its steps and 40% of its time.
        """
        return fold_seconds(t.seconds for t in self.terms if t.provenance.is_refused)

    def seconds_by_species(self) -> Mapping[Species, float]:
        """Seconds grouped by how they were obtained, terms in stored order."""
        mix: dict[Species, float] = {}
        for term in self.terms:
            mix[term.provenance.species] = (
                mix.get(term.provenance.species, 0.0) + term.seconds
            )
        return mix

    def rows(self) -> tuple[tuple[str, float, str], ...]:
        """The breakdown as an artifact writes it: name, seconds, provenance."""
        return tuple((t.name, t.seconds, str(t.provenance)) for t in self.terms)


class ProvenanceMix:
    """The mixture a run reports, accumulated one step at a time.

    Refusals are counted the three ways they are read: how many, what fraction
    of steps, and what fraction of predicted seconds. The seconds fraction is
    the one that decides whether a run is worth anything, and the reason counts
    are what say which measurement would close the gap.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.seconds = 0.0
        self.refused_steps = 0
        self.refused_seconds = 0.0
        self._species_seconds: dict[Species, float] = {}
        self._reasons: dict[tuple[str, str], int] = {}

    def record(self, step: StepCost) -> None:
        self.steps += 1
        self.seconds += step.seconds
        if step.is_refused:
            self.refused_steps += 1
            self.refused_seconds += step.refused_seconds
        for species, seconds in step.seconds_by_species().items():
            self._species_seconds[species] = (
                self._species_seconds.get(species, 0.0) + seconds
            )
        for refusal in step.refusals:
            key = (refusal.source, refusal.reason)
            self._reasons[key] = self._reasons.get(key, 0) + 1

    @property
    def refused_step_fraction(self) -> float:
        return self.refused_steps / self.steps if self.steps else 0.0

    @property
    def refused_second_fraction(self) -> float:
        return self.refused_seconds / self.seconds if self.seconds > 0.0 else 0.0

    def seconds_by_species(self) -> Mapping[Species, float]:
        return dict(self._species_seconds)

    def reasons(self) -> Mapping[tuple[str, str], int]:
        """Distinct (source, reason) pairs with the number of terms each hit."""
        return dict(self._reasons)
