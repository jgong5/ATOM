# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The order cost sources are consulted in, and what a fall-through leaves behind.

A backend rarely has one source of truth. It has several, ranked: an exact
price for the thing being asked about, a nearest-neighbour lookup, a fitted
form, a law computed from geometry alone. `Resolver` walks them in order and
takes the first answer.

Falling through is allowed and is not free. A source that cannot answer says
why, as a `Refusal`, and every refusal collected on the way down is carried on
the `Resolution` and stamped onto the answer's provenance. So a run that drops
from a measured price to a computed one says so in the record for that term:
there is no path through this class that produces an unannotated answer from a
lower source. The failure being designed out is a run that quietly answers from
the bottom of the ladder and reports a tidy number.

When no source answers, nothing is invented. `CostRefused` is raised carrying
every refusal in order, so the caller gets the complete list of what would have
to be measured rather than the first item of it. What the caller does next --
charge the step from a lower-fidelity model and mark it, or stop the run -- is
the caller's policy, and this module deliberately has no opinion beyond
refusing to make the number up.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from atom.compass.backends.cost import CostTerm
from atom.compass.backends.provenance import Refusal


class CostSource(abc.ABC):
    """One rung: something that can price a request, or say why it cannot."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, stable, and specific enough to act on when it declines."""

    @abc.abstractmethod
    def price(self, request: Any) -> CostTerm | Refusal:
        """Answer with a term, or decline with `self.refuse(reason)`."""

    def refuse(self, reason: str) -> Refusal:
        """Build a refusal that names this source, so it cannot name another."""
        return Refusal(self.name, reason)


@dataclass(frozen=True)
class Resolution:
    """Which source answered, and what was passed over to get there."""

    term: CostTerm
    source: str
    declined: tuple[Refusal, ...]

    @property
    def fell_through(self) -> bool:
        return bool(self.declined)


class CostRefused(Exception):
    """No source could answer. Carries the whole list, not the first entry."""

    def __init__(self, request: Any, declined: Sequence[Refusal]) -> None:
        self.request = request
        self.declined = tuple(declined)
        listed = "; ".join(str(r) for r in self.declined) or "no sources were consulted"
        super().__init__(f"no cost source answered for {request!r}: {listed}")


class Resolver:
    """A fixed sequence of sources, consulted highest fidelity first."""

    def __init__(self, sources: Sequence[CostSource]) -> None:
        sources = tuple(sources)
        if not sources:
            raise ValueError(
                "a resolver with no sources can only refuse; give it sources"
            )
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, CostSource):
                raise TypeError(f"not a cost source: {source!r}")
            if source.name in seen:
                raise ValueError(
                    f"duplicate source {source.name!r}: a rung is read by name"
                )
            seen.add(source.name)
        self._sources = sources

    @property
    def sources(self) -> tuple[CostSource, ...]:
        return self._sources

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._sources)

    def resolve(self, request: Any) -> Resolution:
        """The first source that answers, with every refusal above it attached.

        The answer's provenance is stamped with the *first* refusal, because
        that is the source an operator would have to feed to remove the
        fall-through. The rest stay on the resolution, where the full list is
        what a coverage report needs.
        """
        declined: list[Refusal] = []
        for source in self._sources:
            answer = source.price(request)
            if isinstance(answer, Refusal):
                if answer.source != source.name:
                    raise ValueError(
                        f"source {source.name!r} returned a refusal naming "
                        f"{answer.source!r}; a refusal has to name who declined"
                    )
                declined.append(answer)
                continue
            if not isinstance(answer, CostTerm):
                raise TypeError(
                    f"source {source.name!r} returned {answer!r}, not a cost term"
                )
            if declined:
                answer = CostTerm(
                    answer.name, answer.seconds, answer.provenance.after(declined[0])
                )
            return Resolution(answer, source.name, tuple(declined))
        raise CostRefused(request, declined)
