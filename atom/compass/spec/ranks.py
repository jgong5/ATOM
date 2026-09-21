# SPDX-License-Identifier: MIT
"""One reading taken on every rank, reduced to the single number a spec carries.

A memory reading is taken per rank, and the ranks of a symmetric group do
identical work, so the ranks are repetitions of one measurement and the spec
wants one number out of them. Which number is not a detail.

**The minimum is the reading.** A device-wide reading counts every process on
the card, so a rank whose card also carries somebody else's work reads high by
whatever the neighbour holds. No rank ever reads *low* for that reason, which
makes the smallest reading the one least contaminated by a process that has
nothing to do with this engine.

**The spread is reported, never folded away.** Taking a minimum and reporting
only that hides the disagreement, and the disagreement is the single-run
measurement of how quiet the machine was. The spread across ranks is
legitimately non-zero: it was measured at zero for one and two ranks, 192 MiB
across four and 640 MiB across eight, on a machine with nothing else on it. A
tool that refused those would be refusing the hardware for behaving as the
hardware behaves.

**Above a threshold it is refused.** The spread that matters is not a few per
cent. The case this exists to catch is a neighbour holding 152 GB against a
rank's own 2.9 GB, which is a factor of fifty and not a fraction; six runs died
at start-up on a negative memory budget because nothing looked at it. So the
limit sits well above the widest honest spread and far below the hazard: 640
MiB in about 10.4 GiB is six per cent of the smallest reading, and
`SPREAD_LIMIT` accepts four times that while still declining the fifty-fold
case by two orders of magnitude. It is a parameter because it is a judgement
about how quiet a machine has to be, and whoever is running the machine is
better placed to make it than this module is.

The readings themselves are checked the way the schema checks a quantity, so a
reading that could never be written into a spec is declined here, where the
ranks that produced it are still in hand, rather than later where they are not.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .fields import Field, Kind, check
from .rules import Rule, SpecRefusal

#: The widest cross-rank spread accepted, as a fraction of the smallest reading.
SPREAD_LIMIT = 0.25


@dataclass(frozen=True, slots=True)
class RankSpread:
    """One reading per rank: the number a spec takes, and their disagreement."""

    name: str
    tp_width: int
    readings: tuple[float, ...]

    @property
    def minimum(self) -> float:
        """The reading a spec carries: the least contaminated of the ranks."""
        return min(self.readings)

    @property
    def spread(self) -> float:
        """How far the ranks disagreed, in the reading's own unit."""
        return max(self.readings) - self.minimum

    @property
    def relative(self) -> float:
        """The spread as a fraction of the reading, which is what is judged."""
        return self.spread / self.minimum

    def __str__(self) -> str:
        return (
            f"{self.name} at tensor-parallel width {self.tp_width}: "
            f"{self.minimum!r} across {len(self.readings)} ranks, "
            f"spread {self.spread!r} ({self.relative:.1%})"
        )


def across_ranks(
    name: str,
    tp_width: int,
    readings: Sequence[float],
    limit: float = SPREAD_LIMIT,
) -> RankSpread:
    """Reduce one reading per rank to the spec's number, or refuse the spread."""
    readings = tuple(readings)
    if len(readings) != tp_width:
        raise SpecRefusal(
            Rule.RANK_AGREEMENT,
            f"`{name}` has {len(readings)} reading(s) for the {tp_width} ranks "
            "of this group",
            "read every rank: a minimum over some of them is not the minimum, "
            "and the ranks left out are the ones that would have shown a "
            "neighbour",
        )
    for rank, reading in enumerate(readings):
        check(Field(name, Kind.QUANTITY), reading, f"{name}[rank {rank}]")
    spread = RankSpread(name, tp_width, readings)
    if spread.relative > limit:
        raise SpecRefusal(
            Rule.RANK_AGREEMENT,
            f"`{name}` reads {spread.minimum!r} on one of the {tp_width} ranks "
            f"and {max(readings)!r} on another, a spread of "
            f"{spread.relative:.1%} where this accepts {limit:.1%}",
            "ranks of one symmetric group do identical work, so a spread this "
            "wide is another process on the card rather than the engine; find "
            "what else holds memory there and measure again, because a reading "
            "taken beside a neighbour describes the neighbour",
        )
    return spread
