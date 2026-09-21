# SPDX-License-Identifier: MIT
"""How far ahead of a peer a logical process may safely run.

For an ordered pair of logical processes *j -> i*, the lookahead is a floor on
the delay any message from *j* suffers before it can take effect on *i*. It is a
physical property of the path -- an admission delay, a relay plus a cache
transfer, a send and a receive of intermediate tensors -- so every value here is
configured. Nothing in this module asks a device for one, and nothing in it may
start to: a run has to be able to model hardware that is not the hardware it is
running on.

The floor is a commitment, not a constant left for later tuning. Declaring it
per link, up front, is what lets a link be sized by the path it represents
rather than by whatever number made the last run fast.

**A zero floor is correct.** A pair declared at zero still advances in safe
order; what it gives up is overlap, because neither side can then run ahead of
the other's current time, and the whole run collapses towards a single global
event loop. So the floor buys speed, and the amount of speed it buys is the
whole of what it decides. It never buys safety, and code that treats a small
floor as a hazard has the relationship backwards. `LookaheadMatrix.serializing`
exists to report the zero links, not to reject them.

The matrix is addressed by identity throughout. There is no row index and no
position anywhere in the interface, so an arrangement that later inserts a
participant between two existing ones renumbers nothing.
"""

import enum
import math

from .identity import LpId
from .registry import LpRegistry


class LinkClass(enum.Enum):
    """The three kinds of link between logical processes, and the scale of each.

    `scale_seconds` is the order of magnitude the link has been observed or
    modelled at. It is documentation and a sanity reference -- a declared floor
    is never derived from it -- and it is what says which link is the tight one.
    """

    #: Traffic source to engine: the modelled admission delay. The only one of
    #: the three with an end-to-end measurement behind it, and the measurement
    #: is path-specific; see `TRAFFIC_TO_ENGINE_FLOOR_SECONDS`.
    TRAFFIC_TO_ENGINE = ("traffic_to_engine", 1.0e-2)

    #: Prefill role to decode role: a router relay plus a simulated transfer of
    #: the cached keys and values. Modelled at millisecond scale, which is
    #: comfortable -- it is the cheap link to stretch across a node.
    PREFILL_TO_DECODE = ("prefill_to_decode", 1.0e-3)

    #: Pipeline stage to pipeline stage: a modelled send and receive of the
    #: intermediate tensors. Microsecond scale, and the only tight one: stages
    #: belong close together for exactly this reason.
    PIPELINE_STAGE_TO_STAGE = ("pipeline_stage_to_stage", 1.0e-6)

    def __init__(self, label: str, scale_seconds: float) -> None:
        self.label = label
        self.scale_seconds = scale_seconds

    def __str__(self) -> str:
        return self.label


#: Measured admission delay from the traffic source to the engine, per path, in
#: seconds. Earlier work measured 13.7 ms end-to-end -- worth around four points
#: of time-to-first-token -- and found the two serving paths differ enough that
#: one number for both would misprice whichever was not measured.
TRAFFIC_TO_ENGINE_FLOOR_SECONDS = {
    "offline_batch": 13.0e-3,
    "serving": 9.0e-3,
}


class InterLpLink:
    """One declared link: its ends, what kind of path it is, and its floor."""

    __slots__ = ("floor_seconds", "link_class", "source", "target")

    def __init__(
        self,
        source: LpId,
        target: LpId,
        link_class: LinkClass,
        floor_seconds: float,
    ) -> None:
        self.source = source
        self.target = target
        self.link_class = link_class
        self.floor_seconds = floor_seconds

    def __repr__(self) -> str:
        return (
            f"InterLpLink({self.source} -> {self.target}, "
            f"{self.link_class}, floor={self.floor_seconds:g}s)"
        )


class LookaheadMatrix:
    """The floors, addressed as ``L[source -> target]`` by identity.

    Asymmetric on purpose: the delay from a traffic source into an engine is not
    the delay back out, and a link is declared in each direction it carries
    messages.
    """

    def __init__(self, registry: LpRegistry) -> None:
        self._registry = registry
        self._links: dict[tuple[LpId, LpId], InterLpLink] = {}

    def declare(
        self,
        source: LpId,
        target: LpId,
        link_class: LinkClass,
        floor_seconds: float,
    ) -> InterLpLink:
        """Declare the floor on `source -> target`. Zero is accepted; see the module docstring."""
        self._registry.require(source)
        self._registry.require(target)
        if source == target:
            raise ValueError(
                f"{source} has no link to itself; a lookahead floor describes a "
                "delay between two logical processes"
            )
        if not isinstance(link_class, LinkClass):
            raise TypeError(
                f"link_class must be a LinkClass, got {type(link_class).__name__}"
            )
        floor = float(floor_seconds)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError(
                f"the floor on {source} -> {target} must be a finite number of "
                f"seconds and not negative, got {floor_seconds!r}"
            )
        key = (source, target)
        if key in self._links:
            raise ValueError(
                f"{source} -> {target} is already declared as {self._links[key]}"
            )
        link = InterLpLink(source, target, link_class, floor)
        self._links[key] = link
        return link

    def lookahead(self, source: LpId, target: LpId) -> float:
        """``L[source -> target]`` in seconds. Refuses an undeclared pair.

        An undeclared pair is not zero. Zero is a decision to serialize the two;
        silence is a link nobody has sized, and answering it with the value that
        happens to be safe would hide that.
        """
        link = self._links.get((source, target))
        if link is None:
            self._registry.require(source)
            self._registry.require(target)
            raise KeyError(
                f"{source} -> {target} has no declared lookahead floor. Declare "
                "one, at zero if the two are meant to run in lockstep."
            )
        return link.floor_seconds

    def inbound(self, target: LpId) -> tuple[InterLpLink, ...]:
        """Every declared link into `target`, in the registry's total order."""
        self._registry.require(target)
        return tuple(
            self._links[(source, target)]
            for source in self._registry.ids()
            if (source, target) in self._links
        )

    def links(self) -> tuple[InterLpLink, ...]:
        """Every declared link, ordered by source then target."""
        return tuple(self._links[key] for key in sorted(self._links))

    def undeclared(self) -> tuple[tuple[LpId, LpId], ...]:
        """Ordered pairs of registered identities with no floor yet, in order.

        A run checks this once after set-up. An empty result means every pair
        that could exchange a message has been sized.
        """
        ids = self._registry.ids()
        return tuple(
            (source, target)
            for source in ids
            for target in ids
            if source != target and (source, target) not in self._links
        )

    def serializing(self) -> tuple[InterLpLink, ...]:
        """The links declared at a zero floor -- correct, and slow.

        Reported so a run that turns out to be serial can say which pair made it
        so. Not an error, and not a warning about correctness.
        """
        return tuple(link for link in self.links() if link.floor_seconds == 0.0)

    def tightest(self) -> InterLpLink | None:
        """The link with the smallest floor: the one that bounds how far anything may run ahead."""
        links = self.links()
        if not links:
            return None
        return min(
            links, key=lambda link: (link.floor_seconds, link.source, link.target)
        )

    def __len__(self) -> int:
        return len(self._links)
