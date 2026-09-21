# SPDX-License-Identifier: MIT
"""A recorded region, and the statement of where that record is valid.

A trace is a straight-line record of one path through the model. It says
nothing about the paths not taken, so a graph that carried no statement of its
own validity would be priced against steps it never described -- which is how a
flat operator list came to answer for exactly one batch shape while appearing to
answer for all of them.

The statement lives outside the region rather than as branches inside it. It is
produced from what the tracing run itself guarded on, so it cannot drift away
from the control flow it describes, and a step that falls outside every recorded
graph is detected rather than mispriced.

`Applicability` is the slot for that statement and not the statement itself:
deciding a step against recorded guards is the job of the module that evaluates
them, and it chooses how it is asked. What is fixed here is that a graph cannot
be built without one. `None` is not an empty predicate; it is a graph claiming
to apply everywhere, which is the failure this field exists to prevent.
"""

import abc
from dataclasses import dataclass

from .nodes import Region


class Applicability(abc.ABC):
    """Where a recorded graph is valid.

    In two parts, and both are needed: a discrete key matched by equality --
    forward mode, whether the batch has cached context, whether the step
    produces output, speculation width, replayed or eager, the attention backend
    -- and a symbolic domain evaluated from the guards the tracing run
    installed. Guards only capture branches taken on a shape, so a branch on a
    non-shape leaves no guard behind and the key is what covers it.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Graph:
    """One recorded region together with where it is valid.

    A graph is not itself a region. Regions compose into one tree and a graph
    puts a validity statement on the whole of it; nesting one inside another
    would put two statements on one sequence of work, with nothing saying which
    of them governs.
    """

    applicability: Applicability
    region: Region

    def __post_init__(self) -> None:
        if not isinstance(self.applicability, Applicability):
            raise TypeError(
                "a graph states where its record is valid; expected an "
                f"Applicability, got {type(self.applicability).__name__}"
            )
        if not isinstance(self.region, Region):
            raise TypeError(
                f"a graph holds one region, got {type(self.region).__name__}"
            )
