# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The interface a cost backend implements.

Two methods. `estimate` prices one step; `describe` says what this backend is,
in a line that goes into the run record so a result can be read a month later.

`batch_view` is a projection of the scheduled batch, prepared by the caller --
plain numbers and sequences, no engine objects. That is the whole reason this
package imports nothing from the rest of ATOM: a backend is then testable on a
machine with no accelerator and no engine, and the fields it depends on are
visible in the projection rather than reachable by attribute from anywhere in
the scheduler. The projection's shape belongs to the caller that builds it, so
it is not named here.

`tier` says which cost model was asked, which is a different axis from where
each answer inside it came from. A run can ask for the op-level model and still
receive analytically-computed seconds for a term the campaign never covered;
the tier records the intent and the provenance records what happened.
"""

from __future__ import annotations

import abc
import enum
from typing import Any

from atom.compass.backends.cost import StepCost


class Tier(enum.Enum):
    """Which cost model was asked for."""

    ANALYTIC = "0"
    COARSE = "a"
    OP_LEVEL = "b"

    def __str__(self) -> str:
        return self.value


class CostBackend(abc.ABC):
    """Turns a projected batch into a duration with its decomposition."""

    @property
    @abc.abstractmethod
    def tier(self) -> Tier:
        """Which cost model this backend is."""

    @abc.abstractmethod
    def estimate(self, batch_view: Any) -> StepCost:
        """Price one step.

        Returns a `StepCost`, which carries its breakdown by construction.
        Raises `CostRefused` when nothing can price the step: a backend asked
        about something it has no basis for refuses by name and does not
        substitute a number, however plausible one would look.
        """

    @abc.abstractmethod
    def describe(self) -> str:
        """One line naming this backend and what it is answering from."""
