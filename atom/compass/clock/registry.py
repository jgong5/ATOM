# SPDX-License-Identifier: MIT
"""The logical processes taking part in a run, held in a total order.

The order is the point. When two logical processes may advance at the same
virtual time, the one that goes first is decided here, by name, and never by
which of them asked first. Arrival order depends on process start-up, socket
readiness and the host's scheduler, so a run that grants in arrival order is a
different run every time: one earlier measurement produced 126 decode steps once
and 189 another from an unchanged configuration.

Membership lives in a ``dict`` whose values are ``None`` -- an ordered set --
rather than in a ``set``. A ``set`` of strings iterates in hash order, and string
hashing is randomised per process unless ``PYTHONHASHSEED`` is fixed, so
iterating one leaks the process's seed into the schedule. A ``dict`` iterates in
insertion order, and this module never relies even on that: every ordered answer
it gives is sorted at the point of iteration.
"""

from .identity import LpId


class LpRegistry:
    """Which logical processes exist, and in what order they are served.

    Registration is a run's set-up step: every logical process is registered
    before any of them advances. Iterating the registry, or calling `ids`,
    yields the total order.
    """

    def __init__(self) -> None:
        self._members: dict[LpId, None] = {}

    def register(self, lp_id: LpId) -> LpId:
        """Add one logical process. Refuses a duplicate rather than ignoring it.

        A name registered twice means two participants believe they are the same
        one, which would silently merge their clocks.
        """
        if not isinstance(lp_id, LpId):
            raise TypeError(f"register expects an LpId, got {type(lp_id).__name__}")
        if lp_id in self._members:
            raise ValueError(f"{lp_id} is already registered")
        self._members[lp_id] = None
        return lp_id

    def ids(self) -> tuple[LpId, ...]:
        """Every registered identity, in the total order."""
        return tuple(sorted(self._members))

    def require(self, lp_id: LpId) -> LpId:
        """Return `lp_id`, or refuse with the names that are registered.

        Used wherever an unregistered identity would otherwise be accepted and
        turn into a missing row later, far from the call that introduced it. It
        is the entry point the matrix funnels every identity through, so it
        type-checks for the same reason `register` does -- and because a bare
        `str` that reached the check below would be reported as not registered
        alongside the identically-spelled name that is.
        """
        if not isinstance(lp_id, LpId):
            raise TypeError(f"require expects an LpId, got {type(lp_id).__name__}")
        if lp_id not in self._members:
            known = ", ".join(str(known_id) for known_id in self.ids()) or "<none>"
            raise KeyError(f"{lp_id} is not registered; registered: {known}")
        return lp_id

    def __contains__(self, lp_id: object) -> bool:
        return lp_id in self._members

    def __iter__(self):
        return iter(self.ids())

    def __len__(self) -> int:
        return len(self._members)
