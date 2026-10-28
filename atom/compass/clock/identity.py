# SPDX-License-Identifier: MIT
"""The identity of a logical process.

A *logical process* is the unit the virtual-time protocol coordinates. It is
not an OS process: a tensor-parallel group is one logical process however wide,
because its workers are driven by a blocking call and hold no clock of their
own, and a data-parallel group is one because its ranks already synchronise
every step. What creates a new logical process is a role boundary between
prefill and decode, a pipeline stage, or an independent replica.

An identity is a name and nothing else. It does not say where the process runs,
which host or port reaches it, or how deep it sits in any arrangement of clock
authorities -- an authority must be able to appear between a logical process and
another authority without either side changing, and it can only do that if
neither side has written the other's position down.

Identities are compared by name. String comparison is by code point, so the
order two identities fall into is the same in every process, on every run. That
is the only ordering guarantee anything downstream may rely on: never the hash
of a name, which is randomised per process, and never the order the names
arrived in, which is a race.
"""

from dataclasses import dataclass

_BY_NAME: dict = {}


@dataclass(frozen=True, order=True, init=False)
class LpId:
    """The name of one logical process.

    Frozen so it can key a mapping, ordered so a caller can sort or `min` over
    identities directly and get the same answer in every process.

    A process holds one object per name: constructing, copying or unpickling a
    name returns that object. Equality is therefore identity, and both it and
    the hash run in C, which matters because the clock authority looks
    identities up dozens of times per request.
    """

    name: str

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __new__(cls, name: str):
        if not isinstance(name, str):
            raise TypeError(
                f"a logical process name must be a str, got {type(name).__name__}"
            )
        lp_id = _BY_NAME.get(name)
        if lp_id is None:
            if not name:
                raise ValueError("a logical process name must not be empty")
            if any(c.isspace() for c in name):
                raise ValueError(
                    f"a logical process name must not contain whitespace, got {name!r}; "
                    "names are written verbatim into the timeline record, one field per column"
                )
            lp_id = object.__new__(cls)
            object.__setattr__(lp_id, "name", str(name))
            lp_id = _BY_NAME.setdefault(lp_id.name, lp_id)
        return lp_id

    def __reduce__(self):
        return LpId, (self.name,)

    def __str__(self) -> str:
        return self.name
