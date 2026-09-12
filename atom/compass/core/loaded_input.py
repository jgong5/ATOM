"""The identity of an artifact, taken from the bytes that were actually parsed.

Provenance for a run that reads tables off disk has to answer one question:
*what did this process load?* Every answer so far has been derived from the
option string instead -- the API server took the value of
``--compass-oracle-option price=...`` and went looking for files whose names
resembled it. That is a claim about a path, made later, by a different reader,
and it is wrong in three separate ways at once:

* the option is a DSL, not a filename. ``prices.json:graph.json:unregistered``
  is a triple and ``a.json,b.json`` is a list; asking the filesystem about
  either as a single name finds nothing, so a run that loaded two real files
  reported no digest at all and was refused as uncalibrated.
* the option names a *stem*. Every rank writes ``prices.tp0.json``,
  ``prices.tp1.json``; which one this rank got is the thing worth recording,
  and the option cannot say.
* re-reading happens later. Between the load and the digest a file can be
  replaced, or a symlink repointed, and the manifest then describes bytes the
  run never used -- honestly, and wrongly.

So identity is taken *at the read*: one open, one `read()`, the digest of those
bytes, and `json.loads` of those same bytes. There is no second open and no
stat-then-read, which is what makes replacement after loading unable to change
either the payload or its recorded identity. What a consumer retains is this
record, not the path it came from.

Stdlib only, and nothing imported from ATOM outside
:mod:`atom.compass.core.artifacts` -- the same contract
:mod:`atom.compass.core.process_identity` keeps, and for the same reason: a
reader must be able to check a manifest without standing up an engine.

Roles are dotted strings and deliberately not an enumeration. They are
namespaced by *what the input is to the run*, which is a distinction the
validation depends on and which no path can carry:

``oracle.*``
    inputs to constructing the cost oracle -- the price lists, the graphs they
    were measured against, the seeded templates, and the replay target the
    source factory reads to answer AITER's architecture query.
``runtime.*``
    inputs that decide the deployment's actual capacity -- the replay target
    the runner loads, the memory profile, the memory readings. A run can have
    an ``oracle.replay_target`` and no ``runtime.replay_target``: they are
    different files read by different code for different purposes, and the
    first cannot stand in for the second.

Nested inputs get nested roles -- ``runtime.memory_model.collective`` for a
calibration a profile itself names. A closed enum would have forced those to be
flattened into the parent and lost.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from atom.compass.core.artifacts import resolve_rank_path

__all__ = ["LoadedInput", "load_json", "manifest", "roll"]


@dataclass(frozen=True)
class LoadedInput:
    """One artifact, as the process that parsed it can describe it.

    Frozen because a consumer retains it for the life of the run and hands it
    to a reader that will treat it as evidence. A record that could be edited
    after the read would put the reopening problem back, one level up.
    """

    #: What this input is to the run: ``oracle.price``, ``runtime.memory_model``.
    role: str
    #: The path as the option asked for it, before rank resolution. Kept
    #: because a report has to be able to say "rank 3 asked for prices.json",
    #: which is a different statement from which file it got.
    requested: str
    #: The path actually opened.
    path: str
    #: Whether resolution found this rank's own file rather than falling back
    #: to the shared one. False at TP1, where no suffix is written.
    rank_own: bool
    #: Digest of the exact bytes that were parsed.
    sha256: str
    #: Length of those bytes.
    size: int
    #: This rank's coordinates as sorted pairs; empty at TP1.
    rank_coords: tuple = ()

    def as_dict(self) -> dict:
        """JSON-safe, for a manifest that leaves the process."""
        return {
            "role": self.role,
            "requested": self.requested,
            "path": self.path,
            "rank_own": self.rank_own,
            "sha256": self.sha256,
            "size": self.size,
            "rank_coords": {name: index for name, index in self.rank_coords},
        }

    @classmethod
    def from_dict(cls, row: Mapping) -> LoadedInput:
        """Read back a record a reader could not build directly.

        `atom.compass.replay.bootstrap` is the case this exists for: it is
        loaded by path by a child interpreter that must not import `atom`, so
        it records its read as a plain dict of exactly these fields. Reading it
        back here keeps the digest the one taken at that read rather than
        re-deriving anything.
        """
        return cls(
            role=row["role"],
            requested=row["requested"],
            path=row["path"],
            rank_own=bool(row.get("rank_own")),
            sha256=row["sha256"],
            size=int(row.get("size") or 0),
            rank_coords=_coord_pairs(row.get("rank_coords")),
        )


def _coord_pairs(coords: Mapping[str, int] | None) -> tuple:
    return tuple(
        sorted((str(name), int(index)) for name, index in (coords or {}).items())
    )


def load_json(
    requested: str,
    *,
    role: str,
    coords: Mapping[str, int] | None = None,
) -> tuple[Any, LoadedInput]:
    """Parse a JSON artifact and describe the bytes that were parsed.

    ``requested`` is the path **as the option carries it**, unresolved.
    Resolution happens here, once, through
    :func:`~atom.compass.core.artifacts.resolve_rank_path`, so that the record
    can hold both ends of it: the stem a caller asked for and the file this
    rank was actually served. A caller that resolves first and passes the
    result loses the stem and reports ``rank_own`` false for a file that was
    this rank's own.

    Returns ``(payload, LoadedInput)``. The payload is parsed from the same
    `bytes` object the digest was taken over, so the two cannot describe
    different contents. Replacing the file, or repointing a symlink, after this
    returns changes neither.
    """
    if not role:
        raise ValueError(
            "a loaded input needs a role: what it is to the run "
            "is not recoverable from its path"
        )
    if not requested:
        raise ValueError(f"{role}: no path to load")
    path, own = resolve_rank_path(requested, coords)
    with open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    return payload, LoadedInput(
        role=role,
        requested=requested,
        path=path,
        rank_own=bool(own),
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        rank_coords=_coord_pairs(coords),
    )


def roll(inputs: Iterable[LoadedInput]) -> str:
    """One digest over a set of inputs, so a member cannot change unnoticed.

    Over ``role:requested:path:sha256`` and not over the digests alone: two
    runs that loaded the same bytes into different roles did different things,
    and a rolled digest that could not tell them apart would be reporting that
    they were the same run.
    """
    rolled = hashlib.sha256()
    for row in sorted(f"{i.role}:{i.requested}:{i.path}:{i.sha256}" for i in inputs):
        rolled.update(row.encode() + b"\n")
    return rolled.hexdigest()


def manifest(
    inputs: Iterable[LoadedInput],
    *,
    coords: Mapping[str, int] | None = None,
) -> dict:
    """The immutable per-rank record of everything this rank loaded.

    Sorted by ``(role, requested, path)`` so two ranks' manifests are
    comparable line by line, and carrying its own rolled digest so a consumer
    can quote one value for "the inputs this rank ran on".

    ``coords`` names the rank the manifest is of. Left out, it is taken from
    the inputs themselves, which all carry it; a manifest with no inputs and no
    coordinates is a manifest of a rank that loaded nothing, which is a real
    state and is reported as one rather than refused.
    """
    held = list(inputs)
    if coords is None:
        seen = {i.rank_coords for i in held}
        pairs = seen.pop() if len(seen) == 1 else ()
    else:
        pairs = _coord_pairs(coords)
    rows = sorted(
        (i.as_dict() for i in held),
        key=lambda row: (row["role"], row["requested"], row["path"]),
    )
    return {
        "rank_coords": {name: index for name, index in pairs},
        "inputs": rows,
        "rolled_sha256": roll(held),
    }
