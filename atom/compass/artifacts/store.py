# SPDX-License-Identifier: MIT
"""The store: a directory per entry, published once and never again (T19).

**T19 -- the store's physical form -- is decided here as a directory
convention, with no index and no manifest.** A key renders to a path by one
total function, and the entry restates its key, its topology, its provenance
and a digest of every member inside `entry.json`. Resolution is then arithmetic
over the key rather than a lookup, and there is no second record that can
disagree with the first. D41's eight incidents are each a second statement of
one fact drifting away from it; an index is one more of those, and it is the
one a store would have to maintain on every write. Principle 3 -- and principle
6 for what the filesystem gives free: publishing is a rename onto a name that
must not exist, so immutability is enforced by the operating system rather than
promised by this module.

Three refusals the convention buys, each an incident from D41:

**A handed-off entry cannot be quietly replaced.** Publishing to a key that
already has an entry is refused, and the refusal states the stored digest and
whether the incoming members are the same ones -- so a notes-only regeneration,
whose payload is bit-identical and whose digest is not, is visibly the thing it
is. The notes sit inside the digested document precisely so that changing them
changes the identity.

**Every rank of the topology is present, or nothing is published.** The
four-ranks-one-file incident wrote one name four times and left a survivor that
looked complete at 807 operators. Here a publish states its topology and its
members are checked against every rank of it, so a writer that collapses four
coordinates into one name is refused at hand-off with the three missing files
named.

**A file this store did not name cannot answer.** Every file in an entry is
read back through `naming.read_back`; a neighbour that does not parse, is not
listed, or whose digest has moved is refused by name rather than returned.

Not here, and deliberately: the invalidation matrix, fingerprint comparison on
load, and gate state. Those are ART-2's, and what they need is an entry that
can state its key, its digest and what produced it.
"""

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .keys import Key, Kind, refuse_not_a_key
from .naming import RankCoords, Topology, member_name, read_back
from .provenance import Provenance
from .rules import ArtifactRefusal, Rule

SCHEMA_VERSION = 1
ENTRY_FILE = "entry.json"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Entry:
    """One published artifact, as it was read back off the disk."""

    key: Key
    topology: Topology
    provenance: Provenance
    notes: str
    members: Mapping[str, str]
    digest: str
    directory: Path

    def read_member(self, stem: str, coords: RankCoords, extension: str) -> bytes:
        """One rank's file, named by the same function that wrote it."""
        if coords.topology != self.topology:
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"this entry was written at {self.topology.text} and the read "
                f"asks at {coords.topology.text}",
                "read at the topology the entry states; a read side that "
                "assumes a narrower run asks for a name no writer produced, "
                "which is how two workers died on a bare FileNotFoundError",
            )
        name = member_name(stem, coords, extension)
        if name not in self.members:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"`{name}` is not in {self.key}",
                "this entry holds " + ", ".join(sorted(self.members)),
            )
        payload = (self.directory / name).read_bytes()
        seen = _digest(payload)
        if seen != self.members[name]:
            raise ArtifactRefusal(
                Rule.IMMUTABLE,
                f"`{name}` digests {seen}, and this entry records "
                f"{self.members[name]}",
                "the file changed after it was published, so the entry that "
                "was reviewed under this digest no longer exists",
            )
        return payload


class ArtifactStore:
    """A root directory, and the one function from a key to a place in it."""

    def __init__(self, root: os.PathLike | str) -> None:
        self.root = Path(root)

    def directory_for(self, key: object) -> Path:
        """Where an entry lives. The only function that turns a key into a path."""
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        return self.root / key.kind.value / key.dirname()

    def publish(
        self,
        key: object,
        *,
        provenance: Provenance,
        topology: Topology,
        members: Mapping[str, bytes],
        notes: str = "",
    ) -> Entry:
        """Hand an entry off. Once per key, with every rank present."""
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        if not isinstance(provenance, Provenance):
            raise ArtifactRefusal(
                Rule.PROVENANCE,
                f"{provenance!r} is not a provenance stanza",
                "name what produced this entry and out of which trees",
            )
        if not isinstance(topology, Topology):
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"{topology!r} is not a topology",
                "state every axis's width, so each member can name its rank",
            )
        self._check_members(key, topology, members)
        if self.directory_for(key).exists():
            self._refuse_overwrite(key, members)
        self._write(
            self.directory_for(key),
            {
                "schema_version": SCHEMA_VERSION,
                "kind": key.kind.value,
                "key": key.values,
                "topology": topology.widths,
                "provenance": provenance.as_json(),
                "notes": notes,
                "members": {
                    name: _digest(body) for name, body in sorted(members.items())
                },
            },
            members,
        )
        return self.read(key)

    def read(self, key: object) -> Entry:
        """The entry at a key, checked against what it says about itself."""
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        directory = self.directory_for(key)
        if not (directory / ENTRY_FILE).is_file():
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"no entry for {key}",
                f"nothing is published at {directory}; a refusal names the key "
                "that missed, because a missing price is not a missing graph",
            )
        raw = (directory / ENTRY_FILE).read_bytes()
        document = json.loads(raw)
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} states schema_version "
                f"{document.get('schema_version')!r}",
                f"this reader understands version {SCHEMA_VERSION}",
            )
        stored = Key.of(Kind(document["kind"]), **document["key"])
        if stored != key:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} holds {stored}, and was asked for {key}",
                "an entry restates its key, so a directory moved or renamed by "
                "hand is found out rather than believed",
            )
        topology = Topology.from_mapping(document["topology"])
        self._check_directory(directory, topology, document["members"])
        return Entry(
            key,
            topology,
            Provenance.from_json(document["provenance"]),
            document["notes"],
            document["members"],
            _digest(raw),
            directory,
        )

    def _check_members(
        self, key: Key, topology: Topology, members: Mapping[str, bytes]
    ) -> None:
        if not isinstance(members, Mapping) or not members:
            raise ArtifactRefusal(
                Rule.EVERY_RANK_WRITES,
                f"{key} was handed off with no members",
                "an entry holds the files it was published for",
            )
        groups: dict[tuple[str, str], set[str]] = {}
        for name, body in members.items():
            if not isinstance(body, (bytes, bytearray)):
                raise ArtifactRefusal(
                    Rule.EVERY_RANK_WRITES,
                    f"`{name}` holds {type(body).__name__}, not bytes",
                    "publish the bytes that were written, so a digest is of "
                    "what a reader will get",
                )
            stem, coords, extension = read_back(name, topology)
            groups.setdefault((stem, extension), set()).add(coords.suffix)
        wanted = {rank.suffix for rank in topology.ranks()}
        for (stem, extension), seen in sorted(groups.items()):
            if seen != wanted:
                missing = ", ".join(
                    f"{stem}{suffix}.{extension}" for suffix in sorted(wanted - seen)
                )
                raise ArtifactRefusal(
                    Rule.EVERY_RANK_WRITES,
                    f"`{stem}.*.{extension}` covers {len(seen)} of "
                    f"{topology.width} ranks of {topology.text}; missing {missing}",
                    "every rank writes its own file, and a single-writer path "
                    "is indistinguishable from a correct one by inspection",
                )

    def _refuse_overwrite(self, key: Key, members: Mapping[str, bytes]) -> None:
        standing = self.read(key)
        incoming = {name: _digest(body) for name, body in members.items()}
        same = incoming == dict(standing.members)
        raise ArtifactRefusal(
            Rule.IMMUTABLE,
            f"{key} was handed off at {standing.digest} and this publish holds "
            + ("the same members" if same else "different members"),
            "publish a new entry under a key that states what differs; a "
            "regeneration that changes only the notes still destroys the "
            "digest the entry was reviewed as",
        )

    def _check_directory(
        self, directory: Path, topology: Topology, members: Mapping[str, str]
    ) -> None:
        present = {item.name for item in directory.iterdir()} - {ENTRY_FILE}
        for name in sorted(present):
            read_back(name, topology)
        unlisted = sorted(present - set(members))
        if unlisted:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} holds {', '.join(unlisted)}, which it does not list",
                "an entry lists every file it answers from; a neighbour nobody "
                "recorded is what a read that dropped its rank coordinates was "
                "about to answer from",
            )
        absent = sorted(set(members) - present)
        if absent:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} lists {', '.join(absent)}, which is not there",
                "the entry is incomplete; republish it under a new key",
            )

    def _write(
        self, destination: Path, document: dict, members: Mapping[str, bytes]
    ) -> None:
        """Build the entry beside its place, then move it there in one step."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.publishing"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            for name, body in members.items():
                (staging / name).write_bytes(body)
            (staging / ENTRY_FILE).write_bytes(
                json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
            )
            for item in staging.iterdir():
                item.chmod(0o444)
            os.rename(staging, destination)
        except OSError as clash:
            shutil.rmtree(staging, ignore_errors=True)
            raise ArtifactRefusal(
                Rule.IMMUTABLE,
                f"{destination} could not be created: {clash}",
                "an entry is moved into place onto a name that must not exist, "
                "so a second publisher loses the race rather than the entry",
            ) from clash
