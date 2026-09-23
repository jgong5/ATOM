# SPDX-License-Identifier: MIT
"""The artifact store's identity layer: where an entry lives and what made it.

This package exists for eight incidents that share one shape -- the number was
fine and the question of *which artifact answered* was not -- turned into
refusals:

* **A key is a tuple, never a path** (`keys`). The six artifacts and what keys
  each one; a `price_list` is `(model, width, source-root digest)`, because a
  bare or merged price file states no width and silently prices nothing.
* **One naming function** (`naming`). Write and read sides call it, per-rank
  files carry every axis's coordinates, and the suffix is applied at width one
  too -- so a width-2 read cannot answer from a width-1 neighbour.
* **Provenance names every executed source root** (`provenance`). ATOM's by the
  tree `git archive` would ship, aiter's by `git describe --tags --always
  --dirty`, the call `gate_gpu.sh:153-159` already makes. **Only the recording
  half**: the version is recorded, and what a bump *means* is the owner's
  ruling, applied elsewhere.
* **A handed-off entry is immutable** (`store`). **The physical form is a
  directory convention** -- no index, no manifest -- and publishing is a rename
  onto a name that must not exist, so an overwrite is impossible rather than
  discouraged.

The invalidation matrix, fingerprint comparison on load and gate state are not
here. What they need from this package is an entry that can state its key, its
digest and what produced it.
"""

from .keys import KEY_FIELDS, Key, Kind
from .naming import AXES, RankCoords, Topology, member_name, read_back
from .provenance import (
    REQUIRED_ROOTS,
    REVISION_KINDS,
    Provenance,
    SourceRoot,
    git_described_root,
    git_tree_root,
    module_root,
    roots_for,
)
from .rules import ArtifactRefusal, Rule
from .store import SCHEMA_VERSION, ArtifactStore, Entry

__all__ = [
    "AXES",
    "KEY_FIELDS",
    "REQUIRED_ROOTS",
    "REVISION_KINDS",
    "SCHEMA_VERSION",
    "ArtifactRefusal",
    "ArtifactStore",
    "Entry",
    "Key",
    "Kind",
    "Provenance",
    "RankCoords",
    "Rule",
    "SourceRoot",
    "Topology",
    "git_described_root",
    "git_tree_root",
    "member_name",
    "module_root",
    "read_back",
    "roots_for",
]
