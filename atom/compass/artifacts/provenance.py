# SPDX-License-Identifier: MIT
"""What produced an entry: the executed source roots, and there are two of them.

D41's rule is that the source root is a `git archive` digest and never an
rsync, because a correct registry digest once sat beside a stale package for
exactly that reason. The digest taken here is the **tree object** --
`git rev-parse HEAD^{tree}` -- which is the identity of the content
`git archive` ships, and unlike a commit sha it does not move when a message is
amended. That equivalence holds for this repo as it stands: there is no
`export-ignore` or `export-subst` in `.gitattributes` and no `.gitmodules`. If
either is ever added the two decouple silently, and this is the paragraph to
come back to.

A tree staged by `snapshot.sh` has no `.git`, so its `.compass-commit` stamp is
the second way in. **It answers with a commit, not a tree**, and the two are
different object types, so `revision_kind` says which -- a comparator that
cannot tell them apart would refuse two entries published from byte-identical
source, or compare them wrongly. For the same reason the stamp path reports
`dirty` as **unknown** rather than zero: `snapshot.sh` refuses to stage a dirty
tree, but a hand-written stamp is not `snapshot.sh`, and this module cannot see
the difference.

**aiter is the second root** (T86). The MoE kernels and every collective are
aiter's, aiter lives in the container's writable layer outside `/workspace`,
and two versions are in circulation on this project right now. A stanza naming
only ATOM's tree describes half of what ran, so a `Provenance` cannot be built
without both. **What a difference then means is not decided here**: whether an
aiter bump invalidates an artifact or only warns is the owner's ruling, and
this module records the version so a ruling has something to act on.

`git describe --tags --always --dirty` is `gate_gpu.sh:153-159`'s call,
adopted. What is not adopted is how that script finds the checkout: it imports
`aiter` to read `aiter.__file__`, and importing aiter shells out to `rocminfo`,
which hangs uninterruptibly on a node whose driver is wedged. `module_root`
executes nothing at all -- see its own note, because `find_spec` alone is not
enough for a dotted name.

**Every module that can answer is digested, not just the one named.** A
hard-coded `sys.path` once let a stale regions module answer under a current
registry's name, and a stanza naming one root cannot show that. `roots_for`
resolves each module on its own, so two modules landing in different trees
appear as two rows that disagree rather than one row that is wrong. The limit,
stated rather than glossed: a dotted name resolves *inside* its top-level
package, so two modules under one package necessarily share a root. The
divergence this can show is between top-level roots.
"""

import dataclasses
import datetime
import importlib.util
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn

from .rules import ArtifactRefusal, Rule

#: A stanza states at least these: ATOM's tree, and the kernels ATOM called.
REQUIRED_ROOTS = ("atom", "aiter")
#: What a `revision` is. Two entries whose kinds differ are not comparable.
REVISION_KINDS = ("tree", "commit", "describe")
GIT_TIMEOUT_S = 30
STAMP = ".compass-commit"


def _run(argv: Sequence[str]) -> tuple[int, str, str]:
    """Run a command and hand back its outcome; never raises on a non-zero exit."""
    done = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False
    )
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def _refuse(what: str, remedy: str) -> NoReturn:
    raise ArtifactRefusal(Rule.PROVENANCE, what, remedy)


@dataclass(frozen=True, slots=True)
class SourceRoot:
    """One executed source root, what identifies it, and of what kind.

    `root` records *where* the tree was found, beside the revision that
    identifies it. It is descriptive: nothing resolves through it and nothing
    should compare it, because the same tree has a different path in every
    container.

    `dirty` is the number of uncommitted files, or `None` for "not known" --
    which is what a stamped snapshot can honestly say and is not the same
    claim as zero.
    """

    name: str
    root: str
    revision: str
    revision_kind: str
    method: str
    dirty: int | None = None

    def __post_init__(self) -> None:
        for field in ("name", "root", "revision", "method"):
            if not str(getattr(self, field) or "").strip():
                _refuse(
                    f"a source root states no {field}",
                    "a root that cannot name itself is not provenance",
                )
        if self.revision_kind not in REVISION_KINDS:
            _refuse(
                f"`revision_kind` is {self.revision_kind!r}",
                f"say which object `revision` is, one of {', '.join(REVISION_KINDS)}; "
                "a tree and a commit in one field compare unequal for "
                "byte-identical source",
            )
        if self.dirty is not None and (
            isinstance(self.dirty, bool) or not isinstance(self.dirty, int)
        ):
            _refuse(
                f"`dirty` is {self.dirty!r}",
                "state how many files were uncommitted, or None for not known",
            )

    def as_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, document: object) -> "SourceRoot":
        if not isinstance(document, Mapping):
            _refuse(f"{document!r} is not a source root", "republish the entry")
        try:
            return cls(**document)
        except TypeError as bad:
            _refuse(f"a recorded source root is incomplete: {bad}", "republish it")


@dataclass(frozen=True, slots=True)
class Provenance:
    """What produced an entry, when, and out of which trees."""

    produced_by: str
    produced_at: str
    source_roots: tuple[SourceRoot, ...]

    def __post_init__(self) -> None:
        if not str(self.produced_by or "").strip():
            _refuse(
                "this entry does not say what produced it",
                "name the phase or the command, so a number has a source",
            )
        try:
            when = datetime.datetime.fromisoformat(self.produced_at)
        except (TypeError, ValueError):
            _refuse(
                f"`produced_at` is {self.produced_at!r}, which is not a time",
                "write an ISO-8601 instant with its offset",
            )
        if when.tzinfo is None:
            _refuse(
                f"`produced_at` is {self.produced_at!r}, which states no offset",
                "a local time is not comparable across two boxes; write UTC",
            )
        named = [root.name for root in self.source_roots]
        missing = [want for want in REQUIRED_ROOTS if want not in named]
        if missing:
            _refuse(
                f"this stanza names no {', '.join(missing)} source root",
                "record every tree that executed: the MoE kernels and every "
                "collective are aiter's, so a stanza naming only ATOM's tree "
                "describes half of what ran (T86)",
            )
        if len(set(named)) != len(named):
            _refuse(
                f"two source roots share a name in {named}",
                "one row per module that can answer, each named for it",
            )

    def root(self, name: str) -> SourceRoot:
        """One recorded root by name, or a refusal naming what was recorded."""
        for source in self.source_roots:
            if source.name == name:
                return source
        _refuse(
            f"this stanza has no `{name}` source root",
            f"it names {', '.join(root.name for root in self.source_roots)}",
        )

    def as_json(self) -> dict:
        return {
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "source_roots": [root.as_json() for root in self.source_roots],
        }

    @classmethod
    def from_json(cls, document: object) -> "Provenance":
        if not isinstance(document, Mapping) or "source_roots" not in document:
            _refuse(
                f"{document!r} is not a provenance stanza",
                "an entry carries what produced it; one that does not is a "
                "number with no source",
            )
        return cls(
            produced_by=document.get("produced_by", ""),
            produced_at=document.get("produced_at", ""),
            source_roots=tuple(
                SourceRoot.from_json(root) for root in document["source_roots"]
            ),
        )


def module_root(module: str) -> str:
    """Where a module's code sits, resolved without executing anything.

    `importlib.util.find_spec` does not execute a **top-level** module, and it
    **does** import the parents of a dotted one -- measured,
    `find_spec("xml.dom.minidom")` leaves `xml` and `xml.dom` in `sys.modules`.
    Under `aiter` that import is the `rocminfo` shell-out that hangs on a
    wedged driver, so only the head is resolved by the import machinery and
    the remaining segments are walked as paths.

    The walk resolves the two forms `scripts/compass/README.md:265` names for
    the trigger derivation, `X/y/` and `X/y.py`, and refuses a segment that is
    neither -- for the reason that file gives: either the name was misread or
    the module is absent, and both make the stanza wrong by an unknown amount.

    **The precedence here is the opposite of that README line**, deliberately.
    It resolves a name read out of source text and takes `X/y.py` first; this
    resolves a name the import system would, and there a package shadows a
    module of the same name. A tree holding both is digested under the
    directory Python would have executed.
    """
    head, _, rest = module.partition(".")
    try:
        found = importlib.util.find_spec(head)
    except (ImportError, ValueError):
        found = None
    if found is None or not found.origin:
        _refuse(
            f"`{head}` resolves to no file on this interpreter's path",
            "a source root that cannot be located cannot be digested, and a "
            "digest of the wrong tree is what this stanza prevents",
        )
    here = os.path.dirname(found.origin)
    for segment in filter(None, rest.split(".")):
        package = os.path.join(here, segment)
        if os.path.isfile(os.path.join(package, "__init__.py")):
            here = package
        elif not os.path.isfile(f"{package}.py"):
            _refuse(
                f"`{module}` has no `{segment}` under {here}",
                "resolve the name against the tree that is installed, or name "
                "the package instead; this walk imports nothing, so it cannot "
                "follow a module that only exists once something has run",
            )
    return here


def _stamp(root: str) -> str | None:
    """The commit a `snapshot.sh` tree names, found by walking up from `root`."""
    here = os.path.abspath(root)
    while True:
        candidate = os.path.join(here, STAMP)
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as stamped:
                return stamped.readline().strip() or None
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def git_tree_root(name: str, root: str, run=_run) -> SourceRoot:
    """A source root identified by the tree `git archive` would ship."""
    code, tree, _ = run(["git", "-C", root, "rev-parse", "HEAD^{tree}"])
    if code == 0 and tree:
        _, porcelain, _ = run(["git", "-C", root, "status", "--porcelain"])
        dirty = len([line for line in porcelain.splitlines() if line.strip()])
        method = "git rev-parse HEAD^{tree}, the tree git archive ships"
        return SourceRoot(name, root, tree, "tree", method, dirty)
    stamped = _stamp(root)
    if stamped is not None:
        return SourceRoot(
            name,
            root,
            stamped,
            "commit",
            f"{STAMP}, written by snapshot.sh from the same rev-parse that "
            "built the archive; this is a commit, not the tree above",
        )
    _refuse(
        f"`{root}` is neither a git checkout nor a stamped snapshot",
        "stage it with scripts/compass/snapshot.sh, which writes the stamp "
        "from the same rev-parse that builds the archive; an rsync of a tree "
        "cannot say what it is",
    )


def git_described_root(name: str, root: str, run=_run) -> SourceRoot:
    """A source root identified the way `gate_gpu.sh:153-159` identifies aiter."""
    code, version, error = run(
        ["git", "-C", root, "describe", "--tags", "--always", "--dirty"]
    )
    if code != 0 or not version:
        _refuse(
            f"`git describe` could not name `{root}`: {error or 'no output'}",
            "an unreadable version is a mismatch, not a match; record the "
            "checkout this ran against or do not publish the entry",
        )
    return SourceRoot(
        name,
        root,
        version,
        "describe",
        "git describe --tags --always --dirty",
        1 if version.endswith("-dirty") else 0,
    )


def roots_for(
    modules: Mapping[str, str], run=_run, locate=module_root
) -> tuple[SourceRoot, ...]:
    """One row per module that can answer, each resolved on its own.

    `modules` maps the name a row is recorded under to the module it resolves
    from, so `{"atom": "atom", "aiter": "aiter"}` is the stanza D41 and T86
    require, and a module with its own answer adds a row that can disagree
    with ATOM's rather than hiding behind it.
    """
    resolved = []
    for name, module in modules.items():
        root = locate(module)
        described = name == "aiter" or module.split(".")[0] == "aiter"
        resolve = git_described_root if described else git_tree_root
        resolved.append(resolve(name, root, run))
    return tuple(resolved)
