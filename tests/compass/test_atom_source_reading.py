# SPDX-License-Identifier: MIT
"""ATOM's source, as every artifact reads it, is the digest of ATOM's Python
modules less Compass's own code, the host key's ``atom``: not the git tree.

A commit that edits a design doc, a test or Compass code moves no reading and
no key, so memory readings and a step grid bundle published before it load
after it; one that edits ATOM's scheduler refuses both, naming `ATOM src`. A
`git archive` copy of a commit reads what its checkout reads, and a checkout
with uncommitted files still publishes nothing.
"""

import dataclasses
import importlib.util
import io
import subprocess
import tarfile

import pytest
from test_artifact_invalidation import STANZA
from test_memory_readings import DOCUMENT
from test_router_binary import commit, git
from test_run_step_grid import KEY, deployment, publish, readings

import atom
from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Topology,
    git_tree_root,
    member_name,
    module_root,
)
from atom.compass.run import memory, step_grid
from atom.compass.spec import MachineSpec

#: What the checkout holds at its first commit.
FILES = {
    "atom/__init__.py": "",
    "atom/model_engine/scheduler.py": "BUDGET = 1\n",
    "atom/compass/bench.py": "REPEATS = 3\n",
    "atom/compass/design/07.md": "# Design\n",
    "tests/test_scheduler.py": "def test_budget():\n    pass\n",
}


def resolve(monkeypatch, root):
    """Make the `atom` package resolve to `root/atom`, for the digest and for
    the provenance root alike."""
    init = root / "atom" / "__init__.py"
    monkeypatch.setattr(atom, "__file__", str(init))
    spec = importlib.util.spec_from_file_location("atom", init)
    monkeypatch.setattr(atom, "__spec__", spec)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A git checkout of `FILES`, the `atom` package resolving to it."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q")
    for path, text in FILES.items():
        commit(root, path, text)
    resolve(monkeypatch, root)
    return root


def published(tmp_path):
    """A deployment with its memory readings and a step grid bundle published
    under this ATOM source."""
    r = deployment(tmp_path)
    publish(tmp_path, r)
    return r


def load_readings(tmp_path, r):
    machine = MachineSpec.from_mapping(DOCUMENT)
    return memory.load(str(tmp_path / "store"), r.config, machine, 0)


def load_bundle(tmp_path, r):
    key = Key.of(Kind.STEP_GRID_BUNDLE, **KEY)
    store, gates = ArtifactStore(tmp_path / "store"), GateState.of()
    return store.load(key, conditions=readings(r), gates=gates)


@pytest.mark.parametrize(
    "edits",
    [
        {"atom/compass/design/07.md": "# Design, revised\n"},
        {"tests/test_scheduler.py": "def test_budget():\n    assert True\n"},
        {"atom/compass/bench.py": "REPEATS = 5\n"},
    ],
    ids=["design doc", "test", "compass code"],
)
def test_a_child_editing_no_module_atom_runs_moves_no_reading_or_key(
    checkout, tmp_path, edits
):
    r = published(tmp_path)
    key = compass_run.source_root()
    for path, text in edits.items():
        commit(checkout, path, text)

    load_readings(tmp_path, r)
    load_bundle(tmp_path, r)
    assert compass_run.source_root() == key


def test_a_child_editing_the_scheduler_refuses_both_naming_atom_src(checkout, tmp_path):
    r = published(tmp_path)
    key = compass_run.source_root()
    commit(checkout, "atom/model_engine/scheduler.py", "BUDGET = 2\n")

    assert compass_run.source_root() != key
    named = "`{}` x `ATOM src`: recorded"
    with pytest.raises(ArtifactRefusal, match=named.format("memory_readings")):
        load_readings(tmp_path, r)
    with pytest.raises(ArtifactRefusal, match=named.format("step grid bundle")):
        load_bundle(tmp_path, r)


def test_a_git_archive_copy_of_a_commit_reads_what_its_checkout_reads(
    checkout, tmp_path, monkeypatch
):
    r = published(tmp_path)
    head = git(checkout, "rev-parse", "HEAD")
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "HEAD"], capture_output=True, check=True
    ).stdout
    copy = tmp_path / "snapshot"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(copy, filter="data")
    (copy / ".compass-commit").write_text(head + "\n")
    digest = compass_run.atom_digest()
    resolve(monkeypatch, copy)

    assert compass_run.atom_digest() == digest
    load_readings(tmp_path, r)
    load_bundle(tmp_path, r)


def test_a_checkout_with_uncommitted_files_still_publishes_nothing(checkout, tmp_path):
    r = published(tmp_path)
    (checkout / "atom/compass/design/07.md").write_text("# Design, uncommitted\n")
    roots = (git_tree_root("atom", module_root("atom")), STANZA.source_roots[1])

    load_bundle(tmp_path, r)
    with pytest.raises(ArtifactRefusal, match=r"uncommitted files: atom \(1\)"):
        ArtifactStore(tmp_path / "store").publish(
            Key.of(Kind.STEP_GRID_BUNDLE, **{**KEY, "grid": "dirty"}),
            provenance=dataclasses.replace(STANZA, source_roots=roots),
            topology=Topology(),
            conditions=readings(r),
            gates=GateState.of(),
            members={member_name(*step_grid.MEMBER): b"{}"},
        )


def test_the_host_key_s_atom_is_the_reading_s_value():
    host = compass_run.host_key_of("/nonexistent-model", {})
    assert compass_run.atom_source().value == host["atom"]
