# SPDX-License-Identifier: MIT
"""`snapshot.sh`'s two refusals: a ref that did not resolve, and a history that
has no merge-base with it.

Both exit 92, and until 2026-09-22 both printed `REFUSED: no merge-base with
<ref>` -- which reads as a claim about the tree, and was printed just as often
when the ref had never resolved and nothing had been compared at all. Only git's
own `fatal:` on stderr said which had happened, so a reader went to inspect a
tree that was fine. The two are told apart here separately, one test each, so
they cannot merge back into one message.

The third case is the one that made this expensive. The default
`feature/atomcompass_new` is the branch's name on the remote; a linked worktree
and a fresh clone carry only `fork/feature/atomcompass_new`, so with nothing set
the default resolved nowhere and every such tree exited 92.

Each test builds a throwaway git repository under `tmp_path` and copies the
scripts into it, so the tree the script acts on is the fixture and not this
checkout. Nothing here needs a driver or an `import atom`.
"""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "compass"

BASH = shutil.which("bash")
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    BASH is None or GIT is None, reason="needs bash and git"
)

# A fixed identity and no inherited git config: these repositories must be the
# same on any box, and an inherited COMPASS_INTEGRATION_REF would silently
# replace the default this file exists to test.
ENV = {
    k: v for k, v in os.environ.items() if not k.startswith(("COMPASS_", "GIT_"))
} | {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "compass-test",
    "GIT_AUTHOR_EMAIL": "compass@example.invalid",
    "GIT_COMMITTER_NAME": "compass-test",
    "GIT_COMMITTER_EMAIL": "compass@example.invalid",
}


def _git(root, *args, stdin=None):
    return subprocess.run(
        [GIT, "-C", str(root), *args],
        capture_output=True,
        text=True,
        env=ENV,
        input=stdin,
        check=True,
    ).stdout.strip()


def _snapshot(tree, outdir, **env):
    """Run the fixture's own copy, standing in the fixture, as an agent would."""
    return subprocess.run(
        [BASH, str(tree / "scripts" / "compass" / "snapshot.sh"), str(outdir)],
        cwd=str(tree),
        capture_output=True,
        text=True,
        env=ENV | env,
        check=False,
    )


@pytest.fixture
def tree(tmp_path):
    """A committed tree carrying the scripts, and no integration ref of any kind."""
    root = tmp_path / "tree"
    (root / "scripts" / "compass").mkdir(parents=True)
    for name in ("_lib.sh", "snapshot.sh"):
        shutil.copy(SCRIPTS / name, root / "scripts" / "compass" / name)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "tree")
    return root


@pytest.fixture
def outdir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


def test_unresolvable_ref_refuses_at_ref_resolution(tree, outdir):
    """No such ref, bare or remote-qualified: nothing was compared, and the
    refusal says so and quotes git rather than paraphrasing it."""
    r = _snapshot(tree, outdir)
    assert r.returncode == 92, r.stderr
    assert "ref resolution failed" in r.stderr
    assert "fatal" in r.stderr
    assert "merge-base failed" not in r.stderr
    assert "unrelated" not in r.stderr


def test_unrelated_history_refuses_at_merge_base(tree, outdir):
    """The ref resolves and shares no commit with HEAD. This is the claim the
    old message made in both cases, and it is true only here."""
    empty = _git(tree, "hash-object", "-t", "tree", "-w", "--stdin", stdin="")
    orphan = _git(tree, "commit-tree", empty, "-m", "unrelated")
    _git(tree, "update-ref", "refs/heads/feature/atomcompass_new", orphan)
    r = _snapshot(tree, outdir)
    assert r.returncode == 92, r.stderr
    assert "merge-base failed" in r.stderr
    assert "unrelated" in r.stderr
    assert "ref resolution failed" not in r.stderr


def test_remote_qualified_ref_resolves_and_names_itself(tree, outdir):
    """The tree every agent develops in: only `fork/<branch>` exists, nothing is
    set in the environment. It used to exit 92 here."""
    head = _git(tree, "rev-parse", "HEAD")
    _git(tree, "remote", "add", "fork", "https://example.invalid/ATOM.git")
    _git(tree, "update-ref", "refs/remotes/fork/feature/atomcompass_new", head)
    r = _snapshot(tree, outdir)
    assert r.returncode == 0, r.stderr
    assert "fork/feature/atomcompass_new" in r.stdout
    assert (outdir / f"compass-{head[:9]}.tar").exists()


def test_snapshot_carries_both_stamps(tree, outdir):
    """The stamps are the reason to stage with this script rather than a bare
    `git archive`: without them `gate_cpu.sh` cannot answer the GPU question and
    exits 98 on a tree that is otherwise fine."""
    head = _git(tree, "rev-parse", "HEAD")
    _git(tree, "remote", "add", "fork", "https://example.invalid/ATOM.git")
    _git(tree, "update-ref", "refs/remotes/fork/feature/atomcompass_new", head)
    assert _snapshot(tree, outdir).returncode == 0
    with tarfile.open(outdir / f"compass-{head[:9]}.tar") as tar:
        names = tar.getnames()
    assert "ATOM/.compass-commit" in names
    assert "ATOM/.compass-changed" in names


def test_bare_ref_is_preferred_and_the_fallback_is_not_claimed(tree, outdir):
    """A local branch of that name resolves on its own, and the run says nothing
    about a remote -- the fallback is reported only when it happened."""
    _git(tree, "branch", "feature/atomcompass_new")
    r = _snapshot(tree, outdir)
    assert r.returncode == 0, r.stderr
    assert "is not a ref here" not in r.stdout
    assert "(feature/atomcompass_new)" in r.stdout
