# SPDX-License-Identifier: MIT
"""The backend seam itself: two methods, and what it is not allowed to reach.

`estimate` takes a projection of the batch -- numbers the caller prepared --
rather than the scheduled batch object, and the value of that choice is only
real if the package stays free of engine imports. So it is asserted by reading
the sources rather than trusted: a package that imports the engine is testable
only where the engine imports, which on this project means a machine with a
driver, which is where a cheap test stops being run.

The scan walks this package recursively, so a module added under it is covered
the day it is added. It reads nothing outside it, which is the limit worth
stating: the property is a property of a *package*, not something a type
inherits by being passed across the seam. Whoever defines the projection type
somewhere else owns the same assertion over the package that defines it.

Walking a root that does not resolve yields nothing, and a parametrisation of
nothing is a pass. So the non-empty case is asserted on its own: without it a
renamed or moved package collects zero cases and this file reports green, which
is the one outcome a test whose job is to pin something must not have.
"""

import ast
import pathlib

import pytest

from atom.compass.backends import (
    CostBackend,
    CostTerm,
    Provenance,
    Species,
    StepCost,
    Tier,
)

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "atom" / "compass" / "backends"


class TwoTermBackend(CostBackend):
    """The smallest thing that satisfies the interface, used to exercise it."""

    @property
    def tier(self):
        return Tier.COARSE

    def estimate(self, batch_view):
        declared = Provenance(Species.FITTED, "declared coefficients")
        return StepCost(
            [
                CostTerm("launch", 0.001, declared),
                CostTerm("tokens", 2e-6 * batch_view["tokens"], declared),
            ]
        )

    def describe(self):
        return "two-term stub, declared coefficients"


def test_the_tiers_are_the_three_cost_models():
    assert [t.value for t in Tier] == ["0", "a", "b"]


def test_a_backend_must_supply_all_three_members():
    class Partial(CostBackend):
        def estimate(self, batch_view):
            return None

    with pytest.raises(TypeError):
        Partial()


def test_a_backend_returns_a_total_with_its_parts():
    step = TwoTermBackend().estimate({"tokens": 512})
    assert [name for name, _, _ in step.rows()] == ["launch", "tokens"]
    assert step.seconds == pytest.approx(0.001 + 512 * 2e-6)
    assert not step.is_refused


def test_a_backend_says_what_it_is_answering_from():
    backend = TwoTermBackend()
    assert backend.describe()
    assert backend.tier is Tier.COARSE


def _imported_modules(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node, "." * node.level + (node.module or "")


def _backend_modules():
    # rglob, so a module added under the package is covered the day it lands.
    return sorted(PACKAGE.rglob("*.py"))


def test_the_package_was_found():
    assert _backend_modules(), f"no modules under {PACKAGE}"


def test_the_guard_finds_nothing_when_the_root_moves(monkeypatch, tmp_path):
    """The control for the guard above, which otherwise only proves it is alive.

    A guard that has never been seen failing is a liveness check: it passes
    today because the package is where it always was. Pointed at a root that
    does not resolve it must come back empty -- and the sibling module one level
    out is there so a derivation that widened past its own root would be caught
    here instead of quietly keeping the parametrisation non-empty.
    """
    (tmp_path / "sibling.py").write_text("")
    monkeypatch.setitem(globals(), "PACKAGE", tmp_path / "moved")
    assert not _backend_modules()


def test_the_walk_returns_every_module_under_the_root(monkeypatch, tmp_path):
    """Non-empty says the walk found something; this, every module of a tree built here.

    The tree is built here, so the expected set does not move when the package
    gains a module. A walk that is not recursive misses `sub/b.py`, and any
    narrowing or slice that drops a module of the built tree fails the same way.
    """
    for rel in ("__init__.py", "a.py", "sub/b.py"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("")
    monkeypatch.setitem(globals(), "PACKAGE", tmp_path)
    modules = {str(p.relative_to(tmp_path)) for p in _backend_modules()}
    assert modules == {"__init__.py", "a.py", "sub/b.py"}


@pytest.mark.parametrize(
    "path",
    _backend_modules(),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_the_package_imports_nothing_from_the_engine(path):
    """Only its own modules, so it runs anywhere Python does."""
    for node, module in _imported_modules(path):
        if module.startswith(".") or module.split(".")[0] == "atom":
            assert module.startswith(
                "atom.compass.backends"
            ), f"{path.name}:{node.lineno} imports {module}"


def test_every_module_the_walk_returns_is_a_case():
    """The cases are compared to the walk, not only the walk to its tree.

    A slice or filter where the walk is handed to the parametrisation drops
    cases while the walk itself still returns every module.
    """
    (mark,) = test_the_package_imports_nothing_from_the_engine.pytestmark
    assert mark.args[1] == _backend_modules()
