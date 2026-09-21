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


@pytest.mark.parametrize(
    "path",
    sorted(PACKAGE.rglob("*.py")),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_the_package_imports_nothing_from_the_engine(path):
    """Only its own modules, so it runs anywhere Python does."""
    for node, module in _imported_modules(path):
        if module.startswith(".") or module.split(".")[0] == "atom":
            assert module.startswith(
                "atom.compass.backends"
            ), f"{path.name}:{node.lineno} imports {module}"
