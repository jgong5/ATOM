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
import inspect
import pathlib

import pytest

from atom.compass.backends import (
    BatchView,
    CostBackend,
    CostTerm,
    FakeModel,
    HfConfig,
    Provenance,
    RequestShape,
    Species,
    StepCost,
    SyntheticStack,
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
            assert module.startswith("atom.compass.backends"), (
                f"{path.name}:{node.lineno} imports {module}"
            )


@pytest.mark.parametrize(
    "crossing",
    [BatchView, RequestShape, FakeModel, SyntheticStack, HfConfig],
)
def test_the_type_that_crosses_the_seam_is_where_the_scan_can_reach_it(crossing):
    """Every type this package hands out or takes in is defined in a file the
    scan above reads.

    The engine-free property is a property of files: the scan walks this
    package and reads nothing outside it. So a type that leaves the package
    takes its imports out of the scan's reach, and the types listed here are
    the ones whose leaving would matter -- everything this package hands out
    or takes in, not only the batch projection.

    What this catches that the scan does not is one case: a move to a package
    that is not `atom.*` at all. The scan filters on
    `module.split(".")[0] == "atom"` and skips a third-party module entirely,
    so a listed type moved to a vendor package fails here and nowhere else. A
    move that stays under `atom.*` fails the scan too, and there this is a
    second voice on the same fact rather than a new guarantee.

    What it does not catch, and what no import scan can: a config type defined
    beside the engine that reaches `FakeModel` as an argument.
    `FakeModel.__init__(self, config, ...)` is unannotated and reads attribute
    names off whatever it is handed, so that route needs no import in this
    package and leaves nothing here to read. It was tried; neither guard
    fires. The route is open, and closing it needs an instrument that reads
    what is passed rather than what is imported.
    """
    defined_in = pathlib.Path(inspect.getfile(crossing)).resolve()
    assert defined_in.parent == PACKAGE
    assert defined_in in set(PACKAGE.rglob("*.py"))
