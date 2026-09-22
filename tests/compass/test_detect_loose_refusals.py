# SPDX-License-Identifier: MIT
"""The loose-refusal detector's counting rule, pinned over fixture sources.

`tools/compass/detect_loose_refusals.py` counts, for a test's `match=` needle,
how many production raise templates it matches. These tests exercise that rule
against short source strings written here -- never against `atom/compass/`.

Asserting a count for the live tree would freeze an aggregate: the number would
be re-blessed on every change, and it would stay green while the tree grew the
ambiguity the tool exists to find. The tool is run by hand and read by a person;
what a test can pin is the rule it applies, which is what these do.

Both directions are pinned, because a detector that reports nothing passes
forever. Every fixture pair below is one production source that must be flagged
and one that must not, under the same needle.
"""

import ast
import importlib.util
from pathlib import Path

TOOL = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "compass"
    / "detect_loose_refusals.py"
)
_spec = importlib.util.spec_from_file_location("detect_loose_refusals", TOOL)
detector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(detector)

# Two refusals in one function, sharing the words the test asserts on.
COLLIDING = """
def kv_heads_per_rank(heads, ranks):
    if heads > ranks:
        raise ValueError(f"{heads} KV heads do not divide across {ranks} ranks")
    raise ValueError(f"{ranks} ranks do not divide across {heads} KV heads")
"""

# The same two branches, each naming a fault the other cannot.
DISTINGUISHED = """
def kv_heads_per_rank(heads, ranks):
    if heads > ranks:
        raise ValueError(f"{heads} KV heads do not divide across {ranks} ranks")
    raise ValueError(f"{ranks} ranks replicate {heads} KV heads unevenly")
"""

# Two refusals sharing words, but in unrelated functions: broad, never sharp.
UNRELATED = """
def kv_heads_per_rank(heads, ranks):
    raise ValueError(f"{heads} KV heads do not divide across {ranks} ranks")


def block_rows(rows, ranks):
    raise ValueError(f"{rows} rows do not divide across {ranks} ranks")
"""

# A refusal whose first argument is a rule enum, its text in later arguments.
DEFERRED_TEXT = """
def _missing(field):
    if field.runtime:
        raise SpecRefusal(Rule.NO_DEFAULTS, f"`{field.path}` is missing", "measure it")
    raise SpecRefusal(Rule.SHAPE, f"`{field.path}` is missing", "state it in the spec")
"""

OPAQUE = """
def price(term, source):
    raise CostRefused(term, source)
"""

DIVIDE_NEEDLE = """
import pytest


def test_a_width_that_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="do not divide across"):
        kv_heads_per_rank(6, 4)
"""

SPANNING_NEEDLE = """
import pytest


def test_a_width_that_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="6 KV heads do not divide across 4 ranks"):
        kv_heads_per_rank(6, 4)
"""

MISSING_NEEDLE = """
import pytest


def test_an_undeclared_field_is_refused():
    with pytest.raises(SpecRefusal, match="is missing"):
        _missing(field)
"""

RENDERED_NEEDLE = """
import pytest


def test_a_width_that_does_not_divide_is_refused():
    heads = 6
    with pytest.raises(ValueError, match=f"{heads} KV heads do not divide"):
        kv_heads_per_rank(heads, 4)
"""


def _scan(production, tests):
    """Broad flags, sharp flags and unreadable needles for one fixture pair."""
    sites, _ = detector.raise_sites(production, "geometry.py")
    needles, dropped = detector.match_needles(tests, "test_geometry.py")
    broad, sharp = detector.score(sites, needles)
    return broad, sharp, dropped


def _sharp_groups(sharp):
    """The (scope, lines) of every within-function group in a sharp flag."""
    return [
        (scope, [site["line"] for site in group])
        for _, groups in sharp
        for (_, scope), group in groups.items()
    ]


def test_two_refusals_in_one_function_are_flagged_sharp():
    broad, sharp, _ = _scan(COLLIDING, DIVIDE_NEEDLE)
    assert len(broad) == 1
    assert _sharp_groups(sharp) == [("kv_heads_per_rank", [4, 5])]


def test_two_distinguishable_refusals_are_not_flagged():
    broad, sharp, _ = _scan(DISTINGUISHED, DIVIDE_NEEDLE)
    assert broad == []
    assert sharp == []


def test_refusals_in_unrelated_functions_are_broad_but_not_sharp():
    broad, sharp, _ = _scan(UNRELATED, DIVIDE_NEEDLE)
    assert len(broad) == 1
    assert sharp == []


def test_a_message_in_a_later_argument_is_read():
    """A refusal raised as `Refusal(Rule.X, what, remedy)` states nothing in its
    first argument, so reading only that argument drops the site before any
    needle is tried. Every string-bearing argument is read instead."""
    raised = next(
        node
        for node in ast.walk(ast.parse(DEFERRED_TEXT))
        if isinstance(node, ast.Raise)
    )
    assert detector.literal_runs(raised.exc.args[0]) == []
    _, sharp, _ = _scan(DEFERRED_TEXT, MISSING_NEEDLE)
    assert _sharp_groups(sharp) == [("_missing", [4, 5])]


def test_a_needle_spanning_an_interpolation_matches_nothing():
    """An interpolation ends a literal run, so a needle written across a hole
    asserts on a sentence the template never states as one piece."""
    broad, sharp, _ = _scan(COLLIDING, SPANNING_NEEDLE)
    assert broad == []
    assert sharp == []


def test_a_non_constant_needle_is_named_rather_than_dropped():
    """The rendered text of an f-string needle is unknown here, so it leaves the
    population -- and a flag that vanishes that way is not a cleared one."""
    broad, sharp, dropped = _scan(COLLIDING, RENDERED_NEEDLE)
    assert (broad, sharp) == ([], [])
    assert [(entry["line"], entry["why"]) for entry in dropped] == [
        (7, "not a string constant (JoinedStr)")
    ]


def test_a_refusal_stating_no_text_is_named_not_swallowed():
    """Unreadable needles are named in the output; unreadable refusals are the
    other half of the same population and are named the same way."""
    sites, opaque = detector.raise_sites(OPAQUE, "cost.py")
    assert sites == []
    assert [(entry["line"], entry["why"]) for entry in opaque] == [
        (3, "no text in any argument (Name)")
    ]


def _fixture_tree(root):
    """A miniature tree with the two subpaths the tool reads."""
    for part, source in (("atom", COLLIDING), ("tests", DIVIDE_NEEDLE)):
        package = root / part / "compass"
        package.mkdir(parents=True)
        (package / "sample.py").write_text(source)
    return root


def test_a_root_that_resolves_to_nothing_is_refused(tmp_path, capsys):
    """Zero sites and zero needles print as a clean run, and the root is typed by
    hand -- so an empty population is refused rather than reported."""
    assert detector.main(tmp_path) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_a_root_carrying_both_populations_is_read(tmp_path, capsys):
    assert detector.main(_fixture_tree(tmp_path)) == 0
    assert "SHARP  (>= 2 matching raise sites in ONE function): 1" in (
        capsys.readouterr().out
    )
