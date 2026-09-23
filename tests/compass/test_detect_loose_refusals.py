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

What the tool states is pinned on its output, not only on its return values.
A population the tool computes and never prints is one the person reading it
never learns of, so the counts and the per-site lines are asserted as text.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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


PRODUCTION_SOURCES = {"sample.py": COLLIDING, "opaque.py": OPAQUE}
NEEDLE_SOURCES = {"sample.py": DIVIDE_NEEDLE}


def _fixture_tree(root, parts=("atom", "tests")):
    """A miniature tree carrying the named halves of the tool's population.

    The production half carries both kinds of refusal the tool separates: two
    that state text and collide under one needle, and one that states none.
    Naming fewer parts builds the wrong roots a hand-typed argument produces --
    the package without the tests, or the tests without the package.
    """
    for part, sources in (("atom", PRODUCTION_SOURCES), ("tests", NEEDLE_SOURCES)):
        if part not in parts:
            continue
        package = root / part / "compass"
        package.mkdir(parents=True)
        for name, source in sources.items():
            (package / name).write_text(source)
    return root


@pytest.mark.parametrize(
    "parts", [(), ("atom",), ("tests",)], ids=["neither", "atom", "tests"]
)
def test_a_root_missing_either_population_is_refused(tmp_path, capsys, parts):
    """Zero sites or zero needles print as a clean run, and the root is typed by
    hand -- so either half being empty is refused rather than reported.

    The asymmetric roots are the realistic typos: pointing at `atom/` rather
    than the repo root leaves the needles empty, and a tree carrying the package
    without `tests/compass/` leaves them empty the other way. Both must refuse,
    not just the root that is empty on both sides.
    """
    assert detector.main(_fixture_tree(tmp_path, parts)) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_a_root_carrying_both_populations_is_read(tmp_path, capsys):
    """Both populations are counted and each unreadable site is named.

    A count with no per-site line beside it is the aggregate this output exists
    to decompose, so the opaque refusal is asserted as printed text, not only as
    a return value the person running the tool never sees.
    """
    assert detector.main(_fixture_tree(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "production refusals stating no text:  1" in out
    assert "  ! atom/compass/opaque.py:3  no text in any argument (Name)" in out


def test_a_broad_flag_is_printed_with_every_site_it_matched(tmp_path, capsys):
    assert detector.main(_fixture_tree(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "BROAD  (needle matches >= 2 production raise sites): 1" in out
    assert (
        "  tests/compass/sample.py:6  'do not divide across' -> "
        "atom/compass/sample.py:4, atom/compass/sample.py:5\n"
    ) in out


def test_a_sharp_flag_is_printed_with_its_function_and_lines(tmp_path, capsys):
    assert detector.main(_fixture_tree(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "SHARP  (>= 2 matching raise sites in ONE function): 1" in out
    assert "  tests/compass/sample.py:6  'do not divide across'\n" in out
    assert "      atom/compass/sample.py::kv_heads_per_rank at 4, 5\n" in out


# A refusal sharing the fixture needle's words, for a package two levels down.
NESTED_ROWS = """
def block_rows(rows, ranks):
    raise ValueError(f"{rows} rows do not divide across {ranks} ranks")
"""


def test_modules_in_nested_packages_are_read(tmp_path, capsys):
    """Both walks recurse: a raise site and a needle two directories below
    `atom/compass/` and `tests/compass/` are counted and printed like the rest.
    A walk that stops at the top directory drops them without a word."""
    root = _fixture_tree(tmp_path)
    for part, name, source in (
        ("atom", "rows.py", NESTED_ROWS),
        ("tests", "test_rows.py", RENDERED_NEEDLE),
    ):
        package = root / part / "compass" / "sub" / "pkg"
        package.mkdir(parents=True)
        (package / name).write_text(source)
    assert detector.main(root) == 0
    out = capsys.readouterr().out
    assert (
        "  tests/compass/sample.py:6  'do not divide across' -> "
        "atom/compass/sample.py:4, atom/compass/sample.py:5, "
        "atom/compass/sub/pkg/rows.py:3\n"
    ) in out
    assert (
        "  ! tests/compass/sub/pkg/test_rows.py:7  "
        "not a string constant (JoinedStr)\n"
    ) in out


@pytest.mark.parametrize("extra", [False, True], ids=["none", "two"])
def test_a_wrong_argument_count_is_refused_with_a_usage_line(tmp_path, extra):
    """The root is a hand-typed argument, so omitting it must say what to type
    rather than raise `IndexError` out of `sys.argv` with no word about why.
    A second argument is refused too: the tool would read the first and ignore
    the rest, reporting over a root the caller may not have meant."""
    argv = [str(_fixture_tree(tmp_path)), "extra"] if extra else []
    done = subprocess.run(
        [sys.executable, str(TOOL), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert done.stderr.startswith("usage: ")
    assert "REFUSED" not in done.stderr
