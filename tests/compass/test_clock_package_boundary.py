# SPDX-License-Identifier: MIT
"""What `atom.compass.clock` may reach, and what it may write down.

Package-wide rather than per-module, and in a file of its own for that reason:
these were the last two checks in the identity tests and they were never about
identity. A subpackage added later has to be covered by them, which is what
`rglob` is for, and a reader looking for the rule about subpackages should not
have to find it under a heading about names and orderings.

Three separate claims, and they are worth keeping apart because two of them are
much weaker than the third and saying so is the point.

**What the package may reach is an allowlist, and it is a proof.** The claim is
that these structures can be built and read on any machine -- no device
runtime, no wall-clock read, no socket -- and an allowlist keeps it without
having to predict the name of the next import that would break it. It is
widened per module, not package-wide: "somewhere in here a socket is opened" is
a far weaker statement than "this file opens one", and a widening that names a
module or an import that does not exist is itself refused.

**Where a clock is, is confined, and the confinement is close to a proof.**
Something has to name a location or nothing reaches a clock across a container
boundary. What must not name one is the surface a participant uses. So the
calls that turn a written-down endpoint into a way of reaching a clock are
carved out by name, together with the objects they hand back, and the two
modules that carry a frame may say it in their own source; every other export
and every other module may not. Both directions of the carve-out are checked,
so it cannot rot into a blanket permission.

**How deep a clock sits is a vocabulary check, and it is the weakest of the
three. It is not a prohibition and this file should not be read as one.** A
clock inserted between two others is invisible to both sides only while neither
can write down a level, and a participant is addressed by identity only while
nothing numbers one -- but "a level" has no closed spelling. `tier` and `hop`
are in the list because they were found missing; the next synonym will not be.
`stage` cannot be in it at all, because a link class is named for pipeline
stages and the package has to be able to say so. What this buys is that the
ordinary spellings cannot be used without somebody deciding to, which is worth
having and is not the same as a guarantee.

The scans read the exported surface by introspection and the source by syntax,
because neither sees what the other does. A signature never shows a private
attribute or a string used as a name; syntax never shows what an export
actually resolves to.
"""

import ast
import dataclasses
import enum
import inspect
import re
from pathlib import Path

import pytest

from atom.compass import clock
from atom.compass.clock import transport

CLOCK_PACKAGE = Path(__file__).resolve().parents[2] / "atom" / "compass" / "clock"

# Where a peer is. A participant may not write one of these down: it knows a
# name. The calls that turn a written-down endpoint into a way of reaching a
# clock obviously must, and they are the short list below.
PLACING_WORDS = (
    "addr",
    "address",
    "endpoint",
    "host",
    "hostname",
    "ip",
    "node",
    "port",
    "sock",
    "socket",
    "uri",
    "url",
)

# How deep a clock sits, and which of several it is. No module is exempt,
# including the transport: nothing in a carriage needs a level either, since it
# moves a frame between two ends and neither end has a position. A vocabulary
# and not a proof -- see this module's docstring, and do not quote a green run
# of these two tests as one.
RANKING_WORDS = ("depth", "hop", "index", "level", "parent", "rank", "tier")

# The exports that turn a written-down endpoint into something usable, plus the
# objects they hand back. They are the only part of the surface allowed to place
# a clock, they are checked below to actually do so -- an allowance granted for
# no reason weakens the statement -- and a level is still forbidden to them.
RESOLVING_SURFACE = (
    "DEFAULT_ENDPOINT",
    "InProcessCarrier",
    "InProcessServer",
    "carrier_for",
    "connect",
    "serve",
)

# Which modules may write a location down at all. `stream` opens the socket and
# `resolve` picks between a socket and a direct call, so both must; the
# transport's front door re-exports what `resolve` names, so the word travels
# through it. Everything else -- the whole of the clock proper, the wire format,
# the rule's own entry point and the participant's session -- may not.
MAY_PLACE_A_CLOCK = (
    "transport/__init__.py",
    "transport/resolve.py",
    "transport/stream.py",
)

#: The standard library every module in the package may reach.
ALLOWED_IMPORTS = ("dataclasses", "enum", "math")

#: What one module may reach beyond that, and only that module.
EXTRA_IMPORTS = {
    "transport/message.py": ("json",),
    "transport/stream.py": ("socket", "threading"),
}

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _words(identifier):
    """An identifier's lowercase word parts, split on underscores and on case.

    Matching whole words rather than substrings, and carrying the near-spellings
    in the lists above instead. Substring matching catches `hostname`, which is
    worth catching, but it also catches `transport`, `report` and `support`,
    and a guard that fires on an innocent name is a guard people edit around.
    """
    return tuple(part.lower() for part in _WORD.findall(identifier))


def _places(identifier):
    return any(word in PLACING_WORDS for word in _words(identifier))


def _ranks(identifier):
    return any(word in RANKING_WORDS for word in _words(identifier))


# --- the exported surface -----------------------------------------------------


def _parameters(obj):
    """The parameter names of a callable, or none where it has no signature.

    A class deriving straight from a builtin exception has no introspectable
    one. Such a class takes whatever the builtin takes, so there is no name in
    it to read and nothing is lost; its own name is still read by the caller.
    """
    try:
        return list(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return []


def _public_surface(module):
    """(what it is called, every name it writes down) for a module's exports.

    Three declaration forms have to be unwrapped before a name can be read --
    a `classmethod`, a `staticmethod` and a `property` are none of them
    functions, so a scan that filters on `inspect.isfunction` drops all three
    before it looks at anything. The attribute's own name is read as well as its
    parameters, because an alternative constructor and a property are the first
    two places an endpoint gets hung. Enum classes are visited for their
    members and methods; their call is a value lookup rather than a constructor.
    Every exported dataclass is read for its fields, not a hard-coded two, and
    that catches a field declared `init=False`, which no signature shows.
    Annotations are read directly as well: a class whose fields are declared
    and never assigned -- a `NamedTuple`, a `TypedDict`, a bare annotated class
    -- puts nothing in `vars()` and shows nothing in a signature.
    """
    surface = []
    for exported in sorted(module.__all__):
        obj = getattr(module, exported)
        if not inspect.isclass(obj):
            names = [exported]
            if callable(obj):
                names += _parameters(obj)
            surface.append((exported, tuple(names)))
            continue
        if issubclass(obj, enum.Enum):
            surface.append((exported, (exported,)))
        else:
            surface.append((exported, (exported,) + tuple(_parameters(obj))))
        if dataclasses.is_dataclass(obj):
            surface.append(
                (
                    f"{exported} fields",
                    tuple(field.name for field in dataclasses.fields(obj)),
                )
            )
        surface.append(
            (
                f"{exported} annotations",
                tuple(sorted(getattr(obj, "__annotations__", {}))),
            )
        )
        for attr_name, attr in sorted(vars(obj).items()):
            if attr_name.startswith("_"):
                continue
            names = [attr_name]
            if isinstance(attr, (classmethod, staticmethod)):
                attr = attr.__func__
            elif isinstance(attr, property):
                attr = attr.fget
            if inspect.isfunction(attr):
                names += _parameters(attr)
            surface.append((f"{exported}.{attr_name}", tuple(names)))
    return surface


def test_the_surface_a_participant_uses_never_says_where_a_clock_is():
    # The import allowlist below proves the package reaches no device, clock or
    # socket. It does not prove this: a module taking host, port and level and
    # storing them would pass it unchanged. Stated mechanically rather than
    # upheld by inspection, because the pressure to hang an endpoint somewhere
    # convenient arrives with the transport -- and it has now arrived, which is
    # why the resolving calls are carved out by name rather than the whole check
    # being loosened.
    offenders = []
    for module in (clock, transport):
        for name, written in _public_surface(module):
            if name.split(".")[0].split(" ")[0] in RESOLVING_SURFACE:
                continue
            offenders += [f"{name}({word})" for word in written if _places(word)]
    assert not offenders, (
        f"the surface a participant uses says where a clock is: {offenders}. A "
        "participant knows a name; an endpoint belongs to whatever resolves one."
    )


def test_no_public_call_anywhere_uses_the_vocabulary_of_a_level():
    # The half of the old check that keeps no exemption, and the reason it
    # does: the resolving calls need a location, and no part of the package
    # needs a level. A vocabulary rather than a proof -- see the module
    # docstring -- so read a failure as evidence and a pass as silence.
    offenders = []
    for module in (clock, transport):
        for name, written in _public_surface(module):
            offenders += [f"{name}({word})" for word in written if _ranks(word)]
    assert not offenders, f"the interface says how deep a clock sits: {offenders}"


def test_the_resolving_surface_is_the_one_that_places_a_clock():
    # Both directions. Every name carved out is exported, so the allowance
    # cannot go stale; and the three calls really do take an endpoint, so the
    # allowance is not a permission granted for no reason.
    missing = [name for name in RESOLVING_SURFACE if name not in transport.__all__]
    assert not missing, f"{missing} are carved out and not exported"
    for call in ("carrier_for", "connect", "serve"):
        assert "endpoint" in inspect.signature(getattr(transport, call)).parameters


# --- the source ---------------------------------------------------------------


def _clock_modules():
    # rglob, not glob: the transport sits below the top level, and a guard that
    # stopped there would go quiet on exactly the code it exists for. Each
    # module is one parametrised case, so a new one shows up in the gate
    # arithmetic rather than silently.
    return sorted(CLOCK_PACKAGE.rglob("*.py"))


def _named(path):
    # The path within the package, never the bare filename: under rglob the
    # filename is not unique -- there are two `__init__.py` -- and pytest would
    # disambiguate them positionally, so a failure would name neither.
    return path.relative_to(CLOCK_PACKAGE).as_posix()


def test_the_package_was_found():
    assert _clock_modules(), f"no modules under {CLOCK_PACKAGE}"


def test_every_widened_import_names_a_module_that_exists_and_uses_it():
    # An allowance for a module that has been renamed away, or for an import a
    # module no longer makes, is a permission granted for no reason and it
    # weakens every other line of the allowlist.
    present = [_named(path) for path in _clock_modules()]
    for name, extra in sorted(EXTRA_IMPORTS.items()):
        assert name in present, f"{name} is widened and does not exist"
        source = (CLOCK_PACKAGE / name).read_text()
        for root in extra:
            assert f"import {root}" in source, (
                f"{name} is allowed {root} and imports it nowhere"
            )


@pytest.mark.parametrize("module", _clock_modules(), ids=_named)
def test_the_package_imports_only_the_standard_library_it_names(module):
    allowed = ALLOWED_IMPORTS + EXTRA_IMPORTS.get(_named(module), ())
    tree = ast.parse(module.read_text())
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.append((node.module or "").split(".")[0])
    strays = sorted(dict.fromkeys(root for root in roots if root not in allowed))
    assert not strays, (
        f"{_named(module)} imports {strays}; allowed here: {sorted(allowed)}"
    )


def _identifiers(module):
    """Every name a module writes down, in syntax or as a string standing for one.

    The string half is not a refinement. A name reached indirectly is still a
    name: `{"host": ...}` as a payload key, `getattr(clock, "level")`,
    `setattr(self, "endpoint", ...)` -- each writes the word down and none of
    them puts an identifier in the syntax tree. Only strings that could *be* an
    identifier are read, which is what keeps prose out: a docstring or a message
    has spaces in it and is skipped, while `listen_port` is not and is not.
    """
    tree = ast.parse(module.read_text())
    written = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            written.append(node.id)
        elif isinstance(node, ast.arg):
            written.append(node.arg)
        elif isinstance(node, ast.Attribute):
            written.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            written.append(node.name)
        elif isinstance(node, ast.alias):
            written.append(node.asname or node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            written.append(node.arg)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.isidentifier()
        ):
            written.append(node.value)
    return written


@pytest.mark.parametrize("module", _clock_modules(), ids=_named)
def test_only_the_modules_that_carry_a_frame_say_where_a_clock_is(module):
    # The public scan above covers the exported surface. This covers the rest of
    # it: a location hung on a private attribute, or spelled as a string and
    # reached with getattr, is exactly as much of a leak and no signature shows
    # either. Two modules open or choose a carriage and must name a location to
    # do it; the wire format, the rule's entry point and the participant's
    # session are on the other side of that line and stay clean.
    if _named(module) in MAY_PLACE_A_CLOCK:
        return
    offenders = sorted(
        dict.fromkeys(name for name in _identifiers(module) if _places(name))
    )
    assert not offenders, (
        f"{_named(module)} says where a clock is: {offenders}. Only "
        f"{list(MAY_PLACE_A_CLOCK)} may; everything else knows a name."
    )


@pytest.mark.parametrize("module", _clock_modules(), ids=_named)
def test_no_module_at_all_uses_the_vocabulary_of_a_level(module):
    # No exemption, for any module. A vocabulary and not a proof: `stage` is not
    # in the list and cannot be, because a link class is named for pipeline
    # stages.
    offenders = sorted(
        dict.fromkeys(name for name in _identifiers(module) if _ranks(name))
    )
    assert not offenders, (
        f"{_named(module)} says how deep a clock sits, or numbers a "
        f"participant: {offenders}"
    )


@pytest.mark.parametrize("module", _clock_modules(), ids=_named)
def test_the_package_builds_no_set_at_all(module):
    # A `set` of names iterates in hash order, and string hashing is randomised
    # per process, so iterating one makes two runs of one configuration diverge
    # with no error and no warning. Membership against a set would be fine, but
    # "built here and only ever tested against" is not a property this check can
    # see -- and the package has no use for one -- so the rule it enforces is the
    # one it can prove: none is constructed, so none can be iterated.
    tree = ast.parse(module.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.SetComp)):
            offenders.append(f"{type(node).__name__} at line {node.lineno}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("set", "frozenset")
        ):
            offenders.append(f"{node.func.id}() at line {node.lineno}")
    assert not offenders, (
        f"{_named(module)} builds {offenders}; use a dict with None values as "
        "an ordered set, or sort at the point of iteration"
    )
