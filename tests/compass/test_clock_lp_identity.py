# SPDX-License-Identifier: MIT
"""`atom.compass.clock`: identity, the total order, and the lookahead matrix.

These three structures are read by the rule that hands out virtual time, so
their failure modes are the rule's failure modes. What each test here is
defending:

* **Identity, not position.** The matrix is addressed by name at every entry
  point. A participant inserted later must not move anybody else's entry, and
  the test that proves it inserts one alphabetically between two existing names
  and re-reads the entries declared before it arrived.
* **A total order that does not depend on the process.** Sorting identities is a
  code-point comparison, so it gives one answer everywhere. The cross-process
  half of that claim cannot be made in this file -- two runs inside one
  interpreter share one hash seed, so a test written that way would pass against
  the exact defect it exists to catch. It lives in
  `test_clock_order_across_processes.py`, which spawns interpreters with
  different `PYTHONHASHSEED` values.
* **An undeclared floor is a refusal, not a zero, and not a gap either.** Zero
  is a decision; silence is a link nobody sized. A caller minimises over a whole
  row, so a peer quietly missing from that row raises the minimum instead of
  lowering it -- more time handed out, not less, which is the one direction that
  is unsafe. `inbound` refuses an incomplete row and `require_complete` refuses
  an incomplete matrix before a run starts.
* **A declared floor cannot be rewritten.** The accessors hand out the link
  object, so it is frozen; otherwise every refusal in `declare` is reachable
  around it.
* **A zero floor is accepted.** It costs overlap, not correctness, and a module
  that rejected it would be asserting the opposite.
* **The package reaches nothing, and names nothing.** No device runtime, no
  clock, no socket -- an allowlist over the package's own imports, extended per
  module where a module genuinely needs more. That is a different claim from
  naming no location and no level, which a module taking `host`, `port` and
  `level` would satisfy while breaking; the two scans below cover the second.
* **A location is confined; a level is forbidden outright.** These are not the
  same requirement and were one list until a transport arrived and needed one of
  the two words. Something has to say where a clock is or nothing can reach one
  across a container boundary -- so a written-down endpoint is allowed, in the
  handful of calls that turn one into a way of reaching a clock and in the two
  modules that carry it, and nowhere else. How deep a clock sits, and which of
  several it is, stay forbidden everywhere including the transport: a clock
  inserted between two others has to be invisible to both sides, and it stops
  being invisible the moment anything can name a level or number a participant.
"""

import ast
import dataclasses
import enum
import inspect
import re
from pathlib import Path

import pytest

from atom.compass import clock
from atom.compass.clock import (
    TRAFFIC_TO_ENGINE_FLOOR_SECONDS,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
    transport,
)

CLOCK_PACKAGE = Path(__file__).resolve().parents[2] / "atom" / "compass" / "clock"

TRAFFIC = LpId("traffic-source")
PREFILL = LpId("prefill")
DECODE = LpId("decode")


def _registry(*ids):
    registry = LpRegistry()
    for lp_id in ids:
        registry.register(lp_id)
    return registry


# --- identity ----------------------------------------------------------------


def test_an_identity_is_its_name():
    assert str(LpId("decode")) == "decode"
    assert LpId("decode") == LpId("decode")
    assert {LpId("decode"): 1}[LpId("decode")] == 1


@pytest.mark.parametrize("bad", ["", " ", "pp stage 0", "decode\t", "\ndecode"])
def test_a_name_that_would_not_survive_a_timeline_row_is_refused(bad):
    # The timeline record is one field per column, so an embedded space or
    # newline would split a row rather than fail. Refused where it is created.
    with pytest.raises(ValueError):
        LpId(bad)


def test_a_name_must_be_a_string():
    with pytest.raises(TypeError):
        LpId(3)


def test_identities_order_by_name():
    # Ordering is what the tie-break between two eligible participants uses, so
    # it has to be available on the identity itself and not only on a registry.
    assert sorted([DECODE, TRAFFIC, PREFILL]) == [DECODE, PREFILL, TRAFFIC]
    assert min([TRAFFIC, DECODE]) == DECODE


# --- the registry and its total order ----------------------------------------


def test_the_order_does_not_depend_on_the_order_of_registration():
    forward = _registry(TRAFFIC, PREFILL, DECODE)
    backward = _registry(DECODE, PREFILL, TRAFFIC)
    assert forward.ids() == backward.ids() == (DECODE, PREFILL, TRAFFIC)
    assert list(forward) == list(backward)


def test_registering_a_name_twice_is_refused():
    # Two participants that believe they are the same one would have their
    # clocks merged silently, which is the failure this refusal exists for.
    registry = _registry(DECODE)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(LpId("decode"))


def test_the_registry_refuses_a_bare_string():
    # A str would compare against an LpId only by accident, and sorting a mix of
    # the two raises -- better at the call that introduced it.
    registry = LpRegistry()
    with pytest.raises(TypeError):
        registry.register("decode")


def test_an_unregistered_identity_is_named_together_with_the_ones_that_exist():
    registry = _registry(TRAFFIC, DECODE)
    with pytest.raises(KeyError) as excinfo:
        registry.require(PREFILL)
    assert "prefill" in str(excinfo.value)
    assert "decode" in str(excinfo.value) and "traffic-source" in str(excinfo.value)


def test_require_refuses_a_bare_string_rather_than_reporting_it_missing():
    # `require` is what `declare`, `lookahead` and `inbound` all funnel through,
    # so it is the check a caller actually meets. Without the type test a bare
    # "decode" is reported as not registered next to the registered `decode`,
    # which reads as a bug in the registry rather than in the call.
    registry = _registry(DECODE)
    with pytest.raises(TypeError):
        registry.require("decode")


def test_membership_and_size():
    registry = _registry(TRAFFIC, DECODE)
    assert TRAFFIC in registry and PREFILL not in registry
    assert len(registry) == 2


# --- the lookahead matrix ----------------------------------------------------


def test_a_declared_floor_is_what_the_matrix_answers():
    registry = _registry(TRAFFIC, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, DECODE, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    assert matrix.lookahead(TRAFFIC, DECODE) == pytest.approx(9.0e-3)


def test_a_link_is_directional():
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 2.0e-3)
    # The reverse path is a different path and is not implied by this one.
    with pytest.raises(KeyError):
        matrix.lookahead(DECODE, PREFILL)


def test_an_undeclared_pair_is_refused_rather_than_read_as_zero():
    registry = _registry(TRAFFIC, DECODE)
    matrix = LookaheadMatrix(registry)
    with pytest.raises(KeyError, match="no declared lookahead floor"):
        matrix.lookahead(TRAFFIC, DECODE)
    assert matrix.undeclared() == ((DECODE, TRAFFIC), (TRAFFIC, DECODE))


def test_a_zero_floor_is_a_declaration_and_not_an_error():
    # Zero serializes the pair. It stays correct, so it is accepted and reported
    # rather than rejected -- the floor buys speed, never safety.
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    link = matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 0.0)
    assert matrix.lookahead(PREFILL, DECODE) == 0.0
    assert matrix.serializing() == (link,)


def test_a_nonzero_floor_is_not_reported_as_serializing():
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)
    assert matrix.serializing() == ()


@pytest.mark.parametrize("bad", [-1.0e-9, float("nan"), float("inf")])
def test_a_floor_that_is_not_a_finite_non_negative_duration_is_refused(bad):
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    with pytest.raises(ValueError):
        matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, bad)


def test_a_participant_has_no_link_to_itself():
    registry = _registry(DECODE)
    matrix = LookaheadMatrix(registry)
    with pytest.raises(ValueError, match="no link to itself"):
        matrix.declare(DECODE, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)


def test_declaring_the_same_link_twice_is_refused():
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)
    with pytest.raises(ValueError, match="already declared"):
        matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 2.0e-3)


def test_a_link_to_an_unregistered_participant_is_refused():
    registry = _registry(DECODE)
    matrix = LookaheadMatrix(registry)
    with pytest.raises(KeyError):
        matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)


def test_a_link_class_is_not_a_string():
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    with pytest.raises(TypeError):
        matrix.declare(PREFILL, DECODE, "prefill_to_decode", 1.0e-3)


def test_a_row_missing_a_peer_is_refused_rather_than_handed_over_short():
    # The asymmetry that makes this a refusal and not a convenience: a caller
    # takes a minimum over the whole row, over every registered peer. A peer
    # left out of the row does not contribute a small term -- it contributes no
    # term, so the minimum comes out HIGHER than it should and more time is
    # handed out, not less. Zero would have been the safe mistake; silence is
    # not. Same distinction `lookahead` already makes, on the path a caller
    # actually walks.
    registry = _registry(TRAFFIC, PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 2.0e-3)
    with pytest.raises(KeyError) as excinfo:
        matrix.inbound(DECODE)
    assert "traffic-source -> decode" in str(excinfo.value)
    # The declared leg still answers; only the row is refused.
    assert matrix.lookahead(PREFILL, DECODE) == pytest.approx(2.0e-3)


def test_a_complete_row_has_one_entry_for_every_registered_peer():
    registry = _registry(TRAFFIC, PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 2.0e-3)
    matrix.declare(TRAFFIC, DECODE, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    assert len(matrix.inbound(DECODE)) == len(registry) - 1


def test_a_lone_participant_has_an_empty_row_rather_than_a_refusal():
    registry = _registry(DECODE)
    assert LookaheadMatrix(registry).inbound(DECODE) == ()


def test_require_complete_names_every_missing_pair_at_once():
    # Run once when set-up finishes: an incomplete matrix should fail before
    # anything advances, not one step later as an event landing in the past.
    registry = _registry(TRAFFIC, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, DECODE, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    with pytest.raises(KeyError) as excinfo:
        matrix.require_complete()
    assert "decode -> traffic-source" in str(excinfo.value)


def test_require_complete_accepts_a_matrix_with_every_pair_declared():
    registry = _registry(TRAFFIC, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, DECODE, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    matrix.declare(DECODE, TRAFFIC, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    assert matrix.require_complete() is None
    assert matrix.undeclared() == ()


def test_a_declared_floor_cannot_be_rewritten_through_a_handed_out_link():
    # `links`, `inbound` and `declare` all return the object itself. If it were
    # writable, every refusal in `declare` -- negative, NaN, infinite, already
    # declared -- would be reachable around.
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    link = matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 2.0e-3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        link.floor_seconds = 99.0
    assert matrix.lookahead(PREFILL, DECODE) == pytest.approx(2.0e-3)


def test_inbound_links_come_back_in_the_total_order_whatever_order_they_were_declared():
    # This is the shape the grant rule reads: every floor into one participant.
    # It has to be ordered by identity, because a sum or a min taken over it in
    # declaration order would depend on set-up.
    registry = _registry(TRAFFIC, PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, DECODE, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)
    assert tuple(link.source for link in matrix.inbound(DECODE)) == (PREFILL, TRAFFIC)


def test_inserting_a_participant_moves_nothing_that_was_already_declared():
    # `pp-stage-1` sorts between `decode` and `traffic-source`, so under any
    # index-based addressing it would displace one of them. Under identity the
    # entries declared before it arrived answer exactly as before.
    registry = _registry(TRAFFIC, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, DECODE, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    before = matrix.lookahead(TRAFFIC, DECODE)

    inserted = LpId("pp-stage-1")
    registry.register(inserted)
    matrix.declare(inserted, DECODE, LinkClass.PIPELINE_STAGE_TO_STAGE, 5.0e-6)

    assert registry.ids() == (DECODE, inserted, TRAFFIC)
    assert matrix.lookahead(TRAFFIC, DECODE) == before
    assert tuple(link.source for link in matrix.inbound(DECODE)) == (inserted, TRAFFIC)


def test_the_tightest_link_is_the_one_that_bounds_the_run():
    registry = _registry(TRAFFIC, PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, PREFILL, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)
    assert matrix.tightest().link_class is LinkClass.PREFILL_TO_DECODE
    assert LookaheadMatrix(LpRegistry()).tightest() is None


def test_links_are_listed_in_a_stable_order():
    registry = _registry(TRAFFIC, PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(TRAFFIC, PREFILL, LinkClass.TRAFFIC_TO_ENGINE, 9.0e-3)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 1.0e-3)
    assert [(str(link.source), str(link.target)) for link in matrix.links()] == [
        ("prefill", "decode"),
        ("traffic-source", "prefill"),
    ]
    assert len(matrix) == 2


# --- the configured scales ---------------------------------------------------


def test_the_three_link_classes_carry_the_scales_that_were_modelled():
    # Recorded so the tight link is a property of the class rather than folklore:
    # stage-to-stage is three orders below the admission delay, which is why
    # stages belong close together and role boundaries are the cheap ones to
    # stretch.
    scales = {link_class: link_class.scale_seconds for link_class in LinkClass}
    assert (
        scales[LinkClass.PIPELINE_STAGE_TO_STAGE]
        < scales[LinkClass.PREFILL_TO_DECODE]
        < scales[LinkClass.TRAFFIC_TO_ENGINE]
    )
    assert min(scales, key=scales.get) is LinkClass.PIPELINE_STAGE_TO_STAGE


def test_the_admission_delay_is_kept_per_path():
    # One number for both paths would misprice whichever was not measured.
    assert TRAFFIC_TO_ENGINE_FLOOR_SECONDS == {
        "offline_batch": 13.0e-3,
        "serving": 9.0e-3,
    }


# --- what the package is allowed to name -------------------------------------

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

# How deep a clock sits, and which of several it is. Forbidden everywhere,
# including the transport. A clock inserted between two others has to be
# invisible to both sides, which it is only while neither side can write down a
# level; and a participant is addressed by identity, which stops being true the
# moment anything numbers one.
RANKING_WORDS = ("depth", "index", "level", "parent", "rank")

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
    parameters', because an alternative constructor and a property are the first
    two places an endpoint gets hung. Enum classes are visited for their
    members and methods; their call is a value lookup rather than a constructor.
    Every exported dataclass is read for its fields, not a hard-coded two, and
    that catches a field declared `init=False`, which no signature shows.
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
        if not issubclass(obj, enum.Enum):
            surface.append((exported, (exported,) + tuple(_parameters(obj))))
        else:
            surface.append((exported, (exported,)))
        if dataclasses.is_dataclass(obj):
            surface.append(
                (
                    f"{exported} fields",
                    tuple(field.name for field in dataclasses.fields(obj)),
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


def test_an_identity_carries_a_name_and_nothing_that_would_locate_it():
    assert [field.name for field in dataclasses.fields(LpId)] == ["name"]


def test_the_surface_a_participant_uses_never_says_where_a_clock_is():
    # The import allowlist below proves the package reaches no device, clock or
    # socket. It does not prove this: a module taking host, port and level and
    # storing them would pass it unchanged. Stated here so the requirement is
    # mechanical rather than upheld by inspection, because the pressure to hang
    # an endpoint somewhere convenient arrives with the transport -- and it has
    # now arrived, which is why the resolving calls are carved out by name
    # rather than the whole check being loosened.
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


def test_no_public_call_anywhere_says_how_deep_a_clock_sits():
    # The half of the old check that stays absolute, and the reason it does:
    # the resolving calls need a location, and no part of the package needs a
    # level. A clock between two others is reachable exactly while neither side
    # can write one down.
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


# --- what the package is allowed to reach ------------------------------------


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


#: The standard library every module in the package may reach. Short on
#: purpose: these structures have to be buildable and readable on any machine,
#: with no device runtime, no wall-clock read and no socket.
ALLOWED_IMPORTS = ("dataclasses", "enum", "math")

#: What one module may reach beyond that, and only that module. Widened per
#: module rather than package-wide, because "somewhere in here opens a socket"
#: is a much weaker statement than "this file does" -- and the file that does is
#: the one the design says should be the only one.
EXTRA_IMPORTS = {
    "transport/message.py": ("json",),
    "transport/stream.py": ("socket", "threading"),
}


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
    # Stated as an allowlist rather than a denylist. A denylist would have to
    # predict the name of the next thing that breaks the claim. It lists only
    # what each module actually imports: an allowlist naming something unused is
    # a permission granted for no reason.
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
    """Every name a module writes down, from its own syntax."""
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
    return written


@pytest.mark.parametrize("module", _clock_modules(), ids=_named)
def test_only_the_modules_that_carry_a_frame_say_where_a_clock_is(module):
    # The public scan above covers the exported surface. This covers the rest of
    # it: a location hung on a private attribute is exactly as much of a leak
    # and no signature shows it. Two modules open or choose a carriage and must
    # name a location to do it; the wire format, the rule's entry point and the
    # participant's session are on the other side of that line and stay clean.
    offenders = sorted(
        dict.fromkeys(name for name in _identifiers(module) if _places(name))
    )
    if _named(module) in MAY_PLACE_A_CLOCK:
        return
    assert not offenders, (
        f"{_named(module)} says where a clock is: {offenders}. Only "
        f"{list(MAY_PLACE_A_CLOCK)} may; everything else knows a name."
    )


@pytest.mark.parametrize("module", _clock_modules(), ids=_named)
def test_no_module_at_all_says_how_deep_a_clock_sits(module):
    # No exemption, for any module. Nothing in a carriage needs a level either:
    # it carries a frame between two ends and neither end has a position.
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
