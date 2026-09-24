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
  clock, no socket -- an allowlist over the package's own imports. That is a
  different claim from naming no location and no level, which a module taking
  `host`, `port` and `level` would satisfy while breaking; the signature scan
  is what covers the second.
"""

import ast
import dataclasses
import enum
import inspect
from pathlib import Path

import pytest

from atom.compass import clock
from atom.compass.clock import (
    InterLpLink,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
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
    # An unhashable probe is a mistake in the call, so it raises rather than
    # answering False.
    with pytest.raises(TypeError):
        [] in registry


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
    # Zero serializes the pair. It stays correct, so it is accepted rather than
    # rejected -- the floor buys speed, never safety.
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 0.0)
    assert matrix.lookahead(PREFILL, DECODE) == 0.0


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
    # The refusal names the declaration that stands, so the caller can tell which
    # of the two calls to change.
    with pytest.raises(
        ValueError, match=r"already declared as .*prefill_to_decode, floor=0\.001s"
    ):
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
    # `inbound` and `declare` both return the object itself. If it were
    # writable, every refusal in `declare` -- negative, NaN, infinite, already
    # declared -- would be reachable around.
    registry = _registry(PREFILL, DECODE)
    matrix = LookaheadMatrix(registry)
    link = matrix.declare(PREFILL, DECODE, LinkClass.PREFILL_TO_DECODE, 2.0e-3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        link.floor_seconds = 99.0
    assert matrix.lookahead(PREFILL, DECODE) == pytest.approx(2.0e-3)


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


# --- what the package is allowed to name -------------------------------------

# A participant knows a peer by name. Where that peer runs, how it is reached,
# and how deep it sits in an arrangement of authorities are all things it must
# not be able to write down, because an authority inserted between two others
# has to be invisible to both sides.
LOCATING_WORDS = (
    "address",
    "depth",
    "endpoint",
    "host",
    "level",
    "node",
    "parent",
    "port",
    "rank",
    "socket",
    "url",
)


def _public_signatures():
    """(name, signature) for every exported callable and its public methods."""
    found = []
    for exported in sorted(clock.__all__):
        obj = getattr(clock, exported)
        if not inspect.isclass(obj):
            if callable(obj):
                found.append((exported, inspect.signature(obj)))
            continue
        if not issubclass(obj, enum.Enum):
            # An Enum's call is a value lookup, not a constructor.
            found.append((exported, inspect.signature(obj)))
        for attr_name, attr in sorted(vars(obj).items()):
            if attr_name.startswith("_") or not inspect.isfunction(attr):
                continue
            found.append((f"{exported}.{attr_name}", inspect.signature(attr)))
    return found


def test_an_identity_carries_a_name_and_nothing_that_would_locate_it():
    assert [field.name for field in dataclasses.fields(LpId)] == ["name"]


def test_no_public_call_takes_a_location_or_a_level():
    # The import allowlist below proves the package reaches no device, clock or
    # socket. It does not prove this: a module taking host, port and level and
    # storing them would pass it unchanged. Stated here so the requirement is
    # mechanical rather than upheld by inspection, because the pressure to hang
    # an endpoint somewhere convenient arrives with the transport.
    offenders = []
    for name, signature in _public_signatures():
        for parameter in signature.parameters:
            if parameter.lower() in LOCATING_WORDS:
                offenders.append(f"{name}({parameter})")
    for cls in (LpId, InterLpLink):
        for field in dataclasses.fields(cls):
            if field.name.lower() in LOCATING_WORDS:
                offenders.append(f"{cls.__name__}.{field.name}")
    assert not offenders, (
        f"the interface names where a peer is or how deep it sits: {offenders}. "
        "A participant knows a name; an endpoint belongs to whatever resolves it."
    )


# --- what the package is allowed to reach ------------------------------------


def _clock_modules():
    # rglob, not glob: a transport or any other subpackage added later would sit
    # below the top level, and a guard that stopped there would go quiet on
    # exactly the code it exists for. Each module is one parametrised case, so a
    # new one shows up in the gate arithmetic rather than silently.
    return sorted(CLOCK_PACKAGE.rglob("*.py"))


def test_the_package_was_found():
    assert _clock_modules(), f"no modules under {CLOCK_PACKAGE}"


@pytest.mark.parametrize("module", _clock_modules(), ids=lambda p: p.name)
def test_the_package_imports_only_the_standard_library_it_names(module):
    # Stated as an allowlist rather than a denylist. The claim being kept is
    # that these structures can be built and read on any machine: no device
    # runtime, no wall-clock read, no socket. A denylist would have to predict
    # the name of the next thing that breaks it. It lists only what the package
    # actually imports: an allowlist naming something unused is a permission
    # granted for no reason, and it weakens the statement.
    allowed = {"dataclasses", "enum", "math"}
    tree = ast.parse(module.read_text())
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.append((node.module or "").split(".")[0])
    strays = sorted({root for root in roots if root not in allowed})
    assert not strays, f"{module.name} imports {strays}; allowed: {sorted(allowed)}"


@pytest.mark.parametrize("module", _clock_modules(), ids=lambda p: p.name)
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
        f"{module.name} builds {offenders}; use a dict with None values as an "
        "ordered set, or sort at the point of iteration"
    )
