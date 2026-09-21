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
* **An identity carries a name and nothing that would locate it.** An
  exhaustive equality on a closed set of fields, so it fails on any field added
  for any reason.

What is no longer here: the scans over what the whole package may *name* and
may *reach*. They were package-wide from the start -- an allowlist over every
module's imports, and a check that no module writes down a location or a level
-- and they grew a transport subpackage to cover, so they are no longer about
identity, the order or the matrix. They live in
`test_clock_package_boundary.py`, which is where a subpackage added later will
look for them.
"""

import dataclasses

import pytest

from atom.compass.clock import (
    TRAFFIC_TO_ENGINE_FLOOR_SECONDS,
    LinkClass,
    LookaheadMatrix,
    LpId,
    LpRegistry,
)

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


# --- what an identity may carry ----------------------------------------------


def test_an_identity_carries_a_name_and_nothing_that_would_locate_it():
    # An exhaustive equality rather than a scan for forbidden words: it fails on
    # any field added for any reason, which is a proof where a word list is a
    # vocabulary check. The vocabulary checks over the rest of the package are
    # in `test_clock_package_boundary.py`.
    assert [field.name for field in dataclasses.fields(LpId)] == ["name"]
