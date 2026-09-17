"""A later fallback cannot silently inherit an earlier predictor's identity."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.base import StepCost
from atom.compass.core.cost.library import Coverage
from atom.compass.core.cost import composition_extension as extension
from atom.compass.core.cost.composition_qualification import input_identity, source_selection, validate
from atom.compass.core.loaded_input import load_json
from .test_composition_qualification import bundle
from .test_native_ap_work import candidate


@pytest.fixture
def extended(bundle, monkeypatch):
    write, oracle = bundle["write"], bundle["oracle"]
    old_identity = copy.deepcopy(bundle["identity"])
    current_code = {**old_identity["code"], "compass/core/cost/native_mha_prefill.py": "f" * 64}
    monkeypatch.setattr(extension, "code_identity", lambda: current_code)
    fallback = write("fallback", dict(schema="compass.native_mha_prefill_fallback/1",
        old_lookup_first=True, refusal_only=True, modelled_transfer=True,
        base_source_book_changed=False, source_refitted=False, timing_equivalence_proven=False))
    added = load_json(fallback["path"], role=extension.ADDED_PREFIX + "handoff")[1]
    inputs = [*bundle["inputs"], added]
    options = dict(bundle["options"], composition_qualification=bundle["pins"]["receipt"]["path"],
        composition_qualification_sha256=bundle["pins"]["receipt"]["sha256"],
        native_mha_prefill_handoff=fallback["path"], native_mha_prefill_handoff_sha256=fallback["sha256"])

    def estimate(shape):
        parts = bundle["regions"].breakdown(shape)
        return StepCost(seconds=1. + sum(parts.values()), breakdown={"<body>": 1., **parts},
            output_ready_seconds=.25, output_ready_basis={"evidence": "unchanged"},
            preparation_seconds=parts["<prepare>"], model_seconds=1.)

    oracle.estimate = estimate
    oracle.last_coverage = Coverage(operators=1, measured=1, seconds=1., sources={"original": 1})
    source = bundle["pins"]["source"]
    sources = json.loads(Path(source["path"]).read_text())["rows"]
    snapshot = extension.cohort_snapshot(oracle, sources, bundle["heldouts"])
    snapshot.update(schema="compass.original_predictor_quotes/1",
        old_qualification=bundle["pins"]["receipt"], old_predictor_identity=bundle["pins"]["identity"],
        code=old_identity["code"], source_refitted=False, original_qualification_passed=True,
        source=source, heldout=bundle["pins"]["heldout"],
        collector=write("collector", {}), snapshot_helper=write("snapshot_helper", {}))
    data = dict(schema=extension.SCHEMA, new_code_predates_old_heldouts=False,
        source_refitted=False, accepted=False, final_e2e_proof_required=True,
        validation_scope="unchanged_original_geometries", old_qualification=bundle["pins"]["receipt"],
        old_predictor_identity=bundle["pins"]["identity"], old_quotes=write("old_quotes", snapshot),
        new_identity=dict(code=current_code, body_book=dict(loaded_inputs=input_identity(inputs),
                                                          source_selection=source_selection(options))),
        code_changes={"compass/core/cost/native_mha_prefill.py": dict(
            before=old_identity["code"].get("compass/core/cost/native_mha_prefill.py"), after="f" * 64)},
        added_inputs=input_identity([added]), fallback_handoff=fallback)
    pin = write("extension", data)
    # The adapter itself has separate real-source/selector tests. This fixture
    # exercises the extension contracts without rebuilding its source campaign.
    adapter_class = type("NativeMhaPrefillFallback", (), {})
    monkeypatch.setitem(sys.modules, "atom.compass.core.cost.native_mha_prefill",
                        SimpleNamespace(NativeMhaPrefillFallback=adapter_class))
    oracle.library = adapter_class()
    oracle.library.handoff_sha256 = fallback["sha256"]
    oracle.library.review = dict(native_comparisons=[dict(relative_error=-.10076989092815347)])
    oracle.compass_loaded_inputs = inputs
    bundle.update(inputs=inputs, options=options, sources=sources,
                  extension_data=data, snapshot=snapshot, extension_pin=pin,
                  extension=extension.CompositionExtension(**dict(path=pin["path"], sha256=pin["sha256"]), options=options))
    bundle["extension"].calibration_comparisons = []
    return bundle


def qualify(extended, use_extension=True):
    pin = extended["pins"]["receipt"]
    return validate(pin["path"], pin["sha256"], inputs=extended["inputs"], options=extended["options"],
        regions=extended["regions"], oracle=extended["oracle"],
        extension=extended["extension"] if use_extension else None)


def test_explicit_extension_preserves_old_proof_and_discloses_primitive_limit(extended):
    original = Path(extended["pins"]["identity"]["path"]).read_bytes()
    verdict, loaded = qualify(extended)
    result = verdict["qualification_extension"]
    assert verdict["passed"] and result["exact_equality"]
    assert result["original_observations"] == 18
    assert result["primitive_controls_all_under_ten_percent"] is False
    assert result["added_domain_qualified_by_old_heldouts"] is False
    assert Path(extended["pins"]["identity"]["path"]).read_bytes() == original
    assert any(item.role == extension.PREFIX + "receipt" for item in loaded)


def test_default_validator_still_refuses_added_sources(extended):
    with pytest.raises(ValueError, match="complete predictor"):
        qualify(extended, use_extension=False)


def test_unlisted_or_changed_sources_refuse(extended):
    extended["inputs"].append(dict(role="oracle.price", sha256="e" * 64))
    with pytest.raises(ValueError, match="source book"):
        qualify(extended)


def test_new_options_cannot_change_existing_selection(extended):
    extended["options"]["derive"] = False
    with pytest.raises(ValueError, match="source options"):
        qualify(extended)


def test_arbitrary_code_change_refuses_even_when_receipt_is_rewritten(extended, monkeypatch):
    data = extended["extension_data"]
    code = dict(data["new_identity"]["code"], **{"model_engine/scheduler.py": "e" * 64})
    data["new_identity"]["code"] = code
    data["code_changes"]["model_engine/scheduler.py"] = dict(
        before=extended["identity"]["code"].get("model_engine/scheduler.py"), after="e" * 64)
    monkeypatch.setattr(extension, "code_identity", lambda: code)
    pin = extended["write"]("changed_code", data)
    with pytest.raises(ValueError, match="outside the bounded"):
        extension.CompositionExtension(pin["path"], pin["sha256"], options=extended["options"])


@pytest.mark.parametrize("change", ["readiness", "readiness-proof", "coverage", "body", "model", "prepare", "postprocess"])
def test_every_original_timing_boundary_and_coverage_must_match_exactly(extended, change):
    obj, oracle = extended["extension"], extended["oracle"]
    obj.check_identity(extended["identity"], extended["inputs"], extended["options"], oracle)
    original = oracle.estimate
    def estimate(shape):
        cost = original(shape)
        if change == "readiness":
            return replace(cost, output_ready_seconds=cost.output_ready_seconds + 1e-14)
        if change == "readiness-proof":
            return replace(cost, output_ready_basis={"evidence": "different"})
        if change == "coverage":
            oracle.last_coverage = replace(oracle.last_coverage, sources={"different": 1})
            return cost
        if change == "model":
            return replace(cost, model_seconds=cost.model_seconds + 1e-14)
        term = {"body": "<body>", "prepare": "<prepare>", "postprocess": "<postprocess>"}[change]
        return replace(cost, breakdown={**cost.breakdown, term: cost.breakdown[term] + 1e-14})
    oracle.estimate = estimate
    with pytest.raises(ValueError, match="component, timing, readiness or coverage"):
        obj.check_quotes(oracle, extended["sources"], extended["heldouts"], extended["pins"]["source"])


def test_source_calibration_raw_B_is_requoted_exactly(extended):
    obj = extended["extension"]
    quotes = dict(observations=[dict(source_row_index=0, B=1.)])
    obj.check_calibration(extended["oracle"], extended["sources"], quotes)
    quotes["observations"][0]["B"] += 1e-8
    with pytest.raises(ValueError, match="beyond the existing export policy"):
        obj.check_calibration(extended["oracle"], extended["sources"], quotes)


def test_export_roundoff_is_disclosed_but_old_new_equality_stays_exact(extended):
    obj, oracle = extended["extension"], extended["oracle"]
    quotes = dict(observations=[dict(source_row_index=0, B=1. + 1e-14)])
    obj.check_calibration(oracle, extended["sources"], quotes)
    evidence = obj.calibration_comparisons[0]
    assert evidence["old_actual_B"] == evidence["new_actual_B"] == 1.
    assert evidence["absolute_export_difference"] > 0 and evidence["export_difference_ulps"] > 0
    original = oracle.estimate
    def changed(shape):
        cost = original(shape)
        return replace(cost, breakdown={**cost.breakdown, "<body>": 1. + 1e-14})
    oracle.estimate = changed
    with pytest.raises(ValueError, match="source-calibration raw B"):
        obj.check_calibration(oracle, extended["sources"], quotes)


def test_body_identity_only_accepts_the_exact_enumerated_addition(extended):
    obj = extended["extension"]
    book = extended["identity"]["body_book"]
    obj.check_body_identity(book["loaded_inputs"], book["source_selection"],
                            extended["oracle"], extended["options"])
    extended["oracle"].compass_loaded_inputs.append(dict(
        role=extension.ADDED_PREFIX + "unlisted", sha256="e" * 64))
    with pytest.raises(ValueError, match="added source identities"):
        obj.check_body_identity(book["loaded_inputs"], book["source_selection"],
                                extended["oracle"], extended["options"])


def test_body_identity_keeps_original_body_sources_bound(extended):
    obj = extended["extension"]
    book = extended["identity"]["body_book"]
    extended["oracle"].compass_loaded_inputs.append(dict(role="oracle.price", sha256="e" * 64))
    with pytest.raises(ValueError, match="inherited compiled-prefill B identity"):
        obj.check_body_identity(book["loaded_inputs"], book["source_selection"],
                                extended["oracle"], extended["options"])


def test_missing_old_observation_is_not_accepted(extended):
    obj = extended["extension"]
    obj.old_quotes["observations"].pop()
    with pytest.raises(ValueError, match="component, timing, readiness or coverage"):
        obj.check_quotes(extended["oracle"], extended["sources"], extended["heldouts"], extended["pins"]["source"])
