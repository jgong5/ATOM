"""Diagnostic permission never replaces scope, source identity or strict gates."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from atom.compass.core.cost import root_diagnostic as diagnostic
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.core.cost.region_supplement import _candidate_points
from atom.compass.core.cost.root_prefill import RootPrefillPrices
from atom.compass.core.loaded_input import file_digests, manifest


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def header():
    return dict(schema=diagnostic.SCHEMA, diagnostic_only=True, source_qualified=False,
        acceptance_eligible=False, heldout_timings_used_as_fit_inputs=False,
        fit_inputs_are_references_only=True,
        scope=dict(model="Qwen/Qwen3.8-27B", topology={"tp": 1}, dtype="bfloat16", num_sequences=1,
                   request_scope_sha256=diagnostic.REQUEST_SCOPE_SHA256, workload_sha256=diagnostic.WORKLOAD_SHA256),
        evidence={role: {"path": "not-read.json", "sha256": "1" * 64} for role in diagnostic.EVIDENCE})


@pytest.mark.parametrize("mutation", [
    lambda h: h.update(source_qualified=True),
    lambda h: h.update(acceptance_eligible=True),
    lambda h: h.update(diagnostic_only=False),
    lambda h: h.update(heldout_timings_used_as_fit_inputs=True),
    lambda h: h["scope"].update(model="another-model"),
    lambda h: h["scope"].update(topology={"tp": 2}),
    lambda h: h["scope"].update(dtype="float16"),
    lambda h: h["scope"].update(num_sequences=2),
    lambda h: h["scope"].update(workload_sha256="2" * 64),
])
def test_scope_and_qualification_refusal_precedes_reference_reads(tmp_path, monkeypatch, mutation):
    path = tmp_path / "handoff.json"
    value = header()
    path.write_text(json.dumps(value))
    def evidence_reached(*args):
        raise AssertionError("reference reads reached")
    monkeypatch.setattr(diagnostic, "DiagnosticEvidence", evidence_reached)
    options = dict(deployment_scope_sha256=diagnostic.REQUEST_SCOPE_SHA256,
                   workload_sha256=diagnostic.WORKLOAD_SHA256, diagnostic_only=True)
    # The complete valid header reaches reference validation, so a refusal
    # below cannot pass merely because the test omitted another header field.
    with pytest.raises(AssertionError, match="reference reads reached"):
        diagnostic.DiagnosticRootPrefillPrices(PriceLibrary(), str(path), pin(path)["sha256"], **options)
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        diagnostic.DiagnosticRootPrefillPrices(PriceLibrary(), str(path), pin(path)["sha256"], **options)


def test_diagnostic_mode_and_workload_are_required_before_any_file_read():
    for changes in ({"diagnostic_only": False}, {"workload_sha256": "wrong"},
                    {"deployment_scope_sha256": "wrong"}):
        options = dict(deployment_scope_sha256=diagnostic.REQUEST_SCOPE_SHA256,
                       workload_sha256=diagnostic.WORKLOAD_SHA256, diagnostic_only=True)
        options.update(changes)
        with pytest.raises(ValueError, match="explicit mode"):
            diagnostic.DiagnosticRootPrefillPrices(PriceLibrary(), "/not-read", "0" * 64, **options)


@pytest.mark.parametrize("changes", [
    {"diagnostic_only": False},
    {"root_prefill_diagnostic_handoff": None},
    {"root_prefill_diagnostic_handoff_sha256": None},
    {"root_prefill_diagnostic_workload_sha256": None},
    {"root_prefill_handoff": "strict.json", "root_prefill_handoff_sha256": "2" * 64},
    {"region_supplement_handoff": "strict.json", "region_supplement_handoff_sha256": "2" * 64},
])
def test_factory_rejects_incomplete_or_mixed_diagnostic_selection_before_loading(monkeypatch, changes):
    from atom.compass.runtime import cache_region_oracle as factory

    options = dict(model="Qwen/Qwen3.8-27B", tp=1, block_size=16, max_model_len=262144,
        position_rows=3, cudagraph_mode="full", allocation="native", diagnostic_only=True,
        regions="not-read", region_overlay="not-read.json", region_overlay_sha256="1" * 64,
        root_prefill_diagnostic_handoff="diagnostic.json", root_prefill_diagnostic_handoff_sha256="3" * 64,
        root_prefill_diagnostic_workload_sha256=diagnostic.WORKLOAD_SHA256)
    def reached(*args, **kwargs):
        raise AssertionError("file read reached")
    monkeypatch.setattr(factory, "load_json", reached)
    with pytest.raises(AssertionError, match="file read reached"):
        factory.source_cost_oracle(**options)
    options.update(changes)
    with pytest.raises(ValueError):
        factory.source_cost_oracle(**options)


def test_strict_loader_never_accepts_the_diagnostic_schema(tmp_path):
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(header()))
    for mode in (False, True):
        with pytest.raises(ValueError, match="handoff"):
            RootPrefillPrices(PriceLibrary(), str(path), pin(path)["sha256"],
                deployment_scope_sha256=diagnostic.REQUEST_SCOPE_SHA256,
                diagnostic_only=mode, allow_failed_spread=mode)


@pytest.mark.parametrize("filename,json_data", [("JOB.json", True), ("native_steps.jsonl", False)])
def test_actual_reads_and_registry_preserve_duplicate_basenames(tmp_path, filename, json_data):
    from .test_opening_harness import validate

    paths = [tmp_path / name / filename for name in ("first", "second")]
    for index, path in enumerate(paths):
        path.parent.mkdir()
        path.write_text(json.dumps({"source": index}) + "\n")
    reader = diagnostic.DiagnosticEvidence(tmp_path)
    for path in paths:
        reader.read(pin(path), "reference_job", json_data=json_data)
    reader.read(pin(paths[0]), "historical_job", json_data=json_data)
    assert len(reader.inputs) == 3
    identities = {(row.role, row.path, row.sha256) for row in reader.inputs}
    assert len(identities) == 3
    files = file_digests(reader.inputs)
    assert files == {str(path): pin(path)["sha256"] for path in paths}
    ranks = [manifest(reader.inputs)]
    api = Path(__file__).resolve().parents[2] / "atom/entrypoints/openai/api_server.py"
    nodes = [n for n in ast.parse(api.read_text()).body if
        isinstance(n, ast.FunctionDef) and n.name == "_loaded_option_files" or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ROLE_OPTIONS" for t in n.targets)]
    namespace = {}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(api), "exec"), namespace)
    assert namespace["_loaded_option_files"](ranks)["root_prefill_diagnostic_handoff"] == files
    digest = validate._rolled_digest(files)
    entry = dict(kind="source_calibration", measured_at_tp=1, from_target_engine=False,
        sha256=digest, contents=dict(files), sources=[{"path": "/independent/source.json", "sha256": "8" * 64}],
        code={"collector.py": "9" * 64})
    def check(record):
        return validate._check_calibration_records({"root_prefill_diagnostic_handoff": digest},
            {"root_prefill_diagnostic_handoff": files}, {"artifacts": [record]}, 1,
            diagnostic.WORKLOAD_SHA256, {})
    assert check(entry) == []
    damaged = deepcopy(entry)
    damaged["contents"].pop(str(paths[1]))
    assert check(damaged)  # Exactly one colliding basename member is omitted.


def test_diagnostic_raw_registration_allows_only_unambiguous_byte_identical_aliases(tmp_path):
    from .test_opening_harness import opening

    path = tmp_path / "EXECUTABLE_PLAN.json"
    path.write_text('{"plan": "same bytes, another original filename"}')
    reader = diagnostic.DiagnosticEvidence(tmp_path)
    reader.read(pin(path), "plan")
    item = reader.inputs[0]
    entry = dict(kind="source_calibration", measured_at_tp=1, from_target_engine=False,
        sha256=item.sha256, contents={"PLAN.json": item.sha256},
        sources=[{"path": "/independent/source.json", "sha256": "8" * 64}], code={"collector.py": "9" * 64})
    registry = {"artifacts": [entry]}

    def check(value, forbidden=None):
        return opening._check_diagnostic_input_registration(item, value, diagnostic.WORKLOAD_SHA256,
                                                              forbidden or {})

    assert check(registry) == []
    assert entry["contents"] == {"PLAN.json": item.sha256}
    for artifacts in ([], [entry, deepcopy(entry)]):
        with pytest.raises(ValueError, match="one raw-SHA"):
            check({"artifacts": artifacts})
    for contents in ({}, {"PLAN.json": "0" * 64},
                     {"PLAN.json": item.sha256, "EXECUTABLE_PLAN.json": item.sha256}):
        with pytest.raises(ValueError, match="singleton"):
            check({"artifacts": [dict(entry, contents=contents)]})
    assert check({"artifacts": [dict(entry, from_target_engine=True)]})
    assert check({"artifacts": [dict(entry, sources=[])]})
    assert check(registry, {"target step table": item.sha256})
    assert check(registry, {"target source": "8" * 64})


@pytest.mark.parametrize("duplicate", [False, True])
def test_strict_candidate_still_rejects_missing_or_duplicate_original_pin(tmp_path, duplicate):
    from .test_region_supplement import bundle

    _, _, root, _ = bundle(tmp_path)
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert len(_candidate_points(candidate, root)) == 10
    freeze = next(row for row in root.loaded_inputs if row.role.endswith("region_freeze"))
    root.loaded_inputs = tuple(row for row in root.loaded_inputs if row is not freeze)
    if duplicate:
        root.loaded_inputs += (freeze, freeze)
    candidate["original_nine_cell_freeze"]["sha256"] = None
    with pytest.raises(ValueError, match="original nine-cell"):
        _candidate_points(candidate, root)


def test_pair_report_keeps_source_failures_and_cannot_receive_acceptance_credit(tmp_path, monkeypatch):
    from .test_opening_harness import (opening, wrapper_evidence,
        test_pair_route_stays_diagnostic_and_preserves_source_or_memory_failure as exercise_pair)

    evidence = wrapper_evidence.__wrapped__(tmp_path)
    options = evidence[0]["oracle_options"]
    options.update(root_prefill_diagnostic_handoff="diagnostic-fixture.json",
        root_prefill_diagnostic_handoff_sha256="6" * 64,
        root_prefill_diagnostic_workload_sha256=diagnostic.WORKLOAD_SHA256)
    status = dict(source_qualified=False, r5_validation_complete=False,
                  r5_validation_records=171, r5_validation_expected=348, supplement_gates_passed=17)
    def source_check(*args, diagnostic_status=None):
        diagnostic_status.update(status)
        return [], ["FAILED outputless source qualification retained", "FAILED final-query transfer retained"]
    monkeypatch.setattr(opening, "check_source_contract", source_check)
    exercise_pair(tmp_path, monkeypatch, None, evidence)
    report = json.loads((tmp_path / "tp1_aiperf_opening_fixture_c1/opening_diagnostic.json").read_text())
    assert report["accepted"] is False
    assert report["source_contracts"][0]["diagnostic_status"] == status
    assert report["source_contracts"][0]["source_qualified"] is False
