"""Acceptance refuses readiness inputs that cannot be attributed to the core."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "compass_readiness_validation", ROOT / "scripts/compass/cc_traces_validate.py")
validate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validate
spec.loader.exec_module(validate)

WORKLOAD_SHA = "9" * 64
FORBIDDEN_SHA = "8" * 64


@pytest.fixture
def evidence():
    inputs, artifacts = [], []
    names = ("profile", "endpoint_fit", "validation", "plan", "contract", "source")
    for name, digit in zip(names, "abcdef"):
        path = f"/source/readiness/{name}.json"
        digest = digit * 64
        inputs.append({"role": f"runtime.request_readiness.{name}",
                       "requested": path, "path": path, "sha256": digest,
                       "size": 123, "rank_own": False, "rank_coords": {}})
        artifacts.append({"sha256": digest, "kind": "source_calibration",
                          "measured_at_tp": 1, "contents": {f"{name}.json": digest},
                          "sources": [{"path": "/source/independent_probe.json",
                                       "sha256": "6" * 64}],
                          "code": {"source_probe.py": "7" * 64}})
    worker = {"rank_coords": {}, "inputs": [{"role": "oracle.price", "sha256": "0" * 64}],
              "core_inputs": {
                  "reader": {"component": "EngineCore.Scheduler", "pid": 321},
                  "inputs": inputs,
                  "request_readiness": {"resolver": "fixture.Service",
                                        "origin_contract": {"transition": "fixture"},
                                        "support": {"tp": 1}, "source_law": "fixture",
                                        "resolved_requests": 0}}}
    compass = {"request_readiness_profile": "/source/readiness/profile.json",
               "loaded_inputs": {"ranks": [worker]}}
    return SimpleNamespace(manifest={"server": {"compass": compass}}), {"artifacts": artifacts}


def check(evidence):
    modelled, registry = evidence
    # Exercise the public check already called by full cell acceptance.
    return validate.check_calibration(
        modelled, registry, 1, WORKLOAD_SHA, {"real step table": FORBIDDEN_SHA})


def core(evidence):
    return evidence[0].manifest["server"]["compass"]["loaded_inputs"]["ranks"][0]["core_inputs"]


def test_enabled_core_inputs_pass_without_becoming_worker_ranks(evidence):
    before = copy.deepcopy(evidence[0].manifest)
    assert check(evidence) == []
    assert evidence[0].manifest == before
    ranks = validate._rank_records(evidence[0])
    assert len(ranks) == 1 and ranks[0]["inputs"] == before["server"]["compass"]["loaded_inputs"]["ranks"][0]["inputs"]
    # Provenance can be collected before the registered requests have arrived.
    assert core(evidence)["request_readiness"]["resolved_requests"] == 0


def test_disabled_readiness_keeps_legacy_behavior(evidence):
    compass = evidence[0].manifest["server"]["compass"]
    compass.pop("request_readiness_profile")
    compass["loaded_inputs"]["ranks"][0].pop("core_inputs")
    evidence[1]["artifacts"] = []
    assert check(evidence) == []


@pytest.mark.parametrize("missing", ["core", "profile", "source", "reader"])
def test_missing_core_profile_or_source_is_refused(evidence, missing):
    if missing == "core":
        evidence[0].manifest["server"]["compass"]["loaded_inputs"]["ranks"][0].pop("core_inputs")
    elif missing == "reader":
        core(evidence).pop("reader")
    else:
        rows = core(evidence)["inputs"]
        rows[:] = [r for r in rows if r["role"] != f"runtime.request_readiness.{missing}"]
    assert check(evidence)


def test_worker_inputs_cannot_substitute_for_the_core_reader(evidence):
    worker = evidence[0].manifest["server"]["compass"]["loaded_inputs"]["ranks"][0]
    worker["inputs"].extend(worker.pop("core_inputs")["inputs"])
    assert any("core_inputs" in reason for reason in check(evidence))


def test_loaded_profile_must_match_the_enabled_profile(evidence):
    evidence[0].manifest["server"]["compass"]["request_readiness_profile"] = "/source/other-profile.json"
    assert any("does not match enabled profile" in reason for reason in check(evidence))


def test_unregistered_loaded_source_digest_is_refused(evidence):
    core(evidence)["inputs"][-1]["sha256"] = "5" * 64
    assert any("registry does not declare" in reason for reason in check(evidence))


@pytest.mark.parametrize("damage", [None, "wrong_pin", "wrong_role", "source_leak"])
def test_only_exact_typed_opening_workload_is_excluded_from_fitted_sources(evidence, damage):
    modelled, registry = evidence
    compass = modelled.manifest["server"]["compass"]
    compass.update(opening_plan="/workload/opening.json", opening_plan_sha256=WORKLOAD_SHA)
    row = {"role": "runtime.aiperf_opening", "requested": compass["opening_plan"],
           "path": compass["opening_plan"], "sha256": WORKLOAD_SHA, "size": 123}
    core(evidence)["inputs"].append(row)
    core(evidence)["release_calendar"] = {"input": {"sha256": WORKLOAD_SHA}}
    if damage == "wrong_pin":
        row["sha256"] = "5" * 64
    elif damage == "wrong_role":
        row["role"] = "runtime.request_readiness.extra_source"
    elif damage == "source_leak":
        registry["artifacts"][-1]["workload_sha256"] = WORKLOAD_SHA
    errors = validate.check_calibration(
        modelled, registry, 1, WORKLOAD_SHA, {"real step table": FORBIDDEN_SHA},
        expected_opening_plan_sha256=WORKLOAD_SHA)
    assert bool(errors) == (damage is not None)
    # The same input has no exemption in the ordinary registered route.
    assert check(evidence)


def test_registry_contents_must_match_the_core_read(evidence):
    evidence[1]["artifacts"][-1]["contents"]["source.json"] = "5" * 64
    assert any("differs" in reason for reason in check(evidence))


@pytest.mark.parametrize("leak", ["target_engine", "workload", "nested_evaluated_source"])
def test_existing_target_leak_checks_apply_to_core_sources(evidence, leak):
    entry = evidence[1]["artifacts"][-1]
    if leak == "target_engine":
        entry["from_target_engine"] = True
    elif leak == "workload":
        entry["workload_sha256"] = WORKLOAD_SHA
    else:
        entry["sources"] = [{"path": "/evaluated/real_steps.jsonl", "sha256": FORBIDDEN_SHA}]
    assert check(evidence)
