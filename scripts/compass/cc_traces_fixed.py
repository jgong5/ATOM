"""Complete-root fixed-absolute diagnostics through the maintained lifecycle."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import re
import sys


# Retain this instance if another harness loads a module with the same name.
_READER = sys.modules[__name__]
CASE_SCHEMA = "compass.fixed_absolute_case/1"
PLAN_KEY = "fixed_absolute_plan"
EVIDENCE_KEY = "fixed_absolute"
REPORT_SCHEMA = "compass.fixed_absolute_diagnostic/1"
REPORT_NAME = "fixed_diagnostic.json"

# These deployment/source checks are shared with the bounded opening. The
# opening's identity and two-turn release checks remain in its own reader.
_spec = importlib.util.spec_from_file_location(
    "fixed_shared_chat_checks", Path(__file__).with_name("cc_traces_opening.py"))
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)
check_server_configuration = _shared.check_server_configuration
check_source_contract = _shared.check_source_contract


def _plan(path, sha256):
    from atom.compass.fixed_absolute import FixedAbsolutePlan
    return FixedAbsolutePlan.load(path, sha256)


def load_case(path, sha, case_id, *, target_model):
    if not re.fullmatch(r"aiperf_fixed_[A-Za-z0-9_-]+", case_id):
        raise ValueError("fixed case-id must use the explicit aiperf_fixed_ prefix")
    plan = _plan(path, sha)
    if plan.model != target_model:
        raise ValueError("fixed case targets a different model")
    _shared.check_producer(plan.producer, label="fixed")
    evidence = plan.evidence()
    pin = {"path": str(Path(path).resolve()), "sha256": sha}
    return {"schema": CASE_SCHEMA, "case_id": case_id,
            "clients": evidence["clients"], "requests": evidence["requests"],
            "purpose": "diagnostic", "registered_acceptance_cell": False,
            "target_model": target_model, "workload": pin["path"],
            "workload_sha256": sha, PLAN_KEY: pin,
            "workload_inputs": [plan.loaded_input.as_dict(), *evidence["source_roots"]],
            "root_ids": evidence["root_ids"], "cache_policy": plan.cache_policy,
            "producer": evidence["producer"],
            "prompt_token_sha256": evidence["prompt_token_sha256"],
            "profile": evidence["profile"], "dependency_basis": evidence["dependency_basis"],
            "response_delivery": evidence["response_delivery"],
            "qualification": evidence["qualification"]}


def identity(case):
    if case.get("schema") != CASE_SCHEMA:
        raise ValueError("case is not a fixed-absolute complete-root diagnostic")
    return {key: case[key] for key in (
        "schema", "case_id", "clients", "requests", "purpose", "registered_acceptance_cell",
        "target_model", "workload_sha256", "workload_inputs", "root_ids", "producer",
        "prompt_token_sha256", "cache_policy", "profile", "dependency_basis",
        "response_delivery", "qualification")}


def recheck(case):
    pin = case[PLAN_KEY]
    current = load_case(pin["path"], pin["sha256"], case["case_id"],
                        target_model=case["target_model"])
    if identity(current) != identity(case):
        raise ValueError("fixed case identity changed")
    return current


def calibration_options(case):
    return {"expected_fixed_absolute_plan": case[PLAN_KEY]}


def check_result(blob, case):
    recheck(case)
    plan = _plan(**case[PLAN_KEY])
    manifest = blob.get("run") or {}
    attestation = manifest.get(EVIDENCE_KEY) or {}
    if (any(attestation.get(key) != value for key, value in plan.evidence().items())
            or blob.get("workload") != plan.workload()
            or manifest.get("requests") != case["requests"]
            or len(blob.get("results") or []) != case["requests"]
            or manifest.get("complete") is not True
            or (manifest.get("prompt_encoding") or {}).get("kind") != "chat_messages"):
        raise ValueError("replay does not attest to this complete fixed-absolute chat plan")
    errors = plan.observation_errors(manifest.get("server") or {}, blob.get("engine") or {},
                                     blob.get("results") or [])
    if (blob.get("engine") or {}).get("clock") == "wall":
        errors += preparation_errors(manifest, plan)
    elif manifest.get("prepare") is not None:
        errors.append("fixed predictor must remain unprepared")
    if errors:
        raise ValueError("; ".join(errors))


def preparation_errors(manifest, plan):
    """Recheck the exact warm bodies, consumed tokens, serial timing and drain."""
    from atom.compass.fixed_absolute import sequential_preparation_rows
    rows = sequential_preparation_rows(plan)
    receipt = manifest.get("prepare") or {}
    policy = receipt.get("policy") or {}
    expected_payloads = []
    for payload in plan.encode_payloads(declared=False):
        body = json.loads(payload)
        body["max_completion_tokens"] = 2
        expected_payloads.append(hashlib.sha256(json.dumps(body).encode()).hexdigest())
    expected = {"requested": 7, "returned": 7, "drained_records": 7, "drained": True,
                "store_empty_after_drain": True, "declared_workload_size": False,
                "clock": "wall", "within": "preparation", "sequence_order": list(range(7)),
                "plan_sha256": plan.loaded_input.sha256, "failures": [],
                "prompt_token_sha256": [row["prompt_token_sha256"] for row in rows],
                "payload_sha256": expected_payloads,
                "shapes": [{"input_tokens": row["input_tokens"], "output_tokens": 2} for row in rows]}
    bad = [f"fixed preparation {key} differs" for key, value in expected.items()
           if receipt.get(key) != value]
    if policy != {"purpose": "diagnostic", "kind": "sequential_exact_chat", "output_tokens_cap": 2,
                  "prompt_tokens": "exact exported chat", "measured_requests": "unchanged",
                  "cache_between_requests": "retained"}:
        bad.append("fixed preparation policy differs from its bounded exact-chat form")
    responses = receipt.get("responses") or []
    consumed = receipt.get("consumed_prompts") or []
    by_id = {row.get("request_id"): row for row in consumed}
    if (len(responses) != 7 or len(consumed) != len(by_id) or len(by_id) != 7
            or any(row.get("seq_id") is None for row in consumed)
            or len({row.get("seq_id") for row in consumed}) != 7
            or {row.get("request_id") for row in responses} != set(by_id)):
        return bad + ["fixed preparation lacks one consumed record per exact prompt"]
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    started, ended = receipt.get("wall_started_at"), receipt.get("wall_ended_at")
    measured_start = (manifest.get("wall_execution") or {}).get("started_at")
    if not all(finite(value) for value in (started, ended, measured_start)) or not started <= ended <= measured_start:
        return bad + ["fixed preparation is not wholly outside measured execution"]
    previous = started
    for index, response in enumerate(responses):
        timing = response.get("send_timing") or {}
        begin, finish = timing.get("request_started_at"), timing.get("client_response_returned_wall_time")
        source = (by_id.get(response.get("request_id")) or {}).get("shared_preprocessing") or {}
        usage = response.get("usage") or {}
        if (response.get("index") != index or response.get("ok") is not True
                or usage.get("prompt_tokens") != rows[index]["input_tokens"] or usage.get("completion_tokens") != 2
                or source.get("input_tokens") != rows[index]["input_tokens"]
                or source.get("prompt_token_sha256") != rows[index]["prompt_token_sha256"]
                or not finite(begin) or not finite(finish) or not previous <= begin <= finish <= ended):
            bad.append(f"fixed preparation request {index} lost exact identity or sequential completion")
        if finite(finish):
            previous = finish
    return bad


def preparation_policy(case):
    """Only the reviewed seven-request, causally serial N1 form is supported.

    Exact path equivalence is a source preflight result for the selected plan,
    not inferred here for arbitrary roots with the same request count.
    """
    plan = _plan(**case[PLAN_KEY])
    from atom.compass.fixed_absolute import sequential_preparation_rows
    rows = sequential_preparation_rows(plan)
    return {"kind": "sequential_exact_chat", "requests": len(rows), "output_tokens_cap": 2,
            "prompt_token_sha256": [row["prompt_token_sha256"] for row in rows],
            "workload_sha256": case["workload_sha256"], "measured_requests": "unchanged"}


def pair(args):
    try:
        return _shared._pair_case(args, _READER)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"fixed pair refused: {exc}", file=sys.stderr)
        return 2
