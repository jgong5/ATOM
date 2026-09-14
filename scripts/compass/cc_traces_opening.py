"""AIPerf opening identity and paired checks for the maintained side harness."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


CASE_SCHEMA = "compass.aiperf_opening_case/1"
AIPERF_COMMIT = "0d2aa0572ac685943d38c580675c4a61023581d3"
DECLARED_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]
EFFECTIVE_DECODE_BUCKETS = [1, 2, 4, 8, 16, 32]
WORKER_CONFIGURATION = {
    "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
    "max_model_len": 262144, "max_num_seqs": 32, "max_num_batched_tokens": 16384,
    "gpu_memory_utilization": .9, "kv_cache_block_size": 16, "kv_cache_dtype": "bf16",
    "enforce_eager": False, "enable_prefix_caching": True,
    "compilation_level": 3, "cudagraph_mode": "FULL",
    "declared_capture_sizes": DECLARED_CAPTURE_SIZES,
}


@lru_cache
def _script(name):
    spec = importlib.util.spec_from_file_location(
        f"opening_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan(path, sha256):
    from atom.compass.replay_plan import OpeningPlan
    return OpeningPlan.load(path, sha256)


def load_case(path, sha, case_id, *, target_model):
    if not re.fullmatch(r"aiperf_opening_[A-Za-z0-9_-]+", case_id):
        raise ValueError("opening case-id must use the explicit aiperf_opening_ prefix")
    plan = _plan(path, sha)
    if plan.model != target_model:
        raise ValueError("opening case targets a different model")
    export = plan.export_identity
    producer = export.get("producer") or {}
    policy = producer.get("weka_reconstruction") or {}
    effective = policy.get("effective") or {}
    if (producer.get("aiperf_commit") != AIPERF_COMMIT
            or policy.get("defaults_verified") is not True
            or effective != policy.get("pinned_defaults")
            or effective.get("WEKA_LIVE_ASSISTANT_RESPONSES") is not False
            or effective.get("WEKA_SPLIT_FLATTENED_AGENTS") is not True
            or effective.get("WEKA_TOOL_SHAPED_MESSAGES") is not False):
        raise ValueError("opening case lacks verified effective Weka reconstruction defaults")
    source = export.get("source") or {}
    if (source.get("selected_entries") != [0, 1]
            or source.get("full_plan_opening_roles_match") is not True
            or source.get("later_branches_excluded") is not True):
        raise ValueError("opening case lacks its bounded full-root ancestry validation")
    return {"schema": CASE_SCHEMA, "case_id": case_id, "clients": 1, "requests": 2,
            "purpose": "diagnostic", "registered_acceptance_cell": False,
            "target_model": target_model, "workload": str(Path(path).resolve()),
            "workload_sha256": sha, "opening_plan": {"path": str(Path(path).resolve()), "sha256": sha},
            "cache_policy": plan.cache_policy, "export_identity": export,
            "prompt_token_sha256": plan.evidence()["prompt_token_sha256"],
            "profile": plan.evidence()["profile"],
            "response_delivery": plan.evidence()["response_delivery"],
            "qualification": plan.evidence()["qualification"]}


def check_server_configuration(server, case, side):
    """Worker settings/capture and core cache policy retain distinct owners."""
    bad = []
    workers = server.get("worker_runtime") or []
    cores = (server.get("core_cache") or {}).get("ranks") or []
    if len(workers) != 1 or len(cores) != 1:
        return ["opening requires one worker runtime and one core cache-policy record"]
    worker, core = workers[0], cores[0]
    for label, reading, component in (("worker", worker, "ModelRunner"),
                                      ("core", core, "EngineCore.Scheduler")):
        owner = reading.get("reader") or {}
        if owner.get("component") != component or type(owner.get("pid")) is not int or owner["pid"] <= 0:
            bad.append(f"opening {label} settings have no owning reader")
    from atom.compass.core.cache_policy import policy_errors
    bad += policy_errors(core.get("policy"), case["cache_policy"])
    expected = {"model": case["target_model"], **WORKER_CONFIGURATION}
    configuration = worker.get("configuration") or {}
    for name, value in expected.items():
        if json.dumps(configuration.get(name), sort_keys=True) != json.dumps(value, sort_keys=True):
            bad.append(f"opening worker {name} differs from its pinned deployment")
    for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_block_size"):
        if (core.get("scheduler_configuration") or {}).get(name) != expected[name]:
            bad.append(f"opening core {name} differs from its pinned deployment")
    graph = worker.get("graphs") or {}
    if graph.get("effective_decode_buckets") != EFFECTIVE_DECODE_BUCKETS:
        bad.append("opening effective decode buckets differ")
    if side == "real":
        if graph.get("origin") != "native_capture" or graph.get("native_capture_sizes") != EFFECTIVE_DECODE_BUCKETS:
            bad.append("opening native capture ladder was not observed")
    else:
        if (graph.get("origin") != "borrowed_replay_target"
                or graph.get("native_capture_sizes") is not None
                or graph.get("borrowed_source_capture_sizes") != EFFECTIVE_DECODE_BUCKETS
                or not (graph.get("borrowed_target_input") or {}).get("sha256")):
            bad.append("opening predictor lacks its explicitly borrowed layout identity")
    return bad


def identity(case):
    if case.get("schema") != CASE_SCHEMA:
        raise ValueError("case is not an AIPerf opening diagnostic")
    return {key: case[key] for key in (
        "schema", "case_id", "clients", "requests", "purpose", "registered_acceptance_cell",
        "target_model", "workload_sha256", "prompt_token_sha256", "cache_policy",
        "profile", "response_delivery", "qualification")}


def recheck(case):
    current = load_case(case["opening_plan"]["path"], case["opening_plan"]["sha256"],
                        case["case_id"], target_model=case["target_model"])
    if identity(current) != identity(case):
        raise ValueError("opening case identity changed")
    return current


def check_result(blob, case):
    recheck(case)
    plan = _plan(**case["opening_plan"])
    manifest = blob.get("run") or {}
    attestation = manifest.get("aiperf_opening") or {}
    if (attestation.get("input", {}).get("sha256") != case["workload_sha256"]
            or attestation.get("profile") != case["profile"]
            or attestation.get("response_delivery") != case["response_delivery"]
            or blob.get("workload") != plan.workload()
            or manifest.get("requests") != 2 or len(blob.get("results") or []) != 2
            or manifest.get("complete") is not True
            or (manifest.get("prompt_encoding") or {}).get("kind") != "chat_messages"):
        raise ValueError("replay does not attest to this pinned chat opening")
    errors = plan.observation_errors(manifest.get("server") or {}, blob.get("engine") or {},
                                     blob.get("results") or [])
    if errors:
        raise ValueError("; ".join(errors))


def pair(args):
    try:
        return _pair(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"opening pair refused: {exc}", file=sys.stderr)
        return 2


def _pair(args):
    """Compose existing offline checks; this route never issues matrix credit."""
    validate, compare = _script("cc_traces_validate"), _script("compare")
    plan_module = _script("cc_traces_plan")
    case = load_case(args.opening_plan, args.opening_plan_sha256, args.case_id,
                     target_model=plan_module.MODEL)
    cell = Path(args.cell)
    if cell.name != f"tp1_{case['case_id']}_c1":
        raise ValueError("opening directory disagrees with its case identity")
    lock = json.loads((cell / "diagnostic_case.json").read_text())
    if identity(lock) != identity(case):
        raise ValueError("opening directory belongs to different pins")
    registry_path = Path(args.calibration_registry)
    registry = json.loads(registry_path.read_text())
    paths = {side: validate._runs(cell, side) for side in ("real", "modelled")}
    failures, notes, reports, memory = [], [case["qualification"]], [], []
    if len(paths["real"]) != len(paths["modelled"]) or len(paths["real"]) != args.repeats:
        failures.append("opening diagnostic does not contain the requested paired repeats")
    journals = {side: json.loads((cell / f"run.{side}.json").read_text()) for side in paths}
    for side, journal in journals.items():
        if journal.get("ok") is not True or journal.get("purpose") != "diagnostic":
            failures.append(f"{side} lifecycle did not complete as a successful diagnostic")
    failures += validate.check_gpu_free(cell, paths["modelled"])
    isolation_path = cell / "isolation.json"
    isolation = json.loads(isolation_path.read_text()) if isolation_path.exists() else {}
    if not isolation or isolation.get("verdict") == "own_contaminated":
        failures.append("opening has no usable native isolation evidence")
    elif not isolation.get("isolated"):
        notes.append(f"isolation {isolation.get('verdict')}: timings remain advisory")
    forbidden = {str(path.name): validate._digest(path) for pattern in
                 ("*_steps*.jsonl", "real.r*_memory*.json") for path in cell.glob(pattern)}
    rows = _plan(args.opening_plan, args.opening_plan_sha256).workload()
    for side in paths:
        manifests = [json.loads(path.read_text()).get("run") or {} for path in paths[side]]
        failures += validate.check_who_served(journals[side], manifests, side)
    for index, (rp, mp) in enumerate(zip(paths["real"], paths["modelled"]), 1):
        runs = {"real": compare.load_run(str(rp), "real"),
                "modelled": compare.load_run(str(mp), "modelled")}
        for side, path in (("real", rp), ("modelled", mp)):
            blob = json.loads(path.read_text())
            execution = blob.get("execution") or {}
            if execution.get("purpose") != "diagnostic":
                failures.append(f"{side}[{index}] lacks diagnostic execution purpose")
            try:
                if identity(execution.get("diagnostic_case") or {}) != identity(case):
                    raise ValueError("execution carries different opening pins")
                check_result(blob, case)
            except (ValueError, TypeError, KeyError) as exc:
                failures.append(f"{side}[{index}]: {exc}")
            run = runs[side]
            failures += compare.check_run(run, expect_requests=2)
            failures += validate.check_clocks_finite(run)
            failures += validate.check_workload_is_registered(run, rows, side)
            failures += validate.check_usage_against_workload(run, rows, side)
            failures += validate.check_engine(run, 1, side, expected_cache_policy=case["cache_policy"])
            failures += check_server_configuration(run.manifest.get("server") or {}, case, side)
            failures += validate.check_cache_policy_evidence(run.manifest, case["cache_policy"], side)
        real, modelled = runs["real"], runs["modelled"]
        failures += compare.check_pair(real, modelled)
        failures += validate.check_side_roles(real, modelled)
        failures += validate.check_source_factory(modelled, 1, f"repeat {index}")
        failures += validate.check_predictor_device_freedom(modelled, f"repeat {index}")
        failures += validate.check_capacity_inputs(modelled, f"repeat {index}")
        failures += validate.check_reference_budget_is_measured(real, f"repeat {index}")
        bad, measured = validate.check_memory_terms(
            real, modelled, cell, index, f"repeat {index}", expected_cache_policy=case["cache_policy"])
        failures += bad
        memory.append(measured)
        failures += validate.check_calibration(
            modelled, registry, 1, case["workload_sha256"], forbidden,
            expected_opening_plan_sha256=case["workload_sha256"])
        for check in (validate.check_scalar_overheads, validate.check_capacity_provenance,
                      validate.check_region_calibration):
            failures += check(modelled, registry, 1, case["workload_sha256"], forbidden)
        reports.append(compare.compare(real, modelled))
    metrics = validate._across_repeats(reports)
    failures += [f"{metric} exceeds its maintained tolerance" for metric, value in metrics.items()
                 if value.get("within_tolerance") is False]
    failures += [f"{metric} has no evaluable paired tolerance" for metric in validate.TOLERANCE_PCT
                 if metrics.get(metric, {}).get("within_tolerance") is None]
    verdict = {"schema": "compass.aiperf_opening_diagnostic/1", "purpose": "diagnostic",
               "accepted": False, "passed": bool(reports) and not failures,
               "case": identity(case), "repeats": len(reports), "metrics": metrics,
               "memory": memory, "isolation": isolation, "failures": failures, "notes": notes,
               "calibration_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
               "cost_accounting": "Use maintained cc_traces_run costs and its per-side partial costs; speed remains advisory"}
    out = Path(args.out) if args.out else cell / "opening_diagnostic.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({"path": str(out), "passed": verdict["passed"], "accepted": False,
                      "failures": failures}, indent=2))
    return 0 if verdict["passed"] else 1
