"""AIPerf opening identity and paired checks for the maintained side harness."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace


# Retain this instance if another harness loads a module with the same name.
_READER = sys.modules[__name__]
CASE_SCHEMA = "compass.aiperf_opening_case/1"
PLAN_KEY = "opening_plan"
EVIDENCE_KEY = "aiperf_opening"
REPORT_SCHEMA = "compass.aiperf_opening_diagnostic/1"
REPORT_NAME = "opening_diagnostic.json"
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
CACHE_REGION_FACTORY = "atom.compass.runtime.cache_region_oracle.source_cost_oracle"
CACHE_REGION_OPTIONS = frozenset((
    "region_overlay", "region_overlay_sha256", "include_failed_outputless", "include_failed_final",
    "diagnostic_only", "q16_handoff", "q16_handoff_sha256",
    "low_q_handoff", "low_q_handoff_sha256", "low_q_allow_failed_spread", "rank_coords",
    "root_prefill_handoff", "root_prefill_handoff_sha256", "root_prefill_allow_failed_spread",
    "region_supplement_handoff", "region_supplement_handoff_sha256",
    "root_prefill_diagnostic_handoff", "root_prefill_diagnostic_handoff_sha256",
    "root_prefill_diagnostic_workload_sha256",
    "native_prefill_handoff", "native_prefill_handoff_sha256",
    "native_ap_handoff", "native_ap_handoff_sha256",
    "reached_primitive_handoffs",
    "diagnostic_reference_handoff", "diagnostic_reference_handoff_sha256",
))


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


def check_producer(producer, *, label):
    """Both named chat profiles require the same pinned Weka reconstruction."""
    producer = producer if isinstance(producer, dict) else {}
    policy = producer.get("weka_reconstruction") or {}
    policy = policy if isinstance(policy, dict) else {}
    effective = policy.get("effective") or {}
    effective = effective if isinstance(effective, dict) else {}
    if (producer.get("aiperf_commit") != AIPERF_COMMIT
            or policy.get("defaults_verified") is not True
            or effective != policy.get("pinned_defaults")
            or effective.get("WEKA_LIVE_ASSISTANT_RESPONSES") is not False
            or effective.get("WEKA_SPLIT_FLATTENED_AGENTS") is not True
            or effective.get("WEKA_TOOL_SHAPED_MESSAGES") is not False):
        raise ValueError(f"{label} case lacks verified effective Weka reconstruction defaults")


def load_case(path, sha, case_id, *, target_model):
    if not re.fullmatch(r"aiperf_opening_[A-Za-z0-9_-]+", case_id):
        raise ValueError("opening case-id must use the explicit aiperf_opening_ prefix")
    plan = _plan(path, sha)
    if plan.model != target_model:
        raise ValueError("opening case targets a different model")
    export = plan.export_identity
    check_producer(export.get("producer"), label="opening")
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


def _source_view(modelled, **changes):
    """A validator-only view; never alter the recorded factory or its options."""
    manifest = modelled.manifest
    server = manifest.get("server") or {}
    return SimpleNamespace(manifest={**manifest, "server": {
        **server, "compass": {**(server.get("compass") or {}), **changes}}})


def calibration_options(case):
    return {"expected_opening_plan_sha256": case["workload_sha256"]}


def _check_diagnostic_input_registration(item, registry, workload_sha, forbidden):
    """Check raw-byte provenance; the exact read name is bound by the aggregate.

    A retained PLAN.json can also be read as EXECUTABLE_PLAN.json. A single
    raw-SHA entry may keep its original name only when it describes exactly
    those bytes. Multi-file or ambiguous registrations never take this route.
    """
    validate = _script("cc_traces_validate")
    entries = [row for row in registry.get("artifacts", []) if row.get("sha256") == item.sha256]
    if len(entries) != 1:
        raise ValueError(f"root diagnostic input {item.path} needs one raw-SHA registry entry")
    contents = entries[0].get("contents")
    if not isinstance(contents, dict) or len(contents) != 1 or list(contents.values()) != [item.sha256]:
        raise ValueError(f"root diagnostic input {item.path} needs a singleton registration of its actual bytes")
    return validate._check_calibration_records(
        {item.role: item.sha256}, {item.role: contents}, registry, 1, workload_sha, forbidden)


def check_source_contract(modelled, registry, workload_sha, forbidden, label, *, diagnostic_status=None):
    """Check the exact diagnostic wrapper, retaining the base protocol checks.

    Pair validation needs the pinned overlay and optional source bundles mounted
    at their configured paths. Reopening verifies those bytes against the
    worker's original read; it never replaces the worker's attestation. This
    lets the selected region snapshot be rebuilt with the actual flags, rather
    than accepting an arbitrary region name from the record itself.
    """
    validate = _script("cc_traces_validate")
    compass = (modelled.manifest.get("server") or {}).get("compass") or {}
    if compass.get("oracle") != CACHE_REGION_FACTORY:
        return (validate.check_source_factory(modelled, 1, label)
                + validate.check_region_calibration(modelled, registry, 1, workload_sha, forbidden)), []
    options = dict(compass.get("oracle_options") or {})
    base_options = {key: value for key, value in options.items() if key not in CACHE_REGION_OPTIONS}
    bad = validate.check_source_factory(
        _source_view(modelled, oracle=validate.SOURCE_FACTORY, oracle_options=base_options), 1, label)
    notes = []
    try:
        from atom.compass.runtime import cache_region_oracle as wrapper
        from atom.compass.runtime.source_oracle import _flag, _rank_coords, region_snapshot
        from atom.compass.core.loaded_input import load_json

        expected = {"model": "Qwen/Qwen3.8-27B", "tp": 1, "block_size": 16,
                    "max_model_len": 262144, "position_rows": 3,
                    "cudagraph_mode": "full", "allocation": "native"}
        for key, value in expected.items():
            actual = options.get(key)
            if isinstance(value, int):
                actual = int(actual) if actual is not None else None
            elif key == "cudagraph_mode":
                actual = str(actual).lower()
            if actual != value:
                raise ValueError(f"cached-prefill source scope requires {key}={value!r}")
        if any(rank != 0 for rank in _rank_coords(options.get("rank_coords")).values()):
            raise ValueError("cached-prefill source scope requires rank zero")
        ranks = validate._rank_records(modelled)
        if len(ranks) != 1 or any(value != 0 for value in (ranks[0].get("rank_coords") or {}).values()):
            raise ValueError("cached-prefill source needs one TP1 reader at rank zero")
        inputs = ranks[0].get("inputs") or []

        def pinned(option, role):
            path, sha = options.get(option), options.get(option + "_sha256")
            rows = [row for row in inputs if row.get("role") == role]
            if (not isinstance(path, str) or not path or not validate._hexish(sha)
                    or len(rows) != 1 or rows[0].get("requested") != path
                    or rows[0].get("sha256") != sha):
                raise ValueError(f"{option} needs one loaded input matching its explicit path/SHA-256")
            data, loaded = load_json(path, role=role)
            if loaded.sha256 != sha or loaded.size != rows[0].get("size"):
                raise ValueError(f"{option} bytes differ from its pinned worker read")
            # The option summary is not enough for new roles: independently
            # retain the existing source provenance/leakage check on this read.
            bad.extend(validate._check_calibration_records(
                {role: sha}, {role: {Path(rows[0]["path"]).name: sha}},
                registry, 1, workload_sha, forbidden))
            return data

        overlay = pinned("region_overlay", "oracle.region_overlay")
        if options.get("regions") != overlay["base"]["name"]:
            raise ValueError("region overlay and requested base preset differ")
        include_failed = _flag(options.get("include_failed_outputless", False), "include_failed_outputless")
        include_failed_final = _flag(options.get("include_failed_final", False), "include_failed_final")
        diagnostic = _flag(options.get("diagnostic_only", False), "diagnostic_only")
        if options.get("root_prefill_diagnostic_handoff") and (
                not diagnostic or options.get("root_prefill_handoff") or options.get("region_supplement_handoff")):
            raise ValueError("root diagnostic source requires separate explicit diagnostic selection")
        selected = wrapper.model_from_artifact(
            overlay, include_failed_outputless=include_failed,
            include_failed_final=include_failed_final, diagnostic_only=diagnostic)
        selected_name = overlay["name"]
        if include_failed:
            notes.append("FAILED outputless source qualification retained; diagnostic_only=1; no acceptance credit")
        if include_failed_final:
            notes.append("FAILED final-query transfer retained with unchanged q16 formula; diagnostic_only=1; no acceptance credit")
        if bool(options.get("q16_handoff")) != bool(options.get("q16_handoff_sha256")):
            raise ValueError("q16 source handoff and its SHA-256 are required together")
        if options.get("q16_handoff"):
            pinned("q16_handoff", "oracle.q16_sources")
            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if (len(scopes) != 1 or not options.get("attention_scope")
                    or scopes[0].get("requested") != options["attention_scope"]
                    or scopes[0].get("sha256") != (overlay.get("q16_request_scope") or {}).get("sha256")):
                raise ValueError("q16 addition differs from its pinned deployment request scope")
            from atom.compass.core.cost.cached_q16 import CachedQ16Prices
            from atom.compass.core.cost.library import PriceLibrary
            base = PriceLibrary()
            base.launch_charge_seconds = float(options.get("seconds_per_launch", 0))
            q16 = CachedQ16Prices(base, options["q16_handoff"], options["q16_handoff_sha256"])
            for source in q16.loaded_inputs:
                matches = [row for row in inputs if row.get("role") == source.role
                           and row.get("path") == source.path and row.get("sha256") == source.sha256]
                if len(matches) != 1:
                    raise ValueError(f"q16 source {source.path} lacks its exact loaded-input identity")
        elif any(row.get("role") == "oracle.q16_sources" for row in inputs):
            raise ValueError("unconfigured q16 source handoff was loaded")

        if bool(options.get("low_q_handoff")) != bool(options.get("low_q_handoff_sha256")):
            raise ValueError("low-query source handoff and its SHA-256 are required together")
        if options.get("low_q_handoff"):
            pinned("low_q_handoff", "oracle.low_q_sources")
            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if (len(scopes) != 1 or not options.get("attention_scope")
                    or scopes[0].get("requested") != options["attention_scope"]):
                raise ValueError("low-query addition lacks its loaded deployment request scope")
            from atom.compass.core.cost.low_query import LowQueryPrices
            from atom.compass.core.cost.library import PriceLibrary
            base = PriceLibrary()
            base.launch_charge_seconds = float(options.get("seconds_per_launch", 0))
            low_q = LowQueryPrices(base, options["low_q_handoff"], options["low_q_handoff_sha256"],
                deployment_scope_sha256=scopes[0]["sha256"], diagnostic_only=diagnostic,
                allow_failed_spread=_flag(options.get("low_q_allow_failed_spread", False),
                                         "low_q_allow_failed_spread"))
            for source in low_q.loaded_inputs:
                matches = [row for row in inputs if row == source.as_dict()]
                if len(matches) != 1:
                    raise ValueError(f"low-query source {source.path} lacks its exact loaded-input identity")
                bad.extend(validate._check_calibration_records(
                    {source.role: source.sha256},
                    {source.role: {Path(source.path).name: source.sha256}},
                    registry, 1, workload_sha, forbidden))
            if not low_q.source_qualified:
                notes.append("FAILED low-query heldout spread retained; diagnostic_only=1; no acceptance credit: "
                             + ", ".join(low_q.failed_spread_controls))
            elif not low_q.campaign_source_qualified:
                notes.append("Selected low-query sources qualified; full source campaign remains unqualified")
        elif any(str(row.get("role", "")).startswith("oracle.low_q_") for row in inputs):
            raise ValueError("unconfigured low-query source evidence was loaded")
        if bool(options.get("root_prefill_handoff")) != bool(options.get("root_prefill_handoff_sha256")):
            raise ValueError("root prefill handoff and its SHA-256 are required together")
        if options.get("root_prefill_handoff"):
            pinned("root_prefill_handoff", "oracle.root_prefill_sources")
            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if (len(scopes) != 1 or not options.get("attention_scope")
                    or scopes[0].get("requested") != options["attention_scope"]):
                raise ValueError("root prefill addition lacks its loaded deployment request scope")
            from atom.compass.core.cost.root_prefill import ExactPrefillRegions, RootPrefillPrices
            from atom.compass.core.cost.library import PriceLibrary
            base = PriceLibrary()
            base.launch_charge_seconds = float(options.get("seconds_per_launch", 0))
            root_prefill = RootPrefillPrices(base, options["root_prefill_handoff"], options["root_prefill_handoff_sha256"],
                deployment_scope_sha256=scopes[0]["sha256"], diagnostic_only=diagnostic,
                allow_failed_spread=_flag(options.get("root_prefill_allow_failed_spread", False),
                                         "root_prefill_allow_failed_spread"))
            files = {}
            for source in root_prefill.loaded_inputs:
                if sum(row == source.as_dict() for row in inputs) != 1:
                    raise ValueError(f"root prefill source {source.path} lacks its exact loaded-input identity")
                bad.extend(validate._check_calibration_records(
                    {source.role: source.sha256}, {source.role: {Path(source.path).name: source.sha256}},
                    registry, 1, workload_sha, forbidden))
                if source.role.startswith("oracle.root_prefill_"):
                    files[Path(source.path).name] = source.sha256
            digest = next(iter(files.values())) if len(files) == 1 else validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("root_prefill_handoff") != files
                    or (compass.get("oracle_option_sha256") or {}).get("root_prefill_handoff") != digest):
                raise ValueError("root prefill aggregate option does not retain every loaded evidence file")
            bad.extend(validate._check_calibration_records(
                {"root_prefill_handoff": digest}, {"root_prefill_handoff": files},
                registry, 1, workload_sha, forbidden))
            selected = ExactPrefillRegions(selected, root_prefill.region_points, root_prefill.handoff_sha256)
            selected_name = "root-prefill-sources"
            if not root_prefill.source_qualified:
                notes.append("FAILED root prefill heldout spread retained; diagnostic_only=1; no acceptance credit")
        elif any(str(row.get("role", "")).startswith("oracle.root_prefill_") for row in inputs):
            raise ValueError("unconfigured root prefill source evidence was loaded")
        if bool(options.get("region_supplement_handoff")) != bool(options.get("region_supplement_handoff_sha256")):
            raise ValueError("region supplement handoff and its SHA-256 are required together")
        if options.get("region_supplement_handoff"):
            if not options.get("root_prefill_handoff"):
                raise ValueError("region supplement requires the original root prefill handoff")
            from atom.compass.core.cost.region_supplement import PrefillRegionSupplement
            selected = PrefillRegionSupplement.load(selected, root_prefill,
                options["region_supplement_handoff"], options["region_supplement_handoff_sha256"],
                deployment_scope_sha256=scopes[0]["sha256"])
            pinned("region_supplement_handoff", "oracle.region_supplement_sources")
            files = {}
            for source in selected.loaded_inputs:
                if sum(row == source.as_dict() for row in inputs) != 1:
                    raise ValueError(f"region supplement source {source.path} lacks its exact loaded-input identity")
                bad.extend(validate._check_calibration_records(
                    {source.role: source.sha256}, {source.role: {Path(source.path).name: source.sha256}},
                    registry, 1, workload_sha, forbidden))
                files[Path(source.path).name] = source.sha256
            digest = validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("region_supplement_handoff") != files
                    or (compass.get("oracle_option_sha256") or {}).get("region_supplement_handoff") != digest):
                raise ValueError("region supplement aggregate option omits loaded evidence")
            bad.extend(validate._check_calibration_records(
                {"region_supplement_handoff": digest}, {"region_supplement_handoff": files},
                registry, 1, workload_sha, forbidden))
            selected_name = "root-prefill-supplement"
        elif any(str(row.get("role", "")).startswith("oracle.region_supplement_") for row in inputs):
            raise ValueError("unconfigured region supplement evidence was loaded")
        if (bool(options.get("root_prefill_diagnostic_handoff")) != bool(options.get("root_prefill_diagnostic_handoff_sha256"))
                or bool(options.get("root_prefill_diagnostic_handoff")) != bool(options.get("root_prefill_diagnostic_workload_sha256"))):
            raise ValueError("root diagnostic handoff, SHA-256 and fixed workload are required together")
        if options.get("root_prefill_diagnostic_handoff"):
            from atom.compass.core.cost.root_diagnostic import DiagnosticRootPrefillPrices, ROLE_PREFIX
            from atom.compass.core.cost.root_prefill import ExactPrefillRegions
            from atom.compass.core.cost.library import PriceLibrary
            from atom.compass.core.loaded_input import file_digests

            if options["root_prefill_diagnostic_workload_sha256"] != workload_sha:
                raise ValueError("root diagnostic source is bound to another workload")
            pinned("root_prefill_diagnostic_handoff", ROLE_PREFIX + "sources")
            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if len(scopes) != 1 or scopes[0].get("requested") != options.get("attention_scope"):
                raise ValueError("root diagnostic source lacks its loaded deployment request scope")
            base = PriceLibrary()
            base.launch_charge_seconds = float(options.get("seconds_per_launch", 0))
            source = DiagnosticRootPrefillPrices(base, options["root_prefill_diagnostic_handoff"],
                options["root_prefill_diagnostic_handoff_sha256"], deployment_scope_sha256=scopes[0]["sha256"],
                workload_sha256=workload_sha, diagnostic_only=diagnostic)
            if (sum(str(row.get("role", "")).startswith(ROLE_PREFIX) for row in inputs)
                    != sum(item.role.startswith(ROLE_PREFIX) for item in source.loaded_inputs)):
                raise ValueError("root diagnostic worker input roles differ from the actual source reads")
            for item in source.loaded_inputs:
                if sum(row == item.as_dict() for row in inputs) != 1:
                    raise ValueError(f"root diagnostic input {item.path} lacks its exact role/path/digest identity")
                bad.extend(_check_diagnostic_input_registration(item, registry, workload_sha, forbidden))
            files = file_digests(item for item in source.loaded_inputs if item.role.startswith(ROLE_PREFIX))
            digest = validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("root_prefill_diagnostic_handoff") != files
                    or (compass.get("oracle_option_sha256") or {}).get("root_prefill_diagnostic_handoff") != digest):
                raise ValueError("root diagnostic aggregate omits or changes an actual loaded source path")
            bad.extend(validate._check_calibration_records(
                {"root_prefill_diagnostic_handoff": digest}, {"root_prefill_diagnostic_handoff": files},
                registry, 1, workload_sha, forbidden))
            selected = ExactPrefillRegions(selected, source.region_points, source.handoff_sha256)
            selected = ExactPrefillRegions(selected, source.supplement_points, source.handoff_sha256)
            selected_name = "root-reference-diagnostic"
            if diagnostic_status is not None:
                diagnostic_status.update(source.diagnostic_status)
            notes.append("Unqualified root reference diagnostic: R5 validation 171/348 incomplete with two MHA spread failures; "
                         "supplement 17/20 with three failed prepare gates and outer exit 124; no acceptance credit")
        elif any(str(row.get("role", "")).startswith("oracle.root_diagnostic_") for row in inputs):
            raise ValueError("unconfigured root diagnostic evidence was loaded")
        reached_role = "oracle.reached_primitives."
        if options.get("reached_primitive_handoffs"):
            from atom.compass.core.cost.library import PriceLibrary
            from atom.compass.core.cost.reached_primitives import ReachedPrimitivePrices
            from atom.compass.core.loaded_input import file_digests

            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if len(scopes) != 1 or scopes[0].get("requested") != options.get("attention_scope"):
                raise ValueError("reached primitive sources lack their loaded deployment scope")
            reached = ReachedPrimitivePrices(PriceLibrary(), options["reached_primitive_handoffs"],
                deployment_scope_sha256=scopes[0]["sha256"])
            observed = [row for row in inputs if str(row.get("role", "")).startswith(reached_role)]
            if len(observed) != len(reached.loaded_inputs):
                raise ValueError("reached primitive source input inventory differs")
            by_digest = {}
            for item in reached.loaded_inputs:
                if sum(row == item.as_dict() for row in observed) != 1:
                    raise ValueError(f"reached primitive input {item.path} lacks its exact loaded identity")
                by_digest.setdefault(item.sha256, []).append(item)
            for sha, items in by_digest.items():
                contents = file_digests(items)
                digest = sha if len(contents) == 1 else validate._rolled_digest(contents)
                role = reached_role + sha[:16]
                bad.extend(validate._check_calibration_records(
                    {role: digest}, {role: contents}, registry, 1, workload_sha, forbidden))
            files = file_digests(reached.loaded_inputs)
            digest = next(iter(files.values())) if len(files) == 1 else validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("reached_primitive_handoffs") != files
                    or (compass.get("oracle_option_sha256") or {}).get("reached_primitive_handoffs") != digest):
                raise ValueError("reached primitive aggregate omits or changes loaded evidence")
            bad.extend(validate._check_calibration_records(
                {"reached_primitive_handoffs": digest}, {"reached_primitive_handoffs": files},
                registry, 1, workload_sha, forbidden))
        elif any(str(row.get("role", "")).startswith(reached_role) for row in inputs):
            raise ValueError("unconfigured reached primitive source evidence was loaded")
        from atom.compass.core.cost.diagnostic_references import DiagnosticReferencePrices, ROLE_PREFIX

        if bool(options.get("diagnostic_reference_handoff")) != bool(options.get("diagnostic_reference_handoff_sha256")):
            raise ValueError("diagnostic reference handoff and its SHA-256 are required together")
        if options.get("diagnostic_reference_handoff"):
            from atom.compass.core.cost.library import PriceLibrary
            from atom.compass.core.loaded_input import file_digests

            if not diagnostic or not _flag(options.get("require_complete", True), "require_complete"):
                raise ValueError("diagnostic references require diagnostic mode and complete coverage")
            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if len(scopes) != 1 or scopes[0].get("requested") != options.get("attention_scope"):
                raise ValueError("diagnostic references lack their loaded deployment scope")
            source = DiagnosticReferencePrices(PriceLibrary(), options["diagnostic_reference_handoff"],
                options["diagnostic_reference_handoff_sha256"], deployment_scope_sha256=scopes[0]["sha256"],
                diagnostic_only=True)
            observed = [row for row in inputs if str(row.get("role", "")).startswith(ROLE_PREFIX)]
            if len(observed) != len(source.loaded_inputs):
                raise ValueError("diagnostic reference input inventory differs")
            by_digest = {}
            for item in source.loaded_inputs:
                if sum(row == item.as_dict() for row in observed) != 1:
                    raise ValueError(f"diagnostic reference input {item.path} lacks its exact loaded identity")
                by_digest.setdefault(item.sha256, []).append(item)
            for sha, items in by_digest.items():
                contents = file_digests(items)
                digest = sha if len(contents) == 1 else validate._rolled_digest(contents)
                role = ROLE_PREFIX + sha[:16]
                bad.extend(validate._check_calibration_records(
                    {role: digest}, {role: contents}, registry, 1, workload_sha, forbidden))
            files = file_digests(source.loaded_inputs)
            digest = next(iter(files.values())) if len(files) == 1 else validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("diagnostic_reference_handoff") != files
                    or (compass.get("oracle_option_sha256") or {}).get("diagnostic_reference_handoff") != digest):
                raise ValueError("diagnostic reference aggregate omits or changes loaded evidence")
            bad.extend(validate._check_calibration_records(
                {"diagnostic_reference_handoff": digest}, {"diagnostic_reference_handoff": files},
                registry, 1, workload_sha, forbidden))
            notes.append("Diagnostic reference medians and frozen bounded-prefix MHA predictions; "
                         "all original qualification failures retained; no acceptance credit")
        elif any(str(row.get("role", "")).startswith(ROLE_PREFIX) for row in inputs):
            raise ValueError("unconfigured diagnostic reference evidence was loaded")
        if bool(options.get("native_prefill_handoff")) != bool(options.get("native_prefill_handoff_sha256")):
            raise ValueError("native prefill handoff and its SHA-256 are required together")
        native_role = "oracle.native_prefill_regions"
        if options.get("native_prefill_handoff"):
            from atom.compass.core.cost.native_prefill_regions import NativePrefillRegions
            from atom.compass.core.loaded_input import file_digests
            from atom.compass.runtime.templates import NativeAllocation

            pinned("native_prefill_handoff", native_role)
            scopes = [row for row in inputs if row.get("role") == "oracle.attention_scope"]
            if len(scopes) != 1 or scopes[0].get("requested") != options.get("attention_scope"):
                raise ValueError("native prefill sources lack their loaded deployment attention scope")
            allocation = NativeAllocation(block_size=16, max_model_len=262144,
                                          position_rows=3, cudagraph_mode="full")
            selected = NativePrefillRegions.load(
                selected, options["native_prefill_handoff"], options["native_prefill_handoff_sha256"],
                allocation, deployment_scope_sha256=scopes[0]["sha256"])
            if not selected.source_qualified:
                raise ValueError("native prefill source is a review candidate; qualification is required")
            observed = [row for row in inputs if row.get("role") == native_role
                        or str(row.get("role", "")).startswith(native_role + ".")]
            if len(observed) != len(selected.loaded_inputs):
                raise ValueError("native prefill source input inventory differs")
            for item in selected.loaded_inputs:
                if sum(row == item.as_dict() for row in observed) != 1:
                    raise ValueError(f"native prefill input {item.path} lacks its exact loaded identity")
                bad.extend(validate._check_calibration_records(
                    {item.role: item.sha256}, {item.role: {Path(item.path).name: item.sha256}},
                    registry, 1, workload_sha, forbidden))
            files = file_digests(selected.loaded_inputs)
            digest = validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("native_prefill_handoff") != files
                    or (compass.get("oracle_option_sha256") or {}).get("native_prefill_handoff") != digest):
                raise ValueError("native prefill aggregate omits or changes loaded source evidence")
            bad.extend(validate._check_calibration_records(
                {"native_prefill_handoff": digest}, {"native_prefill_handoff": files},
                registry, 1, workload_sha, forbidden))
            selected_name = "native-prefill-sources"
        elif any(row.get("role") == native_role
                 or str(row.get("role", "")).startswith(native_role + ".") for row in inputs):
            raise ValueError("unconfigured native prefill source evidence was loaded")
        if bool(options.get("native_ap_handoff")) != bool(options.get("native_ap_handoff_sha256")):
            raise ValueError("native A/P handoff and its SHA-256 are required together")
        family_role = "oracle.native_ap_regions"
        if options.get("native_ap_handoff"):
            from atom.compass.core.cost.native_ap_regions import NativeAPFamilyRegions

            if not options.get("native_prefill_handoff"):
                raise ValueError("native A/P families lack their retained native-prefill source")
            pinned("native_ap_handoff", family_role)
            selected = NativeAPFamilyRegions.load(selected, options["native_ap_handoff"],
                options["native_ap_handoff_sha256"], allocation, deployment_scope_sha256=scopes[0]["sha256"])
            if not selected.source_qualified:
                raise ValueError("native A/P source is a review candidate; family qualification is required")
            observed = [row for row in inputs if row.get("role") == family_role
                        or str(row.get("role", "")).startswith(family_role + ".")]
            if len(observed) != len(selected.loaded_inputs):
                raise ValueError("native A/P source input inventory differs")
            by_digest = {}
            for item in selected.loaded_inputs:
                if sum(row == item.as_dict() for row in observed) != 1:
                    raise ValueError(f"native A/P input {item.path} lacks its exact loaded identity")
                by_digest.setdefault(item.sha256, []).append(item)
            for sha, items in by_digest.items():
                role = family_role + "." + sha[:16]
                contents = file_digests(items)
                digest = sha if len(contents) == 1 else validate._rolled_digest(contents)
                bad.extend(validate._check_calibration_records(
                    {role: digest}, {role: contents},
                    registry, 1, workload_sha, forbidden))
            files = file_digests(selected.loaded_inputs)
            digest = validate._rolled_digest(files)
            if ((compass.get("oracle_option_files") or {}).get("native_ap_handoff") != files
                    or (compass.get("oracle_option_sha256") or {}).get("native_ap_handoff") != digest):
                raise ValueError("native A/P aggregate omits or changes loaded source evidence")
            bad.extend(validate._check_calibration_records(
                {"native_ap_handoff": digest}, {"native_ap_handoff": files}, registry, 1, workload_sha, forbidden))
            selected_name = "native-ap-families"
        elif any(row.get("role") == family_role
                 or str(row.get("role", "")).startswith(family_role + ".") for row in inputs):
            raise ValueError("unconfigured native A/P source evidence was loaded")
        snapshot = region_snapshot(selected_name, selected)
        if ranks[0].get("regions") != snapshot:
            raise ValueError("selected region snapshot differs from its loaded overlay and flags")
        selected_options = dict(options, regions=selected_name)
        bad.extend(validate.check_region_calibration(
            _source_view(modelled, oracle_options=selected_options), registry, 1, workload_sha, forbidden))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        bad.append(f"{label}: opening source contract: {exc}")
    return bad, notes


def pair(args):
    try:
        return _pair(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"opening pair refused: {exc}", file=sys.stderr)
        return 2


def _pair(args):
    return _pair_case(args, _READER)


def _pair_case(args, case_reader):
    """Compose existing offline checks; this route never issues matrix credit."""
    validate, compare = _script("cc_traces_validate"), _script("compare")
    plan_module = _script("cc_traces_plan")
    case = case_reader.load_case(
        getattr(args, case_reader.PLAN_KEY), getattr(args, case_reader.PLAN_KEY + "_sha256"),
        args.case_id, target_model=plan_module.MODEL)
    cell = Path(args.cell)
    if cell.name != f"tp1_{case['case_id']}_c{case['clients']}":
        raise ValueError("chat diagnostic directory disagrees with its case identity")
    lock = json.loads((cell / "diagnostic_case.json").read_text())
    if case_reader.identity(lock) != case_reader.identity(case):
        raise ValueError("chat diagnostic directory belongs to different pins")
    registry_path = Path(args.calibration_registry)
    registry = json.loads(registry_path.read_text())
    paths = {side: validate._runs(cell, side) for side in ("real", "modelled")}
    failures, notes, reports, memory, sources = [], [case["qualification"]], [], [], []
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
    forbidden.update({item["path"]: item["sha256"] for item in case.get("workload_inputs", [])})
    rows = case_reader._plan(**case[case_reader.PLAN_KEY]).workload()
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
                if case_reader.identity(execution.get("diagnostic_case") or {}) != case_reader.identity(case):
                    raise ValueError("execution carries different chat diagnostic pins")
                case_reader.check_result(blob, case)
            except (ValueError, TypeError, KeyError) as exc:
                failures.append(f"{side}[{index}]: {exc}")
            run = runs[side]
            failures += compare.check_run(run, expect_requests=case["requests"])
            failures += validate.check_clocks_finite(run)
            failures += validate.check_workload_is_registered(run, rows, side)
            failures += validate.check_usage_against_workload(run, rows, side)
            failures += validate.check_engine(run, 1, side, expected_cache_policy=case["cache_policy"])
            failures += case_reader.check_server_configuration(run.manifest.get("server") or {}, case, side)
            failures += validate.check_cache_policy_evidence(run.manifest, case["cache_policy"], side)
        real, modelled = runs["real"], runs["modelled"]
        failures += compare.check_pair(real, modelled)
        failures += validate.check_side_roles(real, modelled)
        compass = (modelled.manifest.get("server") or {}).get("compass") or {}
        options = compass.get("oracle_options") or {}
        diagnostic_source_status = {}
        source_detail = ({"diagnostic_status": diagnostic_source_status}
                         if options.get("root_prefill_diagnostic_handoff") else {})
        source_bad, source_notes = case_reader.check_source_contract(
            modelled, registry, case["workload_sha256"], forbidden, f"repeat {index}", **source_detail)
        failures += source_bad
        notes += [note for note in source_notes if note not in notes]
        sources.append({"repeat": index, "observed_oracle": compass.get("oracle"),
                        "base_validation_factory": validate.SOURCE_FACTORY,
                        "region_overlay_sha256": options.get("region_overlay_sha256"),
                        "q16_handoff_sha256": options.get("q16_handoff_sha256"),
                        "low_q_handoff_sha256": options.get("low_q_handoff_sha256"),
                        "low_q_allow_failed_spread": options.get("low_q_allow_failed_spread", False),
                        "root_prefill_handoff_sha256": options.get("root_prefill_handoff_sha256"),
                        "root_prefill_allow_failed_spread": options.get("root_prefill_allow_failed_spread", False),
                        "region_supplement_handoff_sha256": options.get("region_supplement_handoff_sha256"),
                        "include_failed_outputless": options.get("include_failed_outputless", False),
                        "include_failed_final": options.get("include_failed_final", False),
                        "diagnostic_only": options.get("diagnostic_only", False)})
        if options.get("root_prefill_diagnostic_handoff"):
            sources[-1].update(root_prefill_diagnostic_handoff_sha256=options["root_prefill_diagnostic_handoff_sha256"],
                               diagnostic_status=diagnostic_source_status, source_qualified=False)
        failures += validate.check_predictor_device_freedom(modelled, f"repeat {index}")
        failures += validate.check_capacity_inputs(modelled, f"repeat {index}")
        failures += validate.check_reference_budget_is_measured(real, f"repeat {index}")
        bad, measured = validate.check_memory_terms(
            real, modelled, cell, index, f"repeat {index}", expected_cache_policy=case["cache_policy"])
        failures += bad
        memory.append(measured)
        failures += validate.check_calibration(
            modelled, registry, 1, case["workload_sha256"], forbidden,
            **case_reader.calibration_options(case))
        for check in (validate.check_scalar_overheads, validate.check_capacity_provenance):
            failures += check(modelled, registry, 1, case["workload_sha256"], forbidden)
        reports.append(compare.compare(real, modelled))
    metrics = validate._across_repeats(reports)
    failures += [f"{metric} exceeds its maintained tolerance" for metric, value in metrics.items()
                 if value.get("within_tolerance") is False]
    failures += [f"{metric} has no evaluable paired tolerance" for metric in validate.TOLERANCE_PCT
                 if metrics.get(metric, {}).get("within_tolerance") is None]
    verdict = {"schema": case_reader.REPORT_SCHEMA, "purpose": "diagnostic",
               "accepted": False, "passed": bool(reports) and not failures,
               "case": case_reader.identity(case), "repeats": len(reports), "metrics": metrics,
               "memory": memory, "source_contracts": sources,
               "isolation": isolation, "failures": failures, "notes": notes,
               "calibration_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
               "cost_accounting": "Use maintained cc_traces_run costs and its per-side partial costs; speed remains advisory"}
    out = Path(args.out) if args.out else cell / case_reader.REPORT_NAME
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({"path": str(out), "passed": verdict["passed"], "accepted": False,
                      "failures": failures}, indent=2))
    return 0 if verdict["passed"] else 1
