"""Explicit diagnostic spending of complete, unqualified root reference data.

This loader never qualifies a source. Incomplete/failed validation is evidence
to retain, while missing or inconsistent reference data remains a refusal.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from atom.compass.core.cost.library import PriceLibrary, _signature_of
from atom.compass.core.cost.root_prefill import RootPrefillPrices, validate_region_binding
from atom.compass.core.cost.region_supplement import _candidate_points, _summary
from atom.compass.core.loaded_input import LoadedInput, load_json

SCHEMA = "compass.root_prefill_diagnostic_reference_export/1"
ROLE_PREFIX = "oracle.root_diagnostic_"
WORKLOAD_SHA256 = "c0769dfd6274b26cfb185e34fdb9da0f441ea0f8d61702b5efa81aec30d33d10"
REQUEST_SCOPE_SHA256 = "2e62db60be4f07cddc6af7ae7ad0bc6825f1f780ae778e10e6faa0b741c2ac47"
EVIDENCE = (
    "plan", "proposal", "primitive_freeze", "primitive_reference_phase", "primitive_partial_phase",
    "primitive_preflight", "primitive_failure", "primitive_cpu_failure", "primitive_closeout",
    "prior_plan", "prior_freeze", "prior_verdict", "prior_preflight", "r4_freeze", "r4_failure",
    "region_plan", "region_freeze", "region_verdict", "region_parent_failure",
    *(f"region_event_{i}" for i in range(1, 7)),
    *(f"primitive_event_{i}" for i in range(1, 5)),
    "supplement_candidate", "supplement_plan", "supplement_verdict", "supplement_native_complete",
    "supplement_phase", "supplement_handoff", "supplement_closeout", "supplement_outer_exit",
)


class DiagnosticEvidence:
    """Read each role/path once; retain distinct roles and duplicate basenames."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.inputs, self._values = [], {}

    def read(self, pin, role, *, json_data=True):
        path = str(self.directory / pin["path"])
        role = ROLE_PREFIX + role
        key = role, path, pin["sha256"]
        if key in self._values:
            return self._values[key]
        if json_data:
            value, loaded = load_json(path, role=role)
        else:
            raw = Path(path).read_bytes()
            value = raw
            loaded = LoadedInput(role, path, path, False, hashlib.sha256(raw).hexdigest(), len(raw))
        if loaded.sha256 != pin["sha256"]:
            raise ValueError(f"diagnostic source pin changed: {path}")
        if any(x.path == loaded.path and x.sha256 != loaded.sha256 for x in self.inputs):
            raise ValueError("diagnostic source changed between reads")
        self.inputs.append(loaded)
        self._values[key] = value
        return value


def _child(reader, plan, case, row, raw, phase):
    if case["family"] != "gemm":
        if row.get("child_execution") is not None:
            raise ValueError("non-GEMM diagnostic reference has a child execution")
        return
    execution = reader.read(row["child_execution"], phase + "_execution")
    job = reader.read(execution["job"], phase + "_job")
    source = reader.read(plan, "plan")
    repeat, ordinal = job["repeat"], job["ordinal"]
    orders = source[phase + "_order_by_repeat"]
    if (source.get("gemm_execution_location") != "one_fresh_process_per_observation"
            or job.get("schema") != "compass.gemm_source_job/1"
            or type(repeat) is not int or not 1 <= repeat <= len(orders)
            or type(ordinal) is not int or not 0 <= ordinal < len(orders[repeat - 1])
            or orders[repeat - 1][ordinal] != case["cell_id"]
            or job["seed"] != {"reference": 314159, "heldout": 271828}[phase] + repeat * 1000 + ordinal
            or job["iters"] != case["iters"] or job["source"] != source["source"]
            or job["source_files"] != source["source_files"]
            or job["working_set_budget_bytes"] != source["argument_set_policy"]["working_set_budget_bytes"]
            or job.get("source_consumption_forbidden") is not False
            or job["child_script"] != source["code"]["gemm_child"]):
        raise ValueError("diagnostic child changed its declared source/ordinal/seed/treatment")
    complete = reader.read(execution["child_complete"], phase + "_child_complete")
    child_raw = reader.read(execution["raw"], phase + "_child_raw")
    location = raw["provenance"].get("execution_location") or {}
    if (job["plan"]["sha256"] != plan["sha256"] or job["phase"] != phase
            or job["repeat"] != row["repeat"] or job["seed"] != row["seed"] or job["case"] != case
            or execution["normal_child_exit"] != 0 or execution["source_consumption_forbidden"] is not False
            or complete["job"] != execution["job"] or complete["child"] != execution["child"]
            or complete["raw"] != execution["raw"] or child_raw != raw
            or complete.get("source_consumption_forbidden") is not False
            or execution["raw"]["sha256"] != row["raw"]["sha256"]
            or any(execution[k].get("acknowledged") is not True for k in ("parent_pre_fence", "parent_post_fence"))
            or location.get("timing_and_profile_same_capture") is not True
            or raw.get("schema") != "compass.gemm_source_price/1"
            or raw.get("source_consumption_forbidden") is not False
            or location.get("kind") != "fresh_exec_interpreter"
            or location.get("pid") != execution["child"]["pid"]
            or location.get("source_graph") != case["graph"] or location.get("runtime") != job["runtime"]
            or location.get("seed") != job["seed"]):
        raise ValueError("diagnostic reference child provenance is incomplete or inconsistent")


def _raw_price(reader, case, pin, role):
    raw = reader.read(pin, role)
    if (raw.get("diagnostic_only") is True or raw.get("source_consumption_forbidden") is True
            or raw.get("unpriced") or set(raw.get("prices", {})) != {case["signature"]}):
        raise ValueError("diagnostic oracle requires actual source references, not probe or incomplete prices")
    value = raw["prices"][case["signature"]]
    if (any(value.get(k) != case[v] for k, v in
            (("cache", "observed_cache"), ("arg_sets", "arg_sets"), ("kv_regions", "kv_regions")))
            or type(value.get("seconds")) not in (int, float)
            or not math.isfinite(value["seconds"]) or value["seconds"] <= 0
            or raw["provenance"].get("iters") != case["iters"]
            or raw["provenance"].get("graph") != case["graph"]["path"]
            or raw["provenance"].get("only") != case["only"]
            or raw["provenance"].get("topology") != {"tp": 1}):
        raise ValueError("diagnostic reference timing treatment or value changed")
    return raw, value


def _graph(reader, case, role):
    graph = reader.read(case["graph"], role)
    if len(graph.get("ops", [])) != 1 or _signature_of(graph["ops"][0]) != case["signature"]:
        raise ValueError("diagnostic reference graph differs from its declared signature")
    return graph


def _references(reader, handoff, evidence):
    plan, frozen, proposal = (evidence[k] for k in ("plan", "primitive_freeze", "proposal"))
    pins = handoff["evidence"]
    phase, partial = evidence["primitive_reference_phase"], evidence["primitive_partial_phase"]
    new = {c["cell_id"]: c for c in plan["cases"] if c["phase"] == "reference"}
    prior_cases = {c["cell_id"]: c for c in evidence["prior_plan"]["cases"] if c["phase"] == "reference"}
    retained = set(plan["retained_sources"]["reference_cells"])
    active_old = {c["cell_id"] for c in proposal["reuse"]["active_reference_points"]}
    expected = [(r, name) for r, order in enumerate(plan["reference_order_by_repeat"], 1) for name in order]
    if (plan.get("schema") != "compass.root1493_executable/1"
            or len(new) != 39 or len(retained) != 23 or len(active_old) != 21
            or active_old | set(proposal["reuse"]["dependency_only_reference_cells"]) != retained
            or "reference_gemm_M9_N5120_K17408" not in new
            or "reference_gemm_M9_N5120_K17408" in retained
            or set(frozen["reference_points"]) != set(new) | retained
            or phase.get("complete") is not True or phase.get("phase") != "reference"
            or phase["plan"] != pins["plan"] or phase["collector"] != plan["code"]["collector"]
            or [(r["repeat"], r["cell_id"]) for r in phase["records"]] != expected
            or frozen["reference_evidence"]["phase_result"] != pins["primitive_reference_phase"]
            or frozen["reference_evidence"]["raw_prices"] != [r["raw"] for r in phase["records"]]
            or frozen.get("heldout_timings_read") is not False
            or frozen.get("target_timings_used") is not False or frozen.get("source_qualified") is not False):
        raise ValueError("diagnostic root reference cohort or complete reference boundary differs")
    if (plan["proposal"] != pins["proposal"] or frozen["plan"] != pins["plan"]
            or any(proposal["reuse"][key] != pins[key] for key in ("prior_plan", "prior_freeze", "prior_verdict"))):
        raise ValueError("diagnostic root source lineage differs")
    points = frozen["reference_points"]
    by_cell = {}
    seeds = {(r, name): 314159 + r * 1000 + index
             for r, order in enumerate(plan["reference_order_by_repeat"], 1) for index, name in enumerate(order)}
    for row in phase["records"]:
        case = new[row["cell_id"]]
        raw, value = _raw_price(reader, case, row["raw"], "reference_raw")
        if (row.get("errors") or raw["provenance"].get("observed_group_width") != 1
                or row["seed"] != seeds[row["repeat"], row["cell_id"]]):
            raise ValueError("diagnostic reference source reports an error or wrong width")
        _child(reader, pins["plan"], case, row, raw, "reference")
        by_cell.setdefault(row["cell_id"], []).append(value)
        _graph(reader, case, "reference_graph")
    prior = evidence["prior_freeze"]
    prior_raw = {Path(p["path"]).name: p for p in prior["reference_evidence"]["raw_prices"]}
    if len(prior_raw) != len(prior["reference_evidence"]["raw_prices"]):
        raise ValueError("diagnostic retained reference paths are ambiguous")
    for name in retained:
        point = prior["reference_points"][name]
        if (point.get("source_qualified") is not True or points[name] != dict(point,
                origin_cohort="retained", origin_freeze=pins["prior_freeze"])):
            raise ValueError("diagnostic retained source value or qualification changed")
        case = prior_cases[name]
        by_cell[name] = [_raw_price(reader, case, prior_raw[f"{name}.r{r}.json"], "retained_reference_raw")[1]
                         for r in (1, 2, 3)]
        _graph(reader, case, "retained_reference_graph")
    for name, values in by_cell.items():
        p = points[name]
        numbers = [v["seconds"] for v in values]
        profiles = [sorted(v.get("kernels", {})) for v in values]
        if (len(numbers) != 3 or p["all_three"] != numbers or p["seconds"] != median(numbers)
                or p["range_over_median"] != (max(numbers) - min(numbers)) / median(numbers)
                or p["kernel_profiles"] != profiles
                or p["source_qualified"] is not (p["range_over_median"] <= .05)):
            raise ValueError("diagnostic export changed a frozen three-reference statistic")
        case = new.get(name, prior_cases.get(name))
        if case["family"] == "gemm" and (not profiles[0] or any(x != profiles[0] for x in profiles)):
            raise ValueError("diagnostic reference has unqualified GEMM identity")
    failures = sorted(n for n in new if not points[n]["source_qualified"])
    if failures != ["reference_mha_q1_C2992_l3", "reference_mha_q2_C496_l3"]:
        raise ValueError("diagnostic selection changed the declared R5 reference failures")
    expected_partial = [(r, name) for r, order in enumerate(plan["heldout_order_by_repeat"], 1) for name in order]
    if (partial.get("complete") is not False or partial.get("phase") != "heldout"
            or partial["plan"] != pins["plan"] or partial["collector"] != plan["code"]["collector"]
            or partial.get("target_timings_used") is not False or len(partial["records"]) != 171
            or [(r["repeat"], r["cell_id"]) for r in partial["records"]] != expected_partial[:171]):
        raise ValueError("diagnostic R5 validation must retain its actual incomplete prefix")
    cases = {c["cell_id"]: c for c in plan["cases"]}
    seeds = {(r, name): 271828 + r * 1000 + index
             for r, order in enumerate(plan["heldout_order_by_repeat"], 1) for index, name in enumerate(order)}
    for row in partial["records"]:
        if row.get("errors") or row["seed"] != seeds[row["repeat"], row["cell_id"]]:
            raise ValueError("diagnostic partial validation has an invalid source row or seed")
        case = cases[row["cell_id"]]
        raw, _ = _raw_price(reader, case, row["raw"], "heldout_raw")
        _child(reader, pins["plan"], case, row, raw, "heldout")
    closeout = evidence["primitive_closeout"]
    if (closeout["plan_sha256"] != pins["plan"]["sha256"] or closeout["status"] != "failed"
            or closeout["completed_records"] != {"smoke": 8, "reference": 117, "heldout": 171}
            or closeout["native_complete_present"] is not False or closeout["final_verdict_present"] is not False
            or closeout["reference_source_qualified"] is not False
            or closeout["artifact_sha256"]["NATIVE_FAILURE.json"] != pins["primitive_failure"]["sha256"]
            or closeout["artifact_sha256"]["CPU_FAILURE.json"] != pins["primitive_cpu_failure"]["sha256"]):
        raise ValueError("diagnostic R5 failure/completion status was altered")
    previous = None
    names = ("REFERENCE_CLOSED", "FREEZE_SEALED", "HELDOUT_RELEASED", "HELDOUT_STARTED")
    for index, name in enumerate(names, 1):
        role = f"primitive_event_{index}"
        event = evidence[role]
        if (event["event"] != name or event["index"] != index or event["previous"] != previous
                or event["plan"] != pins["plan"] or event["cohort"] != "primitives"):
            raise ValueError("diagnostic reference-freeze/heldout-start chain changed")
        previous = pins[role]
    if (evidence["primitive_event_1"]["payload"]["phase"] != pins["primitive_reference_phase"]
            or evidence["primitive_event_2"]["payload"]["freeze"] != pins["primitive_freeze"]
            or evidence["primitive_event_3"]["payload"]["freeze"] != pins["primitive_freeze"]
            or evidence["primitive_event_4"]["payload"]["release"] != pins["primitive_event_3"]):
        raise ValueError("diagnostic reference seal is not before heldout release")
    active = {**{name: prior_cases[name] for name in active_old}, **new}
    if set(handoff["selected_reference_cells"]) != set(active):
        raise ValueError("diagnostic reference selection differs from the original declared domain")
    return active, points, failures


def _join_region(observation, timers):
    """The archived region-contract join and unclamped subtraction."""
    ids = tuple(map(str, observation["req_ids"]))
    matches = [row for row in timers if tuple(map(str, row.get("req_ids", ()))) == ids
        and row.get("num_scheduled_tokens") == observation["query_lens"]
        and row.get("num_prefill_tokens") == sum(observation["query_lens"])
        and row.get("context_lens") == observation["context_lens"]
        and row.get("produces_output") is observation["produces_output"]]
    if len(matches) != 1 or matches[0].get("compiled") is not True:
        raise ValueError("diagnostic region timing join is not unique or compiled")
    row = matches[0]
    prepare = row["seconds"] - row["span_seconds"]["run_model"] - row["span_seconds"]["postprocess"]
    return prepare, row["span_seconds"]["postprocess"], timers.index(row)


def _region_fork(observation, case):
    sources, slots = observation["state_fork_srcs"], observation["state_slots"]
    if (len(observation["req_ids"]) != 1 or len(sources) != 1 or len(slots) != 1
            or (sources[0] >= 0) is not (case["cached_prefix"] > 0)
            or sources[0] >= 0 and sources[0] == slots[0]):
        raise ValueError("diagnostic region state-fork structure differs")


def _original_region_references(reader, candidate, evidence):
    frozen = evidence["region_freeze"]
    original_cases = evidence["region_plan"]["region_cases"]
    cases = candidate["cells"] + original_cases
    found = {c["cell_id"]: [] for c in cases}
    chains = candidate["reference_evidence_chains"]
    if [c["repeat"] for c in chains] != [1, 2, 3]:
        raise ValueError("diagnostic region reference chains lost a repeat")
    for chain in chains:
        repeat = chain["repeat"]
        if chain["original_reference_phase"] != frozen["reference_evidence"]["phase_result"]:
            raise ValueError("diagnostic region references escaped the original R3 freeze")
        phase = reader.read(chain["original_reference_phase"], "region_reference_phase")
        anchor = reader.read(chain["original_selected_row_anchor"], "region_reference_anchor")
        ledger = reader.read(chain["fixture_ledger"], "region_reference_ledger")
        final = reader.read(ledger["final"], "region_reference_final")
        reader.read(chain["fixture"], "region_reference_fixture")
        if (not any(row["raw"] == chain["original_selected_row_anchor"] and row["repeat"] == repeat
                    for row in phase["records"])
                or anchor["fixture_ledger"] != chain["fixture_ledger"]
                or ledger["role"] != "reference" or ledger["repeat"] != repeat
                or ledger["native_raw_steps"] != chain["native_timers"]
                or final["structure"] != chain["native_observer"] or ledger["fixture"] != chain["fixture"]):
            raise ValueError("diagnostic region references lost their original native evidence")
        timers = [json.loads(line) for line in reader.read(chain["native_timers"], "region_reference_timers", json_data=False).splitlines()]
        observations = reader.read(chain["native_observer"], "region_reference_observer")["steps"]
        for case in cases:
            matches = [r for r in observations if r["phase"] == f"regions_reference_r{repeat}_request{case['request']}"
                and r["prefill_requests"] == 1 and r["query_lens"] == [case["query"]]
                and r["cached_tokens"] == [case["cached_prefix"]] and r["context_lens"] == [case["total_history"]]
                and r["produces_output"] is case["produces_output"]]
            if len(matches) != 1:
                raise ValueError("diagnostic region reference observer join is not unique")
            _region_fork(matches[0], case)
            prepare, post, ordinal = _join_region(matches[0], timers)
            found[case["cell_id"]].append(dict(repeat=repeat, prepare_seconds=prepare, postprocess_seconds=post,
                timer_ordinal=ordinal, req_ids=matches[0]["req_ids"], observed_phase=matches[0]["phase"]))
    if {c["cell_id"]: found[c["cell_id"]] for c in candidate["cells"]} != candidate["reference_join_rows"]:
        raise ValueError("diagnostic candidate medians are not from the actual thirty R3 reference joins")
    for case in original_cases:
        for component in ("prepare", "postprocess"):
            observed = _summary([r[component + "_seconds"] for r in found[case["cell_id"]]],
                                component == "postprocess" and not case["produces_output"])
            if (observed != frozen["reference_points"][case["cell_id"]][component]
                    or observed["seconds"] != frozen["predictions"][case["cell_id"]][component]):
                raise ValueError("diagnostic original nine region values changed")


def _supplement(reader, handoff, evidence, root):
    pins = handoff["evidence"]
    candidate, plan, verdict, phase, complete = (evidence["supplement_" + name] for name in
        ("candidate", "plan", "verdict", "phase", "native_complete"))
    exported = evidence["supplement_handoff"]
    _original_region_references(reader, candidate, evidence)
    points = _candidate_points(candidate, root, original_evidence={
        "freeze": pins["region_freeze"], "verdict": pins["region_verdict"]})
    if (plan["candidate_freeze"] != pins["supplement_candidate"] or plan["cells"] != candidate["cells"]
            or phase["complete"] is not True or phase["plan"] != pins["supplement_plan"]
            or phase["prediction_freeze"] != pins["supplement_candidate"]
            or verdict["plan"] != pins["supplement_plan"]
            or verdict["prediction_freeze"] != pins["supplement_candidate"]
            or verdict["heldout_evidence"]["phase_result"] != pins["supplement_phase"]
            or complete["plan"] != pins["supplement_plan"] or complete["phase_result"] != pins["supplement_phase"]
            or complete["verdict"] != pins["supplement_verdict"]
            or any(value.get("source_qualified") is not False or value.get("all_error_gates_pass") is not False
                   for value in (verdict, exported))
            or any(value.get("source_requests") != 28 or value.get("primitive_requests") != 0
                   or value.get("target_requests") != 0 for value in (phase, complete))):
        raise ValueError("diagnostic supplement changed its complete but failed validation")
    for key, role in (("candidate", "candidate"), ("plan", "plan"), ("verdict", "verdict"),
                      ("native_complete", "native_complete"), ("heldout_phase", "phase")):
        if exported["evidence"][key] != pins["supplement_" + role]:
            raise ValueError("diagnostic supplement handoff evidence differs")
    cells = {c["cell_id"]: c for c in candidate["cells"]}
    expected = [(name, repeat, candidate["fresh_fixtures"]["heldout"][repeat-1])
                for repeat in (1, 2, 3) for name in cells]
    if [(r["cell_id"], r["repeat"], r["fixture"]) for r in phase["records"]] != expected:
        raise ValueError("diagnostic supplement lost a fresh fixture or repeat")
    observations = {name: {key: [] for key in ("prepare", "postprocess")} for name in cells}
    for row in phase["records"]:
        raw = reader.read(row["raw"], "region_heldout_raw")
        reader.read(raw["fixture"], "region_heldout_fixture")
        ledger = reader.read(raw["fixture_ledger"], "region_heldout_ledger")
        case = cells[row["cell_id"]]
        if (raw["plan"] != pins["supplement_plan"] or raw["prediction_freeze"] != pins["supplement_candidate"]
                or raw["fixture"] != row["fixture"] or raw["repeat"] != row["repeat"]
                or raw.get("phase") != "heldout" or raw.get("target_timings_used") is not False
                or any(raw[k] != case[k] for k in ("cell_id", "query", "cached_prefix", "total_history", "produces_output"))):
            raise ValueError("diagnostic supplement raw source identity differs")
        if (ledger["fixture"] != row["fixture"] or ledger["role"] != "heldout"
                or ledger["repeat"] != row["repeat"] or ledger["plan"] != pins["supplement_plan"]
                or ledger["prediction_freeze"] != pins["supplement_candidate"]
                or ledger["all19_prefill_paths_match"] is not True):
            raise ValueError("diagnostic supplement fixture ledger differs")
        timers = [json.loads(line) for line in reader.read(ledger["native_raw_steps"], "region_heldout_timers", json_data=False).splitlines()]
        final = reader.read(ledger["final"], "region_heldout_final")
        structure = reader.read(final["structure"], "region_heldout_observer")
        observation = raw["native_observation"]
        if (observation not in structure["steps"]
                or observation["phase"] != f"supplement_heldout_r{row['repeat']}_request{case['request']}"
                or observation["prefill_requests"] != 1 or observation["query_lens"] != [case["query"]]
                or observation["cached_tokens"] != [case["cached_prefix"]]
                or observation["context_lens"] != [case["total_history"]]
                or observation["produces_output"] is not case["produces_output"]):
            raise ValueError("diagnostic supplement observation differs from its immutable snapshot")
        _region_fork(observation, case)
        prepare, post, ordinal = _join_region(observation, timers)
        timing = timers[ordinal]
        expected_values = dict(prepare_seconds=prepare, postprocess_seconds=post,
            forward_seconds=timing["seconds"], run_model_seconds=timing["span_seconds"]["run_model"],
            raw_prepare_remainder=prepare, timing_row=timing,
            formula="forward minus run_model minus postprocess; no clamp or fit")
        if any(raw[key] != value for key, value in expected_values.items()):
            raise ValueError("diagnostic supplement values differ from the immutable native timing join")
        for component in observations[row["cell_id"]]:
            observations[row["cell_id"]][component].append(raw[component + "_seconds"])
    expected_checks = {(name, component) for name in cells for component in ("prepare", "postprocess")}
    if len(verdict["checks"]) != 20 or {(c["cell_id"], c["component"]) for c in verdict["checks"]} != expected_checks:
        raise ValueError("diagnostic supplement component coverage changed")
    failures = []
    for check in verdict["checks"]:
        name, component = check["cell_id"], check["component"]
        prediction = candidate["predictions"][name][component]
        observed = _summary(observations[name][component], component == "postprocess" and not cells[name]["produces_output"])
        error = observed["seconds"] - prediction
        passed = abs(error) <= .00011
        if (check["observed"] != observed or check["prediction"] != prediction or check["error_seconds"] != error
                or check["absolute_limit_seconds"] != .00011 or check["pass"] is not passed):
            raise ValueError("diagnostic supplement changed an observed failure or frozen prediction")
        if not passed:
            failures.append({"cell_id": name, "component": component, "error_seconds": error,
                             "absolute_limit_seconds": .00011})
    if {c["cell_id"] for c in failures} != {
            "region_C42000_Q2832_outputless", "region_C44832_Q1728_outputless", "region_C46496_Q10_final"}:
        raise ValueError("diagnostic supplement failure set differs")
    closeout, outer = evidence["supplement_closeout"], evidence["supplement_outer_exit"]
    if (closeout["source_qualified"] is not False or closeout["source_acquisition_completed"] is not True
            or closeout["verdict"] != pins["supplement_verdict"] or closeout["handoff"] != pins["supplement_handoff"]
            or closeout["outer_exit"]["sha256"] != pins["supplement_outer_exit"]["sha256"]
            or closeout["outer_exit_code"] != 124 or outer["exit_code"] != 124
            or outer["cleanup"]["verified"] is not True
            or closeout["isolated_performance_claim"] is not False):
        raise ValueError("diagnostic supplement hid its outer failure or background qualification")
    return points, failures


def _live_abi(evidence, pins):
    for role, plan_role in (("primitive_preflight", "plan"), ("prior_preflight", "prior_plan")):
        live, plan = evidence[role], evidence[plan_role]
        abi = live.get("family_abi") or {}
        layers, families = abi.get("all_layers") or {}, abi.get("families") or {}
        if (live.get("plan", {}).get("sha256") != pins[plan_role]["sha256"]
                or abi.get("flags") != plan["backend_flags"]
                or len(layers) != 64 or set(families) != {"gdn", "mha"}):
            raise ValueError("diagnostic source live layer-family ABI is incomplete")
        for name, value in layers.items():
            family = (live.get("native", {}).get("layers", {}).get(name) or {}).get("family")
            if family not in families or value != families[family]:
                raise ValueError("diagnostic source live layer-family ABI differs")


class DiagnosticRootPrefillPrices(RootPrefillPrices):
    """Same reference arithmetic; an explicit, permanently unqualified source."""

    def __init__(self, base, path, sha256, *, deployment_scope_sha256, workload_sha256,
                 diagnostic_only=False):
        PriceLibrary.__init__(self)
        if (diagnostic_only is not True or workload_sha256 != WORKLOAD_SHA256
                or deployment_scope_sha256 != REQUEST_SCOPE_SHA256):
            raise ValueError("root reference diagnostic requires explicit mode and the fixed ROOT_1493 workload")
        handoff, loaded = load_json(path, role=ROLE_PREFIX + "sources")
        scope = handoff.get("scope") or {}
        if (loaded.sha256 != sha256 or handoff.get("schema") != SCHEMA
                or handoff.get("diagnostic_only") is not True or handoff.get("source_qualified") is not False
                or handoff.get("acceptance_eligible") is not False
                or handoff.get("heldout_timings_used_as_fit_inputs") is not False
                or handoff.get("fit_inputs_are_references_only") is not True
                or scope != {"model": "Qwen/Qwen3.8-27B", "topology": {"tp": 1}, "dtype": "bfloat16",
                    "num_sequences": 1, "request_scope_sha256": deployment_scope_sha256,
                    "workload_sha256": workload_sha256}
                or not deployment_scope_sha256 or getattr(base, "launch_charge_seconds", 0) != 0
                or set(handoff.get("evidence", {})) != set(EVIDENCE)):
            raise ValueError("root diagnostic handoff, source flags or deployment scope differs")
        reader = DiagnosticEvidence(Path(path).parent)
        evidence = {role: reader.read(pin, role) for role, pin in handoff["evidence"].items()}
        _live_abi(evidence, handoff["evidence"])
        active, points, ref_failures = _references(reader, handoff, evidence)
        validate_region_binding(evidence["plan"], evidence, handoff["evidence"])
        region_verdict = evidence["region_verdict"]
        if (region_verdict["source_qualified"] is not True or len(region_verdict["checks"]) != 18
                or not all(c["pass"] is True for c in region_verdict["checks"])):
            raise ValueError("original nine regions lost their separate qualification")
        self._load_reference_tables(base, path, handoff, loaded, tuple(reader.inputs),
            active, points, evidence["plan"]["region_cases"], evidence["region_freeze"])
        self.supplement_points, region_failures = _supplement(reader, handoff, evidence, self)
        # Include later reads made while verifying complete failed region data.
        self.loaded_inputs = base.loaded_inputs + (loaded,) + tuple(reader.inputs) + self.source.loaded_inputs
        self.diagnostic_status = dict(source_qualified=False, acceptance_eligible=False,
            r5_reference_records=117, r5_validation_records=171, r5_validation_expected=348,
            r5_validation_complete=False, r5_native_complete_present=False, r5_final_verdict_present=False,
            r5_failed_references=ref_failures,
            r5_reference_spread_failures=[dict(cell_id=name,range_over_median=points[name]["range_over_median"],
                                              spread_limit=.05) for name in ref_failures],
            historical_reference_failures=[name for name,p in evidence["prior_freeze"]["reference_points"].items()
                                           if not p["source_qualified"]],
            historical_control_failures=[c["cell_id"] for c in evidence["prior_verdict"]["checks"]
                                         if not c["source_qualified"]],
            r4_reference_failures=[name for name,p in evidence["r4_freeze"]["reference_points"].items()
                                   if not p["source_qualified"]], supplement_validation_complete=True,
            supplement_gates_passed=17, supplement_gates_total=20, supplement_failed_checks=region_failures,
            supplement_outer_exit_code=124, isolated_performance_claim=False)
        if handoff["diagnostic_status"] != self.diagnostic_status:
            raise ValueError("diagnostic handoff omits or changes source failure status")
