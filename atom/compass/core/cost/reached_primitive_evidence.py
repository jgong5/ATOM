"""Read the existing reached manifest/freeze/verdict protocol without cohort counts."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from atom.compass.core.cost.cached_q16 import GATHER, GDN, MHA
from atom.compass.core.cost.library import INTERPOLATED_FLAG, _cost_key_of, _layout_fingerprint, _signature_of
from atom.compass.core.cost.low_query import GEMM
from atom.compass.core.cost.root_prefill import QK_NORM
from atom.compass.core.loaded_input import LoadedInput, load_json

SCHEMA = "compass.reached_primitive_reference_export/1"
FAMILIES = {"gemm": GEMM, "gdn": GDN, "mha": MHA, "gather": GATHER, "qk_norm": QK_NORM}
EVENTS = ("REFERENCE_CLOSED", "FREEZE_SEALED", "HELDOUT_RELEASED", "HELDOUT_STARTED", "HELDOUT_CLOSED", "VERDICT_WRITTEN")
EVIDENCE = ("domain_manifest", "plan", "manifest", "freeze", "verdict", "reference_plan",
            "reference_phase", "reference_preflight", "heldout_phase", "heldout_preflight",
            "dispatch", "execution_plan", "terminal", "owner_closeout", "copy_closeout", "original_failure")


class Evidence:
    def __init__(self, directory, index):
        self.directory, self.prefix = Path(directory), f"oracle.reached_primitives.{index}."
        self.inputs, self.values = [], {}

    def path(self, pin):
        return str(self.directory / pin["path"])

    def read(self, pin, role, *, json_data=True):
        path = self.path(pin)
        key = path, pin["sha256"], json_data
        if key in self.values:
            return self.values[key]
        if json_data:
            value, loaded = load_json(path, role=self.prefix + role)
        else:
            value = Path(path).read_bytes()
            loaded = LoadedInput(self.prefix + role, path, path, False, hashlib.sha256(value).hexdigest(), len(value))
        if loaded.sha256 != pin["sha256"]:
            raise ValueError("reached primitive evidence changed: " + role)
        if any(item.path == loaded.path and item.sha256 != loaded.sha256 for item in self.inputs):
            raise ValueError("reached primitive input changed between reads")
        self.inputs.append(loaded)
        self.values[key] = value
        return value


def same_pin(first, second):
    return isinstance(first, dict) and isinstance(second, dict) and first.get("sha256") == second.get("sha256")


def same_case(first, second):
    # Copies retain original graph bytes; their file location is not a treatment.
    return dict(first, graph=first["graph"]["sha256"]) == dict(second, graph=second["graph"]["sha256"])


def graph_for(reader, case, role):
    graph = reader.read(case["graph"], role)
    if (len(graph.get("ops", [])) != 1 or case["family"] not in FAMILIES
            or graph["ops"][0].get("name") != FAMILIES[case["family"]]
            or graph["ops"][0].get("group") is not None
            or _signature_of(graph["ops"][0]) != case["signature"]
            or case.get("portable_cost_key", _cost_key_of(graph["ops"][0])) != _cost_key_of(graph["ops"][0])):
        raise ValueError("reached primitive graph, family or signature differs")
    return graph["ops"][0]


def _point(values, case, profiles=None):
    if len(values) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("reached source requires three finite positive observations")
    value = median(values)
    spread = (max(values) - min(values)) / value
    result = dict(seconds=value, all_three=values, range_over_median=spread,
                  source_qualified=spread <= .05, signature=case["signature"])
    if profiles is not None:
        result["kernel_profiles"] = profiles
    return result


def _raw(reader, case, row, role, *, retained=False):
    raw = reader.read(row["raw"], role)
    settings = row.get("settings") or {}
    provenance = raw.get("provenance") or {}
    if (row.get("errors") or raw.get("unpriced") or set(raw.get("prices", {})) != {case["signature"]}
            or raw.get("source_consumption_forbidden") is True or raw.get("diagnostic_only") is True
            or provenance.get("topology") != {"tp": 1} or provenance.get("observed_group_width") != 1
            or provenance.get("cache") != "graph" or provenance.get("iters") != case["iters"]
            or provenance.get("graph") != case["graph"]["path"]
            or provenance.get("only") != case["only"] or settings.get("GRAPH_BATCH") != case["graph_batch"]
            or settings.get("KV_VARIANTS") != 8 or settings.get("REPLAY_INT_VALUES") is not False
            or settings.get("SYNTH_INT_RANGES") is not True):
        raise ValueError("reached primitive raw observation changes its timing/ABI treatment")
    price = raw["prices"][case["signature"]]
    if any(price.get(field) != case[source] for field, source in
           (("cache", "observed_cache"), ("arg_sets", "arg_sets"), ("kv_regions", "kv_regions"))):
        raise ValueError("reached primitive raw working-set treatment differs")
    if not retained:
        treatment = {field: case[field] for field in
            ("graph_batch", "warmup", "iters", "requested_cache", "observed_cache", "arg_sets", "kv_regions")}
        if (row.get("profiled") is not False or settings.get("PRICE_KERNELS") is not False
                or settings.get("PROFILE_MATCH") != "" or price.get("kernels")
                or row.get("treatment") != treatment):
            raise ValueError("reached primitive timing cannot use profiled or undeclared measurements")
    return raw, price


def _dispatch(reader, pin, plan, plan_pin, role):
    value = reader.read(pin, role)
    if (not same_pin(value.get("plan"), plan_pin) or value.get("complete") is not True
            or value.get("dispatch_qualified") is not True or value.get("timings_used_as_prices") is not False
            or set(value.get("cells", {})) != set(plan["dispatch"]["case_to_probe"])):
        raise ValueError("reached primitive dispatch is incomplete or belongs to another plan")
    cases = {case["cell_id"]: case for case in plan["cases"]}
    cases.update({name: item["case"] for name, item in plan.get("retained_references", {}).items()})
    for name, row in value["cells"].items():
        profile = row.get("kernel_profile")
        if (name not in cases or not same_pin(row.get("graph"), cases[name]["graph"])
                or not isinstance(profile, list) or not profile
                or any(not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                       or type(item[1]) is not int or item[1] <= 0 for item in profile)):
            raise ValueError("reached primitive dispatch graph or kernel identity differs")
    for index, reference in enumerate(value["inputs"]):
        reader.read(reference, f"{role}.input{index}", json_data=False)
    return value


def _seed(reader, plan, phase, repeat, ordinal, name):
    parent = None
    if plan.get("source_seed_ordinals") is not None:
        ordinal = plan["source_seed_ordinals"][phase][repeat - 1][name]
        parent = plan["parent_source_campaign"]
    elif phase == "heldout" and plan.get("continuation"):
        ordinal = plan["continuation"]["seed_ordinals"][str(repeat)][name]
        parent = plan["continuation"]["original_plan"]
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("reached primitive seed ordinal is invalid")
    if parent is not None:
        source = reader.read(parent, phase + ".seed_plan")
        if source[phase + "_order_by_repeat"][repeat - 1][ordinal] != name:
            raise ValueError("reached primitive subset changed an original seed")
    return (314159 if phase == "reference" else 271828) + repeat * 1000 + ordinal


def _phase(reader, phase, plan, plan_pin, phase_pin, preflight_pin):
    value = reader.read(phase_pin, phase + ".phase")
    preflight = reader.read(preflight_pin, phase + ".preflight")
    if (value.get("complete") is not True or value.get("phase") != phase
            or not same_pin(value.get("plan"), plan_pin)
            or value.get("collector") != plan["code"]["collector"]
            or value.get("target_timings_used") is not False or value.get("profiler_tool_loaded") is not False
            or value.get("source_only") is not True or value.get("candidate_activated") is not False
            or not same_pin(preflight.get("plan"), plan_pin) or preflight.get("phase") != phase
            or preflight.get("collector") != plan["code"]["collector"]
            or preflight.get("profiler_tool_loaded") is not False
            or not same_pin(preflight.get("dispatch"), value["dispatch"])):
        raise ValueError("reached primitive phase/preflight identity or completeness differs")
    reader.read(plan["code"]["collector"], phase + ".collector", json_data=False)
    dispatch = _dispatch(reader, value["dispatch"], plan, plan_pin, phase + ".dispatch")
    runtime = preflight.get("runtime_identity") or {}
    if (not isinstance(runtime.get("physical_uuid"), str) or not runtime["physical_uuid"]
            or runtime != dispatch.get("runtime_identity") or runtime.get("backend_flags") != plan["backend_flags"]):
        raise ValueError("reached primitive timing and dispatch runtime/physical UUID differ")
    boundary = preflight.get("boundary") or {}
    if boundary.get("policy") != plan["cache_policy"] or (boundary.get("quiescence") or {}).get("idle") is not True:
        raise ValueError("reached primitive phase lacks its native idle cache boundary")
    abi = preflight.get("family_abi") or {}
    families, layers = abi.get("families") or {}, abi.get("all_layers") or {}
    if (abi.get("flags") != plan["backend_flags"] or set(families) != {"gdn", "mha"}
            or set(layers) != {str(index) for index in range(64)}):
        raise ValueError("reached primitive homogeneous layer-family ABI is incomplete")
    for label, observed in layers.items():
        family = (preflight.get("native", {}).get("layers", {}).get(label) or {}).get("family")
        expected_family = "mha" if int(label) % 4 == 3 else "gdn"
        if family != expected_family or observed != families[family]:
            raise ValueError("reached primitive source layer ABI is not homogeneous")
    cases = {case["cell_id"]: case for case in plan["cases"] if case["phase"] == phase}
    orders = plan[phase + "_order_by_repeat"]
    if len(orders) != 3 or any(len(order) != len(cases) or set(order) != set(cases) for order in orders):
        raise ValueError("reached primitive phase does not contain three complete repetitions")
    expected = [(repeat, name) for repeat, order in enumerate(orders, 1) for name in order]
    if [(row["repeat"], row["cell_id"]) for row in value["records"]] != expected:
        raise ValueError("reached primitive phase records are missing, duplicated or reordered")
    samples, raw_refs = defaultdict(list), defaultdict(list)
    for index, row in enumerate(value["records"]):
        case = cases[row["cell_id"]]
        repeat, ordinal = index // len(cases) + 1, index % len(cases)
        if row.get("seed") != _seed(reader, plan, phase, repeat, ordinal, case["cell_id"]):
            raise ValueError("reached primitive timing seed differs")
        _, price = _raw(reader, case, row, phase + ".raw." + case["cell_id"] + "." + str(repeat))
        samples[case["cell_id"]].append(price["seconds"])
        raw_refs[case["cell_id"]].append(row["raw"])
    points = {name: _point(values, cases[name],
        [dispatch["cells"][name]["kernel_profile"]] * 3 if cases[name]["family"] == "gemm" else [[], [], []])
        for name, values in samples.items()}
    return points, raw_refs, dispatch, runtime


def _retained(reader, plan, dispatch, dispatch_pin):
    selections = plan.get("retained_references", {})
    if not selections:
        return {}, {}, {}
    audit = reader.read(plan["retained_treatment_audit"], "retained.audit")
    entries = {(entry["selected_freeze"]["sha256"], entry["name"]): entry for entry in audit["source_points"]}
    points, cases, samples = {}, {}, {}
    for name, selected in selections.items():
        entry = entries[(selected["freeze"]["sha256"], selected["original_name"])]
        prior = reader.read(entry["plan"], "retained.plan")
        frozen = reader.read(entry["original_freeze"], "retained.freeze")
        phase = reader.read(entry["phase"], "retained.phase")
        old_case = next(case for case in prior["cases"] if case["cell_id"] == selected["original_name"])
        case = selected["case"]
        if (entry.get("source_reference_qualified") is not True or selected["freeze"] != entry["selected_freeze"]
                or not same_case(case, dict(old_case, requested_cache="graph"))
                or phase.get("complete") is not True or phase.get("phase") != "reference"
                or not same_pin(phase.get("plan"), entry["plan"]) or phase["collector"] != entry["collector"]):
            raise ValueError("retained reached source changed its qualified origin or treatment")
        reader.read(entry["collector"], "retained.collector", json_data=False)
        graph_for(reader, case, "retained.graph")
        rows = [row for row in phase["records"] if row["cell_id"] == selected["original_name"]]
        if [row["repeat"] for row in rows] != [1, 2, 3] or [row["raw"] for row in rows] != entry["samples"]:
            raise ValueError("retained reached source sample inventory differs")
        values = [_raw(reader, case, row, "retained.raw", retained=True)[1]["seconds"] for row in rows]
        point = dict(frozen["reference_points"][selected["original_name"]])
        measured = _point(values, case)
        if any(point.get(key) != value for key, value in measured.items()) or not measured["source_qualified"]:
            raise ValueError("retained reached source changes its original observations or spread gate")
        point["origin"] = dict(original_freeze=entry["original_freeze"], selected_freeze=selected["freeze"],
            source_case_id=selected["original_name"], treatment_audit=plan["retained_treatment_audit"])
        if case["family"] == "gemm":
            current = dispatch["cells"][name]["kernel_profile"]
            names = sorted(item[0] for item in current)
            if any(sorted(profile) != names for profile in point["kernel_profiles"]):
                raise ValueError("retained GEMM dispatch no longer matches the original kernel identity")
            point["historical_kernel_profiles"] = point["kernel_profiles"]
            point["kernel_profiles"] = [current] * 3
            point["current_dispatch_validation"] = dispatch_pin
        points[name], cases[name], samples[name] = point, case, entry["samples"]
    return points, cases, samples


def _boundaries(reader, handoff, evidence):
    pins = handoff["evidence"]
    events = handoff.get("boundary_events")
    if not isinstance(events, list) or len(events) != 6:
        raise ValueError("reached primitive export lacks all six source/heldout boundary events")
    previous, values = None, []
    for index, (name, pin) in enumerate(zip(EVENTS, events), 1):
        row = reader.read(pin, f"event{index}")
        if (row.get("schema") != "compass.low_q_boundary_event/1" or row.get("event") != name
                or row.get("index") != index or not same_pin(row.get("plan"), pins["plan"])
                or (row.get("previous") is not None if previous is None else not same_pin(row.get("previous"), previous))):
            raise ValueError("reached primitive boundary chain changed")
        values.append(row["payload"])
        previous = pin
    if (not same_pin(values[0].get("phase"), pins["reference_phase"])
            or not same_pin(values[1].get("reference_phase"), pins["reference_phase"])
            or not same_pin(values[1].get("freeze"), pins["freeze"])
            or not same_pin(values[2].get("freeze"), pins["freeze"])
            or values[2].get("heldout_order_sha256") != evidence["plan"]["heldout_order_sha256"]
            or not same_pin(values[3].get("release"), events[2])
            or not same_pin(values[4].get("phase"), pins["heldout_phase"])
            or not same_pin(values[5].get("verdict"), pins["verdict"])):
        raise ValueError("reached primitive freeze/release/heldout evidence is disconnected")
    if not same_pin(pins["reference_plan"], pins["plan"]):
        if (values[0].get("reused") is not True
                or not same_pin(values[0].get("original_plan"), pins["reference_plan"])
                or not same_pin(evidence["freeze"].get("original_plan"), pins["reference_plan"])
                or evidence["freeze"].get("whole_campaign_source_qualified") is not False):
            raise ValueError("reached subset relabels its original reference phase or whole campaign")


def load_campaign(reference, deployment_scope_sha256, *, index):
    from atom.compass.core.cost.reached_primitives import INVALID_ALIAS, work_identity

    handoff_path = Path(reference["path"]).resolve()
    reader = Evidence(handoff_path.parent, index)
    handoff = reader.read(dict(reference, path=str(handoff_path)), "handoff")
    scope = handoff.get("scope") or {}
    if (handoff.get("schema") != SCHEMA or handoff.get("source_qualified") is not True
            or handoff.get("fit_inputs_are_references_only") is not True
            or handoff.get("heldout_timings_used_as_fit_inputs") is not False
            or handoff.get("candidate_activated") is not False
            or scope.get("model") != "Qwen/Qwen3.8-27B" or scope.get("topology") != {"tp": 1}
            or scope.get("dtype") != "bfloat16" or not deployment_scope_sha256
            or scope.get("request_scope_sha256") != deployment_scope_sha256
            or set(handoff.get("evidence", {})) != set(EVIDENCE)):
        raise ValueError("reached primitive handoff is unqualified or changes its deployment scope")
    pins = handoff["evidence"]
    data = {name: reader.read(pin, name) for name, pin in pins.items()}
    plan, manifest, frozen, verdict = (data[name] for name in ("plan", "manifest", "freeze", "verdict"))
    if (plan.get("schema") != "compass.reached_primitive_executable/2"
            or manifest.get("schema") != "compass.reached_primitive_manifest/1"
            or not same_pin(plan.get("design"), pins["manifest"]) or plan["cases"] != manifest["cases"]
            or frozen.get("schema") != "compass.low_q_reference_freeze/1"
            or verdict.get("schema") != "compass.low_q_source_verdict/1"
            or not same_pin(frozen.get("plan"), pins["plan"]) or not same_pin(verdict.get("plan"), pins["plan"])
            or not same_pin(verdict.get("prediction_freeze"), pins["freeze"])
            or frozen.get("source_qualified") is not True or frozen.get("heldout_timings_read") is not False
            or frozen.get("target_timings_used") is not False or verdict.get("target_timings_used") is not False
            or frozen.get("candidate_activated") is not False or verdict.get("candidate_activated") is not False):
        raise ValueError("reached primitive plan, source freeze or heldout verdict differs")
    from atom.compass.core.cache_policy import cache_on_policy, policy_errors

    for source in (plan, data["reference_plan"]):
        args = source["engine_args"]
        names = [case["cell_id"] for case in source["cases"]]
        if (args.get("model") != scope["model"] or args.get("tensor_parallel_size") != 1
                or args.get("pipeline_parallel_size", 1) != 1 or args.get("kv_cache_dtype") != "bf16"
                or policy_errors(source["cache_policy"], cache_on_policy())
                or len(names) != len(set(names)) or any(case["repetitions"] != 3 for case in source["cases"])):
            raise ValueError("reached primitive source model, cache policy or repetition identity differs")
    if (data["original_failure"].get("source_qualified") is not False
            or not same_pin(data["execution_plan"].get("science_plan"), pins["plan"])
            or data["terminal"].get("plan_sha256") != pins["execution_plan"]["sha256"]
            or type(data["terminal"].get("exit_code")) is not int
            or data["terminal"].get("cleanup") != data["owner_closeout"]
            or data["terminal"].get("collection") != data["copy_closeout"]
            or data["owner_closeout"].get("writers_released", data["owner_closeout"].get("verified")) is not True
            or data["copy_closeout"].get("copy_complete") is not True
            or data["copy_closeout"].get("owned_writers_released_before_collection") is not True):
        raise ValueError("reached primitive original failure or owned copy closeout is missing")
    for phase in ("reference", "heldout"):
        if plan[phase + "_order_by_repeat"] != manifest[phase + "_order_by_repeat"]:
            raise ValueError("reached primitive manifest changes phase order")
    order_sha = hashlib.sha256(json.dumps(plan["heldout_order_by_repeat"], sort_keys=True).encode()).hexdigest()
    if order_sha != plan["heldout_order_sha256"]:
        raise ValueError("reached primitive heldout order digest differs")
    _boundaries(reader, handoff, data)
    new, samples, reference_dispatch, reference_runtime = _phase(reader, "reference", data["reference_plan"],
        pins["reference_plan"], pins["reference_phase"], pins["reference_preflight"])
    retained, retained_cases, retained_samples = _retained(reader, data["reference_plan"],
        reference_dispatch, data["reference_phase"]["dispatch"])
    heldout, _, current_dispatch, runtime = _phase(reader, "heldout", plan, pins["plan"],
        pins["heldout_phase"], pins["heldout_preflight"])
    if runtime != reference_runtime or not same_pin(data["heldout_phase"]["dispatch"], pins["dispatch"]):
        raise ValueError("reached primitive reference/heldout campaigns change physical UUID or runtime")
    if data["dispatch"] != current_dispatch:
        raise ValueError("reached primitive active dispatch evidence differs")
    references = {case["cell_id"]: case for case in plan["cases"] if case["phase"] == "reference"}
    old_references = {case["cell_id"]: case for case in data["reference_plan"]["cases"] if case["phase"] == "reference"}
    if any(name not in old_references or not same_case(case, old_references[name]) for name, case in references.items()):
        raise ValueError("reached primitive reuse changes an original reference")
    if plan.get("retained_references", {}) != data["reference_plan"].get("retained_references", {}):
        raise ValueError("reached primitive retained source selections changed")
    points = {name: new[name] for name in references}
    points.update(retained)
    references.update(retained_cases)
    samples.update(retained_samples)
    expected_raw = [row["raw"] for row in data["reference_phase"]["records"] if row["cell_id"] in references]
    if (not same_pin(frozen["reference_evidence"].get("phase_result"), pins["reference_phase"])
            or frozen["reference_evidence"].get("raw_prices") != expected_raw
            or not same_pin(frozen["reference_evidence"].get("dispatch"), data["reference_phase"]["dispatch"])
            or frozen.get("reuse", {}) != plan.get("retained_references", {})
            or not same_pin(verdict["heldout_evidence"].get("phase_result"), pins["heldout_phase"])
            or verdict["heldout_evidence"].get("raw_prices") != [row["raw"] for row in data["heldout_phase"]["records"]]
            or not same_pin(verdict["heldout_evidence"].get("dispatch"), pins["dispatch"])):
        raise ValueError("reached primitive freeze/verdict changes its recorded source evidence")
    if set(frozen["reference_points"]) != set(points) or any(not point["source_qualified"] for point in points.values()):
        raise ValueError("reached primitive freeze adds failed or unknown reference points")
    for name, point in points.items():
        if any(frozen["reference_points"][name].get(key) != value for key, value in point.items()):
            raise ValueError("reached primitive freeze changes source observations or retained provenance")
    controls = {case["cell_id"]: case for case in plan["cases"] if case["phase"] == "heldout"}
    if set(frozen["predictions"]) != set(controls) or set(heldout) != set(controls):
        raise ValueError("reached primitive frozen/observed heldout membership differs")
    checks = {check["cell_id"]: check for check in verdict["checks"]}
    if len(checks) != len(verdict["checks"]) or set(checks) != set(controls):
        raise ValueError("reached primitive verdict omits or duplicates controls")
    groups = defaultdict(list)
    for name, case in controls.items():
        prediction = frozen["predictions"][name]
        weights = case["frozen_prediction_sources"]
        if (not weights or any(item["reference_cell_id"] not in points for item in weights)
                or any(type(item["weight"]) not in (int, float) or not math.isfinite(item["weight"])
                       or not 0 <= item["weight"] <= 1 for item in weights)
                or not math.isclose(sum(item["weight"] for item in weights), 1, abs_tol=1e-12)):
            raise ValueError("reached primitive control has invalid frozen dependencies")
        seconds = sum(item["weight"] * points[item["reference_cell_id"]]["seconds"] for item in weights)
        limit = .121 if case["family"] == "gather" else .10
        profiles = points[weights[0]["reference_cell_id"]]["kernel_profiles"] if case["family"] == "gemm" else []
        if (prediction.get("signature") != case["signature"] or prediction.get("family") != case["family"]
                or prediction.get("sources") != weights or prediction.get("seconds") != seconds
                or prediction.get("source_qualified") is not True or prediction.get("group") != case["score_group"]
                or prediction.get("limit") != limit or prediction.get("kernel_profiles") != profiles):
            raise ValueError("reached primitive prediction refits heldouts or changes frozen weights")
        observed, check = heldout[name], checks[name]
        error = abs(observed["seconds"] - seconds) / seconds
        kernel_pass = case["family"] != "gemm" or observed["kernel_profiles"] == profiles
        if (check.get("signature") != case["signature"] or check.get("prediction") != seconds
                or any(check["observed"].get(key) != value for key, value in observed.items())
                or check.get("relative_error") != error or check.get("limit") != limit
                or check.get("error_gate_pass") is not (error <= limit)
                or check.get("kernel_identity_pass") is not kernel_pass
                or check.get("source_qualified") is not observed["source_qualified"]
                or check.get("family") != case["family"] or check.get("group") != case["score_group"]):
            raise ValueError("reached primitive heldout gates differ from actual raw measurements")
        groups[case["score_group"]].append(name)
    totals = {row["group"]: row for row in verdict["groups"]}
    if len(totals) != len(verdict["groups"]) or set(totals) != set(groups):
        raise ValueError("reached primitive group aggregate inventory differs")
    qualified = set()
    for group, names in groups.items():
        predicted = sum(frozen["predictions"][name]["seconds"] for name in names)
        observed = sum(heldout[name]["seconds"] for name in names)
        error = abs(observed - predicted) / predicted
        passed = error <= frozen["predictions"][names[0]]["limit"]
        total = totals[group]
        if (total.get("predicted_seconds") != predicted or total.get("observed_seconds") != observed
                or total.get("relative_error") != error or total.get("pass") is not passed):
            raise ValueError("reached primitive group error gate changed")
        if passed and all(checks[name]["error_gate_pass"] and checks[name]["kernel_identity_pass"]
                          and checks[name]["source_qualified"] for name in names):
            qualified.add(group)
    selected = handoff.get("selected_groups")
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or not set(selected) <= qualified):
        raise ValueError("reached primitive export selects a failed or incomplete heldout group")
    domain_manifest = data["domain_manifest"]
    if domain_manifest.get("schema") != "compass.reached_primitive_manifest/1":
        raise ValueError("reached primitive domain manifest schema differs")
    domain_controls = {case["cell_id"]: case for case in domain_manifest["cases"] if case["phase"] == "heldout"}
    domain_groups, domain = defaultdict(set), {}
    for name, case in domain_controls.items():
        op = graph_for(reader, case, "domain.graph." + name)
        key, _ = work_identity(op)
        if key is None or key[1] == INVALID_ALIAS:
            raise ValueError("reached primitive domain has an unsupported work identity")
        group = case["score_group"]
        if key in domain and domain[key] != group:
            raise ValueError("reached primitive domain collapses different groups")
        domain[key] = group
        domain_groups[group].add(name)
    if any(name not in domain_controls or not same_case(case, domain_controls[name]) for name, case in controls.items()):
        raise ValueError("reached primitive campaign changes original domain controls")
    if any(set(names) != domain_groups[group] for group, names in groups.items()):
        raise ValueError("reached primitive campaign contains only part of a score group")
    active_controls = {name for group in selected for name in groups[group]}
    needed = {item["reference_cell_id"] for name in active_controls for item in controls[name]["frozen_prediction_sources"]}
    entries = {entry["reference_cell_id"]: entry for entry in handoff["entries"]}
    if len(entries) != len(handoff["entries"]) or set(entries) != needed:
        raise ValueError("reached primitive entries include unused/unqualified references or omit dependencies")
    source_records, sources = {}, []
    for name, entry in entries.items():
        case, point = references[name], points[name]
        if entry["price"]["sha256"] not in {pin["sha256"] for pin in samples[name]} or not same_pin(entry["graph"], case["graph"]):
            raise ValueError("reached primitive export is not an original reference price/graph")
        raw = reader.read(entry["price"], "entry.price." + name)
        op = graph_for(reader, dict(case, graph=entry["graph"]), "entry.graph." + name)
        price = raw["prices"][case["signature"]]
        if price["seconds"] != point["seconds"]:
            raise ValueError("reached primitive exported repetition is not its frozen median")
        source_key, layer = work_identity(op)
        if source_key is None or source_key[1] == INVALID_ALIAS:
            raise ValueError("reached primitive reference has an unsupported layer or state-alias identity")
        source = reader.path(entry["price"])
        source_records[name] = dict(price, source=source, scope={"topology": {"tp": 1}, "registration": None},
            signature=case["signature"], layout=_layout_fingerprint(op), source_layer=layer,
            reference_cell_id=name, source_price=entry["price"], source_graph=entry["graph"])
        sources.append(source)
    records = {}
    for name in sorted(active_controls):
        case, prediction = controls[name], frozen["predictions"][name]
        op = graph_for(reader, case, "selected.graph." + name)
        key, _ = work_identity(op)
        first = source_records[prediction["sources"][0]["reference_cell_id"]]
        source_op = graph_for(reader, references[first["reference_cell_id"]], "selected.reference_graph")
        source_key, _ = work_identity(source_op)
        record = dict(first, seconds=prediction["seconds"], source_qualified=True,
            source_handoff_sha256=reference["sha256"], score_group=case["score_group"],
            target_signature=case["signature"], reference_prediction=prediction["sources"],
            validation_campaign_physical_uuid=runtime["physical_uuid"],
            retained_reference_origins={item["reference_cell_id"]: retained[item["reference_cell_id"]]["origin"]
                for item in prediction["sources"] if item["reference_cell_id"] in retained})
        if len(prediction["sources"]) != 1 or key != source_key:
            record[INTERPOLATED_FLAG] = True
        if key in records and any(records[key][field] != record[field] for field in
                                  ("seconds", "score_group", "reference_prediction")):
            raise ValueError("reached group controls disagree on one frozen work prediction")
        records.setdefault(key, record)
    return dict(domain_sha256=pins["domain_manifest"]["sha256"], domain=domain, records=records,
        selected_groups=selected, loaded_inputs=tuple(reader.inputs), sources=sources,
        provenance=dict(handoff=reference, plan=pins["plan"], reference_plan=pins["reference_plan"],
            physical_uuid=runtime["physical_uuid"], selected_groups=selected,
            original_failure=pins["original_failure"], terminal=pins["terminal"],
            terminal_exit_code=data["terminal"].get("exit_code"), whole_campaign_requalified=False,
            retained_origins={name: point.get("origin") for name, point in retained.items()}))
