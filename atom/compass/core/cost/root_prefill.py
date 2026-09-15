"""Fixed prefill source cohort, with explicit history and bounded ABI transfer."""

from dataclasses import dataclass
import math
from pathlib import Path

from atom.compass.core.cost.cached_q16 import GDN, MHA, GATHER
from atom.compass.core.cost.library import INTERPOLATED_FLAG, PriceLibrary, _signature_of
from atom.compass.core.cost.low_query import (
    GEMM, GDN_LAYERS, MHA_LAYERS, PREFIXES, _gdn_identity, _key, _layer, _mha_identity,
)
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.loaded_input import load_json

QK_NORM = "triton::_fused_qk_norm_single_kernel"
REPLACED_REFERENCE = "reference_gemm_M9_N5120_K17408"


def _cold_gdn_identity(op):
    context = dict(op.get("context") or ())
    counts = {"num_prefills": 1, "num_prefill_tokens": 32, "num_decodes": 0,
              "num_decode_tokens": 0, "num_spec_decodes": 0,
              "num_spec_decode_tokens": 0, "num_actual_tokens": 32}
    if (any(type(context.get(k)) is not int or context[k] != v for k, v in counts.items())
            or context.get("replayssm") is not False
            or context.get("has_initial_state") != [[0], "bool"]
            or context.get("non_spec_query_start_loc") != [[0, 32], "int32"]):
        return None
    slots = []
    for name in ("non_spec_state_indices_in_tensor", "non_spec_state_indices_tensor"):
        value = context.get(name)
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or value[1] != "int32"
                or not isinstance(value[0], (list, tuple)) or len(value[0]) != 1
                or type(value[0][0]) is not int or not 0 <= value[0][0] < 32):
            return None
        slots.append(value[0][0])
    if slots[0] != slots[1]:
        return None  # Check alias contents before the generic key erases addresses.
    layer = _layer(op, GDN_LAYERS, 0)
    return None if layer is None else (layer[0], _key(layer[1]))


def _evidence(handoff, path):
    values, loaded = {}, []
    for role, pin in handoff["evidence"].items():
        value, identity = load_json(str(Path(path).parent / Path(pin["path"]).name),
                                    role="oracle.root_prefill_" + role)
        if identity.sha256 != pin["sha256"]:
            raise ValueError(f"root prefill {role} differs from its evidence pin")
        values[role] = value
        loaded.append(identity)
    return values, tuple(loaded)


def _validate(handoff, evidence, *, allow_failed_spread, diagnostic_only):
    plan, proposal = evidence["plan"], evidence["proposal"]
    frozen, verdict = evidence["primitive_freeze"], evidence["primitive_verdict"]
    region_freeze, region_verdict = evidence["region_freeze"], evidence["region_verdict"]
    prior_plan, prior_freeze, prior_verdict = (evidence[k] for k in
                                             ("prior_plan", "prior_freeze", "prior_verdict"))
    final = evidence["final_verdict"]
    pins = handoff["evidence"]
    if (plan.get("schema") != "compass.root1493_executable/1"
            or plan["proposal"]["sha256"] != pins["proposal"]["sha256"]
            or frozen.get("schema") != "compass.root1493_primitive_freeze/1"
            or region_freeze.get("schema") != "compass.root1493_region_freeze/1"
            or any(value.get("plan", {}).get("sha256") != pins["plan"]["sha256"]
                   for value in (frozen, verdict, region_freeze, region_verdict, final))
            or verdict["prediction_freeze"]["sha256"] != pins["primitive_freeze"]["sha256"]
            or region_verdict["prediction_freeze"]["sha256"] != pins["region_freeze"]["sha256"]
            or any(value.get("heldout_timings_read") is not False for value in (frozen, region_freeze))
            or any(value.get("target_timings_used") is not False for value in
                   (frozen, verdict, region_freeze, region_verdict, final))
            or any(value.get("candidate_activated") is not False for value in
                   (frozen, verdict, region_freeze, region_verdict, final))
            or any(final["cohorts"][cohort]["sha256"] != pins[role]["sha256"] for cohort, role in
                   (("primitives", "primitive_verdict"), ("regions", "region_verdict")))):
        raise ValueError("root prefill freeze, verdict or reference-only boundary differs")
    reuse = proposal["reuse"]
    if any(reuse[key]["sha256"] != pins[key]["sha256"] for key in
           ("prior_plan", "prior_freeze", "prior_verdict")):
        raise ValueError("root prefill historical source pins differ")
    for role, source_plan, source_pin in (("primitive_preflight", plan, pins["plan"]),
                                          ("prior_preflight", prior_plan, pins["prior_plan"])):
        live = evidence[role]
        abi = live.get("family_abi") or {}
        layers = abi.get("all_layers") or {}
        families = abi.get("families") or {}
        if (live.get("plan", {}).get("sha256") != source_pin["sha256"]
                or abi.get("flags") != source_plan["backend_flags"]
                or len(layers) != 64 or set(families) != {"gdn", "mha"}):
            raise ValueError("root prefill live source layer-family ABI is incomplete")
        for name, value in layers.items():
            family = (live.get("native", {}).get("layers", {}).get(name) or {}).get("family")
            if family not in families or value != families[family]:
                raise ValueError("root prefill live source layer-family ABI differs")
    new = {c["cell_id"]: c for c in plan["cases"] if c["phase"] == "reference"}
    controls = {c["cell_id"]: c for c in plan["cases"] if c["phase"] == "heldout"}
    old = {c["cell_id"]: c for c in prior_plan["cases"] if c["phase"] == "reference"}
    retained = set(plan["retained_sources"]["reference_cells"])
    active_old = {c["cell_id"] for c in reuse["active_reference_points"]}
    points = frozen["reference_points"]
    if (len(new) != 39 or len(controls) != 116 or len(retained) != 23 or len(active_old) != 21
            or set(new) != {c["cell_id"] for c in proposal["primitive_references"]}
            or set(controls) != {c["cell_id"] for c in proposal["primitive_heldouts"]}
            or plan["region_cases"] != proposal["regions"]["cells"]
            or retained != active_old | set(reuse["dependency_only_reference_cells"])
            or set(new) & retained or set(points) != set(new) | retained
            or REPLACED_REFERENCE not in new or REPLACED_REFERENCE in retained
            or set(frozen["predictions"]) != set(controls)
            or set(handoff["selected_reference_cells"]) != set(new) | active_old):
        raise ValueError("root prefill fixed new/retained cohort or M9 replacement differs")
    for name in retained:
        expected = dict(prior_freeze["reference_points"][name], origin_cohort="retained",
                        origin_freeze=reuse["prior_freeze"])
        if points[name] != expected:
            raise ValueError("root prefill retained reference value or qualification changed")
    if any(points[name].get("origin_cohort") != "new" for name in new):
        raise ValueError("root prefill new reference origin changed")
    old_checks = {c["cell_id"]: c for c in prior_verdict["checks"]}
    retained_controls = reuse["retained_qualified_control_cells"]
    if (len(retained_controls) != 84 or len(set(retained_controls)) != 84
            or any(not all(old_checks[name].get(k) is True for k in
                           ("source_qualified", "error_gate_pass", "kernel_identity_pass"))
                   for name in retained_controls)):
        raise ValueError("root prefill retained controls lost their original qualification")
    confirmations = [c for c in controls.values() if "original_failed_control" in c]
    if len(confirmations) != 3:
        raise ValueError("root prefill confirmation cohort differs")
    for case in confirmations:
        original = case["original_failed_control"]
        prediction = prior_freeze["predictions"][original["cell_id"]]["seconds"]
        if (old_checks[original["cell_id"]]["source_qualified"] is not False
                or original["verdict"]["sha256"] != pins["prior_verdict"]["sha256"]
                or case["unchanged_prior_prediction"] != prediction
                or frozen["predictions"][case["cell_id"]]["seconds"] != prediction):
            raise ValueError("root prefill confirmation refits or hides an original failure")
    if handoff["historical_source_qualified"] is not prior_verdict["source_qualified"]:
        raise ValueError("root prefill historical qualification changed")
    historical_failures = {"references": [n for n, p in prior_freeze["reference_points"].items()
                                           if not p["source_qualified"]],
                           "controls": [c["cell_id"] for c in prior_verdict["checks"] if not c["source_qualified"]]}
    if handoff["original_failures_preserved"] != historical_failures:
        raise ValueError("root prefill original failed reference/control list changed")
    checks = verdict["checks"]
    if len(checks) != 116 or {c["cell_id"] for c in checks} != set(controls):
        raise ValueError("root prefill individual heldout coverage differs")
    gates = (verdict.get("all_error_gates_pass") is True
             and all(c.get("error_gate_pass") is True and c.get("kernel_identity_pass") is True for c in checks)
             and all(g.get("pass") is True for g in verdict["groups"])
             and region_verdict.get("all_error_gates_pass") is True
             and len(region_verdict["checks"]) == 18
             and all(c.get("pass") is True for c in region_verdict["checks"]))
    qualified = (all(p.get("source_qualified") is True for p in points.values())
                 and all(c.get("source_qualified") is True for c in checks)
                 and region_verdict.get("source_qualified") is True)
    if (handoff["all_error_gates_pass"] is not gates or final["all_error_gates_pass"] is not gates
            or handoff["source_qualified"] is not qualified or final["source_qualified"] is not qualified):
        raise ValueError("root prefill qualification flags disagree with individual evidence")
    if not gates or not all(p.get("source_qualified") is True for p in points.values()):
        raise ValueError("root prefill consumption requires qualified references and passing error/kernel gates")
    failed = [c for c in checks if c.get("source_qualified") is not True]
    if failed and not (allow_failed_spread and diagnostic_only):
        raise ValueError("root prefill failed heldout spread needs explicit diagnostic selection")
    for check in failed:
        observed = check["observed"]
        values = observed.get("all_three") or []
        if (len(values) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values)
                or observed.get("seconds") != sorted(values)[1]
                or observed.get("range_over_median") != (max(values) - min(values)) / sorted(values)[1]
                or observed["range_over_median"] <= .05
                or observed.get("qualification") != "unqualified_pending_spread_review"
                or frozen["predictions"][check["cell_id"]].get("source_qualified") is not True):
            raise ValueError("root prefill diagnostic selection permits only heldout spread failures")
    return {**{k: old[k] for k in active_old}, **new}, points, region_freeze


class RootPrefillPrices(PriceLibrary):
    """Consume the declared source points; all other work delegates to the base."""

    def __init__(self, base, handoff_path, handoff_sha256, *, deployment_scope_sha256,
                 allow_failed_spread=False, diagnostic_only=False):
        super().__init__()
        if (type(allow_failed_spread) is not bool or type(diagnostic_only) is not bool
                or allow_failed_spread and not diagnostic_only):
            raise ValueError("root prefill spread opt-in requires diagnostic_only=True")
        handoff, loaded = load_json(handoff_path, role="oracle.root_prefill_sources")
        scope = handoff.get("scope") or {}
        if (loaded.sha256 != handoff_sha256 or handoff.get("schema") != "compass.root_prefill_reference_export/1"
                or handoff.get("candidate_activated") is not False
                or handoff.get("fit_inputs_are_references_only") is not True
                or handoff.get("heldout_timings_used_as_fit_inputs") is not False
                or scope.get("model") != "Qwen/Qwen3.8-27B" or scope.get("topology") != {"tp": 1}
                or scope.get("dtype") != "bfloat16" or scope.get("num_sequences") != 1
                or not deployment_scope_sha256 or scope.get("request_scope_sha256") != deployment_scope_sha256
                or getattr(base, "launch_charge_seconds", 0) != 0):
            raise ValueError("root prefill handoff, deployment scope or launch policy differs")
        evidence, evidence_inputs = _evidence(handoff, handoff_path)
        active, points, regions = _validate(handoff, evidence,
            allow_failed_spread=allow_failed_spread, diagnostic_only=diagnostic_only)
        self.base, self.source = base, PriceLibrary()
        self.handoff_sha256, self.source_qualified = loaded.sha256, handoff["source_qualified"]
        self.region_points = tuple((c["query"], c["total_history"], c["produces_output"],
            regions["predictions"][c["cell_id"]]["prepare"], regions["predictions"][c["cell_id"]]["postprocess"])
            for c in evidence["plan"]["region_cases"])
        if len(self.region_points) != 9 or len({r[:3] for r in self.region_points}) != 9:
            raise ValueError("root prefill exact region domain differs")
        for q, history, output, prepare, post in self.region_points:
            if (type(prepare) not in (int, float) or not math.isfinite(prepare) or prepare <= 0
                    or type(post) not in (int, float) or not math.isfinite(post)
                    or (post <= 0 if output else post != 0)):
                raise ValueError("root prefill region is not positive or structural zero")
        self._exact, self._warm_gdn, self._cold_gdn, self._bounded_mha = {}, {}, {}, {}
        seen = set()
        for entry in handoff["entries"]:
            paths = [str(Path(handoff_path).parent / Path(entry[k]["path"]).name) for k in ("price", "graph")]
            blob, graph = self.source._ingest(*paths, None, None)
            if any(item.sha256 != entry[key]["sha256"] for key, item in zip(("price", "graph"), self.source.loaded_inputs[-2:])):
                raise ValueError("root prefill price/graph differs from its pin")
            if (len(entry["cells"]) != len(graph["ops"]) or len(entry["cells"]) != len(blob["prices"])
                    or blob.get("unpriced") or set(blob["prices"]) != {_signature_of(op) for op in graph["ops"]}):
                raise ValueError("root prefill price/graph closure differs")
            for name, op in zip(entry["cells"], graph["ops"]):
                if name in seen or name not in active:
                    raise ValueError("root prefill duplicate or undeclared source cell")
                seen.add(name)
                case, point = active[name], points[name]
                record, _ = self.source.lookup(op, {"tp": 1})
                if (record is None or _signature_of(op) != case["signature"]
                        or record["signature"] != point["signature"] or record["seconds"] != point["seconds"]
                        or any(record.get(k) != case[v] for k, v in
                               (("cache", "observed_cache"), ("arg_sets", "arg_sets"), ("kv_regions", "kv_regions")))):
                    raise ValueError("root prefill exported value differs from its frozen source/treatment")
                record = dict(record, source_qualified=self.source_qualified,
                              source_handoff_sha256=self.handoff_sha256, source_cell_id=name,
                              origin_cohort=point["origin_cohort"])
                if op["name"] == GDN:
                    cold = _cold_gdn_identity(op)
                    warm = _gdn_identity(op)
                    if cold is not None and cold[0] == 0 and point["origin_cohort"] == "new":
                        self._cold_gdn[cold[1]] = record
                    elif warm is not None and warm[1] == 0 and point["origin_cohort"] == "retained":
                        self._warm_gdn[warm[2]] = record
                    else:
                        raise ValueError("root prefill GDN reference mask/alias/query differs")
                elif op["name"] == MHA:
                    layer = _layer(op, MHA_LAYERS, 3)
                    if layer is None or layer[0] != 3:
                        raise ValueError("root prefill MHA reference is not canonical")
                    if point["origin_cohort"] == "retained":
                        identity = _mha_identity(op)
                        if identity is None or identity[1] not in PREFIXES:
                            raise ValueError("root prefill retained MHA endpoint differs")
                        self._bounded_mha.setdefault(identity[3], {})[identity[1]] = record
                    else:
                        self._exact[_key(layer[1])] = record
                elif op["name"] in (GEMM, GATHER, QK_NORM):
                    self._exact[_key(op)] = record
                else:
                    raise ValueError("root prefill unexpected primitive family")
        if seen != set(active) or any(set(p) != set(PREFIXES) for p in self._bounded_mha.values()):
            raise ValueError("root prefill exported source domain is incomplete")
        self.loaded_inputs = base.loaded_inputs + (loaded,) + evidence_inputs + self.source.loaded_inputs
        self.sources = base.sources + self.source.sources
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _source_lookup(self, op, topology):
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        if op.get("group") is not None or any(int(n) != 1 for n in (topology or {}).values()):
            return None
        name, layer, record = op.get("name"), None, None
        if name == GDN:
            cold, warm = _cold_gdn_identity(op), _gdn_identity(op)
            if cold is not None:
                layer, key = cold
                record = self._cold_gdn.get(key)
            elif warm is not None:
                _, layer, key = warm
                record = self._warm_gdn.get(key)
        elif name == MHA:
            normalized = _layer(op, MHA_LAYERS, 3)
            if normalized is not None:
                layer, normalized = normalized
                record = self._exact.get(_key(normalized))
            identity = _mha_identity(op)
            if record is None and identity is not None and identity[3] in self._bounded_mha:
                _, prefix, layer, key = identity
                points = self._bounded_mha[key]
                if prefix in points:
                    record = points[prefix]
                else:
                    weight = (prefix - PREFIXES[0]) / (PREFIXES[1] - PREFIXES[0])
                    low, high = (points[p] for p in PREFIXES)
                    record = dict(low, seconds=(1 - weight) * low["seconds"] + weight * high["seconds"],
                                  **{INTERPOLATED_FLAG: True}, interpolation={
                                      "prefixes": list(PREFIXES), "weights": [1 - weight, weight],
                                      "source_signatures": [low["signature"], high["signature"]]})
        elif name in (GEMM, QK_NORM, GATHER):
            if name == GATHER:
                shapes = op.get("input_shapes") or []
                if (len(shapes) != 2 or len(shapes[0]) != 2 or shapes != [[shapes[0][0], 5120], [1]]
                        or dict(op.get("int_values") or ()).get(1) != [shapes[0][0] - 1]):
                    return None
            record = self._exact.get(_key(op))
        if record is None:
            return None
        if layer is not None and layer != (0 if name == GDN else 3):
            record = dict(record, **{INTERPOLATED_FLAG: True}, layer_transfer={
                "source_layer": 0 if name == GDN else 3, "target_layer": layer,
                "source_handoff_sha256": self.handoff_sha256})
        return record, ("interpolated://root-prefill/bounded-source" if record.get(INTERPOLATED_FLAG) else record["source"])

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching frozen root prefill sources")

    def lookup(self, op, topology=None, registration=None):
        selected = self._source_lookup(op, topology)
        return selected if selected is not None else self.base.lookup(op, topology, registration)

    def _body_lookup(self, op, topology, registration, modelled_memo):
        selected = self._source_lookup(op, topology)
        return selected if selected is not None else self.base._body_lookup(op, topology, registration, modelled_memo)

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        return None if key is None else (key, self.handoff_sha256)

    def describe(self):
        return f"RootPrefillPrices({self.base.describe()}; source={self.handoff_sha256}; qualified={self.source_qualified})"


@dataclass(frozen=True)
class ExactPrefillRegions:
    base: object
    points: tuple
    source_handoff_sha256: str

    @property
    def topologies(self):
        return self.base.topologies

    def _selection(self, shape):
        if (len(shape.num_scheduled_tokens) != 1 or len(shape.context_lens) != 1
                or shape.num_prefill_tokens != shape.total_tokens or not shape.num_prefill_tokens
                or shape.compiled is not True or shape.capture_bucket is not None
                or any(int(n) != 1 for n in (shape.topology or {}).values())
                or any(int(n) != 0 for n in (shape.rank_coords or {}).values())):
            return None
        key = (shape.num_scheduled_tokens[0], shape.context_lens[0], shape.produces_output)
        for query, history, output, prepare, post in self.points:
            if key == (query, history, output):
                return {"<prepare>": prepare, "<postprocess>": post}
        return None

    def refusal(self, shape):
        return None if self._selection(shape) is not None else self.base.refusal(shape)

    def breakdown(self, shape):
        selected = self._selection(shape)
        return selected if selected is not None else self.base.breakdown(shape)

    def seconds(self, shape):
        return sum(self.breakdown(shape).values())

    def band(self, shape):
        if self._selection(shape) is not None:
            raise ValueError("exact prefill sources have no calibrated uncertainty band")
        return self.base.band(shape)

    def describe(self):
        return f"ExactPrefillRegions(source={self.source_handoff_sha256}; {self.base.describe()})"
