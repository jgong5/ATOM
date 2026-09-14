"""Exact small-query cells with independently qualified, bounded layer transfer."""

from copy import deepcopy
import math
from pathlib import Path
import re

from atom.compass.core.cost.cached_q16 import GDN, MHA, GATHER
from atom.compass.core.cost.library import (
    INTERPOLATED_FLAG, PriceLibrary, _cost_key_of, _layout_fingerprint, _signature_of,
)
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.loaded_input import load_json


GEMM = "aiter::gemm_a16w16"
QUERIES = tuple(range(1, 16))
PREFIXES = (32752, 65520)
GDN_LAYERS = tuple(i for i in range(64) if i % 4 != 3)
MHA_LAYERS = tuple(range(3, 64, 4))


def _key(op):
    return (_cost_key_of(op), _layout_fingerprint(op),
            tuple(tuple(x) for x in op.get("output_shapes") or ()),
            tuple(op.get("output_dtypes") or ()), op.get("abi", ""))


def _layer(op, allowed, canonical):
    normalized = deepcopy(op)
    normalized["scalars"] = [[key, value] for key, value in op.get("scalars") or ()]
    found = []
    for scalar in normalized.get("scalars") or ():
        value = scalar[1]
        match = re.fullmatch(r"(.*\.layers\.)(\d+)(\..+)", value) if isinstance(value, str) else None
        if match:
            number = int(match[2])
            if number not in allowed:
                return None
            found.append(number)
            scalar[1] = f"{match[1]}{canonical}{match[3]}"
    return (found[0], normalized) if len(found) == 1 else None


def _gdn_identity(op):
    context = dict(op.get("context") or ())
    q = context.get("num_prefill_tokens")
    if (type(q) is not int or q not in QUERIES or context.get("num_prefills") != 1
            or context.get("num_decodes") != 0 or context.get("num_decode_tokens") != 0
            or context.get("num_actual_tokens") != q or context.get("replayssm") is not False
            or context.get("has_initial_state") != [[1], "bool"]
            or context.get("non_spec_query_start_loc") != [[0, q], "int32"]):
        return None
    slots = []
    for name in ("non_spec_state_indices_in_tensor", "non_spec_state_indices_tensor"):
        value = context.get(name)
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or value[1] != "int32"
                or not isinstance(value[0], (list, tuple)) or len(value[0]) != 1
                or type(value[0][0]) is not int or not 0 <= value[0][0] < 32):
            return None
        slots.append(value[0][0])
    if slots[0] == slots[1]:
        return None  # q16's alias control does not qualify new in-place queries.
    layer = _layer(op, GDN_LAYERS, 0)
    return None if layer is None else (q, layer[0], _key(layer[1]))


def _mha_identity(op):
    context = dict(op.get("context") or ())
    q = context.get("max_seqlen_q")
    cached = context.get("num_cached_tokens")
    history = context.get("context_lens")
    if (type(q) is not int or q not in QUERIES
            or not isinstance(cached, list) or len(cached) != 1
            or not isinstance(history, list) or len(history) != 1):
        return None
    C, H = cached[0], history[0]
    if (type(C) is not int or type(H) is not int or C % 16 or not PREFIXES[0] <= C <= PREFIXES[1]
            or H != C + q or context.get("cu_seqlens_q") != [0, q]
            or context.get("cu_seqlens_k") != [0, H] or context.get("max_seqlen_k") != H
            or context.get("total_kv") != H or context.get("has_cached") is not True
            or context.get("is_prefill") is not True or context.get("state") != "prefill_prefix"):
        return None
    if "positions" in context and context["positions"] != list(range(C, H)) * 3:
        return None
    layer = _layer(op, MHA_LAYERS, 3)
    if layer is None:
        return None
    normalized = layer[1]
    replacements = {"context_lens": [0], "cu_seqlens_k": [0, 0],
                    "max_seqlen_k": 0, "total_kv": 0, "num_cached_tokens": [0]}
    normalized["context"] = [[key, replacements.get(key, value)] for key, value in normalized["context"]]
    return q, C, layer[0], _key(normalized)


class LowQueryPrices(PriceLibrary):
    """A separate source addition; baseline and q16 observations are untouched."""

    def __init__(self, base, handoff_path, handoff_sha256, *, deployment_scope_sha256,
                 allow_failed_spread=False, diagnostic_only=False):
        super().__init__()
        self.base = base
        if type(allow_failed_spread) is not bool or type(diagnostic_only) is not bool:
            raise ValueError("low-query diagnostic flags must be booleans")
        if allow_failed_spread and not diagnostic_only:
            raise ValueError("failed low-query spread requires diagnostic_only=1")
        handoff, loaded = load_json(handoff_path, role="oracle.low_q_sources")
        if loaded.sha256 != handoff_sha256:
            raise ValueError("low-query handoff differs from its explicit SHA-256")
        scope = handoff.get("scope") or {}
        queries = scope.get("queries") or []
        gemms = {tuple(shape) for shape in scope.get("gemm_shapes") or ()}
        if (handoff.get("schema") != "compass.low_q_reference_export/1"
                or type(handoff.get("source_qualified")) is not bool
                or (not handoff["source_qualified"] and not allow_failed_spread)
                or handoff.get("all_error_gates_pass") is not True
                or handoff.get("fit_inputs_are_references_only") is not True
                or handoff.get("heldout_timings_used_as_fit_inputs") is not False
                or scope.get("model") != "Qwen/Qwen3.8-27B" or scope.get("topology") != {"tp": 1}
                or scope.get("dtype") != "bfloat16" or scope.get("num_sequences") != 1
                or not queries or queries != sorted(set(queries))
                or any(type(q) is not int or q not in QUERIES for q in queries)
                or scope.get("cached_prefix") != list(PREFIXES)
                or scope.get("gdn_layers") != list(GDN_LAYERS) or scope.get("mha_layers") != list(MHA_LAYERS)
                or scope.get("gdn_alias") != "fork"
                or not deployment_scope_sha256 or scope.get("request_scope_sha256") != deployment_scope_sha256):
            raise ValueError("low-query model, geometry, layer family or deployment scope differs")
        if getattr(base, "launch_charge_seconds", 0) != 0:
            raise ValueError("low-query sources do not qualify an extra kernel launch count")
        evidence = []
        def read_evidence(name, role):
            item = handoff["evidence"][name]
            path = Path(handoff_path).parent / Path(item["path"]).name
            value, identity = load_json(str(path), role=role)
            if identity.sha256 != item["sha256"]:
                raise ValueError(f"low-query {name} differs from its source pin")
            evidence.append(identity)
            return value, identity
        plan, plan_input = read_evidence("plan", "oracle.low_q_plan")
        frozen, frozen_input = read_evidence("freeze", "oracle.low_q_reference_freeze")
        verdict, _ = read_evidence("verdict", "oracle.low_q_validation")
        live, _ = read_evidence("preflight", "oracle.low_q_live_abi")
        references = {c["cell_id"]: c for c in plan["cases"] if c["phase"] == "reference"}
        controls = {c["cell_id"] for c in plan["cases"] if c["phase"] == "heldout"}
        checks = verdict.get("checks") or []
        active = {name: c for name, c in references.items() if
                  (c["family"] == "gemm" and (c["M"], c["N"], c["K"]) in gemms)
                  or (c["family"] != "gemm" and c["q"] in queries)}
        required_controls = {c["cell_id"] for c in plan["cases"] if c["phase"] == "heldout" and (
            (c["family"] in ("gdn", "mha") and c["q"] in {1, *queries})
            or c["family"] == "gather"
            or (c["family"] == "gemm" and (c["M"], c["N"], c["K"]) in gemms))}
        required_references = set(active)
        for name in required_controls:
            required_references.update(s["reference_cell_id"] for s in frozen["predictions"][name]["sources"])
        selected_checks = [c for c in checks if c["cell_id"] in required_controls]
        groups = {c["group"] for c in selected_checks}
        failed = [c for c in selected_checks if c.get("source_qualified") is not True]
        failed_names = {c["cell_id"] for c in failed}
        if failed and handoff.get("purpose") != "diagnostic":
            raise ValueError("failed low-query sources must be labelled diagnostic")
        # Only heldout measurement spread may be used diagnostically. Frozen
        # reference values, predicted error, dispatch identity and dependency
        # membership remain mandatory, including the shared q1 layer controls.
        for check in failed:
            observed = check.get("observed") or {}
            readings = observed.get("all_three") or []
            if (not allow_failed_spread or len(readings) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in readings)
                    or observed.get("seconds") != sorted(readings)[1]
                    or observed.get("range_over_median") != (max(readings) - min(readings)) / sorted(readings)[1]
                    or observed["range_over_median"] <= .05
                    or observed.get("qualification") != "unqualified_pending_spread_review"
                    or observed.get("source_qualified") is not False
                    or frozen["predictions"][check["cell_id"]].get("source_qualified") is not True):
                raise ValueError("low-query diagnostic selection permits only declared heldout spread failures")
        if (len(references) != 83 or len(controls) != 160
                or frozen.get("heldout_timings_read") is not False
                or set(frozen.get("reference_points") or {}) != set(references)
                or set(frozen.get("predictions") or {}) != controls
                or frozen["plan"]["sha256"] != plan_input.sha256
                or verdict["plan"]["sha256"] != plan_input.sha256
                or verdict["prediction_freeze"]["sha256"] != frozen_input.sha256
                or handoff.get("campaign_source_qualified") is not verdict.get("source_qualified")
                or handoff.get("campaign_all_error_gates_pass") is not verdict.get("all_error_gates_pass")
                or len(checks) != 160 or {c["cell_id"] for c in checks} != controls
                or not all(c.get("error_gate_pass") is True and c.get("kernel_identity_pass") is True
                           for c in selected_checks)
                or handoff["source_qualified"] != (not failed)
                or set(handoff.get("failed_spread_controls") or []) != failed_names
                or {c["cell_id"] for c in selected_checks} != required_controls
                or not groups or not groups.issubset({g["group"] for g in verdict.get("groups") or []})
                or not all(g.get("pass") is True for g in verdict["groups"] if g["group"] in groups)
                or not required_references.issubset(references)
                or not all(frozen["reference_points"][name].get("source_qualified") is True for name in required_references)
                or set(handoff.get("selected_reference_cells") or []) != set(active)
                or set(handoff.get("required_control_cells") or []) != required_controls):
            raise ValueError("low-query source needs every individual reference/control qualification")
        self.queries = tuple(queries)
        self.source_qualified = handoff["source_qualified"]
        self.failed_spread_controls = tuple(sorted(failed_names))
        self.campaign_source_qualified = verdict["source_qualified"]
        abi = live.get("family_abi") or {}
        if (live.get("plan", {}).get("sha256") != plan_input.sha256
                or abi.get("flags") != plan.get("backend_flags")
                or len(abi.get("all_layers") or {}) != 64
                or set(abi.get("families") or {}) != {"gdn", "mha"}):
            raise ValueError("low-query live layer-family ABI receipt is incomplete")
        for name, value in abi["all_layers"].items():
            family = (live.get("native", {}).get("layers", {}).get(name) or {}).get("family")
            if family not in abi["families"] or value != abi["families"][family]:
                raise ValueError("low-query live layer-family ABI is not homogeneous")
        self.source = PriceLibrary()
        self._gdn, self._mha, self._gather, self._gemm = {}, {}, {}, {}
        self.handoff_sha256 = loaded.sha256
        seen, families = set(), set()
        for entry in handoff["entries"]:
            family = entry["family"]
            if family in families or family not in ("gdn", "mha", "gather", "gemm"):
                raise ValueError("unexpected low-query source family")
            families.add(family)
            paths = [Path(handoff_path).parent / Path(entry[k]["path"]).name for k in ("price", "graph")]
            blob, graph = self.source._ingest(str(paths[0]), str(paths[1]), None, None)
            if any(actual.sha256 != entry[key]["sha256"] for key, actual in
                   zip(("price", "graph"), self.source.loaded_inputs[-2:])):
                raise ValueError("low-query price/graph differs from its source pin")
            cells = entry["cells"]
            if (len(cells) != len(graph["ops"]) or len(cells) != len(blob.get("prices") or {})
                    or set(blob["prices"]) != {_signature_of(op) for op in graph["ops"]} or blob.get("unpriced")):
                raise ValueError("low-query price/graph closure differs")
            for cell_id, op in zip(cells, graph["ops"]):
                if cell_id in seen or cell_id not in active:
                    raise ValueError("duplicate or unplanned low-query reference cell")
                seen.add(cell_id)
                case, point = references[cell_id], frozen["reference_points"][cell_id]
                record, why = self.source.lookup(op)
                if (case["family"] != family or record is None or record["signature"] != point["signature"]
                        or _signature_of(op) != case["signature"]
                        or record["seconds"] != point["seconds"] or not math.isfinite(record["seconds"])
                        or record["seconds"] <= 0 or record.get("cache") != case["observed_cache"]
                        or record.get("kv_regions") != case["kv_regions"] or record.get("arg_sets") != case["arg_sets"]):
                    raise ValueError(f"low-query reference differs from frozen point/treatment: {why}")
                if family == "gdn":
                    identity = _gdn_identity(op)
                    if (op["name"] != GDN or identity is None or identity[1] != 0
                            or identity[0] != case["q"]):
                        raise ValueError("GDN reference is outside its canonical query/fork ABI")
                    self._gdn[identity[2]] = record
                elif family == "mha":
                    identity = _mha_identity(op)
                    if (op["name"] != MHA or identity is None or identity[2] != 3
                            or identity[0] != case["q"] or identity[1] != case["cached_prefix"]
                            or identity[1] not in PREFIXES):
                        raise ValueError("MHA reference is outside its canonical query/prefix ABI")
                    self._mha.setdefault(identity[3], {})[identity[1]] = record
                elif family == "gather":
                    if (op["name"] != GATHER or op.get("input_shapes") != [[case["q"], 5120], [1]]
                            or dict(op.get("int_values") or ()).get(1) != [case["q"] - 1]):
                        raise ValueError("gather reference differs from its exact selected final row")
                    self._gather[_key(op)] = record
                else:
                    if (op["name"] != GEMM or op.get("input_shapes") !=
                            [[case["M"], case["K"]], [case["N"], case["K"]]]):
                        raise ValueError("GEMM reference differs from its exact structural shape")
                    self._gemm[_key(op)] = record
        if (seen != set(active) or len(self._gdn) != len(queries) or len(self._mha) != len(queries)
                or any(set(p) != set(PREFIXES) for p in self._mha.values())
                or len(self._gather) != len(queries) or len(self._gemm) != len(gemms)):
            raise ValueError("low-query source grid is incomplete")
        self.loaded_inputs = base.loaded_inputs + (loaded,) + tuple(evidence) + self.source.loaded_inputs
        self.sources = base.sources + self.source.sources
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _source_lookup(self, op, topology):
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        if op.get("group") is not None or any(int(n) != 1 for n in (topology or {}).values()):
            return None
        name = op.get("name")
        if name == GDN:
            identity = _gdn_identity(op)
            if identity is None or identity[2] not in self._gdn:
                return None
            record = self._gdn[identity[2]]
            layer = identity[1]
        elif name == MHA:
            identity = _mha_identity(op)
            if identity is None or identity[3] not in self._mha:
                return None
            _, C, layer, key = identity
            points = self._mha[key]
            if C in points:
                record = points[C]
            else:
                weight = (C - PREFIXES[0]) / (PREFIXES[1] - PREFIXES[0])
                low, high = (points[c] for c in PREFIXES)
                record = dict(low, seconds=(1 - weight) * low["seconds"] + weight * high["seconds"],
                              **{INTERPOLATED_FLAG: True})
                record["interpolation"] = {"basis": "exact query, bounded cached prefix, qualified layer family",
                    "prefixes": list(PREFIXES), "weights": [1 - weight, weight],
                    "source_signatures": [low["signature"], high["signature"]],
                    "source_handoff_sha256": self.handoff_sha256}
        elif name == GATHER:
            shapes = op.get("input_shapes") or []
            if (len(shapes) != 2 or len(shapes[0]) != 2 or shapes[0][0] not in QUERIES
                    or shapes != [[shapes[0][0], 5120], [1]]
                    or dict(op.get("int_values") or ()).get(1) != [shapes[0][0] - 1]):
                return None
            record = self._gather.get(_key(op))
            return (record, record["source"]) if record else None
        elif name == GEMM:
            record = self._gemm.get(_key(op))
            return (record, record["source"]) if record else None
        else:
            return None
        if layer != (0 if name == GDN else 3):
            record = dict(record, **{INTERPOLATED_FLAG: True}, layer_transfer={
                "target_layer": layer, "source_layer": 0 if name == GDN else 3,
                "source_handoff_sha256": self.handoff_sha256})
        return record, ("interpolated://low-query/bounded-layer-family" if record.get(INTERPOLATED_FLAG) else record["source"])

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching frozen low-query sources")

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
        return (f"LowQueryPrices({self.base.describe()}; exact queries={self.queries}, "
                f"bounded layer/prefix transfer; selected_qualified={self.source_qualified}; "
                f"campaign_qualified={self.campaign_source_qualified}; "
                f"source={self.handoff_sha256})")
