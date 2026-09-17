"""Expose original reference measurements for diagnostics, without qualifying them.

Unknown work retains the underlying library's refusal. The only non-exact
prices admitted here are already-frozen, bounded low-query MHA predictions.
"""
from pathlib import Path

from atom.compass.core.cache_policy import cache_on_policy, policy_errors
from atom.compass.core.cost.cached_q16 import GDN, MHA
from atom.compass.core.cost.families import attention
from atom.compass.core.cost.library import INTERPOLATED_FLAG, PriceLibrary, _layout_fingerprint
from atom.compass.core.cost.low_query import PREFIXES, _mha_identity, mha_prefix_interpolation
from atom.compass.core.cost.prepared import PreparedOperator
from atom.compass.core.cost.reached_primitive_evidence import (
    Evidence, _phase, _retained, graph_for, same_pin,
)
from atom.compass.core.cost.reached_primitives import INVALID_ALIAS, work_identity


SCHEMA = "compass.diagnostic_reference_export/1"
ROLE_PREFIX = "oracle.diagnostic_references."


def _bounded_prediction(case, prediction, op, references, points, graphs):
    """Check the frozen interpolation against the existing low-query path class."""
    sources = prediction.get("sources", [])
    identity = _mha_identity(op) if case["family"] == "mha" else None
    if (identity is None or len(sources) != 2
            or case.get("frozen_prediction_sources") != sources):
        raise ValueError("diagnostic interpolation requires a frozen bounded MHA prediction")
    names = [source["reference_cell_id"] for source in sources]
    if len(set(names)) != 2 or any(name not in references for name in names):
        raise ValueError("diagnostic interpolation has unknown reference dependencies")
    source_ids = [_mha_identity(graphs[name]) for name in names]
    if (any(value is None for value in source_ids)
            or [value[1] for value in source_ids] != list(PREFIXES)
            or any((value[0], value[3]) != (identity[0], identity[3]) for value in source_ids)):
        raise ValueError("diagnostic MHA interpolation changes query, layout, bounds or path class")
    seconds, weights = mha_prefix_interpolation(
        points[names[0]]["seconds"], points[names[1]]["seconds"], identity[1])
    if (prediction.get("seconds") != seconds
            or [source.get("weight") for source in sources] != weights
            or prediction.get("source_qualified") is not
            all(points[name]["source_qualified"] for name in names)):
        raise ValueError("diagnostic MHA prediction changes frozen weights, values or precision")
    return names, seconds


def _read_sources(reader, handoff, deployment_scope_sha256):
    scope = handoff.get("scope") or {}
    if (handoff.get("schema") != SCHEMA or handoff.get("purpose") != "diagnostic"
            or handoff.get("accepted") is not False or handoff.get("source_qualified") is not False
            or scope.get("model") != "Qwen/Qwen3.8-27B" or scope.get("topology") != {"tp": 1}
            or scope.get("dtype") != "bfloat16" or not deployment_scope_sha256
            or scope.get("request_scope_sha256") != deployment_scope_sha256):
        raise ValueError("diagnostic references require explicit unqualified diagnostic deployment scope")
    pins = handoff["evidence"]
    plan = reader.read(pins["reference_plan"], "reference_plan")
    args = plan["engine_args"]
    if (plan.get("schema") != "compass.reached_primitive_executable/2"
            or args.get("model") != scope["model"] or args.get("tensor_parallel_size") != 1
            or args.get("pipeline_parallel_size", 1) != 1 or args.get("kv_cache_dtype") != "bf16"
            or policy_errors(plan["cache_policy"], cache_on_policy())):
        raise ValueError("diagnostic reference plan changes model or native cache policy")
    failure = reader.read(pins["original_failure"], "original_failure")
    if (failure.get("accepted") is not False or failure.get("source_qualified") is not False
            or not same_pin(failure.get("science"), pins["reference_plan"])):
        raise ValueError("diagnostic references must preserve the original source failure")
    points, samples, dispatch, _ = _phase(reader, "reference", plan, pins["reference_plan"],
        pins["reference_phase"], pins["reference_preflight"])
    phase = reader.read(pins["reference_phase"], "reference_phase")
    retained, retained_cases, retained_samples = _retained(reader, plan, dispatch, phase["dispatch"])
    references = {case["cell_id"]: case for case in plan["cases"] if case["phase"] == "reference"}
    references.update(retained_cases)
    points.update(retained)
    samples.update(retained_samples)
    frozen = reader.read(pins["prediction_freeze"], "prediction_freeze")
    prediction_plan = reader.read(pins["prediction_plan"], "prediction_plan")
    if (frozen.get("schema") != "compass.low_q_reference_freeze/1"
            or not same_pin(frozen.get("plan"), pins["prediction_plan"])
            or not same_pin(frozen.get("original_plan"), pins["reference_plan"])
            or not same_pin(frozen.get("original_failure"), pins["original_failure"])
            or frozen.get("heldout_timings_read") is not False
            or frozen.get("target_timings_used") is not False
            or frozen.get("candidate_activated") is not False
            or prediction_plan.get("backend_flags") != plan["backend_flags"]
            or prediction_plan.get("resolved_runtime") != plan["resolved_runtime"]):
        raise ValueError("diagnostic predictions change their source, runtime or reference-only freeze")
    for name, point in frozen["reference_points"].items():
        if name not in points or any(point.get(key) != points[name][key] for key in
                ("seconds", "all_three", "range_over_median", "source_qualified", "signature")):
            raise ValueError("diagnostic freeze changes original reference values or precision")
    graphs = {name: graph_for(reader, case, "reference_graph." + name)
              for name, case in references.items()}
    return references, points, samples, graphs, prediction_plan, frozen


class DiagnosticReferencePrices(PriceLibrary):
    """A diagnostic data overlay; qualification and complete coverage stay distinct."""

    def __init__(self, base, handoff_path, handoff_sha256, *, deployment_scope_sha256,
                 diagnostic_only=False):
        super().__init__()
        if diagnostic_only is not True or getattr(base, "launch_charge_seconds", 0) != 0:
            raise ValueError("diagnostic reference prices require diagnostic mode and no extra launch charge")
        self.base = base
        path = Path(handoff_path).resolve()
        reader = Evidence(path.parent, "diagnostic")
        reader.prefix = ROLE_PREFIX
        handoff = reader.read({"path": str(path), "sha256": handoff_sha256}, "handoff")
        references, points, samples, graphs, plan, frozen = _read_sources(
            reader, handoff, deployment_scope_sha256)
        reference_phase = reader.read(handoff["evidence"]["reference_phase"], "reference_phase")
        self._selected = {}
        self.excluded_references = []
        self.exact_reference_count = 0
        records = {}
        for name, case in references.items():
            point, op = points[name], graphs[name]
            selected = None
            for raw_pin in samples[name]:
                price = reader.read(raw_pin, "median." + name)["prices"][case["signature"]]
                if price["seconds"] == point["seconds"] and selected is None:
                    selected = dict(price, source=reader.path(raw_pin), source_price=raw_pin)
            if selected is None:
                raise ValueError("diagnostic source median is not an original measured repetition")
            _, layer = work_identity(op)
            record = dict(selected, source_qualified=False, diagnostic_only=True,
                accepted=False, reference_precision_passed=point["source_qualified"],
                reference_observations=point, reference_cell_id=name,
                source_graph=case["graph"], source_layer=layer,
                source_handoff_sha256=handoff_sha256, original_failure=handoff["evidence"]["original_failure"],
                dispatch_evidence=reference_phase["dispatch"],
                native_path_evidence=handoff["evidence"]["reference_preflight"],
                scope={"topology": {"tp": 1}, "registration": None},
                signature=case["signature"], layout=_layout_fingerprint(op))
            self.exact_reference_count += self._insert(op, record)
            records[name] = record
        self.bounded_prediction_count = 0
        for case in plan["cases"]:
            if case["phase"] != "heldout":
                continue
            prediction = frozen["predictions"][case["cell_id"]]
            if len(prediction["sources"]) != 2:
                continue
            op = graph_for(reader, case, "prediction_graph." + case["cell_id"])
            names, seconds = _bounded_prediction(case, prediction, op, references, points, graphs)
            record = dict(records[names[0]], seconds=seconds, **{INTERPOLATED_FLAG: True},
                frozen_prediction=prediction, prediction_case=case["cell_id"],
                prediction_freeze=handoff["evidence"]["prediction_freeze"],
                source="interpolated://diagnostic-reference/bounded-mha-prefix")
            self.bounded_prediction_count += self._insert(op, record)
        self._gdn_work = {key[0] for key in self._selected if key[0][0].startswith(GDN + "|")}
        self.source_qualified = False
        self.handoff_sha256 = handoff_sha256
        self.loaded_inputs = base.loaded_inputs + tuple(reader.inputs)
        self.sources = base.sources + [str(path)]
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _insert(self, op, record):
        if op.get("name") == MHA:
            regime = attention.regime_of(op)
            if (isinstance(regime, attention.Refusal)
                    and regime.reason == attention.CACHED_ROW_STARTS_REFUSAL):
                self.excluded_references.append(dict(
                    reference_cell_id=record.get("reference_cell_id"),
                    source_graph=record.get("source_graph"), reason=regime.reason))
                return 0
        key, _ = work_identity(op)
        if key is None or key[1] == INVALID_ALIAS:
            raise ValueError("diagnostic reference has an unsupported layer or state-alias identity")
        if key in self._selected:
            if self._selected[key]["seconds"] != record["seconds"]:
                raise ValueError("diagnostic references disagree on one exact work identity")
            return 0
        self._selected[key] = record
        return 1

    def _source_lookup(self, op, topology):
        if isinstance(op, PreparedOperator):
            op = op.as_dict()
        key, layer = work_identity(op)
        record = self._selected.get(key)
        if record is None:
            if op.get("name") == GDN and key is not None and key[0] in self._gdn_work:
                return None, "diagnostic GDN reference does not cover this joint state-alias relation"
            return None
        if op.get("group") is not None or any(int(value) != 1 for value in (topology or {}).values()):
            return None, "diagnostic reference requires its TP1 noncollective work"
        if layer is not None and layer != record["source_layer"]:
            record = dict(record, **{INTERPOLATED_FLAG: True}, layer_transfer={
                "source_layer": record["source_layer"], "target_layer": layer,
                "source_handoff_sha256": self.handoff_sha256})
        return record, record["source"]

    def lookup(self, op, topology=None, registration=None):
        selected = self._source_lookup(op, topology)
        return selected if selected is not None else self.base.lookup(op, topology, registration)

    def _body_lookup(self, op, topology, registration, modelled_memo):
        selected = self._source_lookup(op, topology)
        return selected if selected is not None else self.base._body_lookup(op, topology, registration, modelled_memo)

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching diagnostic references")

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        return None if key is None else (key, self.handoff_sha256)

    def describe(self):
        return (f"DiagnosticReferencePrices({self.exact_reference_count} exact references; "
                f"{self.bounded_prediction_count} frozen MHA predictions; source_qualified=False; "
                f"{len(self.excluded_references)} excluded invalid references; "
                f"base={self.base.describe()})")
