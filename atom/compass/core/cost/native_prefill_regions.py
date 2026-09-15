"""Portable prefill A/P sources, selected by the offered native forward path.

The existing LibraryCostOracle consumes refusal/breakdown unchanged. Selected
cells replace the base A/P terms; caller-clock durations are never prices.
"""
from dataclasses import InitVar, dataclass
import math
from pathlib import Path
from statistics import median

from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.templates import BindRefusal


COMPONENTS = ("prepare", "postprocess")
EVIDENCE = ("rule", "freeze", "verdict", "source", "heldout", "plan", "final",
            "native_complete", "observations", "structure")
LIMIT = .000110


def _identity(document):
    return {key: document[key] for key in ("plan_sha256", "ownership_sha256", "arm")}


def _values(rows, component):
    values = []
    for row in rows:
        seconds = row["seconds"]
        value = seconds[component]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or abs(seconds["prepare"] - (seconds["forward"] - seconds["run_model"] -
                                             seconds["postprocess"])) > 1e-12
                or (not row["selector"]["produces_output"] and seconds["postprocess"] != 0)):
            raise ValueError("native region evidence does not preserve the stream A/P definitions")
        values.append(value)
    if len(values) < 6:
        raise ValueError("native region evidence requires at least six observations per source or heldout cell")
    return values


def _validate_source(cells, rows):
    covered = set()
    for cell in cells:
        points = _endpoints(cell)
        if len(points) not in (1, 2):
            raise ValueError("native region rule must be an exact cell or two source endpoints")
        if len(points) == 2:
            first, last = (point["selector"] for point in points)
            if (cell["method"] != "linear_q_1_to_16_no_extrapolation"
                    or (first["q"], last["q"]) != (1, 16)
                    or {k: v for k, v in first.items() if k != "q"} !=
                       {k: v for k, v in last.items() if k != "q"}):
                raise ValueError("native region interpolation changes path or frozen endpoints")
        elif cell["method"] != "pooled_source_median":
            raise ValueError("unknown native region source rule")
        for point in points:
            selected = [row for row in rows if row["selector"] == point["selector"]]
            if set(point["components"]) != set(COMPONENTS):
                raise ValueError("native region source contains a component other than A/P")
            selector = point["selector"]
            if (selector["n"] != 1 or selector["compiled"] is not True or selector["level"] != 3
                    or selector["capture_bucket"] is not None or selector["cudagraph_mode"] != "FULL"
                    or any(selector[axis] != 1 for axis in ("tp", "pp", "dp"))
                    or selector["path"] != ("within_request_checkpoint" if selector["history"] else "cold")
                    or selector["fork"] is not bool(selector["history"])):
                raise ValueError("native region source selector is outside the measured execution path")
            for component in COMPONENTS:
                values = _values(selected, component)
                statistics = point["components"][component]
                if (statistics["n"] != len(values) or statistics["median"] != median(values)
                        or sorted(statistics["raw"]) != sorted(values)):
                    raise ValueError("native region source rule refits or replaces its source observations")
            covered.update(id(row) for row in selected)
    if covered != {id(row) for row in rows} or any(row["role"] != "source" for row in rows):
        raise ValueError("native region fit includes heldout rows or leaves source rows unexplained")


def _validate_heldout(cells, rows, verdict):
    groups = {(row["case"], row["chunk"]) for row in rows}
    checks = verdict["component_checks"]
    expected = {(case, chunk, component) for case, chunk in groups for component in COMPONENTS}
    if (len(checks) != len(expected) or
            {(c["case"], c["chunk"], c["component"]) for c in checks} != expected
            or any(row["role"] != "heldout" for row in rows)):
        raise ValueError("native region heldout component coverage differs")
    for check in checks:
        selected = [r for r in rows if (r["case"], r["chunk"]) == (check["case"], check["chunk"])]
        selector = selected[0]["selector"]
        cell = _cell_for(cells, selector["q"], selector["history"], selector["produces_output"])
        source = _endpoints(cell)[0]["selector"] if cell is not None else None
        if (source is None or any(r["selector"] != selector for r in selected)
                or {k: v for k, v in selector.items() if k != "q"} !=
                   {k: v for k, v in source.items() if k != "q"}):
            raise ValueError("native region heldout changes the source execution path")
        values = _values(selected, check["component"])
        prediction = _breakdown(cell, selector["q"])["<" + check["component"] + ">"]
        error = abs(prediction - median(values))
        if (check["passed"] is not True or error > LIMIT
                or check["predicted"] != prediction or check["observed"]["median"] != median(values)
                or sorted(check["observed"]["raw"]) != sorted(values)
                or abs(check["absolute_median_error"] - error) > 1e-15):
            raise ValueError("native region heldout median gate failed or changed")


def _scope(plan, observed):
    args = plan["engine_args"]
    runtime = observed["storage"]["resolved_runtime"]
    policy = observed["cache"]["policy"]
    if (policy["state_runtime"]["transfer"]["kind"] != "fork"
            or policy["state_runtime"]["transfer"]["readable_midstep"] is not False):
        raise ValueError("native region source requires the observed non-paged checkpoint fork path")
    if (policy != plan["cache_policy"] or any(runtime[key] != value
            for key, value in plan["required_runtime"].items())):
        raise ValueError("native region observed deployment differs from its frozen source plan")
    # This source is Qwen3_5 MRoPE: the pinned model config has mrope_section,
    # and ModelRunner allocates CpuGpuBuffer(3, max_num_batched_tokens) when
    # that key is present. The handoff retains the model-config/source witness;
    # NativeAllocation's actual position_rows must match this source geometry.
    return dict(model=args["model"], kv_cache_dtype=runtime["kv_cache_dtype"],
        compilation_level=runtime["compilation_level"], cudagraph_mode=runtime["cudagraph_mode"],
        pipeline_parallel_size=args["pipeline_parallel_size"],
        enable_prefix_caching=policy["enable_prefix_caching"],
        checkpoint_demand=policy["state_checkpoint_demand"],
        checkpoint_interval_tokens=policy["state_checkpoint_interval_tokens"],
        block_size=runtime["block_size"], max_model_len=args["max_model_len"], position_rows=3,
        speculative_config_absent=True, num_spec_step=0, state_maintenance_empty=True,
        midstep_saves_empty=True)


def _validate_evidence(data, loaded):
    rule, freeze, verdict = (data[key] for key in ("rule", "freeze", "verdict"))
    sources, heldout = data["source"], data["heldout"]
    complete, observations = data["native_complete"], data["observations"]
    identities = (_identity(sources), _identity(heldout), _identity(complete), _identity(observations))
    if (rule.get("schema") != "compass.region_source_rule/1" or rule.get("source_only") is not True
            or verdict.get("schema") != "compass.region_heldout_validation/1"
            or rule["gate_seconds"] != LIMIT or verdict["gate_seconds"] != LIMIT
            or verdict["passed"] is not True
            or freeze["heldout_validation_started"] is not False
            or freeze["rule"]["sha256"] != loaded["rule"].sha256
            or freeze["source_input"]["sha256"] != loaded["source"].sha256
            or rule["source_sha256"] != loaded["source"].sha256
            or verdict["source_input_sha256"] != loaded["source"].sha256
            or verdict["heldout_input_sha256"] != loaded["heldout"].sha256
            or verdict["frozen_rule_sha256"] != loaded["rule"].sha256
            or any(value != rule["source_identity"] for value in identities)
            or verdict["source_identity"] != rule["source_identity"]
            or rule["source_plan_sha256"] != loaded["plan"].sha256
            or verdict["source_plan_sha256"] != loaded["plan"].sha256
            or freeze["source_plan_sha256"] != loaded["plan"].sha256
            or identities[0]["plan_sha256"] != loaded["plan"].sha256
            or complete["success"] is not True or complete["engine_closed"] is not True
            or complete["observations"]["sha256"] != loaded["observations"].sha256
            or observations["region_sources"]["sha256"] != loaded["source"].sha256
            or observations["region_heldouts"]["sha256"] != loaded["heldout"].sha256
            or observations["final"]["sha256"] != loaded["final"].sha256
            or data["final"]["structure"]["sha256"] != loaded["structure"].sha256
            or data["final"]["cache"]["quiescence"]["idle"] is not True
            or rule["source_rows"] != len(sources["rows"])
            or verdict["heldout_rows"] != len(heldout["rows"])):
        raise ValueError("native region evidence does not bind one source freeze, heldout cohort and completed native run")
    _validate_source(rule["cells"], sources["rows"])
    _validate_heldout(rule["cells"], heldout["rows"], verdict)
    # The recorded manager snapshot witnesses that this source did not execute
    # state maintenance. Request IDs join evidence only; they never select cost.
    records = data["structure"]["steps"]
    for row in sources["rows"] + heldout["rows"]:
        target = row["timeline"]
        matching = [r for r in records if list(map(str, r["req_ids"])) == target["req_ids"]
                    and r["query_lens"] == target["q"] and r["context_lens"] == target["context"]]
        if (len(matching) != 1 or matching[0]["state_maintenance"] !=
                dict(relocations=[], checkpoint_stores=0, checkpoint_restores=0)):
            raise ValueError("native region source state-maintenance path is missing or different")


def _endpoints(cell):
    return cell["endpoints"] if "endpoints" in cell else (cell,)


def _cell_for(cells, query, history, output):
    for cell in cells:
        points = _endpoints(cell)
        first, last = points[0]["selector"], points[-1]["selector"]
        if (first["history"] == history and first["produces_output"] == output
                and first["q"] <= query <= last["q"]):
            return cell
    return None


def _breakdown(cell, query):
    points = _endpoints(cell)
    first, last = points[0], points[-1]
    lo, hi = first["selector"]["q"], last["selector"]["q"]
    fraction = (query - lo) / (hi - lo) if hi != lo else 0
    return {"<" + key + ">": first["components"][key]["median"] + fraction * (
                last["components"][key]["median"] - first["components"][key]["median"])
            for key in COMPONENTS}


@dataclass(frozen=True)
class NativePrefillRegions:
    base: object
    cells: tuple
    scope: dict
    source_handoff_sha256: str
    loaded_inputs: tuple
    allocation: InitVar[object]
    source_qualified: bool = False
    deployment_scope_sha256: str = ""
    version: str = "native-prefill-regions/1"
    provenance: str = "Source-only A/P medians with independent heldout median gates; no uncertainty band"

    def __post_init__(self, allocation):
        if not hasattr(allocation, "region_context_for"):
            raise ValueError("native region sources require a native allocation context provider")
        object.__setattr__(self, "_allocation", allocation)

    @classmethod
    def load(cls, base, handoff_path, handoff_sha256, allocation, *, deployment_scope_sha256):
        handoff, identity = load_json(handoff_path, role="oracle.native_prefill_regions")
        if (identity.sha256 != handoff_sha256 or handoff.get("schema") != "compass.native_prefill_region_sources/1"
                or set(handoff["evidence"]) != set(EVIDENCE)):
            raise ValueError("native region source handoff or evidence inventory differs")
        if (not deployment_scope_sha256 or
                (handoff.get("deployment_scope") or {}).get("sha256") != deployment_scope_sha256):
            raise ValueError("native region sources require the matching pinned deployment attention scope")
        data, loaded = {}, {}
        for role in EVIDENCE:
            reference = handoff["evidence"][role]
            value, source = load_json(str(Path(handoff_path).parent / reference["path"]),
                                      role="oracle.native_prefill_regions." + role)
            if source.sha256 != reference["sha256"]:
                raise ValueError("native region evidence changed: " + role)
            data[role], loaded[role] = value, source
        _validate_evidence(data, loaded)
        scope = _scope(data["plan"], data["final"])
        if handoff["scope"] != scope:
            raise ValueError("native region declared scope differs from its observed deployment")
        allocation.capture_region_context = True
        return cls(base, tuple(data["rule"]["cells"]), scope, identity.sha256,
                   (identity, *loaded.values()), allocation,
                   source_qualified=handoff.get("source_qualified") is True,
                   deployment_scope_sha256=deployment_scope_sha256)

    @property
    def topologies(self):
        return self.base.topologies

    def _cell(self, shape):
        if (shape.batch_size != 1 or len(shape.context_lens) != 1
                or not shape.is_prefill or shape.num_prefill_tokens != shape.total_tokens):
            return None
        query = shape.num_scheduled_tokens[0]
        return _cell_for(self.cells, query, shape.context_lens[0] - query, shape.produces_output)

    def refusal(self, shape):
        cell = self._cell(shape)
        if cell is None:
            return self.base.refusal(shape)
        # This coordinate is newly claimed. An incompatible path must not be
        # answered by the less-specific legacy base merely because it has a row.
        try:
            context = self._allocation.region_context_for(shape)
        except BindRefusal as exc:
            return str(exc)
        for key, expected in self.scope.items():
            if context.get(key) != expected:
                return f"native region scope differs: {key}={context.get(key)!r}, measured {expected!r}"
        source = _endpoints(cell)[0]["selector"]
        if (not shape.topology or any(width != 1 for width in shape.topology.values())
                or any(rank != 0 for rank in shape.rank_coords.values())
                or shape.compiled is not source["compiled"]
                or shape.capture_bucket != source["capture_bucket"]):
            return "native region compiled/capture/topology path differs"
        if context.get("prefill_continuations") != (source["path"] == "within_request_checkpoint",):
            if context.get("prefill_continuations") == (False,) and source["history"]:
                return "native region source does not qualify a first-forward prefix resume"
            return "native region source requires its cold/within-request continuation path"
        expected = {
            "allocation_blocks": (source["allocation_blocks"],),
            "temperatures": (source["temperature"],), "top_ks": (source["top_k"],),
            "top_ps": (source["top_p"],), "return_logprobs": (False,),
            "independent_noise": (False,), "state_rows": (0,),
        }
        for key, value in expected.items():
            if context.get(key) != value:
                return f"native region allocation/sampling path differs: {key}"
        slots, sources = context.get("state_slots"), context.get("state_fork_srcs")
        if (slots is None or sources is None or len(slots) != 1 or len(sources) != 1
                or slots[0] < 0 or (sources[0] >= 0) != source["fork"]
                or (source["fork"] and slots[0] == sources[0])):
            return "native region checkpoint fork source/destination differs"
        return None

    def breakdown(self, shape):
        cell = self._cell(shape)
        if cell is None:
            return self.base.breakdown(shape)
        why = self.refusal(shape)
        if why is not None:
            raise ValueError("no measured native region for this path: " + why)
        return _breakdown(cell, shape.num_scheduled_tokens[0])

    def seconds(self, shape):
        return sum(self.breakdown(shape).values())

    def band(self, shape):
        if self._cell(shape) is not None:
            raise ValueError("native region median gates provide no calibrated uncertainty band")
        return self.base.band(shape)

    def describe(self):
        return f"{self.version}: {self.source_handoff_sha256}; {self.provenance}; base={self.base.describe()}"
