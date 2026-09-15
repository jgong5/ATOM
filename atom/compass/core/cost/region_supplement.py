"""Freshly confirmed exact regions supplement, preserving the original cohort."""

from dataclasses import dataclass
import math
from pathlib import Path

from atom.compass.core.cost.root_prefill import ExactPrefillRegions
from atom.compass.core.loaded_input import load_json


SUPPLEMENT_KEYS = frozenset((
    (16384, 32768, False), (8192, 40960, False), (1040, 42000, False),
    (7, 42007, True), (2832, 44832, False), (1, 44833, True),
    (1728, 46560, False), (9, 46569, True), (1664, 46496, False),
    (10, 46506, True),
))
EVIDENCE_ROLES = ("candidate", "plan", "verdict", "native_complete", "heldout_phase")
COMPONENTS = ("prepare", "postprocess")
LIMIT = .00011


def _summary(values, structural_zero):
    if (len(values) != 3 or any(type(x) not in (int, float) or not math.isfinite(x)
            or (x != 0 if structural_zero else x <= 0) for x in values)):
        raise ValueError("region supplement requires three positive values or structural zeros")
    median = sorted(values)[1]
    return {"seconds": median, "all_three": values, "minimum": min(values),
            "maximum": max(values), "range_over_median": None if structural_zero else
            (max(values) - min(values)) / median, "structural_zero": structural_zero}


def _candidate_points(candidate, root_prefill):
    cells = candidate["cells"]
    names = {c["cell_id"] for c in cells}
    keys = {(c["query"], c["total_history"], c["produces_output"]) for c in cells}
    original = {p[:3] for p in root_prefill.region_points}
    if (candidate.get("schema") != "compass.root1493_region_supplement_candidate/1"
            or candidate.get("fresh_confirmation_timings_read") is not False
            or candidate.get("target_timings_used") is not False
            or candidate.get("candidate_activated") is not False
            or candidate.get("source_consumption_forbidden") is not True
            or candidate.get("old_r3_heldouts_previously_inspected") is not True
            or candidate.get("relative_spread_qualification_gate") is not None
            or candidate.get("absolute_error_limits") != dict.fromkeys(COMPONENTS, LIMIT)
            or len(cells) != 10 or len(names) != 10 or keys != SUPPLEMENT_KEYS
            or len(original) != 9 or original & keys
            or set(candidate["predictions"]) != names
            or set(candidate["reference_points"]) != names
            or set(candidate["reference_join_rows"]) != names):
        raise ValueError("region supplement reference boundary or exact ten-key domain differs")
    for component in ("freeze", "verdict"):
        sources = [s for s in root_prefill.loaded_inputs
                   if s.role == "oracle.root_prefill_region_" + component]
        if (len(sources) != 1 or sources[0].sha256 !=
                candidate["original_nine_cell_" + component]["sha256"]):
            raise ValueError("region supplement replaces the original nine-cell evidence")
    points = []
    for cell in cells:
        name = cell["cell_id"]
        rows = candidate["reference_join_rows"][name]
        if (type(cell["query"]) is not int or type(cell["total_history"]) is not int
                or type(cell["produces_output"]) is not bool
                or cell["cached_prefix"] + cell["query"] != cell["total_history"]
                or len(rows) != 3 or [r["repeat"] for r in rows] != [1, 2, 3]
                or any(r["observed_phase"] !=
                       f"regions_reference_r{r['repeat']}_request{cell['request']}" for r in rows)):
            raise ValueError("region supplement values are not the three original reference joins")
        for component in COMPONENTS:
            summary = _summary([r[component + "_seconds"] for r in rows],
                               component == "postprocess" and not cell["produces_output"])
            if (candidate["reference_points"][name][component] != summary
                    or candidate["predictions"][name][component] != summary["seconds"]):
                raise ValueError("region supplement prediction refits its original reference median")
        points.append((cell["query"], cell["total_history"], cell["produces_output"],
                       *(candidate["predictions"][name][c] for c in COMPONENTS)))
    return tuple(points)


def _validate_fresh(evidence, pins):
    candidate, plan, verdict, complete, phase = (evidence[k] for k in EVIDENCE_ROLES)
    fixtures = candidate["fresh_fixtures"]
    spec = candidate["fresh_fixture_specification"]
    if (plan.get("schema") != "compass.root1493_region_supplement_executable/1"
            or plan.get("candidate_freeze") != pins["candidate"]
            or plan.get("cells") != candidate["cells"] or plan.get("fresh_fixtures") != fixtures
            or spec.get("warm_seed") != 1495000
            or spec.get("heldout_seeds") != [1495001, 1495002, 1495003]
            or len(fixtures["warm"]) != 1 or len(fixtures["heldout"]) != 3
            or len({f["sha256"] for group in fixtures.values() for f in group}) != 4
            or verdict.get("schema") != "compass.root1493_region_supplement_verdict/1"
            or phase.get("schema") != "compass.root1493_region_supplement_phase/1"
            or complete.get("schema") != "compass.root1493_region_supplement_native_complete/1"
            or any(v.get("plan") != pins["plan"] or v.get("prediction_freeze") != pins["candidate"]
                   for v in (verdict, phase, complete))
            or phase.get("phase") != "heldout" or phase.get("fixtures") != fixtures
            or any(v.get("source_requests") != 28 or v.get("primitive_requests") != 0
                   or v.get("target_requests") != 0 for v in (phase, complete))
            or verdict.get("heldout_evidence", {}).get("phase_result") != pins["heldout_phase"]
            or complete.get("phase_result") != pins["heldout_phase"]
            or complete.get("verdict") != pins["verdict"]
            or complete.get("candidate_activated") is not False
            or verdict.get("fresh_confirmation_only") is not True
            or verdict.get("old_r3_heldouts_used_for_validation") is not False
            or any(v.get("source_qualified") is not True or v.get("all_error_gates_pass") is not True
                   or v.get("target_timings_used") is not False or v.get("candidate_activated") is not False
                   for v in (verdict,))):
        raise ValueError("region supplement needs a separate passing fresh source verdict and completion")
    expected = [(c["cell_id"], repeat, fixtures["heldout"][repeat - 1])
                for repeat in (1, 2, 3) for c in candidate["cells"]]
    records = phase["records"]
    if ([(r["cell_id"], r["repeat"], r["fixture"]) for r in records] != expected
            or len({r["raw"]["sha256"] for r in records}) != 30
            or len({r["raw"]["path"] for r in records}) != 30):
        raise ValueError("region supplement fresh fixture/repeat coverage differs")
    checks = verdict["checks"]
    expected_checks = {(c["cell_id"], component) for c in candidate["cells"] for component in COMPONENTS}
    if len(checks) != 20 or {(c["cell_id"], c["component"]) for c in checks} != expected_checks:
        raise ValueError("region supplement requires all twenty distinct component gates")
    cells = {c["cell_id"]: c for c in candidate["cells"]}
    for check in checks:
        name, component = check["cell_id"], check["component"]
        prediction = candidate["predictions"][name][component]
        summary = _summary(check["observed"]["all_three"],
                           component == "postprocess" and not cells[name]["produces_output"])
        error = summary["seconds"] - prediction
        if (check["observed"] != summary or check["prediction"] != prediction
                or check["error_seconds"] != error or check["absolute_limit_seconds"] != LIMIT
                or check["pass"] is not True or abs(error) > LIMIT):
            raise ValueError("region supplement fresh absolute error gate failed or was relabelled")


@dataclass(frozen=True)
class PrefillRegionSupplement(ExactPrefillRegions):
    loaded_inputs: tuple

    @classmethod
    def load(cls, base, root_prefill, handoff_path, handoff_sha256, *, deployment_scope_sha256):
        handoff, identity = load_json(handoff_path, role="oracle.region_supplement_sources")
        scope = handoff.get("scope") or {}
        if (identity.sha256 != handoff_sha256
                or handoff.get("schema") != "compass.root_prefill_region_supplement_export/1"
                or handoff.get("source_qualified") is not True
                or handoff.get("all_error_gates_pass") is not True
                or handoff.get("candidate_activated") is not False
                or handoff.get("target_timings_used") is not False
                or handoff.get("fit_inputs_are_references_only") is not True
                or handoff.get("heldout_timings_used_as_fit_inputs") is not False
                or scope.get("model") != "Qwen/Qwen3.8-27B" or scope.get("topology") != {"tp": 1}
                or scope.get("dtype") != "bfloat16" or scope.get("num_sequences") != 1
                or not deployment_scope_sha256
                or scope.get("request_scope_sha256") != deployment_scope_sha256
                or root_prefill.source_qualified is not True
                or not isinstance(base, ExactPrefillRegions) or len(base.points) != 9
                or base.points != root_prefill.region_points
                or base.source_handoff_sha256 != root_prefill.handoff_sha256):
            raise ValueError("region supplement requires the qualified original root source and deployment scope")
        pins = handoff["evidence"]
        if set(pins) != set(EVIDENCE_ROLES):
            raise ValueError("region supplement evidence closure differs")
        evidence, loaded = {}, [identity]
        for role in EVIDENCE_ROLES:
            value, source = load_json(str(Path(handoff_path).parent / pins[role]["path"]),
                                      role="oracle.region_supplement_" + role)
            if source.sha256 != pins[role]["sha256"]:
                raise ValueError(f"region supplement {role} differs from its evidence pin")
            evidence[role] = value
            loaded.append(source)
        points = _candidate_points(evidence["candidate"], root_prefill)
        _validate_fresh(evidence, pins)
        return cls(base, points, identity.sha256, tuple(loaded))
