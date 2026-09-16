"""Qualified native prefill A/P families over a frozen source-feature hull.

Source labels and absolute allocation IDs never enter the affine price. Native
work and transfer strata guard its applicability; retained exact sources win.
"""
from dataclasses import InitVar, dataclass
from itertools import combinations
from math import gcd, isfinite
from pathlib import Path
from statistics import median

import numpy as np

from atom.compass.core.cost.cache_regions import CachedPrefillRegions
from atom.compass.core.cost.native_prefill_regions import (
    LIMIT,
    NativePrefillRegions,
    _breakdown,
    _cell_for,
)
from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.templates import BindRefusal

SCHEMA = "compass.native_ap_family_sources/1"
EVIDENCE = ("design", "domain", "acquisition_plan", "source", "rule", "freeze",
            "heldout", "verdict", "placement_transfer", "native_complete",
            "initial_runtime", "final_runtime", "copy_closeout", "qualification")
FAMILIES = ("N1_cold_P0", "N1_cached_P0", "N1_cached_P1", "N2_cached_P0", "N2_cached_P1")
ROUND_OFF_SECONDS = 1e-12
PATH_STRATA = {
    "N1_cold_P0": {"prefill_continuations": [[False]], "output_rows": [[False]],
                   "prior_sampled_batch_rows": [1], "shared_prefix_blocks": [None]},
    "N1_cached_P0": {"prefill_continuations": [[False], [True]], "output_rows": [[False]],
                     "prior_sampled_batch_rows": [1], "shared_prefix_blocks": [None]},
    "N1_cached_P1": {"prefill_continuations": [[True]], "output_rows": [[True]],
                     "prior_sampled_batch_rows": [1, 2], "shared_prefix_blocks": [None]},
    "N2_cached_P0": {"prefill_continuations": [[True, False], [True, True]],
                     "output_rows": [[False, False]], "prior_sampled_batch_rows": [1],
                     "shared_prefix_blocks": [2, 3]},
    "N2_cached_P1": {"prefill_continuations": [[True, True]],
                     "output_rows": [[False, True], [True, True]], "prior_sampled_batch_rows": [1],
                     "shared_prefix_blocks": [2, 3]},
}


def work_features(q, history, blocks, output):
    """Read portable work cardinalities; no request identity selects a price."""
    if (len(q) not in (1, 2) or len(q) != len(history) or len(q) != len(blocks)
            or any(type(x) is not int for row in (q, history, blocks) for x in row)
            or any(t <= 0 or h < 0 or h % 16 or b <= 0 or h + t > 16 * b
                   for t, h, b in zip(q, history, blocks))):
        raise ValueError("native A/P work has an unsupported query/history/allocation extent")
    cached = all(h > 0 for h in history)
    if not cached and (len(q) != 1 or history != [0] or output):
        raise ValueError("native A/P source does not qualify mixed-cache or cold-output work")
    family = f"N{len(q)}_{'cached' if cached else 'cold'}_P{int(output)}"
    if family == "N1_cold_P0":
        return family, [1]
    chunks = sum((t + 7) // 8 for t in q)
    return family, [1, sum(q), sum(blocks), chunks if output else max(0, chunks - 1024)]


def source_halfspaces(points):
    """Exact three-dimensional source hull, with primitive integer <= facets."""
    points = sorted({tuple(point) for point in points})
    facets = set()
    for origin, first, second in combinations(points, 3):
        u = [x - y for x, y in zip(first, origin)]
        v = [x - y for x, y in zip(second, origin)]
        normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                  u[0] * v[1] - u[1] * v[0]]
        if not any(normal):
            continue
        bound = sum(a * x for a, x in zip(normal, origin))
        distances = [sum(a * x for a, x in zip(normal, point)) - bound for point in points]
        if max(distances) == min(distances) == 0:
            continue
        if max(distances) <= 0:
            row = normal + [bound]
        elif min(distances) >= 0:
            row = [-x for x in normal] + [-bound]
        else:
            continue
        divisor = gcd(*row)
        facets.add(tuple(x // divisor for x in row))
    if len(facets) < 4:
        raise ValueError("native A/P source anchors do not span a bounded feature hull")
    return [list(row) for row in sorted(facets)]


def _in_domain(domain, q, history, blocks, vector):
    if len(vector) == 1:
        return domain.get("exact") == {"q": q, "history": history, "blocks": blocks}
    return all(sum(a * x for a, x in zip(row[:3], vector[1:])) <= row[3]
               for row in domain["halfspaces"])


def _components(rule, family, vector):
    prepare = sum(x * y for x, y in zip(vector, rule["prepare"][family]["prepare_coefficients"]))
    post = rule["postprocess"][family[1]] if family.endswith("P1") else 0.0
    return {"prepare": prepare, "postprocess": post}


def _stratum_errors(context, domain, n, cached, q, history):
    errors = []
    for key in ("prefill_continuations", "output_rows"):
        actual = context.get(key)
        if actual is None or list(actual) not in domain[key]:
            errors.append(key)
    if context.get("prior_sampled_batch_rows") not in domain["prior_sampled_batch_rows"]:
        errors.append("prior_sampled_batch_rows")
    mask = context.get("output_rows")
    hits = context.get("prefix_cache_hit_tokens")
    if (mask is None or len(mask) != n or any(not 1 <= t <= 16 if output else t % 16 != 0
                                            for t, output in zip(q, mask))):
        errors.append("per-row query/output work")
    if hits is None or len(hits) != n or any(type(x) is not int or not 0 <= x <= h for x, h in zip(hits, history)):
        errors.append("prefix-hit provenance")
    slots, sources = context.get("state_slots"), context.get("state_fork_srcs")
    if (slots is None or sources is None or len(slots) != n or len(sources) != n
            or context.get("state_rows") != tuple(range(n)) or any(x < 0 for x in slots)
            or (cached and (any(x < 0 for x in sources) or len(set(slots + sources)) != 2 * n))
            or (not cached and tuple(sources) != (-1,))):
        errors.append("state read/write alias relation")
    if n == 2 and (context.get("shared_prefix_blocks") not in domain["shared_prefix_blocks"]
                   or context.get("kv_sharing_is_prefix_only") is not True):
        errors.append("shared KV prefix relation")
    return errors


def _source_context(descriptor):
    forward = descriptor["forward_context"]
    if (forward.get("context_source") != "native_runner_cpu_metadata"
            or forward.get("deferred_output") is not True
            or forward.get("pending_token_copies") != 1
            or forward.get("pending_logprob_entries") != 1
            or forward.get("pending_nonnull_logprob_copies") != 0
            or forward.get("pending_mtp_status_copies") != 0):
        raise ValueError("native A/P source has an unqualified native deferred-output queue")
    tables = descriptor["block_tables"]
    shared, prefix_only = None, None
    if len(tables) == 2:
        shared = 0
        for left, right in zip(*tables):
            if left != right:
                break
            shared += 1
        prefix_only = set(tables[0]).intersection(tables[1]) == set(tables[0][:shared])
    return {**forward["scope"], "prefill_continuations": tuple(descriptor["prefill_continuation"]),
            "output_rows": tuple(descriptor["output_rows"]),
            "state_slots": tuple(descriptor["state_slots"]),
            "state_fork_srcs": tuple(descriptor["state_fork_srcs"]),
            "state_rows": tuple(descriptor["state_rows"]),
            "prefix_cache_hit_tokens": tuple(descriptor["prefix_cache_hit_tokens"]),
            "shared_prefix_blocks": shared, "kv_sharing_is_prefix_only": prefix_only,
            "prior_sampled_batch_rows": forward["prior_sampled_batch_rows"]}


def _groups(rows, points, role, scope, domains, *, check_hull=True):
    groups = {}
    for row in rows:
        name = row["point_id"]
        if name not in points or row["role"] != role or row["normal_return"] is not True:
            raise ValueError("native A/P source/heldout cohort or normal return differs")
        point, descriptor = points[name], row["descriptor"]
        for key in ("q", "history", "blocks", "produces_output", "output_rows"):
            if descriptor[key] != point[key]:
                raise ValueError("native A/P observed descriptor differs: " + key)
        if descriptor["prefill_continuation"] != point["prefill_continuation"]:
            raise ValueError("native A/P source transfer stratum changed")
        family, vector = work_features(point["q"], point["history"], point["blocks"], point["produces_output"])
        context = _source_context(descriptor)
        if (any(context.get(key) != expected for key, expected in scope.items())
                or descriptor["blocks"] != [len(row) for row in descriptor["block_tables"]]
                or descriptor["context"] != [t + h for t, h in zip(point["q"], point["history"])]
                or _stratum_errors(context, domains[family], len(point["q"]), "cached" in family,
                                    point["q"], point["history"])
                or descriptor.get("compiled") is not True or descriptor.get("capture_bucket") is not None
                or descriptor.get("prefill_rows") != len(point["q"])
                or not descriptor.get("topology")
                or any(x != 1 for x in descriptor["topology"].values())
                or any(x != 0 for x in descriptor["rank_coords"].values())):
            raise ValueError("native A/P observed scope differs from its frozen source strata")
        if check_hull and not _in_domain(domains[family], point["q"], point["history"], point["blocks"], vector):
            raise ValueError("native A/P source or heldout lies outside its frozen source hull")
        for key, value in (("temperatures", 1.0), ("top_ks", -1), ("top_ps", 1.0),
                           ("return_logprobs", False), ("independent_noise", False)):
            if descriptor.get(key) != [value] * len(point["q"]):
                raise ValueError("native A/P source sampling path differs: " + key)
        seconds = row["seconds"]
        if (set(seconds) != {"prepare", "run_model", "postprocess", "forward"}
                or any(type(x) not in (int, float) or not isfinite(x) or x < 0 for x in seconds.values())
                or abs(seconds["prepare"] - (seconds["forward"] - seconds["run_model"] - seconds["postprocess"])) > 1e-12
                or (not point["produces_output"] and seconds["postprocess"] != 0)):
            raise ValueError("native A/P source does not preserve the measured F/M/P definitions")
        groups.setdefault(name, []).append(row)
    if (set(groups) != set(points) or any(sorted(row["repetition"] for row in group) != list(range(6))
                                         for group in groups.values())):
        raise ValueError("native A/P cohort lacks six independent observations per declared point")
    return groups


def _placement_transfer(data, points, base):
    transfer = data["placement_transfer"]
    selected = {name: point for name, point in points.items() if point["role"] == "transfer"}
    groups = _groups(transfer["rows"], selected, "transfer", base.scope, PATH_STRATA, check_hull=False)
    checks = transfer["checks"]
    expected = {(name, part) for name in groups for part in ("prepare", "postprocess")}
    if (not groups or len(checks) != len(expected) or
            {(c["point_id"], c["component"]) for c in checks} != expected
            or transfer.get("source_refitted") is not False):
        raise ValueError("native A/P placement gate lacks its raw independent control cohort")
    for check in checks:
        point = points[check["point_id"]]
        if len(point["q"]) != 1:
            raise ValueError("native A/P placement control is outside its retained source")
        cell = _cell_for(base.cells, point["q"][0], point["history"][0], point["produces_output"])
        if cell is None:
            raise ValueError("native A/P placement control has no retained source value")
        source = cell.get("endpoints", [cell])[0]["selector"]
        if (point["blocks"] != [source["allocation_blocks"]]
                or point["prefill_continuation"] != [source["path"] == "within_request_checkpoint"]):
            raise ValueError("native A/P placement control changes the retained native path")
        prediction = _breakdown(cell, point["q"][0])["<" + check["component"] + ">"]
        values = [r["seconds"][check["component"]] for r in groups[point["id"]]]
        if (check.get("passed") is not True or abs(prediction - median(values)) > LIMIT
                or check["prediction"] != prediction or check["observed_median"] != median(values)
                or sorted(check["raw"]) != sorted(values)):
            raise ValueError("native A/P normal-path placement transfer failed or changed")


def _validate(data, loaded, base, statuses):
    design, domain, rule, freeze = (data[key] for key in ("design", "domain", "rule", "freeze"))
    if (design.get("target_timing_inputs") != [] or design.get("changes_existing_refusals") is not False
            or rule.get("schema") != "compass.native_ap_family_rule/1" or rule.get("source_only") is not True
            or rule.get("median_gate_seconds") != LIMIT or rule.get("uncertainty_band_claim") is not False
            or freeze.get("heldout_started") is not False or freeze.get("source_only") is not True
            or freeze["rule"]["sha256"] != loaded["rule"].sha256
            or freeze["source_input"]["sha256"] != loaded["source"].sha256
            or freeze["design"]["sha256"] != loaded["design"].sha256
            or rule["source_input"]["sha256"] != loaded["source"].sha256
            or domain.get("schema") != "compass.native_ap_family_domain/1"
            or domain.get("feature_axes") != ["totalQ", "totalBlocks", "convWork"]
            or domain["design"]["sha256"] != loaded["design"].sha256
            or set(domain["families"]) != set(FAMILIES) or set(statuses) != set(FAMILIES)):
        raise ValueError("native A/P source rule, freeze or declared family inventory differs")
    complete, initial, final = (data[key] for key in ("native_complete", "initial_runtime", "final_runtime"))
    if (complete.get("success") is not True or complete.get("engine_closed") is not True
            or complete["plan_sha256"] != loaded["acquisition_plan"].sha256
            or initial.get("inherited_run_model") is not True or final.get("inherited_run_model") is not True
            or initial["attention_scope"] != final["attention_scope"]
            or data["placement_transfer"].get("passed") is not True
            or not data["placement_transfer"].get("checks")
            or any(check.get("passed") is not True for check in data["placement_transfer"]["checks"])):
        raise ValueError("native A/P placement, runtime or completed native source run is unqualified")
    points = {point["id"]: point for point in design["points"]}
    _placement_transfer(data, points, base)
    source_points = {k: v for k, v in points.items() if v["role"] == "source"}
    heldout_points = {k: v for k, v in points.items() if v["role"] == "heldout"}
    sources = _groups(data["source"]["rows"], source_points, "source", base.scope, domain["families"])
    heldouts = _groups(data["heldout"]["rows"], heldout_points, "heldout", base.scope, domain["families"])
    for family, allowed in PATH_STRATA.items():
        region = domain["families"][family]
        rows = [row for groups in (sources, heldouts) for group in groups.values() for row in group
                if work_features(row["descriptor"]["q"], row["descriptor"]["history"],
                    row["descriptor"]["blocks"], row["descriptor"]["produces_output"])[0] == family]
        for key, maximum in allowed.items():
            values = region[key]
            observed = [_source_context(row["descriptor"]).get(key) for row in rows]
            observed = [list(value) if isinstance(value, tuple) else value for value in observed]
            if (any(value not in maximum or value not in observed for value in values)
                    or (key != "shared_prefix_blocks" and not values)):
                raise ValueError("native A/P declared transfer stratum lacks observed source/heldout support: " + key)
    anchors = {name: {part: median(r["seconds"][part] for r in group)
                      for part in ("prepare", "postprocess")} for name, group in sources.items()}
    for name, point in points.items():
        if point["role"] == "retained_source":
            cell = _cell_for(base.cells, point["q"][0], point["history"][0], point["produces_output"])
            if cell is None:
                raise ValueError("native A/P retained anchor has no qualified exact source")
            anchors[name] = {key.strip("<>"): value for key, value in _breakdown(cell, point["q"][0]).items()}
    if set(rule["prepare"]) != set(FAMILIES):
        raise ValueError("native A/P fitted family inventory differs")
    for family in FAMILIES:
        names = rule["prepare"][family]["source_anchor_ids"]
        expected = {name for name in anchors if work_features(points[name]["q"], points[name]["history"],
                        points[name]["blocks"], points[name]["produces_output"])[0] == family}
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("native A/P fit adds, omits or reuses heldout anchors")
        vectors = [work_features(points[name]["q"], points[name]["history"], points[name]["blocks"],
                                  points[name]["produces_output"])[1] for name in names]
        coefficients, _, rank, _ = np.linalg.lstsq(np.asarray(vectors, dtype=float),
            np.asarray([anchors[name]["prepare"] for name in names]), rcond=None)
        frozen = rule["prepare"][family]["prepare_coefficients"]
        verification_vectors = vectors + [work_features(point["q"], point["history"], point["blocks"],
            point["produces_output"])[1] for point in heldout_points.values()
            if work_features(point["q"], point["history"], point["blocks"], point["produces_output"])[0] == family]
        if (rank != len(vectors[0]) or len(frozen) != len(coefficients)
                or any(type(value) not in (int, float) or not isfinite(value) for value in frozen)
                or any(abs(sum(x * (a - b) for x, a, b in zip(vector, frozen, coefficients))) > ROUND_OFF_SECONDS
                       for vector in verification_vectors)):
            raise ValueError("native A/P frozen coefficients differ from source-only least squares")
        region = domain["families"][family]
        domain_names = region["source_anchor_ids"]
        if len(domain_names) != len(expected) or set(domain_names) != expected:
            raise ValueError("native A/P source hull has a different anchor inventory")
        if family == "N1_cold_P0":
            if region.get("exact") != {"q": [32], "history": [0], "blocks": [33]}:
                raise ValueError("native cold source is only Q32/H0/B33")
        else:
            domain_vectors = [work_features(points[name]["q"], points[name]["history"], points[name]["blocks"],
                                            points[name]["produces_output"])[1][1:] for name in domain_names]
            if (region["source_feature_vectors"] != domain_vectors
                    or region["halfspaces"] != source_halfspaces([vector[1:] for vector in vectors])):
                raise ValueError("native A/P declared domain widens or changes the source-anchor hull")
        if any(min(_components(rule, family, vector).values()) < 0 for vector in vectors):
            raise ValueError("native A/P component is negative inside its source hull")
    for n in (1, 2):
        values = [anchors[name]["postprocess"] for name in anchors
                  if len(points[name]["q"]) == n and points[name]["produces_output"]]
        if rule["postprocess"][str(n)] != median(values):
            raise ValueError("native A/P postprocess differs from its source-only median")
    checks = data["verdict"]["checks"]
    if (data["verdict"].get("source_refitted") is not False or len(checks) != 2 * len(heldouts)
            or {(c["point_id"], c["component"]) for c in checks} !=
               {(name, part) for name in heldouts for part in ("prepare", "postprocess")}):
        raise ValueError("native A/P heldout cohort or component inventory differs")
    passed = {family: True for family in FAMILIES}
    observed_families = set()
    for check in checks:
        point = points[check["point_id"]]
        family, vector = work_features(point["q"], point["history"], point["blocks"], point["produces_output"])
        values = [row["seconds"][check["component"]] for row in heldouts[point["id"]]]
        prediction = _components(rule, family, vector)[check["component"]]
        error = abs(prediction - median(values))
        if (check["family"] != family or check["prediction"] != prediction
                or check["observed_median"] != median(values) or sorted(check["raw"]) != sorted(values)
                or abs(check["absolute_error"] - error) > 1e-15 or check["passed"] is not (error <= LIMIT)):
            raise ValueError("native A/P heldout verdict changes its frozen prediction or raw observations")
        passed[family] &= error <= LIMIT
        observed_families.add(family)
    if (observed_families != set(FAMILIES) or any(statuses[name] and not passed[name] for name in FAMILIES)
            or data["verdict"].get("passed") is not all(passed.values())):
        raise ValueError("native A/P family was qualified without passing independent heldouts")
    return passed


@dataclass(frozen=True)
class NativeAPFamilyRegions:
    base: NativePrefillRegions
    rule: dict
    domain: dict
    family_qualified: dict
    source_handoff_sha256: str
    loaded_inputs: tuple
    allocation: InitVar[object]
    source_qualified: bool = False
    verification_roundoff_seconds: float = ROUND_OFF_SECONDS
    version: str = "native-ap-families/1"
    provenance: str = "Source-only affine A/P work features, exact source hulls and independent family heldouts"

    def __post_init__(self, allocation):
        if not isinstance(self.base, NativePrefillRegions) or not self.base.source_qualified:
            raise ValueError("native A/P families require their retained qualified native-prefill adapter")
        object.__setattr__(self, "_allocation", allocation)

    @classmethod
    def load(cls, base, path, sha256, allocation, *, deployment_scope_sha256):
        handoff, identity = load_json(path, role="oracle.native_ap_regions")
        if (identity.sha256 != sha256 or handoff.get("schema") != SCHEMA
                or set(handoff["evidence"]) != set(EVIDENCE)
                or handoff.get("retained_native_handoff_sha256") != base.source_handoff_sha256
                or handoff.get("scope") != base.scope
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("native A/P handoff, retained source or deployment scope differs")
        data, loaded = {}, {}
        for role in EVIDENCE:
            reference = handoff["evidence"][role]
            data[role], loaded[role] = load_json(str(Path(path).parent / reference["path"]),
                                               role="oracle.native_ap_regions." + role)
            if loaded[role].sha256 != reference["sha256"]:
                raise ValueError("native A/P source evidence changed: " + role)
        statuses = {name: value["source_qualified"] for name, value in handoff["families"].items()}
        if any(type(value) is not bool for value in statuses.values()):
            raise ValueError("native A/P family qualification must be explicit")
        _validate(data, loaded, base, statuses)
        qualification = data["qualification"]
        expected = {role: item.sha256 for role, item in loaded.items() if role != "qualification"}
        if (qualification.get("schema") != "compass.native_ap_family_qualification/1"
                or qualification.get("evidence_sha256") != expected
                or qualification.get("families") != statuses
                or data["copy_closeout"].get("passed") is not True
                or data["acquisition_plan"]["design"]["sha256"] != loaded["design"].sha256
                or data["acquisition_plan"]["domain"]["sha256"] != loaded["domain"].sha256
                or data["acquisition_plan"]["request_scope"]["sha256"] != deployment_scope_sha256):
            raise ValueError("native A/P qualification, acquisition or copy closeout does not bind its evidence")
        allocation.capture_region_context = True
        return cls(base, data["rule"], data["domain"], statuses, identity.sha256,
                   (identity, *loaded.values()), allocation,
                   source_qualified=handoff.get("source_qualified") is True and any(statuses.values()))

    @property
    def topologies(self):
        return self.base.topologies

    def _query(self, shape):
        if (self.base._cell(shape) is not None or not shape.is_prefill
                or shape.num_prefill_tokens != shape.total_tokens or shape.batch_size not in (1, 2)):
            return None
        inherited = self.base.base
        while inherited is not None:
            if (isinstance(inherited, CachedPrefillRegions) and inherited.final is not None
                    and shape.batch_size == 1 and shape.produces_output
                    and inherited.final.contains(shape.num_scheduled_tokens[0],
                        shape.context_lens[0] - shape.num_scheduled_tokens[0])
                    and inherited._selection(shape) is not None):
                return None
            inherited = getattr(inherited, "base", None)
        context = self._allocation.region_context_for(shape)
        q = list(shape.num_scheduled_tokens)
        history = [c - t for c, t in zip(shape.context_lens, q)]
        blocks = list(context["allocation_blocks"])
        family, vector = work_features(q, history, blocks, shape.produces_output)
        return family, vector, q, history, blocks, context

    def refusal(self, shape):
        try:
            selected = self._query(shape)
        except (BindRefusal, ValueError) as exc:
            return str(exc)
        if selected is None:
            return self.base.refusal(shape)
        family, vector, q, history, blocks, context = selected
        if not self.family_qualified[family]:
            return "native A/P family is unqualified or failed heldouts: " + family
        domain = self.domain["families"][family]
        if not _in_domain(domain, q, history, blocks, vector):
            return "native A/P work is outside its qualified source-anchor hull"
        if (not shape.topology or any(x != 1 for x in shape.topology.values())
                or any(x != 0 for x in shape.rank_coords.values())
                or shape.compiled is not True or shape.capture_bucket is not None
                or any(context.get(key) != value for key, value in self.base.scope.items())):
            return "native A/P deployment or compiled/capture path differs"
        if (context.get("output_state_representation") != "predictive_deferred_batch"
                or context.get("prior_sampled_has_logprobs") is not False):
            return "native A/P requires the actual predictive deferred batch without prior logprob work"
        for key, value in (("temperatures", 1.0), ("top_ks", -1), ("top_ps", 1.0),
                           ("return_logprobs", False), ("independent_noise", False)):
            if context.get(key) != (value,) * len(q):
                return "native A/P sampling path differs: " + key
        errors = _stratum_errors(context, domain, len(q), "cached" in family, q, history)
        if errors:
            return "native A/P transfer stratum differs: " + ", ".join(errors)
        return None

    def breakdown(self, shape):
        why = self.refusal(shape)
        if why is not None:
            raise ValueError("no qualified native A/P family for this path: " + why)
        selected = self._query(shape)
        if selected is None:
            return self.base.breakdown(shape)
        return {"<" + key + ">": value for key, value in _components(self.rule, selected[0], selected[1]).items()}

    def seconds(self, shape):
        return sum(self.breakdown(shape).values())

    def band(self, shape):
        if self._query(shape) is None:
            return self.base.band(shape)
        raise ValueError("native A/P median gates provide no calibrated uncertainty band")

    def describe(self):
        return f"{self.version}: {self.source_handoff_sha256}; {self.provenance}; base={self.base.describe()}"
