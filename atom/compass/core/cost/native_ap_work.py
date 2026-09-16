"""Source-only native prefill A/P work model, with retained exact-cell priority.

Native request chains supply all observations. This reader preserves signed
F-M-P remainders and never consumes complete-forward or caller-clock residuals
as model targets. Independent composition qualification is a separate input.
"""
from itertools import combinations
from dataclasses import InitVar, dataclass
from math import isfinite
from pathlib import Path
from statistics import median

import numpy as np

from atom.compass.core.cost.cache_regions import CachedPrefillRegions
from atom.compass.core.cost.native_ap_exact import SAMPLING
from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.templates import BindRefusal

SCHEMA = "compass.native_ap_work_sources/1"
MODEL_SCHEMA = "compass.native_ap_work_model_candidate/1"
A_FEATURES = ("constant", "query_tokens", "allocated_block_entries", "rows",
              "state_fork_rows", "any_cached", "produces_output")
A_SCALES = (1., 16384., 16632., 3., 3., 1., 1.)
P_FEATURES = ("constant", "sampler_rows", "prior_sampled_rows")
P_SCALES = (1., 3., 3.)
SOURCE_ROLES = ("acquisition_plan", "design", "source", "model", "initial_runtime")


def work_vectors(q, history, blocks, slots, sources, output, prior_rows, sampler_rows):
    forks = sum(src >= 0 and src != dst for src, dst in zip(sources, slots))
    return ([1, sum(q), sum(blocks), len(q), forks, int(any(history)), int(output)],
            [1, sampler_rows, prior_rows])


def observation_vectors(row):
    d = row["descriptor"]
    sampled = (row.get("forward_context_after_return") or {}).get("previous_sampled_ids")
    if d["produces_output"] and not sampled:
        raise ValueError("native postprocess observation lacks sampled tensor geometry")
    return work_vectors(d["q"], d["history"], d["blocks"], d["state_slots"],
        d["state_fork_srcs"], d["produces_output"],
        d["forward_context"]["prior_sampled_batch_rows"],
        int(sampled["shape"][0]) if sampled else len(d["q"]))


def eligible(row):
    d = row["descriptor"]
    scope = d["forward_context"]["scope"]
    return (d["prefill_rows"] == len(d["q"]) and d["compiled"] is True
            and d["capture_bucket"] is None and scope["speculative_config_absent"] is True
            and scope["num_spec_step"] == 0 and d["midstep_saves_empty"]
            and not any(d["state_maintenance"].values()))


def _nnls(x, y):
    """Reproduce the frozen seven-variable source fit without changing targets."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    best = (float(np.dot(y, y)), np.zeros(x.shape[1]))
    for count in range(1, x.shape[1] + 1):
        for active in combinations(range(x.shape[1]), count):
            values = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            if np.any(values < -1e-12):
                continue
            coefficients = np.zeros(x.shape[1])
            coefficients[list(active)] = np.maximum(values, 0)
            error = float(np.sum((x @ coefficients - y) ** 2))
            if error < best[0]:
                best = error, coefficients
    return best[1].tolist(), int(np.linalg.matrix_rank(x)), best[0]


def validate_source_model(model, rows):
    if (model.get("schema") != MODEL_SCHEMA or model.get("source_only") is not True
            or model.get("heldout_started") is not False or model.get("source_refitted") is not False
            or model.get("source_qualified") is not False or model.get("accepted") is not False
            or model.get("host_return_law_inferred") is not False
            or model.get("whole_forward_relative_limit") != .10
            or model.get("component_microsecond_gate") is not False
            or model.get("signed_prepare_remainders_preserved") is not True
            or model.get("outputless_postprocess_structural_zero") is not True
            or any(row.get("role") != "source" for row in rows)):
        raise ValueError("native A/P work model changes its source-only policy")
    selected = [row for row in rows if eligible(row)]
    if not selected or model.get("observed_rows") != len(rows) or model.get("prefill_fit_rows") != len(selected):
        raise ValueError("native A/P source observation inventory changed")
    for part, names, scales, index in (("prepare", A_FEATURES, A_SCALES, 0),
                                      ("postprocess", P_FEATURES, P_SCALES, 1)):
        groups = {}
        for row in selected:
            if part == "postprocess" and not row["descriptor"]["produces_output"]:
                continue
            vector = tuple(observation_vectors(row)[index])
            groups.setdefault(vector, []).append(row["seconds"][part])
        vectors = list(groups)
        if not vectors:
            raise ValueError("native A/P source model has no observations for " + part)
        x = [[value / scale for value, scale in zip(vector, scales)] for vector in vectors]
        coefficients, rank, error = _nnls(x, [median(groups[v]) for v in vectors])
        fitted = model["models"][part]
        expected_groups = [dict(features=list(v), all_observations=groups[v], median=median(groups[v])) for v in vectors]
        if (fitted.get("features") != list(names) or fitted.get("scales") != list(scales)
                or fitted.get("source_groups") != expected_groups or fitted.get("source_design_rank") != rank
                or rank != len(names) or fitted.get("source_group_count") != len(groups)
                or fitted.get("source_observation_count") != sum(map(len, groups.values()))
                or len(fitted.get("coefficients", ())) != len(names)
                or not np.allclose(fitted["coefficients"], coefficients, rtol=0, atol=1e-12)
                or abs(fitted["source_squared_error"] - error) > 1e-12):
            raise ValueError("native A/P frozen coefficients or original source groups changed: " + part)
    bounds = {}
    vectors = [observation_vectors(row)[0] for row in selected]
    for i, key in enumerate(A_FEATURES[1:], 1):
        bounds[key] = [min(v[i] for v in vectors), max(v[i] for v in vectors)]
    if model.get("observed_feature_bounds") != bounds:
        raise ValueError("native A/P frozen source feature bounds changed")
    return selected


def _source_scope(rows, scope, chains):
    observed = {}
    for row in rows:
        key = row["chain_id"], row["chain_step"]
        if row["chain_id"] not in chains or row.get("normal_return") is not True:
            raise ValueError("native A/P source changes its chain identity or normal return")
        observed.setdefault(key, []).append(row["repetition"])
        seconds = row["seconds"]
        if (set(seconds) != {"prepare", "postprocess", "run_model", "forward"}
                or any(type(v) not in (int, float) or not isfinite(v) for v in seconds.values())
                or any(seconds[k] < 0 for k in ("postprocess", "run_model", "forward"))
                or abs(seconds["prepare"] + seconds["run_model"] + seconds["postprocess"] - seconds["forward"]) > 1e-12):
            raise ValueError("native A/P source changes signed native F/M/P definitions")
        if not eligible(row):
            continue
        d = row["descriptor"]
        n, forward = len(d["q"]), d["forward_context"]
        if (any(forward["scope"].get(k) != v for k, v in scope.items())
                or d["blocks"] != [len(t) for t in d["block_tables"]]
                or d["context"] != [q + h for q, h in zip(d["q"], d["history"])]
                or d["state_rows"] != list(range(n))
                or len(set(d["state_slots"])) != n or any(s < 0 for s in d["state_slots"])
                or any(d.get(k) != [v] * n for k, v in SAMPLING.items())
                or forward.get("deferred_output") is not True
                or forward.get("pending_nonnull_logprob_copies") != 0
                or forward.get("pending_mtp_status_copies") != 0
                or not d["produces_output"] and seconds["postprocess"] != 0):
            raise ValueError("native A/P source changes deployment, allocation, sampling or queue scope")
        if d["produces_output"] and observation_vectors(row)[1][1] != n:
            raise ValueError("native sampled tensor does not witness the declared no-speculation row rule")
    if set(key[0] for key in observed) != chains or any(sorted(reps) != list(range(6)) for reps in observed.values()):
        raise ValueError("native A/P requires six source repetitions of every recorded chain step")


@dataclass(frozen=True)
class NativeAPWorkRegions:
    base: object
    model: dict
    source_handoff_sha256: str
    loaded_inputs: tuple
    allocation: InitVar[object]
    source_qualified: bool = False
    version: str = "native-ap-work/1"
    provenance: str = "Source-only A/P work fit; full-forward composition qualification is separate"

    def __post_init__(self, allocation):
        object.__setattr__(self, "_allocation", allocation)
        allocation.capture_region_context = True

    @property
    def scope(self):
        return self.base.scope

    @property
    def topologies(self):
        return self.base.topologies

    @classmethod
    def load(cls, base, path, sha256, allocation, *, deployment_scope_sha256):
        handoff, identity = load_json(path, role="oracle.native_ap_regions")
        if (identity.sha256 != sha256 or handoff.get("schema") != SCHEMA
                or handoff.get("source_qualified") is not False
                or handoff.get("retained_native_handoff_sha256") != base.source_handoff_sha256
                or handoff.get("scope") != base.scope
                or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256
                or set(handoff["evidence"]) != set(SOURCE_ROLES)):
            raise ValueError("native A/P work handoff changes retained sources or deployment identity")
        data, inputs = {}, {}
        for role in SOURCE_ROLES:
            pin = handoff["evidence"][role]
            data[role], inputs[role] = load_json(str(Path(path).parent / pin["path"]),
                                               role="oracle.native_ap_regions.work." + role)
            if inputs[role].sha256 != pin["sha256"]:
                raise ValueError("native A/P work evidence changed: " + role)
        plan, design, model = data["acquisition_plan"], data["design"], data["model"]
        if (plan["design"]["sha256"] != inputs["design"].sha256
                or plan["request_scope"]["sha256"] != deployment_scope_sha256
                or design.get("target_timing_inputs") != []
                or design.get("source_repetitions") != 6 or design.get("heldout_repetitions") != 6
                or design.get("no_per_forward_fence") is not True
                or model["source_input"]["sha256"] != inputs["source"].sha256
                or data["initial_runtime"].get("inherited_run_model") is not True
                or data["initial_runtime"]["attention_scope"]["native"]["body_flags"] != plan["backend_body_flags"]):
            raise ValueError("native A/P source plan, model or inherited runtime identity changed")
        rows = data["source"]["rows"]
        cohorts = (design["source_repetitions"] + design["heldout_repetitions"]
                   + 2 * design["warmups_per_chain_per_role"])
        if len(rows) * cohorts != design["source_repetitions"] * plan["workload_counts"]["CPU_predicted_forwards"]:
            raise ValueError("native A/P source rows omit part of the CPU-proved chain inventory")
        _source_scope(rows, base.scope, {c["id"] for c in design["chains"]})
        validate_source_model(model, rows)
        return cls(base, model, identity.sha256, (identity, *inputs.values()), allocation)

    def _retained(self, shape):
        if self.base._cell(shape) is not None:
            return True
        inherited = self.base.base
        while inherited is not None:
            if (isinstance(inherited, CachedPrefillRegions) and inherited.final is not None
                    and shape.batch_size == 1 and shape.produces_output
                    and inherited.final.contains(shape.num_scheduled_tokens[0],
                        shape.context_lens[0] - shape.num_scheduled_tokens[0])
                    and inherited._selection(shape) is not None):
                return True
            inherited = getattr(inherited, "base", None)
        return False

    def _vectors(self, shape):
        if self._retained(shape) or not shape.is_prefill or shape.num_prefill_tokens != shape.total_tokens:
            return None
        context = self._allocation.region_context_for(shape)
        q = list(shape.num_scheduled_tokens)
        history = [c - t for c, t in zip(shape.context_lens, q)]
        n, blocks = len(q), context["allocation_blocks"]
        if (not 1 <= n <= 3 or shape.compiled is not True or shape.capture_bucket is not None
                or not shape.topology or any(v != 1 for v in shape.topology.values())
                or any(v != 0 for v in shape.rank_coords.values())
                or any(context.get(k) != v for k, v in self.scope.items())
                or len(blocks) != n or any(t <= 0 or h < 0 or b <= 0 or t + h > 16 * b
                                         for t, h, b in zip(q, history, blocks))):
            raise BindRefusal("native A/P work is outside its prefill deployment/allocation scope")
        slots, sources = context.get("state_slots"), context.get("state_fork_srcs")
        if (slots is None or sources is None or len(slots) != n or len(sources) != n
                or len(set(slots)) != n or any(s < 0 for s in slots)
                or context.get("state_rows") != tuple(range(n))
                or any(src >= 0 and src in slots and src != dst for dst, src in zip(slots, sources))):
            raise BindRefusal("native A/P source does not cover cross-row state aliasing")
        # In this TP1 Qwen/AITER prefill path, pack_rows and slot_mapping
        # copy fixed extents; reused KV IDs change integer addresses only.
        # State aliases remain guarded above. The body reader still receives
        # the actual shared tables and must independently price that work.
        if (context.get("output_state_representation") != "predictive_deferred_batch"
                or context.get("prior_sampled_has_logprobs") is not False
                or any(context.get(k) != (v,) * n for k, v in SAMPLING.items())
                or type(context.get("prior_sampled_batch_rows")) is not int
                or not 0 <= context["prior_sampled_batch_rows"] <= 3):
            raise BindRefusal("native A/P sampling/deferred-output work differs from its sources")
        vectors = work_vectors(q, history, blocks, slots, sources, shape.produces_output,
                               context["prior_sampled_batch_rows"], n)
        for key, value in zip(A_FEATURES[1:], vectors[0][1:]):
            lo, hi = self.model["observed_feature_bounds"][key]
            if not lo <= value <= hi:
                raise BindRefusal("native A/P work exceeds its observed source feature bounds: " + key)
        if shape.produces_output:
            p_groups = self.model["models"]["postprocess"]["source_groups"]
            for i in (1, 2):
                values = [g["features"][i] for g in p_groups]
                if not min(values) <= vectors[1][i] <= max(values):
                    raise BindRefusal("native postprocess work exceeds its observed sampler/queue bounds")
        return vectors

    def refusal(self, shape):
        try:
            vectors = self._vectors(shape)
        except (BindRefusal, ValueError) as exc:
            return str(exc)
        return self.base.refusal(shape) if vectors is None else None

    def breakdown(self, shape):
        why = self.refusal(shape)
        if why is not None:
            raise ValueError("no native A/P source-work prediction: " + why)
        vectors = self._vectors(shape)
        if vectors is None:
            return self.base.breakdown(shape)
        result = {}
        for i, part in enumerate(("prepare", "postprocess")):
            model = self.model["models"][part]
            result["<" + part + ">"] = (0. if i and not shape.produces_output else
                sum(x / scale * coefficient for x, scale, coefficient in
                    zip(vectors[i], model["scales"], model["coefficients"])))
        return result

    def seconds(self, shape):
        return sum(self.breakdown(shape).values())

    def band(self, shape):
        if self._vectors(shape) is None:
            return self.base.band(shape)
        raise ValueError("native A/P work sources provide no calibrated uncertainty band")

    def describe(self):
        return f"{self.version}: {self.source_handoff_sha256}; {self.provenance}; base={self.base.describe()}"
