"""A separately recorded width-four source supplement; old coefficients stay fixed."""
import math
from pathlib import Path
from statistics import median

from atom.compass.core.cost.native_ap_exact import canonical_state_slots
from atom.compass.core.loaded_input import load_json

SCHEMA = "compass.native_ap_width_sources/1"
PREFIX = "oracle.native_ap_regions.width."
PRIMARY = {"sampler_transitions": 1, "allocation4_small": 27,
           "allocation4_query_ceiling": 27, "allocation3_extent": 21}


def primary_rows(rows, role):
    selected = {}
    for row in rows:
        if row["role"] != role:
            raise ValueError("width-four cohort mixes source and heldout roles")
        if row["chain_step"] == PRIMARY.get(row["chain_id"]):
            selected.setdefault(row["chain_id"], []).append(row)
    if set(selected) != set(PRIMARY) or any(sorted(r["repetition"] for r in group) != list(range(6))
                                         for group in selected.values()):
        raise ValueError("width-four controls require six observations of every declared primary")
    return selected


def load_width(pin, *, original_model, original_model_sha256, scope):
    from atom.compass.core.cost.native_ap_work import eligible, observation_vectors, A_FEATURES

    path = Path(pin["path"])
    inputs = []
    def read(reference, role):
        value, loaded = load_json(str(path.parent / reference["path"]), role=PREFIX + role)
        if loaded.sha256 != reference["sha256"]:
            raise ValueError("width-four source input changed: " + role)
        inputs.append(loaded)
        return value
    handoff = read(pin, "handoff")
    if (handoff.get("schema") != SCHEMA or handoff.get("source_only") is not True
            or handoff.get("source_qualified") is not False
            or handoff.get("target_end_to_end_timings_used") is not False
            or handoff.get("original_model_sha256") != original_model_sha256):
        raise ValueError("width-four source changes the original A/P model or source-only scope")
    evidence = {key: read(value, key) for key, value in handoff["evidence"].items()}
    candidate, plan, proof = evidence["candidate"], evidence["plan"], evidence["cpu_proof"]
    if (candidate.get("schema") != "compass.native_width_four_ap_candidate/1"
            or candidate.get("AP_coefficients_unchanged") is not True
            or candidate.get("source_refitted") is not False
            or candidate.get("heldout_started") is not False
            or candidate.get("target_end_to_end_timings_used") is not False
            or candidate.get("original_AP_models") != original_model["models"]
            or candidate["closed_ap_model"]["sha256"] != original_model_sha256
            or candidate["source_input"]["sha256"] != handoff["evidence"]["source"]["sha256"]
            or evidence["waiting"]["source_model"]["sha256"] != handoff["evidence"]["candidate"]["sha256"]
            or evidence["run_start"]["plan_sha256"] != handoff["evidence"]["plan"]["sha256"]
            or plan["cpu_acquisition"]["sha256"] != handoff["evidence"]["cpu_proof"]["sha256"]
            or proof.get("state_pool_entries") != 32 or proof.get("logical_kv_blocks") != 32768
            or evidence["initial_runtime"].get("inherited_run_model") is not True):
        raise ValueError("width-four source lacks its unchanged model and native program proof")
    rows = evidence["source"]["rows"]
    expected = proof["canonical_steps"]
    if set(expected) != set(PRIMARY) or len(rows) != 6 * sum(map(len, expected.values())):
        raise ValueError("width-four source omits native conditioning or primary forwards")
    observed = {}
    for row in rows:
        d = row["descriptor"]
        if row.get("role") != "source" or row.get("normal_return") is not True:
            raise ValueError("width-four source is not an ordinary native source forward")
        key = row["chain_id"], row["chain_step"]
        observed.setdefault(key, []).append(row["repetition"])
        canonical = {k: d[k] for k in ("q", "history", "blocks", "prefill_rows", "produces_output",
                                       "output_rows", "prefill_continuation")}
        canonical.update(state_alias_pattern=canonical_state_slots(d["state_slots"], d["state_fork_srcs"]),
                         prior_sampled_rows=d["forward_context"]["prior_sampled_batch_rows"])
        if (canonical != expected[key[0]][key[1]] or d.get("seq_starts") != [0] * len(d["q"])
                or d["blocks"] != [len(t) for t in d["block_tables"]]
                or any(d["forward_context"]["scope"].get(k) != v for k, v in scope.items())):
            raise ValueError("width-four native work differs from its declared allocation or zero-start scope")
        seconds = row["seconds"]
        if (any(not math.isfinite(v) for v in seconds.values())
                or any(seconds[k] < 0 for k in ("run_model", "postprocess", "forward"))
                or abs(seconds["prepare"] + seconds["run_model"] + seconds["postprocess"] - seconds["forward"]) > 1e-12):
            raise ValueError("width-four source changes the signed native F/M/P definitions")
    if any(sorted(reps) != list(range(6)) for reps in observed.values()):
        raise ValueError("width-four native source repetition inventory differs")
    groups = primary_rows(rows, "source")
    initial = groups["sampler_transitions"]
    values = [r["seconds"]["postprocess"] for r in initial]
    condition = candidate["conditions"]["4"]
    if (set(candidate["conditions"]) != {"4"} or condition["all_observations"] != values
            or condition["postprocess_seconds"] != median(values)
            or any(r["descriptor"]["forward_context"]["prior_sampled_batch_rows"] != 0
                   or r["descriptor"]["q"] != [1] * 4 or r["descriptor"]["history"] != [32] * 4
                   or r["forward_context_after_return"]["previous_sampled_ids"]["shape"] != [4] for r in initial)):
        raise ValueError("initial P4 differs from its six source-only sampler observations")
    vectors = [observation_vectors(r)[0] for r in rows if eligible(r)]
    bounds = {name: [min(original_model["observed_feature_bounds"][name][0], min(v[i] for v in vectors)),
                     max(original_model["observed_feature_bounds"][name][1], max(v[i] for v in vectors))]
              for i, name in enumerate(A_FEATURES[1:], 1)}
    if bounds["rows"] != [1, 4] or bounds["allocated_block_entries"] != [3, 30137]:
        raise ValueError("width-four source changes its bounded work envelope")
    return dict(bounds=bounds, max_rows=4, max_prior_rows=4, initial_P4=condition["postprocess_seconds"],
                source=handoff["evidence"]["source"], candidate=handoff["evidence"]["candidate"],
                plan=handoff["evidence"]["plan"], primary_controls=PRIMARY,
                source_qualified=False), tuple(inputs)
