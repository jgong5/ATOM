"""Exact A/P sources under an explicit complete-forward validation policy.

This is separate from the existing component-precision family policy. Prices
remain source-only A/P medians; no caller-return or host-dispatch law is inferred.
"""
from math import isfinite
from pathlib import Path
from statistics import median

from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.templates import BindRefusal, block_sharing_pairs

SCHEMA = "compass.native_ap_exact_forward_sources/1"
VALIDATION_PREFIX = "validation.native_ap_regions.exact."
POLICY = {"metric": "complete_forward_relative_error", "limit": .10,
          "source_only_components": True, "host_return_law_inferred": False}
EVIDENCE = ("design", "acquisition_plan", "source", "freeze", "heldout", "verdict",
            "body", "native_complete", "initial_runtime", "final_runtime",
            "copy_closeout", "ordinary_holdout", "qualification")
GEOMETRY = ("q", "history", "blocks", "output_rows", "produces_output",
            "prefill_continuation", "state_alias_pattern", "shared_prefix_blocks")
SAMPLING = {"temperatures": 1.0, "top_ks": -1, "top_ps": 1.0,
            "return_logprobs": False, "independent_noise": False}


def canonical_state_slots(destinations, sources):
    names = {}

    def rename(values):
        result = []
        for value in values:
            if type(value) is not int:
                raise ValueError("native state slot is not an integer")
            if value < 0:
                result.append(value)
            else:
                if value not in names:
                    names[value] = len(names)
                result.append(names[value])
        return result

    return dict(destinations=rename(destinations), sources=rename(sources))


def _number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _rows(rows, point, role, scope):
    from atom.compass.core.cost.native_ap_regions import _source_context

    if (len(rows) != 6 or sorted(r["repetition"] for r in rows) != list(range(6))
            or any(r["point_id"] != point["id"] or r["role"] != role
                   or r.get("normal_return") is not True for r in rows)
            or len({tuple(r["descriptor"]["req_ids"]) for r in rows}) != 6):
        raise ValueError("exact native A/P requires six independent observations per role")
    contexts = []
    for row in rows:
        d = row["descriptor"]
        if any(d[key] != point[key] for key in GEOMETRY[:6]):
            raise ValueError("exact native A/P observation changes its declared geometry")
        n = len(point["q"])
        context = _source_context(d)
        if (any(context.get(key) != value for key, value in scope.items())
                or d["blocks"] != [len(table) for table in d["block_tables"]]
                or d["context"] != [q + h for q, h in zip(point["q"], point["history"])]
                or d["state_rows"] != list(range(n))
                or canonical_state_slots(d["state_slots"], d["state_fork_srcs"]) != point["state_alias_pattern"]
                or d.get("prefill_rows") != n or d.get("compiled") is not True or d.get("capture_bucket") is not None
                or not d.get("topology") or any(v != 1 for v in d["topology"].values())
                or any(v != 0 for v in d["rank_coords"].values())
                or any(d.get(key) != [value] * n for key, value in SAMPLING.items())):
            raise ValueError("exact native A/P source scope, state alias or sampling differs")
        pairs = block_sharing_pairs(d["block_tables"])
        if any(prefix != point["shared_prefix_blocks"] or nonprefix for _, _, prefix, nonprefix in pairs):
            raise ValueError("exact native A/P source changes its pairwise KV sharing")
        contexts.append((context["prior_sampled_batch_rows"], pairs))
        seconds = row["seconds"]
        if (set(seconds) != {"prepare", "postprocess", "run_model", "forward"}
                or any(not _number(value) for value in seconds.values())
                or abs(seconds["prepare"] + seconds["postprocess"] + seconds["run_model"] - seconds["forward"]) > 1e-12
                or seconds["postprocess"] != 0):
            raise ValueError("exact native A/P observations change the native F/M/P definitions")
    if any(context != contexts[0] for context in contexts):
        raise ValueError("exact native A/P observations change queue or sharing strata")
    return contexts[0]


def _validate(data, loaded, scope, deployment_scope_sha256):
    design, plan, freeze = (data[key] for key in ("design", "acquisition_plan", "freeze"))
    points = design["points"]
    if (len(points) != 2 or {p["role"] for p in points} != {"source", "heldout"}
            or len({p["id"] for p in points}) != 2
            or design.get("target_timing_inputs") != [] or design.get("changes_existing_refusals") is not False
            or plan.get("collection_only_exact_forward") is not True
            or plan["request_scope"]["sha256"] != deployment_scope_sha256
            or plan["design"]["sha256"] != loaded["design"].sha256
            or plan["fixed_body_prediction"]["sha256"] != loaded["body"].sha256
            or freeze.get("source_only") is not True or freeze.get("heldout_started") is not False
            or freeze.get("target_end_to_end_timings_used") is not False
            or freeze["source_input"]["sha256"] != loaded["source"].sha256
            or freeze["design"]["sha256"] != loaded["design"].sha256
            or freeze["body_prediction"]["sha256"] != loaded["body"].sha256):
        raise ValueError("exact native A/P acquisition or source freeze is not bound to its evidence")
    geometry = {key: points[0][key] for key in GEOMETRY}
    n = len(geometry["q"])
    if (any({key: p[key] for key in GEOMETRY} != geometry for p in points)
            or freeze["geometry"] != geometry or geometry["produces_output"] is not False
            or not n or any(len(geometry[key]) != n for key in ("history", "blocks", "output_rows", "prefill_continuation"))
            or any(type(x) is not int for key in ("q", "history", "blocks") for x in geometry[key])
            or any(q <= 0 or h < 0 or b <= 0 or q + h > 16 * b
                   for q, h, b in zip(geometry["q"], geometry["history"], geometry["blocks"]))
            or any(geometry["output_rows"])
            or scope.get("speculative_config_absent") is not True or scope.get("num_spec_step") != 0):
        raise ValueError("exact native A/P geometry differs or is outside the outputless policy")
    source_point = next(p for p in points if p["role"] == "source")
    heldout_point = next(p for p in points if p["role"] == "heldout")
    sources, heldouts = data["source"]["rows"], data["heldout"]["rows"]
    source_ids = {str(value) for row in sources for value in row["descriptor"]["req_ids"]}
    heldout_ids = {str(value) for row in heldouts for value in row["descriptor"]["req_ids"]}
    if source_ids & heldout_ids:
        raise ValueError("exact native A/P source and heldout requests are not independent")
    queue, pairs = _rows(sources, source_point, "source", scope)
    if _rows(heldouts, heldout_point, "heldout", scope) != (queue, pairs):
        raise ValueError("exact native A/P heldout queue differs from its source")
    components = {part: median(row["seconds"][part] for row in sources) for part in ("prepare", "postprocess")}
    body = data["body"]
    if (body["coverage"].get("complete") is not True or body["coverage"].get("refused") != 0
            or not _number(body["body_only_seconds"]) or freeze["body_seconds"] != body["body_only_seconds"]
            or freeze["components"] != components
            or freeze["source_component_observations"] != {part: [r["seconds"][part] for r in sources] for part in components}):
        raise ValueError("exact native A/P freeze refits source values or lacks a complete body")
    predicted = sum(components.values()) + body["body_only_seconds"]
    values = [row["seconds"]["forward"] for row in heldouts]
    observed = median(values)
    if observed <= 0:
        raise ValueError("exact native forward heldout must have positive duration")
    error = abs(predicted - observed) / observed
    verdict = data["verdict"]
    if (freeze["predicted_forward_seconds"] != predicted or error > POLICY["limit"]
            or verdict.get("source_refitted") is not False or verdict.get("passed") is not True
            or verdict.get("host_return_law_inferred") is not False
            or verdict.get("component_precision_is_not_a_separate_gate") is not True
            or verdict.get("relative_limit") != POLICY["limit"]
            or verdict["predicted_forward_seconds"] != predicted or verdict["observed_forward_median"] != observed
            or verdict["all_forward_observations"] != values or abs(verdict["relative_error"] - error) > 1e-12):
        raise ValueError("exact native complete-forward heldout gate failed or changed")
    ordinary = data["ordinary_holdout"]
    record = ordinary["ordinary_holdout"]["native_record"]
    a = record["decision"]["allocation"]
    ordinary_f = record["seconds"]
    if (ordinary["source_freeze"]["freeze_sha256"] != loaded["freeze"].sha256
            or ordinary.get("source_refitted") is not False or ordinary.get("target_timing_used_to_fit") is not False
            or ordinary.get("target_opened_after_source_freeze") is not True
            or record["num_scheduled_tokens"] != geometry["q"] or a["cached_tokens"] != geometry["history"]
            or [len(t) for t in a["block_tables"]] != geometry["blocks"]
            or canonical_state_slots(a["state_slots"], a["state_fork_srcs"]) != geometry["state_alias_pattern"]
            or block_sharing_pairs(a["block_tables"]) != pairs
            or a["num_prefill_seqs"] != n or a.get("is_dummy_run") is not False
            or record.get("compiled") is not True or record.get("capture_bucket") is not None
            or record["produces_output"] != geometry["produces_output"]
            or not _number(ordinary_f) or ordinary_f <= 0
            or abs(predicted - ordinary_f) / ordinary_f > POLICY["limit"]
            or ordinary["predicted_forward_seconds"] != predicted or ordinary["observed_forward_seconds"] != ordinary_f
            or ordinary.get("forward_comparison_passed") is not True):
        raise ValueError("exact native A/P ordinary-forward holdout differs or failed")
    complete, initial, final = (data[key] for key in ("native_complete", "initial_runtime", "final_runtime"))
    closeout = data["copy_closeout"]
    if (complete.get("success") is not True or complete.get("engine_closed") is not True
            or complete["plan_sha256"] != loaded["acquisition_plan"].sha256
            or initial.get("inherited_run_model") is not True or final.get("inherited_run_model") is not True
            or initial["attention_scope"] != final["attention_scope"]
            or initial["attention_scope"]["native"]["body_flags"] != plan["backend_body_flags"]
            or closeout.get("exit_code") != 0 or closeout["cleanup"].get("verified") is not True
            or closeout["cleanup"].get("writers_released") is not True
            or closeout["collection"].get("copy_complete") is not True
            or closeout["collection"].get("owned_writers_released_before_collection") is not True
            or closeout["terminal"]["unprofiled_control"]["native/NATIVE_COMPLETE.json"]["sha256"]
               != loaded["native_complete"].sha256):
        raise ValueError("exact native A/P runtime or owned source closeout is incomplete")
    qualification = data["qualification"]
    if (qualification.get("schema") != "compass.native_ap_exact_forward_qualification/1"
            or qualification.get("policy") != POLICY or qualification.get("source_qualified") is not True
            or qualification.get("evidence_sha256") != {k: v.sha256 for k, v in loaded.items() if k != "qualification"}):
        raise ValueError("exact native A/P qualification does not bind the explicit forward policy")
    return dict(geometry=geometry, components=components, prior_sampled_batch_rows=queue,
                kv_sharing_pairs=pairs, policy=POLICY,
                queue_invariance="prior sampler buffer is not read or drained by all-prefill P0 without speculation",
                validation={"independent_relative_error": error,
                            "ordinary_relative_error": abs(predicted - ordinary_f) / ordinary_f})


def load_exact(base, handoff, identity, path, *, deployment_scope_sha256):
    if (handoff.get("retained_native_handoff_sha256") != base.source_handoff_sha256
            or handoff.get("scope") != base.scope or handoff.get("source_qualified") is not True
            or handoff["deployment_scope"]["sha256"] != deployment_scope_sha256 or not handoff.get("cells")):
        raise ValueError("exact native A/P handoff changes its retained source or deployment scope")
    cells, inputs = [], [identity]
    for index, cell in enumerate(handoff["cells"]):
        if set(cell["evidence"]) != set(EVIDENCE):
            raise ValueError("exact native A/P evidence inventory differs")
        data, loaded = {}, {}
        for role in EVIDENCE:
            reference = cell["evidence"][role]
            input_role = (f"{VALIDATION_PREFIX}{index}.ordinary_holdout" if role == "ordinary_holdout"
                          else f"oracle.native_ap_regions.exact.{index}.{role}")
            data[role], loaded[role] = load_json(str(Path(path).parent / reference["path"]), role=input_role)
            if loaded[role].sha256 != reference["sha256"]:
                raise ValueError("exact native A/P evidence changed: " + role)
        result = _validate(data, loaded, base.scope, deployment_scope_sha256)
        if any(other["geometry"] == result["geometry"] for other in cells):
            raise ValueError("exact native A/P handoff repeats a geometry")
        cells.append(result)
        inputs.extend(loaded.values())
    return tuple(cells), tuple(inputs)


def query_exact(cells, allocation, shape):
    if (not shape.is_prefill or shape.num_prefill_tokens != shape.total_tokens
            or (shape.batch_size, shape.produces_output) not in
               {(len(c["geometry"]["q"]), c["geometry"]["produces_output"]) for c in cells}):
        return None
    q = list(shape.num_scheduled_tokens)
    history = [c - t for c, t in zip(shape.context_lens, q)]
    context = allocation.region_context_for(shape)
    for cell in cells:
        g = cell["geometry"]
        if g["q"] == q and g["history"] == history and g["produces_output"] == shape.produces_output:
            return "exact", cell, q, history, list(context["allocation_blocks"]), context
    raise BindRefusal("native A/P work is outside its qualified exact geometry")


def exact_refusal(cell, shape, context, scope):
    g, n = cell["geometry"], len(cell["geometry"]["q"])
    if (not shape.topology or any(v != 1 for v in shape.topology.values())
            or any(v != 0 for v in shape.rank_coords.values()) or not shape.compiled or shape.capture_bucket is not None
            or any(context.get(key) != value for key, value in scope.items())):
        return "exact native A/P deployment or compiled path differs"
    if (context.get("output_state_representation") != "predictive_deferred_batch"
            or type(context.get("prior_sampled_batch_rows")) is not int or context["prior_sampled_batch_rows"] < 0
            or context.get("prior_sampled_has_logprobs") is not False):
        return "exact native A/P deferred-output queue differs from its observed source"
    # Native prepare_input_ids returns on prefill before reading prior sampled
    # IDs; pure-middle forward returns before sampler/postprocess and drains no
    # status queue. The native method poison test guards this source-backed
    # invariance. The observed source count remains recorded, but is not work
    # done by this all-prefill P0, non-speculative forward.
    if any(context.get(key) != (value,) * n for key, value in SAMPLING.items()):
        return "exact native A/P sampling path differs"
    if (list(context["allocation_blocks"]) != g["blocks"]
            or list(context.get("prefill_continuations") or ()) != g["prefill_continuation"]
            or list(context.get("output_rows") or ()) != g["output_rows"]
            or context.get("state_rows") != tuple(range(n))
            or canonical_state_slots(context.get("state_slots") or (), context.get("state_fork_srcs") or ()) != g["state_alias_pattern"]
            or [list(row) for row in context.get("kv_sharing_pairs", ())] != cell["kv_sharing_pairs"]):
        return "exact native A/P allocation, continuation or state-sharing selector differs"
    hits = context.get("prefix_cache_hit_tokens")
    if hits is None or len(hits) != n or any(type(x) is not int or not 0 <= x <= h for x, h in zip(hits, g["history"])):
        return "exact native A/P prefix-hit provenance differs"
    return None
