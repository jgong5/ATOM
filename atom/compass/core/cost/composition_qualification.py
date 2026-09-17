"""Qualify one frozen source composition against independent native forwards.

The original source records retain their precision warnings and qualification
status. This receipt validates their composition; it never edits their prices.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from statistics import median

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.native_ap_exact import canonical_state_slots
from atom.compass.core.cost.native_ap_work import _source_scope, eligible
from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.templates import NativeStepAllocation, block_sharing_pairs

SCHEMA = "compass.forward_composition_qualification/1"
ROLE_PREFIX = "validation.forward_composition."
HOST_RULE = {"kind": "existing_forward_timeline", "extra_host_seconds": 0,
             "target_end_to_end_timings_used": False}


def host_rule(oracle):
    model = getattr(oracle, "execution_model", None)
    if model is None:
        return HOST_RULE
    return dict(HOST_RULE, compiled_prefill_execution_sha256=model.sha256,
        mapping="M=max(alpha*B,floor); R=min(A+M,max(A+(R_raw-A)*M/B,A+floor))",
        prefill_preparation_fence_required=True, pure_CPU_time_claim=False)


def input_identity(inputs):
    values = [item.as_dict() if hasattr(item, "as_dict") else item for item in inputs]
    return sorted(({"role": v["role"], "sha256": v["sha256"]} for v in values
                   if not v["role"].startswith("validation.")),
                  key=lambda v: (v["role"], v["sha256"]))


def code_identity():
    # Include derivation, dispatch and native scheduling as well as readers;
    # changing an imported helper must not preserve a stale predictor identity.
    root = Path(__file__).parents[3]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py")) if path.is_file()}


def source_selection(options):
    from atom.compass.runtime.cache_region_oracle import source_cost_oracle
    from atom.compass.runtime.source_oracle import _rank_coords

    arguments = inspect.signature(source_cost_oracle).bind(**options)
    arguments.apply_defaults()
    result = dict(arguments.arguments)
    result.update(result.pop("options"))
    for name in ("diagnostic_only", "composition_qualification", "composition_qualification_sha256",
                 "composition_extension", "composition_extension_sha256"):
        result.pop(name, None)
    result["rank_coords"] = {"tp": 0, **_rank_coords(result.get("rank_coords"))}
    return json.loads(json.dumps({key: value for key, value in result.items() if value is not None}))


def predictor_identity(oracle, options, predictions):
    """Call after source fitting and before releasing the heldout phase gate."""
    scopes = [x for x in oracle.compass_loaded_inputs if x.role == "oracle.attention_scope"]
    if len(scopes) != 1 or oracle.seconds_per_launch != 0:
        raise ValueError("complete predictor identity needs one scope and no extra launch charge")
    identity = dict(schema="compass.complete_predictor_identity/1", complete_identity=True,
        target_end_to_end_timings_used=False,
        body_book=dict(loaded_inputs=input_identity(oracle.compass_loaded_inputs),
                       source_selection=source_selection(options)),
        code=code_identity(), deployment_scope=dict(sha256=scopes[0].sha256),
        host_rule=host_rule(oracle), validation_predictions=predictions)
    if getattr(oracle.regions, "initial_postprocess", None) is not None:
        identity["initial_postprocess_component"] = oracle.regions.initial_postprocess["component_identity"]
    return identity


def geometry(descriptor):
    d = descriptor
    return {**{k: d[k] for k in ("q", "history", "blocks", "prefill_continuation", "output_rows", "produces_output",
                               "prefill_rows", "capture_bucket", "compiled", "topology", "rank_coords")},
            "state_alias_pattern": canonical_state_slots(d["state_slots"], d["state_fork_srcs"]),
            "kv_sharing_pairs": [list(v) for v in block_sharing_pairs(d["block_tables"])]}


def offer_observation(allocation, row):
    """Offer the real prefill or decode geometry to the ordinary native binder."""
    d = row["descriptor"]
    context = dict(d["forward_context"]["scope"],
        prefill_continuations=tuple(d["prefill_continuation"]), output_rows=tuple(d["output_rows"]),
        prefix_cache_hit_tokens=tuple(d["prefix_cache_hit_tokens"]),
        output_state_representation="predictive_deferred_batch",
        prior_sampled_batch_rows=d["forward_context"]["prior_sampled_batch_rows"],
        prior_sampled_has_logprobs=False)
    for key in ("temperatures", "top_ks", "top_ps", "return_logprobs", "independent_noise"):
        context[key] = tuple(d[key])
    allocation.offer(NativeStepAllocation(
        rows=list(zip(d["q"], d["context"])), block_tables=d["block_tables"],
        state_slots=d["state_slots"], state_fork_srcs=d["state_fork_srcs"], state_rows=d["state_rows"],
        num_prefill_seqs=d["prefill_rows"], rank_coords=d["rank_coords"], region_context=context))
    return StepShape(tuple(d["q"]), tuple(d["context"]), num_prefill_tokens=sum(d["q"][:d["prefill_rows"]]),
        produces_output=d["produces_output"], compiled=d["compiled"], capture_bucket=d["capture_bucket"],
        topology=d["topology"], rank_coords=d["rank_coords"])


def observed_components(regions, row):
    """Apply the actual composed A/P selector, including retained tiny cells."""
    shape = offer_observation(regions._allocation, row)
    return regions.breakdown(shape)


def request_namespace(start):
    """Numeric Sequence IDs belong to one owned native process invocation."""
    fields = ("execution_plan_sha256", "plan_sha256", "ownership_token_sha256")
    if (any(not isinstance(start.get(key), str) or len(start[key]) != 64 for key in fields)
            or type(start.get("pid")) is not int or start["pid"] <= 0
            or not isinstance(start.get("started_at"), (int, float))):
        raise ValueError("native request cohort lacks its owned process namespace")
    return tuple(start[key] for key in fields) + (start["pid"], start["started_at"])


def validate(path, sha256, *, inputs, options, regions, oracle, extension=None):
    """Re-quote the actual oracle and recompute every independent forward gate."""
    data, receipt = load_json(path, role=ROLE_PREFIX + "receipt")
    loaded = [receipt]

    def read(pin, role):
        value, item = load_json(str(Path(path).parent / pin["path"]), role=ROLE_PREFIX + role)
        if item.sha256 != pin["sha256"]:
            raise ValueError("composition qualification input changed: " + role)
        loaded.append(item)
        return value

    if (receipt.sha256 != sha256 or data.get("schema") != SCHEMA or data.get("passed") is not True
            or data.get("relative_limit") != .10 or data.get("source_refitted") is not False):
        raise ValueError("composition qualification lacks the unchanged whole-forward policy")
    identity = read(data["predictor_identity"], "predictor_identity")
    frozen = read(data["predictor_freeze"], "predictor_freeze")
    if extension is None:
        unchanged_identity = (
            identity["body_book"]["loaded_inputs"] == input_identity(inputs)
            and identity["body_book"]["source_selection"] == source_selection(options)
            and identity.get("code") == code_identity())
    else:
        extension.check_identity(identity, inputs, options, oracle)
        unchanged_identity = True
    if (identity.get("schema") != "compass.complete_predictor_identity/1"
            or identity.get("complete_identity") is not True
            or identity.get("target_end_to_end_timings_used") is not False
            or not unchanged_identity or identity.get("host_rule") != host_rule(oracle)
            or frozen.get("frozen_before_heldout_warmups") is not True
            or frozen.get("source_refitted") is not False
            or frozen["predictor_identity"]["sha256"] != data["predictor_identity"]["sha256"]):
        raise ValueError("composition is not the complete predictor frozen before heldout warmups")
    model_inputs = [i for i in input_identity(inputs) if i["role"] == "oracle.native_ap_regions.work.model"]
    scopes = [i for i in input_identity(inputs) if i["role"] == "oracle.attention_scope"]
    if (len(model_inputs) != 1 or frozen["source_model"]["sha256"] != model_inputs[0]["sha256"]
            or len(scopes) != 1 or identity["deployment_scope"]["sha256"] != scopes[0]["sha256"]):
        raise ValueError("composition changes the frozen A/P source or deployment scope")
    predictions = read(identity["validation_predictions"], "predictions")
    heldout = read(data["heldout"], "heldout")
    complete = read(data["native_complete"], "native_complete")
    closeout = read(data["copy_closeout"], "copy_closeout")
    source = next(i for i in inputs if (i.role if hasattr(i, "role") else i["role"]) == "oracle.native_ap_regions.work.source")
    source_path = source.path if hasattr(source, "path") else source["path"]
    source_sha = source.sha256 if hasattr(source, "sha256") else source["sha256"]
    sources = read(dict(path=source_path, sha256=source_sha), "source_cohort")["rows"]
    rows = heldout["rows"]
    source_start = read(data["source_run"], "source_run")
    heldout_start = read(data["heldout_run"], "heldout_run")
    source_closeout = read(data["source_closeout"], "source_closeout")
    source_namespace, heldout_namespace = request_namespace(source_start), request_namespace(heldout_start)
    source_plans = [i for i in input_identity(inputs) if i["role"] == "oracle.native_ap_regions.work.acquisition_plan"]
    if (len(source_plans) != 1 or source_start["plan_sha256"] != source_plans[0]["sha256"]
            or heldout_start["plan_sha256"] != complete["plan_sha256"]
            or source_start["started_at"] > heldout_start["started_at"]
            or source_closeout["terminal"]["unprofiled_control"]["RUNNER_START.json"]["sha256"] != data["source_run"]["sha256"]
            or closeout["terminal"]["unprofiled_control"]["RUNNER_START.json"]["sha256"] != data["heldout_run"]["sha256"]
            or source_closeout["plan_sha256"] != source_start["execution_plan_sha256"]
            or closeout["plan_sha256"] != heldout_start["execution_plan_sha256"]
            or source_closeout["cleanup"].get("writers_released") is not True
            or source_closeout["collection"].get("copy_complete") is not True):
        raise ValueError("source/heldout process namespaces do not bind the actual acquisition plans")
    source_ids = {(source_namespace, str(i)) for row in sources for i in row["descriptor"]["req_ids"]}
    if (len(rows) != len(sources)
            or {(r["chain_id"], r["chain_step"]) for r in rows} != {(r["chain_id"], r["chain_step"]) for r in sources}
            or any(row.get("role") != "heldout" for row in rows)
            or source_ids.intersection((heldout_namespace, str(i)) for row in rows for i in row["descriptor"]["req_ids"])
            or complete.get("success") is not True or complete.get("engine_closed") is not True
            or closeout.get("exit_code") != 0 or closeout["cleanup"].get("verified") is not True
            or closeout["cleanup"].get("writers_released") is not True
            or closeout["collection"].get("copy_complete") is not True
            or closeout["terminal"]["unprofiled_control"]["native/NATIVE_COMPLETE.json"]["sha256"] != data["native_complete"]["sha256"]):
        raise ValueError("composition heldouts lack independent requests or successful owned closeout")
    _source_scope(rows, regions.scope, {row["chain_id"] for row in sources})
    groups = {}
    for row in rows:
        groups.setdefault((row["chain_id"], row["chain_step"]), []).append(row)
    predicted = {(row["chain_id"], row["chain_step"]): row for row in predictions["rows"]}
    if (predictions.get("complete") is not True or predictions.get("refused") != 0
            or len(predicted) != len(predictions["rows"]) or set(predicted) != set(groups)):
        raise ValueError("composition predictions omit or repeat native chain steps")
    checks, selected, quotes = [], copy.deepcopy(regions), {}
    for key, observations in sorted(groups.items()):
        prediction = predicted[key]
        if any(geometry(row["descriptor"]) != prediction["geometry"] for row in observations):
            raise ValueError("composition heldout changes the frozen predicted geometry")
        all_terms = [observed_components(selected, row) for row in observations]
        seconds = prediction["seconds"]
        effective_model = seconds.get("run_model", seconds["body"])
        if (any(seconds["prepare"] != terms["<prepare>"] or seconds["postprocess"] != terms["<postprocess>"] for terms in all_terms)
                or seconds["body"] < 0
                or abs(seconds["forward"] - (seconds["prepare"] + seconds["postprocess"] + effective_model)) > 1e-12):
            raise ValueError("composition predictions change their source-only component prices")
        signature = json.dumps(prediction["geometry"], sort_keys=True)
        if signature not in quotes:
            shape = offer_observation(oracle.native_allocation, observations[0])
            quote = oracle.estimate(shape)
            if not oracle.require_complete or not oracle.last_coverage.complete:
                raise ValueError("composition re-quote has incomplete body/head coverage")
            quotes[signature] = quote
        quote = quotes[signature]
        body_and_head = quote.breakdown["<body>"] + quote.breakdown.get("<head>", 0.)
        actual_model = getattr(quote, "model_seconds", None)
        actual_model = body_and_head if actual_model is None else actual_model
        if (abs(seconds["body"] - body_and_head) > 1e-10
                or abs(effective_model - actual_model) > 1e-10
                or abs(seconds["forward"] - quote.seconds) > 1e-10):
            raise ValueError("frozen composition quote differs from the actual loaded body/head predictor")
        measured = median(row["seconds"]["forward"] for row in observations)
        error = abs(seconds["forward"] - measured) / measured if measured > 0 else float("inf")
        if not error < .10:
            raise ValueError("composition independent complete-forward error is not under 10%: " + str(key))
        checks.append(dict(chain_id=key[0], chain_step=key[1], relative_error=error))
    oracle.native_allocation.clear()
    initial_checks = []
    if regions.initial_postprocess is not None:
        from atom.compass.core.cost.native_ap_initial import validate_initial_heldouts

        initial_checks = validate_initial_heldouts(regions.initial_postprocess,
            data["initial_branch"], read, identity)
    elif data.get("initial_branch"):
        raise ValueError("unconfigured initial P qualification evidence")
    extension_check = {}
    if extension is not None:
        extension_check = dict(qualification_extension=extension.check_quotes(
            oracle, sources, rows, dict(path=source_path, sha256=source_sha)))
        loaded.extend(extension.loaded_inputs)
    return dict(passed=True, independent_forward_steps=len(checks),
                independent_prefill_steps=sum(eligible(v[0]) for v in groups.values()),
                unique_geometries_requoted=len(quotes), checks=checks,
                initial_postprocess_checks=initial_checks,
                primitive_source_statuses_unchanged=True, **extension_check), tuple(loaded)
