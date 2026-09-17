"""Fresh complete-forward validation of the four declared width-source controls."""
from pathlib import Path
from statistics import median

from atom.compass.core.cost.composition_qualification import (
    ROLE_PREFIX, code_identity, geometry, host_rule, input_identity,
    offer_observation, request_namespace, source_selection,
)
from atom.compass.core.cost.native_ap_width import PRIMARY, primary_rows
from atom.compass.core.cost.composition_extension import observation_key
from atom.compass.core.loaded_input import load_json

SCHEMA = "compass.native_width_forward_qualification/1"


def validate(data, receipt, *, path, sha256, inputs, options, regions, oracle):
    loaded = [receipt]
    def read(pin, role):
        value, record = load_json(str(Path(path).parent / pin["path"]), role=ROLE_PREFIX + role)
        if record.sha256 != pin["sha256"]:
            raise ValueError("width forward qualification evidence changed: " + role)
        loaded.append(record)
        return value
    if (receipt.sha256 != sha256 or data.get("schema") != SCHEMA
            or data.get("passed") is not True or data.get("source_refitted") is not False
            or data.get("relative_limit") != .10 or data.get("primary_controls") != PRIMARY
            or data.get("old_heldouts_are_regression_only") is not True
            or data.get("final_e2e_proof_required") is not True):
        raise ValueError("width forward qualification changes its declared fresh four-control scope")
    width = getattr(regions, "width_extension", None)
    if width is None:
        raise ValueError("width forward qualification requires its validated source supplement")
    identity = read(data["predictor_identity"], "predictor_identity")
    frozen = read(data["predictor_freeze"], "predictor_freeze")
    if (identity.get("schema") != "compass.complete_predictor_identity/1"
            or identity.get("complete_identity") is not True
            or identity.get("target_end_to_end_timings_used") is not False
            or identity["body_book"]["loaded_inputs"] != input_identity(inputs)
            or identity["body_book"]["source_selection"] != source_selection(options)
            or identity["code"] != code_identity() or identity["host_rule"] != host_rule(oracle)
            or frozen.get("frozen_before_heldout_warmups") is not True
            or frozen.get("source_refitted") is not False
            or frozen["predictor_identity"]["sha256"] != data["predictor_identity"]["sha256"]
            or frozen["source_model"]["sha256"] != width["candidate"]["sha256"]):
        raise ValueError("width controls do not bind the complete predictor frozen before fresh heldouts")
    predictions = read(identity["validation_predictions"], "predictions")
    heldouts = read(data["heldout"], "heldout")
    sources = read(width["source"], "width_source")
    source_run = read(data["source_run"], "source_run")
    heldout_run = read(data["heldout_run"], "heldout_run")
    complete = read(data["native_complete"], "native_complete")
    closeout = read(data["copy_closeout"], "copy_closeout")
    source_copy = read(data["source_copy"], "source_copy")
    original_closeout = read(source_copy["original_closeout"], "source_closeout")
    source_namespace, heldout_namespace = request_namespace(source_run), request_namespace(heldout_run)
    if (source_run["plan_sha256"] != width["plan"]["sha256"]
            or source_run["started_at"] >= heldout_run["started_at"]
            or original_closeout["cleanup"].get("writers_released") is not True
            or source_copy["collection"].get("copy_complete") is not True
            or source_copy["collection"].get("owned_writers_released_before_collection") is not True
            or original_closeout["terminal"]["unprofiled_control"]["RUNNER_START.json"]["sha256"] != data["source_run"]["sha256"]
            or complete.get("success") is not True or complete.get("engine_closed") is not True
            or closeout.get("exit_code") != 0 or closeout["cleanup"].get("verified") is not True
            or closeout["cleanup"].get("writers_released") is not True
            or closeout["collection"].get("copy_complete") is not True
            or closeout["plan_sha256"] != heldout_run["execution_plan_sha256"]
            or complete["plan_sha256"] != heldout_run["plan_sha256"]
            or closeout["terminal"]["unprofiled_control"]["RUNNER_START.json"]["sha256"] != data["heldout_run"]["sha256"]
            or closeout["terminal"]["unprofiled_control"]["native/NATIVE_COMPLETE.json"]["sha256"] != data["native_complete"]["sha256"]):
        raise ValueError("width source/fresh-heldout ownership or closed collection is incomplete")
    source_ids = {(source_namespace, str(i)) for r in sources["rows"] for i in r["descriptor"]["req_ids"]}
    if (len(heldouts["rows"]) != len(sources["rows"])
            or source_ids.intersection((heldout_namespace, str(i)) for r in heldouts["rows"] for i in r["descriptor"]["req_ids"])):
        raise ValueError("width heldouts reuse source requests or omit native forwards")
    groups = primary_rows(heldouts["rows"], "heldout")
    source_groups = primary_rows(sources["rows"], "source")
    predicted = {row["chain_id"]: row for row in predictions["rows"]}
    if (predictions.get("complete") is not True or predictions.get("refused") != 0
            or len(predictions["rows"]) != len(PRIMARY) or set(predicted) != set(PRIMARY)):
        raise ValueError("width predictions omit or duplicate declared controls")
    checks = []
    for name, rows in groups.items():
        prediction = predicted[name]
        if (prediction["chain_step"] != PRIMARY[name]
                or prediction["geometry"] != geometry(source_groups[name][0]["descriptor"])
                or any(observation_key(r) != prediction["observation_key"]
                       or r["descriptor"]["state_rows"] != prediction["state_rows"]
                       for r in [*source_groups[name], *rows])
                or any(geometry(r["descriptor"]) != prediction["geometry"]
                       or r["descriptor"].get("seq_starts") != [0] * len(r["descriptor"]["q"])
                       or r.get("normal_return") is not True for r in rows)):
            raise ValueError("width heldout changes the frozen native control geometry or offered state")
        quote = oracle.estimate(offer_observation(oracle.native_allocation, rows[0]))
        expected = prediction["cost"]
        if not oracle.require_complete or not oracle.last_coverage.complete:
            raise ValueError("width control lacks complete body/head coverage")
        for key in ("operators", "measured", "interpolated", "zero_work", "refused", "sources"):
            if getattr(oracle.last_coverage, key) != prediction["coverage"][key]:
                raise ValueError("width control coverage differs from the frozen predictor")
        for name_part in ("seconds", "preparation_seconds", "model_seconds", "output_ready_seconds"):
            if abs(getattr(quote, name_part) - expected[name_part]) > 1e-10:
                raise ValueError("width control timing/readiness differs from the frozen predictor")
        if (set(quote.breakdown) != set(expected["breakdown"])
                or any(abs(value - expected["breakdown"][key]) > 1e-10 for key, value in quote.breakdown.items())):
            raise ValueError("width control raw body/head or A/P components changed")
        measured = median(r["seconds"]["forward"] for r in rows)
        error = abs(expected["seconds"] - measured) / measured
        if not error < .10:
            raise ValueError("fresh width complete-forward error is not under 10%: " + name)
        checks.append(dict(chain_id=name,chain_step=PRIMARY[name],relative_error=error,
                           all_forward_observations=[r["seconds"]["forward"] for r in rows]))
    oracle.native_allocation.clear()
    return dict(passed=True,independent_forward_steps=4,independent_prefill_steps=4,
        independent_forward_observations=24,checks=checks,primary_controls=PRIMARY,
        conditioning_forward_rows_retained=len(heldouts["rows"])-24,
        old_heldouts_are_regression_only=True,primitive_source_statuses_unchanged=True,
        validation_scope="four cached native controls; fresh E2E proof still required",
        final_e2e_proof_required=True),tuple(loaded)
