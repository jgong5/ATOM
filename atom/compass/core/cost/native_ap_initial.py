"""Measured first-output P prices for the native empty sampler-buffer state.

These three exact sampler-row controls supplement the closed A/P work model.
They do not refit its coefficients or use the short probe's body as a price.
"""
import hashlib
from pathlib import Path
from statistics import median

from atom.compass.core.cost.native_ap_exact import SAMPLING, _number
from atom.compass.core.loaded_input import LoadedInput, load_json

SCHEMA = "compass.native_initial_postprocess_sources/1"
MODEL_SCHEMA = "compass.native_initial_postprocess_candidate/1"
PREFIX = "oracle.native_ap_regions.initial."
PROOF_GEOMETRY = dict(drafter=False, independent_noise=False, logits_dtype="torch.bfloat16",
    logits_stride=[248320, 1], logprobs=False, pp=1, prior_rows=0, speculation=False,
    temperature=1, top_k=-1, top_p=1, tp=1, vocab_size=248320)


def first_output_groups(rows, role, scope):
    groups = {}
    for row in rows:
        if row.get("role") != role:
            raise ValueError("initial P evidence mixes source and heldout roles")
        d = row["descriptor"]
        n = len(d["q"])
        if not d["produces_output"] or d["prefill_rows"] != n:
            continue
        before = d["forward_context"]
        after = row["forward_context_after_return"]
        seconds = row["seconds"]
        if (n not in (1, 2, 3) or d["q"] != [1] * n or d["history"] != [32] * n
                or d["blocks"] != [3] * n or d["output_rows"] != [True] * n
                or any(before["scope"].get(k) != v for k, v in scope.items())
                or any(d.get(k) != [v] * n for k, v in SAMPLING.items())
                or before.get("deferred_output") is not True
                or before.get("prior_sampled_batch_rows") != 0
                or before.get("pending_token_copies") != 0 or before.get("pending_logprob_entries") != 0
                or before.get("pending_mtp_status_copies") != 0 or before.get("previous_sampled_ids") is not None
                or after["previous_sampled_ids"] != dict(shape=[n], dtype="torch.int32", device="cuda:0")
                or row.get("normal_return") is not True or row["returned_request_ids"] != []
                or row["returned_deferred_output"] is not True
                or not _number(seconds["postprocess"])
                or abs(seconds["prepare"] + seconds["run_model"] + seconds["postprocess"] - seconds["forward"]) > 1e-12):
            raise ValueError("initial P observation changes its native sampler/queue work")
        groups.setdefault(n, []).append(row)
    if (len(rows) != 54 or set(groups) != {1, 2, 3}
            or any(sorted(r["repetition"] for r in group) != list(range(6)) for group in groups.values())):
        raise ValueError("initial P source requires six complete native controls per sampler row count")
    return groups


def load_initial(pin, *, scope, closed_model_sha256):
    path = Path(pin["path"])
    inputs = []
    def read(reference, role):
        data, loaded = load_json(str(path.parent / reference["path"]), role=PREFIX + role)
        if loaded.sha256 != reference["sha256"]:
            raise ValueError("initial P input changed: " + role)
        inputs.append(loaded)
        return data
    handoff = read(pin, "handoff")
    if (handoff.get("schema") != SCHEMA or handoff.get("source_qualified") is not False
            or handoff.get("scope") != scope or set(handoff["evidence"]) != {"plan", "model", "source", "runtime", "proof", "component_identity"}):
        raise ValueError("initial P source changes its source-only scope")
    data = {role: read(reference, role) for role, reference in handoff["evidence"].items()}
    proof, model, plan = data["proof"], data["model"], data["plan"]
    component = data["component_identity"]
    if (proof.get("schema") != "compass.initial_postprocess_work_dependency_proof/1"
            or proof.get("geometry") != PROOF_GEOMETRY
            or proof.get("all_output_and_mixed_output_same_P_work_when_any_output_and_sampler_rows_match") is not True
            or proof.get("long_query_history_block_tables_read_by_P") is not False
            or proof.get("current_output_mask_read_by_P") is not False
            or proof.get("warmup_timing_reused") is not False
            or plan["initial_postprocess_work_proof"]["sha256"] != handoff["evidence"]["proof"]["sha256"]
            or model.get("schema") != MODEL_SCHEMA or model.get("source_only") is not True
            or model.get("source_refitted") is not False or model.get("heldout_started") is not False
            or model.get("source_qualified") is not False or model.get("target_end_to_end_timings_used") is not False
            or model["closed_ap_model"]["sha256"] != closed_model_sha256
            or plan["closed_ap_model"]["sha256"] != closed_model_sha256
            or model["source_input"]["sha256"] != handoff["evidence"]["source"]["sha256"]
            or data["runtime"].get("inherited_run_model") is not True
            or data["runtime"]["attention_scope"]["native"]["body_flags"] != plan["backend_body_flags"]):
        raise ValueError("initial P source lacks its unchanged model, native scope or work dependency proof")
    if (component.get("schema") != "compass.initial_postprocess_component_identity/1"
            or component.get("validation_scope") != "initial_postprocess_only"
            or component.get("source_only") is not True or component.get("source_refitted") is not False
            or component.get("target_end_to_end_timings_used") is not False
            or component.get("pricing_rule") != "exact_source_median_by_sampler_rows_at_prior_zero"
            or component["model"]["sha256"] != handoff["evidence"]["model"]["sha256"]
            or component["work_proof"]["sha256"] != handoff["evidence"]["proof"]["sha256"]
            or component["closed_ap_model"]["sha256"] != closed_model_sha256):
        raise ValueError("initial P component identity changes its fixed source rule")
    for index, source in enumerate(proof["sources"]):
        requested = str(path.parent / source["path"])
        actual = Path(requested).resolve();raw = actual.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != source["sha256"]:
            raise ValueError("initial P work-proof source changed")
        inputs.append(LoadedInput(PREFIX + "proof_source." + str(index), requested, str(actual), False, sha, len(raw)))
    groups = first_output_groups(data["source"]["rows"], "source", scope)
    for n, rows in groups.items():
        condition = model["conditions"][str(n)]
        values = [r["seconds"]["postprocess"] for r in rows]
        if (condition.get("sampler_rows") != n or condition.get("prior_sampled_rows") != 0
                or condition.get("postprocess_seconds") != median(values) or condition.get("all_observations") != values):
            raise ValueError("initial P price differs from its complete source-only median")
    source_input = next(item for item in inputs if item.role == PREFIX + "source")
    return dict(conditions=model["conditions"], model_sha256=handoff["evidence"]["model"]["sha256"],
                plan_sha256=handoff["evidence"]["plan"]["sha256"], source=dict(path=source_input.path,sha256=source_input.sha256),
                component_identity=dict(path=str(path.parent / handoff["evidence"]["component_identity"]["path"]),
                                        sha256=handoff["evidence"]["component_identity"]["sha256"]),
                scope=scope, source_qualified=False), tuple(inputs)


def validate_initial_heldouts(initial, evidence, read, predictor_identity):
    """P-only validation is justified by the pinned dependency proof above."""
    heldout = read(evidence["heldout"], "initial.heldout")
    complete = read(evidence["native_complete"], "initial.native_complete")
    closeout = read(evidence["copy_closeout"], "initial.copy_closeout")
    freeze = read(evidence["predictor_freeze"], "initial.predictor_freeze")
    declaration = read(freeze["predictor_identity"], "initial.phase_declaration")
    source = read(initial["source"], "initial.source_cohort")
    if (declaration.get("validation_scope") != "initial_postprocess_only"
            or declaration.get("full_forward_qualified") is not False
            or declaration["initial_postprocess_component"]["sha256"] != initial["component_identity"]["sha256"]
            or predictor_identity["initial_postprocess_component"]["sha256"] != initial["component_identity"]["sha256"]
            or freeze["source_model"]["sha256"] != initial["model_sha256"]
            or freeze.get("frozen_before_heldout_warmups") is not True or freeze.get("source_refitted") is not False
            or complete.get("success") is not True or complete.get("engine_closed") is not True
            or complete["plan_sha256"] != initial["plan_sha256"]
            or closeout["cleanup"].get("writers_released") is not True
            or closeout["collection"].get("copy_complete") is not True
            or closeout["terminal"]["unprofiled_control"]["native/NATIVE_COMPLETE.json"]["sha256"] != evidence["native_complete"]["sha256"]):
        raise ValueError("initial P independent validation lacks frozen identity or native closeout")
    if closeout.get("exit_code") != 0 or closeout["cleanup"].get("verified") is not True:
        _late_foreign_processes(closeout, evidence["window"], heldout["rows"], read)
    source_ids = {str(i) for r in source["rows"] for i in r["descriptor"]["req_ids"]}
    if source_ids.intersection(str(i) for r in heldout["rows"] for i in r["descriptor"]["req_ids"]):
        raise ValueError("initial P source and heldout requests are not independent")
    checks = []
    for n, rows in first_output_groups(heldout["rows"], "heldout", initial["scope"]).items():
        observed = median(r["seconds"]["postprocess"] for r in rows)
        predicted = initial["conditions"][str(n)]["postprocess_seconds"]
        error = abs(predicted-observed)/observed if observed > 0 else float("inf")
        if not error < .10:
            raise ValueError("initial P heldout median differs by at least 10%")
        checks.append(dict(sampler_rows=n, relative_error=error))
    return checks


def _late_foreign_processes(closeout, evidence, rows, read):
    """Keep a failed post-cleanup audit; verify its new process started later."""
    clock = read(evidence["clock"], "initial.window.clock")
    boundary = read(evidence["workload_boundary"], "initial.window.boundary")
    final = read(evidence["final_runtime"], "initial.window.final_runtime")
    cleanup = closeout["cleanup"]
    account = cleanup["gpu_after"]["host_accounting"]
    runner = closeout["terminal"]["unprofiled_control"]["RUNNER_EXIT.json"]["value"]
    if (runner.get("exit_code") != 0 or runner.get("command_exit_code") != 0
            or cleanup["ports"].get("free") is not True
            or any(v.get("live") or v.get("coverage_errors") for v in cleanup["owners"].values())
            or not cleanup.get("errors") or any(e.get("operation") != "post-cleanup resources" for e in cleanup["errors"])
            or account.get("errors") or clock["host_boot_id"] != account["host_boot_id"]
            or abs(clock["boot_after"] - clock["monotonic"]) > .01
            or clock["boot_after"] - clock["boot_before"] > .01
            or final["barrier"].get("acknowledged") is not True
            or final["barrier"].get("kind") != "device_synchronize"
            or final["barrier"]["measurement_journal"]["pending_steps_after"] != 0
            or closeout["terminal"]["unprofiled_control"]["RUNNER_WORKLOAD_BOUNDARY.json"]["sha256"] != evidence["workload_boundary"]["sha256"]):
        raise ValueError("initial P post-cleanup failure cannot be confined to later foreign activity")
    last = max(rows, key=lambda r: r["engine_core_forward_call"]["started_ns"])
    offset = clock["wall"] - clock["boot_after"]
    observed_offset = last["native_journal_boundaries"]["started_at"] - last["engine_core_forward_call"]["started_ns"] / 1e9
    if abs(offset - observed_offset) > .01:
        raise ValueError("initial P clock correlation differs from the measured native journal")
    foreign = [r for r in account["rows"] if any(r["selected_vram_bytes"])
               or any(q["gpu_id"] == account["gpu_id"] for q in r["queues_before"])]
    if not foreign or any(r["start_ticks"][0] != r["start_ticks"][1]
            or r["start_ticks"][0] / clock["clock_ticks_per_second"] + offset <= boundary["observed_at"] + .01
            for r in foreign):
        raise ValueError("observed foreign process may overlap the initial P measurement window")
