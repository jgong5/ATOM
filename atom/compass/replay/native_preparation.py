"""Ordinary marked chat preparation outside the native profiling window."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time


GDN_DISPATCH_CLASSES = ("recompute_varlen_bt64", "output_bt16", "output_bt32", "output_bt64", "amd_fused_ge64")


def gdn_dispatch_coverage(rows, *, native_scope, model, require_complete=False):
    """Check the GDN autotuners and AMD branch against request-backed prefill rows.

    Scope is the TP1 BF16 varlen geometry below, not arbitrary compiler or
    cold-first-use coverage. Dummy graph construction is not execution proof.
    """
    record = native_scope.get("record") or {}
    config = record.get("config") or {}
    layers = [layer for layer in (record.get("layers") or {}).values()
              if layer.get("impl") == "atom.model_ops.attention_gdn.GatedDeltaNet"]
    expected = {"num_v_heads": 48, "num_k_heads": 16, "head_k_dim": 128, "head_v_dim": 128}
    if (model != "Qwen/Qwen3.8-27B" or config.get("kv_cache_dtype") != "bf16"
            or config.get("kv_cache_block_size") != 16
            or (native_scope.get("gdn_dispatch") or {}).get("is_amd") is not True
            or native_scope.get("body_flags") != {"FLA_GDN_FIX_BT": False, "USE_DEFAULT_FLA_NORM": 0}
            or len(layers) != 48):
        raise ValueError("GDN dispatch preparation requires its declared TP1 BF16 geometry/flag path")
    for layer in layers:
        attrs = layer.get("impl_attrs") or {}
        view = (record.get("kv_views") or {}).get("layer_" + str(attrs.get("layer_num")), {})
        if (any(attrs.get(key) != value for key, value in expected.items())
                or view.get("v", {}).get("shape", [])[1:] != [48, 128, 128]
                or any(view.get(part, {}).get("dtype") != "torch.bfloat16" for part in ("k", "v"))):
            raise ValueError("GDN dispatch preparation runtime geometry/dtype differs")
    seen = {}
    for row in rows:
        allocation = (row.get("decision") or {}).get("allocation") or {}
        queries = row.get("num_scheduled_tokens") or []
        if (allocation.get("source") != "ScheduledBatch" or allocation.get("is_dummy_run") is not False
                or not row.get("req_ids") or row.get("topology") != {"tp": 1}
                or row.get("rank_coords") != {"tp": 0} or row.get("compiled") is not True
                or row.get("capture_bucket") is not None
                or not queries or any(type(q) is not int or q <= 0 for q in queries)
                or row.get("num_prefill_tokens") != sum(queries)
                or allocation.get("num_prefill_seqs") != len(queries)
                or len(allocation.get("block_tables") or []) != len(queries)
                or any(not table for table in allocation["block_tables"])
                or allocation.get("state_rows") != list(range(len(queries)))
                or len(allocation.get("state_slots") or []) != len(queries)
                or any(type(slot) is not int or not 0 <= slot < 32 for slot in allocation["state_slots"])):
            continue
        tokens = sum(queries)
        tile = min(64, max(16, 1 << (tokens - 1).bit_length()))
        keys = ["output_bt" + str(tile)]
        if tokens < 64:
            keys.append("recompute_varlen_bt64")
        else:
            keys.append("amd_fused_ge64")
        evidence = {key: row[key] for key in ("req_ids", "num_scheduled_tokens", "context_lens", "topology", "rank_coords")}
        evidence.update(actual_tokens=tokens, allocation=allocation)
        for key in keys:
            seen.setdefault(key, evidence)
    missing = [key for key in GDN_DISPATCH_CLASSES if key not in seen]
    if require_complete and missing:
        raise ValueError("native preparation has no actual GDN dispatch evidence for " + ", ".join(missing))
    return {"observed": seen, "missing": missing, "scope": {
        "model": model, "tp": 1, "dtype": "bfloat16", "varlen": True, "is_amd": True,
        "H": 48, "K": 128, "V": 128, "recompute_BT_BK_BV": [64, 64, 64],
        "body_flags": native_scope["body_flags"], "arbitrary_first_use_covered": False},
        "native_scope_sha256": hashlib.sha256(json.dumps(native_scope, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()}


def gdn_dispatch_prompt(tokens):
    """Distinct fixed prompts; native checkpoint ends yield T16/T32/T64."""
    if tokens not in (16, 32, 64):
        raise ValueError("unknown declared GDN dispatch representative")
    return [1024 + tokens * 128 + i for i in range(tokens + 1)]


def _observed_preparation_rows(journal, offset, records):
    ids = {str(row["seq_id"]) for row in records}
    if len(ids) != len(records):
        raise ValueError("preparation lacks distinct native request identities")
    with Path(journal).open("rb") as stream:
        stream.seek(offset)
        raw = stream.read()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    return [row for row in rows if row.get("req_ids") and set(map(str, row["req_ids"])) <= ids]

def marked_payloads(config, conversations):
    """Static preparation/input audit only; this function never issues a profile."""
    from aiperf.common.enums import CacheBustTarget, CreditPhase
    from aiperf.common.models import ModelEndpointInfo, RequestInfo
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType
    from aiperf.timing.strategies.cache_bust import base_trace_id, build_cache_bust_marker
    from aiperf.workers.session_manager import UserSession
    from aiperf.workers.worker import _inject_marker_into_first_user_turn

    model_endpoint = ModelEndpointInfo.from_user_config(config)
    endpoint = plugins.get_class(PluginType.ENDPOINT, model_endpoint.endpoint.type)(model_endpoint=model_endpoint)
    output = []
    for original in conversations:
        conversation = original.model_copy(deep=True)
        session = UserSession(x_correlation_id="preparation", num_turns=len(conversation.turns),
                              conversation=conversation, context_mode=conversation.context_mode)
        for index in range(session.num_turns):
            session.advance_turn(index)
            info = RequestInfo(
                model_endpoint=model_endpoint, turns=session.turn_list, turn_index=index,
                credit_num=index, credit_phase=CreditPhase.PROFILING,
                x_request_id="", x_correlation_id="preparation",
                conversation_id=conversation.session_id, system_message=conversation.system_message,
                user_context_message=conversation.user_context_message)
            info.endpoint_headers = endpoint.get_endpoint_headers(info)
            info.endpoint_params = endpoint.get_endpoint_params(info)
            body = endpoint.format_payload(info)
            marker = build_cache_bust_marker(
                config.benchmark_id, 0, 0, base_trace_id(conversation.session_id),
                target=CacheBustTarget.FIRST_TURN_PREFIX)
            _inject_marker_into_first_user_turn(body["messages"], marker, is_prefix=True)
            output.append(body)
    return output


def native_provenance(base):
    from urllib.request import urlopen
    with urlopen(base + "/compass/provenance", timeout=120) as response:
        return json.loads(response.read())


def check_runtime(plan, provenance, side, opening):
    from atom.compass.core.cache_policy import cache_on_policy, policy_errors
    bad = policy_errors(plan["cache_policy"], cache_on_policy())
    bad += opening.check_server_configuration(
        provenance, {"target_model": plan["model"], "cache_policy": plan["cache_policy"]}, side)
    if (provenance.get("tensor_parallel_size") != 1
            or provenance.get("pipeline_parallel_size") != 1
            or provenance.get("enable_prefix_caching") is not True):
        bad.append("actual runtime is not TP1/PP1/prefix-cache-on")
    compass = provenance.get("compass") or {}
    if compass.get("mode") != ("measure" if side == "real" else "predict"):
        bad.append("actual runtime uses the wrong native/prediction mode")
    if bad:
        raise ValueError("; ".join(bad))


def prepare_native(base, config, conversations, replay, directory, *, step_journal, native_scope, model):
    """Warm declared dispatch classes, then acknowledge an empty request cache."""
    from atom.compass.core.cache_boundary import flush_receipt_errors, reset_receipt_errors
    from atom.compass.prefix_workload import token_digest

    start, began = time.time(), time.monotonic()
    gdn_dispatch_coverage([], native_scope=native_scope, model=model)
    journal = Path(step_journal)
    offset = journal.stat().st_size if journal.exists() else 0
    payloads = []
    for body in marked_payloads(config, conversations):
        body["max_completion_tokens"] = 2
        body.pop("max_tokens", None)
        payloads.append(json.dumps(body).encode())

    async def submit():
        result = []
        for payload in payloads:
            rows, receipt = await replay._submit_requests(
                base, [payload], [0.0], pace=False, timeout=300,
                endpoint="/v1/chat/completions", streaming=True, expected_outputs=[2])
            result.append({"result": rows[0], "submission": receipt})
            if not rows[0]["ok"]:
                raise ValueError("native preparation request failed")
        return result

    observations = asyncio.run(submit())
    records, admissions, flushes = [], [], []

    def collect(expected):
        flush = replay._flush_measurements(base, 120)
        if flush_receipt_errors(flush):
            raise ValueError("native dispatch preparation measurement flush was not acknowledged")
        flushes.append(flush)
        drained = replay._drain_records(base, 120)
        current = drained.get("requests") or []
        if drained.get("error") or len(current) != expected:
            raise ValueError("native preparation records were not completely drained")
        records.extend(current)
        admissions.extend(drained.get("admissions") or [])
        return current

    collect(len(payloads))
    coverage = gdn_dispatch_coverage(_observed_preparation_rows(journal, offset, records),
                                    native_scope=native_scope, model=model)
    before = coverage
    dispatch_requests = []
    for tokens in (16, 32, 64):
        needed = {"output_bt" + str(tokens)}
        if tokens < 64:
            needed.add("recompute_varlen_bt64")
        else:
            needed.add("amd_fused_ge64")
        if not needed.intersection(coverage["missing"]):
            continue
        prompt = gdn_dispatch_prompt(tokens)
        token_sha = token_digest(prompt)
        body = {"model": model, "prompt": prompt, "max_tokens": 2, "temperature": 1.0,
                "top_k": -1, "top_p": 1.0, "ignore_eos": True, "stream": True}
        payload = json.dumps(body).encode()

        async def submit_dispatch():
            return await replay._submit_requests(base, [payload], [0.0], pace=False, timeout=300,
                endpoint="/v1/completions", streaming=True, expected_outputs=[2])

        results, submission = asyncio.run(submit_dispatch())
        if len(results) != 1 or results[0].get("ok") is not True:
            raise ValueError("GDN dispatch preparation request failed")
        response = results[0].get("response") or {}
        current = collect(1)
        consumed = current[0].get("shared_preprocessing") or {}
        if (current[0].get("request_id") != response.get("id")
                or (response.get("usage") or {}).get("prompt_tokens") != len(prompt)
                or (response.get("usage") or {}).get("completion_tokens") != 2
                or consumed.get("input_tokens") != len(prompt)
                or consumed.get("prompt_token_sha256") != token_sha):
            raise ValueError("GDN dispatch preparation lacks exact consumed token identity")
        coverage = gdn_dispatch_coverage(_observed_preparation_rows(journal, offset, records),
                                        native_scope=native_scope, model=model)
        if not needed <= set(coverage["observed"]):
            raise ValueError("submitted representative did not execute its declared GDN dispatch class")
        dispatch_requests.append({"representative_tokens": tokens, "input_tokens": len(prompt),
            "prompt_token_sha256": token_sha, "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "result": results[0], "submission": submission})
    coverage = gdn_dispatch_coverage(_observed_preparation_rows(journal, offset, records),
                                    native_scope=native_scope, model=model, require_complete=True)
    empty = replay._drain_records(base, 120)
    if (empty.get("error") or empty.get("requests") or empty.get("admissions")
            or empty.get("active_streams") != 0 or empty.get("active_api_requests") != 0):
        raise ValueError("native preparation request/stream teardown is incomplete")
    reset = replay._reset_prefix_cache(base, 120)
    if reset_receipt_errors(reset, expected_worker_kind="device_synchronize"):
        raise ValueError("dispatch preparation did not acknowledge empty KV and state indexes")
    with journal.open("rb") as stream:
        stream.seek(offset)
        journal_bytes = stream.read()
    receipt = {"kind": "ordinary_marked_chat_preparation", "output_cap": 2,
               "outside_profile": True, "started_at": start, "ended_at": time.time(),
               "payload_sha256": [hashlib.sha256(p).hexdigest() for p in payloads],
               "observations": observations, "flush": flushes[-1], "flushes": flushes,
               "drained": {"requests": records, "admissions": admissions}, "empty": empty,
               "cache_boundary": reset, "initialization_regime": "declared_gdn_dispatch_classes_warmed",
               "dispatch_coverage_before": before, "dispatch_coverage": coverage,
               "dispatch_requests": dispatch_requests, "step_journal": str(journal),
               "step_journal_start_offset": offset, "compiled_caches_retained": True,
               "step_journal_end_offset": offset + len(journal_bytes),
               "step_journal_region_sha256": hashlib.sha256(journal_bytes).hexdigest(),
               "setup_elapsed_seconds": time.monotonic() - began, "first_use_latency_fitted": False}
    write(directory / "preparation.json", receipt)
    return receipt



def write(path, value):
    """Publish an immutable receipt atomically, including on failure."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
