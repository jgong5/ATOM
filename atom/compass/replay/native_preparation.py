"""Ordinary marked chat preparation outside the native profiling window."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

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


def prepare_native(base, config, conversations, replay, directory):
    """Warm the pinned input families, drain, then perform the normal empty reset."""
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

    start = time.time()
    observations = asyncio.run(submit())
    flush = replay._flush_measurements(base, 120)
    drained = replay._drain_records(base, 120)
    empty = replay._drain_records(base, 120)
    if len(drained.get("requests") or []) != len(payloads) or empty.get("requests"):
        raise ValueError("native preparation records were not completely drained")
    reset = replay._reset_prefix_cache(base, 120)
    receipt = {"kind": "ordinary_marked_chat_preparation", "output_cap": 2,
               "outside_profile": True, "started_at": start, "ended_at": time.time(),
               "payload_sha256": [hashlib.sha256(p).hexdigest() for p in payloads],
               "observations": observations, "flush": flush, "drained": drained, "empty": empty,
               "cache_boundary": reset}
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
