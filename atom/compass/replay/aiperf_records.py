"""Normalize ordinary raw exports and controlled records without changing clocks."""
import json
from types import SimpleNamespace

from atom.compass.core.proper_replay import content_sha256


def export_controlled_records(records):
    """Use AIPerf's existing metadata and raw-record builders on actual records."""
    from aiperf.common.models import ParsedResponseRecord
    from aiperf.post_processors.raw_record_writer_processor import RawRecordWriterProcessor
    from aiperf.records.record_processor_service import RecordProcessor

    output = []
    for record in records:
        metadata = RecordProcessor._create_metric_record_metadata(
            SimpleNamespace(service_id="controlled-record-export"), record, "controlled-worker",
            last_response_perf_ns=record.responses[-1].perf_ns if record.responses else None)
        raw = RawRecordWriterProcessor._build_export_record(
            None, ParsedResponseRecord(request=record, responses=[]), metadata)
        exported = raw.model_dump(mode="json", exclude_none=True)
        if record.request_info.payload_bytes is not None:
            exported["payload"] = json.loads(record.request_info.payload_bytes)
        output.append(exported)
    return output


def partition_raw_records(raw_records):
    """Retain canonical priming records separately from profiling measurements."""
    phases = {"warmup": [], "profiling": []}
    seen = set()
    for row in raw_records:
        metadata = row["metadata"]
        phase, request_id = metadata.get("benchmark_phase"), metadata.get("x_request_id")
        if phase not in phases or not request_id or request_id in seen:
            raise ValueError("proper raw records have an unknown phase or duplicate request identity")
        seen.add(request_id)
        phases[phase].append(row)
    return phases


def normalize_records(raw_records, *, user_config, tokenizer, model_path,
                      default_chat_template_kwargs=None, consumed, expected_caps, admissions=(),
                      benchmark_phase="profiling"):
    """Keep last packet, first visible content and transport completion separate."""
    from aiperf.common.models import ModelEndpointInfo, RawRecordInfo
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType
    from atom.compass.prefix_workload import token_digest
    from atom.entrypoints.openai.chat_encoders import apply_chat_template, load_custom_message_encoder

    if benchmark_phase not in ("profiling", "warmup"):
        raise ValueError("unsupported proper record phase")
    endpoint_info = ModelEndpointInfo.from_user_config(user_config)
    endpoint = plugins.get_class(PluginType.ENDPOINT, endpoint_info.endpoint.type)(
        model_endpoint=endpoint_info)
    encoder = load_custom_message_encoder(model_path)
    rows = []
    admitted = {}
    for entry in admissions:
        key = entry["client_request_id"]
        if not key or key in admitted:
            raise ValueError("native admissions lack unique actual client request headers")
        admitted[key] = entry
    for value in raw_records:
        raw = RawRecordInfo.model_validate(value)
        meta = raw.metadata
        if str(meta.benchmark_phase) != benchmark_phase:
            raise ValueError("proper replay record belongs to a different benchmark phase")
        if raw.end_perf_ns is None:
            raise ValueError("raw export lacks its actual transport-completion timestamp")
        start, perf = meta.request_start_ns, raw.start_perf_ns
        to_wall = lambda stamp: start + stamp - perf if stamp is not None else None
        end = to_wall(raw.end_perf_ns)
        first = terminal = last = None
        usage, response_ids = {}, set()
        for message in raw.responses:
            stamp = to_wall(message.perf_ns)
            last = stamp
            parsed = endpoint.parse_response(message)
            if parsed is not None and parsed.data is not None and first is None:
                first = stamp
            body = message.get_json() or {}
            if body.get("id"):
                response_ids.add(body["id"])
            if body.get("usage"):
                usage.update(body["usage"])
            if any(choice.get("finish_reason") is not None for choice in body.get("choices", [])):
                terminal = stamp
        if len(response_ids) > 1:
            raise ValueError("one raw request contains different server response identities")
        response_id = next(iter(response_ids), None)
        if not raw.payload:
            raise ValueError("proper replay record lost its actual marked payload")
        sampling = {key: raw.payload.get(key, default) for key, default in (
            ("temperature", 1.0), ("top_p", 1.0), ("top_k", -1), ("ignore_eos", False))}
        if (sampling != {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "ignore_eos": True}
                or raw.payload.get("stop") or raw.payload.get("stop_token_ids")):
            raise ValueError("proper replay changed sampling or enabled content-dependent stopping")
        cap = raw.payload.get("max_completion_tokens", raw.payload.get("max_tokens"))
        source_cap = expected_caps.get((meta.conversation_id, meta.turn_index))
        expected_cap = 1 if benchmark_phase == "warmup" and source_cap is not None else source_cap
        if cap != expected_cap:
            raise ValueError("proper replay changed the source turn output cap")
        kwargs = dict(default_chat_template_kwargs or {})
        kwargs.update(raw.payload.get("chat_template_kwargs") or {})
        rendered = apply_chat_template(tokenizer, encoder, raw.payload["messages"],
                                       tools=raw.payload.get("tools"), **kwargs)
        tokens = tokenizer.encode(rendered)
        prompt_sha = token_digest(tokens)
        admission = admitted.get(meta.x_request_id)
        engine_request_id, seq_id = response_id, None
        observed = consumed.get(response_id)
        if admission is not None:
            engine_request_id, seq_id = admission["request_id"], admission["seq_id"]
            if (response_id is not None and response_id != engine_request_id
                    or admission["max_completion_tokens"] != cap):
                raise ValueError("native admission identity/cap disagrees with the raw request")
            observed = admission["shared_preprocessing"]
        failed = raw.error is not None or meta.was_cancelled
        if observed is not None and (observed["prompt_token_sha256"] != prompt_sha
                                     or observed["input_tokens"] != len(tokens)):
            raise ValueError("marked rendered prompt differs from actual engine consumption")
        if not failed:
            if observed is None or usage.get("prompt_tokens") != len(tokens):
                raise ValueError("marked rendered prompt differs from actual engine consumption")
            if usage.get("completion_tokens") != cap:
                raise ValueError("successful native request did not produce its unchanged source cap")
            if ((benchmark_phase == "profiling" and first is None)
                    or terminal is None or "completion_tokens" not in usage):
                raise ValueError("successful raw record lacks visible content/terminal/usage evidence")
        marker = raw.cache_bust_marker
        if not marker:
            raise ValueError("proper replay record has no ordinary cache-bust marker")
        rows.append({
            "benchmark_phase": benchmark_phase,
            "request_id": meta.x_request_id, "response_id": response_id,
            "engine_request_id": engine_request_id, "engine_seq_id": seq_id,
            "native_tokenized_observed": admission is not None,
            "native_enqueue_observed": admission is not None and admission.get("engine_enqueued") is True,
            "conversation_id": meta.conversation_id, "turn_index": meta.turn_index,
            "source_trace_id": meta.source_trace_id, "source_outer_idx": meta.source_outer_idx,
            "source_inner_idx": meta.source_inner_idx, "source_kind": meta.source_kind,
            "root_correlation_id": meta.root_correlation_id,
            "parent_correlation_id": meta.parent_correlation_id, "agent_depth": meta.agent_depth,
            "cache_bust_marker": marker, "cache_bust_target": str(raw.cache_bust_target),
            "start_ns": start, "first_visible_ns": first, "terminal_sse_ns": terminal,
            "last_sse_ns": last, "end_ns": end,
            "metadata_request_end_ns": meta.request_end_ns,
            "recv_start_ns": to_wall(raw.recv_start_perf_ns),
            "credit_issued_ns": meta.credit_issued_ns,
            "input_tokens": len(tokens), "prompt_token_sha256": prompt_sha,
            "payload_sha256": content_sha256(raw.payload),
            "payload_digest_basis": "canonical JSON of the actual marked payload",
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "sampling": sampling, "max_completion_tokens": cap,
            "cancelled": meta.was_cancelled, "cancellation_time_ns": meta.cancellation_time_ns,
            "error": raw.error.model_dump(mode="json") if raw.error is not None else None,
            "raw_record_sha256": content_sha256(value),
            "transport_completion_basis": "original RequestRecord.end_perf_ns; not last SSE",
        })
    if admitted.keys() - {row["request_id"] for row in rows}:
        raise ValueError("native admissions are missing actual AIPerf raw records")
    return rows
