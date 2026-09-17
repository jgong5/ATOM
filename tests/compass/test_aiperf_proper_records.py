"""Proper pairing retains visible SSE, complete transport time and marked input."""
import json

import pytest

pytest.importorskip("aiperf")
from aiperf.common.models import RawRecordInfo

if "end_perf_ns" not in RawRecordInfo.model_fields:
    pytest.skip("requires the pinned optional raw transport timestamp patch", allow_module_level=True)


def specimen():
    from aiperf.common.config import EndpointConfig, UserConfig
    from aiperf.common.enums import CreditPhase
    from aiperf.common.models import RecordContext, RequestRecord, SSEMessage
    from atom.compass.prefix_workload import token_digest
    from atom.compass.replay.aiperf_records import export_controlled_records
    from .test_aiperf_controlled import _tokenizer

    tokenizer = _tokenizer()
    config = UserConfig(endpoint=EndpointConfig(
        model_names=["fixture"], type="chat", streaming=True, use_server_token_count=True))
    body = {"model": "fixture", "messages": [{"role": "user", "content": "marker token10"}],
            "max_completion_tokens": 2, "ignore_eos": True}
    tokens = tokenizer.encode(tokenizer.apply_chat_template(body["messages"], tokenize=False))
    info = RecordContext(credit_num=1, credit_phase=CreditPhase.PROFILING,
        x_request_id="client-id", x_correlation_id="tree", conversation_id="conv", turn_index=0,
        cache_bust_marker="marker ", cache_bust_target="first_turn_prefix",
        payload_bytes=json.dumps(body).encode())
    record = RequestRecord(timestamp_ns=100_000_000_000, start_perf_ns=1000,
        recv_start_perf_ns=1010, end_perf_ns=1200, request_info=info, status=200,
        responses=[SSEMessage.parse('data: {"object":"chat.completion.chunk","id":"server-id","choices":[{"delta":{"role":"assistant"}}]}', 1020),
                   SSEMessage.parse('data: {"object":"chat.completion.chunk","id":"server-id","choices":[{"delta":{"reasoning_content":"x"}}]}', 1030),
                   SSEMessage.parse('data: {"object":"chat.completion.chunk","id":"server-id","choices":[{"delta":{},"finish_reason":"length"}]}', 1080),
                   SSEMessage.parse('data: ' + json.dumps({"object":"chat.completion.chunk","id":"server-id","choices":[],
                       "usage":{"prompt_tokens":len(tokens),"completion_tokens":2}}), 1090)])
    raw = export_controlled_records([record])
    options = {"user_config": config, "tokenizer": tokenizer, "model_path": "fixture",
               "consumed": {"server-id": {"input_tokens": len(tokens), "prompt_token_sha256": token_digest(tokens)}},
               "expected_caps": {("conv", 0): 2}}
    return raw, options


def test_normalizer_uses_actual_endpoint_visibility_and_transport_end():
    from atom.compass.replay.aiperf_records import normalize_records
    raw, options = specimen()
    row, = normalize_records(raw, **options)
    assert row["first_visible_ns"] == 100_000_000_030
    assert row["terminal_sse_ns"] == 100_000_000_080
    assert row["last_sse_ns"] == row["metadata_request_end_ns"] == 100_000_000_090
    assert row["end_ns"] == 100_000_000_200
    assert row["cache_bust_marker"] == "marker "
    assert row["sampling"]["temperature"] == 1.0


@pytest.mark.parametrize("mutation", ["temperature", "cap", "legacy_end"])
def test_matching_token_hash_cannot_hide_sampling_or_boundary_mismatch(mutation):
    from atom.compass.replay.aiperf_records import normalize_records
    raw, options = specimen()
    if mutation == "temperature":
        raw[0]["payload"]["temperature"] = 0.0
    elif mutation == "cap":
        raw[0]["payload"]["max_completion_tokens"] = 3
    else:
        raw[0].pop("end_perf_ns")
    with pytest.raises(ValueError):
        normalize_records(raw, **options)


def test_cancelled_before_first_sse_retains_actual_native_admission():
    from atom.compass.replay.aiperf_records import normalize_records
    raw, options = specimen()
    raw[0]["responses"] = []
    raw[0]["metadata"]["was_cancelled"] = True
    evidence = options["consumed"]["server-id"]
    options["admissions"] = [{"client_request_id": "client-id", "request_id": "server-id",
        "seq_id": "native-seq-7", "max_completion_tokens": 2, "shared_preprocessing": evidence,
        "aborted": True, "tokenized": True, "engine_enqueued": True}]
    row, = normalize_records(raw, **options)
    assert row["response_id"] is None and row["first_visible_ns"] is None
    assert row["engine_request_id"] == "server-id"
    assert row["engine_seq_id"] == "native-seq-7"
    assert row["native_tokenized_observed"] is True
    assert row["cancelled"] and row["native_enqueue_observed"]
