"""Optional raw-export extension retains real transport boundaries."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("aiperf")
from aiperf.common.enums import CreditPhase
from aiperf.common.models import ParsedResponseRecord, RawRecordInfo, RecordContext, RequestRecord, SSEMessage
from aiperf.post_processors.raw_record_writer_processor import RawRecordWriterProcessor
from aiperf.records.record_processor_service import RecordProcessor

if "end_perf_ns" not in RawRecordInfo.model_fields:
    pytest.skip("requires the pinned optional raw transport timestamp patch", allow_module_level=True)


def test_raw_writer_roundtrip_keeps_last_sse_and_transport_completion_distinct():
    record = RequestRecord(
        timestamp_ns=1_000_000_000_000, start_perf_ns=1000,
        recv_start_perf_ns=1010, end_perf_ns=1200,
        request_info=RecordContext(credit_num=1, credit_phase=CreditPhase.PROFILING,
                                  x_request_id="request1", x_correlation_id="tree1", conversation_id="conversation1",
                                  turn_index=0, payload_bytes=b'{"messages":[]}'),
        responses=[SSEMessage.parse('data: {"choices":[],"usage":{"completion_tokens":1}}', 1100)],
    )
    metadata = RecordProcessor._create_metric_record_metadata(
        SimpleNamespace(service_id="parser"), record, "worker", last_response_perf_ns=1100)
    exported = RawRecordWriterProcessor._build_export_record(
        None, ParsedResponseRecord(request=record, responses=[]), metadata)
    restored = RawRecordInfo.model_validate_json(exported.model_dump_json())
    assert restored.metadata.request_end_ns == record.timestamp_ns + 100
    assert restored.end_perf_ns == 1200
    assert restored.recv_start_perf_ns == 1010
    assert restored.responses[-1].perf_ns == 1100
    assert restored.end_perf_ns - restored.responses[-1].perf_ns == 100

    older = json.loads(exported.model_dump_json())
    older.pop("end_perf_ns")
    older.pop("recv_start_perf_ns")
    legacy = RawRecordInfo.model_validate(older)
    assert legacy.end_perf_ns is None and legacy.recv_start_perf_ns is None
    assert legacy.metadata.request_end_ns == restored.metadata.request_end_ns
