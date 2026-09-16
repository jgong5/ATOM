"""The real malformed fname bytes are tolerated only in their documented field."""
import json

import pytest

from scripts.compass.rocprof_json import raw_json_receipt, read_value


@pytest.fixture
def trace(tmp_path):
    # These bytes and loader identity come from the preserved q3056s1 trace.
    # The complete profiler trace remains an ignored runtime artifact.
    record = dict(kind=21, operation=309, thread_id=2014624,
        correlation_id={"internal": 4340}, start_timestamp=2652740482265075,
        end_timestamp=2652741609959685,
        args=[dict(type="PPi", name="module", value="0x123"),
              dict(type="PKc", name="fname", value="MALFORMED")])
    value = {"rocprofiler-sdk-tool": [dict(metadata={"pid": 2014624},
        buffer_records=dict(hip_api=[record], kernel_dispatch=[], marker_api=[], memory_copy=[]))]}
    path = tmp_path / "example_results.json"
    csv = tmp_path / "example_hip_api_trace.csv"
    csv.write_text("Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n"
        "hipModuleLoad,2014624,2014624,4340,2652740482265075,2652741609959685\n")
    return path, csv, value, {"process": {"pid": 2014624}}


def write_trace(path, value, replacement=bytes.fromhex("40a8ec16")):
    # The profiler escapes control bytes as JSON while leaving invalid UTF-8
    # bytes literal; preserve that distinction in the small captured fixture.
    field = json.dumps(replacement.decode("utf-8", "surrogateescape"), ensure_ascii=False).encode("utf-8", "surrogateescape")
    raw = json.dumps(value).encode().replace(b'"MALFORMED"', field)
    path.write_bytes(raw)
    return raw


def test_actual_malformed_fname_preserves_bytes_records_and_count_contract(trace):
    path, _, value, worker = trace
    raw = write_trace(path, value)
    parsed, audit = read_value(path, worker)
    assert path.read_bytes() == raw
    assert audit["lossless_utf8_roundtrip"] is True
    assert audit["raw_bytes_modified"] is False and audit["records_removed"] == 0
    assert audit["opaque_arguments"][0]["semantic_bytes_hex"] == "40a8ec16"
    assert parsed["rocprofiler-sdk-tool"][0]["buffer_records"]["hip_api"][0]["args"][1]["value"].encode("utf-8", "surrogateescape") == bytes.fromhex("40a8ec16")
    raw_json_receipt(path, worker, {"hip_api": 1, "kernel_dispatch": 0})
    with pytest.raises(ValueError, match="record counts"):
        raw_json_receipt(path, worker, {"hip_api": 2})


@pytest.mark.parametrize("field", ["kernel_name", "device", "timing", "count", "argument_name", "argument_type", "key"])
def test_invalid_utf8_outside_loader_filename_is_refused(trace, field):
    path, _, value, worker = trace
    process = value["rocprofiler-sdk-tool"][0]
    record = process["buffer_records"]["hip_api"][0]
    record["args"][1]["value"] = "valid.so"
    if field == "kernel_name": process["buffer_records"]["kernel_dispatch"] = [{"kernel_name": "MALFORMED"}]
    elif field == "device": process["metadata"]["device"] = "MALFORMED"
    elif field == "timing": record["start_timestamp"] = "MALFORMED"
    elif field == "count": process["metadata"]["count"] = "MALFORMED"
    elif field == "argument_name": record["args"][1]["name"] = "MALFORMED"
    elif field == "argument_type": record["args"][1]["type"] = "MALFORMED"
    else: record["MALFORMED"] = "value"
    write_trace(path, value)
    with pytest.raises(ValueError): read_value(path, worker)


@pytest.mark.parametrize("field", ["kind", "operation", "name", "type", "pid", "csv_function", "csv_time", "csv_duplicate"])
def test_filename_exception_requires_loader_schema_and_independent_csv(trace, field):
    path, csv, value, worker = trace
    process = value["rocprofiler-sdk-tool"][0]
    record = process["buffer_records"]["hip_api"][0]
    if field in ("kind", "operation"): record[field] += 1
    elif field == "name": record["args"][1]["name"] = "kernelName"
    elif field == "type": record["args"][1]["type"] = "other"
    elif field == "pid": process["metadata"]["pid"] += 1
    elif field == "csv_function": csv.write_text(csv.read_text().replace("hipModuleLoad", "hipLaunchKernel"))
    elif field == "csv_time": csv.write_text(csv.read_text().replace("2652741609959685", "2652741609959686"))
    else: csv.write_text(csv.read_text() + csv.read_text().splitlines()[1] + "\n")
    write_trace(path, value)
    with pytest.raises(ValueError): read_value(path, worker)


def test_clean_trace_and_different_opaque_bytes_do_not_require_digest_or_position(trace):
    path, _, value, worker = trace
    write_trace(path, value, b"valid.so")
    assert read_value(path, worker)[1]["opaque_arguments"] == []
    process = value["rocprofiler-sdk-tool"][0]
    process["buffer_records"]["hip_api"].insert(0, dict(kind=21, operation=1, args=[]))
    write_trace(path, value, bytes.fromhex("e0f76011"))  # prior rpg2 defect, same unused field
    _, audit = read_value(path, worker)
    assert audit["opaque_arguments"][0]["path"][4] == 1
    assert audit["opaque_arguments"][0]["semantic_bytes_hex"] == "e0f76011"


@pytest.mark.parametrize("raw", [b'{"x":0,"x":1}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":"\\ud800"}'])
def test_structurally_invalid_or_nonfinite_json_is_refused(trace, raw):
    path, _, _, worker = trace
    path.write_bytes(raw)
    with pytest.raises(ValueError): read_value(path, worker)
