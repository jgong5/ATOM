"""Read rocprofiler JSON while preserving one documented opaque argument field.

ROCm 7.2 can emit non-UTF-8 bytes for hipModuleLoad's unused ``fname``/``PKc``
argument. Only that field may remain opaque, after independent CSV identity
and timestamp checks. No raw bytes or records are rewritten or discarded.
"""
import csv
import hashlib
import io
import json
from pathlib import Path


def _has_surrogate(value):
    return isinstance(value, str) and any(0xD800 <= ord(char) <= 0xDFFF for char in value)


def _object(pairs):
    value = {}
    for key, child in pairs:
        if _has_surrogate(key) or key in value:
            raise ValueError("opaque or duplicate JSON object key")
        value[key] = child
    return value


def _nonfinite(value):
    raise ValueError("nonfinite JSON number: " + value)


def _opaque_strings(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _opaque_strings(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _opaque_strings(child, (*path, index))
    elif _has_surrogate(value):
        yield path, value


def read_value(path, worker):
    """Return parsed data and an audit; all non-loader strings stay strict."""
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", "surrogateescape")
    if text.encode("utf-8", "surrogateescape") != raw:
        raise ValueError("lossless UTF-8 byte round-trip failed")
    value = json.loads(text, object_pairs_hook=_object, parse_constant=_nonfinite)
    opaque = list(_opaque_strings(value))
    csv_rows, csv_pin, admitted = None, None, []
    for location, argument_value in opaque:
        if (len(location) != 8 or location[0] != "rocprofiler-sdk-tool"
                or type(location[1]) is not int or location[2:4] != ("buffer_records", "hip_api")
                or type(location[4]) is not int or location[5] != "args"
                or type(location[6]) is not int or location[7] != "value"):
            raise ValueError("invalid UTF-8 outside the opaque hipModuleLoad fname field")
        process = value[location[0]][location[1]]
        record = process["buffer_records"]["hip_api"][location[4]]
        argument = record["args"][location[6]]
        pid = process["metadata"]["pid"]
        thread = record.get("thread_id")
        correlation = record.get("correlation_id", {}).get("internal")
        start, end = record.get("start_timestamp"), record.get("end_timestamp")
        if (type(pid) is not int or pid != worker["process"]["pid"]
                or type(record.get("kind")) is not int or record["kind"] != 21
                or type(record.get("operation")) is not int or record["operation"] != 309
                or any(type(v) is not int or v < 0 for v in (thread, correlation, start, end))
                or start > end or argument.get("name") != "fname" or argument.get("type") != "PKc"):
            raise ValueError("opaque value is not the documented worker hipModuleLoad fname/PKc field")
        try:
            semantic_bytes = argument_value.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError as exc:
            raise ValueError("opaque fname contains a non-byte Unicode surrogate") from exc
        if csv_rows is None:
            if not path.name.endswith("results.json"):
                raise ValueError("cannot locate the independent HIP API CSV")
            csv_path = path.with_name(path.name.removesuffix("results.json") + "hip_api_trace.csv")
            csv_bytes = csv_path.read_bytes()
            # A malformed CSV never receives the filename exception.
            csv_text = csv_bytes.decode("utf-8", "strict")
            csv_rows = list(csv.DictReader(io.StringIO(csv_text, newline="")))
            csv_pin = {"path": str(csv_path), "sha256": hashlib.sha256(csv_bytes).hexdigest()}
        matches = [row for row in csv_rows if row.get("Process_Id") == str(pid)
                   and row.get("Thread_Id") == str(thread) and row.get("Correlation_Id") == str(correlation)]
        if (len(matches) != 1 or matches[0].get("Function") != "hipModuleLoad"
                or matches[0].get("Start_Timestamp") != str(start)
                or matches[0].get("End_Timestamp") != str(end)):
            raise ValueError("opaque loader field lacks matching independent CSV identity/timestamps")
        admitted.append({"path": list(location), "semantic_bytes_hex": semantic_bytes.hex(),
            "worker_pid": pid, "thread_id": thread, "correlation_id": correlation,
            "kind": record["kind"], "operation": record["operation"],
            "function": matches[0]["Function"], "argument_name": "fname", "argument_type": "PKc"})
    return value, {"original_file_sha256": hashlib.sha256(raw).hexdigest(),
        "lossless_utf8_roundtrip": True, "opaque_arguments": admitted, "hip_api_csv": csv_pin,
        "raw_bytes_modified": False, "records_removed": 0,
        "argument_text_reconstructed": False}


def raw_json_receipt(path, worker, counts):
    """Keep the existing strict worker and JSON/CSV count admission contract."""
    value, audit = read_value(path, worker)
    processes = value.get("rocprofiler-sdk-tool", [])
    hits = [process for process in processes if process.get("metadata", {}).get("pid") == worker["process"]["pid"]]
    if len(hits) != 1:
        raise ValueError("worker JSON output is absent or ambiguous")
    buffers = hits[0].get("buffer_records", {})
    for name, count in counts.items():
        if type(count) is not int or not isinstance(buffers.get(name), list) or len(buffers[name]) != count:
            raise ValueError("JSON and CSV trace record counts do not agree")
    return {"file": {"path": str(path), "sha256": audit["original_file_sha256"]},
        "worker_pid": worker["process"]["pid"], "record_counts": counts,
        "opaque_argument_audit": audit,
        "preservation": "Original JSON bytes and all records retained; only verified hipModuleLoad fname/PKc text may remain opaque."}
