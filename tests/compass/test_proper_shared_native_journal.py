"""A shared server journal must retain exact request and byte attribution per profile."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def cli():
    path = Path(__file__).resolve().parents[2] / "scripts/compass/aiperf_proper_replay.py"
    spec = importlib.util.spec_from_file_location("shared_native_journal_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino]


def line(request):
    return (json.dumps({"req_ids": [request], "seconds": .01}) + "\n").encode()


def test_default_per_run_path_and_acceptance_are_preserved(tmp_path):
    m = cli()
    output = tmp_path / "real.r1.json"
    assert m.native_journal_path({"purpose": "acceptance"}, output) == tmp_path / "real.r1_steps.jsonl"
    shared = str(tmp_path / "shared.jsonl")
    assert m.native_journal_path({"purpose": "diagnostic", "native_step_journal": shared}, output) == Path(shared)
    for purpose, value in [("acceptance", shared), ("diagnostic", "relative.jsonl"), ("diagnostic", "")]:
        with pytest.raises(ValueError, match="absolute diagnostic"):
            m.native_journal_path({"purpose": purpose, "native_step_journal": value}, output)


def test_segment_excludes_previous_and_later_profiles(tmp_path):
    m = cli()
    path = tmp_path / "shared.jsonl"
    before, current, later = line("previous"), line("current") + line("child"), line("later")
    path.write_bytes(before + current + later)
    records = [{"engine_seq_id": key, "error": None, "cancelled": False} for key in ("current", "child")]
    result = m.native_journal_segment(path, len(before), len(before + current), records,
                                     file_identity=identity(path))
    assert result["profile_region_sha256"] == hashlib.sha256(current).hexdigest()
    assert result["scheduled_steps"] == 2
    assert result["scheduled_request_ids"] == ["child", "current"]


@pytest.mark.parametrize("failure", ["foreign_request", "missing_completed", "split_start", "split_end", "replacement", "truncated"])
def test_segment_refuses_wrong_attribution_or_changed_file(tmp_path, failure):
    m = cli()
    path = tmp_path / "shared.jsonl"
    data = line("current")
    path.write_bytes(data)
    start, end, original = 0, len(data), identity(path)
    records = [{"engine_seq_id": "current", "error": None, "cancelled": False}]
    if failure == "foreign_request":
        records[0]["engine_seq_id"] = "other"
    elif failure == "missing_completed":
        records.append({"engine_seq_id": "missing", "error": None, "cancelled": False})
    elif failure == "split_start":
        start = 1
    elif failure == "split_end":
        end -= 1
    elif failure == "replacement":
        original[1] += 1
    else:
        path.write_bytes(data[:-1])
    with pytest.raises(ValueError):
        m.native_journal_segment(path, start, end, records, file_identity=original)


def test_admitted_scheduled_cancellation_without_raw_export_is_not_a_metric_row(tmp_path):
    m = cli()
    raw = [{"metadata": {"x_request_id": "completed", "was_cancelled": False}}]
    admissions = [{"client_request_id": "completed", "seq_id": "1", "aborted": False},
                  {"client_request_id": "cancelled", "seq_id": "2", "aborted": True}]
    phase = {"counts": {"final_requests_cancelled": 1}}
    kept, unrecorded = m.native_record_admissions(raw, admissions, phase, "diagnostic")
    assert kept == admissions[:1] and unrecorded == admissions[1:]
    path = tmp_path / "shared.jsonl"
    path.write_bytes(line("1") + line("2"))
    result = m.native_journal_segment(path, 0, path.stat().st_size,
        [{"engine_seq_id": "1", "cancelled": False, "error": None}],
        file_identity=identity(path), cancelled_admissions=unrecorded)
    assert result["scheduled_request_ids"] == ["1", "2"]
    assert len(raw) == 1 and "output_tokens" not in unrecorded[0]
    for purpose, count, aborted in [("acceptance", 1, True), ("diagnostic", 0, True), ("diagnostic", 1, False)]:
        admissions[1]["aborted"] = aborted
        with pytest.raises(ValueError):
            m.native_record_admissions(raw, admissions, {"counts": {"final_requests_cancelled": count}}, purpose)


def test_summary_permits_only_counted_diagnostic_request_cancellations():
    m = cli()
    error = {"type": "RequestCancellationError", "code": 499, "message": "Request cancelled by external signal"}
    raw = [{"metadata": {"x_request_id": "cancelled", "was_cancelled": True}, "error": error}]
    phase = {"counts": {"final_requests_cancelled": 1}}
    summary = {"was_cancelled": False, "error_summary": [{"error_details": error, "count": 1}]}
    m.check_native_summary(summary, raw, phase, "diagnostic")
    for purpose, altered in [("acceptance", summary), ("diagnostic", dict(summary, was_cancelled=True)),
            ("diagnostic", dict(summary, error_summary=[{"error_details": {"type": "HTTPError"}, "count": 1}])),
            ("diagnostic", dict(summary, error_summary=[{"error_details": error, "count": 2}]))]:
        with pytest.raises(ValueError):
            m.check_native_summary(altered, raw, phase, purpose)
