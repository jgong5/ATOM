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
