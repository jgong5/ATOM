"""Canonical snapshot priming remains initialization, outside profile metrics."""
import copy
import hashlib
import json

import pytest
from atom.compass.core.proper_replay import phase_accounting, validate_records
from atom.compass.replay.aiperf_records import partition_raw_records

from .test_proper_replay_contract import phases, row
from .test_proper_shared_native_journal import cli, identity, line


def primed_phases():
    messages = phases(completed=61, cancelled=1, origin=1789574081833039478)
    start, end = messages[0], messages[1]
    start["config"]["total_expected_requests"] = 3
    start["stats"].update(start_ns=1789573927020843247, total_expected_requests=3)
    end["stats"].update(start_ns=1789573927020843247, total_expected_requests=3,
        requests_end_ns=1789574081706572644, final_requests_sent=3,
        final_requests_completed=3)
    messages[-1]["stats"]["requests_end_ns"] = 1789575011834417429
    return messages


def test_actual_three_credit_shape_keeps_the_full_profile_window():
    phase = phase_accounting(primed_phases())
    assert phase["warmup"]["kind"] == "agentic_snapshot_priming"
    assert phase["warmup_counts"]["final_requests_completed"] == 3
    assert phase["counts"]["final_requests_completed"] == 61
    assert phase["counts"]["final_requests_cancelled"] == 1
    assert phase["observed_duration_seconds"] == pytest.approx(930.001377951)
    assert phase["warmup"]["observed_duration_seconds"] == pytest.approx(154.685729397)


@pytest.mark.parametrize("damage", ["undeclared", "incomplete", "cancelled", "error", "late"])
def test_failed_or_mismatched_priming_is_refused(damage):
    messages = primed_phases()
    if damage == "undeclared":
        messages[0]["config"]["total_expected_requests"] = 4
    elif damage == "incomplete":
        messages[1]["stats"]["final_requests_completed"] = 2
    elif damage == "cancelled":
        messages[1]["stats"]["final_requests_cancelled"] = 1
    elif damage == "error":
        messages[1]["stats"]["final_request_errors"] = 1
    else:
        messages[1]["stats"]["requests_end_ns"] = messages[2]["stats"]["start_ns"] + 1
    with pytest.raises(ValueError):
        phase_accounting(messages)


def test_raw_partition_is_lossless_and_never_retags_warmup():
    warmup = {"metadata": {"benchmark_phase": "warmup", "x_request_id": "warm"},
              "payload": {"max_completion_tokens": 1}}
    profile = {"metadata": {"benchmark_phase": "profiling", "x_request_id": "profile"},
               "payload": {"max_completion_tokens": 987}}
    records = [warmup, profile]
    before = copy.deepcopy(records)
    result = partition_raw_records(records)
    assert result == {"warmup": [warmup], "profiling": [profile]}
    assert result["warmup"][0] is warmup and records == before
    with pytest.raises(ValueError, match="duplicate"):
        partition_raw_records([warmup, dict(profile, metadata=dict(profile["metadata"], x_request_id="warm"))])


def test_non_displayable_priming_token_has_no_profiling_latency():
    phase = phase_accounting(primed_phases())
    records = [dict(row(marker=f"warm{index}", start=phase["warmup"]["origin_ns"]),
                    benchmark_phase="warmup", first_visible_ns=None, output_tokens=1)
               for index in range(3)]
    validate_records(records, phase["warmup"], benchmark_phase="warmup")
    with pytest.raises(ValueError, match="different benchmark phase"):
        validate_records(records, phase["warmup"])
    with pytest.raises(ValueError, match="no visible"):
        validate_records([dict(record, benchmark_phase="profiling") for record in records], phase["warmup"])


def test_normalizer_accepts_special_token_priming_but_keeps_profile_visibility_and_caps():
    from atom.compass.replay.aiperf_records import normalize_records

    from .test_aiperf_proper_records import specimen

    raw, options = specimen()
    raw[0]["metadata"]["benchmark_phase"] = "warmup"
    raw[0]["payload"]["max_completion_tokens"] = 1
    raw[0]["responses"].pop(1)  # A special token has no visible content chunk.
    for response in raw[0]["responses"]:
        for packet in response["packets"]:
            value = json.loads(packet["value"])
            if value.get("usage"):
                value["usage"]["completion_tokens"] = 1
                packet["value"] = json.dumps(value)
    normalized, = normalize_records(raw, **options, benchmark_phase="warmup")
    assert normalized["first_visible_ns"] is None
    assert normalized["terminal_sse_ns"] is not None
    assert normalized["output_tokens"] == normalized["max_completion_tokens"] == 1
    with pytest.raises(ValueError, match="source turn output cap"):
        normalize_records(raw, **dict(options, expected_caps={}), benchmark_phase="warmup")

    raw[0]["metadata"]["benchmark_phase"] = "profiling"
    with pytest.raises(ValueError, match="source turn output cap"):
        normalize_records(raw, **options)
    with pytest.raises(ValueError, match="visible"):
        normalize_records(raw, **dict(options, expected_caps={("conv", 0): 1}))


def test_native_journal_excludes_priming_rows_from_profile_hash_and_count(tmp_path):
    path = tmp_path / "steps.jsonl"
    warm = line("warm-root") + line("warm-child")
    profile = line("current") + line("cancelled")
    path.write_bytes(warm + profile)
    record = lambda key: {"engine_seq_id": key, "cancelled": False, "error": None}
    result = cli().native_journal_segment(path, 0, len(warm + profile), [record("current")],
        warmup_records=[record("warm-root"), record("warm-child")],
        cancelled_admissions=[{"seq_id": "cancelled", "aborted": True}], file_identity=identity(path))
    assert result["profile_start_offset"] == len(warm)
    assert result["profile_region_sha256"] == hashlib.sha256(profile).hexdigest()
    assert result["scheduled_steps"] == 2
    assert result["scheduled_request_ids"] == ["cancelled", "current"]
    assert result["warmup"]["scheduled_steps"] == 2
    assert result["warmup"]["region_sha256"] == hashlib.sha256(warm).hexdigest()


def test_native_journal_refuses_priming_after_profile_start(tmp_path):
    path = tmp_path / "steps.jsonl"
    data = line("current") + line("warm")
    path.write_bytes(data)
    record = lambda key: {"engine_seq_id": key, "cancelled": False, "error": None}
    with pytest.raises(ValueError, match="warmup steps occur after"):
        cli().native_journal_segment(path, 0, len(data), [record("current")],
            warmup_records=[record("warm")], file_identity=identity(path))
