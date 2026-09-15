"""Dynamic proper replay evidence is not a fixed issuance calendar."""
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "atom/compass/core/proper_replay.py"
spec = importlib.util.spec_from_file_location("proper_replay_contract", PATH)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def phases(completed=2, cancelled=1, origin=100_000_000_000):
    result = []
    for phase, count, cancel in (("warmup", 0, 0), ("profiling", completed, cancelled)):
        start = origin - 1 if phase == "warmup" else origin
        stats = dict(phase=phase, start_ns=start, total_expected_requests=0 if phase == "warmup" else None)
        result.append(dict(message_type="credit_phase_start", stats=stats,
                           config={"expected_duration_sec": 900 if phase == "profiling" else None}))
        final = dict(stats, requests_end_ns=start + (900_000_000_000 if phase == "profiling" else 0),
                     final_requests_sent=count + cancel, final_requests_completed=count,
                     final_requests_cancelled=cancel, final_request_errors=0, was_cancelled=False)
        result.append(dict(message_type="credit_phase_complete", stats=final))
    return result


def row(marker="marker", turn=0, start=101_000_000_000):
    return dict(request_id=f"{marker}-{turn}", cache_bust_marker=marker,
                conversation_id="root", turn_index=turn, start_ns=start,
                first_visible_ns=start + 1_000_000, end_ns=start + 3_000_000,
                payload_sha256=f"payload-{marker}-{turn}", prompt_token_sha256=f"tokens-{marker}-{turn}",
                input_tokens=11256, output_tokens=2, cancelled=False, error=None)


def test_credit_conservation_uses_observed_counts_and_keeps_cancellation():
    observed = contract.phase_accounting(phases())
    contract.validate_records([row(turn=0), row(turn=1)], observed)
    assert observed["counts"]["final_requests_sent"] == 3
    assert observed["counts"]["final_requests_cancelled"] == 1
    broken = phases()
    broken[-1]["stats"]["final_requests_completed"] = 1
    with pytest.raises(ValueError, match="credits do not close"):
        contract.phase_accounting(broken)


def test_dynamic_count_and_completion_time_differences_are_reported_without_retiming():
    real = [row(turn=0), row(turn=1)]
    modelled = [row(turn=0, start=102_000_000_000)]
    report = contract.pairing_observations(real, modelled)
    assert report["shared_marker_conversation_turns"] == 1
    assert report["real_only"] == 1 and report["modelled_only"] == 0
    assert not report["identical_completion_driven_timestamps_required"]
    assert real[0]["start_ns"] != modelled[0]["start_ns"]


def test_marked_input_differences_remain_visible():
    real, modelled = row(), row()
    modelled.update(input_tokens=11255, prompt_token_sha256="different")
    result = contract.pairing_observations([real], [modelled])
    assert result["marked_input_differences"][0]["real_input_tokens"] == 11256
    assert result["marked_input_differences"][0]["modelled_input_tokens"] == 11255
    assert result["markers_stripped"] is False
    modelled["cache_bust_marker"] = "new-marker"
    result = contract.pairing_observations([real], [modelled])
    assert result["shared_marker_conversation_turns"] == 0


@pytest.mark.parametrize("change", ["missing_phase", "warmup_credit", "abort"])
def test_incomplete_or_nonempty_warmup_is_refused(change):
    messages = phases()
    if change == "missing_phase":
        messages.pop(0)
    elif change == "warmup_credit":
        messages[1]["stats"]["final_requests_sent"] = 1
    else:
        messages.append(dict(message_type="command", command="profile_cancel", reason="warmup_failure"))
    with pytest.raises(ValueError):
        contract.phase_accounting(messages)


def test_live_assistant_history_is_not_silently_accepted():
    prepared = dict(config=dict(scenario="inferencex-agentx-mvp",
        loadgen=dict(concurrency=1, benchmark_duration=900), input=dict(random_seed=42),
        endpoint=dict(type="chat", streaming=True, use_server_token_count=True), benchmark_id="frozen-marker-input"),
        source={"sha256": "source"}, metadata={"conversations": []},
        conversations=[{"context_mode": "deltas_with_responses"}])
    identity = contract.profile_identity(prepared)
    assert identity["benchmark_id"] == "frozen-marker-input" and identity["request_calendar"] is None
    wrong = copy.deepcopy(prepared)
    wrong["conversations"][0]["context_mode"] = "deltas_without_responses"
    with pytest.raises(ValueError, match="provided-history"):
        contract.profile_identity(wrong)


def test_arbitrary_input_hash_verification_does_not_json_decode_jsonl(tmp_path):
    import hashlib
    path = tmp_path / "source.jsonl"
    path.write_bytes(b'{"one":1}\n{"two":2}\n')
    pin = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert contract.checked_path(pin) == path
    with pytest.raises(ValueError):
        contract.read_pinned(pin)
