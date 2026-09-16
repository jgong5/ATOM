"""Real CLI entrypoint startup failure retains evidence and restores SIGALRM."""
import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_cli():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "proper_cli_startup", root / "scripts/compass/aiperf_proper_replay.py")
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    return cli


def test_preflight_failure_is_recorded_and_restores_alarm(tmp_path):
    cli = load_cli()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": "compass.aiperf_proper_pair/1", "purpose": "diagnostic", "repeats": 1,
        "record_export": {"export_level": "raw", "export_http_trace": True},
        "modelled_environment": {}, "session_wall_timeout_seconds": 30,
    }))
    output = tmp_path / "modelled.r1.json"
    before = signal.getsignal(signal.SIGALRM)
    assert cli.main(["--plan", str(plan), "--plan-sha256",
                     hashlib.sha256(plan.read_bytes()).hexdigest(), "--side", "modelled",
                     "--out", str(output)]) == 1
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.alarm(0) == 0
    result = json.loads(output.read_text())
    failure = json.loads((output.with_suffix(".raw") / "failure.json").read_text())
    exit_receipt = json.loads((output.with_suffix(".raw") / "exit.json").read_text())
    assert result["complete"] is False and result["accepted"] is False
    assert failure == result["failure"]
    assert failure["type"] == "ValueError"
    assert "provided-history environment" in failure["message"]
    assert exit_receipt["success"] is False


@pytest.mark.parametrize("bad_export", [False, True])
def test_source_refusal_survives_nonfinite_or_broken_partial_export(tmp_path, monkeypatch, bad_export):
    cli = load_cli()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": "compass.aiperf_proper_pair/1", "purpose": "diagnostic", "repeats": 1,
        "record_export": {"export_level": "raw", "export_http_trace": True},
        "modelled_environment": {"AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES": "false"},
        "session_wall_timeout_seconds": 30,
    }))
    source_refusal = ValueError("incomplete: 256 unpriced GEMM occurrences")
    sentinel = object() if bad_export else float("inf")
    source_refusal.replay_result = SimpleNamespace(
        events=[{"grace_period_sec": sentinel}], dispatches=[], records=[], messages=[],
        final_time=106.5, cleanup={}, serving={})

    def fail_startup(environment):
        raise source_refusal

    monkeypatch.setattr(cli.os.environ, "update", fail_startup)
    output = tmp_path / "modelled.r1.json"
    assert cli.main(["--plan", str(plan), "--plan-sha256",
                     hashlib.sha256(plan.read_bytes()).hexdigest(), "--side", "modelled",
                     "--out", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["failure"]["message"] == str(source_refusal)
    assert result["complete"] is False
    directory = output.with_suffix(".raw")
    if bad_export:
        export = json.loads((directory / "partial_export_failure.json").read_text())
        assert export["primary_failure_preserved"] is True
    else:
        partial = json.loads((directory / "controlled_failure.json").read_text())
        assert partial["events"][0]["grace_period_sec"] == {"nonfinite_float": "inf"}
        assert partial["final_time"] == 106.5
    with pytest.raises(ValueError, match="Out of range"):
        cli.write(tmp_path / "successful_metrics.json", {"seconds": float("inf")})


def test_unbounded_phase_grace_exports_without_relaxing_measurements(tmp_path):
    cli = load_cli()
    message = {"message_type": "credit_phase_start",
               "config": {"grace_period_sec": float("inf")},
               "stats": {"seconds": 12.5}}
    exported = cli.export_phase_messages([message])
    cli.write(tmp_path / "phase.json", exported)
    assert json.loads((tmp_path / "phase.json").read_text())[0]["config"] == {
        "grace_period_sec": "Infinity"}
    assert message["config"]["grace_period_sec"] == float("inf")
    assert exported[0]["stats"] == {"seconds": 12.5}
    for index, value in enumerate((float("inf"), float("-inf"), float("nan"))):
        bad = dict(message, stats={"seconds": value})
        with pytest.raises(ValueError, match="Out of range"):
            cli.write(tmp_path / f"bad_measurement{index}.json", cli.export_phase_messages([bad]))
