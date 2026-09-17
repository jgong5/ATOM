"""Zero output remains a completed event without inventing token timestamps."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "zero_output_compare", Path(__file__).resolve().parents[2]
    / "scripts/compass/compare.py")
compare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)


def saved_run(tmp_path, outputs=(0, 1, 3), name="real"):
    rows, records, results = [], [], []
    for i, count in enumerate(outputs):
        arrival = 10.0 + i
        first = arrival + 0.5 if count else None
        finish = arrival + 1.0 + count * 0.5 if count else 20.0
        rows.append({"arrival_s": float(i), "input_tokens": 128,
                     "output_tokens": count})
        records.append({"request_id": str(i), "arrive_time": arrival,
                        "first_token_time": first, "finish_time": finish,
                        "ttft": first - arrival if first is not None else None,
                        "latency": finish - arrival})
        results.append({"index": i, "ok": True, "response": {
            "id": str(i), "usage": {"completion_tokens": count}}})
    blob = {"run": {"requests": len(rows), "failed": 0,
                    "prompt_lengths": "passed", "server_revision": "fixture",
                    "model_revision": "fixture", "trace_sha256": "same",
                    "model": "fixture", "time_scale": 1.0},
            "workload": rows, "results": results,
            "engine": {"clock": "wall", "records": records}}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(blob))
    return path, compare.load_run(str(path), name)


def test_mixed_metrics_keep_zero_event_in_latency_and_throughput_window(tmp_path):
    _, run = saved_run(tmp_path)
    assert compare.check_run(run, expect_requests=3) == []
    report = compare.compare(run, run)
    assert report["requests"] == 3
    for metric, indices in [("ttft", [1, 2]), ("tpot", [2]), ("latency", [0, 1, 2])]:
        assert report["metrics"][metric]["request_indices"] == indices
        assert report["metrics"][metric]["real"]["n"] == len(indices)
    throughput = report["metrics"]["throughput_tok_s"]
    assert throughput["output_tokens"] == 4
    assert throughput["real_window_s"] == 10.0  # the zero event finishes last
    assert throughput["real"] == 0.4


def test_zero_only_cli_reports_undefined_token_metrics(tmp_path, capsys):
    real, _ = saved_run(tmp_path, (0,), "real")
    modelled, _ = saved_run(tmp_path, (0,), "modelled")
    summary = tmp_path / "comparison.json"
    assert compare.main(["--real", str(real), "--modelled", str(modelled),
                         "--expect-requests", "1", "--summary-out", str(summary)]) == 0
    assert "undefined (zero reference throughput)" in capsys.readouterr().out
    report = json.loads(summary.read_text())
    assert report["metrics"]["ttft"]["real"] == {"n": 0}
    assert report["metrics"]["tpot"]["real"] == {"n": 0}
    assert report["metrics"]["latency"]["real"]["n"] == 1
    assert report["metrics"]["throughput_tok_s"]["error_pct"] is None


def test_cli_rejects_zero_record_missing_explicit_first_token_null(tmp_path, capsys):
    real, _ = saved_run(tmp_path, (0,), "real")
    modelled, _ = saved_run(tmp_path, (0,), "modelled")
    blob = json.loads(real.read_text())
    del blob["engine"]["records"][0]["first_token_time"]
    real.write_text(json.dumps(blob))
    assert compare.main(["--real", str(real), "--modelled", str(modelled)]) == 1
    assert "must explicitly record first_token_time as null" in capsys.readouterr().out


def test_zero_duration_terminal_interval_has_no_relative_error(tmp_path, capsys):
    _, run = saved_run(tmp_path, (0,))
    run.joined[0].update(finish_time=10.0, latency=0.0)
    assert compare.check_run(run) == []
    report = compare.compare(run, run)
    compare._print(report)
    assert report["metrics"]["latency"]["error_pct"]["mean"] is None
    assert "per req error undefined" in capsys.readouterr().out


@pytest.mark.parametrize("produced", [1, False, None])
def test_zero_cap_requires_exact_zero_usage(tmp_path, produced):
    _, run = saved_run(tmp_path, (0,))
    run.usage[0]["completion_tokens"] = produced
    assert any("zero-output request" in why for why in compare.check_run(run))


@pytest.mark.parametrize("field,value", [("first_token_time", 10.5), ("ttft", 0.0)])
def test_zero_output_cannot_invent_a_first_token_or_zero_ttft(tmp_path, field, value):
    _, run = saved_run(tmp_path, (0,))
    run.joined[0][field] = value
    assert any("undefined" in why for why in compare.check_run(run))


@pytest.mark.parametrize("field", ["arrive_time", "finish_time"])
def test_zero_output_still_requires_terminal_timestamps(tmp_path, field):
    _, run = saved_run(tmp_path, (0,))
    run.joined[0][field] = None
    assert any("missing a timestamp" in why for why in compare.check_run(run))


def test_zero_terminal_order_and_latency_are_still_checked(tmp_path):
    _, run = saved_run(tmp_path, (0,))
    run.joined[0]["finish_time"] = 9.0
    assert any("out of order" in why for why in compare.check_run(run))
    run.joined[0]["finish_time"] = 20.0
    run.joined[0]["latency"] = 9.0
    assert any("reported latency" in why for why in compare.check_run(run))


def test_missing_zero_event_is_not_dropped_from_completeness(tmp_path):
    _, run = saved_run(tmp_path)
    del run.joined[0]
    run.engine_records -= 1
    assert any("no engine record" in why for why in compare.check_run(run))
    assert any("only 2 of 3" in why for why in compare.check_pair(run, run))


@pytest.mark.parametrize("produced", [0, 1, 3])
def test_positive_requested_output_always_requires_first_token(tmp_path, produced):
    _, run = saved_run(tmp_path, (3,))
    run.usage[0]["completion_tokens"] = produced
    run.joined[0]["first_token_time"] = None
    run.joined[0]["ttft"] = None
    assert any("missing a timestamp" in why for why in compare.check_run(run))
