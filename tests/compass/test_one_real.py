# SPDX-License-Identifier: MIT
"""`scripts/compass/one_real.py`: one real run against its simulated runs, each
metric's relative error, and a refusal of runs given other client arguments."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_validate_curves import B, _run

from atom.compass.validate import curves

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "one_real.py"
ROOT = SCRIPT.parents[2]
_spec = importlib.util.spec_from_file_location("one_real", SCRIPT)
one_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(one_real)


def _one_real(*runs):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, runs)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT)},
        check=False,
    )


def test_each_latency_error_is_the_simulated_run_s_scale_less_one(tmp_path):
    real = _run(tmp_path / "real1", 4)
    sim = _run(tmp_path / "step_linear", 4, scale=1.1, server=False)

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    rows = json.loads(done.stdout.splitlines()[-1])
    for name in ("TTFT p50 ms", "TTFT p99 ms", "TPOT p50 ms", "E2E p50 ms"):
        assert rows["step_linear"][name] == pytest.approx(1.1 * rows["real"][name])
    table = {line[:34].strip(): line for line in done.stdout.splitlines()[1:-1]}
    assert table["TTFT p50 ms"].split()[-1] == "+10.0%"
    assert rows["real"]["requests"] == rows["step_linear"]["requests"] > 0


def test_runs_given_other_client_arguments_exit_2(tmp_path):
    real = _run(tmp_path / "real1", 4)
    sim = _run(tmp_path / "step_linear", 8, server=False)

    done = _one_real(real, sim)

    assert done.returncode == 2
    assert "other client arguments" in done.stderr


def test_tails_name_each_simulated_error_against_the_real_run_s_block_interval(
    tmp_path,
):
    real = _run(tmp_path / "real1", 4)
    sim = _run(tmp_path / "step_linear", 4, scale=1.1, server=False)

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    (line,) = [x for x in done.stdout.splitlines() if x.startswith("real tails")]
    assert "TTFT p99 ms" in line and "p90 TTFT" in line and "TPOT p90 ms" in line
    assert "step_linear +10.0%" in line


def _replayed(d, sends, idle=None):
    """A run at c=8 of profiling requests `sends`, each (conversation, turn,
    send s, end s, input tokens[, cached tokens]) with 10 output tokens, its
    aiperf log stating the global idle cap's `idle` (jumps, skipped s) at the
    profiling phase's end after the warmup's none, or no log."""
    run = _run(d, 8, server=False)
    records = [
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "conversation_id": conversation,
                "turn_index": turn,
                "request_start_ns": B + round(start * 1e9),
                "request_end_ns": B + round(end * 1e9),
            },
            "metrics": {
                "time_to_first_token": {"value": 100.0, "unit": "ms"},
                "inter_token_latency": {"value": 5.0, "unit": "ms"},
                "request_latency": {"value": (end - start) * 1e3, "unit": "ms"},
                "input_sequence_length": {"value": tokens, "unit": "tokens"},
                "output_sequence_length": {"value": 10, "unit": "tokens"},
            }
            | (
                {"usage_prompt_cache_read_tokens": {"value": cached[0]}}
                if cached
                else {}
            ),
        }
        for conversation, turn, start, end, tokens, *cached in sends
    ]
    (run / curves.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    if idle is not None:
        log = run / one_real.AIPERF_LOG
        log.parent.mkdir(parents=True)
        summary = (
            "AgenticReplayTiming - INFO - Global system-idle cap summary: limit=10s"
        )
        log.write_text(
            f"{summary}, jumps=0, skipped=0.000s\n"
            f"{summary}, jumps={idle[0]}, skipped={idle[1]:.3f}s\n"
        )
    return run


def _window_line(out, name):
    (line,) = [x for x in out.splitlines() if x.startswith(f"{name}: throughput")]
    return line


def test_throughput_over_the_requests_both_ran_leaves_out_what_one_run_alone_sent(
    tmp_path,
):
    both = [("a", 0, 0.0, 10.0, 1000), ("b", 0, 2.0, 20.0, 3000)]
    real = _replayed(tmp_path / "real1", both, idle=(2, 26.926))
    sim = _replayed(
        tmp_path / "step_linear",
        [*both, ("a", 1, 15.0, 30.0, 500000)],
        idle=(3, 56.813),
    )

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    rows = json.loads(done.stdout.splitlines()[-1])
    assert (
        rows["step_linear"]["throughput per GPU"]
        > 10 * rows["real"]["throughput per GPU"]
    )
    line = _window_line(done.stdout, "step_linear")
    assert "over the 2 requests both ran +0.0%" in line
    assert "(0 ran in the real run only, 1 in step_linear only)" in line
    assert "real 26.9 s in 2 jumps, step_linear 56.8 s in 3 jumps" in line


def test_paired_throughput_is_each_run_s_tokens_of_them_over_its_own_span(tmp_path):
    real = _replayed(
        tmp_path / "real1", [("a", 0, 0.0, 10.0, 990), ("b", 0, 0.0, 8.0, 990)]
    )
    sim = _replayed(
        tmp_path / "step_linear", [("a", 0, 0.0, 8.0, 990), ("b", 0, 0.0, 8.0, 990)]
    )

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    line = _window_line(done.stdout, "step_linear")
    assert "over the 2 requests both ran +25.0%" in line
    assert "idle cap skipped real -, step_linear -" in line


def _hits_line(out, name):
    (line,) = [x for x in out.splitlines() if x.startswith(f"{name}: prefix hits")]
    return line


def test_prefix_hits_over_the_requests_both_ran_are_each_run_s_own_cached_share(
    tmp_path,
):
    real = _replayed(
        tmp_path / "real1",
        [("a", 0, 0.0, 10.0, 1000, 800), ("b", 0, 2.0, 20.0, 3000, 2400)],
    )
    sim = _replayed(
        tmp_path / "step_linear",
        [
            ("a", 0, 0.0, 10.0, 1000, 600),
            ("b", 0, 2.0, 20.0, 3000, 2000),
            ("c", 0, 3.0, 9.0, 5000, 5000),
        ],
    )

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    assert _hits_line(done.stdout, "step_linear") == (
        "step_linear: prefix hits over the 2 requests both ran 65.0% against "
        "the real run's 80.0% (-15.0 pt), their uncached input tokens +75.0%"
    )
    json.loads(done.stdout.splitlines()[-1])  # the table's JSON stays last


def test_prefix_hits_name_none_where_no_record_states_its_cached_tokens(tmp_path):
    real = _replayed(tmp_path / "real1", [("a", 0, 0.0, 10.0, 1000)])
    sim = _replayed(tmp_path / "step_linear", [("a", 0, 0.0, 10.0, 1000, 10)])

    done = _one_real(real, sim)

    assert done.returncode == 0, done.stderr
    assert _hits_line(done.stdout, "step_linear") == (
        "step_linear: prefix hits: no request both ran states its cached tokens"
    )
