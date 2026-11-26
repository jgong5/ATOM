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


def _phased(d, warmup, profiled):
    """A run at c=8 whose export holds warmup requests `warmup` and profiling
    requests `profiled`, each (conversation, turn, input tokens[, cached])."""
    run = _run(d, 8, server=False)
    records = [
        {
            "metadata": {
                "benchmark_phase": phase,
                "conversation_id": conversation,
                "turn_index": turn,
                "request_start_ns": B + i * 10**9,
                "request_end_ns": B + (i + 1) * 10**9,
            },
            "metrics": {
                "time_to_first_token": {"value": 100.0, "unit": "ms"},
                "inter_token_latency": {"value": 5.0, "unit": "ms"},
                "request_latency": {"value": 1000.0, "unit": "ms"},
                "input_sequence_length": {"value": tokens, "unit": "tokens"},
                "output_sequence_length": {"value": 10, "unit": "tokens"},
            }
            | (
                {"usage_prompt_cache_read_tokens": {"value": cached[0]}}
                if cached
                else {}
            ),
        }
        for i, (phase, (conversation, turn, tokens, *cached)) in enumerate(
            [("warmup", w) for w in warmup] + [("profiling", p) for p in profiled]
        )
    ]
    (run / curves.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


def _one_real_env(env, *runs):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, runs)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), **env},
        check=False,
    )


def _warmup_runs(tmp_path):
    real = _phased(
        tmp_path / "real1",
        [("w", 0, 3000, 300), ("w", 1, 3000, 600), ("x", 0, 2000)],
        [("a", 0, 1000, 800), ("b", 0, 1000, 800)],
    )
    sim = _phased(
        tmp_path / "step_linear",
        [("w", 0, 3000, 900), ("w", 1, 3000, 1200), ("x", 0, 2000, 0)],
        [("a", 0, 1000, 800), ("b", 0, 1000, 800)],
    )
    return real, sim


def test_warmup_hits_are_each_run_s_cached_share_of_the_warmup_requests_both_ran(
    tmp_path,
):
    real, sim = _warmup_runs(tmp_path)

    done = _one_real_env({one_real.WARMUP_HITS_ENV: "1"}, real, sim)

    assert done.returncode == 0, done.stderr
    (line,) = [
        x for x in done.stdout.splitlines() if x.startswith("step_linear: warmup")
    ]
    # ("x", 0) states no cached tokens in the real run, so the requests both ran
    # are ("w", 0) and ("w", 1): 2100 of 6000 against 900; the warmup holds 8000
    # of the real run's 10000 prompt tokens, so the +20 pt gap moves +16 pt.
    assert line == (
        "step_linear: warmup prefix hits over the 2 warmup requests both ran "
        "35.0% against the real run's 15.0% (+20.0 pt); the warmup holds 80% "
        "of the real run's prompt tokens, so it moves the GPU hit rate by about "
        "+16.0 pt"
    )
    # the profiled requests' own line is unchanged and the table's JSON stays last
    assert _hits_line(done.stdout, "step_linear").startswith(
        "step_linear: prefix hits over the 2 requests both ran 80.0%"
    )
    json.loads(done.stdout.splitlines()[-1])


def test_without_the_switch_the_output_is_unchanged(tmp_path):
    real, sim = _warmup_runs(tmp_path)

    off = _one_real_env({}, real, sim)
    zero = _one_real_env({one_real.WARMUP_HITS_ENV: "0"}, real, sim)
    on = _one_real_env({one_real.WARMUP_HITS_ENV: "1"}, real, sim)

    assert off.returncode == zero.returncode == on.returncode == 0, off.stderr
    assert off.stdout == zero.stdout
    assert "warmup prefix hits" not in off.stdout
    extra = [x for x in on.stdout.splitlines() if x not in off.stdout.splitlines()]
    assert len(extra) == 1 and extra[0].startswith("step_linear: warmup prefix hits")


def test_warmup_hits_name_none_where_no_warmup_record_states_its_cached_tokens(
    tmp_path,
):
    real = _phased(tmp_path / "real1", [("w", 0, 3000)], [("a", 0, 1000, 800)])
    sim = _phased(tmp_path / "step_linear", [("w", 0, 3000, 10)], [("a", 0, 1000, 800)])

    done = _one_real_env({one_real.WARMUP_HITS_ENV: "1"}, real, sim)

    assert done.returncode == 0, done.stderr
    assert (
        "step_linear: warmup prefix hits: no warmup request both ran states its "
        "cached tokens"
    ) in done.stdout.splitlines()


def test_a_warmup_switch_other_than_0_or_1_exits_2(tmp_path):
    real, sim = _warmup_runs(tmp_path)

    done = _one_real_env({one_real.WARMUP_HITS_ENV: "yes"}, real, sim)

    assert done.returncode == 2
    assert one_real.WARMUP_HITS_ENV in done.stderr


def _timed(d, sends):
    """A run at c=8 of profiling requests `sends`, each (send s, TTFT ms)."""
    run = _run(d, 8, server=False)
    records = [
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "conversation_id": f"c{i}",
                "turn_index": 0,
                "request_start_ns": B + round(start * 1e9),
                "request_end_ns": B + round((start + 20.0) * 1e9),
            },
            "metrics": {
                "time_to_first_token": {"value": ttft, "unit": "ms"},
                "inter_token_latency": {"value": 5.0, "unit": "ms"},
                "request_latency": {"value": 20000.0, "unit": "ms"},
                "input_sequence_length": {"value": 1000, "unit": "tokens"},
                "output_sequence_length": {"value": 10, "unit": "tokens"},
            },
        }
        for i, (start, ttft) in enumerate(sends)
    ]
    (run / curves.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


def _transient_runs(tmp_path):
    # before 300 s: real 3 of 4 under 10 s (TTFT 1, 2, 3 s and 50 s), the
    # simulated run 2 of 4 (2, 4 s; 30, 60 s); from 300 s on: 500 s / 550 s
    real = _timed(
        tmp_path / "real1",
        [
            (0.0, 1000.0),
            (10.0, 2000.0),
            (20.0, 3000.0),
            (30.0, 50000.0),
            (300.0, 500000.0),
            (310.0, 500000.0),
        ],
    )
    sim = _timed(
        tmp_path / "step_linear",
        [
            (0.0, 2000.0),
            (10.0, 4000.0),
            (20.0, 30000.0),
            (30.0, 60000.0),
            (300.0, 550000.0),
            (310.0, 550000.0),
        ],
    )
    return real, sim


def test_transient_splits_ttft_at_w_into_the_early_fast_share_and_each_mode_s_median(
    tmp_path,
):
    real, sim = _transient_runs(tmp_path)

    done = _one_real_env({one_real.TRANSIENT_ENV: "300:10"}, real, sim)

    assert done.returncode == 0, done.stderr
    (line,) = [
        x for x in done.stdout.splitlines() if x.startswith("step_linear: transient")
    ]
    # early fast medians: real 2 s of (1, 2, 3), sim 3 s of (2, 4): +50%;
    # late medians 500 s and 550 s: +10%
    assert line == (
        "step_linear: transient at 300 s: of the requests sent before it (4 against "
        "the real run's 4), 50.0% under 10 s against 75.0% (-25.0 pt), their TTFT "
        "median +50.0%; TTFT median of those sent from 300 s on +10.0%"
    )
    json.loads(done.stdout.splitlines()[-1])  # the table's JSON stays last


def test_without_the_transient_switch_the_output_is_unchanged(tmp_path):
    real, sim = _transient_runs(tmp_path)

    off = _one_real_env({}, real, sim)
    empty = _one_real_env({one_real.TRANSIENT_ENV: ""}, real, sim)
    on = _one_real_env({one_real.TRANSIENT_ENV: "300:10"}, real, sim)

    assert off.returncode == empty.returncode == on.returncode == 0, off.stderr
    assert off.stdout == empty.stdout
    assert "step_linear: transient" not in off.stdout
    extra = [x for x in on.stdout.splitlines() if x not in off.stdout.splitlines()]
    assert len(extra) == 1 and extra[0].startswith("step_linear: transient at 300 s")


def test_transient_names_none_where_a_run_holds_no_profiled_request(tmp_path):
    real = _timed(tmp_path / "real1", [(0.0, 1000.0)])
    sim = _timed(tmp_path / "step_linear", [(0.0, 1000.0)])

    assert one_real.transient(real, sim, 300.0, 10.0).startswith(
        "step_linear: transient at 300 s: of the requests sent before it (1 against"
    )
    assert one_real._modes(tmp_path / "real1", 300.0, 10.0)["late"] is None
    empty = _timed(tmp_path / "empty", [])
    assert one_real.transient(real, empty, 300.0, 10.0) == (
        "empty: transient at 300 s: a run holds no profiled request"
    )


@pytest.mark.parametrize(
    "value", ["300", "a:b", "0:10", "300:-1", "300:10:5", "inf:10"]
)
def test_a_transient_switch_other_than_two_positive_seconds_exits_2(tmp_path, value):
    real, sim = _transient_runs(tmp_path)

    done = _one_real_env({one_real.TRANSIENT_ENV: value}, real, sim)

    assert done.returncode == 2
    assert one_real.TRANSIENT_ENV in done.stderr
