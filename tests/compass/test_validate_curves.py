# SPDX-License-Identifier: MIT
"""Curves and per-point verdicts: `python -m atom.compass.validate --curves OUT ARM...`
on runs laid out as `compass record`, `compass run` and `client.sh` write them.

`RECORDS` and `SERVER_METRICS` are one aiperf export: a warmup record, an errored
one, one naming no phase, one with no inter-token latency, one whose
inter-token latency is 0, one with no output token, and one metric given bare
rather than as {"value": ...}; counters under two of the names InferenceX
accepts, one in two series and one stated as `sum`. `INFERENCEX` is what
InferenceX's agentic results code at SemiAnalysisAI/InferenceX 4b94f096 gives
for them on a TP4 run, worked by hand from that code beside each value:
`artifacts.load_records_with_accounting` keeps the eight records from the third
on; `request_metrics.compute_request_metrics` takes its percentiles linearly
between ranks over the positive values, and `build_result` divides throughput
by 4 GPUs; `backends.atom.AtomBackend.populate` gives the hit rates.
"""

import json

import pytest

from atom.compass.validate import curves
from atom.compass.validate.__main__ import main

B = 1794264000000000000
RECORDS = [
    {
        "metadata": {
            "benchmark_phase": "warmup",
            "request_start_ns": 1794264000000000000,
            "request_end_ns": 1794264000900000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 600.0, "unit": "ms"},
            "inter_token_latency": {"value": 20.0, "unit": "ms"},
            "request_latency": {"value": 900.0, "unit": "ms"},
            "input_sequence_length": {"value": 7000, "unit": "tokens"},
            "output_sequence_length": {"value": 16, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264001000000000,
            "request_end_ns": 1794264061000000000,
        },
        "error": {"type": "TimeoutError", "message": "request timed out"},
        "metrics": {"input_sequence_length": {"value": 5000, "unit": "tokens"}},
    },
    {
        "metadata": {
            "request_start_ns": 1794264001200000000,
            "request_end_ns": 1794264004400000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 812.5, "unit": "ms"},
            "inter_token_latency": {"value": 21.25, "unit": "ms"},
            "request_latency": {"value": 3200.0, "unit": "ms"},
            "input_sequence_length": {"value": 9000, "unit": "tokens"},
            "output_sequence_length": {"value": 112, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264002000000000,
            "request_end_ns": 1794264009500000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 1430.0, "unit": "ms"},
            "inter_token_latency": {"value": 24.5, "unit": "ms"},
            "request_latency": {"value": 7500.0, "unit": "ms"},
            "input_sequence_length": {"value": 31000, "unit": "tokens"},
            "output_sequence_length": {"value": 248, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264004500000000,
            "request_end_ns": 1794264006100000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 240.25, "unit": "ms"},
            "inter_token_latency": {"value": 19.75, "unit": "ms"},
            "request_latency": {"value": 1600.0, "unit": "ms"},
            "input_sequence_length": {"value": 2200, "unit": "tokens"},
            "output_sequence_length": {"value": 69, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264006000000000,
            "request_end_ns": 1794264006350000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 350.0, "unit": "ms"},
            "request_latency": {"value": 350.0, "unit": "ms"},
            "input_sequence_length": {"value": 15000, "unit": "tokens"},
            "output_sequence_length": {"value": 1, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264006200000000,
            "request_end_ns": 1794264007000000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 700.0, "unit": "ms"},
            "inter_token_latency": {"value": 0.0, "unit": "ms"},
            "request_latency": {"value": 800.0, "unit": "ms"},
            "input_sequence_length": {"value": 4000, "unit": "tokens"},
            "output_sequence_length": {"value": 2, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264007100000000,
            "request_end_ns": 1794264012850000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 980.0, "unit": "ms"},
            "inter_token_latency": {"value": 27.0, "unit": "ms"},
            "request_latency": {"value": 5750.0, "unit": "ms"},
            "input_sequence_length": {"value": 48000, "unit": "tokens"},
            "output_sequence_length": {"value": 176, "unit": "tokens"},
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264009600000000,
            "request_end_ns": 1794264011900000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 455.5, "unit": "ms"},
            "inter_token_latency": {"value": 22.0, "unit": "ms"},
            "request_latency": {"value": 2300.0, "unit": "ms"},
            "input_sequence_length": {"value": 6100, "unit": "tokens"},
            "output_sequence_length": 85,
        },
    },
    {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": 1794264010000000000,
            "request_end_ns": 1794264010900000000,
        },
        "metrics": {
            "time_to_first_token": {"value": 900.0, "unit": "ms"},
            "request_latency": {"value": 900.0, "unit": "ms"},
            "input_sequence_length": {"value": 12000, "unit": "tokens"},
            "output_sequence_length": {"value": 0, "unit": "tokens"},
        },
    },
]
SERVER_METRICS = {
    "metrics": {
        "atom:prefix_cache_cached_tokens": {
            "type": "counter",
            "series": [{"stats": {"total": 41000.0, "rate": 11.4}}],
        },
        "atom_prefix_cache_cached_tokens_total": {
            "type": "counter",
            "series": [{"stats": {"sum": 900.0}}],
        },
        "atom:prefix_cache_full_tokens": {
            "type": "counter",
            "series": [
                {"labels": {"endpoint": "a"}, "stats": {"total": 52000.0}},
                {"labels": {"endpoint": "b"}, "stats": {"total": 8000.0}},
            ],
        },
        "atom:lmcache_loaded_tokens": {
            "type": "counter",
            "series": [{"stats": {"total": 6000.0}}],
        },
    }
}
INFERENCEX = {
    # 127300 input and 693 output tokens from 1.2 s to 12.85 s after B.
    "throughput per GPU": (127300 + 693) / 11.65 / 4,
    # Inter-token latencies 19.75, 21.25, 22, 24.5, 27 ms (0 and absent dropped):
    # p90 at rank 3.6, p50 at rank 2.
    "p90 interactivity": 1 / (0.0245 + 0.6 * (0.027 - 0.0245)),
    "median TPOT": 0.022,
    # Latency per output token of seven records (no output token dropped);
    # the two largest, 0.35 and 0.4 s, hold rank 5.4.
    "p90 E2E-normalized interactivity": 1 / (0.35 + 0.4 * (0.4 - 0.35)),
    # Eight TTFTs; 0.98 and 1.43 s hold rank 6.3.
    "p90 TTFT": 0.98 + 0.3 * (1.43 - 0.98),
    "GPU hit rate": (41000 + 900) / (52000 + 8000),
    "CPU hit rate": 6000 / (52000 + 8000),
}
#: The client arguments every run of a point is given, less its concurrency.
ARGUMENTS = {"scenario": "inferencex-agentx-mvp", "phases.profiling.duration": 3600.0}
#: Each value of a run in `RECORDS` that scales with its time.
TIMED = ("time_to_first_token", "inter_token_latency", "request_latency")


def _run(d, concurrency, scale=1.0, server=True, tp=4, **arguments):
    """A run of `RECORDS` at `concurrency` with every time `scale` times as
    long, and `SERVER_METRICS` unless not `server`."""
    (d / "artifacts").mkdir(parents=True)
    (d / "steps").mkdir()
    lines = []
    for r in json.loads(json.dumps(RECORDS)):
        meta = r["metadata"]
        for k in ("request_start_ns", "request_end_ns"):
            meta[k] = B + round((meta[k] - B) * scale)
        for k in TIMED:
            if k in r["metrics"]:
                r["metrics"][k]["value"] *= scale
        lines.append(json.dumps(r) + "\n")
    (d / curves.RECORDS).write_text("".join(lines))
    given = ARGUMENTS | {curves.CONCURRENCY: concurrency} | arguments
    (d / curves.CLIENT_RUN).write_text(json.dumps({"arguments": given}))
    engine = {"tensor_parallel_size": tp, "data_parallel_size": 1}
    (d / curves.ENGINE_RECORD).write_text(json.dumps(engine))
    if server:
        (d / curves.SERVER_METRICS).write_text(json.dumps(SERVER_METRICS))
    return d


def _point(d, concurrency, real=(1.0, 1.04, 0.97), **simulated):
    """A point of real runs at `real` scales and each simulated backend's run
    at its scale, with no server metrics, as a simulated client scrapes none."""
    for k, scale in enumerate(real, start=1):
        _run(d / f"real{k}", concurrency, scale)
    for backend, scale in simulated.items():
        _run(d / backend, concurrency, scale, server=False)
    return d


def _table(out: str) -> dict:
    """The verdict table printed: {(arm, concurrency, run, metric): (value, verdict)}."""
    lines = out.strip().splitlines()
    assert lines[0] == "arm | concurrency | run | metric | value | verdict"
    rows = [line.split(" | ") for line in lines[1:]]
    return {(a, int(c), r, m): (v, verdict) for a, c, r, m, v, verdict in rows}


def test_each_metric_equals_inferencex_on_the_export_fixture(tmp_path):
    got = curves.metrics(_run(tmp_path / "real1", 4))
    assert got.pop("concurrency") == 4
    assert set(got) == set(INFERENCEX)
    for name, value in INFERENCEX.items():
        assert got[name] == pytest.approx(value, rel=1e-12), name


def test_a_run_with_no_server_metrics_has_no_hit_rates(tmp_path):
    got = curves.metrics(_run(tmp_path / "step_linear", 4, server=False))
    assert got["GPU hit rate"] is None and got["CPU hit rate"] is None
    assert got["p90 TTFT"] == pytest.approx(INFERENCEX["p90 TTFT"], rel=1e-12)


def test_one_command_draws_each_arm_and_both_and_judges_every_point(tmp_path, capsys):
    tp4, tp2 = tmp_path / "tp4", tmp_path / "tp2"
    _point(tp4 / "c4", 4, step_linear=1.01, step_grid=1.2)
    _point(tp4 / "c1", 1, step_linear=0.99)
    _point(tp2 / "c2", 2, step_linear=1.06)
    out = tmp_path / "charts"
    main(["--curves", str(out), str(tp4), str(tp2)])
    for name in ("tp4.png", "tp2.png", "arms.png"):
        assert (out / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    table = _table(capsys.readouterr().out)
    # Points by concurrency, each backend's metrics before its point's verdict.
    assert [k[1:3] for k in table if k[0] == "tp4" and k[3] == "point"] == [
        (1, "step_linear"),
        (4, "step_linear"),
        (4, "step_grid"),
    ]
    assert table["tp4", 4, "step_linear", "p90 TTFT (s)"][1] == "pass: in band"
    assert table["tp4", 4, "step_grid", "p90 TTFT (s)"][1] == "fail"
    assert table["tp4", 4, "step_grid", "point"][1].startswith(
        "fail: throughput per GPU"
    )
    # The simulated client scraped no server metrics: hit rates have no value.
    value, verdict = table["tp4", 4, "step_linear", "GPU hit rate"]
    assert (value, verdict) == ("no value in ['simulated']", "refused")
    assert table["tp4", 4, "step_linear", "point"] == (
        "3 real runs",
        "refused: GPU hit rate, CPU hit rate",
    )
    # Outside the band [0.97, 1.04] of times, 1.06 is within 10% of TTFT's mean
    # and not within 5% of median TPOT's.
    ttft = table["tp2", 2, "step_linear", "p90 TTFT (s)"]
    assert ttft[1] == "pass: within 10% of mean" and "limit 10%" in ttft[0]
    tpot = table["tp2", 2, "step_linear", "median TPOT (ms)"]
    assert tpot[1] == "fail" and "real mean 22.07," in tpot[0]
    e2e = "p90 E2E-normalized interactivity (tok/s/user)"
    assert table["tp2", 2, "step_linear", e2e][1] == "not judged"


def test_a_pass_in_a_band_wider_than_the_limit_says_so():
    real = {"real1": 100.0, "real2": 120.0, "real3": 90.0}
    assert curves.judge("p90 TTFT", real, 118.0)[1] == "pass: in band, past 10% of mean"
    assert curves.judge("p90 TTFT", real, 104.0)[1] == "pass: in band"
    assert curves.judge("median TPOT", real, 126.0)[1] == "fail"
    rates = {"real1": 0.60, "real2": 0.61, "real3": 0.62}
    value = (
        "simulated 63.50%; real mean 61.00%, band 60.00% to 62.00%;"
        " off mean 2.50 points, limit 2 points"
    )
    assert curves.judge("GPU hit rate", rates, 0.635) == (value, "fail")
    assert (
        curves.judge("GPU hit rate", rates, 0.592)[1] == "pass: within 2 points of mean"
    )


def test_a_point_is_refused_without_two_real_runs_or_one_set_of_arguments(tmp_path):
    arm = tmp_path / "arm"
    _point(arm / "c1", 1, real=(1.0,), step_linear=1.0)
    with pytest.raises(ValueError, match="holds 1 real"):
        main(["--curves", str(tmp_path / "out"), str(arm)])
    _run(arm / "c1" / "real2", 1, 1.0, **{"phases.profiling.duration": 600.0})
    with pytest.raises(
        ValueError, match=r"different client arguments: \['phases.profiling.duration'\]"
    ):
        main(["--curves", str(tmp_path / "out"), str(arm)])
