"""The curve is two derived numbers, so every way it can lie is arithmetic.

A saturation plot is read at a glance: two lines, one labelled real and one
modelled, and whether they lie on top of each other is the whole result. That
makes it the most dangerous artifact in this harness -- it has no place to put
a caveat. So the caveats live here, as refusals to draw.
"""
import importlib.util
import json
from pathlib import Path


def _module():
    spec = importlib.util.spec_from_file_location(
        "saturation_mod",
        Path(__file__).resolve().parents[2] / "scripts/compass/saturation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(readings, *, clients=4, closed=True, failed=0, extra_records=()):
    """One replay.py artifact from `[(arrive, ttft, latency, produced), ...]`."""
    results, records = [], []
    for i, (arrive, ttft, latency, produced) in enumerate(readings):
        rid = f"cmpl-{i}"
        results.append({"index": i, "ok": True, "response": {
            "id": rid, "usage": {"prompt_tokens": 640,
                                 "completion_tokens": produced}}})
        records.append({"request_id": rid, "arrive_time": arrive,
                        "first_token_time": arrive + ttft,
                        "finish_time": arrive + latency,
                        "ttft": ttft, "latency": latency,
                        "num_cached_tokens": 0})
    records.extend(extra_records)
    return {"run": {"clients": clients, "closed_loop": closed, "failed": failed,
                    "prompt_lengths": "passed"},
            "workload": [], "results": results,
            "engine": {"count": len(records), "clock": "virtual",
                       "requests": records},
            "cache_stats": {"before": {}, "after": {}}}


class TestBothAxesComeOffTheEngineClock:
    def test_per_gpu_is_output_tokens_over_the_engine_span(self):
        mod = _module()
        # Two requests, engine span 0.0 -> 10.0, 200 output tokens.
        point = mod.rung(_artifact([(0.0, 1.0, 5.0, 100),
                                    (5.0, 1.0, 5.0, 100)]), gpus=1)
        assert point["engine_span_s"] == 10.0
        assert point["tokens_s_per_gpu"] == 20.0

    def test_per_gpu_divides_by_the_gpus_it_ran_on(self):
        mod = _module()
        one = mod.rung(_artifact([(0.0, 1.0, 10.0, 100)]), gpus=1)
        four = mod.rung(_artifact([(0.0, 1.0, 10.0, 100)]), gpus=4)
        assert one["tokens_s_per_gpu"] == 4 * four["tokens_s_per_gpu"]

    def test_per_user_is_one_over_tpot_not_tokens_over_latency(self):
        mod = _module()
        # 101 tokens, 1s to the first and 10s of decode -> 0.1 s/token.
        point = mod.rung(_artifact([(0.0, 1.0, 11.0, 101)]), gpus=1)
        assert abs(point["tokens_s_per_user"] - 10.0) < 1e-9
        # The naive reading, tokens over end-to-end latency, would be 9.18.
        assert abs(point["tokens_s_per_user"] - 101 / 11.0) > 0.5


class TestOnlyTheRequestsTheRunExecuted:
    def test_records_from_an_earlier_run_do_not_stretch_the_span(self):
        mod = _module()
        # /compass/requests is cumulative for the server's lifetime. A record
        # with no client result is not this rung's work, and letting it set
        # the span would divide this rung's tokens by someone else's seconds.
        stale = {"request_id": "cmpl-old", "arrive_time": -500.0,
                 "finish_time": -400.0, "ttft": 1.0, "latency": 100.0}
        point = mod.rung(_artifact([(0.0, 1.0, 10.0, 100)],
                                   extra_records=[stale]), gpus=1)
        assert point["engine_span_s"] == 10.0
        assert point["requests"] == 1


class TestIdleTimeIsReportedNotAssumed:
    def test_busy_fraction_sees_a_gap_between_requests(self):
        mod = _module()
        # Ten seconds of span, two one-second requests: the client was
        # thinking for 80% of it. On a real clock that time is in the span; on
        # a virtual one it is not, so the two sides are not comparable unless
        # this is visible.
        point = mod.rung(_artifact([(0.0, 0.1, 1.0, 10),
                                    (9.0, 0.1, 1.0, 10)]), gpus=1)
        assert abs(point["busy_fraction"] - 0.2) < 1e-9

    def test_overlapping_requests_are_not_counted_twice(self):
        mod = _module()
        point = mod.rung(_artifact([(0.0, 0.1, 10.0, 10),
                                    (1.0, 0.1, 5.0, 10)]), gpus=1)
        assert abs(point["busy_fraction"] - 1.0) < 1e-9


class TestACurveThatCannotBeDrawnIsNotDrawn:
    def _sweep(self, tmp_path, mod, *, real_clients=(1, 4), modelled_clients=(1, 4),
               failed=0, closed=True):
        dirs = []
        for i, n in enumerate(real_clients):
            d = tmp_path / f"c{n}"
            d.mkdir(exist_ok=True)
            (d / "real.json").write_text(json.dumps(_artifact(
                [(0.0, 1.0, 11.0, 101)], clients=n, failed=failed,
                closed=closed)))
            dirs.append(str(d))
        for n in modelled_clients:
            d = tmp_path / f"c{n}"
            d.mkdir(exist_ok=True)
            (d / "modelled.json").write_text(json.dumps(_artifact(
                [(0.0, 1.0, 11.0, 101)], clients=n, closed=closed)))
            if str(d) not in dirs:
                dirs.append(str(d))
        return dirs

    def test_a_clean_sweep_draws(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, mod)
        plot = tmp_path / "curve.png"
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json"),
                       "--plot", str(plot)])
        assert rc == 0
        assert plot.exists()

    def test_a_paced_rung_is_refused(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, mod, closed=False)
        plot = tmp_path / "curve.png"
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json"),
                       "--plot", str(plot)])
        assert rc == 1
        assert not plot.exists()
        report = json.loads((tmp_path / "s.json").read_text())
        assert any("closed-loop" in r for r in report["blocking"])

    def test_a_failed_request_is_refused(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, mod, failed=3)
        plot = tmp_path / "curve.png"
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json"),
                       "--plot", str(plot)])
        assert rc == 1
        assert not plot.exists()

    def test_sides_that_swept_different_rungs_are_refused(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, mod, real_clients=(1, 4),
                           modelled_clients=(1, 4, 8))
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json")])
        assert rc == 1
        report = json.loads((tmp_path / "s.json").read_text())
        assert any("different client counts" in r for r in report["blocking"])

    def test_a_missing_side_is_refused_and_named(self, tmp_path):
        mod = _module()
        d = tmp_path / "c1"
        d.mkdir()
        (d / "real.json").write_text(json.dumps(
            _artifact([(0.0, 1.0, 11.0, 101)], clients=1)))
        rc = mod.main([str(d), "--out", str(tmp_path / "s.json")])
        assert rc == 1
        report = json.loads((tmp_path / "s.json").read_text())
        assert any("modelled.json" in r for r in report["blocking"])

    def test_the_report_is_written_even_when_the_curve_is_withheld(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, mod, failed=1)
        assert mod.main([*dirs, "--out", str(tmp_path / "s.json")]) == 1
        # Withholding the verdict must not withhold the numbers; finding out
        # why otherwise costs another two servers.
        report = json.loads((tmp_path / "s.json").read_text())
        assert report["curves"]["real"][0]["tokens_s_per_gpu"] is not None
