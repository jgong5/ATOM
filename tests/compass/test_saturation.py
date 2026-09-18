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


def _artifact(readings, *, clients=4, closed=True, failed=0, extra_records=(),
              sessions=None, dag="dag-0"):
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
                    "prompt_lengths": "passed", "dag_sha256": dag},
            "workload": [{"session": s} for s in
                         (sessions if sessions is not None
                          else range(len(readings)))],
            "results": results,
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


class TestAFrozenArrivalClockIsCaught:
    """The failure this guard exists for did not look like a failure.

    A simulated run keeps one virtual clock per process and only the engine
    core's advances, so an arrival stamped in the API process lands on the
    epoch every time. Every request then reports arriving at the start of the
    run. Nothing errors: TTFT simply absorbs the queueing each request had
    already passed, and the span collapses onto busy time. Measured on a
    one-client rung, that read as +203% TTFT and +68% tokens/s per GPU -- both
    of which look like a cost model being wrong.
    """

    def test_all_requests_at_one_instant_is_refused(self, tmp_path):
        mod = _module()
        d = tmp_path / "c1"
        d.mkdir()
        frozen = [(0.0, 0.03, 0.09, 24), (0.0, 0.13, 0.19, 24),
                  (0.0, 0.23, 0.30, 24)]
        for side in ("real", "modelled"):
            (d / f"{side}.json").write_text(json.dumps(
                _artifact(frozen, clients=1)))
        rc = mod.main([str(d), "--out", str(tmp_path / "s.json")])
        assert rc == 1
        report = json.loads((tmp_path / "s.json").read_text())
        assert any("one arrival instant" in r for r in report["blocking"])

    def test_one_request_per_client_may_share_an_arrival(self, tmp_path):
        """A rung whose clients each ran one request is not evidence of this.

        Four clients starting together genuinely do arrive together, and
        refusing that would withhold a correct curve.
        """
        mod = _module()
        d = tmp_path / "c4"
        d.mkdir()
        together = [(0.0, 0.03, 0.09, 24)] * 4
        for side in ("real", "modelled"):
            (d / f"{side}.json").write_text(json.dumps(
                _artifact(together, clients=4)))
        rc = mod.main([str(d), "--out", str(tmp_path / "s.json")])
        report = json.loads((tmp_path / "s.json").read_text())
        assert not any("one arrival instant" in r for r in report["blocking"])
        assert rc == 0


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
               failed=0, closed=True, real_dag="dag-0", modelled_dag="dag-0"):
        dirs = []
        for i, n in enumerate(real_clients):
            d = tmp_path / f"c{n}"
            d.mkdir(exist_ok=True)
            (d / "real.json").write_text(json.dumps(_artifact(
                [(0.0, 1.0, 11.0, 101)], clients=n, failed=failed,
                closed=closed, dag=real_dag)))
            dirs.append(str(d))
        for n in modelled_clients:
            d = tmp_path / f"c{n}"
            d.mkdir(exist_ok=True)
            (d / "modelled.json").write_text(json.dumps(_artifact(
                [(0.0, 1.0, 11.0, 101)], clients=n, closed=closed,
                dag=modelled_dag)))
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


class TestARungIsLabelledWithTheSlotsItRan:
    """A client count is what the rung was *asked* for. A slot holds one
    session at a time, so when the pool does not divide evenly the last stretch
    of the run is a few slots working alone -- and `tokens_s_per_gpu` divides
    by the whole span, tail included. Reported, so the point is not read as
    "N clients could not fill the GPU".
    """

    def test_slots_that_all_ran_throughout_realise_their_client_count(self):
        mod = _module()
        # Four sessions, one per slot, all spanning the same 10s.
        point = mod.rung(_artifact([(0.0, 1.0, 10.0, 100)] * 4, clients=4),
                         gpus=1)
        assert point["sessions"] == 4
        assert point["realised_clients"] == 4.0
        assert point["slot_utilisation"] == 1.0

    def test_a_slot_that_finished_early_is_not_counted_to_the_end(self):
        mod = _module()
        # Two slots: one runs the whole 10s, one is done at 2s.
        point = mod.rung(_artifact([(0.0, 1.0, 10.0, 100),
                                    (0.0, 1.0, 2.0, 20)], clients=2), gpus=1)
        assert point["engine_span_s"] == 10.0
        assert point["realised_clients"] == 1.2
        assert point["slot_utilisation"] == 0.6

    def test_a_slot_holds_its_session_across_the_gap_between_its_turns(self):
        mod = _module()
        # One session, two turns with 5s of think time between them. The slot
        # is occupied for the gap -- the user is thinking, not gone.
        point = mod.rung(_artifact([(0.0, 1.0, 2.0, 20),
                                    (7.0, 1.0, 3.0, 30)],
                                   clients=1, sessions=[9, 9]), gpus=1)
        assert point["sessions"] == 1
        assert point["engine_busy_s"] == 5.0    # the engine idled the gap
        assert point["realised_clients"] == 1.0  # the slot did not

    def test_a_rung_with_no_workload_rows_reports_no_slots_rather_than_lying(self):
        mod = _module()
        artifact = _artifact([(0.0, 1.0, 10.0, 100)], clients=1)
        artifact["workload"] = []
        point = mod.rung(artifact, gpus=1)
        assert point["sessions"] == 0
        assert point["realised_clients"] == 0.0


class TestBothSidesMustHaveRunTheSameGraph:
    """Two runs can both be closed-loop, both be c8, both be over the same
    trace, and still be two workloads: the deal of sessions to slots is part of
    the experiment. `dag_sha256` names the whole graph, so the check is an
    equality, not a heuristic.
    """

    def _sweep(self, tmp_path, **kw):
        return TestACurveThatCannotBeDrawnIsNotDrawn()._sweep(
            tmp_path, None, **kw)

    def test_two_sides_that_dealt_sessions_differently_are_refused(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, real_dag="dag-a", modelled_dag="dag-b")
        plot = tmp_path / "curve.png"
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json"),
                       "--plot", str(plot)])
        assert rc == 1
        assert not plot.exists()
        report = json.loads((tmp_path / "s.json").read_text())
        assert any("different dependency graphs" in r
                   for r in report["blocking"])

    def test_a_rung_that_named_no_graph_is_refused(self, tmp_path):
        mod = _module()
        # An artifact from before the graph was recorded. Matching Nones are
        # not evidence the two sides agreed.
        dirs = self._sweep(tmp_path, real_dag=None, modelled_dag=None)
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json")])
        assert rc == 1
        report = json.loads((tmp_path / "s.json").read_text())
        assert any("nothing shows the two ran the same graph" in r
                   for r in report["blocking"])

    def test_matching_graphs_draw(self, tmp_path):
        mod = _module()
        dirs = self._sweep(tmp_path, real_dag="dag-x", modelled_dag="dag-x")
        plot = tmp_path / "curve.png"
        rc = mod.main([*dirs, "--out", str(tmp_path / "s.json"),
                       "--plot", str(plot)])
        assert rc == 0
        assert plot.exists()
