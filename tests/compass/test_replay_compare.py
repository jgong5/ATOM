"""The report has to survive the failure that has fooled this project four times.

Two large errors of opposite sign sum to a small one. A cc-traces run landed at
-5.0% on mean latency and was read as a good result for a day; per request it
was +51.6% on TTFT against -30.3% on decode. Every test here is about a number
that looks like agreement without being any.
"""
import importlib.util
import json
from pathlib import Path


def _module():
    spec = importlib.util.spec_from_file_location(
        "replay_compare_mod",
        Path(__file__).resolve().parents[2] / "scripts/compass/replay_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(readings, *, hash_ids=True, ignore_eos=True, failed=0,
              workload=None, cache=None):
    """One replay.py artifact from `[(ttft_s, latency_s, out_tokens), ...]`."""
    results, records, rows = [], [], []
    for i, (ttft, latency, produced) in enumerate(readings):
        rid = f"cmpl-{i}"
        results.append({"index": i, "ok": True, "response": {
            "id": rid, "usage": {"prompt_tokens": 640,
                                 "completion_tokens": produced}}})
        records.append({"request_id": rid, "ttft": ttft, "latency": latency,
                        "num_cached_tokens": 576})
        row = {"arrival_s": float(i), "input_tokens": 640,
               "output_tokens": produced}
        if hash_ids:
            row["hash_ids"] = list(range(10))
            row["session"] = 0
        rows.append(row)
    return {
        "run": {"failed": failed, "prompt_lengths": "passed",
                "ignore_eos": ignore_eos,
                "rows_with_hash_ids": len(rows) if hash_ids else 0},
        "workload": workload if workload is not None else rows,
        "results": results,
        "engine": {"count": len(records), "requests": records},
        "cache_stats": cache or {"before": {"cached_tokens": 0, "full_tokens": 0},
                                 "after": {"cached_tokens": 5760,
                                           "full_tokens": 6400}},
    }


class TestCompensatingErrorsAreVisible:
    def test_a_small_latency_error_does_not_hide_a_large_ttft_one(self):
        """The exact shape of the run that was misread: TTFT up half, decode
        down a third, latency nearly right."""
        mod = _module()
        # real: 1.0s TTFT + 100 tokens at 0.010s = 2.0s latency
        real = _artifact([(1.0, 2.0, 101)] * 20)
        # modelled: 1.5s TTFT + 100 tokens at 0.007s = 2.2s latency
        modelled = _artifact([(1.5, 2.2, 101)] * 20)
        report = mod.compare(real, modelled)
        assert abs(report["latency"]["mean_signed"] - 0.10) < 0.01
        assert abs(report["ttft"]["mean_signed"] - 0.50) < 0.01
        assert report["tpot"]["mean_signed"] < -0.25

    def test_latency_carries_the_warning_in_the_artifact(self):
        """Not only in a printout, which a downstream reader never sees."""
        mod = _module()
        report = mod.compare(_artifact([(1.0, 2.0, 101)]),
                             _artifact([(1.0, 2.0, 101)]))
        assert "not a verdict" in report["latency_note"]


class TestErrorsArePairedNotPooled:
    def test_matching_distributions_with_every_request_wrong(self):
        """Both runs have the same two TTFTs in the same proportion. Pooling
        them compares identical distributions and reports no error at all."""
        mod = _module()
        real = _artifact([(1.0, 2.0, 101), (2.0, 3.0, 101)] * 10)
        modelled = _artifact([(2.0, 3.0, 101), (1.0, 2.0, 101)] * 10)
        report = mod.compare(real, modelled)
        assert report["ttft"]["p50_abs"] >= 0.49
        assert report["ttft"]["max_abs"] >= 0.99


class TestTheDenominatorFloor:
    def test_a_tiny_real_reading_is_excluded_and_counted(self):
        """A 1 ms difference over a 2 ms reading is 50%, and would otherwise
        run the percentiles. Dropping it silently is the other failure."""
        mod = _module()
        real = _artifact([(0.002, 1.0, 101)] + [(1.0, 2.0, 101)] * 9)
        modelled = _artifact([(0.003, 1.0, 101)] + [(1.0, 2.0, 101)] * 9)
        report = mod.compare(real, modelled)
        assert report["ttft"]["below_denominator_floor"] == 1
        assert report["ttft"]["relative_n"] == 9
        assert report["ttft"]["paired"] == 10  # still in the absolute figures


class TestTpot:
    def test_a_single_token_request_has_no_inter_token_interval(self):
        """Charging (latency - ttft) over one token puts a near-zero in the
        middle of the distribution and pulls every percentile with it."""
        mod = _module()
        report = mod.compare(_artifact([(1.0, 2.0, 1)] * 5),
                             _artifact([(1.0, 2.0, 1)] * 5))
        assert report["tpot"]["paired"] == 0

    def test_it_divides_by_tokens_after_the_first(self):
        mod = _module()
        real = _artifact([(1.0, 2.0, 11)] * 5)      # 1.0s over 10 intervals
        modelled = _artifact([(1.0, 3.0, 11)] * 5)  # 2.0s over 10 intervals
        report = mod.compare(real, modelled)
        assert abs(report["tpot"]["mean_signed"] - 1.0) < 1e-6


class TestItRefusesToCallARunAResult:
    def test_different_workloads_block(self):
        mod = _module()
        real = _artifact([(1.0, 2.0, 101)] * 3)
        modelled = _artifact([(1.0, 2.0, 101)] * 2)
        report = mod.compare(real, modelled)
        assert report["verdict"] == "withheld"
        assert any("different workloads" in b for b in report["blocking"])

    def test_different_sharing_blocks(self):
        """Same lengths, different hash_ids, is a different amount of prefill
        work and would read as a model error."""
        mod = _module()
        real = _artifact([(1.0, 2.0, 101)] * 3)
        modelled = _artifact([(1.0, 2.0, 101)] * 3)
        modelled["workload"][1]["hash_ids"] = [99]
        report = mod.compare(real, modelled)
        assert any("hash_ids" in b for b in report["blocking"])

    def test_failed_requests_block(self):
        mod = _module()
        report = mod.compare(_artifact([(1.0, 2.0, 101)] * 3, failed=2),
                             _artifact([(1.0, 2.0, 101)] * 3))
        assert any("failed" in b for b in report["blocking"])

    def test_a_recorded_trace_without_ignore_eos_blocks(self):
        """max_tokens is a ceiling; the trace records a length. Without the
        flag the run is a shorter workload that still reports 0 failed."""
        mod = _module()
        report = mod.compare(_artifact([(1.0, 2.0, 101)] * 3, ignore_eos=False),
                             _artifact([(1.0, 2.0, 101)] * 3))
        assert any("ignore-eos" in b or "ignore_eos" in b
                   for b in report["blocking"])

    def test_a_synthetic_trace_does_not_need_it(self):
        mod = _module()
        report = mod.compare(
            _artifact([(1.0, 2.0, 101)] * 3, hash_ids=False, ignore_eos=False),
            _artifact([(1.0, 2.0, 101)] * 3, hash_ids=False, ignore_eos=False))
        assert report["blocking"] == []
        assert report["verdict"] == "reportable"

    def test_the_barrier_is_reported_as_unchecked_rather_than_passed(self):
        """It lives on the scheduler in another process and is not served over
        HTTP. Saying nothing would read as a pass."""
        mod = _module()
        report = mod.compare(_artifact([(1.0, 2.0, 101)]),
                             _artifact([(1.0, 2.0, 101)]))
        assert report["arrival_barrier"] == "not reported by the server"

    def test_a_reported_barrier_timeout_blocks(self):
        mod = _module()
        real = _artifact([(1.0, 2.0, 101)])
        modelled = _artifact([(1.0, 2.0, 101)])
        modelled["run"]["arrival_barrier_timed_out"] = {"arrived": 1,
                                                        "expected": 9}
        report = mod.compare(real, modelled)
        assert any("arrival barrier" in b for b in report["blocking"])
        assert report["arrival_barrier"] == "checked"


class TestReuseIsMeasuredNotAssumed:
    def test_the_delta_across_the_run_is_reported(self):
        mod = _module()
        report = mod.compare(_artifact([(1.0, 2.0, 101)] * 3),
                             _artifact([(1.0, 2.0, 101)] * 3))
        assert report["reuse"]["real"]["cached_tokens"] == 5760
        assert report["reuse"]["real"]["requests_with_a_hit"] == 3

    def test_a_server_without_the_endpoint_says_so(self):
        mod = _module()
        blind = _artifact([(1.0, 2.0, 101)], cache={"before": {}, "after": {}})
        report = mod.compare(blind, _artifact([(1.0, 2.0, 101)]))
        assert report["reuse"]["real"]["available"] is False


class TestAnAbsentReadingSaysWhichKind:
    """"No readings" reads as a broken join. A run where every request decoded
    faster than the floor is a different fact from one where the two sides
    could not be matched up at all, and only one of them is a bug."""

    def _report(self, capsys, readings, tmp_path):
        mod = _module()
        real = tmp_path / "real.json"
        modelled = tmp_path / "modelled.json"
        real.write_text(json.dumps(_artifact(readings)))
        modelled.write_text(json.dumps(_artifact(readings)))
        rc = mod.main(["--real", str(real), "--modelled", str(modelled),
                       "--out", str(tmp_path / "out.json")])
        return rc, capsys.readouterr().out

    def test_every_tpot_under_the_floor_says_so(self, capsys, tmp_path):
        # 24 tokens in 60ms is 2.6ms each -- real, and below the floor.
        _, out = self._report(capsys, [(0.162, 0.2225, 24)], tmp_path)
        line = [l for l in out.splitlines() if l.strip().startswith("tpot")][0]
        assert "floor" in line and "1" in line

    def test_a_single_token_request_is_not_called_a_floor_case(self, capsys,
                                                               tmp_path):
        # One token has no inter-token interval at all, so it is not paired and
        # not floored either.
        _, out = self._report(capsys, [(0.05, 0.05, 1)], tmp_path)
        line = [l for l in out.splitlines() if l.strip().startswith("tpot")][0]
        assert "nothing paired" in line
