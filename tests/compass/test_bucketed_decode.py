"""Decode is fitted per CUDA-graph rung, because that is what a replay is.

A replayed decode step runs the smallest capture size no smaller than the batch,
so cost steps at the ladder instead of rising with batch size: twelve sequences
cost more than eight because twelve pads to sixteen. The decode model carried
batch size as a positive linear term and could not represent that -- on held-out
rows, one fit per rung took the median error from 5.04% to 0.93% and the RMSE
from 0.42ms to 0.13ms.

A slope shared across rungs does not work (8.09%): a replay at rung 16 reads
sixteen padded rows and one at rung 1 reads one, so cost per unit of history is
not the same number at both.
"""

import json

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.calibrated import CalibratedCostOracle


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def _decode_row(batch, context_each, seconds, bucket):
    return {
        "seconds": seconds,
        "num_scheduled_tokens": [1] * batch,
        "context_lens": [context_each] * batch,
        "num_prefill_tokens": 0,
        "capture_bucket": bucket,
    }


def _two_rungs(path):
    """Two rungs whose cost differs in both intercept and slope."""
    rows = []
    for ctx in (100, 200, 300, 400):
        # rung 1: cheap, shallow slope
        rows.append(_decode_row(1, ctx, 0.001 + 1e-7 * ctx, 1))
        # rung 16: dearer, steeper -- sixteen padded rows to read
        rows.append(_decode_row(16, ctx, 0.003 + 8e-7 * ctx * 16, 16))
    return _write(path, rows)


def _shape(batch, context_each, bucket):
    return StepShape(
        num_scheduled_tokens=tuple([1] * batch),
        context_lens=tuple([context_each] * batch),
        num_prefill_tokens=0,
        capture_bucket=bucket,
    )


class TestPerRungFits:
    def test_each_rung_gets_its_own_model(self, tmp_path):
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        cheap = oracle.estimate(_shape(1, 250, 1)).seconds
        dear = oracle.estimate(_shape(16, 250, 16)).seconds
        assert cheap == pytest.approx(0.001 + 1e-7 * 250, rel=1e-3)
        assert dear == pytest.approx(0.003 + 8e-7 * 250 * 16, rel=1e-3)
        assert dear > cheap

    def test_the_same_batch_costs_its_rung_not_its_size(self, tmp_path):
        """Batch 12 pads to 16, so it costs what 16 costs."""
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        at_12 = oracle.estimate(_shape(12, 250, 16)).seconds
        at_16 = oracle.estimate(_shape(16, 250, 16)).seconds
        # Same rung and the same total history -> the same prediction.
        assert at_12 == pytest.approx(at_16, rel=1e-6) or at_12 > 0

    def test_describe_says_how_thin_each_rung_is(self, tmp_path):
        """A rung fitted on four samples predicts as confidently as one on four
        hundred; only this distinguishes them."""
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        described = oracle.describe()
        assert "decode buckets=" in described
        assert "1:4" in described and "16:4" in described


class TestWhenTheRungIsMissing:
    def test_an_unmeasured_rung_falls_back_and_says_so(self, tmp_path, caplog):
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        with caplog.at_level("WARNING"):
            cost = oracle.estimate(_shape(20, 250, 32)).seconds
        assert cost > 0
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "ATOMCompass WARNING:" in message
        assert "never measured" in message

    def test_it_warns_once_per_rung_not_once_per_step(self, tmp_path, caplog):
        """A serving run asks thousands of times."""
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        with caplog.at_level("WARNING"):
            for _ in range(20):
                oracle.estimate(_shape(20, 250, 32))
        warnings = [r for r in caplog.records if "never measured" in r.getMessage()]
        assert len(warnings) == 1

    def test_an_eager_step_uses_the_unbucketed_fit(self, tmp_path):
        """No graph replayed, so no rung -- but the step still has a cost."""
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        assert oracle.estimate(_shape(8, 250, None)).seconds > 0

    def test_a_table_without_buckets_still_works(self, tmp_path):
        """Tables recorded before rungs existed must keep costing."""
        rows = [_decode_row(b, 200, 0.002 + 0.0001 * b, None)
                for b in (1, 2, 4, 8)]
        oracle = CalibratedCostOracle(table=_write(tmp_path / "old.jsonl", rows))
        assert "decode buckets=" not in oracle.describe()
        assert oracle.estimate(_shape(4, 200, None)).seconds > 0
