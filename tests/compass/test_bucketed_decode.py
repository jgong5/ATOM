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
from atom.compass.core.cost.calibrated import (
    CalibratedCostOracle, _decode_bucket_features)


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

    def test_within_a_rung_only_the_history_matters(self, tmp_path):
        """Eight sequences of 500 and sixteen of 250 read the same history.

        This test used to say "batch 12 pads to 16, so it costs what 16 costs"
        and assert `at_12 == approx(at_16) or at_12 > 0`, which any positive
        number satisfies -- it could not fail. Removing the escape clause showed
        the premise was also wrong: twelve sequences of 250 carry 3000 tokens of
        history and sixteen carry 4000, so the model separates them, and should.
        What the rung fixes is the row count; what varies inside it is the KV
        the step reads, which is the feature.
        """
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        wide = oracle.estimate(_shape(16, 250, 16)).seconds
        deep = oracle.estimate(_shape(8, 500, 16)).seconds
        assert wide == pytest.approx(deep, rel=1e-6)

    def test_less_history_in_the_same_rung_costs_less(self, tmp_path):
        oracle = CalibratedCostOracle(table=_two_rungs(tmp_path / "t.jsonl"))
        assert (oracle.estimate(_shape(12, 250, 16)).seconds
                < oracle.estimate(_shape(16, 250, 16)).seconds)

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


class TestAWorkloadFitsThroughTheClient:
    """A declared workload is posted all at once or it deadlocks.

    The server holds every declared request until all of them have arrived, so
    every one must be in flight at the same time. A client with fewer
    connections than requests waits for responses the server will not produce
    until the requests it is still holding back have been posted. It resolves
    only when the arrival barrier times out, and the run that follows is not
    the declared arrival process -- requests enter as earlier ones finish.

    That happened on a 300-request workload against a 64-connection pool. The
    client reported "0 failed", the server logged the timeout and called its own
    latencies invalid, and the result was read as a measurement for a day. These
    tests are the boundary that was missing.
    """

    def _workers(self, count, pace=False):
        """What the client would open for a workload of `count`."""
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "replay_mod",
            Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return len(range(count)), module.MAX_IN_FLIGHT

    def test_every_request_gets_a_connection(self):
        workers, _ = self._workers(300)
        assert workers == 300

    def test_the_bound_is_above_the_workloads_in_use(self):
        """300 was the one that failed; the bound has to clear it and the
        831-request corpus slice as well."""
        _, cap = self._workers(1)
        assert cap >= 831

    def test_past_the_bound_it_refuses_rather_than_deadlocks(self):
        """Posting fewer than were declared is the deadlock. Refusing is the
        only other honest option until a bulk submission exists."""
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "replay_mod2",
            Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = (Path(__file__).resolve().parents[2]
                  / "scripts/compass/replay.py").read_text()
        # The refusal is a SystemExit on the size check, not a silent clamp.
        assert "MAX_IN_FLIGHT" in source
        assert "min(64, len(workload))" not in source
        assert "raise SystemExit" in source


class TestPaddingIsAFeature:
    """Total context makes [10k,10k,10k] and [1k,1k,28k] the same step.

    They are not, if attention reads a rectangle sized by the longest sequence.
    Real decode batches run 2.8 to 3.9 times ragged while every calibration
    batch ran sequences of one length, so the padding term was zero in every
    sample: a rank deficiency, not a coverage gap, which is why widening the
    context range did nothing for a 21-23% under-prediction.
    """

    def _shape(self, lengths, bucket):
        return StepShape(
            num_scheduled_tokens=tuple([1] * len(lengths)),
            context_lens=tuple(lengths),
            num_prefill_tokens=0,
            capture_bucket=bucket,
        )

    def test_a_ragged_batch_is_told_from_a_uniform_one(self):
        from atom.compass.core.cost.calibrated import _decode_bucket_features

        flat = _decode_bucket_features(self._shape([10000] * 3, 4))
        tall = _decode_bucket_features(self._shape([1000, 1000, 28000], 4))
        assert flat[1] == tall[1], "same total history"
        assert tall[2] > flat[2], "different padded read"

    def test_a_uniform_batch_has_no_padding(self):
        from atom.compass.core.cost.calibrated import _decode_bucket_features

        assert _decode_bucket_features(self._shape([500] * 8, 8))[2] == 0.0

    def test_padding_is_the_rectangle_beyond_the_histories(self):
        from atom.compass.core.cost.calibrated import _decode_bucket_features

        got = _decode_bucket_features(self._shape([1000, 3000], 2))
        assert got[1] == 4000.0          # 1000 + 3000
        assert got[2] == 2000.0          # 2 * 3000 - 4000


class TestAFitDropsWhatItCannotIdentify:
    """A rung calibrated on uniform batches has a dead padding column.

    Fitting it anyway is a singular system; refusing loses the rung entirely.
    Dropping the column fits what the samples can identify and reports the
    coefficient as zero, which is what "this evidence says nothing about it"
    should look like.
    """

    def test_a_constant_column_is_dropped(self):
        from atom.compass.core.cost.calibrated import _drop_constant_columns

        rows = [[1.0, 10.0, 0.0], [1.0, 20.0, 0.0], [1.0, 30.0, 0.0]]
        kept_rows, kept = _drop_constant_columns(rows)
        assert kept == [0, 1]
        assert kept_rows == [[1.0, 10.0], [1.0, 20.0], [1.0, 30.0]]

    def test_a_varying_column_is_kept(self):
        from atom.compass.core.cost.calibrated import _drop_constant_columns

        rows = [[1.0, 10.0, 0.0], [1.0, 20.0, 5.0], [1.0, 30.0, 9.0]]
        _, kept = _drop_constant_columns(rows)
        assert kept == [0, 1, 2]

    def test_the_intercept_survives_being_constant(self):
        from atom.compass.core.cost.calibrated import _drop_constant_columns

        rows = [[1.0, 10.0], [1.0, 20.0], [1.0, 30.0]]
        _, kept = _drop_constant_columns(rows)
        assert 0 in kept

    def test_a_dropped_column_comes_back_as_zero(self):
        """So a caller's feature vector still lines up with the coefficients."""
        from atom.compass.core.cost.calibrated import _least_squares

        rows = [[1.0, float(x), 0.0] for x in (1, 2, 3, 4, 5, 6)]
        targets = [1.0 + 2.0 * x for x in (1, 2, 3, 4, 5, 6)]
        coeffs, _ = _least_squares(rows, targets)
        assert len(coeffs) == 3
        assert coeffs[2] == 0.0
        assert coeffs[1] == pytest.approx(2.0, rel=1e-6)

class TestPaddingIsMeasuredAgainstTheRung:
    """A replay runs the padded rung, so the rectangle it reads is the rung's.

    The padding term was computed over the sequences present, which equals the
    rung exactly when a batch fills it -- and a sweep requesting 2, 4, 8, 16
    sequences always fills it. A serving run does not: as requests finish,
    batches drift below their rung and stay there. On a 27B run the rung-16
    batches were 9-10 and the padding as coded was 5.59 times too small, which
    is the -22.64% that two rounds of sweep widening could not shift.
    """

    @staticmethod
    def _shape(contexts, bucket):
        return StepShape(
            num_scheduled_tokens=tuple(1 for _ in contexts),
            context_lens=tuple(contexts),
            num_prefill_tokens=0,
            capture_bucket=bucket,
        )

    def test_a_batch_below_its_rung_pays_the_rung_s_rectangle(self):
        f = _decode_bucket_features(self._shape([100, 100, 200], bucket=8))
        # 8 rows of 200 read, 400 of which is real history.
        assert f[2] == pytest.approx(8 * 200 - 400)

    def test_a_batch_that_fills_its_rung_is_unchanged(self):
        """Which is why a sweep never saw this."""
        ctx = [100, 100, 100, 200]
        f = _decode_bucket_features(self._shape(ctx, bucket=4))
        assert f[2] == pytest.approx(4 * 200 - sum(ctx))

    def test_a_uniform_batch_filling_its_rung_still_has_no_padding(self):
        """The zero that makes the column droppable where nothing varies."""
        f = _decode_bucket_features(self._shape([256] * 4, bucket=4))
        assert f[2] == pytest.approx(0.0)

    def test_an_eager_step_falls_back_to_its_batch(self):
        """No bucket means nothing was replayed, so the batch is what ran."""
        f = _decode_bucket_features(self._shape([100, 300], bucket=None))
        assert f[2] == pytest.approx(2 * 300 - 400)
