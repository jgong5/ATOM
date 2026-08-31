"""Fitting a cost model to steps that were actually timed.

The failure mode these guard against is not a bad fit — it is a confident one.
Every bug this oracle has had returned a precise number that was wrong, and
none of them raised.
"""

import json

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.calibrated import CalibratedCostOracle, _least_squares


def write_table(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for tokens, context, seconds, prefill in rows:
            fh.write(json.dumps({
                "seconds": seconds,
                "num_scheduled_tokens": [tokens],
                "context_lens": [context],
                "num_prefill_tokens": tokens if prefill else 0,
            }) + "\n")
    return str(path)


def decode(batch=1, context=128):
    return StepShape(
        num_scheduled_tokens=tuple([1] * batch),
        context_lens=tuple([context] * batch),
        num_prefill_tokens=0,
    )


def prefill(tokens):
    return StepShape(
        num_scheduled_tokens=(tokens,), context_lens=(0,),
        num_prefill_tokens=tokens,
    )


class TestOutlierRejection:
    """Triton autotunes per shape, not once per process.

    So a calibration sweep built from deliberately varied shapes pays a one-off
    benchmarking cost on many of its own samples. In the real table one prefill
    row sat at 0.13 s where its neighbour at a larger size took 0.036 s.
    """

    @staticmethod
    def _linear(n, slope=1e-5, intercept=0.005):
        return [[1.0, float(i), float(i * i)] for i in n], [
            intercept + slope * i for i in n
        ]

    def test_a_clean_fit_drops_nothing(self):
        rows, targets = self._linear(range(100, 1100, 100))
        coeffs, dropped = _least_squares(rows, targets)
        assert coeffs is not None
        assert dropped == 0

    def test_one_contaminated_sample_is_dropped(self):
        sizes = list(range(100, 1600, 100))
        rows, targets = self._linear(sizes)
        targets[3] *= 30.0                      # an autotuning launch
        coeffs, dropped = _least_squares(rows, targets)
        assert dropped >= 1
        # And the fit now describes the clean points rather than splitting the
        # difference with the contaminated one.
        predicted = sum(c * f for c, f in zip(coeffs, rows[3]))
        assert predicted == pytest.approx(self._linear([sizes[3]])[1][0], rel=0.2)

    def test_a_fit_too_small_to_survive_dropping_keeps_everything(self):
        """With barely enough points, discarding one leaves nothing to fit.

        Better a fit influenced by an outlier than coefficients derived from an
        underdetermined system, which look like a model and predict nothing.
        """
        rows, targets = self._linear([100, 200, 300])
        coeffs, dropped = _least_squares(rows, targets)
        assert coeffs is not None and dropped == 0

    def test_fewer_samples_than_coefficients_is_refused(self):
        rows, targets = self._linear([100, 200])
        coeffs, dropped = _least_squares(rows, targets)
        assert coeffs is None and dropped == 0


class TestOracle:
    def test_decode_cost_grows_with_context(self, tmp_path):
        """Decode is bandwidth-bound in the KV history it reads."""
        rows = [(1, ctx, 0.001 + ctx * 1e-6, False) for ctx in range(64, 2048, 64)]
        oracle = CalibratedCostOracle(write_table(tmp_path / "t.jsonl", rows))
        assert oracle.estimate(decode(context=1024)).seconds > \
            oracle.estimate(decode(context=128)).seconds

    def test_prefill_cost_grows_with_tokens(self, tmp_path):
        rows = [(n, 0, 0.005 + n * 1e-5, True) for n in range(128, 4096, 128)]
        oracle = CalibratedCostOracle(write_table(tmp_path / "t.jsonl", rows))
        assert oracle.estimate(prefill(2048)).seconds > \
            oracle.estimate(prefill(256)).seconds

    def test_a_kind_never_measured_is_refused_not_invented(self, tmp_path):
        """Returning the mean of no samples is zero.

        That is what made TTFT come back as 0 ms against a real 7.6 s: a
        confident, precise, entirely fictional answer.
        """
        rows = [(1, ctx, 0.001, False) for ctx in range(64, 1024, 64)]
        oracle = CalibratedCostOracle(write_table(tmp_path / "t.jsonl", rows))
        with pytest.raises(ValueError, match="no prefill measurements"):
            oracle.estimate(prefill(512))

    def test_a_prediction_is_never_negative(self, tmp_path):
        """A fit extrapolates, and a negative duration runs the clock backwards."""
        rows = [(n, 0, 0.005 + n * 1e-5, True) for n in range(1024, 8192, 512)]
        oracle = CalibratedCostOracle(write_table(tmp_path / "t.jsonl", rows))
        assert oracle.estimate(prefill(1)).seconds > 0.0

    def test_an_empty_table_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no usable measurements"):
            CalibratedCostOracle(write_table(tmp_path / "t.jsonl", []))

    def test_describe_reports_dropped_samples(self, tmp_path):
        """A fit that discarded evidence must not describe itself like one that
        kept it."""
        rows = [(n, 0, 0.005 + n * 1e-5, True) for n in range(128, 4096, 128)]
        rows[5] = (rows[5][0], 0, 5.0, True)
        oracle = CalibratedCostOracle(write_table(tmp_path / "t.jsonl", rows))
        assert "dropped" in oracle.describe()


class TestExtrapolationIsAnnounced:
    """A fitted model answers anything, including what it has no evidence for.

    This caused the same error twice in two dimensions: a prefill model fitted
    to 1753-16370 tokens asked about 520, and a decode model fitted to batch
    sizes 1-4 asked about 8. Neither said anything; both were simply wrong.
    """

    @staticmethod
    def _decode_table(tmp_path, batches):
        rows = []
        for b in batches:
            for ctx in range(64, 1024, 64):
                rows.append((b, ctx, 0.003 + b * 1e-4 + ctx * 1e-7, False))
        path = tmp_path / "t.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for b, ctx, seconds, _ in rows:
                fh.write(json.dumps({
                    "seconds": seconds,
                    "num_scheduled_tokens": [1] * b,
                    "context_lens": [ctx // b] * b,
                    "num_prefill_tokens": 0,
                }) + "\n")
        return str(path)

    def test_inside_the_calibrated_range_is_silent(self, tmp_path, caplog):
        oracle = CalibratedCostOracle(self._decode_table(tmp_path, [1, 2, 4, 8]))
        caplog.clear()
        oracle.estimate(decode(batch=4, context=100))
        assert not [r for r in caplog.records if "extrapolation" in r.message]

    def test_outside_it_warns(self, tmp_path, caplog):
        import logging

        oracle = CalibratedCostOracle(self._decode_table(tmp_path, [1, 2, 4]))
        with caplog.at_level(logging.WARNING):
            oracle.estimate(decode(batch=64, context=100))
        assert any("extrapolation" in r.message for r in caplog.records)

    def test_repetition_adds_no_further_warnings(self, tmp_path, caplog):
        """A serving run asks this thousands of times.

        More than one feature can be out of range at once -- here both the batch
        size and the total context are -- so the invariant is not "one warning"
        but "no more after the first time each is seen".
        """
        import logging

        oracle = CalibratedCostOracle(self._decode_table(tmp_path, [1, 2, 4]))
        with caplog.at_level(logging.WARNING):
            oracle.estimate(decode(batch=64, context=100))
            first = len([r for r in caplog.records if "extrapolation" in r.message])
            for _ in range(50):
                oracle.estimate(decode(batch=64, context=100))
            after = len([r for r in caplog.records if "extrapolation" in r.message])
        assert first >= 1
        assert after == first
