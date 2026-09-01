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


class TestEventDraining:
    """Timed steps are written once the device has finished them, not before.

    Draining by `query()` rather than `synchronize()` is the whole point: a
    host-side sync on every step destroyed the host/device overlap a serving
    loop runs on, making the measured run 33% slower than the same run
    unmeasured (4.33 ms per output token against 3.26 ms). The table then
    described a machine that only existed while being measured.
    """

    class FakeEvent:
        def __init__(self, ready=True, ms=1.0):
            self._ready, self._ms = ready, ms

        def query(self):
            return self._ready

        def elapsed_time(self, other):
            return other._ms

    @staticmethod
    def _runner():
        import collections

        from atom.compass.config import CompassConfig
        from atom.compass.runtime.runner import CompassModelRunner

        stub = CompassModelRunner.__new__(CompassModelRunner)
        stub.__dict__["_compass_config_cache"] = CompassConfig(
            enabled=True, mode="measure", measure_out="t.jsonl",
        )
        stub._pending = collections.deque()
        stub._measured_steps = 0
        stub._measured_by_kind = {}
        stub._written = []
        stub._record_measurement = (
            lambda shape, seconds, gap=None: stub._written.append(
                (shape, seconds, gap)
            )
        )
        return stub

    def test_an_unfinished_step_is_not_written_yet(self):
        stub = self._runner()
        stub._pending.append(
            (decode(), self.FakeEvent(), self.FakeEvent(ready=False), None))
        stub._drain_pending()
        assert stub._written == []
        assert len(stub._pending) == 1

    def test_finished_steps_are_written_in_order(self):
        stub = self._runner()
        for ms in (2.0, 4.0, 8.0):
            stub._pending.append(
                (decode(), self.FakeEvent(), self.FakeEvent(ms=ms), None)
            )
        stub._drain_pending()
        assert [s for _, s, _g in stub._written] == [0.002, 0.004, 0.008]
        assert not stub._pending

    def test_draining_stops_at_the_first_unfinished_step(self):
        """Order matters: a later step must not be written before an earlier
        one, or the table's rows stop corresponding to the run's sequence."""
        stub = self._runner()
        stub._pending.append(
            (decode(), self.FakeEvent(), self.FakeEvent(ms=2.0), None))
        stub._pending.append(
            (decode(), self.FakeEvent(), self.FakeEvent(ready=False), None))
        stub._pending.append(
            (decode(), self.FakeEvent(), self.FakeEvent(ms=8.0), None))
        stub._drain_pending()
        assert [s for _, s, _g in stub._written] == [0.002]
        assert len(stub._pending) == 2

    def test_warmup_is_counted_per_kind(self):
        """Prefill happens a handful of times in a whole run, so a warmup
        counted in total steps discards every prefill sample there is."""
        from atom.compass.config import CompassConfig

        stub = self._runner()
        stub.__dict__["_compass_config_cache"] = CompassConfig(
            enabled=True, mode="measure", measure_out="t.jsonl",
            measure_warmup_steps=1,
        )
        stub._count_and_record(prefill(128), 0.5)     # first prefill: dropped
        stub._count_and_record(decode(), 0.001)       # first decode: dropped
        stub._count_and_record(prefill(256), 0.05)    # kept
        stub._count_and_record(decode(), 0.002)       # kept
        assert [s for _, s, _g in stub._written] == [0.05, 0.002]


class TestEmpiricalOracle:
    """Answering from nearby measurements instead of a fitted form.

    Its diagnostic value came first: asked about the shape the calibrated oracle
    was getting wrong, it returned 3.84 ms where the fit returned 3.83 ms. Two
    methods with opposite failure modes agreeing meant the fit was faithful to
    its data and the data was wrong — which is what led to F9.
    """

    @staticmethod
    def _oracle(tmp_path, rows):
        from atom.compass.core.cost.empirical import EmpiricalCostOracle

        path = tmp_path / "t.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for batch, ctx, seconds in rows:
                fh.write(json.dumps({
                    "seconds": seconds,
                    "num_scheduled_tokens": [1] * batch,
                    "context_lens": [ctx // batch] * batch,
                    "num_prefill_tokens": 0,
                }) + "\n")
        return EmpiricalCostOracle(str(path), neighbours=3)

    def test_an_exact_match_answers_exactly(self, tmp_path):
        rows = [(4, ctx, 0.001 + ctx * 1e-6) for ctx in range(400, 2000, 400)]
        oracle = self._oracle(tmp_path, rows)
        got = oracle.estimate(decode(batch=4, context=200)).seconds  # ctx total 800
        assert got == pytest.approx(0.001 + 800 * 1e-6, rel=1e-6)

    def test_it_interpolates_between_neighbours(self, tmp_path):
        rows = [(4, ctx, ctx * 1e-6) for ctx in range(400, 4000, 400)]
        oracle = self._oracle(tmp_path, rows)
        got = oracle.estimate(decode(batch=4, context=250)).seconds  # total 1000
        assert 0.0008 < got < 0.0012

    def test_a_kind_never_measured_is_refused(self, tmp_path):
        oracle = self._oracle(tmp_path, [(4, c, 0.001) for c in range(400, 2000, 400)])
        with pytest.raises(ValueError, match="no prefill measurements"):
            oracle.estimate(prefill(512))

    def test_far_outside_the_data_it_warns(self, tmp_path, caplog):
        """Nearest-neighbour degrades to "the closest edge" rather than
        diverging, which is gentler than extrapolating a line but still an
        answer given without evidence."""
        import logging

        oracle = self._oracle(tmp_path, [(4, c, 0.001) for c in range(400, 2000, 400)])
        with caplog.at_level(logging.WARNING):
            oracle.estimate(decode(batch=4, context=100000))
        assert any("outside the measured range" in r.message for r in caplog.records)

    def test_features_are_standardised_before_distances_are_taken(self, tmp_path):
        """Context runs to thousands and batch size to single digits.

        Unstandardised, every neighbour would be chosen by context alone and
        batch size would carry no weight at all.
        """
        rows = [(1, 1000, 0.010), (8, 1000, 0.020)]
        rows += [(1, 1200, 0.010), (8, 1200, 0.020)]
        oracle = self._oracle(tmp_path, rows)
        near_one = oracle.estimate(decode(batch=1, context=1100)).seconds
        near_eight = oracle.estimate(decode(batch=8, context=137)).seconds
        assert near_one < near_eight


def test_every_compass_warning_is_greppable():
    """Warnings must announce themselves in the message, not rely on the level.

    ATOM logs as "[atom.compass.x 00:00:00] ..." and never names the level, so
    anything scanning captured output for warnings has to match on the text.
    `compass/validate.py` does exactly that; a warning added without the prefix
    would be silently invisible in the one workflow that runs all the phases.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "atom" / "compass"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "warning"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            message = node.args[0].value
            if isinstance(message, str) and not message.startswith(
                "ATOMCompass WARNING:"
            ):
                offenders.append(f"{path.name}:{node.lineno} {message[:50]!r}")

    assert not offenders, (
        "these logger.warning calls will not be surfaced by validate.py:\n  "
        + "\n  ".join(offenders)
    )


class TestWarningsReachTheUser:
    """`validate.py` captures each phase, so warnings must be extracted from it.

    Capturing is right — an engine start-up is thousands of lines and none of
    them are the point — but it swallowed the warnings too, and those *are* the
    point. The extrapolation warning exists so a bad number announces itself
    rather than being read off the table as fact; swallowed, it was inert in the
    one workflow that runs all four phases.
    """

    @staticmethod
    def _validate_module():
        import importlib.util
        import pathlib

        path = (pathlib.Path(__file__).resolve().parents[2]
                / "scripts" / "compass" / "validate.py")
        spec = importlib.util.spec_from_file_location("compass_validate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_warning_on_stderr_is_surfaced(self, capsys):
        module = self._validate_module()
        line = "[atom.compass.core.cost.calibrated 00:00:00] " + module.MARKER \
            + " costing a decode step outside the calibrated range"
        module._run(["python", "-c", f"import sys; sys.stderr.write({line!r})"])
        assert "outside the calibrated range" in capsys.readouterr().out

    def test_ordinary_output_is_not_surfaced(self, capsys):
        """Only warnings, or the signal drowns in engine start-up noise."""
        module = self._validate_module()
        module._run(["python", "-c",
                     "print('ATOMCompass active: mode=predict oracle=X')"])
        assert capsys.readouterr().out == ""

    def test_the_same_warning_is_printed_once(self, capsys):
        """A serving run emits per-step; the phase log holds many copies."""
        module = self._validate_module()
        line = module.MARKER + " costing a decode step outside the range"
        module._run(["python", "-c",
                     f"print({line!r}); print({line!r}); print({line!r})"])
        assert capsys.readouterr().out.count("costing a decode step") == 1
