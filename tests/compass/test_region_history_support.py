"""A decode capture extends only the rows and histories it measured."""

from dataclasses import replace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    SOURCE_27B_TP1_PREFILL_SEQS, SOURCE_27B_TP1_HISTORY_64K,
    SOURCE_27B_TP1_HISTORY_DELTA, SOURCE_27B_TP1_HISTORY_2M, Measured)


def _model():
    # Example source support: one long row, and a mixed short/long pair.
    return replace(
        SOURCE_27B_TP1_PREFILL_SEQS,
        decode_context_cells=(
            ((1, False), (1025, 65600, 65600)),
            ((2, False), (513, 65600, 66176)),
        ),
    )


def _shape(histories, bucket):
    return StepShape(num_scheduled_tokens=(1,) * len(histories),
                     context_lens=tuple(histories), capture_bucket=bucket)


def test_source_extension_covers_only_declared_cells():
    model = _model()
    assert model.refusal(_shape([64000], 1)) is None
    assert model.refusal(_shape([641, 63745], 2)) is None
    assert "measured only over [1025, 1152]" in model.refusal(
        _shape([64000] * 4, 4))


def test_mixed_source_does_not_claim_two_long_rows():
    assert "summed history" in _model().refusal(_shape([64000, 64000], 2))


def test_history_support_keeps_both_boundaries_closed():
    model = _model()
    assert model.refusal(_shape([1025], 1)) is None
    assert model.refusal(_shape([65600], 1)) is None
    assert model.refusal(_shape([1024], 1)) is not None
    assert model.refusal(_shape([65601], 1)) is not None


def test_new_support_does_not_move_published_profile():
    assert SOURCE_27B_TP1_PREFILL_SEQS.refusal(
        _shape([641, 63745], 2)) is not None


def test_measured_source_covers_unchanged_development_decode_domain():
    model = SOURCE_27B_TP1_HISTORY_64K
    for short in range(641, 661):
        assert model.refusal(_shape([short, short + 63104], 2)) is None
    for long in range(63765, 64140):
        assert model.refusal(_shape([long], 1)) is None
    assert model.refusal(_shape([64000] * 4, 4)) is not None


def test_native_final_prefill_anchor_keeps_middle_chunk_separate():
    model = SOURCE_27B_TP1_HISTORY_64K
    final = StepShape(num_scheduled_tokens=(16384,), context_lens=(32768,),
                      num_prefill_tokens=16384, produces_output=True)
    middle = replace(final, context_lens=(16384,), produces_output=False)
    assert model.refusal(final) is None
    assert model.breakdown(final)["<postprocess>"] > 0
    assert model.breakdown(middle)["<postprocess>"] == 0


def test_history_delta_preserves_native_coefficients_and_original_answers():
    original, model = SOURCE_27B_TP1_HISTORY_64K, SOURCE_27B_TP1_HISTORY_DELTA
    assert model.prepare_decode_cells == original.prepare_decode_cells
    for n in range(1, 33):
        for context in (1025, 1032, 1152):
            shape = _shape([context] * n, model.bucket_for(n))
            assert model.breakdown(shape) == original.breakdown(shape)
            assert model.band(shape) == original.band(shape)


def test_only_the_history_delta_is_interpolated_after_the_native_plateau():
    model = SOURCE_27B_TP1_HISTORY_DELTA
    native = dict(model.prepare_decode_cells)[(2, False)].seconds
    delta = dict(model.prepare_decode_history_deltas)[(2, False)].seconds
    # Sum 197760 is halfway from the old plateau's 2304 to this cell's
    # measured 393216 sum limit; both request histories remain supported.
    middle = _shape([98880, 98880], 2)
    limit = _shape([196608, 196608], 2)
    assert model.breakdown(middle)["<prepare>"] == pytest.approx(
        native + 0.5 * delta)
    assert model.breakdown(limit)["<prepare>"] == pytest.approx(native + delta)


def test_delta_cannot_clip_unsupported_histories_into_coverage():
    model = SOURCE_27B_TP1_HISTORY_DELTA
    with pytest.raises(ValueError, match="summed history"):
        model.seconds(_shape([196608] * 32, 32))
    assert model.refusal(_shape([128], 1)) is None
    assert model.refusal(_shape([127], 1)) is not None
    assert model.refusal(_shape([196609], 1)) is not None


def _balanced(total, n):
    return [total // n + (i < total % n) for i in range(n)]


def test_high_history_extension_preserves_all_old_segments_and_bands():
    old, extended = SOURCE_27B_TP1_HISTORY_DELTA, SOURCE_27B_TP1_HISTORY_2M
    assert extended.prepare_decode_cells == old.prepare_decode_cells
    assert extended.prepare_decode_history_deltas == old.prepare_decode_history_deltas
    assert extended.prepare_prefill_cells == old.prepare_prefill_cells
    for n in range(1, 33):
        bucket = old.bucket_for(n)
        cell = (bucket, bucket != n)
        lo, hi, limit = dict(old.decode_context_cells)[cell]
        upper = min(limit, n * hi)
        for total in {n * lo, n * 1025, n * 1032, n * 1152,
                      (n * 1152 + upper) // 2, upper}:
            shape = _shape(_balanced(total, n), bucket)
            assert old.refusal(shape) is None
            assert extended.refusal(shape) is None
            assert extended._decode_prepare(shape) == old._decode_prepare(shape)
            assert extended.breakdown(shape) == old.breakdown(shape)
            assert extended.band(shape) == old.band(shape)


def test_high_history_source_covers_c8_refusal_and_keeps_new_bound_closed():
    histories = [83969, 170369, 51329, 62465, 110849, 112129, 33665, 62337,
                 28545, 31681, 32769, 69185, 2945, 47617, 34881, 48001,
                 63297, 17537, 2945, 34945, 26241, 27329, 17729, 116609,
                 78785, 2945, 112961, 2945, 67009, 34881, 26305, 2945]
    shape = _shape(histories, 32)
    assert sum(histories) == 1618144
    assert "summed history" in SOURCE_27B_TP1_HISTORY_DELTA.refusal(shape)
    model = SOURCE_27B_TP1_HISTORY_2M
    assert model.refusal(shape) is None
    assert model.seconds(shape) > 0
    for n in (9, 15, 16, 17, 21, 31, 32):
        upper = min(2097152, n * 196608)
        assert model.refusal(_shape(_balanced(upper, n), model.bucket_for(n))) is None
    assert "summed history" in model.refusal(_shape(_balanced(2097153, 32), 32))
    with pytest.raises(ValueError, match="summed history"):
        model.seconds(_shape(_balanced(2097153, 32), 32))
    assert model.refusal(_shape([196609] + [65536] * 31, 32)) is not None
    assert model.refusal(_shape([127] + [65536] * 31, 32)) is not None
    assert dict(model.decode_context_cells)[(8, False)] == (128, 196608, 1572864)


def test_history_continuation_is_anchored_at_the_old_endpoint():
    # A nonzero continuation verifies the general contract even though the
    # independently measured 2M source chose a conservative zero increment.
    extra = Measured(4e-6, -2e-6, 6e-6, 10, "independent test endpoint")
    model = replace(SOURCE_27B_TP1_HISTORY_2M,
                    prepare_decode_history_extensions=(((32, False), (1572864, extra)),))
    old_endpoint = model._decode_prepare(_shape(_balanced(1572864, 32), 32))
    midpoint = model._decode_prepare(_shape(_balanced(1835008, 32), 32))
    endpoint = model._decode_prepare(_shape(_balanced(2097152, 32), 32))
    assert midpoint.seconds == pytest.approx(old_endpoint.seconds + 2e-6)
    assert midpoint.low == pytest.approx(old_endpoint.low - 1e-6)
    assert midpoint.high == pytest.approx(old_endpoint.high + 3e-6)
    assert endpoint.seconds == pytest.approx(old_endpoint.seconds + 4e-6)
