"""A decode capture extends only the rows and histories it measured."""

from dataclasses import replace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    SOURCE_27B_TP1_PREFILL_SEQS, SOURCE_27B_TP1_HISTORY_64K,
    SOURCE_27B_TP1_HISTORY_DELTA)


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
