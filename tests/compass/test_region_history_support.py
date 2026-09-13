"""A decode capture extends only the rows and histories it measured."""

from dataclasses import replace

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import SOURCE_27B_TP1_PREFILL_SEQS


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
