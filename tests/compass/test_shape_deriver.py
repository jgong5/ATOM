"""Turning a step shape into the batch a trace needs, and the gap between them.

A ``StepShape`` says what the engine was asked to compute. A ``BatchSpec`` says
what a forward runs. The difference is deployment configuration the shape does
not carry and per-request facts nobody can recover from lengths, and every one
of those is either declared to :class:`ShapeDeriver` or refused by it. These
tests exercise the conversion alone -- no model is built, so the tracer is not
touched.
"""

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.runtime.tracer import DeriveRefusal, ShapeDeriver

#: The 27B deployment's declared shape rules, which is what a caller supplies.
DECLARED = dict(block_size=16, max_model_len=262144, position_rows=3)


def deriver(**kw):
    """A deriver with no tracer. ``spec_for`` never needs one."""
    return ShapeDeriver(None, **{**DECLARED, **kw})


def shape(queries, contexts, *, prefill=0, produces=True, bucket=None):
    return StepShape(
        num_scheduled_tokens=tuple(queries),
        context_lens=tuple(contexts),
        num_prefill_tokens=prefill,
        topology={"tp": 1}, rank_coords={"tp": 0},
        capture_bucket=bucket, compiled=None, produces_output=produces)


def test_a_decode_shape_becomes_a_decode_spec():
    spec = deriver().spec_for(shape([1] * 4, [1151] * 4, bucket=32))
    assert spec.kind == "decode"
    assert spec.query_lens == (1, 1, 1, 1)
    assert spec.context_lens == (1151,) * 4
    assert spec.capture_bucket == 32
    assert spec.produces_output() is True


def test_declared_deployment_rules_reach_the_spec():
    """None of these is derivable from a shape, and each changes the graph."""
    spec = deriver().spec_for(shape([1] * 4, [1151] * 4))
    assert spec.block_size == 16
    assert spec.max_model_len == 262144
    assert spec.position_rows == 3


def test_prompt_lens_are_left_to_the_block_policy():
    """Not invented here: ``admitted_lens`` already falls back to cached_lens."""
    spec = deriver().spec_for(shape([1] * 2, [100, 200]))
    assert spec.prompt_lens is None
    assert tuple(spec.admitted_lens) == (99, 199)


def test_a_prefill_shape_becomes_a_prefill_spec():
    spec = deriver().spec_for(shape([4096], [4096], prefill=4096))
    assert spec.kind == "prefill"
    assert spec.produces_output() is True


def test_a_chunk_that_samples_nothing_is_carried_through():
    spec = deriver().spec_for(
        shape([4096], [8192], prefill=4096, produces=False))
    assert spec.produces_output() is False


def test_a_prefill_that_does_not_say_is_refused():
    """Whether a chunk is a request's last decides if the LM head runs at all."""
    with pytest.raises(DeriveRefusal) as exc:
        deriver().spec_for(shape([4096], [8192], prefill=4096, produces=None))
    assert "produces output" in str(exc.value)


def test_a_decode_that_does_not_say_is_not_refused():
    """A decode always samples, so nothing is being guessed."""
    spec = deriver().spec_for(shape([1] * 4, [1151] * 4, produces=None))
    assert spec.produces_output() is True


def test_a_context_past_the_declared_maximum_is_refused():
    with pytest.raises(DeriveRefusal) as exc:
        deriver(max_model_len=4096).spec_for(shape([1], [8192]))
    assert "max_model_len" in str(exc.value)


def test_mismatched_rows_are_refused():
    with pytest.raises(DeriveRefusal) as exc:
        deriver().spec_for(shape([1, 1, 1], [10, 20]))
    assert "row is a pair" in str(exc.value)


def test_an_empty_batch_is_refused():
    with pytest.raises(DeriveRefusal):
        deriver().spec_for(shape([], []))


def test_an_impossible_batch_is_refused_by_the_spec_itself():
    """A query longer than its own context: `BatchSpec.validate` catches it."""
    with pytest.raises(ValueError):
        deriver().spec_for(shape([64], [16], prefill=64))


def test_the_spec_records_what_was_declared():
    spec = deriver().spec_for(shape([1] * 4, [1151] * 4))
    assert spec.notes["declared"]["position_rows"] == 3
    assert "ShapeDeriver" in spec.notes["why"]
