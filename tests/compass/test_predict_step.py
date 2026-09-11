"""Reading a shape list a caller wrote.

The CLI's own body is a composition of things tested elsewhere; keying a saved
graph and reading price entries moved to `test_source_oracle.py` with the code.
What is only here is the boundary between JSON a caller wrote and a
`StepShape`, where a wrong answer is quiet: a mistyped field priced as if the
field were absent.
"""

import importlib.util
import pathlib

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "compass" / "predict_step.py")
_SPEC = importlib.util.spec_from_file_location("predict_step", _PATH)
predict_step = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(predict_step)


# -- the shape list ---------------------------------------------------------

def test_a_shape_is_read_with_its_declared_fields():
    shape = predict_step._shape_from(
        {"num_scheduled_tokens": [1, 1], "context_lens": [10, 20],
         "topology": {"tp": 2}, "rank_coords": {"tp": 1},
         "capture_bucket": 4, "produces_output": False})
    assert shape.num_scheduled_tokens == (1, 1)
    assert shape.context_lens == (10, 20)
    assert shape.capture_bucket == 4
    assert shape.produces_output is False


def test_a_shape_missing_lengths_is_refused():
    with pytest.raises(ValueError) as exc:
        predict_step._shape_from({"num_scheduled_tokens": [1]})
    assert "context_lens" in str(exc.value)


def test_a_misspelled_field_is_refused_rather_than_ignored():
    """`capture_buckets` would otherwise be dropped and read as no bucket."""
    with pytest.raises(ValueError) as exc:
        predict_step._shape_from({"num_scheduled_tokens": [1],
                                  "context_lens": [10],
                                  "capture_buckets": 4})
    assert "capture_buckets" in str(exc.value)
