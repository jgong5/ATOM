# SPDX-License-Identifier: MIT
"""The speculative waits the step recorder wraps are methods ATOM's
`tokenIDProcessor` has.

`StepRecording.__init__` looks each `SPEC_WAITS` name up on the runner's
processor, so a name ATOM renamed stops every speculative `compass record` at
its start, after the weights load. The device window tests drive the recorder
over a fake processor, which cannot tell. Nothing here imports
`atom.model_engine.model_runner`: that import runs aiter's architecture probe,
which raises where there is no GPU, so the class is read from source.
"""

import ast
import pathlib

import pytest

from atom.compass.parity import SPEC_WAITS

MODEL_RUNNER = pathlib.Path(__file__).resolve().parents[2] / (
    "atom/model_engine/model_runner.py"
)


def _methods(source: str, name: str) -> set[str]:
    """The names `source`'s top-level class `name` defines a method for."""
    (cls,) = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    return {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize("key,name", SPEC_WAITS)
def test_each_wrapped_wait_is_a_token_id_processor_method(key, name):
    methods = _methods(MODEL_RUNNER.read_text(), "tokenIDProcessor")
    assert name in methods, (
        f"the recorder stamps {key} on tokenIDProcessor.{name}, which "
        f"{MODEL_RUNNER.name} does not define"
    )
