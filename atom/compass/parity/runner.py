# SPDX-License-Identifier: MIT
"""ATOM's own runner, recording each step: what `--runner-qualname` names on a real run.

Kept apart from the package because importing `ModelRunner` needs a driver.
"""

from atom.compass.parity import StepRecording
from atom.model_engine.model_runner import ModelRunner


class RecordingModelRunner(StepRecording, ModelRunner):
    """`ModelRunner`, unchanged, with each forward call recorded first."""
