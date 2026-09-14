"""A default-looking artifact cannot silently use live assistant responses."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "opening_exporter_policy", Path(__file__).resolve().parents[2] / "scripts/compass/export_aiperf_opening.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class Environment:
    model_fields = {name: SimpleNamespace(default=value) for name, value in {
        "WEKA_LIVE_ASSISTANT_RESPONSES": False,
        "WEKA_SPLIT_FLATTENED_AGENTS": True,
        "WEKA_TOOL_SHAPED_MESSAGES": False,
        "WEKA_SEAM_MAX_GAP_SECONDS": 3600.,
    }.items()}

    def __init__(self):
        for name, field in self.model_fields.items():
            setattr(self, name, field.default)


def test_effective_policy_is_recorded_against_pinned_defaults():
    policy = exporter.weka_reconstruction_policy(Environment())
    assert policy["defaults_verified"] is True
    assert policy["effective"] == policy["pinned_defaults"]
    assert policy["effective"]["WEKA_LIVE_ASSISTANT_RESPONSES"] is False


@pytest.mark.parametrize("name,value", [("WEKA_LIVE_ASSISTANT_RESPONSES", True),
                                      ("WEKA_SPLIT_FLATTENED_AGENTS", False),
                                      ("WEKA_SEAM_MAX_GAP_SECONDS", 10.)])
def test_nondefault_reconstruction_is_refused_before_export(name, value):
    environment = Environment()
    setattr(environment, name, value)
    with pytest.raises(ValueError, match=name):
        exporter.weka_reconstruction_policy(environment)
