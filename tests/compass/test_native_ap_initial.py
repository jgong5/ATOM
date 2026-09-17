"""Optional end-to-end reader checks against preserved native P controls."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from atom.compass.core.cost.native_ap_initial import load_initial, validate_initial_heldouts


@pytest.fixture
def actual_initial():
    path = os.environ.get("ATOMCOMPASS_INITIAL_P_ROOT")
    if not path:
        pytest.skip("set ATOMCOMPASS_INITIAL_P_ROOT to preserved native initial-P artifacts")
    root = Path(path)
    handoff_path = root / "consumer/HANDOFF.json"
    handoff = json.loads(handoff_path.read_text())
    initial, inputs = load_initial(dict(path=str(handoff_path), sha256=hashlib.sha256(handoff_path.read_bytes()).hexdigest()),
        scope=handoff["scope"], closed_model_sha256="c56147d848ad09731d52b6814293bf6647eceee52dc35076a20a1ec0fa885100")
    evidence = json.loads((root / "consumer/VALIDATION_EVIDENCE.json").read_text())
    return root, initial, inputs, evidence


def read(pin, role):
    raw = Path(pin["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
    return json.loads(raw)


def test_real_sources_heldouts_and_late_foreign_process(actual_initial):
    _, initial, inputs, evidence = actual_initial
    checks = validate_initial_heldouts(initial, evidence, read,
                                      dict(initial_postprocess_component=initial["component_identity"]))
    assert len(inputs) == 12 and len(checks) == 3
    assert max(c["relative_error"] for c in checks) < .004
    assert initial["source_qualified"] is False


def test_changed_component_and_possible_overlap_refuse(actual_initial):
    _, initial, _, evidence = actual_initial
    with pytest.raises(ValueError, match="frozen identity"):
        validate_initial_heldouts(initial, evidence, read, dict(initial_postprocess_component=dict(sha256="wrong")))

    def overlap(pin, role):
        value = copy.deepcopy(read(pin, role))
        if role == "initial.copy_closeout":
            for process in value["cleanup"]["gpu_after"]["host_accounting"]["rows"]:
                process["start_ticks"] = [1, 1]
        return value

    with pytest.raises(ValueError, match="may overlap"):
        validate_initial_heldouts(initial, evidence, overlap,
                                  dict(initial_postprocess_component=initial["component_identity"]))
