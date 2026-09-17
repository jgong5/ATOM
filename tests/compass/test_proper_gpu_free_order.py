"""The real device-free validator runs only after all planned outputs exist."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compass"))
spec = importlib.util.spec_from_file_location("proper_gpu_free_order", ROOT / "scripts/compass/cc_traces_proper.py")
proper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = proper
spec.loader.exec_module(proper)


@pytest.mark.parametrize("repeats,concurrency", [(1, 1), (3, 1), (3, 3)])
def test_real_built_side_produces_every_repeat_before_gpu_free_audit(tmp_path, monkeypatch, repeats, concurrency):
    monkeypatch.setattr(proper.core, "read_pinned", lambda _: {"native_engine_args": []})
    case = {"plan": {"path": "/frozen/plan.json", "sha256": "frozen"},
            "case_id": "aiperf_proper_order", "repeats": repeats,
            "modelled_concurrency": concurrency,
            "purpose": "acceptance" if repeats == 3 else "diagnostic",
            "workload": "/frozen/plan.json", "cache_policy": {}}
    built = proper.build_steps(case, tmp_path, port=8850, engine_port=8860, advisory=False)
    validator = proper.lifecycle._load("cc_traces_validate")
    observed = []
    def observe(covers):
        observed.append(dict(covers))
        return {"covers": covers, "device_nodes": {}, "driver_handles": []}
    monkeypatch.setattr(validator, "observe_device_freedom", observe)
    # Reproduce the original failure with the maintained validator, before any
    # fixture worker output exists. Device observation must not run yet.
    assert validator.gpu_free(SimpleNamespace(dir=str(tmp_path))) == 1
    assert not observed
    completed = []
    for step in (item for item in built["steps"] if item["side"] == "modelled"):
        if step["id"] == "gpu-free":
            assert step["command"][1:] == ["scripts/compass/cc_traces_validate.py", "gpu-free", str(tmp_path)]
            assert validator.gpu_free(SimpleNamespace(dir=str(tmp_path))) == 0
            assert sorted(completed) == list(range(1, repeats + 1))
        else:
            sessions = step.get("sessions", [step])
            # Concurrent completion order does not affect which full set the
            # subsequent audit binds; only worker execution is a fixture.
            for session in reversed(sessions):
                command = session["command"]
                output = Path(command[command.index("--out") + 1])
                output.write_text(json.dumps({"repeat": session["repeat"], "fixture": True}))
                completed.append(session["repeat"])
    assert len(observed) == 1
    assert observed[0] == {f"modelled.r{i}.json": validator._digest(tmp_path / f"modelled.r{i}.json")
                           for i in range(1, repeats + 1)}
    native = [item for item in built["steps"] if item["side"] == "real"]
    assert [item["repeat"] for item in native if item["role"] == "serve"] == list(range(1, repeats + 1))
