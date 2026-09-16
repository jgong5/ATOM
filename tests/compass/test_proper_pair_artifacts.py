"""Final pair orchestration over real files, with synthetic validator fixtures.

Adjacent tests cover the numerical/provenance validators. This isolates repeat
discovery, propagation of their verdicts, metric gates, and optional cost history.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compass"))
spec = importlib.util.spec_from_file_location("proper_pair_layout", ROOT / "scripts/compass/cc_traces_proper.py")
paired = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paired
spec.loader.exec_module(paired)
from .test_proper_replay_contract import phases, row


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("damage", [
    None, "missing_derivation", "invalid_clock", "invalid_derivation",
    "source", "memory", "accuracy", "extra_repeat",
])
def test_final_pair_accepts_memory_layout_and_keeps_main_gates(tmp_path, monkeypatch, damage):
    prepared = {"config": {"scenario": "inferencex-agentx-mvp", "benchmark_id": "fixture",
        "loadgen": {"concurrency": 1, "benchmark_duration": 900}, "input": {"random_seed": 42},
        "endpoint": {"type": "chat", "streaming": True, "use_server_token_count": True}},
        "source": {"sha256": "source"}, "metadata": {},
        "conversations": [{"context_mode": "deltas_with_responses"}]}
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    plan = {"schema": "compass.aiperf_proper_pair/1", "purpose": "acceptance", "repeats": 3,
            "record_export": {"export_level": "raw", "export_http_trace": True},
            "prepared": pin(prepared_path), "calibration_registry": pin(registry),
            "native_engine_args": [], "modelled_engine_args": [], "model": "fixture", "cache_policy": {}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    args = SimpleNamespace(plan=str(plan_path), plan_sha256=pin(plan_path)["sha256"],
                           case_id="aiperf_proper_layout", cell=str(tmp_path / "cell"), out=None)
    case = paired.load_case(args.plan, args.plan_sha256, args.case_id)
    cell = Path(args.cell)
    cell.mkdir()
    (cell / "diagnostic_case.json").write_text(json.dumps(case))
    (cell / "isolation.json").write_text(json.dumps({"isolated": True, "verdict": "isolated"}))
    phase = paired.core.phase_accounting(phases(completed=2, cancelled=0))
    window = {"schema": "compass.replay_wall_window/1", "clock": "wall",
              "started_at": 1000., "ended_at": 1001., "seconds": 1.}
    for side in ("real", "modelled"):
        executions = []
        for repeat in range(1, 4):
            records = [row(turn=0), row(turn=1)]
            for index, value in enumerate(records):
                value["response_id"] = f"server-{index}"
            if damage == "accuracy" and side == "modelled":
                for value in records:
                    value["first_visible_ns"] += 1_000_000
            blob = {"schema": "compass.aiperf_proper_run/1", "complete": True,
                    "side": side, "purpose": "acceptance", "repeat": repeat,
                    "plan_sha256": args.plan_sha256, "profile": case["profile"],
                    "record_export": plan["record_export"], "records": records,
                    "phase": phase, "cleanup": {}, "server": {}, "engine": {"requests": []},
                    "execution_wall_window": window,
                    "execution": {"purpose": "acceptance", "diagnostic_case": case}}
            (cell / f"{side}.r{repeat}.json").write_text(json.dumps(blob))
            (cell / f"{side}.r{repeat}.prepare.json").write_text("{}")
            (cell / f"{side}.r{repeat}_steps.json").write_text("{}")
            (cell / f"{side}.r{repeat}_steps.jsonl").write_text("{}\n")
            executions.append({"repeat": repeat, "replay": {"measured_window": window}})
            if side == "real":
                (cell / f"real.r{repeat}_memory.json").write_text(json.dumps(
                    {"valid": damage != "memory", "fixture": True}))
        (cell / f"run.{side}.json").write_text(json.dumps(
            {"ok": True, "purpose": "acceptance", "executions": executions}))
    if damage == "extra_repeat":
        (cell / "real.r4.json").write_text((cell / "real.r3.json").read_text())

    validate = paired.lifecycle._load("cc_traces_validate")
    monkeypatch.setattr(paired.lifecycle, "_load", lambda name: validate)
    monkeypatch.setattr(paired.lifecycle.execution_id, "verify_execution_id", lambda value: True)
    for name in ("check_gpu_free", "check_engine", "check_cache_policy_evidence",
                 "check_calibration", "check_capacity_provenance", "check_scalar_overheads",
                 "check_predictor_device_freedom", "check_capacity_inputs",
                 "check_reference_budget_is_measured", "check_who_served"):
        monkeypatch.setattr(validate, name, lambda *args, **kwargs: [])
    source_calls, memory_calls = [], []
    def source(*args, **kwargs):
        source_calls.append(args)
        return (["source qualification refused"] if damage == "source" else []), []
    def memory(real, modelled, directory, repeat, *args, **kwargs):
        paths = validate._memory_records(directory, repeat)
        memory_calls.extend(paths)
        assert len(paths) == 1 and paths[0].name == f"real.r{repeat}_memory.json"
        valid = json.loads(paths[0].read_text())["valid"]
        return ([] if valid else ["memory terms failed"]), {"checked": paths[0].name}
    monkeypatch.setattr(paired.lifecycle.opening_module, "check_source_contract", source)
    monkeypatch.setattr(validate, "check_memory_terms", memory)
    if damage in ("missing_derivation", "invalid_clock", "invalid_derivation"):
        costs = {"cost_schema": validate.COSTS_SCHEMA,
                 "repeats": {"real": 3, "modelled": 3},
                 "execution_real": 1., "execution_modelled": 1.,
                 "execution_clocks": {"real": "wall", "modelled": "wall"},
                 "execution_by_repeat": {side: {str(i): 1. for i in range(1, 4)}
                                         for side in ("real", "modelled")}}
        if damage == "invalid_clock":
            costs["execution_clocks"]["modelled"] = "virtual"
        elif damage == "invalid_derivation":
            costs["derivation"] = 0.1
        (cell / "costs.json").write_text(json.dumps(costs))

    code = paired.pair(args)
    result = json.loads((cell / "proper_pair.json").read_text())
    expected = damage in (None, "missing_derivation")
    assert (code == 0) == result["passed"] == result["accepted"] == expected
    assert len(source_calls) == len(memory_calls) == 3
    assert len(result["execution_wall_windows"]) == 3
    assert result["execution_wall_windows"][0]["real"] == window
    if expected:
        assert result["failures"] == []
        assert result["speedup"]["replay_ratio"] is None
        assert any("unavailable" in note and "advisory" in note for note in result["notes"])
    elif damage == "source":
        assert "source qualification refused" in result["failures"]
    elif damage == "memory":
        assert "memory terms failed" in result["failures"]
    elif damage == "accuracy":
        assert any("tolerance" in failure for failure in result["failures"])
    elif damage == "extra_repeat":
        assert any("repeat count" in failure for failure in result["failures"])
    else:
        assert any("supplied" in failure for failure in result["failures"])
