"""Opening identity uses the maintained lifecycle without a proxy/cell alias."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


root = Path(__file__).resolve().parents[2]
harness = load("opening_maintained_harness_fixture", Path(__file__).with_name("test_cc_traces_run.py"))
fixtures = load("opening_plan_harness_fixture", Path(__file__).with_name("test_opening_release.py"))
run = harness.run_mod
opening = run.opening_module
validate = opening._script("cc_traces_validate")


def case_file(tmp_path):
    path, _, _, _ = fixtures.opening_fixture(tmp_path, 1.)
    data = json.loads(path.read_text())
    switches = {"WEKA_LIVE_ASSISTANT_RESPONSES": False,
                "WEKA_SPLIT_FLATTENED_AGENTS": True, "WEKA_TOOL_SHAPED_MESSAGES": False}
    data["producer"] = {"aiperf_commit": opening.AIPERF_COMMIT, "weka_reconstruction": {
        "defaults_verified": True, "effective": switches, "pinned_defaults": switches}}
    data["source"] = {"root_id": "fixture-root", "selected_entries": [0, 1],
                      "full_plan_opening_roles_match": True, "later_branches_excluded": True}
    for row in data["requests"]:
        row["body"]["model"] = run.plan_module.MODEL
    path.write_text(json.dumps(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(tmp_path, path, sha, side):
    result = ["opening-side", "--cell", str(tmp_path / "tp1_aiperf_opening_fixture_c1"),
              "--side", side, "--tp", "1", "--case-id", "aiperf_opening_fixture",
              "--opening-plan", str(path), "--opening-plan-sha256", sha]
    if side == "modelled":
        result += ["--replay-target", "/source/target.json", "--memory-model", "/source/memory.json",
                   "--compass-request-readiness-profile", "/source/readiness.json"]
    return result


def runtime_readings(side):
    configuration = {"model": run.plan_module.MODEL, **opening.WORKER_CONFIGURATION}
    graph = {"effective_decode_buckets": opening.EFFECTIVE_DECODE_BUCKETS,
             "origin": "native_capture" if side == "real" else "borrowed_replay_target",
             "native_capture_sizes": opening.EFFECTIVE_DECODE_BUCKETS if side == "real" else None,
             "borrowed_source_capture_sizes": opening.EFFECTIVE_DECODE_BUCKETS if side == "modelled" else None,
             "borrowed_target_input": {"sha256": "b" * 64} if side == "modelled" else None}
    return {"worker_runtime": [{"reader": {"component": "ModelRunner", "pid": 321},
                                 "configuration": configuration, "graphs": graph}],
            "core_cache": {"ranks": [{"reader": {"component": "EngineCore.Scheduler", "pid": 322},
                                       "policy": cache_on_policy(),
                                       "scheduler_configuration": {name: configuration[name] for name in (
                                           "max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_block_size")}}]}}


@pytest.mark.parametrize("side", ["real", "modelled"])
def test_plan_reuses_lifecycle_with_explicit_opening_arguments(tmp_path, capsys, side):
    path, sha = case_file(tmp_path)
    assert run.main(arguments(tmp_path, path, sha, side) + ["--plan-only", "--diagnostic-prepare-output-cap", "32"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["purpose"] == "diagnostic"
    assert plan["diagnostic_case"]["schema"] == opening.CASE_SCHEMA
    assert plan["diagnostic_case"]["workload_sha256"] == sha
    assert plan["cache_policy"] == cache_on_policy()
    for step in plan["steps"]:
        if step["role"] == "replay":
            cmd = step["command"]
            assert "--opening-plan" in cmd and "--opening-plan-sha256" in cmd
            assert not {"--trace", "--prompt-encoding", "--pretokenize"}.intersection(cmd)
            assert ("--prepare" in cmd) == (step["side"] == "real")
        if step["role"] == "serve":
            assert ("--compass-opening-plan" in step["command"]) == (step["side"] == "modelled")
    roles = {step["id"] for step in plan["steps"]}
    assert {"sample-baseline", "sample", "isolation", "gpu-free"} <= roles


def test_unattested_old_export_and_alias_are_rejected_before_launch(tmp_path, monkeypatch):
    path, sha = case_file(tmp_path)
    data = json.loads(path.read_text())
    data["producer"].pop("weka_reconstruction")
    path.write_text(json.dumps(data))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(run, "SideRun", lambda *_a, **_kw: pytest.fail("launched invalid opening"))
    assert run.main(arguments(tmp_path, path, sha, "real")) == 2
    with pytest.raises(ValueError, match="prefix"):
        opening.load_case(path, sha, "clients_short", target_model=run.plan_module.MODEL)


@pytest.mark.parametrize("damage", [None, "engine", "core", "plan"])
def test_live_opening_requires_full_resolved_config_and_runtime_pin(tmp_path, damage):
    path, sha = case_file(tmp_path)
    case = opening.load_case(path, sha, "aiperf_opening_fixture", target_model=run.plan_module.MODEL)
    cell = tmp_path / "tp1_aiperf_opening_fixture_c1"
    cell.mkdir()
    plan = run.plan_module.opening_steps(
        1, case, cell=str(cell), oracle="fixture", options=(), port=8000, repeats=1,
        target="/source/target.json", request_readiness_profile="/source/readiness.json")
    runner = run.SideRun(plan, "modelled")
    step = next(step for step in plan["steps"] if step["id"] == "serve-modelled-1")
    said = {**harness.fake_provenance("predict", tp=1), **validate.ENGINE,
            "enable_prefix_caching": True, "cache_policy": cache_on_policy(), **runtime_readings("modelled")}
    said["compass"].update(opening_plan=str(path), opening_plan_sha256=sha)
    if damage == "engine":
        said["worker_runtime"][0]["configuration"]["max_num_batched_tokens"] = 1
    elif damage == "core":
        said["core_cache"]["ranks"][0]["scheduler_configuration"]["max_num_batched_tokens"] = 1
    elif damage == "plan":
        said["compass"]["opening_plan_sha256"] = "0" * 64
    execution = {"config": {}, "artifacts": {}}
    assert (runner._check_provenance(step, said, execution) is not None) == (damage is None)


def test_case_mount_path_changes_do_not_change_content_identity(tmp_path):
    path, sha = case_file(tmp_path)
    case = opening.load_case(path, sha, "aiperf_opening_fixture", target_model=run.plan_module.MODEL)
    other = copy.deepcopy(case)
    other["workload"] = "/another/mount/opening.json"
    other["opening_plan"]["path"] = other["workload"]
    assert opening.identity(other) == opening.identity(case)


@pytest.mark.parametrize("damage", [None, "codec_kind", "early_release", "missing_result"])
def test_opening_result_requires_real_chat_identity_and_causal_release(tmp_path, damage):
    path, sha = case_file(tmp_path)
    case = opening.load_case(path, sha, "aiperf_opening_fixture", target_model=run.plan_module.MODEL)
    plan = opening._plan(path, sha)
    rows = plan.workload()
    blob = {"run": {"requests": 2, "complete": True, "aiperf_opening": plan.evidence(),
                    "prompt_encoding": {"kind": "chat_messages"}, "server": {}},
            "workload": rows, "results": [], "engine": {"clock": "wall", "requests": []}}
    for index, row in enumerate(rows):
        blob["results"].append({"index": index, "ok": True, "response": {"id": f"r{index}"},
                                "send_timing": {"finished_offset_s": 2., "request_started_offset_s": 2.1 if index else 0.}})
        blob["engine"]["requests"].append({"request_id": f"r{index}", "shared_preprocessing": {
            "input_tokens": row["input_tokens"], "prompt_token_sha256": row["prompt_token_sha256"]}})
    if damage == "codec_kind":
        blob["run"]["prompt_encoding"]["kind"] = "token_ids"
    elif damage == "early_release":
        blob["results"][1]["send_timing"]["request_started_offset_s"] = 1.
    elif damage == "missing_result":
        blob["results"].pop()
    if damage is None:
        opening.check_result(blob, case)
    else:
        with pytest.raises(ValueError):
            opening.check_result(blob, case)


@pytest.mark.parametrize("failure", [None, "calibration", "memory", "journal"])
def test_pair_route_stays_diagnostic_and_preserves_source_or_memory_failure(tmp_path, monkeypatch, failure):
    path, sha = case_file(tmp_path)
    case = opening.load_case(path, sha, "aiperf_opening_fixture", target_model=run.plan_module.MODEL)
    cell = tmp_path / "tp1_aiperf_opening_fixture_c1"
    cell.mkdir()
    (cell / "diagnostic_case.json").write_text(json.dumps(case))
    (cell / "isolation.json").write_text(json.dumps({"isolated": True, "verdict": "isolated"}))
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    plan = opening._plan(path, sha)
    for side in ("real", "modelled"):
        (cell / f"run.{side}.json").write_text(json.dumps({
            "ok": failure != "journal", "purpose": "diagnostic"}))
        blob = {"run": {"server": runtime_readings(side)}, "workload": plan.workload(), "results": [],
                "execution": {"purpose": "diagnostic", "diagnostic_case": case}}
        (cell / f"{side}.r1.json").write_text(json.dumps(blob))
    called = []
    names = ("check_gpu_free", "check_who_served", "check_clocks_finite",
             "check_workload_is_registered", "check_usage_against_workload", "check_engine",
             "check_cache_policy_evidence", "check_side_roles", "check_source_factory",
             "check_predictor_device_freedom", "check_capacity_inputs",
             "check_reference_budget_is_measured", "check_calibration",
             "check_scalar_overheads", "check_capacity_provenance", "check_region_calibration")
    for name in names:
        def check(*args, _name=name, **kwargs):
            called.append((_name, kwargs))
            return ["source provenance failed"] if failure == "calibration" and _name == "check_calibration" else []
        monkeypatch.setattr(validate, name, check)
    def memory(*args, **kwargs):
        called.append(("check_memory_terms", kwargs))
        return (["native memory mismatch"] if failure == "memory" else []), [{"fixture": "memory checked"}]
    monkeypatch.setattr(validate, "check_memory_terms", memory)
    monkeypatch.setattr(opening, "check_result", lambda *_: called.append(("opening_result", {})))
    compare = opening._script("compare")
    monkeypatch.setattr(compare, "check_run", lambda *_a, **_kw: [])
    monkeypatch.setattr(compare, "check_pair", lambda *_a, **_kw: [])
    monkeypatch.setattr(compare, "compare", lambda *_a: {})
    monkeypatch.setattr(validate, "_across_repeats", lambda *_a: {
        key: {"within_tolerance": True} for key in validate.TOLERANCE_PCT})
    args = SimpleNamespace(cell=str(cell), opening_plan=str(path), opening_plan_sha256=sha,
                           case_id=case["case_id"], calibration_registry=str(registry), repeats=1, out=None)
    assert opening.pair(args) == (0 if failure is None else 1)
    report = json.loads((cell / "opening_diagnostic.json").read_text())
    assert report["purpose"] == "diagnostic" and report["accepted"] is False
    assert report["passed"] is (failure is None)
    assert not (cell / "cc_traces_cell.json").exists()
    assert next(kw for name, kw in called if name == "check_calibration")["expected_opening_plan_sha256"] == sha
    assert next(kw for name, kw in called if name == "check_memory_terms")["expected_cache_policy"] == cache_on_policy()
