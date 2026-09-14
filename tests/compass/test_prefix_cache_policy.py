"""Cache-on diagnostics bind configuration without relabelling old evidence."""

import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy, policy_errors
from atom.compass.core.memory_blocks import derived_block_info
from atom.compass.replay.derived_target import derive_target
from atom.compass.replay.runner import TargetRecord


ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run = load(ROOT / "scripts/compass/cc_traces_run.py", "cache_policy_run")
validate = load(ROOT / "scripts/compass/cc_traces_validate.py", "cache_policy_validate")
memory = load(Path(__file__).with_name("test_replay_memory.py"), "cache_policy_memory_fixture")


def case(tmp_path):
    return {"case_id": "cache_fixture", "clients": 1, "workload": str(tmp_path / "rows.jsonl"),
            "cache_policy": cache_on_policy(),
            "prompt_encoding": {"path": str(tmp_path / "encoding.json"), "sha256": "a" * 64},
            "prompt_token_sha256": ["b" * 64]}


def plan(tmp_path, identity, **kwargs):
    return run.plan_module.diagnostic_steps(
        1, identity, cell=str(tmp_path / "tp1_cache_fixture_c1"),
        oracle="source", options=(), port=8000, target="target.json", **kwargs)


def test_cache_on_plan_pins_both_engines_and_encoding(tmp_path):
    identity = case(tmp_path)
    built = plan(tmp_path, identity, enable_prefix_caching=True,
                 prompt_encoding=identity["prompt_encoding"]["path"], prompt_encoding_sha256="a" * 64)
    assert built["purpose"] == "diagnostic" and built["cache_policy"] == cache_on_policy()
    for step in built["steps"]:
        command = step.get("command") or []
        if step["role"] == "serve":
            assert "--no-enable_prefix_caching" not in command
            assert "--enable_prefix_caching" in command
            assert command[command.index("--state-checkpoint-interval-tokens") + 1] == "8192"
            assert "--state-checkpoint-demand" in command
        if step["role"] == "replay":
            assert command[-4:] == ["--prompt-encoding", identity["prompt_encoding"]["path"],
                                    "--prompt-encoding-sha256", "a" * 64]
            assert ("--prepare" in command) == (step["side"] == "real")
    legacy = run.plan_module.cell_steps(1, "clients_short", 1, root=str(tmp_path),
                                        oracle=None, options=(), port=8000, repeats=3)
    assert "cache_policy" not in legacy
    assert all("--no-enable_prefix_caching" in step["command"]
               for step in legacy["steps"] if step["role"] == "serve")


@pytest.mark.parametrize("change", ["flag", "encoding", "demand", "fork"])
def test_plan_refuses_unbound_or_changed_policy(tmp_path, change):
    identity = case(tmp_path)
    if change == "demand":
        identity["cache_policy"]["state_checkpoint_demand"] = False
    if change == "fork":
        identity["cache_policy"]["state_runtime"]["transfer"]["fork_tokens"] = 2
    with pytest.raises(SystemExit):
        plan(tmp_path, identity, enable_prefix_caching=change != "flag",
             prompt_encoding=identity["prompt_encoding"]["path"],
             prompt_encoding_sha256="c" * 64 if change == "encoding" else "a" * 64)


def test_live_engine_check_requires_explicit_new_contract():
    server = {**validate.ENGINE, "enable_prefix_caching": True, "tensor_parallel_size": 1,
              "cache_policy": cache_on_policy()}
    record = SimpleNamespace(manifest={"server": server})
    assert validate.check_engine(record, 1, "test")  # legacy remains cache-off
    assert not validate.check_engine(record, 1, "test", expected_cache_policy=cache_on_policy())
    server["cache_policy"]["state_checkpoint_demand"] = False
    assert validate.check_engine(record, 1, "test", expected_cache_policy=cache_on_policy())


def test_source_policy_disagreement_is_reported_without_relabelling(tmp_path):
    target = Path(memory._target(tmp_path, 1))
    blob = json.loads(target.read_text())
    blob["config"]["enable_prefix_caching"] = False
    target.write_text(json.dumps(blob))
    original = target.read_bytes()
    old = TargetRecord.load(str(target))
    config = memory._config(tmp_path, target=str(target), enable_prefix_caching=True,
                            state_checkpoint_interval_tokens=8192, state_checkpoint_demand=True)
    differences = old.disagreements(config)
    assert any("enable_prefix_caching: captured False" in reason for reason in differences)
    assert any("state_checkpoint_interval_tokens: source capture did not record" in reason
               for reason in differences)
    assert old.cache_policy is None and target.read_bytes() == original


def test_capacity_is_unchanged_but_candidate_policy_is_explicit(tmp_path):
    profile = memory._profile(tmp_path, 1)
    target = TargetRecord.load(memory._target(tmp_path, 1))
    config = memory._config(tmp_path, enable_prefix_caching=False,
                            state_checkpoint_interval_tokens=8192, state_checkpoint_demand=True)
    old = derived_block_info(profile, config, state_runtime=target.blocks["state_runtime"])
    config.enable_prefix_caching = True
    lineage = {}
    enabled = derived_block_info(profile, config, state_runtime=target.blocks["state_runtime"], lineage=lineage)
    assert enabled == old
    assert not policy_errors(lineage["cache_policy"], cache_on_policy())
    assert lineage["cache_policy_assumption"]["status"] == "declared_candidate"
    assert lineage["cache_policy_assumption"]["native_cache_on_memory_validated"] is False
    derived = derive_target(profile, config, layout=target)
    assert derived["blocks"] == enabled and derived["cache_policy"] == cache_on_policy()
    assert derived["config"]["enable_prefix_caching"] is True
    assert derived["derivation"]["borrowed"]["source_cache_policy"]["enable_prefix_caching"] is None
    assert "enable_prefix_caching" not in target.config


def test_memory_and_capture_writers_record_the_same_native_policy(tmp_path):
    # Execute the actual serialization methods without importing the GPU runner.
    tree = ast.parse((ROOT / "atom/compass/runtime/runner.py").read_text())
    mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CompassModelRunner")
    methods = [node for node in mixin.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_write_memory", "_write_replay_target")]
    namespace = {"json": json, "os": os, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "actual_memory_writers", "exec"), namespace)
    config = SimpleNamespace(model="m", tensor_parallel_size=1, max_model_len=262144,
                             max_num_seqs=32, gpu_memory_utilization=0.9, enforce_eager=False,
                             enable_prefix_caching=True, state_checkpoint_interval_tokens=8192,
                             state_checkpoint_demand=True, max_num_batched_tokens=16384,
                             kv_cache_dtype="bf16", kv_cache_block_size=16)
    fake = SimpleNamespace(config=config, rank=0,
        _compass_config=SimpleNamespace(memory_out=str(tmp_path / "memory.json"),
                                        replay_target_out=str(tmp_path / "target.json")),
        _rank_coords=lambda: {"tp": 0}, _topology=lambda: {"tp": 1},
        _hardware_identity=lambda: {"declared_test": True})
    blocks = {"num_kvcache_blocks": 100, "pool_entries": {"state": 32, "kv": 100},
              "pool_entries_per_req": {"state": 1, "kv": 0},
              "state_runtime": cache_on_policy()["state_runtime"]}
    namespace["_write_memory"](fake, {"total": 1}, blocks)
    namespace["_write_replay_target"](fake, blocks=blocks)
    namespace["_write_replay_target"](fake, graph={"capture_sizes": [1]})
    for name in ("memory.json", "target.json"):
        saved = json.loads((tmp_path / name).read_text())
        assert saved["cache_policy"] == cache_on_policy()
        assert saved["config"]["enable_prefix_caching"] is True


def test_cache_on_capacity_reader_keeps_and_checks_candidate_lineage(tmp_path):
    from atom.compass.replay.runner import ReplayModelRunner
    config = memory._config(tmp_path, profile=memory._profile(tmp_path, 1),
                            enable_prefix_caching=True, state_checkpoint_interval_tokens=8192,
                            state_checkpoint_demand=True)
    runner = ReplayModelRunner(0, config)
    runner.get_num_blocks()
    budget = runner.compass_budget_source
    rank = {"inputs": memory.manifest(runner.compass_runtime_inputs)["inputs"],
            "budget_source": budget}
    server = {"enable_prefix_caching": True, "cache_policy": cache_on_policy(),
              "compass": {"loaded_inputs": {"ranks": [rank]}}}
    record = SimpleNamespace(manifest={"server": server})
    assert not validate.check_capacity_inputs(record, "cache-on")
    assert budget["deployment"]["enable_prefix_caching"] is True
    assert budget["lineage"]["borrowed_target_cache_policy"]["enable_prefix_caching"] is None
    budget["lineage"]["cache_policy_assumption"]["native_cache_on_memory_validated"] = True
    assert any("candidate qualification" in reason
               for reason in validate.check_capacity_inputs(record, "cache-on"))


def boundary_manifest(side):
    count = 3 if side == "real" else 0
    quiet = {"idle": True, "reasons": [], "state_readers": 0, "state_deferred_readers": 0}
    before = {"policy": cache_on_policy(), "indexes": {"kv": 0, "state": 0},
              "quiescence": quiet, "counters": {"requests": count}}
    after = json.loads(json.dumps(before))
    rank = {"acknowledged": True, "before": before, "after": after,
            "counter_delta": {"requests": 0}, "worker_barrier": {
                "acknowledged": True, "kind": "device_synchronize" if side == "real" else "modelled_no_device",
                "workers_completed": 1, "core_timeline_drained": True,
                "core_timeline_advance_seconds": 0, "core_virtual_elapsed_seconds": 0,
                "retained_output_requests": 0}}
    return {"server": {"enable_prefix_caching": True, "cache_policy": cache_on_policy()},
            "cache_boundary": {"schema": "compass.cache_reset/1", "acknowledged": True, "ranks": [rank]},
            "cache_state_after": {"schema": "compass.cache_snapshot/1", "ranks": [{"policy": cache_on_policy()}]}}


@pytest.mark.parametrize("side", ["real", "modelled"])
def test_cache_boundary_evidence_binds_policy_and_worker_kind(side):
    manifest = boundary_manifest(side)
    assert not validate.check_cache_policy_evidence(manifest, cache_on_policy(), side)
    manifest["cache_boundary"]["ranks"][0]["worker_barrier"]["kind"] = "unobserved"
    assert validate.check_cache_policy_evidence(manifest, cache_on_policy(), side)


@pytest.mark.parametrize("change", ["policy", "warmed", "retained_output", "elapsed", "final", "missing"])
def test_modelled_cache_boundary_cannot_hide_a_changed_or_used_deployment(change):
    manifest = boundary_manifest("modelled")
    rank = manifest["cache_boundary"]["ranks"][0]
    if change == "policy":
        rank["after"]["policy"]["state_checkpoint_demand"] = False
    elif change == "warmed":
        rank["before"]["counters"]["requests"] = rank["after"]["counters"]["requests"] = 1
    elif change == "retained_output":
        rank["worker_barrier"]["retained_output_requests"] = 1
    elif change == "elapsed":
        rank["worker_barrier"]["core_virtual_elapsed_seconds"] = 0.01
    elif change == "final":
        manifest["cache_state_after"]["ranks"][0]["policy"]["enable_prefix_caching"] = False
    else:
        manifest.pop("cache_boundary")
    assert validate.check_cache_policy_evidence(manifest, cache_on_policy(), "modelled")
