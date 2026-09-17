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


@pytest.fixture
def wrapper_evidence(tmp_path):
    """Real wrapper, base factory, qualified q16 loader and reader manifests."""
    from .test_cache_region_overlay import artifact
    from .test_cached_q16_prices import bundle
    from .test_source_oracle import _template_file
    from .test_attention_call_scopes import PREFILL
    from atom.compass.core.loaded_input import manifest
    from atom.compass.runtime.cache_region_oracle import source_cost_oracle

    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({"attention_scope": {"unified": PREFILL}}))
    overlay = artifact.__wrapped__()
    overlay["q16_request_scope"] = {"sha256": hashlib.sha256(scope.read_bytes()).hexdigest()}
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(overlay))
    q16_path, q16_sha = bundle.__wrapped__(tmp_path)
    template = _template_file(tmp_path)
    options = dict(model=run.plan_module.MODEL, tp=1, block_size=16,
                   max_model_len=262144, position_rows=3, cudagraph_mode="full",
                   allocation="native", require_complete=True, head=True, derive=False,
                   template=template, head_template=template, regions=overlay["base"]["name"],
                   interpolate=4, attention_scope=str(scope), region_overlay=str(path),
                   region_overlay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   include_failed_outputless=1, include_failed_final=1, diagnostic_only=1,
                   q16_handoff=q16_path, q16_handoff_sha256=q16_sha)
    oracle = source_cost_oracle(**options)
    rank = manifest(oracle.compass_loaded_inputs)
    rank["regions"] = oracle.compass_region_snapshot
    # Same option grouping as the reporting boundary, including the new aliases.
    grouped = {}
    for row in rank["inputs"]:
        role = row["role"].removeprefix("oracle.")
        option = {"price_graph": "price", "q16_sources": "q16_handoff"}.get(role, role)
        grouped.setdefault(option, {})[Path(row["path"]).name] = row["sha256"]
    digests = {key: next(iter(files.values())) if len(files) == 1 else validate._rolled_digest(files)
               for key, files in grouped.items()}
    compass = {"oracle": opening.CACHE_REGION_FACTORY, "oracle_options": options,
               "loaded_inputs": {"ranks": [rank]}, "oracle_option_sha256": digests,
               "oracle_option_files": grouped}
    provenance = {"measured_at_tp": 1, "from_target_engine": False,
                  "sources": [{"path": "/isolated/source.json", "sha256": "8" * 64}],
                  "code": {"collector.py": "9" * 64}}
    registry = {"artifacts": [dict(provenance, kind="source_calibration", sha256=sha,
                                    contents=grouped[key]) for key, sha in digests.items()]}
    registry["artifacts"].append(dict(provenance, kind="region_model",
                                      sha256=rank["regions"]["sha256"]))
    return compass, registry


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
                                 "allocation_policy": {"torch_git_version": "3d3aa833db84eed6b7f5595cb5f162c2f78300a4",
                                     "deterministic_algorithms": False, "fill_uninitialized_memory": True},
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


@pytest.mark.parametrize("failure", [None, "calibration", "memory", "journal",
                                      "overlay_pin", "q16_pin", "head", "complete", "allocation"])
def test_pair_route_stays_diagnostic_and_preserves_source_or_memory_failure(
        tmp_path, monkeypatch, failure, wrapper_evidence):
    path, sha = case_file(tmp_path)
    case = opening.load_case(path, sha, "aiperf_opening_fixture", target_model=run.plan_module.MODEL)
    cell = tmp_path / "tp1_aiperf_opening_fixture_c1"
    cell.mkdir()
    (cell / "diagnostic_case.json").write_text(json.dumps(case))
    (cell / "isolation.json").write_text(json.dumps({"isolated": True, "verdict": "isolated"}))
    registry = tmp_path / "registry.json"
    compass, registry_data = wrapper_evidence
    if failure == "calibration":
        registry_data["artifacts"][0]["from_target_engine"] = True
    options = compass["oracle_options"]
    if failure in ("overlay_pin", "q16_pin"):
        options["region_overlay_sha256" if failure == "overlay_pin" else "q16_handoff_sha256"] = "0" * 64
    elif failure in ("head", "complete"):
        options["head" if failure == "head" else "require_complete"] = False
    elif failure == "allocation":
        options["allocation"] = "none"
    registry.write_text(json.dumps(registry_data))
    plan = opening._plan(path, sha)
    for side in ("real", "modelled"):
        (cell / f"run.{side}.json").write_text(json.dumps({
            "ok": failure != "journal", "purpose": "diagnostic"}))
        blob = {"run": {"server": runtime_readings(side)}, "workload": plan.workload(), "results": [],
                "execution": {"purpose": "diagnostic", "diagnostic_case": case}}
        if side == "modelled":
            blob["run"]["server"].update(compass=compass, tensor_parallel_size=1)
        (cell / f"{side}.r1.json").write_text(json.dumps(blob))
    called = []
    names = ("check_gpu_free", "check_who_served", "check_clocks_finite",
             "check_workload_is_registered", "check_usage_against_workload", "check_engine",
             "check_cache_policy_evidence", "check_side_roles",
             "check_predictor_device_freedom", "check_capacity_inputs",
             "check_reference_budget_is_measured",
             "check_scalar_overheads", "check_capacity_provenance")
    for name in names:
        def check(*args, _name=name, **kwargs):
            called.append((_name, kwargs))
            return []
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
    assert next(kw for name, kw in called if name == "check_memory_terms")["expected_cache_policy"] == cache_on_policy()
    assert report["source_contracts"][0]["observed_oracle"] == opening.CACHE_REGION_FACTORY
    if failure is None:
        assert any("FAILED outputless" in note for note in report["notes"])
        assert any("FAILED final-query transfer" in note for note in report["notes"])
        assert report["source_contracts"][0]["include_failed_final"] == 1


@pytest.mark.parametrize("damage, expected", [
    ("base", "base preset"), ("snapshot", "snapshot"), ("diagnostic", "diagnostic_only"),
    ("unknown", "no such option"), ("topology", "tp="), ("rank", "rank zero"),
    ("scope", "request scope"), ("missing_q16_source", "loaded-input identity"),
    ("overlay_bytes", "bytes differ"), ("missing_overlay", "No such file"),
])
def test_wrapper_contract_preserves_selection_scope_and_pinned_reads(wrapper_evidence, damage, expected):
    compass, registry = wrapper_evidence
    options, rank = compass["oracle_options"], compass["loaded_inputs"]["ranks"][0]
    if damage == "base":
        options["regions"] = "source-27b-tp1"
    elif damage == "snapshot":
        rank["regions"]["parameters"]["final"]["prepare_intercept"] += .1
    elif damage == "diagnostic":
        options["diagnostic_only"] = 0
    elif damage == "unknown":
        options["invented_option"] = 1
    elif damage == "topology":
        options["tp"] = 2
    elif damage == "rank":
        options["rank_coords"] = "tp:1"
    elif damage == "scope":
        next(row for row in rank["inputs"] if row["role"] == "oracle.attention_scope")["sha256"] = "0" * 64
    elif damage == "missing_q16_source":
        rank["inputs"] = [row for row in rank["inputs"] if row["role"] != "oracle.price_graph"]
    elif damage == "overlay_bytes":
        path = Path(options["region_overlay"])
        path.write_text(path.read_text() + " ")
    elif damage == "missing_overlay":
        Path(options["region_overlay"]).unlink()
    modelled = SimpleNamespace(manifest={"server": {"compass": compass, "tensor_parallel_size": 1}})
    before = copy.deepcopy(modelled.manifest)
    bad, _ = opening.check_source_contract(modelled, registry, "7" * 64, {}, "fixture")
    assert any(expected in issue for issue in bad), bad
    assert modelled.manifest == before


def test_registered_validation_still_refuses_diagnostic_wrapper(wrapper_evidence):
    compass, _ = wrapper_evidence
    modelled = SimpleNamespace(manifest={"server": {"compass": compass}})
    assert "not a cc-traces acceptance cell" in validate.check_source_factory(modelled, 1, "fixture")[0]


@pytest.mark.parametrize("damage", [None, "missing_read", "unregistered", "opt_in"])
def test_low_query_diagnostic_contract_reopens_every_source(wrapper_evidence, tmp_path, damage):
    from .test_low_query_prices import make_bundle
    from atom.compass.core.loaded_input import manifest
    from atom.compass.runtime.cache_region_oracle import source_cost_oracle

    compass, registry = wrapper_evidence
    options = compass["oracle_options"]
    scope_sha = hashlib.sha256(Path(options["attention_scope"]).read_bytes()).hexdigest()
    path, sha = make_bundle(tmp_path / "low_q", scope_sha, failed=True)
    options.update(low_q_handoff=path, low_q_handoff_sha256=sha, low_q_allow_failed_spread=1)
    oracle = source_cost_oracle(**options)
    rank = manifest(oracle.compass_loaded_inputs)
    rank["regions"] = oracle.compass_region_snapshot
    compass["loaded_inputs"]["ranks"] = [rank]
    provenance = {"kind": "source_calibration", "measured_at_tp": 1, "from_target_engine": False,
                  "sources": [{"path": "/isolated/source.json", "sha256": "8" * 64}],
                  "code": {"collector.py": "9" * 64}}
    grouped = {}
    for row in rank["inputs"]:
        role = row["role"].removeprefix("oracle.")
        option = ("low_q_handoff" if role.startswith("low_q_") else
                  {"price_graph": "price", "q16_sources": "q16_handoff"}.get(role, role))
        grouped.setdefault(option, {})[Path(row["path"]).name] = row["sha256"]
        registry["artifacts"].append(dict(provenance, sha256=row["sha256"],
                                          contents={Path(row["path"]).name: row["sha256"]}))
    digests = {key: next(iter(files.values())) if len(files) == 1 else validate._rolled_digest(files)
               for key, files in grouped.items()}
    registry["artifacts"] += [dict(provenance, sha256=sha, contents=grouped[key])
                              for key, sha in digests.items()]
    compass.update(oracle_option_sha256=digests, oracle_option_files=grouped)
    if damage == "missing_read":
        rank["inputs"] = [row for row in rank["inputs"] if row["role"] != "oracle.low_q_validation"]
    elif damage == "unregistered":
        verdict_sha = next(row["sha256"] for row in rank["inputs"] if row["role"] == "oracle.low_q_validation")
        registry["artifacts"] = [row for row in registry["artifacts"] if row["sha256"] != verdict_sha]
    elif damage == "opt_in":
        options["low_q_allow_failed_spread"] = 0
    modelled = SimpleNamespace(manifest={"server": {"compass": compass, "tensor_parallel_size": 1}})
    before = copy.deepcopy(modelled.manifest)
    bad, notes = opening.check_source_contract(modelled, registry, "7" * 64, {}, "fixture")
    if damage:
        assert bad
    else:
        assert not bad
        assert any("FAILED low-query heldout spread retained" in note for note in notes)
        assert len(grouped["low_q_handoff"]) == 5
        assert "not a cc-traces acceptance cell" in validate.check_source_factory(modelled, 1, "fixture")[0]
    assert modelled.manifest == before
