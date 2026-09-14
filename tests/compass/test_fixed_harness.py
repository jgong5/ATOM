"""Complete-root lifecycle identity, bounded preparation and workload provenance."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from .test_fixed_absolute import bundle, plan_file
from .test_fixed_absolute_evidence import completed_evidence
from .test_opening_harness import run, runtime_readings
from .test_fixed_absolute_transport import endpoint

fixed = run.fixed_module
replay = run.replay_client
validate = fixed._shared._script("cc_traces_validate")


def serial_plan(tmp_path, count=7, clients=1):
    data = bundle(root_times=tuple(float(i) for i in range(count)), child_times=(), clients=clients)
    for row in data["requests"]:
        row["body"]["model"] = run.plan_module.MODEL
    return plan_file(tmp_path, data)


def args_for(tmp_path, path, sha, side):
    args = ["fixed-side", "--cell", str(tmp_path / "tp1_aiperf_fixed_fixture_c1"),
            "--side", side, "--tp", "1", "--case-id", "aiperf_fixed_fixture",
            "--fixed-absolute-plan", str(path), "--fixed-absolute-plan-sha256", sha]
    if side == "modelled":
        args += ["--replay-target", "/source/target.json", "--memory-model", "/source/memory.json",
                 "--compass-request-readiness-profile", "/source/readiness.json"]
    return args


@pytest.mark.parametrize("side", ["real", "modelled"])
def test_fixed_plan_retains_owned_lifecycle_and_distinct_identity(tmp_path, capsys, side):
    path, sha, original = serial_plan(tmp_path)
    assert run.main(args_for(tmp_path, path, sha, side) + ["--plan-only"]) == 0
    planned = json.loads(capsys.readouterr().out)
    case = planned["diagnostic_case"]
    assert case["schema"] == fixed.CASE_SCHEMA and planned["purpose"] == "diagnostic"
    assert case["requests"] == 7 and case["clients"] == 1 and case["registered_acceptance_cell"] is False
    assert {r["role"] for r in case["workload_inputs"]} == {"runtime.fixed_absolute", "runtime.fixed_absolute.source_root"}
    native, modelled = [next(s for s in planned["steps"] if s["id"] == f"replay-{sname}-1")
                        for sname in ("real", "modelled")]
    assert "--flush-measurements" in native["command"]
    assert "--fixed-prepare-sequential" in native["command"]
    assert run._after(native["command"], "--prepare") == "7"
    assert run._after(native["command"], "--diagnostic-prepare-output-cap") == "2"
    assert "--prepare" not in modelled["command"] and "--pace" not in modelled["command"]
    for step in (native, modelled):
        assert run._after(step["command"], "--fixed-absolute-plan-sha256") == sha
        assert "--opening-plan" not in step["command"] and "--trace" not in step["command"]
    serve = next(s for s in planned["steps"] if s["id"] == "serve-modelled-1")
    assert run._after(serve["command"], "--compass-fixed-absolute-plan-sha256") == sha
    assert [r["output_tokens"] for r in original.rows] == [2] * 7


@pytest.mark.parametrize("count,clients", [(6, 1), (7, 2)])
def test_unreviewed_preparation_form_refuses_before_launch(tmp_path, capsys, count, clients):
    path, sha, _ = serial_plan(tmp_path, count, clients)
    assert run.main(args_for(tmp_path, path, sha, "real") + ["--plan-only"]) == 2
    assert "seven causally serial" in capsys.readouterr().err


@pytest.mark.parametrize("damage", [None, "source_read", "profile", "missing_request"])
def test_fixed_result_uses_complete_root_observations(tmp_path, damage):
    plan, server, engine, results = completed_evidence(tmp_path)
    case = fixed.load_case(plan.loaded_input.requested, plan.loaded_input.sha256,
                           "aiperf_fixed_fixture", target_model=plan.model)
    blob = {"run": {"fixed_absolute": plan.evidence(), "requests": len(plan.rows),
                    "complete": True, "prompt_encoding": {"kind": "chat_messages"}, "server": server},
            "engine": engine, "results": results, "workload": plan.workload()}
    if damage == "source_read":
        core = server["compass"]["loaded_inputs"]["ranks"][0]["core_inputs"]
        core["inputs"] = [r for r in core["inputs"] if r["role"] != "runtime.fixed_absolute.source_root"]
    elif damage == "profile":
        blob["run"]["fixed_absolute"]["profile"] = "opening"
    elif damage == "missing_request":
        blob["results"].pop()
    if damage:
        with pytest.raises(ValueError):
            fixed.check_result(blob, case)
    else:
        fixed.check_result(blob, case)


@pytest.mark.parametrize("damage", [None, "missing", "duplicate", "unapproved", "changed_bytes"])
def test_only_verified_fixed_workload_reads_are_exempt_from_fitted_sources(tmp_path, monkeypatch, damage):
    plan, server, _, _ = completed_evidence(tmp_path)
    core = server["compass"]["loaded_inputs"]["ranks"][0]["core_inputs"]
    fixed_rows = [r for r in core["inputs"] if str(r["role"]).startswith("runtime.fixed_absolute")]
    # Supply all six fitted readiness roles; the workload exception must not
    # hide these reads or change the common source-provenance checker.
    profile = server["compass"]["request_readiness_profile"]
    core["inputs"] = fixed_rows + [{"role": role, "requested": profile,
                                   "path": f"/source/{role}.json", "sha256": "b" * 64}
                                  for role in validate.READINESS_INPUT_ROLES]
    if damage == "missing":
        core["inputs"].remove(fixed_rows[-1])
    elif damage == "duplicate":
        core["inputs"].append(copy.deepcopy(fixed_rows[-1]))
    elif damage == "changed_bytes":
        Path(plan.roots[0]["source"]["path"]).write_text("{}")
    checked = []
    monkeypatch.setattr(validate, "_check_calibration_records",
                        lambda digests, *_a: checked.extend(digests) or [])
    pin = {"path": plan.loaded_input.requested, "sha256": plan.loaded_input.sha256}
    errors = validate.check_request_readiness_calibration(
        SimpleNamespace(manifest={"server": server}), {}, 1, "a" * 64, {},
        expected_fixed_absolute_plan=None if damage == "unapproved" else pin)
    assert bool(errors) is (damage is not None), errors
    if damage is None:
        assert len(checked) == 6 and all("request_readiness" in row for row in checked)
    else:
        assert any("fixed_absolute" in row for row in checked)


def prepared_receipt(tmp_path, monkeypatch, *, bad_hash=False, fail_index=None):
    _, _, plan = serial_plan(tmp_path)
    rows = plan.rows
    seen, records = [], []

    async def submit(_base, payloads, arrivals, **options):
        index = len(seen)
        body = json.loads(payloads[0])
        assert len(payloads) == 1 and arrivals == [0.0] and options["pace"] is False
        assert options["endpoint"] == "/v1/chat/completions" and options["streaming"] is True
        assert "fixed_absolute_plan" not in options and "response_gated" not in options
        assert not {"compass_arrival", "compass_workload_size", "compass_workload_index"}.intersection(body)
        assert body["messages"] == rows[index]["body"]["messages"]
        assert body["max_completion_tokens"] == 2
        seen.append(body)
        started = time.time()
        if index == fail_index:
            return [{"index": 0, "ok": False, "error": "fixture failure", "send_timing": {}}], {}
        records.append({"request_id": f"warm{index}", "seq_id": str(index), "finish_time": started,
                        "shared_preprocessing": {"input_tokens": rows[index]["input_tokens"],
                            "prompt_token_sha256": "0" * 64 if bad_hash else rows[index]["prompt_token_sha256"]}})
        return [{"index": 0, "ok": True, "response": {"id": f"warm{index}",
                 "usage": {"prompt_tokens": rows[index]["input_tokens"], "completion_tokens": 2}},
                 "send_timing": {"request_started_at": started,
                                 "client_response_returned_wall_time": time.time()}}], {}

    monkeypatch.setattr(replay, "_submit_requests", submit)
    drains = 0
    def drain(*_):
        nonlocal drains
        drains += 1
        return {"clock": "wall", "requests": records if drains == 1 else []}
    monkeypatch.setattr(replay, "_drain_records", drain)
    args = SimpleNamespace(_fixed_absolute_plan=plan, prepare=7, diagnostic_prepare_output_cap=2,
                           timeout=10., client_memory_budget_mib=4)
    before = plan.rows
    receipt = replay._prepare_fixed("http://fixture", args)
    assert plan.rows == before
    return plan, receipt, seen


@pytest.mark.parametrize("damage", [None, "hash", "serial", "payload", "inside_measurement", "missing_seq"])
def test_exact_preparation_records_support_independent_recheck(tmp_path, monkeypatch, damage):
    plan, receipt, seen = prepared_receipt(tmp_path, monkeypatch, bad_hash=damage == "hash")
    assert len(seen) == 7
    manifest = {"prepare": receipt, "wall_execution": {"started_at": time.time()}}
    if damage == "serial":
        receipt["responses"][1]["send_timing"]["request_started_at"] = receipt["wall_started_at"]
    elif damage == "payload":
        receipt["payload_sha256"][0] = "0" * 64
    elif damage == "inside_measurement":
        manifest["wall_execution"]["started_at"] = receipt["wall_started_at"] - 1
    elif damage == "missing_seq":
        receipt["consumed_prompts"][0]["seq_id"] = None
    assert bool(fixed.preparation_errors(manifest, plan)) is (damage is not None)


def test_failed_warm_request_stops_sequence_and_cannot_measure(tmp_path, monkeypatch):
    _, receipt, seen = prepared_receipt(tmp_path, monkeypatch, fail_index=2)
    assert len(seen) == 3 and receipt["returned"] == 2
    assert receipt["drained"] is False and receipt["failures"]


@pytest.mark.parametrize("fail_first", [False, True])
def test_exact_preparation_uses_real_sse_transport_serially(tmp_path, monkeypatch, fail_first):
    _, _, plan = serial_plan(tmp_path)
    async def exercise():
        async with endpoint({i: .005 for i in range(7)}, fail_first=fail_first) as (base, seen, peak):
            reads = 0
            def drain(*_):
                nonlocal reads
                reads += 1
                return {"clock": "wall", "requests": [{"request_id": f"r{i}", "seq_id": str(i),
                    "finish_time": time.time(), "shared_preprocessing": {
                        "input_tokens": 2, "prompt_token_sha256": plan.rows[i]["prompt_token_sha256"]}}
                    for i in seen] if reads == 1 else []}
            monkeypatch.setattr(replay, "_drain_records", drain)
            args = SimpleNamespace(_fixed_absolute_plan=plan, prepare=7, diagnostic_prepare_output_cap=2,
                                   timeout=2., client_memory_budget_mib=4)
            receipt = await asyncio.to_thread(replay._prepare_fixed, base, args)
            assert seen == ([0] if fail_first else list(range(7)))
            assert peak[0] == 1 and receipt["drained"] is (not fail_first)
            if not fail_first:
                assert fixed.preparation_errors({"prepare": receipt,
                    "wall_execution": {"started_at": time.time()}}, plan) == []
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [None, "memory", "source"])
def test_fixed_pair_shares_checks_without_acceptance_credit(tmp_path, monkeypatch, failure):
    path, sha, plan = serial_plan(tmp_path)
    case = fixed.load_case(path, sha, "aiperf_fixed_fixture", target_model=plan.model)
    cell = tmp_path / "tp1_aiperf_fixed_fixture_c1"
    cell.mkdir()
    (cell / "diagnostic_case.json").write_text(json.dumps(case))
    (cell / "isolation.json").write_text(json.dumps({"verdict": "node_busy", "isolated": False}))
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    for side in ("real", "modelled"):
        (cell / f"run.{side}.json").write_text(json.dumps({"ok": True, "purpose": "diagnostic"}))
        (cell / f"{side}.r1.json").write_text(json.dumps({"run": {"server": runtime_readings(side)},
            "workload": plan.workload(), "results": [],
            "execution": {"purpose": "diagnostic", "diagnostic_case": case}}))
    called = []
    for name in ("check_gpu_free", "check_who_served", "check_clocks_finite", "check_workload_is_registered",
                 "check_usage_against_workload", "check_engine", "check_cache_policy_evidence", "check_side_roles",
                 "check_predictor_device_freedom", "check_capacity_inputs", "check_reference_budget_is_measured",
                 "check_scalar_overheads", "check_capacity_provenance"):
        monkeypatch.setattr(validate, name, lambda *_a, _name=name, **_k: called.append(_name) or [])
    def calibration(*_a, **kwargs):
        assert kwargs["expected_fixed_absolute_plan"] == case[fixed.PLAN_KEY]
        assert "expected_opening_plan_sha256" not in kwargs
        called.append("calibration")
        return []
    monkeypatch.setattr(validate, "check_calibration", calibration)
    monkeypatch.setattr(validate, "check_memory_terms", lambda *_a, **_k:
                        (["memory mismatch"] if failure == "memory" else [], []))
    def source(_modelled, _registry, workload_sha, forbidden, _label):
        assert workload_sha == sha
        assert all(forbidden[row["path"]] == row["sha256"] for row in case["workload_inputs"])
        return (["workload source used for fitting"] if failure == "source" else []), []
    monkeypatch.setattr(fixed, "check_source_contract", source)
    monkeypatch.setattr(fixed, "check_result", lambda *_a: called.append("fixed_result"))
    compare = fixed._shared._script("compare")
    monkeypatch.setattr(compare, "check_run", lambda *_a, **kw: [] if kw["expect_requests"] == 7 else ["wrong count"])
    monkeypatch.setattr(compare, "check_pair", lambda *_a: [])
    monkeypatch.setattr(compare, "compare", lambda *_a: {})
    monkeypatch.setattr(validate, "_across_repeats", lambda *_a:
                        {key: {"within_tolerance": True} for key in validate.TOLERANCE_PCT})
    args = SimpleNamespace(cell=str(cell), case_id=case["case_id"], fixed_absolute_plan=str(path),
        fixed_absolute_plan_sha256=sha, calibration_registry=str(registry), repeats=1, out=None)
    assert fixed.pair(args) == (0 if failure is None else 1)
    report = json.loads((cell / "fixed_diagnostic.json").read_text())
    assert report["schema"] == fixed.REPORT_SCHEMA and report["accepted"] is False
    assert report["passed"] is (failure is None)
    assert "calibration" in called and called.count("fixed_result") == 2
    assert not (cell / "opening_diagnostic.json").exists()


@pytest.mark.parametrize("extra", [[], ["--fixed-prepare-sequential", "--prepare", "3"],
                                  ["--fixed-prepare-sequential", "--prepare", "7", "--diagnostic-prepare-output-cap", "32"]])
def test_fixed_preparation_cli_refuses_unsupported_forms(extra):
    argv = ["--fixed-absolute-plan", "plan", "--fixed-absolute-plan-sha256", "a" * 64,
            "--prepare", "7", "--out", "unused.json", *extra]
    with pytest.raises(SystemExit) as exc:
        replay.main(argv)
    assert exc.value.code == 2
