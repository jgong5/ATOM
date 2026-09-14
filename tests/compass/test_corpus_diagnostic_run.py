"""Pinned diagnostic CLI, execution identity and refusal/lifecycle contracts."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "corpus_diagnostic_harness_fixture", Path(__file__).with_name("test_cc_traces_run.py"))
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


run = harness.run_mod
diagnostic = run.diagnostic_module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_case(tmp_path, count=1, name="corpus_fixture_v1"):
    workload, manifest = tmp_path / "case.jsonl", tmp_path / "case.manifest.json"
    rows = [{"arrival_s": i * 0.1, "input_tokens": 128, "output_tokens": 16}
            for i in range(count)]
    workload.write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = "a" * 36
    stated = {
        "schema": diagnostic.SCHEMA, "class": name, "purpose": "diagnostic",
        "registered_acceptance_cell": False, "sha256": digest(workload),
        "corpus": diagnostic._corpus(), "corpus_sha256_observed": diagnostic._corpus()["sha256"],
        "target_model": run.plan_module.MODEL, "requests": count, "clients": 1,
        "root_requests": count, "descendant_requests": 0, "roots_used": [{"id": root}],
        "input_token_total": 128 * count, "output_token_total": 16 * count,
        "gaps_clipped": 0, "outputs_altered": 0, "subagents_pruned": 0, "requests_serialised": 0,
        "generator": {"path": "/test/fixture-emitter.py", "sha256": "1" * 64},
        "validation": {"synthetic_test_fixture": True}, "selection_scope": "test fixture",
        "provenance": [{**r, "session": root, "json_path": f"/requests/{i}", "actor": "root",
                        "input_blocks": 2, "source_t_s": 100.0 + r["arrival_s"],
                        "origin_shift_s": 100.0} for i, r in enumerate(rows)],
    }
    manifest.write_text(json.dumps(stated))
    return workload, manifest, name


def load(case):
    workload, manifest, name = case
    return diagnostic.load_case(workload, manifest, digest(manifest), name,
                                target_model=run.plan_module.MODEL)


def argv(tmp_path, case, side="modelled", *, tp=2):
    workload, manifest, name = case
    args = ["diagnostic-side", "--cell", str(tmp_path / f"tp{tp}_{name}_c1"),
            "--tp", str(tp), "--side", side, "--case-id", name,
            "--workload", str(workload), "--manifest", str(manifest),
            "--manifest-sha256", digest(manifest)]
    if side == "modelled":
        args += ["--replay-target", "/w/target.json", "--memory-model", "/w/memory.json"]
    return args


@pytest.mark.parametrize("side", ["real", "modelled"])
def test_cli_plans_the_whole_pinned_case_without_matrix_alias(tmp_path, capsys, side):
    case = fixture_case(tmp_path, 95)
    assert run.main(argv(tmp_path, case, side) + [
        "--plan-only", "--pretokenize", "--client-memory-budget-mib", "16384"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["class"] == case[2] and plan["purpose"] == "diagnostic"
    assert plan["diagnostic_case"]["manifest_sha256"] == digest(case[1])
    assert plan["diagnostic_case"]["requests"] == 95
    assert not Path(plan["cell"]).exists()
    assert not any(s["id"] in {"stamp", "verify-workload", "validate"} for s in plan["steps"])
    command = next(s["command"] for s in plan["steps"] if s["role"] == "replay" and s["side"] == side)
    assert command[command.index("--trace") + 1] == str(case[0])
    assert command[command.index("--num-requests") + 1] == "0"
    assert command[command.index("--client-memory-budget-mib") + 1] == "16384"
    assert ("--pace" in command) == (side == "real")
    assert ("--prepare" in command) == (side == "real")


@pytest.mark.parametrize("change", ["manifest_pin", "workload_bytes", "case_id", "acceptance",
                                  "corpus", "source_length", "source_arrival", "source_ancestry"])
def test_bad_case_is_refused_before_any_launch(tmp_path, monkeypatch, change):
    case = fixture_case(tmp_path)
    args = argv(tmp_path, case)
    stated = json.loads(case[1].read_text())
    if change == "manifest_pin":
        args[args.index("--manifest-sha256") + 1] = "0" * 64
    elif change == "workload_bytes":
        case[0].write_text(case[0].read_text() + "\n")
    elif change == "case_id":
        args[args.index("--case-id") + 1] = "corpus_other"
    else:
        if change == "acceptance":
            stated["purpose"] = "acceptance"
        elif change == "corpus":
            stated["corpus"]["sha256"] = "0" * 64
        elif change == "source_length":
            stated["provenance"][0]["input_tokens"] = 256
        elif change == "source_arrival":
            stated["provenance"][0]["source_t_s"] += 1
        elif change == "source_ancestry":
            stated["provenance"][0]["actor"] = "subagent"
        case[1].write_text(json.dumps(stated))
        args[args.index("--manifest-sha256") + 1] = digest(case[1])
    monkeypatch.setattr(run, "SideRun", lambda *a, **k: pytest.fail("invalid case launched"))
    assert run.main(args) == 2


def test_acceptance_override_and_registered_alias_are_refused(tmp_path):
    case = fixture_case(tmp_path)
    with pytest.raises(SystemExit):
        run.main(argv(tmp_path, case) + ["--purpose", "acceptance"])
    alias = fixture_case(tmp_path, name="clients_short")
    assert run.main(argv(tmp_path, alias) + ["--plan-only"]) == 2
    with pytest.raises(ValueError, match="registered cell directory"):
        run.SideRun({"class": "corpus_fixture", "cell": str(tmp_path / "tp1_clients_large_c8")},
                    "modelled", purpose="acceptance")


def test_preparation_cap_changes_only_real_warmup_plan(tmp_path, capsys):
    case = fixture_case(tmp_path)
    command = argv(tmp_path, case) + ["--plan-only"]
    assert run.main(command) == 0
    default = json.loads(capsys.readouterr().out)
    assert run.main(command + ["--diagnostic-prepare-output-cap", "32"]) == 0
    capped = json.loads(capsys.readouterr().out)
    assert default["diagnostic_case"] == capped["diagnostic_case"] == load(case)
    assert capped["purpose"] == "diagnostic"
    assert capped.pop("diagnostic_prepare_output_cap") == 32
    for step in capped["steps"]:
        if step["role"] == "replay" and step["side"] == "real":
            assert step["command"][-2:] == ["--diagnostic-prepare-output-cap", "32"]
            step["command"] = step["command"][:-2]
    assert capped == default


@pytest.mark.parametrize("cap", ["0", "1", "-32"])
def test_preparation_cap_requires_decode_iterations(tmp_path, cap):
    case = fixture_case(tmp_path)
    with pytest.raises(SystemExit, match="at least 2"):
        run.main(argv(tmp_path, case) + ["--plan-only", "--diagnostic-prepare-output-cap", cap])


def test_registered_cli_cannot_select_capped_preparation(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["side", "--cell", str(tmp_path / "tp1_clients_short_c1"),
                  "--side", "real", "--tp", "1", "--class", "clients_short",
                  "--clients", "1", "--diagnostic-prepare-output-cap", "32"])


def test_acceptance_stamp_cannot_hide_capped_preparation(tmp_path):
    validate = run._load("cc_traces_validate")
    journal = {"purpose": "acceptance", "executions": [{"purpose": "acceptance"}]}
    artifact = {"execution": {"purpose": "acceptance"}, "run": {"prepare": {
        "policy": {"purpose": "diagnostic", "output_tokens_cap": 32}}}}
    errors = validate.check_not_diagnostic(tmp_path, {"real": journal}, {"real": [artifact]})
    assert len(errors) == 1 and "capped diagnostic preparation" in errors[0]


class CaseProcesses(harness.FakeProcesses):
    def __init__(self, *args, mutate=None, refuse=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutate, self.refuse = mutate, refuse

    def _write_artifact(self, command):
        super()._write_artifact(command)
        path = Path(command[command.index("--out") + 1])
        blob = json.loads(path.read_text())
        trace = Path(command[command.index("--trace") + 1])
        rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        blob["workload"] = rows
        blob["run"].update(requests=len(rows), completed=len(rows))
        blob["results"] = [{"index": i, "ok": True, "response": {
            "id": str(i), "usage": {"completion_tokens": r["output_tokens"]}}}
            for i, r in enumerate(rows)]
        if self.mutate:
            self.mutate(blob)
        path.write_text(json.dumps(blob))

    def run_observed(self, command, *, observe, **kwargs):
        code = self.run(command, **kwargs)
        if self.refuse:
            directory = Path(self.started[-1].env["COMPASS_REFUSAL_DUMP"])
            (directory / "region_refusal_b1_t128_1.json").write_text(json.dumps({
                "refused_by": "region model", "why": "128 is outside source support",
                "shape": {"total_tokens": 128}}))
        return {"exit": code, "pid": 98765, "model_refusal": observe()}


def runner(tmp_path, monkeypatch, case, processes=None, *, side="modelled", cap=None):
    identity = load(case)
    plan = run.plan_module.diagnostic_steps(
        2, identity, cell=str(tmp_path / f"tp2_{case[2]}_c1"), oracle="transfer",
        options=(), port=8000, repeats=1, target="/w/target.json",
        diagnostic_prepare_output_cap=cap)
    monkeypatch.setattr(harness, "_plan", lambda *_a, **_k: plan)
    return harness._runner(tmp_path, side, processes=processes or CaseProcesses(cell=tmp_path),
                           purpose="acceptance")


@pytest.mark.parametrize("change", [None, "policy", "shape", "unplanned"])
def test_capped_warmup_evidence_matches_the_real_plan(tmp_path, monkeypatch, change):
    case = fixture_case(tmp_path)

    def preparation(blob):
        blob["run"]["prepare"].update(
            policy={"purpose": "diagnostic", "output_tokens_cap": 32},
            shapes=[{"input_tokens": 128, "output_tokens": 16}] * 3)
        if change == "policy":
            blob["run"]["prepare"].pop("policy")
        if change == "shape":
            blob["run"]["prepare"]["shapes"][0] = {"input_tokens": 64, "output_tokens": 16}

    task = runner(tmp_path, monkeypatch, case,
                  CaseProcesses(cell=tmp_path, mutate=preparation),
                  side="real", cap=None if change == "unplanned" else 32)
    assert task.run() == (0 if change is None else 1), task.failures
    result = json.loads((task.cell / "real.r1.json").read_text())
    assert result["execution"]["purpose"] == "diagnostic"
    assert result["execution"]["diagnostic_case"]["workload_sha256"] == digest(case[0])
    if change is not None:
        assert any("preparation" in reason for reason in task.failures)


def test_case_identity_and_forced_purpose_travel_with_execution(tmp_path, monkeypatch):
    case = fixture_case(tmp_path)
    task = runner(tmp_path, monkeypatch, case)
    assert task.purpose == "diagnostic" and task.run() == 0
    result = json.loads((task.cell / "modelled.r1.json").read_text())
    execution = json.loads((task.cell / "execution.modelled.r1.json").read_text())
    assert result["execution"]["purpose"] == "diagnostic"
    assert result["execution"]["diagnostic_case"]["case_id"] == case[2]
    assert execution["source"]["workload"] == str(case[0])
    assert execution["source"]["workload_sha256"] == digest(case[0])
    assert execution["source"]["diagnostic_case"]["manifest_sha256"] == digest(case[1])


def test_truncated_result_with_full_file_hash_is_refused_and_stamped(tmp_path, monkeypatch):
    case = fixture_case(tmp_path, 2)
    def truncate(blob):
        blob["workload"].pop()
        blob["results"].pop()
        blob["run"].update(requests=1, completed=1)
    task = runner(tmp_path, monkeypatch, case, CaseProcesses(cell=tmp_path, mutate=truncate))
    assert task.run() == 1
    assert any("request count" in f for f in task.failures)
    result = json.loads((task.cell / "modelled.r1.json").read_text())
    assert result["execution"]["purpose"] == "diagnostic"
    assert task.processes.stopped == [p.pid for p in task.processes.started]


@pytest.mark.parametrize("exit_code", [3, 4])
def test_failed_replay_output_keeps_diagnostic_pins(tmp_path, monkeypatch, exit_code):
    case = fixture_case(tmp_path)
    task = runner(tmp_path, monkeypatch, case,
                  CaseProcesses(cell=tmp_path, exits={"modelled.r1.json": exit_code}))
    assert task.run() == (run.REFUSAL_EXIT if exit_code == 3 else 1)
    result = json.loads((task.cell / "modelled.r1.json").read_text())
    assert result["execution"]["purpose"] == "diagnostic"
    assert result["execution"]["diagnostic_case"]["workload_sha256"] == digest(case[0])
    assert task.processes.stopped == [p.pid for p in task.processes.started]


def test_changed_pin_before_replay_stops_owned_server_without_sending(tmp_path, monkeypatch):
    case = fixture_case(tmp_path)
    procs = CaseProcesses(cell=tmp_path)
    original = procs.start
    def start(*args, **kwargs):
        proc = original(*args, **kwargs)
        case[0].write_text(case[0].read_text() + "\n")
        return proc
    procs.start = start
    task = runner(tmp_path, monkeypatch, case, procs)
    assert task.run() == run.REFUSAL_EXIT
    assert not procs.ran
    assert procs.stopped == [p.pid for p in procs.started]


def test_model_refusal_is_distinct_from_harness_refusal_and_keeps_marker(tmp_path, monkeypatch):
    case = fixture_case(tmp_path)
    monkeypatch.setenv("COMPASS_REFUSAL_DUMP", str(tmp_path / "foreign-directory"))
    task = runner(tmp_path, monkeypatch, case, CaseProcesses(cell=tmp_path, refuse=True))
    assert task.run() == run.MODEL_REFUSAL_EXIT
    assert task.refused is False
    journal = json.loads((task.cell / "run.modelled.json").read_text())
    assert journal["incomplete"] is True and journal["model_refusals"]
    marker = journal["model_refusals"][0]
    assert marker["exception_text"] == "128 is outside source support"
    assert digest(Path(marker["path"])) == marker["sha256"]
    assert str(task.cell / "refusals") in marker["path"]
    result = json.loads((task.cell / "modelled.r1.json").read_text())
    assert result["execution"]["model_refusal"] == {k: v for k, v in marker.items() if k != "execution_id"}
    assert result["execution"]["incomplete"] is True


def test_existing_evidence_is_not_overwritten(tmp_path, monkeypatch):
    task = runner(tmp_path, monkeypatch, fixture_case(tmp_path))
    task.cell.mkdir(parents=True)
    prior = task.cell / "run.modelled.json"
    prior.write_text('{"existing":"evidence"}')
    before = prior.read_bytes()
    assert task.run() == run.REFUSAL_EXIT
    assert prior.read_bytes() == before and not task.processes.started


def test_case_lock_identity_allows_different_mount_paths(tmp_path, monkeypatch):
    task = runner(tmp_path, monkeypatch, fixture_case(tmp_path))
    task.cell.mkdir(parents=True)
    other = {**task.diagnostic_case, "workload": "/other/mount/case.jsonl",
             "manifest": "/other/mount/case.manifest.json"}
    (task.cell / "diagnostic_case.json").write_text(json.dumps(other))
    assert task.run() == 0
    assert json.loads((task.cell / "diagnostic_case.json").read_text()) == other


def zero_case(tmp_path, semantics=None):
    case = fixture_case(tmp_path, count=2)
    rows = [json.loads(line) for line in case[0].read_text().splitlines()]
    rows[0]["output_tokens"] = 0
    case[0].write_text("".join(json.dumps(row) + "\n" for row in rows))
    stated = json.loads(case[1].read_text())
    stated["sha256"] = digest(case[0])
    stated["provenance"][0]["output_tokens"] = 0
    stated["output_token_total"] = 16
    if semantics is not None:
        stated["zero_output_semantics"] = semantics
    case[1].write_text(json.dumps(stated))
    return case


@pytest.mark.parametrize("semantics", [None, "cancel_at_source_finish", "no_op"])
def test_zero_diagnostic_requires_the_explicit_surrogate_before_launch(
        tmp_path, monkeypatch, semantics):
    case = zero_case(tmp_path, semantics)
    monkeypatch.setattr(run, "SideRun", lambda *a, **k: pytest.fail("invalid case launched"))
    assert run.main(argv(tmp_path, case)) == 2


def test_zero_contract_is_pinned_and_carried_with_the_complete_execution(tmp_path, monkeypatch):
    case = zero_case(tmp_path, diagnostic.ZERO_OUTPUT_SEMANTICS)
    identity = load(case)
    assert diagnostic.identity(identity)["zero_output_semantics"] == diagnostic.ZERO_OUTPUT_SEMANTICS
    task = runner(tmp_path, monkeypatch, case)
    assert task.run() == 0
    result = json.loads((task.cell / "modelled.r1.json").read_text())
    assert result["execution"]["diagnostic_case"]["zero_output_semantics"] == diagnostic.ZERO_OUTPUT_SEMANTICS
    assert result["run"]["requests"] == 2 and len(result["results"]) == 2
    assert result["workload"][0]["output_tokens"] == 0


def test_zero_contract_change_after_planning_is_refused(tmp_path):
    case = zero_case(tmp_path, diagnostic.ZERO_OUTPUT_SEMANTICS)
    identity = load(case)
    stated = json.loads(case[1].read_text())
    stated["zero_output_semantics"] = "no_op"
    case[1].write_text(json.dumps(stated))
    with pytest.raises(ValueError, match="digest"):
        diagnostic.recheck(identity)


def test_positive_diagnostic_identity_needs_no_zero_contract(tmp_path):
    identity = load(fixture_case(tmp_path))
    assert "zero_output_semantics" not in identity
    assert "zero_output_semantics" not in diagnostic.identity(identity)
