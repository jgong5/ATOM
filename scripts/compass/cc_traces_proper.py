"""Proper TP1/C1 AIPerf pairing through the existing lifecycle and protocol.

The frozen plan declares purpose, repeats, symmetric record_export settings,
native/modelled engine arguments and environments, profile/dependency pins, and
the calibration registry. Environment paths may contain {repeat}. Acceptance
uses the protocol repeat floor and existing source, memory and metric checks.
Supplied cost claims remain validated; unavailable speedup accounting is advisory.
cost_args are forwarded unchanged to cc_traces_run costs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cc_traces_run as lifecycle

core = lifecycle._core("proper_replay")
compare = lifecycle.compare
CASE_SCHEMA = "compass.aiperf_proper_case/1"


def load_case(path, sha, case_id):
    plan = core.read_pinned({"path": path, "sha256": sha})
    if plan.get("schema") != "compass.aiperf_proper_pair/1":
        raise ValueError("unsupported proper pair plan")
    if not re.fullmatch(r"aiperf_proper_[A-Za-z0-9_-]+", case_id):
        raise ValueError("proper case-id must use the aiperf_proper_ prefix")
    purpose, repeats = plan.get("purpose"), plan.get("repeats")
    if purpose not in lifecycle.PURPOSES or type(repeats) is not int or repeats < 1:
        raise ValueError("proper pair requires its purpose and repeat count frozen before execution")
    if purpose == lifecycle.ACCEPTANCE and repeats < lifecycle._load("cc_traces_validate").PROTOCOL_REPEATS:
        raise ValueError("acceptance requires the maintained protocol repeat count")
    concurrency = plan.get("modelled_concurrency", 1)
    if type(concurrency) is not int or not 1 <= concurrency <= min(3, repeats):
        raise ValueError("modelled_concurrency must be between one and three, bounded by repeats")
    if concurrency > 1:
        for name in ("TMPDIR", "AIPERF_DATASET_MMAP_BASE_PATH"):
            value = plan.get("modelled_environment", {}).get(name)
            if not isinstance(value, str) or "{repeat}" not in value or not Path(value).is_absolute():
                raise ValueError(f"concurrent modelled sessions require a distinct absolute {name} per repeat")
    if plan.get("record_export") != {"export_level": "raw", "export_http_trace": True}:
        raise ValueError("proper pair requires symmetric raw export settings frozen before execution")
    profile = core.profile_identity(core.read_pinned(plan["prepared"]))
    for args in (plan["native_engine_args"], plan["modelled_engine_args"]):
        if any("fixed-absolute" in arg or "opening-plan" in arg for arg in args):
            raise ValueError("proper pair cannot install a fixed request calendar")
    return {"schema": CASE_SCHEMA, "case_id": case_id, "clients": 1,
            "purpose": purpose, "registered_acceptance_cell": purpose == lifecycle.ACCEPTANCE,
            "repeats": repeats, "modelled_concurrency": concurrency,
            "record_export": dict(plan["record_export"]),
            "target_model": plan["model"], "workload": str(Path(path).resolve()),
            "workload_sha256": sha, "plan": {"path": str(Path(path).resolve()), "sha256": sha},
            "profile": profile, "cache_policy": plan["cache_policy"]}


def identity(case):
    return {"modelled_concurrency": case.get("modelled_concurrency", 1),
            **{k: case[k] for k in ("schema", "case_id", "clients", "purpose",
            "registered_acceptance_cell", "repeats", "record_export", "target_model", "workload_sha256", "profile", "cache_policy")}}


def recheck(case):
    current = load_case(case["plan"]["path"], case["plan"]["sha256"], case["case_id"])
    if identity(current) != identity(case):
        raise ValueError("proper pair input identity changed")
    return current


def check_result(blob, case):
    if (blob.get("schema") != "compass.aiperf_proper_run/1" or blob.get("complete") is not True
            or blob.get("plan_sha256") != case["workload_sha256"]
            or blob.get("profile") != case["profile"]):
        raise ValueError("proper side did not complete its pinned profile")
    if blob.get("purpose") != case["purpose"] or blob.get("repeat") not in range(1, case["repeats"] + 1):
        raise ValueError("proper result changed its predeclared purpose/repeat")
    if blob.get("record_export") != case["record_export"]:
        raise ValueError("proper result changed its predeclared raw export settings")
    core.validate_records(blob["records"], blob["phase"])
    if blob["side"] == "modelled" and any(blob.get("cleanup", {}).get(key, 0)
            for key in ("io_requests", "stream_loops", "request_start_times", "sequence_routes",
                        "stream_callbacks", "callback_tasks", "prepare_cleanup_tasks",
                        "pending_deliveries", "loop_tasks")):
        raise ValueError("controlled proper profile left unsettled execution state")
    if blob["phase"]["counts"]["final_request_errors"]:
        raise ValueError("proper profile contains request errors")
    if blob["phase"]["requested_duration_seconds"] != 900:
        raise ValueError("proper profile duration differs")


def build_steps(case, cell, *, port, engine_port, advisory):
    plan = core.read_pinned(case["plan"])
    pm = lifecycle.plan_module
    before, after = pm._real_monitoring_steps(str(cell), advisory)
    real_steps, modelled_steps = [], []
    for repeat in range(1, case["repeats"] + 1):
        real = pm._lifecycle(
            "real", repeat, tp=1, klass=case["case_id"], clients=1, cell=str(cell), where="gpu",
            port=port, engine_port=engine_port, oracle=None, options=(), target=None,
            engine_args=plan["native_engine_args"], workload_path=case["workload"])
        real[1]["command"] = [sys.executable, "scripts/compass/aiperf_proper_replay.py",
            "--plan", case["plan"]["path"], "--plan-sha256", case["plan"]["sha256"],
            "--side", "real", "--repeat", str(repeat), "--url", f"http://127.0.0.1:{port}",
            "--out", str(cell / f"real.r{repeat}.json")]
        real_steps.extend(real)
        modelled_steps.append({"id": f"proper-modelled-{repeat}", "role": "proper_session",
            "side": "modelled", "repeat": repeat,
            "command": [sys.executable, "scripts/compass/aiperf_proper_replay.py",
                "--plan", case["plan"]["path"], "--plan-sha256", case["plan"]["sha256"],
                "--side", "modelled", "--repeat", str(repeat),
                "--out", str(cell / f"modelled.r{repeat}.json"),
                "--start-signal", str(cell / f"modelled.r{repeat}.raw/start_profile.json")]})
    concurrency = case.get("modelled_concurrency", 1)
    if concurrency > 1:
        modelled_steps = [{"id": f"proper-modelled-group-{start // concurrency + 1}",
                           "role": "proper_sessions", "side": "modelled",
                           "sessions": modelled_steps[start:start + concurrency]}
                          for start in range(0, len(modelled_steps), concurrency)]
    return {"cell": str(cell), "tp": 1, "class": case["case_id"], "clients": 1,
            "repeats": case["repeats"], "purpose": case["purpose"],
            "workload": case["workload"], "diagnostic_case": case, "cache_policy": case["cache_policy"],
            "allow_advisory_isolation": advisory,
            "isolation_qualification": pm.ADVISORY_ISOLATION_QUALIFICATION if advisory else None,
            "steps": before + real_steps + after + [pm._gpu_free_step(str(cell))] + modelled_steps}


class ProperSideRun(lifecycle.SideRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_reader = sys.modules[__name__]
        self.purpose = self.diagnostic_case["purpose"]

    def _check_diagnostic_case(self, step=None):
        if not super()._check_diagnostic_case(step):
            return False
        try:
            plan = core.read_pinned(self.diagnostic_case["plan"])
            for name in ("prepared", "config_builder", "source", "calibration_registry",
                         "replay_target", "memory_model", "readiness_profile"):
                core.checked_path(plan[name])
        except (ValueError, OSError, KeyError) as exc:
            self.refused = True
            self.failures.append(f"proper source qualification refused: {exc}")
            return False
        return True

    def _serve_env(self, step):
        env = dict(super()._serve_env(step) or os.environ)
        plan = core.read_pinned(self.diagnostic_case["plan"])
        requested = {key: value.replace("{repeat}", str(step["repeat"])) for key, value in
                     plan["native_environment" if self.side == "real" else "modelled_environment"].items()}
        env.update(requested)
        if self.side == "real" and step["role"] == "serve":
            if "TMPDIR" in requested and len(os.fsencode(requested["TMPDIR"])) + 37 > 107:
                raise ValueError("native TMPDIR is too long for ATOM UUID IPC socket paths")
            for key in ("TMPDIR", "AIPERF_DATASET_MMAP_BASE_PATH", "ATOM_COMPILE_CACHE_ROOT",
                        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
                path = requested.get(key)
                if path:
                    Path(path).mkdir(parents=True, exist_ok=False)
        return env

    def _source(self, step):
        value = super()._source(step)
        value["proper_plan"] = self.diagnostic_case["plan"]
        return value

    def _check_artifact(self, step, entry, execution):
        path = self.cell / f"{self.side}.r{step['repeat']}.json"
        try:
            blob = json.loads(path.read_text())
            check_result(blob, self.diagnostic_case)
            if blob["repeat"] != step["repeat"]:
                raise ValueError("proper result belongs to another repeat")
            expected = (execution.get("server_process") or {}).get("said") or {}
            actual = blob["server"].get("server_process") or {}
            if any(actual.get(k) != expected.get(k) for k in ("pid", "start_ticks", "boot_id")):
                raise ValueError("proper result came from another engine process")
            execution["replay"]["reported_wall_window"] = blob["wall_window"]
            execution["replay"]["measured_window"] = blob["execution_wall_window"]
            self._stamp(path, blob, execution)
            self._write_execution(execution)
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.failures.append(f"{step['id']}: {exc}")
            entry["ok"] = False
            self._stamp_diagnostic_failure(step, execution)
            return False

    def _command(self, step):
        if step["role"] not in ("proper_session", "proper_sessions"):
            return super()._command(step)
        if not self._check_diagnostic_case(step):
            return False
        plan = core.read_pinned(self.diagnostic_case["plan"])
        sessions = step.get("sessions", [step])
        limit = self.diagnostic_case.get("modelled_concurrency", 1)
        if (not sessions or len(sessions) > limit
                or any(item["role"] != "proper_session" or item["side"] != "modelled" for item in sessions)
                or len({item["repeat"] for item in sessions}) != len(sessions)):
            raise ValueError("proper session group differs from its frozen concurrency")
        # Each command execs a fresh interpreter. The parent only polls owned
        # processes; no simulation clock, runtime globals, or caches are shared.
        active, started_sessions = {}, []
        try:
            for item in sessions:
                started = self.wall()
                env = self._serve_env(item)
                proc = self.processes.start(item["command"], log=self._log(item), env=env)
                execution = self._mint(item, proc, started)
                execution["process"]["role"] = "controlled_session"
                execution["config"]["provenance_transport"] = "owned local startup artifact"
                held = {"proc": proc, "pid": proc.pid, "step": item, "execution": execution,
                        "started": started, "ready": None,
                        "deadline": self.now() + plan.get("session_wall_timeout_seconds", 3600)}
                self.running[item["id"]] = held
                active[item["id"]] = held
                started_sessions.append(held)
            while active:
                for name, held in list(active.items()):
                    item, proc, execution = held["step"], held["proc"], held["execution"]
                    if self.processes.alive(proc):
                        if self.now() > held["deadline"]:
                            raise TimeoutError("controlled proper session exceeded its wall bound")
                        ready_path = self.cell / f"modelled.r{item['repeat']}.raw/startup_ready.json"
                        if held["ready"] is None and ready_path.exists():
                            ready = json.loads(ready_path.read_text())
                            said = ready["server"]
                            if not self._check_server_process(item, said, execution, proc):
                                raise ValueError("controlled session startup identity is unverified")
                            held["ready"] = ready
                            execution["config"]["provenance"] = said
                            execution["process"]["healthy_at"] = ready["at"]
                            execution["process"]["startup_s"] = ready["at"] - held["started"]
                            self._record(dict(item, id=item["id"]+"-ready", role="serve"),
                                         startup_s=ready["at"]-held["started"], ok=True)
                            signal = {"plan_sha256": self.diagnostic_case["workload_sha256"],
                                      "pid": proc.pid, "execution_id": execution["execution_id"]}
                            signal_path = ready_path.with_name("start_profile.json")
                            temporary = signal_path.with_suffix(".writing")
                            temporary.write_text(json.dumps(signal))
                            temporary.replace(signal_path)
                        continue
                    code, ended = proc.returncode, self.wall()
                    seconds = ended - held["started"]
                    execution["process"].update(ended_at=ended, exit=code)
                    execution["replay"] = {"pid": proc.pid, "command": item["command"],
                        "started_at": held["started"], "ended_at": ended, "seconds": seconds, "exit": code}
                    entry = self._record(dict(item, role="replay"), pid=proc.pid, exit=code,
                                         execution_id=execution["execution_id"], seconds=seconds, ok=code == 0)
                    if code != 0 or held["ready"] is None:
                        self.failures.append("controlled proper profile did not complete after a verified startup")
                        self._stamp_diagnostic_failure(item, execution)
                        return False
                    if not self._check_artifact(item, entry, execution):
                        return False
                    active.pop(name)
                if active:
                    self.sleep(.05)
            return True
        except BaseException as exc:
            self.failures.append(f"controlled proper session: {exc}")
            return False
        finally:
            for held in started_sessions:
                code = self.processes.stop(held["proc"])
                execution = held["execution"]
                if "ended_at" not in execution["process"]:
                    execution["process"].update(ended_at=self.wall(), exit=code)
                self.running.pop(held["step"]["id"], None)
                self._write_execution(execution)



def as_metric_run(blob, *, client=True):
    rows = [r for r in blob["records"] if not r["cancelled"] and not r["error"]]
    run = compare.Run("", blob["side"], manifest={
        "server": blob["server"], "paced": blob["side"] == "real",
        "prepare": blob.get("preparation"), "cache_boundary": blob.get("cache_boundary"),
        "cache_state_after": blob.get("cache_after"),
        "prompt_encoding": {"kind": "chat_messages"}},
        clock="wall" if blob["side"] == "real" else "virtual")
    engine = {row["request_id"]: row for row in blob["engine"].get("requests") or []}
    for index, row in enumerate(rows):
        if client:
            joined = {"arrive_time": row["start_ns"] / 1e9,
                      "first_token_time": (row["first_visible_ns"] or row["start_ns"]) / 1e9,
                      "finish_time": row["end_ns"] / 1e9}
        else:
            joined = engine[row["response_id"]]
        run.joined[index] = joined
        run.usage[index] = {"completion_tokens": row["output_tokens"], "prompt_tokens": row["input_tokens"]}
        run.workload.append({"input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]})
    return run


def compare_dynamic(real, modelled):
    observed = {side: compare.metrics(as_metric_run(blob), range(len(as_metric_run(blob).joined)))
                for side, blob in (("real", real), ("modelled", modelled))}
    for side, blob in (("real", real), ("modelled", modelled)):
        phase = blob.get("phase") or {}
        origin, completed = phase.get("origin_ns"), phase.get("completed_ns")
        if type(origin) is not int or type(completed) is not int or completed <= origin:
            raise ValueError("proper throughput requires positive profiling-phase endpoints")
        window = (completed - origin) / 1e9
        if not math.isfinite(window) or window <= 0:
            raise ValueError("proper profiling-phase duration must be finite and positive")
        # Include idle and grace time in the serving clock. The simulator's
        # physical execution window measures speedup, not serving throughput.
        observed[side]["window_s"] = window
        observed[side]["throughput_tok_s"] = observed[side]["output_tokens"] / window
    metrics = {}
    for name in ("ttft", "tpot", "latency"):
        r = compare._quantiles(list(observed["real"][name].values()))
        m = compare._quantiles(list(observed["modelled"][name].values()))
        metrics[name] = {"real": r, "modelled": m, "error_pct": compare._error(r, m)}
    r, m = observed["real"]["throughput_tok_s"], observed["modelled"]["throughput_tok_s"]
    metrics["throughput_tok_s"] = {"real": r, "modelled": m,
        "error_pct": (m-r)/r*100 if r else None,
        "real_window_s": observed["real"]["window_s"], "modelled_window_s": observed["modelled"]["window_s"],
        "real_output_tokens": observed["real"]["output_tokens"],
        "modelled_output_tokens": observed["modelled"]["output_tokens"],
        "window_basis": "profiling phase origin_ns through completed_ns",
        "numerator_basis": "successful completed requests only",
        "real_clock": "wall", "modelled_clock": "virtual"}
    return {"metrics": metrics, "metric_domain": "actual visible client SSE / transport completion",
            "quantile_convention": compare.QUANTILE_CONVENTION}


def pair_input_observations(real, modelled):
    observed = core.pairing_observations(real, modelled)
    if observed["shared_marker_conversation_turns"] == 0:
        raise ValueError("proper pair has no shared marker/conversation/turn identities")
    if observed["marked_input_differences"]:
        raise ValueError("proper pair changed a shared marked payload or consumed token sequence")
    return observed



def cost_claim_errors(costs, validate):
    """Missing accounting is advisory; supplied numerical claims remain checked."""
    if not costs:
        return []
    bad = []
    if costs.get("cost_schema") != validate.COSTS_SCHEMA:
        bad.append("supplied costs use an unsupported cost schema")
    _, reasons = validate._timing(costs)
    bad += reasons
    for name in validate.MEASURED_COST_TERMS:
        value = costs.get(name)
        if value is not None and (type(value) not in (int, float) or not validate._finite(value) or value < 0):
            bad.append(f"supplied cost {name} is not a finite nonnegative duration")
    for side in ("real", "modelled"):
        claimed = costs.get("execution_" + side) is not None or bool(
            (costs.get("execution_by_repeat") or {}).get(side))
        if claimed and (costs.get("execution_clocks") or {}).get(side) != "wall":
            bad.append(f"supplied {side} execution cost is not labelled wall-clock")
    for name in validate.SUPPLIED_COST_TERMS:
        value = costs.get(name)
        if value is None or value == {} or value == [] or validate._unknown_disclosure(value):
            continue
        parts = validate._parts(costs, name)
        if not parts or (isinstance(value, list) and len(parts) != len(value)):
            bad.append(f"supplied {name} cost lacks structured duration provenance")
            continue
        for part in parts:
            if validate._unknown_disclosure(part):
                continue
            if (type(part.get("seconds")) not in (int, float)
                    or not validate._finite(part.get("seconds")) or part["seconds"] < 0
                    or not part.get("source") or "within" not in part
                    or part["within"] not in (None,) + validate.MEASURED_COST_TERMS):
                bad.append(f"supplied {name} cost has invalid duration/provenance/containment")
        known = set(validate._by_repeat(costs, "execution_by_repeat", "modelled" if name == "derivation" else "real"))
        if name in ("derivation", "load") and known:
            if validate._foreign_repeats(costs, name, known):
                bad.append(f"supplied {name} cost names a foreign repeat")
            if name == "derivation" and validate._unplaced_repeats(costs, name) > 0:
                bad.append("supplied derivation duration belongs to no repeat")
    return bad

def pair(args):
    case = load_case(args.plan, args.plan_sha256, args.case_id)
    plan = core.read_pinned(case["plan"])
    cell = Path(args.cell)
    if identity(json.loads((cell / "diagnostic_case.json").read_text())) != identity(case):
        raise ValueError("proper pair directory belongs to different profile pins")
    validate = lifecycle._load("cc_traces_validate")
    registry = core.read_pinned(plan["calibration_registry"])
    journals = {side: json.loads((cell / f"run.{side}.json").read_text())
                for side in ("real", "modelled")}
    paths = {side: [cell / f"{side}.r{repeat}.json" for repeat in range(1, case["repeats"]+1)]
             for side in journals}
    failures, notes, reports, memory, inputs, counts = [], [], [], [], [], []
    execution_windows = []
    for side, journal in journals.items():
        if journal.get("ok") is not True or journal.get("purpose") != case["purpose"]:
            failures.append(f"{side} lifecycle did not complete its predeclared purpose")
        if set(validate._runs(cell, side)) != set(paths[side]):
            failures.append(f"{side} artifacts differ from the frozen repeat count")
    failures += validate.check_gpu_free(cell, paths["modelled"])
    isolation_path = cell / "isolation.json"
    isolation = json.loads(isolation_path.read_text()) if isolation_path.exists() else {}
    if not isolation or isolation.get("verdict") == "own_contaminated":
        failures.append("proper pair has no usable native isolation evidence")
    elif not isolation.get("isolated"):
        notes.append("native isolation is advisory: " + str(isolation.get("verdict")))
    forbidden = {str(path): lifecycle.file_digest(path)["sha256"]
                 for pattern in ("*_steps*.jsonl", "real.r*_memory*.json")
                 for path in cell.glob(pattern)}
    forbidden[plan["prepared"]["path"]] = plan["prepared"]["sha256"]
    manifests = {side: [] for side in journals}
    for repeat in range(1, case["repeats"]+1):
        blobs = {side: json.loads(paths[side][repeat-1].read_text()) for side in journals}
        for side, blob in blobs.items():
            check_result(blob, case)
            validate.replay_client.read_wall_window(blob.get("execution_wall_window"))
            if blob["repeat"] != repeat:
                raise ValueError("proper artifact belongs to another repeat")
            execution = blob.get("execution") or {}
            if (execution.get("purpose") != case["purpose"]
                    or not lifecycle.execution_id.verify_execution_id(execution)
                    or identity(execution.get("diagnostic_case") or {}) != identity(case)):
                failures.append(f"{side}[{repeat}] execution identity/purpose differs")
        inputs.append(pair_input_observations(blobs["real"]["records"], blobs["modelled"]["records"]))
        real, modelled = (as_metric_run(blobs[side]) for side in ("real", "modelled"))
        for side, run in (("real", real), ("modelled", modelled)):
            manifests[side].append(run.manifest)
            failures += validate.check_clocks_finite(run)
            failures += validate.check_engine(run, 1, side, expected_cache_policy=case["cache_policy"])
            failures += validate.check_cache_policy_evidence(run.manifest, case["cache_policy"], side)
        source_bad, source_notes = lifecycle.opening_module.check_source_contract(
            modelled, registry, case["profile"]["source_sha256"], forbidden, f"proper repeat {repeat}")
        failures += source_bad
        notes += [note for note in source_notes if note not in notes]
        for check in (validate.check_calibration, validate.check_capacity_provenance,
                      validate.check_scalar_overheads):
            failures += check(modelled, registry, 1, case["profile"]["source_sha256"], forbidden)
        failures += validate.check_predictor_device_freedom(modelled, f"proper repeat {repeat}")
        failures += validate.check_capacity_inputs(modelled, f"proper repeat {repeat}")
        failures += validate.check_reference_budget_is_measured(real, f"proper repeat {repeat}")
        memory_bad, observed_memory = validate.check_memory_terms(
            real, modelled, cell, repeat, f"proper repeat {repeat}", expected_cache_policy=case["cache_policy"])
        failures += memory_bad
        memory.append(observed_memory)
        reports.append(compare_dynamic(blobs["real"], blobs["modelled"]))
        counts.append({side: blob["phase"]["counts"] for side, blob in blobs.items()})
        execution_windows.append({side: blob["execution_wall_window"] for side, blob in blobs.items()})
    for side in manifests:
        failures += validate.check_who_served(journals[side], manifests[side], side)
    scores = validate._across_repeats(reports)
    failures += [f"{name} lacks a passing maintained tolerance" for name in validate.TOLERANCE_PCT
                 if scores.get(name, {}).get("within_tolerance") is not True]
    costs_path = cell / "costs.json"
    if not costs_path.exists() and all((cell / f"costs.{side}.json").exists() for side in journals):
        lifecycle.main(["costs", str(cell), *plan.get("cost_args", [])])
    costs = json.loads(costs_path.read_text()) if costs_path.exists() else {}
    failures += cost_claim_errors(costs, validate)
    if costs:
        for side in journals:
            if costs.get("execution_" + side) is not None or (costs.get("execution_by_repeat") or {}).get(side):
                failures += validate.check_costs_cover_runs(costs, side, paths[side], journals[side], case["repeats"])
    speedup = validate._speedup(costs, 1)
    if speedup.get("replay_ratio") is None:
        reason = "measured execution/derivation accounting unavailable: " + str(speedup.get("reason"))
        notes.append(reason + "; speedup accounting is advisory")
    passed = bool(reports) and not failures
    output = {"schema": "compass.aiperf_proper_pair_result/1", "case": identity(case),
              "purpose": case["purpose"], "passed": passed,
              "accepted": passed and case["purpose"] == lifecycle.ACCEPTANCE,
              "repeats": case["repeats"], "failures": failures, "notes": notes,
              "metrics": scores, "detail": reports, "input_observations": inputs,
              "counts": counts, "memory": memory, "isolation": isolation,
              "speedup": speedup, "speedup_target_advisory": True,
              "execution_wall_windows": execution_windows,
              "calibration_registry_sha256": plan["calibration_registry"]["sha256"]}
    path = Path(args.out) if args.out else cell / "proper_pair.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cell", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    side = sub.add_parser("side")
    side.add_argument("--side", required=True, choices=("real", "modelled"))
    side.add_argument("--port", type=int, default=8000)
    side.add_argument("--engine-port", type=int, default=29500)
    side.add_argument("--plan-only", action="store_true")
    side.add_argument("--allow-advisory-isolation", action="store_true")
    paired = sub.add_parser("pair")
    paired.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.action == "pair":
            return pair(args)
        case = load_case(args.plan, args.plan_sha256, args.case_id)
        cell = Path(args.cell).resolve()
        expected = f"tp1_{args.case_id}_c1"
        if cell.name != expected:
            raise ValueError(f"proper case directory must be named {expected}")
        built = build_steps(case, cell, port=args.port, engine_port=args.engine_port,
                            advisory=args.allow_advisory_isolation)
        if args.plan_only:
            print(json.dumps(built, indent=2))
            return 0
        return ProperSideRun(built, args.side, purpose=case["purpose"]).run()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"proper replay refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
