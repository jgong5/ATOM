"""Coverage-only ordinary AIPerf run against an owned native ATOM HTTP server.

The frozen C1 profile controls session trees, not in-flight requests. This
driver exports no timing prices and cannot qualify a paired prediction run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cc_traces_run as lifecycle
from atom.compass.core.proper_replay import checked_path, phase_accounting, read_pinned, validate_records
from atom.compass.replay.native_preparation import check_runtime, native_provenance, prepare_native, write

ARM = "unprofiled_control"


def pin(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_plan(plan, output):
    if (plan.get("schema") != "compass.aiperf_native_coverage/1"
            or plan.get("coverage_only") is not True
            or plan.get("fresh_final_pair_required") is not True
            or plan.get("arm") != ARM):
        raise ValueError("native coverage plan has the wrong scope")
    if Path(plan["output_directory"]).resolve() != output.resolve():
        raise ValueError("native output path differs from the sealed plan")
    for item in plan["source_files"]:
        checked_path({"path": str(ROOT / item["path"]), "sha256": item["sha256"]})
    for name in ("prepared", "config_builder", "source", "request_scope", "metadata_audit"):
        checked_path(plan[name])
    if any(key.startswith("AIPERF_") and key not in plan["environment"] for key in os.environ):
        raise ValueError("unsealed AIPerf environment override is present")
    if plan["environment"].get("AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES") != "false":
        raise ValueError("native coverage requires provided assistant history")
    if plan["environment"].get("COMPASS_NATIVE_COVERAGE") != "1":
        raise ValueError("native coverage must retain actual scheduler allocations")
    if plan["service_config"].get("api_port") is not None or "AIPERF_API_SERVER_PORT" in plan["environment"]:
        raise ValueError("native coverage AIPerf API must remain disabled")
    args = plan["server_argv"]
    for flag, expected in (("--server-port", str(plan["ports"]["http"])),
                           ("--port", str(plan["ports"]["engine"])),
                           ("--model", plan["model"]),
                           ("--compass-measure-out", plan["step_journal"])):
        if args.count(flag) != 1 or args[args.index(flag) + 1] != expected:
            raise ValueError(f"native server argument differs from the sealed field: {flag}")
    if plan["url"] != f"http://127.0.0.1:{plan['ports']['http']}":
        raise ValueError("native URL differs from its owned port")
    if not plan["aiperf_dependency"].get("source_files"):
        raise ValueError("native dependency source tree is not pinned")
    if any("fixed-absolute" in arg or "opening-plan" in arg for arg in plan["server_argv"]):
        raise ValueError("native coverage cannot install a fixed request calendar")


def check_native_scope(plan, provenance):
    workers = provenance.get("worker_runtime") or []
    if len(workers) != 1:
        raise ValueError("native coverage requires one actual worker runtime")
    cache_dir = workers[0].get("configuration", {}).get("compilation_cache_dir")
    if not cache_dir or not Path(cache_dir).is_relative_to(plan["environment"]["ATOM_COMPILE_CACHE_ROOT"]):
        raise ValueError("native worker used a different compile-cache root")
    native = workers[0].get("native_attention") or {}
    expected = read_pinned(plan["request_scope"])["attention_scope"]
    actual = native.get("declaration", {}).get("scopes")
    if actual is None:
        raise ValueError("worker did not report its instantiated attention scope")
    for family in ("unified", "gdn"):
        if actual.get(family) != expected.get(family):
            raise ValueError(f"native instantiated {family} scope differs from the pinned source treatment")
    if native.get("body_flags") != plan["backend_body_flags"]:
        raise ValueError("native resolved FLA body flags differ from the source treatment")
    return native


def check_empty_cache(value):
    ranks = value.get("ranks") or []
    if len(ranks) != 1:
        raise ValueError("native cache boundary lacks its single engine snapshot")
    for rank in ranks:
        if rank.get("indexes") != {"kv": 0, "state": 0} or not rank.get("quiescence", {}).get("idle"):
            raise ValueError("native profile did not start from the prepared empty cache boundary")


REQUEST_FACTS = (
    "request_id", "response_id", "conversation_id", "turn_index", "source_trace_id",
    "source_outer_idx", "source_inner_idx", "source_kind", "root_correlation_id",
    "parent_correlation_id", "agent_depth", "cache_bust_marker", "cache_bust_target",
    "input_tokens", "prompt_token_sha256", "payload_sha256", "output_tokens",
    "sampling", "max_completion_tokens", "cancelled",
)
STEP_FACTS = ("num_scheduled_tokens", "context_lens", "num_prefill_tokens",
              "topology", "rank_coords", "capture_bucket", "compiled",
              "produces_output", "req_ids")
DECISION_FACTS = ("kind", "tick", "waiting", "waiting_prefill_outstanding",
                  "waiting_held_for_arrival", "running", "batched_tokens", "token_budget")


def coverage_facts(records, steps, before, after, phase, profile, plan_sha):
    """Allowlist structural fields; raw timings never enter calibration input."""
    requests = [{key: row[key] for key in REQUEST_FACTS} for row in records]
    ids = {row["response_id"] for row in records if row["response_id"]}
    scheduled = []
    for row in steps:
        req_ids = row.get("req_ids") or []
        if not req_ids or not set(req_ids) <= ids:
            raise ValueError("native profiling step lacks attributed raw request identities")
        decision = row.get("decision") or {}
        allocation = decision.get("allocation")
        if allocation is None:
            raise ValueError("native profile step lacks actual ScheduledBatch allocation")
        item = {key: row[key] for key in STEP_FACTS}
        item["ordinal"] = len(scheduled)
        item["decision"] = {key: decision[key] for key in DECISION_FACTS}
        item["allocation"] = {key: allocation[key] for key in (
            "source", "block_tables", "cached_tokens", "state_rows", "state_slots",
            "state_fork_srcs", "num_prefill_seqs")}
        maintenance = allocation["state_maintenance"]
        item["allocation"]["state_maintenance"] = None if maintenance is None else {
            key: maintenance[key] for key in ("relocations", "checkpoint_stores", "checkpoint_restores")}
        scheduled.append(item)
    if not scheduled:
        raise ValueError("native profile produced no attributed scheduler shapes")
    # Cache snapshots contain counters and occupancy. Keep only explicit facts.
    def cache_facts(value):
        return [{"indexes": rank["indexes"],
                 "cache_statistics": {key: rank["cache_statistics"][key] for key in (
                     "requests", "cached_tokens", "compressed_tokens", "wanted_tokens", "reusable_tokens", "full_tokens")},
                 "quiescence": {"idle": rank["quiescence"]["idle"]}}
                for rank in value["ranks"]]
    return {"schema": "compass.native_coverage_facts/1", "arm": ARM,
            "plan_sha256": plan_sha, "coverage_only": True,
            "fresh_final_pair_required": True, "accepted": False,
            "profile": {key: profile[key] for key in (
                "scenario", "clients", "seed", "benchmark_id", "context_mode",
                "source_sha256", "dataset_sha256", "config_sha256")},
            "request_counts": phase["counts"], "requests": requests,
            "scheduled_steps": scheduled, "cache_before": cache_facts(before),
            "cache_after": cache_facts(after),
            "timing_prices_present": False}


def free_ports(ports):
    for port in ports.values():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def process_tree(proc):
    import psutil
    parent = psutil.Process(proc.pid)
    return [(p.pid, p.create_time()) for p in [parent, *parent.children(recursive=True)]]


def tree_closed(identities):
    import psutil
    for pid, created in identities:
        try:
            process = psutil.Process(pid)
            if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                return False
        except psutil.NoSuchProcess:
            pass
    return True


def deadline_handler(signum, frame):
    raise TimeoutError("native coverage driver exceeded the sealed phase wall bound")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    plan = read_pinned({"path": args.plan, "sha256": args.plan_sha256})
    output = Path(args.out).resolve()
    validate_plan(plan, output)
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir()
    receipt_base = {"plan_sha256": args.plan_sha256, "arm": ARM,
                    "coverage_only": True, "accepted": False, "fresh_final_pair_required": True}
    def receipt(name, **values):
        write(output / name, dict(receipt_base, **values))
    processes = lifecycle.Processes()
    proc, owned, failure = None, [], None
    code = None
    started = time.monotonic()
    old_alarm = signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(plan["bounds"]["workload_seconds"])
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        for name in ("tmp", "mmap"):
            (private / name).mkdir()
        os.environ.update(plan["environment"])
        sys.path.insert(0, str(Path(plan["aiperf_dependency"]["checkout"]) / "src"))
        from aiperf.common.config import ServiceConfig
        from aiperf.common.enums import ExportLevel
        from atom.compass.replay.aiperf_profile import compare_native_metadata, load_profile, verify_dependency
        from atom.compass.replay.aiperf_native import run_native_profile
        from atom.compass.replay.aiperf_records import normalize_records
        from atom.compass.prefix_workload import tokenizer_identity
        from atom.model_engine.llm_engine import _load_tokenizer

        dependency = verify_dependency(plan)
        config, conversations, metadata, identity, caps = load_profile(plan)
        tokenizer = _load_tokenizer(plan["model"], False)
        if (tokenizer_identity(tokenizer, plan["model"]) != plan["tokenizer"]
                or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != plan["chat_template_sha256"]):
            raise ValueError("native tokenizer/template differs from frozen input")
        services = ServiceConfig.model_validate(plan["service_config"])
        if services.api_enabled:
            raise ValueError("unsealed AIPerf API is enabled")
        ipc = services.comm_config.path
        ipc.mkdir(parents=True, exist_ok=False)
        cache_root = Path(plan["environment"]["ATOM_COMPILE_CACHE_ROOT"])
        cache_root.mkdir(parents=True, exist_ok=False)
        endpoint = config.endpoint.model_copy(update={"urls": [plan["url"]]})
        export = config.output.model_copy(update={"artifact_directory": private / "aiperf",
                  "export_level": ExportLevel.RAW, "export_http_trace": True})
        config = config.model_copy(update={"endpoint": endpoint, "output": export})
        actual_config = config.model_dump(mode="json")
        actual_config.pop("cli_command", None)
        if actual_config != plan["user_config"]:
            raise ValueError("actual native user configuration differs from the sealed plan")
        if services.model_dump(mode="json") != plan["resolved_service_config"]:
            raise ValueError("actual native service configuration differs from the sealed plan")
        write(private / "actual_config.json", {"user": config.model_dump(mode="json"),
              "services": services.model_dump(mode="json"), "dependency": dependency})
        free_ports(plan["ports"])
        receipt("START.json", driver_pid=os.getpid(), server_argv=plan["server_argv"],
                ports=plan["ports"], started_at=time.time())
        proc = processes.start(plan["server_argv"], log=private / "server.log",
                               cwd=ROOT, env=dict(os.environ))
        limit = started + plan["bounds"]["startup_seconds"]
        signal.alarm(max(1, int(limit - time.monotonic())))
        provenance = None
        while processes.alive(proc):
            if time.monotonic() >= limit:
                raise TimeoutError("owned native HTTP server did not become ready")
            if lifecycle.http_get(plan["url"] + "/health", timeout=2) is not None:
                provenance = native_provenance(plan["url"])
                break
            time.sleep(.2)
        if provenance is None or provenance.get("server_process", {}).get("pid") != proc.pid:
            raise ValueError("native provenance does not belong to the owned HTTP process")
        owned = process_tree(proc)
        check_runtime(plan, provenance, "real", lifecycle.opening_module)
        scope = check_native_scope(plan, provenance)
        write(private / "startup_provenance.json", provenance)
        replay = lifecycle._load("replay")
        preparation = prepare_native(plan["url"], config, conversations, replay, private)
        before = replay._prefix_cache_snapshot(plan["url"], 120)
        check_empty_cache(before)
        preparation_steps = len(Path(plan["step_journal"]).read_text().splitlines())
        if time.monotonic() >= limit:
            raise TimeoutError("native server startup and preparation exceeded their shared bound")
        signal.alarm(max(1, int(started + plan["bounds"]["workload_seconds"] - time.monotonic())))
        messages = run_native_profile(config, services)
        write(private / "phase_messages.json", messages)
        phase = phase_accounting(messages)
        if phase["counts"]["final_request_errors"]:
            raise ValueError("native AIPerf reported request errors")
        observed_metadata = [m["metadata"] for m in messages
                             if m.get("message_type") == "dataset_configured_notification"]
        if len(observed_metadata) != 1:
            raise ValueError("native run lacks one DatasetConfigured observation")
        metadata_comparison = compare_native_metadata(metadata, observed_metadata[0], conversations)
        summary = json.loads(config.output.profile_export_json_file.read_text())
        if summary.get("was_cancelled") or summary.get("error_summary"):
            raise ValueError("ordinary native AIPerf reported cancellation/errors")
        raw_records = [json.loads(line) for line in config.output.profile_export_raw_jsonl_file.read_text().splitlines() if line]
        flush = replay._flush_measurements(plan["url"], 120)
        engine = replay._drain_records(plan["url"], 120)
        after = replay._prefix_cache_snapshot(plan["url"], 120)
        final_provenance = native_provenance(plan["url"])
        check_runtime(plan, final_provenance, "real", lifecycle.opening_module)
        if check_native_scope(plan, final_provenance) != scope:
            raise ValueError("native backend scope changed during the profile")
        consumed = {row["request_id"]: row.get("shared_preprocessing") or {}
                    for row in engine.get("requests") or []}
        records = normalize_records(raw_records, user_config=config, tokenizer=tokenizer,
                  model_path=plan["model"], consumed=consumed, expected_caps=caps,
                  default_chat_template_kwargs=plan["server_options"]["default_chat_template_kwargs"])
        validate_records(records, phase)
        steps = [json.loads(line) for line in Path(plan["step_journal"]).read_text().splitlines() if line]
        facts = coverage_facts(records, steps[preparation_steps:], before, after, phase, identity, args.plan_sha256)
        write(private / "observations.json", {"records": records, "phase": phase,
              "engine": engine, "cache_before": before, "cache_after": after,
              "flush": flush, "metadata_comparison": metadata_comparison,
              "preparation": preparation, "server": final_provenance})
        write(output / "COVERAGE_FACTS.json", facts)
        receipt("WORKLOAD_COMPLETE.json", success=True, completed_at=time.time(),
                coverage=pin(output / "COVERAGE_FACTS.json"),
                request_counts=phase["counts"], scheduled_steps=len(facts["scheduled_steps"]))
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if hasattr(exc, "phase_messages"):
            write(private / "partial_phase_messages.json", exc.phase_messages)
    finally:
        signal.alarm(plan["bounds"]["close_seconds"])
        try:
            if proc is not None:
                if processes.alive(proc):
                    owned = list(set(owned + process_tree(proc)))
                code = processes.stop(proc, grace=plan["bounds"]["close_seconds"])
            else:
                code = None
            closed = tree_closed(owned)
            if not closed and failure is None:
                failure = {"type": "RuntimeError", "message": "owned native engine descendants remain alive"}
        except BaseException as exc:
            closed = False
            failure = failure or {"type": type(exc).__name__, "message": str(exc)}
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        receipt("NATIVE_FAILURE.json" if failure else "NATIVE_COMPLETE.json",
                success=failure is None, engine_closed=closed, server_exit=code,
                ended_at=time.time(), owned_processes=owned, failure=failure)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
