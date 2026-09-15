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
    if Path(plan["source_root"]).resolve() != ROOT.resolve():
        raise ValueError("native driver is not running from the sealed isolated source tree")
    if Path(plan["output_directory"]).resolve() != output.resolve():
        raise ValueError("native output path differs from the sealed plan")
    if Path(plan["environment"]["ATOM_COMPILE_CACHE_ROOT"]).resolve() != output / "private/compile":
        raise ValueError("native compile cache must be owned beneath the private run output")
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
    cache_paths = native.get("generated_cache_paths") or {}
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        expected_path = plan["environment"].get(key)
        if (not expected_path or cache_paths.get(key) != expected_path
                or not Path(expected_path).is_relative_to(plan["environment"]["ATOM_COMPILE_CACHE_ROOT"])):
            raise ValueError(f"native generated cache path is not the owned plan path: {key}")
    expected = read_pinned(plan["request_scope"])["attention_scope"]
    actual = native.get("declaration", {}).get("scopes")
    if actual is None:
        raise ValueError("worker did not report its instantiated attention scope")
    capacity_differences = []
    for family in ("unified", "gdn"):
        observed_family, expected_family = dict(actual.get(family) or {}), dict(expected.get(family) or {})
        if family == "unified":
            observed_layout = observed_family.pop("kv_cache_layout", None)
            expected_layout = expected_family.pop("kv_cache_layout", None)
            if observed_layout is None or expected_layout is None:
                raise ValueError("native unified scope lacks bound KV layout")
            observed_layout, expected_layout = dict(observed_layout), dict(expected_layout)
            if observed_layout.keys() != expected_layout.keys():
                raise ValueError("native bound KV tensor identities differ")
            for part in expected_layout:
                observed_tensor, expected_tensor = dict(observed_layout[part]), dict(expected_layout[part])
                observed_shape = list(observed_tensor.pop("shape"))
                expected_shape = list(expected_tensor.pop("shape"))
                if (not observed_shape or not expected_shape
                        or observed_shape[1:] != expected_shape[1:]
                        or observed_tensor != expected_tensor):
                    raise ValueError("native bound KV trailing geometry/strides/dtype differ")
                if observed_shape[0] != expected_shape[0]:
                    capacity_differences.append({"tensor": part, "axis": 0,
                        "meaning": "physical KV block capacity",
                        "source_scope_capacity": expected_shape[0], "native_capacity": observed_shape[0]})
        if observed_family != expected_family:
            raise ValueError(f"native instantiated {family} scope differs from the pinned source treatment")
    if native.get("body_flags") != plan["backend_body_flags"]:
        raise ValueError("native resolved FLA body flags differ from the source treatment")
    return {"native": native, "kv_capacity_differences": capacity_differences,
            "comparison_scope": "coverage only; leading KV capacity may differ; source qualification unchanged"}


def collect_native_closeout(base, replay, journal, *, evidence, reserve_seconds=120,
                            now=time.monotonic, sleep=time.sleep):
    """Wait for native aborts, then require idle API/core and stable flushed records."""
    import urllib.error
    deadline = now() + reserve_seconds
    records, admissions, attempts = [], [], []
    evidence.update(requests=records, admissions=admissions, attempts=attempts)
    last_signature, stable = None, 0
    while now() < deadline:
        timeout = max(.01, min(10, deadline - now()))
        cache = replay._prefix_cache_snapshot(base, timeout)
        drained = replay._drain_records(base, timeout)
        records.extend(drained.get("requests") or [])
        admissions.extend(drained.get("admissions") or [])
        api_idle = drained.get("active_streams") == drained.get("active_api_requests") == 0
        core_idle = all(r.get("quiescence", {}).get("idle") for r in cache["ranks"])
        attempt = {"api_idle": api_idle, "core_idle": core_idle,
                   "new_requests": len(drained.get("requests") or []),
                   "new_admissions": len(drained.get("admissions") or [])}
        attempts.append(attempt)
        if api_idle and core_idle:
            try:
                flush = replay._flush_measurements(base, max(.01, min(10, deadline - now())))
            except urllib.error.HTTPError as exc:
                if exc.code != 409:
                    raise
                stable = 0
                sleep(.05)
                continue
            cache = replay._prefix_cache_snapshot(base, max(.01, min(10, deadline - now())))
            if all(r.get("quiescence", {}).get("idle") for r in cache["ranks"]):
                signature = (pin(journal)["sha256"], json.dumps(cache, sort_keys=True))
                empty = not drained.get("requests") and not drained.get("admissions")
                stable = stable + 1 if empty and signature == last_signature else 0
                last_signature = signature
                attempt["journal_sha256"], attempt["stable_empty_reads"] = signature[0], stable
                if stable >= 2:
                    engine = dict(drained, requests=records, admissions=admissions, count=len(records))
                    evidence.update(flush=flush, final_cache=cache, stable=True)
                    return flush, engine, cache
            else:
                stable, last_signature = 0, None
        else:
            stable, last_signature = 0, None
        sleep(.05)
    raise TimeoutError("native API/core did not reach a stable flushed record/journal drain")


def check_empty_cache(value):
    ranks = value.get("ranks") or []
    if len(ranks) != 1:
        raise ValueError("native cache boundary lacks its single engine snapshot")
    for rank in ranks:
        if rank.get("indexes") != {"kv": 0, "state": 0} or not rank.get("quiescence", {}).get("idle"):
            raise ValueError("native profile did not start from the prepared empty cache boundary")


REQUEST_FACTS = (
    "request_id", "response_id", "engine_request_id", "engine_seq_id",
    "native_tokenized_observed", "native_enqueue_observed", "conversation_id", "turn_index", "source_trace_id",
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
    ids = {str(row["engine_seq_id"]) for row in records if row["engine_seq_id"] is not None}
    scheduled = []
    for row in steps:
        req_ids = [str(value) for value in row.get("req_ids") or []]
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
    served = {str(seq_id) for step in scheduled for seq_id in step["req_ids"]}
    for request in requests:
        request["native_scheduled_observed"] = str(request["engine_seq_id"]) in served
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
    drain_evidence = {}
    code = None
    started = time.monotonic()
    old_alarm = signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(plan["bounds"]["workload_seconds"])
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        for name in ("tmp", "mmap"):
            (private / name).mkdir()
        cache_root = Path(plan["environment"]["ATOM_COMPILE_CACHE_ROOT"])
        cache_root.mkdir(parents=True, exist_ok=False)
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
            cache_path = Path(plan["environment"][key])
            if not cache_path.is_relative_to(cache_root):
                raise ValueError(f"generated cache escapes the owned root: {key}")
            cache_path.mkdir(parents=True, exist_ok=False)
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
        prepared_empty = replay._drain_records(plan["url"], 120)
        if (prepared_empty.get("requests") or prepared_empty.get("admissions")
                or prepared_empty.get("active_streams") != 0
                or prepared_empty.get("active_api_requests") != 0):
            raise ValueError("native preparation has not completed API/record teardown")
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
        flush, engine, after = collect_native_closeout(
            plan["url"], replay, plan["step_journal"], evidence=drain_evidence,
            reserve_seconds=plan["bounds"]["post_profile_reserve_seconds"])
        write(private / "native_drain.json", drain_evidence)
        final_provenance = native_provenance(plan["url"])
        check_runtime(plan, final_provenance, "real", lifecycle.opening_module)
        if check_native_scope(plan, final_provenance) != scope:
            raise ValueError("native backend scope changed during the profile")
        consumed = {row["request_id"]: row.get("shared_preprocessing") or {}
                    for row in engine.get("requests") or []}
        records = normalize_records(raw_records, user_config=config, tokenizer=tokenizer,
                  model_path=plan["model"], consumed=consumed, expected_caps=caps,
                  admissions=engine["admissions"],
                  default_chat_template_kwargs=plan["server_options"]["default_chat_template_kwargs"])
        validate_records(records, phase)
        steps = [json.loads(line) for line in Path(plan["step_journal"]).read_text().splitlines() if line]
        facts = coverage_facts(records, steps[preparation_steps:], before, after, phase, identity, args.plan_sha256)
        write(private / "observations.json", {"records": records, "phase": phase,
              "engine": engine, "cache_before": before, "cache_after": after,
              "flush": flush, "metadata_comparison": metadata_comparison,
              "preparation": preparation, "server": final_provenance})
        facts["native_runtime_scope"] = {
            "attention_scope": scope["native"]["declaration"]["scopes"],
            "body_flags": scope["native"]["body_flags"],
            "kv_capacity_differences": scope["kv_capacity_differences"]}
        write(output / "COVERAGE_FACTS.json", facts)
        receipt("WORKLOAD_COMPLETE.json", success=True, completed_at=time.time(),
                coverage=pin(output / "COVERAGE_FACTS.json"),
                request_counts=phase["counts"], scheduled_steps=len(facts["scheduled_steps"]))
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if hasattr(exc, "phase_messages"):
            write(private / "partial_phase_messages.json", exc.phase_messages)
        if drain_evidence and not (private / "native_drain.json").exists():
            write(private / "partial_native_drain.json", drain_evidence)
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
