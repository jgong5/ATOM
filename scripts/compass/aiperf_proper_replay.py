"""One ordinary-native or actual controlled AIPerf profile, owned by the pair harness."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import signal
from pathlib import Path
import sys
import time
import traceback
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def script(name):
    spec = importlib.util.spec_from_file_location("proper_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def incomplete_evidence(value):
    """Preserve nonfinite control sentinels explicitly in failure evidence only."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: incomplete_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [incomplete_evidence(item) for item in value]
    return value


def export_phase_messages(messages):
    """Encode only AIPerf's unbounded grace sentinel as the JSON string Infinity."""
    exported = []
    for message in messages:
        value = dict(message)
        config = value.get("config")
        if (value.get("message_type") == "credit_phase_start" and isinstance(config, dict)
                and config.get("grace_period_sec") == math.inf):
            value["config"] = dict(config, grace_period_sec="Infinity")
        exported.append(value)
    return exported


def native_journal_path(plan, output):
    """Diagnostic sweeps may share one server journal; ordinary runs keep their path."""
    shared = plan.get("native_step_journal")
    if shared is None:
        return output.parent / (output.stem + "_steps.jsonl")
    if (plan.get("purpose") != "diagnostic" or not isinstance(shared, str)
            or not shared or not Path(shared).is_absolute()):
        raise ValueError("shared native journal requires an absolute diagnostic plan path")
    return Path(shared)


def native_journal_segment(path, start, end, records, *, file_identity, cancelled_admissions=(),
                           warmup_records=()):
    """Attribute priming and profiling separately within one append interval."""
    expected = {str(row["engine_seq_id"]) for row in records
                if row.get("engine_seq_id") is not None}
    if any(row.get("aborted") is not True for row in cancelled_admissions):
        raise ValueError("unrecorded journal requests require native aborted admissions")
    expected.update(str(row["seq_id"]) for row in cancelled_admissions if row.get("seq_id") is not None)
    warmup_ids = {str(row["engine_seq_id"]) for row in warmup_records
                  if row.get("engine_seq_id") is not None}
    warmup_completed = {str(row.get("engine_seq_id")) for row in warmup_records
                        if not row.get("cancelled") and not row.get("error")}
    completed = {str(row["engine_seq_id"]) for row in records
                 if not row.get("cancelled") and not row.get("error")}
    if ("None" in completed or "None" in warmup_completed or expected & warmup_ids
            or type(start) is not int or type(end) is not int or not 0 <= start <= end):
        raise ValueError("native journal segment lacks a valid request/byte interval")
    digest, observed, count = hashlib.sha256(), set(), 0
    warmup_digest, warmup_observed, warmup_count = hashlib.sha256(), set(), 0
    profile_start = None
    with Path(path).open("rb") as stream:
        stat = os.fstat(stream.fileno())
        if [stat.st_dev, stat.st_ino] != list(file_identity) or stat.st_size < end:
            raise ValueError("native journal was replaced or truncated during the profile")
        if start:
            stream.seek(start - 1)
            if stream.read(1) != b"\n":
                raise ValueError("native journal start cuts through a row")
        stream.seek(start)
        while stream.tell() < end:
            row_start = stream.tell()
            line = stream.readline(end - stream.tell())
            if not line.endswith(b"\n"):
                raise ValueError("native journal end cuts through a row")
            row = json.loads(line)
            ids = {str(value) for value in row.get("req_ids") or []}
            if not ids or not ids <= expected | warmup_ids:
                raise ValueError("native journal contains a request outside this run")
            if ids <= warmup_ids:
                if profile_start is not None:
                    raise ValueError("native warmup steps occur after profiling steps")
                warmup_digest.update(line)
                warmup_observed.update(ids)
                warmup_count += 1
            elif ids <= expected:
                if profile_start is None:
                    profile_start = row_start
                digest.update(line)
                observed.update(ids)
                count += 1
            else:
                raise ValueError("one native forward mixes warmup and profiling requests")
    if not completed <= observed:
        raise ValueError("completed native requests lack attributed journal steps")
    if not warmup_completed <= warmup_observed:
        raise ValueError("completed native warmup requests lack attributed journal steps")
    if profile_start is None:
        profile_start = end
    return {"path": str(path), "profile_start_offset": profile_start, "profile_end_offset": end,
            "profile_region_sha256": digest.hexdigest(), "scheduled_steps": count,
            "scheduled_request_ids": sorted(observed), "file_identity": list(file_identity),
            "warmup": {"start_offset": start, "end_offset": profile_start,
                       "region_sha256": warmup_digest.hexdigest(), "scheduled_steps": warmup_count,
                       "scheduled_request_ids": sorted(warmup_observed)}}


def native_record_admissions(raw_records, admissions, phase, purpose):
    """Keep unsaved cancelled admissions factual, without inventing raw records."""
    if len({row["client_request_id"] for row in admissions}) != len(admissions):
        raise ValueError("native cancellation attribution has duplicate admissions")
    raw_ids = {row["metadata"]["x_request_id"] for row in raw_records}
    missing = [row for row in admissions if row["client_request_id"] not in raw_ids]
    cancelled = sum(row["metadata"].get("was_cancelled") is True for row in raw_records)
    remaining = phase["counts"]["final_requests_cancelled"] - cancelled
    if (remaining < 0 or len(missing) > remaining
            or missing and (purpose != "diagnostic" or any(row.get("aborted") is not True for row in missing))):
        raise ValueError("missing native raw records do not reconcile with aborted admissions and cancelled credits")
    return [row for row in admissions if row["client_request_id"] in raw_ids], missing


def check_native_summary(summary, raw_records, phase, purpose):
    """A diagnostic may retain counted request cancellations, never a failed profile."""
    errors = summary.get("error_summary") or []
    if summary.get("was_cancelled") or errors and purpose != "diagnostic":
        raise ValueError("ordinary native AIPerf reported profile cancellation or errors")
    def error_key(error):
        return tuple(error.get(key) for key in ("type", "code", "message"))
    cancelled = Counter(error_key(row["error"]) for row in raw_records
        if row.get("error") and row["metadata"].get("was_cancelled") is True
        and row["error"].get("type") == "RequestCancellationError" and row["error"].get("code") == 499)
    if sum(cancelled.values()) > phase["counts"]["final_requests_cancelled"]:
        raise ValueError("raw cancellation errors exceed cancelled credits")
    for entry in errors:
        count = entry.get("count")
        key = error_key(entry.get("error_details") or {})
        if type(count) is not int or count <= 0 or count > cancelled[key]:
            raise ValueError("ordinary native AIPerf reported unrelated or unattributed errors")
        cancelled[key] -= count


from atom.compass.replay.native_preparation import (
    check_runtime as _check_runtime, native_provenance, prepare_native,
)


def check_runtime(plan, provenance, side):
    return _check_runtime(plan, provenance, side, script("cc_traces_opening"))


def check_modelled_sources(plan, provenance, cell):
    from types import SimpleNamespace
    from atom.compass.core.proper_replay import read_pinned
    validate = script("cc_traces_validate")
    run = SimpleNamespace(manifest={"server": provenance})
    registry = read_pinned(plan["calibration_registry"])
    source_sha = read_pinned(plan["prepared"])["source"]["sha256"]
    forbidden = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for pattern in ("real.r*_steps*.jsonl", "real.r*_memory*.json")
                 for path in Path(cell).glob(pattern)}
    bad, notes = script("cc_traces_opening").check_source_contract(
        run, registry, source_sha, forbidden, "proper profile startup")
    for check in (validate.check_calibration, validate.check_capacity_provenance,
                  validate.check_scalar_overheads):
        bad += check(run, registry, 1, source_sha, forbidden)
    bad += validate.check_predictor_device_freedom(run, "proper profile startup")
    bad += validate.check_capacity_inputs(run, "proper profile startup")
    if bad:
        raise ValueError("; ".join(bad))
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--side", required=True, choices=("real", "modelled"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--url")
    parser.add_argument("--start-signal")
    args = parser.parse_args(argv)
    raw = Path(args.plan).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise ValueError("proper replay plan changed")
    plan = json.loads(raw)
    if plan.get("schema") != "compass.aiperf_proper_pair/1":
        raise ValueError("unsupported proper replay plan")
    if plan.get("record_export") != {"export_level": "raw", "export_http_trace": True}:
        raise ValueError("proper replay requires predeclared symmetric raw export settings")
    if plan.get("purpose") not in ("acceptance", "diagnostic"):
        raise ValueError("proper replay requires a purpose declared before execution")
    if not 1 <= args.repeat <= plan["repeats"]:
        raise ValueError("proper replay repeat is outside the frozen plan")
    output = Path(args.out)
    if output.name != f"{args.side}.r{args.repeat}.json":
        raise ValueError("proper replay output does not name its declared side/repeat")
    directory = output.with_suffix(".raw")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    core = store = None
    result = None
    failure = None
    closeout = {}
    def deadline_expired(signum, frame):
        raise TimeoutError("proper profile exceeded its declared client wall bound")
    previous_alarm = signal.signal(signal.SIGALRM, deadline_expired)
    signal.alarm(int(plan.get("session_wall_timeout_seconds", 3600)))
    try:
        environment = {key: value.replace("{repeat}", str(args.repeat)) for key, value in
                       plan["native_environment" if args.side == "real" else "modelled_environment"].items()}
        if environment.get("AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES") != "false":
            raise ValueError("proper replay requires the predeclared provided-history environment")
        os.environ.update(environment)
        sys.path.insert(0, str(Path(plan["aiperf_dependency"]["checkout"]) / "src"))
        if args.side == "modelled":
            if Path("/dev/kfd").exists() or Path("/dev/dri").exists():
                raise ValueError("controlled proper replay requires a device-free container")
            if os.environ.get("COMPASS_NATIVE_COVERAGE") == "1":
                raise ValueError("native coverage instrumentation is not a modelled input")
            for key in ("TMPDIR", "AIPERF_DATASET_MMAP_BASE_PATH"):
                if key in environment:
                    Path(environment[key]).mkdir(parents=True, exist_ok=False)
            from atom.compass.replay.bootstrap import install_from_target
            install_from_target(plan["replay_target"]["path"])
        from atom.compass.core.proper_replay import phase_accounting, validate_records, profiling_wall_window
        from atom.compass.replay.aiperf_profile import (
            load_profile, create_modelled_config, controlled_provenance,
            compare_native_metadata, verify_dependency,
        )
        from atom.compass.replay.aiperf_records import (
            export_controlled_records, normalize_records, partition_raw_records,
        )
        from atom.compass.replay.aiperf_runner import ChatServingOptions
        from atom.model_engine.llm_engine import _load_tokenizer
        from atom.compass.prefix_workload import tokenizer_identity

        verify_dependency(plan)
        config, conversations, metadata, identity, caps = load_profile(plan)
        from aiperf.common.enums import ExportLevel
        config = config.model_copy(update={"output": config.output.model_copy(update={
            "artifact_directory": directory / "aiperf", "export_level": ExportLevel.RAW,
            "export_http_trace": plan["record_export"]["export_http_trace"]})})
        tokenizer = _load_tokenizer(plan["model"], False)
        if tokenizer_identity(tokenizer, plan["model"]) != plan["tokenizer"]:
            raise ValueError("proper replay tokenizer differs from the frozen identity")
        if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != plan["chat_template_sha256"]:
            raise ValueError("proper replay chat template changed")
        options = ChatServingOptions(**plan["server_options"])
        replay = script("replay")
        admissions = ()
        source_notes = []
        cache_boundary = None
        journal_segment = None
        cancelled_admissions = []
        if args.side == "real":
            from aiperf.common.config import ServiceConfig
            from aiperf.common.enums import ExportLevel
            from atom.compass.replay.aiperf_native import run_native_profile
            if not args.url:
                raise ValueError("native proper replay requires its owned server URL")
            native_helpers = script("aiperf_native_coverage")
            scope_plan = dict(plan, environment=environment)
            provenance = native_provenance(args.url)
            check_runtime(plan, provenance, "real")
            native_scope = native_helpers.check_native_scope(scope_plan, provenance)
            journal = native_journal_path(plan, output)
            preparation_mode = plan.get("native_preparation", {"mode": "marked_chat_and_dispatch"})
            if preparation_mode.get("mode") not in ("marked_chat_and_dispatch", "dispatch_only"):
                raise ValueError("unknown native preparation mode")
            preparation = prepare_native(args.url, config, conversations, replay, directory,
                step_journal=journal,
                native_scope=native_scope["native"], model=plan["model"],
                dispatch_only=preparation_mode["mode"] == "dispatch_only",
                max_prefill_tokens=preparation_mode.get("max_prefill_tokens"))
            cache_boundary = preparation["cache_boundary"]
            before = replay._prefix_cache_snapshot(args.url, 120)
            native_helpers.check_empty_cache(before)
            empty = replay._drain_records(args.url, 120)
            if (empty.get("requests") or empty.get("admissions")
                    or empty.get("active_streams") != 0 or empty.get("active_api_requests") != 0):
                raise ValueError("native preparation admission/stream teardown is incomplete")
            if plan.get("native_step_journal") is not None:
                journal_stat = journal.stat()
                journal_start = journal_stat.st_size
                journal_identity = [journal_stat.st_dev, journal_stat.st_ino]
            endpoint = config.endpoint.model_copy(update={"urls": [args.url]})
            export = config.output.model_copy(update={"artifact_directory": directory / "aiperf",
                                                      "export_level": ExportLevel.RAW,
                                                      "export_http_trace": True})
            native_config = config.model_copy(update={"endpoint": endpoint, "output": export})
            native_started = time.time()
            service_data = json.loads(json.dumps(plan["service_config"]).replace("{repeat}", str(args.repeat)))
            services = ServiceConfig.model_validate(service_data)
            services.comm_config.path.mkdir(parents=True, exist_ok=False)
            messages = export_phase_messages(run_native_profile(native_config, services))
            wall_finished = time.time()
            write(directory / "phase_messages.json", messages)
            execution_window = profiling_wall_window(messages)
            export_path = native_config.output.profile_export_raw_jsonl_file
            raw_records = [json.loads(line) for line in export_path.read_text().splitlines() if line]
            summary = json.loads(native_config.output.profile_export_json_file.read_text())
            closeout = {}
            flush, engine, after = native_helpers.collect_native_closeout(
                args.url, replay, journal,
                evidence=closeout, reserve_seconds=120)
            if plan.get("native_step_journal") is not None:
                journal_end = journal.stat().st_size
            write(directory / "native_closeout.json", closeout)
            admissions = engine["admissions"]
            provenance = native_provenance(args.url)
            check_runtime(plan, provenance, "real")
            if native_helpers.check_native_scope(scope_plan, provenance) != native_scope:
                raise ValueError("native runtime scope changed during the profile")
            consumed = {row["request_id"]: row.get("shared_preprocessing") or {}
                        for row in engine.get("requests") or []}
            cleanup = {"native_flush": flush, "final_cache": after}
            wall_window = {"started_at": native_started, "ended_at": wall_finished,
                           "seconds": wall_finished - native_started,
                           "scope": "ordinary AIPerf process including setup/export"}
        else:
            from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore
            from atom.compass.replay.aiperf_runner import create_controlled_core, run_controlled_replay
            from atom.compass.core.cache_boundary import snapshot, reset, reset_receipt_errors, RESET_SCHEMA, SNAPSHOT_SCHEMA
            model_config = create_modelled_config(plan, tokenizer, directory)
            core = create_controlled_core(model_config)
            boundary = reset(core)
            cache_boundary = {"schema": RESET_SCHEMA, "acknowledged": boundary["acknowledged"], "ranks": [boundary]}
            errors = reset_receipt_errors(cache_boundary, expected_worker_kind="modelled_no_device",
                                          require_fresh_modelled=True)
            if errors:
                raise ValueError("; ".join(errors))
            before = {"schema": SNAPSHOT_SCHEMA, "ranks": [snapshot(core)]}
            if (before["ranks"][0]["indexes"] != {"kv": 0, "state": 0}
                    or not before["ranks"][0]["quiescence"]["idle"]):
                raise ValueError("controlled proper replay does not start fresh and empty")

            async def materialize():
                backing = MemoryMapDatasetBackingStore(benchmark_id=config.benchmark_id)
                await backing.initialize()
                await backing.add_conversations({c.session_id: c for c in conversations})
                await backing.finalize()
                return backing
            store = asyncio.run(materialize())
            provenance = controlled_provenance(core, tokenizer, options)
            check_runtime(plan, provenance, "modelled")
            source_notes = check_modelled_sources(plan, provenance, output.parent)
            write(directory / "startup_ready.json", {"at": time.time(), "server": provenance})
            if args.start_signal:
                deadline = time.monotonic() + 120
                while not Path(args.start_signal).exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("controlled profile startup was not acknowledged")
                    time.sleep(.01)
                acknowledgement = json.loads(Path(args.start_signal).read_text())
                if acknowledgement.get("plan_sha256") != args.plan_sha256 or acknowledgement.get("pid") != os.getpid():
                    raise ValueError("controlled start signal names another plan/process")
            model_started = time.time()
            result = run_controlled_replay(
                core=core, tokenizer=tokenizer, user_config=config, dataset_metadata=metadata,
                dataset_client_metadata=store.get_client_metadata(), worker_count=1, server_options=options)
            wall_finished = time.time()
            messages = export_phase_messages([m.model_dump(mode="json") for m in result.messages])
            execution_window = profiling_wall_window(result.wall_phase_events)
            raw_records = export_controlled_records(result.records)
            consumed = {row["api_request_id"]: {"input_tokens": row["prompt_tokens"],
                                               "prompt_token_sha256": row["prompt_token_sha256"]}
                        for row in result.dispatches if "api_request_id" in row and "prompt_tokens" in row}
            engine = {"clock": "virtual", "requests": [
                {"request_id": row["api_request_id"], "arrive_time": row["io_processor_arrival"],
                 "first_token_time": row["core_first_token_at"], "finish_time": row["core_completion_at"]}
                for row in result.dispatches if "core_first_token_at" in row and "core_completion_at" in row]}
            provenance = controlled_provenance(core, tokenizer, options)
            check_runtime(plan, provenance, "modelled")
            after = {"schema": SNAPSHOT_SCHEMA, "ranks": [snapshot(core)]}
            if not after["ranks"][0]["quiescence"]["idle"]:
                raise ValueError("controlled profile did not return a quiescent engine")
            cleanup = dict(result.cleanup, final_cache=after)
            preparation = None
            wall_window = {"started_at": model_started, "ended_at": wall_finished,
                           "seconds": wall_finished - model_started,
                           "scope": "controlled AIPerf call including setup/teardown"}
            write(directory / "phase_messages.json", messages)
            write(directory / "controlled_result.json", {"events": result.events,
                  "dispatches": result.dispatches, "serving": result.serving})
        phase = phase_accounting(messages)
        all_raw_records = raw_records
        by_phase = partition_raw_records(all_raw_records)
        raw_records, warmup_raw_records = by_phase["profiling"], by_phase["warmup"]
        write(directory / "raw_records.json", raw_records)
        write(directory / "warmup_raw_records.json", warmup_raw_records)
        if args.side == "real":
            check_native_summary(summary, raw_records, phase, plan["purpose"])
        observed_dataset = [m["metadata"] for m in messages if m.get("message_type") == "dataset_configured_notification"]
        if len(observed_dataset) != 1:
            raise ValueError("actual AIPerf run lacks one dataset metadata observation")
        metadata_comparison = compare_native_metadata(metadata, observed_dataset[0], conversations)
        record_admissions = admissions
        if args.side == "real":
            record_admissions, cancelled_admissions = native_record_admissions(
                all_raw_records, admissions, phase, plan["purpose"])
        warmup_request_ids = {row["metadata"]["x_request_id"] for row in warmup_raw_records}
        warmup_admissions = [row for row in record_admissions if row["client_request_id"] in warmup_request_ids]
        record_admissions = [row for row in record_admissions if row["client_request_id"] not in warmup_request_ids]
        normalized = normalize_records(raw_records, user_config=config, tokenizer=tokenizer,
            model_path=plan["model"], default_chat_template_kwargs=options.default_chat_template_kwargs,
            consumed=consumed, expected_caps=caps, admissions=record_admissions)
        validate_records(normalized, phase)
        warmup_normalized = normalize_records(warmup_raw_records, user_config=config, tokenizer=tokenizer,
            model_path=plan["model"], default_chat_template_kwargs=options.default_chat_template_kwargs,
            consumed=consumed, expected_caps=caps, admissions=warmup_admissions, benchmark_phase="warmup")
        validate_records(warmup_normalized, phase["warmup"], benchmark_phase="warmup")
        write(directory / "warmup_records.json", warmup_normalized)
        if args.side == "real" and plan.get("native_step_journal") is not None:
            journal_segment = native_journal_segment(journal, journal_start, journal_end,
                normalized, file_identity=journal_identity, cancelled_admissions=cancelled_admissions,
                warmup_records=warmup_normalized)
            journal_segment["preparation_start_offset"] = preparation["step_journal_start_offset"]
            journal_segment["preparation_end_offset"] = preparation["step_journal_end_offset"]
            write(directory / "native_journal.json", journal_segment)
        artifact = {"schema": "compass.aiperf_proper_run/1", "side": args.side,
              "purpose": plan["purpose"], "repeat": args.repeat,
              "plan_sha256": args.plan_sha256, "profile": identity, "phase": phase,
              "records": normalized, "engine": engine, "server": provenance,
              "records_phase": "profiling",
              "initialization": {"kind": phase["warmup"]["kind"], "phase": phase["warmup"],
                  "records": warmup_normalized, "native_admissions": warmup_admissions},
              "cache_before": before, "cache_after": after, "cache_boundary": cache_boundary,
              "cache_before_scope": "before canonical AIPerf warmup",
              "preparation": preparation, "dataset_metadata_comparison": metadata_comparison,
              "source_notes": source_notes,
              "record_export": {"export_level": str(config.output.export_level),
                                "export_http_trace": config.output.export_http_trace},
              "cleanup": cleanup, "wall_window": wall_window, "execution_wall_window": execution_window,
              "approximations": {"generated_text": "displayable surrogate" if args.side == "modelled" else "native",
                  "provided_history_excludes_generated_text": True,
                  "frontend_cpu": "unmodelled" if args.side == "modelled" else "actual",
                  "validity_decided_by_workload_invariants_and_e2e_errors": True},
              "complete": True, "accepted": False}
        if journal_segment is not None:
            artifact["native_journal"] = journal_segment
        if args.side == "real":
            artifact["unrecorded_cancelled_admissions"] = cancelled_admissions
        complete = [row for row in normalized if not row["cancelled"] and not row["error"]]
        artifact["run"] = {
            "server": provenance, "complete": True, "requests": len(complete),
            "prompt_lengths": "passed", "trace_sha256": args.plan_sha256,
            "paced": args.side == "real", "prepare": preparation,
            "cache_boundary": cache_boundary, "cache_state_after": after,
            "prompt_encoding": {"kind": "chat_messages"}}
        artifact["workload"] = [{"input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]}
                                for row in complete]
        artifact["results"] = [{"index": index, "ok": True, "response": {
            "id": row["response_id"], "usage": {"prompt_tokens": row["input_tokens"],
                                                "completion_tokens": row["output_tokens"]}}}
            for index, row in enumerate(complete)]
        write(output, artifact)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write(directory / "failure.json", failure)
        if closeout and not (directory / "native_closeout.json").exists():
            write(directory / "partial_native_closeout.json", closeout)
        if not output.exists():
            write(output, {"schema": "compass.aiperf_proper_run/1", "side": args.side,
              "purpose": plan["purpose"], "repeat": args.repeat,
                           "plan_sha256": args.plan_sha256, "complete": False,
                           "failure": failure, "accepted": False})
        partial = getattr(exc, "replay_result", None)
        if partial is not None:
            try:
                write(directory / "controlled_failure.json", incomplete_evidence({
                    "nonfinite_encoding": "explicit nonfinite_float tags; incomplete evidence only",
                    "events": partial.events, "dispatches": partial.dispatches,
                    "records": [r.model_dump(mode="json") for r in partial.records],
                    "messages": [m.model_dump(mode="json") for m in partial.messages],
                    "final_time": partial.final_time, "cleanup": partial.cleanup,
                    "serving": partial.serving, "accepted": False}))
            except Exception as export_exc:
                write(directory / "partial_export_failure.json", {
                    "type": type(export_exc).__name__, "message": str(export_exc),
                    "primary_failure_preserved": True})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if core is not None:
            core.exit()
        if store is not None:
            asyncio.run(store.stop())
        write(directory / "exit.json", {"success": failure is None, "started_at": started,
              "ended_at": time.time(), "accepted": False})
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
