"""Identity and phase accounting for ordinary/controlled AIPerf comparisons.

A duration-driven agentic profile does not prescribe a request calendar.
Completion-driven recycling and cancellation counts remain observed outcomes.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


def checked_path(pin):
    path = Path(pin["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError(f"proper replay input changed: {path}")
    return path


def read_pinned(pin):
    return json.loads(checked_path(pin).read_bytes())


def content_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def profile_identity(prepared):
    config = prepared["config"]
    conversations = prepared["conversations"]
    # Root clients may spawn sub-agents; this is not an in-flight request cap.
    clients = config["loadgen"]["concurrency"]
    if (config.get("scenario") != "inferencex-agentx-mvp"
            or type(clients) is not int or clients not in (1, 2, 4, 8)
            or config["loadgen"]["benchmark_duration"] != 900
            or config["input"]["random_seed"] != 42
            or config["endpoint"]["type"] != "chat"
            or config["endpoint"]["streaming"] is not True
            or config["endpoint"]["use_server_token_count"] is not True
            or not conversations
            or any(c["context_mode"] != "deltas_with_responses" for c in conversations)):
        raise ValueError("proper replay requires a C1/C2/C4/C8, 900s/seed42 provided-history profile")
    # The benchmark identifier is an input: ordinary cache-bust markers hash it.
    benchmark_id = config["benchmark_id"]
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError("proper replay requires its cache-bust benchmark identifier")
    return {
        "scenario": config["scenario"], "clients": clients, "profile_seconds": 900,
        "seed": 42, "benchmark_id": benchmark_id,
        "context_mode": "deltas_with_responses",
        "source_sha256": prepared["source"]["sha256"],
        "dataset_sha256": content_sha256(conversations),
        "dataset_metadata_sha256": content_sha256(prepared["metadata"]),
        "config_sha256": content_sha256(config),
        "request_calendar": None,
    }


def phase_accounting(messages, *, duration_seconds=900):
    starts, ends = {}, {}
    for message in messages:
        kind = message.get("message_type")
        if kind == "command" and message.get("command") == "profile_cancel":
            raise ValueError(f"AIPerf profile was cancelled: {message.get('reason')}")
        if kind not in ("credit_phase_start", "credit_phase_complete"):
            continue
        phase = message["stats"]["phase"]
        destination = starts if kind == "credit_phase_start" else ends
        if phase in destination:
            raise ValueError(f"duplicate AIPerf {kind}: {phase}")
        destination[phase] = message
    if set(starts) != {"warmup", "profiling"} or set(ends) != set(starts):
        raise ValueError("proper replay lacks both native phase starts/completions")
    warmup = ends["warmup"]["stats"]
    names = ("final_requests_sent", "final_requests_completed",
             "final_requests_cancelled", "final_request_errors")
    warmup_start = starts["warmup"]
    declared = warmup_start["stats"].get("total_expected_requests")
    warmup_counts = {key: warmup.get(key) for key in names}
    if (type(declared) is not int or declared < 0
            or warmup_start["config"].get("total_expected_requests", declared) != declared
            or warmup.get("total_expected_requests") != declared
            or any(type(value) is not int or value < 0 for value in warmup_counts.values())
            or warmup_counts["final_requests_sent"] != declared
            or warmup_counts["final_requests_completed"] != declared
            or warmup_counts["final_requests_cancelled"] != 0
            or warmup_counts["final_request_errors"] != 0 or warmup.get("was_cancelled")):
        raise ValueError("proper replay did not complete its declared snapshot-priming warmup")
    start, end = starts["profiling"], ends["profiling"]["stats"]
    if start["config"]["expected_duration_sec"] != duration_seconds:
        raise ValueError("AIPerf profiling duration changed")
    origin, finish = start["stats"]["start_ns"], end["requests_end_ns"]
    if (type(origin) is not int or type(finish) is not int or origin <= 0 or finish < origin
            or end["start_ns"] != origin or end.get("was_cancelled")):
        raise ValueError("AIPerf profiling origin/completion is invalid")
    counts = {key: end[key] for key in names}
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("AIPerf phase counts must be nonnegative integers")
    if counts["final_requests_sent"] != counts["final_requests_completed"] + counts["final_requests_cancelled"]:
        raise ValueError("AIPerf issued credits do not close as completed or cancelled")
    warmup_origin, warmup_finish = warmup_start["stats"]["start_ns"], warmup["requests_end_ns"]
    if (type(warmup_origin) is not int or type(warmup_finish) is not int
            or not warmup_origin <= warmup_finish <= origin
            or warmup["start_ns"] != warmup_origin):
        raise ValueError("AIPerf warmup origin/completion is invalid")
    return {
        "origin_ns": origin, "completed_ns": finish,
        "requested_duration_seconds": duration_seconds,
        "observed_duration_seconds": (finish - origin) / 1e9,
        "counts": counts, "warmup_counts": warmup_counts,
        "warmup": {"origin_ns": warmup_origin, "completed_ns": warmup_finish,
                   "observed_duration_seconds": (warmup_finish - warmup_origin) / 1e9,
                   "declared_requests": declared, "counts": warmup_counts,
                   "kind": "agentic_snapshot_priming" if declared else "empty"},
        "grace_period_timeout_triggered": end.get("grace_period_timeout_triggered", False),
        "branch_stats": ends["profiling"].get("branch_stats"),
        "count_basis": "actual phase counters; no historical request-count target",
    }


def record_key(row):
    """Cross-side content identity includes the ordinary, unmodified marker."""
    return (row["cache_bust_marker"], row["conversation_id"], row["turn_index"])


def validate_records(rows, phase, *, benchmark_phase="profiling"):
    if benchmark_phase not in ("profiling", "warmup"):
        raise ValueError("unsupported proper record phase")
    ids, keys = set(), set()
    for row in rows:
        if row.get("benchmark_phase", benchmark_phase) != benchmark_phase:
            raise ValueError("normalized request belongs to a different benchmark phase")
        if row["request_id"] in ids:
            raise ValueError("duplicate actual request identity")
        ids.add(row["request_id"])
        if not isinstance(row.get("cache_bust_marker"), str) or not row["cache_bust_marker"]:
            raise ValueError("proper replay requires the actual ordinary cache-bust marker")
        key = record_key(row)
        if key in keys:
            raise ValueError("duplicate marker/conversation/turn identity")
        keys.add(key)
        start, end = row["start_ns"], row["end_ns"]
        if (type(start) is not int or type(end) is not int
                or not phase["origin_ns"] <= start <= end):
            raise ValueError("request timestamps are outside the profiling timeline")
        first = row.get("first_visible_ns")
        if first is not None and (type(first) is not int or not start <= first <= end):
            raise ValueError("visible first token is outside the actual request interval")
        if not row.get("payload_sha256") or not row.get("prompt_token_sha256"):
            raise ValueError("request lacks marked payload and consumed-token identity")
        if type(row["output_tokens"]) is not int or row["output_tokens"] < 0:
            raise ValueError("request output-token count is invalid")
        if (benchmark_phase == "profiling" and not row.get("cancelled")
                and not row.get("error") and row["output_tokens"] and first is None):
            raise ValueError("successful output has no visible first-token event")
    successful_or_error = sum(not r.get("cancelled") for r in rows)
    if successful_or_error != phase["counts"]["final_requests_completed"]:
        raise ValueError("exported records do not close the actual completed-credit count")
    if sum(bool(r.get("cancelled")) for r in rows) > phase["counts"]["final_requests_cancelled"]:
        raise ValueError("exported cancellation count exceeds native cancelled credits")


def pairing_observations(real, modelled):
    """Expose dynamic counts and marker/token differences without rewriting input."""
    by_side = {side: {record_key(row): row for row in rows}
               for side, rows in (("real", real), ("modelled", modelled))}
    shared = by_side["real"].keys() & by_side["modelled"].keys()
    different = []
    for key in sorted(shared):
        a, b = by_side["real"][key], by_side["modelled"][key]
        fields = [field for field in ("payload_sha256", "prompt_token_sha256", "input_tokens")
                  if a[field] != b[field]]
        if fields:
            different.append({"key": list(key), "fields": fields,
                              "real_input_tokens": a["input_tokens"],
                              "modelled_input_tokens": b["input_tokens"]})
    return {
        "real_records": len(real), "modelled_records": len(modelled),
        "shared_marker_conversation_turns": len(shared),
        "real_only": len(by_side["real"].keys() - shared),
        "modelled_only": len(by_side["modelled"].keys() - shared),
        "marked_input_differences": different,
        "real_source_turn_counts": dict(Counter(
            f"{r['conversation_id']}:{r['turn_index']}" for r in real)),
        "modelled_source_turn_counts": dict(Counter(
            f"{r['conversation_id']}:{r['turn_index']}" for r in modelled)),
        "identical_completion_driven_timestamps_required": False,
        "identical_dynamic_request_counts_required": False,
        "markers_stripped": False,
    }


def profiling_wall_window(events):
    """The passively observed profiling phase on the physical wall clock."""
    import math
    edges = {}
    for row in events:
        if (row.get("stats") or {}).get("phase") != "profiling":
            continue
        kind = row.get("message_type")
        if kind not in ("credit_phase_start", "credit_phase_complete"):
            continue
        if kind in edges:
            raise ValueError("duplicate physical profiling-phase observation")
        value = row.get("wall_observed_at")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("profiling phase lacks a finite physical wall observation")
        edges[kind] = value
    if set(edges) != {"credit_phase_start", "credit_phase_complete"}:
        raise ValueError("profiling phase lacks both physical wall boundaries")
    start, end = edges["credit_phase_start"], edges["credit_phase_complete"]
    if end <= start:
        raise ValueError("physical profiling wall interval is not positive")
    return {"schema": "compass.replay_wall_window/1",
            "started_at": start, "ended_at": end, "seconds": end-start, "clock": "wall",
            "scope": "passive phase start through phase complete; excludes setup/export and final drain"}
