"""Load the pinned ordinary AIPerf profile and verify its exported dependency."""
import hashlib
import importlib.util
from pathlib import Path

from atom.compass.core.proper_replay import checked_path, profile_identity, read_pinned


def verify_dependency(plan):
    """Check the actual imported optional dependency before any profile starts."""
    import aiperf
    from aiperf.common.models import RawRecordInfo

    dependency = plan["aiperf_dependency"]
    manifest = read_pinned(dependency["manifest"])
    if (manifest.get("schema") != "compass.aiperf_raw_transport_timestamps/1"
            or manifest.get("base_commit") != "0d2aa0572ac685943d38c580675c4a61023581d3"
            or not {"end_perf_ns", "recv_start_perf_ns"} <= RawRecordInfo.model_fields.keys()):
        raise ValueError("proper replay needs the pinned raw transport timestamp dependency")
    root = Path(dependency["checkout"]).resolve()
    if Path(aiperf.__file__).resolve() != root / "src/aiperf/__init__.py":
        raise ValueError("proper replay imported a different AIPerf checkout")
    for item in manifest["files"]:
        checked_path({"path": str(root / item["path"]), "sha256": item["sha256"]})
    for item in dependency.get("source_files", []):
        checked_path({"path": str(root / item["path"]), "sha256": item["sha256"]})
    return manifest


def load_profile(plan):
    """Recreate validator input intent with the frozen original constructor."""
    from aiperf.common.models import Conversation, DatasetMetadata

    prepared = read_pinned(plan["prepared"])
    identity = profile_identity(prepared)
    source = plan.get("source", prepared["source"])
    checked_path(source)
    if source["sha256"] != prepared["source"]["sha256"]:
        raise ValueError("proper replay source alias changes the actual source bytes")
    builder_path = Path(plan["config_builder"]["path"])
    if hashlib.sha256(builder_path.read_bytes()).hexdigest() != plan["config_builder"]["sha256"]:
        raise ValueError("proper config constructor changed")
    spec = importlib.util.spec_from_file_location("proper_frozen_config", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Only a path alias may differ: the bytes above have already been checked.
    builder.SOURCE = Path(source["path"])
    config = builder.config()
    config.benchmark_id = prepared["config"]["benchmark_id"]
    current, expected = config.model_dump(mode="json"), dict(prepared["config"])
    for value in (current, expected):
        value.pop("cli_command", None)
    expected["input"] = dict(expected["input"])
    expected["input"]["file"] = current["input"]["file"]
    if current != expected:
        raise ValueError("proper AIPerf semantic configuration differs from the prepared profile")
    conversations = [Conversation.model_validate(row) for row in prepared["conversations"]]
    metadata = DatasetMetadata.model_validate(prepared["metadata"])
    caps = {(c.session_id, i): turn.max_tokens
            for c in conversations for i, turn in enumerate(c.turns)}
    return config, conversations, metadata, identity, caps



def compare_native_metadata(frozen, observed, conversations):
    """Retain two inactive raw differences; require equality of all active fields."""
    expected = frozen.model_dump(mode="json")
    actual = dict(observed)
    if any(str(c.context_mode) != "deltas_with_responses" for c in conversations):
        raise ValueError("native metadata comparison requires explicit provided history")
    inactive = {}
    for key, values in (("has_timing_data", (True, False)),
                        ("default_context_mode", (None, "deltas_with_responses"))):
        if expected.get(key) != actual.get(key):
            if (expected.get(key), actual.get(key)) != values:
                raise ValueError(f"unexpected native metadata difference: {key}")
            inactive[key] = {"frozen": expected[key], "ordinary_native": actual[key]}
        expected.pop(key, None)
        actual.pop(key, None)
    if expected != actual:
        raise ValueError("ordinary AIPerf dataset metadata changes active profile semantics")
    return {"effective_equal": True, "inactive_raw_differences": inactive,
            "frozen": frozen.model_dump(mode="json"), "ordinary_native": observed,
            "basis": {"has_timing_data": "no consumers in pinned dependency",
                      "default_context_mode": "explicit Conversation.context_mode takes precedence"}}
