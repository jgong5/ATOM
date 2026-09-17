"""Load the pinned ordinary AIPerf profile and verify its exported dependency."""
import hashlib
import importlib.util
import inspect
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
    # Construct with the requested root-client count before dependency validators
    # derive coupled settings. Old pinned C1 constructors remain usable unchanged.
    if "clients" in inspect.signature(builder.config).parameters:
        config = builder.config(clients=identity["clients"])
    else:
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

def _check_source_options(plan, options):
    """Diagnostic execution may retain failed precision; missing work stays fatal."""
    from atom.compass.runtime.source_oracle import _flag

    if not _flag(options.get("require_complete", True), "require_complete"):
        raise ValueError("proper replay requires complete cost coverage; unpriced work cannot be omitted")
    for key in ("diagnostic_only", "include_failed_outputless", "include_failed_final", "low_q_allow_failed_spread",
                "root_prefill_allow_failed_spread"):
        if _flag(options.get(key, False), key) and plan.get("purpose") != "diagnostic":
            raise ValueError(f"proper replay refuses unqualified source opt-in: {key}")
    if options.get("root_prefill_diagnostic_handoff"):
        raise ValueError("proper replay cannot select fixed-workload diagnostic sources")
    composition = options.get("composition_qualification")
    if bool(composition) != bool(options.get("composition_qualification_sha256")):
        raise ValueError("proper replay needs composition qualification and its SHA-256 together")
    if composition:
        from atom.compass.core.cost.composition_qualification import SCHEMA
        from atom.compass.core.loaded_input import load_json

        receipt, identity = load_json(composition, role="validation.forward_composition.preflight")
        if (identity.sha256 != options["composition_qualification_sha256"]
                or receipt.get("schema") != SCHEMA or receipt.get("passed") is not True):
            raise ValueError("proper replay composition qualification is missing, changed or failed")
        # The source factory and final source-contract reader independently
        # recompute this receipt against actual loaded books, code and heldouts.
    if any(options.get(key) for key in ("diagnostic_reference_handoff", "exact_operator_handoff")) and not composition and (
            plan.get("purpose") != "diagnostic"
            or not _flag(options.get("diagnostic_only", False), "diagnostic_only")):
        raise ValueError("proper reference sources require composition qualification or explicit diagnostic execution")


def create_modelled_config(plan, tokenizer, output_directory):
    """Use the normal EngineArgs/Config path; no release calendar is installed."""
    from dataclasses import fields
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    import argparse

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    argv = list(plan["modelled_engine_args"])
    if any("fixed-absolute" in x or "opening-plan" in x for x in argv):
        raise ValueError("proper replay cannot install a fixed request calendar")
    args = EngineArgs.from_cli_args(parser.parse_args(argv))
    kwargs = args._get_engine_kwargs()
    config = Config(args.model, **{k: v for k, v in kwargs.items() if k in {f.name for f in fields(Config)}})
    compass = config.compass_config
    if config.tensor_parallel_size != 1 or config.pipeline_parallel_size != 1:
        raise ValueError("proper paired harness currently supports TP1/PP1")
    if config.enable_prefix_caching is not True:
        raise ValueError("proper paired harness requires prefix caching enabled")
    if compass.oracle_qualname not in (
            "atom.compass.runtime.source_oracle.source_cost_oracle",
            "atom.compass.runtime.cache_region_oracle.source_cost_oracle"):
        raise ValueError("proper replay requires the maintained source oracle factory")
    _check_source_options(plan, compass.oracle_options)
    if (compass.oracle_options.get("compiled_prefill_execution_handoff")
            and not compass.prefill_preparation_fence):
        raise ValueError("compiled-prefill execution requires the source-proven prefill preparation fence")
    compass.epoch = 100.0
    compass.measure_out = str(Path(output_directory).parent / (Path(output_directory).stem + "_steps.jsonl"))
    compass.filler_token_id = plan["surrogate_output"]["token_id"]
    if not compass.enabled or compass.mode != "predict" or not compass.virtual_clock:
        raise ValueError("proper modelled session requires prediction on its virtual clock")
    if compass.fixed_absolute_plan or compass.opening_plan:
        raise ValueError("proper replay has a release calendar")
    for field, option in (("replay_target", "replay_target"), ("memory_model", "memory_model"),
                          ("readiness_profile", "request_readiness_profile")):
        checked_path(plan[field])
        if Path(getattr(compass, option)).resolve() != Path(plan[field]["path"]).resolve():
            raise ValueError(f"modelled {field} differs from its pinned input")
    config.bos_token_id, config.eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    return config


def controlled_provenance(core, tokenizer, server_options):
    """Read existing utility handlers and API provenance without a second server."""
    import asyncio
    import queue
    from atom.compass.replay.aiperf_runner import _serve_with
    from atom.entrypoints.openai import api_server
    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.llm_engine import LLMEngine

    class EvidenceEngine:
        def __init__(self):
            self.config = core.scheduler.config
            self.core_mgr = self

        get_compass_inputs = LLMEngine.get_compass_inputs
        get_compass_cache = LLMEngine.get_compass_cache

        def broadcast_utility_command_sync(self, command, **kwargs):
            if command not in ("get_compass_inputs", "get_compass_cache"):
                raise ValueError("controlled evidence adapter is read-only")
            output = queue.Queue()
            handler = EngineUtilityHandler(core.runner_mgr, output, scheduler=core.scheduler, engine=core)
            getattr(handler, handler._UTILITY_HANDLERS[command])({})
            kind, reply = output.get_nowait()
            if kind != "UTILITY_RESPONSE" or reply["cmd"] != command or not output.empty():
                raise ValueError("controlled provenance utility reply differs")
            return [reply]

    engine = EvidenceEngine()
    with _serve_with(engine, tokenizer, core.scheduler.config.model, server_options):
        return asyncio.run(api_server.compass_provenance())
