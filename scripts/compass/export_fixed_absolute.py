"""Export complete selected Weka roots for the corrected fixed-absolute profile."""

import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.fixed_absolute import (
    DEPENDENCY_BASIS, PROFILE, SCHEMA, FixedAbsolutePlan, source_leaves,
)
from atom.compass.prefix_workload import token_digest, tokenizer_identity
from atom.compass.replay_plan import RESPONSE_DELIVERY


AIPERF_COMMIT = "0d2aa0572ac685943d38c580675c4a61023581d3"


class UnsupportedExport(ValueError):
    def __init__(self, reason, metadata, missing_predecessors=()):
        super().__init__(reason)
        self.metadata = metadata
        self.missing_predecessors = list(missing_predecessors)


def compile_bundle(source_roots, source_data, conversations, tokenizer, producer):
    """Map loader source coordinates exactly; never infer identity from time."""
    metadata = {"conversations": [c.metadata().model_dump(mode="json") for c in conversations]}
    try:
        roots, originals, location = [], [], {}
        for client, (reference, data) in enumerate(zip(source_roots, source_data)):
            leaves = source_leaves(data)
            roots.append({"root_id": data["id"], "client_index": client, "source": reference,
                          "source_paths": list(leaves)})
            for path, original in leaves.items():
                location[data["id"], path] = len(originals)
                originals.append((data["id"], client, path, original))
        origin = min(row[3]["t"] for row in originals)
        coordinates, owner, convs = {}, {}, []
        for conversation in conversations:
            if conversation.agent_depth not in (0, 1) or (conversation.agent_depth and conversation.branches):
                raise ValueError("nested branch orchestration is outside the bounded profile")
            indices = []
            for turn_index, turn in enumerate(conversation.turns):
                if turn.source_inner_idx is not None or type(turn.source_outer_idx) is not int:
                    raise ValueError("loader turn lacks supported original source-leaf coordinates")
                key = (turn.source_trace_id, f"/requests/{turn.source_outer_idx}")
                if key not in location or key in owner:
                    raise ValueError("loader duplicated a source leaf or referenced another source")
                index = location[key]
                original = originals[index][3]
                if (turn.timestamp is None
                        or not math.isclose(turn.timestamp / 1000., original["t"], rel_tol=0, abs_tol=1e-9)
                        or turn.max_tokens != original["out"]):
                    raise ValueError("loader changed an original source timestamp or output maximum")
                owner[key] = conversation.session_id
                coordinates[conversation.session_id, turn_index] = index
                indices.append(index)
            if not indices:
                raise ValueError("empty conversations are outside the complete-root profile")
            root_ids = {originals[i][0] for i in indices}
            if len(root_ids) != 1:
                raise ValueError("loader conversation crosses original roots")
            root_id = next(iter(root_ids))
            if conversation.is_root != (conversation.agent_depth == 0):
                raise ValueError("loader root/client identity is inconsistent")
            convs.append({"conversation_id": conversation.session_id, "root_id": root_id,
                          "agent_depth": conversation.agent_depth,
                          "parent_conversation_id": conversation.parent_conversation_id,
                          "request_indices": indices})
        if set(owner) != set(location):
            raise ValueError("loader omitted source leaves; complete root refused intact")
        by_conversation = {c["conversation_id"]: c for c in convs}
        dependencies = [set() for _ in originals]
        for conversation in convs:
            for before, after in zip(conversation["request_indices"], conversation["request_indices"][1:]):
                dependencies[after].add(before)
        branches = []
        for conversation in conversations:
            declared = {b.branch_id: b for b in conversation.branches}
            for turn in conversation.turns:
                if set(turn.branch_ids) - set(declared):
                    raise ValueError("loader turn references an unknown branch")
                for prerequisite in turn.prerequisites:
                    p = prerequisite.model_dump(mode="json")
                    if (p["kind"] != "spawn_join" or p["branch_id"] not in declared
                            or any(p.get(key) is not None for key in ("timer_seconds", "barrier_id", "event_name"))):
                        raise ValueError("loader prerequisite is outside same-chain/SPAWN/SPAWN_JOIN")
            for original in conversation.branches:
                branch = original.model_dump(mode="json")
                triggers = [i for i, turn in enumerate(conversation.turns) if branch["branch_id"] in turn.branch_ids]
                gates = [(i, p) for i, turn in enumerate(conversation.turns) for p in turn.prerequisites
                         if p.branch_id == branch["branch_id"]]
                if len(triggers) != 1 or len(gates) > 1:
                    raise ValueError("branch requires exactly one spawn and at most one parent join")
                for _, p in gates:
                    if p.child_conversation_ids is not None and set(p.child_conversation_ids) != set(branch["child_conversation_ids"]):
                        raise ValueError("partial-child joins are outside the bounded profile")
                trigger = coordinates[conversation.session_id, triggers[0]]
                gate = coordinates[conversation.session_id, gates[0][0]] if gates else None
                for child_name in branch["child_conversation_ids"]:
                    child = by_conversation[child_name]["request_indices"]
                    dependencies[child[0]].add(trigger)
                    if gate is not None:
                        dependencies[gate].add(child[-1])
                branches.append({"branch_id": branch["branch_id"],
                    "parent_conversation_id": conversation.session_id, "after_request": trigger,
                    "join_before_request": gate, "child_conversation_ids": branch["child_conversation_ids"],
                    "mode": branch["mode"], "is_background": branch["is_background"],
                    "dispatch_timing": branch["dispatch_timing"]})
        rows = [None] * len(originals)
        for conversation in conversations:
            messages = []
            for turn_index, turn in enumerate(conversation.turns):
                index = coordinates[conversation.session_id, turn_index]
                root_id, client, path, original = originals[index]
                if turn.reset_context:
                    messages = []
                messages.extend(turn.raw_messages)
                body = {"model": producer["model"], "messages": copy.deepcopy(messages),
                        "stream": True, "stream_options": {"include_usage": True},
                        "temperature": 0., "ignore_eos": True, "max_completion_tokens": turn.max_tokens,
                        "chat_template_kwargs": {}}
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tokens = tokenizer.encode(rendered)
                predecessors = sorted({coordinates[p.conversation_id, p.turn_index] for p in turn.replay_predecessors})
                rows[index] = {"index": index, "root_id": root_id, "client_index": client,
                    "source_path": path, "source_time_s": original["t"], "source_input_tokens": original["in"],
                    "arrival_s": original["t"] - origin, "conversation_id": conversation.session_id,
                    "turn_index": turn_index, "depends_on": sorted(dependencies[index]),
                    "loader_replay_predecessors": predecessors, "loader_reset_context": turn.reset_context,
                    "input_tokens": len(tokens), "output_tokens": turn.max_tokens,
                    "prompt_token_ids": tokens, "prompt_token_sha256": token_digest(tokens), "body": body}
        payload = {"schema": SCHEMA, "profile": PROFILE, "dependency_basis": DEPENDENCY_BASIS,
            "clients": len(roots), "roots": roots, "conversations": convs, "branches": branches,
            "source_time_origin_s": origin, "time_scale": 1, "initial_cache": "acknowledged_empty",
            "cache_policy": cache_on_policy(), "response_delivery": RESPONSE_DELIVERY,
            "tokenizer": tokenizer_identity(tokenizer, producer["model"]),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "producer": producer, "loader_metadata": metadata, "requests": rows}
        FixedAbsolutePlan(payload, None)  # Full source-byte and dependency proof before publication.
        return payload
    except (ValueError, KeyError, TypeError) as exc:
        raise UnsupportedExport(str(exc), metadata, getattr(exc, "missing", ())) from exc


def export_bundle(source_roots, aiperf_source, model="Qwen/Qwen3.8-27B", seed=42):
    from aiperf.common import random_generator as rng
    from aiperf.common.config import EndpointConfig, InputConfig, LoadGeneratorConfig, TokenizerConfig, UserConfig
    from aiperf.common.environment import Environment
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.generator.coding_content import CodingContentGenerator
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    repo = Path(aiperf_source).resolve()
    git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    loader_file = Path(inspect.getfile(WekaTraceLoader)).resolve()
    if (revision != AIPERF_COMMIT or not loader_file.is_relative_to(repo)
            or subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no", "--", "src/aiperf"], text=True)):
        raise ValueError("export requires the unchanged pinned AIPerf Python source")
    defaults = {name: field.default for name, field in type(Environment.DATASET).model_fields.items() if name.startswith("WEKA_")}
    effective = {name: getattr(Environment.DATASET, name) for name in defaults}
    if effective != defaults:
        raise ValueError("export requires pinned Weka reconstruction defaults")
    source_data = []
    for reference in source_roots:
        raw = Path(reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("source-root bytes changed")
        data = json.loads(raw)
        source_leaves(data)
        source_data.append(data)
    if len(source_data) not in (1, 2, 4, 8) or len({d["id"] for d in source_data}) != len(source_data):
        raise ValueError("export needs C1/C2/C4/C8 distinct original roots")
    config = UserConfig(endpoint=EndpointConfig(model_names=[model], streaming=True, use_server_token_count=True),
        input=InputConfig(file=source_roots[0]["path"], custom_dataset_type="weka_trace", fixed_schedule=True,
                          fixed_schedule_auto_offset=True, random_seed=seed),
        tokenizer=TokenizerConfig(name=model),
        loadgen=LoadGeneratorConfig(concurrency=len(source_data), request_count=sum(len(d["requests"]) for d in source_data)))
    rng.reset()
    rng.init(seed)
    tokenizer = Tokenizer.from_pretrained(model, resolve_alias=False)
    generator = CodingContentGenerator(config=config.input.prompt, tokenizer=tokenizer)
    loader = WekaTraceLoader(filename=source_roots[0]["path"], user_config=config, prompt_generator=generator)
    conversations = loader.convert_to_conversations({d["id"]: [WekaTrace.model_validate(d)] for d in source_data})
    return compile_bundle(source_roots, source_data, conversations, tokenizer._tokenizer, {
        "aiperf_commit": revision, "model": model, "seed": seed, "bootstrap_rng_reset": True,
        "weka_reconstruction": {"defaults_verified": True, "effective": effective, "pinned_defaults": defaults},
        "loader_sha256": hashlib.sha256(loader_file.read_bytes()).hexdigest(),
        "coding_pool_tokens_sha256": token_digest(generator._tool_pool),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_pyproject_sha256": hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--source-root-sha256", action="append", required=True)
    parser.add_argument("--aiperf-source", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.source_root) != len(args.source_root_sha256):
        parser.error("each source root needs its own ordered SHA256")
    sources = [{"path": str(Path(path).resolve()), "sha256": sha}
               for path, sha in zip(args.source_root, args.source_root_sha256)]
    try:
        payload = export_bundle(sources, args.aiperf_source, args.model, args.seed)
    except ValueError as exc:
        refusal = {"schema": "compass.fixed_absolute_export_refusal/1", "source_roots": sources,
                   "reason": str(exc), "unsupported_predecessors": getattr(exc, "missing_predecessors", []),
                   "loader_metadata": getattr(exc, "metadata", {}), "root_pruned": False}
        with open(args.out + ".refusal.json", "x") as stream:
            json.dump(refusal, stream, indent=2)
        print(json.dumps({"refused": True, "reason": str(exc), "metadata": args.out + ".refusal.json"}))
        return 2
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with open(args.out, "xb") as stream:
        stream.write(encoded)
    print(json.dumps({"path": args.out, "sha256": hashlib.sha256(encoded).hexdigest(),
                      "roots": len(payload["roots"]), "requests": len(payload["requests"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
