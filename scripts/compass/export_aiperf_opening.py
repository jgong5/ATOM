"""Export a branch-free first two turns with the pinned published Weka loader."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.prefix_workload import token_digest, tokenizer_identity


AIPERF_COMMIT = "0d2aa0572ac685943d38c580675c4a61023581d3"


def export_opening(source_root, source_sha256, aiperf_source, model, seed):
    from aiperf.common import random_generator as rng
    from aiperf.common.config import (
        EndpointConfig, InputConfig, LoadGeneratorConfig, TokenizerConfig, UserConfig,
    )
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.generator.coding_content import CodingContentGenerator
    from aiperf.dataset.loader.weka_synth_buf import compute_asst_block_caps
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader
    from aiperf.dataset.loader.weka_trace_models import WekaTrace
    import inspect
    import tokenizers
    import transformers

    repo = Path(aiperf_source).resolve()
    git = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != AIPERF_COMMIT:
        raise ValueError("opening exporter requires the audited AIPerf commit")
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no", "--", "src/aiperf"], text=True):
        raise ValueError("AIPerf Python source has tracked modifications")
    loader_file = Path(inspect.getfile(WekaTraceLoader)).resolve()
    if not loader_file.is_relative_to(repo):
        raise ValueError("imported AIPerf loader differs from the pinned checkout")

    raw = Path(source_root).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError("source root digest changed")
    data = json.loads(raw)
    if data.get("hash_id_scope") != "local" or data.get("block_size") != 64:
        raise ValueError("opening requires the declared local 64-token source format")
    source_turns = data["requests"][:2]
    if len(source_turns) != 2 or any(row.get("type") not in ("n", "s") for row in source_turns):
        raise ValueError("the first two source entries must be ordinary requests")
    config = UserConfig(
        endpoint=EndpointConfig(model_names=[model], streaming=True, use_server_token_count=True),
        input=InputConfig(file=str(source_root), custom_dataset_type="weka_trace",
                          fixed_schedule=True, fixed_schedule_auto_offset=True, random_seed=seed),
        tokenizer=TokenizerConfig(name=model),
        loadgen=LoadGeneratorConfig(concurrency=1, request_count=2),
    )
    # Config construction can precede CLI seed selection. Match bootstrap.py.
    rng.reset()
    rng.init(config.input.random_seed)
    tokenizer = Tokenizer.from_pretrained(model, resolve_alias=False)
    generator = CodingContentGenerator(config=config.input.prompt, tokenizer=tokenizer)
    loader = WekaTraceLoader(filename=str(source_root), user_config=config,
                             prompt_generator=generator)
    trace = WekaTrace.model_validate(data)
    full_plan = loader._build_reconstruction_plans({data["id"]: [trace]})
    parent = full_plan.parent_plans[0]
    if [index for index, _ in parent.normals[:2]] != [0, 1]:
        raise ValueError("source opening is not the first two inferred root-chain turns")
    through = source_turns[1]["t"] + source_turns[1].get("api_time", 0)
    child_requests = [request for plan in full_plan.child_plans for request in plan.requests]
    child_requests += [request for plan in full_plan.flat_plans for _, request in plan.requests]
    if any(request.t <= through for request in child_requests):
        raise ValueError("source opening includes a child; this adapter has no branches")
    caps = compute_asst_block_caps(
        [(request.hash_ids, request.input_length) for _, request in parent.normals], 64)
    if caps[:2] != [None, None]:
        raise ValueError("future role-planning constraints change this opening")
    opening = copy.deepcopy(data)
    opening["requests"] = opening["requests"][:2]
    conversations = loader.convert_to_conversations({data["id"]: [WekaTrace.model_validate(opening)]})
    if len(conversations) != 1 or conversations[0].branches:
        raise ValueError("exported opening is not a single branch-free conversation")
    messages, rows = [], []
    origin = float(source_turns[0]["t"])
    for index, turn in enumerate(conversations[0].turns):
        if turn.reset_context:
            messages = []
        messages.extend(turn.raw_messages)
        body = {"model": model, "messages": copy.deepcopy(messages), "stream": True,
                "stream_options": {"include_usage": True}, "temperature": 0.0,
                "ignore_eos": True, "max_completion_tokens": turn.max_tokens,
                "chat_template_kwargs": {}}
        # Match the server's Jinja string render followed by IOProcessor.encode.
        rendered = tokenizer._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer._tokenizer.encode(rendered)
        rows.append({"index": index, "source_path": f"/requests/{index}",
                     "arrival_s": float(source_turns[index]["t"]) - origin,
                     "depends_on": [] if index == 0 else [0],
                     "source_input_tokens": source_turns[index]["in"],
                     "input_tokens": len(tokens), "output_tokens": turn.max_tokens,
                     "prompt_token_ids": tokens, "prompt_token_sha256": token_digest(tokens),
                     "body": body})
    return {
        "schema": "compass.aiperf_opening/1", "profile": "aiperf_fixed_opening_ignore_eos_v1",
        "clients": 1, "branches": [], "initial_cache": "acknowledged_empty", "time_scale": 1,
        "response_delivery": {"kind": "assumed_zero", "seconds": 0.0,
                              "from": "native_engine_finish", "to": "client_response_available"},
        "cache_policy": cache_on_policy(),
        "tokenizer": tokenizer_identity(tokenizer._tokenizer, model),
        "chat_template_sha256": hashlib.sha256(tokenizer._tokenizer.chat_template.encode()).hexdigest(),
        "producer": {"aiperf_commit": revision, "seed": seed,
                     "bootstrap_rng_reset": True,
                     "loader_sha256": hashlib.sha256(loader_file.read_bytes()).hexdigest(),
                     "coding_pool_tokens_sha256": token_digest(generator._tool_pool),
                     "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "environment_pyproject_sha256": hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest(),
                     "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
        "source": {"root_id": data["id"], "path": str(source_root), "sha256": source_sha256,
                   "root_entries": len(data["requests"]), "selected_entries": [0, 1],
                   "full_plan_opening_roles_match": True, "later_branches_excluded": True},
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-root-sha256", required=True)
    parser.add_argument("--aiperf-source", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = export_opening(args.source_root, args.source_root_sha256,
                             args.aiperf_source, args.model, args.seed)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with Path(args.out).open("xb") as output:
        output.write(encoded)
    print(json.dumps({"path": args.out, "sha256": hashlib.sha256(encoded).hexdigest(),
                      "rows": [{key: row[key] for key in ("index", "input_tokens", "output_tokens",
                                                          "arrival_s", "prompt_token_sha256")}
                               for row in payload["requests"]]}))


if __name__ == "__main__":
    main()
