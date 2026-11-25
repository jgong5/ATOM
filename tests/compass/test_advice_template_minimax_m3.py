# SPDX-License-Identifier: MIT
"""The committed MiniMax-M3 template, expanded through `compass advice`, each
point's arguments read back by ATOM's own parser."""

import json
from pathlib import Path

import pytest

from atom.compass import cli
from atom.compass.drive.measure_memory import FRONTEND
from atom.compass.drive.runfile import pd_side
from atom.model_engine.arg_utils import EngineArgs, parse_size_list
from atom.utils.arg_parser import FlexibleArgumentParser

ROOT = Path(__file__).resolve().parents[2]
SPACE = ROOT / "scripts/compass/spaces/minimax-m3-mi355x-crs005.json"
#: Every concurrency the submission ran, whichever arm ran it.
CONCURRENCY = [1, 2, 4, 5, 8, 10, 12, 15, 20, 24, 25, 28, 30, 40, 48]
CHECKPOINTS = Path("/shared_nfs/huggingface_models")
#: What ATOM reads every arm's arguments as, by the name its parser reads each
#: under, decoded as `DECODED` decodes it; prefix caching, the served name and
#: no KV connector are ATOM's defaults.
COMMON = {
    "model": str(CHECKPOINTS / "amd/MiniMax-M3-MXFP4"),
    "trust_remote_code": True,
    "kv_cache_dtype": "fp8",
    "block_size": 128,
    "max_num_batched_tokens": 32768,
    "gpu_memory_utilization": 0.95,
    "cudagraph_capture_sizes": [*range(1, 19), *range(20, 37, 2), 40, 48, 56, 64],
    "index_cache_dtype": "fp8",
    "online_quant_config": {
        "global_quant_config": "ptpc_fp8",
        "exclude_layer": [
            "lm_head",
            "model.embed_tokens",
            "vision_tower",
            "multi_modal_projector",
            "patch_merge_mlp",
            "*block_sparse_moe",
        ],
    },
    "default_chat_template_kwargs": {"thinking_mode": "enabled"},
    "method": "eagle3",
    "draft_model": str(CHECKPOINTS / "Inferact/MiniMax-M3-EAGLE3-GQA"),
    "num_speculative_tokens": 3,
    "spec_decode_acceptance_length": 2.78,
    "enable_prefix_caching": True,
    "served_model_name": None,
    "kv_transfer_config": {},
}
#: The arguments ATOM's parser keeps as strings, and what decodes each.
DECODED = {
    "cudagraph_capture_sizes": parse_size_list,
    "default_chat_template_kwargs": json.loads,
    "kv_transfer_config": json.loads,
}
ENV = {
    "PYTHONNOUSERSITE": "1",
    "AITER_LOG_LEVEL": "WARNING",
    "AITER_SITUV2_A4W4": "1",
    "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
    "AITER_FLYDSL_STAGE2_FP8": "1",
    "ATOM_FORCE_ATTN_TRITON": "1",
    "ATOM_PA_FLYDSL": "1",
    "ATOM_PA_FLYDSL_PLAN": "1",
}
CP = {"ATOM_M3_INDEXER_CP": "1"}
OFFLOAD = {
    "kv_transfer_config": {"kv_connector": "lmcache_offload", "kv_role": "offload"}
}
LMCACHE = {
    "PYTHONHASHSEED": "0",
    "LMCACHE_LOCAL_CPU": "True",
    "LMCACHE_MAX_LOCAL_CPU_SIZE": "257",
    "LMCACHE_CHUNK_SIZE": "256",
    "LMCACHE_CACHE_POLICY": "ATOM_SLRU",
    "ATOM_PREFIX_CACHE_POLICY": "slru",
    "ATOM_PREFIX_CACHE_PROTECTED_RATIO": "0.5",
}
#: Each arm the submission ran, as it names it: its arguments and environment.
ARMS = {
    "TP4": ({"tensor_parallel_size": 4}, {}),
    "TP4_indexer_CP": ({"tensor_parallel_size": 4}, CP),
    "TP2": ({"tensor_parallel_size": 2}, {}),
    "TP2_LMCache": (
        {"tensor_parallel_size": 2, **OFFLOAD},
        {**LMCACHE, "LMCACHE_LOOKUP_SERVER_WORKER_IDS": "0,1"},
    ),
    "TP4_indexer_CP_LMCache": (
        {"tensor_parallel_size": 4, **OFFLOAD},
        {**CP, **LMCACHE, "LMCACHE_LOOKUP_SERVER_WORKER_IDS": "0,1,2,3"},
    ),
}


@pytest.fixture(scope="module")
def expansion(tmp_path_factory):
    out = tmp_path_factory.mktemp("advice")
    cli.main(["advice", str(SPACE), str(out)])
    return json.loads((out / "candidates.json").read_text())


def parsed(args: list[str]) -> dict:
    """`args` as ATOM's server parser reads them, refusing any it does not."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    for flag, kind in FRONTEND.items():
        parser.add_argument(flag, **kind)
    return vars(parser.parse_args(args))


def test_the_template_runs_every_arm_at_every_concurrency_at_one_frozen_commit(
    expansion,
):
    assert [c["name"] for c in expansion["candidates"]] == list(ARMS)
    assert {c["deployment"] for c in expansion["candidates"]} == {"single"}
    assert [(p["candidate"], p["concurrency"]) for p in expansion["points"]] == [
        (arm, n) for arm in ARMS for n in CONCURRENCY
    ]
    assert expansion["commit"] == "ac272852e1e6eae126849b1f1dac2bf0372714e8"
    assert expansion["workload"]["select"] == "agentx"


def test_atom_reads_each_point_as_the_submission_serves_it(expansion):
    for point in expansion["points"]:
        (instance,) = point["instances"].values()
        args, env = ARMS[point["candidate"]]
        stated = {**COMMON, **args, "max_num_seqs": 2 * point["concurrency"]}
        given = parsed(instance["args"])
        written = {w[2:].replace("-", "_") for w in instance["args"] if w[:2] == "--"}

        read = {n: DECODED[n](given[n]) if n in DECODED else given[n] for n in stated}

        assert read == stated
        # Nothing is written that the submission does not state.
        assert written <= stated.keys()
        assert instance["env"] == {**ENV, **env}
        # One server on one node: no prefill or decode side.
        assert pd_side(instance["args"]) is None
