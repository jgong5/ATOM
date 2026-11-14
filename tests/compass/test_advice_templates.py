# SPDX-License-Identifier: MIT
"""The committed DeepSeek-V4-Pro development template, expanded through
`compass advice`, each instance's arguments read back by ATOM's own parser."""

import json
from pathlib import Path

import pytest

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import cli
from atom.compass.run.step_grid import READ_OFF_DEFAULT, keyed_env
from atom.model_engine.arg_utils import EngineArgs, parse_size_list
from atom.utils.arg_parser import FlexibleArgumentParser

ROOT = Path(__file__).resolve().parents[2]
SPACE = ROOT / "scripts/compass/spaces/deepseek-v4-pro-mi355x-crs005.json"
#: The shipped DPA8 pair whose capture ladders the 1P1D configurations take.
LAW = "scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005-{}.json"
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256]
#: Each instance's ATOM arguments the template states, by the name ATOM's
#: parser reads each under, but `--model` and the capture sizes, and its
#: environment.
COMMON = {
    "tensor_parallel_size": 8,
    "kv_cache_dtype": "fp8",
    "enable_prefix_caching": True,
    "max_num_seqs": 512,
}
ONE_NODE = {
    **COMMON,
    "index_cache_dtype": "fp4",
    "method": "mtp",
    "num_speculative_tokens": 3,
    "spec_decode_acceptance_length": 2.49,
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 16384,
    "attn_prefill_chunk_size": 16384,
    "state_checkpoint_interval_tokens": 8192,
    "level": 3,
    "cudagraph_mode": "FULL",
}
DPA = {"enable_dp_attention": True, "enable_tbo": "prefill"}
MESH = {**COMMON, "trust_remote_code": True}
MOE = {"AITER_BF16_FP8_MOE_BOUND": "0", "ATOM_MOE_GU_ITLV": "1"}
TBO = {**MOE, "GPU_MAX_HW_QUEUES": "5", "ATOM_NUMA_BIND": "1"}
DP = {
    **TBO,
    "ATOM_DP_SESSION_AFFINITY": "1",
    "ATOM_DP_LB_REQ_EQUIV": "512",
    "ATOM_ENABLE_PREFILL_DELAYER": "1",
    "ATOM_PREFILL_DECODE_INTERVAL": "10",
}
LMCACHE = {
    "ATOM_KV_OFFLOAD": "lmcache_mp",
    "ATOM_KV_OFFLOAD_EXTRA_CONFIG": '{"lmcache.mp.host": "tcp://127.0.0.1", '
    '"lmcache.mp.port": 5555}',
    "LMCACHE_CHUNK_SIZE": "256",
    "OFFLOAD_MAX_PENDING_SAVES": "8",
}
INSTANCES = {
    "S1": {"server": (ONE_NODE, MOE)},
    "S2": {"server": ({**ONE_NODE, **DPA}, DP)},
    "S5": {"server": ({**ONE_NODE, **DPA}, {**DP, **LMCACHE})},
    "P1": {
        "prefill": ({**MESH, "gpu_memory_utilization": 0.75}, MOE),
        "decode": ({**MESH, "gpu_memory_utilization": 0.75}, MOE),
    },
    "P3": {
        "prefill": ({**MESH, **DPA, "gpu_memory_utilization": 0.72}, TBO),
        "decode": (
            {**MESH, "enable_dp_attention": True, "gpu_memory_utilization": 0.75},
            MOE,
        ),
    },
}


@pytest.fixture
def expansion(tmp_path):
    cli.main(["advice", str(SPACE), str(tmp_path)])
    return json.loads((tmp_path / "candidates.json").read_text())


def parsed(args: list[str]) -> dict:
    """`args` as ATOM's engine parser reads them, refusing any it does not."""
    return vars(EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args(args))


def law(side: str) -> dict:
    """The engine config the shipped pair's `side` law is keyed under."""
    return json.loads((ROOT / LAW.format(side)).read_text())["engine_config"]


def test_the_template_expands_into_its_five_configurations_at_1_to_256_clients(
    expansion,
):
    candidates = expansion["candidates"]
    assert [c["name"] for c in candidates] == list(INSTANCES)
    assert [c["deployment"] for c in candidates] == ["single"] * 3 + ["1p1d"] * 2
    assert [(p["candidate"], p["concurrency"]) for p in expansion["points"]] == [
        (c, n) for c in INSTANCES for n in CONCURRENCY
    ]
    for candidate in candidates:
        assert (ROOT / candidate["basis"].split(",")[0]).is_file()


def test_atom_reads_each_instance_as_the_template_states_it(expansion):
    atom_ladder = parse_size_list(parsed([])["cudagraph_capture_sizes"])
    for point in expansion["points"]:
        for side, instance in point["instances"].items():
            args, env = INSTANCES[point["candidate"]][side]
            given = parsed(instance["args"])
            sizes = parse_size_list(given["cudagraph_capture_sizes"])
            written = {
                w[2:].replace("-", "_") for w in instance["args"] if w[:2] == "--"
            }

            # Each argument the template states is written, ATOM's default or not.
            assert written - {"model", "cudagraph_capture_sizes"} == args.keys()
            assert {name: given[name] for name in args} == args
            assert given["model"] == expansion["model"]
            # ATOM's own on a single node, the shipped pair's on each 1P1D side.
            assert sizes == (atom_ladder if side == "server" else law(side)["ladder"])
            assert instance["env"] == env


def test_p3_keys_its_environment_as_the_shipped_crs005_pair(expansion, monkeypatch):
    p3 = next(c for c in expansion["candidates"] if c["name"] == "P3")
    for side, instance in p3["instances"].items():
        with monkeypatch.context() as m:
            for name in (*PERFORMANCE_KEYS, *READ_OFF_DEFAULT):
                m.delenv(name, raising=False)
            # deploy.sh sets it before it starts either side.
            m.setenv("AITER_LOG_LEVEL", "WARNING")
            for name, value in instance["env"].items():
                m.setenv(name, value)

            assert keyed_env() == law(side)["env"]


def test_compass_advice_reads_no_space_it_is_not_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # OUT alone: refused for want of SPACE, never read as the template.
    with pytest.raises(SystemExit) as refused:
        cli.main(["advice", str(tmp_path / "out")])

    assert refused.value.code == 2
    assert not any(tmp_path.iterdir())
