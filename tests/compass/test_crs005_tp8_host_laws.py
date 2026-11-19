# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's TP8 host laws for crsuse2-m2m-v3-005/013, host terms only,
one per side of the 1P1D TP cell: each side's server, its config as ATOM
resolves it from the cell's arguments and environment, picks its own entry in
a run file that holds both and the DPA8 laws' (`run.tokenizer_term`'s key
check); with the DPA8 laws alone it refuses, naming its parallel layout. A DPA8
side still picks its own law among all four.

No device: aiter's arch probe answers a declared arch.
"""

import json
import re
from types import SimpleNamespace

import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_key_index_cache_dtype import TREE, served

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.backends.json_file import read_json
from atom.compass.run.step_grid import BUCKETS

LAWS = TREE / "scripts" / "compass" / "laws"
TP8 = "deepseek-v4-pro-tp8-mi355x-crs005-{}-host.json.xz"
DPA8 = "deepseek-v4-pro-dpa8-mi355x-crs005-{}.json.xz"
#: Both sides' ATOM arguments and environment in the TP cell: #975's DPA cell
#: less --enable-dp-attention, the prefill at --gpu-memory-utilization 0.65 and
#: the decode at 0.70.
COMMON = (
    "--trust-remote-code --kv_cache_dtype fp8 --block-size 16 "
    "--no-enable_prefix_caching -tp 8 --max-num-seqs 512"
)
CAPTURE = ",".join(map(str, [1, 2, *range(4, 257, 4)]))
ARGS = {
    "prefill": f"{COMMON} --gpu-memory-utilization 0.65 --enable-tbo",
    "decode": (
        f"{COMMON} --gpu-memory-utilization 0.70 --cudagraph-capture-sizes [{CAPTURE}]"
    ),
}
ROLE = {"prefill": "kv_producer", "decode": "kv_consumer"}
ENV = {
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "ATOM_MOE_GU_ITLV": "1",
    "AITER_LOG_LEVEL": "WARNING",
}
ENV_PREFILL = {"GPU_MAX_HW_QUEUES": "5", "ATOM_NUMA_BIND": "1"}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)


def law(name: str) -> dict:
    return read_json(LAWS / name)


def side_config(tmp_path, monkeypatch, side: str, *extra: str):
    """The config ATOM resolves for the cell's `side` under its Mooncake role,
    with `extra` arguments, its quantization hash the device's: with no device
    aiter's fp8 dtype is uint8, which the hash reads."""
    for name, value in (ENV | (ENV_PREFILL if side == "prefill" else {})).items():
        monkeypatch.setenv(name, value)
    transfer = json.dumps({"kv_role": ROLE[side], "kv_connector": "mooncake"})
    args = [*ARGS[side].split(), *extra, "--kv-transfer-config", transfer]
    config = served(tmp_path, monkeypatch, "gfx950", V4_PRO, *args)
    stated = law(DPA8.format(side))["engine_config"]["quantization"]
    monkeypatch.setattr(config.quant_config, "compute_hash", lambda: stated)
    return config


def tokenizer_term(hosts: list, config):
    return compass_run.tokenizer_term(
        {"host": hosts}, SimpleNamespace(is_fast=True), config
    )


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_tp8_side_picks_its_own_host_laws_among_both_sides_and_dpa8s(
    tmp_path, monkeypatch, side
):
    config = side_config(tmp_path, monkeypatch, side)
    names = [f.format(s) for f in (TP8, DPA8) for s in ("prefill", "decode")]

    host, term = tokenizer_term([law(n)["host"] for n in names], config)

    assert host == law(TP8.format(side))["host"]
    assert term == host["tokenizer"]
    assert host["key"]["engine"]["tensor_parallel_size"] == 8


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_a_tp8_side_refuses_the_dpa8_laws_alone_naming_its_layout(
    tmp_path, monkeypatch, side
):
    config = side_config(tmp_path, monkeypatch, side)
    hosts = [law(DPA8.format(s))["host"] for s in ("prefill", "decode")]

    with pytest.raises(ValueError, match="no host entry was fitted under") as e:
        tokenizer_term(hosts, config)
    for field in ("tensor_parallel_size", "data_parallel_size", "enable_dp_attention"):
        assert len(re.findall(rf"\b{field}\b", str(e.value))) == len(hosts)


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_a_dpa8_side_still_picks_its_own_law_beside_the_tp8_ones(
    tmp_path, monkeypatch, side
):
    config = side_config(tmp_path, monkeypatch, side, "--enable-dp-attention")
    names = [f.format(s) for f in (TP8, DPA8) for s in ("prefill", "decode")]

    host, _ = tokenizer_term([law(n)["host"] for n in names], config)

    assert host == law(DPA8.format(side))["host"]
