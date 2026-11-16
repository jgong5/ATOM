# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's MTP host laws for crsuse2-m2m-v3-005/013, one per side of
each 1P1D MTP cell `MTP` names. Each side's server, its config as ATOM
resolves it from the cell's arguments and environment, picks its own entry
among every crs005 V4-Pro law's (`run.pick_fitted`, as a run file's list is
read) and passes `run.refuse_unfitted_host` on the crs005 platform; that side
without its drafter is refused by the law, naming `engine.speculative`.

The host's CPU is crs005's, as the laws' key states it, and the fixture's
config.json is the checkpoint's; the quantization hash is the DPA8 laws', which
the same checkpoint keys. No device: aiter's arch probe answers a declared arch.
"""

import json
from functools import cache
from pathlib import Path

import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_key_index_cache_dtype import TREE, served

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.artifacts import machine_platform
from atom.compass.run import step_grid
from atom.compass.run.step_grid import BUCKETS
from atom.compass.spec import MachineSpec

LAWS = TREE / "scripts" / "compass" / "laws"
MACHINE = Path(__file__).resolve().parent / "machines" / "mi355x-crs005.json"
MTP = {"tp8-mtp3": "deepseek-v4-pro-tp8-mtp3-mi355x-crs005-{}.json"}
DPA8 = "deepseek-v4-pro-dpa8-mi355x-crs005-{}.json"
#: Each cell's ATOM arguments on both sides, its layout's and its drafter's:
#: the 1P1D cell with the recipe's MTP flags on both sides.
COMMON = (
    "--trust-remote-code --kv_cache_dtype fp8 --block-size 16 "
    "--no-enable_prefix_caching -tp 8 --max-num-seqs 512"
)
LAYOUT = {"tp8-mtp3": ""}
DRAFTER = {"tp8-mtp3": "--method mtp --num-speculative-tokens 3"}
CAPTURE = ",".join(map(str, [1, 2, *range(4, 257, 4)]))
SIDE = {"prefill": "--enable-tbo", "decode": f"--cudagraph-capture-sizes [{CAPTURE}]"}
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


@cache
def law(name: str) -> dict:
    return json.loads((LAWS / name).read_text())


def platform() -> dict:
    return machine_platform(MachineSpec.from_mapping(json.loads(MACHINE.read_text())))


def side_config(tmp_path, monkeypatch, cell: str, side: str, drafter: bool = True):
    """The config ATOM resolves for `side` of `cell` under its Mooncake role, its
    drafter's flags only with `drafter`, on crs005's host CPU with the DPA8
    laws' quantization hash."""
    for name, value in (ENV | (ENV_PREFILL if side == "prefill" else {})).items():
        monkeypatch.setenv(name, value)
    transfer = json.dumps({"kv_role": ROLE[side], "kv_connector": "mooncake"})
    flags = [COMMON, LAYOUT[cell], DRAFTER[cell] if drafter else "", SIDE[side]]
    args = [*" ".join(flags).split(), "--kv-transfer-config", transfer]
    config = served(tmp_path, monkeypatch, "gfx950", V4_PRO, *args)
    stated = law(DPA8.format(side))["host"]["key"]
    monkeypatch.setattr(compass_run, "host_cpu", lambda: stated["host_cpu"])
    quantization = stated["engine"]["quantization"]
    monkeypatch.setattr(step_grid, "quantization", lambda _: quantization)
    return config


@pytest.mark.parametrize("cell", MTP)
@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_mtp_side_picks_its_own_host_laws_and_passes_the_host_key_check(
    tmp_path, monkeypatch, cell, side
):
    config = side_config(tmp_path, monkeypatch, cell, side)
    run = {"host": [law(p.name)["host"] for p in sorted(LAWS.glob("*-crs005-*.json"))]}

    compass_run.pick_fitted(run, "host", config)

    assert run["host"] == law(MTP[cell].format(side))["host"]
    compass_run.refuse_unfitted_host(run["host"], config, platform())


@pytest.mark.parametrize("cell", MTP)
@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_a_side_without_its_drafter_is_refused_naming_it(
    tmp_path, monkeypatch, cell, side
):
    config = side_config(tmp_path, monkeypatch, cell, side, drafter=False)
    host = law(MTP[cell].format(side))["host"]

    with pytest.raises(ValueError, match="fitted under another") as e:
        compass_run.refuse_unfitted_host(host, config, platform())
    assert "engine.speculative" in str(e.value)
