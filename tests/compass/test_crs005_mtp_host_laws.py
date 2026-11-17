# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's MTP host laws for crsuse2-m2m-v3-005/013, one per side of
each 1P1D MTP cell `MTP` names. Each side's server, its config as ATOM
resolves it from the cell's arguments and environment, picks its own entry
among every crs005 V4-Pro law's (`run.pick_fitted`, as a run file's list is
read) and passes `run.refuse_unfitted_host` on the crs005 platform; that side
without its drafter is refused by the law, naming `engine.speculative`. The
recurrence a side's runner builds from its law charges one forward of each kind
that side's records ran.

The host's CPU is crs005's, as the laws' key states it, and the fixture's
config.json is the checkpoint's; the quantization hash is the DPA8 laws', which
the same checkpoint keys. No device: aiter's arch probe answers a declared arch.
"""

import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_key_index_cache_dtype import TREE, served

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.artifacts import machine_platform
from atom.compass.backends.json_file import read_json
from atom.compass.backends.launch_features import stated
from atom.compass.run import step_grid
from atom.compass.run.step_grid import BUCKETS
from atom.compass.runner.overrides import ADMISSION_IN_E, HostDeviceRecurrence
from atom.compass.spec import MachineSpec

LAWS = TREE / "scripts" / "compass" / "laws"
MACHINE = Path(__file__).resolve().parent / "machines" / "mi355x-crs005.json"
MTP = {
    "tp8-mtp3": "deepseek-v4-pro-tp8-mtp3-mi355x-crs005-{}.json.xz",
    "dpa8-mtp1": "deepseek-v4-pro-dpa8-mtp1-mi355x-crs005-{}.json.xz",
}
DPA8 = "deepseek-v4-pro-dpa8-mi355x-crs005-{}.json.xz"
#: Each cell's ATOM arguments on both sides, its layout's and its drafter's:
#: the 1P1D cell with the recipe's MTP flags on both sides.
COMMON = (
    "--trust-remote-code --kv_cache_dtype fp8 --block-size 16 "
    "--no-enable_prefix_caching -tp 8 --max-num-seqs 512"
)
LAYOUT = {"tp8-mtp3": "", "dpa8-mtp1": "--enable-dp-attention"}
DRAFTER = {
    "tp8-mtp3": "--method mtp --num-speculative-tokens 3",
    "dpa8-mtp1": "--method mtp --num-speculative-tokens 1",
}
CAPTURE = ",".join(map(str, [1, 2, *range(4, 257, 4)]))
SIDE = {"prefill": "--enable-tbo", "decode": f"--cudagraph-capture-sizes [{CAPTURE}]"}
ROLE = {"prefill": "kv_producer", "decode": "kv_consumer"}
ENV = {
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "ATOM_MOE_GU_ITLV": "1",
    "AITER_LOG_LEVEL": "WARNING",
}
ENV_PREFILL = {"GPU_MAX_HW_QUEUES": "5", "ATOM_NUMA_BIND": "1"}
#: The forward kinds each side's records ran: a prefill side's prefill, middle
#: and decode forwards, a decode side's decodes alone.
KINDS = {"prefill": ("prefill", "middle", "decode"), "decode": ("decode",)}
#: One forward of each kind as `forward_shape` reads it: prefill sequences,
#: tokens, sequences and whether it produces output.
FORWARD = {
    "prefill": (1, 16384, 1, True),
    "middle": (1, 16384, 1, False),
    "decode": (0, 4, 1, True),
}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)


@cache
def law(name: str) -> dict:
    return read_json(LAWS / name)


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
    run = {
        "host": [law(p.name)["host"] for p in sorted(LAWS.glob("*-crs005-*.json.xz"))]
    }

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


@pytest.mark.parametrize("cell", MTP)
@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_kind_a_side_runs_is_charged_by_the_recurrence_its_runner_builds(
    cell, side
):
    """Built as `run.runner` builds it for a V4 P/D side, whose drafter defers
    its output (`tokenIDProcessor`), so each output forward charges `L_mtp`;
    a kind lacking a law for a segment it runs is refused."""
    host = law(MTP[cell].format(side))["host"]
    recurrence = HostDeviceRecurrence(
        host["laws"],
        host.get("skip", ()),
        lambda: 0.0,
        None,
        mtp=True,
        admit=host.get("admit", ADMISSION_IN_E),
        columns=host.get("columns"),
        launch=stated(host.get("key")),
        defer=True,
    )
    for kind in KINDS[side]:
        prefill_seqs, tokens, seqs, output = FORWARD[kind]
        adopted = (host.get("columns") or {}).get(kind, {})
        batch = SimpleNamespace(
            total_seqs_num_prefill=prefill_seqs,
            total_tokens_num=tokens,
            total_seqs_num=seqs,
            produces_output=lambda output=output: output,
            compass_launch={f: 0.0 for c in adopted.values() for f in c},
        )
        recurrence.device_end(batch, 0.01, 0.001 if output else 0.0)
