# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Pro's DPA8 EP8 host laws for crsuse2-m2m-v3-005/013, one per side
of a 1P1D EP8 cell, fitted on lone cells whose records state the graph sets ATOM
recorded. Each side's server, its config as ATOM resolves it from the cells'
arguments and environment under its Mooncake role, picks its own entry among
every crs005 V4-Pro law's (`run.pick_fitted`, as a run file's list is read) and
passes `run.refuse_unfitted_host` on the crs005 platform. The recurrence a
side's runner builds from its law, with the launch provider its key states,
charges one forward of each kind the law prices, its launch columns included.

The host's CPU and the quantization hash are the law's own, as its key states
them. No device: aiter's arch probe answers a declared arch.
"""

import json
from types import SimpleNamespace

import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_crs005_mtp_host_laws import (
    CAPTURE,
    COMMON,
    ENV,
    ENV_PREFILL,
    FORWARD,
    LAWS,
    ROLE,
    law,
    platform,
)
from test_key_index_cache_dtype import served

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.backends.launch_features import stated
from atom.compass.run import step_grid
from atom.compass.run.step_grid import BUCKETS
from atom.compass.runner.overrides import ADMISSION_IN_E, HostDeviceRecurrence

EP8 = "deepseek-v4-pro-dpa8-ep8-mi355x-crs005-{}.json.xz"
#: Both sides' layout, as the cells ran it: DP attention, expert parallelism on
#: the standard MoE backend, and the fp8 index cache a P/D side resolves.
LAYOUT = (
    "--enable-dp-attention --enable-expert-parallel --moe-backend standard "
    "--index_cache_dtype fp8"
)
SIDE = {"prefill": "--enable-tbo", "decode": f"--cudagraph-capture-sizes [{CAPTURE}]"}
ENV_EP = {"MORI_SHMEM_HEAP_SIZE": "16G"}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)


def side_config(tmp_path, monkeypatch, side: str):
    """The config ATOM resolves for `side` of the EP8 cell under its Mooncake
    role, on the host CPU and quantization hash the side's law states."""
    env = ENV | ENV_EP | (ENV_PREFILL if side == "prefill" else {})
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    transfer = json.dumps({"kv_role": ROLE[side], "kv_connector": "mooncake"})
    args = [
        *f"{COMMON} {LAYOUT} {SIDE[side]}".split(),
        "--kv-transfer-config",
        transfer,
    ]
    config = served(tmp_path, monkeypatch, "gfx950", V4_PRO, *args)
    key = law(EP8.format(side))["host"]["key"]
    monkeypatch.setattr(compass_run, "host_cpu", lambda: key["host_cpu"])
    quantization = key["engine"]["quantization"]
    monkeypatch.setattr(step_grid, "quantization", lambda _: quantization)
    return config


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_ep8_side_picks_its_own_host_laws_and_passes_the_host_key_check(
    tmp_path, monkeypatch, side
):
    config = side_config(tmp_path, monkeypatch, side)
    run = {
        "host": [law(p.name)["host"] for p in sorted(LAWS.glob("*-crs005-*.json.xz"))]
    }

    compass_run.pick_fitted(run, "host", config)

    assert run["host"] == law(EP8.format(side))["host"]
    compass_run.refuse_unfitted_host(run["host"], config, platform())


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_kind_an_ep8_side_runs_is_charged_with_its_launch_columns(side):
    """Built as `run.runner` builds it for a V4 P/D side: the provider its key's
    graph sets name must declare every launch feature a column adopts, and a
    forward lacking a feature its kind's columns price would be refused."""
    host = law(EP8.format(side))["host"]
    recurrence = HostDeviceRecurrence(
        host["laws"],
        host.get("skip", ()),
        lambda: 0.0,
        None,
        admit=host.get("admit", ADMISSION_IN_E),
        columns=host.get("columns"),
        launch=stated(host.get("key")),
        defer=True,
    )
    for kind, (prefill_seqs, tokens, seqs, output) in FORWARD.items():
        adopted = (host.get("columns") or {}).get(kind, {})
        batch = SimpleNamespace(
            total_seqs_num_prefill=prefill_seqs,
            total_tokens_num=tokens,
            total_seqs_num=seqs,
            produces_output=lambda output=output: output,
            compass_launch={f: 1.0 for c in adopted.values() for f in c},
        )
        recurrence.device_end(batch, 0.01, 0.0)
