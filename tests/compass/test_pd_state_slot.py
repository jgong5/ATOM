# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer moves the request's state slot beside
its blocks when ATOM's transfer regions carry one, and its staging row in a
second write when they carry one.

Mooncake writes a request's blocks over the PAGE regions and its slot over the
per-request regions (`swa_block_regions`, `slot_regions`) the attention
builder publishes, then its compressor state from the `staging_region` row. Each model below is built driverless in a process of its
own as `compass trace ops` builds one, on a prefill-decode deployment's
config; its fake KV cache is allocated so ATOM's own `get_kv_transfer_tensors`
states the regions, and the runner then publishes what its connector prices.
DeepSeek-V4-Pro publishes its whole slot and stages its compressor state;
DeepSeek-R1-0528 keeps no state; a Qwen3.5 GDN hybrid plans a state slot its
regions do not carry.
"""

import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest
from test_pd_planned_block import (  # noqa: F401
    BUILD,
    _connector,
    build_each,
    publish,
    served,
)

from atom.compass import run as compass_run
from atom.compass.kv.transfer import Link, TransferModel

HERE = pathlib.Path(__file__).parent
CONFIGS = {
    "deepseek-v4-pro": HERE / "deepseek_v4_pro_config.json",
    "deepseek-r1-0528": HERE / "deepseek_r1_0528_config.json",
    "qwen3.5-27b": HERE / "qwen3_5_27b_config.json",
}
#: The stage-4 1P1D cell's prefill side: DPA-TP8, an fp8 cache, the simulated
#: connector in Mooncake's place.
CELL = [
    "--kv_cache_dtype",
    "fp8",
    "--enable-dp-attention",
    "--block-size",
    "16",
    "--kv-transfer-config",
    json.dumps({"kv_connector": "compass", "kv_role": "kv_producer"}),
]
#: A 4096-token prompt in DeepSeek-V4's 256-token blocks.
BLOCKS = 16

PLAN = BUILD + """
with forward._patches(recorder), fake, torch.inference_mode():
    runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
    runner.allocate_kv_cache(runner.pool_plan.paged_entries)
    regions = runner.attn_metadata_builder.get_kv_transfer_tensors()
runner.compass_block_file = pathlib.Path(out)
publish_planned_transfer(runner)
import os
from atom.compass.runner.overrides import _staging_row_bytes
os.environ["ATOM_PD_STAGING_POOL"] = "0"
print(json.dumps({
    "planned": runner.pool_plan.entry_bytes,
    "page_units": sum(r.unit_bytes for r in regions.block_regions),
    "slot_units": sum(
        r.unit_bytes for r in [*regions.swa_block_regions, *regions.slot_regions]
    ),
    "block_size": config.kv_cache_block_size,
    "staging_unit": getattr(regions.staging_region, "unit_bytes", 0),
    "staging_with_no_pool": _staging_row_bytes(runner),
}))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By model, what ATOM planned and its regions state, and the file its
    runner published."""
    return build_each(tmp_path_factory.mktemp("slot"), CONFIGS, PLAN, CELL)


@pytest.mark.parametrize("model", CONFIGS)
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_a_transfer_moves_the_slot_atoms_regions_carry_beside_its_blocks(
    served, tmp_path, built, model, role  # noqa: F811
):
    run = served()
    for engine in ("engine-P", "engine-D"):
        shutil.copy(built[model]["published"], tmp_path / f"now-{engine}.kv_block")
    raw = json.loads(CONFIGS[model].read_text())
    half = _connector(
        role,
        hf_config=SimpleNamespace(**raw),
        kv_cache_block_size=built[model]["block_size"],
        kv_cache_dtype="fp8",
    )
    (transfer,) = half._transfer
    link = run["machine"]["interconnect"]["intra_node"]
    block, slot = built[model]["page_units"], built[model]["slot_units"]
    staging = built[model]["staging_unit"]
    assert (transfer.bytes_per_block, transfer.slot_bytes, transfer.staging_bytes) == (
        block,
        slot,
        staging,
    )
    # A write of the prompt's blocks on this connector's own rank pair, and
    # the staging row's own write after it.
    assert half._written(next(iter(half._socks)), 0.0, BLOCKS) == pytest.approx(
        link["link_latency_s"] * (2 if staging else 1)
        + (BLOCKS * block + slot + staging)
        / (link["link_bandwidth_bytes_per_s"] * link["derate"])
    )


def test_only_deepseek_v4s_regions_carry_the_slot_it_planned(built):
    """V4-Pro's regions carry one planned state entry; Qwen3.5 plans a slot no
    region carries; R1 plans none."""
    v4, r1, qwen = (built[m] for m in CONFIGS)
    assert v4["slot_units"] == v4["planned"]["state"] > 0
    assert qwen["planned"]["state"] > 0 == qwen["slot_units"]
    assert "state" not in r1["planned"] and r1["slot_units"] == 0


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_sides_that_plan_different_slots_refuse_naming_both(
    served, tmp_path, role  # noqa: F811
):
    served()
    publish(tmp_path, "engine-P", 1000)
    compass_run.publish_transfer(tmp_path / "now-engine-D.kv_block", 1000, 1, 300)
    with pytest.raises(
        ValueError,
        match="engine-D at TP 1 plans a state slot of 300 B and engine-P at TP 1 "
        "one of 0 B",
    ):
        _connector(role)


@pytest.mark.parametrize("slot, first, second", [(0, 1.0, 2.5), (20, 3.5, 6.5)])
def test_a_write_of_no_blocks_still_moves_the_slot(slot, first, second):
    """A zero-block write moves the slot, paying the link's setup with it, and
    the next write waits for the slot's bytes to leave the link."""
    model = TransferModel(
        latency_s=1.0,
        bandwidth_bytes_per_s=10.0,
        bytes_per_block=10,
        setup_s=0.5,
        slot_bytes=slot,
    )
    link = Link(model)
    assert (link.start(0.0, 0), link.start(0.0, 1)) == (first, second)


def test_only_deepseek_v4_stages_its_compressor_state_and_no_pool_stages_none(built):
    """V4-Pro stages its compressor state, a row its slot also carries beside
    the SWA rows; R1 and Qwen3.5 stage nothing, and with an empty staging pool
    neither does V4-Pro."""
    v4, r1, qwen = (built[m] for m in CONFIGS)
    assert 0 < v4["staging_unit"] < v4["slot_units"]
    assert r1["staging_unit"] == qwen["staging_unit"] == 0
    assert [built[m]["staging_with_no_pool"] for m in CONFIGS] == [0, 0, 0]


@pytest.mark.parametrize("staging, first, second", [(0, 4.5, 6.5), (10, 6.5, 10.5)])
def test_a_staging_row_follows_in_a_second_write_that_holds_the_link(
    staging, first, second
):
    """The staging row's write pays the latency again, and the next write waits
    for that row's bytes, the first write's latency between them."""
    model = TransferModel(
        latency_s=1.0,
        bandwidth_bytes_per_s=10.0,
        bytes_per_block=10,
        setup_s=0.5,
        slot_bytes=20,
        staging_bytes=staging,
    )
    link = Link(model)
    assert (link.start(0.0, 1), link.start(0.0, 0)) == (first, second)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_sides_that_plan_different_staging_rows_refuse_naming_both(
    served, tmp_path, role  # noqa: F811
):
    served()
    publish(tmp_path, "engine-P", 1000)
    compass_run.publish_transfer(tmp_path / "now-engine-D.kv_block", 1000, 1, 0, 300)
    with pytest.raises(
        ValueError,
        match="engine-D at TP 1 plans a staging row of 300 B and engine-P at TP 1 "
        "one of 0 B",
    ):
        _connector(role)
