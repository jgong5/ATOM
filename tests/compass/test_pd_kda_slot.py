# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer of Kimi-K3 moves the request's KDA
state slot beside its blocks.

On a Mooncake deployment Kimi's attention builder publishes one region per KDA
state cache and layer (`slot_regions`) beside its MLA blocks, and Mooncake
writes one unit of each. Kimi-K3 (`kimi_k3_config.json`, its config.json byte
for byte at revision f831ab66814297da540d832a5235f8e904f29d06) is built
driverless at TP8 in a process of its own as `compass trace ops` builds one,
with ReplaySSM off, as by default, and on, which adds a cursor region the
planned state entry does not count. Its fake KV cache is allocated so ATOM's
own `get_kv_transfer_tensors` states the regions on Mooncake's config, and the
runner publishes what its connector prices.
"""

import json
import pathlib
import shutil

import pytest
from test_pd_planned_block import BUILD, _connector, build_each, served  # noqa: F401

CONFIG = pathlib.Path(__file__).with_name("kimi_k3_config.json")
#: Kimi-K3 at TP8 with an fp8 cache in 128-token blocks, as its recipe serves
#: it, the simulated connector in Mooncake's place.
CELL = [
    "--kv_cache_dtype",
    "fp8",
    "--block-size",
    "128",
    "--kv-transfer-config",
    json.dumps({"kv_connector": "compass", "kv_role": "kv_producer"}),
]
#: A 4096-token prompt in 128-token blocks.
BLOCKS = 32

CONFIGURED = 'config = forward._config(root + "/model", 8, atom_args)\n'
#: The vision tower's config, which ATOM reads through the checkpoint's own
#: configuration code, taken from config.json; ReplaySSM on for a model
#: directory named `*-replayssm`.
VISION = """
import os
from types import SimpleNamespace
os.environ["ATOM_ENABLE_REPLAYSSM"] = str(int(root.endswith("-replayssm")))
raw = json.loads(pathlib.Path(root, "model", "config.json").read_text())
config.multimodal_config = SimpleNamespace(
    vision_config=SimpleNamespace(**raw["vision_config"])
)
"""
#: The build at TP8, not at a DP-attention rank's TP1.
TP8 = BUILD.replace("driverless.group(1,", "driverless.group(8,")
PLAN = TP8.replace(CONFIGURED, CONFIGURED + VISION) + """
runner.compass_block_file = pathlib.Path(out)
publish_planned_transfer(runner)
with forward._patches(recorder), fake, torch.inference_mode():
    runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
    runner.allocate_kv_cache(runner.pool_plan.paged_entries)
    config.kv_transfer_config = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    regions = runner.attn_metadata_builder.get_kv_transfer_tensors()
print(json.dumps({
    "planned": runner.pool_plan.entry_bytes,
    "page_units": sum(r.unit_bytes for r in regions.block_regions),
    "slot_units": sum(r.unit_bytes for r in regions.slot_regions),
    "staging_unit": getattr(regions.staging_region, "unit_bytes", 0),
}))
"""
MODELS = {"kimi-k3": CONFIG, "kimi-k3-replayssm": CONFIG}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By build, what ATOM planned and its regions state, and the file its
    runner published."""
    assert BUILD.count(CONFIGURED) == BUILD.count("driverless.group(1,") == 1
    return build_each(tmp_path_factory.mktemp("kda"), MODELS, PLAN, CELL)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_a_kimi_k3_transfer_moves_its_kda_slot_beside_its_blocks(
    served, tmp_path, built, model, role  # noqa: F811
):
    run = served()
    for engine in ("engine-P", "engine-D"):
        shutil.copy(built[model]["published"], tmp_path / f"now-{engine}.kv_block")
    half = _connector(role)
    (transfer,) = half._transfer
    block, slot = built[model]["page_units"], built[model]["slot_units"]
    assert built[model]["staging_unit"] == 0
    assert (transfer.bytes_per_block, transfer.slot_bytes, transfer.staging_bytes) == (
        block,
        slot,
        0,
    )
    link = run["machine"]["interconnect"]["intra_node"]
    assert half._written(next(iter(half._socks)), 0.0, BLOCKS) == pytest.approx(
        link["link_latency_s"]
        + (BLOCKS * block + slot)
        / (link["link_bandwidth_bytes_per_s"] * link["derate"])
    )


def test_replayssms_cursor_moves_beside_the_planned_kda_state(built):
    """Kimi-K3 plans no kpool tail, so its regions carry its whole planned
    state entry, and ReplaySSM's int32 cursor beside it."""
    off, on = (built[m] for m in MODELS)
    assert off["slot_units"] == off["planned"]["state"] > 0
    assert on["slot_units"] == on["planned"]["state"] + 4
