# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer under the FP4 sparse indexer is refused
where ATOM refuses it, naming ATOM's reason.

DeepSeek-V3.2 (`deepseek_v3_2_config.json`, its config.json byte for byte at
revision a7e62ac04ecb2c0a54d736dc46601c5606cf10a6) is built driverless on the
simulated connector's config, once per index cache dtype, each in a process of
its own, and plans its pool with ATOM's `get_num_blocks`, as a simulated run
does. Its runner then publishes what its connector prices, or refuses; its fake
KV cache is allocated and ATOM's own `get_kv_transfer_tensors` is asked for the
regions on Mooncake's config, the backend the simulated connector stands for,
and on `lmcache_mp`'s. Under fp4 ATOM refuses Mooncake and takes `lmcache_mp`;
under fp8 it takes both.
"""

import json
import pathlib

import pytest
from test_pd_planned_block import BUILD, build_each

CONFIG = pathlib.Path(__file__).with_name("deepseek_v3_2_config.json")
#: The 1P1D cell's prefill side: DPA-TP8, an fp8 cache, 64-token blocks.
CELL = [
    "--kv_cache_dtype",
    "fp8",
    "--enable-dp-attention",
    "--block-size",
    "64",
    "--kv-transfer-config",
    json.dumps({"kv_connector": "compass", "kv_role": "kv_producer"}),
]
#: By index cache dtype, what refuses the transfer: ATOM's builder on the
#: config of the connector named, and the simulated runner (``compass``).
REFUSED = {"fp4": ["compass", "mooncake"], "fp8": []}

#: After `BUILD`, what refuses the transfer and why.
REGIONS = BUILD + """
from atom.compass.runner.overrides import RunnerRefusal
refused = {}
runner.compass_block_file = pathlib.Path(out)
try:
    publish_planned_transfer(runner)
except RunnerRefusal as e:
    refused["compass"] = str(e)
builder = runner.attn_metadata_builder
with forward._patches(recorder), fake, torch.inference_mode():
    config.kv_transfer_config = {}
    runner.allocate_kv_cache(runner.pool_plan.paged_entries)
    for name in ("mooncake", "lmcache_mp"):
        config.kv_transfer_config = {"kv_connector": name}
        try:
            builder.get_kv_transfer_tensors()
        except NotImplementedError as e:
            refused[name] = str(e)
print(json.dumps(refused))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By index cache dtype, what refused DeepSeek-V3.2's transfer and why, and
    the file its runner published."""
    root = tmp_path_factory.mktemp("fp4")
    return {
        dtype: build_each(
            root / dtype,
            {"deepseek-v3.2": CONFIG},
            REGIONS,
            [*CELL, "--index_cache_dtype", dtype],
        )["deepseek-v3.2"]
        for dtype in REFUSED
    }


@pytest.mark.parametrize("dtype", REFUSED)
def test_a_transfer_refuses_under_the_fp4_indexer_where_atom_does(built, dtype):
    refused = built[dtype]
    assert sorted(refused.keys() - {"published"}) == REFUSED[dtype]
    assert refused["published"].exists() == (not REFUSED[dtype])
    if "compass" in refused:
        assert refused["compass"].startswith("ATOM's `")
        assert refused["compass"].partition(": ")[2] in refused["mooncake"]
