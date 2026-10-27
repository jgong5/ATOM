# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer moves the request's state slot beside
its blocks when ATOM's transfer regions carry one.

Mooncake writes a request's blocks over the PAGE regions and its slot over the
per-request regions (`swa_block_regions`, `slot_regions`) the attention
builder publishes. Each model below is built driverless in a process of its
own as `compass trace ops` builds one, on a prefill-decode deployment's
config; its fake KV cache is allocated so ATOM's own `get_kv_transfer_tensors`
states the regions, and the runner then publishes what its connector prices.
DeepSeek-V4-Pro publishes its whole slot; DeepSeek-R1-0528 keeps no state; a
Qwen3.5 GDN hybrid plans a state slot its regions do not carry.
"""

import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_pd_planned_block import _connector, publish, served  # noqa: F401

from atom.compass import run as compass_run
from atom.compass.kv.transfer import Link, TransferModel

TREE = pathlib.Path(__file__).resolve().parents[2]
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

PLAN = """
import json, pathlib, sys, torch
root, out, *atom_args = sys.argv[1:]
from atom.compass.capture import driverless
declared = driverless.install(root, "gfx950")
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from atom.compass.capture import forward
from atom.compass.runner.overrides import publish_paged_block
from atom.config import set_current_atom_config
config = forward._config(root + "/model", 8, atom_args)
parallel = config.parallel_config
declared["group"] = driverless.group(1, parallel.data_parallel_size)
set_current_atom_config(config)
fake = driverless.driverless_mode(ShapeEnv())
recorder = forward._Recorder(config.compilation_config.static_forward_context, fake)
driverless.on_wait = recorder.wait
with forward._patches(recorder):
    runner = forward._build_runner(config, fake)
    with fake, torch.inference_mode():
        runner.get_num_blocks()
        runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
        runner.allocate_kv_cache(runner.pool_plan.paged_entries)
        regions = runner.attn_metadata_builder.get_kv_transfer_tensors()
runner.compass_block_file = pathlib.Path(out)
publish_paged_block(runner)
print(json.dumps({
    "planned": runner.pool_plan.entry_bytes,
    "page_units": sum(r.unit_bytes for r in regions.block_regions),
    "slot_units": sum(
        r.unit_bytes for r in [*regions.swa_block_regions, *regions.slot_regions]
    ),
    "block_size": config.kv_cache_block_size,
}))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By model, what ATOM planned and its regions state, and the file its
    runner published, each built in a process of its own."""
    root = tmp_path_factory.mktemp("slot")

    def build(name):
        (root / name / "model").mkdir(parents=True)
        shutil.copy(CONFIGS[name], root / name / "model" / "config.json")
        out = root / name / "now-engine-P.kv_block"
        done = subprocess.run(
            [sys.executable, "-c", PLAN, str(root / name), str(out), *CELL],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE)},
        )
        assert done.returncode == 0, done.stderr[-4000:]
        return json.loads(done.stdout.splitlines()[-1]) | {"published": out}

    with concurrent.futures.ThreadPoolExecutor(len(CONFIGS)) as pool:
        return dict(zip(CONFIGS, pool.map(build, CONFIGS)))


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
    assert (transfer.bytes_per_block, transfer.slot_bytes) == (block, slot)
    # A write of the prompt's blocks on this connector's own rank pair.
    assert half._written(next(iter(half._socks)), 0.0, BLOCKS) == pytest.approx(
        link["link_latency_s"]
        + (BLOCKS * block + slot)
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
    compass_run.publish_block(tmp_path / "now-engine-D.kv_block", 1000, 1, 300)
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
