# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer of Kimi-K3 moves the request's KDA
state slot in a second write after its blocks.

On a Mooncake deployment Kimi's attention builder publishes one region per KDA
state cache and layer (`slot_regions`) beside its MLA blocks, and Mooncake
writes the blocks, then one unit of each slot region. Kimi-K3 (`kimi_k3_config.json`, its config.json byte
for byte at revision f831ab66814297da540d832a5235f8e904f29d06) is built
driverless at TP8 in a process of its own as `compass trace ops` builds one,
with ReplaySSM off, as by default, and on, which adds a cursor region the
planned state entry does not count. The runner publishes what its connector
prices, then its fake KV cache is allocated on Mooncake's config, where ATOM's
own `get_kv_transfer_tensors` states the regions.
"""

import json
import pathlib
import shutil

import pytest
from test_pd_planned_block import BUILD, _connector, build_each, served  # noqa: F401

from atom.compass.kv.transfer import Link, TransferModel

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
    config.kv_transfer_config = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    runner.allocate_kv_cache(runner.pool_plan.paged_entries)
    regions = runner.attn_metadata_builder.get_kv_transfer_tensors()
print(json.dumps({
    "page_units": sum(r.unit_bytes for r in regions.block_regions),
    "slot_units": sum(r.unit_bytes for r in regions.slot_regions),
    "staging_unit": getattr(regions.staging_region, "unit_bytes", 0),
}))
"""
MODELS = {"kimi-k3": CONFIG, "kimi-k3-replayssm": CONFIG}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By build, what ATOM's regions state and the file its runner
    published."""
    assert BUILD.count(CONFIGURED) == BUILD.count("driverless.group(1,") == 1
    return build_each(tmp_path_factory.mktemp("kda"), MODELS, PLAN, CELL)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_a_kimi_k3_transfer_moves_its_kda_slot_after_its_blocks(
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
        2 * link["link_latency_s"]
        + (BLOCKS * block + slot)
        / (link["link_bandwidth_bytes_per_s"] * link["derate"])
    )


@pytest.mark.parametrize("after, first, second", [(False, 4.0, 6.0), (True, 5.0, 7.0)])
def test_a_slot_after_the_blocks_is_a_second_write_unless_no_block_moves(
    after, first, second
):
    """A slot written after the blocks pays the latency again, and the next
    write waits for it; a write of no blocks posts the slot alone."""
    model = TransferModel(
        latency_s=1.0,
        bandwidth_bytes_per_s=10.0,
        bytes_per_block=10,
        slot_bytes=20,
        slot_after_blocks=after,
    )
    link = Link(model)
    assert (link.start(0.0, 1), link.start(0.0, 0)) == (first, second)


#: A lazy trace of the model under argv's root on argv's cell, `compass run`'s
#: on a store miss (`run._tracer`): ATOM's runner stepped at one decode, its KV
#: pool allocated as a trace allocates it.
TRACE = """
import json, pathlib, sys
from types import SimpleNamespace
root, out, *atom_args = sys.argv[1:]
from atom.compass.capture import forward
raw = json.loads(pathlib.Path(root, "model", "config.json").read_text())
tracer = forward.Tracer(root + "/model", 8, root + "/declared", atom_args, "gfx950")
tracer.config.multimodal_config = SimpleNamespace(
    vision_config=SimpleNamespace(**raw["vision_config"])
)
graph, *_ = next(iter(tracer.trace([forward.decode(1)], [forward.decode(1)])))
print(json.dumps({"kv": tracer.config.kv_transfer_config,
                  "structure": graph.applicability.structure}))
"""


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_a_lazy_trace_of_a_kimi_k3_side_runs_on_mooncake_s_connector(
    monkeypatch, tmp_path, role
):
    """A side's simulated connector is traced as Mooncake's, at the side's
    role, so ATOM's KV allocation, which asks Kimi's builder for the regions
    a P/D backend moves, traces the step. A Mooncake decode serves Kimi-K3's
    mono kernels, which a capture holds no graph of, as for an aggregated
    deployment (#4143): the decode is traced with mono off."""
    if role == "kv_consumer":
        monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    kv = {"kv_connector": "compass", "kv_role": role, "compass_instance": 0}
    cell = [*CELL[:-1], json.dumps(kv)]
    traced = build_each(tmp_path, {"kimi-k3": CONFIG}, TRACE, cell)["kimi-k3"]
    assert traced["kv"] == {"kv_role": role, "kv_connector": "mooncake"}
    assert traced["structure"]


def test_only_the_simulated_connector_is_made_real():
    """The simulated connector, alone or in a `multi`, becomes Mooncake's at
    its role; any other connector stays as it is."""
    from atom.compass.drive.run import real_kv

    store = {"kv_connector": "mooncake_store", "kv_role": "kv_consumer"}
    simulated = {"kv_connector": "Compass", "kv_role": "kv_consumer", "x": 1}
    multi = {"kv_connector": "multi", "connectors": [simulated, store]}
    real = {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
    assert json.loads(real_kv(json.dumps(simulated))) == real
    assert json.loads(real_kv(json.dumps(multi)))["connectors"] == [real, store]
    for other in ({}, store, {"kv_role": "kv_producer"}):
        assert json.loads(real_kv(json.dumps(other))) == other
