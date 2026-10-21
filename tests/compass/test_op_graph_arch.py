# SPDX-License-Identifier: MIT
"""An `op_graph`'s key names the device's arch: one store holds a gfx942 and a
gfx950 graph of one structure and shape, and each reader takes its own arch's.

No device: `torch.cuda` reports the arch as a capture's declared device does,
or as a real one does, with its target features.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import CONDITIONS, GATES, stanza

from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import decode, structure
from atom.compass.run import _op_graphs

ARCHES = ("gfx942", "gfx950")
RANK = Topology().ranks()[0]
#: The engine fields a stand-in config states, which an op_graph key reads.
ENGINE = {
    "kv_cache_block_size": 16,
    "kv_cache_dtype": "bf16",
    "quant_config": SimpleNamespace(compute_hash=lambda: "unquantized-bf16"),
    "enable_dp_attention": False,
    "enable_tbo": False,
    "enable_tbo_decode": False,
}


def _device(monkeypatch, props):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)


@pytest.fixture
def config(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )


@pytest.fixture
def store(tmp_path, config, monkeypatch):
    """A graph of `decode B=2` keyed as a capture keys it on each arch's
    declared device, its region naming the arch, in one store."""
    store = ArtifactStore(str(tmp_path / "store"))
    for arch in ARCHES:
        _device(monkeypatch, driverless._props(arch))
        key = Key.of(
            Kind.OP_GRAPH, structure=structure(config, "decode"), shape="decode B=2"
        )
        store.publish(
            key,
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=CONDITIONS,
            gates=GATES,
            members={
                member_name("graph", RANK, "json"): json.dumps(
                    {"region": {"op": arch}}
                ).encode()
            },
        )
    return store


@pytest.mark.parametrize("arch", ARCHES)
def test_a_run_reads_the_graph_of_its_machine_s_arch(store, config, arch):
    lookup = _op_graphs(store.root, config, arch)

    assert lookup("decode", 2, 2) == {"op": arch}


@pytest.mark.parametrize("arch", ARCHES)
def test_measure_ops_reads_the_graph_of_its_gpu_s_arch(
    store, config, monkeypatch, arch
):
    _device(monkeypatch, SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-"))

    assert ops.graph(store, config, decode(2))["region"] == {"op": arch}


def test_a_plan_traces_a_missing_graph_on_its_gpu_s_arch(config, monkeypatch):
    _device(monkeypatch, SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"))
    called = []
    monkeypatch.setattr(ops, "_child", lambda fn, argv: called.append(argv))

    ops._capture(config, [decode(2)], "store")

    (argv,) = called
    assert argv[argv.index("--arch") + 1] == "gfx950"
