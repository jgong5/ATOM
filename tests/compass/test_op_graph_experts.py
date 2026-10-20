# SPDX-License-Identifier: MIT
"""An `op_graph`'s key names the routed experts' Triton path: one store holds
an aiter and a Triton graph of V4-Pro at EP8 for one shape, and each reader
takes the one its environment selects. Keys of aiter's path and of dense
models are what they were.

No device: `torch.cuda` reports the arch as a capture's declared device does,
and ATOM's own `Mxfp4MoEMethod` decides the path on a stated arch.
"""

import itertools
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import CONDITIONS, GATES, stanza

from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import _check_experts, decode, experts, structure
from atom.compass.run import _op_graphs

RANK = Topology().ranks()[0]
TRITON = (
    "ATOM_USE_TRITON_MOE",
    "ATOM_USE_TRITON_MOE_A4W4",
    "ATOM_USE_TRITON_MOE_DECODE",
)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in (*TRITON, "ATOM_USE_TRITON_GEMM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a, **k: driverless._props("gfx950")
    )


def _config(tmp_path, architecture="DeepseekV4ForCausalLM", ep=True, **hf):
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=[architecture], **hf),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_expert_parallel=ep,
        enforce_eager=True,
    )


@pytest.fixture
def v4(tmp_path):
    return _config(tmp_path, n_routed_experts=384)


@pytest.fixture
def store(tmp_path, v4, monkeypatch):
    """A `decode B=2` graph of V4-Pro at EP8 on gfx950 keyed as a capture keys
    it with ATOM's environment unset, and one under ``ATOM_USE_TRITON_MOE=1``,
    each region naming its path."""
    store = ArtifactStore(str(tmp_path / "store"))
    for path, env in (("aiter", None), ("triton", "1")):
        if env:
            monkeypatch.setenv("ATOM_USE_TRITON_MOE", env)
        key = Key.of(
            Kind.OP_GRAPH, structure=structure(v4, "decode"), shape="decode B=2"
        )
        store.publish(
            key,
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=CONDITIONS,
            gates=GATES,
            members={
                member_name("graph", RANK, "json"): json.dumps(
                    {"region": {"op": path}}
                ).encode()
            },
        )
    monkeypatch.delenv("ATOM_USE_TRITON_MOE")
    return store


@pytest.mark.parametrize("env,path", [(None, "aiter"), ("0", "aiter"), ("1", "triton")])
def test_each_reader_takes_the_graph_its_environment_selects(
    store, v4, monkeypatch, env, path
):
    if env:
        monkeypatch.setenv("ATOM_USE_TRITON_MOE", env)

    assert _op_graphs(store.root, v4, "gfx950")("decode", 2, 2) == {"op": path}
    assert ops.graph(store, v4, decode(2))["region"] == {"op": path}


def test_aiter_and_dense_keys_are_unchanged(tmp_path, v4, monkeypatch):
    digest = structure(v4, "decode").split()[2]
    assert structure(v4, "decode") == (
        f"DeepseekV4ForCausalLM gfx950 {digest} tp=1 dp=8 ep=8 decode eager"
    )
    monkeypatch.setenv("ATOM_USE_TRITON_MOE", "1")
    dense = _config(tmp_path, "Qwen3ForCausalLM", ep=False)
    dense.parallel_config.data_parallel_size = 1

    assert structure(dense, "decode", device="gfx942") == (
        f"Qwen3ForCausalLM gfx942 {digest} tp=1 decode eager"
    )
    assert structure(v4, "decode").endswith(" ep=8 experts=triton decode eager")


#: Arch, EP, EPLB, MegaMoE, ATOM_USE_TRITON_MOE (None unset),
#: ATOM_USE_TRITON_GEMM, ATOM_USE_TRITON_MOE_DECODE.
GRID = list(
    itertools.product(
        ("gfx942", "gfx950", "gfx1250"),
        (False, True),
        (False, True),
        (False, True),
        (None, "0", "1"),
        ("0", "1"),
        ("0", "1"),
    )
)
#: Builds ATOM's own method at every point of `GRID` in a process of its own,
#: since the driverless declarations aiter's import needs are process-global;
#: prints each point's key qualifier, the method's flags, and the capture
#: check's refusal or None.
METHODS = """
import json, os, sys
from types import SimpleNamespace
from atom.compass.capture import driverless
driverless.install(sys.argv[1], "gfx950")
from aiter import QuantType, dtypes
from atom.compass.capture.forward import _check_experts, experts
from atom.model_ops import moe
import torch
quant = SimpleNamespace(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2,
                        quant_method="", is_dynamic=True)
out = []
for arch, ep, eplb, mega, triton, gemm, decoding in json.loads(sys.argv[2]):
    os.environ.pop("ATOM_USE_TRITON_MOE", None)
    if triton is not None:
        os.environ["ATOM_USE_TRITON_MOE"] = triton
    os.environ["ATOM_USE_TRITON_GEMM"] = gemm
    os.environ["ATOM_USE_TRITON_MOE_DECODE"] = decoding
    moe.get_gfx = lambda: arch
    current = SimpleNamespace(eplb_enable=eplb)
    moe.get_current_atom_config = lambda: current
    moe_config = SimpleNamespace(mxfp4_fp8_activations=False, a_quant_dtype=None,
                                 use_ep=ep)
    method = (moe.MegaMxfp4MoEMethod if mega else moe.Mxfp4MoEMethod)(quant, moe_config)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(n_routed_experts=384), tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_expert_parallel=ep, eplb_enable=eplb,
        moe_backend="mega" if mega else "standard")
    named = experts(config, arch)
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.quant_method = method
    try:
        _check_experts(model, named)
        refused = None
    except ValueError as exc:
        refused = str(exc)
    flags = [method.use_triton, method.use_triton_ep, method.use_triton_decode]
    out.append([named, *map(bool, flags), refused])
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def methods(tmp_path_factory):
    env = {k: v for k, v in os.environ.items() if k not in TRITON}
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            METHODS,
            str(tmp_path_factory.mktemp("d")),
            json.dumps(GRID),
        ],
        capture_output=True,
        check=False,
        text=True,
        env=env,
        cwd=pathlib.Path(__file__).resolve().parents[2],
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return dict(zip(GRID, json.loads(done.stdout.splitlines()[-1])))


@pytest.mark.parametrize("point", GRID, ids=lambda p: "-".join(map(str, p)))
def test_the_key_names_the_path_atom_s_method_takes(methods, point):
    named, triton, triton_ep, triton_decode, refused = methods[point]

    assert refused is None
    assert ("triton" in named) == (triton or triton_ep)
    assert ("+decode" in named) == triton_decode


def test_a4w4_is_named_on_the_triton_path(v4, monkeypatch):
    monkeypatch.setenv("ATOM_USE_TRITON_MOE", "1")
    monkeypatch.setenv("ATOM_USE_TRITON_MOE_A4W4", "1")

    assert experts(v4, "gfx950") == " experts=triton+a4w4"


def test_a_text_config_s_routed_experts_count(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_USE_TRITON_MOE", "1")
    text = SimpleNamespace(num_experts=512)
    vision = _config(tmp_path, "Qwen3_5MoeForConditionalGeneration", text_config=text)

    assert experts(vision, "gfx950") == " experts=triton"


def test_a_capture_refuses_experts_on_another_path_than_its_key_names():
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.quant_method = SimpleNamespace(
        use_triton=False, use_triton_ep=True, use_triton_decode=False
    )

    with pytest.raises(
        ValueError,
        match=r"at experts take .*\(True, False\); the op_graph key names aiter",
    ):
        _check_experts(model, "")
