# SPDX-License-Identifier: MIT
"""An `op_graph`'s key holds the engine config a step law is keyed by, less its
ladder: one store holds V4-Pro graphs of one shape that differ only in the KV
cache dtype, or only in ``ATOM_USE_TRITON_MOE`` or ``ATOM_USE_TRITON_MOE_DECODE``,
and each reader takes its own; a reader in another environment refuses naming
each variable that differs.
A step law is checked on the same fields, so one fitted on aiter's experts
refuses a Triton run; a config at ATOM's MoE defaults keys the fields an
existing law states.

No device: `torch.cuda` reports the arch as a capture's declared device does.
"""

import json
import pathlib
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import CONDITIONS, GATES, stanza
from test_backend_step_linear import view
from test_drive_runfile import build, install, step_linear
from test_op_graph_arch import ENGINE
from test_run_step_grid import deployment
from test_work import SPEC

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Key,
    Kind,
    Topology,
    member_name,
)
from atom.compass.backends import CostRefused
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import decode, structure
from atom.compass.run import _op_graphs
from atom.compass.run.step_grid import keyed

RANK = Topology().ranks()[0]
LAW = pathlib.Path(__file__).parents[2] / "scripts/compass/laws"
LAW /= "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json"
#: Each graph's config fields and environment off the first's.
VARIANTS = {
    "bf16": ({}, {}),
    "fp8": ({"kv_cache_dtype": "fp8"}, {}),
    "triton": ({}, {"ATOM_USE_TRITON_MOE": "1"}),
    "triton-decode": (
        {},
        {"ATOM_USE_TRITON_MOE": "1", "ATOM_USE_TRITON_MOE_DECODE": "1"},
    ),
}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in PERFORMANCE_KEYS:
        monkeypatch.delenv(name, raising=False)


def _v4(tmp_path, **fields):
    """DeepSeek-V4-Pro at DPA-TP8 EP8."""
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_expert_parallel=True,
        enforce_eager=True,
        **ENGINE | {"enable_dp_attention": True} | fields,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A `decode B=2` graph of each variant on gfx950, keyed as a capture keys
    it, its region naming the variant."""
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a, **k: driverless._props("gfx950")
    )
    store = ArtifactStore(str(tmp_path / "store"))
    for name, (fields, env) in VARIANTS.items():
        with monkeypatch.context() as m:
            for k, v in env.items():
                m.setenv(k, v)
            at = structure(_v4(tmp_path, **fields), "decode")
        store.publish(
            Key.of(Kind.OP_GRAPH, structure=at, shape="decode B=2"),
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=CONDITIONS,
            gates=GATES,
            members={
                member_name("graph", RANK, "json"): json.dumps(
                    {"region": {"op": name}}
                ).encode()
            },
        )
    return store


@pytest.mark.parametrize("name", VARIANTS)
def test_each_reader_takes_the_graph_of_its_own_engine(
    store, tmp_path, monkeypatch, name
):
    fields, env = VARIANTS[name]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config = _v4(tmp_path, **fields)

    assert _op_graphs(store.root, config, "gfx950")("decode", 2, 2) == {"op": name}
    assert ops.graph(store, config, decode(2))["region"] == {"op": name}


@pytest.mark.parametrize(
    "fitted, here, env, named",
    [
        (
            {},
            {},
            {"ATOM_USE_TRITON_MOE": "1"},
            "env ATOM_USE_TRITON_MOE None in the law and '1' here",
        ),
        (
            {"moe_backend": "mega"},
            {},
            {},
            "moe_backend 'mega' in the law and None here",
        ),
        ({}, {"eplb_enable": True}, {}, "eplb_enable None in the law and True here"),
    ],
    ids=["triton-run", "mega-law", "eplb-run"],
)
def test_a_step_law_refuses_a_run_on_other_experts_naming_the_field(
    monkeypatch, tmp_path, fitted, here, env, named
):
    entry = step_linear(tmp_path, deployment(tmp_path, **fitted))
    run = build(tmp_path, "--step-linear", entry.key.token())
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, run, deployment(tmp_path, **here))
    assert named in str(e.value)


def test_a_step_law_fitted_with_bounded_replay_refuses_a_run_without_it(
    monkeypatch, tmp_path
):
    on = deployment(tmp_path, enable_decoder_swa_bounded_replay=True)
    run = build(tmp_path, "--step-linear", step_linear(tmp_path, on).key.token())
    off = deployment(tmp_path, enable_decoder_swa_bounded_replay=False)

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, run, off)
    assert "enable_decoder_swa_bounded_replay None in the law and False here" in str(
        e.value
    )


def test_atom_s_moe_defaults_key_the_fields_an_existing_law_states(tmp_path):
    law = json.loads(LAW.read_text())["engine_config"]
    default = keyed(deployment(tmp_path, moe_backend="standard").config)
    moved = keyed(deployment(tmp_path, moe_backend="mega", eplb_enable=True).config)

    assert default.keys() == law.keys()
    assert {k: v for k, v in moved.items() if k not in law} == {
        "moe_backend": "mega",
        "eplb_enable": True,
    }


def test_a_graph_traced_in_another_environment_is_refused_naming_the_variable(
    store, tmp_path, monkeypatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    config = _v4(tmp_path)
    lookup = _op_graphs(store.root, config, "gfx950")
    named = (
        "a graph the store holds at this shape and structure differs in engine "
        "config: env OMP_NUM_THREADS None in the graph and '8' here"
    )

    with pytest.raises(CostRefused) as run:
        OpGraphBackend(SPEC, lookup, dp=8, kv="bf16").price(view(*[(1, 9, True)] * 2))
    with pytest.raises(ArtifactRefusal) as bench:
        ops.graph(store, config, decode(2))
    assert lookup("decode", 3, 3) is None
    for refused in (run, bench):
        assert named in str(refused.value)
        assert "kv_cache_dtype 'fp8' in the graph and 'bf16' here" in str(refused.value)
