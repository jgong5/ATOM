# SPDX-License-Identifier: MIT
"""Whether a DP-attention MoE fuses its shared expert into aiter's top-k table
changes its `op_graph`: fused, the fused-MoE leaf takes the shared expert's id
in each row; unfused, the shared MLP runs beside it. ATOM decides it from
``ATOM_FUSE_SHARED_EXPERT``, ``ATOM_DISABLE_MORI_EP``, whether mori imports and
the all-to-all backend (`model_ops/topK.py`). Each changed alone where it flips
ATOM's decision keys the graph apart, and a run, or `measure ops`, under one
refuses a graph traced under the other, naming it; a server at ATOM's
defaults keys as before.

No device: aiter's arch probe answers a declared arch.
"""

import importlib.util
import pathlib
from importlib.machinery import ModuleSpec

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_key_index_cache_dtype import served

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import capture
from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal, ArtifactStore
from atom.compass.bench import ops
from atom.compass.capture.forward import decode, structure
from atom.compass.capture.graph import Traced
from atom.compass.ir import Graph, Op
from atom.compass.run.step_grid import MORI, READ_OFF_DEFAULT, keyed

V3 = pathlib.Path(__file__).with_name("deepseek_v3_config.json")
ARCH = "gfx950"
#: Eager, the mode a run reads its graphs in, so `measure ops` reads the same.
DPA_TP8 = ("-tp", "8", "--kv_cache_dtype", "fp8", "--enable-dp-attention")
DPA_TP8 += ("--enforce-eager",)
FUSE = "ATOM_FUSE_SHARED_EXPERT"
FIND_SPEC = importlib.util.find_spec
#: Each input changed alone from a DP-attention server with mori and
#: `ATOM_FUSE_SHARED_EXPERT=0`, whose MoE ATOM does not fuse: ATOM's arguments,
#: the environment (None unsets a variable), whether mori imports, and the
#: field a refusal names with its value there and here.
CHANGES = {
    "fuse-default": ((), {FUSE: None}, True, (f"env {FUSE}", "0", None)),
    "mori-ep-off": (
        (),
        {"ATOM_DISABLE_MORI_EP": "1"},
        True,
        ("env ATOM_DISABLE_MORI_EP", None, "1"),
    ),
    "no-mori": ((), {}, False, (MORI, None, False)),
    "rccl": (
        ("--all2all-backend", "rccl"),
        {},
        True,
        ("moe_all2all_backend", None, "rccl"),
    ),
    "none": (
        ("--all2all-backend", "none"),
        {},
        True,
        ("moe_all2all_backend", None, "none"),
    ),
}


def server(tmp_path, monkeypatch, args=(), env=None, mori=True):
    """A DP-attention V3 server's config on `ARCH`, ATOM's own fusion switch
    at 0, with ATOM's arguments `args`, the variables `env` set or unset, and
    mori importing or not (`mori`), each held while the test reads the key."""
    for name in (*PERFORMANCE_KEYS, *READ_OFF_DEFAULT):
        monkeypatch.delenv(name, raising=False)
    for name, value in ({FUSE: "0"} | (env or {})).items():
        if value is not None:
            monkeypatch.setenv(name, value)
    spec = ModuleSpec("mori", None) if mori else None
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: spec if name == "mori" else FIND_SPEC(name, *a, **k),
    )
    return served(tmp_path, monkeypatch, ARCH, V3, *DPA_TP8, *args)


def fuses(monkeypatch, config) -> bool:
    """Whether ATOM fuses the shared expert of `config`'s first MoE layer, as
    `FusedMoE` asks it."""
    from aiter.ops.triton.utils._triton import arch_info

    # No device to probe at import; nothing here launches a kernel.
    monkeypatch.setattr(arch_info, "_CACHED_ARCH", arch_info._CACHED_ARCH or ARCH)
    from atom.model_ops.topK import (
        is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config as fusion,
    )
    from atom.model_ops.utils import _has_module

    monkeypatch.setattr("atom.config._current_atom_config", config)
    layer = "model.layers.3.mlp"
    # ATOM probes whether mori imports once a process: here at each call.
    _has_module.cache_clear()
    try:
        return fusion(
            config.quant_config, f"{layer}.shared_experts", f"{layer}.experts"
        )
    finally:
        _has_module.cache_clear()


def published(store, config, leaf):
    """`config`'s decode B=4 graph of one leaf `leaf`, published at the key
    `trace ops` gives it in this process."""
    at = structure(config, "decode", True, ARCH)
    graph = Graph(Traced(at, "decode B=4"), Op(leaf, "opaque_leaf", (), ()))
    model = compass_run._file_digest(config.model, "config.json")
    return capture.publish(store, graph, model, {"arch": ARCH}, 8, [], "trace ops")


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES)
def test_each_input_that_flips_atom_s_fusion_keys_the_graph_apart(
    tmp_path, monkeypatch, change
):
    args, env, mori, _ = change
    base = server(tmp_path, monkeypatch)
    unfused, at = fuses(monkeypatch, base), structure(base, "decode", device=ARCH)
    changed = server(tmp_path, monkeypatch, args, env, mori)

    assert (unfused, fuses(monkeypatch, changed)) == (False, True)
    assert structure(changed, "decode", device=ARCH) != at


@pytest.mark.parametrize("traced", ["base", "changed"])
@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES)
def test_a_reader_under_one_refuses_a_graph_traced_under_the_other_naming_it(
    tmp_path, monkeypatch, change, traced
):
    args, env, mori, (field, *values) = change
    sides = {"base": ((), None, True), "changed": (args, env, mori)}
    [read] = sides.keys() - {traced}
    store = ArtifactStore(tmp_path / "store")
    published(store, server(tmp_path, monkeypatch, *sides[traced]), "aiter.traced")
    reader = server(tmp_path, monkeypatch, *sides[read])
    there, here = values if traced == "base" else values[::-1]
    named = f"{field} {there!r} in the graph and {here!r} here"

    with pytest.raises(ArtifactRefusal) as bench:
        ops.graph(store, reader, decode(4), ARCH)
    assert named in str(bench.value)
    assert named in compass_run._op_graphs(str(store.root), reader, ARCH)(
        "decode", 4, 4
    )


def test_a_server_at_atom_s_defaults_keys_as_before(tmp_path, monkeypatch):
    unset = keyed(server(tmp_path, monkeypatch, env={FUSE: None}))
    spelled = server(
        tmp_path, monkeypatch, env={FUSE: "1", "ATOM_DISABLE_MORI_EP": "0"}
    )

    assert unset["env"] == {}
    assert not {MORI, "moe_all2all_backend"} & unset.keys()
    assert keyed(spelled) == unset
