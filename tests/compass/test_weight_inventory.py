"""The weight term counts what the engine builds, not what the file ships.

`weight_bytes` sums every safetensors header entry. A checkpoint may ship
tensors for a module the configured model class never constructs -- on the
27B, the ``mtp.*`` draft block -- and summing the headers counts them into the
memory budget, where they cost KV blocks that are never recovered.

These tests pin the mechanism rather than the 27B's number: a synthetic
checkpoint with one built tensor and one unbuilt one, and a model that builds
only the first. No device, no real checkpoint, no engine.
"""

import json
import os
import struct
import sys
import types

import pytest
import torch

from atom.compass.core import memory_model
from atom.compass.core.memory_model import (
    built_parameter_bytes, rank_inventory, resident_bytes, weight_bytes)


def _write_shard(path, tensors):
    """One safetensors file with `tensors` as {name: (dtype, shape)}."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for s in shape:
            n *= s
        size = n * (2 if dtype in ("F16", "BF16") else 4)
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)
    return offset


def _checkpoint(tmp_path, tied=False):
    built = _write_shard(str(tmp_path / "model-00001-of-00002.safetensors"),
                         {"model.layers.0.weight": ("BF16", (64, 32))})
    unbuilt = _write_shard(str(tmp_path / "model-00002-of-00002.safetensors"),
                           {"mtp.fc.weight": ("BF16", (16, 32))})
    with open(os.path.join(str(tmp_path), "config.json"), "w") as fh:
        json.dump({"tie_word_embeddings": tied}, fh)
    # A sharded checkpoint is read through its index, as the real one is.
    with open(os.path.join(str(tmp_path),
                           "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": built + unbuilt},
                   "weight_map": {
                       "model.layers.0.weight":
                           "model-00001-of-00002.safetensors",
                       "mtp.fc.weight":
                           "model-00002-of-00002.safetensors"}}, fh)
    return built, unbuilt


class _BuiltModel(torch.nn.Module):
    """Only the tensor the checkpoint's first shard carries."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(64, 32, dtype=torch.bfloat16, device="meta"))


def test_headers_count_a_module_the_model_does_not_build(tmp_path):
    built, unbuilt = _checkpoint(tmp_path)
    headers = weight_bytes(str(tmp_path), 1)
    assert headers == built + unbuilt

    parameters, _ = resident_bytes(_BuiltModel())
    assert parameters == built

    # The whole of the difference is the unbuilt module, and nothing else:
    # no padding, no alias, no sharding rule is involved at TP=1.
    assert headers - parameters == unbuilt


def test_the_two_routes_agree_when_everything_is_built(tmp_path):
    built = _write_shard(str(tmp_path / "model.safetensors"),
                         {"model.layers.0.weight": ("BF16", (64, 32))})
    with open(os.path.join(str(tmp_path), "config.json"), "w") as fh:
        json.dump({"tie_word_embeddings": False}, fh)

    parameters, _ = resident_bytes(_BuiltModel())
    assert weight_bytes(str(tmp_path), 1) == built == parameters


# --------------------------------------------------------------------------
# The build helper itself: the group it runs in, and what it does when the
# build cannot happen. Stubs, not the engine -- the point is the sequencing,
# and importing the real `model_runner` would pull in AITER and a device.


class _HFConfig:
    architectures = ["Stub"]
    tie_word_embeddings = False
    num_hidden_layers = 2
    hidden_size = 8

    def to_dict(self):
        return {"architectures": list(self.architectures)}


class _Config:
    torch_dtype = torch.bfloat16

    def __init__(self, model, tensor_parallel_size=1):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.hf_config = _HFConfig()


def _install_stub_engine(monkeypatch, calls, build=None, config_cls=_Config):
    """Everything `built_parameter_bytes` imports, replaced by a recorder."""
    monkeypatch.setattr(memory_model, "_ensure_meta_group",
                        lambda target=None: calls.append(("group",)))
    monkeypatch.setattr(memory_model, "_restore_simulated_group",
                        lambda: calls.append(("restore",)))

    derive = types.ModuleType("atom.compass.runtime.derive")
    derive.simulate_group_width = (
        lambda logical, physical=1, rank=0:
        calls.append(("patch", logical, rank)))
    monkeypatch.setitem(sys.modules, "atom.compass.runtime.derive", derive)

    config_mod = types.ModuleType("atom.config")
    config_mod.Config = config_cls
    config_mod.set_current_atom_config = lambda config: None
    monkeypatch.setitem(sys.modules, "atom.config", config_mod)

    runner = types.ModuleType("atom.model_engine.model_runner")
    runner.support_model_arch_dict = {"Stub": "stub.Model"}
    monkeypatch.setitem(sys.modules, "atom.model_engine.model_runner", runner)

    class _Stub(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            width = max(1, config.tensor_parallel_size)
            self.weight = torch.nn.Parameter(
                torch.empty(64 // width, 32, dtype=torch.bfloat16))

    utils = types.ModuleType("atom.utils")
    utils.resolve_obj_by_qualname = lambda qualname: build or _Stub
    monkeypatch.setitem(sys.modules, "atom.utils", utils)


def test_every_build_restores_the_group_before_patching_it(monkeypatch):
    # `simulate_group_width` restores only on the path where it patches, and
    # width 1 returns before that. So a process that builds TP=2 and then TP=1
    # would build the second against a group still reporting two ranks, and
    # record a TP2 shape under width 1 without raising. The restore has to be
    # unconditional, and it has to come first.
    calls = []
    _install_stub_engine(monkeypatch, calls)

    for width in (1, 2, 4, 2, 1):
        assert built_parameter_bytes("/ckpt", width) is not None

    patches = [c for c in calls if c[0] == "patch"]
    assert [c[1] for c in patches] == [1, 2, 4, 2, 1]
    for index, call in enumerate(calls):
        if call[0] == "patch":
            assert calls[index - 1] == ("restore",), (
                "the patch at %d was not preceded by a restore" % index)


def test_the_widths_do_not_contaminate_each_other(monkeypatch):
    # The observable consequence of the above, in the emitter's own terms: the
    # answer for a width must not depend on which widths were built before it.
    ascending = {}
    for width in (1, 2, 4):
        calls = []
        _install_stub_engine(monkeypatch, calls)
        ascending[width] = built_parameter_bytes("/ckpt", width)

    calls = []
    _install_stub_engine(monkeypatch, calls)
    descending = {width: built_parameter_bytes("/ckpt", width)
                  for width in (4, 2, 1)}

    assert descending == ascending


def test_a_build_that_cannot_happen_falls_back_and_says_why(monkeypatch):
    # The docstring promises None rather than an exception, and the group
    # setup has to be inside that promise: a caller with a header figure to
    # fall back to should not be taken down by a group that will not
    # initialise. A silent fallback is how a set ends up on the worse number.
    class _Explodes(_Config):
        def __init__(self, model, tensor_parallel_size=1):
            raise RuntimeError("no config here")

    calls, identity = [], {}
    _install_stub_engine(monkeypatch, calls, config_cls=_Explodes)
    assert built_parameter_bytes("/ckpt", 2, identity=identity) is None
    assert "no config here" in identity["error"]

    def _explode(target=None):
        raise RuntimeError("no process group")

    calls, identity = [], {}
    _install_stub_engine(monkeypatch, calls)
    monkeypatch.setattr(memory_model, "_ensure_meta_group", _explode)
    assert built_parameter_bytes("/ckpt", 2, identity=identity) is None
    assert "no process group" in identity["error"]


def test_the_identity_records_what_was_resolved_not_what_was_asked(monkeypatch):
    # A hub name has no revision in it. If the build resolved a different
    # snapshot than the headers were read from, the profile's own provenance
    # has to be able to show it.
    calls, identity = [], {}
    _install_stub_engine(monkeypatch, calls)
    built_parameter_bytes("/ckpt", 1, identity=identity)
    assert identity["requested"] == "/ckpt"
    assert identity["architectures"] == ["Stub"]


def test_rank_inventory_asks_every_rank_and_takes_the_largest(monkeypatch):
    calls = []
    _install_stub_engine(monkeypatch, calls)

    report = rank_inventory("/ckpt", 4)
    assert sorted(report["ranks"]) == [0, 1, 2, 3]
    assert report["uniform"] is True
    assert report["spread"] == 0
    assert report["parameters"] == 64 // 4 * 32 * 2

    # The whole reason to ask every rank: one that is not like the others.
    rank_seen = {"rank": 0}

    def _patch(logical, physical=1, rank=0):
        rank_seen["rank"] = rank

    sys.modules["atom.compass.runtime.derive"].simulate_group_width = _patch

    class _Uneven(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            extra = 16 if rank_seen["rank"] == 0 else 0
            self.weight = torch.nn.Parameter(
                torch.empty(16 + extra, 32, dtype=torch.bfloat16))

    sys.modules["atom.utils"].resolve_obj_by_qualname = lambda q: _Uneven
    report = rank_inventory("/ckpt", 4)
    assert report["uniform"] is False
    assert report["spread"] == 16 * 32 * 2
    # The budget has to hold for the rank that carries the most.
    assert report["parameters"] == 32 * 32 * 2


@pytest.mark.parametrize("width", (1, 2, 4))
def test_the_build_restores_the_default_dtype(monkeypatch, width):
    calls = []
    _install_stub_engine(monkeypatch, calls)
    before = torch.get_default_dtype()
    parameters, _ = built_parameter_bytes("/ckpt", width)
    # dtype is restored even though the build set the config's
    assert torch.get_default_dtype() is before
    assert parameters == 64 // width * 32 * 2
