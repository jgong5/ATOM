# SPDX-License-Identifier: MIT
"""COMPASS_MOE_KEYLOG=1 (#3720): the first aiter MoE tuning lookup of each key, logged with its caller."""

import json
import sys
import types

import torch

from atom.compass.parity import moe_keys


def _fake_aiter(monkeypatch):
    calls = []

    def get_2stage_cfgs(token, model_dim, *rest):
        calls.append(token)
        return ("cfg", token)

    mod = types.ModuleType("aiter.fused_moe")
    mod.get_2stage_cfgs = get_2stage_cfgs
    pkg = types.ModuleType("aiter")
    pkg.fused_moe = mod
    monkeypatch.setitem(sys.modules, "aiter", pkg)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", mod)
    return mod, calls


class Drafter(torch.nn.Module):
    def forward(self, hidden_states, lookup):
        token_num = hidden_states.shape[0]  # noqa: F841  as aiter's fused_moe holds it
        return lookup(256, 6144)


def test_off_leaves_the_lookup_alone(monkeypatch, tmp_path):
    monkeypatch.delenv(moe_keys.ENV, raising=False)
    mod, _ = _fake_aiter(monkeypatch)
    inner = mod.get_2stage_cfgs
    assert not moe_keys.install(types.SimpleNamespace(), tmp_path)
    assert mod.get_2stage_cfgs is inner and not (tmp_path / "moe_keys").exists()


def test_the_first_lookup_of_each_key_is_logged_with_its_caller(monkeypatch, tmp_path):
    monkeypatch.setenv(moe_keys.ENV, "1")
    mod, calls = _fake_aiter(monkeypatch)
    runner = types.SimpleNamespace(_parity_record=types.SimpleNamespace(step=41))
    assert moe_keys.install(runner, tmp_path)
    assert moe_keys.install(runner, tmp_path)  # once
    out = [Drafter()(torch.zeros(183, 4), mod.get_2stage_cfgs) for _ in range(3)]
    assert out == [("cfg", 256)] * 3 and calls == [256, 256, 256]  # the lookup still answers every call
    (log,) = (tmp_path / "moe_keys").glob("*.jsonl")
    (line,) = [json.loads(x) for x in log.read_text().splitlines()]
    assert line["padded_m"] == 256 and line["tokens"] == 183 and line["step"] == 41
    assert line["modules"][0] == "Drafter" and line["key"] == ["256", "6144"]
