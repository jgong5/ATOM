# SPDX-License-Identifier: MIT
"""`prewarm.prewarm_native_a8` first-calls each native A8 GEMM variant once
per weight shape at each tuned M up to the warmup's budget (#3743). CPU only:
the GEMM and the device are stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops import prewarm as blockscale


class Weight(SimpleNamespace):
    pass


def _linear(n, k, rows=32, shuffled=True):
    w = Weight(
        shape=(n, k),
        dtype=torch.float8_e4m3fn,
        is_shuffled=shuffled,
        device=SimpleNamespace(type="cuda"),
    )
    return torch.nn.Module(), w, SimpleNamespace(shape=(n // rows, k // 32)), rows


def _model(*specs):
    model = torch.nn.Module()
    for i, (n, k, rows, shuffled) in enumerate(specs):
        m, w, s, r = _linear(n, k, rows, shuffled)
        m.native_a8_group_rows, m.weight, m.weight_scale = r, w, s
        model.add_module(f"l{i}", m)
    model.add_module("plain", torch.nn.Linear(4, 4))  # not a native A8 linear
    return model


@pytest.fixture
def calls(monkeypatch):
    made = []
    zeros = lambda shape, **k: SimpleNamespace(shape=shape)
    monkeypatch.setattr(torch, "zeros", zeros)

    def gemm(x, w, s, **k):
        made.append((w.shape, x.shape[0], k["weight_preshuffled"]))

    monkeypatch.setattr(blockscale, "_gemm", lambda: gemm)
    return made


def test_each_shape_once_at_each_tuned_m_up_to_the_budget(monkeypatch, calls):
    tuned = {(512, 5120): (1, 16, 64, 2048, 4096, 32768)}
    monkeypatch.setattr(blockscale, "_tuned_ms", lambda n, k, s: tuned.get((n, k), ()))
    model = _model((512, 5120, 32, True), (512, 5120, 32, True), (5120, 2048, 32, True))
    assert blockscale.prewarm_native_a8(model, 3000) == len(calls)
    tuned_shape = [m for s, m, _ in calls if s == (512, 5120)]
    assert tuned_shape == [1, 16, 64, 2048, 4096]  # once, and the row 3000 pads to
    untuned = [m for s, m, _ in calls if s == (5120, 2048)]
    want = [m for m in blockscale.PREWARM_M if m <= 3000] + [4096]
    assert untuned == want


def test_an_unshuffled_weight_takes_the_ladder(monkeypatch, calls):
    monkeypatch.setattr(blockscale, "_tuned_ms", lambda n, k, s: (1, 2))
    blockscale.prewarm_native_a8(_model((512, 5120, 32, False)), 8)
    assert [m for _, m, _ in calls] == [1, 2, 4, 8, 16]
    assert {p for *_, p in calls} == {False}
