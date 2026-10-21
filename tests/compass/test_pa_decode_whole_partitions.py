# SPDX-License-Identifier: MIT
"""run_pa_decode hands gluon's PS kernel a window no context reaches, so the
kernel splits each context on partition edges (#1858). No device: gluon is
replaced by a recorder and every tensor is on the meta device."""

import importlib.abc
import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


class _AiterStandIn(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Every `aiter` module as a stand-in: base_attention imports aiter at
    module scope, and without a driver aiter's import raises."""

    def find_spec(self, name, path, target=None):
        if name == "aiter" or name.startswith("aiter."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda _attr: MagicMock()
        return module

    def exec_module(self, module):
        pass


if torch.cuda.is_available():
    from atom.model_ops import base_attention as ba
else:
    _finder = _AiterStandIn()
    sys.meta_path.insert(0, _finder)
    try:
        from atom.model_ops import base_attention as ba
    finally:
        sys.meta_path.remove(_finder)
        for _name in [n for n in sys.modules if n.split(".")[0] == "aiter"]:
            del sys.modules[_name]


def _window(monkeypatch, *, block=16, sliding_window=-1, ps=True, splits=8):
    """The sliding_window gluon receives for a batch-3 bf16 dense call."""
    monkeypatch.setattr(ba.envs, "ATOM_PA_FLYDSL", False)
    seen = {}
    monkeypatch.setattr(
        torch.ops.aiter,
        "pa_decode_gluon",
        lambda *a, **kw: seen.update(kw),
        raising=False,
    )
    meta = lambda *shape: torch.empty(shape, device="meta")
    ba.run_pa_decode(
        output=meta(3, 24, 256),
        q=meta(3, 24, 256),
        k_cache=meta(64, 4, 32, block, 8),
        v_cache=meta(64, 4, block // 8, 256, 8),
        context_lens=meta(3),
        block_tables=meta(3, 16384),
        softmax_scale=1.0,
        max_seqlen_q=1,
        max_context_partition_num=splits,
        context_partition_size=256,
        compute_type=torch.bfloat16,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        exp_sums=meta(3, 4, splits, 6),
        max_logits=meta(3, 4, splits, 6),
        temporary_output=meta(3, 4, splits, 6, 256),
        sliding_window=sliding_window,
        ps=ps,
    )
    return seen["sliding_window"]


@pytest.mark.parametrize("off", [-1, 0])
def test_an_unwindowed_call_gets_a_window_past_any_context(monkeypatch, off):
    # The block table above addresses 16384 * 16 tokens.
    assert _window(monkeypatch, sliding_window=off) > 16384 * 16


def test_a_real_window_is_passed_through(monkeypatch):
    assert _window(monkeypatch, sliding_window=128) == 128


@pytest.mark.parametrize("kw", [{"block": 1024}, {"ps": False}, {"splits": 1}])
def test_one_split_large_blocks_and_non_ps_keep_no_window(monkeypatch, kw):
    assert _window(monkeypatch, **kw) == -1
