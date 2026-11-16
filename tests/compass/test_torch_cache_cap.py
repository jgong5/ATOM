# SPDX-License-Identifier: MIT
"""The torch cache cap a runner sets once its graph capture ends.

The cap (`atom/model_engine/torch_cache_cap.py`) is read off a stand-in for
`torch.cuda` holding planted readings, so these run with no device. Where the
runner applies it is read from `model_runner.py`'s source, as
`test_runner_rpc_surface.py` reads it: importing that module runs aiter's
architecture probe, which raises where there is no GPU.
"""

import ast
import logging
import pathlib
from types import SimpleNamespace

import pytest

from atom.model_engine.torch_cache_cap import (
    CARD_MARGIN,
    GIB,
    TorchCacheCap,
    cap_torch_cache,
    torch_cache_limit,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
ENGINE = REPO / "atom" / "model_engine"
# DeepSeek-V4.1-Flash DEP2's Post-init memory on MI355X: torch reserved
# 238.19 GB with 34.54 GB free on a 287.98 GB card.
CARD = int(287.98 * GIB)
HELD = int(238.19 * GIB)
FREE = int(34.54 * GIB)


class Cuda:
    """`torch.cuda` as the cap reads it, from planted readings."""

    def __init__(self, free=FREE, held=HELD, card=CARD):
        self.free, self.held, self.card = free, held, card
        self.retries, self.stat_reads = 0, 0
        self.fractions = []

    def mem_get_info(self, device):
        return self.free, self.card

    def memory_reserved(self, device):
        return self.held

    def get_device_properties(self, device):
        return SimpleNamespace(total_memory=self.card)

    def set_per_process_memory_fraction(self, fraction, device):
        self.fractions.append((fraction, device))

    def memory_stats(self, device):
        self.stat_reads += 1
        return {"num_alloc_retries": self.retries}


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_the_cap_is_what_torch_holds_plus_what_is_free_less_the_margin(
    monkeypatch,
):
    monkeypatch.delenv("ATOM_TORCH_CACHE_CAP", raising=False)
    cuda = Cuda()
    cap = cap_torch_cache("cuda:1", "runner", cuda=cuda)
    want = HELD + FREE - int(CARD * CARD_MARGIN)
    assert cap.limit == want
    ((fraction, device),) = cuda.fractions
    assert device == "cuda:1"
    # torch multiplies the fraction back by the card's total.
    assert abs(int(fraction * CARD) - want) <= 1
    # 266.97 GB of torch on this card: 2% of it stays outside torch.
    assert round(want / GIB, 2) == 266.97


def test_a_card_already_inside_the_margin_keeps_what_torch_holds():
    assert torch_cache_limit(HELD, 1 << 30, CARD, CARD_MARGIN) == HELD


@pytest.mark.parametrize("off", ["0", "false", "False"])
def test_ATOM_TORCH_CACHE_CAP_0_leaves_the_allocator_uncapped(monkeypatch, off):
    monkeypatch.setenv("ATOM_TORCH_CACHE_CAP", off)
    cuda = Cuda()
    assert cap_torch_cache("cuda:0", "runner", cuda=cuda) is None
    assert cuda.fractions == []


def test_the_first_free_under_the_cap_is_logged_once(caplog):
    cuda, clock = Cuda(), Clock()
    cap = TorchCacheCap(cuda, "cuda:0", HELD, "runner", every_s=10.0, clock=clock)
    reads = cuda.stat_reads
    caplog.set_level(logging.INFO, logger="atom")

    cuda.retries = 1
    clock.now += 9.0
    cap.log_first_free()
    # Inside the interval a forward reads the clock and nothing else.
    assert cuda.stat_reads == reads and not cap.freed

    clock.now += 1.0
    cap.log_first_free()
    assert cap.freed
    freed = [r for r in caplog.records if "torch freed its cached blocks" in r.message]
    assert len(freed) == 1

    cuda.retries = 5
    for _ in range(3):
        clock.now += 10.0
        cap.log_first_free()
    assert len([r for r in caplog.records if "freed its cached" in r.message]) == 1


def test_checking_reads_the_count_and_never_sets_the_cap():
    cuda, clock = Cuda(), Clock()
    cap = TorchCacheCap(cuda, "cuda:0", HELD, "runner", every_s=10.0, clock=clock)
    for _ in range(50):
        clock.now += 11.0
        cap.log_first_free()
    assert cuda.stat_reads == 51
    assert cuda.fractions == []


def _method(cls, name):
    module = ast.parse((ENGINE / "model_runner.py").read_text())
    owner = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    return next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) == name
    ]


def test_the_cap_is_set_once_after_the_graph_pools():
    capture = _method("ModelRunner", "capture_cudagraph")
    *body, last = capture.body
    assert isinstance(last, ast.Return)
    # The statement before the return: nothing the runner allocates follows it.
    assert _calls(body[-1], "cap_torch_cache")
    assert ast.unparse(body[-1]).startswith("self.torch_cache_cap = ")
    assert len(_calls(capture, "cap_torch_cache")) == 1


def test_the_kv_pool_builder_never_caps():
    """Compass's capture tracer builds a runner's KV pool on a fake device
    (`atom/compass/capture/forward.py`), so a device call there breaks it."""
    assert not _calls(_method("ModelRunner", "allocate_kv_cache"), "cap_torch_cache")


def test_a_forward_logs_the_first_free_and_never_sets_the_cap():
    for cls in ("ModelRunner", "RapidServeModelRunner"):
        forward = _method(cls, "forward")
        assert _calls(forward, "log_first_free"), cls
        assert not _calls(forward, "cap_torch_cache"), cls
