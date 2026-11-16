# SPDX-License-Identifier: MIT
"""The torch cache cap a runner sets when its startup ends.

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
    GIB,
    TorchCacheCap,
    cap_torch_cache,
    torch_cache_limit,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
ENGINE = REPO / "atom" / "model_engine"
MARGIN = 0.02
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
    cap = cap_torch_cache("cuda:1", MARGIN, "runner", cuda)
    want = HELD + FREE - int(CARD * MARGIN)
    assert cap.limit == want
    ((fraction, device),) = cuda.fractions
    assert device == "cuda:1"
    # torch multiplies the fraction back by the card's total.
    assert abs(int(fraction * CARD) - want) <= 1
    # 266.97 GB of torch on this card: the card's 2% stays outside torch.
    assert round(want / GIB, 2) == 266.97


def test_a_card_already_inside_the_margin_keeps_what_torch_holds():
    assert torch_cache_limit(HELD, 1 << 30, CARD, MARGIN) == HELD


@pytest.mark.parametrize("off", ["0", "false", "False"])
def test_ATOM_TORCH_CACHE_CAP_0_leaves_the_allocator_uncapped(monkeypatch, off):
    monkeypatch.setenv("ATOM_TORCH_CACHE_CAP", off)
    cuda = Cuda()
    assert cap_torch_cache("cuda:0", MARGIN, "runner", cuda) is None
    assert cuda.fractions == []


def test_the_first_free_under_the_cap_is_logged_once(caplog):
    cuda, clock = Cuda(), Clock()
    cap = TorchCacheCap(cuda, "cuda:0", HELD, "runner", every_s=10.0, clock=clock)
    reads = cuda.stat_reads
    caplog.set_level(logging.INFO, logger="atom")

    cuda.retries = 1
    clock.now += 9.0
    cap.poll()
    # Inside the interval a forward reads the clock and nothing else.
    assert cuda.stat_reads == reads and not cap.freed

    clock.now += 1.0
    cap.poll()
    assert cap.freed
    freed = [r for r in caplog.records if "torch freed its cached blocks" in r.message]
    assert len(freed) == 1

    cuda.retries = 5
    for _ in range(3):
        clock.now += 10.0
        cap.poll()
    assert len([r for r in caplog.records if "freed its cached" in r.message]) == 1


def test_polling_reads_the_count_and_never_sets_the_cap():
    cuda, clock = Cuda(), Clock()
    cap = TorchCacheCap(cuda, "cuda:0", HELD, "runner", every_s=10.0, clock=clock)
    for _ in range(50):
        clock.now += 11.0
        cap.poll()
    assert cuda.stat_reads == 51
    assert cuda.fractions == []


def _method(path, cls, name):
    module = ast.parse(path.read_text())
    owner = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    return next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _calls(node, attr):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def _last_before_return(method):
    """The statement of `method`'s body just before its final return."""
    *body, last = method.body
    assert isinstance(last, ast.Return)
    return body[-1]


def test_a_graph_runner_caps_after_its_graph_pools():
    capture = _method(ENGINE / "model_runner.py", "ModelRunner", "capture_cudagraph")
    last = _last_before_return(capture)
    assert _calls(last, "_cap_torch_cache") and isinstance(last, ast.Expr)
    assert len(_calls(capture, "_cap_torch_cache")) == 1


def test_an_eager_runner_caps_after_its_kv_pool_and_only_eager():
    allocate = _method(ENGINE / "model_runner.py", "ModelRunner", "allocate_kv_cache")
    last = _last_before_return(allocate)
    assert isinstance(last, ast.If) and "enforce_eager" in ast.unparse(last.test)
    assert not last.orelse and _calls(last, "_cap_torch_cache")
    assert len(_calls(allocate, "_cap_torch_cache")) == 1
    pool = _calls(allocate, "_back_paged_pools")
    assert pool and all(c.lineno < last.lineno for c in pool)


def test_the_engine_captures_graphs_only_for_a_runner_that_did_not_cap_at_its_pool():
    # With the eager guard above, this makes each startup cap exactly once.
    tree = ast.parse((ENGINE / "engine_core.py").read_text())
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    sites = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "call_func"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "capture_cudagraph"
    ]
    assert sites
    for site in sites:
        guards, node = [], site
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                guards.append(ast.unparse(node.test))
        assert any("not config.enforce_eager" in g for g in guards), site.lineno


def test_a_forward_polls_the_cap_and_never_sets_it():
    for cls in ("ModelRunner", "RapidServeModelRunner"):
        forward = _method(ENGINE / "model_runner.py", cls, "forward")
        assert _calls(forward, "poll"), cls
        assert not _calls(forward, "_cap_torch_cache"), cls
