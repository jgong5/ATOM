# SPDX-License-Identifier: MIT
"""V4.1's fixed-order router launches through a compiled function, and `warm`
compiles every width once, so no eager forward traces the kernel (#3743).
CPU only: the launcher and its compile are stand-ins."""

import pytest
import torch

router = pytest.importorskip("atom.model_ops.deepseek_v41.router")


def test_router_logits_goes_through_run_compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "_build", lambda tokens: ("launch", tokens))
    monkeypatch.setattr(
        router, "_run_compiled", lambda exe, *a: calls.append((exe, len(a)))
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    x = torch.zeros((3, router.HIDDEN), dtype=torch.bfloat16)
    w = torch.zeros((router.EXPERTS, router.HIDDEN), dtype=torch.bfloat16)
    out = router.router_logits(x, w)
    # x, w, out and the stream, through the compiled function: never the jit call
    assert calls == [(("launch", 3), 4)]
    assert out.shape == (3, router.EXPERTS)


def test_warm_compiles_each_width_once_and_nothing_off_a_gpu(monkeypatch):
    widths = []
    monkeypatch.setattr(router, "router_supported", lambda x, w: True)
    monkeypatch.setattr(router, "router_logits", lambda x, w: widths.append(x.shape[0]))
    router.warm(torch.zeros((router.EXPERTS, router.HIDDEN), dtype=torch.bfloat16))
    assert widths == []  # a CPU weight: no GPU, nothing to compile

    class W:  # a weight that says it is on a GPU
        device = torch.device("cuda", 0)

    empty = torch.empty
    monkeypatch.setattr(
        torch, "zeros", lambda shape, **k: empty(shape, dtype=k["dtype"])
    )
    router.warm(W())
    assert widths == list(range(1, router.MAX_TOKENS + 1))
