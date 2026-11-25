# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's DSpark draft block under `compass measure ops`: its
drafter's leaves take the anchors ATOM stages at the step's zero rejections,
and the positions and indices ATOM's `draft_step_indices` builds from them,
which Triton's interpreter runs here on the host.
"""

import functools
import os
import subprocess
import sys
from types import SimpleNamespace

import torch
from test_bench_ops import _graph, _leaf, _Runner, _t

from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.utils import forward_context as fctx

I32, I64 = "torch.int32", "torch.int64"
STAGE = "atom.spec_decode.dspark_proposer:DSparkProposer._stage_anchors"
BUILD = "atom.model_ops.deepseek_v41.dspark:draft_step_indices"
ROPE = "atom.model_ops.deepseek_v41.rope_window:_launch_rope_window"
ATTEND = "atom.model_ops.sparse_attn_v4:_sparse_attn_triton"
INVERSE = "aiter._inverse_rope_group_quant_kernel.default"


def _in_own_process(request) -> bool:
    """Whether this process runs the test: V4.1's DSpark kernels import aiter,
    which needs an arch declared, once a process, and run on the host under
    Triton's interpreter, which a process takes at its start."""
    if "COMPASS_ARCH_DECLARED" in os.environ:
        return True
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + [f"{__file__}::{request.node.name}"],
        env={**os.environ, "COMPASS_ARCH_DECLARED": "1", "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-4000:]
    return False


def test_the_draft_block_takes_what_atom_builds_from_the_step_s_anchors(
    monkeypatch, request, tmp_path
):
    # 2 rows of 3 tokens, an anchor and 2 drafts verified, at positions 5 to
    # 7 and 0 to 2: at zero rejections the anchors are rows 2 and 5, at
    # positions 7 and 2, and the block drafts 2 past each, over a ring of 5
    # slots and a window of 4; a RoPE of the target's takes none of it
    if not _in_own_process(request):
        return
    driverless.install(str(tmp_path), "gfx1250")
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from atom.spec_decode.drafter import Drafter

    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner, prepare = _Runner(), _Runner.prepare_model
    runner.model.append(torch.nn.Module())
    runner.model[-1].spec = LayerAttentionSpec(0, 0, AttentionMode.FULL)
    model = torch.nn.Module()
    model._context_layer_ids, model.window_size = None, 4
    runner.drafter = SimpleNamespace(model=model, draft_tokens_per_seq=2)
    runner.drafter.prepare_inputs = functools.partial(Drafter.prepare_inputs, None)
    cache = SimpleNamespace(geometry=SimpleNamespace(ring_slots=5))

    def prepare_model(batch):
        out, fc = prepare(runner, batch), fctx._forward_context
        cu = torch.tensor([0, 3, 6], dtype=torch.int32)
        vars(fc.attn_metadata).update(cache=cache, cu_seqlens_q=cu)
        positions = torch.tensor([5, 6, 7, 0, 1, 2])
        vars(fc.context).update(positions=positions, running_bs=2)
        return out

    rows, positions = _t([2], "activation", dtype=I32), _t([4], "activation", dtype=I64)
    found = ops.leaves(
        _graph(
            _leaf(STAGE, ("positions", _t([6], "metadata", dtype=I64)), ("rows", rows)),
            _leaf(BUILD, ("positions", _t([2], "metadata", dtype=I64)), ("ring", 5)),
            _leaf(ROPE, ("positions", positions)),
            _leaf(INVERSE, ("positions", positions)),
            _leaf(ATTEND, ("topk_idxs", _t([2, 2, 7], "activation", dtype=I32))),
            layers=1,
        )["region"]
    )
    for op, _ in found.values():
        op["drafted"] = True
    target = _leaf(ROPE, ("positions", _t([1], "activation", dtype=I64)))
    [mine] = (target := ops.leaves(_graph(target, layers=1)["region"]))
    runner.prepare_model, stub = prepare_model, ops.Stub(runner, 10_000)
    stub.leaves = found | target
    stub._prepare((0, 0))

    [(_, bound)] = stub.placed
    why = "no replayed prefill here gathers its positions"
    assert bound.pop((mine, "positions")) == (why, None)
    got = {
        (found[sig][0]["op"], path): (name, view.tolist())
        for (sig, path), (name, view) in bound.items()
        if (path, found[sig][0]["op"]) != ("positions", STAGE)
    }
    assert got == {
        (STAGE, "rows"): ("draft_step.rows", [2, 5]),
        (BUILD, "positions"): ("draft_step.anchors", [7, 2]),
        (ROPE, "positions"): ("draft_step.positions", [8, 9, 3, 4]),
        (INVERSE, "positions"): ("draft_step.positions", [8, 9, 3, 4]),
        (ATTEND, "topk_idxs"): (
            "draft_step.indices",
            [[[0, 1, 2, -1, 4, 5, 6]] * 2, [[0, 1, 2, -1, -1, 5, 6]] * 2],
        ),
    }
