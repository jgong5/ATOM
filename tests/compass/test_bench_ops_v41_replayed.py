# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's leaves of a replayed prefill under `compass measure ops`:
layer 39's query rotation and its output's inverse take the positions its
decoder plan gathers for its queries, and its DSpark drafter's draft-KV tail
the step's at the tail's rows; with no plan they refuse by name.
"""

import pytest
import torch
from test_bench_ops import _graph, _leaf, _Runner, _t

from atom.compass.bench import ops
from atom.utils import forward_context as fctx

ROPE = "atom.model_ops.deepseek_v41.rope_window:_launch_rope_window"
INVERSE = "aiter._inverse_rope_group_quant_kernel.default"
TAIL = "atom.model_ops.deepseek_v41.dspark:fused_draft_kv_tail"


@pytest.mark.parametrize("published", [True, False], ids=["replayed", "no_plan"])
def test_a_replayed_prefill_s_leaves_take_the_positions_its_plan_gathers(
    monkeypatch, published
):
    # one request's 6 rows at positions 100 to 105, finishing beside a padding
    # request, a window of 4: the tail's rows 2 to 5, then 5 again, its own
    # positions 102 to 105, then 0; its queries its rows 3 and 7
    from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
    from atom.models.deepseek_v41.bounded_replay import ReplayPlan, lay_out_tail
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec

    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner, prepare = _Runner(), _Runner.prepare_model
    runner.model.append(torch.nn.Module())
    runner.model[-1].spec = LayerAttentionSpec(0, 0, AttentionMode.FULL)
    layout = lay_out_tail([(RequestSpan(0, 100, 0, 6, 0), 4, 0)], 2, 8)
    rows = ("rows", "query_ids", "last", "query_ptrs", "last_rows")
    zeros = (torch.zeros(8, dtype=torch.int32),) * 3
    tail = BatchStep((), torch.tensor([102, 103, 104, 105, 0, 0, 0, 0]), *zeros, None)
    plan = [torch.from_numpy(getattr(layout, r)).int() for r in rows]
    plan = ReplayPlan(tail, *plan, layout.filled, None, None) if published else None
    step = BatchStep((), torch.arange(100, 106), *zeros, None, decoder=plan)

    def prepare_model(batch):
        out, fc = prepare(runner, batch), fctx._forward_context
        vars(fc.attn_metadata).update(step=step)
        vars(fc.context).update(positions=step.positions, running_tokens=6)
        return out

    def leaf(op, rows):
        return _leaf(op, ("positions", _t([rows], "activation", dtype="torch.int64")))

    leaves = leaf(ROPE, 2), leaf(INVERSE, 2), leaf(TAIL, 8)
    found = ops.leaves(_graph(*leaves, layers=1)["region"])
    for op, _ in found.values():
        op["drafted"] = op["op"] == TAIL
    runner.prepare_model, stub = prepare_model, ops.Stub(runner, 10_000)
    stub.leaves = found
    stub._prepare((0, 0))

    [(_, bound)] = stub.placed
    got = {
        found[sig][0]["op"]: (name, view if view is None else view.tolist())
        for (sig, _), (name, view) in bound.items()
    }
    queries = "plan.queries.positions", [105, 0]
    drafted = "plan.tail_rows.positions", [102, 103, 104, 105, 105, 105, 105, 105]
    want = {ROPE: queries, INVERSE: queries, TAIL: drafted}
    why = "no replayed prefill here gathers its positions", None
    assert got == (want if published else dict.fromkeys(want, why))
