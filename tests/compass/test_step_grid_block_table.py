# SPDX-License-Identifier: MIT
"""The step grid's top c nodes under a drafter, held to ATOM's block table.

A replayed row's KV blocks, the positions its draft round writes past it
included, go in a block table of ceil(max_model_len / block_size) columns, the
width `CommonAttentionBuilder` gives it; `block_table_state(...).prepare`
refuses a wider row, which on a bench worker ends `measure steps`. The plan
reads `test_step_grid_drafter`'s scheduler-built steps under DSpark5 and MTP1,
at max_model_len 2048 and 16-token blocks, 128 columns, plus a decode step
and a prefill chunk cached near max_model_len, so its top cells are planned.
"""

import pytest
import torch
from test_bench_replay_drafter import config
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_step_grid_drafter import DEFINITION, SPECULATIVE, recorded

from atom.compass.backends import step_grid_plan
from atom.compass.bench import replay
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state

COLUMNS = 2048 // 16


def table(rows):
    """ATOM's runner taking the block-table `rows` into a buffer of `COLUMNS`
    columns, as `prepare_block_tables` takes a batch's."""
    cpu = {"device": "cpu", "pin_memory": False}
    buf = CpuGpuBuffer(len(rows), COLUMNS, dtype=torch.int32, **cpu)
    return block_table_state(buf).prepare(rows)


@pytest.mark.parametrize("drafter", SPECULATIVE)
def test_the_top_corners_fill_atoms_block_table_and_one_position_more_is_refused(
    model_dir, drafter  # noqa: F811
):
    """DSpark5 verifies 6 tokens a row and writes 4 positions past it, MTP1 2
    and none: decode's top c node is 2048 - 6 - 4 = 2038 or 2046, prefill's
    2048 - 64 - 4 = 1980 or 1984. Every planned point fits the table; one
    position more takes a 129th block, which the planner, the replay rule and
    ATOM's table each refuse."""
    steps, engines = recorded(model_dir, drafter)
    width = SPECULATIVE[drafter][1] + 1
    ahead = max(0, width - 2)
    for row in ([width, 1900 + width, True], [48, 1900 + 48, False]):
        line = {"rows": [row], "rung": 1 if row[2] else None, "produces_output": True}
        steps.append((f"near {row}", line))
    rule = replay.ReplayRule(config(model_dir, drafter))

    planned = step_grid_plan.plan(DEFINITION, steps, engines)

    for line in planned["lines"]:
        table(rule.batch(line).block_tables)
    top = {"decode": (width, 2048 - width - ahead), "prefill": (64, 2048 - 64 - ahead)}
    assert planned["decode"]["nodes"]["c"]["1"][-1] == top["decode"][1]
    assert planned["prefill"]["nodes"]["c"][-1] == top["prefill"][1]
    rows = [line["rows"] for line in planned["lines"]]
    for phase, (q, c) in top.items():
        assert [[q, c + q, phase == "decode"]] in rows
    target = step_grid_plan._target(DEFINITION, engines)
    for phase, (q, c) in top.items():
        decode, held = phase == "decode", c + 1 + q + ahead
        why = step_grid_plan._infeasible(target, 1, q, c + 1, not decode)
        assert why.endswith(f" {held} is above max_model_len")
        with pytest.raises(ValueError, match=f"past (max_model_len|the {COLUMNS} col)"):
            rule.batch({"rows": [[q, c + 1 + q, decode]], "produces_output": True})
        with pytest.raises(ValueError, match="exceeds the destination columns"):
            table([list(range(-(-held // 16)))])
