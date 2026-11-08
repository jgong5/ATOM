# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` on an MI455 card, with no device: a graph the
store lacks is traced on the run's first GPU as that card reads, so its
stepping and compute units pick the kernels, and `trace ops` holds an
`--arch` beside `--gpu` to the card's. A point's second placement begins
over the tentative state DeepSeek-V4.1's verify step left, which no sampler
commits.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_capture_mi455_cells import A0, CELL, TREE, _trace

from atom.compass import capture
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import decode


@pytest.fixture
def started(monkeypatch):
    """Each `trace ops` argv the plan's capture starts, its child stood in."""
    argvs = []
    monkeypatch.setattr(ops, "_child", lambda fn, argv: argvs.append(argv))
    monkeypatch.setattr("atom.compass.capture.forward.device_arch", lambda: "gfx1250")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    return argvs


def test_a_plan_traces_on_its_first_gpu_as_that_card_reads(started):
    config = SimpleNamespace(model="m", tensor_parallel_size=2)
    ops._capture(config, [decode(4)], "s")

    (argv,) = started
    assert argv[argv.index("--gpu") + 1] == "2"
    assert argv[argv.index("--arch") + 1] == "gfx1250"


def test_coverage_traces_the_arch_its_run_recorded_on_no_card(started):
    config = SimpleNamespace(model="m", tensor_parallel_size=2)
    ops._capture(config, [decode(4)], "s", arch="gfx950")

    (argv,) = started
    assert "--gpu" not in argv and argv[argv.index("--arch") + 1] == "gfx950"


def test_an_arch_beside_a_card_is_the_card_s(tmp_path, monkeypatch):
    monkeypatch.setattr(driverless, "card", {0: A0}.get)
    argv = ["--model", "m", "--arch", "gfx950", "--gpu", "0", "--width", "4"]
    with pytest.raises(ValueError, match="card 0 reads as gfx1250, not gfx950"):
        capture.main([*argv, "--store", str(tmp_path)])

    read = "from atom.compass.capture import driverless\n"
    read += f"driverless.card = {{0: {A0!r}}}.get\n"
    where = ["--arch", "gfx1250", "--gpu", "0"]
    traced = _trace(tmp_path / "t", "--width 4" + CELL, where, read)
    for graph, stated in traced.values():
        assert stated["card"] == A0 and graph["structure"].split()[1] == "gfx1250"


#: Two placements of a point on DeepSeek-V4.1's own cache, on the host of a
#: declared gfx1250: a decode row verifying 5 drafts at 192 tokens, its state
#: seated there each time as `replay.seat_states` seats it, and no sampler to
#: commit the first's tentative state before the second begins.
STATE = """
import contextlib, sys
from dataclasses import replace
from atom.compass.capture import driverless
driverless.install(sys.argv[1], "gfx1250")
from atom.compass.bench import ops
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.models.deepseek_v41.config import DeepseekV41TextConfig
from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step, geometry
config = DeepseekV41TextConfig(
    hidden_size=64, head_dim=512, num_attention_heads=8, q_lora_rank=32,
    o_groups=1, o_lora_rank=32, sliding_window=4, index_n_heads=32,
    index_head_dim=32, index_topk=4, candidate_block_size=2,
    candidate_topk_blocks=4, max_position_embeddings=32, num_hidden_layers=5,
    num_nextn_predict_layers=0, compress_ratios=(0, 2, 2, 1, 1),
    kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 4),
    candidate_source_layer_id=3)
geo = replace(geometry(config), window_size=32, speculative_tokens=5)
verify = PagedRequest(0, 192, 0, 6, 0, tuple(range(7)))
for stand_in in (False, True):
    cache = PagedAttentionCache(geo, 16, 4, "cpu")
    try:
        with ops._uncommitted() if stand_in else contextlib.nullcontext():
            for _ in range(2):
                cache.cursor[0, 0] = 192
                cache.prepare_state(begin_step(cache, [verify], tentative=True))
        print("placed twice")
    except RuntimeError as refused:
        print(refused)
"""


def test_a_point_s_second_placement_begins_over_the_first_s_tentative_state(
    tmp_path,
):
    env = {**os.environ, "PYTHONPATH": str(TREE)}
    done = subprocess.run(
        [sys.executable, "-c", STATE, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=TREE,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    refused, placed = done.stdout.strip().splitlines()[-2:]
    # ATOM's own refusal on the second placement, where no sampler committed
    assert refused.startswith("Commit the accepted prefix")
    assert placed == "placed twice"
