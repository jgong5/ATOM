# SPDX-License-Identifier: MIT
"""The pipeline-stage block count, re-derived rather than restated.

The memory and KV design document used to say that the `all_reduce(MIN)`
across pipeline stages in `get_num_blocks` is inert, *because every stage
computes the same number*, and that it *still needs a live process group or a
stub*. Both halves were wrong, and prose is how they stayed wrong: nothing
re-checked them.

So this file checks them, and then checks that the document still says what
was checked. The split is never written out here -- it comes from ATOM's own
`get_pp_indices`, which hands the remainder to the *middle* partitions and so
produces a layout no reader would guess. `VLLM_PP_LAYER_PARTITION` overrides
that partitioner, so every derivation clears it first; left set, this reads a
layer layout out of the environment and calls it ATOM's.

The budget, the block size and the widths are choices, not measurements, and
they are named constants for that reason: what is derived here is how the
count *moves* between stages, not its absolute value. The model is the
64-layer hybrid vendored beside this file -- one full-attention layer in four,
so 16 of the 64 hold paged KV, and a stage's paged-layer count is therefore
not its layer count.

The guard on the reduce is read out of ATOM's source with `ast` rather than by
importing it: the claim is about what the code says, and the runner does not
import on a machine with no driver.
"""

import ast
import json
import pathlib

from transformers import PretrainedConfig

from atom.compass.backends import KvGeometry, Parallelism
from atom.model_ops.attentions.sub_pool_spec import page_pool, plan_pools
from atom.models.utils import get_pp_indices

REPO = pathlib.Path(__file__).resolve().parents[2]
DESIGN = REPO / "atom/compass/design/03_memory_and_kv_model.md"
RUNNER = REPO / "atom/model_engine/model_runner.py"
CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
QWEN = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
BLOCK_SIZE = 64
MAX_NUM_SEQS = 256
KV_BUDGET_BYTES = 200_000_000_000
WIDTHS = range(2, 9)


def stages(pp_size, monkeypatch):
    """(paged layers, blocks) per stage, split the way the runner splits it."""
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    geometries = [
        KvGeometry.from_hf_config(
            QWEN,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=pp_size),
            layer_range=get_pp_indices(int(QWEN.num_hidden_layers), rank, pp_size),
        )
        for rank in range(pp_size)
    ]
    return [
        (
            geometry.layers,
            plan_pools(
                [page_pool(geometry.bytes_per_block)], KV_BUDGET_BYTES, MAX_NUM_SEQS
            ).paged_entries,
        )
        for geometry in geometries
    ]


def test_the_stages_do_not_all_compute_the_same_block_count(monkeypatch):
    """The stages, taken from ATOM's own partitioner.

    Three stages of a 64-layer stack hold 21, 22 and 21 layers, which is 5, 5
    and 6 paged ones -- **two** distinct block counts, not three: the two
    five-layer stages share one. An early paraphrase of this said three, so
    the counts are asserted as a list rather than as a cardinality.

    Every width from 2 to 8 is taken, because the reason given for the
    reduction being inert -- that the readings are identical across stages --
    would hold at all of them. It is inert at 2, 4 and 8, and nowhere else
    here.
    """
    per_width = {pp: stages(pp, monkeypatch) for pp in WIDTHS}
    assert per_width[3] == [(5, 152587), (5, 152587), (6, 127156)]
    assert {
        pp: len({blocks for _, blocks in rows}) for pp, rows in per_width.items()
    } == {2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2, 8: 1}


def test_the_reduce_is_guarded_so_it_needs_neither_a_group_nor_a_stub():
    """The reduce runs only when a process group is already initialised.

    The assertion is not merely that a guard exists somewhere near the
    reduce, which a later edit could leave true while moving the call out from
    under it. It is that the runner's only `torch.distributed.all_reduce` sits
    inside the `is_initialized()` branch -- so with no group nothing runs, and
    nothing has to be stubbed for it.
    """
    tree = ast.parse(RUNNER.read_text())
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "torch.distributed.is_initialized()" in ast.unparse(node.test)
        and "ReduceOp.MIN" in ast.unparse(node)
    ]
    reduces = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.distributed.all_reduce"
    ]
    assert len(guards) == 1 and len(reduces) == 1
    assert {id(node) for node in ast.walk(guards[0])} >= {id(n) for n in reduces}


def test_the_design_document_states_the_table_that_was_derived(monkeypatch):
    """The prose and the derivation, pinned to each other.

    The note this replaces drifted because nothing joined the sentence to the
    behaviour it described. A row edited by hand, or a partitioner that starts
    splitting differently, fails here instead of being believed.
    """
    text = DESIGN.read_text()
    rows = [
        f"| {pp} | {', '.join(str(n) for n, _ in stages(pp, monkeypatch))} |"
        for pp in WIDTHS
    ]
    assert [row for row in rows if row not in text] == []
    assert "152,587 / 152,587 / 127,156" in text
