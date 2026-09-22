# SPDX-License-Identifier: MIT
"""What a pipeline stage's KV share is a fraction *of*.

The parallelism design document used to say that weights and KV both shard
by layer range, so per-stage memory is `layers_in_stage / total_layers` of
the sharded terms. For weights that is right. For KV on a hybrid stack it names
the wrong quantity twice: KV is held only by the layers that cache the whole
history, and inside a span the paged count is not proportional to the span's
length -- so neither the numerator nor the denominator is a layer count.

Nothing re-checked that sentence, which is how it survived: it is true of a
uniform stack, where every layer is paged, and that is the case anyone checks
first. So this file checks it on both, and then checks that the document still
states what was checked.

Neither the split nor the paged count is written out here. The span comes from
ATOM's own `get_pp_indices`, which hands the remainder to the *middle*
partitions, and the paged count comes from the model's own `layer_types`
through the geometry that sizes a block. `VLLM_PP_LAYER_PARTITION` overrides
the partitioner, so every derivation clears it first; left set, this reads a
layer layout out of the environment and calls it ATOM's.

The block size is a choice and not a measurement. It is named so, and it does
not enter a paged-layer count -- only the bytes a block costs, which nothing
here asserts. How many blocks those bytes buy is the memory document's
subject, and no count of them is asserted here.
"""

import json
import pathlib

from transformers import PretrainedConfig

from atom.compass.backends import KvGeometry, Parallelism
from atom.models.utils import get_pp_indices

REPO = pathlib.Path(__file__).resolve().parents[2]
DESIGN = REPO / "atom/compass/design/15_parallelism_support.md"
CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
HYBRID_DICT = json.loads(CONFIG_JSON.read_text())["text_config"]
HYBRID = PretrainedConfig.from_dict(HYBRID_DICT)
UNIFORM = PretrainedConfig.from_dict(
    dict(HYBRID_DICT, layer_types=["full_attention"] * len(HYBRID_DICT["layer_types"]))
)
BLOCK_SIZE = 64
WIDTHS = range(2, 9)


def stages(hf_config, pp_size, monkeypatch):
    """(span, layers held, paged layers) per stage, as ATOM splits the stack."""
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    spans = [
        get_pp_indices(int(hf_config.num_hidden_layers), rank, pp_size)
        for rank in range(pp_size)
    ]
    return [
        (
            span,
            span[1] - span[0],
            KvGeometry.from_hf_config(
                hf_config,
                block_size=BLOCK_SIZE,
                parallelism=Parallelism(pp_size=pp_size),
                layer_range=span,
            ).layers,
        )
        for span in spans
    ]


def agreeing(rows, total_layers, total_paged):
    """The stages whose layer fraction equals their paged fraction.

    Cross-multiplied rather than divided: the two fractions are exact
    rationals and a float comparison would make this a claim about rounding.
    """
    return [
        rank
        for rank, (_, held, paged) in enumerate(rows)
        if held * total_paged == paged * total_layers
    ]


def test_the_kv_share_of_a_stage_is_not_its_layer_share(monkeypatch):
    """The counterexample, in the row where it takes one line to see.

    At six stages, stages 1 and 3 each hold **11** layers and hold **3** and
    **2** paged ones. `layers_in_stage / total_layers` is 11/64 for both and
    their KV is not equal, so that ratio cannot be what the KV term scales by.

    The held and paged counts are asserted as pairs, together, because the
    defect being pinned is precisely the belief that one determines the other:
    a test that asserted only the paged column would stay green if the held
    column were edited to match it.

    At three stages the same thing shows the other way round -- the stage
    holding the most layers holds the fewest paged ones -- and the map over
    every width from 2 to 8 says where the layer ratio happens to be right,
    which is at every stage of 2, 4 and 8 and at stage 4 of 5 alone. If the
    two columns are ever made proportional again, that map is all-stages
    everywhere and this fails.
    """
    per_width = {pp: stages(HYBRID, pp, monkeypatch) for pp in WIDTHS}
    assert [(held, paged) for _, held, paged in per_width[6]] == [
        (10, 2),
        (11, 3),
        (11, 3),
        (11, 2),
        (11, 3),
        (10, 3),
    ]
    assert [(held, paged) for _, held, paged in per_width[3]] == [
        (21, 5),
        (22, 5),
        (21, 6),
    ]
    held, paged = zip(*[(h, p) for _, h, p in per_width[3]])
    assert paged[held.index(max(held))] == min(paged) < max(paged)
    assert {pp: agreeing(rows, 64, 16) for pp, rows in per_width.items()} == {
        2: [0, 1],
        3: [],
        4: [0, 1, 2, 3],
        5: [4],
        6: [],
        7: [],
        8: [0, 1, 2, 3, 4, 5, 6, 7],
    }


def test_a_uniform_stack_is_the_case_the_layer_ratio_gets_right(monkeypatch):
    """The correction does not reach further than the hybrid.

    The same model with every layer declared `full_attention` is a uniform
    stack, and there the layer ratio and the paged ratio are the same number
    at every stage and every width. That is the half of the old sentence that
    was never wrong, and asserting it is what stops this correction from being
    read as "the layer range is never the key".
    """
    total = int(UNIFORM.num_hidden_layers)
    for pp in WIDTHS:
        rows = stages(UNIFORM, pp, monkeypatch)
        assert [held for _, held, _ in rows] == [paged for _, _, paged in rows]
        assert agreeing(rows, total, total) == list(range(pp))


def test_the_document_states_the_split_that_was_derived(monkeypatch):
    """The prose and the derivation, pinned to each other.

    The sentence this replaces drifted because nothing joined it to the
    partitioner it described. Every row of the table is rebuilt here from
    `get_pp_indices` and the model's layer types -- spans, layers held, paged
    layers, and the cell naming the stages where the two fractions agree --
    so a row edited by hand, or a partitioner that starts splitting
    differently, fails here instead of being believed.

    The three sentences asserted beside the table are the ones that bound the
    correction: what the fractions are taken over, that `runtime_constants`
    scale by neither, and that a uniform stack is unaffected. Each is a claim
    a reader would otherwise have to take on trust, and the last two are the
    places an over-reaching edit would show up.
    """
    text = DESIGN.read_text()
    total = int(HYBRID.num_hidden_layers)
    per_width = {pp: stages(HYBRID, pp, monkeypatch) for pp in WIDTHS}
    total_paged = sum(paged for _, _, paged in per_width[2])
    rows = []
    for pp, stage_rows in per_width.items():
        ranks = agreeing(stage_rows, total, total_paged)
        cell = (
            "all"
            if len(ranks) == pp
            else "none"
            if not ranks
            else "stage " + ", ".join(str(rank) for rank in ranks)
        )
        rows.append(
            "| {} | {} | {} | {} | {} |".format(
                pp,
                ", ".join(f"{start}-{end}" for (start, end), _, _ in stage_rows),
                ", ".join(str(held) for _, held, _ in stage_rows),
                ", ".join(str(paged) for _, _, paged in stage_rows),
                cell,
            )
        )
    assert [row for row in rows if row not in text] == []
    assert f"`held/{total}` = `paged/{total_paged}`" in text
    assert "The Class-C `runtime_constants` scale by neither key" in text
    assert "**Uniform stacks are unaffected.**" in text
