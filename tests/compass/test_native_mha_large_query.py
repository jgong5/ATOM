from copy import deepcopy

import pytest

from atom.compass.core.cost.native_mha_large_query import coordinates, argument_set_count
from .test_native_mha_low_query import operator


def test_native_token_capacity_is_abstracted_only_after_exact_layout_validation():
    for q in (512, 8192, 16384):
        op = operator((q,), (8192,))
        assert coordinates(op)["q"] == [q]
        op["layouts"][0][1][2] += 1
        assert coordinates(op) is None


def test_argument_rotation_counts_the_large_native_v_backing_storage():
    assert argument_set_count(coordinates(operator((8192,), (8192,)))) == 3
    assert argument_set_count(coordinates(operator((8192, 8192), (8192, 8192)))) == 2


@pytest.mark.parametrize("queries,history", [
    ((8192, 48, 48), (180224, 0, 0)),
    ((15, 8192, 7504), (195984, 8192, 688)),
    ((512, 512, 512, 512), (8192, 16384, 32768, 49152)),
])
def test_mixed_large_queries_check_every_writable_block(queries, history):
    op = operator(queries, history)
    row = coordinates(op)
    assert row["q"] == list(queries) and row["history"] == list(history)
    context = dict(op["context"])
    # Corrupt a write deep in a query, beyond the one-block low-query case.
    context["slot_mapping"][sum(queries) - 17] -= 16
    assert coordinates(op) is None


@pytest.mark.parametrize("queries,history", [
    ((8144,), (48,)), ((7504,), (688,)), ((432,), (48720,)),
    ((8192, 8192, 48), (8192, 8192, 8192)),
    ((512,) * 5, (8192,) * 5),
])
def test_predeclared_fringe_and_scheduler_limit_remain_outside_model(queries, history):
    assert coordinates(operator(queries, history)) is None


@pytest.mark.parametrize("damage", ["starts", "negative_slot", "short_slots", "tail_alias", "interior_alias"])
def test_invalid_cached_addresses_cannot_be_priced(damage):
    op = operator((8192, 512), (8192, 16384))
    assert coordinates(op) is not None
    context = dict(op["context"])
    width = len(context["block_tables"]) // 2
    if damage == "starts":
        context["seq_starts"][1] = 8192
    elif damage == "negative_slot":
        context["slot_mapping"][0] = -1
    elif damage == "short_slots":
        context["slot_mapping"].pop()
    elif damage == "tail_alias":
        context["block_tables"][width + 1024] = context["block_tables"][512]
    else:
        context["block_tables"][width + 1] = context["block_tables"][3]
    assert coordinates(op) is None


def test_shared_prefix_is_preserved_while_query_blocks_remain_private():
    op = operator((8192, 512), (8192, 16384))
    context = dict(op["context"])
    width = len(context["block_tables"]) // 2
    context["block_tables"][width:width + 100] = context["block_tables"][:100]
    assert coordinates(op)["sharing"] == [100]
    changed = deepcopy(op)
    next(value for key, value in changed["context"] if key == "block_tables")[width + 100] = 101
    assert coordinates(changed) is None
