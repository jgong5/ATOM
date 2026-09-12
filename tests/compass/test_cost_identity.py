"""The cost key: same work priced once, different work still separated.

Four questions, and the third is the one that makes the change usable at all:

1. does a batch whose allocator handed it different addresses match a price
   measured under the old ones?
2. does anything that is *not* an address still separate two operators?
3. does a price file written before this change reindex without recollection?
4. is the normalisation idempotent, so a library may be reindexed twice?

Device-free: these are string and dict operations, no engine and no GPU.
"""
import pytest

from atom.compass.core.cost.identity import (
    ADDRESS_COMPONENTS, cost_key, normalize_component, normalized_context)
from atom.compass.runtime.microbench import cost_key_of, signature_of


def decode_op(slots, *, context=1151, rows=8, positions=None, name=None):
    """A decode attention operator, as a bound graph records one."""
    return {
        "name": name or "aiter::unified_attention_with_output_base",
        "input_shapes": [[rows, 6144], [rows, 1024], [rows, 1024]],
        "dtypes": ["torch.bfloat16", "torch.bfloat16", "torch.bfloat16"],
        "context": [
            ["slot_mapping", list(slots)],
            ["positions", list(positions if positions is not None
                                else range(rows))],
            ["context_lens", [context] * rows],
            ["max_seqlen_k", context],
            ["cu_seqlens_q", list(range(rows + 1))],
            ["block_tables", [[0, 1, 2]] * rows],
            ["block_tables_shape", [rows, 3]],
        ],
        "scalars": [["max_qlen", 1]],
    }


# The measured allocation and the one a replay's scheduler produces: the same
# eight rows at the same history, in different physical slots.
MEASURED = [9102, 9118, 9134, 9150, 9166, 9182, 9198, 9214]
REPLAYED = [1150, 2302, 3454, 4606, 5758, 6910, 8062, 9214]


def test_shifted_allocation_is_the_same_price():
    a, b = decode_op(MEASURED), decode_op(REPLAYED)
    assert signature_of(a) != signature_of(b), (
        "the observation identity must still tell the two calls apart")
    assert cost_key_of(a) == cost_key_of(b)


def test_positions_shift_too():
    """`positions` is an address in the same sense, and bucket 1's mrope
    was unpriced for exactly this reason."""
    a = decode_op(MEASURED, positions=list(range(8)))
    b = decode_op(MEASURED, positions=[1151 + i for i in range(8)])
    assert signature_of(a) != signature_of(b)
    assert cost_key_of(a) == cost_key_of(b)


def test_padding_survives():
    """A padded row carries -1. A bucket of 8 with 5 real rows is not a
    bucket of 8, and collapsing those two is the error in the other
    direction."""
    real = decode_op(MEASURED)
    padded = decode_op(MEASURED[:5] + [-1, -1, -1])
    assert cost_key_of(real) != cost_key_of(padded)
    assert "void=3" in cost_key_of(padded)
    assert "void=0" in cost_key_of(real)


def test_row_count_survives():
    few = decode_op(MEASURED[:4], rows=4)
    many = decode_op(MEASURED, rows=8)
    assert cost_key_of(few) != cost_key_of(many)


@pytest.mark.parametrize("changed", [
    {"context": 4096},
    {"rows": 16},
    {"name": "aiter::linear_attention_with_output_base"},
])
def test_work_still_separates(changed):
    base = decode_op(MEASURED)
    other = decode_op(MEASURED, **changed)
    assert cost_key_of(base) != cost_key_of(other)


def test_history_separates_under_one_allocation():
    """The whole point of `context_lens` being in the key: the same slots at
    a different history are not the same amount of work."""
    assert (cost_key_of(decode_op(MEASURED, context=641))
            != cost_key_of(decode_op(MEASURED, context=64385)))


def test_scalars_and_grid_survive():
    a = decode_op(MEASURED)
    b = dict(a, scalars=[["max_qlen", 16384]])
    assert cost_key_of(a) != cost_key_of(b)
    c = dict(a, launch=[["grid", [64, 1, 1]]])
    d = dict(a, launch=[["grid", [128, 1, 1]]])
    assert cost_key_of(c) != cost_key_of(d)


def test_backward_compatible_reindex():
    """An old price file is keyed by the raw signature. Passing the stored
    key through the same function is the whole reindex -- no recollection,
    no rewritten artifact."""
    stored = {signature_of(decode_op(MEASURED)): {"seconds": 1.25e-4}}
    reindexed = {cost_key(k): v for k, v in stored.items()}
    assert reindexed[cost_key_of(decode_op(REPLAYED))]["seconds"] == 1.25e-4


def test_idempotent():
    once = cost_key(signature_of(decode_op(MEASURED)))
    assert cost_key(once) == once
    assert cost_key(cost_key(once)) == once


def test_state_index_tensor_form_agrees_across_both_paths():
    """The state-index tensors are recorded as ``[values, dtype]``. The
    string path and the operator path must summarise that identically, or a
    price keyed one way is looked up the other."""
    value = [[0, 1, 2, 3], "torch.int32"]
    from_op = normalize_component("non_spec_state_indices_tensor", value)
    from_text = normalize_component("non_spec_state_indices_tensor",
                                    str(value))
    assert from_op == from_text
    assert "torch.int32" in from_op
    assert "n=4" in from_op


def test_state_index_dtype_still_separates():
    a = normalize_component("state_indices", [[0, 1], "torch.int32"])
    b = normalize_component("state_indices", [[0, 1], "torch.int64"])
    assert a != b


def test_normalized_context_matches_the_string_path():
    op = decode_op(MEASURED)
    by_op = dict(normalized_context(op))
    for name, raw in (tuple(x) for x in op["context"]):
        if name in ADDRESS_COMPONENTS:
            assert by_op[name] == normalize_component(name, str(raw))
        else:
            assert by_op[name] == raw


def test_non_address_context_is_untouched():
    op = decode_op(MEASURED)
    by_op = dict(normalized_context(op))
    assert by_op["context_lens"] == [1151] * 8
    assert by_op["max_seqlen_k"] == 1151


def test_empty_signature_is_returned_unchanged():
    assert cost_key("") == ""
