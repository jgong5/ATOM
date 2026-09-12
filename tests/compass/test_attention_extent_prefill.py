"""The unified launch extent is a decode rule, and prefill must not take it.

`_output_rows` reads `q.shape[0] // max_seqlen_q` for the unified wrapper.
That is the Gluon decode kernel's own batch expression and it is the batch
size only because every launched decode row carries the same query length.
A prefill breaks the premise -- its rows *are* the tokens -- so the quotient
is neither a row count nor a token count: one request of 4096 tokens gives 1,
and a ragged (3, 1, 1, 1) gives 2.

That mattered because `geometry_of` folds `executed_rows` into the set of
extents it will abstract to `TOKEN_AXIS`. A prefill reporting 1 makes every
unrelated leading axis of 1 in the same call look like the token axis, and two
operators that are not the same call collapse onto one geometry key.

These tests pin the distinction in both directions: prefill declines, decode
still answers.
"""
import pytest

from atom.compass.core.cost.families import attention


def _op(shapes, *, cu_seqlens_q, max_seqlen_q, is_prefill, context_lens=None,
        dtypes=None):
    context = [["max_seqlen_q", max_seqlen_q],
               ["cu_seqlens_q", cu_seqlens_q],
               ["context_lens", context_lens if context_lens is not None
                else [0] * (len(cu_seqlens_q) - 1)]]
    if is_prefill is not None:
        context.append(["is_prefill", is_prefill])
    return {"name": attention.UNIFIED, "input_shapes": shapes,
            "dtypes": dtypes or ["bf16"] * len(shapes), "context": context}


def test_single_request_prefill_reports_no_launched_rows():
    """4096 // 4096 == 1 is not one launched row; it is the wrong rule."""
    op = _op([[4096, 24, 256], [4096, 4, 256], [4096, 4, 256]],
             cu_seqlens_q=[0, 4096], max_seqlen_q=4096, is_prefill=True)
    assert attention.structure_of(op).executed_rows is None


def test_ragged_prefill_whose_rows_divide_reports_no_launched_rows():
    """(3,1,1,1): six rows, longest three, 6 % 3 == 0, quotient 2.

    Two is neither the four sequences nor the six rows, and the divisibility
    guard cannot catch it because the rows do divide.
    """
    op = _op([[6, 24, 256], [6, 4, 256]], cu_seqlens_q=[0, 3, 4, 5, 6],
             max_seqlen_q=3, is_prefill=True)
    structure = attention.structure_of(op)
    assert list(structure.queries) == [3, 1, 1, 1]
    assert structure.executed_rows is None


def test_prefill_does_not_abstract_an_unrelated_leading_axis_of_one():
    """The regression: a scalar-ish operand must keep its own extent.

    Operand 1 has leading extent 1 and nothing to do with tokens. Under the
    unguarded rule the call reported `executed_rows == 1`, so `geometry_of`
    rewrote that 1 to the token axis.
    """
    op = _op([[4096, 24, 256], [1], [4096, 4, 256]],
             cu_seqlens_q=[0, 4096], max_seqlen_q=4096, is_prefill=True)
    shapes = attention.geometry_of(op)[0]
    assert shapes[1] == (1,), "a leading 1 that is not a token count moved"
    assert shapes[0][0] == attention.TOKEN_AXIS, "the real token axis is lost"


def test_two_prefills_differing_only_in_that_axis_stay_distinct():
    """The consequence the abstraction would have had: a key collision."""
    one = _op([[4096, 24, 256], [1], [4096, 4, 256]],
              cu_seqlens_q=[0, 4096], max_seqlen_q=4096, is_prefill=True)
    two = _op([[4096, 24, 256], [8], [4096, 4, 256]],
              cu_seqlens_q=[0, 4096], max_seqlen_q=4096, is_prefill=True)
    assert attention.geometry_of(one) != attention.geometry_of(two)


def test_decode_still_reports_its_launched_rows():
    """The decode contract this correction must not disturb."""
    op = _op([[4, 24, 256], [4, 4, 256]], cu_seqlens_q=[0, 1, 2, 3, 4],
             max_seqlen_q=1, is_prefill=False, context_lens=[50, 50, 50, 50])
    assert attention.structure_of(op).executed_rows == 4


def test_padded_decode_still_reports_every_launched_row():
    """A bucket of four holding three requests launches four rows."""
    op = _op([[4, 24, 256], [4, 4, 256]], cu_seqlens_q=[0, 1, 2, 3, 3],
             max_seqlen_q=1, is_prefill=False, context_lens=[50, 50, 50, 0])
    structure = attention.structure_of(op)
    assert structure.executed_rows == 4
    assert structure.active_sequences == 3


def test_a_call_that_does_not_say_which_it_is_declines():
    """Unknown beats a plausible number, as elsewhere in this function."""
    op = _op([[4, 24, 256]], cu_seqlens_q=[0, 1, 2, 3, 4], max_seqlen_q=1,
             is_prefill=None)
    assert attention.structure_of(op).executed_rows is None


@pytest.mark.parametrize("rows,max_q", [(6, 4), (5, 2), (7, 3)])
def test_decode_rows_that_do_not_divide_still_decline(rows, max_q):
    """The pre-existing divisibility guard is untouched."""
    op = _op([[rows, 24, 256]], cu_seqlens_q=[0, rows], max_seqlen_q=max_q,
             is_prefill=False)
    assert attention.structure_of(op).executed_rows is None
