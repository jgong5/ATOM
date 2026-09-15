"""Checkpoint reads must survive the scheduler-to-oracle allocation bridge."""

from types import SimpleNamespace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.runtime.batch_spec import BatchSpec
from atom.compass.runtime.predict import CompassPredictMixin
from atom.compass.runtime.templates import BindRefusal, NativeAllocation


def binding(*, sources=(0, -1, 7), slots=(9, 5, 2), rows=(2, 0, 1)):
    allocation = NativeAllocation(
        block_size=16, max_model_len=256, cudagraph_mode="full")
    runner = SimpleNamespace(
        _oracle=SimpleNamespace(native_allocation=allocation),
        _rank_coords=lambda: {"tp": 0})
    batch = SimpleNamespace(
        block_tables=[[0, 1], [2, 3], [4, 5]],
        num_scheduled_tokens=[1, 1, 1], context_lens=[32, 32, 32],
        state_slots_committed=slots, state_rows=rows,
        state_fork_srcs=sources, total_seqs_num_prefill=0)
    CompassPredictMixin._offer_allocation(runner, batch)
    shape = StepShape(num_scheduled_tokens=(1, 1, 1),
                      context_lens=(32, 32, 32), num_prefill_tokens=0,
                      capture_bucket=4, rank_coords={"tp": 0})
    return allocation, shape


def test_checkpoint_source_and_destination_keep_the_same_row_mapping():
    allocation, shape = binding()
    actual = allocation.allocation_for(shape)
    assert actual["non_spec_state_indices_tensor"] == [[5, 2, 9, -1], "int32"]
    assert actual["non_spec_state_indices_in_tensor"] == [[5, 7, 0, -1], "int32"]


@pytest.mark.parametrize("sources", [None, (-1, -1, -1)])
def test_no_fork_stays_in_place(sources):
    allocation, shape = binding(sources=sources)
    actual = allocation.allocation_for(shape)
    assert actual["non_spec_state_indices_tensor"] == [[5, 2, 9, -1], "int32"]
    assert actual["non_spec_state_indices_in_tensor"] == actual[
        "non_spec_state_indices_tensor"]


def test_filtered_state_rows_are_still_refused_instead_of_shifted():
    allocation, shape = binding(sources=(7, -1), slots=(9, 5), rows=(1, 2))
    with pytest.raises(BindRefusal, match=r"rows \[0\].*hold no state"):
        allocation.allocation_for(shape)


def test_short_fork_vector_is_refused():
    allocation, shape = binding(sources=(7,))
    with pytest.raises(BindRefusal, match="one state fork source per state slot"):
        allocation.allocation_for(shape)


def test_prefill_can_read_shared_checkpoint_into_distinct_destinations():
    spec = BatchSpec(kind="prefill", query_lens=(16, 32),
                     context_lens=(128, 144), block_size=16, max_model_len=256)
    actual = dict(spec.gdn_context(state_slots=(9, 5), state_fork_srcs=(3, 3)))
    assert actual["non_spec_state_indices_tensor"] == [[9, 5], "int32"]
    assert actual["non_spec_state_indices_in_tensor"] == [[3, 3], "int32"]
    assert actual["has_initial_state"] == [[1, 1], "bool"]


def test_spec_decode_cannot_silently_use_non_spec_fork_indices():
    spec = BatchSpec(kind="decode", query_lens=(2,), context_lens=(32,),
                     block_size=16, max_model_len=256, num_spec_step=1)
    with pytest.raises(ValueError, match="state fork on the spec-decode path"):
        spec.gdn_context(state_slots=(9,), state_fork_srcs=(3,))


def test_scheduled_checkpoint_fork_reaches_the_oracle():
    from atom.model_engine.scheduler import ScheduledBatch, Scheduler
    from atom.model_engine.sequence import Sequence

    seqs = {}
    for dest, src in [(9, 0), (5, -1)]:
        seq = Sequence(list(range(32)), 16, has_per_req_cache=True)
        seq.state_slots = [dest]
        seq.state_fork_src = src
        seq.block_table.extend([2 * len(seqs), 2 * len(seqs) + 1])
        seqs[seq.id] = seq
    batch = ScheduledBatch(
        seqs, [16, 16], 32, total_tokens_num_prefill=32,
        total_seqs_num=2, total_seqs_num_prefill=2,
        num_cached_tokens=[16, 16])
    # The scheduler consumes sequence flags after taking the batch snapshot.
    Scheduler._consume_state_forks(seqs)
    assert all(seq.state_fork_src == -1 for seq in seqs.values())
    allocation = NativeAllocation(block_size=16, max_model_len=256)
    runner = SimpleNamespace(_oracle=SimpleNamespace(native_allocation=allocation),
                             _rank_coords=lambda: {})
    CompassPredictMixin._offer_allocation(runner, batch)
    shape = StepShape(num_scheduled_tokens=(16, 16), context_lens=(32, 32),
                      num_prefill_tokens=32)
    actual = allocation.allocation_for(shape)
    assert actual["non_spec_state_indices_tensor"] == [[9, 5], "int32"]
    assert actual["non_spec_state_indices_in_tensor"] == [[0, 5], "int32"]
