"""Declared workloads must retain their arrival order after a long prefill."""

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, get_clock, set_clock


@pytest.fixture
def virtual_clock():
    previous = get_clock()
    clock = VirtualClock(epoch=1000.0)
    set_clock(clock)
    yield clock
    set_clock(previous)


def _scheduler():
    return Scheduler(
        MockConfig(
            max_num_seqs=4,
            max_num_batched_tokens=4,
            num_kvcache_blocks=64,
        )
    )


def _request(seq_factory, arrival, total, *, tokens=4, index=None):
    seq = seq_factory(
        [10] * tokens, sampling_params=SamplingParams(max_tokens=1)
    )
    seq.arrive_time = 1000.0 + arrival
    seq.compass_workload_size = total
    seq.compass_workload_index = index
    return seq


def _finish_step(scheduler, batch, seqs):
    scheduler.postprocess(
        seqs,
        ScheduledBatchOutput(
            req_ids=list(batch.req_ids),
            token_ids=[(99,)] * len(batch.req_ids),
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )


def test_later_arrival_cannot_overtake_after_long_prefill(
    virtual_clock, seq_factory
):
    scheduler = _scheduler()
    first = _request(seq_factory, 0, 4, tokens=12)
    later = _request(seq_factory, 20, 4)
    earlier = _request(seq_factory, 5, 4)
    middle = _request(seq_factory, 10, 4)
    # The HTTP receipt order is different from the prescribed arrival order.
    scheduler.extend([first, later, earlier, middle])

    for _ in range(3):
        batch, seqs = scheduler.schedule()
        assert list(batch.req_ids) == [first.id]
        virtual_clock.advance(10)
        _finish_step(scheduler, batch, seqs)

    # All three waiting requests are now due. The later one must not jump
    # ahead just because its HTTP request reached the server earlier.
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [earlier.id]


def test_equal_arrivals_follow_workload_index_not_http_receipt(
    virtual_clock, seq_factory
):
    scheduler = _scheduler()
    last = _request(seq_factory, 0, 3, index=2)
    first = _request(seq_factory, 0, 3, index=0)
    middle = _request(seq_factory, 0, 3, index=1)
    scheduler.extend([last, first, middle])

    actual = []
    for _ in range(3):
        batch, seqs = scheduler.schedule()
        actual.extend(batch.req_ids)
        _finish_step(scheduler, batch, seqs)
    assert actual == [first.id, middle.id, last.id]


def test_future_arrivals_wait_and_idle_clock_jumps_to_the_earliest(
    virtual_clock, seq_factory
):
    scheduler = _scheduler()
    later = _request(seq_factory, 20, 2, index=0)
    earlier = _request(seq_factory, 5, 2, index=1)
    scheduler.extend([later, earlier])

    batch, seqs = scheduler.schedule()
    assert list(batch.req_ids) == [earlier.id]
    assert virtual_clock.elapsed == 5
    _finish_step(scheduler, batch, seqs)
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [later.id]
    assert virtual_clock.elapsed == 20


def test_legacy_equal_arrivals_keep_receipt_order(virtual_clock, seq_factory):
    scheduler = _scheduler()
    first = _request(seq_factory, 0, 2)
    second = _request(seq_factory, 0, 2)
    scheduler.extend([second, first])
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [second.id]


def test_real_clock_keeps_fifo_even_with_declared_metadata(seq_factory):
    from atom.utils.clock import WallClock

    previous = get_clock()
    set_clock(WallClock())
    try:
        scheduler = _scheduler()
        first = _request(seq_factory, 20, 2, index=1)
        second = _request(seq_factory, 0, 2, index=0)
        scheduler.extend([first, second])
        batch, _ = scheduler.schedule()
        assert list(batch.req_ids) == [first.id]
    finally:
        set_clock(previous)


def test_preemptions_keep_requeue_priority_after_initial_ordering(
    virtual_clock, seq_factory
):
    scheduler = _scheduler()
    first = _request(seq_factory, 0, 3, index=0)
    second = _request(seq_factory, 1, 3, index=1)
    third = _request(seq_factory, 2, 3, index=2)
    scheduler.extend([third, second, first])
    virtual_clock.advance(3)

    # Keep both scheduled requests live and then preempt them in order.
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [first.id]
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [second.id]
    for seq in [first, second]:
        scheduler.running.remove(seq)
        assert scheduler.preempt(seq)

    # Preemption puts the most recently preempted request at the front. A
    # per-tick arrival sort would wrongly undo this scheduler priority.
    batch, _ = scheduler.schedule()
    assert list(batch.req_ids) == [second.id]
