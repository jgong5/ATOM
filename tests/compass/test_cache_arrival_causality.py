"""Registering a future cc-traces prefix cannot expose it to native caching."""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, get_clock, set_clock


@pytest.mark.parametrize("admission_delay", [0.0, 3.0])
def test_registered_future_prefix_is_invisible_until_logical_readiness(admission_delay):
    old_clock = get_clock()
    clock = VirtualClock(epoch=1000.0)
    set_clock(clock)
    try:
        scheduler = Scheduler(MockConfig(
            kv_cache_block_size=16, num_kvcache_blocks=8192,
            max_num_batched_tokens=16384, max_num_seqs=32, max_model_len=262144,
            enable_prefix_caching=True, pool_entries={"state": 32},
            state_checkpoint_interval_tokens=8192, state_checkpoint_demand=True,
            compass_config=SimpleNamespace(admission_seconds=admission_delay)),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)))
        first_tokens = list(range(58368))
        later_tokens = first_tokens[:32448] + list(range(100000, 125920))
        first, later = [Sequence(tokens, 16, has_per_req_cache=True,
                                  sampling_params=SamplingParams(max_tokens=1))
                        for tokens in (first_tokens, later_tokens)]
        for index, seq in enumerate((first, later)):
            seq.arrive_time = 1000.0 + (21.434 if index else 0.0)
            seq.compass_workload_index = index
            seq.compass_workload_size = 2
        scheduler.extend([later, first])  # physical registration may be reversed
        bm = scheduler.block_manager
        assert bm.kv.num_indexed == 0 and bm.demands_recorded == 0
        assert later.checkpoint_demand_pos == later.checkpoint_end_pos == 0
        cold_chunks = []
        while first in scheduler.waiting or first in scheduler.running:
            batch, seqs = scheduler.schedule()
            assert batch.req_ids == [first.id]
            if batch.total_seqs_num_prefill:
                cold_chunks.append(int(batch.num_scheduled_tokens[0]))
            scheduler.postprocess(list(seqs.values()), ScheduledBatchOutput(
                req_ids=batch.req_ids, token_ids=[(99,)], num_rejected=None,
                num_bonus=None, draft_token_ids=None), batch=batch)
            clock.advance(0.1)
            assert later.checkpoint_demand_pos == later.checkpoint_end_pos == 0
            assert bm.demands_recorded == 0
        assert cold_chunks == [16384, 16384, 16384, 8192, 1008, 16]

        # The delayer runs before the admission loop. Its peek must have the
        # same readiness gate, including admission time after external arrival.
        ready_at = later.arrive_time + admission_delay
        clock.advance(ready_at - clock.time() - 0.01)
        before = (bm.pool_pressure(), dict(bm.state.hash_to_slot))
        assert not scheduler._can_admit_head_prefill()
        assert scheduler._waiting_new_token_count() == 0
        assert scheduler._oldest_waiting_prefill_age_ms() == 0
        assert (bm.pool_pressure(), dict(bm.state.hash_to_slot)) == before
        assert bm.demands_recorded == 0 and later.checkpoint_demand_pos == 0

        clock.advance(0.01)
        assert scheduler._can_admit_head_prefill()
        assert bm.demands_recorded == 1
        assert later.checkpoint_demand_pos == 32448
        batch, _ = scheduler.schedule()
        assert batch.req_ids == [later.id]
        # The shared prefix is not itself a saved recurrent-state checkpoint.
        # The native ladder reuses 16384, leaving 41984 query tokens in total.
        assert later.num_cached_tokens == 16384
        assert later.num_prompt_tokens - later.num_cached_tokens == 41984
        assert batch.state_fork_srcs[0] >= 0
        assert batch.state_fork_srcs[0] != batch.state_slots_committed[0]
    finally:
        set_clock(old_clock)
