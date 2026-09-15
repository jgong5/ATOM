"""Native batch facts distinguish region source paths without request identities."""
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.runtime.predict import CompassPredictMixin
from atom.compass.runtime.templates import BindRefusal, NativeAllocation
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence, SequenceType
from atom.sampling_params import SamplingParams


def native_context(*, prefix_hit=0, cap=987, ignore_eos=True, stop_strings=None,
                   query=8, history=11248, midstep=False, blocks=704,
                   config_changes=None, batch_changes=None, continuation=None):
    seq = Sequence(list(range(11256)), 16, has_per_req_cache=True,
                   sampling_params=SamplingParams(temperature=1, max_tokens=cap,
                                                  ignore_eos=ignore_eos, stop_strings=stop_strings))
    seq.block_table.extend(range(blocks))
    seq.type = SequenceType.PREFILL
    seq.num_cached_tokens = history
    seq.state_slots = [7]
    seq.state_fork_src = 3 if history else -1
    seq.prefix_cache_hit_tokens = prefix_hit
    seq.is_partial_prefill = (bool(history) and not bool(prefix_hit)) if continuation is None else continuation
    if midstep:
        seq.midstep_reservations = [(4, 8192, 123)]
    batch = ScheduledBatch({seq.id: seq}, [query], query, total_tokens_num_prefill=query,
                           total_seqs_num=1, total_seqs_num_prefill=1,
                           num_cached_tokens=[history], is_final_chunk=[history + query == 11256])
    config = SimpleNamespace(model="Qwen/Qwen3.8-27B", kv_cache_dtype="bf16",
        compilation_config=SimpleNamespace(level=3, cudagraph_mode=SimpleNamespace(name="FULL")),
        pipeline_parallel_size=1, enable_prefix_caching=True,
        state_checkpoint_demand=True, state_checkpoint_interval_tokens=8192,
        speculative_config=None)
    for key, value in (config_changes or {}).items():
        setattr(config, key, value)
    for key, value in (batch_changes or {}).items():
        setattr(batch, key, value)
    allocation = NativeAllocation(block_size=16, max_model_len=262144, position_rows=3,
                                  cudagraph_mode="full", capture_region_context=True)
    runner = SimpleNamespace(config=config, _oracle=SimpleNamespace(native_allocation=allocation),
                             _rank_coords=lambda: {"tp": 0})
    CompassPredictMixin._offer_allocation(runner, batch)
    shape = StepShape((query,), (history + query,), num_prefill_tokens=query, compiled=True,
                      produces_output=history + query == 11256,
                      topology={"tp": 1}, rank_coords={"tp": 0})
    return allocation, shape, seq


def test_region_context_retains_native_allocation_and_within_request_origin():
    allocation, shape, seq = native_context()
    # Later sequence mutations cannot rewrite the batch that was offered.
    seq.prefix_cache_hit_tokens = 11248
    context = allocation.region_context_for(shape)
    assert context["prefix_cache_hit_tokens"] == (0,)
    assert context["prefill_continuations"] == (True,)
    assert context["allocation_blocks"] == (704,)
    assert context["state_slots"] == (7,)
    assert context["state_fork_srcs"] == (3,)
    assert context["temperatures"] == (1.0,)
    assert context["return_logprobs"] == (False,)
    assert context["independent_noise"] == (False,)
    assert context["state_maintenance_empty"] is True
    assert context["midstep_saves_empty"] is True


def test_cold_and_continuing_batches_snapshot_the_actual_sequence_flag():
    cold, cold_shape, _ = native_context(query=8192, history=0)
    continuing, continuing_shape, _ = native_context(query=3056, history=8192)
    assert cold.region_context_for(cold_shape)["prefill_continuations"] == (False,)
    assert cold.region_context_for(cold_shape)["state_fork_srcs"] == (-1,)
    assert continuing.region_context_for(continuing_shape)["prefill_continuations"] == (True,)
    assert continuing.region_context_for(continuing_shape)["state_fork_srcs"] == (3,)


def test_midstep_checkpoint_writes_remain_distinct_from_other_maintenance():
    allocation, shape, _ = native_context(midstep=True)
    context = allocation.region_context_for(shape)
    assert context["state_maintenance_empty"] is True
    assert context["midstep_saves_empty"] is False


def test_prefix_hit_is_distinct_even_when_query_history_and_allocation_match():
    cold, shape, _ = native_context()
    hit, _, _ = native_context(prefix_hit=11248)
    assert cold.region_context_for(shape)["prefix_cache_hit_tokens"] == (0,)
    assert hit.region_context_for(shape)["prefix_cache_hit_tokens"] == (11248,)


def test_request_termination_policy_does_not_change_the_same_native_forward_context():
    original, shape, _ = native_context()
    other, _, _ = native_context(cap=21, ignore_eos=False, stop_strings=["done"])
    assert original.region_context_for(shape) == other.region_context_for(shape)


def test_cleared_or_stale_region_context_is_refused():
    allocation, shape, _ = native_context()
    with pytest.raises(BindRefusal, match="different shape"):
        allocation.region_context_for(StepShape((9,), (11257,), num_prefill_tokens=9))
    allocation.clear()
    with pytest.raises(BindRefusal, match="no native region context"):
        allocation.region_context_for(shape)
