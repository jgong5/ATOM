"""Scheduling follows native host returns while queued GPU work continues."""

import queue
from types import SimpleNamespace

import pytest

from conftest import MockConfig
from atom.compass.core.cost.base import StepCost
from atom.compass.runtime.timeline import ForwardTimeline
from atom.model_engine.engine_core import EngineCore
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, get_clock, reset_clock, set_clock

from .test_serving_timings import TestPredictedOutputIsDeferred as _PredictionFixture


@pytest.fixture(autouse=True)
def _clock():
    set_clock(VirtualClock(epoch=100.0))
    yield
    reset_clock()


def test_next_decode_is_scheduled_before_the_previous_prefill_finishes():
    """An arrival during device work must not preempt an already queued decode.

    Real Scheduler, prediction runner and EngineCore handoff; only numerical
    costs and the worker RPC are replaced. Whole-step clock advancement picks
    the long request second and delays the short token across its chunk train.
    """
    class Oracle:
        def estimate(self, shape):
            return SimpleNamespace(
                seconds=0.3 if shape.is_prefill else 0.02,
                preparation_seconds=0.001, breakdown={},
                output_ready_seconds=0.0, output_ready_basis={})

    sched = Scheduler(MockConfig(num_kvcache_blocks=128, kv_cache_block_size=4,
        max_num_seqs=4, max_num_batched_tokens=8, max_model_len=64))
    short = Sequence([1] * 4, 4,
                     sampling_params=SamplingParams(max_tokens=3))
    long = Sequence([1] * 20, 4,
                    sampling_params=SamplingParams(max_tokens=2))
    short.arrive_time, long.arrive_time = 100.0, 100.25
    sched.add(short)
    sched.add(long)
    runner = _PredictionFixture._runner()
    runner._oracle = Oracle()
    rows = []
    runner._record_measurement = lambda shape, *args, **kw: rows.append(shape)
    core = EngineCore.__new__(EngineCore)
    core.scheduler = sched
    core.runner_mgr = SimpleNamespace(call_func=lambda name, batch, **kw:
                                      runner.forward(batch))
    core.kv_transfer_enabled = False
    core._poll_kv_transfer_progress = lambda: None
    core.output_queue, core.stream_output_queue = queue.Queue(), queue.Queue()

    assert core._process_engine_step_inner()
    assert core._process_engine_step_inner()
    assert rows[0].is_prefill
    assert not rows[1].is_prefill
    assert short.first_token_time == pytest.approx(100.3)
    assert get_clock().time() == pytest.approx(100.3)
    assert core._process_engine_step_inner()
    assert rows[2].is_prefill  # the long request is now eligible


def test_cost_without_preparation_does_not_claim_a_pipeline_boundary():
    assert getattr(StepCost(1.0), "preparation_seconds", None) is None


def test_middle_chunks_obey_staging_and_sync_without_draining_sampled_output():
    timeline = ForwardTimeline()
    first = timeline.submit(100.0, 10.0, 1.0, 0.0, True)
    assert first["host_returned_at"] == 100.0
    cold = timeline.submit(100.0, 20.0, 1.0, 0.0, False)
    assert cold["device_started_at"] == 110.0
    assert cold["host_returned_at"] == 101.0  # not the queued device start
    assert timeline.sample_ready == 110.0
    cached = timeline.submit(101.0, 20.0, 1.0, 5.0, False)
    assert cached["host_after_staging_wait"] == 111.0
    assert cached["host_returned_at"] == 135.0
    assert timeline.sample_ready == 110.0
    final = timeline.submit(135.0, 10.0, 1.0, 5.0, True)
    assert final["previous_sample_ready_at"] == 110.0
    assert final["host_returned_at"] == 155.0
    assert timeline.device_end == 160.0  # each device interval once


def test_device_queue_can_drain_during_an_idle_gap():
    timeline = ForwardTimeline()
    timeline.submit(100.0, 1.0, 0.1, 0.0, True)
    step = timeline.submit(200.0, 1.0, 0.1, 0.0, True)
    assert step["device_started_at"] == 200.0
    assert step["host_returned_at"] == 200.0


@pytest.mark.parametrize("seconds,prepare,prefix", [
    (1.0, 2.0, 0.0), (1.0, 0.0, 2.0), (float("nan"), 0.0, 0.0)])
def test_invalid_boundaries_do_not_advance_the_timeline(seconds, prepare, prefix):
    timeline = ForwardTimeline()
    with pytest.raises(ValueError, match="boundaries"):
        timeline.submit(100.0, seconds, prepare, prefix, True)
    assert timeline.device_end is None
