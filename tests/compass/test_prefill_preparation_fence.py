"""The source-witnessed GDN fence reuses the existing pipeline clock once."""

from types import SimpleNamespace

import pytest

from atom.model_engine.engine_core import _advance_native_pipeline
from atom.utils.clock import VirtualClock, WallClock, get_clock, set_clock


@pytest.fixture
def clock():
    old = get_clock()
    value = VirtualClock(epoch=1000.)
    set_clock(value)
    yield value
    set_clock(old)


def core(enabled=True):
    config = SimpleNamespace(
        compass_config=SimpleNamespace(enabled=True, prefill_preparation_fence=enabled),
        hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
        tensor_parallel_size=1, pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1))
    return SimpleNamespace(scheduler=SimpleNamespace(config=config))


def step(*, produces=True, prefix=0., preparation=1.):
    return SimpleNamespace(compass_preparation_seconds=preparation,
        compass_step_seconds=10., compass_output_ready_seconds=prefix,
        compass_produces_output=produces)


PREFILL = SimpleNamespace(total_tokens_num_prefill=10)


def test_first_and_second_prefill_fence_preserves_device_overlap(clock):
    engine = core()
    assert _advance_native_pipeline(engine, step(), PREFILL)
    assert clock.elapsed == 1
    assert engine._compass_forward_timeline.device_end == 1010
    assert _advance_native_pipeline(engine, step(), PREFILL)
    assert clock.elapsed == 11
    assert engine._compass_forward_timeline.device_end == 1020


def test_outputless_middle_chunk_is_fenced_without_replacing_pending_sample(clock):
    engine = core()
    _advance_native_pipeline(engine, step(), PREFILL)
    pending = engine._compass_forward_timeline.sample_ready
    _advance_native_pipeline(engine, step(produces=False), PREFILL)
    assert clock.elapsed == 11 and engine._compass_forward_timeline.device_end == 1020
    assert engine._compass_forward_timeline.sample_ready == pending == 1010


def test_existing_cached_prefix_is_maximized_not_added_to_preparation(clock):
    engine = core()
    _advance_native_pipeline(engine, step(prefix=4.), PREFILL)
    assert clock.elapsed == 4
    assert engine._compass_forward_timeline.device_end == 1010


def test_unselected_and_decode_paths_keep_existing_rules(clock):
    engine = core(False)
    _advance_native_pipeline(engine, step(), PREFILL)
    assert clock.elapsed == 0
    engine = core(True)
    _advance_native_pipeline(engine, step(), SimpleNamespace(total_tokens_num_prefill=0))
    assert clock.elapsed == 0


def test_missing_preparation_and_unproven_model_are_refused(clock):
    with pytest.raises(ValueError, match="priced preparation"):
        _advance_native_pipeline(core(), step(preparation=None), PREFILL)
    engine = core()
    engine.scheduler.config.hf_config.architectures = ["OtherModel"]
    with pytest.raises(ValueError, match="source-proven"):
        _advance_native_pipeline(engine, step(), PREFILL)


def test_real_clock_remains_unchanged():
    old = get_clock()
    set_clock(WallClock())
    try:
        engine = core()
        assert _advance_native_pipeline(engine, step(), PREFILL) is False
        assert not hasattr(engine, "_compass_forward_timeline")
    finally:
        set_clock(old)
