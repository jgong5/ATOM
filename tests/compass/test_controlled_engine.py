"""Cooperative horizons through the actual core, scheduler and prediction runner."""
import json
import queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import MockConfig

from atom.compass.config import CompassConfig
from atom.compass.runtime.controlled_engine import ControlledEngine
from atom.model_engine.engine_core import EngineCore
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, WallClock, get_clock, reset_clock, set_clock
from .test_native_ingress_service import sources, write
from .test_serving_timings import TestPredictedOutputIsDeferred as PredictionFixture


@pytest.fixture(autouse=True)
def clock():
    clock = VirtualClock(epoch=100.)
    set_clock(clock)
    yield clock
    reset_clock()


@pytest.fixture
def make_core(tmp_path):
    count = 0
    def make(*, seconds=1., preparation=None, output_ready=0., controlled=True, fence=False,
             ingress_service=(0., 0.)):
        nonlocal count
        directory = tmp_path / str(count);directory.mkdir();count += 1
        options = sources(directory, writer=(ingress_service[0] * 1e6, 0.),
                          receiver=(ingress_service[1] * 1e6, 0.))
        fit = json.loads((directory / "fit.json").read_text())
        fit["endpoint_summaries"] = [{"serialized_bytes_median": 10}, {"serialized_bytes_median": 100000}]
        options["fit"] = write(directory / "fit.json", fit)
        validation = json.loads((directory / "validation.json").read_text())
        validation["fit_freeze"] = options["fit"]
        options["validation"] = write(directory / "validation.json", validation)
        profile = write(directory / "profile.json", {
            "schema": "compass.request_readiness_profile/1",
            "resolver": "atom.compass.runtime.native_ingress.NativeWriterReceiver",
            "options": options, "source_law": "constant-service CPU fixture",
            "support": {"fixture": True},
            "origin_contract": {"declared_arrival_event": "issue", "writer_eligibility_event": "issue",
                                "transition": "test-only constant service"},
        })
        config = MockConfig(num_kvcache_blocks=128, kv_cache_block_size=4,
            max_num_seqs=4, max_num_batched_tokens=8, max_model_len=64,
            compass_config=CompassConfig(enabled=True, mode="predict",
                request_readiness_profile=profile["path"] if controlled else "",
                prefill_preparation_fence=fence))
        config.hf_config = SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"])
        sched = Scheduler(config)
        runner = PredictionFixture._runner()
        oracle = SimpleNamespace()
        def estimate(shape):
            return SimpleNamespace(seconds=seconds, preparation_seconds=preparation,
                output_ready_seconds=output_ready, output_ready_basis={}, breakdown={})
        oracle.estimate = estimate
        runner._oracle = oracle
        batches, post_calls = [], []
        core = EngineCore.__new__(EngineCore)
        core.label = "controlled CPU fixture"
        core.scheduler = sched
        core.kv_transfer_enabled = False
        core._next_idle_kv_drain = 0.
        core.output_queue, core.stream_output_queue, core.input_queue = queue.Queue(), queue.Queue(), queue.Queue()
        def worker(name, batch, **kwargs):
            assert name == "forward"
            batches.append((get_clock().time(), tuple(batch.req_ids), tuple(batch.num_scheduled_tokens)))
            return runner.forward(batch)
        core.runner_mgr = SimpleNamespace(call_func=worker)
        core.utility_queue = queue.Queue()
        core._has_pending_utility = False
        core.utility_handler = EngineUtilityHandler(
            core.runner_mgr, core.output_queue, scheduler=sched, engine=core)
        postprocess = sched.postprocess
        def post(*args, **kwargs):
            post_calls.append(get_clock().time())
            return postprocess(*args, **kwargs)
        sched.postprocess = post
        return SimpleNamespace(core=core, scheduler=sched, runner=runner, oracle=oracle,
                               batches=batches, posts=post_calls)
    return make


def request(at, *, prompt=4, outputs=1):
    seq = Sequence([1] * prompt, 4, sampling_params=SamplingParams(max_tokens=outputs, ignore_eos=True))
    seq.arrive_time = at
    seq.compass_workload_size = None
    seq.compass_workload_index = None
    return seq


def drain_outputs(core):
    result = []
    while not core.output_queue.empty():
        result.append(core.output_queue.get_nowait())
    return result


def test_pending_forward_survives_multiple_horizons_and_late_submission(make_core, clock):
    fixture = make_core(seconds=10., output_ready=2.)
    control = ControlledEngine(fixture.core)
    first = request(100., prompt=20)
    control.submit_issued(first, 100.)
    # A middle chunk is not deferred: its entire 10 s forward precedes postprocess.
    assert control.advance_until(103.).reason == "horizon"
    assert len(fixture.batches) == 1 and fixture.posts == []
    selected = fixture.batches[0]
    assert first.num_cached_tokens == 0
    child = request(103.)
    control.submit_issued(child, 103.)
    assert not child.block_table and child.num_cached_tokens == 0
    control.advance_until(106.)
    assert fixture.batches == [selected] and fixture.posts == []
    assert not child.block_table and first.num_cached_tokens == 0
    # Half-open: the host-return boundary itself has not executed.
    control.advance_until(110.)
    assert fixture.posts == [] and clock.time() == 110.
    control.advance_until(110., include_horizon=True)
    assert fixture.posts == [110.] and fixture.batches == [selected]
    assert first.num_cached_tokens == 8 and not child.block_table


def test_deferred_completion_is_published_before_trailing_charge(make_core, clock):
    fixture = make_core(seconds=10., output_ready=2.)
    control = ControlledEngine(fixture.core)
    parent = request(100.)
    control.submit_issued(parent, 100.)
    result = control.advance_until(150.)
    assert result.reason == "outputs" and clock.time() == 112.
    assert {e.kind for e in result.output_events} == {"first_token", "completion"}
    assert all(e.at == 112. for e in result.output_events)
    assert parent.first_token_time == parent.finish_time == 112.
    assert result.next_boundary_at == 120.
    assert len(fixture.batches) == 2 and len(fixture.posts) == 2
    observed = drain_outputs(fixture.core)
    assert any(isinstance(row, list) and parent in row for row in observed)
    child = request(112.)
    control.submit_issued(child, 112.)
    control.advance_until(116.)
    assert len(fixture.batches) == 2 and not child.block_table
    control.advance_until(120.)
    assert len(fixture.batches) == 2
    control.advance_until(120., include_horizon=True)
    assert len(fixture.batches) == 2  # finishing the pending step is not a new batch
    assert not drain_outputs(fixture.core)  # publication is not duplicated


def test_pipeline_reservation_and_sample_wait_are_not_repeated(make_core, clock):
    fixture = make_core(seconds=.3, preparation=.001)
    control = ControlledEngine(fixture.core)
    parent = request(100., outputs=3)
    control.submit_issued(parent, 100.)
    control.advance_until(100.15)
    assert len(fixture.batches) == 2 and fixture.posts == [100.]
    timeline = fixture.core._compass_forward_timeline
    reserved_end = timeline.device_end
    child = request(100.15, prompt=20)
    control.submit_issued(child, 100.15)
    control.advance_until(100.25)
    assert len(fixture.batches) == 2 and timeline.device_end == reserved_end
    assert not child.block_table
    control.advance_until(100.3)
    assert parent.first_token_time == 0.
    output = control.advance_until(100.3, include_horizon=True)
    assert output.reason == "outputs"
    assert [(e.kind, e.at) for e in output.output_events] == [("first_token", 100.3)]
    assert parent.first_token_time == clock.time() == 100.3


def test_pipeline_fence_does_not_publish_prefill_progress_before_boundary(make_core, clock):
    fixture = make_core(seconds=1., preparation=.1, fence=True)
    control = ControlledEngine(fixture.core)
    parent = request(100., prompt=20)
    control.submit_issued(parent, 100.)
    control.advance_until(100.05)
    assert len(fixture.batches) == 1 and fixture.posts == []
    assert parent.num_cached_tokens == 0
    control.advance_until(100.1)
    assert parent.num_cached_tokens == 0
    control.advance_until(100.1, include_horizon=True)
    assert parent.num_cached_tokens == 8 and fixture.posts == [100.1]


def test_empty_engine_and_equal_time_issue_do_not_need_a_finite_workload(make_core, clock):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    result = control.advance_until(105.)
    assert result.idle and clock.time() == 105.
    assert not fixture.batches
    seq = request(105.)
    control.submit_issued(seq, 105.)
    control.advance_until(105.)
    assert not fixture.batches and not seq.block_table
    control.advance_until(105., include_horizon=True)
    assert len(fixture.batches) == 1
    assert seq.compass_workload_size is None


@pytest.mark.parametrize("include_horizon", [False, True])
def test_ready_at_horizon_after_idle_obeys_inclusion(make_core, clock, include_horizon):
    fixture = make_core(ingress_service=(2., 3.))
    control = ControlledEngine(fixture.core)
    seq = request(100., prompt=20)
    record = control.submit_issued(seq, 100.)
    assert record.ready_at == 105.

    result = control.advance_until(105., include_horizon=include_horizon)

    assert result.reason == "horizon" and clock.time() == 105.
    assert len(fixture.batches) == int(include_horizon)
    assert fixture.posts == [] and seq.num_cached_tokens == 0
    assert bool(seq.block_table) is include_horizon


@pytest.mark.parametrize("preparation", [None, .1])
def test_legacy_and_controlled_share_batches_tokens_and_final_time(make_core, preparation):
    legacy = make_core(seconds=1., preparation=preparation, controlled=False)
    seq = request(100., outputs=3)
    legacy.scheduler.add(seq)
    while not legacy.scheduler.is_finished():
        legacy.core._process_engine_step_inner()
    old_clock = get_clock().time()
    old_batches = [(at, sizes) for at, _, sizes in legacy.batches]
    old_tokens = list(seq.token_ids)
    old_times = seq.first_token_time, seq.finish_time

    set_clock(VirtualClock(epoch=100.))
    fixture = make_core(seconds=1., preparation=preparation)
    control = ControlledEngine(fixture.core)
    current = request(100., outputs=3)
    control.submit_issued(current, 100.)
    events = []
    for _ in range(30):
        output = control.advance_until(old_clock, include_horizon=True)
        events.extend(output.output_events)
        if output.idle:
            break
    else:
        pytest.fail("controlled fixture did not drain")
    assert [(at, sizes) for at, _, sizes in fixture.batches] == old_batches
    assert list(current.token_ids) == old_tokens
    assert (current.first_token_time, current.finish_time) == old_times
    assert get_clock().time() == old_clock
    assert all(e.at <= old_clock for e in events)


def test_real_forward_exception_propagates_once_and_cannot_resume(make_core):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    control.submit_issued(request(100.), 100.)
    def failing(_):
        raise ValueError("source oracle refused this shape")
    fixture.oracle.estimate = failing
    with pytest.raises(ValueError, match="source oracle refused"):
        control.advance_until(101.)
    assert len(fixture.batches) == 1
    with pytest.raises(RuntimeError, match="failed"):
        control.advance_until(102.)
    assert len(fixture.batches) == 1


def test_stale_actual_completion_stamp_is_refused_without_substitution(make_core):
    fixture = make_core(seconds=10., output_ready=2.)
    control = ControlledEngine(fixture.core)
    seq = request(100.)
    control.submit_issued(seq, 100.)
    seq.finish_time = 99.  # Deliberately malformed existing stamp; never normalized.
    with pytest.raises(ValueError, match="completion.*99.0.*frontier"):
        control.advance_until(150.)
    assert seq.finish_time == 99.


@pytest.mark.parametrize("kind", ["existing_request", "existing_records", "calendar", "wall", "measure", "kv", "speculation", "tp"])
def test_incompatible_core_is_refused(make_core, kind):
    fixture = make_core()
    if kind == "existing_request":
        fixture.scheduler.add(request(100.))
    elif kind == "existing_records":
        fixture.scheduler._request_readiness.begin_issued_requests()
    elif kind == "calendar":
        fixture.scheduler._release_calendar = object()
    elif kind == "wall":
        set_clock(WallClock())
    elif kind == "measure":
        fixture.scheduler.config.compass_config.mode = "measure"
    elif kind == "kv":
        fixture.core.kv_transfer_enabled = True
    elif kind == "speculation":
        fixture.scheduler.config.speculative_config = object()
    else:
        fixture.scheduler.config.tensor_parallel_size = 2
    with pytest.raises(ValueError):
        ControlledEngine(fixture.core)


def test_exclusive_control_and_half_open_argument_contract(make_core):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    with pytest.raises(ValueError):
        ControlledEngine(fixture.core)
    with pytest.raises(RuntimeError, match="owned"):
        fixture.core._process_engine_step_inner()
    with pytest.raises(ValueError, match="frontier"):
        control.submit_issued(request(101.), 101.)
    for value in (99., float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="horizon"):
            control.advance_until(value)
    control.close()
    with pytest.raises(RuntimeError, match="closed"):
        control.advance_until(101.)


def test_admission_rejection_publishes_terminal_time_without_first_token(make_core, clock):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    seq = request(100., prompt=68)
    control.submit_issued(seq, 100.)
    result = control.advance_until(101.)
    assert result.reason == "outputs"
    assert [(e.kind, e.request_id, e.at) for e in result.output_events] == [("completion", seq.id, 100.)]
    assert clock.time() == seq.finish_time == 100.
    assert seq.first_token_time == 0. and seq.num_completion_tokens == 0
    assert not fixture.batches
    assert drain_outputs(fixture.core) == [[seq]]


def test_zero_output_completion_has_no_fabricated_first_token(make_core, clock):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    seq = request(100., outputs=0)
    control.submit_issued(seq, 100.)
    result = control.advance_until(110.)
    assert result.reason == "outputs"
    assert [(e.kind, e.request_id) for e in result.output_events] == [("completion", seq.id)]
    assert seq.num_completion_tokens == 0 and seq.first_token_time == 0.
    assert result.output_events[0].at == seq.finish_time == clock.time()


def test_abort_before_ingress_does_not_find_a_scheduler_request(make_core, clock):
    fixture = make_core(ingress_service=(2., 3.))
    control = ControlledEngine(fixture.core)
    seq = request(100., prompt=20)
    control.submit_issued(seq, 100.)
    control.abort_issued(seq.id, 100.)
    assert seq.status != SequenceStatus.ABORTED

    result = control.advance_until(100., include_horizon=True)

    assert result.reason == "horizon" and not result.output_events
    assert seq.status != SequenceStatus.ABORTED and seq.first_token_time == 0.
    assert seq not in fixture.scheduler.waiting
    assert not result.idle and result.next_boundary_at == 105.
    assert not fixture.batches and not seq.block_table
    assert clock.time() == 100.


def _native_tick(core, events=()):
    """One actual busy-loop step, with external queue arrivals during waits."""
    original_pull = core.pull_and_process_input_queue
    clock = get_clock()
    original_advance = clock.advance
    pending = list(events)
    pulls = 0

    def bounded_pull():
        nonlocal pulls
        pulls += 1
        return True if pulls > 1 else original_pull()

    def advance_with_arrivals(seconds):
        target = clock.time() + seconds
        while pending and pending[0][0] <= target:
            at, action = pending.pop(0)
            original_advance(at - clock.time())
            action()
        original_advance(target - clock.time())

    core.pull_and_process_input_queue = bounded_pull
    core._is_rl_weights_offloaded = False
    core.utility_handler.push_metrics = lambda: None
    try:
        with patch.object(VirtualClock, "advance", lambda self, seconds: advance_with_arrivals(seconds)):
            core.busy_loop()
    finally:
        core.pull_and_process_input_queue = original_pull
    assert not pending


def _native_abort(core, seq):
    core.utility_queue.put_nowait(("abort_request", {"cmd": "abort_request", "req_id": seq.id}))
    core._has_pending_utility = True


def _request_state(seq):
    return (seq.status, seq.leave_reason, seq.first_token_time, seq.finish_time,
            list(seq.token_ids), list(seq.block_table))


@pytest.mark.parametrize("case", ["before_ready", "same_boundary_first_pull", "after_visible"])
def test_abort_visibility_matches_native_input_pull(make_core, case):
    fixture = make_core(seconds=1., ingress_service=(2., 3.))
    control = ControlledEngine(fixture.core)
    seq = request(100., prompt=20, outputs=5)
    assert control.submit_issued(seq, 100.).ready_at == 105.
    at = {"before_ready": 100., "same_boundary_first_pull": 105., "after_visible": 106.}[case]
    if case == "same_boundary_first_pull":
        control.advance_until(105.)
    elif case == "after_visible":
        control.advance_until(106., include_horizon=True)
        assert fixture.batches
    control.abort_issued(seq.id, at)
    control.advance_until(at if case == "before_ready" else at + 1., include_horizon=True)
    observed = _request_state(seq)
    batches = [(at, sizes) for at, _, sizes in fixture.batches]

    set_clock(VirtualClock(epoch=100.))
    native = make_core(seconds=1., controlled=False)
    native_seq = request(100., prompt=20, outputs=5)
    if case != "before_ready":
        get_clock().advance(5.)
        native.core.input_queue.put_nowait([native_seq])
    if case == "after_visible":
        _native_tick(native.core)
    _native_abort(native.core, native_seq)
    _native_tick(native.core)

    assert _request_state(native_seq) == observed
    assert [(at, sizes) for at, _, sizes in native.batches] == batches


def test_abort_after_visibility_in_waiting_matches_native(make_core):
    fixture = make_core(seconds=1.)
    fixture.scheduler.max_num_seqs = 1
    control = ControlledEngine(fixture.core)
    first, waiting = request(100., prompt=20), request(100., prompt=20)
    control.submit_issued(first, 100.)
    control.submit_issued(waiting, 100.)
    control.advance_until(101., include_horizon=True)
    assert waiting in fixture.scheduler.waiting
    control.abort_issued(waiting.id, 101.)
    control.advance_until(102., include_horizon=True)
    observed = _request_state(waiting)
    assert waiting.status == SequenceStatus.ABORTED

    set_clock(VirtualClock(epoch=100.))
    native = make_core(seconds=1., controlled=False)
    native.scheduler.max_num_seqs = 1
    native_first, native_waiting = request(100., prompt=20), request(100., prompt=20)
    native.core.input_queue.put_nowait([native_first, native_waiting])
    _native_tick(native.core)
    assert native_waiting in native.scheduler.waiting
    _native_abort(native.core, native_waiting)
    _native_tick(native.core)
    assert _request_state(native_waiting) == observed


@pytest.mark.parametrize("phase", ["forward", "trailing_output"])
def test_ready_add_during_an_active_step_stays_invisible_to_abort(make_core, phase):
    fixture = make_core(seconds=10., output_ready=2., ingress_service=(2., 3.))
    control = ControlledEngine(fixture.core)
    first = request(100., prompt=20 if phase == "forward" else 4)
    control.submit_issued(first, 100.)
    if phase == "forward":
        control.advance_until(106.)
        issue_at, ready_at, abort_at, end_at = 106., 111., 112., 115.
    else:
        output = control.advance_until(150.)
        assert output.reason == "outputs" and output.now == 117.
        issue_at, ready_at, abort_at, end_at = 117., 122., 123., 125.
    late = request(issue_at, prompt=20)
    assert control.submit_issued(late, issue_at).ready_at == ready_at
    control.advance_until(abort_at)
    control.abort_issued(late.id, abort_at)
    control.advance_until(end_at, include_horizon=True)
    assert late not in fixture.scheduler.waiting and late not in fixture.scheduler.running
    control.advance_until(end_at + 10., include_horizon=True)
    observed = _request_state(late)
    assert late.status != SequenceStatus.ABORTED and late.leave_reason != "aborted"
    batches = [(at, sizes) for at, _, sizes in fixture.batches]

    set_clock(VirtualClock(epoch=105.))
    native = make_core(seconds=10., output_ready=2., controlled=False)
    native_first = request(100., prompt=20 if phase == "forward" else 4)
    native_late = request(issue_at, prompt=20)
    native.core.input_queue.put_nowait([native_first])
    if phase == "trailing_output":
        _native_tick(native.core)
        assert get_clock().time() == 115.
    _native_tick(native.core, events=[
        (ready_at, lambda: native.core.input_queue.put_nowait([native_late])),
        (abort_at, lambda: _native_abort(native.core, native_late)),
    ])
    assert get_clock().time() == end_at
    _native_tick(native.core)
    assert _request_state(native_late) == observed
    assert [(at, sizes) for at, _, sizes in native.batches] == batches


def test_abort_during_forward_waits_for_the_native_whole_step_boundary(make_core):
    fixture = make_core(seconds=10., output_ready=2.)
    control = ControlledEngine(fixture.core)
    seq = request(100., outputs=5)
    control.submit_issued(seq, 100.)
    control.advance_until(101.)
    selected = list(fixture.batches)
    assert fixture.posts == []
    control.abort_issued(seq.id, 101.)
    assert seq.status != SequenceStatus.ABORTED
    control.advance_until(110., include_horizon=True)
    assert seq.status != SequenceStatus.ABORTED
    assert fixture.batches == selected and fixture.posts == [102.]
    result = control.advance_until(150.)
    controlled_tokens = list(seq.token_ids)
    assert seq.leave_reason == "aborted"
    assert seq.finish_time == result.now == 112.
    control.advance_until(120., include_horizon=True)
    controlled_batches = [(at, sizes) for at, _, sizes in fixture.batches]
    assert fixture.scheduler.is_finished()

    set_clock(VirtualClock(epoch=100.))
    legacy = make_core(seconds=10., output_ready=2., controlled=False)
    native = request(100., outputs=5)
    legacy.scheduler.add(native)
    legacy.core._process_engine_step_inner()
    legacy.core.utility_queue.put_nowait(
        ("abort_request", {"cmd": "abort_request", "req_id": native.id}))
    legacy.core._has_pending_utility = True
    legacy.core.utility_handler.process_queue(legacy.core.utility_queue, legacy.core)
    legacy.core._process_engine_step_inner()
    assert native.leave_reason == seq.leave_reason
    assert (native.first_token_time, native.finish_time) == (seq.first_token_time, seq.finish_time)
    assert list(native.token_ids) == controlled_tokens
    assert [(at, sizes) for at, _, sizes in legacy.batches] == controlled_batches
    assert get_clock().time() == 120.


def test_abort_after_output_waits_for_trailing_charge(make_core):
    fixture = make_core(seconds=10., output_ready=2.)
    control = ControlledEngine(fixture.core)
    seq = request(100., outputs=5)
    control.submit_issued(seq, 100.)
    first = control.advance_until(150.)
    assert first.now == 112. and [e.kind for e in first.output_events] == ["first_token"]
    control.abort_issued(seq.id, 112.)
    control.advance_until(116.)
    assert seq.status != SequenceStatus.ABORTED
    control.advance_until(120., include_horizon=True)
    assert seq.status != SequenceStatus.ABORTED
    result = control.advance_until(150.)
    assert [(e.kind, e.at) for e in result.output_events] == [("completion", 122.)]
    assert seq.leave_reason == "aborted"


def test_abort_after_natural_completion_is_a_drained_noop(make_core):
    fixture = make_core(seconds=10., output_ready=2.)
    control = ControlledEngine(fixture.core)
    seq = request(100.)
    control.submit_issued(seq, 100.)
    finished = control.advance_until(150.)
    assert finished.now == 112.
    control.abort_issued(seq.id, 112.)
    drain_outputs(fixture.core)

    result = control.advance_until(150.)

    assert result.idle and not result.output_events
    assert seq.finish_time == 112. and seq.leave_reason != "aborted"
    assert not drain_outputs(fixture.core)
    assert fixture.core.utility_queue.empty()
    assert not fixture.core._has_pending_utility


def test_abort_requires_the_current_frontier(make_core):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    for at in (99., 101., float("nan"), True):
        with pytest.raises(ValueError, match="frontier"):
            control.abort_issued(1, at)
    assert fixture.core.utility_queue.empty()


def test_in_process_startup_reuses_the_real_replay_runner_and_scheduler(tmp_path):
    from atom.compass.replay.local_proc import LocalProcManager
    from atom.compass.replay.runner import ReplayModelRunner
    from atom.model_engine.state_runtime import StateRuntime
    from .test_replay import _target

    target = _target(tmp_path, blocks={
        "num_kvcache_blocks": 128, "pool_entries": {}, "pool_entries_per_req": {},
        "state_runtime": StateRuntime().to_wire(),
    })
    config = MockConfig(
        num_kvcache_blocks=128, kv_cache_block_size=4,
        max_num_seqs=4, max_model_len=64,
        model="Qwen/Qwen3.8-27B", gpu_memory_utilization=.9,
        tensor_parallel_size=1, pipeline_parallel_size=1,
        prefill_context_parallel_size=1, tp_world_size=1,
        parallel_config=SimpleNamespace(control_address=None, data_parallel_size=1),
        compilation_config=None, kv_transfer_config=None, enforce_eager=False,
        disagg_is_decode=False,
        runner_manager_qualname="atom.compass.replay.local_proc.LocalProcManager",
        runner_qualname="atom.compass.replay.runner.ReplayModelRunner",
        compass_config=CompassConfig(enabled=True, mode="predict", epoch=100.,
                                     replay_target=target),
    )
    core = EngineCore(config, None, None, in_process=True)
    try:
        assert type(core.scheduler) is Scheduler
        assert type(core.runner_mgr) is LocalProcManager
        assert type(core.runner_mgr.runner) is ReplayModelRunner
        assert core.input_thread is core.output_thread is None
        assert core.scheduler.block_manager.max_pool_tokens > 0
        assert core.output_queue.get_nowait()[0] == "READY"
        assert get_clock().time() == 100.
    finally:
        core.exit()
    assert core.output_queue.get_nowait()[0].status == SequenceStatus.EXIT_ENGINE


def test_in_process_startup_rejects_non_replay_before_starting_threads(make_core):
    config = make_core().scheduler.config
    config.runner_manager_qualname = "native"
    config.runner_qualname = "native"
    with pytest.raises(ValueError, match="local virtual prediction replay"):
        EngineCore(config, None, None, in_process=True)


def test_control_rejects_an_active_socket_thread(make_core):
    import threading

    fixture = make_core()
    stopped = threading.Event()
    fixture.core.input_thread = threading.Thread(target=stopped.wait)
    fixture.core.input_thread.start()
    try:
        with pytest.raises(ValueError, match="exclusive ownership"):
            ControlledEngine(fixture.core)
    finally:
        stopped.set()
        fixture.core.input_thread.join()


@pytest.mark.parametrize("feature", [
    "tensor_parallel_size", "pipeline_parallel_size", "prefill_context_parallel_size",
    "decode_context_parallel_size", "data_parallel_size", "kv_transfer_config",
    "speculative_config", "disagg_is_decode",
])
def test_in_process_rejects_unsupported_features_before_initialization(make_core, feature):
    config = make_core().scheduler.config
    config.compass_config.replay_target = "/missing-target-must-not-be-opened"
    config.runner_manager_qualname = "atom.compass.replay.local_proc.LocalProcManager"
    config.runner_qualname = "atom.compass.replay.runner.ReplayModelRunner"
    config.parallel_config = SimpleNamespace(control_address=None, data_parallel_size=1)
    if feature == "data_parallel_size":
        config.parallel_config.data_parallel_size = 2
    else:
        setattr(config, feature, True if feature.endswith("_config") or feature == "disagg_is_decode" else 2)
    clock_before = get_clock()

    with pytest.raises(ValueError, match="TP1/PP1/DP1"):
        EngineCore(config, None, None, in_process=True)

    assert get_clock() is clock_before
