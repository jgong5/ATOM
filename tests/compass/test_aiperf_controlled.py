"""Actual asyncio/AIPerf clocks sharing the controlled core's frontier."""
import asyncio
import threading
import time

import pytest

aiperf_clock = pytest.importorskip("aiperf.common.clock")

from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import RequestRecord
from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle

from atom.compass.replay.aiperf_clock import Acknowledgements, ControlledEventLoop, ReplayClock
from atom.compass.runtime.controlled_engine import ControlledEngine
from .test_controlled_engine import clock, make_core, request


def run(control, acknowledgements, coroutine, on_yield=lambda result: None):
    loop = ControlledEventLoop(control, acknowledgements, on_yield)
    try:
        with aiperf_clock.use_clock(ReplayClock(control.clock)):
            return loop.run_until_complete(coroutine)
    finally:
        loop.close()
        control.close()


def test_semantic_stamps_and_absolute_aiperf_timer_use_the_core_clock(make_core):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    native_time = time.time
    native_perf = time.perf_counter

    async def scenario():
        config = CreditPhaseConfig(phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENTIC_REPLAY, concurrency=1,
            expected_duration_sec=5.)
        lifecycle = PhaseLifecycle(config)
        lifecycle.start()
        record = RequestRecord()
        assert record.timestamp_ns == record.start_perf_ns == 100_000_000_000
        assert lifecycle.started_at_ns == lifecycle.started_at_perf_ns == 100_000_000_000
        scheduler = LoopScheduler()
        finished = asyncio.Event()
        observed = []

        async def at_three():
            observed.append(aiperf_clock.perf_counter_ns())
            finished.set()

        scheduler.schedule_at_perf_ns(103_000_000_000, at_three())
        await finished.wait()
        assert observed == [103_000_000_000]
        assert lifecycle.time_left_in_seconds() == 2.
        assert control.clock.time() == 103.
        assert time.time is native_time and time.perf_counter is native_perf

    run(control, Acknowledgements(), scenario())
    assert abs(aiperf_clock.time() - time.time()) < 1.


@pytest.mark.parametrize("kind", ["registration", "external_delivery"])
def test_pending_acknowledgement_prevents_virtual_progress_during_real_io(make_core, kind):
    fixture = make_core()
    control = ControlledEngine(fixture.core)
    acknowledgements = Acknowledgements()
    gate = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        fired = []
        loop.call_later(2., lambda: fired.append(loop.time()))
        key = (kind, 1)
        acknowledgements.expect(key)
        release = threading.Timer(.02, gate.set)
        release.start()
        try:
            await asyncio.to_thread(gate.wait)
        finally:
            release.join()
        assert loop.time() == 100. and not fired
        acknowledgements.acknowledge(key)
        await asyncio.sleep(2.)
        assert fired == [102.] and loop.time() == 102.

    run(control, acknowledgements, scenario())


def test_equal_time_client_timer_and_child_registration_precede_core_postprocess(make_core):
    fixture = make_core(seconds=1.)
    control = ControlledEngine(fixture.core)
    parent = request(100., prompt=20)
    control.submit_issued(parent, 100.)

    async def scenario():
        await asyncio.sleep(1.)
        assert control.clock.time() == 101.
        assert fixture.posts == [] and parent.num_cached_tokens == 0
        child = request(101.)
        control.submit_issued(child, 101.)
        await asyncio.sleep(.5)
        assert fixture.posts[0] == 101.
        assert fixture.batches[1][0] == 101.
        assert child.id not in fixture.batches[1][1]  # Parent uses the 8-token budget.
        await asyncio.sleep(1.)
        assert child.id in fixture.batches[2][1]

    run(control, Acknowledgements(), scenario())


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({f"token{i}": i for i in range(128)}, unk_token="token0"))
    backend.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="token0")
    result.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}token20"
    return result


def _dataset_and_config(tmp_path, *, branch=False, prefill_concurrency=None,
                        start_ratio=None, grace_period=None):
    from aiperf.common.config import EndpointConfig, InputConfig, LoadGeneratorConfig, TokenizerConfig, UserConfig
    from aiperf.common.enums import ConversationContextMode, ConversationBranchMode
    from aiperf.common.models import Conversation, ConversationBranchInfo, DatasetMetadata, Turn
    from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore
    from aiperf.plugin.enums import DatasetSamplingStrategy

    conversation = Conversation(session_id="root", replay_scope_id="root",
        context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
        turns=[Turn(timestamp=index * 100_000, delay=10,
            api_time_ms=10_000, max_tokens=2,
            raw_messages=[{"role": "user", "content": "token10 token11 token12 token13"}])
            for index in range(3)])
    conversations = [conversation]
    if branch:
        conversation.turns[0].branch_ids = ["fanout"]
        conversation.branches = [ConversationBranchInfo(
            branch_id="fanout", child_conversation_ids=["left", "right"],
            mode=ConversationBranchMode.FORK, start_timestamp_ms=10_000)]
        conversations.extend(Conversation(session_id=name, is_root=False, agent_depth=1,
            parent_conversation_id="root", replay_scope_id=name,
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            turns=[Turn(timestamp=10_000, api_time_ms=10_000, max_tokens=2,
                raw_messages=[{"role": "user", "content": "token30 token31"}])])
            for name in ("left", "right"))
    metadata = DatasetMetadata(conversations=[c.metadata() for c in conversations],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL, has_timing_data=True,
        default_context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES)

    async def prepare():
        store = MemoryMapDatasetBackingStore(benchmark_id=tmp_path.name)
        await store.initialize()
        await store.add_conversations({c.session_id: c for c in conversations})
        await store.finalize()
        return store

    (tmp_path / "fixture.json").write_text(conversation.model_dump_json())
    loadgen = {"concurrency": 1, "benchmark_duration": 900,
               "prefill_concurrency": prefill_concurrency}
    if start_ratio is not None:
        loadgen.update(trajectory_start_min_ratio=start_ratio, trajectory_start_max_ratio=start_ratio)
    if grace_period is not None:
        loadgen["benchmark_grace_period"] = grace_period
    config = UserConfig(scenario="inferencex-agentx-mvp",
        endpoint=EndpointConfig(model_names=["fixture"], type="chat", streaming=True,
                                use_server_token_count=True),
        input=InputConfig(file=str(tmp_path / "fixture.json"), custom_dataset_type="weka_trace",
                          random_seed=42),
        tokenizer=TokenizerConfig(name="fixture"),
        loadgen=LoadGeneratorConfig(**loadgen))
    store = asyncio.run(prepare())
    return store, metadata, config


def test_actual_worker_router_chat_and_prediction_core_complete_a_replay(make_core, tmp_path):
    from atom.compass.replay.aiperf_runner import run_controlled_replay

    fixture = make_core(seconds=10.)
    fixture.scheduler.config.hf_config.model_type = "llama"
    fixture.scheduler.config.hf_config.vocab_size = 128
    store, metadata, config = _dataset_and_config(tmp_path)
    try:
        result = run_controlled_replay(core=fixture.core, tokenizer=_tokenizer(),
            user_config=config, dataset_metadata=metadata,
            dataset_client_metadata=store.get_client_metadata())
    finally:
        asyncio.run(store.stop())

    from collections import Counter, defaultdict

    issued = [e for e in result.events if e["kind"] == "credit_issued"]
    returned = [e for e in result.events if e["kind"] == "CreditReturn"]
    # Numerical 10-second steps: sampled prefix, six full three-turn replays,
    # then two turns before the 900-second phase stops issuing.
    assert Counter(e["phase"] for e in issued) == {"warmup": 1, "profiling": 21}
    assert len(result.records) == len(result.dispatches) == len(returned) == 22
    assert not any(e["cancelled"] for e in returned)
    assert result.final_time == 1040.
    assert all(record.error is None for record in result.records)
    _assert_record_core_times(result, config)
    markers = defaultdict(set)
    for row in result.dispatches:
        markers[row["root_correlation_id"]].add(row["cache_bust_marker"])
        # WITH_RESPONSES uses provided history; sampled token100 output does
        # not get inserted into later prompts.
        assert row["prompt_tokens"] == {0: 10, 1: 14, 2: 18}[row["turn_index"]]
    assert len(markers) == 8 and all(len(values) == 1 for values in markers.values())
    assert len({next(iter(values)) for values in markers.values()}) == 8
    assert not any(result.cleanup.values())
    assert result.frontend_service == "unmodelled" and not result.accepted


def _assert_record_core_times(result, config):
    from aiperf.common.models import ModelEndpointInfo
    from aiperf.endpoints.openai_chat import ChatEndpoint

    endpoint = ChatEndpoint(model_endpoint=ModelEndpointInfo.from_user_config(config))
    dispatches = {(d["phase"], d["credit_id"]): d for d in result.dispatches}
    returned = {(e["phase"], e["credit_id"]): e for e in result.events if e["kind"] == "CreditReturn"}
    record_keys = set()
    for record in result.records:
        key = str(record.request_info.credit_phase), record.request_info.credit_num
        assert key not in record_keys
        record_keys.add(key)
        row = dispatches[key]
        meaningful = [m for m in record.responses
            if (parsed := endpoint.parse_response(m)) is not None and parsed.data is not None]
        assert meaningful
        assert record.timestamp_ns == record.start_perf_ns == row["transport_start_ns"]
        assert meaningful[0].perf_ns == round(row["core_first_token_at"] * 1e9)
        assert record.end_perf_ns == round(row["core_completion_at"] * 1e9)
        assert returned[key]["at"] == row["core_completion_at"]
        assert row["registered_at"] == row["io_processor_arrival"]
        assert row["payload_sha256"] and row["prompt_token_sha256"]
    assert record_keys == {key for key, event in returned.items() if not event["cancelled"]}


def test_actual_worker_fork_fanout_completes_with_one_prefill_slot(make_core, tmp_path):
    from atom.compass.replay.aiperf_runner import run_controlled_replay

    fixture = make_core(seconds=10.)
    fixture.scheduler.config.hf_config.model_type = "llama"
    fixture.scheduler.config.hf_config.vocab_size = 128
    store, metadata, config = _dataset_and_config(
        tmp_path, branch=True, prefill_concurrency=1, start_ratio=0., grace_period=1000.)
    try:
        result = run_controlled_replay(core=fixture.core, tokenizer=_tokenizer(),
            user_config=config, dataset_metadata=metadata,
            dataset_client_metadata=store.get_client_metadata())
    finally:
        asyncio.run(store.stop())

    returned = [e for e in result.events if e["kind"] == "CreditReturn"]
    issued = [e for e in result.events if e["kind"] == "credit_issued"]
    first_tokens = [e for e in result.events if e["kind"] == "FirstToken"]
    assert len(result.records) == len(result.dispatches) == len(returned) == len(issued)
    assert len(first_tokens) == len(result.records)
    assert {d["conversation_id"] for d in result.dispatches} == {"root", "left", "right"}
    assert not any(d["phase"] == "warmup" for d in result.dispatches)
    assert not any(e["cancelled"] for e in returned)
    _assert_record_core_times(result, config)
    active = peak = 0
    for event in result.events:
        if event["kind"] == "credit_issued":
            active += 1
            peak = max(peak, active)
        elif event["kind"] == "CreditReturn":
            active -= 1
    assert active == 0 and peak == 2  # One client tree can have multiple requests.
    children = [d for d in result.dispatches if d["agent_depth"]]
    assert len({(d["root_correlation_id"], d["conversation_id"]) for d in children}) == len(children)
    assert any(at <= d["registered_at"] < at + 10. and d["sequence_id"] not in ids
               for d in children for at, ids, _ in fixture.batches)
    assert not any(result.cleanup.values())


def test_actual_grace_expiry_cancels_once_and_drains_late_core_output(make_core, tmp_path):
    from atom.compass.replay.aiperf_runner import run_controlled_replay

    fixture = make_core(seconds=1000.)
    fixture.scheduler.config.hf_config.model_type = "llama"
    fixture.scheduler.config.hf_config.vocab_size = 128
    store, metadata, config = _dataset_and_config(
        tmp_path, prefill_concurrency=1, start_ratio=0.)
    try:
        result = run_controlled_replay(core=fixture.core, tokenizer=_tokenizer(),
            user_config=config, dataset_metadata=metadata,
            dataset_client_metadata=store.get_client_metadata())
    finally:
        asyncio.run(store.stop())

    issued = [e for e in result.events if e["kind"] == "credit_issued"]
    returned = [e for e in result.events if e["kind"] == "CreditReturn"]
    cancelled = [e for e in result.events if e["kind"] == "client_cancelled"]
    completions = [e for e in result.events if e["kind"] == "core_completion"]
    assert len(issued) == len(returned) == len(cancelled) == len(completions) == 1
    assert not result.records
    assert returned[0]["cancelled"] and not returned[0]["first_token_sent"]
    cancel_at = 100. + config.loadgen.benchmark_duration + config.loadgen.benchmark_grace_period
    assert returned[0]["at"] == cancelled[0]["at"] == cancel_at
    assert completions[0]["event_at"] > cancel_at
    assert result.final_time >= completions[0]["event_at"]
    assert not [e for e in result.events if e["kind"] == "FirstToken"]
    assert not any(result.cleanup.values())


def _one_real_worker_request(make_core, tmp_path, cancellation):
    from concurrent.futures import ThreadPoolExecutor
    from aiperf.common.config import ServiceConfig
    from aiperf.common.enums import CacheBustTarget
    from aiperf.common.messages import DatasetConfiguredNotification
    from aiperf.credit.sticky_router import StickyCreditRouter
    from aiperf.credit.structs import Credit
    from aiperf.plugin.enums import TransportType
    from aiperf.workers.worker import Worker
    from atom.compass.replay.aiperf_runner import _serve_with
    from atom.compass.replay.aiperf_transport import (
        PLUGIN_NAME, ControlledServingEngine, InProcessCommunicationConfig,
        ReplayWire, credit_key, register_plugins,
    )
    from atom.entrypoints.openai import api_server

    fixture = make_core(seconds=1.)
    fixture.scheduler.config.hf_config.model_type = "llama"
    fixture.scheduler.config.hf_config.vocab_size = 128
    store, metadata, config = _dataset_and_config(tmp_path, prefill_concurrency=1)
    register_plugins()
    config = config.model_copy(update={"endpoint": config.endpoint.model_copy(
        update={"transport": TransportType(PLUGIN_NAME)})})
    services = ServiceConfig()
    services._comm_config = InProcessCommunicationConfig()
    control = ControlledEngine(fixture.core)
    acknowledgements = Acknowledgements()
    wire = ReplayWire(control, acknowledgements)
    tokenizer = _tokenizer()
    serving = ControlledServingEngine(wire, tokenizer)
    worker_box = []
    cancelled_at_output = []

    def on_output(result):
        serving.core_mgr.on_engine_yield(result)
        if cancellation == "queued_sse" and not cancelled_at_output:
            if any(e.kind == "first_token" for e in result.output_events):
                cancelled_at_output.append(result.now)
                # The native dispatcher already queued the collector delivery;
                # cancel after it runs but before its consumer gets its next turn.
                loop.call_soon(worker_box[0].credit_tasks[7].cancel)

    loop = ControlledEventLoop(control, acknowledgements, on_output)
    gate = threading.Event()

    async def exercise():
        router = StickyCreditRouter(service_config=services, service_id="one-router")
        worker = Worker(service_config=services, user_config=config, service_id="one-worker")
        worker_box.append(worker)
        blocker = None
        try:
            await router.initialize_and_start()
            await worker.initialize_and_start()
            await wire.join_owned()
            with acknowledgements.delivering(("dataset_load", 0)):
                await wire.publish(DatasetConfiguredNotification(service_id="one-dataset",
                    metadata=metadata, client_metadata=store.get_client_metadata()))
            if cancellation == "prepare_twice":
                loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
            credit = Credit(id=7, phase=CreditPhase.PROFILING, conversation_id="root",
                x_correlation_id="one-root", turn_index=0, num_turns=1,
                issued_at_ns=wire.clock.time_ns(), cache_bust_marker="token40",
                cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX)
            await router.send_credit(credit)
            task = worker.credit_tasks[7]
            if cancellation == "prepare_twice":
                while not wire.dispatches:
                    if task.done():
                        pytest.fail("Worker finished before constructing its transport request")
                    await asyncio.sleep(0)
                assert not api_server._stream_loops
                blocker = loop.run_in_executor(None, gate.wait)
                dispatch = wire.dispatches[0]
                while not api_server._stream_loops:
                    if dispatch.prepare.done():
                        await dispatch.prepare
                        pytest.fail("API preparation completed without its streaming registry")
                    await asyncio.sleep(0)
                assert not dispatch.prepare.done()
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not dispatch.prepare.cancelled()
                assert credit_key(credit.phase, credit.id) in acknowledgements.pending
                assert not fixture.batches and loop.time() == 100.
                gate.set()
                await blocker
            await asyncio.gather(task, return_exceptions=True)
            await wire.join_owned()
            await serving.core_mgr.drain()
        finally:
            gate.set()
            await worker.stop()
            while worker.tasks:
                await worker.wait_for_tasks()
            await wire.join_owned()
            await serving.core_mgr.drain()
            await router.stop()
            await wire.join_owned()
        assert not acknowledgements.pending
        assert not serving.io_processor.requests
        assert not api_server._stream_loops and not api_server._seq_id_to_request_id
        assert not api_server._request_start_times
        assert not wire.callback_tasks and not wire.io_tasks
        return wire.events, wire.dispatches

    try:
        with wire.activate(), aiperf_clock.use_clock(wire.clock), _serve_with(serving, tokenizer, "fixture"):
            events, dispatches = loop.run_until_complete(exercise())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            assert not asyncio.all_tasks(loop)
    finally:
        loop.close()
        control.close()
        asyncio.run(store.stop())
    return events, dispatches, cancelled_at_output


def test_cancellation_beats_already_queued_sse_and_first_token(make_core, tmp_path):
    events, dispatches, cancelled_at = _one_real_worker_request(make_core, tmp_path, "queued_sse")
    returned = [e for e in events if e["kind"] == "CreditReturn"]
    first_tokens = [e for e in events if e["kind"] == "FirstToken"]
    assert len(returned) == 1 and returned[0]["cancelled"]
    assert not first_tokens and not returned[0]["first_token_sent"]
    assert returned[0]["at"] == cancelled_at[0] == dispatches[0].evidence["client_cancelled_at"]


def test_repeated_worker_cancel_keeps_real_prepare_owned_until_cleanup(make_core, tmp_path):
    events, dispatches, _ = _one_real_worker_request(make_core, tmp_path, "prepare_twice")
    returned = [e for e in events if e["kind"] == "CreditReturn"]
    assert len(returned) == 1 and returned[0]["cancelled"]
    assert returned[0]["at"] == dispatches[0].evidence["client_cancelled_at"] == 100.
    assert not [e for e in events if e["kind"] == "FirstToken"]
    assert dispatches[0].prepare.done() and not dispatches[0].prepare.cancelled()
    assert dispatches[0].cleanup.done() and not dispatches[0].cleanup.cancelled()


def test_model_failure_keeps_original_exception_without_resuming_forward(make_core, tmp_path):
    from atom.compass.replay.aiperf_runner import run_controlled_replay
    from atom.entrypoints.openai import api_server

    fixture = make_core()
    fixture.scheduler.config.hf_config.model_type = "llama"
    fixture.scheduler.config.hf_config.vocab_size = 128

    def refuse(_):
        raise ValueError("numerical oracle refused the forward")

    fixture.oracle.estimate = refuse
    store, metadata, config = _dataset_and_config(tmp_path, start_ratio=0.)
    try:
        with pytest.raises(ValueError, match="numerical oracle refused") as failure:
            run_controlled_replay(core=fixture.core, tokenizer=_tokenizer(),
                user_config=config, dataset_metadata=metadata,
                dataset_client_metadata=store.get_client_metadata())
    finally:
        asyncio.run(store.stop())

    assert len(fixture.batches) == 1 and not fixture.posts
    assert not getattr(failure.value, "__notes__", [])
    assert api_server.engine is None
    assert not api_server._stream_loops and not api_server._seq_id_to_request_id
    assert not api_server._request_start_times


def test_real_frontend_rejection_cleans_preparation_and_fails_warmup(make_core, tmp_path):
    from atom.compass.replay.aiperf_runner import run_controlled_replay
    from atom.entrypoints.openai import api_server

    fixture = make_core()
    fixture.scheduler.config.hf_config.model_type = "llama"
    fixture.scheduler.config.hf_config.vocab_size = 128
    fixture.scheduler.config.max_model_len = 1
    store, metadata, config = _dataset_and_config(tmp_path, start_ratio=.5)
    try:
        with pytest.raises(RuntimeError, match="profile aborted: warmup_failure") as failed:
            run_controlled_replay(core=fixture.core, tokenizer=_tokenizer(),
                user_config=config, dataset_metadata=metadata,
                dataset_client_metadata=store.get_client_metadata())
    finally:
        asyncio.run(store.stop())

    result = failed.value.replay_result
    assert [e["reason"] for e in result.events if e["kind"] == "ProfileCancel"] == ["warmup_failure"]
    assert not any(e.get("phase") == "profiling" for e in result.events)
    from aiperf.credit.messages import CreditPhaseStartMessage

    assert [message.config.phase for message in result.messages
            if isinstance(message, CreditPhaseStartMessage)] == [CreditPhase.WARMUP]
    assert len(result.dispatches) == 1 and result.dispatches[0]["phase"] == "warmup"
    assert not any(result.cleanup.values()), result.cleanup
    assert not fixture.batches
    assert api_server.engine is None
    assert not api_server._stream_loops and not api_server._seq_id_to_request_id
    assert not api_server._request_start_times


def test_native_serving_bindings_apply_kwargs_and_restore_all_globals(make_core, tmp_path):
    from atom.compass.replay.aiperf_runner import ChatServingOptions, _serve_with
    from atom.compass.replay.aiperf_transport import ControlledServingEngine, ReplayWire
    from atom.entrypoints.openai import api_server

    fixture = make_core()
    fixture.scheduler.config.hf_config.model_type = "llama"
    control = ControlledEngine(fixture.core)
    wire = ReplayWire(control, Acknowledgements())
    tokenizer = _tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        "token20{% if enable_thinking|default(true) %}<think>{% endif %}{# </think> #}")
    serving = ControlledServingEngine(wire, tokenizer)
    keys = ("engine", "tokenizer", "model_name", "_stream_batch_dispatcher",
            "custom_message_encoder", "default_chat_template_kwargs", "reasoning_dialect",
            "model_starts_in_reasoning", "reasoning_toggle", "tool_call_parser_cls")
    before = {key: getattr(api_server, key) for key in keys}
    options = ChatServingOptions(model_path=str(tmp_path),
                                 default_chat_template_kwargs={"enable_thinking": False})
    try:
        with _serve_with(serving, tokenizer, "fixture-alias", options) as receipt:
            assert api_server.reasoning_toggle == ("enable_thinking", False, True)
            rendered = api_server.apply_chat_template(
                tokenizer, api_server.custom_message_encoder,
                [{"role": "user", "content": "token10"}],
                **api_server.default_chat_template_kwargs)
            assert rendered == "token10 token20"
            assert receipt["served_model_name"] == "fixture-alias"
            assert receipt["default_chat_template_kwargs"] == {"enable_thinking": False}
            assert receipt["tokenizer"]["backend_sha256"]
            assert receipt["chat_template_sha256"]
        assert all(getattr(api_server, key) is value for key, value in before.items())
    finally:
        control.close()


def test_actual_sse_reader_default_stays_native_inside_scoped_semantic_clock(clock):
    from aiperf.transports.sse_utils import AsyncSSEStreamReader

    async def chunks():
        yield b"data: {}\n\n"

    async def observe():
        with aiperf_clock.use_clock(ReplayClock(clock)):
            before = time.perf_counter_ns()
            native = await AsyncSSEStreamReader(chunks()).read_complete_stream()
            after = time.perf_counter_ns()
            virtual = await AsyncSSEStreamReader(chunks(), clock=ReplayClock(clock)).read_complete_stream()
        assert before <= native[0].perf_ns <= after
        assert virtual[0].perf_ns == 100_000_000_000

    asyncio.run(observe())


def test_filler_guard_uses_real_incremental_decoding_without_changing_selection(make_core, tmp_path):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    from atom.compass.config import CompassConfig
    from atom.compass.replay.aiperf_runner import (
        ChatServingOptions, _serve_with, validate_replay_filler,
    )
    from atom.compass.replay.aiperf_transport import ControlledServingEngine, ReplayWire

    backend = Tokenizer(BPE({"[UNK]": 0, "§": 1, "z": 2, "x": 3}, [], unk_token="[UNK]"))
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    assert tokenizer.decode([1]) == "\ufffd"

    fixture = make_core()
    fixture.scheduler.config.hf_config.model_type = "llama"
    control = ControlledEngine(fixture.core)
    wire = ReplayWire(control, Acknowledgements())
    serving = ControlledServingEngine(wire, tokenizer)
    try:
        with _serve_with(serving, tokenizer, "byte-fixture",
                         ChatServingOptions(model_path=str(tmp_path))):
            fixture.scheduler.config.compass_config.filler_token_id = 1
            with pytest.raises(ValueError, match="cannot emit immediate decoded text"):
                validate_replay_filler(fixture.core, tokenizer)
            assert fixture.scheduler.config.compass_config.filler_token_id == 1
            fixture.scheduler.config.compass_config.filler_token_id = 3
            witness = validate_replay_filler(fixture.core, tokenizer)
            assert witness["token_id"] == 3
            assert witness["incremental_deltas"] == ["x", "x", "x"]
            assert all(fields for fields in witness["first_token_fields_by_prompt_reasoning"].values())
        assert not fixture.batches and control.clock.time() == 100.
        assert CompassConfig().filler_token_id == 100
    finally:
        control.close()
