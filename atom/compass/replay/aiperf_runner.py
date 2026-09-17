"""Internal actual-AIPerf composition; no public serving entry point selects it."""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import hashlib
from pathlib import Path

from aiperf.common import clock as aiperf_clock
from aiperf.common import random_generator as rng
from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CommAddress
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.workers.worker import Worker

from atom.compass.replay.aiperf_clock import Acknowledgements, ControlledEventLoop
from atom.compass.prefix_workload import tokenizer_identity
from atom.compass.replay.aiperf_transport import (
    PLUGIN_NAME, ControlledServingEngine, InProcessCommunicationConfig,
    ReplayWire, register_plugins,
)
from atom.compass.runtime.controlled_engine import ControlledEngine
from atom.model_engine.engine_core import EngineCore


@dataclass(frozen=True)
class ChatServingOptions:
    """Native API startup inputs, kept separate from load-generator settings."""

    model_path: str | None = None
    default_chat_template_kwargs: dict = field(default_factory=dict)
    tool_call_parser: str = "auto"


@dataclass
class ReplayResult:
    records: tuple
    events: tuple
    messages: tuple
    dispatches: tuple
    final_time: float
    cleanup: dict = field(default_factory=dict)
    serving: dict = field(default_factory=dict)
    wall_phase_events: tuple = ()
    frontend_service: str = "unmodelled"
    source_qualified: bool = False
    accepted: bool = False


def create_controlled_core(config):
    """Use the same core/runner startup, with its in-process I/O option."""
    return EngineCore(config, None, None, in_process=True)


@contextmanager
def _serve_with(engine, tokenizer, model_name, options=None):
    """Bind the same encoder/reasoning/parser choices as native API startup."""
    from atom.entrypoints.openai import api_server
    from atom.entrypoints.openai.streaming_dispatch import StreamBatchDispatcher

    if api_server.engine is not None:
        raise ValueError("controlled replay requires exclusive ownership of the API module")
    options = options or ChatServingOptions()
    model_path = options.model_path or getattr(engine.config, "model", None) or model_name
    encoder = api_server.load_custom_message_encoder(model_path)
    template = api_server.chat_template_source(tokenizer, encoder)
    dialect, dialect_stated = api_server.resolve_dialect(
        template, api_server.render_probe_prompt(tokenizer, encoder, tools=False) or "")
    changes = {
        "engine": engine, "tokenizer": tokenizer, "model_name": model_name,
        "_stream_batch_dispatcher": StreamBatchDispatcher(tokenizer),
        "custom_message_encoder": encoder,
        "default_chat_template_kwargs": dict(options.default_chat_template_kwargs),
        "reasoning_dialect": dialect,
        "model_starts_in_reasoning": api_server.template_opens_reasoning_implicitly(template),
        "reasoning_toggle": api_server.resolve_reasoning_toggle(tokenizer, encoder),
        "tool_call_parser_cls": api_server.resolve_tool_call_parser(
            options.tool_call_parser, tokenizer, encoder, model=model_path),
    }
    parser = changes["tool_call_parser_cls"]
    dialect_fields = {}
    for item in fields(dialect):
        value = getattr(dialect, item.name)
        dialect_fields[item.name] = (
            f"{value.__module__}.{value.__qualname__}" if callable(value)
            else sorted(value) if isinstance(value, frozenset) else value)
    receipt = {
        "model_path": model_path, "served_model_name": model_name,
        "tokenizer": tokenizer_identity(tokenizer, model_path),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "default_chat_template_kwargs": dict(options.default_chat_template_kwargs),
        "custom_message_encoder": None if encoder is None else {
            "name": encoder.name, "source_path": encoder.source_path,
            "source_sha256": hashlib.sha256(Path(encoder.source_path).read_bytes()).hexdigest()},
        "reasoning_dialect": dialect_fields, "dialect_stated": dialect_stated,
        "model_starts_in_reasoning": changes["model_starts_in_reasoning"],
        "reasoning_toggle": changes["reasoning_toggle"],
        "tool_call_parser_option": options.tool_call_parser,
        "tool_call_parser": None if parser is None else f"{parser.__module__}.{parser.__qualname__}",
    }
    before = {key: getattr(api_server, key) for key in changes}
    for key, value in changes.items():
        setattr(api_server, key, value)
    engine.core_mgr.flush_stream_batch = api_server._stream_batch_dispatcher.flush
    try:
        yield receipt
    finally:
        for key, value in before.items():
            setattr(api_server, key, value)



def validate_replay_filler(core, tokenizer):
    """Refuse surrogate output that would hide first-token visibility."""
    from atom.entrypoints.openai import api_server
    from atom.entrypoints.openai.streaming_dispatch import IncrementalStreamDetokenizer
    from atom.entrypoints.openai.tool_parser import ToolCallStreamParser

    filler = core.scheduler.config.compass_config.filler_token_id
    if (type(filler) is not int or not 0 <= filler < len(tokenizer)
            or filler in tokenizer.all_special_ids):
        raise ValueError("controlled replay filler must be a non-special tokenizer token")
    detokenizer = IncrementalStreamDetokenizer(tokenizer)
    deltas = [detokenizer.update([filler], finished=False) for _ in range(3)]
    if any(not text or "\ufffd" in text for text in deltas):
        raise ValueError(
            f"controlled replay filler {filler} cannot emit immediate decoded text; "
            "choose and pin an explicit displayable filler for this tokenizer")
    visible = {}
    for starts_open in (False, True):
        reasoning = api_server.reasoning_channel(
            starts_open, template_kwargs=api_server.default_chat_template_kwargs).stream()
        tools = ToolCallStreamParser(parser_cls=api_server.tool_call_parser_cls)
        released = []
        for field, text in reasoning.process(deltas[0]):
            if field == "reasoning_content":
                released.append((field, text))
            else:
                released.extend((kind, data) for kind, data in tools.process(text)
                                if kind == "content" and isinstance(data, str))
        if "".join(text for _, text in released) != deltas[0]:
            raise ValueError(
                f"controlled replay filler {filler} is withheld by the configured stream filters")
        visible[str(starts_open)] = released
    return {"token_id": filler, "incremental_deltas": deltas,
            "first_token_fields_by_prompt_reasoning": visible,
            "meaning": "surrogate output only; native generated text is not predicted"}


def run_controlled_replay(
    *, core, tokenizer, user_config, dataset_metadata, dataset_client_metadata,
    service_config=None, worker_count=1, server_options=None,
):
    """Drive actual workers and phase logic against an actual prediction core.

    The caller prepares the real dataset/tokenizer and prediction configuration.
    Request construction and CPU preprocessing execute normally; their wall
    duration is not silently introduced into the model's virtual service law.
    """
    if user_config.endpoint.type != EndpointType.CHAT:
        raise ValueError("controlled replay currently supports the chat endpoint")
    if type(worker_count) is not int or worker_count < 1:
        raise ValueError("worker_count must be positive")
    register_plugins()
    rng.reset()
    rng.init(user_config.input.random_seed)
    endpoint = user_config.endpoint.model_copy(update={"transport": TransportType(PLUGIN_NAME)})
    config = user_config.model_copy(update={"endpoint": endpoint})
    services = (service_config or ServiceConfig()).model_copy(deep=True)
    services._comm_config = InProcessCommunicationConfig()
    control = ControlledEngine(core)
    acknowledgements = Acknowledgements()
    wire = ReplayWire(control, acknowledgements)
    serving = ControlledServingEngine(wire, tokenizer)
    loop = ControlledEventLoop(control, acknowledgements, serving.core_mgr.on_engine_yield)

    async def execute():
        wire.owner_task = asyncio.current_task()
        router = StickyCreditRouter(service_config=services, service_id="controlled-router")
        publisher = PhasePublisher(
            pub_client=router.comms.create_pub_client(CommAddress.EVENT_BUS_PROXY_FRONTEND),
            service_id="controlled-timing")
        workers = []
        try:
            await router.initialize_and_start()
            for index in range(worker_count):
                worker = Worker(service_config=services, user_config=config,
                                service_id=f"controlled-worker-{index}")
                workers.append(worker)
                await worker.initialize_and_start()
            await wire.join_owned()  # WorkerReady deliveries precede credit issuance.
            configured = DatasetConfiguredNotification(
                service_id="controlled-dataset", metadata=dataset_metadata,
                client_metadata=dataset_client_metadata)
            with acknowledgements.delivering(("dataset_load", 0)):
                await wire.publish(configured)
            orchestrator = PhaseOrchestrator(
                config=TimingConfig.from_user_config(config), phase_publisher=publisher,
                credit_router=router, dataset_metadata=dataset_metadata,
                user_config=config)
            wire.orchestrator = orchestrator
            await orchestrator._execute_phases()
            await wire.join_owned()
            await serving.core_mgr.drain()
            for worker in workers:
                if worker.credit_tasks:
                    await asyncio.gather(*tuple(worker.credit_tasks.values()))
        finally:
            wire.tearing_down = True
            try:
                await router.cancel_all_credits()
                for worker in reversed(workers):
                    if not worker.was_stopped:
                        await worker.stop()
                    # Native cancel_all_tasks cancels without joining. Tasks
                    # can create final CreditReturn work while they unwind.
                    while worker.tasks:
                        await worker.wait_for_tasks()
                await wire.join_owned()
                if (control._failed is None and wire.failure is None
                        and not loop.teardown_only):
                    await serving.core_mgr.drain()
            finally:
                await router.stop()
        if wire.failure is not None:
            raise wire.failure
        if acknowledgements.pending:
            raise RuntimeError(f"unsettled controlled deliveries: {acknowledgements.pending}")

    async def finish_owned_work():
        await wire.join_owned()
        if not loop.teardown_only:
            await serving.core_mgr.drain()

    async def teardown_failed_run(main):
        if not main.done():
            main.cancel()
        for task in tuple(wire.callback_tasks):
            task.cancel()
        await asyncio.gather(main, return_exceptions=True)
        # Prepare/cleanup own real executor I/O. Join them before cancelling
        # any remaining component tasks or restoring the API module globals.
        while wire.callback_tasks or wire.io_tasks:
            for task in tuple(wire.callback_tasks):
                task.cancel()
            try:
                await wire.join_owned()
            except BaseException:
                pass  # Preserve the original run failure after owned work settles.
        remaining = asyncio.all_tasks() - {asyncio.current_task()}
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        await loop.shutdown_asyncgens()
        await loop.shutdown_default_executor()

    def snapshot_result(serving_inputs):
        from atom.entrypoints.openai import api_server

        return ReplayResult(
            records=tuple(wire.records), events=tuple(wire.events),
            messages=tuple(wire.messages),
            dispatches=tuple(dict(d.evidence) for d in wire.dispatches),
            final_time=control.clock.time(), serving=serving_inputs,
            wall_phase_events=tuple(wire.wall_phase_events),
            cleanup={
                "io_requests": len(serving.io_processor.requests),
                "stream_loops": len(api_server._stream_loops),
                "request_start_times": len(api_server._request_start_times),
                "sequence_routes": len(api_server._seq_id_to_request_id),
                "stream_callbacks": len(serving.core_mgr._seq_id_to_callback),
                "callback_tasks": len(wire.callback_tasks),
                "prepare_cleanup_tasks": len(wire.io_tasks),
                "pending_deliveries": len(acknowledgements.pending),
                "loop_tasks": len(asyncio.all_tasks(loop)),
            })

    try:
        with wire.activate(), aiperf_clock.use_clock(wire.clock), _serve_with(
            serving, tokenizer, config.endpoint.model_names[0], server_options
        ) as serving_inputs:
            serving_inputs["filler_output"] = validate_replay_filler(core, tokenizer)
            main = loop.create_task(execute())
            try:
                loop.run_until_complete(main)
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(finish_owned_work())
                loop.run_until_complete(loop.shutdown_default_executor())
                result = snapshot_result(serving_inputs)
                if any(result.cleanup.values()):
                    raise RuntimeError(f"controlled replay left live state: {result.cleanup}")
                return result
            except BaseException as original:
                loop.teardown_only = True
                wire.tearing_down = True
                try:
                    loop.run_until_complete(teardown_failed_run(main))
                except BaseException as cleanup_error:
                    original.add_note(f"Controlled replay teardown also failed: {cleanup_error!r}")
                # Keep the original exception type and attach the real protocol
                # trace, including failures that happened before any record.
                failure = (wire.failure if isinstance(original, asyncio.CancelledError)
                           and wire.failure is not None else original)
                failure.replay_result = snapshot_result(serving_inputs)
                if failure is not original:
                    raise failure from original
                raise
    finally:
        loop.close()
        control.close()
