"""Internal protocol adapters from real AIPerf workers to a controlled core."""
import asyncio
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
from itertools import count
import queue
from typing import ClassVar

import orjson
from starlette.requests import Request

from aiperf.common.config import ZMQIPCConfig
from aiperf.common.enums import CommAddress, ServiceRegistrationStatus
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails, RequestRecord, ServiceRunInfo
from aiperf.common.messages import RegisterServiceCommand, CommandAcknowledgedResponse, ProfileCancelCommand
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TransportType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import BaseTransport
from aiperf.transports.sse_utils import AsyncSSEStreamReader
from aiperf.zmq.pub_client import ZMQPubClient
from aiperf.zmq.zmq_defaults import TOPIC_END_LENGTH

from atom.compass.prefix_workload import token_digest
from atom.compass.replay.aiperf_clock import ReplayClock
from atom.compass.runtime.request_readiness import UnsupportedReadiness
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.llm_engine import InputOutputProcessor

_RUNTIME = ContextVar("atomcompass_aiperf_runtime")
_DISPATCH = ContextVar("atomcompass_aiperf_dispatch")
PLUGIN_NAME = "atomcompass_controlled"


def credit_key(phase, number):
    return ("registration", str(phase), number)


@dataclass
class Dispatch:
    request_info: object
    evidence: dict
    sequences: list = field(default_factory=list)
    cancelled: bool = False
    failed: bool = False
    prepare: asyncio.Task | None = None
    consumer: asyncio.Task | None = None
    cleanup: asyncio.Task | None = None


class ReplayWire:
    """Deliver existing protocol messages while tracking their acknowledgements."""

    def __init__(self, control, acknowledgements):
        self.control = control
        self.acknowledgements = acknowledgements
        self.clock = ReplayClock(control.clock)
        self.router_receiver = None
        self.dealer_receivers = {}
        self.subscribers = defaultdict(list)
        self.records = []
        self.messages = []
        self.wall_phase_events = []
        self.events = []
        self.dispatches = []
        self.dispatches_by_credit = {}
        self.callback_tasks = set()
        self.io_tasks = set()
        self.critical_tasks = set()
        self.owner_task = None
        self.orchestrator = None
        self.tearing_down = False
        self.failure = None
        self.registrations = {}
        self._registration_commands = set()
        self._delivery_ids = count()

    def event(self, kind, **fields):
        self.events.append({"kind": kind, "at": self.control.clock.time(), **fields})

    @contextmanager
    def activate(self):
        token = _RUNTIME.set(self)
        try:
            yield self
        finally:
            _RUNTIME.reset(token)

    async def to_worker(self, identity, message):
        key = None
        if isinstance(message, Credit):
            key = credit_key(message.phase, message.id)
            self.acknowledgements.expect(key)
            self.event("credit_issued", phase=str(message.phase), credit_id=message.id,
                       issued_at_ns=message.issued_at_ns,
                       root=message.effective_root_correlation_id)
        try:
            with self.acknowledgements.delivering(("delivery", next(self._delivery_ids))):
                await self.dealer_receivers[identity](message)
        except BaseException:
            if key is not None and key in self.acknowledgements.pending:
                self.acknowledgements.acknowledge(key)
            raise

    def fail(self, error):
        self.failure = self.failure or error
        if (self.owner_task is not None and not self.tearing_down
                and not self.owner_task.done()):
            self.owner_task.cancel()

    def own(self, coroutine, *, callback=False, critical=False):
        task = asyncio.create_task(coroutine)
        (self.callback_tasks if callback else self.io_tasks).add(task)
        if callback or critical:
            self.critical_tasks.add(task)

            def failed(done):
                if not done.cancelled() and done.exception() is not None:
                    self.fail(done.exception())

            task.add_done_callback(failed)
        return task

    async def join_owned(self):
        while self.callback_tasks or self.io_tasks:
            callbacks = tuple(self.callback_tasks)
            tasks = (*callbacks, *tuple(self.io_tasks))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            critical = set(self.critical_tasks)
            self.callback_tasks.difference_update(tasks)
            self.io_tasks.difference_update(tasks)
            self.critical_tasks.difference_update(tasks)
            for task, result in zip(tasks, results):
                if task in critical and isinstance(result, BaseException):
                    raise result

    async def to_router(self, identity, message):
        delivery = ("callback_delivery", next(self._delivery_ids))
        self.acknowledgements.expect(delivery)

        async def receive():
            # Delivery is complete on entry. The native callback may now wait
            # for a timer or first-token capacity that requires core progress.
            self.acknowledgements.acknowledge(delivery)
            if isinstance(message, CreditReturn):
                key = credit_key(message.credit.phase, message.credit.id)
                if (key not in self.dispatches_by_credit
                        and key in self.acknowledgements.pending):
                    self.acknowledgements.acknowledge(key)
            fields = {}
            if isinstance(message, CreditReturn):
                fields = {"phase": str(message.credit.phase), "credit_id": message.credit.id,
                          "cancelled": message.cancelled, "error": message.error,
                          "first_token_sent": message.first_token_sent}
            elif isinstance(message, FirstToken):
                fields = {"phase": str(message.phase), "credit_id": message.credit_id,
                          "ttft_ns": message.ttft_ns}
            self.event(type(message).__name__, worker=identity, **fields)
            await self.router_receiver(identity, message)
            self.event("callback_completed", message_kind=type(message).__name__,
                       worker=identity, **fields)

        task = self.own(receive(), callback=True)

        def delivery_done(_):
            if delivery in self.acknowledgements.pending:
                self.acknowledgements.acknowledge(delivery)

        task.add_done_callback(delivery_done)

    async def publish(self, message):
        self.messages.append(message)
        if str(message.message_type) in ("credit_phase_start", "credit_phase_complete"):
            import time
            self.wall_phase_events.append({
                "message_type": str(message.message_type), "stats": {"phase": str(message.stats.phase)},
                "wall_observed_at": time.time()})
        if isinstance(message, ProfileCancelCommand):
            self.event("ProfileCancel", reason=str(message.reason))
            if message.reason.is_abort:
                self.fail(RuntimeError(f"AIPerf profile aborted: {message.reason}"))
            elif self.owner_task is not None:
                self.owner_task.cancel()
            if self.orchestrator is not None:
                await self.orchestrator.cancel()
            return
        if isinstance(message, RegisterServiceCommand):
            # The internal runner owns these services; acknowledge the same
            # typed startup protocol without launching SystemController's fleet.
            if message.command_id not in self._registration_commands:
                self.registrations[message.service_id] = ServiceRunInfo(
                    registration_status=ServiceRegistrationStatus.REGISTERED,
                    service_type=message.service_type, service_id=message.service_id,
                    state=message.state, first_seen=self.clock.time_ns(),
                    last_seen=self.clock.time_ns())
                self._registration_commands.add(message.command_id)
            await self.publish(CommandAcknowledgedResponse.from_command_message(
                message, "controlled-coordinator"))
            return
        topic = ZMQPubClient._determine_topic(None, message)[:-TOPIC_END_LENGTH]
        for callback in tuple(self.subscribers.get(topic, ())):
            await callback(message)

    def begin_dispatch(self, request_info, payload):
        evidence = {
            "phase": str(request_info.credit_phase),
            "credit_id": request_info.credit_num,
            "request_id": request_info.x_request_id,
            "conversation_id": request_info.conversation_id,
            "turn_index": request_info.turn_index,
            "root_correlation_id": request_info.root_correlation_id,
            "agent_depth": request_info.agent_depth,
            "cache_bust_marker": request_info.cache_bust_marker,
            "cache_bust_target": str(request_info.cache_bust_target),
            "credit_issued_ns": request_info.credit_issued_ns,
            "worker_drop_perf_ns": request_info.drop_perf_ns,
            "transport_start_ns": self.clock.time_ns(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": len(payload),
            "frontend_service": "unmodelled; CPU execution does not advance virtual time",
            "clock": "controlled engine absolute timeline; nanoseconds are a unit projection",
        }
        dispatch = Dispatch(request_info, evidence)
        self.dispatches.append(dispatch)
        self.dispatches_by_credit[credit_key(request_info.credit_phase, request_info.credit_num)] = dispatch
        return dispatch

    def settle_registration(self, dispatch):
        info = dispatch.request_info
        key = credit_key(info.credit_phase, info.credit_num)
        if key in self.acknowledgements.pending:
            self.acknowledgements.acknowledge(key)


class _WireClient(AIPerfLifecycleMixin):
    def __init__(self, wire, kind, address, identity=None):
        super().__init__()
        self.wire, self.kind, self.address, self.identity = wire, kind, address, identity

    def register_receiver(self, callback):
        if self.kind == "router":
            if self.wire.router_receiver is not None:
                raise ValueError("controlled wire already has a credit router")
            self.wire.router_receiver = callback
        elif self.kind == "dealer":
            self.wire.dealer_receivers[self.identity] = callback
        else:
            raise ValueError("receiver registration requires a credit wire client")

    async def send_to(self, identity, message):
        await self.wire.to_worker(identity, message)

    async def send(self, message):
        await self.wire.to_router(self.identity, message)

    async def publish(self, message):
        await self.wire.publish(message)

    async def subscribe(self, topic, callback):
        self.wire.subscribers[str(topic)].append(callback)

    async def subscribe_all(self, subscriptions):
        for topic, callbacks in subscriptions.items():
            for callback in callbacks if isinstance(callbacks, list) else [callbacks]:
                await self.subscribe(topic, callback)

    async def push(self, message):
        if self.address != CommAddress.RAW_INFERENCE_PROXY_FRONTEND:
            raise ValueError(f"unsupported controlled PUSH address: {self.address}")
        self.wire.records.append(message.record)

    async def request(self, *args, **kwargs):
        raise RuntimeError("controlled replay requires the actual configured dataset client")

    async def request_async(self, *args, **kwargs):
        return await self.request(*args, **kwargs)


class InProcessCommunication(AIPerfLifecycleMixin):
    """AIPerf communication plugin using acknowledged in-process delivery."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.wire = _RUNTIME.get()

    def get_address(self, address_type):
        return str(address_type)

    def _client(self, kind, address, identity=None):
        client = _WireClient(self.wire, kind, address, identity)
        self.attach_child_lifecycle(client)
        return client

    def create_pub_client(self, address, **kwargs):
        return self._client("pub", address)

    def create_sub_client(self, address, **kwargs):
        return self._client("sub", address)

    def create_push_client(self, address, **kwargs):
        return self._client("push", address)

    def create_request_client(self, address, **kwargs):
        return self._client("request", address)

    def create_streaming_router_client(self, address, **kwargs):
        if address != CommAddress.CREDIT_ROUTER:
            raise ValueError("controlled router supports only the credit protocol")
        return self._client("router", address)

    def create_streaming_dealer_client(self, address, identity, **kwargs):
        if address != CommAddress.CREDIT_ROUTER:
            raise ValueError("controlled dealer supports only the credit protocol")
        return self._client("dealer", address, identity)


class InProcessCommunicationConfig(ZMQIPCConfig):
    comm_backend: ClassVar[str] = PLUGIN_NAME


class ControlledCoreManager:
    """Use the normal API/IOProcessor contract with an in-process core transport."""

    _deliver_terminal_output = CoreManager._deliver_terminal_output

    def __init__(self, wire):
        self.wire = wire
        self.control = wire.control
        self.label = "ControlledCoreManager"
        self._seq_id_to_callback = {}
        self._dispatches = {}
        self.max_pool_tokens = self.control.scheduler.block_manager.max_pool_tokens
        self.flush_stream_batch = None
        self.engine_idle = True
        self._drained = None

    def add_request(self, sequences):
        dispatch = _DISPATCH.get()
        if len(sequences) != 1:
            raise ValueError("controlled transport currently supports one sequence per request")
        seq = sequences[0]
        dispatch.sequences.append(seq)
        from atom.entrypoints.openai import api_server
        dispatch.evidence["api_request_id"] = api_server._seq_id_to_request_id[seq.id]
        # CoreManager retains callbacks on the frontend before serializing
        # the ADD frame. Readiness describes that same callback-free frame.
        if seq.stream_callback is not None:
            self._seq_id_to_callback[seq.id] = seq.stream_callback
            seq.stream_callback = None
        try:
            record = self.control.submit_issued(seq, self.control.clock.time())
        except BaseException as exc:
            self._seq_id_to_callback.pop(seq.id, None)
            if isinstance(exc, UnsupportedReadiness):
                self.wire.fail(exc)
            raise
        dispatch.evidence.update({
            "sequence_id": seq.id,
            "io_processor_arrival": seq.arrive_time,
            "request_ready_at": record.ready_at,
            "prompt_token_sha256": token_digest(seq.token_ids),
            "prompt_tokens": seq.num_prompt_tokens,
            "max_output_tokens": seq.max_tokens,
            "registered_at": self.control.clock.time(),
        })
        self._dispatches[seq.id] = dispatch
        self.engine_idle = False
        if not dispatch.cancelled:
            self.wire.settle_registration(dispatch)

    def abort_request(self, request_id):
        dispatch = self._dispatches.get(request_id)
        if dispatch is not None:
            dispatch.cancelled = True
        self._seq_id_to_callback.pop(request_id, None)
        self.control.abort_issued(request_id, self.control.clock.time())
        self.engine_idle = False

    def on_engine_yield(self, result):
        for event in result.output_events:
            dispatch = self._dispatches.get(event.request_id)
            if dispatch is not None:
                dispatch.evidence["core_" + event.kind + "_at"] = event.at
            self.wire.event("core_" + event.kind, request_id=event.request_id,
                            event_at=event.at)
        while True:
            try:
                item = self.control.core.output_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                if item[0] == "READY":
                    continue
                if item[0] != "STREAM":
                    raise ValueError(f"unexpected controlled core output: {item[0]}")
                for seq_id, output in item[1]:
                    callback = self._seq_id_to_callback.get(seq_id)
                    dispatch = self._dispatches.get(seq_id)
                    if callback is not None and not (dispatch and dispatch.cancelled):
                        callback(output)
                    if output.finished:
                        self._seq_id_to_callback.pop(seq_id, None)
            else:
                for seq in item:
                    self._deliver_terminal_output(seq)
                    self._dispatches.pop(seq.id, None)
        if self.flush_stream_batch is not None:
            self.flush_stream_batch()
        self.engine_idle = result.idle
        if self.engine_idle and self._drained is not None:
            self._drained.set()

    async def drain(self):
        self._drained = asyncio.Event()
        if self.engine_idle and not self.control.core._has_pending_utility:
            return
        await self._drained.wait()


class ControlledServingEngine:
    def __init__(self, wire, tokenizer):
        self.config = wire.control.scheduler.config
        self.io_processor = InputOutputProcessor(self.config, tokenizer, self.config.kv_cache_block_size)
        self.core_mgr = ControlledCoreManager(wire)


class ControlledTransport(BaseTransport):
    """Run ATOM's actual chat handler and SSE parser over the controlled core."""

    def __init__(self, model_endpoint, **kwargs):
        super().__init__(model_endpoint, **kwargs)
        self.wire = _RUNTIME.get()
        self.clock = self.wire.clock

    @classmethod
    def metadata(cls):
        return TransportMetadata(url_schemes=["atomcompass"])

    def get_url(self, request_info):
        return self.model_endpoint.endpoint.base_url

    def _cancel(self, dispatch):
        dispatch.cancelled = True
        if "client_cancelled_at" not in dispatch.evidence:
            dispatch.evidence["client_cancelled_at"] = self.clock.time()
            self.wire.event("client_cancelled", phase=str(dispatch.request_info.credit_phase),
                            credit_id=dispatch.request_info.credit_num)
        if dispatch.consumer is not None and not dispatch.consumer.done():
            dispatch.consumer.cancel()

    def _cleanup(self, dispatch):
        if dispatch.cleanup is None:
            dispatch.cleanup = self.wire.own(self._cleanup_dispatch(dispatch), critical=True)
        return dispatch.cleanup

    async def _cleanup_dispatch(self, dispatch):
        from atom.entrypoints.openai import api_server

        response = None
        try:
            response = await dispatch.prepare
        except Exception as exc:
            dispatch.failed = True
            dispatch.evidence["preparation_error"] = type(exc).__name__
        if dispatch.consumer is not None:
            if not dispatch.consumer.done():
                dispatch.consumer.cancel()
            await asyncio.gather(dispatch.consumer, return_exceptions=True)
        if response is not None and hasattr(response.body_iterator, "aclose"):
            await response.body_iterator.aclose()
        for seq in dispatch.sequences:
            registered = seq.id in api_server.engine.core_mgr._dispatches
            if (seq.id in api_server._seq_id_to_request_id
                    or seq.id in api_server.engine.io_processor.requests):
                api_server.cleanup_stream(
                    seq.id, aborted=registered and (dispatch.cancelled or dispatch.failed))
        api_request_id = dispatch.evidence.get("api_request_id")
        if api_request_id is not None:
            api_server.cleanup_request(api_request_id)
        self.wire.settle_registration(dispatch)

    async def send_request(self, request_info, payload, *, first_token_callback=None):
        from atom.entrypoints.openai import api_server
        from atom.entrypoints.openai.protocol import ChatCompletionRequest

        if not isinstance(payload, bytes):
            raise ValueError("controlled transport requires InferenceClient's canonical payload bytes")
        request = ChatCompletionRequest.model_validate_json(payload)
        if not request.stream or request.n not in (None, 1):
            raise ValueError("controlled transport requires one streaming chat response")
        dispatch = self.wire.begin_dispatch(request_info, payload)
        token = _DISPATCH.set(dispatch)
        record = RequestRecord(timestamp_ns=self.clock.time_ns(),
                               start_perf_ns=self.clock.perf_counter_ns())
        raw = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
            "headers": [(k.lower().encode(), v.encode()) for k, v in self.build_headers(request_info).items()],
            "query_string": b"", "server": ("controlled", 80), "scheme": "http"})
        dispatch.prepare = self.wire.own(api_server.chat_completions(request, raw))
        try:
            response = await asyncio.shield(dispatch.prepare)
            record.status = response.status_code
            record.recv_start_perf_ns = self.clock.perf_counter_ns()

            async def chunks():
                async for chunk in response.body_iterator:
                    yield chunk.encode() if isinstance(chunk, str) else chunk

            async def consume():
                first_sent = False
                async for message in AsyncSSEStreamReader(chunks(), clock=self.clock):
                    if dispatch.cancelled:
                        raise asyncio.CancelledError
                    AsyncSSEStreamReader.inspect_message_for_error(message)
                    record.responses.append(message)
                    if first_token_callback is not None and not first_sent:
                        first_sent = await first_token_callback(
                            message.perf_ns - record.start_perf_ns, message)

            dispatch.consumer = self.wire.own(consume())
            if request_info.cancel_after_ns is None:
                # Cancellation must reach a runnable SSE consumer immediately.
                # Only preprocessing and cleanup outlive the Worker task.
                await dispatch.consumer
            else:
                try:
                    await asyncio.wait_for(dispatch.consumer, request_info.cancel_after_ns / 1e9)
                except asyncio.TimeoutError:
                    self._cancel(dispatch)
                    record.cancellation_perf_ns = self.clock.perf_counter_ns()
                    record.error = ErrorDetails(type="RequestCancellationError",
                        message="Request cancelled after the configured delay", code=499)
            record.end_perf_ns = self.clock.perf_counter_ns()
            return record
        except asyncio.CancelledError:
            self._cancel(dispatch)
            raise
        except BaseException:
            dispatch.failed = True
            raise
        finally:
            cleanup = self._cleanup(dispatch)
            try:
                if not dispatch.cancelled:
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        self._cancel(dispatch)
                        raise
            finally:
                _DISPATCH.reset(token)


def register_plugins():
    known = {entry.name for entry in plugins.list_entries(PluginType.TRANSPORT)}
    if PLUGIN_NAME not in known:
        plugins.register(PluginType.TRANSPORT, PLUGIN_NAME, ControlledTransport,
                         metadata={"url_schemes": ["atomcompass"]})
    if PLUGIN_NAME not in TransportType.values():
        TransportType.register("ATOMCOMPASS_CONTROLLED", PLUGIN_NAME)
    known = {entry.name for entry in plugins.list_entries(PluginType.COMMUNICATION)}
    if PLUGIN_NAME not in known:
        plugins.register(PluginType.COMMUNICATION, PLUGIN_NAME, InProcessCommunication)
