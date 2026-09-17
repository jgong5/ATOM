"""Resolve registered requests into scheduler-visible events, without scheduling.

The optional profile selects a source-service provider. No service law or
production profile is supplied here: declared arrival, writer eligibility and
native receiver visibility must be connected by that provider's source evidence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import math
import os
import pickle
from types import MappingProxyType

from atom.compass.core.loaded_input import LoadedInput, load_json, manifest
from atom.model_engine.engine_core_protocol import EngineCoreRequestType


class UnsupportedReadiness(ValueError):
    """The selected source service cannot describe this request workload."""


@dataclass(frozen=True)
class IngressDescriptor:
    """Logical single-request ADD bytes reconstructed at registration.

    These are not observed wire bytes or wall-clock receipt data. The envelope
    and protocol match CoreManager.add_request for one ordinary token request.
    A selected source provider must qualify this representation and its range.
    """

    reconstructed_add_bytes: int
    pickle_protocol: int
    frame_request_count: int
    token_typecode: str
    token_itemsize: int
    token_bytes: int
    has_per_req_cache: bool


def ingress_descriptor(sequence) -> IngressDescriptor:
    unsupported = (
        "multimodal_data", "kv_transfer_params", "kv_transfer_params_output",
        "stream_callback", "stop_strings", "stop_token_sequences", "return_logprobs",
        "num_draft_tokens", "needs_independent_noise", "sibling_index",
        "dp_session_id", "dp_parent_session_id",
    )
    if any(getattr(sequence, name, None) for name in unsupported):
        raise UnsupportedReadiness("readiness source layout excludes expanded request metadata")
    if (getattr(sequence, "mrope_positions", None) is not None
            or sequence.num_completion_tokens or sequence.num_cached_tokens
            or sequence.block_table or sequence.state_slots):
        raise UnsupportedReadiness("readiness requires a new token-only request")
    tokens = sequence.token_ids
    if getattr(tokens, "typecode", None) != "i" or tokens.itemsize != 4:
        raise UnsupportedReadiness("readiness source layout requires native int32 token storage")
    payload = pickle.dumps((EngineCoreRequestType.ADD, [sequence]),
                           protocol=pickle.DEFAULT_PROTOCOL)
    return IngressDescriptor(len(payload), pickle.DEFAULT_PROTOCOL, 1,
                             tokens.typecode, tokens.itemsize,
                             len(tokens) * tokens.itemsize,
                             bool(sequence.has_per_req_cache))


@dataclass(frozen=True)
class RegisteredRequest:
    request_id: int | str
    arrived_at: float
    prompt_tokens: int
    workload_index: int | None
    ingress: IngressDescriptor


@dataclass(frozen=True)
class ReadyEvent:
    ready_at: float
    receipt_order: int
    source_service_started_at: float | None = None


@dataclass(frozen=True)
class RequestReadinessRecord:
    arrived_at: float
    ready_at: float
    receipt_order: int
    source_service_started_at: float | None = None


class RequestReadiness:
    """Core-owned provider and once-resolved immutable request records.

    A provider implements ``resolve_closed_workload(tuple[RegisteredRequest])``
    and returns ``{request_id: ReadyEvent}``. It owns service/coverage checks and
    native writer/receiver overlap; this boundary never advances time, chooses
    a batch, rewrites an arrival or assumes a preprocessing queue topology.
    Explicit issued-request streams use the provider's persistent release queue
    without registering future requests.
    """

    def __init__(self, profile_path: str):
        payload, profile_input = load_json(
            profile_path, role="runtime.request_readiness.profile")
        if payload.get("schema") != "compass.request_readiness_profile/1":
            raise UnsupportedReadiness("unsupported request readiness profile schema")
        origin = payload.get("origin_contract", {})
        for name in ("declared_arrival_event", "writer_eligibility_event", "transition"):
            if not isinstance(origin.get(name), str) or not origin[name].strip():
                raise UnsupportedReadiness(f"readiness origin contract needs {name}")
        if not payload.get("support") or not payload.get("source_law"):
            raise UnsupportedReadiness("readiness profile needs support and source_law")
        self._metadata = copy.deepcopy({
            key: payload[key] for key in ("origin_contract", "support", "source_law")})
        qualname = payload.get("resolver", "")
        module_name, _, attribute = qualname.rpartition(".")
        if not module_name or not attribute:
            raise UnsupportedReadiness("readiness profile needs a resolver qualname")
        factory = getattr(importlib.import_module(module_name), attribute)
        self._provider = factory(**payload.get("options", {}))
        if not callable(getattr(self._provider, "resolve_closed_workload", None)):
            raise UnsupportedReadiness("readiness resolver has no closed-workload method")
        provider_inputs = tuple(self._provider.loaded_inputs)
        if not all(isinstance(item, LoadedInput) for item in provider_inputs):
            raise UnsupportedReadiness("readiness resolver must report its loaded inputs")
        self._inputs = (profile_input, *provider_inputs)
        self._resolver = qualname
        self._reader_pid = os.getpid()
        self.records = None
        self._serial_requests = None
        self._causal_requests = None
        self._causal_queue = None
        self._causal_records = None
        self._issued_requests = None
        self._issued_queue = None
        self._issued_records = None

    def begin_issued_requests(self) -> None:
        """Open an empty ingress stream; future requests have no descriptor or demand.

        The caller owns issue ordering and logical time. This mode neither
        declares a final workload size nor advances the engine clock.
        """
        if (self.records is not None or self._serial_requests is not None
                or self._causal_requests is not None or self._issued_requests is not None):
            raise UnsupportedReadiness("request readiness was already registered")
        factory = getattr(self._provider, "new_release_queue", None)
        if not callable(factory):
            raise UnsupportedReadiness("source provider does not qualify persistent issued ingress")
        queue = factory()
        if not callable(getattr(queue, "resolve_release", None)):
            raise UnsupportedReadiness("issued source queue has no release method")
        self._issued_requests, self._issued_queue = {}, queue
        self._issued_records = {}
        self.records = MappingProxyType(self._issued_records)

    def admit_issued_request(self, sequence, issued_at: float) -> RequestReadinessRecord:
        """Resolve one actual issue through the existing persistent source queue.

        Stamp sequence.arrive_time before calling; this method never changes
        the sequence. Equal issue times retain call order. Recycled requests
        need fresh request IDs, independently of their original trace indices.
        """
        if self._issued_requests is None:
            raise UnsupportedReadiness("issued request stream was not initialized")
        if sequence.id in self._issued_requests:
            raise UnsupportedReadiness("issued request identity is duplicated")
        if type(issued_at) not in (int, float):
            raise UnsupportedReadiness("issued request time must be finite numeric time")
        try:
            issued_at = float(issued_at)
        except OverflowError as exc:
            raise UnsupportedReadiness("issued request time must be finite numeric time") from exc
        if (not math.isfinite(issued_at) or type(sequence.arrive_time) not in (int, float)
                or sequence.arrive_time != issued_at):
            raise UnsupportedReadiness("issued request time must be finite and match its declared arrival")
        previous = next(reversed(self._issued_records.values())) if self._issued_records else None
        if previous is not None and issued_at < previous.arrived_at:
            raise UnsupportedReadiness("issued requests must be chronological")
        request = RegisteredRequest(
            sequence.id, issued_at, int(sequence.num_prompt_tokens),
            getattr(sequence, "compass_workload_index", None), ingress_descriptor(sequence))
        event = self._issued_queue.resolve_release(request)
        if (not isinstance(event, ReadyEvent) or not math.isfinite(event.ready_at)
                or event.source_service_started_at is None or not math.isfinite(event.source_service_started_at)
                or not issued_at <= event.source_service_started_at <= event.ready_at
                or (previous is not None and event.ready_at < previous.ready_at)
                or type(event.receipt_order) is not int or event.receipt_order != len(self.records)):
            raise UnsupportedReadiness("issued source queue returned invalid service or receipt order")
        record = RequestReadinessRecord(
            issued_at, event.ready_at, event.receipt_order, event.source_service_started_at)
        self._issued_requests[sequence.id] = request
        self._issued_records[sequence.id] = record
        return record

    def register_causal_workload(self, sequences):
        """Pin all requests without charging service before calendar release."""
        if self.records is not None:
            raise UnsupportedReadiness("request readiness was already registered")
        factory = getattr(self._provider, "new_release_queue", None)
        if not callable(factory):
            raise UnsupportedReadiness("source provider does not qualify persistent causal ingress")
        sequences = tuple(sequences)
        if not sequences or any(seq.compass_workload_size != len(sequences) for seq in sequences):
            raise UnsupportedReadiness("causal readiness requires complete registration")
        requests = {seq.id: RegisteredRequest(
            seq.id, float(seq.arrive_time), int(seq.num_prompt_tokens),
            seq.compass_workload_index, ingress_descriptor(seq)) for seq in sequences}
        if (len(requests) != len(sequences)
                or any(not math.isfinite(r.arrived_at) for r in requests.values())
                or {r.workload_index for r in requests.values()} != set(range(len(sequences)))):
            raise UnsupportedReadiness("causal requests need unique identities and complete indices")
        queue = factory()
        if not callable(getattr(queue, "resolve_release", None)):
            raise UnsupportedReadiness("causal source queue has no release method")
        self._causal_requests, self._causal_queue = requests, queue
        self._causal_records = {}
        self.records = MappingProxyType(self._causal_records)

    def release_causal_request(self, sequence, arrived_at):
        if self._causal_requests is None or sequence.id not in self._causal_requests:
            raise UnsupportedReadiness("causal request was not registered")
        original = self._causal_requests[sequence.id]
        if (sequence.id in self.records or not math.isfinite(arrived_at)
                or arrived_at < original.arrived_at):
            raise UnsupportedReadiness("causal release is duplicated or precedes its source timestamp")
        event = self._causal_queue.resolve_release(RegisteredRequest(
            original.request_id, arrived_at, original.prompt_tokens, original.workload_index, original.ingress))
        previous_ready = next(reversed(self._causal_records.values())).ready_at if self.records else -math.inf
        if (not isinstance(event, ReadyEvent) or not math.isfinite(event.ready_at)
                or event.source_service_started_at is None or not math.isfinite(event.source_service_started_at)
                or not arrived_at <= event.source_service_started_at <= event.ready_at
                or event.ready_at < previous_ready or type(event.receipt_order) is not int
                or event.receipt_order != len(self.records)):
            raise UnsupportedReadiness("causal source queue returned invalid service or receipt order")
        self._causal_records[sequence.id] = RequestReadinessRecord(
            arrived_at, event.ready_at, event.receipt_order, event.source_service_started_at)

    def register_serial_workload(self, sequences):
        """Pin descriptors while leaving unreleased requests outside all service."""
        if self.records is not None or self._serial_requests is not None:
            raise UnsupportedReadiness("request readiness was already registered")
        if not callable(getattr(self._provider, "resolve_serial_release", None)):
            raise UnsupportedReadiness("source provider does not qualify serial causal releases")
        sequences = tuple(sequences)
        if len(sequences) != 2 or any(seq.compass_workload_size != 2 for seq in sequences):
            raise UnsupportedReadiness("serial opening requires complete two-request registration")
        self._serial_requests = {
            seq.id: RegisteredRequest(seq.id, float(seq.arrive_time), int(seq.num_prompt_tokens),
                                      seq.compass_workload_index, ingress_descriptor(seq))
            for seq in sequences
        }
        if len(self._serial_requests) != 2:
            raise UnsupportedReadiness("serial opening request identities must be unique")
        self.records = MappingProxyType({})

    def release_serial_request(self, sequence, arrived_at):
        """Start source service only once its predecessor has completed."""
        if self._serial_requests is None or sequence.id not in self._serial_requests:
            raise UnsupportedReadiness("serial request was not registered")
        if sequence.id in self.records or not math.isfinite(arrived_at):
            raise UnsupportedReadiness("serial release is duplicated or non-finite")
        original = self._serial_requests[sequence.id]
        if arrived_at < original.arrived_at:
            raise UnsupportedReadiness("causal release precedes the source timestamp")
        request = RegisteredRequest(original.request_id, arrived_at, original.prompt_tokens,
                                    original.workload_index, original.ingress)
        event = self._provider.resolve_serial_release(request)
        if (not isinstance(event, ReadyEvent) or not math.isfinite(event.ready_at)
                or event.ready_at < arrived_at
                or event.source_service_started_at != arrived_at):
            raise UnsupportedReadiness("serial source service returned an invalid ready event")
        record = RequestReadinessRecord(arrived_at, event.ready_at, len(self.records),
                                        event.source_service_started_at)
        self.records = MappingProxyType({**self.records, sequence.id: record})

    def resolve_closed_workload(self, sequences) -> None:
        if self.records is not None:
            raise UnsupportedReadiness("request readiness was already resolved")
        sequences = tuple(sequences)
        if not sequences or any(
            seq.compass_workload_size != len(sequences) for seq in sequences
        ):
            raise UnsupportedReadiness("readiness requires a complete declared workload")
        requests = tuple(RegisteredRequest(
            seq.id, float(seq.arrive_time), int(seq.num_prompt_tokens),
            getattr(seq, "compass_workload_index", None), ingress_descriptor(seq))
            for seq in sequences)
        ids = {request.request_id for request in requests}
        if len(ids) != len(requests):
            raise UnsupportedReadiness("readiness requests need unique identities")
        if any(not math.isfinite(request.arrived_at) for request in requests):
            raise UnsupportedReadiness("declared arrivals must be finite")
        events = self._provider.resolve_closed_workload(requests)
        if set(events) != ids:
            raise UnsupportedReadiness("readiness resolver did not cover every request")
        records = {}
        orders = set()
        for request in requests:
            event = events[request.request_id]
            if not isinstance(event, ReadyEvent):
                raise UnsupportedReadiness("readiness resolver must return ReadyEvent records")
            if not math.isfinite(event.ready_at) or event.ready_at < request.arrived_at:
                raise UnsupportedReadiness("ready_at must be finite and no earlier than arrival")
            if (type(event.receipt_order) is not int or event.receipt_order < 0
                    or event.receipt_order in orders):
                raise UnsupportedReadiness("native receipt orders must be unique nonnegative integers")
            orders.add(event.receipt_order)
            records[request.request_id] = RequestReadinessRecord(
                request.arrived_at, event.ready_at, event.receipt_order)
        ordered = sorted(records.values(), key=lambda record: record.receipt_order)
        if any(after.ready_at < before.ready_at for before, after in zip(ordered, ordered[1:])):
            raise UnsupportedReadiness("ready_at contradicts single-receiver receipt order")
        self.records = MappingProxyType(records)

    def record(self, sequence) -> RequestReadinessRecord:
        if self.records is None or sequence.id not in self.records:
            raise UnsupportedReadiness("request has no resolved readiness event")
        record = self.records[sequence.id]
        if sequence.arrive_time != record.arrived_at:
            raise UnsupportedReadiness("declared arrival changed after readiness resolution")
        return record

    def input_manifest(self, extra_inputs=()) -> dict:
        result = {
            **manifest((*self._inputs, *extra_inputs)),
            "reader": {"component": "EngineCore.Scheduler", "pid": self._reader_pid},
            "request_readiness": {
                "resolver": self._resolver,
                **copy.deepcopy(self._metadata),
                "resolved_requests": len(self.records) if self.records is not None else 0,
            },
        }
        if self._serial_requests is not None:
            result["request_readiness"]["serial_releases"] = [
                {"seq_id": str(request_id), "arrived_at": record.arrived_at,
                 "source_service_started_at": record.source_service_started_at,
                 "ready_at": record.ready_at, "receipt_order": record.receipt_order}
                for request_id, record in self.records.items()
            ]
        if self._causal_requests is not None:
            from dataclasses import asdict
            result["request_readiness"]["causal_releases"] = [
                {"seq_id": str(request_id), "arrived_at": record.arrived_at,
                 "source_service_started_at": record.source_service_started_at,
                 "ready_at": record.ready_at, "receipt_order": record.receipt_order,
                 "ingress": asdict(self._causal_requests[request_id].ingress)}
                for request_id, record in self.records.items()]
            if callable(getattr(self._causal_queue, "evidence", None)):
                result["request_readiness"]["causal_queue"] = self._causal_queue.evidence()
        if self._issued_requests is not None:
            from dataclasses import asdict
            result["request_readiness"]["issued_releases"] = [
                {"seq_id": str(request_id), "arrived_at": record.arrived_at,
                 "source_service_started_at": record.source_service_started_at,
                 "ready_at": record.ready_at, "receipt_order": record.receipt_order,
                 "ingress": asdict(self._issued_requests[request_id].ingress)}
                for request_id, record in self.records.items()]
            if callable(getattr(self._issued_queue, "evidence", None)):
                result["request_readiness"]["issued_queue"] = self._issued_queue.evidence()
        return result


def load_for_scheduler(config, clock):
    """The real-clock and unconfigured paths retain their existing behavior."""
    compass = getattr(config, "compass_config", None)
    path = getattr(compass, "request_readiness_profile", "")
    if not path or not (getattr(compass, "enabled", False)
                       and getattr(compass, "mode", None) == "predict"
                       and getattr(compass, "virtual_clock", False)
                       and getattr(clock, "epoch", None) is not None):
        return None
    if getattr(compass, "admission_seconds", 0):
        raise UnsupportedReadiness("readiness and admission_seconds are mutually exclusive")
    parallel = getattr(config, "parallel_config", None)
    for name in ("tensor_parallel_size", "pipeline_parallel_size",
                 "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(config, name, 1) != 1:
            raise UnsupportedReadiness("request readiness currently supports TP1/PP1 only")
    if getattr(parallel, "data_parallel_size", getattr(config, "data_parallel_size", 1)) != 1:
        raise UnsupportedReadiness("request readiness currently supports one engine core")
    if getattr(config, "scheduler_delay_factor", 0):
        raise UnsupportedReadiness("request readiness does not support scheduler delay")
    if getattr(config, "speculative_config", None) or getattr(config, "kv_transfer_config", None):
        raise UnsupportedReadiness("request readiness does not support speculation or KV transfer")
    return RequestReadiness(path)
