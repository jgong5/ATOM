"""Two overlapping native ingress services, priced by a frozen source fit."""

from __future__ import annotations

import math

from atom.compass.core.loaded_input import load_json
from atom.compass.runtime.request_readiness import ReadyEvent, UnsupportedReadiness


class NativeWriterReceiver:
    """A serial writer followed by a serial receiver, with independent clocks.

    The PoC explicitly approximates declared arrival as writer eligibility and
    writer return as receiver eligibility. Extra origin and transport delays
    are zero assumptions, not measurements. Source service durations exclude
    queue waiting; serializing complete handoff elapsed times would charge it
    again and discard writer/receiver overlap.
    """

    def __init__(self, *, fit, validation):
        inputs = []

        def read(reference, role):
            payload, record = load_json(reference["path"], role=role)
            if record.sha256 != reference["sha256"]:
                raise UnsupportedReadiness(f"source identity changed for {role}")
            inputs.append(record)
            return payload

        fitted = read(fit, "runtime.request_readiness.endpoint_fit")
        verdict = read(validation, "runtime.request_readiness.validation")
        if fitted.get("schema") != "compass.frozen_endpoint_service_fit/1":
            raise UnsupportedReadiness("unsupported native ingress source-fit schema")
        if (verdict.get("schema") != "compass.withheld_service_support_verdict/1"
                or verdict.get("fit_freeze", {}).get("sha256") != fit["sha256"]
                or verdict.get("heldout_support_passed") is not True
                or verdict.get("profile_admissible_under_fixed_contract") is not True):
            raise UnsupportedReadiness("native ingress source fit lacks passing heldout support")
        for key, role in (("plan", "plan"), ("contract", "contract"),
                          ("source_result", "source")):
            read(fitted[key], f"runtime.request_readiness.{role}")
        sizes = [row["serialized_bytes_median"] for row in fitted["endpoint_summaries"]]
        self.min_bytes, self.max_bytes = min(sizes), max(sizes)
        self._services = {}
        for name in ("writer", "receiver"):
            model = fitted["models"][name]
            coefficients = (float(model["intercept_us"]), float(model["slope_us_per_byte"]))
            if any(not math.isfinite(value) or value < 0 for value in coefficients):
                raise UnsupportedReadiness("native ingress source coefficients must be finite and nonnegative")
            self._services[name] = coefficients
        self.loaded_inputs = tuple(inputs)

    def _service_times(self, request):
        layout = request.ingress
        if (layout.frame_request_count != 1 or layout.pickle_protocol != 4
                or layout.token_typecode != "i" or layout.token_itemsize != 4):
            raise UnsupportedReadiness("native ingress source requires single-Sequence ADD/protocol4/int32")
        size = layout.reconstructed_add_bytes
        if not self.min_bytes <= size <= self.max_bytes:
            raise UnsupportedReadiness(
                f"native ingress payload {size} bytes is outside source support "
                f"[{self.min_bytes}, {self.max_bytes}]")
        return tuple((base + slope * size) * 1e-6 for base, slope in
                     (self._services["writer"], self._services["receiver"]))

    def resolve_closed_workload(self, requests):
        writer_available = receiver_available = float("-inf")
        result = {}
        for order, request in enumerate(requests):
            writer_seconds, receiver_seconds = self._service_times(request)
            writer_started = max(request.arrived_at, writer_available)
            writer_available = writer_started + writer_seconds
            receiver_started = max(writer_available, receiver_available)
            receiver_available = receiver_started + receiver_seconds
            result[request.request_id] = ReadyEvent(receiver_available, order)
        return result

    def resolve_serial_release(self, request):
        """Reuse the source law after a serial predecessor has fully completed.

        ReleaseCalendar admits only one request at a time. Its previous writer
        and receiver completed before that predecessor's generation, so neither
        can still occupy a service queue at this release.
        """
        event = self.resolve_closed_workload((request,))[request.request_id]
        return ReadyEvent(event.ready_at, event.receipt_order, request.arrived_at)

    def new_release_queue(self):
        """One persistent service queue per finite causal replay, not per client."""
        return NativeIngressReleaseQueue(self)


class NativeIngressReleaseQueue:
    """Advance source writer/receiver availability in matured release order."""

    def __init__(self, provider):
        self.provider = provider
        self.writer_available = self.receiver_available = -math.inf
        self.last_release = -math.inf
        self.count = 0

    def resolve_release(self, request):
        arrival = request.arrived_at
        if not math.isfinite(arrival) or arrival < self.last_release:
            raise UnsupportedReadiness("causal ingress releases must be chronological")
        writer_seconds, receiver_seconds = self.provider._service_times(request)
        writer_started = max(arrival, self.writer_available)
        self.writer_available = writer_started + writer_seconds
        self.receiver_available = max(self.writer_available, self.receiver_available) + receiver_seconds
        event = ReadyEvent(self.receiver_available, self.count, writer_started)
        self.count += 1
        self.last_release = arrival
        return event

    def evidence(self):
        return {"released_requests": self.count,
                "writer_available_at": self.writer_available if self.count else None,
                "receiver_available_at": self.receiver_available if self.count else None}
