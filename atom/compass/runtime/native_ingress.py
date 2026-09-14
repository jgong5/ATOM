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

    def resolve_closed_workload(self, requests):
        writer_available = receiver_available = float("-inf")
        result = {}
        for order, request in enumerate(requests):
            layout = request.ingress
            if (layout.frame_request_count != 1 or layout.pickle_protocol != 4
                    or layout.token_typecode != "i" or layout.token_itemsize != 4):
                raise UnsupportedReadiness("native ingress source requires single-Sequence ADD/protocol4/int32")
            size = layout.reconstructed_add_bytes
            if not self.min_bytes <= size <= self.max_bytes:
                raise UnsupportedReadiness(
                    f"native ingress payload {size} bytes is outside source support "
                    f"[{self.min_bytes}, {self.max_bytes}]")
            writer_base, writer_slope = self._services["writer"]
            receiver_base, receiver_slope = self._services["receiver"]
            writer_started = max(request.arrived_at, writer_available)
            writer_available = writer_started + (writer_base + writer_slope * size) * 1e-6
            receiver_started = max(writer_available, receiver_available)
            receiver_available = receiver_started + (receiver_base + receiver_slope * size) * 1e-6
            result[request.request_id] = ReadyEvent(receiver_available, order)
        return result
