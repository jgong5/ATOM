"""Causal releases for a registered, serial two-turn replay opening."""

from __future__ import annotations

import math

from atom.compass.prefix_workload import token_digest
from atom.compass.replay_plan import OpeningPlan


class ReleaseCalendar:
    """Own release state without changing payloads or the Sequence wire layout."""

    def __init__(self, plan, clock, readiness):
        if readiness is None:
            raise ValueError("opening release requires a source-backed readiness service")
        self.plan, self.clock, self.readiness = plan, clock, readiness
        self.rows = plan.rows
        self.sequences = None
        self.released = {}
        self.completed = {}

    def register(self, sequences):
        if self.sequences is not None or self.clock.time() != self.clock.epoch:
            raise ValueError("opening registration requires a fresh original virtual epoch")
        ordered = sorted(sequences, key=lambda seq: seq.compass_workload_index)
        if len(ordered) != 2:
            raise ValueError("opening registration is incomplete")
        for row, seq in zip(self.rows, ordered):
            if (seq.compass_workload_index != row["index"] or seq.compass_workload_size != 2
                    or seq.arrive_time != self.clock.epoch + row["arrival_s"]
                    or token_digest(seq.token_ids) != row["prompt_token_sha256"]
                    or seq.max_tokens != row["output_tokens"] or not seq.ignore_eos):
                raise ValueError("registered request differs from the pinned opening plan")
        self.readiness.register_serial_workload(ordered)
        self.sequences = ordered
        self._release(0, self.clock.epoch + self.rows[0]["arrival_s"])

    def _release(self, index, released_at):
        seq = self.sequences[index]
        # Arrival for latency accounting is causal release, not the earlier
        # source timestamp. Keep the immutable source timestamp in the plan.
        seq.arrive_time = released_at
        self.readiness.release_serial_request(seq, released_at)
        self.released[seq.id] = {
            "index": index, "seq_id": str(seq.id),
            "source_earliest_at": self.clock.epoch + self.rows[index]["arrival_s"],
            "released_at": released_at,
            "source_service_started_at": self.readiness.record(seq).source_service_started_at,
            "ready_at": self.readiness.record(seq).ready_at,
        }

    def is_released(self, seq):
        return seq.id in self.released

    def complete(self, seq):
        if self.sequences is None or not self.is_released(seq) or seq.id in self.completed:
            raise ValueError("opening terminal notification is out of order")
        index = self.released[seq.id]["index"]
        finished_at = float(seq.finish_time)
        if (not math.isfinite(finished_at) or finished_at < self.released[seq.id]["ready_at"]
                or seq.num_completion_tokens != self.rows[index]["output_tokens"]):
            raise ValueError("opening predecessor did not complete its pinned output")
        response_at = finished_at + self.plan.response_delivery_seconds
        self.completed[seq.id] = {
            "index": index, "seq_id": str(seq.id), "native_engine_finished_at": finished_at,
            "completion_tokens": seq.num_completion_tokens,
            "modelled_client_response_available_at": response_at,
            "response_delivery": self.plan.evidence()["response_delivery"],
        }
        if index == 0:
            self._release(1, max(self.clock.epoch + self.rows[1]["arrival_s"], response_at))

    def evidence(self):
        return {**self.plan.evidence(), "clock": "virtual", "epoch": self.clock.epoch,
                "registered": self.sequences is not None,
                "releases": list(self.released.values()),
                "completions": list(self.completed.values())}


def load_for_scheduler(config, clock, readiness):
    compass = getattr(config, "compass_config", None)
    path = getattr(compass, "opening_plan", "")
    fixed_path = getattr(compass, "fixed_absolute_plan", "")
    if path and fixed_path:
        raise ValueError("opening and fixed-absolute profiles are mutually exclusive")
    if fixed_path:
        if (not getattr(compass, "enabled", False) or getattr(compass, "mode", None) != "predict"
                or getattr(clock, "epoch", None) is None):
            raise ValueError("fixed-absolute calendar requires a virtual predictor")
        from atom.compass.fixed_absolute import FixedAbsolutePlan
        from atom.compass.runtime.fixed_absolute_calendar import FixedAbsoluteCalendar
        plan = FixedAbsolutePlan.load(fixed_path, compass.fixed_absolute_plan_sha256)
        return FixedAbsoluteCalendar(plan, clock, readiness)
    if not path:
        return None
    if (not getattr(compass, "enabled", False) or getattr(compass, "mode", None) != "predict"
            or getattr(clock, "epoch", None) is None):
        raise ValueError("opening release calendar requires a virtual predictor")
    plan = OpeningPlan.load(path, compass.opening_plan_sha256)
    return ReleaseCalendar(plan, clock, readiness)
