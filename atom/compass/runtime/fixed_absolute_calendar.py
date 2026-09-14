"""Core-owned finite releases with persistent source ingress service."""

import math

from atom.compass.fixed_absolute import FixedAbsoluteReleases
from atom.compass.prefix_workload import token_digest


class FixedAbsoluteCalendar:
    def __init__(self, plan, clock, readiness):
        if readiness is None:
            raise ValueError("fixed-absolute replay requires source-backed readiness")
        self.plan, self.clock, self.readiness = plan, clock, readiness
        self.rows = plan.rows
        self.state = FixedAbsoluteReleases(plan, clock.epoch)
        self.sequences = None
        self.released, self.completed = {}, {}

    def register(self, sequences):
        if self.sequences is not None or self.clock.time() != self.clock.epoch:
            raise ValueError("fixed-absolute registration requires a fresh original epoch")
        ordered = sorted(sequences, key=lambda seq: seq.compass_workload_index)
        if len(ordered) != len(self.rows) or len({seq.id for seq in ordered}) != len(ordered):
            raise ValueError("fixed-absolute registration is incomplete or duplicated")
        for row, seq in zip(self.rows, ordered):
            if (seq.compass_workload_index != row["index"] or seq.compass_workload_size != len(ordered)
                    or seq.arrive_time != self.clock.epoch + row["arrival_s"]
                    or token_digest(seq.token_ids) != row["prompt_token_sha256"]
                    or seq.max_tokens != row["output_tokens"] or not seq.ignore_eos):
                raise ValueError("registered request differs from fixed-absolute plan")
        self.readiness.register_causal_workload(ordered)
        self.sequences = ordered

    @property
    def next_release_at(self):
        return self.state.next_due if self.sequences is not None else math.inf

    def drain(self):
        """Mature events before reserving ingress; caller maintains waiting order."""
        if self.sequences is None:
            return set()
        changed = set()
        for index, released_at in self.state.pop_due(self.clock.time()):
            seq = self.sequences[index]
            seq.arrive_time = released_at
            self.readiness.release_causal_request(seq, released_at)
            record = self.readiness.record(seq)
            self.released[seq.id] = {
                "index": index, "seq_id": str(seq.id), "root_id": self.rows[index]["root_id"],
                "source_earliest_at": self.clock.epoch + self.rows[index]["arrival_s"],
                "released_at": released_at, "source_service_started_at": record.source_service_started_at,
                "ready_at": record.ready_at, "receipt_order": record.receipt_order}
            changed.add(seq.id)
        return changed

    def is_released(self, seq):
        return seq.id in self.released

    def complete(self, seq):
        if self.sequences is None or not self.is_released(seq) or seq.id in self.completed:
            raise ValueError("fixed-absolute terminal notification is out of order")
        index = self.released[seq.id]["index"]
        finished_at = float(seq.finish_time)
        if (not math.isfinite(finished_at) or finished_at < self.released[seq.id]["ready_at"]
                or seq.num_completion_tokens != self.rows[index]["output_tokens"]):
            raise ValueError("fixed-absolute request did not complete its pinned output")
        response_at = finished_at + self.plan.response_delivery_seconds
        self.state.complete(index, response_at)
        self.completed[seq.id] = {
            "index": index, "seq_id": str(seq.id), "native_engine_finished_at": finished_at,
            "completion_tokens": seq.num_completion_tokens,
            "modelled_client_response_available_at": response_at}

    def evidence(self):
        return {**self.plan.evidence(), "clock": "virtual", "epoch": self.clock.epoch,
                "registered": self.sequences is not None, "complete": self.state.done,
                "root_completed_at": self.state.root_completion_times(),
                "releases": list(self.released.values()), "completions": list(self.completed.values())}
