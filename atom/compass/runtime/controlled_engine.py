"""Internal cooperative driver for one virtual prediction EngineCore.

No serving entry point selects this driver. The caller owns the external event
agenda and acknowledges each output before granting further virtual progress.
"""
from dataclasses import dataclass
import math

from atom.model_engine.engine_core import EngineCore, _ClockWait, _OutputReady
from atom.utils.clock import VirtualClock, get_clock


@dataclass(frozen=True)
class EngineEvent:
    kind: str
    request_id: int | str
    at: float


@dataclass(frozen=True)
class EngineYield:
    now: float
    reason: str
    output_events: tuple[EngineEvent, ...]
    next_boundary_at: float
    idle: bool


class ControlledEngine:
    def __init__(self, core):
        scheduler = core.scheduler
        config = scheduler.config
        compass = getattr(config, "compass_config", None)
        clock = get_clock()
        parallel = getattr(config, "parallel_config", None)
        if (type(core) is not EngineCore or not isinstance(clock, VirtualClock)
                or not getattr(compass, "enabled", False)
                or getattr(compass, "mode", None) != "predict"
                or not getattr(compass, "virtual_clock", False)
                or any(getattr(config, name, 1) != 1 for name in (
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "prefill_context_parallel_size", "decode_context_parallel_size"))
                or getattr(parallel, "data_parallel_size", getattr(config, "data_parallel_size", 1)) != 1
                or getattr(config, "speculative_config", None)
                or getattr(config, "kv_transfer_config", None)
                or getattr(core, "kv_transfer_enabled", False)):
            raise ValueError("controlled engine requires a virtual prediction TP1/PP1 core without speculation or KV transfer")
        if (getattr(core, "_controlled_owner", None) is not None
                or scheduler.waiting or scheduler.running
                or getattr(scheduler, "deferred_free_blocks", None)
                or getattr(core, "_compass_forward_timeline", None) is not None
                or not core.stream_output_queue.empty()
                or (getattr(core, "input_queue", None) is not None and not core.input_queue.empty())):
            raise ValueError("controlled engine requires exclusive ownership of a fresh empty core")
        # This also refuses finite/serial calendars and previously resolved
        # readiness records. No future request descriptor is constructed.
        scheduler.begin_issued_admission()
        self.core, self.scheduler, self.clock = core, scheduler, clock
        self._frontier = clock.time()
        self._program = self._event = None
        self._reply = None
        self._closed = False
        self._failed = None
        self._first_reported, self._completed = set(), set()
        core._controlled_owner = self

    def _check_active(self):
        if self._closed or self._failed is not None:
            raise RuntimeError("controlled engine is closed or failed") from self._failed
        if (self.core._controlled_owner is not self or get_clock() is not self.clock
                or self.clock.time() != self._frontier):
            raise RuntimeError("controlled engine lost exclusive clock ownership")

    def submit_issued(self, sequence, issued_at):
        self._check_active()
        if type(issued_at) not in (int, float) or issued_at != self._frontier:
            raise ValueError("submit only at the current committed issue frontier")
        record = self.scheduler._request_readiness.admit_issued_request(sequence, issued_at)
        self.scheduler.add(sequence)
        return record

    def _step_program(self):
        try:
            return (yield from self.core._process_engine_step_program(advance_idle=False))
        finally:
            self.core._publish_step_kv_events()

    def _move_to(self, target):
        if not math.isfinite(target) or target < self._frontier:
            raise ValueError("engine clock boundary is invalid or behind the committed frontier")
        self.clock.advance(target - self._frontier)
        self._frontier = self.clock.time()
        if self._frontier != target:
            raise RuntimeError("virtual clock did not reach the requested frontier exactly")

    def _yield(self, reason, events=(), *, next_at=None):
        if next_at is None:
            if isinstance(self._event, _ClockWait):
                next_at = self._event.at
            elif self._program is not None or self.scheduler.running:
                next_at = self._frontier
            else:
                next_at = self.scheduler.next_ready_at
        return EngineYield(self._frontier, reason, tuple(events), next_at,
                           self._program is None and not self.scheduler.running and not self.scheduler.waiting)

    def _output_events(self, output):
        if output.at != self._frontier:
            raise ValueError("output publication is not at the committed frontier")
        first = set(self._first_reported)
        complete = set(self._completed)
        events = []
        for batch in output.streams:
            for request_id, item in batch:
                if request_id != item.request_id:
                    raise ValueError("stream output pair has inconsistent request identity")
                if item.output_tokens and item.request_id not in first:
                    events.append(EngineEvent("first_token", item.request_id, item.first_token_time))
                    first.add(item.request_id)
        for seq in output.finished:
            if seq.id in complete:
                raise ValueError("duplicate controlled completion")
            if seq.num_completion_tokens and seq.id not in first:
                events.append(EngineEvent("first_token", seq.id, seq.first_token_time))
                first.add(seq.id)
            events.append(EngineEvent("completion", seq.id, seq.finish_time))
            complete.add(seq.id)
        for event in events:
            if event.request_id not in self.scheduler._request_readiness.records:
                raise ValueError("output belongs to an unissued request")
            if (type(event.at) not in (int, float) or not math.isfinite(event.at)
                    or event.at < self._frontier or event.at > output.at):
                raise ValueError(
                    f"{event.kind} for {event.request_id} at {event.at} is outside "
                    f"the current publication frontier {self._frontier}")
        self._first_reported, self._completed = first, complete
        return events

    def advance_until(self, horizon, *, include_horizon=False):
        """Run strictly before horizon, unless its boundary is explicitly included.

        Always return after publishing outputs, before a trailing charge or a
        new batch. The external agenda decides equal-time timer/core ordering.
        """
        self._check_active()
        if (type(horizon) not in (int, float) or not math.isfinite(horizon)
                or horizon < self._frontier or type(include_horizon) is not bool):
            raise ValueError("controlled horizon must be finite and nondecreasing")
        try:
            while True:
                if self._frontier == horizon and not include_horizon:
                    return self._yield("horizon")
                if self._event is None:
                    if self._program is None:
                        self._program = self._step_program()
                    try:
                        self._event = self._program.send(self._reply)
                        self._reply = None
                    except StopIteration as done:
                        self._program = None
                        self._reply = None
                        if self._frontier == horizon:
                            return self._yield("horizon")
                        if done.value:
                            continue
                        ready = self.scheduler.next_ready_at
                        if ready <= self._frontier:
                            return self._yield("blocked")
                        self._move_to(min(ready, horizon))
                        if self._frontier == horizon and (not include_horizon or ready != horizon):
                            return self._yield("horizon")
                        continue
                if isinstance(self._event, _ClockWait):
                    target = self._event.at
                    if not math.isfinite(target) or target < self._frontier:
                        raise ValueError("pending forward boundary is behind the committed frontier")
                    if target > horizon or (target == horizon and not include_horizon):
                        self._move_to(horizon)
                        return self._yield("horizon")
                    self._move_to(target)
                    self._event = None
                    continue
                if isinstance(self._event, _OutputReady):
                    events = self._output_events(self._event)
                    self.core._publish_step_output(self._event)
                    after_at = self._event.after_at
                    self._event = None
                    self._reply = True
                    return self._yield("outputs", events, next_at=after_at)
                raise RuntimeError("unknown controlled step checkpoint")
        except BaseException as exc:
            self._failed = exc
            if self._program is not None:
                self._program.close()
                self._program = None
            raise

    def close(self):
        """Abandon control for teardown; a partial step is never retried."""
        if not self._closed:
            self._closed = True
            if self._program is not None:
                self._program.close()
                self._program = None
        # Keep the core claimed: legacy execution cannot resume an abandoned
        # partially applied step. The caller tears down this engine normally.
