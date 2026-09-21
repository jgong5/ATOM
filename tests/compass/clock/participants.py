# SPDX-License-Identifier: MIT
"""What a synthetic participant does between grants, and where each behaviour came from.

Nothing here is invented. Every behaviour is one row of the blocking-call
inventory in `atom/compass/audit/sync_sites.json`, and the row is named in the
docstring of the method that models it. The three shapes the inventory
distinguishes are the three shapes here:

* a wait whose duration **is** modelled time becomes a horizon at `now + d` and
  the work happens when the grant arrives -- the forward pass, and the idle step
  loop that has to jump rather than spin;
* a wait for a message another process will send becomes a parked participant
  with a real block around it, and nothing else -- the engine parking for the
  worker reply, the decode side parking for the prefill side's completion, a
  request's coroutine parking for its next chunk;
* a bound that sets a cadence becomes a timer on simulated time with a real poll
  that returns at once -- the idle transfer drain.

**The engine parks for rank 0, not for every worker.** Only rank 0 is given the
primary output address and only rank 0's replies land on the queue the engine's
step loop reads, so the wait ends on one reply however wide the group is. That
is the whole reason a tensor-parallel group is one participant rather than a
barrier over N of them, and it is why nothing here fans a step out over workers.
The form that does wait for every rank is the aggregation call, which appears
below only where the inventory puts it -- collecting transfer status.
"""

import collections
import enum
import math
import queue
from dataclasses import dataclass

from .deployments import (
    ADMISSION_FLOOR_SECONDS,
    PIPELINE_STAGE_FLOOR_SECONDS,
    ROLE_BOUNDARY_FLOOR_SECONDS,
    TRAFFIC_SOURCE,
    Role,
)

#: How long a real block is allowed to sit before it is a defect. A participant
#: only blocks after the harness has put its message in the queue, so this can
#: only be reached by a message the harness delivered to nobody -- which is the
#: loud half of the failure pair and is meant to stay loud.
REAL_BLOCK_SECONDS = 30.0

#: The cadence of the idle transfer drain, on simulated time.
DRAIN_CADENCE_SECONDS = 1.0e-3


class Message(enum.Enum):
    """What one participant sends another."""

    REQUEST = "request"
    BATCH = "batch"
    TOKENS = "tokens"
    PREFILL_DONE = "prefill-done"
    BLOCKS = "blocks"
    RESPONSE = "response"


@dataclass(frozen=True)
class Workload:
    """The trace a run replays, in steps and modelled seconds.

    The design's sizing is 106 prefill and 4,346 decode steps over 267 s of
    modelled time. Requests arrive in groups rather than one at a time, because
    4,346 decode steps for 106 requests means requests were resident together --
    and because a participant with only ever one event in flight never exercises
    the part of the clock that holds more than one.
    """

    requests: int
    requests_per_arrival: int
    prefill_steps: int
    decode_steps: int
    prefill_step_seconds: float
    decode_step_seconds: float
    arrival_interval_seconds: float


#: The measured prior run: 106 prefill + 4,346 decode steps, last response at
#: about 267 s of modelled time.
DESIGN_WORKLOAD = Workload(
    requests=106,
    requests_per_arrival=2,
    prefill_steps=1,
    decode_steps=41,
    prefill_step_seconds=0.30,
    decode_step_seconds=0.054,
    arrival_interval_seconds=5.0861,
)


def scaled(workload: Workload, requests: int, decode_steps: int = 0) -> Workload:
    """The same trace shape, shorter, for a run that has to finish quickly."""
    return Workload(
        requests=requests,
        requests_per_arrival=workload.requests_per_arrival,
        prefill_steps=workload.prefill_steps,
        decode_steps=decode_steps or workload.decode_steps,
        prefill_step_seconds=workload.prefill_step_seconds,
        decode_step_seconds=workload.decode_step_seconds,
        arrival_interval_seconds=workload.arrival_interval_seconds,
    )


class _Participant:
    """Common to all of them: a name, a real inbox, and a real block on it."""

    def __init__(self, lp_id):
        self.lp_id = lp_id
        self.inbox = queue.Queue()

    def receive(self, count):
        """Take `count` messages out of the inbox with a real blocking read.

        The pair around this is the clock's own: asking to advance is the
        declaration that the participant is idle, and taking up the grant is the
        declaration that it is running again. The call between them is left
        exactly as it is, which is what the inventory's category B says to do
        with `atom/model_engine/async_proc.py::AsyncIOProcManager.call_func::
        self.outputs_queue.get()::queue_get#0` -- the engine's step loop parking
        for rank 0's reply -- and with the eight other unbounded parks modelled
        here.
        """
        return [self.inbox.get(timeout=REAL_BLOCK_SECONDS) for _ in range(count)]

    def poll(self):
        """A real poll that returns at once, whatever the cadence timer says.

        `atom/distributed/pp_transport.py::PPStageTransport.recv_metadata::
        self._meta_recv.poll(timeout_ms)::poll#0` and its sibling on
        `recv_tokens` are bounded polls that pace a loop rather than detect a
        failure, and so is the idle transfer drain at
        `atom/model_engine/engine_core.py:455`. The cadence comes from the
        simulated clock; the poll stays real so the thread is still responsive,
        and spends no simulated time.
        """
        try:
            return self.inbox.get_nowait()
        except queue.Empty:
            return None


class TrafficSource(_Participant):
    """Offers requests at modelled arrival times and waits for their responses."""

    def __init__(self, workload, targets):
        super().__init__(TRAFFIC_SOURCE)
        self.expected = workload.requests
        self.responses = 0
        self.arrivals = collections.deque()
        request = 0
        for slot in range(workload.requests // workload.requests_per_arrival):
            when = slot * workload.arrival_interval_seconds
            target = targets[slot % len(targets)]
            for _ in range(workload.requests_per_arrival):
                self.arrivals.append((when, request, target))
                request += 1

    def horizon(self, now):
        """The next arrival, or nothing at all while it waits for a response.

        `atom/model_engine/llm_engine.py:745` stamps a request's arrival from the
        clock, and every queueing age and reported latency is measured from that
        stamp, so the arrival time is modelled time and the offer happens when
        the clock reaches it. With no arrival left this declares no horizon,
        which is the category-B park of
        `atom/entrypoints/openai/api_server.py::generate_async::token_queue.get()
        ::queue_get#0` -- the request's own coroutine waiting for its next chunk.
        """
        return self.arrivals[0][0] if self.arrivals else math.inf

    def on_time(self, now, delivered, run):
        for message, _request, _seconds, _sender in self.receive(delivered):
            if message is Message.RESPONSE:
                self.responses += 1
        while self.arrivals and self.arrivals[0][0] <= now:
            _when, request, target = self.arrivals.popleft()
            run.send(
                self.lp_id,
                target,
                now + ADMISSION_FLOOR_SECONDS,
                (Message.REQUEST, request, 0.0, self.lp_id),
            )

    @property
    def finished(self):
        return not self.arrivals and self.responses == self.expected


class EngineStage(_Participant):
    """One engine participant: a whole replica, or one stage of a pipelined one.

    Stage 0 is the head and owns the replica's work. With no pipeline it is the
    replica. The widths above it -- tensor-parallel workers, data-parallel ranks
    -- are not modelled as anything, because they hold no clock and the step's
    duration is charged once here however many of them there are.
    """

    def __init__(self, lp_id, role, stage, pp, head, next_stage, handoff, plan):
        super().__init__(lp_id)
        self.role = role
        self.stage = stage
        self.pp = pp
        self.head = head
        self.next_stage = next_stage
        self.handoff = handoff
        self.plan = plan
        self.work = collections.deque()
        self.queued = collections.deque()
        self.remaining = {}
        self.draining = set()
        self.steps = 0
        self._slice = None
        self._step_due = None
        self._drain_due = None

    def horizon(self, now):
        """The end of the step being charged, the next drain tick, or nothing.

        A step's duration is the thing being predicted, so it is charged by
        declaring the time it ends and doing the work when the clock gets there:
        `atom/model_engine/engine_core.py::EngineCore._process_engine_step_inner::
        self.runner_mgr.call_func( "forward", scheduled_batch, wait_out=True )::
        worker_rpc#0`, and its prefill, decode and pipeline-head siblings.

        Declaring nothing is not the same as having nothing to do: it is what
        makes the idle step loop jump instead of spinning, which
        `atom/model_engine/engine_core.py::EngineCore.busy_loop::while True::
        spin_loop#0` says it must, since that loop turns with no pause and no
        blocking call when the scheduler is empty.
        """
        self._start_slice(now)
        dues = [due for due in (self._step_due, self._drain_due) if due is not None]
        return min(dues) if dues else math.inf

    def on_time(self, now, delivered, run):
        for message in self.receive(delivered):
            self._accept(message, now, run)
        if self._drain_due is not None and now >= self._drain_due:
            self._drain(now)
        if self._step_due is not None and now >= self._step_due:
            self._finish_slice(now, run)
        self._start_slice(now)

    @property
    def finished(self):
        return not (
            self.work or self.queued or self.remaining or self.draining or self._slice
        )

    def _accept(self, message, now, run):
        """Take in one message. Which ones arrive is what the role means.

        A decode participant taking in `PREFILL_DONE` is
        `atom/model_engine/engine_core.py::DecodeEngineCore._recv_prefill_done::
        sock.recv()::socket_recv#0`, whose peer the inventory records as the
        other deployment rather than another process of this one; its reply is
        the block assignment the prefill side parks for at
        `PrefillEngineCore._recv_block_assignments::sock.recv()::socket_recv#0`.
        A stage taking in `BATCH` is
        `atom/distributed/pp_transport.py::PPStageTransport.recv_metadata::
        self._meta_recv.recv()::socket_recv#0`, and the head taking in `TOKENS`
        is `recv_tokens::self._token_recv.recv()::socket_recv#0`.
        """
        kind, request, seconds, sender = message
        if kind is Message.BATCH:
            self.queued.append((request, seconds))
        elif kind is Message.TOKENS:
            self._advance_plan(request, now, run)
        elif kind is Message.BLOCKS:
            self.draining.discard(request)
        elif kind is Message.REQUEST:
            self.remaining[request] = collections.deque(self.plan)
            self.work.append(request)
        elif kind is Message.PREFILL_DONE:
            self.remaining[request] = collections.deque(self.plan)
            self.work.append(request)
            run.send(
                self.lp_id,
                sender,
                now + ROLE_BOUNDARY_FLOOR_SECONDS,
                (Message.BLOCKS, request, 0.0, self.lp_id),
            )

    def _start_slice(self, now):
        """Begin charging the next step, if one is waiting and none is running."""
        if self._slice is not None:
            return
        if self.queued:
            request, seconds = self.queued.popleft()
        elif self.work:
            request = self.work.popleft()
            seconds = self.remaining[request][0] / self.pp
        else:
            return
        self._slice = (request, seconds)
        self._step_due = now + seconds

    def _finish_slice(self, now, run):
        """The step is over. Hand it on, or count it against the request's plan."""
        request, seconds = self._slice
        self._slice = None
        self._step_due = None
        self.steps += 1
        if self.next_stage is not None:
            run.send(
                self.lp_id,
                self.next_stage,
                now + PIPELINE_STAGE_FLOOR_SECONDS,
                (Message.BATCH, request, seconds, self.lp_id),
            )
        elif self.stage:
            run.send(
                self.lp_id,
                self.head,
                now + PIPELINE_STAGE_FLOOR_SECONDS,
                (Message.TOKENS, request, 0.0, self.lp_id),
            )
        else:
            self._advance_plan(request, now, run)

    def _advance_plan(self, request, now, run):
        """One step of this request is done. Either more follow, or it is finished."""
        self.remaining[request].popleft()
        if self.remaining[request]:
            self.work.append(request)
            return
        del self.remaining[request]
        if self.role is Role.PREFILL:
            run.send(
                self.lp_id,
                self.handoff,
                now + ROLE_BOUNDARY_FLOOR_SECONDS,
                (Message.PREFILL_DONE, request, 0.0, self.lp_id),
            )
            self.draining.add(request)
            self._drain_due = now + DRAIN_CADENCE_SECONDS
        else:
            run.send(
                self.lp_id,
                self.handoff,
                now + ADMISSION_FLOOR_SECONDS,
                (Message.RESPONSE, request, 0.0, self.lp_id),
            )

    def _drain(self, now):
        """A tick of the idle transfer drain: poll for real, re-arm on the clock.

        The poll finds nothing, which is what an idle drain usually finds, and it
        costs no simulated time. The cadence re-arms only while a transfer is
        outstanding, so a run with none of them pays for none of them.
        """
        self.poll()
        self._drain_due = now + DRAIN_CADENCE_SECONDS if self.draining else None


def build(deployment, workload):
    """Every participant of a deployment, ready to be driven."""
    heads = {
        role: tuple(replica.stage_ids()[0] for replica in deployment.of_role(role))
        for role in Role
    }
    people = {}
    for replica in deployment.replicas:
        stages = replica.stage_ids()
        handoff = (
            heads[Role.DECODE][replica.index]
            if replica.role is Role.PREFILL
            else TRAFFIC_SOURCE
        )
        plan = _plan(replica.role, workload)
        for stage, lp_id in enumerate(stages):
            people[lp_id] = EngineStage(
                lp_id,
                replica.role,
                stage,
                replica.pp,
                stages[0],
                stages[stage + 1] if stage + 1 < replica.pp else None,
                handoff,
                plan,
            )
    entry = heads[Role.ENGINE] or heads[Role.PREFILL]
    people[TRAFFIC_SOURCE] = TrafficSource(workload, entry)
    return people


def _plan(role, workload):
    """The step durations one request costs a replica of this role."""
    prefill = [workload.prefill_step_seconds] * workload.prefill_steps
    decode = [workload.decode_step_seconds] * workload.decode_steps
    if role is Role.PREFILL:
        return tuple(prefill)
    if role is Role.DECODE:
        return tuple(decode)
    return tuple(prefill + decode)
