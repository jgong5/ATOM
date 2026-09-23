# SPDX-License-Identifier: MIT
"""A KV connector that moves no bytes and charges for the ones it would have.

Registered in the engine's connector factory under the name `compass`, so a
simulated deployment selects it the way it selects any other backend and
nothing in the engine changes. It implements the same eight methods as the
RDMA backends: a transfer is announced to the worker, it is priced, and it is
reported finished once the clock has passed its deadline.

**The clock is handed in and there is no other.** This connector never reads a
wall clock, not as a default and not as a fallback, because a simulated run
that quietly advanced on real time would produce plausible numbers that mean
nothing at all. The harness binds two entries in `kv_transfer_config` before
the engine starts:

| key | what it holds |
|---|---|
| `compass_clock` | a callable of no arguments returning the current time in seconds |
| `compass_transfer` | a `TransferModel`, priced from the machine spec |

Either one missing is refused by name at construction, on the side that needs
it -- the worker needs both, the scheduler side holds no timing and asks for
neither.

**Completion is reported per worker, and the request finishes when every
worker has reported it.** That conjunction is not built here: the engine's own
output aggregator already holds a transfer incomplete until all workers report
the same identity, so this connector reports only what its own rank has
finished and lets the layer above combine them. The rule it lands on is
therefore the one the push backend uses -- every rank must report before a
request completes -- rather than the pull backend's, which reads only the last
status of a transfer's list and would release a request on one rank's word.
The two agree whenever every rank holds the same blocks and disagree the
moment they do not: an asymmetric or pipelined producer releases earlier under
the last-status rule, and a comparison between the two backends has to
account for that rather than assume it away.

**Nothing may turn on the order of the returned sets.** A transfer is released
by its deadline against the clock and by nothing else, so two transfers issued
in either order at one instant release together, and the sets carry no
sequence for anything downstream to read.

**Filling a request from another deployment is two halves, and they land
together.** The scheduler side claims the whole prompt as already held
elsewhere, which is what suspends the request, and offers the receive that
the workers then carry and report. The offer becomes an announcement only for
a request the engine did suspend, so a report always has a suspension to
resolve and never names a request nothing was waiting on. A producer that
finishes a request hands back the parameters the router relays to the
deployment that will decode it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atom.compass.kv.handoff import transfer_params
from atom.compass.kv.transfer import TransferModel
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, ReqId, ReqMeta
from atom.model_engine.sequence import SequenceStatus

#: Where the harness binds the clock this connector reads.
CLOCK_KEY = "compass_clock"

#: Where the harness binds the priced transfer model.
TRANSFER_KEY = "compass_transfer"


class UnboundSeam(RuntimeError):
    """The harness did not bind something the connector cannot invent."""


def _kv_config(config) -> dict:
    return getattr(config, "kv_transfer_config", {}) or {}


def _is_producer(kv_config: dict) -> bool:
    return kv_config.get("kv_role", "kv_producer") == "kv_producer"


def _bound_clock(kv_config: dict):
    clock = kv_config.get(CLOCK_KEY)
    if not callable(clock):
        raise UnboundSeam(
            f"kv_transfer_config[{CLOCK_KEY!r}] holds {clock!r}, where this "
            "connector needs a callable of no arguments returning the current "
            "time in seconds. It reads no other clock: a run that fell back to "
            "a wall clock would report transfer times that belong to the "
            "machine it ran on rather than the one it describes"
        )
    return clock


def _bound_transfer(kv_config: dict) -> TransferModel:
    model = kv_config.get(TRANSFER_KEY)
    if not isinstance(model, TransferModel):
        raise UnboundSeam(
            f"kv_transfer_config[{TRANSFER_KEY!r}] holds {model!r}, where this "
            "connector needs a TransferModel priced from the machine spec's "
            "interconnect and the KV geometry of the model being simulated"
        )
    return model


@dataclass(frozen=True, slots=True)
class _InFlight:
    """One announced transfer: what it carries and when it may be reported."""

    blocks: int
    release_at_s: float


class SimulatedKVConnector(KVConnectorBase):
    """Worker side: prices what it is told to move, and releases it on time.

    There are no device bytes behind any of this, so there is nothing to
    register and nothing to fence after a receive -- the base class's empty
    answer for the fence is the correct one here rather than an omission.
    """

    def __init__(self, config) -> None:
        kv_config = _kv_config(config)
        self.is_producer = _is_producer(kv_config)
        self._clock = _bound_clock(kv_config)
        self._transfer = _bound_transfer(kv_config)
        self._sending: dict[ReqId, _InFlight] = {}
        self._recving: dict[ReqId, _InFlight] = {}

    def register_kv_caches(
        self, kv_caches, transfer_tensors=None, num_blocks=None
    ) -> None:
        """Nothing to register: no KV tensor exists to expose to a remote."""

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Announce this step's transfers and price each one from now.

        A producer is told which completed prefills the other side will take,
        a consumer which requests it is waiting on; each announcement starts
        one transfer's clock.
        """
        if metadata is None:
            return
        now = self._clock()
        if self.is_producer:
            for req_id, meta in metadata.reqs_to_save.items():
                self._issue(self._sending, req_id, meta, now)
            return
        for req_id, meta in metadata.reqs_to_recv.items():
            self._issue(self._recving, req_id, meta, now)

    def _issue(
        self, pending: dict[ReqId, _InFlight], req_id: ReqId, meta: ReqMeta, now: float
    ) -> None:
        if req_id in pending:
            raise ValueError(
                f"request {req_id!r} was announced again while its transfer is "
                "still in flight; taking either issue time would be a guess, "
                "and the announcement is cleared once a request is queued, so "
                "a second one means two transfers were queued for one request"
            )
        blocks = len(meta.local_block_ids) - meta.num_computed_blocks
        pending[req_id] = _InFlight(
            blocks=blocks, release_at_s=self._transfer.release_at(now, blocks)
        )

    def get_finished(self) -> tuple[set, set]:
        """The requests whose deadline the clock has reached, once each."""
        now = self._clock()
        return self._matured(self._sending, now), self._matured(self._recving, now)

    @staticmethod
    def _matured(pending: dict[ReqId, _InFlight], now: float) -> set:
        done = {
            req_id for req_id, flight in pending.items() if flight.release_at_s <= now
        }
        for req_id in done:
            del pending[req_id]
        return done


class SimulatedKVConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler side: it suspends a remote fill, announces it, and hands off.

    It holds no clock. Timing belongs to the workers, which are where a
    transfer is announced and where it is reported finished; what is decided
    here is only which requests have one and what the other deployment is
    told.

    **The prompt is claimed once per request, and the request keeps its own
    intent.** The engine asks on every admission attempt, and a second claim
    on a request it has already suspended would suspend it again. Both real
    backends stop that by clearing the request's `do_remote_prefill` as they
    queue; the pull backend additionally sets a mark on the request and reads
    it back. This connector takes the mark and not the clearing, because the
    flag is what the request said about itself and the clearing happens at a
    point where nothing in that step has read it yet. The cost of declining it
    is named rather than claimed away: under the composite backend this call
    fans out to every sub-connector, so a second consumer beside this one
    would still see the flag and queue its own receive. That is a
    configuration this design does not contemplate -- a simulated deployment
    has no second consumer to pair with -- but it is the thing the clearing
    buys, and it is given up here.

    **The cost of the mark, which is the pull backend's.** A request matched
    on a step where it cannot be admitted -- the pool is full, or the batch is
    -- has spent its claim, and is prefilled locally on a later step rather
    than suspended. That is reproduced rather than improved on, because the
    engine's reads of the suspended state were written against it.

    **A transfer is announced only for a request the engine actually
    suspended.** What `update_state_after_alloc` takes is an offer, not an
    announcement; the announcement is made in `build_connector_meta`, which
    the engine calls once its whole admission pass is over. By then every
    suspension decision in that step is final and can be read off the
    request's own status, and an offer whose request was not suspended is
    dropped.

    That is deliberately not the same moment as clearing the flag, and cannot
    collapse into it. Clearing acts **on the request**, before the suspension
    has been decided, and destroys what the request said about itself on the
    way past. This acts **on this connector's own queue**, after the
    suspension has happened, and leaves the request exactly as it arrived.

    The case it exists for is the one the mark above creates: a request whose
    claim was spent on an earlier step reaches the allocation with its flag
    still set and is not suspended. Both real backends queue a receive for it
    anyway -- the guard on both is the flag alone -- and the workers then
    report a transfer finished against a request the scheduler never
    suspended, whose id lands on a list with no reachable pop, after which the
    engine rebuilds its waiting queue on every step for the rest of the run.
    Nothing is announced for it here. The divergence has a cost and it is one
    term wide: a deployment that really did issue that read spends the
    bandwidth, and this does not charge for it, so a run containing such a
    request under-reports by that request's block table -- once, because the
    offer is dropped rather than carried.

    **How much of the block table moves is not decided here, and the reason
    is the blob.** A consumer that already holds a prefix could take only the
    blocks past it, but the two deployments' block tables only correspond if
    they were launched with the same block size, and the field set this
    connector emits -- the pull backend's -- carries no block size for the
    consumer to compare against. The push backend, whose blob does carry one,
    transfers the whole table whenever that comparison cannot be made. So does
    this connector, and so does the pull backend in every case: the count of
    already-held blocks is named nowhere in its package. The consequence is
    worth stating in the direction it errs -- a consumer whose prefix cache
    holds part of the prompt is still charged for the whole of it, so a
    simulated run of that request reads slower than a deployment that could
    skip -- but no deployment on this field set can skip, so there is nothing
    here to compute that would not be an invention.
    """

    def __init__(self, config) -> None:
        kv_config = _kv_config(config)
        self.is_producer = _is_producer(kv_config)
        self._tp_size = config.tensor_parallel_size
        self._dp_rank = config.parallel_config.data_parallel_rank
        self._offered: dict[ReqId, tuple[Any, list[int]]] = {}

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        """Claim a remote-filled prompt whole, once, so the engine suspends it.

        The whole prompt is the claim because the producer computed all of it:
        there is no partial remote fill. The second element is what the engine
        reads to suspend the request, and it is the point of the method.
        """
        params = seq.kv_transfer_params or {}
        if params.get("do_remote_prefill") and not hasattr(seq, "kv_async_tagged"):
            seq.kv_async_tagged = True
            return len(seq.prompt_token_ids), True
        return 0, False

    def update_state_after_alloc(self, seq) -> None:
        """Offer the receive that a suspension would wait for.

        The engine calls this immediately after it allocates the request's
        blocks and immediately before it decides whether to suspend it, so the
        block table read here is the one a transfer would fill -- and whether
        there is going to be a transfer at all is not known yet. Nothing is
        announced from here, and the request is not touched.
        """
        params = seq.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return
        if self.is_producer:
            raise ValueError(
                f"request {seq.id!r} asks to be filled from another deployment "
                "while this connector was built as the producer side. Only the "
                "decoding side takes a remote fill on; a producer queueing one "
                "would wait for a transfer it is itself supposed to serve"
            )
        self._offered[seq.id] = (seq, list(seq.block_table))

    def build_connector_meta(self) -> ConnectorMetadata:
        """Announce the offers the engine suspended, and drop the rest.

        Called once the admission pass is over, so the status read here is
        the engine's settled answer. Cleared on the way out either way,
        because the workers own each announced transfer from here -- a queue
        that survived its own announcement is how one request gets two.
        """
        meta = ConnectorMetadata()
        for req_id, (seq, block_ids) in self._offered.items():
            if seq.status is not SequenceStatus.WAITING_FOR_REMOTE_KVS:
                continue
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=seq.kv_transfer_params or {},
            )
        self._offered.clear()
        return meta

    def request_finished(self, seq) -> None:
        """Attach the parameters the router relays to the next deployment.

        Attached on both sides, as both real backends do: the router reads it
        from the prefilling leg only, and a decode response carrying one costs
        nothing and keeps this to one path.
        """
        seq.kv_transfer_params_output = transfer_params(
            seq, tp_size=self._tp_size, dp_rank=self._dp_rank
        )
