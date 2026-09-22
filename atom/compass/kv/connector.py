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
elsewhere, which is what suspends the request, and queues the receive that the
workers then carry and report; the suspension has something waiting on it and
the report has somewhere to go. A producer that finishes a request hands back
the parameters the router relays to the deployment that will decode it.
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
    """Scheduler side: it suspends a remote fill, queues it, and hands off.

    It holds no clock. Timing belongs to the workers, which are where a
    transfer is announced and where it is reported finished; what is decided
    here is only which requests have one and what the other deployment is
    told.

    **The prompt is claimed once per request.** The engine asks on every
    admission attempt, and a second claim on a request it has already
    suspended would suspend it again. The mark that makes the claim once-only
    is the request attribute the pull backend sets, and it is the reason this
    connector never clears the request's own `do_remote_prefill`: the push
    backend guards the same thing by clearing that flag instead, which
    destroys the request's stated intent on its way past and hides it from
    anything downstream that had not looked yet. The cost of the mark is that
    a request matched on a step where it cannot be admitted -- the pool is
    full, or the batch is -- has spent its claim, and is prefilled locally on
    a later step rather than suspended. That is the pull backend's behaviour
    at this head and it is reproduced rather than improved on, because the
    engine's five reads of the suspended state were written against it.

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
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}

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
        """Queue the receive the suspension is waiting for.

        The engine calls this immediately after it allocates the request's
        blocks and immediately before it decides to suspend it, so the block
        table read here is the one the transfer will fill.
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
        self._reqs_need_recv[seq.id] = (seq, list(seq.block_table))

    def build_connector_meta(self) -> ConnectorMetadata:
        """Hand this step's queued receives to the workers, and forget them.

        Cleared on the way out because the workers own each transfer from
        here: announcing one twice is refused on their side, and a queue that
        survived its own announcement is how that happens.
        """
        meta = ConnectorMetadata()
        for req_id, (seq, block_ids) in self._reqs_need_recv.items():
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=seq.kv_transfer_params or {},
            )
        self._reqs_need_recv.clear()
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
