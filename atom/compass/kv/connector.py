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

Two things are deliberately absent, and a deployment driven through the HTTP
router will not work until they arrive: the parked remote-prefill path, where
a consumer declares the whole prompt matched and waits, and the transfer
parameter blob the router relays between the two deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from atom.compass.kv.transfer import TransferModel
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, ReqId, ReqMeta

logger = logging.getLogger("atom")

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
    """Scheduler side: queues what the workers are to move, and nothing more.

    It holds no clock. Timing belongs to the workers, which are where the
    transfer is announced and where it is reported finished.
    """

    def __init__(self, config) -> None:
        kv_config = _kv_config(config)
        self.is_producer = _is_producer(kv_config)
        self._reqs_need_recv: dict[ReqId, tuple] = {}

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        """No remote match is claimed here, so no request is parked.

        The path that declares the whole prompt already held on the producer,
        and makes the scheduler wait for it, is not implemented yet.
        """
        return 0, False

    def update_state_after_alloc(self, seq) -> None:
        """Queue a consumer's request for the receive its blocks are for."""
        params = seq.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return
        assert not self.is_producer, "only the consumer side receives remote KV"
        self._reqs_need_recv[seq.id] = (seq, list(seq.block_table))
        params["do_remote_prefill"] = False

    def build_connector_meta(self) -> ConnectorMetadata:
        """This step's announcements, handed to the workers and then dropped."""
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
        """Nothing is attached to a finished request yet.

        The producer's answer here is the parameter blob the HTTP router
        relays to the other deployment, which is not written yet; a partial
        one would be relayed and believed, so none is written.
        """
