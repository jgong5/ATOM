# SPDX-License-Identifier: MIT
"""A KV connector that moves no bytes and charges for the ones it would have.

Registered in the engine's connector factory under the name `compass`. It
stands in for Mooncake, the push backend: decode asks prefill to write a
prompt's KV into decode's blocks, and both learn when the write is done. Only
the scheduler half takes part, on the engine's step loop and the LP clock of
the `LPRuntime` installed in the engine's process:

- Decode, in `update_state_after_alloc`, sends the write request on the
  ``kv_write_req`` channel (arrival ``a``) and is ready at ``c + notify``,
  where the notify latency is that channel's lookahead.
- Prefill records a finished request's data-ready time ``r`` in
  `request_finished`, takes the write requests inline in `process_completions`
  and is done at ``c``. A write request for a request that has not
  finished (``r > a``) is refused: the router forwards to decode only after
  prefill has returned. A claimed request aborted before its write request
  is done at the abort.

Both report from `process_completions`, which the engine calls at every step
and every idle drain tick, one call after the first whose clock reached their
time: the engine's KV aggregation keeps one poll in flight, so a real poll
returns what the workers had seen at the previous one. ``c`` is when the
write completes on its rank pair's `Link`, which moves the pair's writes one
at a time, each its blocks and, as Mooncake moves a DeepSeek-V4 request's, its
state slot and then its staging row, priced by the `TransferModel` bound at
``kv_transfer_config[TRANSFER_KEY]``, or returned, as the connector is built, by
a callable bound there. Both ends feed it the same writes in the
same order, so write-done is not a message: both already know its time. The write
request travels over the zmq address at ``kv_transfer_config[WRITE_REQ_KEY]``,
which prefill binds and decode connects to.

Above DP width one, or above one deployment of a side, each decode and
prefill rank pair has its own channel and address, the configured port plus
the pair's index among the run's write-request channels. Decode sends to the
prefill deployment the relayed ``remote_engine_id`` names (`engine_id`), at
the rank its ``dp_rank`` names, and a prefill rank refuses a write request
for another rank.

Above prefill pipeline width one, the transfer model bound is a tuple, one
per prefill stage, each pricing that stage's layers on a `Link` of its own
per rank pair, and ``c`` is the latest stage's write-done. The write request
reaches the head stage only, which reports at ``c + 2 x notify``: decode is
ready at ``c + notify`` and sends Mooncake's release, which both ends time
as they time write-done. Stages above the head build an inert connector.

The worker half reports nothing. The engine still builds and polls it.
"""

from __future__ import annotations

import collections
import logging
import pickle

from atom.compass.kv.handoff import SIMULATED_ENGINE_ID, transfer_params, whole_number
from atom.compass.kv.transfer import Link, TransferModel
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata

logger = logging.getLogger(__name__)

#: Where the harness binds the priced transfer model, or a callable that
#: returns it once the engine's runner has planned its pool.
TRANSFER_KEY = "compass_transfer"

#: The zmq address the write requests travel over.
WRITE_REQ_KEY = "compass_kv_write_req"


def engine_id(lp: str) -> str:
    """The engine id a producer that is engine LP `lp` gives: one label, with
    the deployment's index when it is one of several (``engine-P1``), on any
    of its pipeline stages (``engine-P1-pp0``)."""
    lp = lp.partition("-pp")[0]
    index = lp.removeprefix("engine-P") if lp.startswith("engine-P") else ""
    return f"{SIMULATED_ENGINE_ID}-p{index}" if index else SIMULATED_ENGINE_ID


class UnboundSeam(RuntimeError):
    """The harness did not bind something the connector cannot invent."""


def _kv_config(config) -> dict:
    return getattr(config, "kv_transfer_config", {}) or {}


def _is_producer(kv_config: dict) -> bool:
    return kv_config.get("kv_role", "kv_producer") == "kv_producer"


def _bound_transfer(kv_config: dict) -> tuple[TransferModel, ...]:
    """The bound transfer model of each prefill stage, from a callable bound
    in its place called now."""
    model = kv_config.get(TRANSFER_KEY)
    if callable(model):
        model = model()
    models = (model,) if isinstance(model, TransferModel) else model
    if not (
        isinstance(models, tuple)
        and models
        and all(isinstance(m, TransferModel) for m in models)
    ):
        raise UnboundSeam(
            f"kv_transfer_config[{TRANSFER_KEY!r}] holds {model!r}, where this "
            "connector needs a TransferModel, or one per prefill stage, priced "
            "from the machine spec's interconnect and the paged KV block the "
            "engine's runner planned"
        )
    return models


class SimulatedKVConnector(KVConnectorBase):
    """Worker side: no device bytes, nothing to start, nothing to report."""

    def __init__(self, config) -> None:
        pass

    def register_kv_caches(
        self, kv_caches, transfer_tensors=None, num_blocks=None
    ) -> None:
        """Nothing to register: no KV tensor exists to expose to a remote."""

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        pass

    def get_finished(self) -> tuple[set, set]:
        return set(), set()


class SimulatedKVConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler side: Mooncake's write request, and both ends' completion times."""

    def __init__(self, config) -> None:
        import zmq

        from atom.utils import clock

        kv_config = _kv_config(config)
        self.is_producer = _is_producer(kv_config)
        # Checked here so a malformed width refuses the connector before any
        # request is served, not at the first request_finished.
        parallel = config.parallel_config
        self._tp_size = whole_number("tp_size", config.tensor_parallel_size)
        self._dp_rank = whole_number("dp_rank", parallel.data_parallel_rank)
        self._pp_size = whole_number("pp_size", config.pipeline_parallel_size)
        self._block_size = config.kv_cache_block_size
        self._dcp_size = config.decode_context_parallel_size
        self._hash_block_size = self._block_size * self._dcp_size
        self._transfer = _bound_transfer(kv_config)
        if self.is_producer and len(self._transfer) != self._pp_size:
            raise ValueError(
                f"the bound transfer models price {len(self._transfer)} prefill "
                f"stages, and this deployment runs {self._pp_size}"
            )
        self._rt = clock.installed()
        if self._rt is None:
            raise UnboundSeam("no LP runtime is installed to time transfers on")
        self._ready: dict = {}  # prefill: request -> r
        self._due: dict = {}  # request -> when its transfer is done
        # Each pair's links, one per prefill stage, by the rank at the other end.
        self._links = collections.defaultdict(lambda: [Link(m) for m in self._transfer])
        self._snapshot: list = []  # done at the last poll, reported at the next
        # Mooncake's send claim: a producer keeps a request's blocks from the
        # pool from `request_finished` until its write is reported done.
        self._awaiting_send: set[str] = set()
        # Requests aborted while claimed and before any write request.
        self._aborted: set = set()
        self._socks: dict[tuple[str, int], clock.WrappedSocket] = {}
        if getattr(parallel, "pipeline_parallel_rank", 0):
            return  # Mooncake reports a send on the head stage only.
        address = kv_config.get(WRITE_REQ_KEY)
        if not isinstance(address, str):
            raise UnboundSeam(
                f"kv_transfer_config[{WRITE_REQ_KEY!r}] holds {address!r}, where "
                "this connector needs the zmq address of its write requests"
            )
        table, me = self._rt.table, self._rt.me
        pairs = sorted(
            c.name
            for lp in table.registry.ids()
            for c in table.channels_from(lp)
            if c.name.partition(":")[2].startswith("kv_write_req")
        )
        # This rank's end of each pair, by the engine LP and the rank at the
        # other end; the k-th pair's address is the configured one's port plus k.
        for k, channel in enumerate(pairs):
            ends = table.channel(channel).source, table.channel(channel).target
            if me not in ends:
                continue
            d, _, p = channel.partition("#")[2].replace("dp", "").partition(".")
            mine, other = (p, d) if self.is_producer else (d, p)
            if mine and int(mine) != self._dp_rank:
                continue
            head, _, port = address.rpartition(":")
            raw = zmq.Context.instance().socket(
                zmq.PULL if self.is_producer else zmq.PUSH
            )
            (raw.bind if self.is_producer else raw.connect)(
                f"{head}:{int(port) + k}" if k else address
            )
            peer = ends[ends[0] == me].name
            self._socks[peer, int(other or 0)] = clock.WrappedSocket(
                self._rt, raw, channel
            )
        if not self._socks:
            raise UnboundSeam(
                f"{me} at DP rank {self._dp_rank} is no end of a kv_write_req "
                f"channel; its table declares {pairs}"
            )
        self._wide = "#" in pairs[0]
        self._peers = {engine_id(peer): peer for peer, _ in self._socks}
        # The write request's latency, and the write-done notice's.
        self._latency_s = table.lookahead(pairs[0])

    def _written(self, pair, a: float, blocks: int) -> float:
        """When the last prefill stage's write of `blocks`, arriving at `a` on
        rank pair `pair`, is done."""
        return max(link.start(a, blocks) for link in self._links[pair])

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        """Claim a remote-filled prompt whole, so the engine parks it."""
        params = seq.kv_transfer_params or {}
        if params.get("do_remote_prefill") and not getattr(
            seq, "kv_async_tagged", False
        ):
            return len(seq.prompt_token_ids), True
        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        return ConnectorMetadata()

    def _pages_line_up(self, params) -> bool:
        """Mooncake's rule for skipping decode's cached prefix: the producer
        has this side's block size and runs DCP 1 or this side's DCP. A blob
        without both fields lines up when its ``hash_block_size`` is this
        side's block size or its hash block size."""
        block_size, dcp_size = params.get("block_size"), params.get("dcp_size")
        if block_size is not None and dcp_size is not None:
            return int(block_size) == self._block_size and int(dcp_size) in (
                1,
                self._dcp_size,
            )
        return params.get("hash_block_size") in (
            self._block_size,
            self._hash_block_size,
        )

    def update_state_after_alloc(self, seq) -> None:
        """Decode: send the write request and clear the flag, as Mooncake does.

        The engine parks the request right after this call, so every write
        request has a parked request waiting for it.
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
        params["do_remote_prefill"] = False
        stages = params.get("remote_pp_size", 1)
        if stages != len(self._transfer):
            raise ValueError(
                f"request {seq.id!r} was prefilled over {stages} pipeline stages, "
                f"and this connector prices {len(self._transfer)}"
            )
        computed = 0
        if (
            self._pages_line_up(params)
            and self._hash_block_size > 0
            and not seq.has_per_req_cache
        ):
            computed = seq.num_cached_tokens // self._hash_block_size
        blocks = len(seq.block_table) - computed
        # The arrival `stamp_send` gives this send.
        a = self._rt.now + self._latency_s
        # To the prefill deployment and rank that served the request.
        peers = self._peers
        peer = (
            peers[params["remote_engine_id"]]
            if len(peers) > 1
            else next(iter(peers.values()))
        )
        key = peer, params["dp_rank"] if self._wide else 0
        self._socks[key].send(pickle.dumps((params["transfer_id"], blocks, a, key[1])))
        self._due[seq.id] = self._written(key, a, blocks) + self._latency_s

    def request_finished(self, seq) -> None:
        """Record prefill's data-ready time, once, and attach the relayed blob.

        A producer claims the blocks of a request it will send, on Mooncake's
        condition: not aborted, and `do_remote_decode` set.
        """
        if self.is_producer:
            self._ready.setdefault(seq.id, self._rt.read_clock())
            if getattr(seq, "leave_reason", None) != "aborted" and (
                getattr(seq, "kv_transfer_params", None) or {}
            ).get("do_remote_decode"):
                self._awaiting_send.add(str(seq.id))
        seq.kv_transfer_params_output = transfer_params(
            seq,
            tp_size=self._tp_size,
            dp_rank=self._dp_rank,
            pp_size=self._pp_size,
            block_size=self._block_size,
            dcp_size=self._dcp_size,
            engine_id=engine_id(self._rt.me.name),
        )

    def should_defer_free(self, seq) -> bool:
        return str(seq.id) in self._awaiting_send

    def send_finished(self, req_id) -> None:
        self._awaiting_send.discard(str(req_id))

    def abort_send(self, req_id) -> None:
        """Mooncake's abort of a claimed request: a write request taken by now,
        as Mooncake's listener takes one on arrival, reports when it would
        have; with none, the send is done now."""
        self._take_write_requests()
        pending = req_id in self._due or req_id in self._snapshot
        if str(req_id) not in self._awaiting_send or pending:
            return
        self._ready.pop(req_id, None)
        self._aborted.add(req_id)
        self._due[req_id] = self._rt.read_clock()

    def source_blocks_released(self, seq) -> None:
        """No block-lifetime state remains after the send claim is retired."""

    def _write_requests(self):
        """Prefill: each write request released so far, pair by pair."""
        for other, sock in self._socks.items() if self.is_producer else ():
            while sock.poll(0):
                yield other, pickle.loads(sock.recv())

    def _take_write_requests(self) -> None:
        """Prefill: time each write request released so far on its link."""
        for other, (req_id, blocks, a, rank) in self._write_requests():
            if rank != self._dp_rank:
                raise ValueError(
                    f"the write request for {req_id!r} is for prefill DP rank "
                    f"{rank} and reached rank {self._dp_rank}"
                )
            if req_id in self._aborted:
                # Mooncake refuses it. Decode, which cannot learn that, timed
                # the write on its end of the link, and this end keeps in step.
                self._written(other, a, blocks)
                logger.info("compass: a write request for aborted %s", req_id)
                continue
            r = self._ready.pop(req_id, None)
            if r is None or r > a:
                raise ValueError(
                    f"the write request for {req_id!r} arrived at {a}, before "
                    f"its prefill finished (at {r}); the router sends a request "
                    "to decode only after prefill has returned"
                )
            # Above one stage, decode's write-done notice and its release back.
            release = 2 * self._latency_s if len(self._transfer) > 1 else 0.0
            self._due[req_id] = self._written(other, a, blocks) + release
            logger.info(
                "compass: KV transfer %s, %d blocks, from %r to %r simulated s",
                req_id,
                blocks,
                a,
                self._due[req_id],
            )

    def process_completions(self, output):
        """Take the write requests released so far; report what the clock had
        reached at the previous call."""
        self._take_write_requests()
        # The engine's aggregation keeps one poll in flight, so a poll returns
        # the snapshot the workers took at the previous poll: report that one.
        now = self._rt.read_clock()
        done, self._snapshot = self._snapshot, [
            req_id for req_id, t in self._due.items() if t <= now
        ]
        for req_id in self._snapshot:
            del self._due[req_id]
        if self.is_producer:
            output.finished_sending.update(done)
        else:
            output.finished_recving.update(done)
        return output

    def has_pending_work(self) -> bool:
        """Keeps an idle decode engine polling until its parked requests are ready."""
        return bool(self._due or self._snapshot)
