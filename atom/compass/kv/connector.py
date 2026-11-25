# SPDX-License-Identifier: MIT
"""A KV connector that moves no bytes and charges for the ones it would have.

Registered in the engine's connector factory under the name `compass`. It
stands in for Mooncake, the push backend: decode asks prefill to write a
prompt's KV into decode's blocks, and both learn when the write is done. Only
the scheduler half takes part, on the engine's step loop and the LP clock of
the `LPRuntime` installed in the engine's process:

- Decode, in `update_state_after_alloc`, queues the write request, and
  sends it on the ``kv_write_req`` channel (arrival ``a``) once the device
  has run the forward its engine dispatches next, the admission step's or,
  above DP width one, the group's dummy that follows an admission step with
  nothing to run (`device_done`); with no forward to follow, at its KV
  poll, once the last forward's device work is done. Mooncake's
  write-request thread synchronizes the device before it posts what was
  queued (`_post_queued_write_requests`), and the engine has launched the
  admission step's forward by then, so under deferred output the request
  leaves as the step after the admission step returns. Decode is ready at
  ``c + notify``, where the notify latency is that channel's lookahead.
- Prefill records a finished request's data-ready time ``r`` in
  `request_finished`, takes the write requests inline in `process_completions`
  and is done at ``c``. A write request for a request that has not
  finished (``r > a``) is refused: the router forwards to decode only after
  prefill has returned. A claimed request aborted before its write request
  is done at the abort.
- A decode request that finishes before its write request, aborted while it
  waited, sends prefill a notice on the same channel instead, as Mooncake's
  consumer does, and prefill's send is done at the notice's arrival.

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

Under a link that states a NIC's bandwidth (`nic_bandwidth_bytes_per_s`,
opt-in; `SharedNic`), the writes into one decode rank share its NIC rather
than queue on their rank pair's `Link`: decode, which posts every write into
it, both P ranks' alike, times them on one `SharedNic` and learns a write's
write-done once it has advanced that NIC past it; prefill times its own writes
into that rank on one of its own, so it frees its blocks no later than decode
learns of the write, and earlier by what another prefill rank's writes into
the same NIC cost. One prefill stage only.

Under ``enable_chunked_transfer`` on the producer, a request the router sent
both sides at once (``chunked_transfer``, the transfer id shared) transfers as
Mooncake's chunked path does: decode admits it and sends its write request
(the 5-tuple's last field marks it chunked); prefill's `admission_ready`
holds the request until that write request is in, past
``chunked_admission_timeout_s`` aborting it; once allocated, its prefill
publishes the prompt's blocks at each forward's device end
(`forward_returned`, the scheduled batch's ``context_lens``), and a
`ChunkedSend` reads them onto the pair's link as they come
(`transfer.chunk_writes`). At the handoff, prefill's send is done at the last
write and the handoff, ``c``, and it sends decode a write-done notice on
``kv_write_done`` (prefill to decode, the address at
``kv_transfer_config[WRITE_DONE_KEY]``), arriving at ``c + notify``, when
decode is ready. Decode takes the handoff's first token as the prompt's last.

Above one prefill stage each stage writes its own layers, as each of
Mooncake's stage workers serves its own reads: the head relays each admitted
chunked request (its transfer id, the write request's arrival and the blocks
to write) in its connector metadata, which the scheduled batch carries to every
stage (`relayed`); a stage's `ChunkedSend` reads its forwards' publications onto
its own stage link and sends its own notice on its ``kv_write_done``. Decode is
ready once every stage's notice is in, and tells the head (a release on the
write-request channel), whose blocks are freed then, as Mooncake's head frees
the shared page table on the consumer's release. On a `Link`; a shared NIC is
refused.

The worker half reports nothing. The engine still builds and polls it.
"""

from __future__ import annotations

import collections
import logging
import math
import pickle
import weakref
from typing import ClassVar

from atom.compass.kv.handoff import SIMULATED_ENGINE_ID, transfer_params, whole_number
from atom.compass.kv.transfer import (
    ChunkedSend,
    Link,
    SharedNic,
    TransferModel,
    chunk_writes,
)
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

#: The zmq address a chunked transfer's write-done notices travel over, from
#: prefill to decode, which binds it.
WRITE_DONE_KEY = "compass_kv_write_done"

#: Mooncake's default for ``chunked_admission_timeout_s``.
CHUNKED_ADMISSION_TIMEOUT_S = 300.0


def engine_id(lp: str) -> str:
    """The engine id a producer that is engine LP `lp` gives: one label, with
    the deployment's index when it is one of several (``engine-P1``), on any
    of its pipeline stages (``engine-P1-pp0``)."""
    lp = lp.partition("-pp")[0]
    index = lp.removeprefix("engine-P") if lp.startswith("engine-P") else ""
    return f"{SIMULATED_ENGINE_ID}-p{index}" if index else SIMULATED_ENGINE_ID


#: The consumers built in this process, each told of every forward its
#: engine's workers return (`forward_returned`).
_CONSUMERS: weakref.WeakSet[SimulatedKVConnectorScheduler] = weakref.WeakSet()
#: The chunked producers built in this process, each told of every forward and
#: the batch it ran.
_CHUNKED: weakref.WeakSet[SimulatedKVConnectorScheduler] = weakref.WeakSet()


def forward_returned(reply, batch=None) -> None:
    """Tell each consumer built in this process that its engine's workers
    returned the forward, or the DP group's dummy, whose reply is `reply`: its
    device work ends at the reply's ``device_end``, the runner's
    `HostDeviceRecurrence` end, or with none at its return; and each chunked
    producer which prompts the forward's `batch` computed, by then.
    `atom.compass.run.engine` has the engine's `call_func` call this."""
    end = getattr(reply, "device_end", None)
    for consumer in list(_CONSUMERS):
        consumer.device_done(end)
    if batch is not None:
        for producer in list(_CHUNKED):
            producer.prefill_done(end, batch)


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

    # The opt-in switches' state, off: what `__init__` replaces when one is on,
    # and what a connector built without it reads (no shared NIC, nothing in
    # flight on a NIC).
    _shared = False
    _nics: ClassVar[dict] = {}
    _extra: ClassVar[dict] = {}

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
        # Opt-in: the writes into one decode rank's NIC share it (`SharedNic`).
        # Decode times every write into its rank on one NIC, prefill its own
        # writes into each decode rank on one NIC per pair.
        self._shared = self._transfer[0].nic_bandwidth_bytes_per_s > 0
        if self._shared and len(self._transfer) > 1:
            raise ValueError(
                "a shared NIC prices one prefill stage's writes; this deployment "
                f"prices {len(self._transfer)}"
            )
        self._nics = collections.defaultdict(lambda: SharedNic(self._transfer[0]))
        self._extra: dict = {}  # request -> what its due adds to its write-done
        self._snapshot: list = []  # done at the last poll, reported at the next
        # Mooncake's send claim: a producer keeps a request's blocks from the
        # pool from `request_finished` until its write is reported done.
        self._awaiting_send: set[str] = set()
        # Requests aborted while claimed and before any write request.
        self._aborted: set = set()
        # Decode: (request, blob, blocks) of each write request queued for
        # the device, and when the last forward's device work ends.
        self._queued: list = []
        self._device_end = -math.inf
        # Whether the last KV poll found write requests queued with no forward
        # since: above DP width one the group's dummy normally follows it.
        self._polled_queued = False
        # Above DP width one an admission step with nothing to run is followed
        # by the group's dummy forward.
        self._dummies = getattr(parallel, "data_parallel_size", 1) > 1
        self._socks: dict[tuple[str, int], clock.WrappedSocket] = {}
        # Mooncake's chunked transfer, on the producer's switch.
        self._chunked = bool(kv_config.get("enable_chunked_transfer", False))
        self._admission_timeout_s = float(
            kv_config.get("chunked_admission_timeout_s", CHUNKED_ADMISSION_TIMEOUT_S)
        )
        self._reads: dict = {}  # prefill: transfer id -> (pair, blocks, a)
        self._waiting: dict = {}  # prefill: request -> its admission deadline
        # prefill: request -> (ChunkedSend, pair, transfer id, prompt tokens)
        self._sends: dict = {}
        self._notices: list = []  # prefill: (c, pair, transfer id) to send
        self._done_socks: dict = {}  # each pair's write-done notice socket
        self._listening: dict = (
            {}
        )  # decode: transfer id -> [request, notices due, latest]
        self._stage = int(getattr(parallel, "pipeline_parallel_rank", 0) or 0)
        self._relay: dict = {}  # prefill head: what the stages are told of new sends
        self._released: dict = (
            {}
        )  # prefill head: request -> its send, done, awaiting release
        if self._stage:
            # Mooncake reports a send on the head stage only; under a chunked
            # transfer a later stage writes its own layers (`relayed`).
            if self.is_producer and self._chunked:
                table = self._rt.table
                pairs = [
                    c.name
                    for c in table.channels_from(self._rt.me)
                    if ":kv_write_done" in c.name
                ]
                if pairs:
                    self._latency_s = table.lookahead(pairs[0])
                self._bind_write_done(kv_config, table, self._rt.me)
                _CHUNKED.add(self)
            return
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
        if not self.is_producer:
            _CONSUMERS.add(self)
        self._bind_write_done(kv_config, table, me)
        if self.is_producer and self._chunked:
            if self._shared:
                raise ValueError(
                    "a chunked transfer is priced on its rank pair's link; this "
                    "deployment's link states a shared NIC"
                )
            _CHUNKED.add(self)

    def _bind_write_done(self, kv_config: dict, table, me) -> None:
        """Each rank pair's write-done notice socket, where the run declares
        ``kv_write_done`` channels: prefill sends, decode receives."""
        import zmq

        from atom.utils import clock

        pairs = sorted(
            c.name
            for lp in table.registry.ids()
            for c in table.channels_from(lp)
            if c.name.partition(":")[2].startswith("kv_write_done")
        )
        if not pairs:
            return
        address = kv_config.get(WRITE_DONE_KEY)
        if not isinstance(address, str):
            raise UnboundSeam(
                f"kv_transfer_config[{WRITE_DONE_KEY!r}] holds {address!r}, where "
                "a chunked transfer's connector needs the zmq address of its "
                "write-done notices"
            )
        for k, channel in enumerate(pairs):
            ends = table.channel(channel).source, table.channel(channel).target
            if me not in ends:
                continue
            p, _, d = channel.partition("#")[2].replace("dp", "").partition(".")
            mine, other = (p, d) if self.is_producer else (d, p)
            if mine and int(mine) != self._dp_rank:
                continue
            head, _, port = address.rpartition(":")
            raw = zmq.Context.instance().socket(
                zmq.PUSH if self.is_producer else zmq.PULL
            )
            (raw.connect if self.is_producer else raw.bind)(
                f"{head}:{int(port) + k}" if k else address
            )
            peer = ends[ends[0] == me].name
            self._done_socks[peer, int(other or 0)] = clock.WrappedSocket(
                self._rt, raw, channel
            )

    def _written(self, pair, a: float, blocks: int) -> float:
        """When the last prefill stage's write of `blocks`, arriving at `a` on
        rank pair `pair`, is done."""
        return max(link.start(a, blocks) for link in self._links[pair])

    def _write(self, pair, req_id, a: float, blocks: int, extra: float) -> None:
        """Time `req_id`'s write of `blocks`, arriving at `a` on rank pair
        `pair`: due `extra` after its write-done, at once on the pair's link,
        or under a shared NIC once `_settle` has advanced the NIC past it. A
        `req_id` of None is timed and reported to no one."""
        if not self._shared:
            done = self._written(pair, a, blocks)
            if req_id is not None:
                self._due[req_id] = done + extra
            return
        # Decode's writes all share its own NIC; prefill's, its pair's.
        self._nics[None if not self.is_producer else pair].add(req_id, a, blocks)
        if req_id is not None:
            self._extra[req_id] = extra

    def _settle(self, now: float) -> None:
        """Under a shared NIC: each write whose bytes have moved by `now`
        is due at its write-done."""
        for nic in self._nics.values():
            for req_id, done in nic.advance(now):
                if req_id is not None:
                    self._due[req_id] = done + self._extra.pop(req_id)

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        """Claim a remote-filled prompt whole, so the engine parks it."""
        params = seq.kv_transfer_params or {}
        if params.get("do_remote_prefill") and not getattr(
            seq, "kv_async_tagged", False
        ):
            return len(seq.prompt_token_ids), True
        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        meta = ConnectorMetadata()
        if self._relay:
            # The stages' share of each chunked send admitted since (`relayed`).
            meta.compass_chunked, self._relay = self._relay, {}
        return meta

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
        """Decode: queue the write request for the device and clear the flag,
        as Mooncake does; it leaves at `device_done` or at the KV poll.

        The engine parks the request right after this call, so every write
        request has a parked request waiting for it.
        """
        params = seq.kv_transfer_params or {}
        if (
            self.is_producer
            and params.get("chunked_transfer")
            and params.get("do_remote_decode")
            and seq.id not in self._sends
        ):
            self._chunked_alloc(seq, params)
            return
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
        if params.get("chunked_transfer"):
            # Its prompt pages, not the lookahead past them (`_prompt_block_ids`).
            page = self._block_size * self._dcp_size
            blocks = max(0, -(-seq.num_prompt_tokens // page) - computed)
            self._listening[params["transfer_id"]] = [
                seq,
                params.get("remote_pp_size", 1),
                -math.inf,
            ]
        self._queued.append((seq.id, params, blocks))

    def device_done(self, end: float | None) -> None:
        """Decode: the forward the engine dispatched last returned, its device
        work ending at `end`, or with None at its return: every write request
        queued before it leaves then, no earlier than now. Mooncake's
        write-request thread synchronizes the device, and with it this
        forward, which the engine launched first, before it posts them."""
        now = self._rt.now
        self._device_end = max(self._device_end, now if end is None else end)
        self._post(self._device_end)
        self._polled_queued = False

    def _post(self, t: float) -> None:
        """Decode: send every queued write request, leaving at `t`, no earlier
        than now, and time its write from its arrival."""
        a = max(t, self._rt.now) + self._latency_s
        for req_id, params, blocks in self._queued:
            key, _ = self._send(params, blocks, a)
            if not params.get("chunked_transfer"):
                self._write(key, req_id, a, blocks, self._latency_s)
            # A chunked transfer's write-done comes from prefill (`_heard`).
        self._queued.clear()

    def _send(self, params, blocks, a=None) -> tuple[tuple[str, int], float]:
        """Decode: send the request's write request, or with `blocks` None its
        notice that none will come, to the prefill deployment and rank that
        served it, arriving at `a`, by default after the channel's latency;
        its rank pair and its arrival there."""
        peers = self._peers
        peer = (
            peers[params["remote_engine_id"]]
            if len(peers) > 1
            else next(iter(peers.values()))
        )
        chunked = bool(params.get("chunked_transfer"))
        # A chunked request was sent here by the router, which names prefill's
        # DP rank as remote_dp_rank, never prefill's own dp_rank.
        rank = params.get("remote_dp_rank", 0) if chunked else params.get("dp_rank")
        key = peer, rank if self._wide else 0
        a = self._rt.now + self._latency_s if a is None else a
        said = (params["transfer_id"], blocks, a, key[1])
        message = pickle.dumps(said + (True,) if chunked else said)
        self._socks[key].send(message, at=a)
        return key, a

    def request_finished(self, seq) -> None:
        """Record prefill's data-ready time, once, and attach the relayed blob.

        A producer claims the blocks of a request it will send, on Mooncake's
        condition: not aborted, and `do_remote_decode` set. A consumer request
        that finishes with `do_remote_prefill` still set never sent its write
        request, so decode tells prefill that none will come, once.
        """
        params = getattr(seq, "kv_transfer_params", None) or {}
        if self.is_producer and params.get("chunked_transfer"):
            self._handoff(seq)
            return
        if self.is_producer:
            self._ready.setdefault(seq.id, self._rt.read_clock())
            if getattr(seq, "leave_reason", None) != "aborted" and params.get(
                "do_remote_decode"
            ):
                self._awaiting_send.add(str(seq.id))
        elif params.get("do_remote_prefill"):
            params["do_remote_prefill"] = False
            self._send(params, None)
        seq.kv_transfer_params_output = transfer_params(
            seq,
            tp_size=self._tp_size,
            dp_rank=self._dp_rank,
            pp_size=self._pp_size,
            block_size=self._block_size,
            dcp_size=self._dcp_size,
            engine_id=engine_id(self._rt.me.name),
        )

    # -- Mooncake's chunked transfer ------------------------------------------

    def admission_ready(self, seq) -> bool:
        """Prefill: whether a chunked request may allocate, which it may once
        its consumer's write request is in (Mooncake's ``pd_admission_ready``);
        past ``chunked_admission_timeout_s`` waiting it is aborted, as
        Mooncake's scheduler aborts it. Any other request may at once."""
        from atom.model_engine.sequence import SequenceStatus

        params = seq.kv_transfer_params or {}
        if not (self.is_producer and params.get("chunked_transfer")):
            return True
        if not self._chunked or not params.get("transfer_id"):
            raise ValueError(
                "chunked admission requires enable_chunked_transfer and transfer_id"
            )
        if str(seq.id) in self._awaiting_send:
            return True
        if seq.status == SequenceStatus.ABORTED:
            return False
        self._take_write_requests()
        if params["transfer_id"] in self._reads:
            self._waiting.pop(seq.id, None)
            return True
        now = self._rt.now
        deadline = self._waiting.setdefault(seq.id, now + self._admission_timeout_s)
        if now < deadline:
            return False
        logger.warning(
            "compass: chunked P/D admission timed out for request %s", seq.id
        )
        self._waiting.pop(seq.id, None)
        seq.status = SequenceStatus.ABORTED
        seq.leave_reason = "aborted"
        self.request_finished(seq)
        return False

    def _chunked_alloc(self, seq, params) -> None:
        """Prefill: a chunked request allocated its prompt; its consumer's
        reads start from the write request's arrival, its blocks pinned."""
        pair, blocks, a = self._reads.pop(params["transfer_id"])
        total = -(-seq.num_prompt_tokens // (self._block_size * self._dcp_size))
        send = ChunkedSend(self._links[pair][0], a, max(0, total - blocks), total)
        self._sends[seq.id] = (send, pair, params["transfer_id"], seq.num_prompt_tokens)
        self._awaiting_send.add(str(seq.id))
        if self._pp_size > 1:
            self._relay[seq.id] = (
                params["transfer_id"],
                pair,
                a,
                blocks,
                seq.num_prompt_tokens,
            )

    def relayed(self, batch) -> None:
        """A later stage: start a send for each chunked request the head
        admitted, its metadata riding the batch (`build_connector_meta`)."""
        meta = getattr(batch, "connector_meta_output", None)
        metas = getattr(meta, "metas", None) or ([meta] if meta is not None else [])
        page = self._block_size * self._dcp_size
        for m in metas:
            for req_id, (tid, pair, a, blocks, prompt) in (
                getattr(m, "compass_chunked", None) or {}
            ).items():
                total = -(-prompt // page)
                link = self._links[pair][self._stage]
                send = ChunkedSend(link, a, max(0, total - blocks), total)
                self._sends[req_id] = (send, pair, tid, prompt)

    def prefill_done(self, end, batch) -> None:
        """Prefill: the forward that ran `batch` returned, its device work
        ending at `end` (None: now): each chunked request's prompt computed
        through its ``context_lens`` is published then."""
        if self._stage:
            self.relayed(batch)
        if not self._sends:
            return
        t = self._rt.now if end is None else end
        prefills = getattr(batch, "total_seqs_num_prefill", len(batch.req_ids))
        for req_id, tokens in zip(
            batch.req_ids[:prefills], list(batch.context_lens)[:prefills]
        ):
            if (held := self._sends.get(req_id)) is not None:
                page = self._block_size * self._dcp_size
                held[0].publish(t, int(tokens), held[3], page)
                if self._stage and int(tokens) >= held[3]:
                    # A later stage hands off with its last forward: the head
                    # samples once the last stage's has run.
                    held[0].finish(t)
        if self._stage:
            # A later stage takes no step loop's poll: it settles at its
            # forwards, deciding every send to its end as far as what is
            # published allows -- one whose prompt is all published to its
            # end, its notice stamped to arrive then -- in time order across
            # them (`ChunkedSend.step`: a later publication's write queues
            # behind what is decided).
            chunk_writes([v[0] for v in self._sends.values()], math.inf)
            self._settle_chunked(self._rt.now)

    def _handoff(self, seq) -> None:
        """Prefill: a chunked request finished its prefill, or was aborted;
        its handoff time, with nothing relayed (the router drains the reply)."""
        held = self._sends.get(seq.id)
        aborted = getattr(seq, "leave_reason", None) == "aborted"
        if held is not None and not aborted:
            held[0].finish(self._rt.read_clock())
        elif held is not None:
            # Its reads stop at once; what is written stays written.
            del self._sends[seq.id]
            self._due[seq.id] = self._rt.read_clock()
        self._waiting.pop(seq.id, None)
        seq.kv_transfer_params_output = None

    def _settle_chunked(self, now: float) -> None:
        """Prefill: decide the chunked sends up to `now`; each one done is
        due at its write-done, and its notice leaves for decode."""
        if not self._sends:
            return
        sends = {id(v[0]): (req_id, v) for req_id, v in self._sends.items()}
        chunk_writes([v[0] for v in self._sends.values()], now)
        # Each send done, now or decided to its end at its last forward.
        finished = sorted(
            (v[0] for v in self._sends.values() if v[0].done is not None),
            key=lambda send: send.done,
        )
        for send in finished:
            req_id, (_, pair, transfer_id, _) = sends[id(send)]
            del self._sends[req_id]
            if self._stage:
                pass  # the head's scheduler holds the blocks; it frees them
            elif self._pp_size > 1:
                # Freed on decode's release, once every stage's write is done.
                self._released[transfer_id] = req_id
            else:
                self._due[req_id] = send.done
            self._notices.append((send.done, pair, transfer_id))
            logger.info(
                "compass: chunked KV transfer %s, %d writes, %d blocks, from %r "
                "to %r simulated s%s",
                req_id,
                len(send.writes),
                sum(w[2] for w in send.writes),
                send.writes[0][0] if send.writes else send.done,
                send.done,
                (
                    f", queued {send.clamped:.6f} s behind writes decided before"
                    if send.clamped
                    else ""
                ),
            )
        for c, pair, transfer_id in self._notices:
            sock = self._done_socks.get(pair)
            if sock is None:
                raise UnboundSeam(
                    f"no kv_write_done channel to {pair}: the run declares none, "
                    "so a chunked transfer cannot tell decode its write is done"
                )
            at = max(c, self._rt.now) + self._latency_s
            sock.send(pickle.dumps((transfer_id, at)), at=at)
        self._notices.clear()

    def _heard(self) -> None:
        """Decode: each write-done notice released so far: its request is
        ready at the notice's arrival, its first token the prompt's last."""
        for sock in self._done_socks.values() if not self.is_producer else ():
            while sock.poll(0):
                transfer_id, at = pickle.loads(sock.recv())
                held = self._listening.get(transfer_id)
                if held is None:
                    continue
                seq, left, latest = held
                latest = max(latest, at)
                if left > 1:
                    self._listening[transfer_id] = [seq, left - 1, latest]
                    continue
                del self._listening[transfer_id]
                params = seq.kv_transfer_params
                params.setdefault("first_token_id", int(seq.prompt_token_ids[-1]))
                params.setdefault("draft_token_ids", [])
                self._due[seq.id] = latest
                if params.get("remote_pp_size", 1) > 1:
                    # Mooncake's release: the head may reuse its page table.
                    self._send(
                        params, "release", max(latest, self._rt.now) + self._latency_s
                    )

    @property
    def supports_pd_admission(self) -> bool:
        """Decode admits a receive under ``pd_admission`` as Mooncake's
        consumer does (`PdAdmissionController`): this connector's decode side
        receives the range `pd_receive_block_offset` names."""
        return not self.is_producer

    def pd_receive_block_offset(self, seq, num_blocks, num_cached_blocks):
        """The first block of `seq`'s receive, which `pd_admission` charges up
        to `num_blocks`, or None where it receives nothing: Mooncake's rule
        (`_receive_block_offset`). A local prefix of `num_cached_blocks` is
        skipped where the producer's pages line up with this side's
        (`_pages_line_up`, as `update_state_after_alloc` skips it) and a
        legacy transfer's source holds more blocks than the prefix; a prefix
        that is the whole prompt is received whole."""
        params = seq.kv_transfer_params or {}
        if self.is_producer or not params.get("do_remote_prefill"):
            return None
        if seq.has_per_req_cache:
            raise ValueError("PD admission supports paged KV receives only")
        cached = num_cached_blocks if self._pages_line_up(params) else 0
        sources = len(params.get("remote_block_ids") or [])
        if (
            cached < 0
            or cached >= num_blocks
            or (not params.get("chunked_transfer") and cached >= sources)
        ):
            return 0
        return cached

    def should_defer_free(self, seq) -> bool:
        return str(seq.id) in self._awaiting_send

    def send_finished(self, req_id) -> None:
        self._awaiting_send.discard(str(req_id))

    def abort_parked_send(self, req_id) -> None:
        """Mooncake's abort of a claimed request: a write request taken by now,
        as Mooncake's listener takes one on arrival, reports when it would
        have; with none, the send is done now."""
        self._take_write_requests()
        self._drop_claim(req_id, self._rt.read_clock())

    def _drop_claim(self, req_id, t: float) -> None:
        """A claimed request no write request is taken for is done at `t`, and
        a later write request for it is refused."""
        pending = (
            req_id in self._due or req_id in self._snapshot or req_id in self._extra
        )
        if str(req_id) not in self._awaiting_send or pending:
            return
        self._ready.pop(req_id, None)
        self._aborted.add(req_id)
        self._due[req_id] = t

    def source_blocks_released(self, seq) -> None:
        """No block-lifetime state remains after the send claim is retired."""

    def _write_requests(self):
        """Prefill: each write request released so far, pair by pair."""
        for other, sock in self._socks.items() if self.is_producer else ():
            while sock.poll(0):
                yield other, pickle.loads(sock.recv())

    def _take_write_requests(self) -> None:
        """Prefill: time each write request released so far on its link; a
        chunked one opens its request's admission (`admission_ready`)."""
        for other, (req_id, blocks, a, rank, *chunked) in self._write_requests():
            if blocks == "release":
                # Decode received every stage's chunked write.
                if (done := self._released.pop(req_id, None)) is not None:
                    self._due[done] = a
                continue
            if chunked and blocks is not None:
                if not self._chunked:
                    raise ValueError(
                        f"a chunked write request for {req_id!r} reached a "
                        "producer without enable_chunked_transfer"
                    )
                self._reads[req_id] = (other, blocks, a)
                continue
            if rank != self._dp_rank:
                raise ValueError(
                    f"the write request for {req_id!r} is for prefill DP rank "
                    f"{rank} and reached rank {self._dp_rank}"
                )
            if blocks is None:
                # Decode finished the request before asking for its write.
                self._drop_claim(req_id, a)
                continue
            if req_id in self._aborted:
                # Mooncake refuses it. Decode, which cannot learn that, timed
                # the write on its end of the link, and this end keeps in step.
                self._write(other, None, a, blocks, 0.0)
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
            self._write(other, req_id, a, blocks, release)
            logger.info(
                "compass: KV transfer %s, %d blocks, from %r to %r simulated s",
                req_id,
                blocks,
                a,
                self._due.get(req_id, "a shared NIC's write-done"),
            )

    def process_completions(self, output):
        """Take the write requests released so far; post the ones queued with
        no forward to follow, at width one; report what the clock had reached
        at the previous call.

        Write requests still queued here were queued by a step that
        dispatched no forward: at DP width one none follows, and the device
        has only the last forward's work to finish."""
        self._take_write_requests()
        if self._queued and not self._dummies:
            self._post(self._device_end)
        elif self._queued and self._polled_queued:
            # A second poll and still no dummy: no rank of the group had a
            # forward to run, so none follows, and the device is idle now.
            self._post(self._device_end)
        self._polled_queued = bool(self._queued)
        # The engine's aggregation keeps one poll in flight, so a poll returns
        # the snapshot the workers took at the previous poll: report that one.
        now = self._rt.read_clock()
        if self._sends or self._notices:
            self._settle_chunked(now)
        if self._done_socks and not self.is_producer:
            self._heard()
        if self._shared:
            self._settle(now)
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
        return bool(
            self._due
            or self._snapshot
            or self._queued
            or self._sends
            or self._notices
            or self._listening
            or self._released
            # A chunked request's write request is in: its admission is due.
            or bool(self._reads)
            or any(nic.busy() for nic in self._nics.values())
        )
