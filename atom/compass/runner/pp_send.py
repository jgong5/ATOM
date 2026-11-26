# SPDX-License-Identifier: MIT
"""A pipeline stage's send to the next stage, and when it completes on whose clock.

A send is several parts, each on one carrier: ``gloo`` for the size and the
pickled metadata, ``rccl`` for each device tensor, in the order the receiver
takes them. A part up to its carrier's eager threshold completes on the
sender's clock at ``t_send + o + bytes x G``; a larger one is rendezvous and
completes on both sides at ``max(t_send, r - E x G) + T``, ``r`` the time the
receiver posts its receive, ``E`` the threshold and ``T = rendezvous_fixed_s +
bytes x G``: up to a threshold's bytes leave before the receive is posted. A
part leaves when the part before it is done on the sender. The receiver
posts each part's receive when the part before it has arrived. The ``gloo``
parts leave at ``t_send``, the sender's host send; the ``rccl`` parts at
``t_ready``, when the sender's device work ends, or when the previous send's
``rccl`` parts are done, whichever is later; their receive waits for the receiver's device end.
At TP above 1 a tensor sent as its shard is a part ``(carrier, bytes, True)``:
the receiver all-gathers it over its TP group before it posts the next
part's receive.

`StageSends` is the stage loop's side. Before a downstream stage's forward it
receives that forward's input on ``pp_data``, the call and the metadata's arrival
as the batch's ``compass_pp_host`` and
the tensors' as the batch's ``compass_pp_release``, and, when the ``gloo`` parts
are rendezvous, returns their completion, and the ``rccl`` parts', on ``pp_ack`` stamped at it; at a
non-last stage's forward's launch end it waits for its previous send's ``gloo``
parts, then sends. A host that blocked on them wakes
``wake_s x (1 - exp(-b / wake_rise_s))`` after they complete, gloo's wake,
``b`` the time since the send before them completed: the pair's idle, not
the block.
The ``rccl`` parts hold the sender's stream, not its host: its
forwards after that wait start their device work no earlier. An idle
flush waits for the previous send the same way. The frames move between the
stages' TP-rank-0 workers over the PP CPU group.

Under `PP_IPC` a boundary is ATOM's HIP IPC one (`atom.distributed.pp_ipc`,
``ATOM_PP_IPC``) in place of the above. The sender's copy stream, not its
compute stream, copies the tensors into a ring of ``SLOTS`` slots on the
receiver, at the ``rccl`` carrier's ``overhead_s`` and ``per_byte_s`` a part
and its ``latency_s`` once: the first two (``hidden_states``, ``residual``)
from ``t_ready``, the previous frame's copies and the slot's release, whichever
is latest, then the rest (DSpark's aux block). The receiver's stream waits for
the first two only, posting nothing; a later stage's own send waits for its
input's rest too, as ATOM packs the aux it relays after its forward. The
metadata is a non-blocking ``gloo`` send: the receiver's host takes it as
before, and the sender waits for no ack of it. Neither stream is held by a
send. The receiver frees each slot at its forward's end, on ``pp_ack`` with
that forward's device end; a sender waits for the ack of the frame ``SLOTS``
sends back before it sends, its copy for that slot's device end.
"""

import math
import os
from dataclasses import dataclass, fields

CARRIERS = ("rccl", "gloo")
PREFIX = "interconnect.intra_node.pp"
#: Set to 1 (opt-in; unset or 0: the gloo and RCCL boundary above), every stage
#: boundary is priced as ATOM's HIP IPC one (the module text): MiniMax-M3.1's P
#: (PP4 x TP1, ATOM_PP_IPC=1) on #3720's CAL ran it, and its last stage's input
#: waited for 24 KiB a token, where RCCL's three tensors held the next stage's
#: device for 36-72 KiB a token and its sender's stream for the rendezvous.
PP_IPC = "COMPASS_PP_IPC"
#: The ring's slots, `atom.distributed.pp_ipc.SLOTS`; held here so a stage loop
#: imports no torch for it (tests/compass/test_pp_ipc.py checks it).
SLOTS = 3
#: The leading tensor parts the receiver's forward waits for (pp_ipc._KEYS).
IPC_LEAD = 2


def ipc_on() -> bool:
    """`PP_IPC` set to anything but 0."""
    return os.environ.get(PP_IPC, "0") not in ("", "0")


@dataclass(frozen=True)
class CarrierModel:
    eager_threshold_bytes: int
    latency_s: float
    overhead_s: float
    per_byte_s: float
    rendezvous_fixed_s: float

    def eager(self, nbytes: int) -> bool:
        return nbytes <= self.eager_threshold_bytes

    def rendezvous_s(self, nbytes: int) -> float:
        return self.rendezvous_fixed_s + nbytes * self.per_byte_s


def ipc_copies(model: CarrierModel, start: float, tensors) -> tuple[float, float]:
    """``(lead, end)``: the end of the copy of `tensors`' first `IPC_LEAD`
    parts and of all of them, on one copy stream from `start`, each part
    ``overhead_s + bytes x per_byte_s`` and the stream's ``latency_s`` once."""
    lead = end = start + model.latency_s
    for i, (_, n, *_) in enumerate(tensors):
        end += model.overhead_s + n * model.per_byte_s
        lead = end if i < IPC_LEAD else lead
    return lead, end


def carrier_models(spec) -> dict[str, CarrierModel]:
    """Each carrier's model from a `MachineSpec`; a missing field is refused by name."""
    return {
        c: CarrierModel(
            *(spec.value(f"{PREFIX}.{c}.{f.name}") for f in fields(CarrierModel))
        )
        for c in CARRIERS
    }


def lookaheads(models: dict[str, CarrierModel]) -> tuple[float, float]:
    """``(L_data, L_ack)``: the least carrier latency, and the least rendezvous
    fixed part, which a rendezvous outlasts after its receive is posted, less
    it. A non-positive either is refused."""
    for c, m in models.items():
        if not m.latency_s > 0:
            raise ValueError(f"the {c} carrier's latency_s is {m.latency_s!r}")
    l_data = min(m.latency_s for m in models.values())
    t_min = min(m.rendezvous_fixed_s for m in models.values())
    if not t_min > l_data:
        raise ValueError(
            f"T_min {t_min!r} is not above L_data {l_data!r}: a rendezvous would "
            "complete before its request reaches the receiver, so pp_ack has no "
            "positive lookahead"
        )
    return l_data, t_min - l_data


def gloo_wake(spec) -> tuple[float, float]:
    """``(wake_s, wake_rise_s)``, gloo's wake from a `MachineSpec`; a missing
    field is refused by name."""
    return tuple(spec.value(f"{PREFIX}.gloo.{f}") for f in ("wake_s", "wake_rise_s"))


def woken(wake, idle: float) -> float:
    """How long after its gloo parts complete a host blocked on them goes on,
    the pair having carried nothing for `idle` seconds before."""
    return wake[0] * -math.expm1(-idle / wake[1]) if idle > 0 else 0.0


def is_rendezvous(models, parts) -> bool:
    return any(not models[c].eager(n) for c, n, *_ in parts)


def recv_done(models, t_send: float, t_recv_posted: float, parts, gather=0.0):
    """``(done, end)``: the send's latest sender-side completion, and the
    receive's end, folding the parts in the receiver's order; a sharded part's
    all-gather takes `gather` seconds after it arrives."""
    r, done = t_recv_posted, -math.inf
    for c, n, *sharded in parts:
        m = models[c]
        if m.eager(n):
            sent = t_send + m.overhead_s + n * m.per_byte_s
            r = max(r, sent + m.latency_s)
            done = max(done, sent)
        else:
            pushed = m.eager_threshold_bytes * m.per_byte_s
            r = max(t_send, done, r - pushed) + m.rendezvous_s(n)
            done = max(done, r)
        if sharded:
            r += gather
    return done, r


def split(parts) -> tuple[list, list]:
    """A frame's ``gloo`` parts and its ``rccl`` parts."""
    return [p for p in parts if p[0] == "gloo"], [p for p in parts if p[0] != "gloo"]


def arrivals(models, frame, t_recv_posted: float):
    """``(sent, meta, end)`` of a ``pp_data`` frame: the ``gloo`` parts'
    sender-side completion, their arrival, and the ``rccl`` parts' after it,
    which are posted when the metadata has arrived."""
    gloo, tensors = split(frame["parts"])
    sent, meta = recv_done(models, frame["t_send"], t_recv_posted, gloo)
    return sent, meta, recv_done(models, frame["t_ready"], meta, tensors)[1]


class StageSends:
    """One stage loop's ``pp_data`` and ``pp_ack``, around its runner calls.

    `call` is the runner manager's own `call_func`; `__call__` stands in for
    it, and acts only on ``forward`` and ``flush_pp_send`` inside the run.
    """

    def __init__(self, rt, call, rank: int, pp: int, models, wake=(0.0, 1.0)) -> None:
        self.rt, self.call, self.models, self.wake = rt, call, models, wake
        # The pipeline's LPs are this one's name with its stage's index swapped.
        engine = rt.me.name.rpartition("-pp")[0]

        def ch(a, b, kind):
            return f"{engine}-pp{a}->{engine}-pp{b}:{kind}#dp0"

        self.data_in = rank > 0 and ch(rank - 1, rank, "pp_data")
        self.ack_out = rank > 0 and ch(rank, rank - 1, "pp_ack")
        self.data_out = rank < pp - 1 and ch(rank, rank + 1, "pp_data")
        self.ack_in = rank < pp - 1 and ch(rank + 1, rank, "pp_ack")
        self.received = self.sent = 0
        self.pending = None  # the last send: ("ack", seq, _) or ("eager", done, held)
        self.tensors_done = -math.inf  # the previous frame received's rccl parts
        self.held = -math.inf  # the rccl parts' end of the send last waited for
        self.acked = -math.inf  # the gloo parts' end of the send last acked
        self.ipc = ipc_on()
        self.copy_free = -math.inf  # under PP_IPC: the copy stream's last copy's end
        self.slot_free = [-math.inf] * SLOTS  # each slot's release, device side
        self.acks = 0  # the acks read
        self.aux_in = -math.inf  # the rest of the input last received

    def __call__(self, name, *args, wait_out=False):
        if not self.rt.in_run or name not in ("forward", "flush_pp_send"):
            return self.call(name, *args, wait_out=wait_out)
        if name == "flush_pp_send":
            self.settle()
            return self.call(name, *args, wait_out=wait_out)
        if self.ipc:
            return self._ipc_forward(name, *args, wait_out=wait_out)
        if self.held is None:  # an eager send's rendezvous rccl part
            raise RuntimeError(
                f"{self.data_out}: no ack says when the last send's rccl parts end"
            )
        release = self.held
        if self.data_in:  # the tensors' arrival, no earlier than `held`
            args[0].compass_pp_host, release = self.receive(args[0])
        if release > -math.inf:
            args[0].compass_pp_release = release
        t0 = self.rt.now
        reply = self.call(name, *args, wait_out=wait_out)
        if self.data_out:
            ready_s, post_s, parts = reply.pp_send
            self.send(t0 + ready_s, parts)
            if post_s > 0:
                self.rt.advance_to(self.rt.now + post_s)
        return reply

    def _frame(self, kind: str, ch: str, seq: int) -> dict:
        """Receive (`ch`, `seq`) inline, then read its frame off the worker."""
        self.rt.receive(ch, seq)
        frame = self.call("recv_pp_frame", kind, wait_out=True)
        if frame["seq"] != seq:
            raise RuntimeError(f"{ch}: read frame {frame['seq']}, released {seq}")
        return frame

    def receive(self, batch) -> tuple:
        """``(called, meta)`` and the tensors' arrival, posted at `batch`'s prepare end.
        The tensors wait for the stage's device end, and leave once the previous
        frame's are done: one stream sends both. Each sharded tensor's
        all-gather is priced by the worker's cost backend."""
        from atom.compass.runner.overrides import forward_shape

        seq, called = self.received, self.rt.now
        self.received += 1
        frame = self._frame("pp_data", self.data_in, seq)
        shape = forward_shape(batch)
        posted, device = self.call("pp_recv_posted", called, *shape, wait_out=True)
        gloo, tensors = split(frame["parts"])
        sent, meta = recv_done(self.models, frame["t_send"], posted, gloo)
        ready = max(frame["t_ready"], self.tensors_done)
        posted = max(meta, device, self.held)
        gather = 0.0
        if any(len(p) > 2 for p in tensors):
            rows = batch.total_tokens_num
            gather = self.call("pp_all_gather_s", rows, wait_out=True)
        self.tensors_done, end = recv_done(self.models, ready, posted, tensors, gather)
        if is_rendezvous(self.models, gloo):
            ack = {"seq": seq, "tensors_done": self.tensors_done, "gloo_done": sent}
            self.call("send_pp_frame", "pp_ack", ack, wait_out=True)
            # The ack says when the parts completed (`gloo_done`), and reaches the
            # sender no earlier than this stage's clock and the channel's lookahead
            # allow: a receive this stage posted late (the 1P2D repro at C240, a
            # threshold of bytes left before it) completes before its own clock,
            # and the sender, granted up to that promise, may already be past it.
            lookahead = self.rt.table.lookahead(self.ack_out)
            self.rt.stamp_send(self.ack_out, max(sent, self.rt.now + lookahead))
        return (called, meta), end

    def send(self, ready: float, parts) -> None:
        """Send once the previous send's ``gloo`` parts are done: the metadata
        now, the tensors once the forward's device work, at `ready`, is done."""
        self.settle()
        seq, t_send = self.sent, self.rt.now
        self.sent += 1
        frame = {"seq": seq, "t_send": t_send, "t_ready": ready, "parts": parts}
        self.call("send_pp_frame", "pp_data", frame, wait_out=True)
        self.rt.stamp_send(
            self.data_out, t_send + self.rt.table.lookahead(self.data_out)
        )
        gloo, tensors = split(parts)
        if is_rendezvous(self.models, gloo):
            self.pending = ("ack", seq, None)
        else:
            done = None  # a rendezvous rccl part ends at the receiver's post
            if self.held is not None and not is_rendezvous(self.models, tensors):
                done = recv_done(self.models, max(ready, self.held), ready, tensors)[0]
            self.pending = ("eager", arrivals(self.models, frame, t_send)[0], done)

    def settle(self) -> None:
        """Wait for the previous send's ``gloo`` parts: their ack if
        rendezvous, else their local end, and take its rccl parts' end as `held`."""
        pending, self.pending = self.pending, None
        if pending is None:
            return
        kind, at, self.held = pending
        if kind == "ack":
            entry, last = self.rt.now, self.acked
            ack = self._frame("pp_ack", self.ack_in, at)
            self.held, self.acked = ack["tensors_done"], ack["gloo_done"]
            if self.rt.now > entry and (late := woken(self.wake, self.acked - last)):
                self.rt.advance_to(self.rt.now + late)
        elif at > self.rt.now:
            self.rt.advance_to(at)

    # ---- under PP_IPC ----

    def _ipc_forward(self, name, *args, wait_out=False):
        """`__call__`'s forward under `PP_IPC`: the input's metadata holds the
        host and its first `IPC_LEAD` tensors' copy the device; the slot is
        freed at the forward's end, then the output is sent, its copy waiting
        for the input's rest as well as the device work."""
        if self.data_in:
            args[0].compass_pp_host, release = self._ipc_receive(args[0])
            args[0].compass_pp_release = release
        t0 = self.rt.now
        reply = self.call(name, *args, wait_out=wait_out)
        if self.data_in:
            self._ipc_free(reply)
        if self.data_out:
            ready_s, post_s, parts = reply.pp_send
            self._ipc_send(max(t0 + ready_s, self.aux_in), parts)
            if post_s > 0:
                self.rt.advance_to(self.rt.now + post_s)
        return reply

    def _ipc_receive(self, batch) -> tuple:
        """``(called, meta)``, the metadata taken at `batch`'s prepare end as
        `receive` takes it, and the end of the copy of the input's first
        `IPC_LEAD` tensors into this stage's ring, which posts nothing."""
        from atom.compass.runner.overrides import forward_shape

        seq, called = self.received, self.rt.now
        self.received += 1
        frame = self._frame("pp_data", self.data_in, seq)
        shape = forward_shape(batch)
        posted, _ = self.call("pp_recv_posted", called, *shape, wait_out=True)
        gloo, tensors = split(frame["parts"])
        if any(len(p) > 2 for p in tensors):
            raise RuntimeError(
                f"{self.data_in}: a sharded tensor crosses an IPC boundary, which "
                "ATOM runs at TP 1 only"
            )
        meta = recv_done(self.models, frame["t_send"], posted, gloo)[1]
        lead, self.aux_in = frame["ipc"]
        return (called, meta), lead

    def _ipc_free(self, reply) -> None:
        """Free the slot the forward just run read, at its device end, on ``pp_ack``."""
        end = getattr(reply, "device_end", None)
        ack = {"seq": self.received - 1, "freed": self.rt.now if end is None else end}
        self.call("send_pp_frame", "pp_ack", ack, wait_out=True)
        self.rt.stamp_send(self.ack_out)

    def _ipc_send(self, ready: float, parts) -> None:
        """Send the metadata now, after the ack freeing this frame's slot, and
        the tensors' copies, from `ready`, the previous copies' end and the slot's
        release, whichever is latest, as the frame's ``ipc``: ``(lead, end)``."""
        seq = self.sent
        self.sent += 1
        self._read_acks(seq - SLOTS)
        tensors = split(parts)[1]
        start = max(ready, self.copy_free, self.slot_free[seq % SLOTS])
        lead, self.copy_free = ipc_copies(self.models["rccl"], start, tensors)
        t_send = self.rt.now
        frame = {
            "seq": seq,
            "t_send": t_send,
            "t_ready": ready,
            "parts": parts,
            "ipc": (lead, self.copy_free),
        }
        self.call("send_pp_frame", "pp_data", frame, wait_out=True)
        self.rt.stamp_send(
            self.data_out, t_send + self.rt.table.lookahead(self.data_out)
        )

    def _read_acks(self, upto: int) -> None:
        """Read every ack released so far and each one through `upto`, waiting
        for those; an ack's slot is free from its ``freed``."""
        while True:
            seq = self.acks
            if seq > upto:
                with self.rt.lock:
                    if not self.rt.is_released(self.ack_in, seq):
                        return
            ack = self._frame("pp_ack", self.ack_in, seq)
            self.acks += 1
            self.slot_free[seq % SLOTS] = ack["freed"]


def install(engine, spec) -> None:
    """Put `StageSends` in front of a pipeline stage's runner calls, and make
    the stage's bounded polls essential while it holds a send or KV work."""
    from atom.utils import clock

    rt = clock.installed()
    mgr = engine.runner_mgr
    sends = StageSends(
        rt,
        mgr.call_func,
        engine.pp_rank,
        engine.pp_size,
        carrier_models(spec),
        gloo_wake(spec),
    )
    mgr.call_func = sends
    rt.holds_work = lambda: sends.pending is not None or engine.has_pending_kv_work()
    transport = getattr(engine, "pp_transport", None)
    if engine.pp_rank > 0 and transport is not None:
        transport.recv_metadata = _looks_again(rt, transport.recv_metadata)


def _looks_again(rt, recv_metadata):
    """`recv_metadata` such that a batch it returns counts as the loop pass's
    work. `clock.idle` waits for the next event only after a pass that made no
    clock call; a later stage's pass that takes an empty, connector-only batch
    makes none, and its idle point then waits with the head's later batches
    already released and unread, while the head waits on this stage's ack: the
    1P2D C240 sim's client run ended at ~367 s, every LP waiting."""

    def taken(*args, **kwargs):
        batch = recv_metadata(*args, **kwargs)
        if batch is not None:
            rt.calls += 1
        return batch

    return taken
