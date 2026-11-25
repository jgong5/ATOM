# SPDX-License-Identifier: MIT
"""What a KV transfer costs, and when it is therefore released.

One transfer is priced as `latency + bytes / bandwidth`, and the three terms
come from three different places on purpose.

**The bytes are exact.** A transfer moves a whole number of paged blocks, and
what one block costs is the entry of the paged class the engine's runner
planned its pool with, not a measurement -- so the byte count carries no error
at all and nothing here re-derives a KV layout. A model whose transfer also
carries the request's state slot moves one planned slot beside its blocks, and
one that stages its compressor state for prefill-decode moves that staging row
in a second write, which pays the latency again.

**The bandwidth is a spec peak, and it is spent at the derate the spec
declares.** A datasheet number spent as an achievable one would make every
transfer faster than any wire has ever been. The schema obliges a derate
wherever a peak appears and refuses a document that omits one, so this module
reads both and multiplies; it re-checks nothing.

**The latency and the bandwidth are named per side of the node boundary**, and
the caller says which side it is on. There is no default and no fall back to
the other side: a transfer between two nodes priced on the intra-node link is
wrong by roughly an order of magnitude and would look entirely plausible.

The numbers are as good as the document they came from. A spec's provenance
says per value whether it came from a datasheet, was assumed or was probed. A
probed RDMA link is
`atom/compass/networks/crs005-013-ionic.json`, measured on the rails between
two MI355X nodes; its provenance names the bench and the run.

It charges the same duration to both ends of the same wire. A push and a pull
of the same blocks cost the same here, which is a claim about the fabric being
symmetric that this model cannot check.

Writes on one `Link` move one at a time, in arrival order. A link's first
write that moves bytes also pays the link's one-off setup, when the spec
declares one; a write that moves no block and no slot makes no transfer call
and pays none. Writes on different links do not contend, even where real ones
share a rail or a NIC, so writes that fan in on one are each priced at a lone
writer's rate -- unless the link states a NIC's bandwidth
(`nic_bandwidth_bytes_per_s`, opt-in): then the writes into one NIC share it
(`SharedNic`). On a recorded 1P1D run at c=256 (#3720), a write alone on its
D rank's NIC moved 7.9 GB/s and two to five at once 5.7 to 3.1 GB/s each,
whatever the prefill rank's other writes or forwards did: one write is held
to the link's rate, several share a NIC's ~12 GB/s.

Nothing in this module imports the engine or a tensor library. Reaching it
through the package around it is not the same import: `atom.compass.kv` loads
the connector beside it, and that brings the engine's connector interface, its
factory, and whatever those pull in.
None of it is a tensor library and none of it is a device runtime, so the
arithmetic still runs where no driver does -- which is the property worth
having, and it is not a bare import of this file.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from dataclasses import dataclass


class Scope(enum.Enum):
    """Which side of the node boundary a transfer crosses.

    The spec carries a latency and a bandwidth for each, and they differ by
    enough that picking the wrong one is not a small error.
    """

    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True, slots=True)
class TransferModel:
    """The price of one transfer: a fixed cost, a rate, and a block size.

    `bandwidth_bytes_per_s` is what a transfer actually reaches, not what the
    link is rated at -- `from_spec` applies the derate before it gets here, so
    a model built by hand in a test states an achievable number too.
    `slot_bytes` is the request's state slot, which every transfer moves beside
    its blocks, or after them when `slot_after_blocks`; 0 for a model whose
    transfer carries none. `staging_bytes` is the staging row a second write
    moves once the first completes; 0 for a model that stages nothing.
    """

    latency_s: float
    bandwidth_bytes_per_s: float
    bytes_per_block: int
    setup_s: float = 0.0
    slot_bytes: int = 0
    staging_bytes: int = 0
    slot_after_blocks: bool = False
    #: What the writes into one NIC reach together, 0 when the link states no
    #: NIC: each write is then priced alone on its rank pair's `Link`.
    nic_bandwidth_bytes_per_s: float = 0.0

    def __post_init__(self) -> None:
        if self.latency_s < 0:
            raise ValueError(f"latency_s must not be negative, got {self.latency_s}")
        if self.setup_s < 0:
            raise ValueError(f"setup_s must not be negative, got {self.setup_s}")
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError(
                "bandwidth_bytes_per_s must be positive, got "
                f"{self.bandwidth_bytes_per_s}"
            )
        if self.bytes_per_block < 1:
            raise ValueError(
                f"bytes_per_block must be at least 1, got {self.bytes_per_block}"
            )
        if self.slot_bytes < 0:
            raise ValueError(f"slot_bytes must not be negative, got {self.slot_bytes}")
        if self.staging_bytes < 0:
            raise ValueError(
                f"staging_bytes must not be negative, got {self.staging_bytes}"
            )
        if self.nic_bandwidth_bytes_per_s < 0:
            raise ValueError(
                "nic_bandwidth_bytes_per_s must not be negative, got "
                f"{self.nic_bandwidth_bytes_per_s}"
            )

    @classmethod
    def from_spec(
        cls,
        spec,
        bytes_per_block: int,
        scope: Scope,
        slot_bytes: int = 0,
        staging_bytes: int = 0,
        slot_after_blocks: bool = False,
    ) -> TransferModel:
        """Price a block of *bytes_per_block*, and a state slot of
        *slot_bytes* and a staging row of *staging_bytes* per transfer, on the
        *scope* link of *spec*.

        The spec refuses a link it does not carry, and refuses a peak
        bandwidth whose derate is missing; both refusals travel out of here
        unchanged, naming the field and the rule that declined it. A link
        that states `nic_bandwidth_bytes_per_s` (an attained rate, not a peak)
        has its writes into one NIC share it; one that does not prices each
        write alone on its pair's link.
        """
        link = f"interconnect.{scope.value}"
        peak = spec.value(f"{link}.link_bandwidth_bytes_per_s")
        derate = spec.value(f"{link}.derate")
        return cls(
            latency_s=spec.value(f"{link}.link_latency_s"),
            bandwidth_bytes_per_s=peak * derate,
            bytes_per_block=bytes_per_block,
            setup_s=spec.values.get(f"{link}.link_setup_s", 0.0),
            slot_bytes=slot_bytes,
            staging_bytes=staging_bytes,
            slot_after_blocks=slot_after_blocks,
            nic_bandwidth_bytes_per_s=spec.values.get(
                f"{link}.nic_bandwidth_bytes_per_s", 0.0
            ),
        )

    def bytes_moved(self, blocks: int) -> int:
        """The bytes a transfer of *blocks* moves: its blocks, the state slot
        and the staging row."""
        return blocks * self.bytes_per_block + self.slot_bytes + self.staging_bytes

    def writes(self, blocks: int) -> int:
        """The writes a transfer of *blocks* takes, each paying the latency:
        the blocks, with the slot unless it follows them, then the staging row
        and a slot that follows the blocks. A write with no bytes is not
        posted, and a transfer that moves no byte still takes one."""
        after = self.staging_bytes + (self.slot_bytes if self.slot_after_blocks else 0)
        return max(1, (self.bytes_moved(blocks) > after) + (after > 0))

    def duration_s(self, blocks: int) -> float:
        """How long *blocks* take on this link.

        Zero blocks is a real case -- a consumer whose prefix cache already
        holds the whole prompt still has a transfer to complete -- and it
        costs the latency, because the two ends still have to say so, and
        the state slot, which no prefix cache holds.
        """
        if blocks < 0:
            raise ValueError(
                f"a transfer of {blocks} blocks is not a transfer; a block "
                "count below zero means the computed-block count exceeded the "
                "allocation, which would price a transfer that finishes "
                "before it starts"
            )
        return (
            self.writes(blocks) * self.latency_s
            + self.bytes_moved(blocks) / self.bandwidth_bytes_per_s
        )

    def release_at(self, issue_time_s: float, blocks: int) -> float:
        """The time this transfer may be reported finished, and not before."""
        return issue_time_s + self.duration_s(blocks)


class Link:
    """The writes on one link, moved one at a time in arrival order.

    A write arriving at `a` starts once the writes before it have moved their
    bytes, holds the link for `bytes / bandwidth`, and completes the latency
    after its last byte. Its completion is fixed when it arrives, so two
    holders fed the same arrivals in the same order compute the same times. A
    write alone on the link completes at `a + duration_s(blocks)`, and the
    link's first write that moves bytes starts `setup_s` after it would
    otherwise. A transfer whose staging row or slot follows in a second write
    holds the link until it has moved, the first write's latency included.
    """

    def __init__(self, model: TransferModel) -> None:
        self._model = model
        # A chunked prefill's middle reads move its blocks alone: the slot and
        # the staging row go with its last (`ChunkedSend`).
        self._bare = dataclasses.replace(model, slot_bytes=0, staging_bytes=0)
        self._last = -math.inf  # the latest arrival
        self._free = -math.inf  # when the link has moved every byte so far
        self._set_up = False  # whether a write that moves bytes has paid setup_s

    @property
    def last(self) -> float:
        """The latest arrival so far: a write may not arrive before it."""
        return self._last

    def start(self, a: float, blocks: int, *, bare: bool = False) -> float:
        """A write of `blocks` arrives at `a`; when it completes. With `bare`,
        it moves the blocks alone, no slot or staging row."""
        if a < self._last:
            raise ValueError(
                f"a write arrives at {a}, before one that arrived at "
                f"{self._last}; a link takes its writes in time order"
            )
        self._last = a
        model = self._bare if bare else self._model
        start = max(a, self._free)
        moved = model.bytes_moved(blocks)
        if moved > 0 and not self._set_up:
            self._set_up = True
            start += model.setup_s
        done = start + model.duration_s(blocks)
        # A second write leaves when the first write completes.
        gap = (model.writes(blocks) - 1) * model.latency_s
        self._free = start + gap + moved / model.bandwidth_bytes_per_s
        return done


class SharedNic:
    """The writes into one NIC, which share it (`nic_bandwidth_bytes_per_s`).

    A write arriving at `a` moves its bytes from `a` on, each write moving
    bytes at `min(bandwidth, nic / n)` while `n` are: alone it reaches the
    link's rate, beside others they split the NIC's. It completes the latency
    after its last byte, once per write it takes (`TransferModel.writes`), as
    on a `Link`, and the NIC's first write that moves bytes pays `setup_s` on
    top. A write with no bytes completes at its arrival plus the latency.

    Unlike a `Link`'s, a write's completion depends on the writes that arrive
    after it, so it is known once the NIC has been advanced past it:
    `advance(t)` returns each write whose last byte moved by `t`, once, with
    its completion, and is exact so long as every write arriving before `t`
    was added first. Writes are added in arrival order.
    """

    def __init__(self, model: TransferModel) -> None:
        if model.nic_bandwidth_bytes_per_s <= 0:
            raise ValueError("a shared NIC needs nic_bandwidth_bytes_per_s above 0")
        self._model = model
        self._t = -math.inf  # how far the bytes have been moved
        self._last = -math.inf  # the latest arrival
        self._left: dict = {}  # key -> bytes still to move, of the writes moving
        self._tail: dict = {}  # key -> what it pays after its last byte
        self._found: list = []  # (key, completion) not yet taken
        self._set_up = False

    def add(self, key, a: float, blocks: int) -> None:
        """A write of `blocks`, named `key`, arrives at `a`."""
        if a < self._last:
            raise ValueError(
                f"a write arrives at {a}, before one that arrived at "
                f"{self._last}; a NIC takes its writes in time order"
            )
        self._last = a
        self._move(a)
        m = self._model
        moved = m.bytes_moved(blocks)
        tail = m.writes(blocks) * m.latency_s
        if moved <= 0:
            self._found.append((key, a + tail))
            return
        if not self._set_up:
            self._set_up = True
            tail += m.setup_s
        self._left[key] = float(moved)
        self._tail[key] = tail

    def _move(self, t: float) -> None:
        """Move every write's bytes up to `t`, finding each that completes."""
        m = self._model
        while self._left:
            rate = min(
                m.bandwidth_bytes_per_s, m.nic_bandwidth_bytes_per_s / len(self._left)
            )
            key = min(self._left, key=self._left.get)
            finish = self._t + max(self._left[key], 0.0) / rate
            if finish > t:
                if t > self._t:
                    for k in self._left:
                        self._left[k] -= rate * (t - self._t)
                break
            done = self._left.pop(key)
            for k in self._left:
                self._left[k] -= done
            self._t = finish
            self._found.append((key, finish + self._tail.pop(key)))
        self._t = max(self._t, t)

    def advance(self, t: float) -> list:
        """`(key, completion)` of each write whose last byte moved by `t`,
        each once."""
        self._move(t)
        found, self._found = self._found, []
        return found

    def busy(self) -> bool:
        """Whether a write added is still to be returned by `advance`."""
        return bool(self._left or self._found)


def shared_completions(model: TransferModel, writes) -> list:
    """The completion of each `(arrival, blocks)` of `writes`, in arrival
    order, all into one `SharedNic` of `model`: what a record's writes on one
    NIC replay to, and what the run prices them at."""
    nic = SharedNic(model)
    for i, (a, blocks) in enumerate(writes):
        nic.add(i, a, blocks)
    done = dict(nic.advance(math.inf))
    return [done[i] for i in range(len(writes))]


class ChunkedSend:
    """One consumer's reads of one chunked prefill on one prefill rank:
    Mooncake's send task under ``enable_chunked_transfer``
    (`ChunkedPrefill.wait_chunk`'s loop, `_dispatch_ready_chunked_requests`).

    The write request arrives at `start`, before the prefill has run: the task
    waits from then. At each decision it takes every block the prefill has
    published by then past its cursor and writes them, one write on its
    `Link`, and decides again when that write completes; with none new, it
    waits for the next publication. A publication is a forward's end on the
    device: the prompt's whole blocks computed by then, every block once the
    prompt is (`publish`, `ChunkedPrefill.publish`'s rule). The last write
    carries the transfer's slot and staging row; the middle ones move blocks
    alone. Once the prompt's last block is written and the prefill has handed
    off (`finish`, its sampled token), the send is `done`, the later of the
    two: what the producer's write-done notice says.

    `cursor` is the first block it writes, past the consumer's cached prefix,
    and `total` the prompt's blocks. Decisions need every publication up to
    their time, so the caller publishes forwards as they end and asks
    `chunk_writes` to decide only up to a time it has seen.
    """

    def __init__(self, link: Link, start: float, cursor: int, total: int) -> None:
        if not 0 <= cursor <= total:
            raise ValueError(f"cursor {cursor} outside a prompt of {total} blocks")
        self.link, self.at, self.cursor, self.total = link, start, cursor, total
        self.pubs: list = []  # (time, blocks ready), both rising
        self.handoff: float | None = None
        self.writes: list = []  # (start, end, blocks)
        self.done: float | None = None
        self.clamped = 0.0  # seconds its writes queued behind ones decided earlier

    def publish(self, t: float, tokens: int, prompt_tokens: int, block_size: int):
        """A forward ending at `t` computed the prompt through `tokens`: its
        whole blocks, or every block once the prompt is done."""
        tokens = min(tokens, prompt_tokens)
        blocks = (
            -(-tokens // block_size)
            if tokens == prompt_tokens
            else tokens // block_size
        )
        blocks = min(blocks, self.total)
        if self.pubs and (t < self.pubs[-1][0] or blocks < self.pubs[-1][1]):
            raise ValueError(
                f"a publication of {blocks} blocks at {t} after one of "
                f"{self.pubs[-1][1]} at {self.pubs[-1][0]}: a prefill publishes "
                "forward in time and in blocks"
            )
        self.pubs.append((t, blocks))

    def finish(self, t: float) -> None:
        """The prefill handed off at `t`: it sampled the first token."""
        self.handoff = t

    def _ready(self, t: float) -> int:
        """The blocks published by `t`."""
        ready = 0
        for when, blocks in self.pubs:
            if when > t:
                break
            ready = blocks
        return ready

    def next_at(self) -> float | None:
        """When its next decision is, or None while it waits for what it has
        not been told: a later publication, or the handoff."""
        if self.done is not None:
            return None
        if self._ready(self.at) > self.cursor:
            return self.at
        if self.cursor >= self.total:
            return None if self.handoff is None else self.at
        later = [t for t, blocks in self.pubs if t > self.at and blocks > self.cursor]
        return later[0] if later else None

    def step(self) -> None:
        """Make its next decision: write what is ready, or finish."""
        t = self.next_at()
        if t is None:
            raise ValueError("a chunked send stepped with nothing to decide")
        self.at = t
        ready = self._ready(t)
        if ready > self.cursor:
            # A later PP stage decides a send that handed off to its end at
            # once (it takes no poll, `CompassKVConnector.prefill_done`): a
            # write it decides afterwards from an earlier publication queues
            # behind those, arriving with the link's latest. Where every
            # publication is known first -- the head stage, a replay -- the
            # sends are decided in time order and this changes nothing.
            if t < self.link.last:
                self.clamped += self.link.last - t
                t = self.at = self.link.last
                ready = self._ready(t)
            n, last = ready - self.cursor, ready >= self.total
            end = self.link.start(t, n, bare=not last)
            self.writes.append((t, end, n))
            self.cursor, self.at = ready, end
            return
        self.done = max(t, self.handoff)


def chunk_writes(sends, until: float) -> list:
    """Decide every chunked send of `sends` up to `until`, in time order
    across them, as their links take writes in arrival order: each that is
    done by then, once, in the order it finished. One function for the
    simulated run, which calls it at each poll with the time it has seen, and
    for a record's replay, which calls it once with ``inf``."""
    done = []
    while True:
        due = [(t, i) for i, s in enumerate(sends) if (t := s.next_at()) is not None]
        due = [d for d in due if d[0] <= until]
        if not due:
            return done
        _, i = min(due)
        sends[i].step()
        if sends[i].done is not None:
            done.append(sends[i])
