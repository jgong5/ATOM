# SPDX-License-Identifier: MIT
"""The logical-process side of the simulated clock: time requests, sends and release.

One `LPRuntime` per logical process (LP), in the process of its clock owner: the
engine step loop or the frontend event loop. The owner is the only thread that
moves the LP clock and the only one that produces a cross-LP message. It asks the
clock authority for time over an injected connection, ``conn.send((kind, t,
log))`` then ``conn.recv() -> (G, released)``, and every request carries the
sends registered since the previous one, so the authority knows each message an
LP produced before it moves that LP's clock.

A grant names the messages it releases as ``{channel: [(seq, arrival)]}``.
`_step_through` releases them one at a time in ``(arrival, channel, seq)``
order, setting the LP clock to each arrival. On a ``thread`` channel it then
waits until the receiving thread is back at its wait point, so a handler runs
with the clock at its message's arrival and nothing else in the LP moving. An
``inline`` channel is read by the owner itself at its own receive point, so
there is nothing to wait for. Messages are counted by ``(channel, seq)``, so a
channel need not be FIFO. The ``(channel, seq)`` sets here are only ever tested
for membership, never iterated.

The receive side (socket and poller wrappers) calls `check_arrival` for every
stamped frame it reads, holds a frame back until `is_released`, and brackets
handing one to ATOM with `handed_over` and `back_at_wait_point`.
"""

import logging
import threading

from atom.compass.clock import ChannelTable, LpId
from atom.compass.clock.channels import ReceiveMode

logger = logging.getLogger("atom")

#: Wall seconds a released message may stay unhandled before one diagnostic is
#: printed. The wait itself goes on: it costs no simulated time.
DIAG_S = 30.0


class Straggler(Exception):
    """An unreleased message whose arrival this LP has already released past."""


class LPRuntime:
    def __init__(
        self,
        me: LpId,
        table: ChannelTable,
        conn,
        *,
        owner: threading.Thread | None = None,
        diag_s: float = DIAG_S,
    ) -> None:
        self.me, self.table, self.conn, self.diag_s = me, table, conn, diag_s
        self.owner = owner or threading.current_thread()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.in_run = False
        self.now = 0.0
        self.send_log: list[tuple[str, int, float]] = []
        self.next_seq = {c.name: 0 for c in table.channels_from(me)}
        into = [c.name for c in table.channels_into(me)]
        self.arrived: dict[str, set[int]] = {ch: set() for ch in into}
        self.released: dict[str, set[int]] = {ch: set() for ch in into}
        self.handled: dict[str, set[int]] = {ch: set() for ch in into}
        # Frames read but not released, (channel, seq) -> arrival: checked at each drain.
        self.unreleased: dict[tuple[str, int], float] = {}
        self.taken_by: dict[threading.Thread, tuple[str, int]] = {}
        self.wakes: dict[threading.Thread, threading.Event] = {}

    def start_run(self) -> None:
        self.in_run = True

    def end_run(self) -> None:
        self.in_run = False

    # ---- clock owner ----

    def read_clock(self) -> float:
        return self.now

    def stamp_send(self, ch: str) -> tuple[float, int]:
        """Register one message on `ch` as produced now; returns its ``(arrival, seq)``."""
        self._require_owner("stamp_send")
        with self.lock:
            seq = self.next_seq[ch]
            self.next_seq[ch] = seq + 1
            arrival = self.now + self.table.lookahead(ch)
            self.send_log.append((ch, seq, arrival))
        return arrival, seq

    def advance_to(self, T: float) -> None:
        """Price an event: block until granted `T`, releasing what arrives up to it."""
        self._require_owner("advance_to")
        if T < self.now:
            raise ValueError(f"{self.me} cannot advance to {T}, it is at {self.now}")
        self._step_through(*self._ca_call("TAR", T))

    def next_event(self, t: float) -> float:
        """Idle until `t` or the earliest arrival, whichever the grant is; returns it."""
        self._require_owner("next_event")
        G, released = self._ca_call("NER", t)
        self._step_through(G, released)
        return G

    def end_workload(self) -> None:
        self._require_owner("end_workload")
        self._ca_call("END", float("inf"))

    def _require_owner(self, what: str) -> None:
        if threading.current_thread() is not self.owner:
            raise RuntimeError(
                f"{what} from thread {threading.current_thread().name!r}: only the clock "
                f"owner {self.owner.name!r} of {self.me} moves its clock or sends"
            )

    def _ca_call(self, kind: str, t: float):
        with self.lock:
            log, self.send_log = self.send_log, []
        self.conn.send((kind, t, log))
        return self.conn.recv()

    def _step_through(self, G: float, released: dict) -> None:
        order = sorted((a, ch, seq) for ch, msgs in released.items() for seq, a in msgs)
        with self.lock:
            for a, ch, seq in order:
                self.now = max(self.now, a)
                self.released[ch].add(seq)
                self.unreleased.pop((ch, seq), None)
                for w in self.wakes.values():
                    w.set()
                if self.table.recv_mode(ch) is ReceiveMode.INLINE:
                    continue
                reported = False
                while seq not in self.handled[ch]:
                    if not self.cv.wait(self.diag_s) and not reported:
                        logger.warning(
                            "%s: released %s seq %d (arrival %s) is not handled after "
                            "%s wall seconds; still waiting",
                            self.me,
                            ch,
                            seq,
                            a,
                            self.diag_s,
                        )
                        reported = True
            self.now = G
            late = [(a, ch, s) for (ch, s), a in self.unreleased.items() if a < G]
            if late:
                self._straggler(*min(late))

    def _straggler(self, arrival: float, ch: str, seq: int) -> None:
        raise Straggler(
            f"{ch} seq {seq} arrives at {arrival} but {self.me} has released up to "
            f"{self.now} without it (lookahead {self.table.lookahead(ch)})"
        )

    def inline_pending(self) -> bool:
        """Is there a released inline message not yet taken?"""
        with self.lock:
            return any(
                self.released[ch] - self.handled[ch]
                for ch in self.released
                if self.table.recv_mode(ch) is ReceiveMode.INLINE
            )

    # ---- receive side ----

    def check_arrival(self, ch: str, arrival: float, seq: int) -> None:
        """Called once per stamped frame read; frames may arrive in any order."""
        with self.lock:
            if seq in self.arrived[ch]:
                raise RuntimeError(f"{ch} seq {seq} arrived twice")
            self.arrived[ch].add(seq)
            if seq not in self.released[ch]:
                self.unreleased[ch, seq] = arrival
                if arrival < self.now:
                    self._straggler(arrival, ch, seq)

    def my_wakeup(self) -> threading.Event:
        """This thread's event, set whenever a message is released."""
        with self.lock:
            return self.wakes.setdefault(threading.current_thread(), threading.Event())

    def is_released(self, ch: str, seq: int) -> bool:
        """Caller holds `lock`."""
        return seq in self.released[ch]

    def handed_over(self, ch: str, seq: int) -> None:
        with self.lock:
            self.taken_by[threading.current_thread()] = (ch, seq)

    def back_at_wait_point(self) -> None:
        with self.lock:
            taken = self.taken_by.pop(threading.current_thread(), None)
            if taken is not None:
                self.count_done_locked(*taken)

    def count_done_locked(self, ch: str, seq: int) -> None:
        """Caller holds `lock`."""
        self.handled[ch].add(seq)
        self.cv.notify_all()
