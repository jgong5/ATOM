# SPDX-License-Identifier: MIT
"""The logical-process side of the simulated clock: time requests, sends and release.

One `LPRuntime` per logical process (LP), in the process of its clock owner: the
engine step loop or the frontend event loop. The owner is the only thread that
moves the LP clock and the only one that produces a cross-LP message. It asks the
clock authority for time over an injected connection, ``conn.send((kind, t,
log, t_daemon))`` then ``conn.recv() -> (G, released)``, and every request
carries the sends registered since the previous one, so the authority knows each
message an LP produced before it moves that LP's clock.

A grant to ``+inf`` closes the simulation window (`end_run`) before the process
begins to shut down: the run is finished, and the owner's loop must exit rather
than run its timers at ``+inf``. A clock call after it raises, `stamp_send` keeps
returning ``+inf`` arrivals for shutdown sends, and `close` raises if the owner
leaves before it.

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

#: The runtime of the LP this process belongs to, or None on a real run.
_installed: "LPRuntime | None" = None


def install(runtime: "LPRuntime | None") -> None:
    """Make every `now` in this process read `runtime`; None restores the real clock."""
    global _installed
    _installed = runtime


def now(real) -> float:
    """The LP clock while a runtime is installed; otherwise `real()`.

    A serving-path read passes the machine clock it read before, so a real run
    reads exactly what it did, and a simulated one never calls it.
    """
    runtime = _installed
    return real() if runtime is None else runtime.read_clock()


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
        self._require_open("advance_to")
        if T < self.now:
            raise ValueError(f"{self.me} cannot advance to {T}, it is at {self.now}")
        self._step_through(*self._ca_call("TAR", T))

    def next_event(self, t: float, t_daemon: float = float("inf")) -> float:
        """Idle until `t`, the daemon deadline `t_daemon` or the earliest arrival,
        whichever the grant is; returns it.

        A daemon deadline is a housekeeping timer that does not keep the run
        alive: it fires only once essential work reaches it.
        """
        self._require_open("next_event")
        G, released = self._ca_call("NER", t, t_daemon)
        self._step_through(G, released)
        if G == float("inf"):
            self.end_run()
        return G

    def close(self) -> None:
        """The owner leaves its loop; refused before the ``+inf`` grant."""
        if self.now != float("inf"):
            raise RuntimeError(
                f"{self.me} left its loop at {self.now}, before the +inf grant "
                "finished the run"
            )

    def _require_owner(self, what: str) -> None:
        if threading.current_thread() is not self.owner:
            raise RuntimeError(
                f"{what} from thread {threading.current_thread().name!r}: only the clock "
                f"owner {self.owner.name!r} of {self.me} moves its clock or sends"
            )

    def _require_open(self, what: str) -> None:
        self._require_owner(what)
        if self.now == float("inf"):
            raise RuntimeError(f"{what} from {self.me} after the +inf grant")

    def _ca_call(self, kind: str, t: float, t_daemon: float = float("inf")):
        with self.lock:
            log, self.send_log = self.send_log, []
        self.conn.send((kind, t, log, t_daemon))
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
