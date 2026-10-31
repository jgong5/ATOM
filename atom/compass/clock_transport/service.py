# SPDX-License-Identifier: MIT
"""The Clock Authority's serve side, and the in-process carrier to it.

`serve` makes one serve side per endpoint. Each request is served on the
thread that brings it, the requester's own in-process or a socket carrier's
reader, under one lock, so requests reach `ClockAuthority.on_request` one at a
time, in the order they take the lock. An address is an LP, or
``(lp, member)`` for a member of an LP declared with members; each member binds
its own connection. Each reply that call returns goes to its address's own
reply slot, whether or not it is waiting on it, from the thread that served
the request: no other thread carries a request in or a reply out. A grant only
ever goes to an LP waiting in TAR or NER; the finish at the end of a run comes
when every LP waits, and answers each one's request with ``+inf``. A requester
whose reply is held gets nothing until a later request makes it due, and
nothing times it out. A bind is answered with what the LP starts from
(`ClockAuthority.on_bind`).

The end of a client run before the last answers the client with ``+inf``: it
has resigned, so its address is unbound, its connection may end without ending
the run, and the next client binds the same LP.

A `KeyError` or `ValueError` from the authority refuses the request: the
requester's `recv` raises it as the same type, and the server goes on serving.
A `ClockAbort`, such as `BackdatedEvent`, ends the run: its refusal goes to
every bound address's slot, waiting or not, and answers every later request.
Any other exception from the authority, or while framing a reply, ends the run
the same way, as a `ClockAbort` that names it; its table is empty when the LP
table cannot be read or framed. So does the end of a bound socket connection,
as a `ClockAbort` naming its address: no finish can come without it. A resigned
client's connection is no longer bound, and its end ends nothing.

Frames and closes reach the serve side as bytes: a carrier hands it encoded
frames and it decodes them, so the in-process carrier here moves the same bytes
a socket would; a socket's end reaches it as a ``None`` request. An LP waits for
its reply on its own slot and under no lock shared with another connection, so
one parked LP holds up no other.

Once every LP has sent a request, a wait of `diag_s` wall seconds with none
arriving is a stall the authority cannot see: a watch thread logs the LP table
once, counts it in the authority's `diagnostics` and keeps waiting. It logs
again only after a request arrives and a new stall begins, never after the run
has ended. It looks every quarter of `diag_s`, so a stall is logged between
`diag_s` and a quarter more after the last request.
"""

import contextlib
import logging
import math
import os
import queue
import threading

from atom.compass.clock import ClockAbort, LpId, lp_table_dump
from atom.utils.clock import DIAG_S

from .wire import (
    BIND,
    GRANT,
    REFUSALS,
    REFUSED,
    REQUESTS,
    MalformedMessage,
    decode,
    encode,
)

logger = logging.getLogger("atom")

IN_PROCESS_SCHEME = "inproc"
DEFAULT_ENDPOINT = f"{IN_PROCESS_SCHEME}:clock"

#: How many times a `diag_s` the watch thread looks for a request served.
_LOOKS = 4

#: The authorities served in this process, by endpoint.
_SERVED: dict[str, "_Server"] = {}


class _Server:
    """The serve side of one endpoint. It takes in frames and closes."""

    def __init__(self, authority, endpoint: str, diag_s: float = DIAG_S) -> None:
        self.endpoint = endpoint
        self._authority = authority
        self._diag_s = diag_s
        self._lps = [row.lp for row in authority.lp_table()]
        self._slots = {}
        # Reentrant: a socket carrier binds and sends the bind's reply under it.
        self._lock = threading.RLock()
        self._ended = None  # the refusal frame that ended the run, once one has
        self._unasked = set(self._lps)  # LPs yet to send a request
        self._quiet = True  # no stall diagnostic is due
        self._served = 0  # requests and closes served, for the watch
        self._closed_at = threading.Event()
        self._watch = threading.Thread(
            target=self._watching, name=f"clock {endpoint}", daemon=True
        )
        self._watch.start()

    def bind(self, frame: bytes, slot=None) -> tuple:
        """Bind a connection to the LP, or LP and member, its first frame names:
        that addressee, its reply slot and what it starts from. A member's name
        is checked by the authority at its first request. The slot is `slot`,
        anything with a `put` of a frame, or a new queue the LP takes its
        replies from."""
        kind, *rest = decode(frame)
        if kind != BIND:
            raise MalformedMessage(
                f"the first frame on a connection binds it to an LP, got {kind}"
            )
        lp, *member = rest
        address = (lp, *member) if member else lp
        slot = queue.Queue() if slot is None else slot
        with self._lock:
            taken = self._slots.setdefault(address, slot) is not slot
            if lp not in self._lps or taken:
                state, why = "not a participant", ""
                if lp in self._lps:
                    state = "already bound"
                    why = (
                        "two servers claim one deployment's LP; under xPyD each "
                        "names its own compass_instance, below the run file's "
                        "prefill_instances or decode_instances; "
                    )
                raise KeyError(
                    f"{' member '.join(map(str, rest))} is {state} at "
                    f"{self.endpoint}; {why}participants: "
                    + ", ".join(map(str, self._lps))
                )
            return address, slot, self._authority.on_bind(lp)

    def submit(self, address, frame: bytes) -> None:
        """Serve one request frame from the connection bound to `address`."""
        message = decode(frame)
        if message[0] not in REQUESTS:
            raise MalformedMessage(
                f"{address} sent {message[0]}; a bound connection sends only "
                + ", ".join(REQUESTS)
            )
        self._serve(address, message, None)

    def _closed(self, address, slot) -> None:
        """The connection bound to `address`, replying through `slot`, closed."""
        self._serve(address, None, slot)

    def close(self) -> None:
        """Stop serving and give up the endpoint."""
        if _SERVED.get(self.endpoint) is self:
            del _SERVED[self.endpoint]
        self._closed_at.set()
        self._watch.join()

    def _serve(self, address, request, slot) -> None:
        """Serve one request, or the close of the connection replying through
        `slot` when `request` is None, and put each reply in its slot."""
        lp, member = address if isinstance(address, tuple) else (address, None)
        with self._lock:
            if self._closed_at.is_set():
                return  # nothing is served once the endpoint is closed
            self._served += 1
            self._unasked.discard(lp)
            self._quiet = bool(self._unasked)
            replies, resigned = [(address, self._ended)], []
            if request is None:
                replies = []
                if self._ended is None and self._slots.get(address) is slot:
                    name = f"{lp} member {member}" if member else f"{lp}"
                    self._ended = self._ending(
                        f"{name} closed its connection, so the run ends here"
                    )
                    replies = [(i, self._ended) for i in list(self._slots)]
            elif self._ended is None:
                kind, t, log, t_daemon = request
                try:
                    try:
                        grants = self._authority.on_request(
                            lp, kind, t, log, t_daemon, member
                        )
                    except (KeyError, ValueError) as refused:
                        replies = [(address, encode(_refusal(refused)))]
                    else:
                        # Outside the refusing branch: a grant that cannot be
                        # framed ends the run, and no other grant goes out.
                        replies = [(i, encode((GRANT, *reply))) for i, *reply in grants]
                        # A client run ended: its client, granted +inf, resigned.
                        if any(len(reply) > 3 for reply in grants):
                            resigned = [i for i, g, *_ in grants if g == math.inf]
                except Exception as fault:  # noqa: BLE001 - ends the run, named
                    self._ended = self._ending(fault)
                    replies = [(i, self._ended) for i in list(self._slots)]
            for i, frame in replies:
                self._slots[i].put(frame)
            for i in resigned:
                del self._slots[i]

    def _watching(self) -> None:
        """Log a stall once no request has been served for `diag_s`."""
        idle = 0  # looks in a row that saw no request served
        seen = self._served
        while not self._closed_at.wait(self._diag_s / _LOOKS):
            with self._lock:
                idle = idle + 1 if self._served == seen else 0
                seen = self._served
                if idle < _LOOKS:
                    continue
                if (
                    not self._quiet
                    and self._ended is None
                    and self._authority.final_clocks is None
                ):
                    self._stalled()
                self._quiet = True

    def _stalled(self) -> None:
        """Log the LP table for a stall, once; reading it never ends the run."""
        self._authority.diagnostics += 1
        try:
            table = lp_table_dump(self._authority)
        except Exception as unread:  # noqa: BLE001 - a diagnostic never aborts
            table = f"no LP table: {unread!r}"
        logger.warning(
            "the clock authority at %s has had no request for %s wall seconds; "
            "still waiting. The LP table:\n%s",
            self.endpoint,
            self._diag_s,
            table,
        )

    def _ending(self, fault: Exception | str) -> bytes:
        """The refusal frame that ends the run on `fault`, an exception or a
        reason. Anything but a `ClockAbort` travels as one naming it, with the
        LP table, or with an empty table naming why the table could not be read
        or framed."""
        if isinstance(fault, ClockAbort):
            return encode(_refusal(fault))
        reason = (
            fault if isinstance(fault, str) else f"the clock authority raised {fault!r}"
        )
        try:
            return encode(_refusal(ClockAbort(reason, self._authority.lp_table())))
        except Exception as unread:  # noqa: BLE001 - the run still ends, named
            return encode(
                _refusal(ClockAbort(f"{reason}; no LP table: {unread!r}", ()))
            )


def _refusal(refused: Exception) -> tuple:
    error = next(name for name, cls in REFUSALS.items() if isinstance(refused, cls))
    if isinstance(refused, ClockAbort):
        return REFUSED, error, refused.reason, refused.table
    return REFUSED, error, " ".join(map(str, refused.args)), None


def _reply(frame: bytes) -> tuple:
    """The message `frame` carries; a refusal raises as its own type."""
    kind, *rest = decode(frame)
    if kind == REFUSED:
        error, reason, table = rest
        cls = REFUSALS[error]
        raise cls(reason, table) if issubclass(cls, ClockAbort) else cls(reason)
    return kind, *rest


class _Connection:
    """One LP's or member's connection: ``send((kind, t, log, t_daemon))``,
    ``recv() -> (G, released)``, ``close()``."""

    def __init__(self, server: _Server, lp: LpId, member: str | None = None) -> None:
        self._server = server
        # `joined`: the clock and next sequence numbers the LP starts from.
        self._lp, self._slot, self.joined = server.bind(encode((BIND, lp, member)))

    def send(self, request: tuple) -> None:
        self._server.submit(self._lp, encode(request))

    def recv(self) -> tuple:
        """Block until this LP's reply exists, ``(G, released)`` with the client
        run's index appended when it marks one's end; a refusal raises."""
        _, *reply = _reply(self._slot.get())
        return tuple(reply)

    def close(self) -> None:
        """Nothing to release: the reply slot is the LP's, not the connection's."""


class _ReadableSlot:
    """A reply slot an LP can also wait on with `select`: each reply makes its
    pipe readable by one byte, which `get` takes with the reply."""

    def __init__(self) -> None:
        self._replies = queue.SimpleQueue()
        self._read, self._write = os.pipe()

    def put(self, frame: bytes) -> None:
        self._replies.put(frame)
        with contextlib.suppress(OSError):  # the LP has closed its connection
            os.write(self._write, b"\0")

    def get(self) -> bytes:
        os.read(self._read, 1)
        return self._replies.get_nowait()

    def fileno(self) -> int:
        return self._read

    def close(self) -> None:
        os.close(self._read)
        os.close(self._write)


class _HereConnection(_Connection):
    """A connection from the process that serves the endpoint, whatever its
    scheme: no socket, and `fileno` is readable while a reply waits, as a
    socket's is."""

    def __init__(self, server: _Server, lp: LpId, member: str | None = None) -> None:
        self._server = server
        self._lp, self._slot, self.joined = server.bind(
            encode((BIND, lp, member)), _ReadableSlot()
        )

    def fileno(self) -> int:
        return self._slot.fileno()

    def close(self) -> None:
        self._slot.close()


def serve(
    authority, endpoint: str = DEFAULT_ENDPOINT, diag_s: float = DIAG_S
) -> _Server:
    """Serve `authority` at `endpoint`, logging a stall after `diag_s` wall
    seconds. `close()` on the result stops it."""
    if endpoint in _SERVED:
        raise ValueError(
            f"{endpoint} is already served in this process; two authorities at "
            "one endpoint would grant time from two states"
        )
    _SERVED[endpoint] = _Server(authority, endpoint, diag_s)
    return _SERVED[endpoint]


def connect(
    lp: LpId, endpoint: str = DEFAULT_ENDPOINT, member: str | None = None
) -> _Connection:
    """`lp`'s connection, or its `member`'s, to the authority served at `endpoint`."""
    server = _SERVED.get(endpoint)
    if server is None:
        raise KeyError(
            f"nothing is served at {endpoint} in this process; served here: "
            + (", ".join(sorted(_SERVED)) or "<none>")
        )
    return _Connection(server, lp, member)


def connect_here(
    lp: LpId, endpoint: str = DEFAULT_ENDPOINT, member: str | None = None
) -> _HereConnection:
    """`connect`, with a `fileno` to wait on."""
    server = _SERVED.get(endpoint)
    if server is None:
        raise KeyError(f"nothing is served at {endpoint} in this process")
    return _HereConnection(server, lp, member)
