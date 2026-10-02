# SPDX-License-Identifier: MIT
"""The Clock Authority's serve loop, and the in-process carrier to it.

`serve` starts one thread that takes ``(lp, request)`` off a single queue, in
arrival order, and passes it to `ClockAuthority.on_request`. Each reply that
call returns goes to its LP's own reply slot, whether or not that LP is waiting
on it. A grant only ever goes to an LP waiting in TAR or NER, but the finish at
the end of a run answers LPs that are still running too, and each of them reads
its ``+inf`` on its next request. A requester whose reply is held gets nothing
until a later request makes it due, and nothing times it out.

A refused request, a `ClockAbort` such as `BackdatedEvent` or a `KeyError` or
`ValueError` from the authority, is answered with a refusal frame. The
requester's `recv` raises it as the same type, and the loop goes on serving.

Only frames reach the loop: a carrier hands the serve side encoded frames and
the serve side decodes them onto the queue, so the in-process carrier here moves
the same bytes a socket would. An LP waits for its reply on its own slot and
under no lock shared with another connection, so one parked LP holds up no
other.
"""

import queue
import threading

from atom.compass.clock import ClockAbort, LpId

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

IN_PROCESS_SCHEME = "inproc"
DEFAULT_ENDPOINT = f"{IN_PROCESS_SCHEME}:clock"

#: The authorities served in this process, by endpoint.
_SERVED: dict[str, "_Server"] = {}


class _Server:
    """The serve side of one endpoint. Everything it takes in is a frame."""

    def __init__(self, authority, endpoint: str) -> None:
        self.endpoint = endpoint
        self._authority = authority
        self._slots = {row.lp: queue.Queue() for row in authority.lp_table()}
        self._unbound = dict(self._slots)
        self._requests = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name=f"clock {endpoint}", daemon=True
        )
        self._thread.start()

    def bind(self, frame: bytes) -> tuple[LpId, queue.Queue]:
        """Bind a connection to the LP its first frame names: that LP and its reply slot."""
        kind, *rest = decode(frame)
        if kind != BIND:
            raise MalformedMessage(
                f"the first frame on a connection binds it to an LP, got {kind}"
            )
        (lp,) = rest
        slot = self._unbound.pop(lp, None)
        if slot is None:
            state = "already bound" if lp in self._slots else "not a participant"
            raise KeyError(
                f"{lp} is {state} at {self.endpoint}; participants: "
                + ", ".join(map(str, self._slots))
            )
        return lp, slot

    def submit(self, lp: LpId, frame: bytes) -> None:
        """Queue one request frame from the connection bound to `lp`."""
        message = decode(frame)
        if message[0] not in REQUESTS:
            raise MalformedMessage(
                f"{lp} sent {message[0]}; a bound connection sends only "
                + ", ".join(REQUESTS)
            )
        self._requests.put((lp, message))

    def close(self) -> None:
        """Stop the loop and give up the endpoint."""
        if _SERVED.get(self.endpoint) is self:
            del _SERVED[self.endpoint]
        self._requests.put(None)
        self._thread.join()

    def _loop(self) -> None:
        while (item := self._requests.get()) is not None:
            lp, (kind, t, log) = item
            try:
                replies = [
                    (i, (GRANT, g, released))
                    for i, g, released in self._authority.on_request(lp, kind, t, log)
                ]
            except (ClockAbort, KeyError, ValueError) as refused:
                replies = [(lp, _refusal(refused))]
            for i, reply in replies:
                self._slots[i].put(encode(reply))


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
    """One LP's connection: ``send((kind, t, log))``, ``recv() -> (G, released)``, ``close()``."""

    def __init__(self, server: _Server, lp: LpId) -> None:
        self._server = server
        self._lp, self._slot = server.bind(encode((BIND, lp)))

    def send(self, request: tuple) -> None:
        self._server.submit(self._lp, encode(request))

    def recv(self) -> tuple:
        """Block until this LP's reply exists; a refusal raises."""
        _, g, released = _reply(self._slot.get())
        return g, released

    def close(self) -> None:
        """Nothing to release: the reply slot is the LP's, not the connection's."""


def serve(authority, endpoint: str = DEFAULT_ENDPOINT) -> _Server:
    """Serve `authority` at `endpoint`. `close()` on the result stops it."""
    _require_carried(endpoint)
    if endpoint in _SERVED:
        raise ValueError(
            f"{endpoint} is already served in this process; two authorities at "
            "one endpoint would grant time from two states"
        )
    _SERVED[endpoint] = _Server(authority, endpoint)
    return _SERVED[endpoint]


def connect(lp: LpId, endpoint: str = DEFAULT_ENDPOINT) -> _Connection:
    """`lp`'s connection to the authority served at `endpoint`."""
    _require_carried(endpoint)
    server = _SERVED.get(endpoint)
    if server is None:
        raise KeyError(
            f"nothing is served at {endpoint} in this process; served here: "
            + (", ".join(sorted(_SERVED)) or "<none>")
        )
    return _Connection(server, lp)


def _require_carried(endpoint: str) -> None:
    scheme, separator, _ = endpoint.partition(":")
    if not separator or scheme != IN_PROCESS_SCHEME:
        raise ValueError(f"{endpoint!r} is not {IN_PROCESS_SCHEME}:<name>")
