# SPDX-License-Identifier: MIT
"""The ``tcp`` carrier: the Clock Authority's serve side reached over a socket.

``serve(authority, "tcp://host:port")`` listens there; port 0 binds a free one,
and the result's ``endpoint`` names the port it got. Each accepted connection
gets a reader thread. Its first frame binds it to an LP through the serve side,
which answers with what the LP starts from, or a refusal. After that, the reader
serves each request frame itself (`service`), and whichever thread serves a
request writes each reply it makes straight onto its LP's socket: no thread
stands between a socket and the serve side. A reply is a few hundred bytes and
its LP reads every reply it is owed, so a write does not wait on a peer. The
reader waits for no reply, so when a bound connection ends, however it ends and
even while its LP waits for a grant, it tells the serve side then, which ends
the run.

``serve(authority, endpoint, select=True)`` reads every connection on one
thread instead, through a selector, and serves each frame there as soon as it
is whole, the way a reader serves it, so no request waits for the serve side's
lock behind another connection's reader. Each connection's frames are still
served in the order they were sent, each after it arrived and one at a time,
so the order it serves two connections' frames in is one their readers could
also take the lock in; frames that end a connection, or that the serve side
refuses, end or refuse it as a reader does.

``connect(lp, "tcp://host:port")`` returns the in-process carrier's connection
with a socket in place of the serve side: the same ``send``, ``recv`` and ``close``,
and the same frames, each preceded by its length as four big-endian bytes, plus
``fileno`` to wait for a reply alongside other sockets. A
connection that ends inside a frame is a `MalformedMessage`. A frame the serve side
will not take, such as a reply kind sent as a request, is refused at ``recv``
here and at ``send`` in-process, with the same reason, and reaches no authority.
"""

import contextlib
import logging
import selectors
import socket
import threading

from .service import _Connection, _HereConnection, _refusal, _reply, _Server
from .wire import JOINED, MalformedMessage, encode

logger = logging.getLogger("atom")

STREAM_SCHEME = "tcp"
#: The servers this process runs, by the endpoint each bound and the one it was
#: asked for.
_SERVED: dict[str, "_StreamServer"] = {}
#: A frame is a few hundred bytes. A length far above that is a peer that is not
#: speaking this protocol, and is refused before anything is allocated for it.
LARGEST_FRAME = 1 << 22


def _framed(frame: bytes) -> bytes:
    return len(frame).to_bytes(4, "big") + frame


def _send(stream, frame: bytes) -> None:
    stream.write(_framed(frame))
    stream.flush()


def _receive(stream) -> bytes | None:
    """The next frame, or `None` when the peer closed between frames."""
    header = stream.read(4)
    if not header:
        return None
    size = int.from_bytes(header, "big")
    if size > LARGEST_FRAME:
        raise MalformedMessage(f"a {size}-byte frame is over {LARGEST_FRAME}")
    frame = stream.read(size)
    if len(header) < 4 or len(frame) < size:
        raise MalformedMessage("the connection ended inside a frame")
    return frame


def _end(sock: socket.socket) -> None:
    """Wake any thread reading `sock`, then close it."""
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    sock.close()


class _Incoming:
    """A connection as the serve side reads it: the bytes of a frame not yet
    whole, and once its first frame has bound it, its address and reply slot."""

    __slots__ = ("address", "buffer", "slot")

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.address = self.slot = None


class _StreamServer:
    """A serve side that LPs reach over TCP."""

    def __init__(self, authority, host: str, port: int) -> None:
        self._listener = socket.create_server((host, port))
        host, port = self._listener.getsockname()[:2]
        self.endpoint = f"{STREAM_SCHEME}://{host}:{port}"
        self._server = _Server(authority, self.endpoint)
        self._open = []
        threading.Thread(target=self._admit, daemon=True).start()

    def close(self) -> None:
        """Stop listening, end every connection and stop serving."""
        for name in [k for k, v in _SERVED.items() if v is self]:
            del _SERVED[name]
        for sock in [self._listener, *self._open]:
            _end(sock)
        self._server.close()

    def _admit(self) -> None:
        while True:
            try:
                sock, _ = self._listener.accept()
            except OSError:
                return
            self._open.append(sock)
            threading.Thread(target=self._read, args=(sock,), daemon=True).start()

    def _read(self, sock: socket.socket) -> None:
        incoming = _Incoming()
        try:
            with sock.makefile("rb") as stream:
                while (frame := _receive(stream)) is not None:
                    self._take(sock, incoming, frame)
        except (MalformedMessage, OSError):
            pass
        finally:
            self._ended(sock, incoming)

    def _take(self, sock: socket.socket, incoming: _Incoming, frame: bytes) -> None:
        """Serve `frame`, the next one `sock` sent: its first binds it."""
        try:
            if incoming.slot is None:
                # Bound and answered under the serve side's lock, so no reply
                # reaches the socket before the bind's.
                with self._server._lock:
                    incoming.address, incoming.slot, joined = self._server.bind(
                        frame, _SocketSlot(sock)
                    )
                    try:
                        reply = encode((JOINED, *joined))
                    except MalformedMessage as refused:
                        reply = encode(_refusal(refused))
                    incoming.slot.put(reply)
            else:
                self._server.submit(incoming.address, frame)
        except (KeyError, ValueError) as refused:
            reply = encode(_refusal(refused))
            if incoming.slot is None:
                sock.sendall(_framed(reply))
            else:
                incoming.slot.put(reply)

    def _ended(self, sock: socket.socket, incoming: _Incoming) -> None:
        """`sock` ended, however it did: end it, and a bound one's run."""
        if incoming.slot is None:
            _end(sock)
        else:
            # Ended first: the peer has gone, so the end of the run its
            # close makes is for every other LP.
            incoming.slot.put(None)
            self._server._closed(incoming.address, incoming.slot)


class _SelectServer(_StreamServer):
    """A serve side whose one thread accepts and reads every connection through
    a selector, in place of a reader thread a connection; each frame is served
    with `_take` once whole, and each end with `_ended`, as a reader does. A
    fault no reader expects ends its connection alone, as it would end that
    connection's reader."""

    def __init__(self, authority, host: str, port: int) -> None:
        self._selector = selectors.DefaultSelector()
        # Written once by `close`, so the selector's thread leaves.
        self._wake, self._woken = socket.socketpair()
        self._selector.register(self._woken, selectors.EVENT_READ, self._woken)
        super().__init__(authority, host, port)

    def close(self) -> None:
        super().close()
        with contextlib.suppress(OSError):
            self._wake.send(b"\0")

    def _admit(self) -> None:
        self._selector.register(self._listener, selectors.EVENT_READ, self._listener)
        try:
            while True:
                for key, _ in self._selector.select():
                    if key.data is self._woken:
                        return
                    if key.data is not self._listener:
                        self._readable(key.fileobj, key.data)
                        continue
                    try:
                        sock, _ = self._listener.accept()
                    except OSError:  # admits no more, as a reader's server
                        self._selector.unregister(self._listener)
                        continue
                    self._open.append(sock)
                    self._selector.register(sock, selectors.EVENT_READ, _Incoming())
        finally:
            self._selector.close()
            for sock in (self._wake, self._woken):
                sock.close()

    def _readable(self, sock: socket.socket, incoming: _Incoming) -> None:
        """Serve, in order, every frame `sock`'s next bytes complete; end it
        when it ends or sends what is not a frame."""
        try:
            data = sock.recv(1 << 16)
            buffer = incoming.buffer
            buffer += data
            while len(buffer) >= 4:
                size = int.from_bytes(buffer[:4], "big")
                if size > LARGEST_FRAME:
                    raise MalformedMessage(
                        f"a {size}-byte frame is over {LARGEST_FRAME}"
                    )
                if len(buffer) < 4 + size:
                    break
                frame = bytes(buffer[4 : 4 + size])
                del buffer[: 4 + size]
                self._take(sock, incoming, frame)
            if data:
                return
        except Exception as ended:  # ends this connection alone
            if not isinstance(ended, (MalformedMessage, OSError)):
                logger.exception(
                    "the clock's serve side ends a connection it failed on"
                )
        self._selector.unregister(sock)
        self._ended(sock, incoming)


class _SocketSlot:
    """An LP's reply slot on the serve side: `put` writes the frame onto the
    LP's socket at once, and ``put(None)`` ends the socket, after which a put
    writes nothing; nor does one to a peer that has gone. Its own lock keeps a
    refusal the reader writes from interleaving with a reply another thread
    writes."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._lock = threading.Lock()
        self._open = True

    def put(self, frame: bytes | None) -> None:
        with self._lock:
            if not self._open:
                return
            if frame is None:
                self._open = False
                _end(self._sock)
                return
            with contextlib.suppress(OSError):
                self._sock.sendall(_framed(frame))


class _Remote:
    """The LP's end of a TCP connection, in the server-and-slot shape
    `_Connection` drives."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.create_connection((host, port))
        # Each frame goes at once: a REPORT waits for no reply, so Nagle would
        # hold the call after it until the report's delayed ACK, 40 ms.
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stream = self._sock.makefile("rwb")

    def bind(self, frame: bytes) -> tuple:
        _send(self._stream, frame)
        try:
            _, *joined = _reply(self.get())
        except Exception:
            self.close()
            raise
        return None, self, tuple(joined)

    def submit(self, _lp, frame: bytes) -> None:
        _send(self._stream, frame)

    def get(self) -> bytes:
        frame = _receive(self._stream)
        if frame is None:
            raise MalformedMessage("the clock closed the connection with no reply")
        return frame

    def close(self) -> None:
        self._stream.close()
        _end(self._sock)


class _StreamConnection(_Connection):
    def fileno(self) -> int:
        """The socket's, readable once a reply is on its way."""
        return self._server._sock.fileno()

    def close(self) -> None:
        """End the socket."""
        self._server.close()


def serve(authority, endpoint: str, select: bool = False) -> _StreamServer:
    """Serve `authority` at ``tcp://host:port``, every connection on one
    selector thread with `select`, else each on a reader thread of its own."""
    host, port = _address(endpoint)
    server = (_SelectServer if select else _StreamServer)(authority, host, port)
    _SERVED[server.endpoint] = server
    if port:  # port 0 names no one server
        _SERVED[endpoint] = server
    return server


def connect_here(lp, endpoint: str, member: str | None = None):
    """`lp`'s connection, or its `member`'s, to the authority this process
    serves at ``tcp://host:port``, with no socket (`service._HereConnection`)."""
    server = _SERVED.get(endpoint)
    if server is None:
        raise KeyError(f"nothing is served at {endpoint} in this process")
    return _HereConnection(server._server, lp, member)


def connect(lp, endpoint: str, member: str | None = None) -> _StreamConnection:
    """`lp`'s connection, or its `member`'s, to the authority served at
    ``tcp://host:port``."""
    return _StreamConnection(_Remote(*_address(endpoint)), lp, member)


def _address(endpoint: str) -> tuple[str, int]:
    prefix = f"{STREAM_SCHEME}://"
    host, _, port = endpoint.removeprefix(prefix).rpartition(":")
    if not endpoint.startswith(prefix) or not host or not port.isdigit():
        raise ValueError(f"{endpoint!r} is not {prefix}<host>:<port>")
    return host, int(port)
