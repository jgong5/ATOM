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

``connect(lp, "tcp://host:port")`` returns the in-process carrier's connection
with a socket in place of the serve side: the same ``send``, ``recv`` and ``close``,
and the same frames, each preceded by its length as four big-endian bytes, plus
``fileno`` to wait for a reply alongside other sockets. A
connection that ends inside a frame is a `MalformedMessage`. A frame the serve side
will not take, such as a reply kind sent as a request, is refused at ``recv``
here and at ``send`` in-process, with the same reason, and reaches no authority.
"""

import contextlib
import socket
import threading

from .service import _Connection, _HereConnection, _refusal, _reply, _Server
from .wire import JOINED, MalformedMessage, encode

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
        slot = None
        try:
            with sock.makefile("rb") as stream:
                while (frame := _receive(stream)) is not None:
                    try:
                        if slot is None:
                            # Bound and answered under the serve side's lock,
                            # so no reply reaches the socket before the bind's.
                            with self._server._lock:
                                address, slot, joined = self._server.bind(
                                    frame, _SocketSlot(sock)
                                )
                                try:
                                    reply = encode((JOINED, *joined))
                                except MalformedMessage as refused:
                                    reply = encode(_refusal(refused))
                                slot.put(reply)
                        else:
                            self._server.submit(address, frame)
                    except (KeyError, ValueError) as refused:
                        reply = encode(_refusal(refused))
                        if slot is None:
                            sock.sendall(_framed(reply))
                        else:
                            slot.put(reply)
        except (MalformedMessage, OSError):
            pass
        finally:
            if slot is None:
                _end(sock)
            else:
                # Ended first: the peer has gone, so the end of the run its
                # close makes is for every other LP.
                slot.put(None)
                self._server._closed(address, slot)


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


def serve(authority, endpoint: str) -> _StreamServer:
    """Serve `authority` at ``tcp://host:port``."""
    host, port = _address(endpoint)
    server = _StreamServer(authority, host, port)
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
