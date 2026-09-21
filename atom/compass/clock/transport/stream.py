# SPDX-License-Identifier: MIT
"""Moving a frame over a socket, for the arrangement that spans containers.

This is the one module in the package that is allowed to say where something
is, and it says it here so that nothing else has to. A participant names a
peer; a location belongs to whatever turns a written-down address into a way of
reaching it, which is this and the module that picks between this and a direct
call.

What is here is only carriage. The protocol, the stamps and the rule are
elsewhere and are shared with the arrangement that runs inside one process:
this module writes bytes it does not read and reads bytes it does not
interpret. That division is what keeps the two arrangements one implementation
rather than two, and it is why the interesting code is not here.

Frames are length-prefixed -- a decimal count, a newline, then that many bytes
-- because a stream has no message boundaries of its own and a reader that
guesses one will eventually split a grant. Each connection gets a buffered file
over the socket, so a short read is the buffer's problem rather than a
truncated grant.

Requests are answered one at a time under a lock. The rule holds every
participant's clock and is not written to be entered twice at once, and serving
grants concurrently would buy nothing: the work per grant is a minimum over a
handful of numbers, and the run's cost is in what the participants do between
grants.
"""

import socket
import threading

#: How an address for this carriage is written down.
STREAM_SCHEME = "tcp"

#: Refuse a frame larger than this rather than allocate whatever was asked for.
#: A grant message is a few hundred bytes; a count far above that is a fault in
#: the sender or a stream that is not carrying this protocol at all.
LARGEST_FRAME_BYTES = 1 << 22


class _Frames:
    """Length-prefixed frames over one open connection."""

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._stream = connection.makefile("rwb")

    def send(self, frame: bytes) -> None:
        self._stream.write(b"%d\n" % len(frame))
        self._stream.write(frame)
        self._stream.flush()

    def receive(self) -> bytes | None:
        """The next frame, or `None` once the far side has finished."""
        header = self._stream.readline()
        if not header:
            return None
        count = int(header)
        if count < 0 or count > LARGEST_FRAME_BYTES:
            raise ValueError(
                f"a frame of {count} bytes was announced, above the "
                f"{LARGEST_FRAME_BYTES} this carries; the stream is not this "
                "protocol, or the sender is faulty"
            )
        frame = self._stream.read(count)
        if len(frame) != count:
            raise ConnectionError(
                f"{len(frame)} of {count} bytes arrived before the connection ended"
            )
        return frame

    def close(self) -> None:
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._stream.close()
        self._connection.close()


class StreamCarrier:
    """Puts a frame on a socket and waits for the frame that answers it."""

    def __init__(self, host: str, port: int) -> None:
        self._frames = _Frames(socket.create_connection((host, port)))

    def exchange(self, frame: bytes) -> bytes:
        self._frames.send(frame)
        reply = self._frames.receive()
        if reply is None:
            raise ConnectionError(
                "the clock closed the connection without answering; a run "
                "cannot continue without knowing whether it was granted time"
            )
        return reply

    def close(self) -> None:
        self._frames.close()


class StreamServer:
    """A clock in a process of its own, reachable over a socket.

    Started before the participants, since a participant's first message is
    what tells it where its clock stands.
    """

    def __init__(self, service, host: str, port: int) -> None:
        self._service = service
        self._lock = threading.Lock()
        self._open = []
        self._running = True
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(64)
        self._bound = self._listener.getsockname()
        self._door = threading.Thread(target=self._admit, daemon=True)
        self._door.start()

    @property
    def endpoint(self) -> str:
        """Where this clock is reachable, with whatever was left to be chosen.

        Asking for port zero binds a free one, so this is the value to hand a
        participant rather than the value that was requested.
        """
        return f"{STREAM_SCHEME}://{self._bound[0]}:{self._bound[1]}"

    def close(self) -> None:
        self._running = False
        self._listener.close()
        while self._open:
            self._open.pop().close()

    def _admit(self) -> None:
        while self._running:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            self._open.append(_Frames(connection))
            threading.Thread(
                target=self._serve, args=(self._open[-1],), daemon=True
            ).start()

    def _serve(self, frames: _Frames) -> None:
        while True:
            try:
                frame = frames.receive()
            except (ConnectionError, OSError, ValueError):
                return
            if frame is None:
                return
            with self._lock:
                reply = self._service.handle(frame)
            try:
                frames.send(reply)
            except OSError:
                return

    def __repr__(self) -> str:
        return f"StreamServer({self.endpoint})"


def split_location(rest: str) -> tuple[str, int]:
    """``//host:port`` -> the two parts of it. Refuses anything else."""
    if not rest.startswith("//") or ":" not in rest[2:]:
        raise ValueError(
            f"{STREAM_SCHEME}:{rest!r} is not //<host>:<port>; a socket needs "
            "both, and a missing port is not a default worth guessing"
        )
    host, _, port = rest[2:].rpartition(":")
    try:
        return host, int(port)
    except ValueError as fault:
        raise ValueError(f"{port!r} is not a port number") from fault


def open_carrier(rest: str) -> StreamCarrier:
    """Reach a clock written down as ``//host:port``."""
    host, port = split_location(rest)
    return StreamCarrier(host, port)


def open_server(service, rest: str) -> StreamServer:
    """Serve a clock at ``//host:port``. Port zero binds a free one."""
    host, port = split_location(rest)
    return StreamServer(service, host, port)
