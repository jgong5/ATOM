# SPDX-License-Identifier: MIT
"""A simulated frontend's streamed body messages through anyio's memory object
streams and uvicorn's h11 send, with less generic machinery per call.

Starlette's ``BaseHTTPMiddleware`` hands each body message of a response on
through an anyio memory object stream, and uvicorn's H11 protocol frames each
one through h11's state machine; jgong5/ATOM#4214 measures their share of a
decode frontend's own work. A simulated frontend does that work while its
LP's clock stands still, so every other LP waits on its wall time.

`install` replaces, once a process:

- ``MemoryObjectReceiveStream.receive`` and ``MemoryObjectSendStream.send``
  with anyio's own code on its asyncio backend, that backend looked up once
  rather than for each checkpoint, event and task. A waiting receiver is built
  without calling the subscripted generic, which only records the alias on it.
- uvicorn's H11 ``RequestResponseCycle.send``, for a body message with more
  to come on a chunked response under way: the chunk h11 frames for it,
  ``<length in hex> CRLF <body> CRLF``, in the one ``transport.write`` uvicorn
  makes of it. h11 changes no state for such a send. Any other message, and
  any sent while writing is paused, takes uvicorn's own send.

Each keeps every await, so the loop's task switches and their order are the
libraries' own, and every byte. Each is written against the library versions
`CHECKED` names, and installs nothing on any other. anyio picks its backend by
the event loop running, and a simulated frontend runs asyncio's alone.
"""

import functools
import logging
from importlib import metadata

logger = logging.getLogger("atom")

#: The library versions the fast paths are written against.
CHECKED = {"anyio": "4.16.0", "h11": "0.16.0", "uvicorn": "0.54.0"}
#: The attribute marking a method this module installed; its ``__wrapped__``
#: is the method it replaced.
_INSTALLED = "compass_fe_fastpath"


def install() -> None:
    """Install each fast path whose libraries are at their `CHECKED`
    versions, once a process; log each library that is not."""
    if _checked("anyio"):
        from anyio.streams.memory import (
            MemoryObjectReceiveStream,
            MemoryObjectSendStream,
        )

        receive, send = _memory_stream_paths()
        _replace(MemoryObjectReceiveStream, "receive", lambda _: receive)
        _replace(MemoryObjectSendStream, "send", lambda _: send)
    if _checked("h11", "uvicorn"):
        from uvicorn.protocols.http.h11_impl import RequestResponseCycle

        _replace(RequestResponseCycle, "send", _with_chunk_fast_path)


@functools.cache
def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _checked(*dists: str) -> bool:
    """Whether each of `dists` is installed at its `CHECKED` version."""
    off = [d for d in dists if _version(d) != CHECKED[d]]
    for d in off:
        logger.info(
            "compass: %s is %s, not the %s its fast path is written against; "
            "it runs its own code",
            d,
            _version(d),
            CHECKED[d],
        )
    return not off


def _replace(owner, name: str, make) -> None:
    """Replace `owner`'s `name` with ``make(current)``, unless this module
    did already."""
    current = getattr(owner, name)
    if not getattr(current, _INSTALLED, False):
        new = make(current)
        new.__wrapped__ = current
        setattr(new, _INSTALLED, True)
        setattr(owner, name, new)


def _memory_stream_paths():
    """anyio's ``receive`` and ``send`` of a memory object stream, as its
    asyncio backend runs them."""
    from anyio import (
        BrokenResourceError,
        ClosedResourceError,
        EndOfStream,
        WouldBlock,
    )
    from anyio._backends._asyncio import AsyncIOBackend as backend
    from anyio.streams.memory import _MemoryObjectItemReceiver as Receiver

    async def receive(self):
        await backend.checkpoint()
        try:
            return self.receive_nowait()
        except WouldBlock:
            if self._waiters is None:
                self._waiters = set()

            receive_event = backend.create_event()
            self._waiters.add(receive_event)
            # What Receiver's __init__ does, with the task from the backend.
            receiver = object.__new__(Receiver)
            receiver.task_info = backend.get_current_task()
            self._state.waiting_receivers[receive_event] = receiver

            try:
                await receive_event.wait()
            finally:
                self._state.waiting_receivers.pop(receive_event, None)
                if self._waiters is not None:
                    self._waiters.remove(receive_event)

            try:
                return receiver.item
            except AttributeError:
                if self._closed:
                    raise ClosedResourceError from None
                else:
                    raise EndOfStream from None

    async def send(self, item):
        await backend.checkpoint()
        try:
            self.send_nowait(item)
        except WouldBlock:
            if self._waiters is None:
                self._waiters = set()

            send_event = backend.create_event()
            self._waiters.add(send_event)
            self._state.waiting_senders[send_event] = item
            try:
                await send_event.wait()
            finally:
                if item_not_consumed := send_event in self._state.waiting_senders:
                    del self._state.waiting_senders[send_event]

                withdrawn = self._closed and send_event in self._waiters
                self._waiters.discard(send_event)

            if item_not_consumed:
                raise BrokenResourceError from None
            elif withdrawn:
                raise ClosedResourceError from None

    return receive, send


def _with_chunk_fast_path(send):
    """`send`, uvicorn's H11 ``RequestResponseCycle.send``, writing a body
    message with more to come on a chunked response under way as the chunk
    h11 frames for it."""
    from h11 import SEND_BODY
    from h11._writers import ChunkedWriter

    async def send_chunk_fast(cycle, message):
        if (
            cycle.response_started
            and not cycle.response_complete
            and not cycle.disconnected
            and not cycle.flow.write_paused
            and message["type"] == "http.response.body"
            and message.get("more_body", False)
            and cycle.scope["method"] != "HEAD"
            and type(body := message.get("body", b"")) is bytes
            and body
            and cycle.conn.our_state is SEND_BODY
            and type(cycle.conn._writer) is ChunkedWriter
        ):
            # h11's Connection.send of Data(body) here: ChunkedWriter's three
            # writes, joined, and the state SEND_BODY it was in.
            cycle.transport.write(b"%x\r\n%b\r\n" % (len(body), body))
            return
        await send(cycle, message)

    return send_chunk_fast
