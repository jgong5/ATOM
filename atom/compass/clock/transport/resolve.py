# SPDX-License-Identifier: MIT
"""Turning a written-down endpoint into a way of reaching the clock.

The whole difference between the two arrangements lives in this module, and it
is four lines of it. Co-hosted in the API-server process, which is what almost
every run wants: the endpoint resolves to a direct call and there is no socket
and no second process to start, supervise or leak. Standalone, for a run split
across containers or nodes, where neither container is the obvious owner of the
clock and making one of them the owner would give the two roles different
failure behaviour from the one being modelled: the endpoint resolves to a
socket.

Both arrangements carry the same frames to the same code. The direct call is
not a shortcut past the protocol -- it hands over the bytes a socket would have
carried, stamps and all -- and that is deliberate, because the cheap
arrangement is the one almost every run uses and it has to be evidence about
the expensive one.

A participant asks for an endpoint and gets a session. It is never told which
of the two it got, and there is nothing on the session to find out with. The
same two calls serve a clock that has participants of its own, which is what an
arrangement with a clock between two others would be built from.
"""

from ..identity import LpId
from .session import ClockSession
from .stream import STREAM_SCHEME, open_carrier, open_server

#: How a clock inside this process is written down.
IN_PROCESS_SCHEME = "inproc"

#: Where a run puts its clock when nothing says otherwise: in this process.
DEFAULT_ENDPOINT = f"{IN_PROCESS_SCHEME}:clock"

#: The clocks served inside this process, by endpoint. A dict rather than a
#: set-like structure, and never iterated for anything a run depends on.
_SERVED_HERE: dict[str, "InProcessServer"] = {}


class InProcessCarrier:
    """Hands the frame to a clock in this process, and takes the answer back.

    Bytes in, bytes out, exactly as a socket would: the frame is built and
    parsed by the same code in both arrangements. Anything faster would have to
    stop carrying the frame, and a frame that is not carried is a stamp that is
    not carried -- at which point a single-container run no longer tests what a
    multi-container run does.
    """

    def __init__(self, server: "InProcessServer") -> None:
        self._server = server

    def exchange(self, frame: bytes) -> bytes:
        return self._server.handle(frame)

    def close(self) -> None:
        """Nothing to let go of. The clock outlives the participant's session."""

    def __repr__(self) -> str:
        return f"InProcessCarrier({self._server.endpoint})"


class InProcessServer:
    """A clock co-hosted in this process, reachable without a socket."""

    def __init__(self, service, endpoint: str) -> None:
        self._service = service
        self._endpoint = endpoint

    @property
    def endpoint(self) -> str:
        """Where this clock is reachable."""
        return self._endpoint

    def handle(self, frame: bytes) -> bytes:
        return self._service.handle(frame)

    def close(self) -> None:
        """Give up the name, unless somebody else has taken it since.

        A second close of a clock already taken down would otherwise unregister
        whichever clock is serving that name now -- and a run set up twice in
        one process, which is what a test session is, reuses names.
        """
        if _SERVED_HERE.get(self._endpoint) is self:
            del _SERVED_HERE[self._endpoint]

    def __repr__(self) -> str:
        return f"InProcessServer({self._endpoint})"


def serve(service, endpoint: str = DEFAULT_ENDPOINT):
    """Make a clock reachable, and say where it ended up.

    Read `.endpoint` off the result rather than reusing the argument: a socket
    asked to take any free port answers with the one it got.
    """
    scheme, rest = _split(endpoint)
    if scheme == IN_PROCESS_SCHEME:
        if endpoint in _SERVED_HERE:
            raise ValueError(
                f"{endpoint} already has a clock in this process; two clocks "
                "at one endpoint would hand out time from two sets of state"
            )
        _SERVED_HERE[endpoint] = InProcessServer(service, endpoint)
        return _SERVED_HERE[endpoint]
    if scheme == STREAM_SCHEME:
        return open_server(service, rest)
    raise _unknown(scheme, endpoint)


def connect(lp_id: LpId, endpoint: str = DEFAULT_ENDPOINT) -> ClockSession:
    """Join the run as `lp_id`, through whichever carriage the endpoint names."""
    session = ClockSession(lp_id, carrier_for(endpoint))
    session.attach()
    return session


def carrier_for(endpoint: str):
    """The carriage an endpoint names, without joining the run through it."""
    scheme, rest = _split(endpoint)
    if scheme == IN_PROCESS_SCHEME:
        server = _SERVED_HERE.get(endpoint)
        if server is None:
            here = ", ".join(sorted(_SERVED_HERE)) or "<none>"
            raise KeyError(
                f"no clock is served at {endpoint} in this process; served "
                f"here: {here}. A co-hosted clock has to be served before a "
                "participant can reach it"
            )
        return InProcessCarrier(server)
    if scheme == STREAM_SCHEME:
        return open_carrier(rest)
    raise _unknown(scheme, endpoint)


def _split(endpoint: str) -> tuple[str, str]:
    if not isinstance(endpoint, str):
        raise TypeError(f"an endpoint is a str, got {type(endpoint).__name__}")
    scheme, separator, rest = endpoint.partition(":")
    if not separator:
        raise ValueError(
            f"{endpoint!r} names no carriage; write "
            f"{IN_PROCESS_SCHEME}:<name> or {STREAM_SCHEME}://<where>"
        )
    return scheme, rest


def _unknown(scheme: str, endpoint: str) -> ValueError:
    return ValueError(
        f"{endpoint!r} asks for {scheme!r}, which nothing here carries; "
        f"{IN_PROCESS_SCHEME} and {STREAM_SCHEME} are what there is"
    )
