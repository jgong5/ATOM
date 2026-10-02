# SPDX-License-Identifier: MIT
"""How a logical process (LP) reaches the Clock Authority.

``serve(authority, endpoint)`` starts the authority's serve loop at an endpoint;
``connect(lp, endpoint)`` returns the LP's connection to it, whose whole surface
is ``send((kind, t, log, t_daemon))``, ``recv() -> (G, released)`` and
``close()``. The endpoint is the only place a location appears, and its scheme
picks the carrier: ``inproc:<name>`` serves in this process (`service`),
``tcp://<host>:<port>`` over a socket (`stream`). Both move the same encoded
frames into one serve loop.

This sits beside `atom.compass.clock` rather than inside it because it starts a
thread and queues frames, and that package reaches nothing.
"""

from . import service, stream
from .service import DEFAULT_ENDPOINT
from .wire import MalformedMessage, decode, encode

#: The module that carries each endpoint scheme.
_CARRIERS = {service.IN_PROCESS_SCHEME: service, stream.STREAM_SCHEME: stream}


def serve(authority, endpoint: str = DEFAULT_ENDPOINT):
    """Serve `authority` at `endpoint`. `close()` on the result stops it."""
    return _carrier(endpoint).serve(authority, endpoint)


def connect(lp, endpoint: str = DEFAULT_ENDPOINT):
    """`lp`'s connection to the authority served at `endpoint`."""
    return _carrier(endpoint).connect(lp, endpoint)


def _carrier(endpoint: str):
    scheme, separator, _ = endpoint.partition(":")
    if not separator or scheme not in _CARRIERS:
        raise ValueError(
            f"{endpoint!r} is not inproc:<name> or tcp://<host>:<port>; "
            "no carrier is guessed"
        )
    return _CARRIERS[scheme]


__all__ = [
    "DEFAULT_ENDPOINT",
    "MalformedMessage",
    "connect",
    "decode",
    "encode",
    "serve",
]
