# SPDX-License-Identifier: MIT
"""How a logical process (LP) reaches the Clock Authority.

``serve(authority, endpoint)`` starts the authority's serve loop at an endpoint;
``connect(lp, endpoint)`` returns the LP's connection to it, whose whole surface
is ``send((kind, t, log))``, ``recv() -> (G, released)`` and ``close()``. The
endpoint is the only place a location appears. ``inproc:<name>`` serves in this
process and moves the same encoded frames a socket would.

This sits beside `atom.compass.clock` rather than inside it because it starts a
thread and queues frames, and that package reaches nothing.
"""

from .service import DEFAULT_ENDPOINT, connect, serve
from .wire import MalformedMessage, decode, encode

__all__ = [
    "DEFAULT_ENDPOINT",
    "MalformedMessage",
    "connect",
    "decode",
    "encode",
    "serve",
]
