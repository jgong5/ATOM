# SPDX-License-Identifier: MIT
"""Ports held for a server that must be told them before it binds them.

Each port is taken by a plain bind on every address, which fails on a port any
other socket holds, so no two holders, and no port draw such as
`atom.compass.drive.serve._held_ports`, take the same one. The holder then sets
SO_REUSEADDR and never listens: a listener that sets it too binds the port
beside the holder, as uvicorn's, libzmq's, `socket.create_server`'s, torch's
TCPStore's and Rust's `std::net::TcpListener` do, while a plain bind elsewhere
still fails on it.
"""

import contextlib
import itertools
import random
import socket

from atom.compass.drive.serve import SERVE_PORTS, below_ephemeral


def _hold(stack: contextlib.ExitStack, port: int) -> bool:
    """Hold `port` until `stack` closes: whether it was free."""
    s = socket.socket()
    try:
        s.bind(("", port))
    except OSError:
        s.close()
        return False
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stack.enter_context(s)
    return True


@contextlib.contextmanager
def held_ports(n: int, derived=lambda ports: ()):
    """`n` ports drawn at random from `SERVE_PORTS` below the ephemeral range,
    and the ports `derived` names from them, all held until leaving; drawn
    again until the derived ones are in that range and free too."""
    room = below_ephemeral(SERVE_PORTS)
    while True:
        with contextlib.ExitStack() as stack:
            draw = (p for p in random.sample(room, len(room)) if _hold(stack, p))
            ports = list(itertools.islice(draw, n))
            assert len(ports) == n, f"fewer than {n} free ports in {room}"
            if all(p in room and _hold(stack, p) for p in derived(ports)):
                yield ports
                return
