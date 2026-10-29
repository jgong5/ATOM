# SPDX-License-Identifier: MIT
"""`held_ports`: another process's bind and a port draw take no held port, and
each listener the tests' servers start binds one beside its holder."""

import contextlib
import errno
import socket
import subprocess
import sys
import time

import pytest
import zmq
from torch.distributed import TCPStore

from atom.compass import clock_transport
from atom.compass.clock import ClockAuthority, single_engine_table
from atom.compass.drive import serve
from tests.compass.held_ports import held_ports

#: Another process: a listener on each port it is given, on every address and
#: then on 127.0.0.1, printing each bind's errno, 0 when it bound.
SQUAT = """
import socket, sys
for port in map(int, sys.argv[1:]):
    for host in ("", "127.0.0.1"):
        with socket.socket() as s:
            try:
                s.bind((host, port))
                s.listen()
                print(0)
            except OSError as e:
                print(e.errno)
"""
UVICORN = """
import sys, uvicorn
async def app(scope, receive, send):
    pass
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), lifespan="off")
"""


def _connects(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port)).close()
        return True
    except ConnectionRefusedError:
        return False


def test_another_process_and_a_port_draw_take_no_held_port(monkeypatch):
    with held_ports(1, lambda ports: [ports[0] + 1]) as [port]:
        squat = subprocess.run(
            [sys.executable, "-c", SQUAT, str(port), str(port + 1)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert squat.stdout.split() == [str(errno.EADDRINUSE)] * 4
        monkeypatch.setattr(serve, "SERVE_PORTS", range(port, port + 2))
        with (
            pytest.raises(serve.ServeError, match="fewer than 1 free ports"),
            serve._held_ports(1),
        ):
            pass


def _zmq(stack, port):
    sock = zmq.Context.instance().socket(zmq.ROUTER)
    stack.callback(sock.close, linger=0)
    sock.bind(f"tcp://127.0.0.1:{port}")


def _clock(stack, port):
    table = single_engine_table(admission_path="serving", ipc_s=1e-3, stream_s=2e-3)
    endpoint = f"tcp://127.0.0.1:{port}"
    stack.callback(clock_transport.serve(ClockAuthority(table), endpoint).close)


def _tcp_store(stack, port):
    store = TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False)
    stack.callback(lambda: store)  # keeps it serving until the stack closes


def _uvicorn(stack, port):
    server = subprocess.Popen([sys.executable, "-c", UVICORN, str(port)])
    stack.callback(server.wait)
    stack.callback(server.kill)
    deadline = time.monotonic() + 60
    while not _connects(port):
        assert server.poll() is None, f"uvicorn exited {server.returncode}"
        assert time.monotonic() < deadline, "uvicorn did not listen in 60 s"
        time.sleep(0.1)


@pytest.mark.parametrize(
    "listen", [_zmq, _clock, _tcp_store, _uvicorn], ids=lambda f: f.__name__[1:]
)
def test_each_server_listener_binds_a_held_port(listen):
    with held_ports(1) as [port], contextlib.ExitStack() as stack:
        listen(stack, port)
        assert _connects(port)
