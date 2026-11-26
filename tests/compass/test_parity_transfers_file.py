# SPDX-License-Identifier: MIT
"""A decode instance's transfers file (`parity.TRANSFERS`), written through
`carry_response_ids` over ATOM's own core manager: opened once a process, it
holds the bytes the writer that reopened it for each line wrote."""

import json
import os
import signal

import pytest

from atom.compass import parity
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.sequence import Sequence

BLOCK = 16
LINES = [
    {"response_id": "chatcmpl-0", "remote_dp_rank": 2, "transfer_id": 40},
    {"response_id": "chatcmpl-1", "remote_dp_rank": 0, "transfer_id": 41},
    {
        "response_id": "chatcmpl-2",
        "remote_dp_rank": 1,
        "transfer_id": 40,
        "remote_engine_id": "compass-simulated-p1",
    },
]


@pytest.fixture
def send(monkeypatch, tmp_path):
    """Sends `LINES[i]`'s request as the API server hands it to the engine."""
    for cls in (CoreManager, DisaggCoreManager):  # ATOM's back after the test
        monkeypatch.setattr(cls, "add_request", cls.add_request)
    monkeypatch.setenv(parity.ENV, str(tmp_path / "decode"))
    parity.carry_response_ids()
    manager = CoreManager.__new__(CoreManager)
    manager.label, manager.pp_size, manager.local_engine_count = "t", 1, 1
    manager._seq_id_to_callback = {}
    manager._send_request = lambda _rank, payload: None

    def send(i):
        params = {k: v for k, v in LINES[i].items() if k != "response_id"}
        if params["remote_dp_rank"] == 0:
            del params["remote_dp_rank"]  # a transfer naming no prefill rank
        rid = LINES[i]["response_id"]
        manager.add_request(
            [Sequence([1, 2], BLOCK, kv_transfer_params=params, request_id=rid)]
        )
        manager.add_request([Sequence([4], BLOCK)])  # no transfer, no line

    return send


def _reopened(lines, path) -> bytes:
    """`lines` as the writer that reopened the file for each one wrote them."""
    for line in lines:
        with path.open("a") as f:
            f.write(json.dumps(line) + "\n")
    return path.read_bytes()


def _descriptors(path) -> list[int]:
    """This process's descriptors open on `path`."""
    key = os.stat(path).st_dev, os.stat(path).st_ino
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            st = os.stat(f"/proc/self/fd/{fd}")
        except OSError:  # the listing's own descriptor, closed by now
            continue
        if (st.st_dev, st.st_ino) == key:
            found.append(int(fd))
    return found


def test_a_process_keeps_one_file_holding_each_line_once_its_request_is_sent(
    send, tmp_path
):
    path = tmp_path / "decode" / parity.TRANSFERS
    for i in range(len(LINES)):
        send(i)
        assert path.read_bytes() == _reopened(LINES[: i + 1], tmp_path / f"r{i}")
        assert len(_descriptors(path)) == 1


def test_a_forked_child_writes_through_a_file_of_its_own(send, tmp_path):
    path = tmp_path / "decode" / parity.TRANSFERS
    send(0)
    child = os.fork()
    if child == 0:
        try:
            # Its parent's file closed here, a line written through it fails.
            for fd in _descriptors(path):
                os.close(fd)
            send(1)
            os.kill(os.getpid(), signal.SIGKILL)  # a crash: nothing closed
        finally:
            os._exit(1)
    _, status = os.waitpid(child, 0)
    send(2)

    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
    assert path.read_bytes() == _reopened(LINES, tmp_path / "reopened")
