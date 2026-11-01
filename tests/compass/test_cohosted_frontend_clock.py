# SPDX-License-Identifier: MIT
"""A frontend that co-hosts the clock authority reaches it in-process.

`compass run` serves the authority in the API server at the run file's
``tcp://`` endpoint, which the engine and the client reach over a socket. The
frontend LP is in the serving process: its connection holds no socket, and its
`fileno`, which the event loop polls for its grant, is a pipe.
"""

import json
import os
import stat

import pytest
from held_ports import held_ports
from test_dp_slice import _deployment

from atom.compass import run as compass_run
from atom.compass.clock_transport import stream
from atom.model_engine.engine_core_mgr import CoreManager
from atom.utils import clock
from atom.utils.distributed import utils


@pytest.fixture
def tcp_run(monkeypatch, tmp_path):
    with held_ports(1) as (port,):
        run = {
            "admission_path": "serving",
            "ipc_s": 0.001,
            "stream_s": 0.002,
            "bound_s": 10.0,
            "clock_endpoint": f"tcp://127.0.0.1:{port}",
            "out_dir": str(tmp_path),
        }
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        monkeypatch.delenv(compass_run.CLOCK_ENV, raising=False)
        for module, name in [
            (compass_run, "_authority"),
            (compass_run, "_member"),
            (clock, "_installed"),
            (utils, "LP_OF_RANK"),
        ]:
            monkeypatch.setattr(module, name, None)
        # The frontend wraps this method for the process; put ATOM's back after.
        monkeypatch.setattr(
            CoreManager,
            "broadcast_utility_command_sync",
            CoreManager.broadcast_utility_command_sync,
        )
        yield run
        # End the co-hosted run, so neither the early-end poller nor the
        # authority's server outlives the test.
        if compass_run._authority is not None:
            compass_run._authority.done.set()
            compass_run._early_end.join()
            stream._SERVED[run["clock_endpoint"]].close()


def test_a_frontend_cohosting_a_tcp_authority_holds_no_socket_to_it(tcp_run):
    with compass_run.frontend(_deployment()):
        pass
    rt = clock.installed()
    assert rt.me == compass_run.FRONTEND
    assert not stat.S_ISSOCK(os.fstat(rt.conn.fileno()).st_mode)
    assert stat.S_ISFIFO(os.fstat(rt.conn.fileno()).st_mode)
