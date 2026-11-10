# SPDX-License-Identifier: MIT
"""client.sh --stop gives a trace run's command serve's TRACE_STOP_S more than
WAIT_S to leave: it writes every rank's trace and reduces them first, about
11 min for a 120 s V4.1-Flash TP4 trace run that keeps every profiler row
(#3106)."""

import subprocess
import sys
import time

import pytest
from test_client_sh import COMMAND, GOOD, _client, _setup

#: The stand-in compass command, leaving 3 s after SIGTERM.
SLOW = COMMAND.replace("import json,", "import time, json,").replace(
    "    server.kill()\n", "    server.kill()\n    time.sleep(3)\n"
)


@pytest.mark.parametrize("trace", [True, False])
def test_a_trace_runs_command_gets_longer_to_leave_than_wait_s(tmp_path, trace):
    """With WAIT_S at 1 s, the command leaving 3 s after SIGTERM is waited for
    when its OUT holds a trace run's `trace` directory, and refused when not."""
    assert SLOW.count("time.sleep(3)") == 1 and "import time" in SLOW
    out, env = _setup(tmp_path)
    if trace:
        (out / "trace").mkdir()
    (tmp_path / "compass_command.py").write_text(SLOW)
    command = subprocess.Popen([sys.executable, tmp_path / "compass_command.py", out])
    while not (out / "server.json").exists():
        assert command.poll() is None
        time.sleep(0.1)
    done = _client(out, *GOOD, "--stop", env=env | {"WAIT_S": "1"})
    assert command.wait(timeout=30) == 0
    if trace:
        assert done.returncode == 0, done.stdout + done.stderr
        assert "client: stopped the server" in done.stdout
    else:
        assert done.returncode == 1
        assert "is up 1 s after SIGTERM" in done.stderr
