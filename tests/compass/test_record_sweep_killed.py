# SPDX-License-Identifier: MIT
"""A record's lines on disk after its server's worker dies by SIGKILL mid-run.

`compass record` starts `test_drive_record.py`'s planted server, here also the
worker: it answers ``/health`` once, runs `test_parity_device_window.py`'s
planted real runner over the batches ATOM's scheduler builds for four prompts,
recording where the server's environment points, and dies by SIGKILL with its
last forward's device window still open, so it never reaches `exit`.
"""

import json
import os
import signal
from pathlib import Path

import pytest
from test_drive_identity import GIT
from test_drive_record import planted_gpu  # noqa: F401
from test_parity_device_window import _recorded
from test_parity_record import _config, _drive

from atom.compass import cli
from atom.compass.parity import WRITE_THROUGH

HERE = Path(__file__).resolve().parent
KILLED = """\
import http.server, os, sys
sys.path[:0] = os.environ["PLANTED_PATH"].split(os.pathsep)
from test_record_sweep_killed import forward_until_killed
port = int(sys.argv[sys.argv.index("--server-port") + 1])

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

http.server.HTTPServer(("127.0.0.1", port), Health).handle_request()
forward_until_killed()
"""


def forward_until_killed():
    """Record each batch the scheduler builds for four prompts, the simulated
    record of them in ``PLANTED_SIMULATED``, then die by SIGKILL. A record that
    waits on the last forward's window, which never closes, dies by SIGALRM."""
    with pytest.MonkeyPatch.context() as patch:
        simulated = Path(os.environ["PLANTED_SIMULATED"])
        batches = _drive(simulated, patch, {0: [0, 1, 2, 3]})[0][0]
    signal.alarm(60)
    runner_class, _ = _recorded(batches[-1])
    runner = runner_class(0, _config())
    for batch in batches:
        runner.forward(batch)
    os.kill(os.getpid(), signal.SIGKILL)


@pytest.mark.skipif(GIT is None, reason="needs git")
@pytest.mark.parametrize("sweep, exported", [(True, None), (False, None), (False, "1")])
def test_a_killed_sweep_server_keeps_each_line_its_worker_finished(
    tmp_path, planted_gpu, monkeypatch, sweep, exported  # noqa: F811
):
    """The named result: a sweep's record holds every line but the last, whose
    window was open at the kill; a timing record is held to an `exit` the
    worker never reached, and holds none, as before, even when the command's
    environment exports `WRITE_THROUGH`."""
    monkeypatch.delenv(WRITE_THROUGH, raising=False)
    if exported:
        monkeypatch.setenv(WRITE_THROUGH, exported)
    (tmp_path / "server.py").write_text(KILLED)
    monkeypatch.setenv("PLANTED_PATH", f"{HERE}{os.pathsep}{HERE.parent}")
    monkeypatch.setenv("PLANTED_SIMULATED", str(tmp_path / "simulated"))
    out = tmp_path / "out"
    argv = ["record", "--gpu", "7", *(["--sweep"] if sweep else []), "--out", out]
    with pytest.raises(SystemExit, match="the server exited -9"):
        cli.main([*map(str, argv), "--", "--model", str(planted_gpu)])

    ran = (tmp_path / "simulated" / "dp0.jsonl").read_text().splitlines()
    kept = (out / "steps" / "dp0.jsonl").read_text().splitlines()
    assert len(ran) > 1
    assert [json.loads(line)["step"] for line in kept] == (
        list(range(len(ran) - 1)) if sweep else []
    )
