# SPDX-License-Identifier: MIT
"""The grant-cost probe drives a real clock authority over loopback TCP."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/compass/ca_grant_cost.py"


def test_three_lps_are_granted_every_round_and_finish_the_run():
    server = subprocess.Popen(
        [sys.executable, SCRIPT, "serve", "3", "tcp://127.0.0.1:0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        endpoint = server.stdout.readline().strip()
        lps = subprocess.run(
            [sys.executable, SCRIPT, "lps", "3", endpoint, "--rounds", "20"],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        served = json.loads(server.communicate(timeout=120)[0])
    finally:
        server.kill()
    rtt = json.loads(lps.stdout)["rtt_s"]
    assert 0 < rtt["p10"] <= rtt["p50"] <= rtt["p90"]
    # Each LP's 20 TARs and its NER(+inf), all served before the finish.
    assert served["requests"] == 3 * 21
    assert served["cpu_per_request_s"] > 0
