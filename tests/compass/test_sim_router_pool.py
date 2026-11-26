# SPDX-License-Identifier: MIT
"""COMPASS_SIM_ROUTER_LOG (scripts/compass/_lib.sh): a simulated router runs the request-preparation pool its
recorded router's log states (#3720). Unset, COMPASS_SIM_ROUTER_BOUNDS is as before, word for word. The library is
sourced by bash alone; no router runs."""

import os
import pathlib
import subprocess

TREE = pathlib.Path(__file__).resolve().parents[2]
LIB = TREE / "scripts/compass/_lib.sh"
#: The recorded router's line as atomesh's tracing writes it, its ANSI colours in.
POOL = (
    "\x1b[2m2026-11-26 05:04:06\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mmesh::server\x1b[0m\x1b[2m:\x1b[0m "
    "Request preparation pool started \x1b[3mworkers\x1b[0m\x1b[2m=\x1b[0m64 "
    "\x1b[3mtokio_workers\x1b[0m\x1b[2m=\x1b[0m64 \x1b[3mqueue_capacity\x1b[0m\x1b[2m=\x1b[0m640 "
    "\x1b[3mmax_tokenize_bytes\x1b[0m\x1b[2m=\x1b[0m1048576\n"
)
YEAR = [
    "--request-timeout-secs",
    "31536000",
    "--prepare-queue-timeout-ms",
    "31536000000",
    "--prepare-timeout-ms",
    "31536000000",
]


def _bash(script: str, log=None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "COMPASS_SIM_ROUTER_LOG"}
    if log is not None:
        env["COMPASS_SIM_ROUTER_LOG"] = str(log)
    return subprocess.run(
        ["bash", "-c", f"source {LIB}; {script}"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _bounds(log=None) -> list[str]:
    return _bash('printf "%s\\n" "${COMPASS_SIM_ROUTER_BOUNDS[@]}"', log).stdout.split()


def test_unset_the_bounds_are_as_before():
    assert _bounds() == YEAR


def test_a_recorded_log_sets_its_prepare_pool(tmp_path):
    log = tmp_path / "router.log"
    log.write_text(
        "starting\n" + POOL.replace("=\x1b[0m64 ", "=\x1b[0m8 ", 1) + "more\n" + POOL
    )
    assert _bounds(log) == YEAR + [
        "--prepare-workers",
        "64",
        "--prepare-queue-capacity",
        "640",
        "--prepare-max-tokenize-bytes",
        "1048576",
    ]


def test_a_log_with_no_pool_line_adds_nothing_and_says_so(tmp_path):
    log = tmp_path / "router.log"
    log.write_text("no pool here\n")
    got = _bash('printf "%s\\n" "${COMPASS_SIM_ROUTER_BOUNDS[@]}"', log)
    assert got.stdout.split() == YEAR
    assert "states no request-preparation pool" in got.stderr


def test_a_callers_own_pool_flag_is_refused_only_under_a_recorded_log(tmp_path):
    log = tmp_path / "router.log"
    log.write_text(POOL)
    refusal = "compass_sim_router_refusal 0 --policy random --prepare-workers 8"
    assert "the recorded router's prepare pool" in _bash(refusal, log).stdout
    assert _bash(refusal).stdout == ""
