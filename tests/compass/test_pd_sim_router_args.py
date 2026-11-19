# SPDX-License-Identifier: MIT
"""`pd_sim.sh` launches the router on `ROUTER_ARGS` as given, as `deploy.sh`
takes a recipe's, adding the wall-clock bounds a simulated router gets, and
refuses before anything starts what `deploy.sh` refuses.

`python3` is stubbed as in `test_pd_sim_sides`; the router binary writes its
arguments and exits 1, so the script exits 1 once both servers have answered.
Nothing needs a driver.
"""

import json
import os
import subprocess

import pytest
from test_pd_sim_sides import PYTHON3, TREE

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from tests.compass.held_ports import held_ports

#: The 6P16D C1228 share-cache recipe's router flags, less those `pd_sim.sh`
#: gives (host, ports, workers) or sets itself (its request and prepare
#: deadlines), and with the wall-clock detectors off, as a simulated router
#: needs them; `--dp-aware` and `--model-path` aside, which a run adds.
RECIPE = "--prefill-policy cache_aware --decode-policy cache_aware"
RECIPE += " --cache-threshold 1.0 --balance-abs-threshold 0 --balance-rel-threshold 1.0"
RECIPE += " --atom-pd-rank-mapping-policy none --backend atom"
RECIPE += " --disable-circuit-breaker --disable-health-check --disable-retries"
YEAR_MS = str(365 * 24 * 3600 * 1000)


def _pd_sim(tmp_path, router_args: str) -> tuple[subprocess.CompletedProcess, list]:
    """`pd_sim.sh` on a 1P1D run file with `ROUTER_ARGS`, and the arguments
    the router was launched with, empty when it was not."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(PYTHON3)
    argv = tmp_path / "router.json"
    (bin_ / "atomesh").write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {argv}\nexit 1\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"router_s": 1e-5}))
    env = {k: v for k, v in os.environ.items() if k not in PERFORMANCE_KEYS}
    env.update(
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        ATOMESH=str(bin_ / "atomesh"),
        WAIT_S="60",
        ROUTER_ARGS=router_args,
    )
    with held_ports(4) as ports:
        names = ("CLOCK_PORT", "PREFILL_PORT", "DECODE_PORT", "ROUTER_PORT")
        env.update(zip(names, map(str, ports)))
        done = subprocess.run(
            ["bash", str(TREE / "scripts/compass/pd_sim.sh"), str(run)],
            env=env,
            cwd=TREE,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    return done, argv.read_text().split("\n")[:-1] if argv.exists() else []


def test_the_router_takes_a_recipe_s_arguments_as_given_and_a_year_s_bounds(
    tmp_path,
):
    done, argv = _pd_sim(tmp_path, f"{RECIPE} --model-path /m")

    assert done.returncode == 1, done.stderr
    recipe = [*RECIPE.split(), "--model-path", "/m"]
    at = argv.index(recipe[0])
    assert argv[at : at + len(recipe)] == recipe
    for flag, bound in (
        ("--request-timeout-secs", str(365 * 24 * 3600)),
        ("--prepare-queue-timeout-ms", YEAR_MS),
        ("--prepare-timeout-ms", YEAR_MS),
    ):
        assert argv.count(flag) == 1, flag
        assert argv[argv.index(flag) + 1] == bound, flag
    assert argv.count("--backend") == argv.count("--disable-retries") == 1
    # The script states the command it launched.
    [said] = [x for x in done.stdout.splitlines() if x.startswith("pd_sim: router ")]
    assert said.split()[3:] == argv


@pytest.mark.parametrize(
    "extra, refusal",
    [
        ("--prepare-queue-timeout-ms 5000", "carry --prepare-queue-timeout-ms,"),
        ("--prepare-timeout-ms=15000", "carry --prepare-timeout-ms,"),
        ("--request-timeout-secs 1800", "carry --request-timeout-secs,"),
        ("--prefill http://127.0.0.1:9", "carry --prefill,"),
    ],
    ids=lambda v: v.split()[0].split("=")[0] if v.startswith("--") else "",
)
def test_router_args_deploy_sh_refuses_exit_2_before_anything_starts(
    tmp_path, extra, refusal
):
    done, argv = _pd_sim(tmp_path, f"{RECIPE} {extra}")

    assert done.returncode == 2, done.stderr
    assert f"pd_sim: the router's arguments {refusal}" in done.stderr
    assert argv == [] and not list(tmp_path.glob("*.log"))


def test_a_dp_aware_router_without_disable_retries_exits_2(tmp_path):
    done, _ = _pd_sim(tmp_path, "--dp-aware --policy random")

    assert done.returncode == 2, done.stderr
    assert "carry --dp-aware but not --disable-retries:" in done.stderr
