# SPDX-License-Identifier: MIT
"""`deploy.sh run` picks each side's host entry under the environment its
instance starts in: the prefill side's under the script's own, the decode
side's under the one its `--on` prefix gives, which `PREFIX env -0` shows.

The crs005 DPA8 pair as shipped keys its prefill entry to `ATOM_NUMA_BIND=1
GPU_MAX_HW_QUEUES=5`, which the README's local prefix strips from the decode.
Here the decode entry's hops move, so the build picks each side's entry, each
side's config resolved on the pair's gfx950 in a process of its own. The
authority, the instances and the router are `test_deploy_sh`'s planted ones.
"""

import json
import shutil
import signal
import subprocess

import platforms
import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_deploy_sh import ROUTER_ARGS, SCRIPT, _until, planted  # noqa: F401
from test_memory_readings import DOCUMENT
from test_run_check_side import PAIR, SHARED_ENV, SIDES

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.backends.json_file import read_json

#: The README's local prefix, which starts the decode without the prefill
#: side's own variables.
ON = "env -u GPU_MAX_HW_QUEUES -u ATOM_NUMA_BIND"


def pair(tmp_path, planted, on):  # noqa: F811
    """`deploy.sh run`'s command for the crs005 DPA8 pair, its decode entry's
    hops doubled and its decode started through `on`, the environment it runs
    in, the prefill side's, and each side's hops."""
    env, _ = planted
    env = {k: v for k, v in env.items() if k not in PERFORMANCE_KEYS}
    env |= SHARED_ENV | SIDES["prefill"][1]
    hops, laws = {}, []
    for side, path in PAIR.items():
        host = read_json(path)["host"]
        if side == "decode":
            host["hops"] = {k: 2 * v for k, v in host["hops"].items()}
        hops[side] = host["hops"]
        (tmp_path / f"{side}.json").write_text(json.dumps({"host": host}))
        laws += ["--host-laws", str(tmp_path / f"{side}.json")]
    store = tmp_path / "store"
    flags = ["--step-stub", str(tmp_path / "stub.json"), *laws, "--store", str(store)]
    flags += platforms.flags(store, DOCUMENT, tmp_path, host["key"])
    flags += platforms.network(DOCUMENT, tmp_path)
    model = tmp_path / "DeepSeek-V4-Pro"
    model.mkdir()
    shutil.copy(V4_PRO, model / "config.json")
    common = ["--model", str(model), "-tp", "8", "--kv_cache_dtype", "fp8"]
    common += ["--enable-dp-attention"]
    words = [
        "bash", str(SCRIPT), str(tmp_path / "d"), "run", *flags,
        "--prefill", "--", *common, *SIDES["prefill"][0],
        "--decode", "--on", on, "--", *common, *SIDES["decode"][0],
        "--router", "--", *ROUTER_ARGS,
    ]  # fmt: skip
    return words, env, hops


# A prefix that unsets AITER_LOG_LEVEL too starts a decode that compass_env
# sets it WARNING in again, as its entry's key states it.
@pytest.mark.parametrize("on", [ON, f"{ON} -u AITER_LOG_LEVEL"])
def test_each_side_of_the_crs005_pair_takes_the_hops_fitted_under_its_own_environment(
    tmp_path, planted, on  # noqa: F811
):
    words, env, hops = pair(tmp_path, planted, on)
    d = tmp_path / "d"
    script = subprocess.Popen(words, env=env, cwd=tmp_path, stdout=subprocess.PIPE)
    _until(d / "server.json", script)

    run = json.loads((d / "run_file.json").read_text())

    assert {k: run[k] for k in hops["prefill"]} == hops["prefill"]
    assert run[compass_run.DECODE_HOPS] == hops["decode"] != hops["prefill"]
    assert run["kv_link"] == "inter_node"

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out


def test_a_prefix_that_cannot_show_the_decode_s_environment_fails_naming_it(
    tmp_path, planted  # noqa: F811
):
    # The prefix runs `false env -0`, as an unreachable node's would fail.
    words, env, _ = pair(tmp_path, planted, f"{ON} false")

    done = subprocess.run(
        words,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert done.returncode == 1, done.stderr
    assert "deploy: --on's prefix exited 1 showing the decode's environment" in (
        done.stderr
    )
    assert not (tmp_path / "d" / "run_file.json").exists()
