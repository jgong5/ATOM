# SPDX-License-Identifier: MIT
"""The pit2 compass-advice scripts' pure parts (scripts/compass/pit2): a
printed calibration command localized to a node's Store, its directory and its
variables; a finished point's recorded run under a send-order seed, and at
another concurrency, each into its own directory on the tree given."""

import importlib.util
import json
import shlex
from pathlib import Path

import pytest

PIT2 = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "pit2"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"pit2_{name}", PIT2 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


printed, seed_sims, rerun_points = map(_load, ("printed", "seed_sims", "rerun_points"))

OFFLOAD = json.dumps(
    {
        "mooncake_store.master": "10.19.0.127:26051",
        "mooncake_store.metadata": "http://10.19.0.127:26080/metadata",
        "mooncake_store.local_hostname": "10.19.0.127",
    }
)
CMD = (
    "AITER_LOG_LEVEL=WARNING MODEL=/m PREFILL_GPU=0,1,2,3 "
    + shlex.quote(
        f"PREFILL_ENV=ATOM_KV_OFFLOAD=mooncake_store ATOM_KV_OFFLOAD_EXTRA_CONFIG={OFFLOAD}"
    )
    + " ROUTER_ARGS='--backend atom' /t/scripts/compass/calibrate.sh /d/cal/A '1 --x 0' 'trace 16'"
)


def _saved(tmp_path, text):
    path = tmp_path / "A.cmd"
    path.write_text(text)
    return str(path)


def test_a_printed_command_moves_its_store_to_the_node_and_names_its_directory(
    tmp_path,
):
    path = _saved(tmp_path, CMD + "\n  expected time: unknown\n")
    cmd = printed.command(path)
    assert cmd == CMD, "the indented notes are not the command"
    here = printed.localize(cmd, "10.19.0.152")
    assert "10.19.0.127" not in here and here.count("10.19.0.152") == 3
    assert printed.split(cmd)[1] == "/d/cal/A"
    words, _ = printed.split(here)
    assert all("=" in w for w in words) and words[-1] == "ROUTER_ARGS=--backend atom"
    assert (
        printed.localize("A=1 /t/calibrate.sh /o", "1.2.3.4")
        == "A=1 /t/calibrate.sh /o"
    )


def test_a_command_with_no_calibrate_sh_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no calibrate.sh OUT"):
        printed.split("A=1 /t/deploy.sh /o")
    with pytest.raises(SystemExit, match="no command"):
        printed.command(_saved(tmp_path, "\n   indented only\n"))


def _point(tmp_path, name="A-c64"):
    point = tmp_path / "search" / "runs" / name
    point.mkdir(parents=True)
    ran = {
        "tree": {"revision": "old", "dirty": 0},
        "server": ["/t/deploy.sh", str(point), "run", "--prefill"],
        "env": {"X": "1"},
        "client": [
            "/t/client.sh",
            str(point),
            "--agentx",
            "--concurrency",
            "64",
            "--stop",
        ],
        "client_env": {"MODEL": "/m", "TRACES": "/tr"},
    }
    (point / "point.json").write_text(json.dumps({"ran": ran, "state": "finished"}))
    return point, ran


def test_a_seeded_run_is_the_point_s_own_in_its_directory_on_the_tree_given(tmp_path):
    point, ran = _point(tmp_path)
    out = tmp_path / "seeds"
    p = seed_sims.plan(point, 1, out, tree=("new", 0))
    at = out / "runs" / "A-c64-s1"
    assert (p["name"], p["at"]) == ("A-c64-s1", at)
    assert p["ran"]["server"] == ["/t/deploy.sh", str(at), "run", "--prefill"]
    assert p["ran"]["client"][1] == str(at) and p["ran"]["client"][4] == "64"
    assert p["ran"]["client_env"] == {
        **ran["client_env"],
        "COMPASS_HARNESS_SEND_ORDER_SEED": "1",
    }
    assert p["ran"]["env"] == ran["env"]
    assert p["ran"]["tree"] == {
        "revision": "new",
        "dirty": 0,
    }, "read afresh, not copied"


def test_a_rerun_takes_its_concurrency_and_a_point_json_of_another_directory_is_refused(
    tmp_path,
):
    point, _ = _point(tmp_path)
    assert rerun_points.parse(f"A-c256={point}:256") == ("A-c256", point, 256)
    p = rerun_points.plan("A-c256", point, 256, tmp_path / "fix", tree=("new", 0))
    assert p["ran"]["client"][4] == "256" and p["ran"]["client"][1] == str(p["at"])
    assert "COMPASS_HARNESS_SEND_ORDER_SEED" not in p["ran"]["client_env"]
    with pytest.raises(ValueError, match="NAME=BASE_POINT_DIR:C"):
        rerun_points.parse(f"{point}:256")
    moved = json.loads((point / "point.json").read_text())
    moved["ran"]["server"][1] = "/elsewhere"
    (point / "point.json").write_text(json.dumps(moved))
    for make in (lambda: seed_sims.plan(point, 0, tmp_path, ("t", 0)),
                 lambda: rerun_points.plan("x", point, 8, tmp_path, ("t", 0))):  # fmt: skip
        with pytest.raises(ValueError, match="runs another directory"):
            make()
