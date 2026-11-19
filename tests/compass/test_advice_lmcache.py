# SPDX-License-Identifier: MIT
"""A candidate offloading to an lmcache server, from its space file to LP
lmcache's launch: its server's flags expanded (`space.expand`), each point's
own port and run-file entry (`check.serving`), the run file `compass run`
builds from ``--lmcache`` (`runfile.build`), and the launch that takes it
(`run._launch_lmcache`)."""

import argparse
import dataclasses
import json
from pathlib import Path

import platforms
import pytest
from test_lmcache_lp import _config, _run

from atom.compass import run as compass_run
from atom.compass.advice import check, space
from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import LpId
from atom.compass.drive import runfile
from atom.compass.kv.lmcache_server import DECLARED_LAWS, parse_server_flags
from atom.model_engine.arg_utils import kv_offload_connector_config

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "scripts/compass/spaces/deepseek-v4-pro-mi355x-crs005.json"
SERVER = {
    "--chunk-size": 256,
    "--l1-size-gb": 8,
    "--eviction-policy": "LRU",
    "--separate-object-groups": True,
}
FLAGS = [
    "--chunk-size",
    "256",
    "--l1-size-gb",
    "8",
    "--eviction-policy",
    "LRU",
    "--separate-object-groups",
]
OFFLOAD = {"ATOM_KV_OFFLOAD": "lmcache_mp"}
#: The laws each instance's verdicts name, which a point's command takes.
VERDICTS = {"law": {"server": {"run": ["law.json", "host.json"]}}}


def a_space(**on) -> dict:
    """Two TP widths, offloading or not, at two concurrencies; the
    offloading value's block updated with `on`."""
    return {
        "model": "/m",
        "machine": {"device": "mi355x", "ptl": "disabled"},
        "store": "/s",
        "workload": {"traces": "/t.jsonl", "select": "agentx", "concurrency": [1, 4]},
        "slo": {},
        "common": {"basis": "a recipe"},
        "dimensions": {
            "tp": {
                "tp4": {"args": {"--tensor-parallel-size": 4}},
                "tp8": {"args": {"--tensor-parallel-size": 8}},
            },
            "lm": {"off": {}, "on": {"env": OFFLOAD, "lmcache_server": SERVER} | on},
        },
    }


def expand(tmp_path, given: dict) -> dict:
    path = tmp_path / "space.json"
    path.write_text(json.dumps(given))
    return space.expand(path)


def served(expansion: dict, name: str, at: Path) -> tuple[list, dict]:
    """The command and variables `check.serving` runs point `name` under."""
    point = next(p for p in expansion["points"] if p["name"] == name)
    candidate = next(
        c for c in expansion["candidates"] if c["name"] == point["candidate"]
    )
    return check.serving(expansion["machine"], candidate, point, VERDICTS, "/s", at)


def test_each_point_of_an_offloading_candidate_takes_its_own_port(tmp_path):
    expansion = expand(tmp_path, a_space())

    flags = {c["name"]: c.get("lmcache_server") for c in expansion["candidates"]}
    assert flags == {"tp4-off": None, "tp4-on": FLAGS, "tp8-off": None, "tp8-on": FLAGS}
    ports = {p["name"]: p.get("lmcache_port") for p in expansion["points"]}
    assert ports == {
        "tp4-off-c1": None,
        "tp4-off-c4": None,
        "tp4-on-c1": 5555,
        "tp4-on-c4": 5556,
        "tp8-off-c1": None,
        "tp8-off-c4": None,
        "tp8-on-c1": 5557,
        "tp8-on-c4": 5558,
    }
    # The port is the point's, not its instances': points alike start once.
    on = [p["instances"] for p in expansion["points"] if p["candidate"] == "tp4-on"]
    assert on[0] == on[1]


def test_server_flags_join_across_blocks_in_the_blocks_order(tmp_path):
    given = a_space(lmcache_server={"--l1-size-gb": 8})
    given["common"]["lmcache_server"] = {"--chunk-size": 256}

    expansion = expand(tmp_path, given)

    assert expansion["candidates"][1]["lmcache_server"] == FLAGS[:4]


@pytest.mark.parametrize(
    "common, on, named",
    [
        ({}, {"--host": "0.0.0.0"}, "on.lmcache_server: --host is compass advice's"),
        ({}, {"--port": 5555}, "on.lmcache_server: --port is compass advice's"),
        ({}, {"-p": 1}, "lmcache_server: '-p' takes a long flag"),
        ({}, {"--chunk-size": [256]}, "lmcache_server.--chunk-size takes a string"),
        (
            {"--l1-size-gb": 8},
            {"--l1-size-gb": 16},
            "--l1-size-gb 8 in common.lmcache_server, but --l1-size-gb 16 in dim",
        ),
        (
            {"--l1-use-lazy": True},
            {"--no-l1-use-lazy": True},
            "--l1-use-lazy in common.lmcache_server, but --no-l1-use-lazy in",
        ),
    ],
)
def test_the_server_s_address_or_a_flag_set_apart_is_refused_by_name(
    tmp_path, common, on, named
):
    given = a_space(lmcache_server=on)
    given["common"]["lmcache_server"] = common

    with pytest.raises(ValueError) as refused:
        expand(tmp_path, given)

    assert named in str(refused.value)


def test_a_side_block_holds_no_server_flags(tmp_path):
    given = a_space()
    given["dimensions"]["lm"]["on"] = {
        "deployment": "1p1d",
        "prefill": {"lmcache_server": SERVER},
    }

    with pytest.raises(ValueError, match="on.prefill: unknown field 'lmcache_server'"):
        expand(tmp_path, given)


def test_more_offloading_points_than_ports_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(space, "LMCACHE_PORTS", range(5555, 5558))

    with pytest.raises(ValueError, match="4 points offload to an lmcache server"):
        expand(tmp_path, a_space())


def test_each_template_s5_point_reaches_its_own_server(tmp_path):
    expansion = space.expand(TEMPLATE)
    s5 = next(c for c in expansion["candidates"] if c["name"] == "S5")
    held = []

    for name in ("S5-c1", "S5-c2"):
        at = tmp_path / "runs" / name
        command, env = served(expansion, name, at)
        entry = Path(command[command.index("--lmcache") + 1])
        held.append((entry, json.loads(entry.read_text()), env))

    ports = []
    for (entry, written, env), name in zip(held, ("S5-c1", "S5-c2")):
        port = int(written["server"][-1])
        ports.append(port)
        assert entry == tmp_path / "runs" / f"{name}.lmcache.json"
        assert written == {
            "server": [
                *s5["lmcache_server"],
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            **DECLARED_LAWS,
        }
        # The stand-in takes every flag of the recipe's server.
        parsed = parse_server_flags(written["server"])
        assert (parsed.host, parsed.port, parsed.l1_size_gb) == (
            "127.0.0.1",
            port,
            1200,
        )
        reached = json.loads(env[check.EXTRA])
        assert reached == {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": port,
        }
    assert ports == [5555, 5556]


@pytest.mark.parametrize(
    "given",
    ['{"lmcache.mp.port": 5555}', '{"lmcache.mp.host": "tcp://h"}', "[]", "{"],
)
def test_a_variable_naming_the_server_s_address_or_no_object_is_refused(
    tmp_path, given
):
    expansion = expand(tmp_path, a_space(env=OFFLOAD | {check.EXTRA: given}))

    with pytest.raises(ValueError, match="takes a JSON object naming neither"):
        served(expansion, "tp4-on-c1", tmp_path / "at")


def test_the_variable_keeps_its_other_keys(tmp_path):
    other = {"lmcache.mp.tp_rank_collapse": True}
    expansion = expand(
        tmp_path, a_space(env=OFFLOAD | {check.EXTRA: json.dumps(other)})
    )

    _, env = served(expansion, "tp8-on-c4", tmp_path / "at")

    reached = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": 5558}
    assert json.loads(env[check.EXTRA]) == other | reached


def test_a_1p1d_start_reaches_the_server_from_its_offloading_side_alone(tmp_path):
    given = a_space()
    given["machine"]["network"] = "a-network"
    given["dimensions"]["lm"]["on"] = {
        "deployment": "1p1d",
        "prefill": {"env": OFFLOAD},
        "lmcache_server": SERVER,
    }
    expansion = expand(tmp_path, given)

    command, env = served(expansion, "tp4-on-c1", tmp_path / "at")

    flags = command[: command.index("--prefill")]
    assert flags[flags.index("--lmcache") + 1] == tmp_path / "at.lmcache.json"
    assert json.loads(env[check.EXTRA])["lmcache.mp.port"] == 5555
    # ATOM refuses the variable with no offload, so the decode side drops it.
    on = command[command.index("--on") + 1]
    assert on == f"env -u ATOM_KV_OFFLOAD -u {check.EXTRA}"
    assert command[command.index("--router") :] == ["--router", "--", *check.ROUTER]


@pytest.mark.parametrize("dp_attention", [True, False], ids=["dpa", "tp8"])
def test_compass_run_takes_the_entry_and_lp_lmcache_launches_on_it(
    tmp_path, monkeypatch, dp_attention
):
    """At TP8, under DP attention or with every TP rank a writer: the entry
    names no writer count, and neither does the channel table it builds."""
    expansion = expand(tmp_path, a_space())
    command, env = served(expansion, "tp8-on-c4", tmp_path / "at")
    entry = command[command.index("--lmcache") + 1]
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    store = tmp_path / "store"
    # A machine whose host link prices LP lmcache's copies.
    platform = platforms.flags(store, _run(tmp_path)["machine"], tmp_path)
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    args = parser.parse_args(
        ["--step-stub", str(stub), *platform, "--store", str(store)]
        + ["--lmcache", str(entry)]
    )

    run = runfile.build(args, ["--model", str(tmp_path)])

    assert run["lmcache"] == json.loads(Path(entry).read_text())
    started = []
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda a, **k: started.append(a)
    )
    # ATOM's own reading of the variables the point runs under.
    kv = kv_offload_connector_config(env["ATOM_KV_OFFLOAD"], env[check.EXTRA])
    config = _config(kv, tensor_parallel_size=8, enable_dp_attention=dp_attention)
    compass_run._launch_lmcache(config, run, None)
    [argv] = started
    assert argv[-1] == str(config.kv_cache_block_size)
    lm = compass_run.channel_table(run).channels_into(LpId("lmcache"))
    assert {c.name for c in lm} == {
        "engine->lmcache:lookup#dp0",
        "engine->lmcache:transfer#dp0",
    }
    stations = compass_run._lmcache_stations(run)
    assert stations["lookup_s"] == tuple(DECLARED_LAWS["lookup_s"])
    assert stations["request_s"] == DECLARED_LAWS["request_s"]
    lp = parse_server_flags(run["lmcache"]["server"])
    assert (lp.host, lp.port) == ("127.0.0.1", 5558)
    assert kv["kv_connector_extra_config"]["lmcache.mp.port"] == lp.port
