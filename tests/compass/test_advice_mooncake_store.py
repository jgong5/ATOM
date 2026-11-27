# SPDX-License-Identifier: MIT
"""A 1P1D candidate whose prefill offloads to a Mooncake Store, from its space
file to LP mooncake_store's launch: the Store's entry its blocks name
(`space.expand`), each point's `deploy.sh run` taking that file
(`check.serving`), the run file `deploy.sh` builds from it
(`runfile.prefill_decode`), and the launch that reads it
(`run._launch_mooncake_store`)."""

import argparse
import dataclasses
import json
import shlex

import platforms
import pytest
from test_memory_readings import DOCUMENT
from test_mooncake_store_lp import ENTRY, STORE, _config

from atom.compass import run as compass_run
from atom.compass.advice import check, space
from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import LpId
from atom.compass.drive import runfile
from atom.compass.drive.run import SIMULATED, side_args
from atom.model_engine.arg_utils import (
    compose_kv_offload_config,
    kv_offload_connector_config,
)

#: The prefill side's variables for the Store, as a recipe gives them.
OFFLOAD = {
    "ATOM_KV_OFFLOAD": "mooncake_store",
    check.EXTRA: json.dumps(STORE["kv_connector_extra_config"]),
}
#: Each side's laws, which a point's command takes.
VERDICTS = {
    "law": {s: {"run": [f"{s}.json", f"{s}-host.json"]} for s in ("prefill", "decode")}
}


def a_space(entry, **on) -> dict:
    """1P1D at two concurrencies, its prefill offloading to the Store whose
    entry is at `entry`, or not; the offloading value's block updated with
    `on`."""
    offloading = {"prefill": {"env": OFFLOAD}, "mooncake_store": entry} | on
    return {
        "model": "/m",
        "machine": {"device": "mi355x", "ptl": "disabled", "network": "a-network"},
        "store": "/s",
        "workload": {"traces": "/t.jsonl", "select": "agentx", "concurrency": [1, 4]},
        "slo": {},
        "common": {"basis": "a recipe", "deployment": "1p1d"},
        "dimensions": {"kv": {"off": {}, "on": offloading}},
    }


def expand(tmp_path, given: dict) -> dict:
    path = tmp_path / "space.json"
    path.write_text(json.dumps(given))
    return space.expand(path)


def served(expansion: dict, name: str, at) -> tuple[list[str], dict]:
    """The command and variables `check.serving` runs point `name` under."""
    point = next(p for p in expansion["points"] if p["name"] == name)
    candidate = next(
        c for c in expansion["candidates"] if c["name"] == point["candidate"]
    )
    command, env = check.serving(
        expansion["machine"], candidate, point, VERDICTS, "/s", at
    )
    return list(map(str, command)), env


def test_each_point_of_a_candidate_naming_an_entry_runs_with_that_file(tmp_path):
    entry = str(tmp_path / "store.json")
    expansion = expand(tmp_path, a_space(entry))

    named = {c["name"]: c.get("mooncake_store") for c in expansion["candidates"]}
    assert named == {"off": None, "on": entry}
    for point in expansion["points"]:
        command, _ = served(expansion, point["name"], tmp_path / point["name"])
        flags = command[: command.index("--prefill")]
        given = [flags[i + 1] for i, w in enumerate(flags) if w == "--mooncake-store"]
        # Points run at once share the file: each deployment's Store draws its port.
        assert given == ([entry] if point["candidate"] == "on" else [])
        if point["name"] == "on-c1":
            print(shlex.join(command))


@pytest.mark.parametrize(
    "common, on, named",
    [
        ({}, {"mooncake_store": "store.json"}, "on.mooncake_store takes its entry"),
        ({}, {"mooncake_store": 1}, "on.mooncake_store takes its entry"),
        (
            {"mooncake_store": "/a.json"},
            {"mooncake_store": "/b.json"},
            "mooncake_store /a.json in common, but mooncake_store /b.json in dim",
        ),
        (
            {},
            {"prefill": {"env": OFFLOAD, "mooncake_store": "/a.json"}},
            "on.prefill: unknown field 'mooncake_store'",
        ),
    ],
    ids=["relative", "not-a-path", "set-apart", "on-a-side"],
)
def test_an_entry_of_the_wrong_kind_or_set_apart_is_refused_by_name(
    tmp_path, common, on, named
):
    given = a_space("/a.json", **on)
    given["common"] |= common

    with pytest.raises(ValueError) as refused:
        expand(tmp_path, given)

    assert named in str(refused.value)


def test_the_run_file_holds_the_entry_and_lp_mooncake_store_launches_on_it(
    tmp_path, monkeypatch
):
    entry = tmp_path / "store.json"
    entry.write_text(json.dumps(ENTRY))
    expansion = expand(tmp_path, a_space(str(entry)) | {"model": str(tmp_path)})
    command, env = served(expansion, "on-c4", tmp_path / "at")
    named = command.index("--mooncake-store")
    prefill = command[command.index("--prefill") + 2 : command.index("--decode")]
    # --decode --on PREFIX -- <ATOM args>
    decode = command[command.index("--decode") + 4 : command.index("--router")]
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    store = tmp_path / "store"
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    args = parser.parse_args(
        ["--step-stub", str(stub), *platforms.flags(store, DOCUMENT, tmp_path)]
        + ["--store", str(store), *command[named : named + 2]]
    )

    run = runfile.prefill_decode(args, prefill, decode, "intra_node")

    assert run["mooncake_store"] == ENTRY
    started = []
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda a, **k: started.append(a)
    )
    # Each side as ATOM reads it: deploy.sh's KV config, the prefill's offload.
    offload = kv_offload_connector_config(env["ATOM_KV_OFFLOAD"], env[check.EXTRA])
    for side, atom_args, more in (
        ("prefill", prefill, offload),
        ("decode", decode, None),
    ):
        kv = compose_kv_offload_config(side_args(side, atom_args, SIMULATED)[-1], more)
        config = _config(json.loads(kv), kv_cache_block_size=256)
        compass_run._launch_mooncake_store(config, run, side[0].upper())
    [argv] = started
    assert argv[-2:] == [compass_run.MOONCAKE_STORE_MAIN, "256"]
    into = compass_run.channel_table(run).channels_into(LpId("mooncake_store"))
    assert {c.source.name for c in into} == {"engine-P"}
