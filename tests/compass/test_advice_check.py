# SPDX-License-Identifier: MIT
"""`compass advice SPACE OUT`'s check: each candidate gets one state, each
point missing readings the command that measures them, and each other point of
a searchable candidate a start with no traffic, all written into OUT.

The configuration is `test_advice_reuse`'s, its law the re-keyed DPA8 law as a
law file; ATOM's arguments resolve it changed by each of `CHANGED` they state.
`compass run` and `deploy.sh` are one stub that records each start and
answers, or, under `--enable-dynamic-chunking`, raises the simulated runner's
own refusal of it into its server log and leaves as `compass run` does, or,
under ``STUB_STALL``, never answers.
"""

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import platforms
import pytest
from test_advice_reuse import COVERED, law, law_file
from test_memory_readings import publish_readings
from test_run_lacking import served
from test_run_step_grid import DOCUMENT, deployment

from atom.compass import cli
from atom.compass import run as compass_run
from atom.compass.advice import check
from atom.compass.drive import reuse
from atom.compass.drive import run as drive_run

#: The ATOM arguments the stubbed resolution reads, by the config field each sets.
CHANGED = {
    "--max-num-seqs": "max_num_seqs",
    "--max-model-len": "max_model_len",
    "--kv-cache-dtype": "kv_cache_dtype",
}
STUB = """#!{python}
import json, os, signal, subprocess, sys, time
from pathlib import Path

argv = sys.argv[1:]
at = Path(argv[0] if argv[0] != "run" else argv[argv.index("--out") + 1])
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{"argv": argv, "env": dict(os.environ)}}) + "\\n")
at.mkdir(parents=True)
if "--enable-dynamic-chunking" in argv:
    refuse = (
        "from atom.compass.runner.overrides import RefusedDynamicChunking; "
        "RefusedDynamicChunking().profile_dynamic_chunking()"
    )
    with open(at / "server.log", "w") as log:
        subprocess.run([sys.executable, "-c", refuse], stderr=log, cwd={tree!r})
    sys.exit(
        "compass run: the server exited 1 before /health answered; see "
        + str(at / "server.log")
    )
signal.signal(signal.SIGTERM, lambda *_: ((at / "stopped").touch(), sys.exit(1)))
if "STUB_STALL" not in os.environ:
    (at / "server.json").write_text(json.dumps({{"url": "http://127.0.0.1:1"}}))
while True:
    time.sleep(0.1)
"""
REFUSED = (
    "atom.compass.runner.overrides.RunnerRefusal: --enable-dynamic-chunking sizes "
    "prefill chunks from timed real forwards, and a simulated runner runs none to "
    "time"
)
#: Why a 1P1D candidate on one node whose sides' variables differ is refused.
ONE_ENVIRONMENT = (
    "deploy.sh starts both sides of a deployment on one node under one "
    "environment, and the decode side's differs from the prefill side's by -u "
    "STUB_PREFILL STUB_SIDE=decode"
)
#: The network the 1P1D candidates read.
NETWORK = "crs005-013-ionic"


def resolved(config, atom_args):
    """`config` changed by each of `CHANGED` that ATOM's arguments state."""
    given = dict(itertools.pairwise(atom_args))
    changed = {
        field: type(getattr(config, field))(given[flag])
        for flag, field in CHANGED.items()
        if flag in given
    }
    return SimpleNamespace(**vars(config) | changed)


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """The configuration, resolved as `resolved` resolves it for every
    instance, at the head whose digest is the tree's, with no law file; each
    start the stub's, given `check.START_S` 2 s."""
    r = deployment(tmp_path, **COVERED)
    served(monkeypatch, r.config)
    monkeypatch.setattr(
        drive_run, "_config", lambda a, arch=None: resolved(r.config, a)
    )
    head = {"commit": "c0ffee" * 6 + "c0ff", "ref": "origin/" + reuse.INTEGRATION}
    head["atom"] = compass_run.atom_digest()
    monkeypatch.setattr(reuse, "head", lambda commit=None: head)
    monkeypatch.setattr(reuse, "LAWS", tmp_path / "laws")
    (tmp_path / "laws").mkdir()
    stub = tmp_path / "stub"
    tree = str(Path(check.__file__).resolve().parents[3])
    stub.write_text(STUB.format(python=sys.executable, tree=tree))
    stub.chmod(0o755)
    monkeypatch.setattr(check, "COMPASS", stub)
    monkeypatch.setattr(check, "DEPLOY", stub)
    monkeypatch.setattr(check, "START_S", 2)
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.delenv("STUB_STALL", raising=False)
    return r


def advise(tmp_path, dimensions: dict, network=None) -> dict:
    """`compass advice` on `dimensions`' candidates at concurrencies 1 and 4,
    on the suite's platform, with `network` where given; the states it writes."""
    flags = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    machine = {"device": flags[1], "ptl": flags[3]}
    space = {
        "model": str(tmp_path),
        "machine": machine | ({"network": network} if network else {}),
        "store": str(tmp_path / "store"),
        "workload": {
            "traces": str(tmp_path / "traces.jsonl"),
            "select": "agentx",
            "concurrency": [1, 4],
        },
        "slo": {},
        "common": {"basis": "a recipe"},
        "dimensions": {"config": dimensions},
    }
    (tmp_path / "space.json").write_text(json.dumps(space))
    cli.main(["advice", str(tmp_path / "space.json"), str(tmp_path / "out")])
    return json.loads((tmp_path / "out" / check.STATES).read_text())


def starts(tmp_path) -> list[dict]:
    """Each start the stub recorded, in order."""
    calls = tmp_path / "calls.jsonl"
    lines = calls.read_text().splitlines() if calls.exists() else []
    return [json.loads(line) for line in lines]


def test_each_candidate_gets_one_state_and_each_point_missing_readings_its_command(
    tmp_path, stubbed, capsys
):
    where = law_file(tmp_path, law(stubbed))
    store, out = tmp_path / "store", tmp_path / "out"

    found = advise(
        tmp_path,
        {
            "a": {"per_concurrency": {"--max-model-len": 512}},
            "b": {"args": {"--max-num-seqs": 12}},
            "c": {"args": {"--enable-dynamic-chunking": True}},
            "d": {"env": {"STUB_STALL": "1"}},
        },
    )

    assert {name: s["state"] for name, s in found.items()} == {
        "a": "searchable",
        "b": "missing a law",
        "c": "refused",
        "d": "refused",
    }
    a = found["a"]["points"]
    assert a["a-c4"] == {"state": "searchable", "why": [], "made_by": []}
    assert a["a-c1"]["state"] == "missing readings"
    (measure,) = a["a-c1"]["made_by"]
    assert measure == (
        f"compass measure memory --gpu G --out D --store {store} -- --model "
        f"{tmp_path} --max-model-len 512"
    )
    (why,) = found["b"]["why"]
    assert "has no decode vector for rung eager, which" in why
    (calibration,) = found["b"]["made_by"]
    assert f"TRACES={tmp_path / 'traces.jsonl'} " in calibration
    assert f"calibrate.sh {out / 'cal' / 'b'} " in calibration
    log = out / "starts" / "c-c1" / "server.log"
    left = f"compass run: the server exited 1 before /health answered; see {log}"
    assert found["c"]["why"] == [f"{left}; {REFUSED}"]
    assert found["d"]["why"] == [f"no {out / 'starts/d-c1/server.json'} in 2 s"]
    assert all(s["points"] == {} for name, s in found.items() if name != "a")
    machine = json.loads((out / "candidates.json").read_text())["machine"]
    ran = [call["argv"] for call in starts(tmp_path)]
    each = ["run", "--device", machine["device"], "--ptl", machine["ptl"]]
    each += ["--store", str(store)]
    each += ["--step-linear", where, "--host-laws", where, "--out"]
    model = ["--", "--model", str(tmp_path)]
    assert ran == [
        [*each, str(out / "starts/a-c4"), *model, "--max-model-len", "2048"],
        [*each, str(out / "starts/c-c1"), *model, "--enable-dynamic-chunking"],
        [*each, str(out / "starts/d-c1"), *model],
    ]
    assert (out / "starts/a-c4/stopped").is_file()
    assert not (out / "cal").exists()
    printed = capsys.readouterr().out
    assert "  state: searchable, 1 of 2 points searchable\n" in printed
    assert f"    a-c1: missing readings, made by: {measure}\n" in printed
    assert f"    made by: {calibration}\n" in printed
    assert f"    refused: {found['c']['why'][0]}\n" in printed
    assert f"1 points searchable, each state in {out / check.STATES}\n" in printed


@pytest.mark.parametrize("lands", ["a law", "readings"])
def test_rerunning_after_an_input_lands_moves_only_what_it_serves_to_searchable(
    tmp_path, stubbed, lands
):
    # Under readings, a-c1 reads readings of its own.
    scaled = {"per_concurrency": {"--max-model-len": 512}}
    dimensions = {
        "a": scaled if lands == "readings" else {},
        "b": {"args": {"--kv-cache-dtype": "fp8"}},
    }
    if lands == "readings":
        law_file(tmp_path, law(stubbed))
    before = advise(tmp_path, dimensions)
    expanded = (tmp_path / "out" / "candidates.json").read_text()
    if lands == "a law":
        law_file(tmp_path, law(stubbed))
    else:
        config = resolved(stubbed.config, ["--max-model-len", "512"])
        publish_readings(tmp_path / "store", config)

    after = advise(tmp_path, dimensions)

    assert (tmp_path / "out" / "candidates.json").read_text() == expanded
    assert after["b"] == before["b"]
    assert before["b"]["state"] == "missing a law"
    if lands == "a law":
        assert before["a"]["state"] == "missing a law"
    else:
        assert before["a"]["points"]["a-c1"]["state"] == "missing readings"
        assert before["a"]["points"]["a-c4"] == after["a"]["points"]["a-c4"]
    assert {p: v["state"] for p, v in after["a"]["points"].items()} == {
        "a-c1": "searchable",
        "a-c4": "searchable",
    }


@pytest.mark.parametrize("network", [NETWORK, None])
def test_a_1p1d_start_takes_the_decode_side_s_own_variables_in_a_local_prefix(
    tmp_path, stubbed, monkeypatch, network
):
    # A side's keys hold the AITER_LOG_LEVEL deploy.sh sets; these variables none.
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    publish_readings(tmp_path / "store", stubbed.config)
    where = law_file(tmp_path, law(stubbed))
    out = tmp_path / "out"
    side = {
        "deployment": "1p1d",
        "env": {"STUB_BOTH": "1"},
        "prefill": {"args": {"--enable-tbo": True}, "env": {"STUB_PREFILL": "1"}},
        "decode": {"env": {"STUB_SIDE": "decode"}},
    }

    found = advise(tmp_path, {"p": side}, network)

    if network is None:
        assert found["p"] == {
            "state": "refused",
            "why": [ONE_ENVIRONMENT],
            "made_by": [],
            "points": {},
        }
        assert starts(tmp_path) == []
        return
    assert found["p"]["state"] == "searchable"
    machine = json.loads((out / "candidates.json").read_text())["machine"]
    model = ["--model", str(tmp_path)]
    (call,) = starts(tmp_path)
    assert call["argv"] == [
        str(out / "starts/p-c1"),
        "run",
        *("--device", machine["device"], "--ptl", machine["ptl"]),
        *("--store", str(tmp_path / "store")),
        *("--step-linear", where, "--host-laws", where, "--network", NETWORK),
        *("--prefill", "--", *model, "--enable-tbo"),
        *("--decode", "--on", "env -u STUB_PREFILL STUB_SIDE=decode"),
        *("--", *model, "--router", "--"),
    ]
    held = ("STUB_BOTH", "STUB_PREFILL", "STUB_SIDE")
    assert [call["env"].get(k) for k in held] == ["1", "1", None]


def test_a_tree_that_is_not_the_head_refuses_every_candidate_verbatim(
    tmp_path, stubbed, monkeypatch
):
    def another(commit=None):
        raise ValueError("the tree is not the head")

    monkeypatch.setattr(reuse, "head", another)

    found = advise(tmp_path, {"a": {}})

    assert found == {
        "a": {
            "state": "refused",
            "why": ["the tree is not the head"],
            "made_by": [],
            "points": {},
        }
    }
    assert starts(tmp_path) == []
