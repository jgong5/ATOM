# SPDX-License-Identifier: MIT
"""`compass advice SPACE OUT`'s check: each candidate gets one state, each
point missing readings the command that measures them, and each other point of
a searchable candidate a start with no traffic, all written into OUT.

The configuration is `test_advice_reuse`'s, its law the re-keyed DPA8 law as a
law file; ATOM's arguments resolve it in this process, changed by each of
`CHANGED` they state, but where a test resolves them in a real child.
`compass run` and `deploy.sh` are one stub that records each start and
answers, or, under `--enable-dynamic-chunking`, raises the simulated runner's
own refusal of it into its server log, logs its stop after it and leaves as
`compass run` does, or, under ``STUB_STALL``, never answers.
"""

import itertools
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import platforms
import pytest
from test_advice_reuse import COVERED, law, law_file, past, resolved_here
from test_capture_v4_pro import CONFIG as V4_PRO
from test_memory_readings import publish_readings
from test_run_lacking import served
from test_run_step_grid import DOCUMENT, deployment

from atom.compass import cli
from atom.compass import run as compass_run
from atom.compass.advice import check
from atom.compass.artifacts import Kind
from atom.compass.bench import ops
from atom.compass.drive import reuse
from atom.compass.drive import run as drive_run
from atom.compass.spec.layers import stack_of

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
    calls.write(
        json.dumps({{"argv": argv, "env": dict(os.environ), "cwd": os.getcwd()}}) + "\\n"
    )
at.mkdir(parents=True)
if "--enable-dynamic-chunking" in argv:
    refuse = (
        "from atom.compass.runner.overrides import RefusedDynamicChunking; "
        "RefusedDynamicChunking().profile_dynamic_chunking()"
    )
    with open(at / "server.log", "w") as log:
        subprocess.run([sys.executable, "-c", refuse], stderr=log, cwd={tree!r})
    with open(at / "server.log", "a") as log:
        log.write("the server stopped\\n")
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
#: The check's child and the resolution it runs, which the fixture keeps here.
CHILD, DECLARED = ops._child, drive_run._declared
#: Why Compass refuses `compass_refusing`'s side.
REFUSING = (
    f"{drive_run.KV_FLAG} is among the prefill side's ATOM arguments; --check "
    f"prefill adds its {drive_run.KV_FLAG}, as deploy.sh does"
)


class Unpicklable(Exception):
    """An error its own arguments do not rebuild, so it fails to unpickle, as
    huggingface_hub's `StrictDataclassFieldValidationError` does."""

    def __init__(self, field: str, cause: str):
        super().__init__(f"Validation error for field {field!r}: {cause}")


def unpicklable(atom_args, arch):
    """A resolution raising an `Unpicklable`."""
    raise Unpicklable("num_hidden_layers", "expected int, got str")


def compass_refusing(atom_args, arch):
    """A resolution Compass's own code refuses."""
    return drive_run.side_args("prefill", [drive_run.KV_FLAG], drive_run.SIMULATED)


def compass_failing(atom_args, arch):
    """A resolution failing in Compass's own code: no side is a server."""
    return drive_run.side_args("server", atom_args, drive_run.SIMULATED)


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
    start the stub's, given `check.START_S` 60 s, a bound only a stalling
    start reaches; with ``HARNESS_PYTHON`` unset, so no searchable point is
    simulated."""
    r = deployment(tmp_path, **COVERED)
    served(monkeypatch, r.config)
    resolved_here(monkeypatch, lambda atom_args: resolved(r.config, atom_args))
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
    monkeypatch.setattr(check, "START_S", 60)
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.delenv("STUB_STALL", raising=False)
    monkeypatch.delenv("HARNESS_PYTHON", raising=False)
    return r


def advise(tmp_path, dimensions: dict, out: str | None = None, **machine) -> dict:
    """`compass advice` on `dimensions`' candidates at concurrencies 1 and 4,
    on the suite's platform, with each of `machine`'s fields given, into OUT
    `out`, `tmp_path`'s ``out`` when None; the states it writes there."""
    flags = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    machine = {k: v for k, v in machine.items() if v}
    space = {
        "model": str(tmp_path),
        "machine": {"device": flags[1], "ptl": flags[3]} | machine,
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
    out = out or str(tmp_path / "out")
    cli.main(["advice", str(tmp_path / "space.json"), out])
    return json.loads((Path(out) / check.STATES).read_text())


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
        },
    )

    assert {name: s["state"] for name, s in found.items()} == {
        "a": "searchable",
        "b": "missing a law",
        "c": "refused",
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
    ]
    assert (out / "starts/a-c4/stopped").is_file()
    assert not (out / "cal").exists()
    printed = capsys.readouterr().out
    assert "  state: searchable, 1 of 2 points searchable\n" in printed
    assert f"    a-c1: missing readings, made by: {measure}\n" in printed
    assert f"    made by: {calibration}\n" in printed
    assert f"    refused: {found['c']['why'][0]}\n" in printed
    assert f"1 points searchable, each state in {out / check.STATES}\n" in printed


def test_a_start_that_does_not_answer_in_start_s_refuses_its_candidate(
    tmp_path, stubbed, monkeypatch
):
    monkeypatch.setattr(check, "START_S", 2)
    law_file(tmp_path, law(stubbed))

    found = advise(tmp_path, {"d": {"env": {"STUB_STALL": "1"}}})

    ready = tmp_path / "out" / "starts" / "d-c1" / "server.json"
    assert found["d"] == {
        "state": "refused",
        "why": [f"no {ready} in 2 s"],
        "made_by": [],
        "points": {},
    }


def test_a_refusal_at_one_point_outranks_another_point_s_missing_law(tmp_path, stubbed):
    # Past the gate on records at the head: c1, whose ladder the law covers,
    # is refused; c4, whose bound of 32 passes the ladder, lacks the eager rung.
    law_file(tmp_path, past(stubbed)[0])

    found = advise(tmp_path, {"a": {"per_concurrency": {"--max-num-seqs": 8}}})

    (why,) = found["a"]["why"]
    assert (found["a"]["state"], found["a"]["made_by"]) == ("refused", [])
    assert "past 5%, on records run at the head" in why
    assert starts(tmp_path) == []


def test_a_value_atom_s_parser_rejects_refuses_its_candidate_with_the_parser_s_error(
    tmp_path, stubbed
):
    # The stubbed resolution reads no --block-size; ATOM's parser types it.
    found = advise(tmp_path, {"a": {}, "b": {"args": {"--block-size": "big"}}})

    rejected = "argument --block-size/--block_size: invalid int value: 'big'"
    assert found["a"]["state"] == "missing a law"
    assert found["b"] == {
        "state": "refused",
        "why": [f"ATOM's parser: {rejected}"],
        "made_by": [],
        "points": {},
    }
    assert starts(tmp_path) == []


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("tp_size (8) must be divisible by dcp_size (3)"),
        TypeError("ATOM_KV_OFFLOAD_EXTRA_CONFIG must be a JSON object"),
    ],
    ids=["assertion", "type"],
)
def test_a_configuration_atom_refuses_refuses_its_candidate_with_atom_s_error(
    tmp_path, stubbed, monkeypatch, error
):
    def refusing(atom_args):
        if "--decode-context-parallel-size" in atom_args:
            raise error
        return resolved(stubbed.config, atom_args)

    resolved_here(monkeypatch, refusing)

    found = advise(
        tmp_path, {"a": {}, "b": {"args": {"--decode-context-parallel-size": 3}}}
    )

    assert found["a"]["state"] == "missing a law"
    assert found["b"] == {
        "state": "refused",
        "why": [f"ATOM's config: {type(error).__name__}: {error}"],
        "made_by": [],
        "points": {},
    }
    assert starts(tmp_path) == []


@pytest.mark.parametrize(
    ("declared", "args", "why"),
    [
        (
            DECLARED,
            {"--kv-transfer-config": "{bad"},
            "ATOM's config: SyntaxError: '{' was never closed",
        ),
        (
            DECLARED,
            {"--tensor-parallel-size": 8, "--decode-context-parallel-size": 3},
            (
                "ATOM's config: AssertionError: tp_size (8) must be divisible by "
                "dcp_size (3)"
            ),
        ),
        (
            unpicklable,
            {},
            (
                "ATOM's config: test_advice_check.Unpicklable: Validation error for "
                "field 'num_hidden_layers': expected int, got str"
            ),
        ),
        (compass_refusing, {}, REFUSING),
    ],
    ids=["atom-stdlib", "atom-config", "unpicklable", "compass"],
)
def test_what_the_child_raises_resolving_refuses_its_candidate_as_its_line(
    tmp_path, stubbed, monkeypatch, declared, args, why
):
    # A real child; ATOM's own resolution reads DeepSeek-V4-Pro and fails in
    # the standard library at {bad, or in atom/config.py at DCP 3 of TP 8.
    monkeypatch.setattr(ops, "_child", CHILD)
    monkeypatch.setattr(drive_run, "_declared", declared)
    shutil.copy(V4_PRO, tmp_path / "config.json")

    found = advise(tmp_path, {"a": {"args": args}})

    assert found["a"] == {"state": "refused", "why": [why], "made_by": [], "points": {}}
    assert starts(tmp_path) == []


def test_compass_s_own_defect_resolving_fails_compass_advice_with_its_traceback(
    tmp_path, stubbed, monkeypatch
):
    monkeypatch.setattr(ops, "_child", CHILD)
    monkeypatch.setattr(drive_run, "_declared", compass_failing)

    with pytest.raises(RuntimeError) as failed:
        advise(tmp_path, {"a": {}})

    said = str(failed.value)
    assert said.startswith(
        f"Compass's own code raised resolving ['--model', '{tmp_path}'"
    )
    assert "\nTraceback (most recent call last):\n" in said
    assert ", in side_args\n" in said
    assert said.endswith("\nKeyError: 'server'\n")
    assert not (tmp_path / "out" / check.STATES).exists()


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


@pytest.mark.parametrize("given", ["whole", "relative"])
def test_a_start_runs_here_taking_out_whole_the_store_holding_its_inputs_and_stack(
    tmp_path, stubbed, monkeypatch, given
):
    # Only the store a calibration of a writes into holds the readings.
    store, own = tmp_path / "store", tmp_path / "out/cal/a/store"
    law_file(tmp_path, law(stubbed))
    platforms.flags(own, DOCUMENT, tmp_path)
    publish_readings(own, stubbed.config)
    shutil.rmtree(store / Kind.MEMORY_READINGS.value)
    (entry,) = (own / Kind.PLATFORM_READINGS.value).glob("*/entry.json")
    stack = stack_of(json.loads(entry.read_text())["key"])
    monkeypatch.chdir(tmp_path)

    out = "out" if given == "relative" else None
    found = advise(tmp_path, {"a": {}}, out=out, stack=stack)

    assert found["a"]["state"] == "searchable"
    (call,) = starts(tmp_path)
    argv = call["argv"]
    # Where the check resolved its arguments, such as a relative model.
    assert call["cwd"] == os.getcwd()
    assert argv[argv.index("--out") + 1] == str(tmp_path / "out/starts/a-c1")
    assert argv[argv.index("--store") + 1] == str(own)
    assert argv[argv.index("--stack") + 1] == stack


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

    found = advise(tmp_path, {"p": side}, network=network)

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
        *("--", *model, "--router", "--", *check.ROUTER),
    ]
    held = ("STUB_BOTH", "STUB_PREFILL", "STUB_SIDE")
    assert [call["env"].get(k) for k in held] == ["1", "1", None]


def test_a_1p1d_decode_side_variable_holding_whitespace_is_refused_unstarted(
    tmp_path, stubbed, monkeypatch
):
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    publish_readings(tmp_path / "store", stubbed.config)
    law_file(tmp_path, law(stubbed))
    side = {"deployment": "1p1d", "decode": {"env": {"STUB_SIDE": "a b"}}}

    found = advise(tmp_path, {"p": side}, network=NETWORK)

    (why,) = found["p"]["why"]
    assert why == (
        "deploy.sh takes the decode side's own variables as one --on prefix split "
        "on whitespace, and STUB_SIDE=a b holds whitespace"
    )
    assert (found["p"]["state"], starts(tmp_path)) == ("refused", [])


def test_a_1p1d_side_missing_readings_is_measured_on_the_connector_a_real_start_takes(
    tmp_path, stubbed, monkeypatch
):
    # The law is keyed under the AITER_LOG_LEVEL a side runs under.
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    law_file(tmp_path, law(stubbed))
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)

    found = advise(tmp_path, {"p": {"deployment": "1p1d", "env": {"STUB_BOTH": "1"}}})

    assert found["p"]["state"] == "searchable"
    measured = "AITER_LOG_LEVEL=WARNING STUB_BOTH=1 compass measure memory --gpu G "
    measured += f"--out D --store {tmp_path / 'store'} -- --model {tmp_path} "
    point = found["p"]["points"]["p-c1"]
    assert point["state"] == "missing readings"
    assert point["made_by"] == [
        f"{measured}--kv-transfer-config "
        + shlex.quote(json.dumps({"kv_role": role, "kv_connector": "mooncake"}))
        for role in ("kv_producer", "kv_consumer")
    ]
    assert starts(tmp_path) == []


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
