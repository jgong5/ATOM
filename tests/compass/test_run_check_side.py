# SPDX-License-Identifier: MIT
"""`compass run --check prefill|decode`: one side of a `deploy.sh run`
deployment checked alone, its flags and ATOM arguments as the script takes
them.

DeepSeek-V4-Pro's DPA8 sides resolve the fp8 index cache under the
`--kv-transfer-config` the script adds, and fp4 without one at gfx950. The
shipped crs005 DPA8 pair is keyed to fp8, so each side's check finds its own
law and host laws there and lacks only the decode rungs its capture ladder
schedules that its own law has no vector for, and its readings, which the
command it prints measures; the same arguments checked as a lone server
resolve fp4, and both laws refuse them, naming it.

Each check resolves its config where the machine's gfx950 is declared, a
process of its own (`run._config`). The host laws' CPU is the laws', and ATOM's
server loads a fast tokenizer.
"""

import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import platforms
import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_drive_run import stub
from test_memory_readings import DOCUMENT, publish_readings
from test_run_lacking import lacked, served
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.drive import run as drive_run
from atom.compass.drive import runfile
from atom.model_engine import llm_engine

LAWS = Path(__file__).resolve().parents[2] / "scripts/compass/laws"
#: The crs005 DPA8 pair, prefill then decode.
PAIR = {
    side: LAWS / f"deepseek-v4-pro-dpa8-mi355x-crs005-{side}.json"
    for side in ("prefill", "decode")
}
#: The machine of the pair's records: an MI355X at the stack their laws
#: state, PTL off.
MACHINE = {
    **DOCUMENT,
    "device": {
        **DOCUMENT["device"],
        "software_pinned_to": {
            "rocm": "7.2.4",
            "aiter": "41d4d8d7e",
            "rccl": "2.27.7",
            "torch": "2.10.0+rocm7.2.4.git3d3aa833",
        },
    },
}
DECODE_SIZES = "[1,2," + ",".join(map(str, range(4, 257, 4))) + "]"
#: Each side's own ATOM arguments and environment, as its records ran.
SIDES = {
    "prefill": (["--enable-tbo"], {"ATOM_NUMA_BIND": "1", "GPU_MAX_HW_QUEUES": "5"}),
    "decode": (["--cudagraph-capture-sizes", DECODE_SIZES], {}),
}
#: Both sides' environment less AITER_LOG_LEVEL, which deploy.sh sets.
SHARED_ENV = {"AITER_BF16_FP8_MOE_BOUND": "0", "ATOM_MOE_GU_ITLV": "1"}
ROLES = {"prefill": "kv_producer", "decode": "kv_consumer"}
#: The decode rungs each side's capture ladder schedules that its law in the
#: pair has no vector for.
UNCOVERED = {
    "prefill": "8, 16, 48, 64, 128, 256, 512",
    "decode": ", ".join(map(str, range(32, 257, 4))) + ", eager",
}
COVERAGE = "step linear law: stale (coverage)"


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    # Set first, so each is restored after the test, AITER_LOG_LEVEL too,
    # which a side's check sets.
    for name in PERFORMANCE_KEYS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def platform(tmp_path) -> list[str]:
    """The platform flags of the pair's machine, its readings in the store."""
    store = tmp_path / "store"
    return [*platforms.flags(store, MACHINE, tmp_path), "--store", str(store)]


def pair(tmp_path) -> list[str]:
    """`deploy.sh run`'s flags for the crs005 pair."""
    laws = [
        [flag, str(PAIR[s])] for flag in ("--step-linear", "--host-laws") for s in PAIR
    ]
    return [word for law in laws for word in law] + platform(tmp_path)


def side_args(tmp_path, monkeypatch, side) -> list[str]:
    """`side`'s ATOM arguments as deploy.sh takes them, serving DeepSeek-V4-Pro,
    in its environment, on a host with the laws' CPU, its server loading a
    fast tokenizer."""
    model = tmp_path / "DeepSeek-V4-Pro"
    model.mkdir(exist_ok=True)
    shutil.copy(V4_PRO, model / "config.json")
    own, env = SIDES[side]
    for name, value in {**SHARED_ENV, **env}.items():
        monkeypatch.setenv(name, value)
    cpu = json.loads(PAIR[side].read_text())["host"]["key"]["host_cpu"]
    monkeypatch.setattr(compass_run, "host_cpu", lambda: cpu)
    tok = SimpleNamespace(is_fast=True, name_or_path=None)
    monkeypatch.setattr(llm_engine, "_load_tokenizer", lambda *args: tok)
    common = ["-tp", "8", "--kv_cache_dtype", "fp8", "--enable-dp-attention"]
    return ["--model", str(model), *common, *own]


def checked(flags, side, atom_args) -> int:
    """`compass run --check`'s exit status: of `side`, or with `side` None of
    a lone server."""
    named = ["--check", *([side] if side else [])]
    with pytest.raises(SystemExit) as exited:
        drive_run.main([*flags, *named, "--", *atom_args])
    return exited.value.code


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_side_holds_the_fp8_index_cache_its_crs005_laws_state(
    tmp_path, monkeypatch, capsys, side
):
    atom_args = side_args(tmp_path, monkeypatch, side)

    assert checked(pair(tmp_path), side, atom_args) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [COVERAGE, "memory readings: missing"]
    assert f"no decode vector for rungs {UNCOVERED[side]}, which" in out
    assert '"index_cache_dtype": "fp8"' in out
    assert '"AITER_LOG_LEVEL": "WARNING"' in out
    kv = json.dumps({"kv_role": ROLES[side], "kv_connector": "mooncake"})
    assert out.endswith(f" --kv-transfer-config '{kv}'\n")


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_a_side_checked_as_a_lone_server_resolves_fp4_and_both_laws_refuse_it(
    tmp_path, monkeypatch, capsys, side
):
    atom_args = side_args(tmp_path, monkeypatch, side)
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")

    assert checked(pair(tmp_path), None, atom_args) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [
        "step linear law: stale",
        "host laws: stale",
        "memory readings: missing",
    ]
    own = f"entry {list(PAIR).index(side)} in ['index_cache_dtype']"
    for what in ("step_linear", "host"):
        assert f"no {what} entry was fitted under this engine config" in out
    assert out.count(own) == 2
    assert '"index_cache_dtype": "fp4"' in out


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_readings_the_printed_command_measures_complete_the_side_s_check(
    tmp_path, monkeypatch, capsys, side
):
    atom_args = side_args(tmp_path, monkeypatch, side)
    flags = pair(tmp_path)
    assert checked(flags, side, atom_args) == 1
    words = shlex.split(capsys.readouterr().out.rsplit("made by: ", 1)[1])
    env, measured = words[: words.index("compass")], words[words.index("--") + 1 :]
    # The command runs in the shell the check ran in, AITER_LOG_LEVEL unset.
    monkeypatch.delenv("AITER_LOG_LEVEL")
    for name, value in (word.split("=", 1) for word in env):
        monkeypatch.setenv(name, value)
    config = drive_run._config(measured, "gfx950")
    publish_readings(tmp_path / "store", config, MACHINE)
    monkeypatch.delenv("AITER_LOG_LEVEL", raising=False)

    assert checked(flags, side, atom_args) == 1

    assert lacked(capsys.readouterr().out) == [COVERAGE]


@pytest.mark.parametrize(
    "given",
    [
        ["--kv-transfer-config", '{"kv_role":"kv_producer","kv_connector":"compass"}'],
        ["--kv-transfer-config={}"],
    ],
    ids=["compass", "equals"],
)
def test_a_side_s_own_kv_transfer_config_refuses_as_deploy_sh_refuses_it(
    tmp_path, given
):
    flags = ["--step-stub", stub(tmp_path), *platform(tmp_path)]

    with pytest.raises(ValueError, match="adds its --kv-transfer-config") as e:
        drive_run.main([*flags, "--check", "prefill", "--", "--model", "m", *given])
    assert str(e.value).startswith(f"{given[0]} is among the prefill side's")


def test_the_config_a_side_adds_overrides_another_spelling_as_deploy_sh_s_does():
    given = ["--model", "m", "--kv_transfer_config", '{"kv_connector":"mooncake"}']

    resolved = runfile._engine_args(drive_run.side_args("decode", given, "compass"))

    kv = {"kv_role": "kv_consumer", "kv_connector": "compass"}
    assert json.loads(resolved.kv_transfer_config) == kv


def test_a_side_reads_a_network_file_across_nodes_and_a_lone_server_refuses_one(
    tmp_path, monkeypatch, capsys
):
    served(monkeypatch, deployment(tmp_path).config)
    flags = ["--step-stub", stub(tmp_path), *platform(tmp_path)]
    flags += platforms.network(MACHINE, tmp_path)
    atom_args = ["--model", str(tmp_path)]

    # The deployment's readings are at another platform's key.
    assert checked(flags, "decode", atom_args) == 1
    assert lacked(capsys.readouterr().out) == ["memory readings: missing"]

    assert checked(flags, None, atom_args) == 1
    out = capsys.readouterr().out
    assert lacked(out) == ["platform: missing"]
    assert "a run on one node reads no network" in out
