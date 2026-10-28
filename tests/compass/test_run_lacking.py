# SPDX-License-Identifier: MIT
"""`compass run --check`: every input a run lacks, listed before any server.

The run is `test_drive_run`'s: its store and deployment, a law and host laws
published as `test_drive_runfile` publishes them, a bundle as `bundle` does,
and an op graph run as `op_graph_run` sets one up. ATOM's server loads a fast
tokenizer with no tokenizer.json, which the suite's machine spec holds an
entry for under ``Qwen3ForCausalLM``.
"""

import json
import re
import shutil
from types import SimpleNamespace

import pytest
from test_artifact_invalidation import moved
from test_drive_run import bundle, op_graph_run, recorded, stub, unstarted
from test_drive_runfile import LAWS, host_laws, install, published, step_linear
from test_memory_readings import DOCUMENT
from test_run_load_invalidation import fitted
from test_run_step_grid import SPEC_DIGEST, deployment, host_laws_key, readings
from test_step_grid_plan import MIXED, SHORT, located, measured, plan

from atom.compass import run as compass_run
from atom.compass.artifacts import Axis, Kind
from atom.compass.backends.step_roofline import UNCLASSIFIED, Module
from atom.compass.bench import ops
from atom.compass.drive import run as drive_run
from atom.compass.drive import runfile, serve
from atom.compass.validate import bound
from atom.model_engine import llm_engine


def checked(tmp_path, *flags, machine=None):
    """`compass run --check`'s exit status with `flags`, the suite's machine,
    or the file `machine`, and the store, serving the model in `tmp_path`."""
    if machine is None:
        machine = tmp_path / "machine.json"
        machine.write_text(json.dumps(DOCUMENT))
    common = ["--machine", str(machine), "--store", str(tmp_path / "store")]
    with pytest.raises(SystemExit) as exited:
        drive_run.main([*flags, *common, "--check", "--", "--model", str(tmp_path)])
    return exited.value.code


def lacked(out: str) -> list[str]:
    """Each input `out` names lacking, with its state."""
    return re.findall(r"^lacking (.*)$", out, re.MULTILINE)


def served(monkeypatch, config, architecture="Qwen3ForCausalLM"):
    """`config` resolved from ATOM's arguments, a model of `architecture`, its
    tokenizer a fast one with no tokenizer.json; no server starts."""
    config.hf_config.architectures = [architecture]
    monkeypatch.setattr(drive_run, "_config", lambda atom_args: config)
    tok = SimpleNamespace(is_fast=True, name_or_path=None)
    monkeypatch.setattr(llm_engine, "_load_tokenizer", lambda *args: tok)
    monkeypatch.setattr(serve, "serve", unstarted)


def test_one_check_lists_every_missing_input_with_what_makes_it(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path)
    served(monkeypatch, r.config, "Qwen3MoeForCausalLM")
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
    flags = ["--step-linear", "0123456789ab", "--host-laws", "ba9876543210"]
    flags += ["--acceptance-record", str(tmp_path / "speculative")]

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [
        "tokenizer entry: missing",
        "step linear law 0123456789ab: missing",
        "host laws ba9876543210: missing",
        "memory readings: missing",
        "acceptance record: missing",
    ]
    assert "no tokenizer probed for 'Qwen3MoeForCausalLM'" in out
    assert "0 step_linear_law entries in" in out
    assert "made by: compass record, then compass fit host" in out
    made = f"--store {tmp_path / 'store'} -- --model {tmp_path}"
    assert "made by: compass measure memory --gpu G --out D --machine " in out
    assert made in out
    assert not (tmp_path / "out").exists()


def test_a_stale_law_names_each_axis_and_stale_host_laws_each_field(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path)
    served(monkeypatch, r.config)
    law = fitted(tmp_path, r, moved(readings(r), Axis.ATOM_SRC)).key.token()
    key = {**compass_run.host_key(r.config), "atom": "0" * 64}
    key["machine_spec"] = SPEC_DIGEST
    params = {"key": key, "laws": LAWS}
    host = published(tmp_path, r, Kind.HOST_LAWS, host_laws_key(r), params)

    assert checked(tmp_path, "--step-linear", law, "--host-laws", host.key.token())

    out = capsys.readouterr().out
    assert lacked(out) == [
        f"step linear law {law}: stale (ATOM src)",
        "host laws: stale",
    ]
    assert "host laws were fitted under another ['atom']" in out


def test_check_meets_the_runner_s_own_refusal_of_a_law_from_another_engine_config(
    tmp_path, monkeypatch, capsys
):
    entry = step_linear(tmp_path, deployment(tmp_path))
    run = runfile.build(*_parsed(tmp_path, "--step-linear", entry.key.token()))
    here = deployment(tmp_path, kv_cache_block_size=32)
    with pytest.raises(ValueError) as installed:
        install(tmp_path, monkeypatch, run, here)
    served(monkeypatch, here.config)

    assert checked(tmp_path, "--step-linear", entry.key.token()) == 1

    out = capsys.readouterr().out
    assert lacked(out) == ["step linear law: stale"]
    assert f"  refused: {installed.value}\n" in out
    assert "kv_cache_block_size 16 in the law and 32 here" in out


def _parsed(tmp_path, *flags):
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    common = ["--machine", str(tmp_path / "machine.json")]
    argv = [*flags, *common, "--store", str(tmp_path / "store"), "--check"]
    return drive_run._parser().parse_args(argv), ["--model", str(tmp_path)]


def test_a_run_that_lacks_nothing_exits_0_and_needs_no_out(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path)
    served(monkeypatch, r.config)
    law, host = step_linear(tmp_path, r), host_laws(tmp_path, r)
    flags = ["--step-linear", law.key.token(), "--host-laws", host.key.token()]

    assert checked(tmp_path, *flags) == 0

    out = capsys.readouterr().out
    assert out == "nothing lacking: the run meets no refusal before it serves\n"
    assert not (tmp_path / "out").exists()


def test_a_missing_machine_spec_is_listed_alone(tmp_path, monkeypatch, capsys):
    served(monkeypatch, deployment(tmp_path).config)

    code = checked(tmp_path, "--step-stub", stub(tmp_path), machine="absent.json")

    assert code == 1
    assert lacked(capsys.readouterr().out) == ["machine spec: missing"]


def test_a_stale_bundle_is_listed_and_a_held_one_s_coverage_printed_apart(
    tmp_path, monkeypatch, capsys
):
    # Each check builds the model on the meta device; the suite's model
    # classifies every module, so no line names one.
    directory = recorded(tmp_path, monkeypatch, MIXED)
    planned = plan(directory / serve.STEPS)
    points = located(tmp_path, planned)
    member = measured(planned, points, drop=("decode", points["decode"][0]))
    stale, held = tmp_path / "stale", tmp_path / "held"
    stale.mkdir(), held.mkdir()
    token, _ = bundle(stale, monkeypatch, member, kv_cache_dtype="fp8")
    served(monkeypatch, drive_run._config([]))

    assert checked(stale, "--step-grid", token) == 1
    out = capsys.readouterr().out
    assert lacked(out) == ["step grid bundle: stale"]
    assert "kv_cache_dtype 'bf16' in the bundle and 'fp8' here" in out

    token, _ = bundle(held, monkeypatch, member)
    served(monkeypatch, drive_run._config([]))

    assert checked(held, "--step-grid", token, "--coverage", str(directory)) == 0
    out = capsys.readouterr().out
    assert lacked(out) == [] and "unclassified" not in out
    covered = re.search(
        r"^apart: coverage of (.*): (\d+) of (\d+) steps", out, re.MULTILINE
    )
    assert covered[1] == str(directory) and 0 < int(covered[2]) < int(covered[3])


def test_an_absent_price_list_prints_apart_and_counts_under_coverage(
    tmp_path, monkeypatch, capsys
):
    directory = recorded(tmp_path, monkeypatch, SHORT)
    op_graph_run(tmp_path, monkeypatch, directory / serve.STEPS, None)
    served(monkeypatch, drive_run._config([]))

    assert checked(tmp_path, "--op-graph") == 0
    out = capsys.readouterr().out
    assert lacked(out) == [] and "apart: price list: none at price_list" in out

    assert checked(tmp_path, "--op-graph", "--coverage", str(directory)) == 1
    out = capsys.readouterr().out
    assert lacked(out) == ["price list: missing"]
    assert "refused: --coverage: the store holds no price_list" in out


def test_modules_the_step_roofline_leaves_unclassified_print_apart(
    tmp_path, monkeypatch, capsys
):
    served(monkeypatch, deployment(tmp_path).config)
    walked = [Module("gemm at a", "gemm"), Module("Mystery at b", UNCLASSIFIED)]
    monkeypatch.setattr(bound, "walked", lambda atom_args, arch: walked)
    monkeypatch.setattr(ops, "_child", lambda fn, *args: fn(*args))

    flags = ["--step-stub", stub(tmp_path), "--step-roofline"]
    assert checked(tmp_path, *flags) == 0

    out = capsys.readouterr().out
    assert out.endswith("leaving these modules unclassified: Mystery at b\n")


def test_check_refuses_beside_join_and_a_run_without_it_still_needs_out(tmp_path):
    join = ["--join", str(tmp_path / "run_file.json"), "--check"]
    with pytest.raises(SystemExit) as refused:
        drive_run.main([*join, "--", "--model", str(tmp_path)])
    assert refused.value.code == 2

    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    argv = ["--step-stub", stub(tmp_path), "--machine", str(tmp_path / "machine.json")]
    with pytest.raises(SystemExit) as refused:
        drive_run.main([*argv, "--store", str(tmp_path), "--", "--model", "m"])
    assert refused.value.code == 2
