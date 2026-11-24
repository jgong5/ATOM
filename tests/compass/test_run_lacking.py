# SPDX-License-Identifier: MIT
"""`compass run --check`: every input a run lacks, listed before any server.

The run is `test_drive_run`'s: its store and deployment, a law and host laws
published as `test_drive_runfile` publishes them, a bundle as `bundle` does,
and an op graph run as `op_graph_run` sets one up. ATOM's server loads a fast
tokenizer with no tokenizer.json.
"""

import json
import re
import shutil
from types import SimpleNamespace

import platforms
import pytest
from test_artifact_invalidation import moved
from test_drive_run import bundle, op_graph_run, recorded, stub, unstarted
from test_drive_runfile import LAWS, host_laws, install, published, step_linear
from test_memory_readings import CONFIG_JSON, DOCUMENT
from test_run_load_invalidation import fitted
from test_run_step_grid import PLATFORM, deployment, host_laws_key, readings
from test_step_grid_plan import MIXED, SHORT, located, measured, plan
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.artifacts import Axis, Kind, keyed_platform
from atom.compass.backends.step_roofline import UNCLASSIFIED, Module
from atom.compass.bench import ops
from atom.compass.drive import run as drive_run
from atom.compass.drive import runfile, serve
from atom.compass.fit import device
from atom.compass.kv.lmcache_offload import OffloadShell
from atom.compass.validate import bound
from atom.model_engine import llm_engine
from atom.utils import clock

#: Capture sizes whose rungs `step_linear`'s law has decode vectors for: rung 2,
#: and the eager rung past it.
COVERED = [2]
#: An ``lmcache_offload`` entry the connector reads.
OFFLOAD_ENTRY = {
    "chunk_size": 256,
    "tensor_parallel_size": 1,
    "chunk_bytes": 2**20,
    **{term: [0.0, 0.0] for term in ("lookup_s", "store_s", "load_s")},
}
#: A deployment offloading KV through ATOM's ``lmcache_offload``.
OFFLOADS = {"kv_connector": "lmcache_offload", "kv_role": "offload"}


def checked(tmp_path, *flags, platform=None):
    """`compass run --check`'s exit status with `flags`, the suite's platform,
    or the `platform` flags, and the store, serving the model in `tmp_path`."""
    common = platform or platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    common += ["--store", str(tmp_path / "store")]
    with pytest.raises(SystemExit) as exited:
        drive_run.main([*flags, *common, "--check", "--", "--model", str(tmp_path)])
    return exited.value.code


def lacked(out: str) -> list[str]:
    """Each input `out` names lacking, with its state."""
    return re.findall(r"^lacking (.*)$", out, re.MULTILINE)


def served(monkeypatch, config):
    """`config` resolved from ATOM's arguments, its tokenizer a fast one with no
    tokenizer.json; no server starts."""
    monkeypatch.setattr(drive_run, "_config", lambda atom_args, arch=None: config)
    tok = SimpleNamespace(is_fast=True, name_or_path=None)
    monkeypatch.setattr(llm_engine, "load_tokenizer", lambda *args: tok)
    monkeypatch.setattr(serve, "serve", unstarted)


def test_one_check_lists_every_missing_input_with_what_makes_it(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path)
    served(monkeypatch, r.config)
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
    flags = ["--step-linear", "0123456789ab", "--host-laws", "ba9876543210"]
    flags += ["--acceptance-record", str(tmp_path / "speculative")]

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [
        "step linear law 0123456789ab: missing",
        "host laws ba9876543210: missing",
        "memory readings: missing",
        "acceptance record: missing",
    ]
    assert "0 step_linear_law entries in" in out
    assert "made by: compass record, then compass fit host" in out
    made = f"--store {tmp_path / 'store'} -- --model {tmp_path}"
    assert "made by: compass measure memory --gpu G --out D --store " in out
    assert made in out
    assert not (tmp_path / "out").exists()


def test_a_stale_law_names_each_axis_and_stale_host_laws_each_field(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path)
    served(monkeypatch, r.config)
    law = fitted(tmp_path, r, moved(readings(r), Axis.ATOM_SRC)).key.token()
    key = {**compass_run.host_key(r.config), "atom": "0" * 64}
    key |= keyed_platform(Kind.HOST_LAWS, PLATFORM)
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
    entry = step_linear(tmp_path, deployment(tmp_path, capture_sizes=COVERED))
    run = runfile.build(*_parsed(tmp_path, "--step-linear", entry.key.token()))
    here = deployment(tmp_path, kv_cache_block_size=32, capture_sizes=COVERED)
    with pytest.raises(ValueError) as installed:
        install(tmp_path, monkeypatch, run, here)
    served(monkeypatch, here.config)

    assert checked(tmp_path, "--step-linear", entry.key.token()) == 1

    out = capsys.readouterr().out
    assert lacked(out) == ["step linear law: stale"]
    assert f"  refused: {installed.value}\n" in out
    assert "kv_cache_block_size 16 in the law and 32 here" in out


def test_check_meets_the_server_s_refusal_of_a_tokenizer_from_another_backend(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path)
    served(monkeypatch, r.config)
    key = compass_run.host_key(r.config) | keyed_platform(Kind.HOST_LAWS, PLATFORM)
    term = {"backend": "slow", "fingerprint": "sha256:" + "a" * 64}
    params = {"key": key, "laws": LAWS, "tokenizer": term}
    host = published(tmp_path, r, Kind.HOST_LAWS, host_laws_key(r), params)
    flags = ["--step-stub", stub(tmp_path), "--host-laws", host.key.token()]
    (tmp_path / "run_file.json").write_text(
        json.dumps(runfile.build(*_parsed(tmp_path, *flags)))
    )
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run_file.json"))
    with pytest.raises(ValueError) as refused:
        compass_run.tokenizer(llm_engine.load_tokenizer(str(tmp_path)), r.config)
    monkeypatch.delenv(compass_run.ENV)

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == ["host laws: stale"]
    assert f"  refused: {refused.value}\n" in out
    assert "fitted on the slow backend and the server loaded the fast one" in out
    assert "made by: compass record, then compass fit host" in out


def _parsed(tmp_path, *flags):
    common = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    argv = [*flags, *common, "--store", str(tmp_path / "store"), "--check"]
    return drive_run._parser().parse_args(argv), ["--model", str(tmp_path)]


def test_a_run_that_lacks_nothing_exits_0_and_needs_no_out(
    tmp_path, monkeypatch, capsys
):
    r = deployment(tmp_path, capture_sizes=COVERED)
    served(monkeypatch, r.config)
    law, host = step_linear(tmp_path, r), host_laws(tmp_path, r)
    flags = ["--step-linear", law.key.token(), "--host-laws", host.key.token()]

    assert checked(tmp_path, *flags) == 0

    out = capsys.readouterr().out
    assert out == "nothing lacking: every input the run reads resolves and holds\n"
    assert not (tmp_path / "out").exists()


def test_missing_platform_readings_are_listed_alone(tmp_path, monkeypatch, capsys):
    served(monkeypatch, deployment(tmp_path).config)
    monkeypatch.setattr(device, "COPIES", tmp_path / "no copies")

    platform = ["--device", "mi355x", "--ptl", "disabled"]
    code = checked(tmp_path, "--step-stub", stub(tmp_path), platform=platform)

    assert code == 1
    assert lacked(capsys.readouterr().out) == ["platform: missing"]


def test_a_platform_the_store_lacks_resolves_from_its_committed_copy(
    tmp_path, monkeypatch, capsys
):
    served(monkeypatch, deployment(tmp_path).config)

    platform = ["--device", "mi355x", "--ptl", "disabled"]
    code = checked(tmp_path, "--step-stub", stub(tmp_path), platform=platform)

    assert code == 1
    assert lacked(capsys.readouterr().out) == ["memory readings: missing"]


@pytest.mark.parametrize(
    "text", [None, "{", '{"prefill_tokens": 1}'], ids=["absent", "not JSON", "field"]
)
def test_a_step_stub_the_run_cannot_build_is_listed_missing(
    tmp_path, monkeypatch, capsys, text
):
    r = deployment(tmp_path)
    path = tmp_path / "typo.json"
    if text is not None:
        path.write_text(text)
    with pytest.raises((OSError, TypeError, ValueError)) as refused:
        run = runfile.build(*_parsed(tmp_path, "--step-stub", str(path)))
        install(tmp_path, monkeypatch, run, r)
    served(monkeypatch, r.config)

    assert checked(tmp_path, "--step-stub", str(path)) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [f"step stub {path}: missing"]
    assert f"  refused: {refused.value}\n" in out


@pytest.mark.parametrize("text", [None, "{"], ids=["absent", "not JSON"])
def test_an_lmcache_offload_file_the_run_cannot_read_is_listed_missing(
    tmp_path, monkeypatch, capsys, text
):
    r = deployment(tmp_path, kv_transfer_config=OFFLOADS)
    path = tmp_path / "typo.json"
    if text is not None:
        path.write_text(text)
    flags = ["--step-stub", stub(tmp_path), "--lmcache-offload", str(path)]
    with pytest.raises((OSError, ValueError)) as refused:
        runfile.build(*_parsed(tmp_path, *flags))
    served(monkeypatch, r.config)

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [f"lmcache_offload entry {path}: missing"]
    assert f"  refused: {refused.value}\n" in out


@pytest.mark.parametrize(
    "kv, entry, state",
    [
        ({}, OFFLOAD_ENTRY, "stale"),
        (OFFLOADS, None, "missing"),
        (OFFLOADS, [], "stale"),
    ],
    ids=["offloads none", "no entry", "not a mapping"],
)
def test_check_meets_the_engine_s_refusal_of_an_lmcache_offload_entry_or_none(
    tmp_path, monkeypatch, capsys, kv, entry, state
):
    r = deployment(tmp_path, kv_transfer_config=kv)
    flags, name = ["--step-stub", stub(tmp_path)], "lmcache_offload entry"
    if entry is not None:
        path = tmp_path / "offload.json"
        path.write_text(json.dumps(entry))
        flags += ["--lmcache-offload", str(path)]
        name += f" {path}"
    run = runfile.build(*_parsed(tmp_path, *flags))
    with pytest.raises((TypeError, ValueError)) as refused:
        compass_run._offload_tier(r.config, run)
    served(monkeypatch, r.config)

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [f"{name}: {state}"]
    assert f"  refused: {refused.value}\n" in out


@pytest.mark.parametrize(
    "model_type, kv, fitted",
    [
        ("qwen3_5_text", {}, {}),
        ("qwen3", {}, {"chunk_size": 16}),
        ("qwen3", {}, {"tensor_parallel_size": 2}),
        ("qwen3", {"lmcache.local_cpu": False}, {}),
        ("qwen3", {"lmcache.save_unfull_chunk": True}, {}),
    ],
    ids=["state layout", "chunk size", "TP width", "no CPU tier", "unmodelled"],
)
def test_check_meets_the_offload_shell_s_refusal_of_the_deployment_s_lmcache_config(
    tmp_path, monkeypatch, capsys, model_type, kv, fitted
):
    text = json.loads(CONFIG_JSON.read_text())["text_config"]
    hf = PretrainedConfig.from_dict(text | {"model_type": model_type})
    kv = OFFLOADS | {"lmcache.chunk_size": OFFLOAD_ENTRY["chunk_size"]} | kv
    r = deployment(tmp_path, kv_transfer_config=kv, hf_config=hf)
    entry, path = OFFLOAD_ENTRY | fitted, tmp_path / "offload.json"
    path.write_text(json.dumps(entry))
    with monkeypatch.context() as m, pytest.raises(ValueError) as refused:
        m.setattr(clock, "_installed", object())
        m.setattr(compass_run, "spec", lambda: {"lmcache_offload": entry})
        OffloadShell(r.config)
    served(monkeypatch, r.config)

    flags = ["--step-stub", stub(tmp_path), "--lmcache-offload", str(path)]
    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [f"lmcache_offload entry {path}: stale"]
    assert f"  refused: {refused.value}\n" in out


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

    argv = [
        "--step-stub",
        stub(tmp_path),
        *platforms.flags(tmp_path, DOCUMENT, tmp_path),
    ]
    with pytest.raises(SystemExit) as refused:
        drive_run.main([*argv, "--store", str(tmp_path), "--", "--model", "m"])
    assert refused.value.code == 2
