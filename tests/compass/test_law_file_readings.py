# SPDX-License-Identifier: MIT
"""A law file carries the memory readings its law's engine config was measured
with, each stating `assumed`, which a run reads where its store has no entry at
their key, through the run file `drive.runfile` builds: keyed as a store entry
is, at the ATOM digest of the law's host key. An entry at the key that does not
read refuses, readings at another key refuse, naming the fields they differ in,
and readings at another ATOM digest name both. Host laws fitted on another host
CPU serve a run, which predicts the host they timed. ``python -m
atom.compass.run.memory`` writes a store entry's readings into a law file.
"""

import json
import shutil

import pytest
from test_drive_run import stub
from test_drive_runfile import LAWS, build, install
from test_drive_runfile_law_file import law_file
from test_memory_readings import DOCUMENT, READINGS
from test_run_check_side import COVERAGE, pair, side_args
from test_run_check_side import checked as side_checked
from test_run_lacking import checked, lacked, served
from test_run_step_grid import PLATFORM, deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Kind,
    keyed_platform,
    machine_platform,
    member_name,
)
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.backends.json_file import read_json, write_json
from atom.compass.run import memory
from atom.compass.run.step_grid import keyed
from atom.compass.spec import MachineSpec

MACHINE = MachineSpec.from_mapping(DOCUMENT)
ASSUMED = "measured at another ATOM digest, carried by this law"


@pytest.fixture
def bare(tmp_path):
    """A deployment whose store holds no memory readings."""
    r = deployment(tmp_path)
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
    return r


def law_member(r, **key) -> dict:
    """A law file's readings of `r`'s, at their key moved by `key`."""
    at = memory.key_of(r.config, MACHINE).values | key
    ranks = memory.topology(r.config).ranks()
    members = {
        member_name(memory.STEM, rank, memory.EXTENSION): READINGS[1]
        | {"assumed": ASSUMED}
        for rank in ranks
    }
    return {"key": at, "members": members}


def carried(r, atom=None, **key):
    """The run file's readings of a law file carrying `r`'s, at their key
    moved by `key`, its host key at ATOM digest `atom` or this tree's."""
    law = {
        memory.LAW_MEMBER: law_member(r, **key),
        "host": {"key": {"atom": atom or compass_run.atom_digest()}},
    }
    return [memory.carried_by(law, "law.json")]


def load(tmp_path, r, carried):
    return memory.load(str(tmp_path / "store"), r.config, MACHINE, 0, carried)


def test_a_store_with_no_readings_reads_the_law_file_s_assumed(tmp_path, bare):
    read = load(tmp_path, bare, carried(bare))

    assert read.assumed == ASSUMED
    assert read.values["activation_bytes"] == READINGS[1]["activation_bytes"]


def test_a_store_with_no_readings_and_no_law_file_s_refuses(tmp_path, bare):
    with pytest.raises(ArtifactRefusal, match="no entry for memory_readings"):
        load(tmp_path, bare, ())


def test_a_store_s_readings_come_before_a_law_file_s(tmp_path):
    r = deployment(tmp_path)

    assert load(tmp_path, r, carried(r)).assumed is None


@pytest.mark.parametrize(
    "broken, named",
    [("schema_version", "states schema_version 0"), ("member", "which is not there")],
)
def test_a_store_entry_that_does_not_read_refuses_beside_a_law_file_s(
    tmp_path, broken, named
):
    r = deployment(tmp_path)
    key = memory.key_of(r.config, MACHINE)
    entry = ArtifactStore(tmp_path / "store").directory_for(key) / ENTRY_FILE
    if broken == "schema_version":
        entry.write_text(json.dumps(json.loads(entry.read_text()) | {broken: 0}))
    else:
        next(entry.parent.glob(f"{memory.STEM}*")).unlink()

    with pytest.raises(ArtifactRefusal, match=named):
        load(tmp_path, r, carried(r))


def test_a_run_file_of_a_law_file_hands_the_runner_its_readings(
    tmp_path, monkeypatch, bare
):
    path, _ = law_file(tmp_path, bare, **{memory.LAW_MEMBER: law_member(bare)})
    run = build(tmp_path, "--step-linear", path, "--host-laws", path)

    install(tmp_path, monkeypatch, run, bare)

    assert bare.compass_memory["assumed"] == ASSUMED


def test_readings_at_another_key_refuse_naming_the_field(tmp_path, bare):
    values = memory.key_of(bare.config, MACHINE).values
    engine = json.loads(values["engine_config"])
    engine["max_num_seqs"] += 1
    moved = json.dumps(engine, sort_keys=True)

    with pytest.raises(ArtifactRefusal, match=r"law.json carries differ in tp\."):
        load(tmp_path, bare, carried(bare, tp=values["tp"] + 1))
    with pytest.raises(
        ArtifactRefusal, match=r"differ in engine_config\.max_num_seqs\."
    ):
        load(tmp_path, bare, carried(bare, engine_config=moved))


def test_readings_at_another_atom_digest_refuse_naming_both(tmp_path, bare):
    here = compass_run.atom_digest()

    with pytest.raises(ArtifactRefusal) as refused:
        load(tmp_path, bare, carried(bare, atom="0" * 64))
    assert f"ATOM digest {'0' * 64}" in str(refused.value)
    assert f"this tree's is {here}" in str(refused.value)


def test_host_laws_fitted_on_another_cpu_serve_the_run(tmp_path):
    r = deployment(tmp_path)
    platform = machine_platform(MACHINE)
    key = compass_run.host_key(r.config) | keyed_platform(Kind.HOST_LAWS, platform)
    host = {"laws": {"decode": {"E": [1e-4, 0.0, 0.0]}}}

    compass_run.refuse_unfitted_host(
        host | {"key": key | {"host_cpu": "another CPU"}}, r.config, platform
    )
    with pytest.raises(ValueError, match=r"another \['model'\]"):
        compass_run.refuse_unfitted_host(
            host | {"key": key | {"model": "another model"}}, r.config, platform
        )


@pytest.mark.parametrize(
    "moved", [{}, {"kv_cache_dtype": "fp8"}], ids=["same", "other"]
)
def test_the_writer_carries_a_store_s_entry_into_the_law_file(tmp_path, capsys, moved):
    r = deployment(tmp_path)
    path, token = tmp_path / "law.json", memory.key_of(r.config, MACHINE).token()
    atom = {"key": {"atom": compass_run.atom_digest()}}
    write_json({"engine_config": keyed(r.config) | moved, "host": atom}, path)
    argv = [str(path), "--store", str(tmp_path / "store")]
    argv += ["--token", token, "--assumed", ASSUMED]

    if moved:
        with pytest.raises(SystemExit):
            memory.main(argv)
        assert f"measured under another {list(moved)}" in capsys.readouterr().err
        return
    memory.main(argv)
    shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)

    given = [memory.carried_by(read_json(path), str(path))]
    assert load(tmp_path, r, given).assumed == ASSUMED


def test_check_lacks_no_readings_a_law_file_of_another_cpu_carries(
    tmp_path, monkeypatch, capsys, bare
):
    served(monkeypatch, bare.config)
    key = compass_run.host_key(bare.config) | keyed_platform(Kind.HOST_LAWS, PLATFORM)
    host = {"key": key | {"host_cpu": "another CPU"}, "laws": LAWS}
    path = tmp_path / "law.json"
    write_json({"host": host, memory.LAW_MEMBER: law_member(bare)}, path)

    assert (
        checked(tmp_path, "--step-stub", stub(tmp_path), "--host-laws", str(path)) == 0
    )
    assert capsys.readouterr().out.startswith("nothing lacking")


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_crs005_side_reads_the_readings_its_law_file_carries(
    tmp_path, monkeypatch, capsys, side
):
    # The side's environment alone, and AITER_LOG_LEVEL, which its check sets,
    # restored after it.
    for name in PERFORMANCE_KEYS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    atom_args = side_args(tmp_path, monkeypatch, side)

    # The store holds the platform's readings and no memory readings. The
    # prefill side, a producer, schedules no decode rung its law could lack.
    lacking = [COVERAGE] if side == "decode" else []
    assert side_checked(pair(tmp_path), side, atom_args) == (1 if lacking else 0)

    assert lacked(capsys.readouterr().out) == lacking
