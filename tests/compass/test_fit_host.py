# SPDX-License-Identifier: MIT
"""`compass fit host`, run through `cli.main` on `test_fit_steps.py`'s record
directories, and the host laws it publishes handed to `run.runner` in a run
file on `test_run_step_grid.py`'s deployment."""

import json
import re

import numpy as np
import pytest
from test_backend_step_labels import pulled
from test_fit_samples import edit
from test_fit_steps import (  # noqa: F401
    DIGEST,
    ENGINE,
    HOST,
    clients,
    fit,
    member,
    planted_roots,
    record,
    served,
)
from test_memory_readings import META_8B
from test_run_step_grid import B37, decode, deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.artifacts import Key, Kind
from atom.compass.backends.step_labels import digest
from atom.compass.backends.step_linear_fit import fit_host, label_records
from atom.compass.drive.aiter_jit import BUILDS
from atom.compass.runner.overrides import OUT_SEGMENT, forward_shape


def host_laws_key(key) -> Key:
    return Key.of(
        Kind.HOST_LAWS,
        host_cpu=key["host_cpu"],
        atom_src=key["atom"],
        model=key["model"],
        engine_config=json.dumps(key["engine"], sort_keys=True),
        machine_spec=DIGEST,
    )


def test_the_host_laws_published_are_fit_hosts_and_a_run_file_carrying_them_runs(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key, trace="trace-a")
    b = record(tmp_path / "b", 20, key=key, trace="trace-b")
    fit(tmp_path, "fit host", a, b, *clients(a, b))

    host = member(tmp_path / "store", host_laws_key(key), "params")
    # In run digest order, as the command reads them: a least-squares fit's
    # last bits depend on the order of its samples.
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    laws, report = fit_host(label_records(paths)[0])
    assert host == {**laws, "key": {**laws["key"], "machine_spec": DIGEST}}
    assert member(tmp_path / "store", host_laws_key(key), "report") == {
        "provenance": served(key, {a: "trace-a", b: "trace-b"}),
        **report,
    }
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == host["laws"]


def test_a_record_without_its_host_key_refuses(tmp_path):
    a = record(tmp_path / "a", 19, keyed=False)

    named = f"{digest(a / 'steps/dp0.jsonl')} holds steps with no host labels under a host key"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit host", a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_a_record_whose_server_built_an_aiter_module_refuses_naming_it(tmp_path):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.update({BUILDS: ["mha_varlen_fwd"]}))

    named = f"{a} is not fitted: aiter ran a JIT build of ['mha_varlen_fwd'] while"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit host", a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_records_with_pull_stamps_give_every_kind_an_E_out_law_a_run_prices(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key)
    pulled(a / "steps/dp0.jsonl")
    fit(tmp_path, "fit host", a, *clients(a))

    host = member(tmp_path / "store", host_laws_key(key), "params")
    assert {k: OUT_SEGMENT in v for k, v in host["laws"].items()} == {
        k: True for k in ("decode", "prefill", "middle")
    }
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    batch = decode(B37)
    kind, x = forward_shape(batch)
    out_s = np.dot(host["laws"][kind][OUT_SEGMENT], x)
    assert r.forward(batch).out_s == pytest.approx(out_s) and out_s > 0


def test_a_dp_pair_s_host_laws_start_E_at_the_group_s_latest_pull(tmp_path):
    key = {**HOST, "engine": {**ENGINE, "data_parallel_size": 2}}
    a = record(tmp_path / "a", 19, key=key)
    steps = a / "steps"
    for suffix in (".key.json", ".engine.json"):
        (steps / f"dp1{suffix}").write_text((steps / f"dp0{suffix}").read_text())
    rank0 = [json.loads(x) for x in (steps / "dp0.jsonl").open()]
    rank1 = "".join(json.dumps(x | {"dp_rank": 1}) + "\n" for x in rank0)
    (steps / "dp1.jsonl").write_text(rank1)
    # The same forwards on both ranks; rank 1 always pulls after rank 0.
    paths = [steps / "dp0.jsonl", steps / "dp1.jsonl"]
    for path, parts in zip(paths, ((3,), (2,))):
        pulled(path, parts)
    fit(tmp_path, "fit host", a, *clients(a))

    laws = member(tmp_path / "store", host_laws_key(key), "params")["laws"]
    alone = [fit_host(label_records([p])[0])[0]["laws"] for p in paths]
    assert set(laws) == {"decode", "prefill", "middle"}
    for kind, law in laws.items():
        assert law["E"] == pytest.approx(alone[1][kind]["E"], rel=1e-9)
        assert law["E"] != pytest.approx(alone[0][kind]["E"], rel=1e-3)
