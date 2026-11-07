# SPDX-License-Identifier: MIT
"""`compass run` prices by a law file, as `step_linear_fit` writes one and
`scripts/compass/laws/` holds them: its step law at the machine spec and engine
config the file states, its host laws at the host key the file states.

A law re-keyed to this tree's `atom` digest runs here, where the `host_laws`
entry its records' fit published, keyed by their digest, refuses at engine
start. The run file is read back through `run.spec` and installed by
`run.runner`, as `test_drive_runfile` installs a store entry's. A law file
names no fit inputs, so grading checks none and takes its run; a store entry's
report names them, and grading refuses its run where one served a trace.
"""

import hashlib
import json
import re

import pytest
from test_backend_step_linear import law
from test_drive_runfile import LAWS, REPORT, build, install, published, step_linear
from test_run_step_grid import SPEC_DIGEST, deployment, host_laws_key

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind
from atom.compass.backends import StepLinearBackend
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve
from atom.compass.run.step_grid import keyed
from atom.compass.validate import runs

#: The `atom` digest of the tree a law's records ran at, before its re-key.
RECORDS = "829f117e" * 8


def law_file(tmp_path, r, name="law.json", **written):
    """A law file fitted under `r`'s engine config, its host key re-keyed to
    this tree's `atom` digest; `written` replaces its fields."""
    host = {**compass_run.host_key(r.config), "machine_spec": SPEC_DIGEST}
    document = {
        **law(),
        "engine_config": keyed(r.config),
        "host": {"key": host, "laws": LAWS},
        "host_fit": {"unfitted": {}},
        **written,
    }
    (tmp_path / name).write_text(json.dumps(document))
    return str(tmp_path / name), document


def test_a_law_rekeyed_here_lends_its_host_laws_where_its_records_entry_refuses(
    monkeypatch, tmp_path
):
    r = deployment(tmp_path)
    path, document = law_file(tmp_path, r)
    records = {"key": {**document["host"]["key"], "atom": RECORDS}, "laws": LAWS}
    key = {**host_laws_key(r), "atom_src": RECORDS}
    fitted = published(tmp_path, r, Kind.HOST_LAWS, key, records).key.token()
    law_token = step_linear(tmp_path, r).key.token()
    refused = build(tmp_path, "--step-linear", law_token, "--host-laws", fitted)
    with pytest.raises(ValueError, match=r"fitted under another \['atom'\]"):
        install(tmp_path, monkeypatch, refused, r)

    run = build(tmp_path, "--step-linear", law_token, "--host-laws", path)
    install(tmp_path, monkeypatch, run, r)

    assert run["host"] == document["host"] and r.compass_host.laws == LAWS


def test_a_law_file_stands_in_for_both_tokens(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    path, document = law_file(tmp_path, r)
    run = build(tmp_path, "--step-linear", path, "--host-laws", path)

    install(tmp_path, monkeypatch, run, r)

    assert isinstance(r.compass_backend, StepLinearBackend)
    assert r.compass_host.laws == LAWS
    del document["host"], document["host_fit"]
    assert run["step_linear"] == document
    stated = json.dumps(document, sort_keys=True).encode()
    digest = "sha256:" + hashlib.sha256(stated).hexdigest()
    assert run["backend"] == {"digest": digest, "provenance": None}


def test_a_law_file_with_no_host_laws_refuses_naming_it(tmp_path):
    path, document = law_file(tmp_path, deployment(tmp_path))
    del document["host"], document["host_fit"]
    (tmp_path / "law.json").write_text(json.dumps(document))

    with pytest.raises(ValueError, match=re.escape(f"--host-laws {path} states no")):
        build(tmp_path, "--step-linear", path, "--host-laws", path)


def test_two_law_files_install_each_at_its_own_engine_config(monkeypatch, tmp_path):
    flags, runners = [], []
    for i, block in enumerate((16, 32)):
        r = deployment(tmp_path, kv_cache_block_size=block)
        path, _ = law_file(tmp_path, r, f"law{i}.json", provenance=f"law {i}")
        flags += ["--step-linear", path, "--host-laws", path]
        runners.append(r)
    run = build(tmp_path, *flags)

    assert run["backend"]["provenance"] == {"entries": [None, None], "inputs": []}
    for i, r in enumerate(runners):
        install(tmp_path, monkeypatch, run, r)
        assert r.compass_backend.law.fitted_on == f"law {i}"
        assert r.compass_host.laws == LAWS


#: The traces the report of each store entry `test_drive_runfile` publishes names.
FITTED = [t for i in REPORT["provenance"] for t in i["traces"]]


@pytest.mark.parametrize(
    "given, served, why",
    [
        (["token"], ["trace-z"], []),
        (["token"], FITTED, [f"its backend was fitted on traces it served: {FITTED}"]),
        (["file"], FITTED, []),
        (["file", "file"], FITTED, []),
    ],
)
def test_grading_takes_a_law_files_run_and_refuses_a_fit_on_a_trace_it_served(
    tmp_path, given, served, why
):
    """Each step law `given` by store token or law file, two as a deployment's
    sides run them, each at its own engine config; the run served `served`."""
    flags = []
    for i, (form, block) in enumerate(zip(given, (16, 32))):
        r = deployment(tmp_path, kv_cache_block_size=block)
        path, _ = law_file(tmp_path, r, f"law{i}.json")
        step = step_linear(tmp_path, r).key.token() if form == "token" else path
        flags += ["--step-linear", step, "--host-laws", path]
    (tmp_path / serve.RUN_FILE).write_text(json.dumps(build(tmp_path, *flags)))
    (tmp_path / serve.RUN_JSON).write_text("{}")

    drive_run.write_grading(tmp_path, None)

    written = json.loads((tmp_path / serve.RUN_JSON).read_text())
    written["refusals"] = {"count": 0, "steps": 1}
    assert runs._not_graded(written, served) == why
