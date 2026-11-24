# SPDX-License-Identifier: MIT
"""A `host_laws` key holds the digest of the code that fitted it
(`run.fit_digest`): `compass fit host`'s module and the `step_linear_fit`
functions it fits with. Two fits of the same records by two fit codes are two
entries, a run takes either, and the reuse check looks host laws up at this
tree's fit code."""

import json
from pathlib import Path

import pytest
from test_advice_reuse import (  # noqa: F401  fixtures
    HOST_LAWS,
    advise,
    at_head,
    law,
    publish,
    searched,
    stored,
    verdicts,
)
from test_fit_steps import clients, fit, member, record
from test_memory_readings import META_8B
from test_run_step_grid import deployment, host_laws_key, run_file
from test_run_step_grid import publish as publish_deployment

import atom
from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.fit import host as fit_host

SLF, HOST = compass_run.FIT_CODE[1], compass_run.FIT_CODE[0]


def fit_code(root: Path, *edits) -> Path:
    """`root`, a package directory holding this tree's fit code with each
    ``(path, old, new)`` of `edits` made."""
    for path in compass_run.FIT_CODE:
        text = (Path(atom.__file__).parent / path).read_text()
        for at, old, new in edits:
            assert at != path or text.count(old) == 1
            text = text.replace(old, new) if at == path else text
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    return root


@pytest.mark.parametrize(
    "path, old, new, moves",
    [
        (SLF, "MAD_K = 3.5", "MAD_K = 3.0", True),
        (HOST, '("router_s", "kv_write_req_s")', '("router_s",)', True),
        (
            SLF,
            "def fill_rungs(law: dict, ladder)",
            "def fill_rungs(law, ladder)",
            False,
        ),
    ],
    ids=["host fit's outlier cut", "fit host's module", "a step law's rung fill"],
)
def test_the_digest_moves_with_the_code_fit_host_fits_with_alone(
    tmp_path, path, old, new, moves
):
    here = compass_run.fit_digest()
    assert compass_run.fit_digest(fit_code(tmp_path / "copy")) == here
    edited = compass_run.fit_digest(fit_code(tmp_path / "edited", (path, old, new)))
    assert (edited != here) is moves


def test_a_refit_of_the_same_records_by_changed_code_is_another_entry_a_run_takes(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish_deployment(tmp_path, r)
    a = record(tmp_path / "a", 19, key=compass_run.host_key(r.config), trace="t")

    def token() -> str:
        fit(tmp_path, "fit host", a, *clients(a))
        return json.loads((tmp_path / "out/host_laws.json").read_text())["token"]

    tokens = [token()]
    changed = fit_code(tmp_path / "changed", (SLF, "MAD_K = 3.5", "MAD_K = 3.0"))
    digest = compass_run.fit_digest(changed)
    monkeypatch.setattr(fit_host, "fit_digest", lambda: digest, raising=False)
    tokens.append(token())

    store = ArtifactStore(tmp_path / "store")
    keys = [store.key_from_token(Kind.HOST_LAWS, t) for t in tokens]
    assert [k.values["fit_src"] for k in keys] == [compass_run.fit_digest(), digest]
    host = member(tmp_path / "store", keys[1], "params")
    assert host["key"]["fit_src"] == digest
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == host["laws"]


def test_host_laws_beside_a_store_law_fitted_by_other_code_are_not_reused(
    tmp_path, searched  # noqa: F811
):
    stored(tmp_path, searched, host=False)
    params = {"key": law(searched)["host"]["key"], "laws": HOST_LAWS}
    key = host_laws_key(searched) | {"fit_src": "0" * 64}
    publish(tmp_path, searched, Kind.HOST_LAWS, key, params)

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [("calibrate", 1)] * 2
    assert "fitted by this tree's fit code" in found[0]["reason"]
