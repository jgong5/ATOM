# SPDX-License-Identifier: MIT
"""A `host_laws` key holds ``fit_src``, the sha256 of the laws without their
key (`fit.host.fit_src`): two fits of the same records whose laws differ are
two entries, a run takes either, and the reuse check takes the newer beside a
store law, skipping an entry whose key states none."""

import json

import pytest
from test_advice_reuse import (  # noqa: F401  fixtures
    advise,
    at_head,
    searched,
    stored,
    verdicts,
)
from test_fit_steps import clients, fit, member, planted_roots, record  # noqa: F401
from test_memory_readings import META_8B
from test_run_step_grid import deployment, run_file
from test_run_step_grid import publish as publish_deployment

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.fit import host as fit_host


def refits(tmp_path, monkeypatch, directory, changed_first=False) -> list[str]:
    """The tokens of two `compass fit host` runs over the record `directory`
    into `tmp_path`'s store, in order: one by this tree's fit code, one by code
    that prices every host law's intercept a microsecond higher, that one first
    when `changed_first`."""
    fitted = fit_host.fit_host

    def changed(samples):
        host, report = fitted(samples)
        for segments in host["laws"].values():
            for coefficients in segments.values():
                coefficients[0] += 1e-6
        return host, report

    tokens = []
    for code in (changed, fitted) if changed_first else (fitted, changed):
        monkeypatch.setattr(fit_host, "fit_host", code)
        fit(tmp_path, "fit host", directory, *clients(directory))
        out = json.loads((tmp_path / "out/host_laws.json").read_text())
        tokens.append(out["token"])
    return tokens


@pytest.mark.parametrize("taken", [0, 1], ids=["the first", "the second"])
def test_two_fits_of_the_same_records_whose_laws_differ_are_two_entries_a_run_takes(
    tmp_path, monkeypatch, taken
):
    r = deployment(tmp_path)
    publish_deployment(tmp_path, r)
    a = record(tmp_path / "a", 19, key=compass_run.host_key(r.config), trace="t")

    tokens = refits(tmp_path, monkeypatch, a)

    store = ArtifactStore(tmp_path / "store")
    keys = [store.key_from_token(Kind.HOST_LAWS, t) for t in tokens]
    hosts = [member(tmp_path / "store", k, "params") for k in keys]
    assert hosts[0]["laws"] != hosts[1]["laws"]
    assert [k.values["fit_src"] for k in keys] == list(map(fit_host.fit_src, hosts))
    host = hosts[taken]
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == host["laws"]


@pytest.mark.parametrize(
    "changed_first", [False, True], ids=["changed code second", "changed code first"]
)
def test_the_reuse_check_takes_the_newer_of_two_fits_beside_a_store_law(
    tmp_path, monkeypatch, searched, changed_first  # noqa: F811
):
    stored(tmp_path, searched, host=False)
    host_key = compass_run.host_key(searched.config)
    a = record(tmp_path / "a", 19, key=host_key, trace="t")
    tokens = refits(tmp_path, monkeypatch, a, changed_first)

    found = verdicts(advise(tmp_path), "law")

    assert [v["run"][1] for v in found] == [tokens[1]] * 2


def test_host_laws_whose_key_states_no_fit_src_are_not_taken(
    tmp_path, searched  # noqa: F811
):
    stored(tmp_path, searched)
    for entry in (tmp_path / "store" / Kind.HOST_LAWS.value).glob(f"*/{ENTRY_FILE}"):
        document = json.loads(entry.read_text())
        del document["key"]["fit_src"]
        entry.write_text(json.dumps(document))

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [("calibrate", 1)] * 2
    assert "has no host laws beside it" in found[0]["reason"]
