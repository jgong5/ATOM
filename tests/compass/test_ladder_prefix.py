# SPDX-License-Identifier: MIT
"""A step linear law, host laws and a step grid bundle serve a run whose capture
ladder is a non-empty prefix of their own: the runner installs them, `compass
run --check` holds them, the reuse check finds them and the grader picks them.
A ladder that is no prefix, or an eager run under a replayed entry, refuses
naming the ladder.

Each entry is published at `test_run_step_grid`'s deployment, whose capture
sizes `LADDER` its `max_num_seqs` clips; a run takes a lower `max_num_seqs`, so
its ladder is a prefix of the entry's, as ATOM clips one
`--cudagraph-capture-sizes` at each point of an arm.
"""

import json
import re

import pytest
from test_advice_reuse import (  # noqa: F401  fixtures
    advise,
    at_head,
    law_file,
    searched,
    stored,
    verdicts,
)
from test_advice_reuse import law as reuse_law
from test_backend_step_grid import LADDER
from test_backend_step_linear import TRIANGLE, law
from test_drive_runfile import LAWS, build, install, published
from test_memory_readings import META_8B, publish_readings
from test_run_lacking import checked, lacked, served
from test_run_step_grid import (
    NEW_KINDS,
    PLATFORM,
    deployment,
    host_laws_key,
    publish,
    run_file,
)

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind, keyed_platform
from atom.compass.backends import StepGridBackend, StepLinearBackend
from atom.compass.run.step_grid import keyed
from atom.compass.validate import __main__ as validate

#: A decode vector at each rung of `LADDER` clipped at 8, and no eager one.
RUNGS = {str(rung): TRIANGLE for rung in (1, 2, 4, 8)}
#: A run whose ladder is no prefix of the entries': rung 3 in place of 4.
NO_PREFIX = {"capture_sizes": [1, 2, 3, 4], "max_num_seqs": 4}


def entries(tmp_path, max_num_seqs: int) -> list[str]:
    """The flags naming a step linear law with `RUNGS`' vectors and host laws,
    both fitted on the deployment at `max_num_seqs`."""
    r = deployment(tmp_path, max_num_seqs=max_num_seqs)
    params = law(**RUNGS)
    del params["decode"]["eager"]
    key = NEW_KINDS[Kind.STEP_LINEAR_LAW]
    key |= {"engine_config": json.dumps(keyed(r.config), sort_keys=True)}
    step = published(tmp_path, r, Kind.STEP_LINEAR_LAW, key, params)
    host = compass_run.host_key(r.config) | keyed_platform(Kind.HOST_LAWS, PLATFORM)
    params = {"key": host, "laws": LAWS}
    hosts = published(tmp_path, r, Kind.HOST_LAWS, host_laws_key(r), params)
    return ["--step-linear", step.key.token(), "--host-laws", hosts.key.token()]


@pytest.mark.parametrize(
    "run, named",
    [({"max_num_seqs": 4}, None), (NO_PREFIX, [1, 2, 3, 4])]
    + [({"max_num_seqs": 4, "enforce_eager": True}, [])],
    ids=["prefix", "no-prefix", "eager"],
)
def test_a_run_at_a_prefix_of_the_law_s_ladder_installs_it_and_no_other_does(
    monkeypatch, tmp_path, run, named
):
    built = build(tmp_path, *entries(tmp_path, 8))
    here = deployment(tmp_path, **run)

    if named is None:
        install(tmp_path, monkeypatch, built, here)
        assert isinstance(here.compass_backend, StepLinearBackend)
        assert here.compass_host.laws == LAWS
        return
    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, built, here)
    assert f"ladder [1, 2, 4, 8] in the law and {named} here" in str(e.value)


def test_two_entries_serving_one_run_refuse_naming_what_differs(monkeypatch, tmp_path):
    tokens = entries(tmp_path, 8)[:2] + entries(tmp_path, 48)[:2]

    with pytest.raises(
        ValueError,
        match=r"^2 step_linear entries were fitted under this engine config, and "
        r"they differ in \['engine_config.ladder'\]$",
    ):
        install(
            tmp_path,
            monkeypatch,
            build(tmp_path, *tokens),
            deployment(tmp_path, max_num_seqs=4),
        )


@pytest.mark.parametrize(
    "run, lacking, rungs",
    [
        ({"max_num_seqs": 8}, [], []),
        ({"max_num_seqs": 4}, [], []),
        # Batches 5 and 6 run eager, where the law's records replay them at 8.
        ({"max_num_seqs": 6}, ["step linear law: stale (coverage)"], ["eager"]),
        (
            NO_PREFIX,
            ["step linear law: stale", "step linear law: stale (coverage)"]
            + ["host laws: stale"],
            ["3"],
        ),
    ],
)
def test_check_holds_the_entries_at_a_prefix_naming_the_rungs_the_run_schedules(
    tmp_path, monkeypatch, capsys, run, lacking, rungs
):
    flags = entries(tmp_path, 8)
    served(monkeypatch, deployment(tmp_path, **run).config)

    assert checked(tmp_path, *flags) == (1 if lacking else 0)

    out = capsys.readouterr().out
    assert lacked(out) == lacking
    assert re.findall(r"for rungs? (.*), which this configuration", out) == rungs
    if run is NO_PREFIX:
        assert "ladder [1, 2, 4, 8] in the law and [1, 2, 3, 4] here" in out
        assert "engine.ladder" in out


@pytest.mark.parametrize("where", ["law file", "store"])
def test_the_reuse_check_finds_a_law_and_host_laws_at_a_prefix_of_their_ladder(
    tmp_path, searched, where  # noqa: F811
):
    # The law at the searched config's ladder, [1, 2, 4, 8]; then a run at 4.
    if where == "law file":
        found = law_file(tmp_path, reuse_law(searched))
    else:
        found = stored(tmp_path, searched)
    searched.config.max_num_seqs = 4
    publish_readings(tmp_path / "store", searched.config)

    judged = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"], v.get("law")) for v in judged] == [
        ("reuse", None, found)
    ] * 2


def test_the_grader_picks_the_law_whose_ladder_its_record_s_is_a_prefix_of(tmp_path):
    here = keyed(deployment(tmp_path, max_num_seqs=4).config)
    (tmp_path / "steps").mkdir()
    engine = json.dumps({**here, "max_num_seqs": 4})
    (tmp_path / "steps" / "dp0.engine.json").write_text(engine)
    laws = [
        {"engine_config": {**here, "ladder": ladder}, "provenance": str(ladder)}
        for ladder in ([1, 2, 3, 4], [1, 2, 4, 8])
    ]

    assert validate._law({"step_linear": laws}, tmp_path) == laws[1]

    laws[0]["engine_config"]["ladder"] = LADDER
    with pytest.raises(ValueError, match=r"differ in \['engine_config.ladder', "):
        validate._law({"step_linear": laws}, tmp_path)


def test_a_run_at_a_prefix_of_a_bundle_s_ladder_loads_it_at_the_bundle_s(
    monkeypatch, tmp_path
):
    publish(tmp_path, deployment(tmp_path))
    run_file(tmp_path, monkeypatch)
    r = deployment(tmp_path, max_num_seqs=16)

    compass_run.runner(r, META_8B)

    assert isinstance(r.compass_backend.first, StepGridBackend)
