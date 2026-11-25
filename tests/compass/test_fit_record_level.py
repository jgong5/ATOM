# SPDX-License-Identifier: MIT
"""A record whose engine and host key files name no compilation level, as every
file did before `keyed` named it, keyed at the `--level` its run.json's ATOM
arguments name by `compass fit steps`, `compass fit host`, `step_linear_fit`
and `measure steps`, on `test_fit_steps.py`'s record directories; and read as
`keyed` names the level of a server under those arguments. No device: aiter's
arch probe answers a declared arch.
"""

import json
import re

import pytest
from test_backend_step_linear_fit import platform_arg
from test_fit_host import host_laws_key
from test_fit_samples import edit
from test_fit_stated_engine import _engine
from test_fit_steps import (  # noqa: F401
    ENGINE,
    HOST,
    LINEAR,
    clients,
    fit,
    law_key,
    member,
    planted_roots,
    record,
)
from test_key_index_cache_dtype import QWEN, served
from test_memory_readings import DOCUMENT
from test_parity_served import SAMPLING

from atom.compass.backends import step_linear_fit
from atom.compass.backends.step_labels import digest
from atom.compass.bench import measure
from atom.compass.run.step_grid import LEVEL, keyed


def ran_at(directory, *level):
    """`directory`, its run.json naming ATOM arguments after --, `level` among them."""
    atom = ["--", "--model", "m", "-tp", "1", *level]
    edit(directory / "run.json", lambda r: r.update(argv=r["argv"] + atom))
    return directory


@pytest.mark.parametrize(
    "level, keyed",
    [(("--level", "0"), {"level": 0}), (("--level", "3"), {}), ((), {})],
    ids=["level-0", "level-3", "no-level"],
)
def test_every_fit_keys_its_law_at_the_level_the_records_ran(tmp_path, level, keyed):
    a, b = (ran_at(record(tmp_path / n, s), *level) for n, s in (("a", 19), ("b", 20)))
    fit(tmp_path, "fit host", a, b, *clients(a, b))
    fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    out = tmp_path / "law.json"
    step_linear_fit.main(
        [*map(str, paths), "--form", "history", *platform_arg(DOCUMENT)]
        + ["--out", str(out)]
    )

    engine = {**ENGINE, **keyed}
    written = json.loads(out.read_text())
    assert written["engine_config"] == written["host"]["key"]["engine"] == engine
    law = member(
        tmp_path / "store", law_key(tmp_path / "store", engine=engine), "params"
    )
    assert law["engine_config"] == engine
    host_key = host_laws_key({**HOST, "engine": engine}, tmp_path)
    assert member(tmp_path / "store", host_key, "params")["key"]["engine"] == engine


@pytest.mark.parametrize(
    "command", ["fit host", "fit steps --backend step-linear", "step_linear_fit"]
)
def test_an_engine_file_omitting_the_level_a_record_ran_at_refuses(tmp_path, command):
    a = ran_at(record(tmp_path / "a", 19), "--level", "0")
    engine = _engine(tmp_path)

    named = f"{a / 'steps/dp0.engine.json'} contradicts {engine} in ['level']"
    with pytest.raises((SystemExit, ValueError), match=re.escape(named)):
        if command == "step_linear_fit":
            step_linear_fit.main(
                [str(a / "steps/dp0.jsonl"), *platform_arg(DOCUMENT)]
                + ["--out", str(tmp_path / "law.json"), "--engine", str(engine)]
            )
        else:
            fit(tmp_path, command, a, *clients(a), "--engine", engine)
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "law.json").exists()


def test_measure_steps_reads_a_grid_record_at_the_level_it_ran(tmp_path):
    a = ran_at(record(tmp_path / "a", 19), "--level", "0")

    def read(engine):
        return measure._read(str(a), None, [a / "client.json"], None, engine, SAMPLING)

    stated = json.loads((a / "steps/dp0.engine.json").read_text())
    assert read({**ENGINE, "level": 0})[2] == stated
    named = f"{a} ran under another engine than ATOM's arguments resolve: level 0 there"
    with pytest.raises(ValueError, match=re.escape(named)):
        read(ENGINE)


@pytest.mark.parametrize(
    "args",
    [[], ["--level", "3"], ["--level", "0"], ["--level=1"]],
    ids=["no-level", "level-3", "level-0", "level=1"],
)
def test_a_record_reads_the_level_keyed_names_for_its_server(
    tmp_path, monkeypatch, args
):
    config = served(tmp_path, monkeypatch, "gfx950", QWEN, *args)
    a = ran_at(record(tmp_path / "a", 19), *args)

    named = {k: v for k, v in keyed(config).items() if k == LEVEL}
    assert step_linear_fit.recorded_level(a / "steps/dp0.jsonl") == named
