# SPDX-License-Identifier: MIT
"""`compass fit steps`, `compass fit host` and `step_linear_fit` pool records
whose capture ladders are non-empty prefixes of the longest among them, and key
the law and the host entry there, with or without `--engine`; the runner then
installs both at a prefix point. Any other ladder refuses, naming it.

Records are `test_fit_steps.py`'s, each at its point's ladder: a decode batch
past the ladder's top rung runs eager, as ATOM runs it. No device.
"""

import json
import re

import pytest
from test_backend_step_linear_fit import _price, _steps, _timed_record, platform_arg
from test_drive_runfile import build, install
from test_fit_host import host_laws_key
from test_fit_steps import (  # noqa: F401
    ENGINE,
    HOST,
    LINEAR,
    as_record,
    clients,
    fit,
    law_key,
    member,
    planted_roots,
)
from test_memory_readings import DOCUMENT
from test_run_step_grid import deployment

from atom.compass import run as compass_run
from atom.compass.backends import StepLinearBackend, step_linear_fit
from atom.compass.backends.step_labels import digest
from atom.compass.run.step_grid import BOUNDS

#: An arm's two points' ladders, the first a prefix of the second.
SHORT, LONG = [1, 2], [1, 2, 3, 4, 5, 6, 7, 8]
COMMANDS = ["fit steps", "fit host", "step_linear_fit"]
#: The run file flags naming a law and host laws, and the fits' readable copies.
FLAGS = ("--step-linear", "--host-laws")
OUT = ("step_linear_law.json", "host_laws.json")


def at(ladder) -> dict:
    """`HOST` at the capture ladder `ladder`."""
    return {**HOST, "engine": {**ENGINE, "ladder": ladder}}


def point(directory, seed, key):
    """A record of `_steps(seed)` under the host key `key`, each decode batch
    past its ladder's top rung run eager."""
    top = max(key["engine"]["ladder"], default=0)
    steps = [
        (rows, None, _price(rows, None)) if rung and rung > top else (rows, rung, s)
        for rows, rung, s in _steps(seed)
    ]
    (directory / "steps").mkdir(parents=True)
    _timed_record(directory / "steps/dp0.jsonl", steps, key=key)
    return as_record(directory, key)


def engine_file(tmp_path, ladder):
    """An engine file stating `ENGINE` at `ladder`."""
    path = tmp_path / "engine.json"
    path.write_text(json.dumps({**at(ladder)["engine"], **dict.fromkeys(BOUNDS, 64)}))
    return path


def run(tmp_path, command, *records, flags=()):
    """`command` on `records`, publishing to `tmp_path/store`, or for
    `step_linear_fit` writing `tmp_path/law.json`."""
    if command != "step_linear_fit":
        backend = LINEAR if command == "fit steps" else ()
        return fit(tmp_path, command, *backend, *records, *clients(*records), *flags)
    paths = sorted((d / "steps/dp0.jsonl" for d in records), key=digest)
    step_linear_fit.main(
        [*map(str, paths), "--form", "history", *platform_arg(DOCUMENT)]
        + ["--out", str(tmp_path / "law.json"), *map(str, flags)]
    )


@pytest.mark.parametrize("stated", [False, True], ids=["own", "--engine"])
def test_records_at_a_prefix_fit_one_law_and_host_entry_keyed_at_the_longest(
    tmp_path, stated, capsys
):
    a, b = point(tmp_path / "a", 19, at(SHORT)), point(tmp_path / "b", 20, at(LONG))
    flags = ("--engine", engine_file(tmp_path, LONG)) if stated else ()
    for command in COMMANDS:
        run(tmp_path, command, a, b, flags=flags)

    keyed = at(LONG)["engine"]
    law = member(
        tmp_path / "store", law_key(tmp_path / "store", engine=keyed), "params"
    )
    host = member(tmp_path / "store", host_laws_key(at(LONG), tmp_path), "params")
    written = json.loads((tmp_path / "law.json").read_text())
    assert law["engine_config"] == written["engine_config"] == keyed
    assert host["key"]["engine"] == written["host"]["key"]["engine"] == keyed
    # Rung 4 from b's steps alone; a ran those batches eager. The ladder's other
    # rungs, which no record reached, are filled (`step_linear_fit.fill_rungs`).
    for got in (law, written):
        fitted = sorted(k for k, v in got["decode"].items() if "filled" not in v)
        assert fitted == ["2", "4", "eager"]
        assert sorted(got["filled"]) == ["1", "3", "5", "6", "7", "8"]
        assert got["filled_rungs"]["decode"] == [1, 3, 5, 6, 7, 8]
    # Each law-writing command names the rungs it filled on stderr (#3796).
    warning = step_linear_fit.filled_warning(law)
    assert warning.startswith(
        "WARNING: no record reached decode rungs 1, 3, 5, 6, 7, 8"
    )
    assert capsys.readouterr().err.splitlines().count(warning) == 2
    assert len(law["runs"]) == 2


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    "ladders", [([1, 2, 4], [1, 2, 3]), ([], [1, 2, 4])], ids=["no-prefix", "eager"]
)
def test_a_ladder_no_prefix_of_the_longest_refuses_naming_it(
    tmp_path, command, ladders
):
    a = point(tmp_path / "a", 19, at(ladders[0]))
    b = point(tmp_path / "b", 20, at(ladders[1]))

    if command == "step_linear_fit":
        named = "the records ran under two engine configs, differing in ['ladder']"
    else:
        named = f"engine_config ladder: {ladders[0]} in {a}; {ladders[1]} in {b}"
    with pytest.raises((SystemExit, ValueError), match=re.escape(named)):
        run(tmp_path, command, a, b)
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "law.json").exists()


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    "ladders, stated",
    [((SHORT, [1, 2, 4]), LONG), ((SHORT, LONG), SHORT), (([1, 2, 4], LONG), LONG)],
    ids=["longer", "shorter", "no-prefix"],
)
def test_an_engine_file_off_the_records_longest_ladder_refuses_naming_it(
    tmp_path, command, ladders, stated
):
    a = point(tmp_path / "a", 19, at(ladders[0]))
    b = point(tmp_path / "b", 20, at(ladders[1]))
    engine = engine_file(tmp_path, stated)

    contradicts = re.escape(f" contradicts {engine} in ['ladder']")
    with pytest.raises((SystemExit, ValueError), match=contradicts):
        run(tmp_path, command, a, b, flags=("--engine", engine))
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "law.json").exists()


@pytest.mark.parametrize("where", ["store", "law file"])
def test_the_runner_installs_the_pooled_law_and_host_laws_at_a_prefix_point(
    monkeypatch, tmp_path, where
):
    keys = {
        n: compass_run.host_key(deployment(tmp_path, max_num_seqs=n).config)
        for n in (2, 8)
    }
    records = [point(tmp_path / f"c{n}", 17 + n, key) for n, key in keys.items()]
    if where == "store":
        run(tmp_path, "fit steps", *records)
        run(tmp_path, "fit host", *records)
        out = [json.loads((tmp_path / "out" / f).read_text()) for f in OUT]
        flags = [x for f, entry in zip(FLAGS, out) for x in (f, entry["token"])]
    else:
        run(tmp_path, "step_linear_fit", *records)
        flags = [x for f in FLAGS for x in (f, str(tmp_path / "law.json"))]
    r = deployment(tmp_path, max_num_seqs=4)
    built = build(tmp_path, *flags)

    install(tmp_path, monkeypatch, built, r)

    assert isinstance(r.compass_backend, StepLinearBackend)
    assert r.compass_host.laws == built["host"]["laws"]
    assert built["step_linear"]["engine_config"] == keys[8]["engine"]
    assert built["host"]["key"]["engine"] == keys[8]["engine"]
