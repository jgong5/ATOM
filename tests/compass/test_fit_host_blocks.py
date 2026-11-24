# SPDX-License-Identifier: MIT
"""`compass fit host --decode-client-run`: a client run whose host samples are
its decode drains alone, the steps of its span in which no rank prefills
(`fit.host.drain_cut`).

The record is `test_fit_steps.record`'s, alternating prefill and decode steps
but at `LONG`, a decode, served by two client runs one after another:
`steady.json` sends the requests of the steps before `SPLIT`, `decode.json`
the rest.
"""

import dataclasses
import json

import pytest
from test_backend_step_linear_fit import _steps, _timed_record
from test_fit_host import host_laws_key
from test_fit_steps import (  # noqa: F401
    HOST,
    PLATFORM,
    as_record,
    fit,
    member,
    planted_roots,
)

from atom.compass.artifacts import Kind, keyed_platform
from atom.compass.backends.step_linear_fit import fit_host, label_records
from atom.compass.fit.host import drain_cut
from atom.compass.validate.runs import SIDES

SPLIT = 120
#: A step of the decode client run's span that decodes where the others
#: prefill: its drain runs from the step before it to the step after.
LONG = SPLIT + 2


def _two_runs(directory):
    steps = directory / "steps"
    steps.mkdir(parents=True)
    shapes = _steps(19)
    shapes[LONG] = shapes[LONG - 1]
    _timed_record(steps / "dp0.jsonl", shapes, idle_after=(SPLIT - 1,), key=HOST)
    as_record(directory)
    sent = json.loads((directory / "client.json").read_text())["requests"]
    lines = [json.loads(x) for x in (steps / "dp0.jsonl").read_text().splitlines()]
    late = {f"r{r[0]}" for line in lines[SPLIT:] for r in line["batch"]}
    for name, part in (("steady", sent.keys() - late), ("decode", late)):
        exported = {"requests": {k: sent[k] for k in part}}
        (directory / f"{name}.json").write_text(json.dumps(exported))
    return directory


def _drained(samples) -> list:
    """`samples` less the decode client run's prefill steps, its even ones
    but `LONG`."""
    return [s for s in samples if s.step < SPLIT or s.step % 2 or s.step == LONG]


def test_a_decode_client_run_keeps_only_its_drains_and_their_dummies(tmp_path):
    a = _two_runs(tmp_path / "a")
    samples = label_records([a / "steps/dp0.jsonl"])[0]
    dummies = ((2, None, {}, {}), (SPLIT, None, {}, {}))
    samples = [dataclasses.replace(s, dummy_host=dummies) for s in samples]
    kept, report = drain_cut(samples, [(str(a), str(a / "decode.json"))])

    assert kept == [
        dataclasses.replace(s, dummy_host=dummies[:1]) for s in _drained(samples)
    ]
    # Each odd step of the span is a drain of its own, between two prefills,
    # but the three from LONG - 1 to LONG + 1.
    drained = [s for s in _drained(samples) if s.step >= SPLIT]
    assert report == [
        {
            "record": str(a),
            "export": str(a / "decode.json"),
            "span": [SPLIT, 239],
            "drains": 59,
            "drain_steps": 61,
            "dropped": len(samples) - len(_drained(samples)),
            "kept": len(drained),
        }
    ]


def test_the_command_fits_a_decode_client_run_on_its_drains(tmp_path):
    a = _two_runs(tmp_path / "a")
    fit(tmp_path, "fit host", a, "--client-run", a, a / "steady.json")
    whole = member(tmp_path / "store", host_laws_key(HOST), "params")
    runs = ("--client-run", a, a / "steady.json", "--decode-client-run")
    fit(tmp_path / "d", "fit host", a, *runs, a, a / "decode.json")

    host = member(tmp_path / "d" / "store", host_laws_key(HOST), "params")
    laws, _ = fit_host(_drained(label_records([a / "steps/dp0.jsonl"])[0]))
    platform = keyed_platform(Kind.HOST_LAWS, PLATFORM)
    assert host == {**laws, "key": {**laws["key"], **platform}}
    assert host["laws"] != whole["laws"]
    report = member(tmp_path / "d" / "store", host_laws_key(HOST), "report")
    (run,) = report["decode_client_runs"]
    assert run["export"] == str(a / "decode.json") and run["span"] == [SPLIT, 239]


def test_a_decode_client_run_of_no_record_is_refused_by_its_own_flag(tmp_path):
    a = _two_runs(tmp_path / "a")
    runs = ("--client-run", a, a / "steady.json", "--decode-client-run")
    with pytest.raises(SystemExit, match="--decode-client-run names .*b, which"):
        fit(tmp_path, "fit host", a, *runs, tmp_path / "b", a / "decode.json")


def test_a_prefill_side_refuses_as_a_decode_client_run(tmp_path):
    for side in SIDES:
        (tmp_path / side).mkdir()
    with pytest.raises(ValueError, match="a prefill side, which decodes nothing"):
        drain_cut([], [(str(tmp_path / SIDES[0]), "client.json")])
