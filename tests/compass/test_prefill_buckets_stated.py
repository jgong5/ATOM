# SPDX-License-Identifier: MIT
"""Where a fit finds the prefill token-graph buckets a record's runner
recorded: its engine file, else, on a record whose engine file predates that
field, its host key's graph sets. Records stating them nowhere fit no `padding`
and the law says so; a record whose runner recorded none fits without either.

Both commands are run: `step_linear_fit`'s, and `compass fit steps`, which
publishes the same law. No device.
"""

import json

import pytest
from test_fit_steps import LINEAR, clients, fit, law_key, member, record
from test_memory_readings import spec_file
from test_step_linear_token_padding import BUCKETS

from atom.compass.backends import step_linear_fit
from atom.compass.backends.launch_features import GRAPHS
from atom.compass.backends.step_labels import digest


def _state(directory, engine=None, key=None) -> None:
    """Make `directory`'s record state `engine` as its engine file's buckets
    and `key` as its host key's graph sets' prefill buckets, each left
    unstated where None."""
    steps = directory / "steps"
    if engine is not None:
        held = json.loads((steps / "dp0.engine.json").read_text())
        held["prefill_buckets"] = engine
        (steps / "dp0.engine.json").write_text(json.dumps(held))
    if key is not None:
        held = json.loads((steps / "dp0.key.json").read_text())
        graphs = {"architecture": None, "token": {"prefill": key, "decode": []}}
        held["engine"] = {**held["engine"], GRAPHS: graphs}
        (steps / "dp0.key.json").write_text(json.dumps(held))


def _laws(tmp_path, a, b, *args) -> tuple[dict, dict]:
    """The law `compass fit steps` publishes on `a` and `b`, and the one
    `step_linear_fit`'s command writes, less its host laws."""
    fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))
    published = member(tmp_path / "store", law_key(), "params")
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    out = tmp_path / "law.json"
    step_linear_fit.main(
        [*map(str, paths), "--form", "history", "--machine", spec_file(tmp_path)]
        + ["--out", str(out), *args]
    )
    written = json.loads(out.read_text())
    return published, {
        k: v for k, v in written.items() if k not in ("host", "host_fit")
    }


def test_a_record_whose_engine_file_predates_its_buckets_reads_them_off_its_host_key(
    tmp_path,
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    for d in (a, b):
        _state(d, key=BUCKETS)
    published, written = _laws(tmp_path, a, b)

    assert published == written
    assert written["prefill"]["buckets"] == BUCKETS
    assert "padding" not in written["unfitted"]
    stated = tmp_path / "engine.json"
    held = json.loads((a / "steps/dp0.engine.json").read_text())
    stated.write_text(json.dumps({**held, "prefill_buckets": BUCKETS}))
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    out = tmp_path / "engine_law.json"
    step_linear_fit.main(
        [*map(str, paths), "--form", "history", "--machine", spec_file(tmp_path)]
        + ["--out", str(out), "--engine", str(stated)]
    )
    assert json.loads(out.read_text())["prefill"] == written["prefill"]


def test_records_stating_their_buckets_nowhere_fit_no_padding_and_the_law_says_so(
    tmp_path, capsys
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    published, written = _laws(tmp_path, a, b)

    for law in (published, written):
        assert "padding" not in law["prefill"]["coefficients"]
        assert law["unfitted"]["padding"] == step_linear_fit.NO_BUCKETS
    assert f'"padding": "{step_linear_fit.NO_BUCKETS}"' in capsys.readouterr().out

    # An engine file stating them for the records names nothing.
    stated = tmp_path / "engine.json"
    held = json.loads((a / "steps/dp0.engine.json").read_text())
    stated.write_text(json.dumps({**held, "prefill_buckets": BUCKETS}))
    paths = [d / "steps/dp0.jsonl" for d in (a, b)]
    out = tmp_path / "engine_law.json"
    step_linear_fit.main(
        [*map(str, paths), "--machine", spec_file(tmp_path), "--out", str(out)]
        + ["--engine", str(stated)]
    )
    law = json.loads(out.read_text())
    assert law["prefill"]["buckets"] == BUCKETS
    assert "padding" not in law["unfitted"]


@pytest.mark.parametrize("where", ["engine", "key"])
def test_a_record_whose_runner_recorded_no_prefill_graphs_fits_unnamed(tmp_path, where):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    for d in (a, b):
        _state(d, **{where: []})
    published, written = _laws(tmp_path, a, b)

    for law in (published, written):
        assert "padding" not in law["prefill"]["coefficients"]
        assert "padding" not in law["unfitted"]
