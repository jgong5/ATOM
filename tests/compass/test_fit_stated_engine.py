# SPDX-License-Identifier: MIT
"""`compass fit host` and `compass fit steps` under `--engine FILE`, an engine
file stated from the records' server logs in place of each record's own, as
`step_linear_fit --engine` takes it, on `test_fit_steps.py`'s record
directories. No device.
"""

import json
import re

import pytest
from test_backend_step_linear_fit import platform_arg
from test_fit_host import host_laws_key
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
from test_memory_readings import DOCUMENT
from test_prefill_buckets_stated import _state
from test_step_linear_token_padding import BUCKETS

from atom.compass.backends import step_linear_fit
from atom.compass.backends.launch_features import GRAPHS
from atom.compass.backends.step_labels import digest
from atom.compass.fit.samples import load
from atom.compass.run.step_grid import BOUNDS

#: The engine config the records' server logs state, a field their engine
#: files and host keys predate among it.
STATED = {**ENGINE, "indexer_dcp_only": True}


def _engine(tmp_path, **fields):
    """An engine file stating `STATED`, with `fields` over it."""
    path = tmp_path / "engine.json"
    path.write_text(json.dumps({**STATED, **dict.fromkeys(BOUNDS, 4096), **fields}))
    return path


def test_both_fits_key_their_entries_by_the_stated_engine_as_step_linear_fit_does(
    tmp_path,
):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    for d in (a, b):
        _state(d, key=BUCKETS)
    owns = [(d / "steps/dp0.engine.json").read_bytes() for d in (a, b)]
    engine = _engine(tmp_path, prefill_buckets=BUCKETS)
    fit(tmp_path, "fit host", a, b, *clients(a, b), "--engine", engine)
    fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b), "--engine", engine)

    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    out = tmp_path / "law.json"
    step_linear_fit.main(
        [*map(str, paths), "--form", "history", *platform_arg(DOCUMENT)]
        + ["--out", str(out), "--engine", str(engine)]
    )
    written = json.loads(out.read_text())
    graphs = json.loads((a / "steps/dp0.key.json").read_text())["engine"][GRAPHS]
    keyed = {**STATED, GRAPHS: graphs}
    assert written["host"]["key"]["engine"] == keyed
    host_key = host_laws_key({**HOST, "engine": keyed})
    assert member(tmp_path / "store", host_key, "params")["key"]["engine"] == keyed
    report = member(tmp_path / "store", host_key, "report")
    assert report["stated_engine"] == json.loads(engine.read_text())
    law = member(tmp_path / "store", law_key(engine=STATED), "params")
    assert law["engine_config"] == written["engine_config"] == STATED
    assert [(d / "steps/dp0.engine.json").read_bytes() for d in (a, b)] == owns


@pytest.mark.parametrize("command", ["fit host", "fit steps --backend step-linear"])
def test_an_engine_file_a_record_contradicts_refuses_naming_the_field(
    tmp_path, command
):
    a = record(tmp_path / "a", 19)
    engine = _engine(tmp_path, kv_cache_dtype="fp8")

    named = f"{a / 'steps/dp0.engine.json'} contradicts {engine} in ['kv_cache_dtype']"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, command, a, *clients(a), "--engine", engine)
    assert not (tmp_path / "store").exists()


def test_a_record_stating_no_buckets_takes_the_file_s_and_one_stating_others_refuses(
    tmp_path,
):
    a = record(tmp_path / "a", 19)
    runs = {a: [str(a / "client.json")]}
    engine = _engine(tmp_path, prefill_buckets=[64, 128])
    assert load(runs, engine=engine)[1]["prefill_buckets"] == [64, 128]

    _state(a, engine=[64])
    named = (
        "the records state other prefill buckets: "
        f"[64, 128] in {engine}; [64] in {a / 'steps'}"
    )
    with pytest.raises(ValueError, match=re.escape(named)):
        load(runs, engine=engine)
