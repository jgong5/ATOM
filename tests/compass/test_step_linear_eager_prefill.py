# SPDX-License-Identifier: MIT
"""A prefill its group runs eagerly pays no token-graph `padding`. ATOM runs a
step eagerly when an attention backend of any rank refuses a graph for it, as
V4.1's does for rows its packed score workspace cannot hold in one band, and
then replays no token graph: the step runs its own rows, not its bucket's.

The record is `test_step_linear_token_padding.py`'s trace-run record under the
law #2976 fitted there, with deep-history prefills its lines state the group
ran eagerly; the run is a simulated runner whose attention refuses a graph for
every prefill.

No device.
"""

import json
from types import SimpleNamespace

import pytest
from test_backend_step_linear_fit import ENGINE, _engine_file, _record
from test_parity_record import _config, _drive, _Runner
from test_step_linear_token_padding import (
    BUCKETS,
    PADDED,
    _bucket,
    _law_of,
    _path,
    _priced,
    _trace_steps,
    _view,
)

from atom.compass.backends import StepLinearBackend, StepLinearLaw, step_linear_fit
from atom.compass.backends.json_file import read_jsonl
from atom.compass.parity import ENV
from atom.compass.runner.overrides import install_cost_backend

#: One-request prefills on deep histories, each in a bucket well short of its top.
EAGER = [[(n, n + 30000, False)] for n in (300, 1100, 2100, 4200, 6200)]


def _unpadded(rows):
    """`PADDED`'s price of `rows` run eagerly: its own rows, not its bucket's."""
    tokens = sum(r[0] for r in rows)
    return _priced(rows) - PADDED["padding"] * (_bucket(tokens) - tokens)


def test_a_prefill_its_group_runs_eagerly_pads_nothing_in_a_fit_or_a_price(tmp_path):
    steps = _trace_steps(0) + [(rows, None, _unpadded(rows)) for rows in EAGER]
    path = _record(_path(tmp_path / "eager"), steps)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for line in lines[-len(EAGER) :]:
        line["group_needs_eager"] = True
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    _engine_file(path, {**ENGINE, "prefill_buckets": BUCKETS})

    samples = step_linear_fit.label_records([path])[0]
    eager = [s.step for s in samples if s.view.group_needs_eager]
    assert eager == list(range(len(steps) - len(EAGER), len(steps)))
    law = _law_of(tmp_path, [path])
    assert law["prefill"]["coefficients"] == pytest.approx(PADDED, rel=1e-6)
    assert law["prefill"]["fit"]["dropped"] == 0
    backend = StepLinearBackend(StepLinearLaw(law))
    for rows in EAGER:
        cost = backend.price(_view(rows, group_needs_eager=True))
        assert cost.seconds == pytest.approx(_unpadded(rows), rel=1e-6)
        assert [t.seconds for t in cost.terms if t.name == "prefill.padding"] == [0.0]
    # A data-parallel step run eagerly has no bucket, so needs no largest rank.
    dp = backend.price(_view(EAGER[0], moe_rows=4000, group_needs_eager=True))
    assert [t.seconds for t in dp.terms if t.name == "prefill.padding"] == [0.0]


def test_a_simulated_prefill_its_attention_runs_eagerly_records_so_and_pads_nothing(
    tmp_path, monkeypatch
):
    batches = _drive(tmp_path / "source", monkeypatch, {0: [0]})[0][0]
    vector = {
        "coefficients": PADDED,
        "hull": {"equal": dict.fromkeys(list(PADDED)[1:], 0), "facets": []},
        "buckets": BUCKETS,
    }
    law = StepLinearLaw({"provenance": "planted", "prefill": vector, "decode": {}})
    padding = {}
    for refused in (False, True):
        directory = tmp_path / ("eager" if refused else "graph")
        monkeypatch.setenv(ENV, str(directory))
        runner = _Runner(_config())
        install_cost_backend(runner, StepLinearBackend(law))
        runner.attn_metadata_builder = SimpleNamespace(
            step_needs_eager=lambda b, r=refused: r and b.total_seqs_num_prefill > 0,
            supports_token_graph=False,
        )
        for batch in batches:
            runner.forward(batch)
        prefills = {
            s["step"]: s.get("group_needs_eager", False)
            for s in read_jsonl(directory / "dp0.jsonl")
            if not any(r[2] for r in s["rows"])
        }
        assert set(prefills.values()) == {refused}
        padding[refused] = [
            t["seconds"]
            for c in read_jsonl(directory / "dp0.cost.jsonl")[1:]
            if c["step"] in prefills
            for t in c["terms"]
            if t["name"] == "prefill.padding"
        ]
    assert len(padding[True]) == len(padding[False]) > 1
    assert min(padding[False]) > 0 and set(padding[True]) == {0.0}
