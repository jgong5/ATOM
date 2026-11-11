# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` under DP, with no device: a drafter's later
pass on a prefill is measured at the G a run asks it at.

With the stand-in bench and MTP graphs of `test_bench_ops_drafter.py`, on DP
rank 0 of two beside an idle peer, each prefill's drafter runs a later pass
at its batch, 4. `OpGraphBackend` asks that pass at the rows it gathers, its
batch on each rank, 8, not at the rows its first pass gathers: the plan's
point there prices the step's drafter as measured, and `--coverage` reads it
where the run does.
"""

import json
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops_drafter import CHUNK, MTP, _mtp, _ranks
from test_bench_ops_plan import bench  # noqa: F401  a fixture
from test_work import SPEC

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.provenance import Method
from atom.compass.bench import ops
from atom.compass.capture.forward import prefill
from atom.compass.fit import ops as fit_ops

#: `MTP`, of two drafts, at DP 2.
DP = SimpleNamespace(**vars(MTP), parallel_config=SimpleNamespace(data_parallel_size=2))
#: Each prefill step's line, where the plan measures its drafter's later pass,
#: and the drafter's microseconds: a stand-in prefill leaf times a microsecond
#: a token, and the drafter runs its first pass, its later pass and, on a
#: prompt, its sampler. ATOM records a prefill's DP gathers alike, its first
#: pass's and its target's, every rank's tokens.
STEPS = {
    # a middle chunk of 60 tokens beside a peer's 60, or its 30
    "middle-chunk": (
        CHUNK | {"moe_rows": 120, "draft_moe_rows": 120},
        "prefill T=60 B=1 W=4 G=8",
        120,
    ),
    "middle-chunk-g90": (
        CHUNK | {"moe_rows": 90, "draft_moe_rows": 90},
        "prefill T=60 B=1 W=4 G=8",
        120,
    ),
    # a prompt whose LM head pads too: its drafter's shape is the step's
    "padded-prompt": (
        CHUNK
        | {"produces_output": True, "target_bs": 4}
        | {"moe_rows": 120, "draft_moe_rows": 120},
        "prefill T=60 B=1 W=4 G=8",
        180,
    ),
    # a prompt of 4 rows, its batch: a later pass of 4 leads with its B too
    "prompt-at-its-batch": (
        {"rows": [[10, 10, False]] * 4, "produces_output": True, "draft_bs": 4}
        | {"moe_rows": 80, "draft_moe_rows": 80},
        "prefill T=40 B=4 G=8",
        120,
    ),
}


@pytest.mark.parametrize("name", STEPS)
def test_a_dp_drafter_s_later_pass_is_measured_at_the_g_a_run_asks_it_at(
    bench, tmp_path, monkeypatch, name  # noqa: F811
):
    step, at, us = STEPS[name]
    monkeypatch.setattr(ops, "graph", _mtp)
    decode = {"rows": [[3, 1000, True], [3, 3000, True]], "rung": 2}
    record = _ranks(tmp_path / "steps", [decode, step], [{"is_dummy_run": True}])
    store, root, key, run = bench
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(plan=[record], store=root, llc_mib=256)

    ops.plan(store, key, DP, run, args, "compass measure ops --plan", out)

    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]
    later = {
        r["shape"]
        for r in records
        if r["op"] == "aiter.step"
        and r["shape"].startswith("prefill")
        and json.loads(r["leaf"])["args"]["x"]["shape"][0] == 4
    }
    assert later == {at}
    # A run prices the step's drafter from the plan's points alone.
    params = fit_ops.fit([r for r in records if "refused" not in r])
    fit_ops.held_out(params)

    def graph(phase, t, b, output, *_, draft=False, width=None):
        shape = prefill(t, b, None, width)
        return _mtp(store, DP, shape, None, output, draft)["region"]

    listed = fit_ops.PriceList(params)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", dp=2, drafts=2, price_list=listed)
    fields = ("moe_rows", "draft_moe_rows", "draft_bs", "target_bs")
    view = BatchView(
        tuple(RequestShape(*row) for row in step["rows"]),
        produces_output=step["produces_output"],
        **{f: step.get(f) for f in fields},
    )
    cost = backend.price(view)
    assert {t.provenance.method for t in cost.draft_terms} == {Method.MEASURED}
    assert cost.draft_seconds == pytest.approx(us * 1e-6)
    report = ops.coverage(store, root, DP, [record], records, "gfx942")
    assert report["by part"] == {
        "target": {"steps": 2, "covered": 2},
        "draft": {"steps": 2, "covered": 2},
    }
