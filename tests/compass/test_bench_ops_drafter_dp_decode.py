# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` under DP, with no device: a drafter's later
pass on a decode is measured at the batch and G a run asks it at.

With the stand-in bench and MTP graphs of `test_bench_ops_drafter.py`, the
drafter also staging its rows into its batch, on DP rank 0 of two beside an
idle peer: a decode's drafter runs its later pass at its batch, 4, which
`OpGraphBackend` asks at that batch and the rows the pass gathers, its batch
on each rank, 8, and the staging, which leads with both, at the step's rows
and that gather. The plan's points there price the step's drafter as
measured, and `--coverage` reads them where the run does.
"""

import json
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import _graph, _leaf, _t
from test_bench_ops_drafter import MTP, _mtp, _ranks
from test_bench_ops_plan import bench  # noqa: F401  a fixture
from test_work import SPEC

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.provenance import Method
from atom.compass.bench import ops
from atom.compass.capture.forward import decode
from atom.compass.fit import ops as fit_ops

#: `MTP`, of two drafts, at DP 2.
DP = SimpleNamespace(**vars(MTP), parallel_config=SimpleNamespace(data_parallel_size=2))
#: A decode row verifying two drafts.
ROW = [3, 1000, True]
#: The target's leaves and its drafter's, each of the drafter's a microsecond.
TARGET = {"aiter.attn", "aiter.gemm", "aiter.head"}
DRAFTER = {"aiter.propose", "aiter.step", "aiter.sample", "aiter.stage"}
#: Each leaf at each rung of the ladder, 1, 2 and 4, as the plan measures them.
RUNGS = {(f"decode B={b}", op) for b in (1, 2, 4) for op in TARGET | DRAFTER}


def _staging(store, config, shape, arch=None, output=True, draft=False):
    """`_mtp`'s graphs, a decode's drafter's also staging its rows, T, into
    its batch, B: a leaf that leads with both."""
    got = _mtp(store, config, shape, arch, output, draft)
    if draft is True and shape.phase == "decode":
        x = ("x", _t([shape.seqs, 8], "activation"))
        rows = ("rows", _t([shape.tokens], "metadata", dtype="torch.int32"))
        stage = _graph(_leaf("aiter.stage", x, rows), layers=1)["region"]
        got = got | {"region": {"seq": [got["region"], stage]}}
    return got


def _plan(bench, tmp_path, monkeypatch, step):  # noqa: F811
    """`--plan` under `DP` of a record of `step` on rank 0 beside an idle rank
    1: its points' records, and `--coverage`'s report of the record."""
    monkeypatch.setattr(ops, "graph", _staging)
    record = _ranks(tmp_path / "steps", [step], [{"is_dummy_run": True}])
    store, root, key, run = bench
    (out := tmp_path / "out").mkdir()
    args = SimpleNamespace(plan=[record], store=root, llc_mib=256)
    ops.plan(store, key, DP, run, args, "compass measure ops --plan", out)
    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]
    return records, ops.coverage(store, root, DP, [record], records, "gfx942")


#: Each decode at rung 4, its drafter's later pass at 4: its line, and the
#: points past the rungs' where the plan measures the drafter's leaves.
STEPS = {
    # 3 rows: its drafter's own shape holds the staging of its 3 rows alone
    "padded": (
        {"rows": [ROW] * 3, "moe_rows": 24, "draft_moe_rows": 18},
        {("decode B=4 G=8", op) for op in DRAFTER - {"aiter.stage"}}
        | {("decode T=3 B=4 G=8", "aiter.stage")},
    ),
    # 4 rows: its drafter's shape is the step's, a rung, which keeps each leaf
    "full": (
        {"rows": [ROW] * 4, "moe_rows": 24, "draft_moe_rows": 24},
        {("decode B=4 G=8", op) for op in DRAFTER},
    ),
}


@pytest.mark.parametrize("name", STEPS)
def test_a_dp_drafter_s_later_pass_on_a_decode_is_measured_where_a_run_asks_it(
    bench, tmp_path, monkeypatch, name  # noqa: F811
):
    line, moved = STEPS[name]
    step = line | {"rung": 4, "draft_bs": 4}

    records, report = _plan(bench, tmp_path, monkeypatch, step)

    assert {(r["shape"], r["op"]) for r in records} == RUNGS | moved
    # A run prices the step from the plan's points alone.
    params = fit_ops.fit([r for r in records if "refused" not in r])
    fit_ops.held_out(params)

    def graph(phase, t, b, output, *_, draft=False, width=None):
        return _staging(bench[0], DP, decode(b, None, t), None, output, draft)["region"]

    listed = fit_ops.PriceList(params)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", dp=2, drafts=2, price_list=listed)
    view = BatchView(
        tuple(RequestShape(*row) for row in step["rows"]),
        capture_rung=4,
        produces_output=True,
        **{f: step[f] for f in ("moe_rows", "draft_moe_rows", "draft_bs")},
    )
    cost = backend.price(view)
    assert {t.provenance.method for t in cost.draft_terms} == {Method.MEASURED}
    assert cost.draft_seconds == pytest.approx(len(DRAFTER) * 1e-6)
    assert report["by part"] == {
        "target": {"steps": 1, "covered": 1},
        "draft": {"steps": 1, "covered": 1},
    }


def test_a_drafter_s_later_pass_at_its_rows_beside_a_prefilling_peer_is_a_rung_s(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # One row decoded eagerly beside a peer prefilling 17 tokens over 5 rows,
    # under a drafter with no draft graph: its later pass runs at its row and
    # gathers the widest rank's rows and one, 6, the first pass's gather at
    # its default, so a run asks a leaf leading with the row at rung 1, whose
    # points of the target's leaves the plan keeps.
    step = {"rows": [ROW], "moe_rows": 20, "draft_moe_rows": 20, "draft_rows": 6}

    records, report = _plan(bench, tmp_path, monkeypatch, step)

    assert {(r["shape"], r["op"]) for r in records} == RUNGS
    assert report["by part"]["draft"] == {"steps": 1, "covered": 1}
