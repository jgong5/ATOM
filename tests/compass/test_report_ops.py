# SPDX-License-Identifier: MIT
"""An op-level run's op-family split, on its report: an instance's record and
cost lines written by the simulated runner pricing ATOM's scheduled batches
with the op-level backend, and a deployment's written by hand.
"""

import collections
import json

import pytest
from test_op_graph import GATHER, GEMM, g
from test_op_work import leaf, t
from test_parity_record import _config, _Runner
from test_report_html import _Page
from test_runner_step_semantics import _drive
from test_work import SPEC

from atom.compass import parity
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.report import html, ops
from atom.compass.runner import overrides
from atom.compass.runner.overrides import HostDeviceRecurrence, install_cost_backend

GATING = "aiter.topk_gating_fwd.default"
INDPTR = "atom.model_ops.v4_kernels.paged_decode_indices:build_v4_paged_decode_indptr"
NORM = "aiter.rmsnorm2d_fwd_.default"
FILL = "atom.model_ops.decode_input_ids:fill_deferred_decode_ids"
#: Each leaf's term, by the family the split puts it in.
FAMILY = {
    f"op.{GEMM}": "dense GEMM",
    f"op.{NORM}": "norms and elementwise",
    f"op.{GATING}": "MoE",
    f"op.{INDPTR}": "attention",
    f"op.{FILL}": "other",
}
REGION = {
    "seq": [
        g(2, 64, 64),
        leaf(NORM, [[2, 64]], input=t([2, 64]), weight=t([64], role="weight")),
        leaf(GATING, [[2, 8]], gating_output=t([2, 8])),
        leaf(INDPTR, [[3]], seq_lens=t([2], "torch.int32")),
        leaf(FILL, [[2]], ids=t([2], "torch.int32")),
        {"par": [g(2, 128, 64), g(2, 256, 64)], "join": "resource_bound"},
    ]
}
PAR = "par.resource_bound"


def _op_level_instance(run, monkeypatch) -> list[dict]:
    """Write a `compass run --op-graph` instance's record and cost lines, its
    forwards priced from `REGION` but those that sample nothing, whose shapes
    no graph holds, and return its cost lines."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        batches = _drive().batches
    monkeypatch.setenv(parity.ENV, str(run / "steps"))
    r, clock = _Runner(_config()), [10.0]
    graph = OpGraphBackend(SPEC, lambda *shape, **_: shape[3] and REGION, kv="bf16")
    install_cost_backend(r, graph)
    r.compass_host = HostDeviceRecurrence({}, (), lambda: clock[0])
    for b in batches:
        b.external_request_ids = [f"r{i}" for i in b.req_ids]
        clock[0] += r.forward(b).predicted_s
    r._cost_table.close()
    (run / "run.json").write_text(json.dumps({"run_file": {"op_graph": True}}))
    costs = (run / "steps" / "dp0.cost.jsonl").read_text().splitlines()[1:]
    return [json.loads(c) for c in costs]


def test_an_op_level_instance_s_families_sum_its_terms_and_reconcile_with_its_time(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    costs = _op_level_instance(run, monkeypatch)
    steps = [
        json.loads(s) for s in (run / "steps" / "dp0.jsonl").read_text().splitlines()
    ]

    [(side, ranks, phases)] = ops.split(run)

    assert (side, ranks) == ("", 1)
    assert {"prefill, no output", "prefill", "decode"} <= phases.keys()
    by_family, predicted = collections.Counter(), 0.0
    for phase in phases.values():
        *parts, (last, total, _) = ops.rows(phase)
        assert last == "predicted"
        assert sum(s for _, s, _ in parts) == pytest.approx(total, rel=1e-12)
        by_family.update({name: s for name, s, _ in parts})
        predicted += total
    terms = collections.Counter()
    for c in costs:
        for term in c["terms"]:
            terms[FAMILY.get(term["name"], term["name"])] += term["seconds"]
    assert set(terms) == {*FAMILY.values(), PAR}
    assert {f: by_family[f] for f in terms} == pytest.approx(terms, rel=1e-12)
    assert predicted == pytest.approx(sum(c["group_s"] for c in costs), rel=1e-12)
    # Every forward that samples nothing is refused, and only those.
    quiet = sum(not s["produces_output"] for s in steps)
    assert quiet == sum(phases["prefill, no output"].refused.values()) > 0
    assert sum(sum(p.refused.values()) for p in phases.values()) == quiet
    [why] = phases["prefill, no output"].refused
    assert why.startswith("cost:op-graph: no op_graph holds a prefill step")


def _step(step, *rows, output=True, dummy=False) -> dict:
    line = {"step": step, "is_dummy_run": dummy, "produces_output": output}
    return line | {"rows": [list(r) for r in rows]}


def _cost(step, terms, *, charge=None, group=None, refused=None) -> dict:
    """A cost line of `terms`, each (name, seconds), a drafter's named so."""
    device = sum(s for n, s in terms if not n.startswith("draft "))
    draft = sum(s for n, s in terms if n.startswith("draft "))
    charge = device + draft if charge is None else charge
    line = {"step": step, "t_s": float(step), "device_s": device, "draft_s": draft}
    line |= {"charge_s": charge, "group_s": charge if group is None else group}
    named = [{"name": n, "seconds": s, "method": "measured"} for n, s in terms]
    return line | {"terms": named, "refused": refused}


def _side(side, ranks) -> None:
    """`side`'s DP ranks, each a list of its steps' record and cost lines."""
    steps = side / "steps"
    steps.mkdir(parents=True)
    for rank, lines in enumerate(ranks):
        record = "".join(json.dumps(s) + "\n" for s, _ in lines)
        (steps / f"dp{rank}.jsonl").write_text(record)
        cost = [{"version": overrides.COST_TABLE_VERSION}, *(c for _, c in lines)]
        (steps / f"dp{rank}.cost.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cost)
        )


MLA = "op.aiter.mla:mla_decode_fwd_v4_nm"
EXTRACTOR = "cost:op-graph: no extractor for the op kind x"


def _deployment(run) -> None:
    """A 1P1D run: two prefill ranks, each step one two-batch-overlap term,
    rank 1 leaving at rank 0's charge, and rank 0's next step a decode row
    beside a prefill row running only the MoE layer's DP gather; one decode
    rank with a drafter, its host returning after its device, a DP dummy and a
    refused step."""
    run.mkdir()
    meta = {"kv_link": "inter_node", "run_file": {"op_graph": True}}
    (run / "run.json").write_text(json.dumps(meta))
    _side(
        run / "prefill",
        [
            [
                (_step(0, (64, 64, False)), _cost(0, [("op.tbo", 2.0)])),
                (
                    _step(1, (1, 65, True), (32, 96, False)),
                    _cost(1, [(f"op.{GATHER}", 0.25)]),
                ),
            ],
            [
                (
                    _step(0, (32, 32, False)),
                    _cost(0, [("op.tbo.price_list", 1.5)], group=2.0),
                )
            ],
        ],
    )
    terms = [(f"op.{GEMM}.price_list", 0.5), (MLA, 0.25), (f"draft op.{GEMM}", 0.125)]
    _side(
        run / "decode",
        [
            [
                (_step(0, (1, 65, True)), _cost(0, terms, charge=1.125)),
                (_step(1, dummy=True), _cost(1, [(PAR, 0.75)])),
                (
                    _step(2, (1, 66, True)),
                    _cost(2, [], charge=0.0625, refused=EXTRACTOR),
                ),
            ]
        ],
    )


def test_a_deployment_s_report_splits_each_side_with_its_whole_terms_and_the_rest(
    tmp_path,
):
    run = tmp_path / "run"
    _deployment(run)

    page = _Page(html.write(run).parent)

    shown = page.parts["Op-family split"]
    whom = "steps over {} DP ranks, {} of them refused; seconds summed over the"
    whom += " ranks."
    assert shown["text"] == [
        "prefill side, prefill: 3 " + whom.format(2, 0),
        "decode side, decode: 2 " + whom.format(1, 1),
        "decode side, DP dummy: 1 " + whom.format(1, 0),
    ]
    head = ["seconds", "share of predicted", "op kinds, or why"]
    summed = "group_s, summed over the side's DP ranks"
    assert page.rows("Op-family split", 0) == {
        "part": head,
        "collectives": ["0.25", "5.882%", f"op.{GATHER} 0.25"],
        "op.tbo": ["3.5", "82.35%", ops.WHOLE["op.tbo"]],
        "the DP group's slowest rank": ["0.5", "11.76%", ops.GROUP],
        "predicted": ["4.25", "100%", summed],
    }
    gemm = f"op.{GEMM} 0.5; draft op.{GEMM} 0.125"
    assert page.rows("Op-family split", 1) == {
        "part": head,
        "attention": ["0.25", "21.05%", f"{MLA} 0.25"],
        "dense GEMM": ["0.625", "52.63%", gemm],
        "host time": ["0.3125", "26.32%", ops.HOST],
        "predicted": ["1.188", "100%", summed],
    }
    assert page.rows("Op-family split", 2) == {"refused": ["steps"], EXTRACTOR: ["1"]}
    assert page.rows("Op-family split", 3) == {
        "part": head,
        PAR: ["0.75", "100%", ops.WHOLE[PAR]],
        "predicted": ["0.75", "100%", summed],
    }


def test_a_step_whose_terms_do_not_fold_to_its_device_time_refuses_the_split(
    tmp_path,
):
    run = tmp_path / "run"
    _deployment(run)
    path = run / "decode" / "steps" / "dp0.cost.jsonl"
    lines = path.read_text().splitlines()
    lines[1] = json.dumps(json.loads(lines[1]) | {"device_s": 0.5})
    path.write_text("\n".join(lines) + "\n")

    page = _Page(html.write(run).parent)

    why = f"None: {path}'s step 0: its terms fold to (0.75, 0.125), not its"
    assert page.parts["Op-family split"]["text"] == [f"{why} device_s and draft_s"]


@pytest.mark.parametrize(
    "meta, why",
    [
        ({}, "a real run is measured, not priced: it has no op split"),
        (
            {"run_file": {"step_linear": {}}},
            "a step-level run prices whole steps: it has no op split",
        ),
    ],
    ids=["real", "step-level"],
)
def test_a_real_or_step_level_run_says_it_has_no_op_split(tmp_path, meta, why):
    (tmp_path / "run.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match=why):
        ops.split(tmp_path)
