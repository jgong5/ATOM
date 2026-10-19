# SPDX-License-Identifier: MIT
"""The analytical bound gate: a real run's steps priced at step and op level,
none above its measured device time.

A step priced above measured fails the gate, named by its costliest term; a
law seeded to price one op kind above its measured time fails it naming that
kind. Refusals are counted by reason and make the result a coverage report;
measured over analytical is reported per phase, and the records are ranked
by tokens a second.
"""

import json

import pytest
from test_backend_step_roofline import MACHINE, STACK
from test_op_graph import g
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.backends import BatchView, RequestShape, op_work
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.step_roofline import Module, StepRooflineBackend
from atom.compass.backends.work import Work
from atom.compass.validate import bound

SILU = "aiter.silu_and_mul.default"
# A decode step at rung 2, a prefill, a mixed step, a DP dummy run, and a
# DP MoE step; one second of device time each.
LINES = [
    {"rows": [[1, 900, True], [1, 300, True]], "rung": 2},
    {"rows": [[4096, 4096, False]], "rung": None},
    {"rows": [[100, 100, False], [1, 50, True]], "rung": None},
    {"rows": [[1, 9, True]], "rung": None, "is_dummy_run": True},
    {"rows": [[1, 9, True]], "rung": None, "moe_rows": 8},
]


def record(path, lines=LINES, busy_ns=10**9):
    (path / "steps").mkdir(parents=True)
    body = [
        {
            "step": i,
            "is_dummy_run": False,
            "produces_output": True,
            "d_busy_ns": busy_ns,
            **s,
        }
        for i, s in enumerate(lines)
    ]
    (path / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in body)
    )
    return path


def graph(*shape):
    return {
        "seq": [
            g(shape[1], 4096, 4096),
            leaf(SILU, [[shape[1], 4096]], input=t([shape[1], 8192])),
        ]
    }


BACKENDS = {
    "step": StepRooflineBackend(MACHINE, STACK),
    "op": OpGraphBackend(SPEC, graph, kv="bf16"),
}


def test_every_step_under_its_measured_time_passes_with_its_reports(tmp_path):
    report = bound.gate([record(tmp_path)], BACKENDS)

    for level in ("step", "op"):
        r = report[level]
        assert (r["verdict"], r["steps"], r["violations"]) == ("coverage report", 3, [])
        assert r["refused"] == {f"refused({bound.SOURCE}: {bound.DP_MOE})": 1}
        assert set(r["measured_over_analytical"]) == {"decode", "prefill", "mixed"}
        assert all(q["min"] > 1 for q in r["measured_over_analytical"].values())
    one = BACKENDS["op"].price(BatchView((RequestShape(4096, 4096, False),)))
    assert report["op"]["measured_over_analytical"]["prefill"][
        "median"
    ] == pytest.approx(1 / one.seconds)


def test_a_law_pricing_one_op_kind_above_its_measured_time_fails_naming_it(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(op_work.EXTRACTORS, SILU, lambda *a: Work({"bf16": 1e18}))

    r = bound.gate([record(tmp_path)], {"op": BACKENDS["op"]})["op"]

    assert r["verdict"] == "fail"
    assert [v["named"] for v in r["violations"]] == [f"op.{SILU}"] * 3
    assert {v["phase"] for v in r["violations"]} == {"decode", "prefill", "mixed"}


def test_a_step_level_violation_names_its_module_kind(tmp_path):
    norm = Module("layers.0.norm", "norm", (10**12,))
    seeded = StepRooflineBackend(MACHINE, [*STACK, norm])

    r = bound.gate([record(tmp_path)], {"step": seeded})["step"]

    assert r["verdict"] == "fail"
    assert {v["named"] for v in r["violations"]} == {"roofline.norm"}


def test_a_shape_with_no_graph_is_counted_by_its_reason(tmp_path):
    backend = OpGraphBackend(
        SPEC, lambda *shape: None if shape[0] == "decode" else graph(*shape), kv="bf16"
    )

    r = bound.gate([record(tmp_path)], {"op": backend})["op"]

    assert (r["verdict"], r["steps"]) == ("coverage report", 2)
    assert sum(r["refused"].values()) == 2
    assert any("no op_graph holds a decode step of 2 tokens" in k for k in r["refused"])


def test_records_are_ranked_by_tokens_a_second(tmp_path):
    slow = record(tmp_path / "slow", LINES[:3], busy_ns=2 * 10**9)
    fast = record(tmp_path / "fast", LINES[:3], busy_ns=10**9)

    k = bound.gate([slow, fast], {"op": BACKENDS["op"]})["op"]["ranking"]

    # Both price alike, so analytical's order is the records' order.
    assert k["analytical_order"] == [str(slow), str(fast)]
    assert (k["measured_fastest"], k["top_1"], k["top_3"]) == (str(fast), False, True)


@pytest.mark.parametrize(
    "measured_over_priced,verdict", [(0.99, "fail"), (1.0, "pass")]
)
def test_the_gate_is_at_most_measured(tmp_path, measured_over_priced, verdict):
    prefill = BatchView((RequestShape(4096, 4096, False),))
    priced = BACKENDS["op"].price(prefill).seconds
    busy = int(priced * 1e9 * measured_over_priced) + (measured_over_priced == 1.0)

    r = bound.gate([record(tmp_path, LINES[1:2], busy)], {"op": BACKENDS["op"]})["op"]

    assert r["verdict"] == verdict
