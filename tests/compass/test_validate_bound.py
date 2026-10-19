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
from types import SimpleNamespace

import pytest
from test_backend_step_roofline import MACHINE, ROOFLINE, STACK
from test_op_graph import g
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.backends import BatchView, RequestShape, op_work
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.step_roofline import Module, StepRooflineBackend
from atom.compass.backends.work import Work
from atom.compass.validate import bound

SILU = "aiter.silu_and_mul.default"
# A decode step at rung 2, a prefill, a mixed step, a DP dummy run, and a
# DP step of a one-rank group; one second of device time each.
LINES = [
    {"rows": [[1, 900, True], [1, 300, True]], "rung": 2},
    {"rows": [[4096, 4096, False]], "rung": None},
    {"rows": [[100, 100, False], [1, 50, True]], "rung": None},
    {"rows": [[1, 9, True]], "rung": None, "is_dummy_run": True},
    {"rows": [[1, 9, True]], "rung": None, "moe_rows": 8},
]


def record(path, lines=LINES, busy_ns=10**9, rank=0):
    (path / "steps").mkdir(parents=True, exist_ok=True)
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
    (path / "steps" / f"dp{rank}.jsonl").write_text(
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
        assert (r["verdict"], r["steps"], r["violations"]) == ("pass", 4, [])
        assert (r["refused"], r["steps_refused"]) == ({}, 0)
        assert set(r["measured_over_analytical"]) == {"decode", "prefill", "mixed"}
        assert all(q["min"] > 1 for q in r["measured_over_analytical"].values())
    prefill = (RequestShape(4096, 4096, False),)
    one = BACKENDS["op"].price(BatchView(prefill, produces_output=True))
    assert report["op"]["measured_over_analytical"]["prefill"][
        "median"
    ] == pytest.approx(1 / one.seconds)


def test_a_law_pricing_one_op_kind_above_its_measured_time_fails_naming_it(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(op_work.EXTRACTORS, SILU, lambda *a: Work({"bf16": 1e18}))

    r = bound.gate([record(tmp_path)], {"op": BACKENDS["op"]})["op"]

    assert r["verdict"] == "fail"
    assert [v["named"] for v in r["violations"]] == [f"op.{SILU}"] * 4
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


class OneSecond:
    """Every step at exactly one second."""

    def price(self, view):
        term = CostTerm("one", 1.0, Provenance(Method.ANALYTICAL, "", "test"))
        return StepCost((term,))


@pytest.mark.parametrize("busy_ns,verdict", [(10**9 - 1, "fail"), (10**9, "pass")])
def test_the_gate_is_at_most_measured(tmp_path, busy_ns, verdict):
    r = bound.gate([record(tmp_path, LINES[1:2], busy_ns)], {"op": OneSecond()})["op"]

    assert r["verdict"] == verdict


def test_records_rank_by_tokens_not_by_rows(tmp_path):
    # One 4096-token row a second against two one-token rows a second.
    long = record(tmp_path / "long", LINES[1:2])
    wide = record(tmp_path / "wide", [{"rows": [[1, 9, True]] * 2, "rung": None}])

    k = bound.gate([wide, long], {"op": OneSecond()})["op"]["ranking"]

    assert (k["measured_fastest"], k["analytical_order"][0]) == (str(long), str(long))


def _line(rows, moe_rows=8, **more):
    return {"rows": rows, "rung": None, "moe_rows": moe_rows, **more}


def test_a_dp_step_joins_every_ranks_line_by_atoms_rules(tmp_path):
    prefill, short, decode = [[9000, 9000, False]], [[100, 100, False]], [[1, 50, True]]
    dummy = _line([[1, 1, True]], is_dummy_run=True)
    rank0 = [_line(prefill)] * 2 + [_line(short)] + [_line(prefill)] * 2
    rank1 = [_line(short), dummy, _line([[200, 200, False]]), _line(decode)]
    record(tmp_path, rank0, rank=0)
    record(tmp_path, rank1, rank=1)

    got = list(bound.steps(tmp_path, tbo=True))
    views = {name.split(" ", 1)[1]: v for name, v, _ in got}

    # 9000 clears the 8192-token bar and both ranks split at their midpoints.
    assert views["dp0 step 0"].tbo_tokens == ((4500, 50), (4500, 50))
    assert views["dp1 step 0"].moe_tokens == 9100
    # A dummy rank counts one token and cannot split; nor can a decoding one.
    assert views["dp0 step 1"].moe_tokens == 9001
    assert views["dp0 step 1"].tbo_tokens is None
    assert "dp1 step 1" not in views
    assert views["dp0 step 3"].tbo_tokens is None
    # No rank clears the bar.
    assert views["dp0 step 2"].tbo_tokens is None
    # A step one rank's record lacks.
    assert views["dp0 step 4"] == Refusal(bound.SOURCE, bound.PARTIAL)
    assert next(bound.steps(tmp_path))[1].tbo_tokens is None


def test_an_mtp_record_is_measured_by_its_target(tmp_path):
    lines = [
        {**LINES[0], "d_target_ns": 4 * 10**8},
        {**LINES[1]},
    ]
    record(tmp_path, lines)

    (_, view, measured), (_, refused, _) = bound.steps(tmp_path)

    assert (view.decode != (), measured) == (True, pytest.approx(0.4))
    assert refused == Refusal(bound.SOURCE, bound.UNSPLIT)


def _main(monkeypatch, tmp_path, records, tp=1):
    config = SimpleNamespace(
        tensor_parallel_size=tp,
        kv_cache_dtype="bf16",
        enable_tbo=False,
        enable_tbo_decode=False,
        speculative_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    monkeypatch.setattr(bound, "_build", lambda *a: (config, None))
    monkeypatch.setattr("atom.compass.run.modules.walk", lambda tree: STACK)
    monkeypatch.setattr("atom.compass.run._op_graphs", lambda *a: graph)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(ROOFLINE))
    argv = ["--model", "m", "--machine", str(spec), "--store", "s", *map(str, records)]
    bound.main(argv)


def test_main_exits_1_on_a_step_above_its_measured_time(monkeypatch, tmp_path, capsys):
    fast = record(tmp_path / "fast", LINES[:3], busy_ns=1)
    engine = {"tensor_parallel_size": 1, "kv_cache_dtype": "bf16"}
    (fast / "steps" / "dp0.engine.json").write_text(json.dumps(engine))

    with pytest.raises(SystemExit) as exit:
        _main(monkeypatch, tmp_path, [fast])

    assert exit.value.code == 1
    assert "step level: fail" in capsys.readouterr().out


def test_main_refuses_a_record_its_key_states_another_tp_for(monkeypatch, tmp_path):
    slow = record(tmp_path / "slow", LINES[:3])
    engine = {"engine": {"tensor_parallel_size": 2, "kv_cache_dtype": "bf16"}}
    (slow / "steps" / "dp0.key.json").write_text(json.dumps(engine))

    with pytest.raises(ValueError, match="tensor_parallel_size 2 there, 1 here"):
        _main(monkeypatch, tmp_path, [slow])
