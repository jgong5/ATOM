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
from atom.compass.backends.work import Work, price
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


def _main(monkeypatch, tmp_path, records, tp=1, more=(), ep=1):
    config = SimpleNamespace(
        tensor_parallel_size=tp,
        kv_cache_dtype="bf16",
        enable_tbo=False,
        enable_tbo_decode=False,
        speculative_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    moe = SimpleNamespace(moe_parallel_config=SimpleNamespace(ep_size=ep))
    tree = SimpleNamespace(modules=lambda: [SimpleNamespace(), moe])
    monkeypatch.setattr(bound, "_build", lambda *a: (config, tree))
    monkeypatch.setattr("atom.compass.run.modules.walk", lambda tree: STACK)
    monkeypatch.setattr("atom.compass.run._op_graphs", lambda *a: graph)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(ROOFLINE))
    argv = ["--model", "m", "--machine", str(spec), "--store", "s", *more]
    argv += map(str, records)
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


def plan(path, points):
    path.mkdir(parents=True, exist_ok=True)
    (path / "plan.jsonl").write_text("".join(json.dumps(p) + "\n" for p in points))
    return path


def point(op, shape, rows, cold_ns, **more):
    record = {"shape": shape, "context": None, "op": op["op"], "rows": rows}
    return record | {"leaf": json.dumps(op), "cold_ns": cold_ns, **more}


ATTN = "aiter.unified_attention_with_output_base.default"


def test_each_measured_point_is_its_cold_time_over_its_leafs_price(tmp_path):
    short, long = [[1, 900, True], [1, 300, True]], [[1, 200000, True]] * 2
    silu = leaf(SILU, [[2, 4096]], input=t([2, 8192]))
    attn = leaf(ATTN, [[2, 6144]], q=t([2, 6144]), k=t([2, 1024]), v=t([2, 1024]))
    attn["args"] |= {"layer_name": {"context": None}, "use_mla": False}

    def priced(op, rows):
        view = BatchView(tuple(RequestShape(*r) for r in rows), produces_output=True)
        return price(op_work.work(op, view, 1, "bf16"), SPEC) * 1e9

    decode, empty = "decode T=2 B=2", leaf("aten.empty.memory_format", [[2, 4096]])
    out = plan(
        tmp_path,
        [
            point(silu, decode, short, 3 * priced(silu, short)),
            point(silu, decode, short, 0.5 * priced(silu, short)),
            point(silu, decode, short, 0.8 * priced(silu, short)),
            point(silu, "prefill T=2 B=1", [[2, 2, False]], 2 * priced(silu, short)),
            point(attn, decode, long, 1.5 * priced(attn, long)),
            point(silu, decode, short, 0, refused="ValueError: no room"),
            point(empty, decode, short, 0),
            point(empty, decode, short, 1000),
            point(leaf("x.unknown.default"), decode, short, 1000),
        ],
    )

    r = bound.ops(out, SPEC, 1, "bf16")

    assert r["points"] == 9
    assert r["counts"] == {"unmeasured": 1, "no kernel": 1, "no kernel, timed": 1}
    assert r["refused"] == {
        "refused(op-work: no extractor for the op kind x.unknown.default)": 1
    }
    q = r["measured_over_analytical"]
    assert list(q) == [f"{SILU} decode", f"{SILU} prefill", f"{ATTN} decode"]
    # Attention reads the KV of the point's rows, not of its shape alone.
    assert priced(attn, long) > 10 * priced(attn, short)
    assert q[f"{ATTN} decode"]["median"] == pytest.approx(1.5)
    assert (q[f"{SILU} decode"]["points"], q[f"{SILU} decode"]["above"]) == (3, 2)
    assert (q[f"{SILU} decode"]["min"], q[f"{SILU} decode"]["max"]) == pytest.approx(
        (0.5, 3)
    )
    assert q[f"{SILU} prefill"]["median"] == pytest.approx(2)
    assert "2 of a leaf priced as launching no kernel (1 of them timed" in (
        text := bound.ops_text(r)
    )
    assert "2 priced above their measured time" in text


def test_main_reports_an_ops_run_and_refuses_one_of_another_engine(
    monkeypatch, tmp_path, capsys
):
    slow = record(tmp_path / "slow", LINES[:3])
    engine = {"tensor_parallel_size": 1, "kv_cache_dtype": "bf16"}
    (slow / "steps" / "dp0.engine.json").write_text(json.dumps(engine))
    out = plan(tmp_path / "plan", [])
    (out / "engine.json").write_text(json.dumps(engine))

    _main(monkeypatch, tmp_path, [slow], more=["--ops", str(out)])
    assert "op points: 0 planned" in capsys.readouterr().out

    (out / "engine.json").write_text(json.dumps(engine | {"kv_cache_dtype": "fp8"}))
    with pytest.raises(ValueError, match="kv_cache_dtype 'fp8' there, 'bf16' here"):
        _main(monkeypatch, tmp_path, [slow], more=["--ops", str(out)])


def test_main_prices_ops_and_points_at_the_model_s_ep_width(monkeypatch, tmp_path):
    slow = record(tmp_path / "slow", LINES[:3])
    engine = {"tensor_parallel_size": 1, "kv_cache_dtype": "bf16"}
    (slow / "steps" / "dp0.engine.json").write_text(json.dumps(engine))
    out = plan(tmp_path / "plan", [])
    (out / "engine.json").write_text(json.dumps(engine))
    seen = {}
    monkeypatch.setattr(
        bound, "gate", lambda _, b, *a: seen.update(op=b["op"].ep) or {}
    )
    monkeypatch.setattr(bound, "ops", lambda *a: seen.update(points=a[-1]))
    monkeypatch.setattr(bound, "text", lambda r: "")
    monkeypatch.setattr(bound, "ops_text", lambda r: "")

    _main(monkeypatch, tmp_path, [slow], more=["--ops", str(out)], ep=8)

    assert seen == {"op": 8, "points": 8}


def test_a_measured_point_prices_at_the_ep_width(tmp_path):
    op = leaf(
        "mori.ops.dispatch_combine:EpDispatchCombineOp.combine",
        [[16384, 7168]],
        input=t([16384, 7168]),
        weights=None,
        indices=t([2048, 6], "torch.int32"),
    )
    rows = [[2048, 2048, False]]
    view = BatchView(tuple(RequestShape(*r) for r in rows), produces_output=True)
    cold = price(op_work.work(op, view, 1, "bf16", 8), SPEC) * 1e9
    out = plan(tmp_path, [point(op, "prefill T=2048 B=1", rows, cold)])

    [q] = bound.ops(out, SPEC, 1, "bf16", 8)["measured_over_analytical"].values()

    assert q["median"] == pytest.approx(1)
