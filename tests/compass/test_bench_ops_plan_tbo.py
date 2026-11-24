# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `--coverage` over a record whose DP group
split a prefill step under two-batch overlap: each reads the step as its two
micro-batches at their own rows, as `trace ops --population` traces it and
`OpGraphBackend` prices it, and the plan refuses a graph only that call traces
before it traces or measures anything. A micro-batch's point keeps its mark in
its record, and the plan densifies such points, measured or held in its store,
at the micro-batch's own graph; a plan without TBO that reads them back
densifies at its own steps' graphs.
The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

import json
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import _measuring
from test_bench_ops_plan import CONFIG, _bench, _plan, bench  # noqa: F401  a fixture

from atom.compass.artifacts import ArtifactRefusal, Rule
from atom.compass.bench import ops
from atom.compass.capture import forward, population_shapes

#: Rank 0 prefills 9000 tokens, past the 8192-token bar, and rank 1 100: each
#: cuts its own at the midpoint, each micro-batch gathering both ranks' rows of
#: it (`test_op_graph_gathered`); then each decodes a row at its context.
RANKS = ([[9000, 9000, False]], 1000), ([[100, 100, False]], 3000)
HALVES = {"prefill T=4500 B=1 G=4550", "prefill T=50 B=1 G=4550"}
WHOLE = {"prefill T=9000 B=1 G=9100", "prefill T=100 B=1 G=9100"}


@pytest.fixture
def split(tmp_path, monkeypatch):
    """The record's directory, read under `--enable-tbo`."""
    monkeypatch.setattr(CONFIG, "enable_tbo", True)
    steps = tmp_path / "plan" / "steps"
    steps.mkdir(parents=True)
    (steps.parent / "run.json").write_text("{}")
    for rank, (rows, context) in enumerate(RANKS):
        prefill = {"step": 0, "rows": rows, "moe_rows": 9100, "rung": None}
        decode = {"step": 1, "rows": [[1, context, True]], "moe_rows": 2, "rung": 1}
        # 4500 tokens on each rank, under the bar: one step, its gather the default
        even = prefill | {"step": 2, "rows": [[4500, 4500, False]], "moe_rows": 9000}
        lines = (
            json.dumps(s | {"is_dummy_run": False}) for s in (prefill, decode, even)
        )
        (steps / f"dp{rank}.jsonl").write_text("\n".join(lines) + "\n")
    return str(steps.parent)


def test_a_plan_loads_a_split_step_s_micro_batches_graphs_not_the_whole_step_s(
    bench, tmp_path, split, monkeypatch  # noqa: F811
):
    loaded, stand_in = set(), ops.graph

    def graph(store, config, shape, arch=None):
        loaded.add(str(shape))
        return stand_in(store, config, shape, arch)

    monkeypatch.setattr(ops, "graph", graph)
    _plan(bench, tmp_path, split)

    assert HALVES <= loaded and not loaded & WHOLE


def test_a_plan_reads_a_micro_batch_s_cached_tokens_off_its_own_rows(
    bench, tmp_path, split  # noqa: F811
):
    records = _plan(bench, tmp_path, split)

    # Rank 0's second micro-batch attends the 4500 tokens of its first.
    prefill = [r for r in records if r["shape"].startswith("prefill")]
    assert 4500 in {r["context"] for r in prefill if r["op"] == "aiter.attn"}


def test_coverage_counts_a_split_step_as_its_micro_batches(
    bench, tmp_path, split  # noqa: F811
):
    records = _plan(bench, tmp_path, split)
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [split], records, "gfx942")

    # Each rank's two micro-batches and its even step, each at points of its own
    # shape: `fit ops` prices a shape that states G apart from one that does not.
    assert report["by phase"]["prefill"] == {"steps": 6, "covered": 6}


def test_a_plan_refuses_a_micro_batch_no_graph_holds_before_tracing_or_measuring(
    bench, tmp_path, split, monkeypatch  # noqa: F811
):
    """`_bench` stands in for every child the plan starts, and takes no capture."""
    store, root, key, run = bench

    def empty(store, config, shape, arch=None):
        raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")

    monkeypatch.setattr(ops, "graph", empty)
    monkeypatch.setattr(forward, "device_arch", lambda: "gfx942")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    config = SimpleNamespace(**vars(CONFIG), model="m")
    args = SimpleNamespace(plan=[split], store=root, llc_mib=256)

    with pytest.raises(ArtifactRefusal) as refused:
        ops.plan(store, key, config, run, args, "plan", tmp_path, ["--enable-tbo"])

    # `prefill T=4500 B=1`, the even step's, is one `--prefill 4500 1` names.
    assert refused.value.what.startswith("no op_graph holds prefill T=4500 B=1 G=4550")
    call = "compass trace ops --model m --tp 1 --arch gfx942 --gpu 3 --population"
    call += f" {split} --store {root} -- --enable-tbo"
    assert refused.value.remedy == f"trace it with {call}"
    assert _bench.calls == 0


def test_a_micro_batch_s_measured_point_keeps_its_mark_in_its_record(monkeypatch):
    _, todo = _measuring(monkeypatch)
    half = forward.prefill(4, 1)._replace(micro_batch=True)

    got = ops.measure(None, [(half, None, todo[1][2])], 10_000)

    assert [(r["shape"], r.get("micro_batch")) for r in got] == [
        ("prefill T=4 B=1", True)
    ] * 2


def _switching(fn, config, todo, llc):
    """`_bench` with prefill attention's kernel switching at 1000 cached tokens,
    between the 50 and the 4500 the micro-batches' rows cache, and each
    micro-batch's record marked as `Stub.point` marks it."""
    got = []
    for point in todo:
        for r in _bench(fn, config, [point], llc):
            if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
                r["kernels"] = [f"p{int((r['context'] or 0) >= 1000)}"]
            got.append(r | ({"micro_batch": True} if point[0].micro_batch else {}))
    return got


def _unmarked(fn, config, todo, llc):
    """`_switching` as measured before a record kept a micro-batch's mark."""
    got = _switching(fn, config, todo, llc)
    return [{k: v for k, v in r.items() if k != "micro_batch"} for r in got]


@pytest.mark.parametrize("held", [False, True])
def test_a_plan_densifies_a_micro_batch_s_points_at_its_own_graph(
    bench, tmp_path, split, monkeypatch, held  # noqa: F811
):
    """The store holds the graphs of the record's shapes and of any shape that
    states no gather, which `trace ops --prefill` traces: at a micro-batch's
    gather only the micro-batch's own, so a whole step's there refuses. With
    `held` it also holds rounds 0 to 2's points, unmarked, which the plan reads
    back."""
    graphs, stand_in = set(population_shapes([split], True)), ops.graph

    def graph(store, config, shape, arch=None):
        if shape.gathered and shape not in graphs:
            raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")
        return stand_in(store, config, shape, arch)

    monkeypatch.setattr(ops, "graph", graph)
    # what the `trace ops` call of a refusal names
    monkeypatch.setattr(CONFIG, "model", "m", raising=False)
    monkeypatch.setattr(forward, "device_arch", lambda: "gfx942")
    if held:  # a plan that stopped once it published round 2
        publish = ops.publish

        def stopped(*args):
            publish(*args)
            if args[-1].name == "round2":
                raise KeyboardInterrupt

        monkeypatch.setattr(ops, "publish", stopped)
        monkeypatch.setattr(ops, "_child", _unmarked)
        with pytest.raises(KeyboardInterrupt):
            _plan(bench, tmp_path, split)
        monkeypatch.setattr(ops, "publish", publish)
    monkeypatch.setattr(ops, "_child", _switching)

    records = _plan(bench, tmp_path, split)

    # Rank 1's micro-batch's attention, halved over cached tokens to the switch.
    attn = {
        r["context"]
        for r in records
        if r["op"] == "aiter.attn" and r["shape"] == "prefill T=50 B=1 G=4550"
    }
    assert {999, 1000} <= attn


def _whole(path):
    """A record whose ranks prefill 4500 fresh tokens and 50 past 5000 cached,
    each under the bar: with no TBO its steps are `prefill T=4500 B=1 G=4550`
    and `prefill T=50 B=1 G=4550`, the texts of `split`'s micro-batches."""
    steps = path / "whole" / "steps"
    steps.mkdir(parents=True)
    (steps.parent / "run.json").write_text("{}")
    ranks = ([[4500, 4500, False]], 1000), ([[50, 5050, False]], 3000)
    for rank, (rows, context) in enumerate(ranks):
        prefill = {"step": 0, "rows": rows, "moe_rows": 4550, "rung": None}
        decode = {"step": 1, "rows": [[1, context, True]], "moe_rows": 2, "rung": 1}
        lines = (json.dumps(s | {"is_dummy_run": False}) for s in (prefill, decode))
        (steps / f"dp{rank}.jsonl").write_text("\n".join(lines) + "\n")
    return str(steps.parent)


def test_a_plan_without_tbo_densifies_a_point_a_tbo_plan_stored_at_its_own_graph(
    bench, tmp_path, split, monkeypatch  # noqa: F811
):
    """A point's key holds no TBO switch, so a plan without `--enable-tbo` reads
    back the points a plan under it stored at its micro-batches' texts; a
    graph's structure does, so it holds no micro-batch's graph."""
    monkeypatch.setattr(CONFIG, "model", "m", raising=False)
    monkeypatch.setattr(forward, "device_arch", lambda: "gfx942")
    monkeypatch.setattr(ops, "_child", _switching)
    (tmp_path / "tbo").mkdir()
    stored = _plan(bench, tmp_path / "tbo", split)
    assert any(r.get("micro_batch") for r in stored)
    monkeypatch.setattr(CONFIG, "enable_tbo", False)
    whole = _whole(tmp_path)
    graphs, stand_in = set(population_shapes([whole], False)), ops.graph

    def graph(store, config, shape, arch=None):
        if shape.micro_batch or shape.gathered and shape not in graphs:
            raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")
        return stand_in(store, config, shape, arch)

    monkeypatch.setattr(ops, "graph", graph)

    records = _plan(bench, tmp_path, whole)

    attn = {
        r["context"]
        for r in records
        if r["op"] == "aiter.attn" and r["shape"] == "prefill T=50 B=1 G=4550"
    }
    assert {999, 1000} <= attn
