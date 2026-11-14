# SPDX-License-Identifier: MIT
"""An `op_measurement` keys the parallel layout only for a collective leaf.

A plan at a new layout over a store an earlier layout measured, that store
moved to today's keys by `scripts/compass/migrate_keys.py`, measures only the
leaf signatures the store lacks and the collectives; `fit ops` over that plan's
`--out` alone publishes the new layout's price list. Points are published to
and read back from a real store, by `measure ops`' own planner with a
stand-in bench.
"""

import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import _graph, _leaf, _t
from test_bench_ops_plan import CONFIG, _records, bench  # noqa: F401  a fixture
from test_bench_ops_ranks import NEG, _dpa_config, _dpa_main

from atom.compass import cli
from atom.compass.artifacts import ArtifactStore, Kind, Topology, member_name
from atom.compass.bench import ops
from atom.compass.fit import ops as fit_ops
from atom.compass.run import PRICE_LIST
from atom.compass.validate.__main__ import _warm_over_cold

MIGRATE = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts/compass/migrate_keys.py"
)
#: Two DP-attention ranks, expert parallelism off and on.
EP_OFF = {"dp": 2, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
EP_OFF |= {"enable_expert_parallel": False, "enable_dp_attention": True}
EP_ON = EP_OFF | {"enable_expert_parallel": True}
GATHER = "_c10d_functional.all_gather_into_tensor.default"
#: An all-reduce fused with a norm: no collective `exchange` families, on the
#: group its `group_name` names.
FUSED = "aiter.fused_allreduce_rmsnorm_.default"


def _graphs(ep: bool):
    """`ops.graph` at one layout: attention and a GEMM at either, a MoE whose
    experts expert parallelism cuts, an all-gather across the ranks and an
    all-reduce fused with a norm on the TP group."""

    def graph(store, config, shape, arch=None):
        x = ("x", _t([shape.seqs, 8], "activation"))
        return _graph(
            _leaf("aiter.attn", x),
            _leaf("aiter.gemm", x),
            _leaf("aiter.moe", x, ("w", _t([4 if ep else 8, 8], "weight"))),
            _leaf(GATHER, x),
            _leaf(FUSED, x, ("group_name", "tp:0")),
            layers=1,
        )

    return graph


def _bench(fn, config, todo, llc):
    """`measure` as a device would time each point, a collective's record
    naming its family as the bench's does (`exchange`)."""
    got = []
    for shape, context, found in todo:
        for sig, (op, _) in found.items():
            record = {"shape": str(shape), "context": context, "op": op["op"]}
            record |= {"leaf": sig, "kernels": ["k"], "cold_ns": 1e3}
            record |= {"exchange": ex} if (ex := ops.exchange(op)) else {}
            got.append(record)
    _bench.measured += [(r["op"], r["shape"], r["context"]) for r in got]
    return got


def _plan_at(layout, bench, monkeypatch, out, population, root=None):  # noqa: F811
    """`measure ops --plan` of `population` at `layout` into `out` over the
    store at `root`, the fixture's when None; what it measured, by op."""
    _, fixture_root, _, run = bench
    root = root or fixture_root
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: layout)
    monkeypatch.setattr(ops, "graph", _graphs(layout["enable_expert_parallel"]))
    monkeypatch.setattr(ops, "_child", _bench)
    _bench.measured = []
    out.mkdir()
    (out / "run.json").write_text(json.dumps(run))
    (out / "engine.json").write_text(json.dumps({"level": 0}))
    args = SimpleNamespace(plan=[population], store=root, llc_mib=256)
    store = ArtifactStore(root)
    ops.plan(store, ops.keys(None, run), CONFIG, run, args, "measure ops", out)
    return {op for op, *_ in _bench.measured}


def _as_keyed_before(store_root, old_root, layout):
    """Each `op_measurement` in the store at `store_root` written into
    `old_root` as `measure ops` keyed it before: the layout in its key and not
    in its point's digest, a member on every rank of the layout."""
    ranks = Topology(**{a: layout[a] for a in ("dp", "pp", "pcp", "tp")})
    folder = pathlib.Path(store_root) / Kind.OP_MEASUREMENT.value
    for n, entry in enumerate(sorted(folder.glob("*/entry.json"))):
        document = json.loads(entry.read_text())
        (member,) = document["members"]
        body = (entry.parent / member).read_bytes()
        record = json.loads(body)
        point = ops.point_digest(record["shape"], record["context"], record["leaf"])
        document["key"] |= layout | {"point": point}
        document["topology"] = ranks.widths
        names = [member_name("point", r, "json") for r in ranks.ranks()]
        document["members"] = {name: document["members"][member] for name in names}
        place = pathlib.Path(old_root) / Kind.OP_MEASUREMENT.value / f"before{n}"
        place.mkdir(parents=True)
        (place / "entry.json").write_text(json.dumps(document))
        for name in names:
            (place / name).write_bytes(body)


def _migrated(old, new) -> list[str]:
    done = subprocess.run(
        [sys.executable, str(MIGRATE), "--store", str(old), str(new)],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.splitlines()


def test_a_point_keys_the_layout_only_when_its_leaf_is_a_collective(
    bench, monkeypatch  # noqa: F811
):
    *_, run = bench
    key = {}
    for name, layout in (("off", EP_OFF), ("on", EP_ON)):
        fields = "atom.compass.run.memory.key_fields"
        monkeypatch.setattr(fields, lambda c, layout=layout: layout)
        key[name] = ops.keys(None, run)
    point = ("decode B=2", 1000, '{"op":"aiter.gemm"}')
    x = ("x", _t([2, 8], "activation"))
    fused = _graph(_leaf(FUSED, x, ("group_name", "tp:0")), layers=1)
    (sig,) = ops.leaves(fused["region"])
    on_group = ("decode B=2", 1000, sig)

    assert key["off"](*point) == key["on"](*point)
    assert key["off"](*point, True) != key["on"](*point, True)
    assert key["off"](*on_group) != key["on"](*on_group)
    # a leaf's digest is the one its point had when every key stated the layout
    assert key["on"](*point).values["point"] == ops.point_digest(*point)
    assert not set(EP_ON) & set(key["on"](*point, True).values)


def test_an_ep_plan_over_a_migrated_ep_off_store_measures_only_new_leaves_and_collectives(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    population = _records(tmp_path / "plan", ([[1, 1000, True]], 1))
    measured = _plan_at(EP_OFF, bench, monkeypatch, tmp_path / "off", population)
    assert measured == {"aiter.attn", "aiter.gemm", "aiter.moe", GATHER, FUSED}
    _, today, _, _ = bench
    _as_keyed_before(today, tmp_path / "old", EP_OFF)
    folder = Kind.OP_MEASUREMENT.value
    published = {p.parent.name for p in pathlib.Path(today, folder).glob("*/entry.*")}

    lines = _migrated(tmp_path / "old", tmp_path / "new")

    # the store EP off measured, moved to the keys it is published at today
    assert lines[-1] == f"{len(published)} moved, 0 listed"
    moved = {p.parent.name for p in (tmp_path / "new" / folder).glob("*/entry.*")}
    assert moved == published
    new = str(tmp_path / "new")
    ep = tmp_path / "on"

    measured = _plan_at(EP_ON, bench, monkeypatch, ep, population, root=new)

    assert measured == {"aiter.moe", GATHER, FUSED}
    held = [json.loads(x) for x in (ep / "plan.jsonl").read_text().splitlines()]
    assert {r["op"] for r in held} == {"aiter.attn", "aiter.gemm", "aiter.moe"} | {
        GATHER,
        FUSED,
    }
    assert all(r["layout"] == EP_ON for r in held)
    fit = tmp_path / "fit"
    cli.main(["fit", "ops", str(ep), "--store", new, "--out", str(fit)])
    listed = json.loads((fit / "price_list.json").read_text())
    assert EP_ON.items() <= listed["key"].items()
    assert listed["report"]["points"] == len(held)


def test_fit_ops_refuses_a_line_that_states_no_layout(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    population = _records(tmp_path / "plan", ([[1, 1000, True]], 1))
    out = tmp_path / "off"
    _plan_at(EP_OFF, bench, monkeypatch, out, population)
    store, *_ = bench
    lines = out / "round0" / "ops.jsonl"
    old = [json.loads(x) for x in lines.read_text().splitlines()]
    for r in old:  # as measure ops wrote a line before it stated its layout
        del r["layout"]
    lines.write_text("".join(json.dumps(r) + "\n" for r in old))
    (out / "plan.jsonl").unlink()

    with pytest.raises(ValueError, match="with no parallel layout"):
        fit_ops.read(store, [out])


def test_a_run_without_plan_lists_each_point_the_store_held_at_its_own_layout(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # One DP-attention rank on one GPU at two and then four ranks: the second
    # finds the negation the first measured, and its gather refuses each time.
    for dp in (2, 4):
        with pytest.raises(ValueError, match="1 points refused"):
            _dpa_main([], bench, tmp_path / f"dp{dp}", monkeypatch, _dpa_config(dp))
    out = tmp_path / "dp4" / "out"

    lines = [json.loads(x) for x in (out / "ops.jsonl").read_text().splitlines()]

    (held,) = [r for r in lines if r["op"] == NEG]
    assert "refused" not in held and held["layout"]["dp"] == 4
    store, *_ = bench
    assert [p["op"] for p in fit_ops.read(store, [out])[0]] == [NEG]


def test_warm_over_cold_reads_a_point_on_a_group_only_at_its_price_list_s_layout(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # A GEMM and a fused all-reduce measured EP off, ratio 0.5; the all-reduce
    # again EP on, ratio 1.5, where the GEMM's point answers already.
    store, root, _, run = bench
    x = ("x", _t([2, 8], "activation"))
    region = _graph(_leaf("aiter.gemm", x), _leaf(FUSED, x, ("group_name", "tp:0")))
    point = {"shape": "decode B=2", "context": 9, "cold_ns": 1.0}
    for n, layout in enumerate((EP_OFF, EP_ON)):
        at = "atom.compass.run.memory.key_fields"
        monkeypatch.setattr(at, lambda c, layout=layout: layout)
        records = [
            point | {"op": op["op"], "leaf": sig, "warm_over_cold": 0.5 + n}
            for sig, (op, _) in ops.leaves(region["region"]).items()
            if n == 0 or op["op"] == FUSED
        ]
        (tmp_path / str(n)).mkdir()
        key = ops.keys(None, run)
        ops.publish(store, key, None, run, "measure ops", records, tmp_path / str(n))
    stated = key("decode B=2", 9, "x").values
    listed = {k: v for k, v in stated.items() if k != "point"} | EP_OFF

    rows = _warm_over_cold({"store": root, PRICE_LIST: listed})

    one = "1 of 1 points timed warm; min 0.500, median 0.500, max 0.500"
    assert {op: value for _, op, value, _ in rows} == {"aiter.gemm": one, FUSED: one}
