# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `--coverage`, with no device.

The bench is a stand-in whose kernels switch where a real one's tile or split
choice would: a GEMM at 100 tokens, an attention at 5000 tokens of context.
Graphs are built by `trace ops`'s own folding; points are published to and
read back from a real store.
"""

import json
import os
import re
import subprocess
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_artifact_store import CONDITIONS, GATES, stanza
from test_bench_ops import KEY, TREE, _graph, _leaf, _Runner, _t
from test_op_graph_arch import config as model_config  # noqa: F401  a fixture

from atom.compass import capture, cli, ir
from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.bench import ops
from atom.compass.capture import forward
from atom.compass.capture.forward import decode, structure
from atom.compass.fit import ops as fit_ops
from atom.compass.run import memory, step_grid
from atom.compass.runner.overrides import schedulable_ladder
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import PINNED
from atom.config import Config
from atom.utils import forward_context as fctx

CONFIG = SimpleNamespace(
    capture_sizes=[1, 2, 4],
    max_num_seqs=4,
    max_num_batched_tokens=8192,
    max_model_len=262144,
    speculative_config=None,
)


def _graph_at(store, config, shape, arch=None):
    rows = shape.tokens if shape.phase == "prefill" else shape.seqs
    attn = _leaf(
        "aiter.attn",
        ("q", _t([rows, 8], "activation")),
        ("layer", ir.ContextRef(KEY)),
        key=KEY,
    )
    gemm = _leaf("aiter.gemm", ("x", _t([rows, 8], "activation")))
    return _graph(attn, gemm, layers=2)


def _bench(fn, config, todo, llc):
    """`measure` as a device would time each point, a microsecond per unit: the
    GEMM by its tokens, the attention by its context, each kernel switching at
    its threshold."""
    assert fn is ops.measure
    _bench.calls += 1
    got = []
    for shape, context, found in todo:
        for sig, (op, _) in found.items():
            if shape.phase == "prefill":
                n = shape.tokens
                kernel = "p" if op["op"] == "aiter.attn" else f"gemm{int(n >= 100)}"
            elif op["op"] == "aiter.attn":  # by its longest row, ragged or not
                n = ops._split(context)[0]
                kernel = f"attn{int(n >= 5000)}"
            else:
                n, kernel = 1, "gemm"
            got.append(
                {"shape": str(shape), "context": context, "op": op["op"], "leaf": sig}
                | {"kernels": [kernel], "cold_ns": n * 1e3}
            )
    return got


def _records(path, *steps):
    """A record directory of `steps`, each `(rows, rung)`."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    lines = [
        {"step": i, "is_dummy_run": False, "rows": rows, "rung": rung}
        for i, (rows, rung) in enumerate(steps)
    ]
    (path / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return str(path)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A store, the key of a point measured on hjbog-srdc-39, the run, and
    the stand-in bench and graphs in place."""
    spec = MachineSpec.from_mapping(
        json.loads((TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text())
    )
    run = {
        "device": [{"index": 6, "name": spec.value("device.name")}],
        "stack": {c: spec.value(f"device.software_pinned_to.{c}") for c in PINNED},
        "ptl": "disabled",
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu"},
        "source": {"revision_kind": "tree", "revision": "a" * 40},
    }
    run["device"][0]["arch"] = spec.value("device.arch")
    layout = {"dp": 1, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
    layout |= {"enable_expert_parallel": False, "enable_dp_attention": False}
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: layout)
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda c: {"level": 0})
    monkeypatch.setattr(ops, "graph", _graph_at)
    monkeypatch.setattr(ops, "_child", _bench)
    _bench.calls = 0
    root = str(tmp_path / "store")
    return ArtifactStore(root), root, ops.keys(None, run), run


def _plan(bench, tmp_path, population):
    store, root, key, run = bench
    args = SimpleNamespace(plan=[population], store=root, llc_mib=256)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    ops.plan(store, key, CONFIG, run, args, "compass measure ops --plan", out)
    return [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]


def test_a_plan_densifies_each_leaf_to_adjacent_points_where_its_kernel_switches(
    bench, tmp_path
):
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
        ([[40, 40, False]], None),
        ([[150, 150, False]], None),
        ([[30, 30, False], [30, 30, False]], None),
    )

    records = _plan(bench, tmp_path, population)

    assert ops._dependent(records) == {"aiter.attn": True, "aiter.gemm": False}
    at = {}
    for r in records:
        if isinstance(r["context"], list):  # a ragged decode point
            continue
        at.setdefault((r["op"], r["shape"].split()[0]), set()).add(
            r["context"] if r["shape"].startswith("decode") else r["shape"]
        )
    # The context-free GEMM keeps the ends of the steps' contexts at every rung.
    assert at["aiter.gemm", "decode"] == {2000, 12000}
    # Attention takes the ladder of means and maxes, then halves to 4999 | 5000.
    attn = sorted(at["aiter.attn", "decode"])
    assert attn[:2] == [2000, 3000] and attn[-1] == 12000
    assert {4999, 5000} <= set(attn) and len(attn) < 20
    # GEMM rows over every B: 40, 60, 150, then halves down to 99 | 100.
    tokens = sorted(
        int(s.split("T=")[1].split()[0]) for s in at["aiter.gemm", "prefill"]
    )
    assert tokens[:2] == [40, 60] and tokens[-1] == 150
    assert {99, 100} <= set(tokens) and len(tokens) < 12
    # Attention depends on context, so its prefill points keep their rows.
    assert at["aiter.attn", "prefill"] == {
        "prefill T=40 B=1",
        "prefill T=150 B=1",
        "prefill T=60 B=2",
    }

    calls = _bench.calls
    assert _plan(bench, tmp_path, population) == records
    assert _bench.calls == calls  # every point is read back from the store


def test_coverage_counts_a_step_only_when_every_leaf_sits_in_one_kernel_s_band(
    bench, tmp_path
):
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
        ([[40, 40, False]], None),
        ([[150, 150, False]], None),
    )
    records = _plan(bench, tmp_path, plan)
    graded = _records(
        tmp_path / "graded",
        ([[1, 3500, True], [1, 4500, True]], 2),  # inside attention's first band
        ([[1, 3000, True], [1, 6000, True]], 2),  # mean 4500, max across 5000
        ([[1, 20000, True]], 1),  # past the largest context measured
        ([[120, 120, False]], None),  # GEMM between 100 and 150
        ([[120, 1120, False]], None),  # a cached row
        ([[40, 40, False]] * 3, None),  # no leaf measured at three rows
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert (report["steps"], report["covered"]) == (6, 2)
    assert report["by phase"] == {
        "decode": {"steps": 3, "covered": 1},
        "prefill": {"steps": 3, "covered": 1},
    }
    assert report["left out, by leaf kind"] == {
        "aiter.attn": {
            "between two kernels": 1,
            "outside the measured range": 1,
            "a cached row, measured on fresh rows only": 1,
            "no measured point": 1,
        },
        "aiter.gemm": {"no measured point": 1},
    }


def _cached_bench(fn, config, todo, llc):
    """`_bench` with prefill attention's kernel switching at 5000 cached tokens."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            r["kernels"] = [f"p{int((r['context'] or 0) >= 5000)}"]
    return got


def test_a_plan_measures_prefill_attention_over_cached_tokens_and_coverage_reads_them(
    bench, tmp_path, monkeypatch
):
    monkeypatch.setattr(ops, "_child", _cached_bench)
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[40, 40, False]], None),
        ([[100, 2100, False]], None),
        ([[100, 2500, False]], None),  # 2400 is under 1.25 times 2000: off the ladder
        ([[100, 3100, False]], None),
        ([[150, 8150, False]], None),
    )
    records = _plan(bench, tmp_path, plan)

    attn = {}
    for r in records:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            attn.setdefault(r["shape"], set()).add(r["context"])
    # Every token count over the cached ladder, halved to 4999 | 5000.
    assert set(attn) == {f"prefill T={t} B=1" for t in (40, 100, 150)}
    for cached in attn.values():
        assert {None, 2000, 3000, 4999, 5000, 8000} <= cached and 2400 not in cached
        assert len(cached) < 20
    # GEMM does not depend on context: its prefill points stay fresh.
    gemm = [r for r in records if r["op"] == "aiter.gemm"]
    assert {r["context"] for r in gemm if r["shape"].startswith("prefill")} == {None}
    graded = _records(
        tmp_path / "graded",
        ([[120, 1120, False]], None),  # between fresh and 2000 cached
        ([[120, 6120, False]], None),  # between 5000 and 8000, past the switch
        ([[120, 9120, False]], None),  # past the largest cached count measured
        ([[160, 1160, False]], None),  # past the largest token count measured
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 4, "covered": 2}}
    assert report["left out, by leaf kind"] == {
        "aiter.attn": {"outside the measured range": 2},
        "aiter.gemm": {"outside the measured range": 1},
    }


def _moving_bench(fn, config, todo, llc):
    """`_bench` with prefill attention's time growing with its cached tokens."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            r["cold_ns"] += (r["context"] or 0) * 1e3
    return got


def _two_mode_bench(fn, config, todo, llc):
    """`_bench` with prefill attention a short kernel that reads in one of two
    modes, 0.4 us apart, past `CONTEXT_MOVES` between fresh and cached rows."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            r["cold_ns"] = 3300.0 if r["context"] else 3700.0
    return got


@pytest.mark.parametrize(
    "stand_in, cached, covered, round2",
    [
        # one kernel and time: only the far end, and round 2 densifies
        (_bench, {None, 8000}, 2, {("aiter.gemm", None)}),
        # its two modes, under the floor: as one time
        (_two_mode_bench, {None, 8000}, 2, {("aiter.gemm", None)}),
        # its time grows: round 2 is the ladder alone, densifying after it
        (
            _moving_bench,
            {None, 2000, 3000, 8000},
            1,
            {("aiter.attn", 2000), ("aiter.attn", 3000)},
        ),
    ],
)
def test_a_plan_ladders_cached_tokens_only_for_a_prefill_kind_that_moves_there(
    bench, tmp_path, monkeypatch, stand_in, cached, covered, round2
):
    monkeypatch.setattr(ops, "_child", stand_in)
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),  # decode attention moves
        ([[40, 40, False]], None),
        ([[100, 2100, False]], None),
        ([[100, 3100, False]], None),
        ([[150, 8150, False]], None),
    )
    records = _plan(bench, tmp_path, plan)

    attn = {}
    for r in records:
        if r["op"] == "aiter.attn" and r["shape"].startswith("prefill"):
            attn.setdefault(r["shape"], set()).add(r["context"])
    assert attn == {f"prefill T={t} B=1": cached for t in (40, 100, 150)}
    lines = (tmp_path / "out" / "round2" / "ops.jsonl").read_text().splitlines()
    assert {(r["op"], r["context"]) for r in map(json.loads, lines)} == round2
    graded = _records(
        tmp_path / "graded",
        ([[120, 1120, False]], None),
        ([[120, 9120, False]], None),  # past the most cached tokens measured
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 2, "covered": covered}}


@pytest.mark.parametrize(
    "max_model_len, at_8000", [(8100, {40, 100}), (8150, {40, 100, 150})]
)
def test_a_plan_places_no_cached_prefill_point_past_max_model_len(
    bench, tmp_path, monkeypatch, max_model_len, at_8000
):
    monkeypatch.setattr(ops, "_child", _moving_bench)
    monkeypatch.setattr(CONFIG, "max_model_len", max_model_len)
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[40, 8040, False]], None),  # the most cached tokens, on a short chunk
        ([[100, 2100, False]], None),
        ([[150, 150, False]], None),
    )
    records = _plan(bench, tmp_path, plan)

    prefill = [r for r in records if r["shape"].startswith("prefill")]
    for r in prefill:
        shape = ops._shape(r["shape"])
        row = -(-shape.tokens // shape.seqs) + (r["context"] or 0)
        assert row <= max_model_len, r
    attn = {
        ops._shape(r["shape"]).tokens
        for r in prefill
        if r["op"] == "aiter.attn" and r["context"] == 8000
    }
    # At 8100, 150 tokens past 8000 cached is a row past max_model_len.
    assert attn == at_8000


@pytest.mark.parametrize(
    "tokens, lo, hi, why",
    [
        (120, 0, 1000, ""),  # fresh to 1000 cached, both of one kernel
        (120, 1000, 2000, "between two kernels"),  # 1000 and 2000 differ
        (120, 1000, 1000, ""),
        (200, 0, 0, "outside the measured range"),  # past the token counts
        (120, 1000, 5000, "outside the measured range"),  # past the cached counts
        (60, 1000, 1000, "outside the measured range"),  # 1000 holds no 60
    ],
)
def test_a_cached_bracket_holds_between_counts_and_tokens_of_one_kernel(
    tokens, lo, hi, why
):
    cuts = {
        0: ([40, 100, 150], ["a", "a", "a"]),
        1000: ([100, 150], ["a", "a"]),
        2000: ([40, 150], ["b", "b"]),
    }

    assert ops._bracket_cached(cuts, tokens, lo, hi) == why


@pytest.mark.parametrize(
    "lo, hi, why",
    [
        (10, 10, ""),  # on a point
        (12, 18, ""),  # between two of one kernel
        (18, 22, "between two kernels"),
        (5, 10, "outside the measured range"),
        (30, 31, "outside the measured range"),
        (30, 30, "between two kernels"),  # two points there disagree
    ],
)
def test_a_bracket_holds_only_between_points_of_one_kernel(lo, hi, why):
    ats, kernels = [10, 20, 25, 30, 30], ["a", "a", "b", "b", "c"]

    assert ops._bracket(ats, kernels, lo, hi) == why


@pytest.mark.parametrize(
    "rows, rung, lo",
    [
        ([4000] * 5, 8, 2500),  # three rows of padding: the rung's mean
        ([6, 10, 6], 3, 6),  # rows 0 and 2 against row 1: 12 over two
        ([1000, 3000], 2, 2000),  # the mean
        ([3], 8, 1),  # at least a token
    ],
)
def test_a_decode_step_s_lowest_context_is_over_the_rung_s_rows(rows, rung, lo):
    line = {"rows": [[1, c, True] for c in rows]}

    assert ops._rows(line, rung) == (lo, max(rows))


def test_a_tp2_plan_reads_back_every_point_it_holds(bench, tmp_path, monkeypatch):
    store, root, _, run = bench
    layout = memory.key_fields(None) | {"tp": 2}
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: layout)
    at_tp2 = (store, root, ops.keys(None, run), run)
    population = _records(tmp_path / "plan", ([[1, 3000, True]], 2))

    records = _plan(at_tp2, tmp_path, population)
    calls = _bench.calls

    assert _plan(at_tp2, tmp_path, population) == records
    assert _bench.calls == calls


def test_a_planned_decode_point_holds_the_rows_of_the_ragged_step_it_stands_for(
    bench, tmp_path
):
    # Rows of 1000 and 4000 at rung 2: their mean is 625 permille of their max.
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 4000, True]], 2),
        ([[1, 12000, True]], 1),
    )

    records = _plan(bench, tmp_path, population)

    ragged = {}
    for r in records:
        if isinstance(r["context"], list):
            assert r["op"] == "aiter.attn"  # only a kind that moves with context
            ragged.setdefault(r["shape"], set()).add(tuple(r["context"]))
    # Each context of the ladder at each `GRID`th of the max from the least the
    # rows hold, the step's own 625 among them, halved where the kernel
    # switches; a single row is never ragged.
    assert ragged.keys() == {"decode B=2", "decode B=4"}
    for shape, least in (("decode B=2", 500), ("decode B=4", 250)):
        grid = (round(1000 * k / ops.GRID) for k in range(1, ops.GRID))
        means = {least} | {m for m in grid if m > least}
        assert {m for _, m in ragged[shape]} == means
        ladder = {(c, m) for c in (2500, 4000, 12000) for m in means}
        assert ladder | {(4999, 625), (5000, 625)} <= ragged[shape]
    # The point at the step's own context holds the step's rows.
    [point] = [
        r for r in records if r["shape"] == "decode B=2" and r["context"] == [4000, 625]
    ]
    rows = ops.point_rows(decode(2), point["context"], 1 << 20, 16)
    assert rows == [[1, 4000, True], [1, 1000, True]]

    calls = _bench.calls
    assert _plan(bench, tmp_path, population) == records
    assert _bench.calls == calls  # every ragged point is read back from the store


def test_a_plan_reads_a_padded_step_s_lowest_context_over_its_rung(bench, tmp_path):
    # One row at rung 2: the lowest context is 3000 over two rows.
    population = _records(tmp_path / "plan", ([[1, 3000, True]], 2))

    records = _plan(bench, tmp_path, population)

    assert {r["context"] for r in records if not isinstance(r["context"], list)} == {
        1500,
        3000,
    }


def test_coverage_keeps_a_leaf_s_reason_whichever_leaf_of_its_op_comes_later(
    bench, tmp_path, monkeypatch
):
    def two_gemms(store, config, shape, arch=None):
        rows = shape.tokens if shape.phase == "prefill" else shape.seqs
        x = [_t([rows, width], "activation") for width in (8, 16)]
        return _graph(*(_leaf("aiter.gemm", ("x", t)) for t in x), layers=1)

    monkeypatch.setattr(ops, "graph", two_gemms)
    plan = _records(
        tmp_path / "plan", ([[1, 1000, True]], 1), ([[40, 40, False]], None)
    )
    records = _plan(bench, tmp_path, plan)
    sigs = list(
        dict.fromkeys(r["leaf"] for r in records if r["shape"] == "prefill T=40 B=1")
    )
    assert len(sigs) == 2
    graded = _records(tmp_path / "graded", ([[40, 40, False]], None))
    store, root, _, _ = bench

    # Each leaf in turn has no point; the other sits on one.
    for sig in sigs:
        kept = [r for r in records if r["leaf"] != sig]
        report = ops.coverage(store, root, CONFIG, [graded], kept, "gfx942")

        assert report["by phase"] == {"prefill": {"steps": 1, "covered": 0}}
        assert report["left out, by leaf kind"] == {
            "aiter.gemm": {"no measured point": 1}
        }


def test_coverage_reads_a_padded_step_s_lowest_context_over_its_rung(bench, tmp_path):
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
    )
    records = _plan(bench, tmp_path, plan)
    # One row at rung 2: 1500, under the lowest context measured.
    graded = _records(tmp_path / "graded", ([[1, 3000, True]], 2))
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["left out, by leaf kind"] == {
        "aiter.attn": {"outside the measured range": 1}
    }


def test_a_decode_family_halves_either_side_of_a_point_off_its_neighbours_line():
    def family(times, kernels="kkkk", shape="decode B=2"):
        return [
            (at, (k,), {"shape": shape, "context": at, "cold_ns": ns})
            for (at, ns), k in zip(times, kernels)
        ]

    families = {
        "line": family([(1000, 10.0), (2000, 20.0), (3000, 30.0)]),
        # 26 at 2000 against 20 on the line from 1000 to 3000; 30 at 3000 is on
        # the line from 2000 to 4000.
        "bent": family([(1000, 10.0), (2000, 26.0), (3000, 30.0), (4000, 34.0)]),
        # The line prices 1000 at 1051: 5.1% over 1000, 4.9% over 1051.
        "under": family([(1000, 1000.0), (2000, 1000.0), (3000, 1102.0)]),
        # The same bend across a kernel switch halves only the switch.
        "switch": family([(1000, 10.0), (2000, 26.0), (3000, 30.0)], "kkj"),
    }
    families = {json.dumps(["decode B=2", sig]): f for sig, f in families.items()}
    prefill = json.dumps(["prefill", None, "leaf"], sort_keys=True)
    families[prefill] = family(
        [(10, 1.0), (20, 9.0), (30, 3.0)], shape="prefill T=10 B=1"
    )

    want = ops._densify(families, set())

    assert {at: {json.loads(f)[1] for f in fams} for at, fams in want.items()} == {
        (decode(2), 1500): {"bent", "under"},
        (decode(2), 2500): {"bent", "under", "switch"},
    }


def test_a_plan_halves_decode_attention_until_no_point_bends_off_its_neighbours(
    bench, tmp_path, monkeypatch
):
    def steeper(fn, config, todo, llc):
        # Decode attention turns three times steeper at 3500 tokens, one kernel.
        got = _bench(fn, config, todo, llc)
        for r in got:
            if r["op"] == "aiter.attn" and r["shape"].startswith("decode"):
                r["cold_ns"] += 2e3 * max(ops._split(r["context"])[0] - 3500, 0)
        return got

    monkeypatch.setattr(ops, "_child", steeper)
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
    )

    records = _plan(bench, tmp_path, population)

    by = {}
    for r in records:
        if isinstance(r["context"], list):  # a ragged line keeps to the ladder
            continue
        if r["op"] == "aiter.attn" and r["shape"].startswith("decode"):
            by.setdefault(r["shape"], []).append(
                (r["context"], r["kernels"], r["cold_ns"])
            )
    for points in by.values():
        points.sort()
        for p, q, s in zip(points, points[1:], points[2:]):
            line = p[2] + (s[2] - p[2]) * (q[0] - p[0]) / (s[0] - p[0])
            if p[1] == q[1] == s[1] and s[0] - p[0] > 2:
                assert abs(line - q[2]) <= 0.05 * q[2], (p, q, s)
    at = {c for c, *_ in by["decode B=1"]}
    assert any(3000 < c < 4125 for c in at) and {4999, 5000} <= at


def test_a_plan_finds_a_step_its_points_hold_on_a_line_where_the_kind_steps_elsewhere(
    bench, tmp_path, monkeypatch
):
    def stepped(fn, config, todo, llc):
        # Decode attention steps up past 2500 tokens under one kernel: by 1500
        # us at width 1, which bends its points; by 150 us, 6%, at the other
        # widths, too little to bend theirs over 5% about a point the plan holds.
        got = _bench(fn, config, todo, llc)
        for r in got:
            if r["op"] == "aiter.attn" and r["shape"].startswith("decode"):
                high = 1500e3 if r["shape"] == "decode B=1" else 150e3
                r["cold_ns"] += high * (ops._split(r["context"])[0] > 2500)
        return got

    monkeypatch.setattr(ops, "_child", stepped)
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
    )

    records = _plan(bench, tmp_path, population)

    at = {}
    for r in records:
        if isinstance(r["context"], list):  # a ragged line keeps to the ladder
            continue
        if r["op"] == "aiter.attn" and r["shape"].startswith("decode"):
            at.setdefault(r["shape"], set()).add(r["context"])
    assert len(at) > 1
    for shape, contexts in at.items():
        assert {2500, 2501} <= contexts, shape
    # A ragged line is halved where it bends, not where only its kind steps.
    ragged = {r["context"][0] for r in records if isinstance(r["context"], list)}
    assert ragged and 2501 not in ragged


def test_a_plan_over_rows_alike_measures_no_ragged_point(bench, tmp_path):
    population = _records(
        tmp_path / "plan",
        ([[1, 3000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
    )

    records = _plan(bench, tmp_path, population)

    assert not [r for r in records if isinstance(r["context"], list)]


@pytest.mark.parametrize(
    "points, rows",
    [
        # one token count at three row counts, its time growing with them
        (
            [
                (64, 1, 150.4, "k"),
                (64, 2, 179.9, "k"),
                (64, 4, 245.0, "k"),
                (48, 4, 120.0, "k"),
            ],
            [[1], [2], [4]],
        ),
        ([(64, 1, 150.0, "a"), (64, 2, 150.0, "b"), (48, 2, 120.0, "b")], [[1], [2]]),
        # flat across rows: one family
        (
            [
                (64, 1, 150.0, "k"),
                (64, 2, 152.0, "k"),
                (64, 4, 151.0, "k"),
                (48, 4, 120.0, "k"),
            ],
            [[1, 2, 4]],
        ),
        # no token count at two row counts: nothing shows the rows do not matter
        ([(64, 1, 150.0, "k"), (48, 4, 120.0, "k")], [[1], [4]]),
    ],
)
def test_a_prefill_family_keeps_its_rows_unless_its_points_show_they_do_not_move_it(
    points, rows
):
    def point(shape, context, ns, kernel):
        record = {"shape": shape, "context": context, "op": "lin", "leaf": "{}"}
        return record | {"cold_ns": ns * 1e3, "kernels": [kernel]}

    # Flat over decode context, so the decode check reads it context-independent.
    records = [point("decode B=2", c, 10.0, "d") for c in (1, 9)]
    records += [point(f"prefill T={t} B={b}", None, ns, k) for t, b, ns, k in points]

    families = ops._families(records, ops._dependent(records))

    got = [
        sorted({ops._shape(r["shape"]).seqs for *_, r in found})
        for fam, found in families.items()
        if fam.startswith('["prefill"')
    ]
    assert sorted(got) == rows


def _rows_bench(fn, config, todo, llc):
    """`_bench` with the prefill GEMM's time growing with its rows."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.gemm" and r["shape"].startswith("prefill"):
            r["cold_ns"] *= ops._shape(r["shape"]).seqs
    return got


@pytest.mark.parametrize("stand_in, densified", [(_bench, 1), (_rows_bench, 2)])
def test_a_plan_densifies_a_prefill_leaf_at_each_row_count_only_where_rows_move_it(
    bench, tmp_path, monkeypatch, stand_in, densified
):
    monkeypatch.setattr(ops, "_child", stand_in)
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[40, 40, False]], None),
        ([[150, 150, False]], None),
        ([[20, 20, False]] * 2, None),
        ([[75, 75, False]] * 2, None),
    )

    records = _plan(bench, tmp_path, population)

    tokens = {1: set(), 2: set()}
    for r in records:
        if r["op"] == "aiter.gemm" and r["shape"].startswith("prefill"):
            shape = ops._shape(r["shape"])
            tokens[shape.seqs].add(shape.tokens)
    # Both row counts hold 40 and 150; the halving to 99 | 100 runs at one of
    # them when the GEMM reads flat across rows, at each when it moves.
    assert all({40, 150} <= t for t in tokens.values())
    assert sum({99, 100} <= t for t in tokens.values()) == densified


def test_a_ladder_keeps_values_a_step_apart_and_the_largest():
    assert ops._ladder([16384, 1, 2, 3, 4, 5, 100, 101, 16383]) == [
        1,
        2,
        3,
        4,
        5,
        100,
        16383,
        16384,
    ]


def test_a_kind_depends_on_context_when_its_times_or_its_kernels_move():
    def point(op, context, ns, kernel):
        record = {"shape": "decode B=2", "context": context, "op": op, "leaf": op}
        return record | {"cold_ns": ns * 1e3, "kernels": [kernel]}

    records = [
        *(point("state", c, ns, "s") for c, ns in ((1, 10.0), (9, 20.0))),
        *(point("attn", c, 10.0, k) for c, k in ((1, "a"), (9, "b"))),
        *(point("gemm", c, ns, "g") for c, ns in ((1, 10.0), (9, 10.5))),
        point("once", 1, 1.0, "o"),
    ]

    assert ops._dependent(records) == {"state": True, "attn": True, "gemm": False}


def test_an_integer_activation_indexes_a_dimension_of_one():
    spec = {"shape": [64], "dtype": "torch.int32", "stride": [1]}

    index = ops._random(spec, "cpu")

    assert torch.zeros(1, 3)[index].shape == (64, 3)


def _publish_graph(store, config, arch, shape):
    """A graph of `shape` keyed on `arch`, its one leaf named for the arch."""
    key = Key.of(
        Kind.OP_GRAPH,
        structure=structure(config, shape.phase, device=arch),
        shape=str(shape),
    )
    leaf = _leaf(f"aiter.{arch}", ("x", _t([shape.seqs, 8], "activation")))
    store.publish(
        key,
        provenance=stanza("compass trace ops"),
        topology=Topology(),
        conditions=CONDITIONS,
        gates=GATES,
        members={
            member_name("graph", ops.RANK, "json"): json.dumps(
                _graph(leaf, layers=1)
            ).encode()
        },
    )


@pytest.mark.parametrize("held", [("gfx942", "gfx950"), ("gfx942",)])
def test_coverage_reads_the_graphs_of_the_plan_s_arch_with_no_gpu(
    tmp_path, monkeypatch, held, model_config  # noqa: F811
):
    model_config.pipeline_parallel_size = model_config.prefill_context_parallel_size = 1
    store = ArtifactStore(str(tmp_path / "store"))
    for arch in held:
        _publish_graph(store, model_config, arch, decode(1))

    def no_gpu(*args, **kwargs):
        raise RuntimeError("No HIP GPUs are available")

    traced = []

    def trace(fn, argv):
        """`trace ops` as the capture publishes: on the arch `--arch` names."""
        traced.append(argv[argv.index("--arch") + 1])
        _publish_graph(store, model_config, traced[-1], decode(1))

    monkeypatch.setattr(torch.cuda, "get_device_properties", no_gpu)
    monkeypatch.setattr(
        "atom.compass.bench.measure.atom_config", lambda a: model_config
    )
    monkeypatch.setattr(ops, "_child", trace)
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.jsonl").write_text("")
    device = {"index": 0, "name": "AMD Instinct MI355X", "arch": "gfx950"}
    (out / "run.json").write_text(json.dumps({"device": [device]}))
    graded = _records(tmp_path / "graded", ([[1, 100, True]], 1))

    argv = ["--coverage", graded, "--store", str(store.root), "--out", str(out)]
    report = ops._coverage_main(argv, [])

    assert report["left out, by leaf kind"] == {
        "aiter.gfx950": {"no measured point": 1}
    }
    assert traced == ([] if "gfx950" in held else ["gfx950"])


@pytest.fixture(autouse=True)
def _declared_arch_ends_with_its_test(monkeypatch):
    """`--coverage` declares the plan's arch to the whole process, in its
    environment and Triton's driver; each test's declaration ends with it."""
    import triton

    monkeypatch.setattr(triton.runtime.driver, "_active", triton.runtime.driver._active)
    with patch.dict(os.environ):
        yield


def test_coverage_declares_the_plan_s_arch_to_aiter_before_it_builds_the_config(
    tmp_path, monkeypatch, model_config  # noqa: F811
):
    """ATOM's config imports aiter, which asks `GPU_ARCHS` and `rocminfo` for
    the arch; with no driver, `rocminfo` names none."""
    model_config.pipeline_parallel_size = model_config.prefill_context_parallel_size = 1
    driverless_bin = tmp_path / "bin"
    driverless_bin.mkdir()
    (driverless_bin / "rocminfo").write_text("#!/bin/sh\nexit 0\n")
    (driverless_bin / "rocminfo").chmod(0o755)
    monkeypatch.setenv("PATH", f"{driverless_bin}:{os.environ['PATH']}")
    asked = []

    def atom_config(args):
        found = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True)
        listing = found.stdout
        asked.append((re.findall(r"\bgfx\w+", listing), os.environ.get("GPU_ARCHS")))
        return model_config

    monkeypatch.setattr("atom.compass.bench.measure.atom_config", atom_config)
    store = ArtifactStore(str(tmp_path / "store"))
    _publish_graph(store, model_config, "gfx950", decode(1))
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.jsonl").write_text("")
    (out / "run.json").write_text(json.dumps({"device": [{"arch": "gfx950"}]}))
    graded = _records(tmp_path / "graded", ([[1, 100, True]], 1))

    argv = ["--coverage", graded, "--store", str(store.root), "--out", str(out)]
    report = ops._coverage_main(argv, [])

    assert asked == [(["gfx950"], "gfx950")]
    assert report["steps"] == 1


def test_coverage_refuses_a_plan_arch_with_no_declared_device_by_name(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.jsonl").write_text("")
    (out / "run.json").write_text(json.dumps({"device": [{"arch": "gfx90a"}]}))
    argv = ["--coverage", "graded", "--store", str(tmp_path), "--out", str(out)]

    with pytest.raises(ValueError, match="no declared device for gfx90a"):
        ops._coverage_main(argv, [])


@pytest.mark.parametrize("mode", ["--width", "--plan", "--coverage"])
def test_measure_ops_without_enforce_eager_refuses_by_name_and_traces_nothing(
    tmp_path, monkeypatch, mode, model_config  # noqa: F811
):
    graph_mode = SimpleNamespace(**vars(model_config))
    graph_mode.enforce_eager, graph_mode.tp_world_size = False, 1
    graph_mode.pipeline_parallel_size = graph_mode.prefill_context_parallel_size = 1
    graph_mode.enable_dp_attention = False
    graph_mode.compilation_config = SimpleNamespace(level=0)
    traced = []
    monkeypatch.setattr("atom.compass.bench.measure.atom_config", lambda a: graph_mode)
    monkeypatch.setattr("atom.compass.drive.gpu.hip_visible", lambda g: "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setattr(ops, "_child", lambda fn, argv: traced.append(argv))
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.jsonl").write_text("")
    (out / "run.json").write_text(json.dumps({"device": [{"arch": "gfx950"}]}))
    graded = _records(tmp_path / "graded", ([[1, 100, True]], 1))
    common = ["--store", str(tmp_path / "store"), "--out", str(out)]
    argv = {
        "--width": ["--width", "1", "--context", "100"],
        "--plan": ["--plan", graded],
        "--coverage": ["--coverage", graded],
    }[mode] + common
    if mode != "--coverage":
        argv += ["--llc-mib", "256", "--gpu", "0"]

    with pytest.raises(ValueError, match="measure with --enforce-eager"):
        ops.main([*argv, "--", "--model", model_config.model, "--level", "0"])

    assert traced == []


def test_measure_ops_learns_its_kfd_pid_before_atom_config_opens_a_gpu(
    tmp_path, monkeypatch
):
    order = []

    def atom_config(args):
        order.append("atom_config")
        raise InterruptedError

    monkeypatch.setattr("atom.compass.bench.measure.atom_config", atom_config)
    monkeypatch.setattr("atom.compass.drive.gpu.hip_visible", lambda g: "0")
    monkeypatch.setattr(
        "atom.compass.drive.gpu.kfd_pid", lambda: order.append("kfd_pid")
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    argv = ["--plan", "records", "--store", str(tmp_path / "store")]
    argv += ["--out", str(tmp_path / "out"), "--llc-mib", "256", "--gpu", "0"]

    with pytest.raises(InterruptedError):
        ops.main([*argv, "--", "--model", "m", "--enforce-eager"])

    assert order == ["kfd_pid", "atom_config"]


def test_measure_ops_plan_hands_plan_the_atom_arguments_after_the_separator(
    tmp_path, monkeypatch, model_config  # noqa: F811
):
    config = SimpleNamespace(**vars(model_config))
    config.pipeline_parallel_size = config.prefill_context_parallel_size = 1
    config.enable_dp_attention = False
    config.compilation_config = SimpleNamespace(level=0)
    got = []
    monkeypatch.setattr("atom.compass.bench.measure.atom_config", lambda a: config)
    monkeypatch.setattr("atom.compass.drive.gpu.hip_visible", lambda g: "0")
    monkeypatch.setattr("atom.compass.drive.gpu.quiet", lambda g, out: nullcontext())
    monkeypatch.setattr("atom.compass.drive.identity.identity", lambda *a: {})
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setattr(step_grid, "keyed", lambda c: {})
    monkeypatch.setattr(ops, "keys", lambda c, r: None)
    monkeypatch.setattr(ops, "plan", lambda *a: got.append(a[7:]))
    atom_args = ["--model", config.model, "--enforce-eager", "--enable-expert-parallel"]
    argv = ["--plan", "records", "--store", str(tmp_path / "store")]
    argv += ["--out", str(tmp_path / "out"), "--llc-mib", "256", "--gpu", "0"]

    ops.main([*argv, "--", *atom_args])

    assert got == [(atom_args,)]


def _dies(*finished):
    """`_bench` as a child that dies once for each entry of `finished`, having
    journaled that many of its points, and then lives; and the records it was
    measuring when it died."""
    deaths, dead = list(finished), []

    def child(fn, config, todo, llc):
        got = _bench(fn, config, todo, llc)
        if not deaths:
            return got
        n = deaths.pop(0)
        dead.append(got[n])
        ops._journal(os.environ[ops.JOURNAL], got[:n])
        raise BrokenProcessPool("killed")

    return child, dead


POPULATION = ([[1, 1000, True], [1, 3000, True]], 2), ([[40, 40, False]], None)


def test_a_child_that_dies_costs_the_plan_only_the_point_it_was_measuring(
    bench, tmp_path, monkeypatch
):
    population = _records(tmp_path / "plan", *POPULATION)
    *_, key, run = bench
    clean = str(tmp_path / "clean")
    whole = _plan((ArtifactStore(clean), clean, key, run), tmp_path, population)
    # The first dies on a point's attention leaf, its GEMM leaf still to come;
    # the second dies on that GEMM leaf having finished nothing.
    child, dead = _dies(2, 0)
    monkeypatch.setattr(ops, "_child", child)
    records = _plan(bench, tmp_path, population)

    assert [r for r in records if "refused" in r] == [
        {k: d[k] for k in ("shape", "context", "op", "leaf")}
        | {"refused": "the measuring process died"}
        for d in dead
    ]
    assert [r for r in records if "refused" not in r] == [
        r for r in whole if {k: r[k] for k in r if k != "token"} not in dead
    ]
    monkeypatch.setattr(ops, "_child", _bench)
    again = _plan(bench, tmp_path, population)  # the next plan retries them
    assert sorted(again, key=json.dumps) == sorted(whole, key=json.dumps)


def test_a_child_that_dies_at_startup_every_time_fails_the_plan(
    bench, tmp_path, monkeypatch
):
    population = _records(tmp_path / "plan", *POPULATION)
    child, dead = _dies(0, 0)
    monkeypatch.setattr(ops, "_child", child)

    with pytest.raises(BrokenProcessPool):
        _plan(bench, tmp_path, population)
    assert len(dead) == 2


def test_a_plan_resumed_into_its_out_keeps_both_runs_points_for_fit_ops(
    bench, tmp_path, monkeypatch
):
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
        ([[150, 150, False]], None),
    )
    store, root, key, run = bench
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.json").write_text(json.dumps(run))
    (out / "engine.json").write_text(json.dumps({"level": 0}))
    publish = ops.publish

    def stopped_after_round1(*args):
        publish(*args)
        if args[-1].name == "round1":
            raise KeyboardInterrupt

    monkeypatch.setattr(ops, "publish", stopped_after_round1)
    with pytest.raises(KeyboardInterrupt):
        _plan(bench, tmp_path, population)
    first = (out / "plan.jsonl").read_text()  # written after each round
    monkeypatch.setattr(ops, "publish", publish)
    records = _plan(bench, tmp_path, population)

    tokens = {r["token"] for r in records if "refused" not in r}
    assert {json.loads(x)["token"] for x in first.splitlines()} < tokens
    assert (out / "round0" / "ops.jsonl").read_text()  # the first run's, kept
    got = fit_ops.read(store, [out])[0]
    assert {key(p["shape"], p["context"], p["leaf"]).token() for p in got} == tokens
    fit = tmp_path / "fit"
    cli.main(["fit", "ops", str(out), "--store", root, "--out", str(fit)])
    assert json.loads((fit / "price_list.json").read_text())["report"]["points"] == len(
        tokens
    )


def test_measure_journals_each_record_as_it_finishes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    runner, journal = _Runner(), tmp_path / "points.jsonl"
    runner.exit = lambda: None
    placed = SimpleNamespace(block_size=16, batch=lambda view, block, slot: (0, 0))
    monkeypatch.setattr(ops, "_runner", lambda config, rank: runner)
    monkeypatch.setattr("atom.compass.bench.replay.ReplayRule", lambda c: placed)
    monkeypatch.setenv(ops.JOURNAL, str(journal))
    x = ("self", _t([2, 3], "activation"))
    found = ops.leaves(
        _graph(_leaf("aten.neg.default", x), _leaf("aten.abs.default", x))["region"]
    )

    got = ops.measure(None, [(decode(2), 20, found)], 10_000)

    assert len(got) == 2
    assert [json.loads(line) for line in journal.read_text().splitlines()] == got


#: What the `bench` fixture stands in for: the op_graph reader and the engine
#: config an op_graph's key states.
GRAPH, KEYED = ops.graph, step_grid.keyed

#: The architecture of the model whose config.json the test writes.
ARCH = "Qwen3ForCausalLM"


class _Config(Config):
    """ATOM's `Config` with its fields as ATOM's arguments resolve them, the
    model and quantization left unread: reading them imports aiter, which a
    process with no driver cannot do twice."""

    def __post_init__(self):
        self.hf_config = SimpleNamespace(architectures=[ARCH])
        self.quant_config = SimpleNamespace(compute_hash=lambda: "unquantized")


@pytest.mark.parametrize(
    "parallel, widths",
    [("--enable-dp-attention", "tp=1 dp=8"), ("--enable-expert-parallel", "tp=8 ep=8")],
)
def test_a_plan_traces_a_missing_graph_under_atom_s_arguments_and_reads_it_back(
    bench, tmp_path, monkeypatch, parallel, widths
):
    """`trace ops` runs as `--plan` starts it, its trace left out: the config
    it builds from its arguments keys the graph it publishes."""
    store, root, key, run = bench
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": [ARCH]}))
    atom_args = ["--model", str(model), "-tp", "8", "--enforce-eager", parallel]
    monkeypatch.setattr("atom.config.Config", _Config)
    config = forward._config(str(model), 8, atom_args)  # laid out, as `main` does
    config.max_num_seqs, config.max_model_len = 4, CONFIG.max_model_len
    published = []

    def trace(model, tp, shapes, tmpdir, atom_args, arch, population=()):
        traced = forward._config(model, tp, atom_args)
        for shape in shapes:
            at = structure(traced, shape.phase, device=arch)
            yield SimpleNamespace(
                applicability=SimpleNamespace(structure=at, shape=shape)
            ), None, {}

    def publish(store, graph, model, declared, tp, atom_args, command):
        at = graph.applicability
        published.append((at.structure, tp))
        key = Key.of(Kind.OP_GRAPH, structure=at.structure, shape=str(at.shape))
        store.publish(
            key,
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=CONDITIONS,
            gates=GATES,
            members={
                member_name("graph", ops.RANK, "json"): json.dumps(
                    _graph_at(store, None, at.shape)
                ).encode()
            },
        )
        return key

    def child(fn, *args):
        if fn is not capture.main:
            return _bench(fn, *args)
        with (
            patch.object(forward, "capture", trace),
            patch.object(capture, "publish", publish),
        ):
            return fn(*args)

    monkeypatch.setattr(ops, "graph", GRAPH)
    monkeypatch.setattr(step_grid, "keyed", KEYED)
    monkeypatch.setattr(ops, "_child", child)
    monkeypatch.setattr(forward, "device_arch", lambda: "gfx942")
    steps = ([[1, 1000, True]], 1), ([[40, 40, False]], None)
    args = SimpleNamespace(
        plan=[_records(tmp_path / "plan", *steps)], store=root, llc_mib=256
    )

    ops.plan(
        store, key, config, run, args, "compass measure ops --plan", tmp_path, atom_args
    )

    assert published
    assert {(s.split("} ")[-1], tp) for s, tp in published} == {
        (f"{widths} {phase} eager", 8) for phase in ("decode", "prefill")
    }
    records = [
        json.loads(x) for x in (tmp_path / "plan.jsonl").read_text().splitlines()
    ]
    rungs = {f"decode B={b}" for b in schedulable_ladder(config)}
    shapes = {r["shape"] for r in records}
    assert {s for s in shapes if s.startswith("decode")} == rungs
    assert "prefill T=40 B=1" in shapes
    assert not [r for r in records if "refused" in r]


def test_coverage_traces_a_dpa_tp8_graph_and_reads_it_at_tp1_dp8(tmp_path, monkeypatch):
    """`--coverage` builds ATOM's config from its arguments, lays it out as
    `CoreManager` does, and traces a graph the store lacks under the same
    arguments, keyed by `trace ops`'s own parsing; the read after finds it."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": [ARCH]}))
    atom_args = ["--model", str(model), "-tp", "8", "--enforce-eager"]
    atom_args.append("--enable-dp-attention")
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2)
    monkeypatch.setattr("atom.config.Config", _Config)
    monkeypatch.setattr(
        "atom.model_engine.llm_engine._load_tokenizer", lambda *a: tokenizer
    )
    store, traced = ArtifactStore(str(tmp_path / "store")), set()

    def trace(model, tp, shapes, tmpdir, atom_args, arch, population=()):
        config = forward._config(model, tp, atom_args)
        for shape in shapes:
            traced.add(structure(config, shape.phase, device=arch).split("} ")[-1])
            _publish_graph(store, config, arch, shape)
        return []

    monkeypatch.setattr(forward, "capture", trace)
    monkeypatch.setattr(ops, "_child", lambda fn, argv: fn(argv))
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.jsonl").write_text("")
    (out / "run.json").write_text(json.dumps({"device": [{"arch": "gfx942"}]}))
    steps = ([[1, 1000, True]], 1), ([[40, 40, False]], None)
    graded = _records(tmp_path / "graded", *steps)

    argv = ["--coverage", graded, "--store", str(store.root), "--out", str(out)]
    report = ops._coverage_main(argv, atom_args)

    assert traced == {"tp=1 dp=8 decode eager", "tp=1 dp=8 prefill eager"}
    assert report["by phase"] == {
        "decode": {"steps": 1, "covered": 0},
        "prefill": {"steps": 1, "covered": 0},
    }


def _kinked_bench(fn, config, todo, llc):
    """`_bench` with the prefill GEMM five times steeper past 150 tokens, one
    kernel either side."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.gemm" and r["shape"].startswith("prefill"):
            r["cold_ns"] += 4e3 * max(ops._shape(r["shape"]).tokens - 150, 0)
    return got


def test_coverage_leaves_out_a_step_inside_a_gap_the_fit_declines(
    bench, tmp_path, monkeypatch
):
    monkeypatch.setattr(ops, "_child", _kinked_bench)
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[40, 40, False]], None),
        ([[150, 150, False]], None),
        ([[400, 400, False]], None),
    )
    records = _plan(bench, tmp_path, population)
    params = fit_ops.fit(records)
    fit_ops.held_out(params)
    # T = 150 reads +67% off 122 and 400, its neighbours along the GEMM's T
    graded = _records(
        tmp_path / "graded", ([[120, 120, False]], None), ([[200, 200, False]], None)
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records, "gfx942")

    assert report["by phase"] == {"prefill": {"steps": 2, "covered": 1}}
    assert report["left out, by leaf kind"] == {"aiter.gemm": {"the fit declined T": 1}}
    gaps = [d[2:4] for f in params["families"].values() for d in f["declined"]]
    assert gaps == [[122, 400]]
