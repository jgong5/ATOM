# SPDX-License-Identifier: MIT
"""`compass fit ops`: a price_list from measured points, and what it declines.

Points are published by `measure ops`'s own `publish` into the store the fit
reads, beside the run.json and engine.json `measure ops` writes.
"""

import json
import pathlib

import pytest

from atom.compass import cli, run
from atom.compass.artifacts import ArtifactRefusal, ArtifactStore, GateState, Key, Kind
from atom.compass.bench import ops as bench
from atom.compass.capture.peers import MORI
from atom.compass.fit import ops, publish
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import PINNED

TREE = pathlib.Path(__file__).resolve().parents[2]
SPEC = MachineSpec.from_mapping(
    json.loads((TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text())
)
LAYOUT = {"dp": 1, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
LAYOUT |= {"enable_expert_parallel": False, "enable_dp_attention": False}
BF16 = "torch.bfloat16"


def _leaf(m, n, cu=False):
    """A GEMM's signature at `m` rows into `n` outputs, as `measure ops` keys it;
    with `cu`, also an `m + 1` offsets vector, as attention's signatures hold."""
    x = {"shape": [m, 64], "stride": [64, 1], "dtype": BF16, "role": "activation"}
    w = {"shape": [n, 64], "stride": [64, 1], "dtype": BF16, "role": "weight"}
    sig = {"op": "aiter.gemm", "args": {"x": x, "w": w}, "out": [[m, n]]}
    if cu:
        sig["args"]["cu"] = {"shape": [m + 1], "dtype": "torch.int32"}
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _point(b, ns, kernel="k", n=128, context=1024, cu=False):
    return {
        "shape": f"decode B={b}",
        "context": context,
        "op": "aiter.gemm",
        "leaf": _leaf(b, n, cu),
        "cold_ns": ns,
        "kernels": [kernel],
    }


def _run(ptl="disabled", root="a" * 40):
    stack = {c: SPEC.value(f"device.software_pinned_to.{c}") for c in PINNED}
    card = {"index": 6, "name": SPEC.value("device.name")}
    return {
        "device": [card | {"arch": SPEC.value("device.arch")}],
        "stack": stack,
        "ptl": ptl,
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu"},
        "source": {"revision_kind": "tree", "revision": root},
    }


@pytest.fixture
def measured(tmp_path, monkeypatch):
    """A `measure ops --out` directory of `records`, published to the store;
    with `plan`, a `--plan` one, `records` a list of rounds each in round<n>."""
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: LAYOUT)
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda c: {"level": 0})

    def make(name, records, plan=False, **run):
        run, out = _run(**run), tmp_path / name
        out.mkdir()
        (out / "run.json").write_text(json.dumps(run))
        (out / "engine.json").write_text(json.dumps({"level": 0}))
        store = ArtifactStore(str(tmp_path / "store"))
        key = bench.keys(None, run)
        for n, got in enumerate(records if plan else [records]):
            at = out / f"round{n}" if plan else out
            at.mkdir(exist_ok=True)
            bench.publish(store, key, None, run, "compass measure ops", got, at)
        return out, key

    return make


def _families(points):
    got = ops.fit(points)["families"].values()
    return sorted((f["dims"], [p[:2] for p in f["points"]]) for f in got)


def test_a_family_masks_the_rows_and_keeps_the_weight():
    points = [_point(2, 10.0), _point(8, 16.0), _point(2, 20.0, n=256)]
    points += [_point(8, 26.0, n=256), _point(2, 30.0, cu=True)]
    points += [_point(8, 36.0, cu=True)]

    assert _families(points) == [
        (["B", "context"], [[[2, 1024], 10.0], [[8, 1024], 16.0]]),
        (["B", "context"], [[[2, 1024], 20.0], [[8, 1024], 26.0]]),
        (["B", "context"], [[[2, 1024], 30.0], [[8, 1024], 36.0]]),
    ]


def test_two_leaves_each_measured_at_one_width_are_two_families():
    # N differs between the two, and N is no count of either shape
    points = [_point(2, 10.0, n=128), _point(8, 26.0, n=256)]

    assert _families(points) == [
        (["B", "context"], [[[2, 1024], 10.0]]),
        (["B", "context"], [[[8, 1024], 26.0]]),
    ]


LINE = {(2,): (10.0, ["a"]), (8,): (16.0, ["a"]), (32,): (40.0, ["b"])}


@pytest.mark.parametrize("at, ns", [(2, 10.0), (4, 12.0), (8, 16.0), (32, 40.0)])
def test_a_query_takes_a_measured_price_or_interpolates_one_kernel(at, ns):
    assert ops.interpolate(LINE, (at,), ["B"]) == (ns, ["b" if at == 32 else "a"])


@pytest.mark.parametrize(
    "at, why",
    [
        (1, "B = 1 is outside the measured 2 to 32"),
        (33, "B = 33 is outside the measured 2 to 32"),
        (16, "B = 16 lies between 8 and 32, which launched different kernels: a and b"),
    ],
)
def test_a_query_outside_the_hull_or_between_kernels_declines_with_its_range(at, why):
    with pytest.raises(ops.Declined, match=f"^{why}$"):
        ops.interpolate(LINE, (at,), ["B"])


def test_a_grid_interpolates_multilinearly_and_declines_a_missing_corner():
    grid = {(2, 1024): (10.0, ["a"]), (8, 1024): (16.0, ["a"])}
    grid |= {(2, 2048): (20.0, ["a"]), (8, 2048): (32.0, ["a"])}
    dims = ["B", "context"]

    assert ops.interpolate(grid, (4, 1536), dims) == (18.0, ["a"])
    del grid[8, 2048]
    with pytest.raises(ops.Declined, match="^at B = 8, context = 1536 is outside"):
        ops.interpolate(grid, (4, 1536), dims)


def _fit(out, store, *dirs):
    cli.main(["fit", "ops", *map(str, dirs), "--store", str(store), "--out", str(out)])
    return json.loads((out / "price_list.json").read_text())


def test_fit_ops_publishes_its_points_key_and_their_held_out_error(measured, tmp_path):
    # 4 sits inside one band; 12 launched x between two k neighbours
    points = [_point(2, 10.0), _point(4, 12.5), _point(8, 16.0)]
    points += [_point(12, 9.0, kernel="x"), _point(16, 24.0), _point(32, 50.0, "b")]
    d, key = measured("m", points + [_point(4, 1.0, n=512) | {"refused": "no"}])

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    fields = key("decode B=2", 1024, _leaf(2, 128)).values
    assert got["key"] == {k: v for k, v in fields.items() if k != "point"}
    assert got["report"]["by_op"] == {
        "decode aiter.gemm": {"held_out": 4, "declined": 2, "on_declined_lines": 4}
    }
    declined = [r["declined"] for r in got["report"]["held_out"] if "declined" in r]
    assert declined == [
        "B = 8 lies between 4 and 12, which launched different kernels: k and x",
        "B = 16 lies between 12 and 32, which launched different kernels: x and b",
    ]
    (f,) = got["params"]["families"].values()
    assert f["declined"] == [
        ["B", [1024], "B = 12 launched other kernels than its neighbours"]
    ]


def test_fit_ops_reads_a_plan_run_s_rounds_as_one_directory(measured, tmp_path):
    # 4 is held out between 2 and 8, each measured in another round
    rounds = [[_point(2, 10.0), _point(8, 16.0)], [_point(4, 12.5)]]
    rounds += [[_point(4, 1.0, n=512) | {"refused": "no"}]]
    d, key = measured("plan", rounds, plan=True)

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    fields = key("decode B=2", 1024, _leaf(2, 128)).values
    assert got["key"] == {k: v for k, v in fields.items() if k != "point"}
    (f,) = got["params"]["families"].values()
    assert f["points"] == [
        [[b, 1024], ns, ["k"]] for b, ns in ((2, 10.0), (8, 16.0), (4, 12.5))
    ]
    assert f["brackets"] == [["B", [1024], 2, 8]]


def test_a_line_holding_an_error_over_5pct_either_way_is_declined_with_its_reason():
    # held out along B, B = 4 reads -4% at context 1024 and -20% at 2048;
    # along context at B = 2, context 2048 reads +25%
    points = [_point(2, 10e3), _point(4, 12.5e3), _point(8, 16e3)]
    points += [
        _point(b, ns, context=2048) for b, ns in ((2, 20e3), (4, 30e3), (8, 32e3))
    ]
    points += [_point(2, 40e3, context=3072)]
    params = ops.fit(points)
    rows = ops.held_out(params)
    (f,) = params["families"].values()

    assert f["declined"] == [
        ["context", [2], "context = 2048 reads +25.0% off its neighbours"],
        ["B", [2048], "B = 4 reads -20.0% off its neighbours"],
    ]
    assert ops.summary(rows)["decode aiter.gemm"]["max_error"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "rows, contexts, declined",
    [
        ((2, 2, 2), (511, 512, 513), []),
        ((2, 2, 2), (510, 512, 514), ["context = 512 reads +10.0%"]),
        ((2, 2, 2), (511, 512, 514), ["context = 512 reads +6.7%"]),
        ((2, 2, 2), (510, 512, 513), ["context = 512 reads +13.3%"]),
        ((1, 2, 3), (512, 512, 512), []),
    ],
)
def test_fit_ops_holds_out_no_point_between_neighbours_one_apart(
    rows, contexts, declined, measured, tmp_path
):
    # one kernel stepping just above the middle point, which its neighbours price
    # 6.7% to 13.3% high unless both are one away along that dimension
    points = zip(rows, contexts, (1e4, 1e4, 12e3))
    d, _ = measured("m", [_point(b, ns, context=c) for b, c, ns in points])

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    (f,) = got["params"]["families"].values()
    assert [why.split(" off")[0] for _, _, why in f["declined"]] == declined
    held = [r["at"] for r in got["report"]["held_out"]]
    assert held == ([[2, 512]] if declined else [])


BEND = "bytes = 8192 reads +10.0%"
MOE = "atom.model_ops.moe:"


@pytest.mark.parametrize(
    "rows, nbytes, declined, shape, op",
    [
        ((7, 8, 9), (7, 8, 9), [], "decode B={}", "_c10d_functional.all_"),
        ((6, 8, 10), (6, 8, 10), [BEND], "decode B={}", "_c10d_functional.all_"),
        ((7, 8, 10), (7, 8, 9), [BEND], "decode B={}", "_c10d_functional.all_"),
        ((7, 8, 9), (7, 8, 9), [], "prefill T={} B=1", "_c10d_functional.all_"),
        ((7, 8, 9), (7, 8, 9), [BEND], "decode B={}", bench.DP_GROUP[0]),
        ((7, 8, 9), (7, 8, 9), [BEND], "prefill T={} B=1", MORI + ".dispatch"),
        ((7, 8, 9), (7, 8, 9), [BEND], "decode B={}", bench.DP_GROUP[1]),
        ((7, 8, 9), (7, 8, 9), [BEND], "prefill T={} B=1", MORI + ".combine"),
        ((7, 8, 9), (7, 8, 9), [], "decode B={}", MOE + "all_gather_with_padding"),
        (
            (7, 8, 9),
            (7, 8, 9),
            [],
            "decode B={}",
            MOE + "reduce_scatter_with_unpadding",
        ),
    ],
)
def test_fit_ops_holds_out_no_collective_point_between_neighbours_one_row_apart(
    rows, nbytes, declined, shape, op, measured, tmp_path
):
    # nbytes in rows of 1024 bytes; the third case's points move other bytes per
    # row, so a row's bytes are not known; all_gatherv's, reduce_scatterv's and
    # MORI's ranks' rows can differ, so a step can hold between two points' rows;
    # the padded pair pads every rank to the same rows, so a row keeps its bytes
    fam = json.dumps(["exchange", {"op": op}])
    points = [
        {"shape": shape.format(b), "context": 1024, "op": op}
        | {"leaf": "{}", "exchange": [fam, n * 1024], "cold_ns": ns, "kernels": ["k"]}
        for b, n, ns in zip(rows, nbytes, (1e4, 1e4, 12e3))
    ]
    d, _ = measured("m", points)

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    (f,) = got["params"]["families"].values()
    assert [why.split(" off")[0] for _, _, why in f["declined"]] == declined


@pytest.mark.parametrize(
    "moved, cell",
    [({"ptl": "disabled"}, "PTL"), ({"source_root": "tree-" + "b" * 40}, "ATOM src")],
)
def test_a_price_list_refuses_at_load_on_another_ptl_state_or_source_root(
    measured, tmp_path, moved, cell
):
    d, _ = measured("m", [_point(2, 10.0), _point(8, 16.0)], ptl="enabled")
    got = _fit(tmp_path / "fit", tmp_path / "store", d)
    store = ArtifactStore(str(tmp_path / "store"))
    key = Key.of(Kind.PRICE_LIST, **got["key"])
    _, config, _ = ops.read(store, [d])
    gates = GateState.of()

    assert store.load(key, conditions=publish.conditions(config), gates=gates)
    with pytest.raises(ArtifactRefusal, match=f"`price_list` x `{cell}`: recorded"):
        store.load(key, conditions=publish.conditions(config | moved), gates=gates)


def test_tp2_points_publish_a_tp2_price_list_that_a_tp2_run_loads(
    measured, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "atom.compass.run.memory.key_fields", lambda c: LAYOUT | {"tp": 2}
    )
    d, _ = measured("m", [_point(2, 10.0), _point(8, 16.0)])

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    store = ArtifactStore(str(tmp_path / "store"))
    _, config, _ = ops.read(store, [d])
    monkeypatch.setattr(
        "atom.compass.run.step_grid.conditions",
        lambda c, m: publish.conditions(config),
    )
    loaded = run._price_list(
        {"store": str(tmp_path / "store"), "price_list": got["key"]}, None, None
    )
    assert got["key"]["tp"] == 2
    assert store.read(Key.of(Kind.PRICE_LIST, **got["key"])).topology.tp == 2
    assert loaded.params == got["params"]


def test_points_naming_two_price_lists_refuse(measured, tmp_path):
    a, _ = measured("a", [_point(2, 10.0)])
    b, _ = measured("b", [_point(8, 16.0)], root="b" * 40)

    with pytest.raises(SystemExit, match="^compass fit ops: the points name 2 price"):
        _fit(tmp_path / "fit", tmp_path / "store", a, b)
    assert not (tmp_path / "store" / "price_list").exists()


def _prefill(t, b, ns, op="aiter.gemm", context=None, rows=None):
    """A prefill point of `t` tokens over `b` rows, its GEMM over the tokens."""
    point = _point(b, ns, context=context) | {"shape": f"prefill T={t} B={b}"}
    point |= {"op": op, "leaf": _leaf(t, 128)}
    return point | ({"rows": rows} if rows else {})


def test_a_kind_that_does_not_move_with_context_runs_over_the_counts_it_follows():
    # the GEMM reads within 10% at two contexts of each width, so the decode
    # points at one B are one point; the prefill points at one T too, B aside
    points = [_point(2, 100e3), _point(2, 104e3, context=2048)]
    points += [_point(8, 160e3), _point(8, 160e3, context=2048)]
    points += [_prefill(64, 1, 300e3), _prefill(64, 2, 320e3), _prefill(128, 1, 5e5)]

    assert _families(points) == [
        (["B"], [[[2], 102e3], [[8], 160e3]]),
        (["T"], [[[64], 310e3], [[128], 5e5]]),
    ]


def test_a_decode_point_sits_at_its_longest_row_a_prefill_at_its_rows_mean_cached():
    # the decode GEMM moves with context; the prefill kind with cached tokens
    points = [_point(2, 100e3)]
    points += [
        _point(2, 200e3, context=3000) | {"rows": [[1, 3000, True], [1, 1000, True]]}
    ]
    points += [_prefill(64, 2, 300e3, "attn"), _prefill(64, 2, 310e3, "attn", 0)]
    uneven = [[32, 32 + 256, False], [32, 32 + 768, False]]
    points += [_prefill(64, 2, 400e3, "attn", 512, rows=uneven)]

    assert _families(points) == [
        (["B", "context"], [[[2, 1024], 100e3], [[2, 3000], 200e3]]),
        (["T", "B", "context"], [[[64, 2, 0], 305e3], [[64, 2, 512], 400e3]]),
    ]


def test_a_prefill_kind_reads_its_own_prefill_points_on_context_first():
    # flat over a decode's context, yet twice the time past 1024 cached tokens
    points = [_point(2, 100e3), _point(2, 100e3, context=2048)]
    points += [_prefill(64, 1, 300e3, context=0), _prefill(64, 1, 6e5, context=1024)]

    assert _families(points) == [
        (["B"], [[[2], 100e3]]),
        (["T", "B", "context"], [[[64, 1, 0], 300e3], [[64, 1, 1024], 6e5]]),
    ]


def test_a_prefill_kind_its_rows_move_runs_over_its_rows():
    # flat in cached tokens, yet twice the time over two rows of one T
    points = [_prefill(64, 1, 300e3, context=0), _prefill(64, 1, 300e3, context=1024)]
    points += [_prefill(64, 2, 6e5, context=0)]

    assert _families(points) == [(["B"], [[[1], 300e3], [[2], 6e5]])]


def test_a_stated_gathered_count_is_a_dimension_its_points_kept_apart():
    # an MoE leaf over the rows the group gathers, at two G of each T
    points = [
        _prefill(t, 1, ns)
        | {"shape": f"prefill T={t} B=1 G={g}", "leaf": _leaf(g, 128)}
        for t, g, ns in (
            (64, 128, 1e5),
            (64, 256, 2e5),
            (128, 256, 21e4),
            (128, 512, 4e5),
        )
    ]

    assert _families(points) == [
        (
            ["T", "B", "G", "context"],
            [
                [[64, 1, 128, 0], 1e5],
                [[64, 1, 256, 0], 2e5],
                [[128, 1, 256, 0], 21e4],
                [[128, 1, 512, 0], 4e5],
            ],
        )
    ]


def test_points_at_one_coordinate_that_launched_other_kernels_refuse():
    points = [_point(2, 10.0), _point(2, 10.0, "x") | {"rows": [[1, 1024, True]] * 2}]

    with pytest.raises(ValueError, match=r"launched \['k'\] and other kernels at"):
        ops.fit(points)


@pytest.mark.parametrize(
    "ns,brackets",
    [
        ({1: 1e3, 64: 64e3}, []),
        ({1: 1e3, 8: 8e3, 64: 64e3}, [["B", [1024], 1, 64]]),
        (
            {1: 1e3, 8: 8e3, 16: 16e3, 64: 64e3},
            [["B", [1024], 1, 16], ["B", [1024], 8, 64]],
        ),
        ({1: 1e3, 8: 4e3, 64: 64e3}, []),
    ],
    ids=["two-points", "one-held-out", "two-held-out", "declined"],
)
def test_a_line_that_interpolates_records_each_held_out_point_s_bracket(ns, brackets):
    params = ops.fit([_point(b, t) for b, t in ns.items()])
    ops.held_out(params)
    (f,) = params["families"].values()

    assert f["brackets"] == brackets
