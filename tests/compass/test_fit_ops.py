# SPDX-License-Identifier: MIT
"""`compass fit ops`: a price_list from measured points, and what it declines.

Points are published by `measure ops`'s own `publish` into the store the fit
reads, beside the run.json and engine.json `measure ops` writes.
"""

import json
import pathlib

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_op_work import leaf, t

from atom.compass import cli, run
from atom.compass.artifacts import ArtifactRefusal, ArtifactStore, GateState, Key, Kind
from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import LISTED, OpGraphBackend
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.resolver import CostRefused
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
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu", "atom": root},
        "source": {"revision_kind": "tree", "revision": "a" * 40},
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
    assert got["key"] == {k: v for k, v in fields.items() if k != "point"} | LAYOUT
    assert got["report"]["by_op"] == {
        "decode aiter.gemm": {"held_out": 4, "declined": 2, "declining": 1}
        | {"median_error": pytest.approx(0.04), "max_error": pytest.approx(0.04)}
    }
    declined = [r["declined"] for r in got["report"]["held_out"] if "declined" in r]
    assert declined == [
        "B = 8 lies between 4 and 12, which launched different kernels: k and x",
        "B = 16 lies between 12 and 32, which launched different kernels: x and b",
    ]
    (f,) = got["params"]["families"].values()
    assert f["declined"] == [
        ["B", [1024], 8, 16, "B = 12 launched other kernels than its neighbours"]
    ]


def test_fit_ops_reads_a_plan_run_s_rounds_as_one_directory(measured, tmp_path):
    # 4 is held out between 2 and 8, each measured in another round
    rounds = [[_point(2, 10.0), _point(8, 16.0)], [_point(4, 12.5)]]
    rounds += [[_point(4, 1.0, n=512) | {"refused": "no"}]]
    d, key = measured("plan", rounds, plan=True)

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    fields = key("decode B=2", 1024, _leaf(2, 128)).values
    assert got["key"] == {k: v for k, v in fields.items() if k != "point"} | LAYOUT
    (f,) = got["params"]["families"].values()
    assert f["points"] == [
        [[b, 1024], ns, ["k"]] for b, ns in ((2, 10.0), (8, 16.0), (4, 12.5))
    ]
    assert f["brackets"] == [["B", [1024], 2, 8]]


def test_a_point_off_by_over_5pct_either_way_declines_its_gaps_with_its_reason():
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
        ["context", [2], 1024, 3072, "context = 2048 reads +25.0% off its neighbours"],
        ["B", [2048], 2, 8, "B = 4 reads -20.0% off its neighbours"],
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
    assert [why.split(" off")[0] for *_, why in f["declined"]] == declined
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
    assert [why.split(" off")[0] for *_, why in f["declined"]] == declined


@pytest.mark.parametrize(
    "moved, cell",
    [
        ({"ptl": "disabled"}, "PTL"),
        ({"source_root": "atom_py-" + "b" * 40}, "ATOM src"),
    ],
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


DECLINED = (
    "aiter.gemm: the fit declined B at context = 1024: "
    "B = 16 reads -6.5% off its neighbours"
)


@pytest.mark.parametrize(
    "b, priced",
    [(3, 3e-6), (10, DECLINED), (16, 17.12e-6), (20, DECLINED), (29, 29e-6)],
)
def test_a_point_over_5pct_off_declines_only_the_two_gaps_beside_it(b, priced):
    # B = 16 reads -6.5% off its neighbours 4 and 28, which read within 5%
    points = [(2, 2e3), (4, 4e3), (16, 17.12e3), (28, 28e3), (30, 30e3)]
    params = ops.fit([_point(*p) for p in points])
    ops.held_out(params)
    leaf = json.loads(_leaf(b, 128))

    got = ops.PriceList(params).price((leaf, ("decode", None, b, None, 1024, 1024), 0))

    if isinstance(priced, str):
        assert got.reason == priced
    else:
        assert got.seconds == pytest.approx(priced)
        assert got.provenance.validated is (b != 16)  # inside 2 to 16, 16 to 30


GATHER = "_c10d_functional.all_gather_into_tensor.default"


def _gathered(rows):
    """An all-gather of `rows` rows from each of eight ranks, 1024 bytes a row."""
    x = {"shape": [rows, 64], "stride": [64, 1], "dtype": BF16, "role": "activation"}
    return {"op": GATHER, "args": {"x": x}, "out": [[8 * rows, 64]]}


@pytest.mark.parametrize(
    "rows, at, priced",
    [
        (2, ("decode", None, 2, None, 1024, 1024), 2e-6),
        (3, ("prefill", 3, 1, None, 0, 0), 3e-6),
        (
            8,
            ("decode", None, 8, None, 1024, 1024),
            f"{GATHER}: bytes = 8192 is outside the measured 1024 to 4096",
        ),
    ],
)
def test_a_price_list_prices_a_collective_between_two_measured_sizes(rows, at, priced):
    # measured at decode widths 1 and 4: 1024 and 4096 bytes, one kernel
    points = [
        {"shape": f"decode B={b}", "context": 1024, "op": GATHER, "cold_ns": ns}
        | {"leaf": bench.signature(_gathered(b)), "kernels": ["ag"]}
        | {"exchange": bench.exchange(_gathered(b))}
        for b, ns in ((1, 1e3), (4, 4e3))
    ]
    params = ops.fit(points)
    ops.held_out(params)

    got = ops.PriceList(params).price((_gathered(rows), at, 0))

    if isinstance(priced, str):
        assert got.reason == priced
    else:
        assert got.seconds == pytest.approx(priced)


V4_DECODE = "aiter.mla:mla_decode_fwd_v4_nm"


def _layer(base):
    """DeepSeek-V4's decode attention over 4 rows in a layer whose view of the
    one KV pool starts at row `base`, as its `op_graph` holds it."""
    q = {"shape": [4, 8, 64], "stride": [512, 64, 1], "dtype": BF16}
    kv = {"shape": [4096 - base, 64], "stride": [64, 1], "dtype": BF16}
    args = {"q": q | {"role": "activation"}, "kv_buffer": kv | {"role": "kv"}}
    return {"op": V4_DECODE, "args": args, "out": []}


def test_each_layer_s_attention_prices_from_the_point_one_layer_s_view_measured():
    # measured on the first layer's view at two contexts, its time moving
    points = [
        {"shape": "decode B=4", "context": c, "op": V4_DECODE, "cold_ns": ns}
        | {"leaf": bench.signature(_layer(0)), "kernels": ["mla"]}
        for c, ns in ((1024, 10e3), (4096, 40e3))
    ]
    params = ops.fit(points)
    ops.held_out(params)
    at = ("decode", None, 4, None, 4096, 4096)

    got = [ops.PriceList(params).price((_layer(b), at, 0)) for b in (0, 64, 128)]

    assert [getattr(g, "seconds", g) for g in got] == [pytest.approx(40e-6)] * 3


V4_WRITE = "atom.model_ops.v4_kernels.state_writes:swa_write"


def _write(layer):
    """DeepSeek-V4's window write of 4 rows in layer `layer`, which ATOM passes
    that layer's name, as its `op_graph` holds it."""
    kv = {"shape": [4, 512], "stride": [512, 1], "dtype": BF16, "role": "activation"}
    args = {"kv": kv, "prefix": f"layers.{layer}.attn.swa_write"}
    return {"op": V4_WRITE, "args": args, "out": []}


def test_each_layer_s_write_prices_from_the_point_one_layer_s_call_measured():
    # measured on the first layer's call at two contexts
    points = [
        {"shape": "decode B=4", "context": c, "op": V4_WRITE, "cold_ns": 5e3}
        | {"leaf": bench.signature(_write(0)), "kernels": ["swa"]}
        for c in (1024, 4096)
    ]
    params = ops.fit(points)
    ops.held_out(params)
    at = ("decode", None, 4, None, 4096, 4096)

    got = [ops.PriceList(params).price((_write(n), at, 0)) for n in (0, 1, 60)]

    assert [getattr(g, "seconds", g) for g in got] == [pytest.approx(5e-6)] * 3


@pytest.mark.parametrize(
    "mean, priced",
    [
        (3000, 57.5e-6),
        (
            6400,
            (
                "aiter.gemm: at B = 4, at mean = 1000, context = 8000 is outside "
                "the measured 1024 to 1024"
            ),
        ),
    ],
)
def test_a_family_with_ragged_points_leaves_out_a_uniform_point_the_pool_cut(
    mean, priced
):
    # At 8000 tokens the pool cut the uniform point's other three rows to 4000:
    # its rows hold a mean of 625 permille, not the 1000 its context names.
    cut = _point(4, 82.5e3, context=8000)
    cut["rows"] = [[1, 8000, True]] + [[1, 4000, True]] * 3
    ragged = [_point(4, 20e3 + 100 * m, context=[8000, m]) for m in (250, 500)]
    params = ops.fit([_point(4, 10e3), *ragged, cut])
    ops.held_out(params)
    (f,) = params["families"].values()
    leaf = json.loads(_leaf(4, 128))

    got = ops.PriceList(params).price((leaf, ("decode", None, 4, None, 8000, mean), 0))

    assert sorted(c for c, *_ in f["points"]) == [
        [4, 250, 8000],
        [4, 500, 8000],
        [4, 1000, 1024],
    ]
    if isinstance(priced, str):
        assert got.reason == priced
    else:
        assert got.seconds == pytest.approx(priced)


def _later(rows):
    """A drafter's later pass over `rows` rows."""
    return leaf("aiter.step", x=t([rows, 8]))


def _mixed(rows, width):
    """A product of `rows` rows broadcast across `width`."""
    out = [[width, rows, 64]]
    return leaf("aten.mul.Tensor", out, self=t([rows, 64]), other=t([width, 1, 64]))


def _at(op, shape, ns, context=None, kernel="k"):
    """`measure ops`'s point of the leaf `op` at `shape`."""
    point = {"shape": shape, "context": context, "op": op["op"], "cold_ns": ns}
    return point | {"leaf": bench.signature(op), "kernels": [kernel]}


def _priced(points, step, graph, drafts=0):
    """`step`'s terms, the region `graph(draft, width)` gives each part priced
    by the fit of `points`."""
    params = ops.fit(points)
    ops.held_out(params)
    backend = OpGraphBackend(
        SPEC,
        lambda *_, draft=False, width=None: graph(draft, width),
        kv="bf16",
        drafts=drafts,
        price_list=ops.PriceList(params),
    )
    cost = backend.price(step)
    return {term.name: term.seconds for term in (*cost.terms, *cost.draft_terms)}


@pytest.mark.parametrize(
    "queries, width, kernel, priced",
    [
        ((60,), None, "k", 1e-6),
        ((40, 40), None, "k", 2e-6),
        ((60,), 4, "k", 4e-6),
        ((60,), 4, "k4", 4e-6),
    ],
    ids=["one-row", "two-rows", "padded-to-4", "padded-to-4-on-its-own-kernel"],
)
def test_a_drafter_s_later_pass_at_a_prefill_s_width_prices_from_its_width_s_point(
    queries, width, kernel, priced
):
    # 1 us a row and flat over a decode's context, measured at a prompt's rows,
    # 1 and 2, and at 4, the width a prompt of one row padded its later pass to,
    # there on the rows' kernel or on one of its own
    points = [_at(_later(2), "decode B=2", 2e3, c) for c in (1000, 3000)]
    points += [_at(_later(1), "prefill T=60 B=1", 1e3)]
    points += [_at(_later(2), "prefill T=80 B=2", 2e3)]
    points += [_at(_later(4), "prefill T=60 B=1 W=4", 4e3, kernel=kernel)]
    prompts = tuple(RequestShape(q, q, False) for q in queries)
    step = BatchView(prompts, produces_output=True, draft_bs=width)

    got = _priced(
        points,
        step,
        lambda draft, _: _later(width or len(queries)) if draft else {"seq": []},
        drafts=2,
    )

    assert got == {"draft op.aiter.step.price_list": pytest.approx(priced)}


@pytest.mark.parametrize("width, priced", [(4, 1e-6), (8, 2e-6)])
def test_a_leaf_over_a_prefill_s_rows_and_its_width_is_priced_at_each_width_apart(
    width, priced
):
    # A prompt of one row whose LM head pads to 4 or 8: a leaf over both its
    # row and that width is read at its rows, and each width is its own point
    points = [_at(_mixed(1, w), f"prefill T=60 B=1 W={w}", w * 250) for w in (4, 8)]
    prompt = (RequestShape(60, 60, False),)
    step = BatchView(prompt, produces_output=True, target_bs=width)

    got = _priced(points, step, lambda _, at: _mixed(1, at))

    assert got == {"op.aten.mul.Tensor.price_list": pytest.approx(priced)}


#: A drafter's later pass, 1 us a row and flat over a decode's context, at a
#: prompt's rows, 1 and 2, and at 4, the width a prompt of 60 tokens and one of
#: 4, its T, padded its later pass to: there it ran 1 row or 4.
LATER = [_at(_later(2), "decode B=2", 2e3, c) for c in (1000, 3000)]
LATER += [
    _at(_later(b), f"prefill T={t} B={b}", b * 1e3) for t, b in ((60, 1), (80, 2))
]
LATER += [_at(_later(4), f"prefill T={t} B=1 W=4", 4e3) for t in (60, 4)]
EITHER = "it leads with its T, a count of its width too, so it ran at "


def test_fit_ops_leaves_out_by_name_a_point_that_ran_its_leaf_at_either_b():
    params = ops.fit(LATER)

    prefill = [f for name, f in params["families"].items() if "prefill" in name]
    assert [(f["dims"], sorted(c for c, *_ in f["points"])) for f in prefill] == [
        (["B"], [[1], [2], [4]])
    ]
    assert params["left_out"] == [
        {
            "shape": "prefill T=4 B=1 W=4",
            "op": "aiter.step",
            "why": EITHER + "T=4 B=1 or T=4 B=4",
        }
    ]


@pytest.mark.parametrize(
    "queries, width, priced",
    [((60,), None, 1e-6), ((40, 40), None, 2e-6), ((60,), 4, 4e-6), ((4,), 4, None)],
    ids=["one-row", "two-rows", "padded-to-4", "t-is-its-w"],
)
def test_a_prefill_whose_t_is_its_w_declines_its_later_pass_and_blends_no_width(
    queries, width, priced
):
    prompts = tuple(RequestShape(q, q, False) for q in queries)
    step = BatchView(prompts, produces_output=True, draft_bs=width)
    rows = width or len(queries)

    def graph(draft, _):
        return _later(rows) if draft else {"seq": []}

    if priced is None:
        with pytest.raises(CostRefused) as refused:
            _priced(LATER, step, graph, drafts=2)
        why = "aiter.step: " + EITHER + "T=4 B=1 or T=4 B=4"
        assert Refusal(LISTED, why) in refused.value.declined
    else:
        got = _priced(LATER, step, graph, drafts=2)
        assert got == {"draft op.aiter.step.price_list": pytest.approx(priced)}


def _tokens(n):
    """A leaf over a prompt's `n` tokens."""
    return leaf("aiter.embed", x=t([n, 8]))


@pytest.mark.parametrize("tokens, width", [(60, None), (4, 4)], ids=["60", "4-of-4"])
def test_a_leaf_over_a_prompt_s_tokens_keeps_its_t_is_w_point_and_prices_there(
    tokens, width
):
    # 10 ns a token, flat over a decode's context, at 60 and 80 tokens and at 4
    # padded to 4: its family reads its T alone, so either B prices it alike
    points = [_at(_tokens(2), "decode B=2", 20.0, c) for c in (1000, 3000)]
    points += [_at(_tokens(n), f"prefill T={n} B=1", n * 10.0) for n in (60, 80)]
    points += [_at(_tokens(4), "prefill T=4 B=1 W=4", 40.0)]
    prompt = (RequestShape(tokens, tokens, False),)
    step = BatchView(prompt, produces_output=True, target_bs=width)

    got = _priced(points, step, lambda *_: _tokens(tokens))

    assert got == {"op.aiter.embed.price_list": pytest.approx(tokens * 1e-8)}
