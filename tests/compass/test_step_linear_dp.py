# SPDX-License-Identifier: MIT
"""The step linear law under DP-attention with EP off: one law per rank, priced at
the deployment's DP width, the group's step the max over its ranks. A rung's hull
holds each fitted step at the rank its vector prices it as, MoE column included,
and at its rank of larger features where its ranks price alike; a rung that
adopts a column is solved on those rows.
The checked-in DeepSeek-V4-Pro EP8 laws install at their own side's engine config
only, and validate a step at every decode rung they fit.

Each rank's record is written from a known per-rank law, every line labelled
with the group's step, the max over the ranks' prices, as a real DP group's
ranks each measure it. The fitted law is read by `StepLinearLaw` and priced by
`StepLinearBackend.price`.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_step_linear_moe_column as routed
from test_backend_step_linear import _installed, _without_step_stub, law
from test_backend_step_linear_fit import SPEC, _engine_file, _fit, _record
from test_memory_readings import META_8B, publish_readings
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.shape import sum_context
from atom.compass.backends.step_linear import UNIFORM_ROUTING
from atom.compass.backends.step_linear_fit import priced_rank

DP = 4
PREFILL = {
    "step": 0.02,
    "tokens": 6e-5,
    "query_square": 2e-10,
    "query_cached": 3e-10,
    "moe_rows": 2e-5,
}
DECODE = {"step": 0.021, "context": 1.2e-7, "graph_padding": 3e-8}


def _price(v: BatchView) -> float:
    """One rank's step under the known law, from its rows by hand."""
    if v.prefill:
        q = np.array([r.query_tokens for r in v.prefill])
        n = np.array([r.context_tokens for r in v.prefill])
        x = (1, q.sum(), (q * q).sum(), (q * (n - q)).sum(), v.moe_rows)
        return float(np.dot(list(PREFILL.values()), x))
    n = sum(r.context_tokens for r in v.decode)
    return float(np.dot(list(DECODE.values()), (1, n, v.graph_padding)))


def _groups(seed, n=300):
    """Steps of `DP` ranks, all prefilling or all decoding at rung 4, rank
    `i % DP` the slowest by construction; each with the group's step."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow, ranks = i % DP, []
        if i % 2:
            for r in range(DP):
                lo, hi = (40000, 60000) if r == slow else (1, 20000)
                width = 4 if r == slow else int(rng.integers(1, 5))
                rows = [
                    RequestShape(1, int(rng.integers(lo, hi)), True)
                    for _ in range(width)
                ]
                ranks.append((rows, 4, DP * 4))
        else:
            per_rank = []
            for r in range(DP):
                lo, hi = (4096, 8192) if r == slow else (16, 2048)
                rows = []
                for _ in range(3 if r == slow else int(rng.integers(1, 4))):
                    q = int(rng.integers(lo, hi))
                    rows.append(RequestShape(q, q + int(rng.integers(0, 30000)), False))
                per_rank.append(rows)
            moe = sum(r.query_tokens for rows in per_rank for r in rows)
            ranks = [(rows, None, moe) for rows in per_rank]
        views = [BatchView(tuple(rows), rung, moe) for rows, rung, moe in ranks]
        prices = [_price(v) for v in views]
        assert int(np.argmax(prices)) == slow
        out.append((views, max(prices)))
    return out


def _records(tmp_path, groups):
    """Rank `r`'s record, its labels stretched by `1 + 0.01 r`."""
    paths = []
    for r in range(len(groups[0][0])):
        steps = [
            (
                [
                    (row.query_tokens, row.context_tokens, row.decode)
                    for row in v[r].requests
                ],
                v[r].capture_rung,
                seconds,
            )
            for v, seconds in groups
        ]
        stretch = dict.fromkeys(range(len(steps)), 1 + 0.01 * r)
        path = _record(tmp_path / f"dp{r}.jsonl", steps, stretch)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for line, (v, _) in zip(lines, groups):
            line["dp_rank"], line["moe_rows"] = r, v[r].moe_rows
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(path)
    return paths


def test_the_alternating_fit_recovers_the_per_rank_law_and_every_row_does_not(
    tmp_path,
):
    groups = _groups(0)
    # One step whose ranks fall to different vectors: rank 0 prefills.
    mixed = groups[1][0].copy()
    mixed[0] = groups[0][0][0]
    law_ = _fit(_records(tmp_path, groups + [(mixed, 1.0)]), dp=DP)

    assert "key" not in law_
    assert law_["cut"]["ranks"] == 1
    assert law_["spread"] == pytest.approx([0.03, 0.03], abs=1e-6)
    fitted = [law_["prefill"], law_["decode"]["4"]]
    assert fitted[0]["coefficients"] == pytest.approx(PREFILL, rel=1e-6)
    assert fitted[1]["coefficients"] == pytest.approx(DECODE, rel=1e-6)
    for vector in fitted:
        assert vector["fit"]["mad"] < 1e-6
        assert vector["fit"]["all_rows"]["mad"] > 0.01

    backend = StepLinearBackend(StepLinearLaw(law_), DP)
    for views, seconds in groups:
        priced = max(backend.price(v).seconds for v in views)
        assert priced == pytest.approx(seconds, rel=1e-6)
    without = BatchView(groups[0][0][0].requests)
    with pytest.raises(CostRefused, match="no MoE gather rows"):
        backend.price(without)


def test_a_law_stating_key_refuses_by_name():
    with pytest.raises(
        ValueError, match=r"the law states key \[8, 1\]; its engine_config"
    ):
        StepLinearLaw(dict(law(), key=[8, 1]))


@pytest.mark.parametrize("cap", [20, 4, 3])
def test_the_fit_records_its_passes_and_a_cap_reached_refuses_it(
    tmp_path, monkeypatch, cap
):
    from atom.compass.backends import step_linear_fit

    monkeypatch.setattr(step_linear_fit, "FIT_PASSES", cap)
    records = _records(tmp_path, _groups(0))
    if cap < 4:
        with pytest.raises(ValueError, match="still moved after 3 passes"):
            _fit(records, dp=DP)
        return
    law_ = _fit(records, dp=DP)
    assert law_["prefill"]["fit"]["passes"]["count"] == 1
    passes = law_["decode"]["4"]["fit"]["passes"]
    assert (passes["count"], passes["stop"]) == (4, "settled")
    assert passes["F"] == pytest.approx(0.0, abs=1e-20)


#: A rank's wide rows and another's one tall row cost about the same.
WIDE_OR_TALL = {
    "step": 0.015,
    "context": 2e-8,
    "graph_padding": 1e-8,
    "moe_experts": 1.5e-4,
}


def _wide_or_tall(seed, widths, wide):
    """200 decode steps at rung 4, each the group's step under `WIDE_OR_TALL`:
    rank 0 runs 4 rows of one context drawn from `wide`, rank 1 one row 1.4 to
    1.9 times as long, which the rung pads, and the others `widths` short rows,
    so that the group's MoE tokens vary from step to step."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(200):
        n = int(rng.uniform(*wide))
        contexts = [[n] * routed.RUNG, [int(n * rng.uniform(1.4, 1.9))]]
        for _ in range(DP - 2):
            contexts.append(rng.integers(1, 1000, rng.integers(*widths)).tolist())
        views = [routed._view(c, sum(map(len, contexts))) for c in contexts]
        out.append((views, max(routed._price(v, WIDE_OR_TALL) for v in views)))
    return out


def test_a_routed_rung_hulls_each_step_at_the_rank_it_prices(tmp_path):
    """The alternation takes each step's row on the rung's features alone; the
    experts column the rung then adopts moves some steps' costliest rank from
    the wide one to the tall one. Each step prices inside the fitted hull at
    the rank it is priced as."""
    full = _wide_or_tall(0, routed.FULL, (110_000, 130_000))
    sparse = _wide_or_tall(1, routed.SPARSE, (80_000, 100_000))
    records = routed._records(tmp_path / "full", full)
    records += routed._records(tmp_path / "sparse", sparse)
    law_ = routed._routed(records)[0]
    vector = law_["decode"][str(routed.RUNG)]
    assert vector["fit"]["moe_column"]["adopted"] == "moe_experts"
    assert vector["fit"]["dropped"] == 0

    laws = [_fit(records, dp=DP), law_]
    backends = [StepLinearBackend(StepLinearLaw(x), DP) for x in laws]
    moved, outside = 0, []
    for i, (views, _) in enumerate(full + sparse):
        costs = [[b.price(v) for v in views] for b in backends]
        top = [int(np.argmax([c.seconds for c in cost])) for cost in costs]
        moved += top[0] != top[1]
        if any(t.provenance.extrapolated for t in costs[1][top[1]].terms):
            outside.append(i)
    assert moved
    assert outside == []


@pytest.mark.parametrize("cap", [20, 9, 8])
def test_a_routed_rung_is_fitted_again_under_its_column(tmp_path, monkeypatch, cap):
    """The experts column moves some steps' costliest rank off the row the
    rung's features alone pick, so the rung is fitted again under it, from
    every rank's row. Once no step moves, it is solved on the rows it
    prices: the per-rank law. A cap reached refuses the fit."""
    from atom.compass.backends import step_linear_fit

    monkeypatch.setattr(step_linear_fit, "FIT_PASSES", cap)
    full = _wide_or_tall(2, routed.FULL, (110_000, 130_000))
    sparse = _wide_or_tall(3, routed.SPARSE, (80_000, 100_000))
    records = routed._records(tmp_path / "full", full)
    records += routed._records(tmp_path / "sparse", sparse)
    if cap < 9:
        with pytest.raises(ValueError, match="still moved after 8 passes"):
            routed._routed(records)
        return
    vector = routed._routed(records)[0]["decode"][str(routed.RUNG)]
    passes = vector["fit"]["passes"]
    assert (passes["count"], passes["stop"]) == (9, "settled")
    assert vector["coefficients"] == pytest.approx(WIDE_OR_TALL, rel=1e-6)


#: Two ranks' (context, graph_padding) in thousands of tokens and the group's
#: step in centiseconds, six steps on which the alternation by full refits
#: cycles between two states that differ at the last step's rank.
CYCLE = (
    ((3, 4), (1, 2), 3.68),
    ((3, 5), (1, 5), 3.9),
    ((6, 5), (1, 5), 4.79),
    ((1, 5), (1, 5), 3.36),
    ((2, 1), (8, 2), 4.81),
    ((7, 1), (5, 4), 4.22),
)


def test_an_alternation_that_cycles_on_full_refits_ends_at_a_tie_below_both(
    tmp_path,
):
    """Solved on the rows one state picks, each state's vector picks the
    other's. The fit's line search ends between them where the last step's
    two ranks price alike, `F` below either state."""
    from scipy.optimize import nnls

    groups = [
        ([_decode_step(8, 800 * c, 800 * p) for c, p in ranks], y / 100)
        for *ranks, y in CYCLE
    ]
    x = np.array([[(1, 800 * c, 800 * p) for c, p in ranks] for *ranks, _ in CYCLE])
    y = np.array([g[1] for g in groups])

    def state(pick):
        coef = nnls(x[np.arange(len(y)), pick] / y[:, None], np.ones(len(y)))[0]
        return coef, tuple(np.argmax(x @ coef, axis=1))

    def cost(coef):
        return float((((x @ coef).max(axis=1) / y - 1) ** 2).sum())

    a, b = (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 1, 1)
    (coef_a, after_a), (coef_b, after_b) = state(a), state(b)
    assert (after_a, after_b) == (b, a)

    law_ = _fit(_records(tmp_path, groups), dp=2)
    passes = law_["decode"]["8"]["fit"]["passes"]
    assert passes["stop"] == "tie"
    assert passes["F"] < min(cost(coef_a), cost(coef_b))
    backend = StepLinearBackend(StepLinearLaw(law_), 2)
    last = [backend.price(v).seconds for v in groups[-1][0]]
    assert last[0] == pytest.approx(last[1], rel=1e-6)
    for views, _ in groups:
        costs = [backend.price(v) for v in views]
        top = max(costs, key=lambda c: c.seconds)
        assert not any(t.provenance.extrapolated for t in top.terms)


def _alike(seed, n=300):
    """`n` decode steps at rung 4, each rank 1 to 4 rows of a random context,
    each step 20 ms whatever its rows."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        views = []
        for _ in range(DP):
            width = int(rng.integers(1, 5))
            rows = [
                RequestShape(1, int(rng.integers(1, 50_000)), True)
                for _ in range(width)
            ]
            views.append(BatchView(tuple(rows), 4, DP * 4))
        out.append((views, 0.02))
    return out


def test_ranks_priced_alike_hull_each_step_at_its_rank_of_larger_features(tmp_path):
    """Every step takes 20 ms, so the rung's vector prices a step's ranks alike
    up to rounding, and the step is priced as its costliest rank of larger
    features (`priced_rank`), which some steps' first costliest rank is not.
    Each step prices inside the fitted hull at that rank."""
    runs = [_alike(seed) for seed in (0, 1)]
    records = []
    for i, steps in enumerate(runs):
        (tmp_path / str(i)).mkdir()
        records += _records(tmp_path / str(i), steps)
    backend = StepLinearBackend(StepLinearLaw(_fit(records, dp=DP)), DP)
    tied, outside = 0, []
    for i, (views, _) in enumerate(runs[0] + runs[1]):
        costs = [backend.price(v) for v in views]
        ranks = {
            j: (c.target_seconds, (1, sum_context(v.decode), v.graph_padding))
            for j, (c, v) in enumerate(zip(costs, views))
        }
        top = priced_rank(ranks)
        tied += top != max(ranks, key=lambda j: ranks[j][0])
        if any(t.provenance.extrapolated for t in costs[top].terms):
            outside.append(i)
    assert tied
    assert outside == []


def test_a_law_fitted_on_dp_records_joins_at_their_width_and_refuses_at_dp_1(
    tmp_path,
    monkeypatch,
):
    from atom.compass.backends.step_linear_fit import main

    records = _records(tmp_path, _groups(0))
    engine = {"data_parallel_size": DP, "enable_dp_attention": True}
    for path in records:
        _engine_file(path, engine)
    main([*map(str, records), "--out", str(tmp_path / "law.json"), *SPEC])
    law_ = json.loads((tmp_path / "law.json").read_text())

    assert law_["engine_config"] == engine
    assert law_["prefill"]["coefficients"] == pytest.approx(PREFILL, rel=1e-6)
    with pytest.raises(ValueError) as refused:
        _installed(monkeypatch, tmp_path, step_linear=law_)
    assert f"data_parallel_size {DP} in the law and 1 here" in str(refused.value)


def test_records_predating_the_width_fit_at_the_width_an_engine_file_states(tmp_path):
    from atom.compass.backends.step_linear_fit import main

    records = _records(tmp_path, _groups(0))
    for path in records:
        _engine_file(path, {"enable_dp_attention": True})
    out = ["--out", str(tmp_path / "law.json"), *SPEC]
    with pytest.raises(ValueError, match="states no data_parallel_size"):
        main([*map(str, records), *out])
    stated = tmp_path / "stated.engine.json"
    engine = {"data_parallel_size": DP, "enable_dp_attention": True}
    stated.write_text(json.dumps(engine | {"max_num_seqs": 64}))
    main([*map(str, records), *out, "--engine", str(stated)])
    law_ = json.loads((tmp_path / "law.json").read_text())

    assert law_["engine_config"] == engine
    assert law_["prefill"]["coefficients"] == pytest.approx(PREFILL, rel=1e-6)


def test_a_stated_engine_file_contradicting_the_records_own_refuses(tmp_path):
    from atom.compass.backends.step_linear_fit import main

    records = _records(tmp_path, _groups(0))
    for path in records:
        _engine_file(path, {"data_parallel_size": DP, "enable_dp_attention": True})
    stated = tmp_path / "stated.engine.json"
    stated.write_text(
        json.dumps({"data_parallel_size": 1, "enable_dp_attention": True})
    )
    out = ["--out", str(tmp_path / "law.json"), *SPEC]
    with pytest.raises(ValueError, match=r"contradicts .* in \['data_parallel_size'\]"):
        main([*map(str, records), *out, "--engine", str(stated)])
    assert not (tmp_path / "law.json").exists()


COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
GPU45 = json.loads((COMPASS / "machines" / "mi355x-gpu45.json").read_text())
CRS005 = json.loads((COMPASS / "machines" / "mi355x-crs005.json").read_text())
#: Each DeepSeek-V4-Pro EP8 law pair's file stem and the spec it was fitted under.
EP8_PAIRS = [
    pytest.param(
        "deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661", GPU45, id="gpu45"
    ),
    pytest.param("deepseek-v4-pro-dpa8-ep8-mi355x-crs005", CRS005, id="crs005"),
]


@pytest.mark.parametrize("side", ["prefill", "decode"])
@pytest.mark.parametrize("stem,spec", EP8_PAIRS)
def test_each_v4_pro_ep8_law_installs_at_its_side_and_refuses_ep_off(
    monkeypatch, tmp_path, stem, spec, side
):
    fitted = json.loads((COMPASS / "laws" / f"{stem}-{side}.json").read_text())
    engine = fitted["engine_config"]
    for key in PERFORMANCE_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in engine["env"].items():
        monkeypatch.setenv(key, value)
    run = _without_step_stub(tmp_path, step_linear=fitted, machine=spec)
    monkeypatch.setenv(compass_run.ENV, str(run))

    def installed(ep):
        r = deployment(
            tmp_path,
            **{k: engine[k] for k in ("kv_cache_block_size", "kv_cache_dtype")},
            quant_config=SimpleNamespace(compute_hash=lambda: engine["quantization"]),
            parallel_config=SimpleNamespace(
                data_parallel_size=8, data_parallel_rank=0, pipeline_parallel_rank=0
            ),
            capture_sizes=engine["ladder"],
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            enable_dp_attention=True,
            enable_expert_parallel=ep,
            enable_tbo=engine["enable_tbo"],
        )
        publish_readings(tmp_path / "store", r.config, spec)
        compass_run.runner(r, META_8B)
        return r.compass_backend

    assert installed(True).treatments == (UNIFORM_ROUTING,)
    with pytest.raises(
        ValueError, match="enable_expert_parallel True in the law and False here"
    ):
        installed(False)


def _inside(hull: dict, features: tuple) -> dict:
    """The centre of the largest ball inside `hull`, a law's HULL over `features`."""
    from scipy.optimize import linprog

    vary = [f for f in features if f not in hull["equal"]]
    a = np.array(hull["facets"], dtype=np.float64).reshape(-1, len(vary) + 1)
    rows = np.column_stack([a[:, :-1], np.linalg.norm(a[:, :-1], axis=1)])
    free = [(None, None)] * len(vary) + [(0, None)]
    x = linprog([0] * len(vary) + [-1], A_ub=rows, b_ub=-a[:, -1], bounds=free).x
    return {**hull["equal"], **dict(zip(vary, x[:-1]))}


def _decode_step(rung: int, context: float, graph_padding: float) -> BatchView:
    """A step replaying `rung` at about that point: its widest row sets
    `rung * widest = Sum N_KV + padding`, and the others share the rest."""
    widest = math.ceil((context + graph_padding) / rung)
    rest = max(0, round(context) - widest)
    n = math.ceil(rest / widest)
    shares = [rest // n + (i < rest % n) for i in range(n)]
    return BatchView(tuple(RequestShape(1, c, True) for c in [widest, *shares]), rung)


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_each_v4_pro_ep8_law_validates_a_step_at_every_decode_rung_it_fits(side):
    name = f"deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661-{side}.json"
    fitted = json.loads((COMPASS / "laws" / name).read_text())
    backend = StepLinearBackend(StepLinearLaw(fitted))
    for rung, vector in fitted["decode"].items():
        assert "held_out" in vector, f"rung {rung} has no held-out hull"
        at = _inside(vector["held_out"]["hull"], ("context", "graph_padding"))
        cost = backend.price(_decode_step(int(rung), **at))
        assert all(t.provenance.validated for t in cost.terms), f"rung {rung}"
