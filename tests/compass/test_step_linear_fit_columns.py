# SPDX-License-Identifier: MIT
"""The step linear law's opt-in columns (#3720): decode `row_padding` and
`peer_tokens`, prefill `gather_rows` and `moe_imbalance`, and the in-sample
rule a decode rung picks its MoE column by.

Each column is counted by one `step_linear` function that `fit` and
`StepLinearBackend` both call, so a law prices a simulated step as it was
fitted. Each is off unless asked for, and a fit with every switch off writes
the law it wrote before them.

Records are written from a known per-rank law, every line labelled with the
group's step, as in `test_step_linear_dp`. No device.
"""

import json

import numpy as np
import pytest
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_backend_step_linear_fit import _engine_file, _record
from test_fit_steps import ENGINE, HOST, LINEAR, as_record, clients, law_key, member
from test_fit_steps import fit as compass_fit
from test_step_linear_moe_column import EXPERTS, FULL, _groups
from test_step_linear_moe_column import ROUTING as MOE
from test_step_linear_moe_column import _records as moe_records

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import (
    gather_rows,
    moe_imbalance,
    peer_tokens,
    row_padding,
    token_bucket,
    token_padding,
)
from atom.compass.backends.step_linear_fit import (
    SATURATED,
    env_options,
    fit,
    label_records,
)
from atom.compass.backends.work import activated_experts
from atom.compass.fit import publish

DP = 4
BUCKETS = (256, 512, 1024, 2048, 4096)


def _decode(contexts, rung, **fields):
    rows = tuple(RequestShape(1, n, True) for n in contexts)
    return BatchView(rows, capture_rung=rung, **fields)


def _prefill(tokens, peak=None, moe=None, **fields):
    rows = (RequestShape(tokens, tokens, False),)
    return BatchView(
        rows, moe_rows=moe, moe_tokens=moe, group_max_tokens=peak, **fields
    )


def _anywhere(features: int) -> dict:
    """A hull every point is inside: one facet, `0 . x - 1 <= 0`."""
    return {"equal": {}, "facets": [[0.0] * features + [-1.0]]}


def _vector(coefficients: dict, **spec) -> dict:
    free = sum(1 for f in coefficients if f != "step")
    return {"coefficients": coefficients, "hull": _anywhere(free), **spec}


def test_row_padding_is_the_rows_own_rectangle_and_the_rung_adds_the_rest():
    view = _decode([100, 40, 10], 8)
    assert row_padding(view.decode) == 3 * 100 - 150
    assert view.graph_padding - row_padding(view.decode) == (8 - 3) * 100
    assert row_padding(()) == 0


def test_peer_tokens_are_the_group_tokens_less_the_rank_s_own():
    rows = [100, 200]
    assert peer_tokens(_decode(rows, 2, moe_rows=8, moe_tokens=7)) == 5
    assert peer_tokens(_decode(rows, 2)) == 0  # one rank: no peers
    assert peer_tokens(_decode(rows, 2, moe_rows=8)) is None


def test_gather_rows_carry_every_rank_s_bucket_under_a_token_graph():
    graphed = _prefill(200, peak=300, moe=350)
    assert token_bucket(graphed, BUCKETS) == 512
    assert token_padding(graphed, BUCKETS) == 512 - 200
    assert gather_rows(graphed, BUCKETS, DP) == DP * 512
    # Past the largest bucket, or refused a graph, the gather is variable-length.
    eager = _prefill(2000, peak=5000, moe=6000)
    assert token_bucket(eager, BUCKETS) == 0 and token_padding(eager, BUCKETS) == 0
    assert gather_rows(eager, BUCKETS, DP) == 6000
    refused = _prefill(200, peak=300, moe=350, group_needs_eager=True)
    assert gather_rows(refused, BUCKETS, DP) == 350
    assert gather_rows(_prefill(200), BUCKETS, DP) == 0  # one rank
    assert gather_rows(_prefill(200, moe=350), BUCKETS, DP) is None


def test_moe_imbalance_is_the_idle_rows_of_an_uneven_variable_length_gather():
    lone = _prefill(5000, peak=5000, moe=5000)  # one rank prefills
    assert moe_imbalance(lone, BUCKETS, DP, 0.2) == (DP - 1) * 5000
    even = _prefill(5000, peak=5200, moe=20000)  # 4 x 5200 - 20000: 4%
    assert moe_imbalance(even, BUCKETS, DP, 0.2) == 0
    assert moe_imbalance(even, BUCKETS, DP, 0.0) == 800
    # Every rank runs the bucket under a token graph: even by construction.
    assert moe_imbalance(_prefill(200, peak=300, moe=350), BUCKETS, DP, 0.2) == 0
    assert moe_imbalance(_prefill(200), BUCKETS, DP, 0.2) == 0
    assert moe_imbalance(_prefill(5000, moe=5000), BUCKETS, DP, 0.2) is None


PREFILL = {
    "step": 0.02,
    "tokens": 6e-5,
    "query_square": 2e-10,
    "query_cached": 3e-10,
    "moe_rows": 2e-5,
    "gather_rows": 4e-6,
    "moe_imbalance": 3e-6,
    "padding": 1e-5,
}
DECODE = {
    "step": 0.021,
    "context": 1.2e-7,
    "graph_padding": 3e-8,
    "row_padding": 5e-8,
    "moe_tokens": 1.5e-4,
    "peer_tokens": 4e-5,
}


def _law(prefill=PREFILL, decode=DECODE, **prefill_spec) -> dict:
    spec = {"buckets": list(BUCKETS), "ranks": DP, "min_spread": 0.2}
    return {
        "provenance": "the test's",
        "prefill": _vector(prefill, **(spec | prefill_spec)),
        "decode": {"4": _vector(decode)},
    }


def test_the_backend_prices_each_column_by_the_function_the_fit_counts_it_with():
    backend = StepLinearBackend(StepLinearLaw(_law()), DP)
    view = _decode([90_000, 20_000], 4, moe_rows=DP * 4, moe_tokens=9)
    terms = {t.name: t.seconds for t in backend.price(view).terms}
    assert terms["decode.row_padding"] == pytest.approx(5e-8 * 70_000)
    assert terms["decode.peer_tokens"] == pytest.approx(4e-5 * (9 - 2))
    assert terms["decode.graph_padding"] == pytest.approx(3e-8 * (4 * 90_000 - 110_000))
    for view in (_prefill(200, peak=300, moe=350), _prefill(5000, peak=5000, moe=5000)):
        terms = {t.name: t.seconds for t in backend.price(view).terms}
        rows = gather_rows(view, BUCKETS, DP)
        assert terms["prefill.gather_rows"] == pytest.approx(4e-6 * rows)
        idle = moe_imbalance(view, BUCKETS, DP, 0.2)
        assert terms["prefill.moe_imbalance"] == pytest.approx(3e-6 * idle)


def test_a_step_stating_no_group_count_a_column_reads_is_refused():
    backend = StepLinearBackend(StepLinearLaw(_law()), DP)
    with pytest.raises(CostRefused, match="no MoE group tokens"):
        backend.price(_decode([10, 20], 4, moe_rows=DP * 4))
    with pytest.raises(CostRefused, match="gather_rows is counted from"):
        backend.price(_prefill(200, moe=350))


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"ranks": None}, "group's width"),
        ({"ranks": 1}, "group's width"),
        ({"min_spread": None}, "min_spread"),
        ({"min_spread": -0.1}, "min_spread"),
    ],
)
def test_a_law_whose_gather_columns_state_no_width_or_spread_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        StepLinearLaw(_law(**spec))


def test_gather_rows_without_padding_is_refused():
    prefill = {k: v for k, v in PREFILL.items() if k != "padding"}
    with pytest.raises(ValueError, match="gather_rows and no padding"):
        StepLinearLaw(_law(prefill))


def test_the_switches_read_from_the_environment():
    assert env_options({}) == {}
    got = env_options(
        {
            "COMPASS_FIT_ROW_PADDING": "1",
            "COMPASS_FIT_GATHER_ROWS": "1",
            "COMPASS_FIT_MOE_IMBALANCE": "1",
            "COMPASS_FIT_MOE_SELECT": "insample",
            "COMPASS_FIT_PEER_TOKENS": "16",
            "COMPASS_FIT_DRAFT_PEER_TOKENS": "1",
        }
    )
    assert got == {
        "row_padding": True,
        "gather_rows": True,
        "moe_imbalance": 0.2,
        "moe_select": "insample",
        "peer_tokens": (16,),
        "draft_peer_tokens": "all",
    }
    spread = {"COMPASS_FIT_MOE_IMBALANCE": "1", "COMPASS_FIT_MOE_MIN_SPREAD": "0.5"}
    assert env_options(spread) == {"moe_imbalance": 0.5}
    assert env_options({"COMPASS_FIT_PEER_TOKENS": "1"}) == {"peer_tokens": "saturated"}
    with pytest.raises(ValueError, match="COMPASS_FIT_MOE_IMBALANCE=1"):
        env_options({"COMPASS_FIT_MOE_MIN_SPREAD": "0.5"})
    with pytest.raises(ValueError, match="not one of"):
        env_options({"COMPASS_FIT_MOE_SELECT": "median"})


RUNG = 8
ROUTING = (64, 4)


def _decode_price(view, law, routing=ROUTING) -> float:
    rows = view.decode
    features = {
        "step": 1,
        "context": sum(r.context_tokens for r in rows),
        "graph_padding": view.graph_padding,
        "row_padding": row_padding(rows),
        "moe_tokens": view.moe_tokens,
        "moe_experts": activated_experts(*routing, view.moe_tokens),
        "peer_tokens": peer_tokens(view),
    }
    return float(sum(c * features[f] for f, c in law.items()))


def _decode_groups(seed, n, law, widths=(1, RUNG + 1), routing=ROUTING):
    """`n` decode steps at rung `RUNG` over `DP` ranks, rank `i % DP` the
    slowest, with 2 to `RUNG` rows, one of them long, the next rank with
    `RUNG` short rows, which set the rung, and each other rank of `widths`
    short rows; each step labelled with its group's price under `law`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow, full = i % DP, (i + 1) % DP
        rows = [
            (
                int(rng.integers(2, RUNG + 1))
                if r == slow
                else RUNG if r == full else int(rng.integers(*widths))
            )
            for r in range(DP)
        ]
        contexts = [
            [
                int(rng.integers(150_000, 400_000) if r == slow and j == 0 else 0)
                or int(rng.integers(1, 120_000 if r == slow else 30_000))
                for j in range(w)
            ]
            for r, w in enumerate(rows)
        ]
        views = [
            _decode(c, RUNG, moe_rows=DP * RUNG, moe_tokens=sum(rows)) for c in contexts
        ]
        prices = [_decode_price(v, law, routing) for v in views]
        assert int(np.argmax(prices)) == slow
        out.append((views, max(prices)))
    return out


def _decode_records(path, groups):
    path.mkdir()
    paths = []
    for r in range(DP):
        steps = [
            ([(1, row.context_tokens, True) for row in v[r].requests], RUNG, seconds)
            for v, seconds in groups
        ]
        record = _record(path / f"dp{r}.jsonl", steps)
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line, (v, _) in zip(lines, groups):
            line.update(dp_rank=r, moe_rows=v[r].moe_rows, moe_tokens=v[r].moe_tokens)
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        paths.append(record)
    return paths


def _fit(paths, **options):
    samples = label_records(paths, DP)[0]
    return fit(samples, {"form": "base", "dp": DP, **options})[0]


def test_row_padding_is_fitted_when_asked_and_off_by_default(tmp_path):
    truth = {k: DECODE[k] for k in ("step", "context", "graph_padding", "row_padding")}
    groups = _decode_groups(0, 200, truth)
    paths = _decode_records(tmp_path / "run", groups)
    law = _fit(paths, row_padding=True)
    vector = law["decode"][str(RUNG)]
    assert vector["coefficients"] == pytest.approx(truth, rel=1e-5)
    assert list(StepLinearLaw(law).decode[RUNG].features) == list(truth)
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    for views, seconds in groups[:20]:
        priced = max(backend.price(v).seconds for v in views)
        assert priced == pytest.approx(seconds, rel=1e-5)
    # Off, the rung keeps the base columns, and the law is the one every switch
    # left unset writes.
    off = _fit(paths)
    assert list(off["decode"][str(RUNG)]["coefficients"]) == [
        "step",
        "context",
        "graph_padding",
    ]
    unset = {"row_padding": False, "moe_imbalance": None, "peer_tokens": None}
    assert json.dumps(_fit(paths, **unset)) == json.dumps(off)


def test_insample_adopts_a_moe_column_where_held_out_records_cannot_choose(tmp_path):
    """One run whose ranks are all busy: one group composition, no record or
    composition to hold out, so the held-out rule adopts none; in-sample
    adopts the column the cost follows."""
    truth = {k: DECODE[k] for k in ("step", "context", "graph_padding", "moe_tokens")}
    paths = _decode_records(tmp_path / "run", _decode_groups(1, 200, truth, (4, 9)))
    held = _fit(paths, moe_routing=ROUTING)["decode"][str(RUNG)]
    assert held["fit"]["moe_column"]["adopted"] is None
    law = _fit(paths, moe_routing=ROUTING, moe_select="insample")
    vector = law["decode"][str(RUNG)]
    figures = vector["fit"]["moe_column"]
    assert figures["adopted"] == "moe_tokens" and figures["selected_by"] == "insample"
    assert set(figures["in_sample"]) == {"moe_tokens", "moe_experts"}
    assert vector["coefficients"] == pytest.approx(truth, rel=1e-5)


def test_peer_tokens_join_the_rungs_named_after_their_moe_column(tmp_path):
    """The step pays for the experts its group's tokens reach and, apart, for
    the tokens its peers bring; its own rows vary, so the two separate."""
    truth = {
        "step": 0.021,
        "context": 1.2e-7,
        "graph_padding": 3e-8,
        "moe_experts": 6e-5,
        "peer_tokens": 4e-5,
    }
    paths = _decode_records(tmp_path / "run", _decode_groups(2, 240, truth))
    options = {"moe_routing": ROUTING, "moe_select": "insample"}
    law = _fit(paths, peer_tokens=(RUNG,), **options)
    vector = law["decode"][str(RUNG)]
    assert vector["coefficients"] == pytest.approx(truth, rel=1e-5)
    assert list(StepLinearLaw(law).decode[RUNG].features) == list(truth)
    # Named for another rung, or under the saturation rule at a rung whose
    # experts column is not adopted, the rung takes none.
    for peers in ((RUNG * 2,), "saturated"):
        other = _fit(paths, peer_tokens=peers, **options)["decode"][str(RUNG)]
        assert "peer_tokens" not in other["coefficients"]


def test_the_saturation_rule_adds_peer_tokens_where_the_experts_column_is_flat(
    tmp_path,
):
    """64 experts, top 8: the 30 or so tokens of a group whose ranks are all
    busy reach 98% of them, past `SATURATED`, and the rung takes the column;
    the 12 or so of one whose ranks are mostly idle reach 80%, and it does not."""
    routing = (64, 8)
    truth = {
        "step": 0.021,
        "context": 1.2e-7,
        "graph_padding": 3e-8,
        "moe_experts": 6e-5,
    }
    options = {"moe_routing": routing, "moe_select": "insample"}
    reached = {}
    for name, widths, seed in (("many", (6, 9), 3), ("few", (1, 2), 4)):
        groups = _decode_groups(seed, 240, truth, widths, routing)
        tokens = np.median([g[0][0].moe_tokens for g in groups])
        reached[name] = activated_experts(*routing, tokens) / routing[0]
        paths = _decode_records(tmp_path / name, groups)
        law = _fit(paths, peer_tokens="saturated", **options)
        vector = law["decode"][str(RUNG)]
        assert vector["fit"]["moe_column"]["adopted"] == "moe_experts"
        assert ("peer_tokens" in vector["coefficients"]) == (name == "many")
    assert reached["many"] >= SATURATED > reached["few"]


def _prefill_price(rows, moe, peak, law) -> float:
    q = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows])
    view = _prefill(int(q.sum()), peak=peak, moe=moe)
    features = {
        "step": 1,
        "tokens": q.sum(),
        "query_square": (q * q).sum(),
        "query_cached": (q * (n - q)).sum(),
        "moe_rows": moe,
        "gather_rows": gather_rows(view, BUCKETS, DP),
        "moe_imbalance": moe_imbalance(view, BUCKETS, DP, 0.2),
        "padding": token_padding(view, BUCKETS),
    }
    return float(sum(c * features[f] for f, c in law.items()))


def _prefill_groups(seed, n, law):
    """`n` prefill steps over `DP` ranks, rank `i % DP` the busiest, a third
    of them under a token graph, the rest past its largest bucket, some even
    and some uneven."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow = (
            int(rng.integers(100, 4000))
            if i % 3 == 0
            else int(rng.integers(4200, 9000))
        )
        ranks = []
        for r in range(DP):
            share = 1.0 if r == i % DP else rng.choice([0.01, 0.3, 0.85, 0.97])
            q = max(int(slow * share), 1)
            ranks.append(
                [(q, q + int(rng.integers(0, 2)) * int(rng.integers(0, 9000)), False)]
            )
        moe = sum(rows[0][0] for rows in ranks)
        peak = max(rows[0][0] for rows in ranks)
        out.append(
            (ranks, moe, peak, max(_prefill_price(r, moe, peak, law) for r in ranks))
        )
    return out


def _prefill_records(path, groups):
    path.mkdir()
    paths = []
    for r in range(DP):
        steps = [(ranks[r], None, seconds) for ranks, _, _, seconds in groups]
        record = _record(path / f"dp{r}.jsonl", steps)
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line, (_, moe, _, _) in zip(lines, groups):
            line |= {"dp_rank": r, "moe_rows": moe, "moe_tokens": moe}
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        _engine_file(
            record, {"data_parallel_size": DP, "prefill_buckets": list(BUCKETS)}
        )
        paths.append(record)
    return paths


def test_the_gather_columns_are_fitted_and_the_vector_states_width_and_spread(tmp_path):
    groups = _prefill_groups(5, 360, PREFILL)
    paths = _prefill_records(tmp_path / "run", groups)
    options = {"buckets": BUCKETS, "gather_rows": True, "moe_imbalance": 0.2}
    law = _fit(paths, **options)
    vector = law["prefill"]
    assert vector["coefficients"] == pytest.approx(PREFILL, rel=1e-5)
    assert vector["ranks"] == DP and vector["min_spread"] == 0.2
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    for ranks, moe, peak, seconds in groups[:30]:
        views = [
            BatchView(
                (RequestShape(*rows[0]),),
                moe_rows=moe,
                moe_tokens=moe,
                group_max_tokens=peak,
            )
            for rows in ranks
        ]
        priced = max(backend.price(v).seconds for v in views)
        assert priced == pytest.approx(seconds, rel=1e-5)
    off = _fit(paths, buckets=BUCKETS)["prefill"]
    assert "gather_rows" not in off["coefficients"] and "ranks" not in off


def test_a_gather_column_on_one_rank_or_without_buckets_is_refused(tmp_path):
    groups = _prefill_groups(6, 60, PREFILL)
    paths = _prefill_records(tmp_path / "run", groups)
    with pytest.raises(ValueError, match="state none"):
        _fit(paths, gather_rows=True)
    one = label_records(paths[:1], 1)[0]
    with pytest.raises(ValueError, match="run one rank"):
        fit(one, {"dp": 1, "moe_imbalance": 0.2})
    with pytest.raises(ValueError, match="run one rank"):
        fit(one, {"dp": 1, "peer_tokens": (4,)})


def test_moe_records_still_load_with_peer_rows(tmp_path):
    """The MoE-column test's records label and fit as before; each decode row
    carries its peers' tokens."""
    paths = moe_records(tmp_path / "full", _groups(0, 60, FULL, EXPERTS))
    samples = label_records(paths, DP)[0]
    assert all(peer_tokens(s.view) >= 0 for s in samples)


def test_compass_fit_steps_reads_the_switches_from_the_environment(
    tmp_path, monkeypatch
):
    """The command fits the columns and rule the environment asks for, under
    the key the law has without them, so each such law lives in a store of its
    own; unset, it publishes the law it did."""
    monkeypatch.setattr(publish, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT))
    key = {**HOST, "engine": {**ENGINE, "data_parallel_size": DP}}
    (tmp_path / "full").mkdir()
    moe_records(tmp_path / "full" / "steps", _groups(0, 200, FULL, EXPERTS))
    run = as_record(tmp_path / "full", key)
    routing = ("--moe-routing", ",".join(map(str, MOE)))
    laws = {}
    for name, env in (
        ("unset", {}),
        ("set", {"COMPASS_FIT_ROW_PADDING": "1", "COMPASS_FIT_MOE_SELECT": "insample"}),
    ):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        (tmp_path / name).mkdir()
        compass_fit(tmp_path / name, "fit steps", *LINEAR, *routing, run, *clients(run))
        store = tmp_path / name / "store"
        laws[name] = member(store, law_key(engine=key["engine"]), "params")
    rung = "4"
    unset = laws["unset"]["decode"][rung]
    assert "row_padding" not in unset["coefficients"]
    assert unset["fit"]["moe_column"]["adopted"] is None  # one record, one composition
    got = laws["set"]["decode"][rung]
    assert list(got["coefficients"])[:4] == [
        "step",
        "context",
        "graph_padding",
        "row_padding",
    ]
    assert got["fit"]["moe_column"]["selected_by"] == "insample"
    StepLinearLaw(laws["set"])


def test_a_draft_law_takes_peer_tokens_from_its_own_switch_alone(tmp_path):
    """A draft's MoE runs on its group's rows too: `draft_peer_tokens` adds the
    column to the draft law's rungs it names, `peer_tokens` to the target's
    only, and the draft law takes none of the other columns."""
    truth = {
        "step": 0.021,
        "context": 1.2e-7,
        "graph_padding": 3e-8,
        "peer_tokens": 4e-5,
    }
    paths = _decode_records(tmp_path / "run", _decode_groups(7, 200, truth))
    samples = label_records(paths, DP)[0]
    options = {"form": "base", "dp": DP, "row_padding": True}
    draft = fit(samples, {**options, "draft_peer_tokens": (RUNG,)}, draft=True)[0]
    assert draft["decode"][str(RUNG)]["coefficients"] == pytest.approx(truth, rel=1e-5)
    target = fit(samples, {**options, "peer_tokens": (RUNG,)}, draft=True)[0]
    assert list(target["decode"][str(RUNG)]["coefficients"]) == list(truth)[:3]
    every = fit(samples, {**options, "draft_peer_tokens": "all"}, draft=True)[0]
    assert "peer_tokens" in every["decode"][str(RUNG)]["coefficients"]
