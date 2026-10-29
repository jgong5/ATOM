# SPDX-License-Identifier: MIT
"""The step grid under a drafter: a verify step of K + 1 tokens a row, read in
a grid whose every point holds a target value and a drafter value, priced as a
target term and a draft term, each by its own residual.

The lookup runs on `test_backend_step_grid`'s replay grid, a drafter value
`g(B, c)` beside each point's. The planner reads steps ATOM's own `Scheduler`
builds under a DSpark drafter of five drafts and an MTP drafter of one, as
`test_bench_replay_drafter` drives it; its limits are checked against the
replay rule, which allocates as ATOM's scheduler does. The fit runs through
`compass fit steps` on a DeepSeek-V4.1-Flash DSpark5 record fixture: the
published V4.1-Flash config.json at TP4 with V4's 256-token KV blocks, each
step planted at its lookup times a factor, as `test_backend_step_grid_fit`
plants its record, the drafter's part at a lookup and a factor of its own.
"""

import json
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import test_backend_step_grid_fit as grid_fit
from test_artifact_store import stanza
from test_backend_step_grid import (
    REPLAY,
    UNIT,
    bundle,
    f,
    fitted,
    linear,
    view,
)
from test_backend_step_grid_fit import BENCHED, clients, loaded
from test_backend_step_grid_fit import fitted as fit_steps
from test_backend_step_grid_fit import view as line_view
from test_backend_step_roofline_v41 import CONFIG as V41_CONFIG
from test_bench_replay_drafter import config, scheduled
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_fit_steps import planted_roots  # noqa: F401
from test_parity_served import SAMPLING
from test_run_step_grid import deployment, readings

from atom.compass.artifacts import ArtifactStore, Axis, GateState, Topology, member_name
from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepGrid,
    StepGridBackend,
    step_grid_plan,
)
from atom.compass.backends.step_grid import equivalent_shape
from atom.compass.backends.step_grid_residual import Residual, order
from atom.compass.bench import replay
from atom.compass.bench.measure import measurement_key
from atom.compass.fit.samples import MEMBER
from atom.compass.run import step_grid

K = 5  # DSpark5's drafts


def g(b, c):
    """A drafter value, linear in each coordinate, so a cell reproduces it."""
    return 300 + 2 * b + c / 8 + b * c / 64


def drafted(points=REPLAY, alpha=None, draft_alpha=None):
    """`test_backend_step_grid`'s replay grid under DSpark5, each point holding
    `g` as its drafter value."""
    grid = bundle("replay", [{**p, "draft": g(*p["at"][1:])} for p in points])
    grid["draft_width"] = K
    for key, value in (("alpha", alpha), ("draft_alpha", draft_alpha)):
        if value:
            grid["decode"][key] = value
    return grid


HALF = Residual(fitted({"half": linear(0.5)}, {"half": 1.0}), drafter=True)


def backend(grid):
    drafter = (StepGrid(grid, drafter=True), HALF)
    return StepGridBackend(StepGrid(grid), UNIT, "drafted", drafter)


def verify(*cached, rung=32):
    """A verify step: a row of K + 1 tokens on each cached context."""
    return view(*[(K + 1, c + K + 1, True) for c in cached], rung=rung)


def test_a_verify_step_is_priced_as_a_target_and_a_draft_term_each_from_its_own_value():
    cost = backend(drafted()).price(verify(*[600] * 20))

    (target,), (draft,) = cost.terms, cost.draft_terms
    assert (target.name, draft.name) == ("forward", "draft")
    assert target.seconds == pytest.approx((f(20, 600) + 3200) * 1e-6, rel=1e-12)
    assert draft.seconds == pytest.approx(0.5 * g(20, 600) * 1e-6, rel=1e-12)
    assert "; decode rung 32 cell" in target.provenance.detail
    assert "; drafter decode rung 32 cell" in draft.provenance.detail
    assert draft.provenance.detail.endswith("residual 0.5 = 1.0 x half 0.5")
    assert cost.seconds == pytest.approx(target.seconds + draft.seconds)


@pytest.mark.parametrize("width", [1, 2, 7])
def test_a_decode_row_of_another_width_is_refused(width):
    rows = [(width, 600 + width, True)] * 20
    with pytest.raises(
        CostRefused,
        match="computes other than the 6 tokens of a verify step of 5 drafts",
    ):
        backend(drafted()).price(view(*rows, rung=32))


def test_each_value_reads_a_ragged_batch_by_its_own_alpha():
    """The drafter's alpha reads 18 rows at c 560, one at 600 and one at 1000
    at the longest row; the target's grid holds none and reads their mean, 584."""
    alpha = {"32": {"intercept": 1.0, "ratio": 0.0, "spread": 0.0}}
    cost = backend(drafted(draft_alpha=alpha)).price(verify(*[560] * 18, 600, 1000))

    assert cost.terms[0].seconds == pytest.approx((f(20, 584) + 3200) * 1e-6)
    assert cost.draft_terms[0].seconds == pytest.approx(0.5 * g(20, 1000) * 1e-6)
    assert "longest row's cell" in cost.draft_terms[0].provenance.detail
    assert "longest row's cell" not in cost.terms[0].provenance.detail


def test_past_the_pool_a_row_holds_its_verify_tokens_and_the_draft_rounds_positions():
    """2304 blocks of 16 hold 48 rows to c 16 * 48 - 6 - 4 = 758 under five
    drafts: a corner at c 762 is past the pool, so a step at c 760 extends B=48
    from 512 and 758; were a row's need c + 1, 762 would be inside, unmeasured."""
    at = [(48, b, c) for b in (33, 48) for c in (512, 758, 762) if (b, c) != (48, 762)]
    grid = drafted(
        [{"at": list(p), "time": f(*p[1:]), "statistic": "m", "repeats": 1} for p in at]
    )
    grid["decode"]["nodes"] = {"B": {"48": [33, 48]}, "c": {"48": [512, 758, 762]}}
    grid["decode"]["pool"] = {"block_size": 16, "blocks": 2304, "lookahead": 4}

    cost = backend(grid).price(verify(*[760] * 48, rung=48))

    past = "step-grid-extrapolated: c 2 past 758, the pool's edge at B=48"
    for term in (*cost.terms, *cost.draft_terms):
        assert term.provenance.source == past
    assert cost.terms[0].seconds == pytest.approx(f(48, 760) * 1e-6, rel=1e-12)
    assert cost.draft_terms[0].seconds == pytest.approx(0.5 * g(48, 760) * 1e-6)


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda m: m.pop("draft_width"), "states no draft width, so it holds no"),
        (lambda m: m.update(draft_width=-1), "draft width -1 is not a count"),
        (lambda m: m["decode"]["points"][0].pop("draft"), "'draft'"),
        (
            lambda m: m["decode"]["points"][0].update(draft=0.0),
            "the drafter decode rung 32 point at .* takes 0.0, which is not a",
        ),
    ],
    ids=["no-width", "negative-width", "no-draft", "zero-draft"],
)
def test_a_grid_without_a_drafter_value_fails_loading_for_the_drafter(edit, match):
    mapping = drafted()
    edit(mapping)
    with pytest.raises((ValueError, KeyError), match=match):
        StepGrid(mapping, drafter=True)


def test_its_line_names_the_drafters_residual():
    assert (
        backend(drafted())
        .describe()
        .endswith(
            "; drafter recipe step-grid-residual v3, members decode: half; "
            "prefill_output: half; prefill_no_output: half"
        )
    )


#: The planner's definition, and the engine the scheduler-driven steps state.
DEFINITION = {
    "prefill": {"B": [1, 2], "T": [8, 16, 32, 64], "c": [0, 32, 64, 128]},
    "decode": {"c": [0, 64, 128, 256]},
    "margin": 0,
}
SPECULATIVE = {"dspark5": ["dspark", 5], "mtp1": ["mtp", 1]}
LADDER = [1, 2, 4, 8]


def recorded(model_dir, drafter, blocks=256):  # noqa: F811
    """The steps ATOM's scheduler builds under `drafter`, as named step-record
    lines, each decode step on the smallest rung of `LADDER` holding its rows,
    and the engine they ran on, at a pool of `blocks` blocks."""
    built = config(model_dir, drafter)
    lines = []
    for i, (batch, line, _) in enumerate(scheduled(built)):
        decode = line["rows"][0][2]
        rung = min(r for r in LADDER if r >= len(line["rows"])) if decode else None
        lines.append((f"{drafter} step {i}", {**line, "rung": rung}))
    engine = {
        "max_num_seqs": built.max_num_seqs,
        "max_num_batched_tokens": built.max_num_batched_tokens,
        "max_model_len": built.max_model_len,
        "kv_cache_block_size": built.kv_cache_block_size,
        "num_kvcache_blocks": blocks,
        "ladder": LADDER,
        "speculative": SPECULATIVE[drafter],
    }
    return lines, {f"{drafter} dp0": engine}


@pytest.mark.parametrize("drafter", SPECULATIVE)
def test_a_drafted_record_plans_without_refusal_and_its_points_are_verify_steps(
    model_dir, drafter  # noqa: F811
):
    steps, engines = recorded(model_dir, drafter)
    width = SPECULATIVE[drafter][1] + 1

    planned = step_grid_plan.plan(DEFINITION, steps, engines)

    assert planned["unplaced"] == [] and planned["infeasible"] == []
    assert planned["draft_width"] == width - 1
    grid = StepGrid({**planned, "unit": "s"})
    decode = [x for x in planned["lines"] if x["rung"]]
    assert decode and len(decode) < len(planned["lines"])
    for line in decode:
        rows = line["rows"]
        assert {tuple(r) for r in rows} == {(width, rows[0][1], True)}
        v = BatchView(tuple(RequestShape(*r) for r in rows), capture_rung=line["rung"])
        cells, at = grid.locate(v)
        assert cells.corners(v, at) == [((len(rows), rows[0][1] - width), 1)]


@pytest.mark.parametrize("drafter", SPECULATIVE)
def test_each_kv_limit_is_the_largest_context_the_replay_rule_fits_in_the_pool(
    model_dir, drafter  # noqa: F811
):
    """At a 30-block pool each rung's c nodes add each B node's KV limit, 16 *
    (30 // B) less the verify step's tokens and the positions the draft round
    writes past them; B rows on it fit the pool the replay rule allocates, as
    ATOM's scheduler does, and on one more token do not."""
    steps, engines = recorded(model_dir, drafter, blocks=30)
    width = SPECULATIVE[drafter][1] + 1
    built = config(model_dir, drafter)
    built.num_kvcache_blocks = 30
    rule = replay.ReplayRule(built)

    planned = step_grid_plan.plan(DEFINITION, steps, engines)

    nodes = planned["decode"]["nodes"]
    lookahead = width - 2 if width > 2 else 0
    for rung, bs in nodes["B"].items():
        for b in bs:
            limit = 16 * (30 // b) - width - lookahead
            assert limit in nodes["c"][rung] or limit >= 2048 - width
            fits = {"rows": [[width, limit + width, True]] * b, "produces_output": True}
            rule.batch(fits)
            over = {**fits, "rows": [[width, limit + 1 + width, True]] * b}
            with pytest.raises(
                ValueError, match="KV blocks of 16 tokens, and the pool"
            ):
                rule.batch(over)
        assert 2048 - width in nodes["c"][rung]
    assert planned["decode"]["pool"] == {
        "block_size": 16,
        "blocks": 30,
        **({"lookahead": lookahead} if lookahead else {}),
    }


def test_a_step_of_another_width_than_the_engines_drafter_refuses_the_plan(
    model_dir,  # noqa: F811
):
    steps, engines = recorded(model_dir, "dspark5")
    name, line = next((n, x) for n, x in steps if x["rows"][0][2])
    line["rows"] = [[1, c - K, True] for _, c, _ in line["rows"]]

    with pytest.raises(
        ValueError,
        match=f"^{name} decodes 1 token in a row, and the engine its inputs "
        "state verifies 5 drafts$",
    ):
        step_grid_plan.plan(DEFINITION, steps, engines)


#: The fixture's bounds and drafter: DeepSeek-V4.1-Flash under DSpark5, as
#: stage 4's MI455 cells serve it, at `test_backend_step_grid_fit`'s bounds.
BOUNDS = grid_fit.BOUNDS
SPEC = SimpleNamespace(
    method="dspark", num_speculative_tokens=K, use_dspark=lambda: True
)


@pytest.fixture
def flash(tmp_path):
    """The deployment, at TP4 with V4's 256-token KV blocks under DSpark5,
    serving the published V4.1-Flash config.json; its readings and the
    inputs' configuration."""
    r = deployment(
        tmp_path,
        **BOUNDS,
        capture_sizes=[8],
        speculative_config=SPEC,
        kv_cache_block_size=256,
        tensor_parallel_size=4,
    )
    shutil.copy(V41_CONFIG, tmp_path / "config.json")
    here = readings(r)
    configuration = {
        "model": here.reading(Axis.MODEL).value,
        "device": "MI355X gfx950",
        "source_root": "{0.kind}-{0.value}".format(here.reading(Axis.ATOM_SRC)),
    }
    return r, here, configuration


def draft_time(line) -> float:
    """The drafter's planted value at a node, and so its lookup anywhere."""
    b, n_q, c = map(float, equivalent_shape(line_view(line).requests))
    if line["rows"][0][2]:
        return 4e-4 + 3e-6 * b + 2e-8 * c + 1e-10 * b * c
    head = 2e-4 if line["produces_output"] else 1e-4
    return head + 1e-6 * b + 3e-7 * b * n_q + 1e-7 * c + 1e-10 * b * n_q * c


def draft_factor(line) -> float:
    """The drafter residual's planted factor, on order alone."""
    return 0.8 * math.exp(0.1 * float(order(line_view(line).requests)))


def target(line) -> float:
    return grid_fit.planted_time(line) * grid_fit.factor(line)


def flash_nodes() -> list[dict]:
    """Every node batch of the grid the planner places for `BOUNDS` under
    DSpark5: decode's top c node is max_model_len less a verify step's 6."""
    out = [
        grid_fit.batch([(K + 1, c + K + 1, True)] * b)
        for b in (1, 8)
        for c in (0, 512, 1018)
    ]
    return out + [x for x in grid_fit.nodes() if not x["rows"][0][2]]


def flash_mixed(seed=1, n=20) -> list[dict]:
    """`n` verify steps of 2 to 8 rows on cached contexts 0 to 1018, and
    `test_backend_step_grid_fit.mixed`'s prefill chunks."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ks = rng.integers(K + 1, 1025, rng.integers(2, 9))
        out.append(grid_fit.batch([(K + 1, int(k), True) for k in ks]))
    return out + grid_fit.mixed(seed, n)[n:]


def flash_record(tmp_path, run, lines, monkeypatch):
    """`test_backend_step_grid_fit.record` of `lines`, each step's device window
    its target's part and its drafter's, the target ending at `target`."""
    with monkeypatch.context() as m:
        whole = lambda x: (target(x) + draft_time(x) * draft_factor(x))
        m.setattr(grid_fit, "time", lambda x: whole(x) / grid_fit.factor(x))
        directory = grid_fit.record(tmp_path, run, lines)
    path = directory / "steps" / "dp0.jsonl"
    written = [json.loads(x) for x in path.read_text().splitlines()]
    for line, x in zip(written, lines):
        line["d_target_ns"] = round(target(x) * 1e9)
    path.write_text("".join(json.dumps(x) + "\n" for x in written))
    return directory


def flash_measured(tmp_path, run, line):
    """`line`'s step measurement as `measure steps` publishes one under a
    drafter: the target's value and the drafter's."""
    r, here, configuration = run
    key = measurement_key(configuration, step_grid.keyed(r.config), SAMPLING, line)
    member = {
        **line,
        "seconds": grid_fit.planted_time(line),
        "draft_seconds": draft_time(line),
        "statistic": "median",
        "repeats": BENCHED["repeats"],
        "origin": "plan",
        "sources": ["planted"],
        "traces": ["trace-7"],
    }
    ArtifactStore(tmp_path / "store").publish(
        key,
        provenance=stanza("compass measure steps"),
        topology=Topology(),
        conditions=here,
        gates=GateState.of(),
        members={member_name(*MEMBER): json.dumps(member).encode()},
    )


def test_a_v41_flash_dspark5_verify_step_is_priced_from_a_fitted_bundle(
    tmp_path, flash, monkeypatch
):
    """The named result: `fit steps` plans the record's grid, reads each point's
    two values, fits each value's residual on the record's labels, the target's
    and the drafter's, and a run's backend prices a verify step of the record
    at each value's lookup times that value's planted factor."""
    lines = flash_nodes() * 4 + flash_mixed()
    a = flash_record(tmp_path, flash, lines, monkeypatch)
    for line in flash_nodes():
        flash_measured(tmp_path, flash, line)

    copy = fit_steps(tmp_path, a, *clients(a))
    backend = loaded(tmp_path, flash, copy)

    params, report = copy["params"], copy["report"]
    assert params["grid"]["draft_width"] == K
    assert json.loads(copy["key"]["engine_config"])["speculative"] == ["dspark", K]
    assert (report["unpriced"], report["draft"]["unpriced"]) == ([], [])
    draft_residual = Residual(params["draft_residual"], drafter=True)
    for line in flash_nodes() + flash_mixed():
        cost = backend.price(line_view(line))
        (forward,), (draft,) = cost.terms, cost.draft_terms
        assert forward.seconds == pytest.approx(target(line), rel=1e-3)
        part = draft_time(line) * draft_factor(line)
        assert draft.seconds == pytest.approx(part, rel=1e-3)
        factor = draft_residual.factor(line_view(line))[0]
        assert draft.seconds == pytest.approx(draft_time(line) * factor, rel=1e-9)
    step = flash_mixed()[0]
    cost = backend.price(line_view(step))
    for term, residual in zip(
        (*cost.terms, *cost.draft_terms), ("residual", "draft_residual")
    ):
        (member,) = params[residual]["kinds"]["decode"]["selected"]
        assert f" x {member} " in term.provenance.detail.split("; residual ")[1]


@pytest.mark.parametrize("c, fits", [(60, True), (61, False)])
def test_a_point_is_planned_only_where_the_replay_rule_fits_the_draft_rounds_positions(
    model_dir, c, fits  # noqa: F811
):
    """At an 8-block pool one prefill row of 64 tokens on 61 cached fills 8
    blocks of 16, and its draft round's 4 positions past it a ninth."""
    target = {
        **recorded(model_dir, "dspark5")[1]["dspark5 dp0"],
        "num_kvcache_blocks": 8,
        "draft_width": K,
        "lookahead": K - 1,
    }
    built = config(model_dir, "dspark5")
    built.num_kvcache_blocks = 8
    line = {"rows": [[64, c + 64, False]], "produces_output": False}

    why = step_grid_plan._infeasible(target, 1, 64, c, prefill=True)

    assert (why == "") == fits
    if fits:
        replay.ReplayRule(built).batch(line)
        return
    assert why == "9 KV blocks are above num_kvcache_blocks"
    with pytest.raises(ValueError, match="hold 9 KV blocks of 16 tokens"):
        replay.ReplayRule(built).batch(line)
